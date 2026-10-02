"""构建 v2 语料：真实开源中文指令数据 + 现有语料，统一清洗去重。

数据来源
--------
* ``BelleGroup/train_0.5M_CN``（52 万条真实中文指令，Belle 开源）
* 现有 ``data/corpus_100m.jsonl``（真实指令 + 合成 + 教师蒸馏语料）

清洗规则（借鉴 DeepSeek 报告中的数据构建思路，裁剪到我们这个规模适用的子集）
--------------------------------------------------------------------------
1. 语言过滤：回答里中文字符占比过低（< 0.15）的丢弃 —— 我们要的是中文模型，
   英文/代码为主的数据对它只会造成分布污染。
2. 长度过滤：instruction 3~1500 字符，response 4~1600 字符。
   过短没有信息量，过长会被 512 截断浪费算力。
3. 退化样本过滤：回答为空、只有标点、与提问雷同、含大量重复字符的丢弃。
4. **按归一化 instruction 去重**（沿用 v1 规则），避免同题多答案冲突。
5. 合并后打乱，保证教师/合成语料不被大语料淹没（Belle 上限可配）。

用法::

    python scripts/build_corpus_v2.py \
        --belle data/raw/Belle_open_source_0.5M.json \
        --extra data/corpus_100m.jsonl \
        --out data/corpus_v2.jsonl --max_belle 600000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter

CN_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCT_RE = re.compile(r"^[\s\W_a-zA-Z0-9]+$")
REPEAT_RE = re.compile(r"(.)\1{8,}")


def normalize(text: str) -> str:
    """归一化用于去重：全半角统一、去空白与标点。"""
    t = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s\W]+", "", t.lower())


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(CN_RE.findall(text)) / len(text)


def is_degenerate(inst: str, resp: str) -> bool:
    if PUNCT_RE.match(resp):
        return True
    if REPEAT_RE.search(resp):
        return True
    if normalize(resp) and normalize(resp) == normalize(inst):
        return True
    return False


def clean(inst: str, resp: str) -> tuple[str, str] | None:
    inst = re.sub(r"\s+", " ", inst.replace("\r", " ")).strip()
    resp = resp.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not (3 <= len(inst) <= 1500 and 4 <= len(resp) <= 1600):
        return None
    if chinese_ratio(resp) < 0.15:
        return None
    if is_degenerate(inst, resp):
        return None
    return inst, resp


def load_belle(path: str, limit: int, rng: random.Random) -> list[dict]:
    """读 Belle json（每行一个对象），水库抽样到 limit 条再清洗。"""
    raw: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            inst = str(obj.get("instruction", "")) + (
                "\n" + str(obj.get("input", "")).strip()
                if str(obj.get("input", "")).strip()
                else ""
            )
            resp = str(obj.get("output", ""))
            raw.append((inst, resp))
    rng.shuffle(raw)
    out, seen = [], 0
    for inst, resp in raw:
        c = clean(inst, resp)
        if c is None:
            continue
        out.append({"instruction": c[0], "response": c[1], "category": "Belle真实"})
        seen += 1
        if seen >= limit:
            break
    return out


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            inst = str(obj.get("instruction", "")).strip()
            resp = str(obj.get("response", obj.get("output", ""))).strip()
            cat = str(obj.get("category", "合成")).strip() or "合成"
            c = clean(inst, resp)
            if c is None:
                continue
            out.append({"instruction": c[0], "response": c[1], "category": cat})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 v2 语料")
    ap.add_argument("--belle", default="data/raw/Belle_open_source_0.5M.json")
    ap.add_argument("--extra", nargs="*", default=["data/corpus_100m.jsonl"])
    ap.add_argument("--out", default="data/corpus_v2.jsonl")
    ap.add_argument("--max_belle", type=int, default=600000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items: list[dict] = []
    if os.path.isfile(args.belle):
        belle = load_belle(args.belle, args.max_belle, rng)
        print(f"Belle 清洗后: {len(belle)}")
        items += belle
    for p in args.extra:
        if os.path.isfile(p):
            got = load_jsonl(p)
            print(f"{p} 清洗后: {len(got)}")
            items += got

    # 按 instruction 去重（保留先出现的：extra 里教师/合成语料优先）
    seen: set[str] = set()
    deduped = []
    for it in items:
        key = normalize(it["instruction"])
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    rng.shuffle(deduped)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_tok_chars = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for it in deduped:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            n_tok_chars += len(it["instruction"]) + len(it["response"])

    cats = Counter(it["category"] for it in deduped)
    print(f"\n输出 {args.out}: {len(deduped)} 条, 约 {n_tok_chars/1e6:.1f}M 字符")
    for k, v in cats.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
