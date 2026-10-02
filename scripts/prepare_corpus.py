"""构建最终训练语料：真实指令数据 + 合成问答 + 教师语料。

数据来源：
1. **真实指令数据**：``data/raw/alpaca_gpt4_data_zh.json``（GPT-4 生成的中文指令，
   约 4.9 万条），是模型获得「泛化能力」的关键。
2. **合成问答**：由 ``make_corpus.py`` 生成的常识/科学/逻辑等问答。
3. **教师语料**：由 ``teacher_generate.py`` 生成的教师蒸馏数据。

输出统一为我方 jsonl 格式：``{"instruction":..., "response":..., "category":...}``

用法::

    python scripts/prepare_corpus.py --out data/corpus_100m.jsonl \
        --alpaca data/raw/alpaca_gpt4_data_zh.json \
        --synth data/corpus_synth.jsonl \
        --teacher data/teacher_qa.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Dict, List

# 清洗：折叠空白、去掉异常控制符
_WS = re.compile(r"[ \t\u3000]+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(text: str) -> str:
    text = _CTRL.sub("", text)
    text = _WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_alpaca(path: str, max_chars: int, min_resp_chars: int) -> List[Dict[str, str]]:
    """读取 alpaca 格式（instruction/input/output）并转为统一格式。"""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: List[Dict[str, str]] = []
    for rec in raw:
        inst = _clean(str(rec.get("instruction", "")))
        extra = _clean(str(rec.get("input", "")))
        resp = _clean(str(rec.get("output", "")))
        if extra:
            inst = f"{inst}\n{extra}" if inst else extra
        if not inst or not resp:
            continue
        if len(resp) < min_resp_chars:
            continue
        # 过滤过长样本（超过 max_seq_len 会被大量截断，不如丢弃）
        if len(inst) + len(resp) > max_chars:
            continue
        out.append({"instruction": inst, "response": resp, "category": "真实指令"})
    return out


def load_alpaca_many(paths: List[str], max_chars: int, min_resp_chars: int) -> List[Dict[str, str]]:
    """读取多个 alpaca 格式文件并合并（后续统一去重）。"""
    out: List[Dict[str, str]] = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"[警告] 未找到 {p}，跳过")
            continue
        items = load_alpaca(p, max_chars, min_resp_chars)
        print(f"  {os.path.basename(p)}: {len(items)} 条")
        out.extend(items)
    return out


def load_jsonl(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.isfile(path):
        return []
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inst = str(obj.get("instruction", "")).strip()
            resp = str(obj.get("response", "")).strip()
            if inst and resp:
                out.append({
                    "instruction": inst,
                    "response": resp,
                    "category": str(obj.get("category", "合成")).strip() or "合成",
                })
    return out


def dedup(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """按 instruction 去重。

    同一问题若出现多个回答，会造成训练目标冲突（模型学到互相矛盾的答案），
    因此统一保留首次出现的样本。
    """
    seen = set()
    out = []
    for it in items:
        key = it["instruction"].strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 hotsteel0.1 训练语料")
    ap.add_argument("--out", default="data/corpus_100m.jsonl")
    ap.add_argument(
        "--alpaca", nargs="+",
        default=[
            "data/raw/alpaca_gpt4_data_zh.json",
            "data/raw/alpaca_data_51k.json",
        ],
        help="一个或多个 alpaca 格式数据文件",
    )
    ap.add_argument("--synth", default="data/corpus_zh.jsonl")
    ap.add_argument("--teacher", default="data/teacher_qa.jsonl")
    ap.add_argument("--max_chars", type=int, default=700, help="单样本最大字符数（超长丢弃）")
    ap.add_argument("--min_resp_chars", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    records: List[Dict[str, str]] = []
    print("真实指令数据:")
    records.extend(load_alpaca_many(args.alpaca, args.max_chars, args.min_resp_chars))

    synth = load_jsonl(args.synth)
    print(f"合成问答: {len(synth)} 条")
    records.extend(synth)

    teacher = load_jsonl(args.teacher)
    print(f"教师语料: {len(teacher)} 条")
    records.extend(teacher)

    records = dedup(records)
    rng = random.Random(args.seed)
    rng.shuffle(records)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cats: Dict[str, int] = {}
    for it in records:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    print(f"\n合计 {len(records)} 条 -> {args.out}")
    for c, n in sorted(cats.items(), key=lambda x: -x[1])[:12]:
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
