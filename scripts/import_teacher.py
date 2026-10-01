"""教师蒸馏数据导入：把「教师模型」产出的问答对并入训练语料。

hotsteel0.1 的定位是**学生模型**——它以更强的「教师模型」（如对话式大模型）
对同一批问题的回答作为学习目标。教师可用任意方式生成回答，只要落到
``data/teacher_qa.jsonl`` 即可，本脚本负责校验、清洗、去重并合并进主语料。

教师数据格式（每行一条 JSON，与主语料一致）::

    {"instruction": "为什么天空是蓝色的？", "response": "<教师给出的回答>", "category": "科学"}

用法::

    # 合并教师数据到主语料（保留原有数据）
    python scripts/import_teacher.py --teacher data/teacher_qa.jsonl \
        --corpus data/corpus_zh.jsonl --out data/corpus_zh.jsonl

    # 只查看统计，不写回
    python scripts/import_teacher.py --teacher data/teacher_qa.jsonl --dry_run
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [跳过] {path}:{lineno} JSON 错误: {exc}")
                continue
            inst = str(obj.get("instruction", "")).strip()
            resp = str(obj.get("response", "")).strip()
            if not inst or not resp:
                print(f"  [跳过] {path}:{lineno} 缺少 instruction/response")
                continue
            items.append({
                "instruction": inst,
                "response": resp,
                "category": str(obj.get("category", "教师")).strip() or "教师",
            })
    return items


def dedup(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for it in items:
        key = (it["instruction"], it["response"])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def merge(teacher: List[Dict[str, str]], corpus: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """教师数据优先：同一问题时以教师回答为准。"""
    by_q: Dict[str, Dict[str, str]] = {}
    for it in corpus:
        by_q.setdefault(it["instruction"], it)
    for it in teacher:
        by_q[it["instruction"]] = it  # 覆盖为学生提供教师答案
    return list(by_q.values())


def write_jsonl(items: List[Dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def summarize(items: List[Dict[str, str]], title: str) -> None:
    cats: Dict[str, int] = {}
    for it in items:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    print(f"{title}: {len(items)} 条")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")


def main() -> None:
    ap = argparse.ArgumentParser(description="导入教师蒸馏数据到 hotsteel0.1 语料")
    ap.add_argument("--teacher", default="data/teacher_qa.jsonl", help="教师问答 jsonl")
    ap.add_argument("--corpus", default="data/corpus_zh.jsonl", help="主语料 jsonl")
    ap.add_argument("--out", default="data/corpus_zh.jsonl", help="合并输出路径")
    ap.add_argument("--dry_run", action="store_true", help="只统计不写回")
    args = ap.parse_args()

    teacher = dedup(load_jsonl(args.teacher))
    summarize(teacher, "教师数据")
    corpus = dedup(load_jsonl(args.corpus)) if os.path.isfile(args.corpus) else []
    merged = dedup(merge(teacher, corpus))
    summarize(merged, "合并后语料")

    if args.dry_run:
        print("(dry_run) 未写回文件")
        return
    write_jsonl(merged, args.out)
    print(f"已写回 -> {args.out}")
    print("提示：语料变更后需重新运行 train_tokenizer.py 与 train.py")


if __name__ == "__main__":
    main()
