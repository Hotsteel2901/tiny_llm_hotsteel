"""构建 SFT 语料：在大语料预训练之后，做一轮窄域高质量对齐。

背景
----
v2 直接把 44 万条 Belle 和 ~2100 条教师/合成核心语料混在一起训练，
结果核心问答被稀释（"彩虹是由太阳和月亮组成的"）。这正是
DeepSeek 报告里 SFT 阶段存在的意义：**宽域学语言，窄域学回答**。

配比策略
--------
* 核心语料（corpus_100m 里的教师/合成问答，约 2100 条）**上采样 12 倍**
  —— 这部分是知识主干，必须占足够比重
* 保留一部分真实指令（默认 3 万条）防止灾难性遗忘
* 打乱后输出

用法::

    python scripts/build_corpus_sft.py \
        --core data/corpus_100m.jsonl \
        --pool data/corpus_v2.jsonl \
        --out data/corpus_sft.jsonl --core_repeat 12 --pool_n 30000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_corpus_v2 import load_jsonl, normalize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 SFT 语料")
    ap.add_argument("--core", default="data/corpus_100m.jsonl",
                    help="教师/合成核心语料（知识主干）")
    ap.add_argument("--pool", default="data/corpus_v2.jsonl",
                    help="大语料池，用于抽取防遗忘样本")
    ap.add_argument("--out", default="data/corpus_sft.jsonl")
    ap.add_argument("--core_repeat", type=int, default=12)
    ap.add_argument("--pool_n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    core = [it for it in load_jsonl(args.core) if it["category"] != "真实指令"]
    print(f"核心语料: {len(core)} 条 (类别: "
          f"{dict(Counter(it['category'] for it in core))})")

    # 池子里剔除与核心重复的题，避免上采样后自我重复
    core_keys = {normalize(it["instruction"]) for it in core}
    pool = load_jsonl(args.pool)
    pool = [it for it in pool
            if normalize(it["instruction"]) not in core_keys]
    rng.shuffle(pool)
    picked = pool[: args.pool_n]
    print(f"防遗忘样本: {len(picked)} 条 (从 {len(pool)} 条池中抽取)")

    items = core * args.core_repeat + picked
    rng.shuffle(items)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\n输出 {args.out}: {len(items)} 条")
    for k, v in Counter(it["category"] for it in items).most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
