"""固定题集评测：衡量模型在「核心问答 / 同义改写 / 超纲泛化」三层的表现。

为什么要这个脚本
----------------
只看 val_loss 会误导：语料变大后 val_loss 反而可能上升（分布更难），
但模型泛化能力实际变强。真正决定「聪不聪明」的是：
  L1 核心问答   —— 训练语料直接覆盖的事实（应答对）
  L2 同义改写   —— 同一事实换种问法（考验是否背模板）
  L3 超纲泛化   —— 语料里没有的知识（看是否胡言乱语）

对每层输出贪婪解码结果，人工抽看即可横向对比不同版本。结果同时存 JSON，
方便版本间 diff。

用法::

    python scripts/eval_qa.py --ckpt checkpoints_100m/final.pt \
        --config configs/hotsteel_100m.yaml --out eval_v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hotsteel.config import load_config
from hotsteel.generate import chat_once
from hotsteel.model import HotsteelLM, ModelConfig
from hotsteel.tokenizer import HotsteelTokenizer
from hotsteel.utils import pick_device

# L1：教师语料直接覆盖的核心事实
L1_CORE = [
    "人是什么", "你是谁", "太阳是什么", "为什么天空是蓝色的",
    "地球为什么有白天和黑夜", "什么是人工智能", "怎样保护眼睛",
    "水为什么会结冰", "人类与人猿的区别", "彩虹是怎么形成的",
]

# L2：换种问法，考是否只会背模板
L2_PARAPHRASE = [
    "请介绍一下太阳", "简述人类和猿类的不同点",
    "天空为什么是蓝颜色的呢", "人工智能指的是什么",
    "冰是怎么变成水的",
]

# L3：语料大概率没有覆盖，观察是否胡编
L3_OOD = [
    "请解释量子纠缠", "宋朝的开国皇帝是谁",
    "怎么用 Python 读一个 CSV 文件", "抑郁症有哪些常见表现",
    "为什么飞机能飞起来",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="固定题集评测")
    ap.add_argument("--config", default="configs/hotsteel_100m.yaml")
    ap.add_argument("--ckpt", default="checkpoints_100m/final.pt")
    ap.add_argument("--out", default=None, help="结果存 JSON（可选）")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device or cfg.train.device)
    tokenizer = HotsteelTokenizer.load(cfg.paths.tokenizer_dir)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = HotsteelLM(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gen = cfg.generate
    gen.greedy = True
    gen.max_new_tokens = args.max_new_tokens

    suites = [("L1_核心问答", L1_CORE), ("L2_同义改写", L2_PARAPHRASE), ("L3_超纲泛化", L3_OOD)]
    results: dict = {"ckpt": args.ckpt, "config": args.config, "suites": {}}
    for name, qs in suites:
        print(f"\n===== {name} =====")
        acc = []
        for q in qs:
            ans = chat_once(model, tokenizer, q, gen, device=device)
            acc.append({"q": q, "a": ans.strip()})
            print(f"  Q: {q}\n  A: {ans.strip()}\n")
        results["suites"][name] = acc

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"已保存 -> {args.out}")


if __name__ == "__main__":
    main()
