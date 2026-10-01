"""hotsteel0.1 命令行对话 / 批量推理。

交互模式::

    python scripts/chat.py --config configs/small.yaml --ckpt checkpoints/final.pt

单次提问::

    python scripts/chat.py --ckpt checkpoints/final.pt --prompt "中国首都是哪里？"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hotsteel.config import load_config
from hotsteel.generate import chat_once
from hotsteel.model import HotsteelLM, ModelConfig
from hotsteel.tokenizer import HotsteelTokenizer
from hotsteel.utils import human_params, pick_device


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = HotsteelLM(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="hotsteel0.1 对话")
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--ckpt", default="checkpoints/final.pt")
    ap.add_argument("--prompt", default=None, help="单次提问；不填进入交互模式")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gen = cfg.generate
    if args.max_new_tokens is not None:
        gen.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        gen.temperature = args.temperature
    if args.top_k is not None:
        gen.top_k = args.top_k
    if args.top_p is not None:
        gen.top_p = args.top_p
    if args.greedy:
        gen.greedy = True

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"找不到权重 {args.ckpt}，请先训练或指定 --ckpt")

    device = pick_device(args.device or cfg.train.device)
    tokenizer = HotsteelTokenizer.load(cfg.paths.tokenizer_dir)
    model = load_model(args.ckpt, device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[hotsteel0.1] 已加载 {args.ckpt} | 参数 {human_params(n)} | 设备 {device}")

    def ask(text: str) -> str:
        return chat_once(model, tokenizer, text, gen, device=device)

    if args.prompt:
        print(f"\n问: {args.prompt}")
        print(f"答: {ask(args.prompt)}")
        return

    print("进入交互模式，输入问题回车；输入 :q 退出。\n")
    while True:
        try:
            text = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text in (":q", ":quit", "exit", "quit"):
            break
        if not text:
            continue
        print(f"模型 > {ask(text)}\n")


if __name__ == "__main__":
    main()
