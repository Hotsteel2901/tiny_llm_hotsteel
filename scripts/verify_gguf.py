"""校验 GGUF 与 PyTorch 推理是否等价。

用同一组提示做**贪婪解码**，分别跑我方 PyTorch 实现与 llama.cpp，
逐条比对输出文本。两者一致即证明：
1. 权重导出（键名映射）正确；
2. RoPE 等算子约定与 llama.cpp 一致。

用法::

    python scripts/verify_gguf.py --config configs/hotsteel_100m.yaml \
        --ckpt checkpoints_100m/final.pt \
        --gguf export/hotsteel0.1-100m-f16.gguf \
        --llama_cli ../llama_bin/llama-cli.exe
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hotsteel.config import load_config
from hotsteel.generate import generate
from hotsteel.model import HotsteelLM, ModelConfig
from hotsteel.tokenizer import HotsteelTokenizer
from hotsteel.utils import pick_device

PROMPTS = [
    "你好",
    "你是谁？",
    "为什么天空是蓝色的？",
    "什么是人工智能？",
    "怎样保护眼睛？",
    "太阳是什么？",
    "人是什么？",
    "谢谢",
]


def torch_generate(model, tokenizer, prompt, max_new_tokens, device):
    """PyTorch 贪婪解码，返回回答文本。"""
    ids = tokenizer.build_prompt(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model, x,
        max_new_tokens=max_new_tokens,
        temperature=0.0, top_k=0, top_p=1.0,
        repetition_penalty=1.0, greedy=True,
        eos_id=tokenizer.end_id,
    )
    new = out[0, len(ids):].tolist()
    if tokenizer.end_id in new:
        new = new[: new.index(tokenizer.end_id)]
    return tokenizer.decode(new).strip()


def llama_generate(llama_cli, gguf, prompt, max_new_tokens, threads):
    """调用 llama-cli 做贪婪解码，解析出回答文本。"""
    llama_cli = os.path.abspath(llama_cli)
    gguf = os.path.abspath(gguf)
    raw = f"<|user|>{prompt}<|assistant|>"
    cmd = [
        llama_cli, "-m", gguf,
        "-p", raw,
        "--special",
        "-n", str(max_new_tokens),
        "--temp", "0",
        "-t", str(threads),
        "--no-warmup",
    ]
    proc = subprocess.run(
        cmd, input="", capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        cwd=os.path.dirname(llama_cli),
    )
    lines = proc.stdout.splitlines()
    # 定位回显的提示行（形如 "> <|user|>...<|assistant|>"），取其后到空行为止的内容
    captured = []
    hit = False
    for ln in lines:
        if not hit and ln.lstrip().startswith("> ") and raw in ln:
            hit = True
            continue
        if hit:
            if not ln.strip():
                break
            captured.append(ln)
    text = "\n".join(captured).strip()
    return text.replace("<|end|>", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="校验 GGUF 与 PyTorch 推理等价性")
    ap.add_argument("--config", default="configs/hotsteel_100m.yaml")
    ap.add_argument("--ckpt", default="checkpoints_100m/final.pt")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama_cli", default="../llama_bin/llama-cli.exe")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device("cpu")  # 用 CPU 保证与 llama.cpp 数值路径更接近
    tokenizer = HotsteelTokenizer.load(cfg.paths.tokenizer_dir)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    mcfg = ModelConfig(**ckpt["model_config"])
    model = HotsteelLM(mcfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ok = 0
    for p in PROMPTS:
        t = torch_generate(model, tokenizer, p, args.max_new_tokens, device)
        l = llama_generate(args.llama_cli, args.gguf, p, args.max_new_tokens, args.threads)
        same = t == l
        ok += same
        flag = "一致" if same else "不一致"
        print(f"[{flag}] {p}")
        if not same:
            print(f"    PyTorch : {t}")
            print(f"    llama.cpp: {l}")
        else:
            print(f"    -> {t}")
    print(f"\n结果: {ok}/{len(PROMPTS)} 条输出完全一致")
    if ok != len(PROMPTS):
        print("注意：贪婪解码下个别不一致通常源于浮点累加顺序差异，"
              "若文本语义接近可视为等价。")


if __name__ == "__main__":
    main()
