"""与 llama-perplexity 做数值对照，验证 GGUF 实现正确性。

关键点：llama.cpp 的 ``llama-perplexity`` **不是**逐 token 遍历全部文本，而是
把文本切成 ``n_ctx`` 大小的窗口，每个窗口只评估**后半段**的 token
（``first = n_ctx/2``，给模型留出上下文），并且**丢弃**不够一个窗口的尾部。

参见 llama.cpp ``tools/perplexity/perplexity.cpp``::

    const int first = n_ctx/2;
    ...
    const int n_chunk_max = tokens.size() / n_ctx;
    ...
    process_logits(..., tokens_data, n_ctx - 1 - first, ...);
    count += n_ctx - first - 1;

所以直接拿「全 token 逐位 NLL」去比对必然对不上。本脚本支持两种协议：

* ``--protocol full``  —— 逐 token 全量 NLL（旧行为）
* ``--protocol llama`` —— 复刻 llama-perplexity（窗口 = ``--chunk``，
  只算 ``[skip_first, chunk-2]`` 的预测，尾部余数丢弃），并打印与
  llama 相同的逐窗口累计 PPL，便于逐行比对。

用法::

    # 复刻 llama-perplexity -c 512
    python scripts/compare_ppl.py --text data/ppl_sample.txt \
        --protocol llama --chunk 512 --skip-first 256

    # 逐 token 全量
    python scripts/compare_ppl.py --text data/ppl_sample.txt --protocol full
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from hotsteel.config import load_config
from hotsteel.model import HotsteelLM, ModelConfig
from hotsteel.tokenizer import HotsteelTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="PyTorch 侧 perplexity 计算")
    ap.add_argument("--config", default="configs/hotsteel_100m.yaml")
    ap.add_argument("--ckpt", default="checkpoints_100m/final.pt")
    ap.add_argument("--text", required=True, help="纯文本文件（与 llama-perplexity 同一份）")
    ap.add_argument("--protocol", default="full", choices=["full", "llama"])
    ap.add_argument("--chunk", type=int, default=512, help="llama 协议的窗口大小 n_ctx")
    ap.add_argument("--skip-first", type=int, default=None,
                    help="窗口内跳过前 N 个位置（llama 为 chunk//2）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tokenizer = HotsteelTokenizer.load(cfg.paths.tokenizer_dir)

    with open(args.text, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()

    ids = tokenizer.encode_ids(text)
    print(f"文本字符数={len(text)}  token 数={len(ids)}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = HotsteelLM(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.protocol == "full":
        skip_first, chunk, n_chunk = 0, args.chunk, None
    else:
        chunk = args.chunk
        skip_first = args.chunk // 2 if args.skip_first is None else args.skip_first
        n_chunk = len(ids) // chunk           # llama: n_chunk_max = tokens.size()/n_ctx
        print(f"[llama 协议] 窗口={chunk} 跳过前{skip_first}位置 "
              f"窗口数={n_chunk} (丢弃尾部 {len(ids) - n_chunk * chunk} token)")

    total_nll = 0.0
    total_tok = 0
    windows = range(0, len(ids) - chunk - 1, chunk) if n_chunk is None else range(n_chunk)
    with torch.no_grad():
        for wi, start in enumerate(windows):
            base = wi * chunk if n_chunk is not None else start
            seg = ids[base: base + chunk]
            if len(seg) < chunk:
                break
            x = torch.tensor([seg[:-1]], dtype=torch.long, device=device)
            logits = model(x)                                  # 位置 0..chunk-2
            logits = logits[0, skip_first:, :]                 # 只取 [skip_first, chunk-2]
            y = torch.tensor([seg[skip_first + 1:]], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, y[0], reduction="sum")
            total_nll += loss.item()
            total_tok += y.numel()
            if n_chunk is not None:
                cur = math.exp(total_nll / total_tok)
                print(f"[{wi + 1}]{cur:.4f},", end="\n" if (wi + 1) % 7 == 0 else "")

    print()
    ppl = math.exp(total_nll / total_tok)
    print(f"PyTorch  PPL = {ppl:.4f}   (平均 NLL {total_nll/total_tok:.4f}, "
          f"评测 token 数 {total_tok})")


if __name__ == "__main__":
    main()
