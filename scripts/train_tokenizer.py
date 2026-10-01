"""训练 BPE 分词器。

运行::

    python scripts/train_tokenizer.py --config configs/small.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hotsteel.config import load_config
from hotsteel.tokenizer import train_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 hotsteel0.1 分词器")
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--corpus", default=None, help="覆盖配置中的语料路径")
    args = ap.parse_args()

    cfg = load_config(args.config)
    corpus = args.corpus or cfg.paths.corpus
    if not os.path.isfile(corpus):
        raise SystemExit(f"找不到语料文件 {corpus}，请先运行 scripts/make_corpus.py")

    path = train_tokenizer(
        corpus_path=corpus,
        out_dir=cfg.paths.tokenizer_dir,
        vocab_size=cfg.tokenizer.vocab_size,
        min_frequency=cfg.tokenizer.min_frequency,
        special_tokens=list(cfg.tokenizer.special_tokens),
    )
    print(f"分词器已保存 -> {path}")


if __name__ == "__main__":
    main()
