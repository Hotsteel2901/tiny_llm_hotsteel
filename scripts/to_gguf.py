"""把导出的 HF 格式模型转换为 GGUF（供 llama.cpp 加载）。

为什么需要这层封装：
llama.cpp 的 ``convert_hf_to_gguf.py`` 通过「用分词器编码一段固定探针串并取哈希」
来识别 BPE 预分词器类型。自训练的分词器哈希不在其内置表中，会直接报错。
本脚本在不修改 llama.cpp 源码的前提下，运行时为其补上兜底逻辑：
自训 ByteLevel(GPT-2 正则) 分词器映射到 ``gpt2`` 预分词器类型。

用法::

    python scripts/to_gguf.py --hf export/hotsteel0.1-100m-hf \
        --out export/hotsteel0.1-100m-f16.gguf --outtype f16

    # 可选：量化
    python scripts/to_gguf.py --hf ... --out ... --outtype f16
    ../llama_bin/llama-quantize.exe export/xxx-f16.gguf export/xxx-q4_k_m.gguf Q4_K_M
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys

DEFAULT_LLAMA_DIR = os.environ.get("LLAMA_CPP_DIR", "D:/AI_LLM_Train/llama.cpp")


def main() -> None:
    ap = argparse.ArgumentParser(description="转换 HF 模型为 GGUF")
    ap.add_argument("--hf", required=True, help="HF 格式模型目录")
    ap.add_argument("--out", required=True, help="输出 .gguf 路径")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0", "auto"])
    ap.add_argument("--llama_dir", default=DEFAULT_LLAMA_DIR)
    args = ap.parse_args()

    llama_dir = os.path.abspath(args.llama_dir)
    conv = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    if not os.path.isfile(conv):
        raise SystemExit(f"找不到 {conv}，请设置 --llama_dir 或环境变量 LLAMA_CPP_DIR")

    sys.path.insert(0, llama_dir)
    from conversion import base  # noqa: E402

    # 兜底：自训 ByteLevel 分词器 → gpt2 风格预分词器
    _orig = base.TextModel.get_vocab_base_pre

    def patched(self, tokenizer):  # type: ignore[no-untyped-def]
        try:
            name = _orig(self, tokenizer)
            print(f"[to_gguf] 预分词器识别为: {name}")
            return name
        except NotImplementedError:
            print("[to_gguf] 自训分词器未在内置表中，回退为 gpt-2 预分词器（ByteLevel 正则一致）")
            return "gpt-2"

    base.TextModel.get_vocab_base_pre = patched

    sys.argv = [
        "convert_hf_to_gguf.py",
        args.hf,
        "--outfile", args.out,
        "--outtype", args.outtype,
    ]
    runpy.run_path(conv, run_name="__main__")


if __name__ == "__main__":
    main()
