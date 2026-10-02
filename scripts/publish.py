"""训练完成后一键产出可分发模型：HF 导出 → GGUF → 量化 → 等价性验证。

把 export_hf.py / to_gguf.py / llama-quantize / verify_gguf.py 串起来，
避免每次训练完都手敲四条命令。任一步失败立即终止并保留现场。

用法::

    python scripts/publish.py --config configs/hotsteel_v2.yaml \
        --ckpt checkpoints_v2/final.pt --name hotsteel0.1-v2

    # 只做导出不做验证
    python scripts/publish.py --config ... --ckpt ... --name ... --skip_verify
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DEFAULT_LLAMA_BIN = os.path.join(os.path.dirname(HERE), "llama_bin")


def run(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===")
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"[publish] {desc} 失败，退出码 {r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="一键导出与验证")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="产物基名，如 hotsteel0.1-v2")
    ap.add_argument("--outtype", default="f16")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--llama_bin", default=DEFAULT_LLAMA_BIN)
    ap.add_argument("--skip_verify", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    export_dir = os.path.join(HERE, "export")
    os.makedirs(export_dir, exist_ok=True)
    hf_dir = os.path.join(export_dir, f"{args.name}-hf")
    f16 = os.path.join(export_dir, f"{args.name}-{args.outtype}.gguf")
    quant = os.path.join(export_dir, f"{args.name}-q4_k_m.gguf")
    llama_cli = os.path.join(args.llama_bin, "llama-cli.exe")
    quantize = os.path.join(args.llama_bin, "llama-quantize.exe")

    # 1) HF Llama 格式
    run([PY, "scripts/export_hf.py", "--config", args.config,
         "--ckpt", args.ckpt, "--out", hf_dir], "1/4 导出 HF Llama 格式")

    # 2) GGUF
    run([PY, "scripts/to_gguf.py", "--hf", hf_dir,
         "--out", f16, "--outtype", args.outtype], f"2/4 转换 GGUF ({args.outtype})")

    # 3) 量化
    if os.path.isfile(quantize):
        run([quantize, f16, quant, args.quant], f"3/4 量化 {args.quant}")
    else:
        print(f"[publish] 找不到 {quantize}，跳过量化")

    # 4) 等价性验证（对未量化权重做，排除量化误差干扰）
    if not args.skip_verify and os.path.isfile(llama_cli):
        run([PY, "scripts/verify_gguf.py", "--config", args.config,
             "--ckpt", args.ckpt, "--gguf", f16,
             "--llama_cli", llama_cli, "--threads", str(args.threads)],
            "4/4 贪婪解码等价性验证")

    print("\n=== 产物 ===")
    for p in (hf_dir, f16, quant):
        if os.path.exists(p):
            size = (os.path.getsize(p) / 1e6) if os.path.isfile(p) else 0
            print(f"  {p}" + (f"  ({size:.1f} MB)" if size else "/"))


if __name__ == "__main__":
    main()
