#!/bin/bash
# 等训练完成 -> 自动导出 HF -> 转 GGUF -> 量化 -> 校验
cd "D:/AI_LLM_Train/hotsteel0.1" || exit 1
PY=./.venv/Scripts/python.exe
i=0
while ! grep -q "训练完成" train_100m.log; do
  sleep 20
  i=$((i+1))
  if [ $i -gt 500 ]; then echo "TIMEOUT_WAITING"; exit 2; fi
done
echo "=== TRAIN DONE, exporting ==="
$PY scripts/export_hf.py --config configs/hotsteel_100m.yaml \
    --ckpt checkpoints_100m/final.pt --out export/hotsteel0.1-100m-hf || exit 3
echo "=== convert to gguf f16 ==="
$PY scripts/to_gguf.py --hf export/hotsteel0.1-100m-hf \
    --out export/hotsteel0.1-100m-f16.gguf --outtype f16 || exit 4
echo "=== quantize q4_k_m ==="
../llama_bin/llama-quantize.exe export/hotsteel0.1-100m-f16.gguf \
    export/hotsteel0.1-100m-q4_k_m.gguf Q4_K_M 2>&1 | tail -4
echo "=== verify equivalence ==="
$PY scripts/verify_gguf.py --config configs/hotsteel_100m.yaml \
    --ckpt checkpoints_100m/final.pt \
    --gguf export/hotsteel0.1-100m-f16.gguf \
    --llama_cli ../llama_bin/llama-cli.exe --max_new_tokens 80 2>&1 | tail -30
echo "PIPELINE_DONE"
