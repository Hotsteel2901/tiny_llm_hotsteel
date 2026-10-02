"""把 hotsteel 训练好的 checkpoint 导出为 **HuggingFace Llama 格式**目录，
以便用 llama.cpp 的 ``convert_hf_to_gguf.py`` 转成 GGUF。

输出目录结构::

    <out_dir>/
        config.json              # LlamaConfig（model_type=llama）
        model.safetensors        # 权重（键名与 HF Llama 对齐）
        tokenizer.json           # BPE 词表（HF tokenizers 格式）
        tokenizer_config.json    # eos/bos 与 chat_template
        special_tokens_map.json

用法::

    python scripts/export_hf.py --config configs/hotsteel_100m.yaml \
        --ckpt checkpoints_100m/final.pt --out export/hotsteel0.1-100m-hf
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hotsteel.config import load_config
from hotsteel.model import HotsteelLM, ModelConfig
from hotsteel.tokenizer import (
    ASSISTANT,
    END,
    PAD,
    TOKENIZER_FILE,
    UNK,
    USER,
)

# 我方模块名 → HF Llama 键名
_LAYER_MAP = [
    ("blocks.{i}.norm1.weight", "model.layers.{i}.input_layernorm.weight"),
    ("blocks.{i}.attn.q_proj.weight", "model.layers.{i}.self_attn.q_proj.weight"),
    ("blocks.{i}.attn.k_proj.weight", "model.layers.{i}.self_attn.k_proj.weight"),
    ("blocks.{i}.attn.v_proj.weight", "model.layers.{i}.self_attn.v_proj.weight"),
    ("blocks.{i}.attn.o_proj.weight", "model.layers.{i}.self_attn.o_proj.weight"),
    ("blocks.{i}.norm2.weight", "model.layers.{i}.post_attention_layernorm.weight"),
    ("blocks.{i}.mlp.w1.weight", "model.layers.{i}.mlp.gate_proj.weight"),
    ("blocks.{i}.mlp.w3.weight", "model.layers.{i}.mlp.up_proj.weight"),
    ("blocks.{i}.mlp.w2.weight", "model.layers.{i}.mlp.down_proj.weight"),
]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|user|>' + message['content'] + '<|assistant|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + '<|end|>' }}"
    "{% endif %}{% endfor %}"
)


def to_hf_state_dict(model: HotsteelLM) -> dict:
    """把我方 state_dict 映射为 HF Llama 键名。"""
    sd = model.state_dict()
    out: dict = {}
    out["model.embed_tokens.weight"] = sd["tok_emb.weight"]
    out["model.norm.weight"] = sd["norm_f.weight"]
    if not model.cfg.tie_embeddings:
        out["lm_head.weight"] = sd["lm_head.weight"]
    for i in range(model.cfg.n_layer):
        for src, dst in _LAYER_MAP:
            out[dst.format(i=i)] = sd[src.format(i=i)]
    return out


def write_config_json(model: HotsteelLM, path: str) -> None:
    c = model.cfg
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": c.vocab_size,
        "hidden_size": c.d_model,
        "intermediate_size": c.d_ff,
        "num_hidden_layers": c.n_layer,
        "num_attention_heads": c.n_head,
        "num_key_value_heads": c.n_head,
        "hidden_act": "silu",
        "max_position_embeddings": c.max_seq_len,
        "initializer_range": 0.02,
        "rms_norm_eps": c.rms_norm_eps,
        "rope_theta": c.rope_theta,
        "tie_word_embeddings": c.tie_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": "float32",
        "use_cache": True,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)


def write_tokenizer_files(tokenizer_dir: str, out_dir: str) -> None:
    """复制 tokenizer.json 并补充 tokenizer_config / special_tokens_map。"""
    src = os.path.join(tokenizer_dir, TOKENIZER_FILE)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"找不到分词器: {src}")
    shutil.copyfile(src, os.path.join(out_dir, TOKENIZER_FILE))

    specials = [PAD, USER, ASSISTANT, END, UNK]
    tok_cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 4096,
        "eos_token": END,
        "bos_token": USER,
        "pad_token": PAD,
        "unk_token": UNK,
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "chat_template": CHAT_TEMPLATE,
        "additional_special_tokens": specials,
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        json.dump(tok_cfg, fh, ensure_ascii=False, indent=2)

    special_map = {
        "eos_token": END,
        "bos_token": USER,
        "pad_token": PAD,
        "unk_token": UNK,
        "additional_special_tokens": specials,
    }
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as fh:
        json.dump(special_map, fh, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="导出 HF Llama 格式（供 GGUF 转换）")
    ap.add_argument("--config", default="configs/hotsteel_100m.yaml")
    ap.add_argument("--ckpt", default="checkpoints_100m/final.pt")
    ap.add_argument("--out", default="export/hotsteel0.1-100m-hf")
    args = ap.parse_args()

    from safetensors.torch import save_file

    cfg = load_config(args.config)
    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"找不到权重: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    mcfg = ModelConfig(**ckpt["model_config"])
    model = HotsteelLM(mcfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    os.makedirs(args.out, exist_ok=True)

    sd = to_hf_state_dict(model)
    # safetensors 要求张量连续且为 fp32/fp16/bf16；统一转 float32 保证兼容
    sd = {k: v.contiguous().to(torch.float32) for k, v in sd.items()}
    save_file(sd, os.path.join(args.out, "model.safetensors"))

    write_config_json(model, os.path.join(args.out, "config.json"))
    write_tokenizer_files(cfg.paths.tokenizer_dir, args.out)

    n = sum(v.numel() for v in sd.values())
    print(f"已导出 HF 格式 -> {args.out}")
    print(f"  参数量: {n:,} ({n/1e6:.1f}M)")
    print(f"  文件: config.json, model.safetensors, tokenizer.json, "
          f"tokenizer_config.json, special_tokens_map.json")


if __name__ == "__main__":
    main()
