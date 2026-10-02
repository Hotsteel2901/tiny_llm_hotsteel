"""热钢0.1 Transformer 语言模型。

结构要点：
- Decoder-only 因果语言模型（类似 GPT）
- Pre-Norm + RMSNorm，训练更稳定
- RoPE 旋转位置编码，无需学习位置 embedding
- SwiGLU 前馈（可选）
- 权重共享（输入 embedding 与输出投影）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10000.0
    max_seq_len: int = 256
    tie_embeddings: bool = True
    use_swiglu: bool = True
    rms_norm_eps: float = 1e-6
    # 供 HF / GGUF 导出记录（不影响训练）
    model_type: str = "hotsteel"

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model 必须能被 n_head 整除")


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm（与 Llama 一致，eps 默认 1e-6）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(x.dtype) * self.weight


def precompute_rope(
    head_dim: int, max_seq_len: int, theta: float, device=None, dtype=torch.float32
):
    """预计算 RoPE 的 cos / sin 表，形状 (max_seq_len, head_dim)。

    严格遵循 **HuggingFace Llama** 的约定（half-split / rotate_half），
    以保证导出的 GGUF 在 llama.cpp 中结果正确。
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                 # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)          # (T, head_dim) —— 复制为前后两半
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Llama 风格旋转：前半与后半交换并取负。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 (B, H, T, D) 的 q/k 施加 Llama 标准 RoPE。"""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """多头因果自注意力（可选 KV 无关，仅训练用）。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y))


class MLP(nn.Module):
    """前馈层：SwiGLU 或 GELU。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.use_swiglu = cfg.use_swiglu
        if cfg.use_swiglu:
            # SwiGLU: 下投影两路，其中一路做门控
            hidden = cfg.d_ff
            self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
            self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
            self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)
        else:
            self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
            self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        else:
            x = self.proj(self.act(self.fc(x)))
        return self.dropout(x)


class Block(nn.Module):
    """Pre-Norm Transformer 块。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class HotsteelLM(nn.Module):
    """热钢0.1 语言模型主体。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(
            cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 残差分支缩放，缓解深层训练不稳
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight") or name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """前向计算；给定 labels 时返回 (logits, loss)，否则返回 logits。"""
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"序列长度 {T} 超过配置上限 {self.cfg.max_seq_len}"
            )
        x = self.drop(self.tok_emb(input_ids))
        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # 延迟导入，避免环形依赖
        from .generate import generate as _generate
        return _generate(self, *args, **kwargs)
