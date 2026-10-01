"""文本生成：贪婪 / 温度 / top-k / top-p 采样。"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def _apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """对已生成的 token 施加重复惩罚。"""
    if penalty == 1.0:
        return logits
    for b in range(logits.size(0)):
        uniq = torch.unique(generated[b])
        logits[b, uniq] = torch.where(
            logits[b, uniq] > 0,
            logits[b, uniq] / penalty,
            logits[b, uniq] * penalty,
        )
    return logits


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0 or p <= 0.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    mask = cumulative - probs > p  # 保留累计概率首次超过 p 的最小集合
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    greedy: bool = False,
    stop_ids: Optional[List[int]] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """自回归生成，返回包含前缀的完整 id 序列 (B, T+n)。

    ``stop_ids`` 中的任意 token 出现即停止生成。
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    stop_ids = stop_ids or []
    if eos_id is not None:
        stop_ids = list(stop_ids) + [eos_id]
    stop_set = set(stop_ids)

    cur = input_ids
    max_ctx = getattr(model.cfg, "max_seq_len", 1024)

    for _ in range(max_new_tokens):
        ctx = cur[:, -max_ctx:]
        logits = model(ctx)
        next_logits = logits[:, -1, :].float()

        if repetition_penalty and repetition_penalty != 1.0:
            next_logits = _apply_repetition_penalty(next_logits, cur, repetition_penalty)

        if greedy or temperature <= 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            next_logits = _top_k_filter(next_logits, top_k)
            next_logits = _top_p_filter(next_logits, top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        cur = torch.cat([cur, next_token], dim=1)
        if stop_set and next_token.item() in stop_set:
            break

    return cur


@torch.no_grad()
def chat_once(
    model,
    tokenizer,
    instruction: str,
    gen_cfg,
    device: torch.device | None = None,
    history: Optional[List[List[int]]] = None,
) -> str:
    """单轮问答：构造 prompt → 生成 → 解码回答。"""
    device = device or next(model.parameters()).device
    if history:
        ids = list(history[-1]) if history else []
    else:
        ids = tokenizer.build_prompt(instruction)

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model,
        input_ids,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_k=gen_cfg.top_k,
        top_p=gen_cfg.top_p,
        repetition_penalty=gen_cfg.repetition_penalty,
        greedy=gen_cfg.greedy,
        eos_id=tokenizer.end_id,
    )
    new_ids = out[0, len(ids):].tolist()
    # 截断到 <|end|>
    if tokenizer.end_id in new_ids:
        new_ids = new_ids[: new_ids.index(tokenizer.end_id)]
    return tokenizer.decode(new_ids)
