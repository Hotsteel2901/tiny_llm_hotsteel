"""数据集构建：把 jsonl 问答语料编码为因果 LM 训练样本。

两种策略：
1. **逐样本切齐（默认，推荐）**：每个问答对单独编码、补齐到 ``max_seq_len``，
   并用 ``-100`` 屏蔽填充与跨样本边界。模型只在「同一条样例内部」学习，
   不会把上一条回答的结尾与下一条提问粘连起来。
2. **拼接打包（可选）**：把所有样例首尾相接再切块，token 利用率高，
   但样本间会发生跨边界污染（易导致答案串题）。

默认采用策略 1，以获得更干净的对齐与更连贯的回答。
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import HotsteelTokenizer


def load_corpus(path: str) -> List[Dict[str, str]]:
    """读取 jsonl 语料，返回 ``[{"instruction":..., "response":...}, ...]``。"""
    items: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 解析失败: {exc}") from exc
            inst = str(obj.get("instruction", "")).strip()
            resp = str(obj.get("response", "")).strip()
            if inst and resp:
                items.append({"instruction": inst, "response": resp})
    if not items:
        raise ValueError("语料为空或格式不正确: " + path)
    return items


class ExampleLMDataset(Dataset):
    """逐样本因果 LM 数据集。

    - 每个样例单独编码为 ``<|user|>问<|assistant|>答<|end|>``
    - 补齐到 ``max_seq_len + 1``（多一位用于标签错位）
    - ``labels`` 中，pad 位置与序列头部（提问部分）置为 -100：
      只对「回答」部分计算损失，让模型专注学会如何作答。
    """

    def __init__(
        self,
        items: List[Dict[str, str]],
        tokenizer: HotsteelTokenizer,
        max_seq_len: int,
        mask_instruction: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_instruction = mask_instruction
        self.samples: List[Tuple[List[int], int]] = []

        for it in items:
            ids = tokenizer.build_example(it["instruction"], it["response"])
            if len(ids) < 2:
                continue
            # 截断：保留 <|end|> 结尾以保证有完整回答
            if len(ids) > max_seq_len:
                ids = ids[: max_seq_len - 1] + [tokenizer.end_id]
            # 记录「回答起始位置」：<|assistant|> 之后
            ans_start = len(ids)
            for idx, t in enumerate(ids):
                if t == tokenizer.assistant_id:
                    ans_start = idx + 1
                    break
            self.samples.append((ids, ans_start))

        if not self.samples:
            raise ValueError("没有可用样本（语料过短或 max_seq_len 太小）")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids, ans_start = self.samples[idx]
        L = self.max_seq_len

        x = ids[:-1]
        y = ids[1:]
        # 标签中，提问部分（含 <|assistant|>）不计损失
        if self.mask_instruction:
            cut = max(0, min(ans_start - 1, len(y)))
            y = [-100] * cut + y[cut:]

        x = x[:L]
        y = y[:L]
        # 不在此处补齐；交给 collate_batch 按 batch 内最长序列动态 padding
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0):
    """按 batch 内最长序列动态 padding，显著减少无效计算。"""
    max_len = max(item["input_ids"].numel() for item in batch)
    xs, ys = [], []
    for item in batch:
        x, y = item["input_ids"], item["labels"]
        pad = max_len - x.numel()
        if pad > 0:
            x = torch.cat([x, torch.full((pad,), pad_id, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad,), -100, dtype=torch.long)])
        xs.append(x)
        ys.append(y)
    return {"input_ids": torch.stack(xs), "labels": torch.stack(ys)}


def pack_tokens(
    items: List[Dict[str, str]],
    tokenizer: HotsteelTokenizer,
    max_seq_len: int,
    repeat_until_chunks: int = 0,
    add_eos_between: bool = True,
) -> torch.Tensor:
    """把所有样例编码并拼接成一个长 id 序列（备用的打包策略）。

    用 ``array.array`` 累积而不是 Python list：大语料（数千万 token）下
    list 每个元素要 ~28 字节对象开销，array 只要 8 字节，内存差 4 倍以上。
    """
    from array import array

    stream: "array[int]" = array("q")
    for item in items:
        stream.extend(tokenizer.build_example(item["instruction"], item["response"]))
        if add_eos_between:
            stream.append(tokenizer.end_id)

    if repeat_until_chunks > 0:
        need = repeat_until_chunks * max_seq_len + 1
        rounds = 0
        while len(stream) < need and rounds < 100 and items:
            for item in items:
                stream.extend(tokenizer.build_example(item["instruction"], item["response"]))
                if add_eos_between:
                    stream.append(tokenizer.end_id)
            rounds += 1

    return torch.frombuffer(stream, dtype=torch.int64).clone()


class PackedLMDataset(Dataset):
    """拼接打包数据集（备用策略，样本间无边界屏蔽）。"""

    def __init__(self, tokens: torch.Tensor, max_seq_len: int):
        self.max_seq_len = max_seq_len
        n_tokens = tokens.numel()
        n_chunks = (n_tokens - 1) // max_seq_len
        if n_chunks <= 0:
            raise ValueError(
                f"语料 token 数 ({n_tokens}) 少于一个序列长度 ({max_seq_len})，"
                "请增大语料或减小 max_seq_len"
            )
        usable = n_chunks * max_seq_len
        self.tokens = tokens[: usable + 1]

    def __len__(self) -> int:
        return (self.tokens.numel() - 1) // self.max_seq_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        L = self.max_seq_len
        start = idx * L
        x = self.tokens[start : start + L]
        y = self.tokens[start + 1 : start + 1 + L]
        return {"input_ids": x, "labels": y}


def build_datasets(
    corpus_path: str,
    tokenizer: HotsteelTokenizer,
    max_seq_len: int,
    val_ratio: float,
    seed: int = 1337,
    packing: bool = False,
    mask_instruction: bool = True,
):
    """返回 ``(train_ds, val_ds, stats)``。

    ``packing=False``（默认）使用逐样本对齐数据集，回答更连贯；
    ``packing=True`` 使用拼接打包，token 利用率更高。
    """
    items = load_corpus(corpus_path)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(items), generator=g).tolist()
    n_val = max(1, int(len(items) * val_ratio))
    val_items = [items[i] for i in perm[:n_val]]
    train_items = [items[i] for i in perm[n_val:]]

    if packing:
        train_tokens = pack_tokens(train_items, tokenizer, max_seq_len)
        val_tokens = pack_tokens(val_items, tokenizer, max_seq_len, repeat_until_chunks=2)
        train_ds = PackedLMDataset(train_tokens, max_seq_len)
        val_ds = PackedLMDataset(val_tokens, max_seq_len)
        train_tokens_n = int(train_tokens.numel())
        val_tokens_n = int(val_tokens.numel())
    else:
        train_ds = ExampleLMDataset(train_items, tokenizer, max_seq_len, mask_instruction)
        val_ds = ExampleLMDataset(val_items, tokenizer, max_seq_len, mask_instruction)
        train_tokens_n = sum(len(s[0]) for s in train_ds.samples)
        val_tokens_n = sum(len(s[0]) for s in val_ds.samples)

    stats = {
        "n_examples": len(items),
        "n_train_examples": len(train_items),
        "n_val_examples": len(val_items),
        "train_tokens": train_tokens_n,
        "val_tokens": val_tokens_n,
        "train_chunks": len(train_ds),
        "val_chunks": len(val_ds),
        "packing": packing,
    }
    return train_ds, val_ds, stats
