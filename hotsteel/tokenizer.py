"""ByteLevel BPE 分词器：训练与加载。

用 HuggingFace ``tokenizers`` 训练一个 byte-level BPE，
对中文友好且不依赖额外词表文件。
"""
from __future__ import annotations

import os
from typing import List

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# 特殊 token 语义
PAD = "<pad>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
UNK = "<unk>"

DEFAULT_SPECIALS = [PAD, USER, ASSISTANT, END, UNK]

TOKENIZER_FILE = "tokenizer.json"


def train_tokenizer(
    corpus_path: str,
    out_dir: str,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    special_tokens: List[str] | None = None,
) -> str:
    """在语料文本上训练 BPE 并保存到 ``out_dir/tokenizer.json``。

    返回保存路径。
    """
    special_tokens = special_tokens or DEFAULT_SPECIALS
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # 以流式迭代器喂入文本，避免一次性载入大文件
    def text_iter():
        with open(corpus_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line

    tokenizer.train_from_iterator(text_iter(), trainer=trainer)
    save_path = os.path.join(out_dir, TOKENIZER_FILE)
    tokenizer.save(save_path)
    return save_path


class HotsteelTokenizer:
    """封装特殊 token 的编解码便利方法。"""

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.pad_id = tokenizer.token_to_id(PAD)
        self.user_id = tokenizer.token_to_id(USER)
        self.assistant_id = tokenizer.token_to_id(ASSISTANT)
        self.end_id = tokenizer.token_to_id(END)
        self.unk_id = tokenizer.token_to_id(UNK)
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def load(cls, tokenizer_dir: str) -> "HotsteelTokenizer":
        path = os.path.join(tokenizer_dir, TOKENIZER_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到分词器 {path}，请先运行 scripts/train_tokenizer.py"
            )
        return cls(Tokenizer.from_file(path))

    # ---- 编码 ----

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_ids(self, text: str) -> List[int]:
        """纯文本 → id（不带任何特殊 token）。"""
        return self._tok.encode(text, add_special_tokens=False).ids

    def build_example(self, instruction: str, response: str) -> List[int]:
        """构造训练序列：<|user|>问<|assistant|>答<|end|>。"""
        ids: List[int] = [self.user_id]
        ids += self.encode_ids(instruction)
        ids.append(self.assistant_id)
        ids += self.encode_ids(response)
        ids.append(self.end_id)
        return ids

    def build_prompt(self, instruction: str) -> List[int]:
        """构造推理前缀：<|user|>问<|assistant|>（等待模型续写）。"""
        ids: List[int] = [self.user_id]
        ids += self.encode_ids(instruction)
        ids.append(self.assistant_id)
        return ids

    # ---- 解码 ----

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            specials = {self.pad_id, self.user_id, self.assistant_id, self.end_id}
            ids = [i for i in ids if i not in specials]
        return self._tok.decode(ids, skip_special_tokens=False)
