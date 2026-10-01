"""通用工具：随机种子、设备选择、日志。"""
from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定所有随机源，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def pick_device(pref: str = "auto") -> torch.device:
    """按偏好自动选择设备。"""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(pref: str, device: torch.device) -> torch.dtype:
    """选择训练精度。CPU/不支持时回落 float32。"""
    if pref == "float32" or device.type == "cpu":
        return torch.float32
    if pref == "bfloat16":
        return torch.bfloat16
    if pref == "float16":
        return torch.float16
    # auto
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def get_logger(name: str = "hotsteel") -> logging.Logger:
    """构建统一格式的日志器。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
