"""配置加载与工具函数。

用法::

    from hotsteel.config import load_config
    cfg = load_config("configs/small.yaml")
    cfg.model.d_model  # 属性式访问
"""
from __future__ import annotations

import os
from typing import Any

import yaml


class Config(dict):
    """支持属性访问的嵌套字典配置。"""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(f"配置项不存在: {name}") from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, *keys: str, default: Any = None) -> Any:
        node: Any = self
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def load_config(path: str) -> Config:
    """从 YAML 文件加载配置。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw)


def to_plain(obj: Any) -> Any:
    """递归把 Config 转成普通 dict，便于 yaml/json 序列化。"""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def save_config(cfg: Config, path: str) -> None:
    """把配置写回 YAML（用于把本次训练实际参数存档）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_plain(cfg), fh, allow_unicode=True, sort_keys=False)
