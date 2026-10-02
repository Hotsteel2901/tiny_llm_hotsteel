"""hotsteel0.1 训练脚本。

流程：加载配置 → 构建数据/模型 → 训练循环（AMP + 梯度累积 +
cosine 学习率 + 梯度裁剪）→ 定期评估/保存。

运行::

    python scripts/train.py --config configs/small.yaml
    python scripts/train.py --config configs/small.yaml --max_iters 300   # 冒烟测试
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from hotsteel.config import load_config, save_config
from hotsteel.config import to_plain
from hotsteel.dataset import build_datasets
from hotsteel.model import HotsteelLM, ModelConfig
from hotsteel.tokenizer import HotsteelTokenizer
from hotsteel.utils import (
    count_parameters,
    get_logger,
    human_params,
    pick_device,
    pick_dtype,
    set_seed,
)

logger = get_logger("train")


def get_lr(iter_num: int, cfg) -> float:
    """线性 warmup + cosine 衰减。"""
    if iter_num < cfg.warmup_iters:
        return cfg.lr * (iter_num + 1) / max(1, cfg.warmup_iters)
    if iter_num > cfg.max_iters:
        return cfg.min_lr
    ratio = (iter_num - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model, val_loader, cfg, device, ctx) -> float:
    """在验证集上估计平均 loss。"""
    model.eval()
    losses = []
    it = iter(val_loader)
    for _ in range(cfg.eval_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(val_loader)
            try:
                batch = next(it)
            except StopIteration:
                break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    if not losses:
        return float("inf")
    return sum(losses) / len(losses)


def save_checkpoint(model, cfg, path: str, extra: dict | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "model_config": model.cfg.__dict__,
        "config": to_plain(cfg),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def build_model_config(cfg, vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=cfg.model.d_model,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        rope_theta=cfg.model.rope_theta,
        rms_norm_eps=cfg.model.get("rms_norm_eps", 1e-6),
        max_seq_len=cfg.data.max_seq_len,
        tie_embeddings=cfg.model.tie_embeddings,
        use_swiglu=cfg.model.use_swiglu,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 hotsteel0.1")
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--max_iters", type=int, default=None, help="覆盖训练步数")
    ap.add_argument("--resume", default=None, help="从 checkpoint 继续训练")
    ap.add_argument("--start_iter", type=int, default=0,
                    help="起始迭代步（分段训练用：恢复权重后从该步继续，"
                         "LR 调度仍按 --max_iters 总长计算，保证多段拼接后曲线连续）")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_iters is not None:
        cfg.train.max_iters = args.max_iters
        cfg.train.warmup_iters = min(cfg.train.warmup_iters, max(1, args.max_iters // 10))
    if args.device:
        cfg.train.device = args.device

    set_seed(cfg.project.seed)
    device = pick_device(cfg.train.device)
    dtype = pick_dtype(cfg.train.dtype, device)
    logger.info(f"设备={device}  精度={dtype}  种子={cfg.project.seed}")

    # --- 分词器 ---
    tokenizer = HotsteelTokenizer.load(cfg.paths.tokenizer_dir)
    logger.info(f"词表大小={tokenizer.vocab_size}")

    # --- 数据 ---
    train_ds, val_ds, stats = build_datasets(
        corpus_path=cfg.paths.corpus,
        tokenizer=tokenizer,
        max_seq_len=cfg.data.max_seq_len,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.project.seed,
        packing=bool(cfg.data.get("packing", False)),
        mask_instruction=bool(cfg.data.get("mask_instruction", True)),
    )
    logger.info(
        f"样本: 训练={stats['n_train_examples']} 验证={stats['n_val_examples']} | "
        f"tokens: 训练={stats['train_tokens']} 验证={stats['val_tokens']} | "
        f"序列数: 训练={stats['train_chunks']} 验证={stats['val_chunks']}"
    )

    from functools import partial
    from hotsteel.dataset import collate_batch

    collate = partial(collate_batch, pad_id=tokenizer.pad_id)
    train_bs = max(1, min(cfg.train.batch_size, len(train_ds)))
    if train_bs != cfg.train.batch_size:
        logger.info(
            f"训练集较小({len(train_ds)} 样本)，batch_size 由 {cfg.train.batch_size} 调整为 {train_bs}"
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=False,
        collate_fn=collate,
    )
    val_bs = max(1, min(cfg.train.batch_size, len(val_ds)))
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=False,
        collate_fn=collate,
    )

    # --- 模型 ---
    mcfg = build_model_config(cfg, tokenizer.vocab_size)
    model = HotsteelLM(mcfg).to(device)
    n_params = count_parameters(model)
    logger.info(f"模型参数量={human_params(n_params)} ({n_params:,})")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        logger.info(f"已从 {args.resume} 恢复权重")

    if cfg.train.compile and hasattr(torch, "compile"):
        logger.info("启用 torch.compile")
        model = torch.compile(model)

    # --- 优化器 ---
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 else decay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.train.lr,
        betas=tuple(cfg.train.betas),
    )

    use_amp = dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else torch.autocast(device_type="cpu", enabled=False)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.paths.checkpoint_dir, "train_config.yaml"))

    # --- 训练循环 ---
    model.train()
    t0 = time.time()
    tokens_per_iter = cfg.train.batch_size * cfg.data.max_seq_len * cfg.train.grad_accum_steps
    best_val = float("inf")
    data_iter = iter(train_loader)

    for it in range(args.start_iter, cfg.train.max_iters + 1):
        lr = get_lr(it, cfg.train)
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.train.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.train.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        if cfg.train.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optim)
        scaler.update()

        if it % cfg.train.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_s = tokens_per_iter * cfg.train.log_interval / max(dt, 1e-6)
            logger.info(
                f"iter {it:5d}/{cfg.train.max_iters} | loss {accum_loss:.4f} | "
                f"lr {lr:.2e} | {tok_s:,.0f} tok/s"
            )

        if it > 0 and it % cfg.train.eval_interval == 0:
            val_loss = evaluate(model, val_loader, cfg.train, device, ctx)
            ppl = math.exp(min(val_loss, 20))
            logger.info(f"  [eval] iter {it} | val_loss {val_loss:.4f} | ppl {ppl:.2f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model, cfg,
                    os.path.join(cfg.paths.checkpoint_dir, "best.pt"),
                    extra={"iter": it, "val_loss": val_loss},
                )
            model.train()

        if it > 0 and it % cfg.train.save_interval == 0:
            save_checkpoint(
                model, cfg,
                os.path.join(cfg.paths.checkpoint_dir, f"ckpt_{it}.pt"),
                extra={"iter": it},
            )
            logger.info(f"  已保存 checkpoint ckpt_{it}.pt")

    # --- 收尾保存 ---
    save_checkpoint(
        model, cfg,
        os.path.join(cfg.paths.checkpoint_dir, "final.pt"),
        extra={"iter": cfg.train.max_iters},
    )
    logger.info(f"训练完成，最终权重 -> {cfg.paths.checkpoint_dir}/final.pt")


if __name__ == "__main__":
    main()
