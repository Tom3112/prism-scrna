"""
Self-supervised pretraining with Masked Gene Prediction (MGP).
"""

from __future__ import annotations

import math
import os
import random
import time

import anndata as ad
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import load_datasets
from model.transformer import scRNAEncoder
from model.gnn import build_gene_gat
from model.heads import MaskedGenePredictionHead
from train.config import DataConfig, ModelConfig, PretrainConfig


# ---------------------------------------------------------------------------
# Scheduler helper
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Masked prediction accuracy (token-level, masked positions only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == labels[mask]).float().sum().item()
    return correct / mask.sum().item()


# ---------------------------------------------------------------------------
# Training and validation steps
# ---------------------------------------------------------------------------

def train_epoch(
    encoder: scRNAEncoder,
    head: MaskedGenePredictionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    encoder.train()
    head.train()
    total_loss = 0.0
    total_acc = 0.0

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        bin_ids = batch["bin_ids"].to(device) if "bin_ids" in batch else None

        hidden = encoder(input_ids, attention_mask, bin_ids)
        loss, logits = head(hidden, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(head.parameters()), grad_clip
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += masked_accuracy(logits, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def val_epoch(
    encoder: scRNAEncoder,
    head: MaskedGenePredictionHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    encoder.eval()
    head.eval()
    total_loss = 0.0
    total_acc = 0.0

    for batch in tqdm(loader, desc="  val  ", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        bin_ids = batch["bin_ids"].to(device) if "bin_ids" in batch else None

        hidden = encoder(input_ids, attention_mask, bin_ids)
        loss, logits = head(hidden, labels)

        total_loss += loss.item()
        total_acc += masked_accuracy(logits, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main pretraining loop
# ---------------------------------------------------------------------------

def pretrain(
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
    cfg: PretrainConfig | None = None,
) -> scRNAEncoder:
    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    cfg = cfg or PretrainConfig()

    # Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(
        cfg.device if cfg.device != "cpu" else
        ("cuda" if torch.cuda.is_available() else
         "mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Datasets
    print("Loading datasets...")
    splits = load_datasets(
        data_cfg.processed_path,
        max_seq_len=data_cfg.max_seq_len,
        mask_ratio=data_cfg.mask_ratio,
        train_frac=data_cfg.train_frac,
        val_frac=data_cfg.val_frac,
        seed=data_cfg.seed,
        tokenization=data_cfg.tokenization,
        n_bins=data_cfg.n_bins,
    )
    vocab_size = splits["vocab_size"]
    print(f"Vocab size: {vocab_size}  |  Train cells: {len(splits['pretrain_train'])}  |  Val cells: {len(splits['pretrain_val'])}")

    train_loader = DataLoader(
        splits["pretrain_train"],
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        splits["pretrain_val"],
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Model
    gene_gat = None
    if model_cfg.use_gnn:
        print("Building PPI graph for GeneGAT gene embeddings...")
        tmp_adata = ad.read_h5ad(data_cfg.processed_path)
        gene_names = list(tmp_adata.var_names)
        del tmp_adata
        gene_gat = build_gene_gat(gene_names, model_cfg, data_cfg.processed_path, device)

    encoder = scRNAEncoder(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        ffn_dim=model_cfg.ffn_dim,
        dropout=model_cfg.dropout,
        max_seq_len=data_cfg.max_seq_len,
        n_expr_bins=(data_cfg.n_bins if data_cfg.tokenization == "expr_bin" else None),
        gene_gat=gene_gat,
    ).to(device)
    # expr_bin mode predicts the masked expression bin (0..n_bins); gene identity
    # is always visible (fixed panel), so predicting it would be trivial from
    # position alone. rank mode predicts the masked gene identity as usual.
    n_pred_classes = (data_cfg.n_bins + 1) if data_cfg.tokenization == "expr_bin" else vocab_size
    head = MaskedGenePredictionHead(model_cfg.hidden_dim, n_pred_classes).to(device)

    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimiser & scheduler
    optimizer = AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            encoder, head, train_loader, optimizer, scheduler, device, cfg.grad_clip
        )
        val_loss, val_acc = val_epoch(encoder, head, val_loader, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state": encoder.state_dict(),
                    "head_state": head.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_cfg": model_cfg.__dict__,
                    "data_cfg": data_cfg.__dict__,
                    "vocab_size": vocab_size,
                },
                cfg.best_ckpt,
            )
            print(f"  → saved best checkpoint (val_loss={best_val_loss:.4f})")

    # Save training history alongside checkpoint
    history_path = os.path.join(cfg.checkpoint_dir, "pretrain_history.npy")
    np.save(history_path, history)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return encoder


if __name__ == "__main__":
    pretrain()
