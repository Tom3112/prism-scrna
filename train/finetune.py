"""
Supervised fine-tuning for cell-type classification.

Loads the pretrained encoder, attaches a classification head, and trains
on 80% of labelled cells.
"""

from __future__ import annotations

import math
import os
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import anndata as ad

from data.dataset import load_datasets
from model.transformer import scRNAEncoder
from model.gnn import build_gene_gat
from model.heads import CellTypeClassificationHead, CellGATClassificationHead
from model.gene_graph import build_gene_graph
from train.config import DataConfig, ModelConfig, FinetuneConfig
from train.pretrain import get_cosine_schedule_with_warmup
from eval.metrics import _forward_head, evaluate


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Encoder freezing utilities
# ---------------------------------------------------------------------------

def apply_freeze(encoder: scRNAEncoder, strategy: str, n_freeze_layers: int) -> None:
    if strategy == "none":
        return
    if strategy == "full":
        for p in encoder.parameters():
            p.requires_grad = False
        return
    if strategy == "partial":
        # Freeze embeddings and first n_freeze_layers transformer layers
        if encoder.token_emb is not None:
            for p in encoder.token_emb.parameters():
                p.requires_grad = False
        for p in encoder.pos_emb.parameters():
            p.requires_grad = False
        for i, layer in enumerate(encoder.encoder.layers):
            if i < n_freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        return
    raise ValueError(f"Unknown freeze strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Train / val step
# ---------------------------------------------------------------------------

def train_epoch(
    encoder: scRNAEncoder,
    head,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    grad_clip: float,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, float]:
    encoder.train()
    head.train()
    total_loss, total_acc = 0.0, 0.0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cell_type      = batch["cell_type"].to(device)
        bin_ids        = batch["bin_ids"].to(device) if "bin_ids" in batch else None

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            loss, logits = _forward_head(encoder, head, input_ids, attention_mask, cell_type, bin_ids)

        params = list(encoder.parameters()) + list(head.parameters())
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc  += batch_accuracy(logits, cell_type)

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def val_epoch(
    encoder: scRNAEncoder,
    head,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> tuple[float, float]:
    encoder.eval()
    head.eval()
    total_loss, total_acc = 0.0, 0.0

    for batch in tqdm(loader, desc="  val  ", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cell_type      = batch["cell_type"].to(device)
        bin_ids        = batch["bin_ids"].to(device) if "bin_ids" in batch else None

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            loss, logits = _forward_head(encoder, head, input_ids, attention_mask, cell_type, bin_ids)

        total_loss += loss.item()
        total_acc  += batch_accuracy(logits, cell_type)

    n = len(loader)
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
    cfg: FinetuneConfig | None = None,
) -> tuple[scRNAEncoder, CellTypeClassificationHead]:
    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    cfg = cfg or FinetuneConfig()

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
    num_classes = splits["num_classes"]
    label_names = splits["label_names"]
    print(f"Classes ({num_classes}): {label_names}")

    train_loader = DataLoader(
        splits["finetune_train"], batch_size=cfg.batch_size, shuffle=True,
        num_workers=data_cfg.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        splits["finetune_val"], batch_size=cfg.batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        splits["finetune_test"], batch_size=cfg.batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=device.type == "cuda",
    )

    # Build encoder
    gene_gat = None
    if model_cfg.use_gnn:
        print("Building PPI graph for GeneGAT gene embeddings...")
        gene_gat = build_gene_gat(model_cfg, data_cfg.processed_path, device)

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

    # Load pretrained weights if checkpoint exists
    if os.path.exists(cfg.pretrain_ckpt):
        ckpt = torch.load(cfg.pretrain_ckpt, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])
        print(f"Loaded pretrained encoder from {cfg.pretrain_ckpt}")
    else:
        print("WARNING: no pretrain checkpoint found — training from scratch.")

    apply_freeze(encoder, cfg.freeze_encoder, cfg.freeze_layers)

    if model_cfg.use_gat_head:
        print("Building PPI graph for GAT classification head...")
        tmp_adata  = ad.read_h5ad(data_cfg.processed_path)
        gene_names = list(tmp_adata.var_names)
        del tmp_adata
        edge_index, edge_weight = build_gene_graph(
            gene_names,
            cache_dir=os.path.dirname(data_cfg.processed_path),
            min_score=model_cfg.string_min_score,
            processed_path=data_cfg.processed_path,
        )
        head = CellGATClassificationHead(
            hidden_dim=model_cfg.hidden_dim,
            num_classes=num_classes,
            dropout=model_cfg.dropout,
            n_genes=vocab_size - 3,
            ppi_edge_index=edge_index.to(device),
            ppi_edge_weight=edge_weight.to(device),
            n_gat_heads=model_cfg.gat_head_n_heads,
        ).to(device)
        head_type = "CellGAT"
    else:
        head = CellTypeClassificationHead(
            hidden_dim=model_cfg.hidden_dim,
            num_classes=num_classes,
            dropout=model_cfg.dropout,
        ).to(device)
        head_type = "CLS"

    print(f"Classification head: {head_type}")

    # Only pass trainable parameters to the optimiser
    trainable = [p for p in list(encoder.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(encoder, head, train_loader, optimizer, scheduler, device, cfg.grad_clip, scaler)
        val_loss, val_acc = val_epoch(encoder, head, val_loader, device, amp_enabled=scaler.is_enabled())
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state": encoder.state_dict(),
                    "head_state": head.state_dict(),
                    "val_acc": val_acc,
                    "label_names": label_names,
                    "model_cfg": model_cfg.__dict__,
                    "vocab_size": vocab_size,
                },
                cfg.best_ckpt,
            )
            print(f"  → saved best checkpoint (val_acc={best_val_acc:.4f})")

    # Final test evaluation
    print("\nEvaluating on test set with best checkpoint...")
    ckpt = torch.load(cfg.best_ckpt, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state"])
    head.load_state_dict(ckpt["head_state"])
    test_metrics = evaluate(encoder, head, test_loader, device, label_names)

    history_path = os.path.join(cfg.checkpoint_dir, "finetune_history.npy")
    np.save(history_path, history)
    return encoder, head, test_metrics


if __name__ == "__main__":
    finetune()
