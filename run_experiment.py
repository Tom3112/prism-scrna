"""
Full pipeline runner: download → preprocess → pretrain → finetune → report.

Usage:
    uv run python run_experiment.py [--run-name NAME]

All artifacts (logs, checkpoints, report) are saved to experiments/<run_name>/.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import asdict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))


def setup_run_dir(run_name: str) -> str:
    run_dir = os.path.join(ROOT, "experiments", run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Tee:
    """Write to both stdout and a log file simultaneously."""

    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1)

    def write(self, msg: str):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "Apple Silicon",
        "python": sys.version.split()[0],
    }
    info["torch"] = torch.__version__
    if torch.backends.mps.is_available():
        info["accelerator"] = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        info["accelerator"] = torch.cuda.get_device_name(0)
    else:
        info["accelerator"] = "CPU"
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "Chip" in line or "Memory" in line or "Cores" in line:
                key, _, val = line.strip().partition(": ")
                info[key.strip()] = val.strip()
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Data steps
# ---------------------------------------------------------------------------

def run_download(run_dir: str) -> float:
    print("\n" + "="*60)
    print("STEP 1: Download data")
    print("="*60)
    processed_path = os.path.join(ROOT, "data", "processed.h5ad")
    raw_path = os.path.join(ROOT, "data", "raw.h5ad")
    if os.path.exists(processed_path):
        print("processed.h5ad already exists — skipping download & preprocess.")
        return 0.0
    t0 = time.time()
    if not os.path.exists(raw_path):
        sys.path.insert(0, ROOT)
        from data.download import download_pbmc3k
        download_pbmc3k()
    return time.time() - t0


def run_preprocess(run_dir: str) -> float:
    processed_path = os.path.join(ROOT, "data", "processed.h5ad")
    if os.path.exists(processed_path):
        return 0.0
    print("\n" + "="*60)
    print("STEP 2: Preprocess")
    print("="*60)
    t0 = time.time()
    sys.path.insert(0, ROOT)
    from data.preprocess import preprocess
    preprocess()
    return time.time() - t0


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def run_pretrain(run_dir: str, model_cfg, data_cfg) -> tuple[float, dict]:
    print("\n" + "="*60)
    print("STEP 3: Pretraining")
    print("="*60)
    from train.config import PretrainConfig
    from train.pretrain import pretrain

    cfg = PretrainConfig(
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        best_ckpt=os.path.join(run_dir, "checkpoints", "pretrain_best.pt"),
    )

    t0 = time.time()
    pretrain(data_cfg=data_cfg, model_cfg=model_cfg, cfg=cfg)
    elapsed = time.time() - t0

    history = np.load(
        os.path.join(run_dir, "checkpoints", "pretrain_history.npy"),
        allow_pickle=True
    ).item()
    # Move history file into run dir
    return elapsed, history


def run_finetune(run_dir: str, model_cfg, data_cfg) -> tuple[float, dict]:
    print("\n" + "="*60)
    print("STEP 4: Fine-tuning")
    print("="*60)
    from train.config import FinetuneConfig
    from train.finetune import finetune

    cfg = FinetuneConfig(
        pretrain_ckpt=os.path.join(run_dir, "checkpoints", "pretrain_best.pt"),
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        best_ckpt=os.path.join(run_dir, "checkpoints", "finetune_best.pt"),
    )

    t0 = time.time()
    encoder, head = finetune(data_cfg=data_cfg, model_cfg=model_cfg, cfg=cfg)
    elapsed = time.time() - t0

    history = np.load(
        os.path.join(run_dir, "checkpoints", "finetune_history.npy"),
        allow_pickle=True
    ).item()
    return elapsed, history


# ---------------------------------------------------------------------------
# Full evaluation (metrics + UMAP)
# ---------------------------------------------------------------------------

def run_evaluation(run_dir: str, model_cfg, data_cfg) -> dict:
    print("\n" + "="*60)
    print("STEP 5: Evaluation")
    print("="*60)
    import torch
    from torch.utils.data import DataLoader
    from data.dataset import load_datasets
    from model.transformer import scRNAEncoder
    from model.heads import CellTypeClassificationHead
    from eval.metrics import evaluate
    from eval.visualise import extract_cls_embeddings, umap_reduce, plot_comparison

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    splits = load_datasets(
        data_cfg.processed_path,
        max_seq_len=data_cfg.max_seq_len,
        mask_ratio=data_cfg.mask_ratio,
        seed=data_cfg.seed,
    )
    test_loader = DataLoader(splits["finetune_test"], batch_size=64, shuffle=False, pin_memory=False)
    label_names = splits["label_names"]

    ckpt = torch.load(
        os.path.join(run_dir, "checkpoints", "finetune_best.pt"),
        map_location=device,
    )
    encoder = scRNAEncoder(
        vocab_size=ckpt["vocab_size"],
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        ffn_dim=model_cfg.ffn_dim,
        dropout=model_cfg.dropout,
        max_seq_len=data_cfg.max_seq_len,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])

    head = CellTypeClassificationHead(
        hidden_dim=model_cfg.hidden_dim,
        num_classes=splits["num_classes"],
    ).to(device)
    head.load_state_dict(ckpt["head_state"])

    metrics = evaluate(encoder, head, test_loader, device, label_names)

    # UMAP: pretrained vs random init
    print("Generating UMAP comparison...")
    rand_encoder = scRNAEncoder(
        vocab_size=ckpt["vocab_size"],
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        ffn_dim=model_cfg.ffn_dim,
        dropout=model_cfg.dropout,
        max_seq_len=data_cfg.max_seq_len,
    ).to(device)

    pre_embs, labels = extract_cls_embeddings(encoder, test_loader, device)
    rand_embs, _ = extract_cls_embeddings(rand_encoder, test_loader, device)

    coords_pre = umap_reduce(pre_embs)
    coords_rand = umap_reduce(rand_embs)

    umap_path = os.path.join(run_dir, "umap_comparison.png")
    plot_comparison(pre_embs, rand_embs, labels, label_names, save_path=umap_path)

    np.save(os.path.join(run_dir, "embeddings_pretrained.npy"), pre_embs)
    np.save(os.path.join(run_dir, "embeddings_random.npy"), rand_embs)
    np.save(os.path.join(run_dir, "umap_coords_pretrained.npy"), coords_pre)
    np.save(os.path.join(run_dir, "umap_coords_random.npy"), coords_rand)
    np.save(os.path.join(run_dir, "test_labels.npy"), labels)

    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    run_dir: str,
    run_name: str,
    hw: dict,
    model_cfg,
    data_cfg,
    pretrain_cfg,
    finetune_cfg,
    pretrain_time: float,
    finetune_time: float,
    pretrain_history: dict,
    finetune_history: dict,
    metrics: dict,
    n_params: int,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _best(vals, mode="min"):
        return min(vals) if mode == "min" else max(vals)

    lines = [
        f"# Experiment Report — {run_name}",
        f"\n**Date:** {now}",
        "",
        "---",
        "",
        "## Hardware",
        "",
    ]
    for k, v in hw.items():
        lines.append(f"- **{k}:** {v}")

    lines += [
        "",
        "---",
        "",
        "## Model Configuration",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| hidden_dim | {model_cfg.hidden_dim} |",
        f"| num_layers | {model_cfg.num_layers} |",
        f"| num_heads | {model_cfg.num_heads} |",
        f"| ffn_dim | {model_cfg.ffn_dim} |",
        f"| dropout | {model_cfg.dropout} |",
        f"| max_seq_len | {data_cfg.max_seq_len} |",
        f"| vocab_size | {data_cfg.max_seq_len} |",
        f"| **Total parameters** | **{n_params:,}** |",
        "",
        "---",
        "",
        "## Data Configuration",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| dataset | PBMC 3k |",
        f"| n_hvg | 2000 |",
        f"| mask_ratio | {data_cfg.mask_ratio} |",
        f"| train_frac | {data_cfg.train_frac} |",
        f"| val_frac | {data_cfg.val_frac} |",
        f"| seed | {data_cfg.seed} |",
        "",
        "---",
        "",
        "## Pretraining",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| epochs | {pretrain_cfg.epochs} |",
        f"| batch_size | {pretrain_cfg.batch_size} |",
        f"| lr | {pretrain_cfg.lr} |",
        f"| weight_decay | {pretrain_cfg.weight_decay} |",
        f"| warmup_steps | {pretrain_cfg.warmup_steps} |",
        f"| grad_clip | {pretrain_cfg.grad_clip} |",
        f"| total time | {pretrain_time/60:.1f} min |",
        "",
        "### Results",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Best val loss | {_best(pretrain_history['val_loss']):.4f} |",
        f"| Final train loss | {pretrain_history['train_loss'][-1]:.4f} |",
        f"| Best val masked accuracy | {_best(pretrain_history['val_acc'], 'max'):.4f} |",
        "",
        "### Loss curve (val)",
        "",
        "```",
    ]
    # ASCII loss curve
    val_losses = pretrain_history["val_loss"]
    max_l, min_l = max(val_losses), min(val_losses)
    height = 8
    for row in range(height, -1, -1):
        threshold = min_l + (max_l - min_l) * row / height
        bar = ""
        for v in val_losses:
            bar += "█" if v >= threshold else " "
        lines.append(f"{threshold:.3f} |{bar}|")
    lines += [
        "       " + "-" * len(val_losses),
        f"       Epoch 1{' ' * (len(val_losses)-8)}Epoch {len(val_losses)}",
        "```",
        "",
        "---",
        "",
        "## Fine-tuning",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| epochs | {finetune_cfg.epochs} |",
        f"| batch_size | {finetune_cfg.batch_size} |",
        f"| lr | {finetune_cfg.lr} |",
        f"| weight_decay | {finetune_cfg.weight_decay} |",
        f"| warmup_steps | {finetune_cfg.warmup_steps} |",
        f"| freeze_encoder | {finetune_cfg.freeze_encoder} |",
        f"| total time | {finetune_time/60:.1f} min |",
        "",
        "### Results",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Best val accuracy | {_best(finetune_history['val_acc'], 'max'):.4f} |",
        f"| Final train accuracy | {finetune_history['train_acc'][-1]:.4f} |",
        "",
        "---",
        "",
        "## Test Set Evaluation",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Macro F1 | {metrics['macro_f1']:.4f} |",
        "",
        "### Per-class F1",
        "",
    ]

    # Load label names from checkpoint
    ckpt_path = os.path.join(run_dir, "checkpoints", "finetune_best.pt")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        label_names = ckpt.get("label_names", [f"Class {i}" for i in range(len(metrics["per_class_f1"]))])
    except Exception:
        label_names = [f"Class {i}" for i in range(len(metrics["per_class_f1"]))]

    lines.append("| Cell type | F1 |")
    lines.append("|---|---|")
    for name, f1 in zip(label_names, metrics["per_class_f1"]):
        lines.append(f"| {name} | {f1:.4f} |")

    lines += [
        "",
        "### Classification Report",
        "",
        "```",
        metrics["classification_report"],
        "```",
        "",
        "---",
        "",
        "## Artifacts",
        "",
        "| File | Description |",
        "|---|---|",
        "| `checkpoints/pretrain_best.pt` | Best pretraining checkpoint |",
        "| `checkpoints/finetune_best.pt` | Best fine-tuning checkpoint |",
        "| `checkpoints/pretrain_history.npy` | Pretraining loss/acc history |",
        "| `checkpoints/finetune_history.npy` | Fine-tuning loss/acc history |",
        "| `umap_comparison.png` | UMAP: pretrained vs random init encoder |",
        "| `embeddings_pretrained.npy` | CLS embeddings from pretrained encoder |",
        "| `embeddings_random.npy` | CLS embeddings from random-init encoder |",
        "| `run.log` | Full console output |",
        "",
    ]

    report = "\n".join(lines)
    report_path = os.path.join(run_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    run_dir = setup_run_dir(args.run_name)
    log_path = os.path.join(run_dir, "run.log")
    tee = Tee(log_path)
    sys.stdout = tee

    print(f"Run: {args.run_name}")
    print(f"Directory: {run_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sys.path.insert(0, ROOT)
    from train.config import DataConfig, ModelConfig, PretrainConfig, FinetuneConfig

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    pretrain_cfg = PretrainConfig(
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        best_ckpt=os.path.join(run_dir, "checkpoints", "pretrain_best.pt"),
    )
    finetune_cfg = FinetuneConfig(
        pretrain_ckpt=os.path.join(run_dir, "checkpoints", "pretrain_best.pt"),
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        best_ckpt=os.path.join(run_dir, "checkpoints", "finetune_best.pt"),
    )

    hw = hardware_info()
    print("\nHardware:")
    for k, v in hw.items():
        print(f"  {k}: {v}")

    # Count model params
    encoder_tmp = __import__("model.transformer", fromlist=["scRNAEncoder"]).scRNAEncoder(
        vocab_size=2003,  # approximate; updated after dataset load
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        ffn_dim=model_cfg.ffn_dim,
        dropout=model_cfg.dropout,
        max_seq_len=data_cfg.max_seq_len,
    )
    n_params = sum(p.numel() for p in encoder_tmp.parameters())

    t_download = run_download(run_dir)
    t_preprocess = run_preprocess(run_dir)
    t_pretrain, pretrain_history = run_pretrain(run_dir, model_cfg, data_cfg)
    t_finetune, finetune_history = run_finetune(run_dir, model_cfg, data_cfg)
    metrics = run_evaluation(run_dir, model_cfg, data_cfg)

    total = t_download + t_preprocess + t_pretrain + t_finetune
    print(f"\nTotal wall time: {total/60:.1f} min")

    generate_report(
        run_dir=run_dir,
        run_name=args.run_name,
        hw=hw,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        pretrain_cfg=pretrain_cfg,
        finetune_cfg=finetune_cfg,
        pretrain_time=t_pretrain,
        finetune_time=t_finetune,
        pretrain_history=pretrain_history,
        finetune_history=finetune_history,
        metrics=metrics,
        n_params=n_params,
    )

    sys.stdout = tee.terminal
    tee.close()
    print(f"\nDone. Results in experiments/{args.run_name}/")


if __name__ == "__main__":
    main()
