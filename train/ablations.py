"""
Runs the 7 ablation experiments listed in README.md's "Ablation Experiments" table.

Each experiment varies one axis of DataConfig/ModelConfig/PretrainConfig/FinetuneConfig and
reuses pretrain()/finetune() from train/pretrain.py and train/finetune.py. Pretrain and finetune
runs are cached on disk (keyed by the config fields that affect their outcome) so that shared
baseline runs are not recomputed across experiments.

Usage:
    uv run python train/ablations.py                       # all 7 experiments
    uv run python train/ablations.py --experiment masking_ratio
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

import anndata as ad

from data.dataset import load_datasets
from eval.metrics import evaluate
from eval.visualise import extract_cls_embeddings, umap_reduce
from model.transformer import scRNAEncoder
from model.gnn import build_gene_gat
from model.heads import CellTypeClassificationHead
from train.config import DataConfig, ModelConfig, PretrainConfig, FinetuneConfig
from train.pretrain import pretrain
from train.finetune import finetune

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "experiments", "ablations")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
NO_CKPT = os.path.join(CACHE_DIR, "__no_checkpoint__.pt")  # deliberately never created


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _hash_key(*parts: dict) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cached pretrain / finetune
# ---------------------------------------------------------------------------

def cached_pretrain(data_cfg: DataConfig, model_cfg: ModelConfig, epochs: int = 30) -> tuple[str, dict]:
    """Returns (pretrain_ckpt_path, history). Skips training if a cached run exists."""
    key = _hash_key(asdict(data_cfg), asdict(model_cfg), {"epochs": epochs, "stage": "pretrain"})
    run_dir = os.path.join(CACHE_DIR, key)
    ckpt_path = os.path.join(run_dir, "pretrain_best.pt")
    history_path = os.path.join(run_dir, "pretrain_history.npy")

    if os.path.exists(ckpt_path) and os.path.exists(history_path):
        print(f"  [cache hit] pretrain {key}")
        return ckpt_path, np.load(history_path, allow_pickle=True).item()

    print(f"  [cache miss] pretrain {key} -> {run_dir}")
    cfg = PretrainConfig(checkpoint_dir=run_dir, best_ckpt=ckpt_path, epochs=epochs)
    pretrain(data_cfg=data_cfg, model_cfg=model_cfg, cfg=cfg)
    history = np.load(history_path, allow_pickle=True).item()
    return ckpt_path, history


def cached_finetune(
    pretrain_ckpt: str | None,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    freeze_encoder: str = "none",
    freeze_layers: int = 2,
    epochs: int = 15,
) -> dict:
    """Returns test metrics dict (accuracy, macro_f1, ...). Skips training if cached."""
    key = _hash_key(
        {"pretrain_ckpt": pretrain_ckpt or "scratch"},
        asdict(data_cfg), asdict(model_cfg),
        {"freeze_encoder": freeze_encoder, "freeze_layers": freeze_layers,
         "epochs": epochs, "stage": "finetune"},
    )
    run_dir = os.path.join(CACHE_DIR, key)
    metrics_path = os.path.join(run_dir, "test_metrics.json")

    if os.path.exists(metrics_path):
        print(f"  [cache hit] finetune {key}")
        with open(metrics_path) as f:
            return json.load(f)

    print(f"  [cache miss] finetune {key} -> {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    cfg = FinetuneConfig(
        pretrain_ckpt=pretrain_ckpt or NO_CKPT,
        checkpoint_dir=run_dir,
        best_ckpt=os.path.join(run_dir, "finetune_best.pt"),
        freeze_encoder=freeze_encoder,
        freeze_layers=freeze_layers,
        epochs=epochs,
    )
    _, _, metrics = finetune(data_cfg=data_cfg, model_cfg=model_cfg, cfg=cfg)
    serializable = {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }
    with open(metrics_path, "w") as f:
        json.dump(serializable, f)
    return serializable


def embedding_silhouette(pretrain_ckpt: str, data_cfg: DataConfig, model_cfg: ModelConfig, device: torch.device) -> float:
    """Silhouette score of the pretrained encoder's CLS embeddings (2D UMAP) against cell_type labels."""
    from sklearn.metrics import silhouette_score

    splits = load_datasets(
        data_cfg.processed_path, max_seq_len=data_cfg.max_seq_len, mask_ratio=data_cfg.mask_ratio,
        train_frac=data_cfg.train_frac, val_frac=data_cfg.val_frac, seed=data_cfg.seed,
        tokenization=data_cfg.tokenization, n_bins=data_cfg.n_bins,
    )
    loader = DataLoader(splits["finetune_test"], batch_size=64, shuffle=False)

    ckpt = torch.load(pretrain_ckpt, map_location=device)

    gene_gat = None
    if model_cfg.use_gnn:
        tmp_adata = ad.read_h5ad(data_cfg.processed_path)
        gene_names = list(tmp_adata.var_names)
        del tmp_adata
        gene_gat = build_gene_gat(gene_names, model_cfg, data_cfg.processed_path, device)

    encoder = scRNAEncoder(
        vocab_size=ckpt["vocab_size"], hidden_dim=model_cfg.hidden_dim, num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads, ffn_dim=model_cfg.ffn_dim, dropout=model_cfg.dropout,
        max_seq_len=data_cfg.max_seq_len,
        n_expr_bins=(data_cfg.n_bins if data_cfg.tokenization == "expr_bin" else None),
        gene_gat=gene_gat,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])

    embs, labels = extract_cls_embeddings(encoder, loader, device)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    coords = umap_reduce(embs)
    return float(silhouette_score(coords, labels))


# ---------------------------------------------------------------------------
# The 7 experiments
# ---------------------------------------------------------------------------

def experiment_masking_ratio(device: torch.device) -> list[dict]:
    results = []
    for mask_ratio in [0.05, 0.15, 0.25, 0.40]:
        print(f"\n-- masking_ratio={mask_ratio} --")
        data_cfg = DataConfig(mask_ratio=mask_ratio)
        model_cfg = ModelConfig()
        ckpt, history = cached_pretrain(data_cfg, model_cfg)
        metrics = cached_finetune(ckpt, data_cfg, model_cfg)
        results.append({
            "mask_ratio": mask_ratio,
            "val_loss": min(history["val_loss"]),
            "test_macro_f1": metrics["macro_f1"],
            "test_accuracy": metrics["accuracy"],
        })
    return results


def experiment_tokenization(device: torch.device) -> list[dict]:
    results = []
    for tokenization in ["rank", "expr_bin"]:
        print(f"\n-- tokenization={tokenization} --")
        data_cfg = DataConfig(tokenization=tokenization)
        model_cfg = ModelConfig()
        ckpt, history = cached_pretrain(data_cfg, model_cfg)
        silhouette = embedding_silhouette(ckpt, data_cfg, model_cfg, device)
        results.append({
            "tokenization": tokenization,
            "val_loss": min(history["val_loss"]),
            "umap_silhouette": silhouette,
        })
    return results


def experiment_model_depth(device: torch.device) -> list[dict]:
    results = []
    for num_layers in [2, 4, 6]:
        print(f"\n-- num_layers={num_layers} --")
        data_cfg = DataConfig()
        model_cfg = ModelConfig(num_layers=num_layers)
        ckpt, history = cached_pretrain(data_cfg, model_cfg)
        metrics = cached_finetune(ckpt, data_cfg, model_cfg)
        results.append({
            "num_layers": num_layers,
            "val_loss": min(history["val_loss"]),
            "test_accuracy": metrics["accuracy"],
        })
    return results


def experiment_pretrain_vs_scratch(device: torch.device) -> list[dict]:
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    ckpt, _ = cached_pretrain(data_cfg, model_cfg)

    results = []
    for label, use_ckpt in [("pretrained", ckpt), ("scratch", None)]:
        print(f"\n-- pretrain_vs_scratch={label} --")
        metrics = cached_finetune(use_ckpt, data_cfg, model_cfg)
        results.append({"condition": label, "test_accuracy": metrics["accuracy"]})
    return results


def experiment_freeze_vs_finetune(device: torch.device) -> list[dict]:
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    ckpt, _ = cached_pretrain(data_cfg, model_cfg)

    results = []
    for freeze in ["none", "full"]:
        print(f"\n-- freeze_encoder={freeze} --")
        metrics = cached_finetune(ckpt, data_cfg, model_cfg, freeze_encoder=freeze)
        results.append({"freeze_encoder": freeze, "test_accuracy": metrics["accuracy"]})
    return results


def experiment_gene_embeddings(device: torch.device) -> list[dict]:
    data_cfg = DataConfig()
    conditions = [
        ("baseline", ModelConfig(use_gnn=False)),
        ("gnn_frozen", ModelConfig(use_gnn=True, gnn_freeze=True)),
        ("gnn_joint", ModelConfig(use_gnn=True, gnn_freeze=False)),
    ]
    results = []
    for label, model_cfg in conditions:
        print(f"\n-- gene_embeddings={label} --")
        ckpt, _ = cached_pretrain(data_cfg, model_cfg)
        metrics = cached_finetune(ckpt, data_cfg, model_cfg)
        silhouette = embedding_silhouette(ckpt, data_cfg, model_cfg, device)
        results.append({
            "condition": label,
            "test_accuracy": metrics["accuracy"],
            "umap_silhouette": silhouette,
        })
    return results


def experiment_classification_head(device: torch.device) -> list[dict]:
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    ckpt, _ = cached_pretrain(data_cfg, model_cfg)

    results = []
    for label, head_model_cfg in [("cls", ModelConfig()), ("gat", ModelConfig(use_gat_head=True))]:
        print(f"\n-- classification_head={label} --")
        metrics = cached_finetune(ckpt, data_cfg, head_model_cfg)
        results.append({"head": label, "test_accuracy": metrics["accuracy"]})
    return results


EXPERIMENTS = {
    "masking_ratio": experiment_masking_ratio,
    "tokenization": experiment_tokenization,
    "model_depth": experiment_model_depth,
    "pretrain_vs_scratch": experiment_pretrain_vs_scratch,
    "freeze_vs_finetune": experiment_freeze_vs_finetune,
    "gene_embeddings": experiment_gene_embeddings,
    "classification_head": experiment_classification_head,
}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(all_results: dict[str, list[dict]]) -> str:
    lines = ["# Ablation Results\n"]
    for name, rows in all_results.items():
        lines.append(f"## {name}\n")
        if not rows:
            lines.append("(no results)\n")
            continue
        cols = list(rows[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in rows:
            vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    report = "\n".join(lines)
    path = os.path.join(OUT_DIR, "summary.md")
    with open(path, "w") as f:
        f.write(report)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()), default=None,
                        help="Run a single experiment; omit to run all 7.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = _get_device()
    print(f"Device: {device}")

    results_path = os.path.join(OUT_DIR, "results.json")
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)

    to_run = [args.experiment] if args.experiment else list(EXPERIMENTS.keys())
    for name in to_run:
        print(f"\n{'='*70}\nEXPERIMENT: {name}\n{'='*70}")
        all_results[name] = EXPERIMENTS[name](device)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    summary_path = write_summary(all_results)
    print(f"\nResults: {results_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
