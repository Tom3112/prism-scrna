"""
5-fold cross-validation evaluation on standard benchmark datasets.

Compares our model against published baselines:
  - Original 5 (scBiGNN, Ma et al. 2023, Table 2) — verified exact match
  - Segerstolpe/Muraro (ACTINN, Ma & Pellegrini 2020) — from the same Abdelaal
    et al. 2019 benchmark suite (Zenodo 3357167); dataset stats verified against
    Abdelaal's Table 2, accuracy values not independently re-derived from the
    source figure
  - Zeisel/Macosko/Klein: no verified baseline. Despite prior attribution to
    Abdelaal et al. 2019, these 3 datasets do not appear in that paper or in
    scBiGNN's — reported standalone rather than against an unconfirmed number.

  Dataset      | Baseline | Method   | ours
  -------------|----------|----------|-----
  Zheng68K     |  0.760   | scBiGNN  |  ?
  Zhengsorted  |  0.867   | scBiGNN  |  ?
  BaronHuman   |  0.983   | scBiGNN  |  ?
  BaronMouse   |  0.983   | scBiGNN  |  ?
  AMB          |  0.994   | scBiGNN  |  ?
  Segerstolpe  |  0.886   | ACTINN   |  ?
  Muraro       |  0.962   | ACTINN   |  ?
  Zeisel       |    —     |    —     |  ?
  Macosko      |    —     |    —     |  ?
  Klein        |    —     |    —     |  ?

Usage:
    uv run python train/benchmark_eval.py                    # plain baseline (no GNN)
    uv run python train/benchmark_eval.py --dataset BaronHuman
    uv run python train/benchmark_eval.py --gnn frozen        # GeneGAT frozen (Option A)
    uv run python train/benchmark_eval.py --gnn joint         # GeneGAT joint (Option A)
    uv run python train/benchmark_eval.py --head gat          # CellGAT head (Option B)
    uv run python train/benchmark_eval.py --dataset Zeisel --epochs 20

Results are saved per-variant to experiments/benchmark_results_<head>_<gnn>.npy so
different --head/--gnn runs don't overwrite each other.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.benchmark_utils import load_benchmark, make_kfold_splits, BENCHMARK_FILES, BENCH_DIR
from data.dataset import scRNADataset
from model.transformer import scRNAEncoder
from model.heads import CellTypeClassificationHead, CellGATClassificationHead
from model.gene_graph import build_gene_graph
from model.gnn import build_gene_gat
from train.config import DataConfig, ModelConfig, FinetuneConfig
from train.pretrain import get_cosine_schedule_with_warmup
from eval.metrics import collect_predictions, compute_metrics, _forward_head

# Published baselines for 5-fold CV accuracy.
#
# Original 5: scBiGNN (Ma et al. 2023, Table 2) — verified exact match, direct
#             comparison target.
# Segerstolpe/Muraro: ACTINN (Ma & Pellegrini 2020), from the same Abdelaal
#             et al. 2019 benchmark suite.
# Zeisel/Macosko/Klein have no entry here — no verified baseline (see module
# docstring); evaluate_dataset() reports these standalone via BASELINES.get()'s
# (None, None) default.
BASELINES: dict[str, tuple[str, float]] = {
    # dataset          method      accuracy
    "Zheng68K":    ("scBiGNN",  0.760),
    "Zhengsorted": ("scBiGNN",  0.867),
    "BaronHuman":  ("scBiGNN",  0.983),
    "BaronMouse":  ("scBiGNN",  0.983),
    "AMB":         ("scBiGNN",  0.994),
    "Segerstolpe": ("ACTINN",   0.886),
    "Muraro":      ("ACTINN",   0.962),
}

# Keep old name as alias for backward compatibility
SCBIGNN_BASELINE = {k: v for k, (_, v) in BASELINES.items() if _[0] == "scBiGNN"}

# STRING species per dataset — BaronMouse and AMB are mouse; STRING won't match
# mouse gene symbols against the human (9606) network, so PPI lookups for those
# two would silently return near-empty graphs without this.
STRING_SPECIES_BY_DATASET = {
    "BaronMouse": 10090,  # Mus musculus
    "AMB":        10090,  # Mus musculus
}


def _species_for(name: str) -> int:
    return STRING_SPECIES_BY_DATASET.get(name, 9606)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_fold(
    train_adata,
    test_adata,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    ft_cfg: FinetuneConfig,
    device: torch.device,
    pretrain_ckpt: str | None,
    ppi_edge_index: torch.Tensor | None = None,
    ppi_edge_weight: torch.Tensor | None = None,
    gene_gat=None,
) -> tuple[float, float]:
    """Train on one fold, return (accuracy, macro_f1)."""

    # Use combined label set (test may have types not in train)
    all_cats = sorted(set(train_adata.obs["cell_type"].cat.categories) |
                      set(test_adata.obs["cell_type"].cat.categories))
    for a in [train_adata, test_adata]:
        a.obs["cell_type"] = a.obs["cell_type"].cat.set_categories(all_cats)
    num_classes = len(all_cats)

    train_ds = scRNADataset(train_adata, max_seq_len=data_cfg.max_seq_len,
                            mask_ratio=data_cfg.mask_ratio, mode="finetune")
    test_ds  = scRNADataset(test_adata,  max_seq_len=data_cfg.max_seq_len,
                            mask_ratio=data_cfg.mask_ratio, mode="finetune")

    train_loader = DataLoader(train_ds, batch_size=ft_cfg.batch_size,
                              shuffle=True, num_workers=data_cfg.num_workers,
                              pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=ft_cfg.batch_size,
                              shuffle=False, num_workers=data_cfg.num_workers,
                              pin_memory=device.type == "cuda")

    vocab_size = train_ds.vocab_size
    encoder = scRNAEncoder(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        ffn_dim=model_cfg.ffn_dim,
        dropout=model_cfg.dropout,
        max_seq_len=data_cfg.max_seq_len,
        gene_gat=gene_gat,
    ).to(device)

    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        ckpt = torch.load(pretrain_ckpt, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"], strict=False)

    if model_cfg.use_gat_head and ppi_edge_index is not None:
        head = CellGATClassificationHead(
            hidden_dim=model_cfg.hidden_dim,
            num_classes=num_classes,
            dropout=model_cfg.dropout,
            n_genes=vocab_size - 3,
            ppi_edge_index=ppi_edge_index.to(device),
            ppi_edge_weight=ppi_edge_weight.to(device),
            n_gat_heads=model_cfg.gat_head_n_heads,
        ).to(device)
    else:
        head = CellTypeClassificationHead(
            hidden_dim=model_cfg.hidden_dim,
            num_classes=num_classes,
            dropout=model_cfg.dropout,
        ).to(device)

    optimizer = AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=ft_cfg.lr, weight_decay=ft_cfg.weight_decay,
    )
    total_steps = ft_cfg.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, ft_cfg.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, ft_cfg.epochs + 1):
        encoder.train(); head.train()
        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            amsk = batch["attention_mask"].to(device)
            ct   = batch["cell_type"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                loss, _ = _forward_head(encoder, head, ids, amsk, ct)
            params = list(encoder.parameters()) + list(head.parameters())
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, ft_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, ft_cfg.grad_clip)
                optimizer.step()
            scheduler.step()

    preds, labels = collect_predictions(encoder, head, test_loader, device)
    metrics = compute_metrics(preds, labels, all_cats)
    return metrics["accuracy"], metrics["macro_f1"], metrics["median_f1"]


def evaluate_dataset(
    name: str,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    ft_cfg: FinetuneConfig,
    device: torch.device,
    pretrain_ckpt: str | None,
    k: int = 5,
) -> dict:
    gnn_label = "none"
    if model_cfg.use_gnn:
        gnn_label = "frozen" if model_cfg.gnn_freeze else "joint"
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  ({k}-fold CV)  head={'GAT' if model_cfg.use_gat_head else 'CLS'}  gnn={gnn_label}")
    print(f"{'='*60}")

    adata   = load_benchmark(name)
    splits  = make_kfold_splits(adata, k=k)
    species = _species_for(name)

    # processed_path needed as co-expression fallback if STRING API is unreachable
    proc_path = os.path.join(BENCH_DIR, BENCHMARK_FILES[name]).replace(".h5ad", "_processed.h5ad")

    # Build PPI graph once for the whole dataset (same HVGs across all folds)
    ppi_edge_index = ppi_edge_weight = None
    if model_cfg.use_gat_head:
        gene_names   = list(adata.var_names)
        ppi_cache_dir = os.path.join(BENCH_DIR, f"{name}_ppi")
        print(f"  Building PPI graph for {len(gene_names)} genes (species={species}) …")
        ppi_edge_index, ppi_edge_weight = build_gene_graph(
            gene_names,
            cache_dir=ppi_cache_dir,
            min_score=model_cfg.string_min_score,
            species=species,
            processed_path=proc_path,
        )

    # GeneGAT (Option A) is trainable in joint mode, so it must be rebuilt fresh
    # per fold — reusing one instance across folds would leak fold N's trained
    # GAT weights into fold N+1, breaking CV independence. The underlying STRING
    # edge_index/edge_weight are still disk-cached, so rebuilding the nn.Module
    # wrapper each fold is cheap.
    if not model_cfg.use_gnn:
        proc_path = None

    fold_accs, fold_f1s, fold_median_f1s = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        t0 = time.time()
        train_a = adata[train_idx].copy()
        test_a  = adata[test_idx].copy()
        gene_gat = None
        if model_cfg.use_gnn:
            gene_gat = build_gene_gat(model_cfg, proc_path, device, species=species)
        acc, f1, median_f1 = train_one_fold(
            train_a, test_a, model_cfg, data_cfg, ft_cfg, device, pretrain_ckpt,
            ppi_edge_index=ppi_edge_index, ppi_edge_weight=ppi_edge_weight,
            gene_gat=gene_gat,
        )
        fold_accs.append(acc); fold_f1s.append(f1); fold_median_f1s.append(median_f1)
        print(f"  Fold {fold}/{k}  acc={acc:.4f}  macro_f1={f1:.4f}  median_f1={median_f1:.4f}  ({time.time()-t0:.1f}s)")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    mean_f1  = np.mean(fold_f1s)
    mean_median_f1 = np.mean(fold_median_f1s)

    ref_method, ref_acc = BASELINES.get(name, (None, None))
    delta = mean_acc - ref_acc if ref_acc is not None else None

    print(f"\n  Mean acc       : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean macro F1  : {mean_f1:.4f}")
    print(f"  Mean median F1 : {mean_median_f1:.4f}  (Abdelaal et al. 2019's primary metric — robust to rare-class outliers)")
    if ref_acc is not None:
        sign = "▲" if delta > 0 else "▼"
        print(f"  vs {ref_method}: {sign} {abs(delta)*100:.2f}%  "
              f"({ref_acc:.4f} → {mean_acc:.4f})")

    return {
        "dataset":        name,
        "mean_acc":       mean_acc,
        "std_acc":        std_acc,
        "mean_f1":        mean_f1,
        "mean_median_f1": mean_median_f1,
        "fold_accs":      fold_accs,
        "fold_median_f1s": fold_median_f1s,
        "ref_method":     ref_method,
        "ref_acc":        ref_acc,
        "delta":          delta,
    }


def print_summary_table(results: list[dict]):
    print(f"\n{'='*90}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*90}")
    print(f"{'Dataset':<15} {'Ours (acc)':<18} {'Median F1':<11} {'Baseline':<10} {'Method':<12} {'Δ':>8}")
    print("-" * 78)
    for r in results:
        delta_str  = f"{r['delta']*100:+.2f}%" if r["delta"] is not None else "—"
        ref_str    = f"{r['ref_acc']:.3f}" if r["ref_acc"] is not None else "—"
        method_str = r["ref_method"] or "—"
        print(f"{r['dataset']:<15} {r['mean_acc']:.4f} ± {r['std_acc']:.4f}  "
              f"{r['mean_median_f1']:<11.4f} {ref_str:<10} {method_str:<12} {delta_str:>8}")
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        help="Dataset name or 'all'")
    parser.add_argument("--pretrain-ckpt",
                        default=os.path.join(os.path.dirname(__file__), "..",
                                             "checkpoints", "pretrain_best.pt"),
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--k", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--head", choices=["cls", "gat"], default="cls",
                        help="cls = [CLS] linear probe; gat = CellGAT (PPI graph during classification)")
    parser.add_argument("--gnn", choices=["none", "frozen", "joint"], default="none",
                        help="none = plain nn.Embedding; frozen/joint = GeneGAT gene embeddings (Option A)")
    args = parser.parse_args()

    device     = _get_device()
    model_cfg  = ModelConfig(
        use_gat_head=(args.head == "gat"),
        use_gnn=(args.gnn != "none"),
        gnn_freeze=(args.gnn == "frozen"),
    )
    data_cfg   = DataConfig()
    ft_cfg     = FinetuneConfig(epochs=args.epochs)

    print(f"Device: {device}")
    print(f"Pretrain ckpt: {args.pretrain_ckpt} "
          f"({'found' if os.path.exists(args.pretrain_ckpt) else 'NOT FOUND — training from scratch'})")

    datasets = list(BENCHMARK_FILES.keys()) if args.dataset == "all" else [args.dataset]
    results  = []
    for name in datasets:
        try:
            r = evaluate_dataset(name, model_cfg, data_cfg, ft_cfg, device,
                                 args.pretrain_ckpt, k=args.k)
            results.append(r)
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")

    if results:
        print_summary_table(results)
        variant = f"{args.head}_{args.gnn}"
        out = os.path.join(os.path.dirname(__file__), "..", "experiments",
                           f"benchmark_results_{variant}.npy")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, results)
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
