"""
Shared utilities for benchmark preprocessing and k-fold CV splits.

Applies the same preprocessing pipeline as our PBMC data so embeddings
are comparable across experiments.
"""

from __future__ import annotations

import os
import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.model_selection import StratifiedKFold

BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks")

BENCHMARK_FILES = {
    # scBiGNN baseline
    "BaronHuman":  "baron_human.h5ad",
    "BaronMouse":  "baron_mouse.h5ad",
    "AMB":         "amb.h5ad",
    "Zheng68K":    "zheng68k.h5ad",
    "Zhengsorted": "zhengsorted.h5ad",
    # ACTINN baseline — same Zenodo archive (Abdelaal et al. 2019)
    "Segerstolpe": "segerstolpe.h5ad",  # human pancreas, 13 types
    "Muraro":      "muraro.h5ad",       # human pancreas, 9 types
    # Zeisel/Macosko/Klein dropped: not present in this Zenodo archive at all
    # (verified by listing its full contents) — see README.md's Benchmarks section.
}

N_HVG       = 2000
MIN_GENES   = 200
MIN_COUNTS  = 500
MIN_CELLS   = 3


def preprocess_benchmark(adata: ad.AnnData, dataset_name: str) -> ad.AnnData:
    """
    Apply our standard preprocessing pipeline to a benchmark dataset.
    Lighter filters than PBMC to preserve small datasets (BaronMouse = 1,886 cells).
    """
    print(f"Preprocessing {dataset_name}: {adata.n_obs} cells × {adata.n_vars} genes")

    # Back up labels before filtering (filtering drops cells)
    if "cell_type" not in adata.obs.columns:
        raise ValueError(f"{dataset_name}: missing cell_type column")

    # Adaptive QC — relax thresholds for small datasets
    min_genes  = MIN_GENES if adata.n_obs > 3000 else 50
    min_counts = MIN_COUNTS if adata.n_obs > 3000 else 100

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_hvg = min(N_HVG, adata.n_vars - 1)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)

    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    print(f"  After QC: {adata.n_obs} cells × {adata.n_vars} HVGs  "
          f"| {adata.obs['cell_type'].nunique()} cell types")
    return adata


def load_benchmark(name: str, reprocess: bool = False) -> ad.AnnData:
    """
    Load a benchmark dataset, applying preprocessing if needed.
    Saves processed version to benchmarks/<name>_processed.h5ad.
    """
    raw_path  = os.path.join(BENCH_DIR, BENCHMARK_FILES[name])
    proc_path = raw_path.replace(".h5ad", "_processed.h5ad")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found. Run: uv run python data/download_benchmarks.py"
        )

    if os.path.exists(proc_path) and not reprocess:
        return ad.read_h5ad(proc_path)

    adata = ad.read_h5ad(raw_path)
    adata = preprocess_benchmark(adata, name)
    adata.write_h5ad(proc_path)
    return adata


def make_kfold_splits(
    adata: ad.AnnData,
    k: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return k stratified (train_idx, test_idx) pairs.
    Stratified by cell_type to preserve class balance in each fold.
    """
    labels = adata.obs["cell_type"].cat.codes.to_numpy()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(train, test) for train, test in skf.split(np.zeros(len(labels)), labels)]
