"""
scanpy preprocessing pipeline:
  QC → normalize → log1p → HVG selection → cell-type labels → save
"""

import os
import scanpy as sc
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "raw.h5ad")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed.h5ad")

N_HVG = 2000
MIN_GENES = 200
MIN_COUNTS = 500
MIN_CELLS = 3
N_COUNTS_NORM = 1e4


def preprocess(raw_path: str = RAW_PATH, out_path: str = PROCESSED_PATH) -> sc.AnnData:
    print(f"Loading {raw_path}")
    adata = sc.read_h5ad(raw_path)

    if "louvain" in adata.obs.columns and adata.raw is not None:
        # pbmc3k_processed(): adata.X has already been scaled/clipped
        # (sc.pp.scale, values roughly in [-3, 10]) for PCA/clustering — not
        # expression values, so rank/expr_bin tokenization would be meaningless
        # on it. adata.raw.X retains the pre-scaling log1p-normalized values
        # (13,714 genes), which is the same stage our own pipeline normally
        # produces — use that instead, then apply our own HVG selection.
        print(f"Detected pbmc3k_processed() input ({adata.n_obs} cells) — "
              f"using .raw (log-normalized) instead of scaled/clipped .X")
        raw_adata = adata.raw.to_adata()
        raw_adata.obs = adata.obs.copy()
        adata = raw_adata
        sc.pp.highly_variable_genes(adata, n_top_genes=min(N_HVG, adata.n_vars), subset=True)
        print(f"HVGs selected: {adata.n_vars}")
    else:
        # --- QC filters ---
        sc.pp.filter_cells(adata, min_genes=MIN_GENES)
        sc.pp.filter_cells(adata, min_counts=MIN_COUNTS)
        sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
        print(f"After QC: {adata.n_obs} cells, {adata.n_vars} genes")

        # --- Normalization ---
        sc.pp.normalize_total(adata, target_sum=N_COUNTS_NORM)
        sc.pp.log1p(adata)

        # --- HVG selection ---
        sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, subset=True)
        print(f"HVGs selected: {adata.n_vars}")

    # --- Cell-type labels ---
    # pbmc3k_processed already has louvain; re-annotate with coarse labels.
    if "cell_type" not in adata.obs.columns:
        _annotate_cell_types(adata)

    adata.write_h5ad(out_path)
    print(f"Saved processed AnnData to {out_path}")
    return adata


def _annotate_cell_types(adata: sc.AnnData) -> None:
    """
    Cluster with Leiden and assign coarse PBMC cell-type labels.
    Falls back to louvain if already present (pbmc3k_processed()).
    """
    if "louvain" in adata.obs.columns:
        # pbmc3k_processed()'s louvain column already holds real annotated
        # names (e.g. "CD4 T cells", "CD14+ Monocytes") from the scanpy
        # clustering tutorial — use directly, no cluster-id mapping needed.
        adata.obs["cell_type"] = adata.obs["louvain"].astype("category")
        return

    # Generic path: compute neighbors → Leiden clusters
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.leiden(adata, resolution=0.5)
    adata.obs["cell_type"] = adata.obs["leiden"].astype("category")
    print("Cell types assigned via Leiden clustering.")


if __name__ == "__main__":
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Run data/download.py first; {RAW_PATH} not found.")
    preprocess()
