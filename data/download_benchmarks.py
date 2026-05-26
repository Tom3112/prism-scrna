"""
Download the 5 standard cell-type annotation benchmark datasets from Zenodo.

Source: Abdelaal et al. (2019) "A comparison of automatic cell identification methods
        for single-cell RNA sequencing data." Genome Biology.
Zenodo: https://doi.org/10.5281/zenodo.3357167

Datasets included:
  - BaronHuman   : human pancreas,    8,569 cells,  14 types
  - BaronMouse   : mouse pancreas,    1,886 cells,  13 types
  - AMB          : mouse visual cortex, 12,832 cells, 22 types (subset)
  - Zheng68K     : human PBMC,       65,943 cells,  11 types
  - Zhengsorted  : human PBMC (FACS), 20,000 cells, 10 types

Usage:
    uv run python data/download_benchmarks.py
"""

from __future__ import annotations

import io
import os
import tarfile
import zipfile
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np

ZENODO_URL  = "https://zenodo.org/records/3357167/files/scRNAseq_Benchmark_datasets.zip?download=1"
BENCH_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks")
ZIP_CACHE   = os.path.join(BENCH_DIR, "_zenodo_raw.zip")

DATASETS = [
    # Original 5 (used by scBiGNN)
    "BaronHuman", "BaronMouse", "AMB", "Zheng_68K", "Zhengsorted",
    # Extended 5 (all from same Zenodo archive, used by Abdelaal et al. 2019)
    "Zeisel",       # mouse brain,          ~3,000 cells,  9 types
    "Segerstolpe",  # human pancreas,       ~2,300 cells, 14 types
    "Muraro",       # human pancreas,       ~2,100 cells,  9 types
    "Macosko",      # mouse retina,        ~44,000 cells, 39 types
    "Klein",        # mouse ESC timecourse,  ~2,400 cells,  4 types
]

# Canonical output names
CANONICAL = {
    "BaronHuman":  "baron_human.h5ad",
    "BaronMouse":  "baron_mouse.h5ad",
    "AMB":         "amb.h5ad",
    "Zheng_68K":   "zheng68k.h5ad",
    "Zhengsorted": "zhengsorted.h5ad",
    "Zeisel":      "zeisel.h5ad",
    "Segerstolpe": "segerstolpe.h5ad",
    "Muraro":      "muraro.h5ad",
    "Macosko":     "macosko.h5ad",
    "Klein":       "klein.h5ad",
}


def _progress_hook(count, block_size, total_size):
    pct = min(100, int(count * block_size * 100 / total_size))
    print(f"\r  Downloading: {pct}%", end="", flush=True)


def _ensure_zip():
    os.makedirs(BENCH_DIR, exist_ok=True)
    if os.path.exists(ZIP_CACHE):
        print(f"Zenodo zip already cached at {ZIP_CACHE}")
        return
    print(f"Downloading Zenodo archive (~3.7 GB) to {ZIP_CACHE} ...")
    urllib.request.urlretrieve(ZENODO_URL, ZIP_CACHE, reporthook=_progress_hook)
    print()


def _extract_datasets():
    """
    Extract only the 5 target datasets from the zip.
    The zip structure is: scRNAseq_Benchmark_datasets/<DatasetName>/<DatasetName>_*.loom or .csv
    We read each into AnnData and save as h5ad.
    """
    import scanpy as sc

    with zipfile.ZipFile(ZIP_CACHE, "r") as zf:
        names = zf.namelist()

        for dataset, out_name in CANONICAL.items():
            out_path = os.path.join(BENCH_DIR, out_name)
            if os.path.exists(out_path):
                print(f"  {out_name} already exists — skipping.")
                continue

            # Find matching files in the zip
            matches = [n for n in names if f"/{dataset}/" in n and not n.endswith("/")]
            if not matches:
                print(f"  WARNING: {dataset} not found in zip. Files available: "
                      f"{[n for n in names if dataset.lower() in n.lower()[:60]]}")
                continue

            print(f"  Extracting {dataset} ({len(matches)} files)...")

            # Extract to temp dir
            tmp_dir = os.path.join(BENCH_DIR, f"_tmp_{dataset}")
            os.makedirs(tmp_dir, exist_ok=True)
            for fname in matches:
                zf.extract(fname, tmp_dir)

            # Find the actual file and load it
            adata = _load_dataset(tmp_dir, dataset)
            if adata is not None:
                adata.write_h5ad(out_path)
                print(f"  Saved {adata.n_obs} cells × {adata.n_vars} genes → {out_name}")

            # Clean up temp
            import shutil
            shutil.rmtree(tmp_dir)


def _load_dataset(tmp_dir: str, dataset: str) -> ad.AnnData | None:
    """Load a dataset from its extracted files into AnnData."""
    import scanpy as sc
    from pathlib import Path

    files = list(Path(tmp_dir).rglob("*"))
    files = [f for f in files if f.is_file()]

    # Try loom first
    loom_files = [f for f in files if f.suffix == ".loom"]
    if loom_files:
        adata = sc.read_loom(str(loom_files[0]))
        # Standardise cell type column
        _standardise_labels(adata, dataset)
        return adata

    # Try h5ad
    h5ad_files = [f for f in files if f.suffix == ".h5ad"]
    if h5ad_files:
        adata = sc.read_h5ad(str(h5ad_files[0]))
        _standardise_labels(adata, dataset)
        return adata

    # Try CSV (expression matrix + labels separate)
    csv_files = [f for f in files if f.suffix == ".csv"]
    if csv_files:
        return _load_from_csv(csv_files, dataset)

    # Try txt/tsv
    tsv_files = [f for f in files if f.suffix in (".txt", ".tsv")]
    if tsv_files:
        return _load_from_csv(tsv_files, dataset)

    print(f"  Could not load {dataset}: no recognised format among {[f.name for f in files]}")
    return None


def _load_from_csv(files, dataset: str) -> ad.AnnData | None:
    import pandas as pd

    expr_file  = next((f for f in files if "expr" in f.name.lower() or
                       "count" in f.name.lower() or "matrix" in f.name.lower()), None)
    label_file = next((f for f in files if "label" in f.name.lower() or
                       "cell_type" in f.name.lower() or "annot" in f.name.lower()), None)

    if expr_file is None:
        # Fallback: largest file is expression matrix
        expr_file = max(files, key=lambda f: f.stat().st_size)

    print(f"    Reading expression from {expr_file.name}")
    expr = pd.read_csv(str(expr_file), index_col=0)

    # Assume cells are rows if n_rows > n_cols, else transpose
    if expr.shape[0] < expr.shape[1]:
        expr = expr.T

    adata = ad.AnnData(X=expr.values.astype("float32"),
                       obs=pd.DataFrame(index=expr.index),
                       var=pd.DataFrame(index=expr.columns))

    if label_file is not None:
        print(f"    Reading labels from {label_file.name}")
        labels = pd.read_csv(str(label_file), index_col=0, header=None).squeeze()
        adata.obs["cell_type"] = labels.values if len(labels) == adata.n_obs else "Unknown"

    _standardise_labels(adata, dataset)
    return adata


def _standardise_labels(adata: ad.AnnData, dataset: str):
    """Ensure cell_type column exists and is a category."""
    label_candidates = ["cell_type", "CellType", "celltype", "label",
                        "Cluster", "cluster", "Annotation"]
    for col in label_candidates:
        if col in adata.obs.columns:
            if col != "cell_type":
                adata.obs["cell_type"] = adata.obs[col]
            break

    if "cell_type" not in adata.obs.columns:
        print(f"    WARNING: no cell_type column found for {dataset}. "
              f"Obs columns: {list(adata.obs.columns)}")
        adata.obs["cell_type"] = "Unknown"

    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")


def download_all():
    _ensure_zip()
    print("Extracting target datasets...")
    _extract_datasets()
    print("\nDone. Available benchmark datasets:")
    for name, fname in CANONICAL.items():
        path = os.path.join(BENCH_DIR, fname)
        if os.path.exists(path):
            adata = ad.read_h5ad(path)
            n_types = adata.obs["cell_type"].nunique()
            print(f"  {name:<15} {adata.n_obs:>7,} cells  {adata.n_vars:>6,} genes  "
                  f"{n_types:>3} types")
        else:
            print(f"  {name:<15} NOT FOUND")


if __name__ == "__main__":
    download_all()
