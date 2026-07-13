"""
Download the 7 standard cell-type annotation benchmark datasets from Zenodo.

Source: Abdelaal et al. (2019) "A comparison of automatic cell identification methods
        for single-cell RNA sequencing data." Genome Biology.
Zenodo: https://doi.org/10.5281/zenodo.3357167

Datasets included (all from the same Zenodo archive):
  - BaronHuman   : human pancreas,    8,569 cells,  14 types
  - BaronMouse   : mouse pancreas,    1,886 cells,  13 types
  - AMB          : mouse visual cortex, 12,832 cells, 22 types (Subclass column)
  - Zheng68K     : human PBMC,       65,943 cells,  11 types
  - Zhengsorted  : human PBMC (FACS), 20,000 cells, 10 types
  - Segerstolpe  : human pancreas,    2,133 cells,  13 types
  - Muraro       : human pancreas,    2,122 cells,   9 types

Zeisel, Macosko, and Klein are NOT part of this Zenodo record (verified by
listing its full archive contents) and are not fetched here — see README.md's
Benchmarks section for why they were dropped from the comparison entirely
rather than reported without a source.

Usage:
    uv run python data/download_benchmarks.py
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ZENODO_URL = "https://zenodo.org/records/3357167/files/scRNAseq_Benchmark_datasets.zip?download=1"
BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks")
ZIP_CACHE = os.path.join(BENCH_DIR, "_zenodo_raw.zip")

CHUNK_ROWS = 2000  # rows per chunk when streaming large CSVs into a sparse matrix

# Real paths inside the Zenodo zip (verified by listing the archive — the
# top-level dataset names in DATASETS/CANONICAL below don't appear verbatim as
# folder names; several have spaces or live under Pancreatic_data/).
DATASET_SPECS = {
    "BaronHuman": dict(
        out="baron_human.h5ad",
        dir="Intra-dataset/Pancreatic_data/Baron Human",
        expr="Filtered_Baron_HumanPancreas_data.csv",
        labels="Labels.csv",
    ),
    "BaronMouse": dict(
        out="baron_mouse.h5ad",
        dir="Intra-dataset/Pancreatic_data/Baron Mouse",
        expr="Filtered_MousePancreas_data.csv",
        labels="Labels.csv",
    ),
    "AMB": dict(
        out="amb.h5ad",
        dir="Intra-dataset/AMB",
        expr="Filtered_mouse_allen_brain_data.csv",
        labels="Labels.csv",
        label_col="Subclass",  # Class=4, Subclass=22, cluster=110 — 22 matches our doc
    ),
    "Zheng68K": dict(
        out="zheng68k.h5ad",
        dir="Intra-dataset/Zheng 68K",
        expr="Filtered_68K_PBMC_data.csv",
        labels="Labels.csv",
    ),
    "Zhengsorted": dict(
        out="zhengsorted.h5ad",
        dir="Intra-dataset/Zheng sorted",
        expr="Filtered_DownSampled_SortedPBMC_data.csv",
        labels="Labels.csv",
    ),
    "Segerstolpe": dict(
        out="segerstolpe.h5ad",
        dir="Intra-dataset/Pancreatic_data/Segerstolpe",
        expr="Filtered_Segerstolpe_HumanPancreas_data.csv",
        labels="Labels.csv",
    ),
    "Muraro": dict(
        out="muraro.h5ad",
        dir="Intra-dataset/Pancreatic_data/Muraro",
        expr="Filtered_Muraro_HumanPancreas_data.csv",
        labels="Labels.csv",
    ),
}


def _progress_hook(count, block_size, total_size):
    pct = min(100, int(count * block_size * 100 / total_size))
    print(f"\r  Downloading: {pct}%", end="", flush=True)


def _ensure_zip():
    import urllib.request

    os.makedirs(BENCH_DIR, exist_ok=True)
    if os.path.exists(ZIP_CACHE):
        print(f"Zenodo zip already cached at {ZIP_CACHE}")
        return
    print(f"Downloading Zenodo archive (~3.7 GB) to {ZIP_CACHE} ...")
    urllib.request.urlretrieve(ZENODO_URL, ZIP_CACHE, reporthook=_progress_hook)
    print()


def _read_expr_csv_sparse(path: str, chunksize: int = CHUNK_ROWS):
    """
    Stream a (cells x genes) CSV into a sparse CSR matrix in row chunks, so
    peak memory stays around one chunk's dense size instead of the whole
    matrix (some of these files are 65,943 x 20,387 dense floats — reading
    them whole with pandas' default float64 plus a float32 cast copy
    exceeds available RAM). Index column = cell barcode, header = gene name;
    rows are always cells for this dataset collection (not auto-detected —
    verified directly against known cell counts per dataset).
    """
    # Passing a bare dtype together with index_col confuses pandas' C parser
    # into trying to cast the (string) index column too — build an explicit
    # per-column dtype dict from the header instead, keyed by gene name.
    with open(path) as f:
        header = next(f).rstrip("\n").split(",")
    gene_cols = [c.strip('"') for c in header[1:]]
    dtype_map = {c: np.float32 for c in gene_cols}

    chunks = []
    obs_names: list[str] = []
    var_names = None
    for chunk in pd.read_csv(path, index_col=0, dtype=dtype_map, chunksize=chunksize):
        if var_names is None:
            var_names = chunk.columns.astype(str).tolist()
        chunks.append(sp.csr_matrix(chunk.values))
        obs_names.extend(chunk.index.astype(str).tolist())
    X = sp.vstack(chunks, format="csr")
    return X, obs_names, var_names


def _extract_datasets():
    with zipfile.ZipFile(ZIP_CACHE, "r") as zf:
        for dataset, spec in DATASET_SPECS.items():
            out_path = os.path.join(BENCH_DIR, spec["out"])
            if os.path.exists(out_path):
                print(f"  {spec['out']} already exists — skipping.")
                continue

            expr_zip_path = f"{spec['dir']}/{spec['expr']}"
            labels_zip_path = f"{spec['dir']}/{spec['labels']}"

            print(f"  Extracting {dataset}...")
            tmp_dir = os.path.join(BENCH_DIR, f"_tmp_{dataset}")
            os.makedirs(tmp_dir, exist_ok=True)
            zf.extract(expr_zip_path, tmp_dir)
            zf.extract(labels_zip_path, tmp_dir)

            expr_file = os.path.join(tmp_dir, expr_zip_path)
            labels_file = os.path.join(tmp_dir, labels_zip_path)

            print(f"    Reading {spec['expr']} (chunked, sparse)...")
            X, obs_names, var_names = _read_expr_csv_sparse(expr_file)

            adata = ad.AnnData(
                X=X,
                obs=pd.DataFrame(index=obs_names),
                var=pd.DataFrame(index=var_names),
            )

            labels_df = pd.read_csv(labels_file)
            label_col = spec.get("label_col", labels_df.columns[0])
            adata.obs["cell_type"] = labels_df[label_col].astype("category").values

            adata.write_h5ad(out_path)
            n_types = adata.obs["cell_type"].nunique()
            print(f"  Saved {adata.n_obs:,} cells x {adata.n_vars:,} genes, "
                  f"{n_types} types -> {spec['out']}")

            shutil.rmtree(tmp_dir)


def download_all():
    _ensure_zip()
    print("Extracting target datasets...")
    _extract_datasets()
    print("\nDone. Available benchmark datasets:")
    for dataset, spec in DATASET_SPECS.items():
        path = os.path.join(BENCH_DIR, spec["out"])
        if os.path.exists(path):
            adata = ad.read_h5ad(path)
            n_types = adata.obs["cell_type"].nunique()
            print(f"  {dataset:<15} {adata.n_obs:>7,} cells  {adata.n_vars:>6,} genes  "
                  f"{n_types:>3} types")
        else:
            print(f"  {dataset:<15} NOT FOUND")


if __name__ == "__main__":
    download_all()
