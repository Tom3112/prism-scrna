"""
Download scRNA data for pretraining.

Uses scanpy.datasets.pbmc3k_processed() as a smoke-test dataset (~2.6k cells) —
the canonical scanpy clustering-tutorial PBMC3k with real expert-derived
louvain cell-type annotations (CD4 T, CD14+ Monocytes, B, CD8 T, NK,
FCGR3A+ Monocytes, Dendritic, Megakaryocytes), not unsupervised pseudo-labels.
For the full 10k PBMC dataset, set USE_10K=True and provide a valid
download directory; the script will fetch it from 10x Genomics.
"""

import os
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "raw.h5ad")
USE_10K = False  # flip to True to download the full 10k dataset


def download_pbmc3k() -> sc.AnnData:
    print("Loading PBMC 3k (processed, annotated) dataset via scanpy...")
    adata = sc.datasets.pbmc3k_processed()
    adata.write_h5ad(RAW_PATH)
    print(f"Saved {adata.n_obs} cells to {RAW_PATH}")
    return adata


def download_pbmc10k(out_path: str = RAW_PATH) -> None:
    """
    Downloads the 10x Genomics PBMC 10k filtered feature matrix.
    Requires `requests` and ~500 MB of disk space.
    """
    import urllib.request

    url = (
        "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
        "10k_PBMC_3p_nextgem_Chromium_X/10k_PBMC_3p_nextgem_Chromium_X"
        "_filtered_feature_bc_matrix.h5"
    )
    h5_path = out_path.replace(".h5ad", ".h5")
    print(f"Downloading 10k PBMC to {h5_path} ...")
    urllib.request.urlretrieve(url, h5_path)
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    adata.write_h5ad(out_path)
    print(f"Saved {adata.n_obs} cells to {out_path}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    if USE_10K:
        download_pbmc10k()
    else:
        download_pbmc3k()
