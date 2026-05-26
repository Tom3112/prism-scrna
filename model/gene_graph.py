"""
Build a gene-gene interaction graph from the STRING PPI database.

Nodes  = our 2,000 HVGs
Edges  = STRING protein-protein interactions (high confidence ≥ min_score)
Weights = normalised STRING combined score (0–1)

The graph is fetched via the STRING API and cached locally so subsequent runs
are instant. Falls back to a co-expression graph if STRING is unreachable.
"""

from __future__ import annotations

import os
import numpy as np
import torch

SPECIAL_TOKENS = 3  # [PAD], [CLS], [MASK]
STRING_API = "https://string-db.org/api/tsv/network"
STRING_SPECIES = 9606  # Homo sapiens


# ---------------------------------------------------------------------------
# STRING fetch
# ---------------------------------------------------------------------------

def _fetch_string(gene_names: list[str], min_score: int, cache_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (src, dst, weight) numpy arrays for edges between gene_names.
    Results are cached to cache_path as a .npz file.
    """
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"Loaded STRING graph from cache ({data['src'].shape[0]} edges).")
        return data["src"], data["dst"], data["weight"]

    print("Fetching STRING PPI network (this may take ~30s)...")
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required: uv add requests")

    gene_set = set(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    response = requests.post(
        STRING_API,
        data={
            "identifiers": "\r".join(gene_names),
            "species": STRING_SPECIES,
            "required_score": min_score,
            "caller_identity": "prism_portfolio",
        },
        timeout=120,
    )
    response.raise_for_status()

    src_list, dst_list, w_list = [], [], []
    for line in response.text.strip().splitlines()[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        gene_a, gene_b = parts[2], parts[3]
        score = float(parts[5])
        if gene_a in gene_set and gene_b in gene_set and gene_a != gene_b:
            i, j = gene_to_idx[gene_a], gene_to_idx[gene_b]
            # Undirected: add both directions
            src_list += [i, j]
            dst_list += [j, i]
            w_list   += [score / 1000.0, score / 1000.0]

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    weight = np.array(w_list, dtype=np.float32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, src=src, dst=dst, weight=weight)
    print(f"STRING graph: {len(gene_names)} genes, {len(src)//2} unique interactions "
          f"(min_score={min_score}). Cached to {cache_path}.")
    return src, dst, weight


# ---------------------------------------------------------------------------
# Co-expression fallback
# ---------------------------------------------------------------------------

def _build_coexpr_graph(
    processed_path: str,
    gene_names: list[str],
    threshold: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fallback: build gene-gene graph from Pearson correlation of expression.
    Edges = |corr| > threshold.
    """
    print(f"Falling back to co-expression graph (|r| > {threshold})...")
    import anndata as ad
    import scipy.sparse as sp

    adata = ad.read_h5ad(processed_path)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Compute correlation matrix (n_genes × n_genes)
    X_c = X - X.mean(0, keepdims=True)
    norms = np.linalg.norm(X_c, axis=0, keepdims=True) + 1e-8
    X_c = X_c / norms
    corr = (X_c.T @ X_c) / X.shape[0]  # (G, G)

    # Threshold
    mask = (np.abs(corr) > threshold) & ~np.eye(len(gene_names), dtype=bool)
    src_arr, dst_arr = np.where(mask)
    w_arr = corr[src_arr, dst_arr].clip(0, 1).astype(np.float32)

    print(f"Co-expression graph: {len(src_arr)//2} unique edges (threshold={threshold}).")
    return src_arr.astype(np.int64), dst_arr.astype(np.int64), w_arr


# ---------------------------------------------------------------------------
# Add self-loops so every gene participates in message passing
# ---------------------------------------------------------------------------

def _add_self_loops(
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    n_genes: int,
    self_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loop_idx = np.arange(n_genes, dtype=np.int64)
    loop_w   = np.full(n_genes, self_weight, dtype=np.float32)
    src = np.concatenate([src, loop_idx])
    dst = np.concatenate([dst, loop_idx])
    weight = np.concatenate([weight, loop_w])
    return src, dst, weight


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gene_graph(
    gene_names: list[str],
    cache_dir: str,
    min_score: int = 700,
    processed_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (edge_index, edge_weight) tensors for the gene PPI graph.

    edge_index : (2, E) long tensor — [src_indices; dst_indices]
    edge_weight: (E,)  float tensor — normalised STRING scores

    Gene indices are 0-based (NOT offset by SPECIAL_TOKENS).
    The caller is responsible for the token offset when embedding.
    """
    cache_path = os.path.join(cache_dir, f"string_graph_score{min_score}.npz")
    n_genes = len(gene_names)

    try:
        src, dst, weight = _fetch_string(gene_names, min_score, cache_path)
    except Exception as e:
        print(f"STRING fetch failed ({e}). Using co-expression fallback.")
        if processed_path is None:
            raise ValueError("processed_path required for co-expression fallback.")
        src, dst, weight = _build_coexpr_graph(processed_path, gene_names)

    src, dst, weight = _add_self_loops(src, dst, weight, n_genes)

    edge_index  = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(weight, dtype=torch.float32)
    return edge_index, edge_weight
