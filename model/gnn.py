"""
Graph Attention Network for gene embeddings.

GeneGAT runs over the STRING PPI gene graph and produces one context-aware
embedding per gene. These replace the plain nn.Embedding lookup in scRNAEncoder.

No external GNN library required — uses only base PyTorch scatter operations.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scatter softmax (MPS-safe, no torch_geometric required)
# ---------------------------------------------------------------------------

def _scatter_softmax(
    scores: torch.Tensor,   # (E, H) or (E,)
    dst: torch.Tensor,      # (E,) destination node indices
    n_nodes: int,
) -> torch.Tensor:
    """
    Per-node softmax over incoming edge scores.
    Uses a global max shift — safe on all PyTorch backends including MPS.
    """
    is_1d = scores.dim() == 1
    if is_1d:
        scores = scores.unsqueeze(-1)   # (E, 1)

    H = scores.size(1)

    # Global max shift for numerical stability
    shift = scores.detach().max()
    exp_scores = (scores - shift).exp()                             # (E, H)

    denom = torch.zeros(n_nodes, H, device=scores.device, dtype=scores.dtype)
    denom.scatter_add_(0, dst.unsqueeze(1).expand(-1, H), exp_scores)  # (N, H)

    alpha = exp_scores / (denom[dst] + 1e-8)                       # (E, H)
    return alpha.squeeze(-1) if is_1d else alpha


# ---------------------------------------------------------------------------
# Single GAT layer
# ---------------------------------------------------------------------------

class GATConv(nn.Module):
    """
    Multi-head sparse Graph Attention layer.

    Follows the original Velickovic et al. (2018) formulation:
        e_ij  = LeakyReLU( a^T [ W h_i || W h_j ] )
        α_ij  = softmax over {j ∈ N(i)} of e_ij  ×  edge_weight_ij
        h_i'  = σ( Σ_j α_ij  W h_j )

    Heads are concatenated then projected back to out_dim.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads

        self.W       = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, n_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, n_heads, self.head_dim))
        self.leaky   = nn.LeakyReLU(negative_slope=0.2)
        self.drop    = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.att_src.view(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.att_dst.view(n_heads, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,            # (N, in_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_weight: torch.Tensor,  # (E,)
    ) -> torch.Tensor:              # (N, out_dim)
        N = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]  # each (E,)

        # Linear transform → (N, H, D)
        Wh = self.W(x).view(N, self.n_heads, self.head_dim)

        # Attention scores per edge per head: (E, H)
        e_src = (Wh[src_idx] * self.att_src).sum(-1)
        e_dst = (Wh[dst_idx] * self.att_dst).sum(-1)
        e = self.leaky(e_src + e_dst)                       # (E, H)

        # Scale by STRING confidence (higher confidence → stronger attention)
        e = e * edge_weight.unsqueeze(-1)                   # (E, H)

        # Softmax per destination node
        alpha = _scatter_softmax(e, dst_idx, N)             # (E, H)
        alpha = self.drop(alpha)

        # Weighted aggregation: out[dst] += alpha * Wh[src]
        out = torch.zeros(N, self.n_heads, self.head_dim,
                          device=x.device, dtype=x.dtype)
        idx = dst_idx.view(-1, 1, 1).expand(-1, self.n_heads, self.head_dim)
        out.scatter_add_(0, idx, alpha.unsqueeze(-1) * Wh[src_idx])

        return out.view(N, -1)      # (N, out_dim)


# ---------------------------------------------------------------------------
# GeneGAT — 2-layer GAT over the PPI graph
# ---------------------------------------------------------------------------

class GeneGAT(nn.Module):
    """
    Produces one embedding per gene using the STRING PPI graph.
    Replaces nn.Embedding(vocab_size, hidden_dim) for gene tokens.

    Architecture:
        learnable gene features  →  GATConv × n_layers (with residual + LN)  →  gene embeddings

    Args:
        n_genes    : number of HVGs (vocabulary size minus special tokens)
        hidden_dim : embedding dimension (matches transformer hidden_dim)
        edge_index : (2, E) graph connectivity (from gene_graph.build_gene_graph)
        edge_weight: (E,)  STRING confidence scores (0–1)
        n_layers   : number of GAT layers (default 2)
        n_heads    : attention heads per layer (default 4)
        dropout    : dropout in attention weights
        freeze     : if True, GAT weights are frozen after init (Option A1)
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        freeze: bool = False,
    ):
        super().__init__()
        self.n_genes    = n_genes
        self.hidden_dim = hidden_dim

        # Learnable initial features (analogous to an embedding table)
        self.gene_features = nn.Parameter(torch.empty(n_genes, hidden_dim))
        nn.init.normal_(self.gene_features, std=0.02)

        # Graph topology — registered as buffers (move with .to(device), not trainable)
        self.register_buffer("edge_index",  edge_index)
        self.register_buffer("edge_weight", edge_weight)

        # GAT layers with residual connections
        self.layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self) -> torch.Tensor:
        """Returns context-aware gene embeddings (n_genes, hidden_dim)."""
        x = self.gene_features
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x, self.edge_index, self.edge_weight))   # residual
        return x


def _build_knn_graph(embeddings: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    k-NN graph over a batch of embeddings via cosine similarity — directed
    edges from each node to its k nearest neighbors (self excluded).

    Edge weight is cosine similarity rescaled from [-1, 1] to [0, 1] so it
    plays the same role GATConv already expects from STRING confidence
    scores (higher weight -> stronger attention).

    Graph structure is built from detached embeddings (no gradient through
    which edges exist, only through the GAT's message values) — same
    convention as k-NN-graph-based GNN methods generally use.
    """
    B = embeddings.size(0)
    k = min(k, B - 1)
    if k <= 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=embeddings.device),
            torch.zeros(0, device=embeddings.device),
        )
    norm = F.normalize(embeddings.detach(), dim=-1)
    sim = norm @ norm.t()
    sim.fill_diagonal_(-float("inf"))
    topk_sim, topk_idx = sim.topk(k, dim=-1)
    src = torch.arange(B, device=embeddings.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = topk_idx.reshape(-1)
    weight = ((topk_sim.reshape(-1) + 1) / 2).clamp(0, 1)
    return torch.stack([src, dst]), weight


class CellCellGAT(nn.Module):
    """
    Option C: refines a batch of cell embeddings using a k-NN graph built
    from cosine similarity between cells IN THAT BATCH.

    This is the cell-level counterpart to GeneGAT/CellGAT, which both only
    ever model gene-gene structure — no PRISM variant models cell-cell
    structure at all until this one. It's a cheap approximation of
    scBiGNN's actual cell-level GNN, which builds one graph over the WHOLE
    dataset via an EM loop (pseudo-labels from a gene-level GNN determine
    which cells are "close"); this version rebuilds a small graph fresh
    every forward pass from whatever cells happen to share a batch, so
    quality depends on batch composition and there's no EM refinement.

    Args:
        hidden_dim: cell embedding dimension.
        k         : neighbors per cell (clamped to batch_size - 1).
        n_heads   : GAT attention heads.
        dropout   : dropout in attention weights.
    """

    def __init__(self, hidden_dim: int, k: int = 5, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.gat = GATConv(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, cell_emb: torch.Tensor) -> torch.Tensor:
        """cell_emb: (B, hidden_dim) -> (B, hidden_dim), residual-refined."""
        edge_index, edge_weight = _build_knn_graph(cell_emb, self.k)
        if edge_index.numel() == 0:
            return cell_emb
        delta = self.gat(cell_emb, edge_index, edge_weight)
        return self.norm(cell_emb + delta)


class GlobalCellGraph:
    """
    Full-dataset cell-cell k-NN graph, periodically refreshed — the "E-step"
    of an EM-style refinement loop. This is what CellCellGAT approximates at
    batch scale: instead of a k-NN graph over whichever cells happen to
    share a training minibatch, `refresh()` computes each cell's TRUE
    nearest neighbors across the ENTIRE training split, from a full-dataset
    embedding snapshot taken with the current encoder.

    Call `refresh(embeddings)` once before training and again periodically
    (e.g. every epoch) as the encoder improves — each refresh is the E-step
    (rebuild the graph from current understanding); the training steps that
    follow, using that fixed graph, are the M-step. Neighbor embeddings are
    cached (detached) between refreshes, so no gradient flows through
    "which cells are neighbors," same convention as `_build_knn_graph`.

    Not a drop-in nn.Module — this is plain host-side state, not a layer,
    since it needs to persist and mutate across an entire epoch rather than
    being reconstructed per forward pass.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.neighbor_emb: torch.Tensor | None = None  # (N, k, D), detached

    @torch.no_grad()
    def refresh(self, embeddings: torch.Tensor) -> None:
        """
        embeddings: (N, D) full-dataset snapshot, in dataset-index order
        (index i must correspond to the same cell scRNADataset.__getitem__(i)
        would return as "idx": i). Rebuilds neighbor cache via cosine
        similarity. Chunked to avoid an O(N^2) memory blowup on large
        datasets (e.g. Zheng68K's ~53k training cells).
        """
        N = embeddings.size(0)
        k = min(self.k, N - 1)
        norm = F.normalize(embeddings, dim=-1)
        chunk = 2048
        idx_chunks = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            sim = norm[start:end] @ norm.t()                     # (chunk, N)
            rows = torch.arange(end - start, device=embeddings.device)
            cols = torch.arange(start, end, device=embeddings.device)
            sim[rows, cols] = -float("inf")                       # exclude self
            _, topk = sim.topk(k, dim=-1)                          # (chunk, k)
            idx_chunks.append(topk)
        neighbor_idx = torch.cat(idx_chunks, dim=0)                # (N, k)
        self.neighbor_emb = embeddings[neighbor_idx].detach()      # (N, k, D)

    def get_neighbors(self, dataset_indices: torch.Tensor) -> torch.Tensor:
        """dataset_indices: (B,) -> (B, k, D) cached neighbor embeddings."""
        if self.neighbor_emb is None:
            raise RuntimeError("GlobalCellGraph.refresh() must be called before get_neighbors()")
        return self.neighbor_emb[dataset_indices.to(self.neighbor_emb.device)]


def build_gene_gat(
    model_cfg,
    processed_path: str,
    device: torch.device,
    species: int = 9606,
) -> GeneGAT:
    """
    Builds a GeneGAT wired to the STRING PPI graph for the genes in
    `processed_path`, using the GNN-related fields of ModelConfig (hidden_dim,
    gnn_layers, gnn_heads, gnn_freeze, string_min_score). Shared by
    pretrain.py, finetune.py, ablations.py, and benchmark_eval.py so all
    construct an identical architecture (required to load a checkpoint trained
    with use_gnn=True). species=9606 (human) default; pass 10090 for mouse
    datasets (BaronMouse, AMB) — STRING won't match mouse gene symbols against
    the human network.

    Reads gene names directly (backed mode — skips loading the expression
    matrix) rather than requiring the caller to have already loaded the adata.
    """
    import anndata as ad
    from model.gene_graph import build_gene_graph

    gene_names = list(ad.read_h5ad(processed_path, backed="r").var_names)

    edge_index, edge_weight = build_gene_graph(
        gene_names,
        cache_dir=os.path.dirname(processed_path),
        min_score=model_cfg.string_min_score,
        processed_path=processed_path,
        species=species,
    )
    return GeneGAT(
        n_genes=len(gene_names),
        hidden_dim=model_cfg.hidden_dim,
        edge_index=edge_index.to(device),
        edge_weight=edge_weight.to(device),
        n_layers=model_cfg.gnn_layers,
        n_heads=model_cfg.gnn_heads,
        dropout=model_cfg.dropout,
        freeze=model_cfg.gnn_freeze,
    ).to(device)
