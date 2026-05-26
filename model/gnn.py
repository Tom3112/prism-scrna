"""
Graph Attention Network for gene embeddings.

GeneGAT runs over the STRING PPI gene graph and produces one context-aware
embedding per gene. These replace the plain nn.Embedding lookup in scRNAEncoder.

No external GNN library required — uses only base PyTorch scatter operations.
"""

from __future__ import annotations

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
