"""
Task-specific heads that attach to scRNAEncoder.

Two classification heads are provided:

  CellTypeClassificationHead  — baseline: linear probe on [CLS] token.
  CellGATClassificationHead   — GNN-as-classifier: uses the STRING PPI graph
      *during* the classification forward pass (not just embedding init).
      Gene hidden states are refined by a GATConv layer (PPI edges) then
      pooled via attention → cell embedding → linear head.
      This closes the architectural gap with scBiGNN-style methods.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.gnn import GATConv

_SPECIAL = 3   # [PAD]=0, [CLS]=1, [MASK]=2 — must match transformer.py


class MaskedGenePredictionHead(nn.Module):
    """
    Predicts the original gene token at masked positions.
    Loss is CrossEntropy computed only on positions where labels != -100.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        hidden: torch.Tensor,   # (B, L, hidden_dim)
        labels: torch.Tensor,   # (B, L), -100 for unmasked
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss, logits). logits shape: (B, L, vocab_size)."""
        logits = self.proj(self.norm(hidden))          # (B, L, vocab_size)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),           # (B*L, V)
            labels.view(-1),                            # (B*L,)
            ignore_index=-100,
        )
        return loss, logits


class CellTypeClassificationHead(nn.Module):
    """
    Classifies cells from the [CLS] token embedding.
    """

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        cls_emb: torch.Tensor,         # (B, hidden_dim)
        labels: torch.Tensor | None = None,  # (B,) integer class ids
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Returns (loss_or_None, logits). logits shape: (B, num_classes)."""
        logits = self.proj(self.drop(self.norm(cls_emb)))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits


class CellGATClassificationHead(nn.Module):
    """
    GNN-as-classifier: uses the STRING PPI graph *during* cell type classification.

    Closes the architectural gap with scBiGNN-style methods. Instead of reading
    only the [CLS] token, all gene hidden states from the transformer are refined
    by a GATConv layer (whose edges come from the PPI graph over the genes
    expressed in each cell), then pooled via learned attention to a cell embedding.

    Pipeline:
        gene_hidden (B, L, D)  ← transformer full hidden states
            ↓ build per-cell sparse PPI subgraph from input_ids
            ↓ GATConv (residual) — PPI edges between co-expressed genes
        refined gene states (B, L, D)
            ↓ attention-weighted pool: cell attends over its gene nodes
        cell_emb (B, D)
            ↓ LN → Dropout → Linear
        logits (B, num_classes)

    When no PPI edges exist (has_ppi=False), the head degenerates to attention
    pooling over gene states — still stronger than a bare [CLS] linear probe.

    Args:
        hidden_dim      : transformer hidden dimension.
        num_classes     : number of cell types.
        dropout         : dropout rate.
        n_genes         : HVG vocabulary size (vocab_size - 3 special tokens).
        ppi_edge_index  : (2, E) long tensor, gene indices 0..n_genes-1.
        ppi_edge_weight : (E,) float tensor, STRING confidence (0–1).
        n_gat_heads     : number of attention heads in the GAT layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        n_genes: int,
        ppi_edge_index: torch.Tensor | None = None,
        ppi_edge_weight: torch.Tensor | None = None,
        n_gat_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_genes    = n_genes

        # Filter PPI edges to genes within our HVG vocabulary
        if ppi_edge_index is not None and ppi_edge_index.numel() > 0:
            valid = (ppi_edge_index[0] < n_genes) & (ppi_edge_index[1] < n_genes)
            self.register_buffer("ppi_src", ppi_edge_index[0][valid])
            self.register_buffer("ppi_dst", ppi_edge_index[1][valid])
            self.register_buffer("ppi_w",   ppi_edge_weight[valid])
            self.has_ppi = True
            self.gat = GATConv(hidden_dim, hidden_dim, n_heads=n_gat_heads, dropout=dropout)
        else:
            self.has_ppi = False

        # Attention pooling: cell query from mean gene state → attends over gene keys
        self.attn_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, num_classes)

    def _build_batch_graph(
        self,
        input_ids: torch.Tensor,  # (B, L)
        gene_mask: torch.Tensor,  # (B, L) bool — True at valid gene positions
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Constructs a batched sparse PPI graph over expressed gene tokens.

        For every PPI edge (g_src, g_dst) and every cell b in the batch:
          if both g_src and g_dst appear in cell b's sequence → add the edge.

        Nodes are identified by their flat position in (B × L) space:
          node (b, k)  →  flat index b*L + k

        Returns:
            src_flat  : (E_batch,) flat source node indices
            dst_flat  : (E_batch,) flat destination node indices
            w         : (E_batch,) PPI edge weights
        """
        B, L   = input_ids.shape
        device = input_ids.device

        # gene_ids[b, k] = 0-based gene index if a gene token, else -1
        gene_ids = torch.where(
            gene_mask,
            input_ids - _SPECIAL,
            input_ids.new_full((), -1),
        )  # (B, L)

        # pos_table[b, g] = sequence position k of gene g in cell b; -1 if absent
        pos_table = input_ids.new_full((B, self.n_genes), -1)
        gm_nz        = gene_mask.nonzero(as_tuple=False)    # (N_valid, 2)
        b_v, k_v     = gm_nz[:, 0], gm_nz[:, 1]
        g_v          = gene_ids[b_v, k_v]
        pos_table[b_v, g_v] = k_v

        # For each PPI edge, find which cells have both endpoint genes
        pos_src = pos_table[:, self.ppi_src]   # (B, E_ppi)
        pos_dst = pos_table[:, self.ppi_dst]   # (B, E_ppi)

        valid    = (pos_src >= 0) & (pos_dst >= 0)      # (B, E_ppi)
        b_e, e_e = valid.nonzero(as_tuple=True)          # (E_batch,) each

        src_flat = b_e * L + pos_src[b_e, e_e]
        dst_flat = b_e * L + pos_dst[b_e, e_e]
        w        = self.ppi_w[e_e]
        return src_flat, dst_flat, w

    def forward(
        self,
        hidden: torch.Tensor,          # (B, L, D) — full transformer output
        attention_mask: torch.Tensor,  # (B, L), 1=real 0=pad
        input_ids: torch.Tensor,       # (B, L) — needed to identify gene positions
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Returns (loss_or_None, logits). logits shape: (B, num_classes)."""
        B, L, D = hidden.shape

        # gene_mask: True at positions that hold a real gene token
        gene_mask = attention_mask.bool() & (input_ids >= _SPECIAL)  # (B, L)

        # ── PPI-guided GAT refinement ──────────────────────────────────────
        if self.has_ppi:
            src_f, dst_f, w = self._build_batch_graph(input_ids, gene_mask)
            if src_f.numel() > 0:
                h_flat   = hidden.reshape(B * L, D)
                edge_idx = torch.stack([src_f, dst_f])         # (2, E_batch)
                delta    = self.gat(h_flat, edge_idx, w)       # (B*L, D)
                hidden   = (h_flat + delta).view(B, L, D)      # residual

        # ── Attention-weighted pooling: cell attends over gene nodes ───────
        gf       = gene_mask.float().unsqueeze(-1)                       # (B, L, 1)
        mean_g   = (hidden * gf).sum(1) / gf.sum(1).clamp(min=1)        # (B, D)

        q        = self.attn_q(mean_g).unsqueeze(1)                      # (B, 1, D)
        k        = self.attn_k(hidden)                                   # (B, L, D)
        scores   = (q * k).sum(-1) / D ** 0.5                           # (B, L)
        scores   = scores.masked_fill(~gene_mask, float("-inf"))
        alpha    = torch.softmax(scores, dim=-1)                         # (B, L)

        cell_emb = (alpha.unsqueeze(-1) * hidden).sum(1)                 # (B, D)
        logits   = self.proj(self.drop(self.norm(cell_emb)))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits
