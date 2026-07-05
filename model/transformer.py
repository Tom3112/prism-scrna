"""
BERT-style transformer encoder for scRNA data.

Genes are treated as tokens; expression rank encodes positional information.

Two gene embedding modes:
  use_gnn=False (default) : plain nn.Embedding — learns from pretraining data only
  use_gnn=True            : GeneGAT over STRING PPI — biologically-informed prior
"""

from __future__ import annotations

import torch
import torch.nn as nn

SPECIAL_TOKENS = 3  # [PAD]=0, [CLS]=1, [MASK]=2


class scRNAEncoder(nn.Module):
    """
    Args:
        vocab_size  : number of HVGs + 3 special tokens.
        hidden_dim  : embedding / model dimension.
        num_layers  : number of TransformerEncoder layers.
        num_heads   : number of attention heads.
        ffn_dim     : feed-forward hidden dimension (default hidden_dim * 4).
        dropout     : dropout probability.
        max_seq_len : maximum sequence length (including [CLS]).
        pad_token_id: token id used for padding (default 0).
        gene_gat    : optional GeneGAT instance; if provided, replaces nn.Embedding
                      for gene tokens (use_gnn=True path).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
        gene_gat=None,          # GeneGAT | None
        n_expr_bins: int | None = None,  # set for "expr_bin" tokenization
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4
        self.hidden_dim   = hidden_dim
        self.pad_token_id = pad_token_id
        self.use_gnn      = gene_gat is not None

        # bins 0..n_expr_bins (n_expr_bins + 1 values) plus one MASK_BIN sentinel
        self.bin_emb = (
            nn.Embedding(n_expr_bins + 2, hidden_dim) if n_expr_bins is not None else None
        )

        if self.use_gnn:
            # GNN path: GAT produces gene embeddings; special tokens use a small lookup
            self.gene_gat    = gene_gat
            self.special_emb = nn.Embedding(SPECIAL_TOKENS, hidden_dim)
            self.token_emb   = None
        else:
            # Baseline path: plain learned embedding table
            self.token_emb   = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
            self.gene_gat    = None
            self.special_emb = None

        # Rank/position embedding — encodes expression rank as a biological prior
        self.pos_emb  = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_norm = nn.LayerNorm(hidden_dim)
        self.emb_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    # ------------------------------------------------------------------
    # Gene embedding lookup — switches between GNN and plain paths
    # ------------------------------------------------------------------

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, L, hidden_dim) token embeddings.

        Baseline : token_emb[input_ids]
        GNN      : [special_emb || gene_gat()] indexed by input_ids
        """
        if not self.use_gnn:
            return self.token_emb(input_ids)

        # GNN path — rebuild full vocab embedding matrix each forward pass
        gene_embs  = self.gene_gat()                                        # (n_genes, D)
        all_embs   = torch.cat([self.special_emb.weight, gene_embs], dim=0) # (vocab_size, D)
        return all_embs[input_ids]                                          # (B, L, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,   # (B, L), 1=real 0=pad
        bin_ids: torch.Tensor | None = None,  # (B, L), only for "expr_bin" tokenization
    ) -> torch.Tensor:
        """Returns (B, L, hidden_dim) hidden states."""
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)

        x = self._embed_tokens(input_ids) + self.pos_emb(positions)        # (B, L, D)
        if self.bin_emb is not None:
            assert bin_ids is not None, "bin_ids required when n_expr_bins is set"
            x = x + self.bin_emb(bin_ids)
        x = self.emb_drop(self.emb_norm(x))

        pad_mask = attention_mask == 0                                      # True where padded
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x

    def get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bin_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the [CLS] token embedding (B, hidden_dim)."""
        return self.forward(input_ids, attention_mask, bin_ids)[:, 0, :]
