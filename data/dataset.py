"""
PyTorch Dataset: tokenize cells into rank-ordered gene sequences, following the Geneformer approach.

Each cell becomes a sequence of gene indices sorted by descending expression.
During pretraining, 15% of positions are randomly masked.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
import anndata as ad
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

MASK_RATIO = 0.15
PAD_TOKEN = 0   # index 0 reserved for [PAD]
CLS_TOKEN = 1   # index 1 reserved for [CLS]
MASK_TOKEN = 2  # index 2 reserved for [MASK]
SPECIAL_TOKENS = 3  # number of special tokens prepended to gene vocab


class scRNADataset(Dataset):
    """
    Args:
        adata: preprocessed AnnData (cells × HVGs, log-normalised).
        max_seq_len: maximum number of gene tokens (including [CLS]).
        mask_ratio: fraction of tokens to mask during pretraining.
        mode: "pretrain" returns (input_ids, attention_mask, labels);
              "finetune" also returns cell-type label.
        label_col: obs column name for cell type labels.
        tokenization: "rank" (default, Geneformer-style: genes ordered by
            descending expression, order encodes magnitude) or "expr_bin"
            (scBERT-style: fixed gene panel shared by every cell, expression
            discretized into quantile bins, order carries no information).
        n_bins: number of non-zero expression bins for "expr_bin" mode.
        panel: precomputed fixed gene-index panel (from the training split) for
            "expr_bin" mode, reused for val/test so every split shares the same
            gene set. If None, computed from this adata.
        bin_edges: precomputed quantile edges (n_bins - 1,) for "expr_bin"
            mode, e.g. reused from the training split for val/test to avoid
            leakage. If None, computed from this adata.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        max_seq_len: int = 2048,
        mask_ratio: float = MASK_RATIO,
        mode: str = "pretrain",
        label_col: str = "cell_type",
        tokenization: str = "rank",
        n_bins: int = 10,
        panel: np.ndarray | None = None,
        bin_edges: np.ndarray | None = None,
        index_offset: int = 0,
    ):
        if tokenization not in ("rank", "expr_bin"):
            raise ValueError(f"Unknown tokenization: {tokenization!r}")
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.tokenization = tokenization
        self.n_bins = n_bins
        self.n_genes = adata.n_vars  # vocabulary size before special tokens
        # Added to __getitem__'s "idx" — lets two datasets built from disjoint
        # splits (e.g. train/test) share one combined index space, so a
        # GlobalCellGraph built over both can be queried consistently
        # regardless of which split a given cell came from.
        self.index_offset = index_offset

        # Convert to dense numpy array for fast per-cell access
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        self.X = X.astype(np.float32)

        # Cell-type integer labels (None in unsupervised mode)
        if mode == "finetune" and label_col in adata.obs.columns:
            cats = adata.obs[label_col].astype("category")
            self.cell_type_labels = cats.cat.codes.to_numpy()
            self.label_names = list(cats.cat.categories)
            self.num_classes = len(self.label_names)
        else:
            self.cell_type_labels = None
            self.num_classes = None

        # vocab_size = n_genes + SPECIAL_TOKENS; gene i maps to token (i + SPECIAL_TOKENS)
        self.vocab_size = self.n_genes + SPECIAL_TOKENS

        if self.tokenization == "expr_bin":
            if panel is not None:
                self.panel = panel
            else:
                # Fixed gene panel shared by every cell: top (max_seq_len - 1)
                # genes by mean expression, in gene-index order (no rank info).
                panel_size = min(self.n_genes, self.max_seq_len - 1)
                self.panel = np.argsort(-self.X.mean(axis=0))[:panel_size]
                self.panel.sort()  # canonical gene-index order

            if bin_edges is not None:
                self.bin_edges = bin_edges
            else:
                nonzero = self.X[:, self.panel]
                nonzero = nonzero[nonzero > 0]
                if nonzero.size == 0:
                    self.bin_edges = np.linspace(0, 1, n_bins - 1)
                else:
                    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
                    self.bin_edges = np.percentile(nonzero, quantiles)

            # Gene identity and padding are identical for every cell in this
            # mode (fixed panel) — precompute once rather than rebuilding on
            # every __getitem__ call.
            self._panel_size = len(self.panel)
            pad_len = (self.max_seq_len - 1) - self._panel_size
            gene_tokens = self.panel + SPECIAL_TOKENS
            self._base_input_ids = np.concatenate(
                [[CLS_TOKEN], gene_tokens, np.full(pad_len, PAD_TOKEN)]
            ).astype(np.int64)
            self._base_attention_mask = np.concatenate(
                [np.ones(self._panel_size + 1), np.zeros(pad_len)]
            ).astype(np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict:
        if self.tokenization == "expr_bin":
            return self._getitem_expr_bin(idx)
        return self._getitem_rank(idx)

    def _getitem_rank(self, idx: int) -> dict:
        expr = self.X[idx]  # (n_genes,)

        # Rank genes by descending expression; only keep expressed genes
        expressed = np.where(expr > 0)[0]
        order = expressed[np.argsort(-expr[expressed])]

        # Map gene indices to token ids (offset by SPECIAL_TOKENS)
        gene_tokens = (order + SPECIAL_TOKENS).tolist()

        # Truncate to (max_seq_len - 1) to leave room for [CLS]
        gene_tokens = gene_tokens[: self.max_seq_len - 1]
        seq_len = len(gene_tokens)

        # Build input: [CLS] + gene tokens + [PAD] * padding
        pad_len = (self.max_seq_len - 1) - seq_len
        input_ids = [CLS_TOKEN] + gene_tokens + [PAD_TOKEN] * pad_len
        attention_mask = [1] * (seq_len + 1) + [0] * pad_len

        input_ids = np.array(input_ids, dtype=np.int64)
        attention_mask = np.array(attention_mask, dtype=np.int64)

        if self.mode == "pretrain":
            input_ids, labels = self._apply_masking(input_ids, seq_len, MASK_TOKEN)
        else:
            labels = np.full(self.max_seq_len, -100, dtype=np.int64)

        out = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "idx": torch.tensor(idx + self.index_offset, dtype=torch.long),
        }

        if self.mode == "finetune" and self.cell_type_labels is not None:
            out["cell_type"] = torch.tensor(self.cell_type_labels[idx], dtype=torch.long)

        return out

    def _getitem_expr_bin(self, idx: int) -> dict:
        expr = self.X[idx][self.panel]  # (panel_size,), fixed gene order

        # bin 0 = not expressed; bins 1..n_bins = quantile bins of nonzero expression
        bin_ids = np.digitize(expr, self.bin_edges) + 1
        bin_ids = np.where(expr > 0, bin_ids, 0)
        pad_len = (self.max_seq_len - 1) - self._panel_size
        bin_ids = np.pad(bin_ids, (1, pad_len), constant_values=0).astype(np.int64)

        input_ids = self._base_input_ids
        attention_mask = self._base_attention_mask

        if self.mode == "pretrain":
            # Gene identity at each position is fixed and known (same panel for
            # every cell), so masking input_ids would be trivially recoverable
            # from the position embedding alone. Instead mask the expression
            # bin — the only cell-specific information in this tokenization —
            # and predict the original bin id at masked positions.
            bin_ids, labels = self._apply_masking(bin_ids, self._panel_size, self.n_bins + 1)
        else:
            labels = np.full(self.max_seq_len, -100, dtype=np.int64)

        out = {
            "input_ids": torch.tensor(input_ids),
            "bin_ids": torch.tensor(bin_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "idx": torch.tensor(idx + self.index_offset, dtype=torch.long),
        }

        if self.mode == "finetune" and self.cell_type_labels is not None:
            out["cell_type"] = torch.tensor(self.cell_type_labels[idx], dtype=torch.long)

        return out

    def _apply_masking(
        self, array: np.ndarray, seq_len: int, mask_value: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly mask mask_ratio of non-[CLS] positions in `array` (gene
        tokens for rank mode, expression bins for expr_bin mode), replacing
        masked entries with `mask_value`.
        Returns the modified array and labels (-100 for unmasked positions,
        the original value at masked positions).
        """
        labels = np.full_like(array, -100)
        # Positions 1..seq_len are maskable (skip [CLS] at 0 and padding)
        maskable = np.arange(1, seq_len + 1)
        n_mask = max(1, int(len(maskable) * self.mask_ratio))
        mask_positions = np.random.choice(maskable, size=n_mask, replace=False)

        labels[mask_positions] = array[mask_positions]
        array = array.copy()
        array[mask_positions] = mask_value
        return array, labels


def load_datasets(
    processed_path: str,
    max_seq_len: int = 2048,
    mask_ratio: float = MASK_RATIO,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    tokenization: str = "rank",
    n_bins: int = 10,
    label_col: str = "cell_type",
) -> dict:
    """
    Load processed.h5ad and return train/val/test splits as Dataset objects.
    Returns a dict with keys: pretrain_train, pretrain_val,
                               finetune_train, finetune_val, finetune_test,
                               num_classes, label_names, vocab_size.

    For tokenization="expr_bin", the gene panel and bin edges are fit on the
    train split and reused for val/test to avoid leakage.

    The split is stratified by `label_col` when that column is present, so
    every class — including small ones — is proportionally represented in
    train/val/test rather than left to chance. Falls back to a plain random
    split if `label_col` is absent (e.g. unlabeled pretrain-only data).
    """
    adata = ad.read_h5ad(processed_path)
    n = adata.n_obs

    if label_col in adata.obs.columns:
        labels = adata.obs[label_col].to_numpy()
        all_idx = np.arange(n)
        train_idx, rest_idx = train_test_split(
            all_idx, train_size=train_frac, stratify=labels, random_state=seed,
        )
        # val_frac and the remainder (1 - train_frac) are both fractions of
        # the full n, so re-express val_frac as a fraction of what's left.
        val_frac_of_rest = val_frac / (1 - train_frac)
        val_idx, test_idx = train_test_split(
            rest_idx, train_size=val_frac_of_rest,
            stratify=labels[rest_idx], random_state=seed,
        )
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx = idx[:n_train]
        val_idx = idx[n_train: n_train + n_val]
        test_idx = idx[n_train + n_val:]

    def _subset(indices):
        return adata[indices].copy()

    ds_kwargs = dict(
        max_seq_len=max_seq_len, mask_ratio=mask_ratio,
        tokenization=tokenization, n_bins=n_bins,
    )

    pt_train = scRNADataset(_subset(train_idx), mode="pretrain", **ds_kwargs)
    shared = dict(ds_kwargs)
    if tokenization == "expr_bin":
        shared["panel"] = pt_train.panel
        shared["bin_edges"] = pt_train.bin_edges

    pt_val = scRNADataset(_subset(val_idx), mode="pretrain", **shared)

    ft_train = scRNADataset(_subset(train_idx), mode="finetune", **ds_kwargs)
    ft_val = scRNADataset(_subset(val_idx), mode="finetune", **shared)
    ft_test = scRNADataset(_subset(test_idx), mode="finetune", **shared)

    return {
        "pretrain_train": pt_train,
        "pretrain_val": pt_val,
        "finetune_train": ft_train,
        "finetune_val": ft_val,
        "finetune_test": ft_test,
        "num_classes": ft_train.num_classes,
        "label_names": ft_train.label_names,
        "vocab_size": pt_train.vocab_size,
    }
