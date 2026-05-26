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
    """

    def __init__(
        self,
        adata: ad.AnnData,
        max_seq_len: int = 2048,
        mask_ratio: float = MASK_RATIO,
        mode: str = "pretrain",
        label_col: str = "cell_type",
    ):
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.n_genes = adata.n_vars  # vocabulary size before special tokens

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

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict:
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
            input_ids, labels = self._apply_masking(input_ids, seq_len)
        else:
            labels = np.full(self.max_seq_len, -100, dtype=np.int64)

        out = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

        if self.mode == "finetune" and self.cell_type_labels is not None:
            out["cell_type"] = torch.tensor(self.cell_type_labels[idx], dtype=torch.long)

        return out

    def _apply_masking(
        self, input_ids: np.ndarray, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly mask mask_ratio of non-[CLS] gene tokens.
        Returns modified input_ids and labels (-100 for unmasked positions).
        """
        labels = np.full_like(input_ids, -100)
        # Positions 1..seq_len are maskable (skip [CLS] at 0 and padding)
        maskable = np.arange(1, seq_len + 1)
        n_mask = max(1, int(len(maskable) * self.mask_ratio))
        mask_positions = np.random.choice(maskable, size=n_mask, replace=False)

        labels[mask_positions] = input_ids[mask_positions]
        input_ids = input_ids.copy()
        input_ids[mask_positions] = MASK_TOKEN
        return input_ids, labels


def load_datasets(
    processed_path: str,
    max_seq_len: int = 2048,
    mask_ratio: float = MASK_RATIO,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Load processed.h5ad and return train/val/test splits as Dataset objects.
    Returns a dict with keys: pretrain_train, pretrain_val,
                               finetune_train, finetune_val, finetune_test,
                               num_classes, label_names, vocab_size.
    """
    adata = ad.read_h5ad(processed_path)
    n = adata.n_obs
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train: n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def _subset(indices):
        return adata[indices].copy()

    ds_kwargs = dict(max_seq_len=max_seq_len, mask_ratio=mask_ratio)

    pt_train = scRNADataset(_subset(train_idx), mode="pretrain", **ds_kwargs)
    pt_val = scRNADataset(_subset(val_idx), mode="pretrain", **ds_kwargs)

    ft_train = scRNADataset(_subset(train_idx), mode="finetune", **ds_kwargs)
    ft_val = scRNADataset(_subset(val_idx), mode="finetune", **ds_kwargs)
    ft_test = scRNADataset(_subset(test_idx), mode="finetune", **ds_kwargs)

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
