"""
Hyperparameters and path configuration for pretraining and fine-tuning.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# macOS + MPS requires num_workers=0 due to multiprocessing fork conflicts.
_DEFAULT_NUM_WORKERS = 0 if sys.platform == "darwin" else 4


@dataclass
class DataConfig:
    processed_path: str = os.path.join(ROOT, "data", "processed.h5ad")
    max_seq_len: int = 512
    mask_ratio: float = 0.15
    train_frac: float = 0.8
    val_frac: float = 0.1
    num_workers: int = _DEFAULT_NUM_WORKERS
    seed: int = 42
    # Tokenization scheme: "rank" (Geneformer-style) or "expr_bin" (scBERT-style)
    tokenization: str = "rank"
    n_bins: int = 10  # non-zero expression bins, only used for "expr_bin"


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 1024       # hidden_dim * 4
    dropout: float = 0.1
    # vocab_size is determined at runtime from the dataset

    # GNN gene embedding (Option A — gene tokens use GAT instead of nn.Embedding)
    use_gnn: bool = False               # False = plain nn.Embedding (baseline)
    gnn_layers: int = 2                 # GAT depth
    gnn_heads: int = 4                  # GAT attention heads
    gnn_freeze: bool = False            # True = Option A1 (frozen), False = A2 (joint)
    string_min_score: int = 700         # STRING confidence threshold (0–1000)

    # GNN classifier head (Option B — PPI graph used DURING classification)
    use_gat_head: bool = False          # False = CLS linear probe, True = CellGATHead
    gat_head_n_heads: int = 4           # attention heads in the classification GAT

    # Cell-cell GNN classifier head (Option C — batch-level cell k-NN graph)
    use_cell_graph: bool = False        # True = CellGraphClassificationHead
    cell_graph_k: int = 5               # neighbors per cell in the k-NN graph
    cell_graph_heads: int = 4           # attention heads in the cell-cell GAT

    # Cell-cell GNN, EM-refined (Option C2 — full-dataset graph, not batch-level)
    use_em_cell_graph: bool = False      # True = EMCellGraphClassificationHead
    em_graph_k: int = 5                  # neighbors per cell in the full-dataset graph
    em_graph_refresh_every: int = 1      # rebuild the graph every N epochs (the "E-step")


@dataclass
class PretrainConfig:
    # Paths
    checkpoint_dir: str = os.path.join(ROOT, "checkpoints")
    best_ckpt: str = os.path.join(ROOT, "checkpoints", "pretrain_best.pt")
    # Optimisation
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    # Reproducibility
    seed: int = 42
    # Device
    device: str = "cpu"       # overridden at runtime if GPU available


@dataclass
class FinetuneConfig:
    # Paths
    pretrain_ckpt: str = os.path.join(ROOT, "checkpoints", "pretrain_best.pt")
    checkpoint_dir: str = os.path.join(ROOT, "checkpoints")
    best_ckpt: str = os.path.join(ROOT, "checkpoints", "finetune_best.pt")
    # Optimisation
    batch_size: int = 64
    epochs: int = 15
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    # Encoder freezing strategy: "none" | "full" | "partial"
    freeze_encoder: str = "none"
    freeze_layers: int = 2    # used only when freeze_encoder == "partial"
    # Reproducibility
    seed: int = 42
    # Device
    device: str = "cpu"
