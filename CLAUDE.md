# CLAUDE.md — PRISM

## Python environment

Always run Python via `uv run python` — never call `python` or `python3` directly.
The virtual environment is at `.venv` (Python 3.12, created with `uv sync`).

## Running the pipeline

```bash
# 1. Download data
uv run python data/download.py

# 2. Preprocess
uv run python data/preprocess.py

# 3. Pretrain
uv run python train/pretrain.py

# 4. Fine-tune
uv run python train/finetune.py

# 5. Full experiment with logging and report
uv run python run_experiment.py
```

Checkpoints are saved to `checkpoints/`.
Experiment artifacts (logs, UMAP plots, reports) are saved to `experiments/<run_name>/`.

## Project structure

```
data/
  download.py         # fetch PBMC 3k / 10k dataset
  preprocess.py       # QC → normalize → HVG selection → cell-type labels
  dataset.py          # PyTorch Dataset with rank tokenization and masking

model/
  tokenizer.py        # GeneVocab: gene ↔ token id mapping
  transformer.py      # scRNAEncoder (BERT-style, supports GNN embedding path)
  heads.py            # MaskedGenePredictionHead + CellTypeClassificationHead + CellGATClassificationHead
  gnn.py              # GATConv + GeneGAT (STRING PPI gene embeddings)
  gene_graph.py       # STRING API fetch, co-expression fallback, graph construction

train/
  config.py           # DataConfig / ModelConfig / PretrainConfig / FinetuneConfig
  pretrain.py         # self-supervised pretraining loop
  finetune.py         # supervised cell-type classification (CLS head or CellGAT head)
  benchmark_eval.py   # 5-fold CV on 10 benchmark datasets vs scBiGNN / ACTINN
  ablations.py        # runs the 7 ablation experiments (README "Ablation Experiments"), cached per-config

eval/
  metrics.py          # accuracy, F1, confusion matrix
  visualise.py        # UMAP of CLS embeddings

notebooks/
  01_data_exploration.ipynb
  02_pretraining_curves.ipynb
  03_ablations.ipynb

experiments/          # per-run output: logs, checkpoints, UMAP plots, report.md
run_experiment.py     # full pipeline runner
```

## Key design decisions

- Rank-based tokenization (default, `DataConfig.tokenization="rank"`): genes sorted by descending expression per cell, following Geneformer
- Expression-bin tokenization (`tokenization="expr_bin"`, scBERT-style): fixed gene panel shared by every cell, expression discretized into `n_bins` quantile bins; MGP pretraining masks the *bin id*, not gene identity (gene identity is fixed/known per position, so masking it would be trivially recoverable from the position embedding alone)
- Special tokens: [PAD]=0, [CLS]=1, [MASK]=2; gene vocab starts at index 3
- Pre-LayerNorm transformer for training stability
- Masking applied only at dataset level during pretraining (labels=-100 for unmasked positions)
- All hyperparameters live in `train/config.py` dataclasses
- `num_workers=0` in DataLoaders — required on macOS to avoid MPS/multiprocessing conflicts
- `pin_memory=False` for MPS (only True for CUDA)
- `max_seq_len=512` — sufficient for pbmc3k (95th percentile of expressed HVGs is 285)

## GNN integration

PRISM uses the STRING PPI graph in two distinct ways:

### Option A — GeneGAT (gene embeddings, `ModelConfig.use_gnn`)

| `use_gnn` | `gnn_freeze` | Behaviour |
|---|---|---|
| `False` | — | Plain `nn.Embedding` — baseline |
| `True` | `True` | GeneGAT frozen — STRING prior baked in, no gradient through GAT |
| `True` | `False` | GeneGAT joint — STRING prior + end-to-end training |

PPI graph cached to `data/string_graph_score700.npz`. Falls back to co-expression graph if STRING unreachable.

### Option B — CellGAT head (GNN-as-classifier, `ModelConfig.use_gat_head`)

Uses PPI during classification, not just embedding init.
- Gene hidden states from transformer → GATConv over PPI edges within each cell → attention pool → cell embedding → linear.
- Enable: `ModelConfig(use_gat_head=True)` or `--head gat` in benchmark_eval.py.
