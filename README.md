# PRISM

**PPI-augmented RNA Inference and Single-cell Modeling**

A hybrid **Transformer + GAT** foundation model for single-cell RNA-seq data. Pretrained with Masked Gene Prediction (MGP), then fine-tuned for cell-type classification. The protein-protein interaction (PPI) graph from STRING is used in two places: to produce biologically-informed gene embeddings (GeneGAT), and during classification itself (CellGAT head).

Inspired by Geneformer, scGPT, and scBERT. Benchmarked against scBiGNN and ACTINN on 10 standard datasets.

---

## Architecture

![Pipeline](assets/pipeline.png)
![Encoder](assets/encoder.png)

```
Input cell  →  [CLS] g₁ g₂ … gₙ  (ranked by expression, padded to max_seq_len)
                    ↓
       Gene Embedding  +  Rank (Position) Embedding
       ┌──────────────┴──────────────────┐
  nn.Embedding                      GeneGAT (Option A)
  (baseline)               STRING PPI prior → gene embeddings
       └──────────────┬──────────────────┘
                      ↓
           TransformerEncoder  (Pre-LN, GELU, N layers)
                      ↓
        ┌─────────────┴──────────────────────┐
   All positions                        CLS token
        ↓                                    ↓
MaskedGenePredHead              ┌─────────────┴──────────────┐
(pretraining)             CLS linear head          CellGAT head (Option B)
                          (baseline)          PPI GAT + attention pool
                                              (GNN-as-classifier)
```

**Default config (small):** hidden=256, layers=4, heads=4, ffn=1024, dropout=0.1

---

## Tokenization

![Tokenization](assets/tokenization.png)

Rank-based, as in Geneformer:
1. For each cell, take the top-2,000 HVGs after log-normalization.
2. Sort genes by descending expression → ordered token sequence.
3. Prepend `[CLS]`; pad/truncate to `max_seq_len=512`.
4. Gene index in the sequence **is** the positional encoding — expression rank is the biological prior.

Vocabulary: `[PAD]=0, [CLS]=1, [MASK]=2, gene_0=3, …, gene_1999=2002`

---

## Pretraining Objective

Masked Gene Prediction (MGP):
- Randomly mask 15% of non-`[CLS]` gene tokens → replace with `[MASK]`.
- Model predicts the original gene identity at each masked position.
- Loss = cross-entropy on masked positions only (`labels=-100` elsewhere).

---

## GNN Integration

![GeneGAT](assets/gene_gat.png)

The STRING PPI graph connects genes whose proteins physically interact (confidence ≥ 700).
PRISM uses this graph in two distinct ways:

### Option A — GeneGAT (gene embeddings)
Replaces the plain `nn.Embedding` table. A 2-layer GAT over the PPI graph produces
one context-aware embedding per gene before the transformer sees them.

| Condition | `use_gnn` | `gnn_freeze` | Description |
|---|---|---|---|
| Baseline | `False` | — | Plain `nn.Embedding`, learns from data only |
| GNN frozen | `True` | `True` | STRING prior baked in, GAT weights fixed |
| GNN joint | `True` | `False` | String prior + end-to-end fine-tuning |

### Option B — CellGAT head (GNN-as-classifier)
Closes the architectural gap with scBiGNN-style methods. Instead of reading only
the `[CLS]` token, gene hidden states from the transformer are:
1. Refined by a GATConv layer using PPI edges between co-expressed genes in each cell.
2. Pooled via learned attention → cell embedding → linear classifier.

Enable with `--head gat` in `benchmark_eval.py` or `ModelConfig(use_gat_head=True)`.

---

## Quick Start

```bash
uv sync

# 1. Download PBMC data
uv run python data/download.py

# 2. Preprocess
uv run python data/preprocess.py

# 3. Pretrain
uv run python train/pretrain.py

# 4. Fine-tune (CLS head)
uv run python train/finetune.py

# 5. Fine-tune (GNN-as-classifier head)
uv run python train/finetune.py  # set ModelConfig(use_gat_head=True) in config

# 6. Full experiment with report
uv run python run_experiment.py
```

```bash
# Download all 10 benchmark datasets (same Zenodo archive)
uv run python data/download_benchmarks.py

# Run 5-fold CV against published baselines
uv run python train/benchmark_eval.py               # CLS head vs scBiGNN / ACTINN
uv run python train/benchmark_eval.py --head gat    # GNN-as-classifier
uv run python train/benchmark_eval.py --dataset BaronHuman --epochs 20
```

---

## Benchmarks

10 datasets from Abdelaal et al. 2019 (Zenodo 3357167):

| Dataset | Cells | Types | Baseline | Method |
|---|---|---|---|---|
| Zheng68K | 65,943 | 11 | 0.760 | scBiGNN |
| Zhengsorted | 20,000 | 10 | 0.867 | scBiGNN |
| BaronHuman | 8,569 | 14 | 0.983 | scBiGNN |
| BaronMouse | 1,886 | 13 | 0.983 | scBiGNN |
| AMB | 12,832 | 22 | 0.994 | scBiGNN |
| Zeisel | ~3,000 | 9 | 0.944 | ACTINN |
| Segerstolpe | ~2,300 | 14 | 0.886 | ACTINN |
| Muraro | ~2,100 | 9 | 0.962 | ACTINN |
| Macosko | ~44,000 | 39 | 0.798 | ACTINN |
| Klein | ~2,400 | 4 | 0.979 | ACTINN |

---

## Ablation Experiments

| Experiment | Variable | Metric |
|---|---|---|
| Masking ratio | 5%, 15%, 25%, 40% | Val loss, downstream F1 |
| Tokenization | Rank-based vs raw expression bins | Embedding UMAP quality |
| Model depth | 2 / 4 / 6 layers | Val loss, fine-tune accuracy |
| Pretrain vs scratch | Pretrained encoder vs random init | Fine-tune accuracy |
| Freeze vs fine-tune | Frozen encoder vs full fine-tune | Fine-tune accuracy |
| Gene embeddings | Baseline vs GNN frozen vs GNN joint | Fine-tune accuracy, UMAP |
| Classification head | CLS linear vs CellGAT (PPI) | Fine-tune accuracy |

Run ablations in `notebooks/03_ablations.ipynb`.

---

## Project Structure

```
prism/
├── data/
│   ├── download.py              # Fetch PBMC 3k / 10k
│   ├── preprocess.py            # QC → normalize → HVG → labels
│   ├── dataset.py               # PyTorch Dataset with rank tokenization + masking
│   ├── download_benchmarks.py   # Download 10 benchmark datasets from Zenodo
│   └── benchmark_utils.py       # Preprocessing + k-fold splits for benchmarks
├── model/
│   ├── tokenizer.py             # GeneVocab: gene ↔ token id mapping
│   ├── transformer.py           # scRNAEncoder (BERT-style, supports GNN embedding)
│   ├── heads.py                 # MGP head, CLS head, CellGAT head
│   ├── gnn.py                   # GATConv + GeneGAT (STRING PPI gene embeddings)
│   └── gene_graph.py            # STRING API fetch, co-expression fallback
├── train/
│   ├── config.py                # DataConfig / ModelConfig / PretrainConfig / FinetuneConfig
│   ├── pretrain.py              # MGP pretraining loop
│   ├── finetune.py              # Supervised cell-type classification
│   └── benchmark_eval.py        # 5-fold CV on 10 benchmark datasets
├── eval/
│   ├── metrics.py               # Accuracy, F1, confusion matrix
│   └── visualise.py             # UMAP of CLS embeddings
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pretraining_curves.ipynb
│   └── 03_ablations.ipynb
├── experiments/                 # Per-run logs, UMAP plots, reports
├── run_experiment.py            # Full pipeline runner
└── checkpoints/
```

---

## References

- **Geneformer** — Theodoris et al., *Nature* 2023. Rank-based gene tokenization.
- **scGPT** — Cui et al., *Nature Methods* 2024. Generative pretraining on scRNA.
- **scBERT** — Yang et al., *Nature Machine Intelligence* 2022. BERT for scRNA.
- **scBiGNN** — Ma et al., 2023. Bipartite GNN for cell-type annotation (direct baseline).
- **ACTINN** — Chen et al., *Bioinformatics* 2020. Supervised neural network baseline.
- **STRING** — Szklarczyk et al., *Nucleic Acids Research* 2023. PPI network.
- **GAT** — Velickovic et al., *ICLR* 2018. Graph Attention Networks.
- **Abdelaal et al.** — *Genome Biology* 2019. Benchmark suite (Zenodo 3357167).
