# PRISM

**PPI-augmented RNA Inference and Single-cell Modeling**

A hybrid **Transformer + GAT** foundation model for single-cell RNA-seq data. Pretrained with Masked Gene Prediction (MGP), then fine-tuned for cell-type classification. The protein-protein interaction (PPI) graph from STRING is used in two places: to produce biologically-informed gene embeddings (GeneGAT), and during classification itself (CellGAT head).

Inspired by Geneformer, scGPT, and scBERT. Benchmarked against scBiGNN and ACTINN on 7 standard datasets.

> **Scope note:** the original project brief specified a plain transformer (no graph
> component) with 5 ablation experiments and 3 references (Geneformer, scGPT, scBERT).
> The GeneGAT/CellGAT architecture, the scBiGNN/ACTINN benchmark suite, and 3 of the
> 8 ablations below (gene embeddings, GNN depth, classification head) extend beyond
> that brief — motivated by the GNN-in-single-cell-omics review cited below, which is
> the actual source for the graph-based design, not the original brief.

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
uv run python train/benchmark_eval.py               # CLS head, no GNN vs scBiGNN / ACTINN
uv run python train/benchmark_eval.py --gnn frozen  # GeneGAT frozen (Option A)
uv run python train/benchmark_eval.py --gnn joint    # GeneGAT joint (Option A)
uv run python train/benchmark_eval.py --head gat    # CellGAT head (Option B)
uv run python train/benchmark_eval.py --dataset BaronHuman --epochs 20
```

---

## Benchmarks

7 datasets, all from the Abdelaal et al. 2019 Zenodo archive (3357167), evaluated via
`train/benchmark_eval.py` (5-fold CV, from-scratch supervised training — no MGP
pretraining transfer, matching how scBiGNN/ACTINN themselves are evaluated). All 7
verified: both their presence in that exact archive (confirmed by listing its full
contents directly) and their baseline numbers against the cited paper.

**Four architecture variants run below**, all on the same 7 datasets / 5-fold CV:
plain PRISM baseline (`use_gnn=False`, CLS head), GeneGAT frozen (Option A,
`gnn_freeze=True`), GeneGAT joint (Option A, `gnn_freeze=False`), and CellGAT head
(Option B, `use_gat_head=True`). See [GNN Variants on the Benchmark Suite](#gnn-variants-on-the-benchmark-suite)
below for the head-to-head comparison.

**Metrics — not all one type, read the Method column:**
- **scBiGNN** rows: baseline is **accuracy**, confirmed directly from Ma et al.'s
  Table 2 (caption: *"Classification accuracy of all the methods on the five
  datasets"*) — the 5 datasets in that table are exactly Zheng68K, Zhengsorted,
  BaronHuman, BaronMouse, AMB (our baseline numbers are an exact match to their
  bold `scBiGNN (p_θ)` row). PRISM's accuracy is the right comparison here.
- **ACTINN** rows (Segerstolpe, Muraro): scBiGNN's paper doesn't include these two
  datasets at all, so the 0.886/0.962 baselines must trace to **Abdelaal et al. 2019
  directly**, whose Methods section states its metric explicitly: *"For each cell
  population in the dataset, the F1-score is reported. The median of these F1-scores
  is used as a measure for the performance on the dataset."* Median F1 is their sole
  reported metric — so 0.886/0.962 are **median F1**, not accuracy. PRISM's median F1
  (not accuracy) is the correct comparison for these two rows.
- **median F1** here means the median (not mean) of per-class F1 scores — robust to
  one or two badly-performing rare classes dragging down a macro-average, which is
  exactly why Abdelaal et al. use it instead of accuracy or macro-F1. Computed in
  `eval/metrics.py` the same way: per-class F1 via `f1_score(average=None)`, then
  `np.median()` over classes.

| Dataset | Cells | Types | Baseline | Method | PRISM accuracy | PRISM median F1 | Δ |
|---|---|---|---|---|---|---|---|
| Zheng68K | 65,943 | 11 | 0.760 (acc) | scBiGNN | **0.840 ± 0.003** | 0.796 | ▲ 8.04% (acc) |
| Zhengsorted | 20,000 | 10 | 0.867 (acc) | scBiGNN | 0.821 ± 0.004 | 0.841 | ▼ 4.60% (acc) |
| BaronHuman | 8,569 | 14 | 0.983 (acc) | scBiGNN | 0.986 ± 0.002 | 0.983 | ▲ 0.25% (acc) |
| BaronMouse | 1,886 | 13 | 0.983 (acc) | scBiGNN | 0.959 ± 0.003 | 0.884 | ▼ 2.44% (acc) |
| AMB | 12,832 | 22 | 0.994 (acc) | scBiGNN | 0.989 ± 0.002 | 0.990 | ▼ 0.51% (acc) |
| Segerstolpe | 2,133 | 13 | 0.886 (median F1) | ACTINN | 0.972 ± 0.006 | 0.963 | ▲ 7.70% (median F1) |
| Muraro | 2,122 | 9 | 0.962 (median F1) | ACTINN | 0.978 ± 0.010 | 0.971 | ▲ 0.94% (median F1) |

Among the 5 scBiGNN rows (accuracy baseline), PRISM beats it on 2 of 5 (Zheng68K
+8.04pp, BaronHuman +0.25pp) and trails on 3 (Zhengsorted, BaronMouse, AMB), all
within ~0.5–4.6pp. One caveat: BaronMouse's accuracy (0.959) looks solid but median
F1 is much lower (0.884, mean macro F1 0.661) — a per-class breakdown shows the 5
largest classes (91% of the dataset) all score F1 ≥ 0.90, while the smallest classes
do badly (schwann, 6 cells, F1 = 0.000; T_cell, 7 cells, F1 = 0.25) — classic
class-imbalance masking, the same failure mode Abdelaal et al.'s median-F1 choice is
designed to catch. On the 2 ACTINN rows (median F1 baseline, now correctly matched
to PRISM's median F1), PRISM wins both: Segerstolpe +7.70pp, Muraro +0.94pp.

### GNN Variants on the Benchmark Suite

Same 7 datasets, 5-fold CV, same from-scratch supervised training — only the gene
embedding / classification head changes. Accuracy for all 4 variants:

| Dataset | Baseline | GNN frozen | GNN joint | CellGAT | Best variant |
|---|---|---|---|---|---|
| BaronHuman | 0.9855 | 0.9742 (▼1.13pp) | 0.9854 (▼0.01pp) | **0.9856** (▲0.01pp) | CellGAT (tied) |
| BaronMouse | 0.9586 | 0.9205 (▼3.81pp) | 0.9677 (▲0.91pp) | **0.9751** (▲1.65pp) | CellGAT |
| AMB | 0.9889 | 0.9778 (▼1.11pp) | 0.9890 (▲0.01pp) | **0.9914** (▲0.25pp) | CellGAT |
| Zheng68K | 0.8404 | 0.8293 (▼1.11pp) | **0.8429** (▲0.25pp) | 0.8383 (▼0.21pp) | GNN joint |
| Zhengsorted | **0.8210** | 0.7517 (▼6.93pp) | 0.7436 (▼7.74pp) | 0.8200 (▼0.10pp) | Baseline (tied w/ CellGAT) |
| Segerstolpe | 0.9723 | 0.9334 (▼3.89pp) | 0.9709 (▼0.14pp) | **0.9756** (▲0.33pp) | CellGAT |
| Muraro | **0.9779** | 0.9383 (▼3.96pp) | 0.9736 (▼0.43pp) | 0.9731 (▼0.48pp) | Baseline |

Three clear patterns:
- **GNN frozen (Option A) loses on all 7 datasets**, sometimes badly (Zhengsorted
  ▼6.93pp, Muraro ▼3.96pp, Segerstolpe ▼3.89pp). A STRING PPI prior baked into frozen
  gene embeddings, with no ability to adapt to the task, is a net negative here —
  the plain learned embedding table adapts better than a fixed biological prior.
- **GNN joint (Option A) is roughly a wash against baseline** — small wins on
  BaronMouse/Zheng68K/AMB, small losses on Segerstolpe/Muraro, **except Zhengsorted,
  where it craters (▼7.74pp)**, worse even than frozen. Something about this dataset's
  gene panel or class structure makes end-to-end GAT training actively harmful, not
  just neutral.
- **CellGAT head (Option B) is the strongest variant overall** — best or tied-best on
  5/7 datasets, and critically, **does not share frozen/joint's Zhengsorted collapse**
  (▼0.10pp, essentially noise, vs ▼6.93/▼7.74pp). Since CellGAT uses PPI edges during
  classification on top of transformer hidden states (not as the gene embedding
  itself), it isn't vulnerable to whatever makes a STRING-initialized embedding table
  fail on this dataset — the transformer's own representations carry the signal, and
  the GAT layer adds a task-relevant refinement rather than replacing the embedding.

Notes:
- The 5 scBiGNN baselines are verified exact matches against Ma et al.'s scBiGNN paper
  (arXiv:2312.10310, Table 2) — confirmed accuracy (table caption), confirmed those 5
  datasets only (Segerstolpe/Muraro aren't in scBiGNN's paper at all). Segerstolpe and
  Muraro's cell/type counts are corrected here against Abdelaal et al. 2019's actual
  Table 2 (previously listed as ~2,300/14 and ~2,100/9); their ACTINN values (0.886,
  0.962) are median F1 — confirmed via Abdelaal et al. 2019's Methods section, which
  states median F1-score as its sole reported metric (see Benchmarks metrics note
  above for the exact quote).
- **Zeisel, Macosko, and Klein were previously listed here** (attributed to "Abdelaal
  et al. 2019 (Zenodo 3357167)") but don't actually exist anywhere in that Zenodo
  archive — confirmed by listing its full directory contents (`Inter-dataset/`,
  `Intra-dataset/`, `Rejection/`, `Scalability/`, none of which contain these 3
  datasets) as well as by the earlier literature check that found them absent from
  both Abdelaal et al. 2019 and the scBiGNN paper. They're legitimate, widely-used
  datasets (Zeisel et al. 2015 *Science*; Macosko et al. 2015 *Cell*; Klein et al.
  2015 *Cell*) but from other sources entirely, so they were dropped from this
  benchmark suite rather than fetched from elsewhere and reported without a
  comparable baseline.
- `data/download_benchmarks.py`'s CSV loader streams each file in row chunks into a
  sparse matrix rather than loading it dense — the largest file here (Zheng68K,
  2.7 GB as text) expands to well over 10 GB as a naive dense float64 array, enough
  to OOM-kill the process on a 14 GB box; chunked+sparse loading keeps peak memory
  to roughly one chunk's size regardless of dataset size.

---

## Ablation Experiments

Run with `uv run python train/ablations.py` (or `--experiment NAME` for a single one).
Pretrain/fine-tune runs are cached on disk keyed by config, so shared baseline runs
aren't recomputed across experiments. Results below are on PBMC 3k (`pbmc3k_processed()`,
2,638 cells, 8 real expert-annotated cell types — CD4 T, CD14+ Monocytes, B, CD8 T, NK,
FCGR3A+ Monocytes, Dendritic, Megakaryocytes — not unsupervised Leiden pseudo-labels);
see `notebooks/03_ablations.ipynb` for plots and `experiments/ablations/summary.md` for
the raw tables.

| Experiment | Variable | Metric | Result |
|---|---|---|---|
| Masking ratio | 5%, 15%, 25%, 40% | Val loss, downstream F1 | F1 rises with mask ratio (0.854 → 0.868 → 0.863 → **0.864**), accuracy flat (0.913–0.924) — no clear optimum, differences are within noise |
| Tokenization | Rank-based vs raw expression bins | Embedding UMAP quality (silhouette) | Rank: **-0.139** vs expr_bin: -0.158 — a small edge for rank ordering, much narrower than seen with pseudo-labels |
| Model depth | 2 / 4 / 6 layers | Val loss, fine-tune accuracy | Val loss falls monotonically with depth (6.47 → 6.39 → 6.18); accuracy non-monotonic (0.909 → **0.917** → 0.913) — depth helps pretraining loss but not classification here |
| Pretrain vs scratch | Pretrained encoder vs random init | Fine-tune accuracy | Scratch (**0.924**) still edges out pretrained (0.917) even with real labels — the earlier "confounded by circular Leiden labels" caveat no longer applies, and MGP pretraining still isn't clearly helping fine-tune accuracy on this small a dataset |
| Freeze vs fine-tune | Frozen encoder vs full fine-tune | Fine-tune accuracy | Full fine-tune **0.917** vs frozen 0.602 — the one result that stays decisive under real labels, same ~32-point gap as before |
| Gene embeddings | Baseline vs GNN frozen vs GNN joint | Fine-tune accuracy, UMAP | Baseline and GNN joint tie on accuracy (both 0.917); GNN frozen worse (0.871). Silhouette no longer favors joint training (-0.138 baseline vs -0.100 frozen vs -0.134 joint) — the clear GNN-joint win seen with Leiden pseudo-labels does not replicate on the harder, real-label task |
| GNN depth | GeneGAT at 1 / 2 / 3 layers (joint) | Fine-tune accuracy, UMAP | Noisy: accuracy peaks at 2 layers (0.917) and drops at 3 (0.864); silhouette peaks at 1 layer (0.037) and is negative at both 2 and 3. Depth's effect on embedding quality doesn't move consistently with either metric here — treat the earlier over-smoothing story as unconfirmed until repeated across seeds |
| Classification head | CLS linear vs CellGAT (PPI) | Fine-tune accuracy | CellGAT (**0.928**) ahead of CLS (0.917) by 1.1 points — small but the largest CellGAT-vs-CLS gap seen across both label sources |

Notes:
- Switching from unsupervised Leiden clusters to `pbmc3k_processed()`'s real expert
  annotations changes both `data/download.py` (fetches `pbmc3k_processed()` instead of
  raw `pbmc3k()`) and `data/preprocess.py` (uses `.raw`, the pre-scaling log-normalized
  matrix, since `.X` on the processed object is already z-scored/clipped for PCA and
  isn't valid input for rank/expr_bin tokenization; also fixed a dormant labeling bug
  where the louvain→cell-type mapping assumed numeric cluster IDs but the real column
  already contains descriptive names like `"CD4 T cells"`).
- Real annotation resolves the label-circularity caveat on the pretrain-vs-scratch
  experiment, but accuracy across the board is lower and noisier than with Leiden
  pseudo-labels (test set is the same size, 264 cells, but the task itself is harder) —
  several deltas here that looked clean before (GNN joint's win, the depth-vs-silhouette
  over-smoothing pattern) don't hold up under the real, harder task and should be read
  as inconclusive on a single run rather than confirmed effects.
- All numbers above are from the stratified train/val/test split (see `data/dataset.py`'s
  `load_datasets()`) with the corrected STRING edge-weight normalization (see
  `model/gene_graph.py`) — gene-embedding and CellGAT experiments run against live
  STRING PPI data, not the co-expression fallback.
- `val_loss` is only comparable *within* an experiment, not between tokenization
  schemes: rank predicts over the full gene vocabulary (~2,000-way) while expr_bin
  predicts over expression bins (11-way), so their losses live on different scales.

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

- **Geneformer** — Theodoris et al., "Transfer learning enables predictions in network biology," *Nature* 618:616–624, 2023. Rank-based gene tokenization. [doi.org/10.1038/s41586-023-06139-9](https://doi.org/10.1038/s41586-023-06139-9)
- **scGPT** — Cui et al., "scGPT: toward building a foundation model for single-cell multi-omics using generative AI," *Nature Methods*, 2024. Generative pretraining on scRNA. [doi.org/10.1038/s41592-024-02201-0](https://doi.org/10.1038/s41592-024-02201-0)
- **scBERT** — Yang et al., "scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data," *Nature Machine Intelligence* 4, 2022. BERT for scRNA. [doi.org/10.1038/s42256-022-00534-z](https://doi.org/10.1038/s42256-022-00534-z)
- **scBiGNN** — Ma et al., "scBiGNN: Bilevel Graph Representation Learning for Cell Type Classification from Single-cell RNA Sequencing Data," 2023. Bipartite GNN for cell-type annotation (direct baseline). [arxiv.org/abs/2312.10310](https://arxiv.org/abs/2312.10310)
- **ACTINN** — Ma & Pellegrini, "ACTINN: automated identification of cell types in single cell RNA sequencing," *Bioinformatics* 36(2):533–538, 2020. Supervised neural network baseline. [doi.org/10.1093/bioinformatics/btz592](https://doi.org/10.1093/bioinformatics/btz592)
- **STRING** — Szklarczyk et al., "The STRING database in 2023: protein-protein association networks and functional enrichment analyses for any sequenced genome of interest," *Nucleic Acids Research* 51(D1):D638–D646, 2023. PPI network. [doi.org/10.1093/nar/gkac1000](https://doi.org/10.1093/nar/gkac1000)
- **GAT** — Veličković et al., "Graph Attention Networks," *ICLR* 2018. [arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- **GNNs for single-cell omics** — Li, Hua & Chen, "Graph neural networks for single-cell omics data: a review of approaches and applications," *Briefings in Bioinformatics* 26(2):bbaf109, 2025. The actual source for PRISM's GeneGAT/CellGAT design (PPI-informed gene embeddings, GAT-based classification) and for the GNN-depth ablation's over-smoothing motivation — not the original project brief, which specified a plain transformer only. [doi.org/10.1093/bib/bbaf109](https://doi.org/10.1093/bib/bbaf109)
- **Abdelaal et al.** — "A comparison of automatic cell identification methods for single-cell RNA sequencing data," *Genome Biology* 20:194, 2019. Benchmark suite (Zenodo 3357167). [doi.org/10.1186/s13059-019-1795-z](https://doi.org/10.1186/s13059-019-1795-z)
