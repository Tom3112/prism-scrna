"""
Generate architecture diagrams for PRISM.

Saves PNG files to assets/:
  pipeline.png     — full training pipeline
  encoder.png      — scRNAEncoder with dual embedding paths
  gene_gat.png     — GeneGAT over the STRING PPI graph
  tokenization.png — how a cell becomes a ranked token sequence

Usage:
    uv run python make_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C = dict(
    blue   = "#4A90D9",
    green  = "#5BAD6F",
    orange = "#E8834A",
    purple = "#9B6BB5",
    red    = "#D95B5B",
    grey   = "#8A9BB0",
    light  = "#F0F4F8",
    dark   = "#2C3E50",
    white  = "#FFFFFF",
    yellow = "#F5C842",
)

def _box(ax, x, y, w, h, label, sublabel=None, color=C["blue"],
         fontsize=11, radius=0.04):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=color, edgecolor=C["dark"], linewidth=1.5, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + (0.07 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=C["white"], zorder=4)
    if sublabel:
        ax.text(x, y - 0.13, sublabel,
                ha="center", va="center", fontsize=8,
                color=C["white"], alpha=0.9, zorder=4)

def _arrow(ax, x0, y0, x1, y1, color=C["dark"], lw=1.8, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0"))

def _label(ax, x, y, text, fontsize=9, color=C["grey"], ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, style="italic")


# ── Figure 1: Full Pipeline ───────────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_facecolor(C["white"])
    fig.patch.set_facecolor(C["white"])

    steps = [
        (1.1,  2.0, 1.6, 0.9, "Raw\nCounts",      None,            C["grey"]),
        (3.1,  2.0, 1.6, 0.9, "Preprocess",        "QC→HVG→log1p", C["blue"]),
        (5.1,  2.0, 1.6, 0.9, "Tokenize",          "rank-based",   C["blue"]),
        (7.1,  2.0, 1.6, 0.9, "scRNAEncoder",      "Transformer",  C["purple"]),
        (9.5,  2.8, 1.6, 0.9, "MGP Head",          "pretraining",  C["orange"]),
        (9.5,  1.2, 1.6, 0.9, "Class. Head",       "fine-tuning",  C["green"]),
        (11.9, 2.8, 1.6, 0.9, "Gene\nPrediction",  None,           C["orange"]),
        (11.9, 1.2, 1.6, 0.9, "Cell Type\nLabel",  None,           C["green"]),
    ]
    for x, y, w, h, label, sub, col in steps:
        _box(ax, x, y, w, h, label, sub, col, fontsize=10)

    # Main horizontal arrows
    for x0, x1 in [(1.9, 2.3), (3.9, 4.3), (5.9, 6.3), (7.9, 8.1)]:
        _arrow(ax, x0, 2.0, x1, 2.0)

    # Split arrow from encoder to two heads
    _arrow(ax, 8.1, 2.4, 8.65, 2.8)
    _arrow(ax, 8.1, 1.6, 8.65, 1.2)

    # Arrows from heads to outputs
    _arrow(ax, 10.3, 2.8, 11.1, 2.8)
    _arrow(ax, 10.3, 1.2, 11.1, 1.2)

    # Brace labels
    _label(ax, 6.1, 0.55, "Pretraining objective: Masked Gene Prediction",
           fontsize=9, color=C["orange"])
    _label(ax, 6.1, 0.25, "Fine-tuning objective: Cross-Entropy on cell-type labels",
           fontsize=9, color=C["green"])

    ax.set_title("PRISM — Training Pipeline",
                 fontsize=13, fontweight="bold", color=C["dark"], pad=10)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "pipeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["white"])
    plt.close()
    print(f"Saved {out}")


# ── Figure 2: scRNAEncoder (dual embedding paths) ────────────────────────────
def fig_encoder():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(C["white"])

    # Input
    _box(ax, 6, 6.3, 3.0, 0.7, "[CLS]  g₁  g₂  …  gₙ  [PAD]",
         "input_ids  (B × L)", C["grey"], fontsize=10)

    # Two embedding paths
    _box(ax, 2.8, 4.8, 3.2, 0.8, "nn.Embedding",
         "plain lookup table", C["blue"], fontsize=10)
    _box(ax, 9.2, 4.8, 3.2, 0.8, "GeneGAT",
         "STRING PPI prior", C["purple"], fontsize=10)

    # Labels above paths
    ax.text(2.8, 5.45, "Baseline path", ha="center", fontsize=9,
            color=C["blue"], fontweight="bold")
    ax.text(9.2, 5.45, "GNN path  (use_gnn=True)", ha="center", fontsize=9,
            color=C["purple"], fontweight="bold")

    # Arrows from input to both paths
    _arrow(ax, 4.5, 5.95, 3.2, 5.22)
    _arrow(ax, 7.5, 5.95, 8.8, 5.22)

    # Pos embedding
    _box(ax, 6, 4.8, 2.4, 0.8, "Pos Embedding",
         "rank → vector", C["blue"], fontsize=10)
    _label(ax, 6, 4.3, "+ (element-wise sum)", fontsize=9)

    # Merge arrows to sum
    _arrow(ax, 2.8, 4.4, 4.8, 4.05)
    _arrow(ax, 6.0, 4.4, 6.0, 4.12)
    _arrow(ax, 9.2, 4.4, 7.2, 4.05)

    # LayerNorm + Dropout
    _box(ax, 6, 3.6, 3.2, 0.65, "LayerNorm  +  Dropout", None,
         C["grey"], fontsize=10)
    _arrow(ax, 6, 4.1, 6, 3.94)

    # Transformer layers
    for i, y in enumerate([2.9, 2.2, 1.55]):
        alpha = 1.0 - i * 0.12
        col = tuple(c * alpha + (1 - alpha) for c in
                    (int(C["orange"][1:3], 16)/255,
                     int(C["orange"][3:5], 16)/255,
                     int(C["orange"][5:7], 16)/255))
        col_hex = "#{:02x}{:02x}{:02x}".format(
            int(col[0]*255), int(col[1]*255), int(col[2]*255))
        label = f"TransformerEncoderLayer {i+1}" if i < 2 else "TransformerEncoderLayer N"
        _box(ax, 6, y, 5.5, 0.55, label,
             "MultiHeadAttention  ·  FFN  ·  Pre-LN",
             C["orange"], fontsize=9)
        if i == 0:
            _arrow(ax, 6, 3.28, 6, 3.19)
        else:
            _arrow(ax, 6, y + 0.58, 6, y + 0.3)
        if i == 1:
            ax.text(6, 1.85, "⋮", ha="center", fontsize=14, color=C["orange"])

    # Output split
    _arrow(ax, 6, 1.27, 6, 1.0)
    _box(ax, 6, 0.65, 5.5, 0.55,
         "(B, L, hidden_dim)  hidden states",
         "CLS token [:, 0, :]  →  cell embedding", C["green"], fontsize=9)

    ax.set_title("scRNAEncoder Architecture", fontsize=13,
                 fontweight="bold", color=C["dark"], pad=8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "encoder.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["white"])
    plt.close()
    print(f"Saved {out}")


# ── Figure 3: GeneGAT ─────────────────────────────────────────────────────────
def fig_gene_gat():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor(C["white"])

    # ── Left: gene graph ──
    ax.text(2.5, 5.6, "STRING PPI Gene Graph", ha="center",
            fontsize=11, fontweight="bold", color=C["dark"])
    ax.text(2.5, 5.25, "nodes = 2,000 HVGs  ·  edges = known interactions",
            ha="center", fontsize=8.5, color=C["grey"])

    np.random.seed(7)
    n_vis = 12
    angles = np.linspace(0, 2 * np.pi, n_vis, endpoint=False)
    cx, cy, r = 2.5, 2.8, 1.6
    positions = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    edges_vis = [(0,3),(0,7),(1,4),(2,5),(3,6),(4,8),(5,9),(6,10),(7,11),(8,2),(9,1),(10,4)]
    for i, j in edges_vis:
        x0, y0 = positions[i]; x1, y1 = positions[j]
        ax.plot([x0, x1], [y0, y1], color=C["grey"], lw=1.0, alpha=0.5, zorder=1)
    colors_node = [C["blue"], C["green"], C["orange"], C["purple"],
                   C["red"], C["blue"], C["green"], C["orange"],
                   C["purple"], C["red"], C["blue"], C["green"]]
    for i, (x, y) in enumerate(positions):
        circ = plt.Circle((x, y), 0.18, color=colors_node[i], zorder=2, ec=C["dark"], lw=1)
        ax.add_patch(circ)
        ax.text(x, y, f"g{i}", ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", zorder=3)

    # ── Middle: GAT layers ──
    ax.text(6.0, 5.6, "GeneGAT", ha="center",
            fontsize=11, fontweight="bold", color=C["dark"])

    _box(ax, 6.0, 4.5, 2.8, 0.75,
         "Learnable gene features",
         "(n_genes × hidden_dim)", C["grey"], fontsize=9)

    for i, (y, label) in enumerate([(3.35, "GATConv  Layer 1"), (2.2, "GATConv  Layer 2")]):
        _box(ax, 6.0, y, 2.8, 0.75, label,
             "multi-head attention  ·  scatter softmax",
             C["purple"], fontsize=9)
        if i == 0:
            _arrow(ax, 6.0, 4.13, 6.0, 3.74)
        _box(ax, 6.0, y - 0.62, 2.8, 0.35,
             "LayerNorm (residual)",
             None, C["blue"], fontsize=8)
        _arrow(ax, 6.0, y - 0.38, 6.0, y - 0.45)
    _arrow(ax, 6.0, 3.05, 6.0, 2.59)
    _arrow(ax, 6.0, 1.90, 6.0, 1.4)

    _box(ax, 6.0, 1.1, 2.8, 0.6,
         "Gene embeddings",
         "(n_genes × hidden_dim)", C["green"], fontsize=9)

    # attention formula
    ax.text(6.0, 0.55,
            r"$\alpha_{ij}$ = softmax$_j$( LeakyReLU( $\mathbf{a}^T$[$\mathbf{W}h_i \| \mathbf{W}h_j$] ) · $w_{ij}$ )",
            ha="center", fontsize=9, color=C["dark"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["light"],
                      edgecolor=C["grey"], lw=1))

    # ── Right: how it plugs in ──
    ax.text(10.0, 5.6, "Plugs into Encoder", ha="center",
            fontsize=11, fontweight="bold", color=C["dark"])

    _box(ax, 10.0, 4.6, 3.0, 0.75,
         "special_emb.weight",
         "[PAD], [CLS], [MASK] → (3, D)", C["grey"], fontsize=8.5)
    _box(ax, 10.0, 3.5, 3.0, 0.75,
         "gene_gat()",
         "HVG embeddings → (2000, D)", C["purple"], fontsize=8.5)

    ax.text(10.0, 2.85, "concat  ↓", ha="center", fontsize=10, color=C["dark"])

    _box(ax, 10.0, 2.4, 3.0, 0.75,
         "all_embs[input_ids]",
         "vocab embedding lookup (2003, D)", C["blue"], fontsize=8.5)

    _arrow(ax, 10.0, 2.03, 10.0, 1.5)

    _box(ax, 10.0, 1.15, 3.0, 0.65,
         "TransformerEncoder",
         "(unchanged)", C["orange"], fontsize=9)

    # Cross arrow from GAT output to right panel
    _arrow(ax, 7.4, 1.1, 8.5, 3.5, color=C["purple"])

    ax.set_title("GeneGAT — STRING PPI Gene Embedding",
                 fontsize=13, fontweight="bold", color=C["dark"], pad=8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "gene_gat.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["white"])
    plt.close()
    print(f"Saved {out}")


# ── Figure 4: Tokenization ────────────────────────────────────────────────────
def fig_tokenization():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor(C["white"])

    ax.set_title("Rank-Based Gene Tokenization", fontsize=13,
                 fontweight="bold", color=C["dark"], pad=8)

    # ── Step 1: raw expression ──
    ax.text(1.8, 4.5, "① Raw expression\n(single cell)", ha="center",
            fontsize=10, fontweight="bold", color=C["dark"])
    genes  = ["geneA", "geneB", "geneC", "geneD", "geneE"]
    values = [0.0,     3.2,     0.1,     5.8,     1.4]
    colors = [C["grey"] if v == 0 else C["blue"] for v in values]
    for i, (g, v, col) in enumerate(zip(genes, values, colors)):
        y = 3.7 - i * 0.52
        bar_w = v / 7.0 * 2.2
        rect = FancyBboxPatch((0.3, y - 0.18), bar_w, 0.36,
                              boxstyle="round,pad=0.02",
                              facecolor=col, alpha=0.85, ec="none", zorder=2)
        ax.add_patch(rect)
        ax.text(0.28, y, g, ha="right", va="center", fontsize=9, color=C["dark"])
        if v > 0:
            ax.text(0.32 + bar_w, y, f"{v}", ha="left", va="center",
                    fontsize=8.5, color=C["dark"])
        else:
            ax.text(0.32, y, "0  (dropped)", ha="left", va="center",
                    fontsize=8, color=C["grey"], style="italic")

    # ── Arrow ──
    _arrow(ax, 2.8, 2.3, 3.5, 2.3, color=C["dark"], lw=2)
    ax.text(3.15, 2.6, "sort desc.", ha="center", fontsize=8.5, color=C["grey"])

    # ── Step 2: ranked sequence ──
    ax.text(5.3, 4.5, "② Ranked sequence\n(expressed genes only)", ha="center",
            fontsize=10, fontweight="bold", color=C["dark"])

    ranked = [("geneD", 5.8, 0), ("geneB", 3.2, 1), ("geneE", 1.4, 2), ("geneC", 0.1, 3)]
    for rank, (g, v, _) in enumerate(ranked):
        x = 3.8 + rank * 0.85
        _box(ax, x, 2.9, 0.72, 0.55, g, None, C["blue"], fontsize=8)
        ax.text(x, 2.38, f"rank {rank+1}", ha="center", fontsize=7.5, color=C["grey"])

    ax.text(3.8 + 4 * 0.85, 2.9, "…", ha="center", fontsize=16, color=C["grey"])

    # ── Arrow ──
    _arrow(ax, 7.6, 2.3, 8.3, 2.3, color=C["dark"], lw=2)
    ax.text(7.95, 2.6, "map to IDs", ha="center", fontsize=8.5, color=C["grey"])

    # ── Step 3: token ids ──
    ax.text(10.2, 4.5, "③ Token sequence\n(model input)", ha="center",
            fontsize=10, fontweight="bold", color=C["dark"])

    tokens = [("[CLS]", "1", C["green"]),
              ("geneD", "44", C["blue"]),
              ("geneB", "12", C["blue"]),
              ("geneE", "6",  C["blue"]),
              ("geneC", "203",C["blue"]),
              ("[PAD]", "0",  C["grey"])]
    for i, (name, tid, col) in enumerate(tokens):
        x = 8.5 + i * 0.82
        _box(ax, x, 3.15, 0.7, 0.5, name, None, col, fontsize=7.5)
        ax.text(x, 2.7, tid, ha="center", fontsize=8, color=C["dark"],
                fontweight="bold")

    ax.text(8.5 + 6 * 0.82, 3.15, "…", ha="center", fontsize=16, color=C["grey"])

    # ── Token id legend ──
    ax.text(6.5, 1.55,
            "Vocabulary:  [PAD] = 0   [CLS] = 1   [MASK] = 2   "
            "gene_0 = 3   …   gene_1999 = 2002   (vocab_size = 2,003)",
            ha="center", fontsize=9, color=C["dark"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["light"],
                      edgecolor=C["grey"], lw=1))

    # ── Position = rank note ──
    ax.text(6.5, 0.75,
            "Position index = expression rank  →  pos_emb(1) learns to mean "
            "\"most expressed gene in this cell\"",
            ha="center", fontsize=9, color=C["orange"], style="italic")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "tokenization.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["white"])
    plt.close()
    print(f"Saved {out}")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_pipeline()
    fig_encoder()
    fig_gene_gat()
    fig_tokenization()
    print("Done. Figures saved to assets/")
