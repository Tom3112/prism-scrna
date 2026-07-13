"""
UMAP visualisation of CLS embeddings coloured by cell type.

Key ablation: pretrained encoder vs random-init encoder — call plot_comparison().
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model.transformer import scRNAEncoder


@torch.no_grad()
def extract_cls_embeddings(
    encoder: scRNAEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (embeddings, labels) arrays from the test loader."""
    encoder.eval()
    embs, labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bin_ids = batch["bin_ids"].to(device) if "bin_ids" in batch else None
        emb = encoder.get_cls_embedding(input_ids, attention_mask, bin_ids).cpu().numpy()
        embs.append(emb)
        if "cell_type" in batch:
            labels.append(batch["cell_type"].numpy())
    embs = np.concatenate(embs)
    labels = np.concatenate(labels) if labels else np.zeros(len(embs), dtype=int)
    return embs, labels


def umap_reduce(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed, verbose=False)
    except ImportError:
        from sklearn.decomposition import PCA
        print("umap-learn not found, falling back to PCA for visualisation.")
        reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(embeddings)


def plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    title: str = "UMAP of CLS Embeddings",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Axes:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    cmap = cm.get_cmap("tab10", len(label_names))
    for i, name in enumerate(label_names):
        mask = labels == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], label=name, s=8, alpha=0.7, linewidths=0,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        else:
            plt.show()
    return ax


def plot_comparison(
    pretrained_embs: np.ndarray,
    random_embs: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    save_path: str | None = None,
) -> None:
    """Side-by-side UMAP: pretrained encoder vs random-init encoder."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    coords_pre = umap_reduce(pretrained_embs)
    coords_rand = umap_reduce(random_embs)

    plot_umap(coords_pre, labels, label_names, title="Pretrained Encoder", ax=axes[0])
    plot_umap(coords_rand, labels, label_names, title="Random Init Encoder", ax=axes[1])

    # Remove duplicate legends — keep only the right panel's
    axes[0].get_legend().remove()

    plt.suptitle("CLS Embedding UMAP: Pretrained vs Random Init", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot: {save_path}")
    else:
        plt.show()


def run_visualisation(
    pretrain_ckpt: str,
    finetune_ckpt: str,
    test_loader: DataLoader,
    label_names: list[str],
    device: torch.device,
    out_dir: str = ".",
) -> None:
    """
    Full pipeline: load both checkpoints, extract embeddings, plot comparison.
    """
    import torch
    from model.transformer import scRNAEncoder

    def _load_encoder(ckpt_path: str) -> scRNAEncoder:
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["model_cfg"]
        enc = scRNAEncoder(
            vocab_size=ckpt["vocab_size"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            ffn_dim=cfg["ffn_dim"],
            dropout=cfg["dropout"],
        ).to(device)
        enc.load_state_dict(ckpt["encoder_state"])
        return enc

    pretrained_enc = _load_encoder(finetune_ckpt)
    rand_enc = scRNAEncoder(
        vocab_size=pretrained_enc.token_emb.num_embeddings,
        hidden_dim=pretrained_enc.hidden_dim,
        num_layers=len(pretrained_enc.encoder.layers),
        num_heads=pretrained_enc.encoder.layers[0].self_attn.num_heads,
    ).to(device)

    pre_embs, labels = extract_cls_embeddings(pretrained_enc, test_loader, device)
    rand_embs, _ = extract_cls_embeddings(rand_enc, test_loader, device)

    save_path = os.path.join(out_dir, "umap_comparison.png")
    plot_comparison(pre_embs, rand_embs, labels, label_names, save_path=save_path)
