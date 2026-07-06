"""
Evaluation metrics for cell-type classification.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from model.transformer import scRNAEncoder
from model.heads import CellTypeClassificationHead, CellGATClassificationHead


def _forward_head(encoder, head, input_ids, attention_mask, labels=None, bin_ids=None):
    """Unified forward pass for CLS head and GAT head."""
    if isinstance(head, CellGATClassificationHead):
        hidden = encoder(input_ids, attention_mask, bin_ids)
        return head(hidden, attention_mask, input_ids, labels)
    else:
        cls_emb = encoder.get_cls_embedding(input_ids, attention_mask, bin_ids)
        return head(cls_emb, labels)


@torch.no_grad()
def collect_predictions(
    encoder: scRNAEncoder,
    head,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (all_preds, all_labels) as numpy arrays."""
    encoder.eval()
    head.eval()
    preds_list, labels_list = [], []

    amp_enabled = device.type == "cuda"
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cell_type      = batch["cell_type"].to(device)
        bin_ids        = batch["bin_ids"].to(device) if "bin_ids" in batch else None

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, logits = _forward_head(encoder, head, input_ids, attention_mask, bin_ids=bin_ids)
        preds_list.append(logits.argmax(dim=-1).cpu().numpy())
        labels_list.append(cell_type.cpu().numpy())

    return np.concatenate(preds_list), np.concatenate(labels_list)


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    # Explicit `labels=` keeps class count/order fixed at len(label_names) even
    # when a fold has zero test examples of some (rare) class — without it,
    # sklearn infers the class set from what actually appears in this fold,
    # which silently shrinks per_class_f1/confusion_matrix and crashes
    # classification_report's target_names length check.
    all_labels = np.arange(len(label_names)) if label_names is not None else None

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, labels=all_labels, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, preds, labels=all_labels, average=None, zero_division=0)
    # Median (not mean) of per-class F1 — the metric Abdelaal et al. 2019 (source
    # of this benchmark suite) use as their primary score specifically because it's
    # robust to one or two badly-performing rare classes dragging down a macro-average.
    median_f1 = float(np.median(per_class_f1))
    cm = confusion_matrix(labels, preds, labels=all_labels)
    report = classification_report(labels, preds, labels=all_labels, target_names=label_names, zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "median_f1": median_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def evaluate(
    encoder: scRNAEncoder,
    head: CellTypeClassificationHead,
    loader: DataLoader,
    device: torch.device,
    label_names: list[str] | None = None,
) -> dict:
    preds, labels = collect_predictions(encoder, head, loader, device)
    metrics = compute_metrics(preds, labels, label_names)
    print(f"Accuracy: {metrics['accuracy']:.4f}  |  Macro F1: {metrics['macro_f1']:.4f}")
    print(metrics["classification_report"])
    return metrics
