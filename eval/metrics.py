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


def _forward_head(encoder, head, input_ids, attention_mask, labels=None):
    """Unified forward pass for CLS head and GAT head."""
    if isinstance(head, CellGATClassificationHead):
        hidden = encoder(input_ids, attention_mask)
        return head(hidden, attention_mask, input_ids, labels)
    else:
        cls_emb = encoder.get_cls_embedding(input_ids, attention_mask)
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

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cell_type      = batch["cell_type"].to(device)

        _, logits = _forward_head(encoder, head, input_ids, attention_mask)
        preds_list.append(logits.argmax(dim=-1).cpu().numpy())
        labels_list.append(cell_type.cpu().numpy())

    return np.concatenate(preds_list), np.concatenate(labels_list)


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=label_names, zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
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
