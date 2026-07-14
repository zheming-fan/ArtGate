"""Evaluation helpers for ArtGate."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from data import create_dataloader_test_artgate


def tpr_at_fpr(y_true, y_scores, target_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return float(np.interp(target_fpr, fpr, tpr))


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    index = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[index] + fnr[index]) / 2)


def compute_metrics(y_true, y_pred, threshold=0.5):
    """Compute all reported metrics from labels and fake probabilities."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("labels and predictions must be non-empty and equally sized")
    if np.unique(y_true).size != 2:
        raise ValueError("evaluation requires both real (0) and fake (1) samples")

    predicted_labels = y_pred > threshold
    real = y_true == 0
    fake = y_true == 1
    return {
        "f1": float(f1_score(y_true, predicted_labels)),
        "auc": float(roc_auc_score(y_true, y_pred)),
        "tpr_at_fpr_10": tpr_at_fpr(y_true, y_pred, target_fpr=0.1),
        "tpr_at_fpr_1": tpr_at_fpr(y_true, y_pred, target_fpr=0.01),
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "average_precision": float(average_precision_score(y_true, y_pred)),
        "real_accuracy": float(accuracy_score(y_true[real], predicted_labels[real])),
        "fake_accuracy": float(accuracy_score(y_true[fake], predicted_labels[fake])),
        "eer": compute_eer(y_true, y_pred),
    }


def validate_artgate(
    model,
    opt,
    max_real_size=None,
    max_fake_size=None,
    *,
    device=None,
):
    data_loader = create_dataloader_test_artgate(
        opt, max_real_size=max_real_size, max_fake_size=max_fake_size
    )
    device = torch.device(device or next(model.parameters()).device)
    y_true, y_pred = [], []

    model.eval()
    with torch.inference_mode():
        for index, (images, labels, artifact_images) in enumerate(data_loader, 1):
            print(f"batch number {index}/{len(data_loader)}", end="\r")
            logits = model(images.to(device), artifact_images.to(device))
            probabilities = (
                torch.softmax(logits, dim=1)[:, 1]
                if opt.fc_class2
                else logits.sigmoid().flatten()
            )
            y_pred.extend(probabilities.cpu().tolist())
            y_true.extend(labels.flatten().tolist())

    metrics = compute_metrics(y_true, y_pred)
    # Preserve the public return shape used by existing callers.
    return (
        metrics["f1"],
        metrics["auc"],
        metrics["tpr_at_fpr_10"],
        metrics["tpr_at_fpr_1"],
        metrics["accuracy"],
        metrics["average_precision"],
        metrics["real_accuracy"],
        metrics["fake_accuracy"],
        np.asarray(y_true),
        np.asarray(y_pred),
        metrics["eer"],
    )
