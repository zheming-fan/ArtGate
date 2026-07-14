"""Dataset evaluation loop for ArtGate."""

from __future__ import annotations

import numpy as np
import torch

from data import create_dataloader_test_artgate
from .metrics import compute_metrics


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
