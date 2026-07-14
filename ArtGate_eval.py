"""Command-line evaluation entry point for ArtGate."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile

from models import ArtGateCLIP
from options import TestOptions
from validate import validate_artgate

ImageFile.LOAD_TRUNCATED_IMAGES = True

METRIC_NAMES = [
    "accuracy",
    "average_precision",
    "real_accuracy",
    "fake_accuracy",
    "f1",
    "auc",
    "tpr_at_fpr_10",
    "tpr_at_fpr_1",
    "eer",
]


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(opt, device):
    model = ArtGateCLIP(num_classes=2 if opt.fc_class2 else 1)
    checkpoint = torch.load(opt.model_path, map_location="cpu", weights_only=True)
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The file passed to --model_path must contain the complete ArtGate "
            "state_dict (CLIP, LoRA, artifact branch, and classifier)."
        ) from error
    return model.to(device).eval()


def evaluate(opt):
    set_random_seed(opt.seed)
    if not opt.device.startswith("cuda"):
        raise ValueError("ArtGate evaluation is GPU-only; --device must be cuda or cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError("ArtGate evaluation requires an NVIDIA GPU with CUDA support")
    device = torch.device(opt.device)
    model = load_model(opt, device)
    model_name = Path(opt.model_path).stem
    rows = [["testset", *METRIC_NAMES]]
    all_metrics = []

    for testset in opt.testsets:
        opt.dataroot = str(Path(opt.dataset_root) / testset)
        values = validate_artgate(
            model,
            opt,
            max_real_size=opt.max_test_image,
            max_fake_size=opt.max_test_image,
            device=device,
        )
        f1, auc, t10, t1, acc, ap, real_acc, fake_acc, _, _, eer = values
        metrics = [acc, ap, real_acc, fake_acc, f1, auc, t10, t1, eer]
        rows.append([testset, *metrics])
        all_metrics.append(metrics)
        print(
            f"{testset}: "
            + "; ".join(
                f"{name}={value:.4f}"
                for name, value in zip(METRIC_NAMES, metrics)
            )
        )

    mean_metrics = np.mean(np.asarray(all_metrics, dtype=np.float64), axis=0)
    rows.append(["Average", *mean_metrics.tolist()])
    output_dir = Path(opt.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_{opt.noise_type or 'clean'}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    print(f"Results written to {output_path}")
    return output_path


def main():
    evaluate(TestOptions().parse(print_options=True))


if __name__ == "__main__":
    main()
