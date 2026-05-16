import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import FiveCrop
import torchmetrics

from src.utils.progress import NestedProgressBar


def _tta_predict(
    model: nn.Module, inputs: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """10-crop TTA: FiveCrop + horizontal flip of each crop. Returns averaged logits."""
    five_crop = FiveCrop(224)
    crops = five_crop(inputs)
    flipped = [torch.flip(c, dims=[-1]) for c in crops]
    all_crops = list(crops) + flipped
    logits_sum = None
    for crop in all_crops:
        with torch.inference_mode():
            logits = model(crop.to(device))
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / len(all_crops)


def evaluate_trial(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 101,
    pbar: NestedProgressBar | None = None,
) -> tuple[float, float]:
    """
    Fast evaluation used per Optuna trial.

    Returns:
        (top1_accuracy, top5_accuracy)
    """
    top1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
    top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

    model.eval()
    with torch.inference_mode():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Discard soft labels from Mixup/CutMix — use hard labels for eval
            if labels.ndim == 2:
                labels = labels.argmax(dim=1)

            outputs = model(inputs)
            top1.update(outputs, labels)
            top5.update(outputs, labels)

            if pbar is not None:
                pbar.update_batch(batch_idx + 1)

    return top1.compute().item(), top5.compute().item()


def evaluate_final(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_classes: int = 101,
    use_tta: bool = False,
) -> dict[str, float]:
    """
    Full evaluation used by train.py after finding the best model.
    Computes top-1, top-5, macro F1/precision/recall and saves metrics.json.

    Returns:
        dict with keys: top1, top5, f1, precision, recall
    """
    top1_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes, top_k=1
    ).to(device)
    top5_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes, top_k=5
    ).to(device)
    f1_metric = torchmetrics.F1Score(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    precision_metric = torchmetrics.Precision(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    recall_metric = torchmetrics.Recall(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)

    pbar = NestedProgressBar(total_epochs=1, total_batches=len(loader), mode="eval")

    model.eval()
    with torch.inference_mode():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if labels.ndim == 2:
                labels = labels.argmax(dim=1)

            outputs = (
                _tta_predict(model, inputs, device)
                if use_tta
                else model(inputs)
            )
            top1_metric.update(outputs, labels)
            top5_metric.update(outputs, labels)
            f1_metric.update(outputs, labels)
            precision_metric.update(outputs, labels)
            recall_metric.update(outputs, labels)
            pbar.update_batch(batch_idx + 1)

    pbar.close("Evaluation complete.")

    results = {
        "top1": top1_metric.compute().item(),
        "top5": top5_metric.compute().item(),
        "f1": f1_metric.compute().item(),
        "precision": precision_metric.compute().item(),
        "recall": recall_metric.compute().item(),
    }

    # Print summary
    print(
        f"Top-1: {results['top1']:.4f} | Top-5: {results['top5']:.4f} | "
        f"F1: {results['f1']:.4f} | Precision: {results['precision']:.4f} | "
        f"Recall: {results['recall']:.4f}"
    )

    # Save metrics to disk
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
