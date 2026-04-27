from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import optuna
from omegaconf import DictConfig
from src.training.evaluate import evaluate_trial

from src.utils.progress import NestedProgressBar


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    gradient_clip_val: float,
    scheduler: object | None,
    pbar: NestedProgressBar,
) -> tuple[float, float]:
    """
    Trains the model for one epoch.

    Returns:
        (avg_loss, top1_accuracy) for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(inputs)
            # labels may be soft ([B, C]) from Mixup/CutMix or hard ([B]) indices
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()

        if gradient_clip_val > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        scaler.step(optimizer)
        scaler.update()

        # OneCycleLR steps every batch
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)

        # Accuracy: use argmax for soft labels, labels directly for hard
        hard_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels
        predicted = outputs.argmax(dim=1)
        correct += predicted.eq(hard_labels).sum().item()
        total += inputs.size(0)

        pbar.update_batch(batch_idx + 1, {"loss": f"{loss.item():.4f}"})

    return running_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    cfg_training: DictConfig,
    device: torch.device,
    scheduler: object | None = None,
    trial: optuna.Trial | None = None,
    checkpoint_path: Path | None = None,
    num_classes: int = 101,
) -> float:
    """
    Full training loop with AMP, early stopping, Optuna pruning, and scheduler stepping.

    Returns:
        Best validation top-1 accuracy achieved across all epochs.
    """

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg_training.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=cfg_training.amp and device.type == "cuda")

    n_epochs: int = cfg_training.n_epochs
    patience: int = cfg_training.early_stopping.patience
    min_delta: float = cfg_training.early_stopping.min_delta

    best_val_acc: float = 0.0
    epochs_without_improvement: int = 0

    pbar = NestedProgressBar(
        total_epochs=n_epochs,
        total_batches=len(train_loader),
        mode="train",
        epoch_message_freq=5,
    )

    for epoch in range(1, n_epochs + 1):
        pbar.begin_epoch()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            cfg_training.gradient_clip_val,
            scheduler,
            pbar,
        )

        pbar.start_validation(len(val_loader))
        val_acc, val_top5 = evaluate_trial(model, val_loader, device, num_classes=num_classes, pbar=pbar)
        pbar.end_validation()

        # Epoch-level scheduler stepping
        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        pbar.update_epoch(
            epoch,
            {
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.3f}",
                "val_acc": f"{val_acc:.3f}",
            },
        )
        pbar.maybe_log_epoch(
            epoch,
            f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"train {train_acc:.3f} | val {val_acc:.3f} (top5 {val_top5:.3f})",
        )

        # Optuna pruning
        if trial is not None and cfg_training.pruning:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                pbar.close()
                raise optuna.TrialPruned()

        # Early stopping + checkpoint
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if (
                cfg_training.early_stopping.enabled
                and epochs_without_improvement >= patience
            ):
                pbar.maybe_log_epoch(epoch, f"Early stopping at epoch {epoch}")
                break

    pbar.close()
    return best_val_acc
