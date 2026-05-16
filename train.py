"""
Single-run entrypoint. Trains one model with fixed config values (no Optuna).
Saves the best checkpoint to results/<model>/best_model.pt and evaluates on
the official Food-101 test set, writing metrics to results/<model>/metrics.json.

Usage:
    python train.py                                      # defaults from config.yaml
    python train.py model=resnet18 scheduler=one_cycle
    python train.py model=flexible_cnn training.n_epochs=30
"""

from pathlib import Path
import torch
import torch.optim as optim
import hydra
from omegaconf import DictConfig

from src.data.dataset import get_dataset_loaders
from src.models.flexible_cnn import build_flexible_cnn
from src.models.transfer import build_transfer_model, TransferModel
from src.training.scheduler import build_scheduler
from src.training.train import train_model
from src.training.evaluate import evaluate_final


def _resolve_device(cfg: DictConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> optim.Optimizer:
    opt_cfg = cfg.optimizer

    if isinstance(model, TransferModel):
        param_groups = model.param_groups(opt_cfg.lr, cfg.model.lr_multiplier_backbone)
    else:
        param_groups = model.parameters()

    if opt_cfg.type == "adam":
        return optim.Adam(
            param_groups, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay
        )

    if opt_cfg.type == "adamw":
        return optim.AdamW(
            param_groups, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay
        )

    if opt_cfg.type == "sgd":
        return optim.SGD(
            param_groups,
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
        )

    raise ValueError(f"Unknown optimizer type: {opt_cfg.type!r}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(
        f"Device: {device} | Model: {cfg.model.type} | "
        f"Optimizer: {cfg.optimizer.type} | Scheduler: {cfg.scheduler.type}"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataset_loaders(
        cfg.dataset,
        batch_size=cfg.dataset.batch_size,
        augmentation_level=cfg.dataset.augmentation.level,
        device=str(device),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if cfg.model.type == "flexible_cnn":
        model = build_flexible_cnn(cfg.model, num_classes=cfg.dataset.num_classes)
    else:
        model = build_transfer_model(cfg.model, num_classes=cfg.dataset.num_classes)
    model = model.to(device)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = _build_optimizer(cfg, model)
    scheduler = build_scheduler(
        cfg.scheduler,
        optimizer,
        n_epochs=cfg.training.n_epochs,
        steps_per_epoch=len(train_loader),
    )

    save_dir = Path("results") / cfg.model.get("backbone", cfg.model.type)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        cfg_training=cfg.training,
        device=device,
        scheduler=scheduler,
        trial=None,
        checkpoint_path=save_dir / "best_model.pt",
        num_classes=cfg.dataset.num_classes,
    )

    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))

    print("\nEvaluating on official Food-101 test set...")
    evaluate_final(
        model,
        test_loader,
        device,
        save_dir=save_dir,
        num_classes=cfg.dataset.num_classes,
        use_tta=cfg.training.get("tta", False),
    )


if __name__ == "__main__":
    main()
