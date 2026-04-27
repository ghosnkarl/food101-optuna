import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from omegaconf import DictConfig


def build_scheduler(
    cfg: DictConfig,
    optimizer: optim.Optimizer,
    n_epochs: int,
    steps_per_epoch: int,
) -> LRScheduler | ReduceLROnPlateau | None:
    """
    Factory that builds a scheduler from a Hydra scheduler config.

    Stepping conventions (handled by train_model, not here):
      - OneCycleLR:        step after every batch
      - ReduceLROnPlateau: step after every epoch with scheduler.step(val_acc)
      - All others:        step after every epoch with scheduler.step()

    Returns None when type is 'none'.
    """
    scheduler_type: str = cfg.type

    if scheduler_type == "none":
        return None

    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            eta_min=cfg.eta_min,
        )

    if scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
        )

    if scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=cfg.mode,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )

    if scheduler_type == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            pct_start=cfg.pct_start,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor,
        )

    raise ValueError(
        f"Unknown scheduler type: {scheduler_type!r}. "
        "Expected one of: none, cosine, step, reduce_on_plateau, one_cycle"
    )
