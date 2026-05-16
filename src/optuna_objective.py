import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
import optuna

from src.data.dataset import get_dataset_loaders
from src.models.flexible_cnn import FlexibleCNN
from src.models.transfer import TransferModel
from src.training.scheduler import build_scheduler
from src.training.train import train_model


def _build_model(trial: optuna.Trial, cfg: DictConfig) -> torch.nn.Module:
    ss = cfg.model.search_space

    if cfg.model.type == "flexible_cnn":
        n_layers = trial.suggest_int("n_layers", ss.n_layers.low, ss.n_layers.high)
        n_filters_start = trial.suggest_int(
            "n_filters_start", ss.n_filters_start.low, ss.n_filters_start.high
        )
        filter_growth = trial.suggest_categorical(
            "filter_growth", list(ss.filter_growth)
        )
        kernel_size = trial.suggest_categorical("kernel_size", list(ss.kernel_sizes))
        dropout = trial.suggest_float(
            "dropout_rate", ss.dropout_rate.low, ss.dropout_rate.high
        )
        fc_size = trial.suggest_int("fc_size", ss.fc_size.low, ss.fc_size.high)

        n_filters = [
            min(n_filters_start * (filter_growth**i), 512) for i in range(n_layers)
        ]

        return FlexibleCNN(
            n_layers=n_layers,
            n_filters=n_filters,
            kernel_sizes=[kernel_size] * n_layers,
            dropout_rate=dropout,
            fc_size=fc_size,
            num_classes=cfg.dataset.num_classes,
        )

    if cfg.model.type == "transfer":
        freeze_backbone = trial.suggest_categorical(
            "freeze_backbone", list(ss.freeze_backbone)
        )
        fine_tune_from_layer = trial.suggest_categorical(
            "fine_tune_from_layer", list(ss.fine_tune_from_layer)
        )
        fc_hidden_size = trial.suggest_int(
            "fc_hidden_size", ss.fc_hidden_size.low, ss.fc_hidden_size.high
        )
        dropout = trial.suggest_float(
            "dropout_rate", ss.dropout_rate.low, ss.dropout_rate.high
        )
        lr_multiplier = trial.suggest_float(
            "lr_multiplier_backbone",
            ss.lr_multiplier_backbone.low,
            ss.lr_multiplier_backbone.high,
            log=True,
        )

        return (
            TransferModel(
                backbone=cfg.model.backbone,
                freeze_backbone=freeze_backbone,
                fine_tune_from_layer=fine_tune_from_layer,
                fc_hidden_size=fc_hidden_size,
                dropout_rate=dropout,
                num_classes=cfg.dataset.num_classes,
                pretrained=cfg.model.pretrained,
            ),
            lr_multiplier,
        )

    raise ValueError(f"Unknown model type: {cfg.model.type!r}")


def _build_optimizer(
    trial: optuna.Trial,
    model: torch.nn.Module,
    optimizer_name: str,
    lr_multiplier_backbone: float | None,
    cfg: DictConfig,
) -> tuple[optim.Optimizer, float]:
    """Returns (optimizer, lr) — lr is needed by OneCycleLR."""
    ss = cfg.optimizers[optimizer_name].search_space

    if optimizer_name == "adam":
        lr = trial.suggest_float("lr", ss.lr.low, ss.lr.high, log=ss.lr.log)
        weight_decay = trial.suggest_float(
            "weight_decay",
            ss.weight_decay.low,
            ss.weight_decay.high,
            log=ss.weight_decay.log,
        )
        if isinstance(model, TransferModel) and lr_multiplier_backbone is not None:
            param_groups = model.param_groups(lr, lr_multiplier_backbone)
        else:
            param_groups = model.parameters()
        return optim.Adam(param_groups, lr=lr, weight_decay=weight_decay), lr

    if optimizer_name == "sgd":
        lr = trial.suggest_float("lr", ss.lr.low, ss.lr.high, log=ss.lr.log)
        momentum = trial.suggest_float("momentum", ss.momentum.low, ss.momentum.high)
        weight_decay = trial.suggest_float(
            "weight_decay",
            ss.weight_decay.low,
            ss.weight_decay.high,
            log=ss.weight_decay.log,
        )
        if isinstance(model, TransferModel) and lr_multiplier_backbone is not None:
            param_groups = model.param_groups(lr, lr_multiplier_backbone)
        else:
            param_groups = model.parameters()
        return (
            optim.SGD(
                param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay
            ),
            lr,
        )

    if optimizer_name == "adamw":
        ss = cfg.optimizers["adamw"].search_space
        lr = trial.suggest_float("lr", ss.lr.low, ss.lr.high, log=ss.lr.log)
        weight_decay = trial.suggest_float(
            "weight_decay",
            ss.weight_decay.low,
            ss.weight_decay.high,
            log=ss.weight_decay.log,
        )
        if isinstance(model, TransferModel) and lr_multiplier_backbone is not None:
            param_groups = model.param_groups(lr, lr_multiplier_backbone)
        else:
            param_groups = model.parameters()
        return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay), lr

    raise ValueError(f"Unknown optimizer: {optimizer_name!r}")


def _build_scheduler_from_trial(
    trial: optuna.Trial,
    scheduler_name: str,
    optimizer: optim.Optimizer,
    lr: float,
    n_epochs: int,
    steps_per_epoch: int,
    cfg: DictConfig,
):
    ss = cfg.schedulers[scheduler_name].search_space

    if scheduler_name == "cosine":
        eta_min = trial.suggest_float(
            "eta_min", ss.eta_min.low, ss.eta_min.high, log=ss.eta_min.log
        )
        sched_cfg = OmegaConf.create({"type": "cosine", "eta_min": eta_min})

    elif scheduler_name == "step":
        step_size = trial.suggest_categorical("step_size", list(ss.step_size))
        gamma = trial.suggest_categorical("gamma", list(ss.gamma))
        sched_cfg = OmegaConf.create(
            {"type": "step", "step_size": step_size, "gamma": gamma}
        )

    elif scheduler_name == "reduce_on_plateau":
        factor = trial.suggest_categorical("factor", list(ss.factor))
        patience = trial.suggest_categorical("patience", list(ss.patience))
        sched_cfg = OmegaConf.create(
            {
                "type": "reduce_on_plateau",
                "mode": "max",
                "factor": factor,
                "patience": patience,
                "min_lr": 1e-6,
            }
        )

    elif scheduler_name == "one_cycle":
        pct_start = trial.suggest_categorical("pct_start", list(ss.pct_start))
        div_factor = trial.suggest_categorical("div_factor", list(ss.div_factor))
        sched_cfg = OmegaConf.create(
            {
                "type": "one_cycle",
                "max_lr": lr,
                "pct_start": pct_start,
                "div_factor": div_factor,
                "final_div_factor": 1e4,
            }
        )

    elif scheduler_name == "warmup_cosine":
        ss = cfg.schedulers["warmup_cosine"].search_space
        warmup_epochs = trial.suggest_categorical(
            "warmup_epochs", list(ss.warmup_epochs)
        )
        eta_min = trial.suggest_float(
            "eta_min", ss.eta_min.low, ss.eta_min.high, log=ss.eta_min.log
        )
        sched_cfg = OmegaConf.create(
            {
                "type": "warmup_cosine",
                "warmup_epochs": warmup_epochs,
                "eta_min": eta_min,
            }
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name!r}")

    return build_scheduler(sched_cfg, optimizer, n_epochs, steps_per_epoch)


def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    """
    Optuna objective function for a single experiment.

    Samples: model hyperparams, optimizer (adam/sgd), scheduler type + hyperparams,
             batch size, and augmentation level. Returns best validation top-1 accuracy.
    """
    device = torch.device(
        "cuda"
        if cfg.device == "auto" and torch.cuda.is_available()
        else cfg.device if cfg.device != "auto" else "cpu"
    )

    # ── Sample dataset params ──────────────────────────────────────────────────
    batch_size = trial.suggest_categorical(
        "batch_size", list(cfg.dataset.search_space.batch_size)
    )
    augmentation_level = trial.suggest_categorical(
        "augmentation_level", list(cfg.dataset.search_space.augmentation_level)
    )

    train_loader, val_loader, _ = get_dataset_loaders(
        cfg.dataset, batch_size, augmentation_level, device=str(device)
    )

    # ── Sample model ───────────────────────────────────────────────────────────
    lr_multiplier_backbone = None
    result = _build_model(trial, cfg)
    if isinstance(result, tuple):
        model, lr_multiplier_backbone = result
    else:
        model = result
    model = model.to(device)

    # ── Sample optimizer ───────────────────────────────────────────────────────
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    optimizer, lr = _build_optimizer(
        trial, model, optimizer_name, lr_multiplier_backbone, cfg
    )

    # ── Sample scheduler ───────────────────────────────────────────────────────
    scheduler_name = trial.suggest_categorical(
        "scheduler",
        ["cosine", "warmup_cosine", "step", "reduce_on_plateau", "one_cycle"],
    )
    scheduler = _build_scheduler_from_trial(
        trial,
        scheduler_name,
        optimizer,
        lr,
        n_epochs=cfg.training.n_epochs,
        steps_per_epoch=len(train_loader),
        cfg=cfg,
    )

    # ── Print trial config before training starts ──────────────────────────────
    def _fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    col1, col2 = [], []
    items = list(trial.params.items())
    mid = (len(items) + 1) // 2
    for k, v in items[:mid]:
        col1.append(f"  {k:<28} {_fmt(v)}")
    for k, v in items[mid:]:
        col2.append(f"  {k:<28} {_fmt(v)}")

    print(f"\n{'─'*60}")
    print(f"  Trial #{trial.number}")
    print(f"{'─'*60}")
    for i in range(max(len(col1), len(col2))):
        left = col1[i] if i < len(col1) else ""
        right = col2[i] if i < len(col2) else ""
        print(f"{left:<40}{right}")
    print(f"{'─'*60}")

    # ── Train ──────────────────────────────────────────────────────────────────
    best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        cfg_training=cfg.training,
        device=device,
        scheduler=scheduler,
        trial=trial,
        num_classes=cfg.dataset.num_classes,
    )

    return best_val_acc
