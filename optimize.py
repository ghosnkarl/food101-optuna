"""
Optuna entrypoint. Runs a separate study per experiment, then generates a comparison report.

Usage:
    python optimize.py                                           # all experiments
    python optimize.py experiments=[scratch_cnn,resnet18_finetune]
    python optimize.py optuna.n_trials=10 training.n_epochs=5   # quick run
    python optimize.py experiments=[resnet18_finetune]          # single experiment
"""

from pathlib import Path
from functools import partial
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState
import hydra
from omegaconf import DictConfig, OmegaConf

from src.optuna_objective import objective
from src.utils.plotting import plot_comparison, save_comparison_csv, save_study_plots


def _resolve_device(cfg: DictConfig) -> str:
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def _compose_experiment_cfg(experiment: str, cfg: DictConfig) -> DictConfig:
    """
    Manually compose the experiment config without Hydra's composition system.

    The experiment yamls use Hydra's `defaults` list (e.g. `- override /model: resnet18`)
    which is a Hydra-only directive — not a real config key. OmegaConf.merge() rejects it
    because the composed cfg struct doesn't have a `defaults` field.

    Fix: parse the defaults list ourselves, load the referenced model config directly,
    then merge everything as plain dicts (bypassing struct restriction).
    """
    raw: dict = OmegaConf.to_container(
        OmegaConf.load(Path("configs/experiment") / f"{experiment}.yaml"),
        resolve=False,
    )

    # Parse defaults list to find model override: `- override /model: resnet18`
    model_name: str | None = None
    for item in raw.get("defaults", []):
        if isinstance(item, dict):
            for key, val in item.items():
                if "model" in key:
                    model_name = val

    # Convert base config to plain dict (removes struct restriction)
    cfg_dict: dict = OmegaConf.to_container(cfg, resolve=True)

    # Apply model override
    if model_name:
        cfg_dict["model"] = OmegaConf.to_container(
            OmegaConf.load(f"configs/model/{model_name}.yaml"), resolve=True
        )

    # Load all optimizer and scheduler configs so optuna_objective can read their
    # search_space sections instead of using hardcoded values.
    cfg_dict["optimizers"] = {
        name: OmegaConf.to_container(
            OmegaConf.load(f"configs/optimizer/{name}.yaml"), resolve=False
        )
        for name in ["adam", "adamw", "sgd"]
    }
    cfg_dict["schedulers"] = {
        name: OmegaConf.to_container(
            OmegaConf.load(f"configs/scheduler/{name}.yaml"), resolve=False
        )
        for name in ["cosine", "warmup_cosine", "step", "reduce_on_plateau", "one_cycle"]
    }

    # Apply remaining non-defaults fields (optuna overrides etc.)
    for key, val in raw.items():
        if key == "defaults":
            continue
        if (
            isinstance(val, dict)
            and key in cfg_dict
            and isinstance(cfg_dict[key], dict)
        ):
            cfg_dict[key] = {**cfg_dict[key], **val}
        else:
            cfg_dict[key] = val

    return OmegaConf.create(cfg_dict)


def _run_study(experiment: str, cfg: DictConfig, device: str) -> dict:
    """Load the experiment override, run its Optuna study, return best trial summary."""

    exp_cfg = _compose_experiment_cfg(experiment, cfg)
    exp_cfg.device = device

    study_name: str = exp_cfg.optuna.study_name
    n_trials: int = exp_cfg.optuna.n_trials
    storage: str = exp_cfg.optuna.storage

    Path("results").mkdir(exist_ok=True)

    sampler = TPESampler(seed=cfg.seed)
    pruner = MedianPruner(
        n_startup_trials=exp_cfg.optuna.pruner.n_startup_trials,
        n_warmup_steps=exp_cfg.optuna.pruner.n_warmup_steps,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=exp_cfg.optuna.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Mark any stale RUNNING trials (from a previous crashed run) as failed so they get re-run
    for t in study.trials:
        if t.state == TrialState.RUNNING:
            study.tell(t.number, state=TrialState.FAIL)

    prior = study.trials
    n_complete = sum(1 for t in prior if t.state == TrialState.COMPLETE)
    n_pruned   = sum(1 for t in prior if t.state == TrialState.PRUNED)
    n_failed   = sum(1 for t in prior if t.state == TrialState.FAIL)
    n_done     = n_complete + n_pruned
    n_remaining = max(0, n_trials - n_done)

    print(f"\n{'='*60}")
    print(f"  Experiment : {experiment}")
    print(f"  Study      : {study_name}")
    if prior:
        print(f"  Prior      : {len(prior)} trials  "
              f"({n_complete} complete, {n_pruned} pruned, {n_failed} failed)")
    print(f"  Target     : {n_trials} trials  |  Done: {n_done}  |  Remaining: {n_remaining}")
    print(f"  Epochs/trial: {exp_cfg.training.n_epochs}  |  Next trial: #{len(prior)}")
    print(f"{'='*60}")

    study.optimize(
        partial(objective, cfg=exp_cfg),
        n_trials=n_remaining,
        show_progress_bar=False,
    )

    # Save per-study artifacts
    save_dir = Path("results") / study_name
    save_study_plots(study, save_dir)

    best = study.best_trial
    top5 = best.user_attrs.get("top5", float("nan"))

    print(f"\n  Best trial #{best.number}: val_acc={best.value:.4f}  top5={top5:.4f}")
    print(f"  Params: {best.params}")

    return {
        "experiment": experiment,
        "study_name": study_name,
        "best_trial": best.number,
        "top1": best.value,
        "top5": top5,
        **{f"param_{k}": v for k, v in best.params.items()},
    }


def _collect_all_trials(cfg: DictConfig, experiments: list[str]) -> list[dict]:
    """Gather every trial from every study into a flat list for all_trials.csv."""
    storage = cfg.optuna.storage
    rows = []
    for exp in experiments:
        exp_cfg = _compose_experiment_cfg(exp, cfg)
        try:
            study = optuna.load_study(
                study_name=exp_cfg.optuna.study_name, storage=storage
            )
            for t in study.trials:
                if t.value is not None:
                    rows.append(
                        {
                            "experiment": exp,
                            "trial": t.number,
                            "top1": t.value,
                            "top5": t.user_attrs.get("top5", float("nan")),
                            **t.params,
                        }
                    )
        except Exception:
            pass
    return rows


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    experiments: list[str] = list(cfg.experiments)

    print(f"Device: {device}")
    print(f"Running {len(experiments)} experiment(s): {experiments}")

    results: list[dict] = []
    for experiment in experiments:
        summary = _run_study(experiment, cfg, device)
        results.append(summary)

    # ── Comparison report ─────────────────────────────────────────────────────
    comparison_dir = Path("results/comparison")
    all_trials = _collect_all_trials(cfg, experiments)

    plot_comparison(results, comparison_dir)
    save_comparison_csv(results, all_trials, comparison_dir)

    print(f"\n{'='*60}")
    print("  All studies complete. Results saved to results/comparison/")
    print(f"{'='*60}")
    print(f"\n  {'Experiment':<25} {'Top-1':>8} {'Top-5':>8}")
    print(f"  {'-'*43}")
    for r in sorted(results, key=lambda x: x["top1"], reverse=True):
        print(f"  {r['experiment']:<25} {r['top1']:>8.4f} {r['top5']:>8.4f}")

    print(f"\n  Dashboard: optuna-dashboard {cfg.optuna.storage}")


if __name__ == "__main__":
    main()
