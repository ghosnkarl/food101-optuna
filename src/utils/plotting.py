from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import optuna
import pandas as pd


# ── Per-study plots ────────────────────────────────────────────────────────────


def plot_trial_history(study: optuna.Study, save_dir: Path) -> None:
    """Top-1 accuracy of every trial, with the running best highlighted."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get all completed trial values (skip pruned or failed trials)
    # Values is a list of floats (top-1 accuracy) or empty if no completed trials
    values = [t.value for t in study.trials if t.value is not None]
    if not values:
        return

    best_so_far = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(values)), values, s=20, alpha=0.6, label="Trial accuracy")
    ax.plot(
        range(len(values)), best_so_far, color="red", linewidth=1.5, label="Best so far"
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Val accuracy (top-1)")
    ax.set_title(f"Optimization history — {study.study_name}")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    fig.tight_layout()
    fig.savefig(save_dir / "trial_history.png", dpi=150)
    plt.close(fig)


def plot_param_importances(study: optuna.Study, save_dir: Path) -> None:
    """Bar chart of hyperparameter importances (requires ≥2 completed trials)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    completed = [t for t in study.trials if t.value is not None]
    if len(completed) < 2:
        return

    importances = optuna.importance.get_param_importances(study)
    if not importances:
        return

    params = list(importances.keys())
    scores = list(importances.values())

    fig, ax = plt.subplots(figsize=(8, max(3, len(params) * 0.4)))
    ax.barh(params[::-1], scores[::-1])
    ax.set_xlabel("Importance score")
    ax.set_title(f"Param importances — {study.study_name}")
    fig.tight_layout()
    fig.savefig(save_dir / "param_importances.png", dpi=150)
    plt.close(fig)


def save_study_csv(study: optuna.Study, save_dir: Path) -> None:
    """Save all trials to results.csv."""
    save_dir.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(save_dir / "results.csv", index=False)


def save_study_plots(study: optuna.Study, save_dir: Path) -> None:
    """Generate and save all per-study plots + CSV."""
    plot_trial_history(study, save_dir)
    plot_param_importances(study, save_dir)
    save_study_csv(study, save_dir)


# ── Cross-experiment comparison plots ─────────────────────────────────────────


def plot_comparison(results: list[dict], save_dir: Path) -> None:
    """
    Bar chart comparing best top-1 and top-5 accuracy across all experiments.

    Each entry in `results` should be:
        {"experiment": str, "top1": float, "top5": float}
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    experiments = [r["experiment"] for r in results]
    top1 = [r["top1"] for r in results]
    top5 = [r["top5"] for r in results]

    x = np.arange(len(experiments))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(experiments) * 1.4), 5))
    ax.bar(x - width / 2, top1, width, label="Top-1 accuracy")
    ax.bar(x + width / 2, top5, width, label="Top-5 accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Best trial accuracy per experiment")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    fig.tight_layout()
    fig.savefig(save_dir / "comparison_accuracy.png", dpi=150)
    plt.close(fig)


def save_comparison_csv(
    results: list[dict], all_trials: list[dict], save_dir: Path
) -> None:
    """
    Save two CSVs to save_dir:
      - comparison.csv  — best trial per experiment
      - all_trials.csv  — every trial across all studies
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(save_dir / "comparison.csv", index=False)
    pd.DataFrame(all_trials).to_csv(save_dir / "all_trials.csv", index=False)
