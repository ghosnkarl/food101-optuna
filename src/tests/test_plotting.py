"""
Visual test for src/utils/plotting.py.

Run with:
    python -m src.tests.test_plotting

Outputs are saved to: results/test_plotting/
Open the PNGs to verify each plot looks correct.
"""

from pathlib import Path
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.utils.plotting import (
    plot_trial_history,
    plot_param_importances,
    save_study_csv,
    save_study_plots,
    plot_comparison,
    save_comparison_csv,
)

SAVE_DIR = Path("results/test_plotting")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_study(name: str, n_trials: int = 20) -> optuna.Study:
    """
    Create a synthetic Optuna study with realistic-looking trial values.

    Each trial samples two params (lr and dropout) and returns a fake
    accuracy value so we can test all plots without real training.
    A few trials are 'pruned' (value stays None) to test the None-filtering.
    """
    study = optuna.create_study(study_name=name, direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        # Simulate accuracy improving over trials with noise
        base = 0.3 + trial.number * 0.02 + np.random.normal(0, 0.05)
        # Penalise high dropout and reward low lr a bit
        acc = base - dropout * 0.1 + (1 - lr) * 0.05
        acc = float(np.clip(acc, 0.1, 0.95))

        # Prune every 5th trial to test None-filtering
        if trial.number % 5 == 4:
            raise optuna.exceptions.TrialPruned()

        return acc

    study.optimize(objective, n_trials=n_trials)
    return study


# ── Test 1: plot_trial_history ─────────────────────────────────────────────


def test_trial_history() -> None:
    print("\n[Test 1] plot_trial_history")
    study = _make_study("test_history", n_trials=20)

    out = SAVE_DIR / "test1_trial_history"
    plot_trial_history(study, out)

    png = out / "trial_history.png"
    assert png.exists(), f"Expected {png} to be created"
    print(f"  Saved: {png}")
    print("  Open it — you should see:")
    print("    • Blue dots: one per completed trial (pruned trials are absent)")
    print("    • Red line:  running best, only goes flat or up, never down")


# ── Test 2: plot_trial_history with no completed trials ────────────────────


def test_trial_history_empty() -> None:
    print("\n[Test 2] plot_trial_history (empty study — all trials pruned)")
    study = optuna.create_study(study_name="empty_study", direction="maximize")

    def always_prune(trial: optuna.Trial) -> float:
        raise optuna.exceptions.TrialPruned()

    study.optimize(always_prune, n_trials=5)

    out = SAVE_DIR / "test2_empty"
    plot_trial_history(study, out)  # should return early, no file created
    png = out / "trial_history.png"
    assert not png.exists(), "No PNG should be created when all trials are pruned"
    print("  Correctly skipped (no PNG created)")


# ── Test 3: plot_param_importances ────────────────────────────────────────


def test_param_importances() -> None:
    print("\n[Test 3] plot_param_importances")
    study = _make_study("test_importances", n_trials=20)

    out = SAVE_DIR / "test3_param_importances"
    plot_param_importances(study, out)

    png = out / "param_importances.png"
    assert png.exists(), f"Expected {png} to be created"
    print(f"  Saved: {png}")
    print("  Open it — you should see:")
    print("    • Horizontal bar chart, one bar per hyperparam (lr, dropout, scheduler)")
    print("    • Bars sorted so the most important param is at the top")


# ── Test 4: save_study_csv ────────────────────────────────────────────────


def test_save_study_csv() -> None:
    print("\n[Test 4] save_study_csv")
    study = _make_study("test_csv", n_trials=10)

    out = SAVE_DIR / "test4_csv"
    save_study_csv(study, out)

    csv = out / "results.csv"
    assert csv.exists(), f"Expected {csv} to be created"
    print(f"  Saved: {csv}")
    print("  Open it — each row is one trial: number, value, params, state, duration")


# ── Test 5: save_study_plots (all-in-one) ────────────────────────────────


def test_save_study_plots() -> None:
    print("\n[Test 5] save_study_plots (all three outputs at once)")
    study = _make_study("test_all_plots", n_trials=20)

    out = SAVE_DIR / "test5_all"
    save_study_plots(study, out)

    for fname in ("trial_history.png", "param_importances.png", "results.csv"):
        assert (out / fname).exists(), f"Missing: {out / fname}"
        print(f"  Saved: {out / fname}")


# ── Test 6: plot_comparison ───────────────────────────────────────────────


def test_plot_comparison() -> None:
    print("\n[Test 6] plot_comparison")

    # Simulate best-trial results from 7 experiments
    results = [
        {"experiment": "scratch_cnn", "top1": 0.42, "top5": 0.68},
        {"experiment": "resnet18_finetune", "top1": 0.71, "top5": 0.92},
        {"experiment": "resnet50_finetune", "top1": 0.74, "top5": 0.93},
        {"experiment": "efficientnet_b0", "top1": 0.73, "top5": 0.92},
        {"experiment": "mobilenet_v3", "top1": 0.68, "top5": 0.90},
        {"experiment": "convnext_tiny", "top1": 0.76, "top5": 0.94},
        {"experiment": "vit_b_16", "top1": 0.78, "top5": 0.95},
    ]

    out = SAVE_DIR / "test6_comparison"
    plot_comparison(results, out)

    png = out / "comparison_accuracy.png"
    assert png.exists(), f"Expected {png} to be created"
    print(f"  Saved: {png}")
    print("  Open it — you should see:")
    print("    • Grouped bar chart, one pair of bars per experiment")
    print("    • Left bar = Top-1, Right bar = Top-5 (always higher)")
    print("    • Y-axis formatted as percentages")


# ── Test 7: save_comparison_csv ──────────────────────────────────────────


def test_save_comparison_csv() -> None:
    print("\n[Test 7] save_comparison_csv")

    results = [
        {
            "experiment": "resnet18_finetune",
            "top1": 0.71,
            "top5": 0.92,
            "best_trial": 7,
        },
        {"experiment": "convnext_tiny", "top1": 0.76, "top5": 0.94, "best_trial": 3},
    ]
    all_trials = [
        {"experiment": "resnet18_finetune", "trial": 0, "value": 0.60},
        {"experiment": "resnet18_finetune", "trial": 1, "value": 0.65},
        {"experiment": "resnet18_finetune", "trial": 7, "value": 0.71},
        {"experiment": "convnext_tiny", "trial": 3, "value": 0.76},
    ]

    out = SAVE_DIR / "test7_comparison_csv"
    save_comparison_csv(results, all_trials, out)

    assert (out / "comparison.csv").exists()
    assert (out / "all_trials.csv").exists()
    print(f"  Saved: {out / 'comparison.csv'}  — best trial per experiment")
    print(f"  Saved: {out / 'all_trials.csv'}   — every trial (flat)")


# ── Runner ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print(f"Saving all test outputs to: {SAVE_DIR.resolve()}")

    test_trial_history()
    test_trial_history_empty()
    test_param_importances()
    test_save_study_csv()
    test_save_study_plots()
    test_plot_comparison()
    test_save_comparison_csv()

    print("\nAll tests passed.")
    print(f"\nOpen the PNGs in:  {SAVE_DIR.resolve()}")
