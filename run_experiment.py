"""
Experiment Pipeline for the MoE Mid-Price Movement Prediction Framework.

Runs the pipeline on a single file (Train/Test_Dst_NoAuction_ZScore_CF_1)
from the FI-2010 benchmark dataset:
  - Stage 1: Train experts (LR, XGBoost, MLP)
  - Stage 2: Generate stacked expert probabilities
  - Stage 3: Train gating network
  - Stage 4: Strategy + backtracking + backtesting
  - Stage 5: Report metrics

Usage:
    python run_experiment.py

Configure DATASET_ROOT below to point to your FI-2010 dataset location.
"""

import os
import sys
import time
import random
import numpy as np
import torch

# ============================================================
# CONFIGURATION — Edit these as needed
# ============================================================
DATASET_ROOT = "BenchmarkDatasets"          # Path to FI-2010 dataset root
NORMALIZATION = "NoAuction_Zscore"          # Normalization variant
RESULTS_DIR = "results"                     # Directory for saving outputs
# ----------------------------------------------------------------
# ORIGINAL: Multi-fold configuration
# ----------------------------------------------------------------
# N_FOLDS = 9                                 # Number of CV folds
# ----------------------------------------------------------------
SEED = 42                                   # Global random seed
SINGLE_FILE = "NoAuction_ZScore_CF_1"       # Single file used for train/test
# ============================================================


def set_global_seed(seed):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    """Run the experiment pipeline on a single file."""
    set_global_seed(SEED)

    # ----------------------------------------------------------------
    # ORIGINAL: Header for multi-fold pipeline
    # ----------------------------------------------------------------
    # print("=" * 70)
    # print("  Mixture-of-Experts Framework for Mid-Price Movement Prediction")
    # print("  FI-2010 Benchmark Dataset — Full Pipeline")
    # print("=" * 70)
    # print(f"  Dataset root:    {DATASET_ROOT}")
    # print(f"  Normalization:   {NORMALIZATION}")
    # print(f"  Results dir:     {RESULTS_DIR}")
    # print(f"  Number of folds: {N_FOLDS}")
    # print(f"  Random seed:     {SEED}")
    # print("=" * 70)
    # ----------------------------------------------------------------

    # SINGLE FILE: Header
    print("=" * 70)
    print("  Mixture-of-Experts Framework for Mid-Price Movement Prediction")
    print("  FI-2010 Benchmark Dataset — Single File Pipeline")
    print("=" * 70)
    print(f"  Dataset root:    {DATASET_ROOT}")
    print(f"  Normalization:   {NORMALIZATION}")
    print(f"  Train/Test file: {SINGLE_FILE}")
    print(f"  Results dir:     {RESULTS_DIR}")
    print(f"  Random seed:     {SEED}")
    print("=" * 70)

    # Verify dataset exists
    expected_path = os.path.join(
        DATASET_ROOT, "NoAuction", "1.NoAuction_Zscore",
        "NoAuction_Zscore_Training"
    )
    if not os.path.isdir(expected_path):
        print(f"\n  ERROR: Dataset directory not found at '{expected_path}'")
        print(f"  Please ensure the FI-2010 dataset is placed at '{DATASET_ROOT}'")
        print(f"  with the correct directory structure.")
        print(f"  See README.md for setup instructions.")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # ORIGINAL: Multi-fold pipeline loop
    # ----------------------------------------------------------------
    # # Import pipeline modules
    # from data.cross_val import anchored_forward_cv
    # from train import train_fold
    # from evaluate import evaluate_fold
    #
    # # Storage for all fold metrics
    # all_eval_metrics = []
    # all_fold_results = []
    #
    # total_start = time.time()
    #
    # # ========== Process each fold ==========
    # for fold_idx in anchored_forward_cv(N_FOLDS):
    #     fold_start = time.time()
    #     set_global_seed(SEED + fold_idx)
    #
    #     # Stages 1–3: Training
    #     fold_results = train_fold(
    #         fold_idx=fold_idx,
    #         dataset_root=DATASET_ROOT,
    #         normalization=NORMALIZATION,
    #         results_dir=RESULTS_DIR
    #     )
    #     all_fold_results.append(fold_results)
    #
    #     # Stage 4: Evaluation
    #     eval_metrics = evaluate_fold(
    #         fold_idx=fold_idx,
    #         fold_results=fold_results,
    #         results_dir=RESULTS_DIR
    #     )
    #     all_eval_metrics.append(eval_metrics)
    #
    #     fold_time = time.time() - fold_start
    #     print(f"  Fold {fold_idx}/9 completed in {fold_time:.1f}s")
    #
    # total_time = time.time() - total_start
    # ----------------------------------------------------------------

    # SINGLE FILE: Process only CF_1
    from train import train_fold
    from evaluate import evaluate_fold

    total_start = time.time()

    fold_idx = 1
    set_global_seed(SEED + fold_idx)

    # Stages 1–3: Training
    fold_results = train_fold(
        fold_idx=fold_idx,
        dataset_root=DATASET_ROOT,
        normalization=NORMALIZATION,
        results_dir=RESULTS_DIR
    )

    # Stage 4: Evaluation
    eval_metrics = evaluate_fold(
        fold_idx=fold_idx,
        fold_results=fold_results,
        results_dir=RESULTS_DIR
    )

    run_time = time.time() - total_start
    print(f"  {SINGLE_FILE} completed in {run_time:.1f}s")

    total_time = run_time

    # ----------------------------------------------------------------
    # ORIGINAL: Stage 5 — Aggregated results across all folds
    # ----------------------------------------------------------------
    # # ========== Stage 5: Aggregation ==========
    # print("\n" + "=" * 70)
    # print("  STAGE 5: AGGREGATED RESULTS ACROSS ALL FOLDS")
    # print("=" * 70)
    #
    # # Collect individual expert metrics
    # lr_accs = [m['lr_metrics']['accuracy'] for m in all_eval_metrics]
    # lr_f1s = [m['lr_metrics']['macro_f1'] for m in all_eval_metrics]
    # xgb_accs = [m['xgb_metrics']['accuracy'] for m in all_eval_metrics]
    # xgb_f1s = [m['xgb_metrics']['macro_f1'] for m in all_eval_metrics]
    # mlp_accs = [m['mlp_metrics']['accuracy'] for m in all_eval_metrics]
    # mlp_f1s = [m['mlp_metrics']['macro_f1'] for m in all_eval_metrics]
    #
    # # MoE without backtracking
    # moe_no_bt_accs = [m['no_backtracking']['accuracy'] for m in all_eval_metrics]
    # moe_no_bt_f1s = [m['no_backtracking']['macro_f1'] for m in all_eval_metrics]
    # moe_no_bt_returns = [m['no_backtracking']['cum_return'] for m in all_eval_metrics]
    # moe_no_bt_sharpes = [m['no_backtracking']['sharpe'] for m in all_eval_metrics]
    # moe_no_bt_mdds = [m['no_backtracking']['mdd'] for m in all_eval_metrics]
    # moe_no_bt_wrs = [m['no_backtracking']['win_rate'] for m in all_eval_metrics]
    # moe_no_bt_das = [m['no_backtracking']['decision_acc'] for m in all_eval_metrics]
    #
    # # MoE with backtracking
    # moe_bt_accs = [m['with_backtracking']['accuracy'] for m in all_eval_metrics]
    # moe_bt_f1s = [m['with_backtracking']['macro_f1'] for m in all_eval_metrics]
    # moe_bt_returns = [m['with_backtracking']['cum_return'] for m in all_eval_metrics]
    # moe_bt_sharpes = [m['with_backtracking']['sharpe'] for m in all_eval_metrics]
    # moe_bt_mdds = [m['with_backtracking']['mdd'] for m in all_eval_metrics]
    # moe_bt_wrs = [m['with_backtracking']['win_rate'] for m in all_eval_metrics]
    # moe_bt_das = [m['with_backtracking']['decision_acc'] for m in all_eval_metrics]
    #
    # # Buy-and-hold
    # bh_returns = [m['buy_and_hold_return'] for m in all_eval_metrics]
    #
    # # Per-class F1 (MoE with backtracking)
    # per_class_f1s = np.array([m['with_backtracking']['per_class_f1']
    #                           for m in all_eval_metrics])
    #
    # def fmt(values):
    #     """Format mean ± std."""
    #     return f"{np.mean(values):.4f} ± {np.std(values):.4f}"
    #
    # # Print Classification Metrics
    # print("\n  --- CLASSIFICATION METRICS (mean ± std across folds) ---\n")
    # print(f"  {'Model':<25} {'Accuracy':<20} {'Macro-F1':<20}")
    # print(f"  {'-'*65}")
    # print(f"  {'LR (Expert 1)':<25} {fmt(lr_accs):<20} {fmt(lr_f1s):<20}")
    # print(f"  {'XGBoost (Expert 2)':<25} {fmt(xgb_accs):<20} {fmt(xgb_f1s):<20}")
    # print(f"  {'MLP (Expert 3)':<25} {fmt(mlp_accs):<20} {fmt(mlp_f1s):<20}")
    # print(f"  {'MoE (no backtracking)':<25} {fmt(moe_no_bt_accs):<20} "
    #       f"{fmt(moe_no_bt_f1s):<20}")
    # print(f"  {'MoE (with backtracking)':<25} {fmt(moe_bt_accs):<20} "
    #       f"{fmt(moe_bt_f1s):<20}")
    #
    # print(f"\n  Per-class F1 (MoE with backtracking):")
    # print(f"    Up (class 0):         {np.mean(per_class_f1s[:, 0]):.4f} "
    #       f"± {np.std(per_class_f1s[:, 0]):.4f}")
    # print(f"    Stationary (class 1): {np.mean(per_class_f1s[:, 1]):.4f} "
    #       f"± {np.std(per_class_f1s[:, 1]):.4f}")
    # print(f"    Down (class 2):       {np.mean(per_class_f1s[:, 2]):.4f} "
    #       f"± {np.std(per_class_f1s[:, 2]):.4f}")
    #
    # # Print Financial Metrics
    # print("\n  --- FINANCIAL / STRATEGY METRICS (mean ± std across folds) ---\n")
    # print(f"  {'Metric':<25} {'No Backtracking':<25} {'With Backtracking':<25}")
    # print(f"  {'-'*75}")
    # print(f"  {'Cumulative Return':<25} {fmt(moe_no_bt_returns):<25} "
    #       f"{fmt(moe_bt_returns):<25}")
    # print(f"  {'Sharpe Ratio':<25} {fmt(moe_no_bt_sharpes):<25} "
    #       f"{fmt(moe_bt_sharpes):<25}")
    # print(f"  {'Max Drawdown':<25} {fmt(moe_no_bt_mdds):<25} "
    #       f"{fmt(moe_bt_mdds):<25}")
    # print(f"  {'Win Rate':<25} {fmt(moe_no_bt_wrs):<25} "
    #       f"{fmt(moe_bt_wrs):<25}")
    # print(f"  {'Decision Accuracy':<25} {fmt(moe_no_bt_das):<25} "
    #       f"{fmt(moe_bt_das):<25}")
    #
    # print(f"\n  Buy-and-Hold Return:    {fmt(bh_returns)}")
    #
    # # Action stability
    # stab_no_bt = [m['action_stability_no_bt'] for m in all_eval_metrics]
    # stab_bt = [m['action_stability_bt'] for m in all_eval_metrics]
    # print(f"\n  Action Stability (lower = more stable):")
    # print(f"    No backtracking:      {fmt(stab_no_bt)}")
    # print(f"    With backtracking:    {fmt(stab_bt)}")
    #
    # # Trade counts
    # n_trades_no = [m['n_trades_no_bt'] for m in all_eval_metrics]
    # n_trades_bt = [m['n_trades_bt'] for m in all_eval_metrics]
    # print(f"\n  Average trades per fold:")
    # print(f"    No backtracking:      {np.mean(n_trades_no):.1f}")
    # print(f"    With backtracking:    {np.mean(n_trades_bt):.1f}")
    #
    # print(f"\n  Total experiment time: {total_time:.1f}s")
    # print("=" * 70)
    # print("  Experiment complete. Results saved to:", RESULTS_DIR)
    # print("=" * 70)
    # ----------------------------------------------------------------

    # SINGLE FILE: Stage 5 — Results for CF_1
    print("\n" + "=" * 70)
    print(f"  STAGE 5: RESULTS FOR {SINGLE_FILE}")
    print("=" * 70)

    m = eval_metrics

    # Print Classification Metrics
    print(f"\n  --- CLASSIFICATION METRICS ({SINGLE_FILE}) ---\n")
    print(f"  {'Model':<25} {'Accuracy':<20} {'Macro-F1':<20}")
    print(f"  {'-'*65}")
    print(f"  {'LR (Expert 1)':<25} {m['lr_metrics']['accuracy']:<20.4f} {m['lr_metrics']['macro_f1']:<20.4f}")
    print(f"  {'XGBoost (Expert 2)':<25} {m['xgb_metrics']['accuracy']:<20.4f} {m['xgb_metrics']['macro_f1']:<20.4f}")
    print(f"  {'MLP (Expert 3)':<25} {m['mlp_metrics']['accuracy']:<20.4f} {m['mlp_metrics']['macro_f1']:<20.4f}")
    print(f"  {'MoE (no backtracking)':<25} {m['no_backtracking']['accuracy']:<20.4f} "
          f"{m['no_backtracking']['macro_f1']:<20.4f}")
    print(f"  {'MoE (with backtracking)':<25} {m['with_backtracking']['accuracy']:<20.4f} "
          f"{m['with_backtracking']['macro_f1']:<20.4f}")

    per_class_f1 = m['with_backtracking']['per_class_f1']
    print(f"\n  Per-class F1 (MoE with backtracking):")
    print(f"    Up (class 0):         {per_class_f1[0]:.4f}")
    print(f"    Stationary (class 1): {per_class_f1[1]:.4f}")
    print(f"    Down (class 2):       {per_class_f1[2]:.4f}")

    # Print Financial Metrics
    print(f"\n  --- FINANCIAL / STRATEGY METRICS ({SINGLE_FILE}) ---\n")
    print(f"  {'Metric':<25} {'No Backtracking':<25} {'With Backtracking':<25}")
    print(f"  {'-'*75}")
    print(f"  {'Cumulative Return':<25} {m['no_backtracking']['cum_return']:<25.4f} "
          f"{m['with_backtracking']['cum_return']:<25.4f}")
    print(f"  {'Sharpe Ratio':<25} {m['no_backtracking']['sharpe']:<25.4f} "
          f"{m['with_backtracking']['sharpe']:<25.4f}")
    print(f"  {'Max Drawdown':<25} {m['no_backtracking']['mdd']:<25.4f} "
          f"{m['with_backtracking']['mdd']:<25.4f}")
    print(f"  {'Win Rate':<25} {m['no_backtracking']['win_rate']:<25.4f} "
          f"{m['with_backtracking']['win_rate']:<25.4f}")
    print(f"  {'Decision Accuracy':<25} {m['no_backtracking']['decision_acc']:<25.4f} "
          f"{m['with_backtracking']['decision_acc']:<25.4f}")

    print(f"\n  Buy-and-Hold Return:    {m['buy_and_hold_return']:.4f}")

    # Action stability
    print(f"\n  Action Stability (lower = more stable):")
    print(f"    No backtracking:      {m['action_stability_no_bt']:.4f}")
    print(f"    With backtracking:    {m['action_stability_bt']:.4f}")

    # Trade counts
    print(f"\n  Trade count:")
    print(f"    No backtracking:      {m['n_trades_no_bt']}")
    print(f"    With backtracking:    {m['n_trades_bt']}")

    print(f"\n  Total experiment time: {total_time:.1f}s")
    print("=" * 70)
    print("  Experiment complete. Results saved to:", RESULTS_DIR)
    print("=" * 70)


if __name__ == '__main__':
    main()
