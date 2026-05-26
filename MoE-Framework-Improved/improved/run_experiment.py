"""
Experiment Pipeline for the MoE Mid-Price Movement Prediction Framework.

Runs the full pipeline on the FI-2010 benchmark dataset:
  Stage 1: Train experts (LR, XGBoost, MLP)
  Stage 2: Generate stacked expert probabilities (inner CV)
  Stage 3: Train gating network
  Stage 4: Strategy + backtracking + backtesting
  Stage 5: Aggregate and report results

Usage:
    python run_experiment.py
"""

import os

# Prevent macOS OpenMP segfaults
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import random
import numpy as np
import torch

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = "data/BenchmarkDatasets"
NORMALIZATION = "NoAuction_Zscore"
RESULTS_DIR = "results"
N_FOLDS = 9
SEED = 42
# ============================================================


def set_global_seed(seed):
    """Set seeds for full reproducibility."""
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
    set_global_seed(SEED)

    print("=" * 70, flush=True)
    print("  Mixture-of-Experts Framework — Mid-Price Movement Prediction")
    print("  FI-2010 Benchmark Dataset — Full Pipeline (Improved)")
    print("=" * 70)
    print(f"  Dataset root:    {DATASET_ROOT}")
    print(f"  Normalization:   {NORMALIZATION}")
    print(f"  Results dir:     {RESULTS_DIR}")
    print(f"  Folds:           {N_FOLDS}")
    print(f"  Seed:            {SEED}")
    print(f"  GPU available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device:      {torch.cuda.get_device_name(0)}")
    print("=" * 70, flush=True)

    # Verify dataset
    expected_path = os.path.join(
        DATASET_ROOT, "NoAuction", "1.NoAuction_Zscore",
        "NoAuction_Zscore_Training"
    )
    if not os.path.isdir(expected_path):
        print(f"\n  ERROR: Dataset not found at '{expected_path}'")
        print(f"  See README.md for setup instructions.")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    from data.cross_val import anchored_forward_cv
    from train import train_fold
    from evaluate import evaluate_fold

    all_eval_metrics = []
    all_fold_results = []

    total_start = time.time()

    for fold_idx in anchored_forward_cv(N_FOLDS):
        fold_start = time.time()
        set_global_seed(SEED + fold_idx)

        fold_results = train_fold(
            fold_idx=fold_idx,
            dataset_root=DATASET_ROOT,
            normalization=NORMALIZATION,
            results_dir=RESULTS_DIR
        )
        all_fold_results.append(fold_results)

        eval_metrics = evaluate_fold(
            fold_idx=fold_idx,
            fold_results=fold_results,
            results_dir=RESULTS_DIR
        )
        all_eval_metrics.append(eval_metrics)

        fold_time = time.time() - fold_start
        print(f"  Fold {fold_idx}/{N_FOLDS} completed in {fold_time:.1f}s",
              flush=True)

    total_time = time.time() - total_start

    # ========== Stage 5: Aggregation ==========
    print("\n" + "=" * 70)
    print("  STAGE 5: AGGREGATED RESULTS ACROSS ALL FOLDS")
    print("=" * 70)

    # Helper
    def fmt(values):
        return f"{np.mean(values):.4f} ± {np.std(values):.4f}"
    def fmt_int(values):
        return f"{np.mean(values):.1f} ± {np.std(values):.1f}"

    # Classification
    lr_accs  = [m['lr_metrics']['accuracy'] for m in all_eval_metrics]
    lr_f1s   = [m['lr_metrics']['macro_f1'] for m in all_eval_metrics]
    xgb_accs = [m['xgb_metrics']['accuracy'] for m in all_eval_metrics]
    xgb_f1s  = [m['xgb_metrics']['macro_f1'] for m in all_eval_metrics]
    mlp_accs = [m['mlp_metrics']['accuracy'] for m in all_eval_metrics]
    mlp_f1s  = [m['mlp_metrics']['macro_f1'] for m in all_eval_metrics]

    moe_no_bt_accs = [m['no_backtracking']['accuracy'] for m in all_eval_metrics]
    moe_no_bt_f1s  = [m['no_backtracking']['macro_f1'] for m in all_eval_metrics]
    moe_no_bt_rets = [m['no_backtracking']['cum_return'] for m in all_eval_metrics]
    moe_no_bt_sharpes = [m['no_backtracking']['sharpe'] for m in all_eval_metrics]
    moe_no_bt_mdds = [m['no_backtracking']['mdd'] for m in all_eval_metrics]

    moe_bt_accs = [m['with_backtracking']['accuracy'] for m in all_eval_metrics]
    moe_bt_f1s  = [m['with_backtracking']['macro_f1'] for m in all_eval_metrics]
    moe_bt_rets = [m['with_backtracking']['cum_return'] for m in all_eval_metrics]
    moe_bt_sharpes = [m['with_backtracking']['sharpe'] for m in all_eval_metrics]
    moe_bt_mdds    = [m['with_backtracking']['mdd'] for m in all_eval_metrics]
    moe_bt_wrs     = [m['with_backtracking']['win_rate'] for m in all_eval_metrics]
    moe_bt_das     = [m['with_backtracking']['decision_acc'] for m in all_eval_metrics]

    bh_returns = [m['buy_and_hold_return'] for m in all_eval_metrics]

    xgb_returns = [m['xgb_trading']['cum_return'] for m in all_eval_metrics]
    xgb_sharpes = [m['xgb_trading']['sharpe'] for m in all_eval_metrics]
    xgb_mdds    = [m['xgb_trading']['mdd'] for m in all_eval_metrics]
    xgb_trades  = [m['xgb_trading']['n_trades'] for m in all_eval_metrics]

    mlp_returns = [m['mlp_trading']['cum_return'] for m in all_eval_metrics]
    mlp_sharpes = [m['mlp_trading']['sharpe'] for m in all_eval_metrics]
    mlp_mdds    = [m['mlp_trading']['mdd'] for m in all_eval_metrics]
    mlp_trades  = [m['mlp_trading']['n_trades'] for m in all_eval_metrics]

    n_trades_no = [m['n_trades_no_bt'] for m in all_eval_metrics]
    n_trades_bt = [m['n_trades_bt'] for m in all_eval_metrics]

    print("\n  --- CLASSIFICATION METRICS (mean ± std) ---\n")
    print(f"  {'Model':<30} {'Accuracy':<20} {'Macro-F1':<20}")
    print(f"  {'-'*70}")
    print(f"  {'LR (Expert 1)':<30} {fmt(lr_accs):<20} {fmt(lr_f1s):<20}")
    print(f"  {'XGBoost (Expert 2)':<30} {fmt(xgb_accs):<20} {fmt(xgb_f1s):<20}")
    print(f"  {'MLP (Expert 3)':<30} {fmt(mlp_accs):<20} {fmt(mlp_f1s):<20}")
    print(f"  {'MoE (no BT)':<30} {fmt(moe_no_bt_accs):<20} {fmt(moe_no_bt_f1s):<20}")
    print(f"  {'MoE (with BT)':<30} {fmt(moe_bt_accs):<20} {fmt(moe_bt_f1s):<20}")

    print("\n  --- TRADING METRICS (mean ± std) ---\n")
    print(f"  {'Strategy':<22} {'Return':<20} {'Sharpe':<20} {'MaxDD':<20} {'Trades':<15}")
    print(f"  {'-'*97}")
    print(f"  {'XGB alone':<22} {fmt(xgb_returns):<20} {fmt(xgb_sharpes):<20} {fmt(xgb_mdds):<20} {fmt_int(xgb_trades):<15}")
    print(f"  {'MLP alone':<22} {fmt(mlp_returns):<20} {fmt(mlp_sharpes):<20} {fmt(mlp_mdds):<20} {fmt_int(mlp_trades):<15}")
    print(f"  {'MoE (no BT)':<22} {fmt(moe_no_bt_rets):<20} {fmt(moe_no_bt_sharpes):<20} {fmt(moe_no_bt_mdds):<20} {fmt_int(n_trades_no):<15}")
    print(f"  {'MoE (with BT)':<22} {fmt(moe_bt_rets):<20} {fmt(moe_bt_sharpes):<20} {fmt(moe_bt_mdds):<20} {fmt_int(n_trades_bt):<15}")
    print(f"  {'Buy & Hold':<22} {fmt(bh_returns):<20}")

    print(f"\n  MoE+BT Win Rate:        {fmt(moe_bt_wrs)}")
    print(f"  MoE+BT Decision Acc:    {fmt(moe_bt_das)}")

    print(f"\n  Total time: {total_time:.1f}s")
    print("=" * 70)
    print("  Experiment complete. Results saved to:", RESULTS_DIR)
    print("=" * 70)


if __name__ == '__main__':
    main()
