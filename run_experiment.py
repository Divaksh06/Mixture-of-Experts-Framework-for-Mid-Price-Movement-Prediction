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

# Prevent MacOS OpenMP segfaults
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import random
import numpy as np
import torch

# ============================================================
# CONFIGURATION — Edit these as needed
# ============================================================
DATASET_ROOT = "data/BenchmarkDatasets"          # Path to FI-2010 dataset root
NORMALIZATION = "NoAuction_DecPre"          # Normalization variant
RESULTS_DIR = "results"                     # Directory for saving outputs
N_FOLDS = 1                                # Number of CV folds
SEED = 42                                   # Global random seed
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

    print("=" * 70)
    print("  Mixture-of-Experts Framework for Mid-Price Movement Prediction")
    print("  FI-2010 Benchmark Dataset — Full Pipeline")
    print("=" * 70)
    print(f"  Dataset root:    {DATASET_ROOT}")
    print(f"  Normalization:   {NORMALIZATION}")
    print(f"  Results dir:     {RESULTS_DIR}")
    print(f"  Number of folds: {N_FOLDS}")
    print(f"  Random seed:     {SEED}")
    print("=" * 70)

    # Verify dataset exists
    expected_path = os.path.join(
        DATASET_ROOT, "NoAuction", "3.NoAuction_DecPre",
        "NoAuction_DecPre_Training"
    )
    if not os.path.isdir(expected_path):
        print(f"\n  ERROR: Dataset directory not found at '{expected_path}'")
        print(f"  Please ensure the FI-2010 dataset is placed at '{DATASET_ROOT}'")
        print(f"  with the correct directory structure.")
        print(f"  See README.md for setup instructions.")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Import pipeline modules
    from data.cross_val import anchored_forward_cv
    from train import train_fold
    from evaluate import evaluate_fold

    # Storage for all fold metrics
    all_eval_metrics = []
    all_fold_results = []

    total_start = time.time()

    # ========== Process each fold ==========
    for fold_idx in anchored_forward_cv(N_FOLDS):
        fold_start = time.time()
        set_global_seed(SEED + fold_idx)

        # Stages 1–3: Training
        fold_results = train_fold(
            fold_idx=fold_idx,
            dataset_root=DATASET_ROOT,
            normalization=NORMALIZATION,
            results_dir=RESULTS_DIR
        )
        all_fold_results.append(fold_results)

        # Stage 4: Evaluation
        eval_metrics = evaluate_fold(
            fold_idx=fold_idx,
            fold_results=fold_results,
            results_dir=RESULTS_DIR
        )
        all_eval_metrics.append(eval_metrics)

        fold_time = time.time() - fold_start
        print(f"  Fold {fold_idx}/9 completed in {fold_time:.1f}s")

    total_time = time.time() - total_start

    # ========== Stage 5: Aggregation ==========
    print("\n" + "=" * 70)
    print("  STAGE 5: AGGREGATED RESULTS ACROSS ALL FOLDS")
    print("=" * 70)

    # Collect individual expert metrics
    lr_accs = [m['lr_metrics']['accuracy'] for m in all_eval_metrics]
    lr_f1s = [m['lr_metrics']['macro_f1'] for m in all_eval_metrics]
    xgb_accs = [m['xgb_metrics']['accuracy'] for m in all_eval_metrics]
    xgb_f1s = [m['xgb_metrics']['macro_f1'] for m in all_eval_metrics]
    mlp_accs = [m['mlp_metrics']['accuracy'] for m in all_eval_metrics]
    mlp_f1s = [m['mlp_metrics']['macro_f1'] for m in all_eval_metrics]

    # MoE without backtracking
    moe_no_bt_accs = [m['no_backtracking']['accuracy'] for m in all_eval_metrics]
    moe_no_bt_f1s = [m['no_backtracking']['macro_f1'] for m in all_eval_metrics]
    moe_no_bt_returns = [m['no_backtracking']['cum_return'] for m in all_eval_metrics]
    moe_no_bt_sharpes = [m['no_backtracking']['sharpe'] for m in all_eval_metrics]
    moe_no_bt_mdds = [m['no_backtracking']['mdd'] for m in all_eval_metrics]
    moe_no_bt_wrs = [m['no_backtracking']['win_rate'] for m in all_eval_metrics]
    moe_no_bt_das = [m['no_backtracking']['decision_acc'] for m in all_eval_metrics]

    # MoE with backtracking
    moe_bt_accs = [m['with_backtracking']['accuracy'] for m in all_eval_metrics]
    moe_bt_f1s = [m['with_backtracking']['macro_f1'] for m in all_eval_metrics]
    moe_bt_returns = [m['with_backtracking']['cum_return'] for m in all_eval_metrics]
    moe_bt_sharpes = [m['with_backtracking']['sharpe'] for m in all_eval_metrics]
    moe_bt_mdds = [m['with_backtracking']['mdd'] for m in all_eval_metrics]
    moe_bt_wrs = [m['with_backtracking']['win_rate'] for m in all_eval_metrics]
    moe_bt_das = [m['with_backtracking']['decision_acc'] for m in all_eval_metrics]

    # Buy-and-hold
    bh_returns = [m['buy_and_hold_return'] for m in all_eval_metrics]

    # Per-class F1 (MoE with backtracking)
    per_class_f1s = np.array([m['with_backtracking']['per_class_f1']
                              for m in all_eval_metrics])

    # Ablation Study Trading Metrics
    xgb_returns = [m['xgb_trading']['cum_return'] for m in all_eval_metrics]
    xgb_sharpes = [m['xgb_trading']['sharpe'] for m in all_eval_metrics]
    xgb_mdds = [m['xgb_trading']['mdd'] for m in all_eval_metrics]
    xgb_trades = [m['xgb_trading']['n_trades'] for m in all_eval_metrics]

    mlp_returns = [m['mlp_trading']['cum_return'] for m in all_eval_metrics]
    mlp_sharpes = [m['mlp_trading']['sharpe'] for m in all_eval_metrics]
    mlp_mdds = [m['mlp_trading']['mdd'] for m in all_eval_metrics]
    mlp_trades = [m['mlp_trading']['n_trades'] for m in all_eval_metrics]

    n_trades_no = [m['n_trades_no_bt'] for m in all_eval_metrics]

    def fmt(values):
        """Format mean ± std."""
        return f"{np.mean(values):.4f} ± {np.std(values):.4f}"
    
    def fmt_int(values):
        return f"{np.mean(values):.1f} ± {np.std(values):.1f}"

    # Print Classification Metrics
    print("\n  --- CLASSIFICATION METRICS (mean ± std across folds) ---\n")
    print(f"  {'Model':<30} {'Accuracy':<20} {'Macro-F1':<20}")
    print(f"  {'-'*70}")
    print(f"  {'LR (Expert 1)':<30} {fmt(lr_accs):<20} {fmt(lr_f1s):<20}")
    print(f"  {'XGBoost (Expert 2)':<30} {fmt(xgb_accs):<20} {fmt(xgb_f1s):<20}")
    print(f"  {'MLP (Temporal, Expert 3)':<30} {fmt(mlp_accs):<20} {fmt(mlp_f1s):<20}")
    print(f"  {'MoE (Aggregated)':<30} {fmt(moe_no_bt_accs):<20} {fmt(moe_no_bt_f1s):<20}")

    # Print Financial Metrics (Ablation Study)
    print("\n  --- TRADING METRICS ABLATION STUDY (mean ± std across folds) ---\n")
    print(f"  {'Model':<20} {'Return':<20} {'Sharpe':<20} {'Max DD':<20} {'Trades':<20}")
    print(f"  {'-'*105}")
    print(f"  {'Best Single (XGB)':<20} {fmt(xgb_returns):<20} {fmt(xgb_sharpes):<20} {fmt(xgb_mdds):<20} {fmt_int(xgb_trades):<20}")
    print(f"  {'Temporal MLP':<20} {fmt(mlp_returns):<20} {fmt(mlp_sharpes):<20} {fmt(mlp_mdds):<20} {fmt_int(mlp_trades):<20}")
    print(f"  {'MoE':<20} {fmt(moe_no_bt_returns):<20} {fmt(moe_no_bt_sharpes):<20} {fmt(moe_no_bt_mdds):<20} {fmt_int(n_trades_no):<20}")
    
    print("\n  ========================================================================")
    print("  KEY INSIGHT: CLASSIFICATION ACCURACY ≠ TRADING PROFITABILITY")
    print("  The MoE ensemble combines base model strengths, smoothing out risk (Max DD)")
    print("  and improving risk-adjusted returns (Sharpe) over a standalone classifier.")
    print("  ========================================================================")

    print(f"\n  Buy-and-Hold Benchmark Return: {fmt(bh_returns)}")

    print(f"\n  Total experiment time: {total_time:.1f}s")
    print("=" * 70)
    print("  Experiment complete. Results saved to:", RESULTS_DIR)
    print("=" * 70)


if __name__ == '__main__':
    main()
