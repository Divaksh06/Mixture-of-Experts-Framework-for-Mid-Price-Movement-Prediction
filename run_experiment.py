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
DATASET_ROOT = "data/BenchmarkDatasets"     # Path to FI-2010 dataset root
NORMALIZATION = "NoAuction_DecPre"          # Normalization variant
RESULTS_DIR = "results"                     # Directory for saving outputs
N_FOLDS = 9                                 # Number of CV folds
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

    # ========== Stage 5: Aggregation with Statistical Rigor ==========
    from scipy.stats import ttest_rel, spearmanr
    import sys

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open(os.path.join(RESULTS_DIR, "aggregated_metrics.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)

    print("\n" + "=" * 70)
    print("  STAGE 5: AGGREGATED RESULTS ACROSS ALL FOLDS")
    print("  We report results across 9 chronological folds.")
    print("=" * 70)

    def fmt(v):
        return f"{np.mean(v):.4f} ± {np.std(v):.4f}"

    def fmt_trade(v):
        return f"{np.mean(v):.1f} ± {np.std(v):.1f}"

    def fmt_ret(v):
        return f"{np.mean(v):.3f}±{np.std(v):.3f} (md {np.median(v):.3f})"

    def fmt_pct(v):
        return f"{np.mean(v)*100:.1f}%"

    # --- Classification Metrics ---
    lr_accs = [m['lr_metrics']['accuracy'] for m in all_eval_metrics]
    lr_f1s = [m['lr_metrics']['macro_f1'] for m in all_eval_metrics]
    xgb_accs = [m['xgb_metrics']['accuracy'] for m in all_eval_metrics]
    xgb_f1s = [m['xgb_metrics']['macro_f1'] for m in all_eval_metrics]
    mlp_accs = [m['mlp_metrics']['accuracy'] for m in all_eval_metrics]
    mlp_f1s = [m['mlp_metrics']['macro_f1'] for m in all_eval_metrics]
    moe_accs = [m['moe_1bp']['accuracy'] for m in all_eval_metrics]
    moe_f1s = [m['moe_1bp']['macro_f1'] for m in all_eval_metrics]

    print("\n  --- CLASSIFICATION METRICS (mean ± std) ---\n")
    print(f"  {'Model':<30} {'Accuracy':<20} {'Macro-F1':<20}")
    print(f"  {'-'*70}")
    print(f"  {'LR (Expert 1)':<30} {fmt(lr_accs):<20} {fmt(lr_f1s):<20}")
    print(f"  {'XGBoost (Expert 2)':<30} {fmt(xgb_accs):<20} {fmt(xgb_f1s):<20}")
    print(f"  {'MLP (Expert 3)':<30} {fmt(mlp_accs):<20} {fmt(mlp_f1s):<20}")
    print(f"  {'Gated MoE':<30} {fmt(moe_accs):<20} {fmt(moe_f1s):<20}")

    # --- Gate Weight Diagnostics (aggregated across folds) ---
    all_gate_weights = np.array([m['gate_mean_weights'] for m in all_eval_metrics])
    all_route_pcts = np.array([m['gate_route_pcts'] for m in all_eval_metrics])
    print("\n  --- GATING NETWORK DIAGNOSTICS ---\n")
    print(f"  Mean soft weights:   LR: {all_gate_weights[:,0].mean():.4f} ± {all_gate_weights[:,0].std():.4f}"
          f"  |  XGB: {all_gate_weights[:,1].mean():.4f} ± {all_gate_weights[:,1].std():.4f}"
          f"  |  MLP: {all_gate_weights[:,2].mean():.4f} ± {all_gate_weights[:,2].std():.4f}")
    print(f"  Argmax routing:      LR: {all_route_pcts[:,0].mean():.1%}"
          f"  |  XGB: {all_route_pcts[:,1].mean():.1%}"
          f"  |  MLP: {all_route_pcts[:,2].mean():.1%}")
    favoured_idx = np.argmax(all_gate_weights.mean(axis=0))
    print(f"  Dominant expert:     {['LR', 'XGB', 'MLP'][favoured_idx]}")

    # --- Trading Metrics ---
    xgb_ret = [m['xgb_1bp']['cum_return'] for m in all_eval_metrics]
    mlp_ret = [m['mlp_1bp']['cum_return'] for m in all_eval_metrics]
    moe_ret = [m['moe_1bp']['cum_return'] for m in all_eval_metrics]
    moe_bt_ret = [m['moe_bt_1bp']['cum_return'] for m in all_eval_metrics]
    moe_cr_ret = [m['moe_cr_1bp']['cum_return'] for m in all_eval_metrics]
    bh_ret = [m['bh_return'] for m in all_eval_metrics]

    xgb_s = [m['xgb_1bp']['sortino'] for m in all_eval_metrics]
    mlp_s = [m['mlp_1bp']['sortino'] for m in all_eval_metrics]
    moe_s = [m['moe_1bp']['sortino'] for m in all_eval_metrics]
    moe_bt_s = [m['moe_bt_1bp']['sortino'] for m in all_eval_metrics]
    moe_cr_s = [m['moe_cr_1bp']['sortino'] for m in all_eval_metrics]

    xgb_dd = [m['xgb_1bp']['mdd'] for m in all_eval_metrics]
    mlp_dd = [m['mlp_1bp']['mdd'] for m in all_eval_metrics]
    moe_dd = [m['moe_1bp']['mdd'] for m in all_eval_metrics]
    moe_bt_dd = [m['moe_bt_1bp']['mdd'] for m in all_eval_metrics]
    moe_cr_dd = [m['moe_cr_1bp']['mdd'] for m in all_eval_metrics]

    xgb_t = [m['xgb_1bp']['n_trades'] for m in all_eval_metrics]
    mlp_t = [m['mlp_1bp']['n_trades'] for m in all_eval_metrics]
    moe_t = [m['moe_1bp']['n_trades'] for m in all_eval_metrics]
    moe_bt_t = [m['moe_bt_1bp']['n_trades'] for m in all_eval_metrics]
    moe_cr_t = [m['moe_cr_1bp']['n_trades'] for m in all_eval_metrics]
    
    xgb_wr = [m['xgb_1bp']['win_rate'] for m in all_eval_metrics]
    moe_wr = [m['moe_1bp']['win_rate'] for m in all_eval_metrics]
    moe_bt_wr = [m['moe_bt_1bp']['win_rate'] for m in all_eval_metrics]
    moe_cr_wr = [m['moe_cr_1bp']['win_rate'] for m in all_eval_metrics]
    xgb_da = [m['xgb_1bp']['decision_accuracy'] for m in all_eval_metrics]
    moe_da = [m['moe_1bp']['decision_accuracy'] for m in all_eval_metrics]
    moe_bt_da = [m['moe_bt_1bp']['decision_accuracy'] for m in all_eval_metrics]
    moe_cr_da = [m['moe_cr_1bp']['decision_accuracy'] for m in all_eval_metrics]

    print("\n  --- TRADING METRICS (mean ± std) ---\n")
    print(f"  {'Strategy':<18} {'Return (std/med)':<28} {'Sortino':<16} {'MaxDD':<16} {'Trades':<16} {'Win Rate':<12} {'Dec Acc':<12}")
    print(f"  {'-'*125}")
    print(f"  {'XGB alone':<18} {fmt_ret(xgb_ret):<28} {fmt(xgb_s):<16} {fmt(xgb_dd):<16} {fmt_trade(xgb_t):<16} {fmt_pct(xgb_wr):<12} {fmt_pct(xgb_da):<12}")
    print(f"  {'MLP alone':<18} {fmt_ret(mlp_ret):<28} {fmt(mlp_s):<16} {fmt(mlp_dd):<16} {fmt_trade(mlp_t):<16} {'-':<12} {'-':<12}")
    print(f"  {'Gated MoE':<18} {fmt_ret(moe_ret):<28} {fmt(moe_s):<16} {fmt(moe_dd):<16} {fmt_trade(moe_t):<16} {fmt_pct(moe_wr):<12} {fmt_pct(moe_da):<12}")
    print(f"  {'MoE + BT':<18} {fmt_ret(moe_bt_ret):<28} {fmt(moe_bt_s):<16} {fmt(moe_bt_dd):<16} {fmt_trade(moe_bt_t):<16} {fmt_pct(moe_bt_wr):<12} {fmt_pct(moe_bt_da):<12}")
    print(f"  {'MoE + CR':<18} {fmt_ret(moe_cr_ret):<28} {fmt(moe_cr_s):<16} {fmt(moe_cr_dd):<16} {fmt_trade(moe_cr_t):<16} {fmt_pct(moe_cr_wr):<12} {fmt_pct(moe_cr_da):<12}")
    print(f"  {'Buy & Hold':<18} {fmt_ret(bh_ret):<28} {'-':<16} {'-':<16} {'-':<16} {'-':<12} {'-':<12}")
    print("=" * 70)
    print(f"  Experiment complete. Results saved to {os.path.join(RESULTS_DIR, 'aggregated_metrics.txt')}")
    print("=" * 70)

    log_file.close()

if __name__ == '__main__':
    main()
