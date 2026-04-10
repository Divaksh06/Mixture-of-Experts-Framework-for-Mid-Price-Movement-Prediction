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
N_FOLDS = 9                                # Number of CV folds
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
        """Format mean ± std."""
        return f"{np.mean(v):.4f} ± {np.std(v):.4f}"

    def fmt_int(v):
        return f"{np.mean(v):.1f} ± {np.std(v):.1f}"

    # --- Classification Metrics ---
    lr_accs = [m['lr_metrics']['accuracy'] for m in all_eval_metrics]
    lr_f1s = [m['lr_metrics']['macro_f1'] for m in all_eval_metrics]
    xgb_accs = [m['xgb_metrics']['accuracy'] for m in all_eval_metrics]
    xgb_f1s = [m['xgb_metrics']['macro_f1'] for m in all_eval_metrics]
    mlp_accs = [m['mlp_metrics']['accuracy'] for m in all_eval_metrics]
    mlp_f1s = [m['mlp_metrics']['macro_f1'] for m in all_eval_metrics]
    moe_accs = [m['moe_1bp']['accuracy'] for m in all_eval_metrics]
    moe_f1s = [m['moe_1bp']['macro_f1'] for m in all_eval_metrics]

    print("\n  --- CLASSIFICATION METRICS (mean ± std across folds) ---\n")
    print(f"  {'Model':<30} {'Accuracy':<20} {'Macro-F1':<20}")
    print(f"  {'-'*70}")
    print(f"  {'LR (Expert 1)':<30} {fmt(lr_accs):<20} {fmt(lr_f1s):<20}")
    print(f"  {'XGBoost (Expert 2)':<30} {fmt(xgb_accs):<20} {fmt(xgb_f1s):<20}")
    print(f"  {'MLP (Temporal, Expert 3)':<30} {fmt(mlp_accs):<20} {fmt(mlp_f1s):<20}")
    print(f"  {'Gated MoE':<30} {fmt(moe_accs):<20} {fmt(moe_f1s):<20}")

    # --- 1. Benchmarks ---
    bh_returns = [m['bh_return'] for m in all_eval_metrics]
    passive_returns = [m['passive_return'] for m in all_eval_metrics]

    print("\n  --- 1. BENCHMARKS ---\n")
    print(f"  Directional Benchmark (Long-Only, Always Invested, No Costs): {fmt(bh_returns)}")
    print(f"  Passive Signal Benchmark (Perfect Direction, No Timing):      {fmt(passive_returns)}")
    print("  * Note: Buy-and-Hold is not directly comparable due to continuous")
    print("    exposure and absence of transaction costs.")

    # --- 2. Transaction Cost Sensitivity ---
    moe_1bp_ret = [m['moe_1bp']['cum_return'] for m in all_eval_metrics]
    xgb_1bp_ret = [m['xgb_1bp']['cum_return'] for m in all_eval_metrics]
    moe_3bp_ret = [m['moe_3bp']['cum_return'] for m in all_eval_metrics]
    xgb_3bp_ret = [m['xgb_3bp']['cum_return'] for m in all_eval_metrics]

    print("\n  --- 2. TRANSACTION COST SENSITIVITY ---\n")
    print(f"  {'Cost':<10} {'MoE Return':<22} {'XGB Return':<22}")
    print(f"  {'-'*55}")
    print(f"  {'1 bp':<10} {fmt(moe_1bp_ret):<22} {fmt(xgb_1bp_ret):<22}")
    print(f"  {'3 bp':<10} {fmt(moe_3bp_ret):<22} {fmt(xgb_3bp_ret):<22}")
    print("  * Transaction costs are varied as a stress test rather than an exact")
    print("    simulation of real execution conditions.")

    # --- 3. Gating Ablation ---
    eq_1bp_ret = [m['eq_1bp']['cum_return'] for m in all_eval_metrics]
    wt_1bp_ret = [m['wt_1bp']['cum_return'] for m in all_eval_metrics]

    print("\n  --- 3. GATING ABLATION (CRITICAL) ---\n")
    print(f"  {'Model':<25} {'Return':<22}")
    print(f"  {'-'*48}")
    print(f"  {'XGB Solo':<25} {fmt(xgb_1bp_ret):<22}")
    print(f"  {'Equal Ensemble (1/3)':<25} {fmt(eq_1bp_ret):<22}")
    print(f"  {'Weighted Ensemble (F1)':<25} {fmt(wt_1bp_ret):<22}")
    print(f"  {'Gated MoE (Ours)':<25} {fmt(moe_1bp_ret):<22}")
    gating_val = "shows potential benefit" if np.mean(moe_1bp_ret) > np.mean(wt_1bp_ret) else "does not consistently improve"
    print(f"  * Interpretation: MoE performs comparably to the best single model (XGBoost).")
    print(f"    Gated MoE vs Weighted Ensemble suggests gating {gating_val}.")

    # --- 4. Return-to-Volatility & Sortino ---
    moe_r2v = [m['moe_1bp']['return_to_volatility'] for m in all_eval_metrics]
    xgb_r2v = [m['xgb_1bp']['return_to_volatility'] for m in all_eval_metrics]
    moe_sortino = [m['moe_1bp']['sortino'] for m in all_eval_metrics]
    xgb_sortino = [m['xgb_1bp']['sortino'] for m in all_eval_metrics]

    print("\n  --- 4. RISK-ADJUSTED METRICS ---\n")
    print(f"  {'Model':<20} {'R-to-V Ratio':<22} {'Sortino':<22}")
    print(f"  {'-'*65}")
    print(f"  {'XGB Solo':<20} {fmt(xgb_r2v):<22} {fmt(xgb_sortino):<22}")
    print(f"  {'Gated MoE':<20} {fmt(moe_r2v):<22} {fmt(moe_sortino):<22}")
    print("  * All reported ratios are computed on tick-level returns (non-annualized).")

    # --- 5. Statistical Rigor ---
    wins = sum(1 for m, x in zip(moe_1bp_ret, xgb_1bp_ret) if m > x)

    print("\n  --- 5. STATISTICAL RIGOR ---\n")
    print(f"  Win Count: MoE outperforms XGB in {wins}/{len(moe_1bp_ret)} folds")
    print(f"  Mean Return:   MoE = {np.mean(moe_1bp_ret):.4f} | XGB = {np.mean(xgb_1bp_ret):.4f}")
    print(f"  Median Return: MoE = {np.median(moe_1bp_ret):.4f} | XGB = {np.median(xgb_1bp_ret):.4f}")

    if len(moe_1bp_ret) > 1:
        t_stat, p_val = ttest_rel(moe_1bp_ret, xgb_1bp_ret)
        print(f"  Paired t-test (MoE vs XGB): t = {t_stat:.4f}, p-value = {p_val:.4g}")
        if p_val > 0.05:
            print(f"  → Does not show statistically significant improvement (p > 0.05)")
            print(f"    Performance differences are within variance across folds.")
        else:
            print(f"  → Statistically significant at p < 0.05")
    else:
        print("  Paired t-test: insufficient folds (need >= 2)")
    print("  * Due to temporal dependence between folds, statistical tests are approximate.")

    # --- 6. Predictor Paradox Formalization ---
    all_f1 = xgb_f1s + mlp_f1s + moe_f1s
    all_ret = xgb_1bp_ret + [m['mlp_1bp']['cum_return'] for m in all_eval_metrics] + moe_1bp_ret

    print("\n  --- 6. PREDICTOR PARADOX FORMALIZATION ---\n")
    if len(all_f1) > 2:
        rho, rho_p = spearmanr(all_f1, all_ret)
        print(f"  Spearman correlation (F1 vs Return): rho = {rho:.4f} (p = {rho_p:.4g})")
        if rho_p > 0.05:
            print(f"  → Does not show a strong correlation between classification and")
            print(f"    profitability in this dataset (p > 0.05).")
    else:
        print("  Spearman correlation: insufficient data points")
    print("  * Weak or non-significant correlation is consistent with the hypothesis")
    print("    that classification accuracy alone does not determine profitability.")

    print(f"\n  Total experiment time: {total_time:.1f}s")
    print("=" * 70)
    print(f"  Experiment complete. Results saved to {os.path.join(RESULTS_DIR, 'aggregated_metrics.txt')}")
    print("=" * 70)

    log_file.close()

if __name__ == '__main__':
    main()
