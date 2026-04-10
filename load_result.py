"""
Load and display results from all 9 folds of the MoE experiment.

Usage (from the project root):
    python load_result.py

This script loads the saved fold_results.pkl files, re-runs the
evaluation pipeline (Stage 4) for each fold, and prints the same
aggregated metrics that run_experiment.py produces.
"""

import os
import sys
import pickle
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that unpickling the expert
# objects (which reference experts.*, moe.*, etc.) works correctly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
N_FOLDS = 9


def fmt(values):
    """Format mean ± std."""
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"


def main():
    # ------------------------------------------------------------------
    # 1. Load all fold_results.pkl files
    # ------------------------------------------------------------------
    all_fold_results = {}
    for fold_idx in range(1, N_FOLDS + 1):
        pkl_path = os.path.join(RESULTS_DIR, f"fold_{fold_idx}", "fold_results.pkl")
        if not os.path.isfile(pkl_path):
            print(f"  [SKIP] Fold {fold_idx}: {pkl_path} not found.")
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        all_fold_results[fold_idx] = data
        print(f"  [OK]   Fold {fold_idx} loaded  —  "
              f"MoE Acc: {data['moe_accuracy']:.4f}, "
              f"MoE F1: {data['moe_macro_f1']:.4f}")

    if not all_fold_results:
        print("\nNo fold results found. Run the experiment first.")
        sys.exit(1)

    loaded_folds = sorted(all_fold_results.keys())
    print(f"\nLoaded {len(loaded_folds)} fold(s): {loaded_folds}")

    # ------------------------------------------------------------------
    # 2. Re-run Stage 4 (evaluation) for each loaded fold
    # ------------------------------------------------------------------
    from evaluate import evaluate_fold          # imported here so sklearn
                                                # is only needed at this point

    all_eval_metrics = []
    for fold_idx in loaded_folds:
        eval_metrics = evaluate_fold(
            fold_idx=fold_idx,
            fold_results=all_fold_results[fold_idx],
            results_dir=RESULTS_DIR,
        )
        all_eval_metrics.append(eval_metrics)

    # ------------------------------------------------------------------
    # 3. Aggregate and print results (mirrors run_experiment.py Stage 5)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  AGGREGATED RESULTS ACROSS ALL LOADED FOLDS")
    print("=" * 70)

    # Individual expert metrics
    lr_accs  = [m['lr_metrics']['accuracy']  for m in all_eval_metrics]
    lr_f1s   = [m['lr_metrics']['macro_f1']  for m in all_eval_metrics]
    xgb_accs = [m['xgb_metrics']['accuracy'] for m in all_eval_metrics]
    xgb_f1s  = [m['xgb_metrics']['macro_f1'] for m in all_eval_metrics]
    mlp_accs = [m['mlp_metrics']['accuracy'] for m in all_eval_metrics]
    mlp_f1s  = [m['mlp_metrics']['macro_f1'] for m in all_eval_metrics]

    # MoE without backtracking
    moe_no_bt_accs    = [m['no_backtracking']['accuracy']     for m in all_eval_metrics]
    moe_no_bt_f1s     = [m['no_backtracking']['macro_f1']     for m in all_eval_metrics]
    moe_no_bt_returns = [m['no_backtracking']['cum_return']    for m in all_eval_metrics]
    moe_no_bt_sharpes = [m['no_backtracking']['sharpe']        for m in all_eval_metrics]
    moe_no_bt_mdds    = [m['no_backtracking']['mdd']           for m in all_eval_metrics]
    moe_no_bt_wrs     = [m['no_backtracking']['win_rate']      for m in all_eval_metrics]
    moe_no_bt_das     = [m['no_backtracking']['decision_acc']  for m in all_eval_metrics]

    # MoE with backtracking
    moe_bt_accs    = [m['with_backtracking']['accuracy']     for m in all_eval_metrics]
    moe_bt_f1s     = [m['with_backtracking']['macro_f1']     for m in all_eval_metrics]
    moe_bt_returns = [m['with_backtracking']['cum_return']    for m in all_eval_metrics]
    moe_bt_sharpes = [m['with_backtracking']['sharpe']        for m in all_eval_metrics]
    moe_bt_mdds    = [m['with_backtracking']['mdd']           for m in all_eval_metrics]
    moe_bt_wrs     = [m['with_backtracking']['win_rate']      for m in all_eval_metrics]
    moe_bt_das     = [m['with_backtracking']['decision_acc']  for m in all_eval_metrics]

    # Buy-and-hold
    bh_returns = [m['buy_and_hold_return'] for m in all_eval_metrics]

    # Per-class F1 (MoE with backtracking)
    per_class_f1s = np.array([m['with_backtracking']['per_class_f1']
                              for m in all_eval_metrics])

    # ---- Classification Metrics ----
    print("\n  --- CLASSIFICATION METRICS (mean ± std across folds) ---\n")
    print(f"  {'Model':<25} {'Accuracy':<20} {'Macro-F1':<20}")
    print(f"  {'-'*65}")
    print(f"  {'LR (Expert 1)':<25} {fmt(lr_accs):<20} {fmt(lr_f1s):<20}")
    print(f"  {'XGBoost (Expert 2)':<25} {fmt(xgb_accs):<20} {fmt(xgb_f1s):<20}")
    print(f"  {'MLP (Expert 3)':<25} {fmt(mlp_accs):<20} {fmt(mlp_f1s):<20}")
    print(f"  {'MoE (no backtracking)':<25} {fmt(moe_no_bt_accs):<20} "
          f"{fmt(moe_no_bt_f1s):<20}")
    print(f"  {'MoE (with backtracking)':<25} {fmt(moe_bt_accs):<20} "
          f"{fmt(moe_bt_f1s):<20}")

    print(f"\n  Per-class F1 (MoE with backtracking):")
    print(f"    Up (class 0):         {np.mean(per_class_f1s[:, 0]):.4f} "
          f"± {np.std(per_class_f1s[:, 0]):.4f}")
    print(f"    Stationary (class 1): {np.mean(per_class_f1s[:, 1]):.4f} "
          f"± {np.std(per_class_f1s[:, 1]):.4f}")
    print(f"    Down (class 2):       {np.mean(per_class_f1s[:, 2]):.4f} "
          f"± {np.std(per_class_f1s[:, 2]):.4f}")

    # ---- Financial Metrics ----
    print("\n  --- FINANCIAL / STRATEGY METRICS (mean ± std across folds) ---\n")
    print(f"  {'Metric':<25} {'No Backtracking':<25} {'With Backtracking':<25}")
    print(f"  {'-'*75}")
    print(f"  {'Cumulative Return':<25} {fmt(moe_no_bt_returns):<25} "
          f"{fmt(moe_bt_returns):<25}")
    print(f"  {'Sharpe Ratio':<25} {fmt(moe_no_bt_sharpes):<25} "
          f"{fmt(moe_bt_sharpes):<25}")
    print(f"  {'Max Drawdown':<25} {fmt(moe_no_bt_mdds):<25} "
          f"{fmt(moe_bt_mdds):<25}")
    print(f"  {'Win Rate':<25} {fmt(moe_no_bt_wrs):<25} "
          f"{fmt(moe_bt_wrs):<25}")
    print(f"  {'Decision Accuracy':<25} {fmt(moe_no_bt_das):<25} "
          f"{fmt(moe_bt_das):<25}")

    print(f"\n  Buy-and-Hold Return:    {fmt(bh_returns)}")

    # Action stability
    stab_no_bt = [m['action_stability_no_bt'] for m in all_eval_metrics]
    stab_bt    = [m['action_stability_bt']    for m in all_eval_metrics]
    print(f"\n  Action Stability (lower = more stable):")
    print(f"    No backtracking:      {fmt(stab_no_bt)}")
    print(f"    With backtracking:    {fmt(stab_bt)}")

    # Trade counts
    n_trades_no = [m['n_trades_no_bt'] for m in all_eval_metrics]
    n_trades_bt = [m['n_trades_bt']    for m in all_eval_metrics]
    print(f"\n  Average trades per fold:")
    print(f"    No backtracking:      {np.mean(n_trades_no):.1f}")
    print(f"    With backtracking:    {np.mean(n_trades_bt):.1f}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
