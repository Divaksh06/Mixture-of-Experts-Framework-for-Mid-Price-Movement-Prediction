"""
Evaluation Pipeline (Stage 4) for the MoE framework.

For each fold: runs inference on the test set, applies the strategy layer
with backtracking, runs the backtesting engine, and collects all
classification and financial metrics.
"""

import os
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from moe.mixture import compute_pfinal
from strategy.strategy_layer import StrategyLayer
from strategy.backtracking import BacktrackingModule
from backtest.engine import BacktestEngine
from backtest.metrics import (
    cumulative_return, sharpe_ratio, max_drawdown,
    win_rate, decision_accuracy, buy_and_hold_return
)


def evaluate_fold(fold_idx, fold_results, results_dir='results'):
    """
    Execute Stage 4 for a single fold: inference + strategy + backtracking + backtesting.

    Parameters
    ----------
    fold_idx : int
        Fold index (1 through 9).
    fold_results : dict
        Output from train_fold() containing models, test data, and probabilities.
    results_dir : str
        Directory for saving results.

    Returns
    -------
    eval_metrics : dict
        Dictionary with all classification and financial metrics,
        including with and without backtracking.
    """
    print(f"\n  Fold {fold_idx}/9: Stage 4 — Strategy + Backtest...")

    X_test = fold_results['X_test']
    y_test = fold_results['y_test']
    test_probs_lr = fold_results['test_probs_lr']
    test_probs_xgb = fold_results['test_probs_xgb']
    test_probs_mlp = fold_results['test_probs_mlp']
    pfinal_test = fold_results['pfinal_test']

    lr_expert = fold_results['lr_expert']
    xgb_expert = fold_results['xgb_expert']
    mlp_expert = fold_results['mlp_expert']

    n_test = len(y_test)

    # Generate per-sample mid-price returns (deterministic from label direction)
    # Since we don't have actual prices, we simulate returns based on labels:
    # Up (0) -> fixed positive return, Stationary (1) -> 0, Down (2) -> fixed negative
    BASE_RETURN = 0.0002
    mid_price_returns = np.zeros(n_test)
    for i in range(n_test):
        if y_test[i] == 0:    # Up
            mid_price_returns[i] = BASE_RETURN
        elif y_test[i] == 2:  # Down
            mid_price_returns[i] = -BASE_RETURN
        else:                 # Stationary
            mid_price_returns[i] = 0.0

    # Get individual expert predictions for backtracking
    lr_preds = np.argmax(test_probs_lr, axis=1)
    xgb_preds = np.argmax(test_probs_xgb, axis=1)
    mlp_preds = np.argmax(test_probs_mlp, axis=1)

    # ===== Run WITHOUT backtracking =====
    strategy_no_bt = StrategyLayer(theta_buy=0.55, theta_sell=0.55, delta=0.10)
    engine_no_bt = BacktestEngine(initial_capital=10000.0, transaction_cost=0.0001)

    actions_no_bt = []
    for i in range(n_test):
        action = strategy_no_bt.decide(pfinal_test[i])
        actions_no_bt.append(action)
        ret = mid_price_returns[i]
        engine_no_bt.step(action, ret)
    engine_no_bt.close_position(mid_price_returns[-1] if len(mid_price_returns) > 0 else 0.0)

    # Metrics without backtracking
    moe_preds = np.argmax(pfinal_test, axis=1)
    no_bt_metrics = {
        'accuracy': accuracy_score(y_test, moe_preds),
        'macro_f1': f1_score(y_test, moe_preds, average='macro', zero_division=0),
        'per_class_f1': f1_score(y_test, moe_preds, average=None, labels=[0, 1, 2], zero_division=0),
        'cum_return': cumulative_return(engine_no_bt.get_values_history()),
        'sharpe': sharpe_ratio(engine_no_bt.get_step_returns()),
        'mdd': max_drawdown(engine_no_bt.get_values_history()),
        'win_rate': win_rate(engine_no_bt.trade_returns),
        'decision_acc': decision_accuracy(actions_no_bt, y_test),
    }

    # ===== Run WITH backtracking =====
    bt_module = BacktrackingModule(
        initial_weights=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
        theta_buy=0.55, theta_sell=0.55, delta=0.10,
        N_buf=100, N_upd=50
    )
    strategy_bt = StrategyLayer(theta_buy=0.55, theta_sell=0.55, delta=0.10)
    engine_bt = BacktestEngine(initial_capital=10000.0, transaction_cost=0.0001)

    actions_bt = []
    pfinal_bt_all = np.zeros_like(pfinal_test)

    for i in range(n_test):
        # Get current backtracking weights
        bt_weights = bt_module.get_weights()

        # Compute weighted P_final with backtracking-adjusted weights
        pfinal_i = (
            bt_weights[0] * test_probs_lr[i] +
            bt_weights[1] * test_probs_xgb[i] +
            bt_weights[2] * test_probs_mlp[i]
        )
        pfinal_bt_all[i] = pfinal_i

        y_pred = np.argmax(pfinal_i)

        # Update strategy thresholds from backtracking
        thresholds = bt_module.get_thresholds()
        strategy_bt.update_thresholds(
            theta_buy=thresholds['theta_buy'],
            theta_sell=thresholds['theta_sell'],
            delta=thresholds['delta']
        )

        # Decide action
        action = strategy_bt.decide(pfinal_i)

        # Check for action correction (reversal override)
        action = bt_module.check_action_correction(pfinal_i, action)

        actions_bt.append(action)

        # Set previous action BEFORE adding observation (BUG 10 fix)
        bt_module.set_prev_action(action, y_pred)

        # Simulate portfolio step
        ret = mid_price_returns[i]
        engine_bt.step(action, ret)

        # Add observation to backtracking buffer
        expert_preds_i = np.array([lr_preds[i], xgb_preds[i], mlp_preds[i]])
        bt_module.add_observation(y_pred, pfinal_i, action, y_test[i], expert_preds_i)

        # Update backtracking module if needed
        if bt_module.should_update():
            bt_module.update()

    engine_bt.close_position(mid_price_returns[-1] if len(mid_price_returns) > 0 else 0.0)
    engine_bt.close_short(mid_price_returns[-1] if len(mid_price_returns) > 0 else 0.0)

    # Metrics with backtracking
    moe_preds_bt = np.argmax(pfinal_bt_all, axis=1)
    bt_metrics = {
        'accuracy': accuracy_score(y_test, moe_preds_bt),
        'macro_f1': f1_score(y_test, moe_preds_bt, average='macro', zero_division=0),
        'per_class_f1': f1_score(y_test, moe_preds_bt, average=None, labels=[0, 1, 2], zero_division=0),
        'cum_return': cumulative_return(engine_bt.get_values_history()),
        'sharpe': sharpe_ratio(engine_bt.get_step_returns()),
        'mdd': max_drawdown(engine_bt.get_values_history()),
        'win_rate': win_rate(engine_bt.trade_returns),
        'decision_acc': decision_accuracy(actions_bt, y_test),
    }

    # Buy-and-hold benchmark
    bh_return = buy_and_hold_return(mid_price_returns)

    # Count action stability
    action_changes_no_bt = sum(
        1 for j in range(1, len(actions_no_bt))
        if actions_no_bt[j] != actions_no_bt[j-1]
    )
    action_changes_bt = sum(
        1 for j in range(1, len(actions_bt))
        if actions_bt[j] != actions_bt[j-1]
    )

    eval_metrics = {
        'no_backtracking': no_bt_metrics,
        'with_backtracking': bt_metrics,
        'buy_and_hold_return': bh_return,
        'action_stability_no_bt': action_changes_no_bt / max(n_test - 1, 1),
        'action_stability_bt': action_changes_bt / max(n_test - 1, 1),
        'lr_metrics': fold_results['lr_metrics'],
        'xgb_metrics': fold_results['xgb_metrics'],
        'mlp_metrics': fold_results['mlp_metrics'],
        'n_trades_no_bt': len(engine_no_bt.trade_returns),
        'n_trades_bt': len(engine_bt.trade_returns),
    }

    print(f"    Without Backtracking — Acc: {no_bt_metrics['accuracy']:.4f}, "
          f"F1: {no_bt_metrics['macro_f1']:.4f}, "
          f"Return: {no_bt_metrics['cum_return']:.6f}, "
          f"Sharpe: {no_bt_metrics['sharpe']:.4f}")
    print(f"    With Backtracking    — Acc: {bt_metrics['accuracy']:.4f}, "
          f"F1: {bt_metrics['macro_f1']:.4f}, "
          f"Return: {bt_metrics['cum_return']:.6f}, "
          f"Sharpe: {bt_metrics['sharpe']:.4f}")
    print(f"    Buy-and-Hold Return:   {bh_return:.6f}")

    return eval_metrics
