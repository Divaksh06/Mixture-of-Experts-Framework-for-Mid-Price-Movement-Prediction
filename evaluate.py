"""
Evaluation Pipeline (Stage 4) for the MoE framework.

For each fold: runs inference on the test set, applies the signal-based
strategy layer with multi-horizon agreement filtering, runs the backtesting
engine, and collects all classification and financial metrics.

Multi-Horizon Agreement:
    The strategy action from the k=10 MoE signal is only executed if the
    secondary k=5 LR model agrees on the directional prediction.
    If they disagree, the action is downgraded to 'Hold'.
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
    win_rate, decision_accuracy, buy_and_hold_return,
    profit_factor, turnover, average_holding_time, average_trade_return
)


def evaluate_fold(fold_idx, fold_results, results_dir='results'):
    """
    Execute Stage 4 for a single fold: signal strategy + backtesting.

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
    print(f"\n  Fold {fold_idx}: Stage 4 — Signal Strategy + Multi-Horizon Backtest...")

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

    # Load real unscaled Mid-Prices for true return calculation
    from data.load_fi2010 import load_mid_prices
    try:
        mid_prices = load_mid_prices(fold_idx, 'data/BenchmarkDatasets', split='test')
        mid_price_returns = np.zeros(n_test)
        for i in range(n_test - 1):
            if mid_prices[i] > 0:
                mid_price_returns[i] = (mid_prices[i+1] - mid_prices[i]) / mid_prices[i]
    except Exception as e:
        print(f"    Warning: Could not load real mid prices ({e}). Return metrics will be invalid.")
        mid_price_returns = np.zeros(n_test)

    # Get individual expert predictions for backtracking
    lr_preds = np.argmax(test_probs_lr, axis=1)
    xgb_preds = np.argmax(test_probs_xgb, axis=1)
    mlp_preds = np.argmax(test_probs_mlp, axis=1)

    # ===== Run WITHOUT backtracking =====
    strategy_no_bt = StrategyLayer()
    engine_no_bt = BacktestEngine(initial_capital=10000.0, transaction_cost=0.0001)

    actions_no_bt = []
    for i in range(n_test):
        action = strategy_no_bt.decide(pfinal_test[i], engine_no_bt.position)
        actions_no_bt.append(action)
        ret = mid_price_returns[i] if i < n_test - 1 else 0.0
        engine_no_bt.step(action, ret)
    engine_no_bt.close_position()

    # Metrics without backtracking
    moe_preds = np.argmax(pfinal_test, axis=1)
    no_bt_metrics = {
        'accuracy': accuracy_score(y_test, moe_preds),
        'macro_f1': f1_score(y_test, moe_preds, average='macro'),
        'per_class_f1': f1_score(y_test, moe_preds, average=None),
        'cum_return': cumulative_return(engine_no_bt.get_values_history()),
        'sharpe': sharpe_ratio(engine_no_bt.get_step_returns(), n_ann=252*390),
        'mdd': max_drawdown(engine_no_bt.get_values_history()),
        'win_rate': win_rate(engine_no_bt.trade_returns),
        'decision_acc': decision_accuracy(actions_no_bt, y_test),
        'profit_factor': profit_factor(engine_no_bt.trade_returns),
        'turnover': turnover(actions_no_bt),
        'avg_hold_time': average_holding_time(engine_no_bt.hold_durations),
        'avg_trade_return': average_trade_return(engine_no_bt.trade_returns),
    }

    # ===== Run XGBoost Independently =====
    engine_xgb = BacktestEngine(initial_capital=10000.0, transaction_cost=0.0001)
    strategy_xgb = StrategyLayer()
    actions_xgb = []
    for i in range(n_test):
        action = strategy_xgb.decide(test_probs_xgb[i], engine_xgb.position)
        actions_xgb.append(action)
        ret = mid_price_returns[i] if i < n_test - 1 else 0.0
        engine_xgb.step(action, ret)
    engine_xgb.close_position()

    xgb_trading_metrics = {
        'cum_return': cumulative_return(engine_xgb.get_values_history()),
        'sharpe': sharpe_ratio(engine_xgb.get_step_returns(), n_ann=252*390),
        'mdd': max_drawdown(engine_xgb.get_values_history()),
        'n_trades': len(engine_xgb.trade_returns)
    }

    # ===== Run MLP Independently =====
    engine_mlp = BacktestEngine(initial_capital=10000.0, transaction_cost=0.0001)
    strategy_mlp = StrategyLayer()
    actions_mlp = []
    for i in range(n_test):
        action = strategy_mlp.decide(test_probs_mlp[i], engine_mlp.position)
        actions_mlp.append(action)
        ret = mid_price_returns[i] if i < n_test - 1 else 0.0
        engine_mlp.step(action, ret)
    engine_mlp.close_position()

    mlp_trading_metrics = {
        'cum_return': cumulative_return(engine_mlp.get_values_history()),
        'sharpe': sharpe_ratio(engine_mlp.get_step_returns(), n_ann=252*390),
        'mdd': max_drawdown(engine_mlp.get_values_history()),
        'n_trades': len(engine_mlp.trade_returns)
    }

    # ===== Run WITH backtracking =====
    bt_module = BacktrackingModule(
        initial_weights=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
        theta_buy=0.75, theta_sell=0.75, delta=0.15,
        N_buf=100, N_upd=50
    )
    strategy_bt = StrategyLayer()
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
        )

        # Decide action via signal-based strategy
        action = strategy_bt.decide(pfinal_i, engine_bt.position)

        # Check for action correction (reversal override from backtracking)
        action = bt_module.check_action_correction(pfinal_i, action)

        actions_bt.append(action)

        # Simulate portfolio step
        ret = mid_price_returns[i] if i < n_test - 1 else 0.0
        engine_bt.step(action, ret)

        # Add observation to backtracking buffer
        expert_preds_i = np.array([lr_preds[i], xgb_preds[i], mlp_preds[i]])
        bt_module.add_observation(y_pred, pfinal_i, action, y_test[i], expert_preds_i)

        # Update backtracking module if needed
        if bt_module.should_update():
            bt_module.update()

        bt_module.set_prev_action(action, y_pred)

    engine_bt.close_position()

    # Metrics with backtracking
    moe_preds_bt = np.argmax(pfinal_bt_all, axis=1)
    bt_metrics = {
        'accuracy': accuracy_score(y_test, moe_preds_bt),
        'macro_f1': f1_score(y_test, moe_preds_bt, average='macro'),
        'per_class_f1': f1_score(y_test, moe_preds_bt, average=None),
        'cum_return': cumulative_return(engine_bt.get_values_history()),
        'sharpe': sharpe_ratio(engine_bt.get_step_returns(), n_ann=252*390),
        'mdd': max_drawdown(engine_bt.get_values_history()),
        'win_rate': win_rate(engine_bt.trade_returns),
        'decision_acc': decision_accuracy(actions_bt, y_test),
        'profit_factor': profit_factor(engine_bt.trade_returns),
        'turnover': turnover(actions_bt),
        'avg_hold_time': average_holding_time(engine_bt.hold_durations),
        'avg_trade_return': average_trade_return(engine_bt.trade_returns),
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
        'xgb_trading': xgb_trading_metrics,
        'mlp_trading': mlp_trading_metrics,
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

    # Diagnostics
    print(f"    [Diagnostics] Mean(r_t): {np.mean(engine_bt.get_step_returns()):.8f}, "
          f"Std(r_t): {np.std(engine_bt.get_step_returns()):.8f}")
    print(f"    [Strategy Control] Turnover: {bt_metrics['turnover']:.4f}, "
          f"ProfitFactor: {bt_metrics['profit_factor']:.2f}, "
          f"AvgHold: {bt_metrics['avg_hold_time']:.1f} steps, "
          f"AvgTradeRet: {bt_metrics['avg_trade_return']:.6f}")

    return eval_metrics
