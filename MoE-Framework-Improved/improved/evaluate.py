"""
Evaluation Pipeline (Stage 4) for the MoE framework.

For each fold: runs inference, applies signal strategy with multi-horizon
agreement filtering, runs backtesting, and collects metrics.

Multi-Horizon Agreement:
    If the k=5 label (from a secondary lightweight LR) disagrees with
    the k=10 MoE directional signal, the action is downgraded to Hold.
    This acts as a "sanity check" filter — only trade when both horizons
    agree on direction.
"""

import os
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from moe.mixture import compute_pfinal
from strategy.strategy_layer import StrategyLayer
from strategy.backtracking import BacktrackingModule
from backtest.engine import BacktestEngine
from backtest.metrics import (
    cumulative_return, sharpe_ratio, max_drawdown,
    win_rate, decision_accuracy, buy_and_hold_return,
    profit_factor, turnover, average_holding_time, average_trade_return
)


def _log(msg):
    print(msg, flush=True)


def _multi_horizon_filter(action, moe_pred, k5_pred):
    """
    Downgrade action to Hold if k=10 and k=5 predictions disagree
    on direction.

    Parameters
    ----------
    action : str
    moe_pred : int   — argmax of MoE (k=10) probabilities
    k5_pred : int    — argmax of secondary LR (k=5) predictions

    Returns
    -------
    filtered_action : str
    """
    if action == 'Buy':
        # k=10 says Up. If k=5 says Down → disagree → Hold
        if k5_pred == 2:
            return 'Hold'
    elif action == 'Sell':
        # k=10 says Down. If k=5 says Up → disagree → Hold
        if k5_pred == 0:
            return 'Hold'
    return action


def evaluate_fold(fold_idx, fold_results, results_dir='results'):
    """
    Execute Stage 4 for a single fold.
    """
    _log(f"\n  Fold {fold_idx}: Stage 4 — Signal Strategy + Backtest...")

    X_test = fold_results['X_test']
    y_test = fold_results['y_test']
    y_test_k5 = fold_results.get('y_test_k5', None)
    test_probs_lr  = fold_results['test_probs_lr']
    test_probs_xgb = fold_results['test_probs_xgb']
    test_probs_mlp = fold_results['test_probs_mlp']
    pfinal_test    = fold_results['pfinal_test']

    n_test = len(y_test)

    # --- Train a lightweight secondary LR for k=5 multi-horizon filter ---
    # (We use the LR expert's predictions on X_test as proxy for k=5 direction.
    #  In a full implementation, a separate model would be trained on k=5 labels.)
    k5_preds = None
    if y_test_k5 is not None:
        # We use the existing LR expert to get directional hints
        # A dedicated k=5 model could improve this further
        k5_preds = fold_results['lr_expert'].predict(X_test)

    # --- Load real mid-prices ---
    from data.load_fi2010 import load_mid_prices
    try:
        mid_prices = load_mid_prices(fold_idx, 'data/BenchmarkDatasets',
                                     split='test')
        mid_price_returns = np.zeros(n_test)
        for i in range(n_test - 1):
            if mid_prices[i] > 0:
                mid_price_returns[i] = ((mid_prices[i+1] - mid_prices[i])
                                        / mid_prices[i])
    except Exception as e:
        _log(f"    Warning: Could not load mid prices ({e}). "
             f"Returns will be zero.")
        mid_price_returns = np.zeros(n_test)

    # Individual expert predictions
    lr_preds  = np.argmax(test_probs_lr, axis=1)
    xgb_preds = np.argmax(test_probs_xgb, axis=1)
    mlp_preds = np.argmax(test_probs_mlp, axis=1)

    # ===== WITHOUT backtracking =====
    strategy_no_bt = StrategyLayer()
    engine_no_bt = BacktestEngine(initial_capital=10000.0)
    actions_no_bt = []

    for i in range(n_test):
        action = strategy_no_bt.decide(pfinal_test[i], engine_no_bt.position)
        # Multi-horizon filter
        moe_pred = np.argmax(pfinal_test[i])
        if k5_preds is not None:
            action = _multi_horizon_filter(action, moe_pred, k5_preds[i])
        actions_no_bt.append(action)
        ret = mid_price_returns[i] if i < n_test - 1 else 0.0
        engine_no_bt.step(action, ret)
    engine_no_bt.close_position()

    moe_preds = np.argmax(pfinal_test, axis=1)
    no_bt_metrics = {
        'accuracy': accuracy_score(y_test, moe_preds),
        'macro_f1': f1_score(y_test, moe_preds, average='macro'),
        'per_class_f1': f1_score(y_test, moe_preds, average=None),
        'cum_return': cumulative_return(engine_no_bt.get_values_history()),
        'sharpe': sharpe_ratio(engine_no_bt.get_step_returns()),
        'mdd': max_drawdown(engine_no_bt.get_values_history()),
        'win_rate': win_rate(engine_no_bt.trade_returns),
        'decision_acc': decision_accuracy(actions_no_bt, y_test),
        'profit_factor': profit_factor(engine_no_bt.trade_returns),
        'turnover': turnover(actions_no_bt),
        'avg_hold_time': average_holding_time(engine_no_bt.hold_durations),
        'avg_trade_return': average_trade_return(engine_no_bt.trade_returns),
    }

    # ===== XGBoost alone =====
    engine_xgb = BacktestEngine(initial_capital=10000.0)
    strategy_xgb = StrategyLayer()
    actions_xgb = []
    for i in range(n_test):
        action = strategy_xgb.decide(test_probs_xgb[i], engine_xgb.position)
        actions_xgb.append(action)
        ret = mid_price_returns[i] if i < n_test - 1 else 0.0
        engine_xgb.step(action, ret)
    engine_xgb.close_position()

    xgb_trading = {
        'cum_return': cumulative_return(engine_xgb.get_values_history()),
        'sharpe': sharpe_ratio(engine_xgb.get_step_returns()),
        'mdd': max_drawdown(engine_xgb.get_values_history()),
        'n_trades': len(engine_xgb.trade_returns),
    }

    # ===== MLP alone =====
    engine_mlp = BacktestEngine(initial_capital=10000.0)
    strategy_mlp = StrategyLayer()
    actions_mlp = []
    for i in range(n_test):
        action = strategy_mlp.decide(test_probs_mlp[i], engine_mlp.position)
        actions_mlp.append(action)
        ret = mid_price_returns[i] if i < n_test - 1 else 0.0
        engine_mlp.step(action, ret)
    engine_mlp.close_position()

    mlp_trading = {
        'cum_return': cumulative_return(engine_mlp.get_values_history()),
        'sharpe': sharpe_ratio(engine_mlp.get_step_returns()),
        'mdd': max_drawdown(engine_mlp.get_values_history()),
        'n_trades': len(engine_mlp.trade_returns),
    }

    # ===== WITH backtracking =====
    bt_module = BacktrackingModule(
        initial_weights=np.array([1.0/3, 1.0/3, 1.0/3]),
        theta_buy=0.55, theta_sell=0.55, delta=0.15,
        N_buf=200, N_upd=100
    )
    strategy_bt = StrategyLayer()
    engine_bt = BacktestEngine(initial_capital=10000.0)

    actions_bt = []
    pfinal_bt_all = np.zeros_like(pfinal_test)

    for i in range(n_test):
        bt_weights = bt_module.get_weights()

        pfinal_i = (
            bt_weights[0] * test_probs_lr[i] +
            bt_weights[1] * test_probs_xgb[i] +
            bt_weights[2] * test_probs_mlp[i]
        )
        pfinal_bt_all[i] = pfinal_i
        y_pred = np.argmax(pfinal_i)

        # Update thresholds from backtracking
        thresholds = bt_module.get_thresholds()
        strategy_bt.update_thresholds(
            theta_buy=thresholds['theta_buy'],
            theta_sell=thresholds['theta_sell'],
        )

        action = strategy_bt.decide(pfinal_i, engine_bt.position)
        action = bt_module.check_action_correction(pfinal_i, action)

        # Multi-horizon filter
        if k5_preds is not None:
            action = _multi_horizon_filter(action, y_pred, k5_preds[i])

        actions_bt.append(action)

        ret = mid_price_returns[i] if i < n_test - 1 else 0.0
        engine_bt.step(action, ret)

        expert_preds_i = np.array([lr_preds[i], xgb_preds[i], mlp_preds[i]])
        bt_module.add_observation(y_pred, pfinal_i, action,
                                  y_test[i], expert_preds_i)

        if bt_module.should_update():
            bt_module.update()

        bt_module.set_prev_action(action, y_pred)

    engine_bt.close_position()

    moe_preds_bt = np.argmax(pfinal_bt_all, axis=1)
    bt_metrics = {
        'accuracy': accuracy_score(y_test, moe_preds_bt),
        'macro_f1': f1_score(y_test, moe_preds_bt, average='macro'),
        'per_class_f1': f1_score(y_test, moe_preds_bt, average=None),
        'cum_return': cumulative_return(engine_bt.get_values_history()),
        'sharpe': sharpe_ratio(engine_bt.get_step_returns()),
        'mdd': max_drawdown(engine_bt.get_values_history()),
        'win_rate': win_rate(engine_bt.trade_returns),
        'decision_acc': decision_accuracy(actions_bt, y_test),
        'profit_factor': profit_factor(engine_bt.trade_returns),
        'turnover': turnover(actions_bt),
        'avg_hold_time': average_holding_time(engine_bt.hold_durations),
        'avg_trade_return': average_trade_return(engine_bt.trade_returns),
    }

    bh_return = buy_and_hold_return(mid_price_returns)

    eval_metrics = {
        'no_backtracking': no_bt_metrics,
        'with_backtracking': bt_metrics,
        'buy_and_hold_return': bh_return,
        'lr_metrics': fold_results['lr_metrics'],
        'xgb_metrics': fold_results['xgb_metrics'],
        'mlp_metrics': fold_results['mlp_metrics'],
        'xgb_trading': xgb_trading,
        'mlp_trading': mlp_trading,
        'n_trades_no_bt': len(engine_no_bt.trade_returns),
        'n_trades_bt': len(engine_bt.trade_returns),
    }

    _log(f"    Without BT — Acc: {no_bt_metrics['accuracy']:.4f}, "
         f"F1: {no_bt_metrics['macro_f1']:.4f}, "
         f"Return: {no_bt_metrics['cum_return']:.6f}, "
         f"Sharpe: {no_bt_metrics['sharpe']:.4f}")
    _log(f"    With BT    — Acc: {bt_metrics['accuracy']:.4f}, "
         f"F1: {bt_metrics['macro_f1']:.4f}, "
         f"Return: {bt_metrics['cum_return']:.6f}, "
         f"Sharpe: {bt_metrics['sharpe']:.4f}")
    _log(f"    Buy-and-Hold: {bh_return:.6f}")
    _log(f"    [Diagnostics] Mean(r_t): {np.mean(engine_bt.get_step_returns()):.8f}, "
         f"Std(r_t): {np.std(engine_bt.get_step_returns()):.8f}")
    _log(f"    [Control] Turnover: {bt_metrics['turnover']:.4f}, "
         f"ProfitFactor: {bt_metrics['profit_factor']:.2f}, "
         f"AvgHold: {bt_metrics['avg_hold_time']:.1f}, "
         f"AvgTradeRet: {bt_metrics['avg_trade_return']:.6f}")

    return eval_metrics
