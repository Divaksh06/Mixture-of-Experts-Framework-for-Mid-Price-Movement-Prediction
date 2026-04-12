"""
Evaluation Pipeline (Stage 4) for the MoE framework.
"""

import os
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score

from moe.mixture import compute_pfinal, get_stacked_probs
from strategy.strategy_layer import StrategyLayer
from strategy.backtracking import BacktrackingModule
from backtest.engine import BacktestEngine
from backtest.metrics import (
    cumulative_return, return_to_volatility_ratio, sortino_ratio, max_drawdown,
    win_rate, decision_accuracy, buy_and_hold_return, passive_signal_return,
    profit_factor, turnover, average_holding_time, average_trade_return
)

def run_static_strategy(probs, y_true, mid_price_returns, cost=0.0001, track_regime=False):
    strategy = StrategyLayer()
    engine = BacktestEngine(initial_capital=10000.0, transaction_cost=cost)
    actions = []
    avg_s_list = []
    
    n = len(y_true)
    for i in range(n):
        action = strategy.decide(probs[i], engine.position)
        actions.append(action)
        if track_regime:
            q = list(strategy.signal_queue) if hasattr(strategy, 'signal_queue') else [0]
            avg_s = sum(q)/len(q) if q else 0.0
            avg_s_list.append(avg_s)
            
        ret = mid_price_returns[i] if i < n - 1 else 0.0
        engine.step(action, ret)
    engine.close_position()
    
    preds = np.argmax(probs, axis=1)
    metrics = {
        'cum_return': cumulative_return(engine.get_values_history()),
        'return_to_volatility': return_to_volatility_ratio(engine.get_step_returns()),
        'sortino': sortino_ratio(engine.get_step_returns()),
        'mdd': max_drawdown(engine.get_values_history()),
        'win_rate': win_rate(engine.trade_returns),
        'n_trades': len(engine.trade_returns),
        'accuracy': accuracy_score(y_true, preds),
        'macro_f1': f1_score(y_true, preds, average='macro'),
        'turnover': turnover(actions),
        'profit_factor': profit_factor(engine.trade_returns),
        'decision_accuracy': decision_accuracy(actions, y_true),
    }
    return metrics, avg_s_list, preds

def run_backtracking_strategy(probs, y_true, mid_price_returns, cost=0.0001):
    strategy = StrategyLayer()
    backtracker = BacktrackingModule(theta_buy=0.75, theta_sell=0.75,
                                     tau_entry=0.60, tau_exit=0.15)
    engine = BacktestEngine(initial_capital=10000.0, transaction_cost=cost)
    actions = []
    
    n = len(y_true)
    for i in range(n):
        if backtracker.should_update():
            backtracker.update()
            
        # Wire ALL thresholds from BT → strategy (including tau_entry/tau_exit)
        thresh = backtracker.get_thresholds()
        strategy.theta_buy = thresh['theta_buy']
        strategy.theta_sell = thresh['theta_sell']
        strategy.tau_entry = thresh['tau_entry']
        strategy.tau_exit = thresh['tau_exit']
        
        action = strategy.decide(probs[i], engine.position)
        action = backtracker.check_action_correction(probs[i], action)
        
        y_pred = np.argmax(probs[i])
        backtracker.set_prev_action(action, y_pred)
        
        actions.append(action)
        
        ret = mid_price_returns[i] if i < n - 1 else 0.0
        engine.step(action, ret)
        
        backtracker.add_observation(y_pred, probs[i], action, y_true[i], expert_preds=None)
        
    engine.close_position()
    
    preds = np.argmax(probs, axis=1)
    metrics = {
        'cum_return': cumulative_return(engine.get_values_history()),
        'return_to_volatility': return_to_volatility_ratio(engine.get_step_returns()),
        'sortino': sortino_ratio(engine.get_step_returns()),
        'mdd': max_drawdown(engine.get_values_history()),
        'win_rate': win_rate(engine.trade_returns),
        'n_trades': len(engine.trade_returns),
        'accuracy': accuracy_score(y_true, preds),
        'macro_f1': f1_score(y_true, preds, average='macro'),
        'turnover': turnover(actions),
        'profit_factor': profit_factor(engine.trade_returns),
        'decision_accuracy': decision_accuracy(actions, y_true),
    }
    return metrics


def compute_confidence_routed_pfinal(probs_xgb, probs_mlp, probs_lr, confidence_thresh=0.60):
    """
    Confidence-based routing: bypass gating when an expert is highly confident.
    
    If max(XGB probs) > threshold → use XGB directly
    elif max(MLP probs) > threshold → use MLP directly  
    else → fallback to XGB (strongest single expert)
    
    Returns pfinal with the same shape as input expert probs.
    """
    n = len(probs_xgb)
    pfinal = np.zeros_like(probs_xgb)
    route_counts = np.zeros(3)  # [xgb_conf, mlp_conf, fallback_xgb]
    
    for i in range(n):
        max_xgb = np.max(probs_xgb[i])
        max_mlp = np.max(probs_mlp[i])
        
        if max_xgb > confidence_thresh:
            pfinal[i] = probs_xgb[i]
            route_counts[0] += 1
        elif max_mlp > confidence_thresh:
            pfinal[i] = probs_mlp[i]
            route_counts[1] += 1
        else:
            pfinal[i] = probs_xgb[i]  # fallback
            route_counts[2] += 1
    
    total = route_counts.sum()
    print(f"    [Confidence Routing] XGB-conf: {route_counts[0]/total:.1%} | "
          f"MLP-conf: {route_counts[1]/total:.1%} | Fallback-XGB: {route_counts[2]/total:.1%}")
    
    return pfinal


def evaluate_fold(fold_idx, fold_results, results_dir='results'):
    print(f"\n  Fold {fold_idx}: Stage 4 — Signal Strategy + Multi-Horizon Backtest...")

    X_test = fold_results['X_test']
    y_test = fold_results['y_test']
    test_probs_lr = fold_results['test_probs_lr']
    test_probs_xgb = fold_results['test_probs_xgb']
    test_probs_mlp = fold_results['test_probs_mlp']
    pfinal_test = fold_results['pfinal_test']

    n_test = len(y_test)

    mid_price_returns = np.zeros(n_test)
    from data.load_fi2010 import load_mid_prices
    try:
        mid_prices = load_mid_prices(fold_idx, 'data/BenchmarkDatasets', split='test')
        for i in range(n_test - 1):
            if mid_prices[i] > 0:
                mid_price_returns[i] = (mid_prices[i+1] - mid_prices[i]) / mid_prices[i]
    except Exception as e:
        print(f"    Warning: Could not load real mid prices ({e}).")

    # Benchmarks
    bh_return = buy_and_hold_return(mid_price_returns)
    passive_return = passive_signal_return(y_test, mid_price_returns)

    # Gate Weights Diagnostics
    gating_trainer = fold_results['gating_trainer']
    gating_scaler = fold_results['gating_scaler']
    stacked_test = get_stacked_probs(test_probs_lr, test_probs_xgb, test_probs_mlp)
    gate_input_test = np.hstack([X_test, stacked_test])
    gate_input_test = gating_scaler.transform(gate_input_test)
    test_weights = gating_trainer.get_weights(gate_input_test)
    mean_weights = test_weights.mean(axis=0)
    # Argmax routing: which expert would be selected under hard routing
    favoured = np.argmax(test_weights, axis=1)
    expert_names = ['LR', 'XGB', 'MLP']
    route_counts = np.bincount(favoured, minlength=3)
    route_pcts = route_counts / route_counts.sum()
    print(f"    [Gate Weights]  LR: {mean_weights[0]:.4f} | XGB: {mean_weights[1]:.4f} | MLP: {mean_weights[2]:.4f}")
    print(f"    [Argmax Route]  LR: {route_pcts[0]:.1%} | XGB: {route_pcts[1]:.1%} | MLP: {route_pcts[2]:.1%}  (favoured: {expert_names[np.argmax(route_counts)]})")

    # Confidence-routed pfinal
    pfinal_cr = compute_confidence_routed_pfinal(test_probs_xgb, test_probs_mlp, test_probs_lr)

    # Standard 1bp Evaluations
    moe_1bp, moe_avg_s, moe_preds = run_static_strategy(pfinal_test, y_test, mid_price_returns, cost=0.0001, track_regime=True)
    moe_bt_1bp = run_backtracking_strategy(pfinal_test, y_test, mid_price_returns, cost=0.0001)
    moe_cr_1bp, _, _ = run_static_strategy(pfinal_cr, y_test, mid_price_returns, cost=0.0001)
    xgb_1bp, _, _ = run_static_strategy(test_probs_xgb, y_test, mid_price_returns, cost=0.0001)
    mlp_1bp, _, _ = run_static_strategy(test_probs_mlp, y_test, mid_price_returns, cost=0.0001)

    # 3bp Sensitivity Iterations
    moe_3bp, _, _ = run_static_strategy(pfinal_test, y_test, mid_price_returns, cost=0.0003)
    xgb_3bp, _, _ = run_static_strategy(test_probs_xgb, y_test, mid_price_returns, cost=0.0003)

    # Gating Ablation
    f1_lr = fold_results['lr_metrics']['macro_f1']
    f1_xgb = fold_results['xgb_metrics']['macro_f1']
    f1_mlp = fold_results['mlp_metrics']['macro_f1']
    total_f1 = f1_lr + f1_xgb + f1_mlp
    
    w_lr = f1_lr / total_f1
    w_xgb = f1_xgb / total_f1
    w_mlp = f1_mlp / total_f1

    pfinal_eq = (test_probs_lr + test_probs_xgb + test_probs_mlp) / 3.0
    pfinal_weighted = w_lr*test_probs_lr + w_xgb*test_probs_xgb + w_mlp*test_probs_mlp

    eq_1bp, _, _ = run_static_strategy(pfinal_eq, y_test, mid_price_returns, cost=0.0001)
    wt_1bp, _, _ = run_static_strategy(pfinal_weighted, y_test, mid_price_returns, cost=0.0001)


    eval_metrics = {
        'bh_return': bh_return,
        'passive_return': passive_return,
        
        'moe_1bp': moe_1bp,
        'moe_bt_1bp': moe_bt_1bp,
        'moe_cr_1bp': moe_cr_1bp,
        'moe_3bp': moe_3bp,
        'xgb_1bp': xgb_1bp,
        'xgb_3bp': xgb_3bp,
        'mlp_1bp': mlp_1bp,
        
        'eq_1bp': eq_1bp,
        'wt_1bp': wt_1bp,
        
        'lr_metrics': fold_results['lr_metrics'],
        'xgb_metrics': fold_results['xgb_metrics'],
        'mlp_metrics': fold_results['mlp_metrics'],
        
        'gate_mean_weights': mean_weights,
        'gate_route_pcts': route_pcts,
    }

    print(f"    [Benchmarks] B&H: {bh_return:.4f} | Passive Signal: {passive_return:.4f}")
    print(f"    [MoE 1bp] Return: {moe_1bp['cum_return']:.6f} | R2V: {moe_1bp['return_to_volatility']:.4f}")
    print(f"    [MoE+BT]  Return: {moe_bt_1bp['cum_return']:.6f} | Trades: {moe_bt_1bp['n_trades']}")
    print(f"    [MoE+CR]  Return: {moe_cr_1bp['cum_return']:.6f} | Trades: {moe_cr_1bp['n_trades']}")
    print(f"    [XGB 1bp] Return: {xgb_1bp['cum_return']:.6f} | R2V: {xgb_1bp['return_to_volatility']:.4f}")

    return eval_metrics

