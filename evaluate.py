"""
Evaluation Pipeline (Stage 4) for the MoE framework.
"""

import os
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score

from moe.mixture import compute_pfinal
from strategy.strategy_layer import StrategyLayer
from strategy.backtracking import BacktrackingModule
from backtest.engine import BacktestEngine
from backtest.metrics import (
    cumulative_return, return_to_volatility_ratio, sortino_ratio, max_drawdown,
    win_rate, decision_accuracy, buy_and_hold_return,
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
    }
    return metrics, avg_s_list, preds

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
    passive_pos = np.where(y_test == 0, 1, np.where(y_test == 2, -1, 0))
    passive_return = float(np.sum(passive_pos[:-1] * mid_price_returns[:-1]))

    # Standard 1bp Evaluations
    moe_1bp, moe_avg_s, moe_preds = run_static_strategy(pfinal_test, y_test, mid_price_returns, cost=0.0001, track_regime=True)
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
        'moe_3bp': moe_3bp,
        'xgb_1bp': xgb_1bp,
        'xgb_3bp': xgb_3bp,
        'mlp_1bp': mlp_1bp,
        
        'eq_1bp': eq_1bp,
        'wt_1bp': wt_1bp,
        
        'lr_metrics': fold_results['lr_metrics'],
        'xgb_metrics': fold_results['xgb_metrics'],
        'mlp_metrics': fold_results['mlp_metrics'],
    }

    print(f"    [Benchmarks] B&H: {bh_return:.4f} | Passive Signal: {passive_return:.4f}")
    print(f"    [MoE 1bp] Return: {moe_1bp['cum_return']:.6f} | R2V: {moe_1bp['return_to_volatility']:.4f}")
    print(f"    [XGB 1bp] Return: {xgb_1bp['cum_return']:.6f} | R2V: {xgb_1bp['return_to_volatility']:.4f}")
    print(f"    [MoE 3bp] Return: {moe_3bp['cum_return']:.6f}")
    print(f"    [EQ Ens]  Return: {eq_1bp['cum_return']:.6f}")

    return eval_metrics

