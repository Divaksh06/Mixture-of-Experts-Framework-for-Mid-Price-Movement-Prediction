"""
Financial Performance Metrics for the Backtesting Engine.

Computes cumulative return, annualised Sharpe ratio, maximum drawdown,
win rate, decision accuracy, and various trade-level statistics.
"""

import numpy as np


def cumulative_return(values):
    """(V_T − V_0) / V_0"""
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / values[0]


def sharpe_ratio(returns, n_ann=252*390, rf=0.0):
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns : array-like  — per-step returns
    n_ann   : int — periods per year (252 trading days × 390 minutes)
    rf      : float — risk-free rate per period
    """
    returns = np.array(returns)
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess = returns - rf
    return float((np.mean(excess) / np.std(excess)) * np.sqrt(n_ann))


def max_drawdown(values):
    """Maximum drawdown as a positive fraction (0 to 1)."""
    values = np.array(values)
    if len(values) < 2:
        return 0.0
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / peak
    return float(np.max(dd))


def win_rate(trade_returns):
    """Fraction of profitable trades."""
    if len(trade_returns) == 0:
        return 0.0
    tr = np.array(trade_returns)
    return float(np.sum(tr > 0)) / len(tr)


def decision_accuracy(actions, y_true):
    """
    Directional accuracy of non-Hold actions.
    Buy correct if true=Up(0), Sell correct if true=Down(2).
    """
    correct = total = 0
    for a, yt in zip(actions, y_true):
        if a == 'Buy':
            total += 1
            if yt == 0:
                correct += 1
        elif a == 'Sell':
            total += 1
            if yt == 2:
                correct += 1
    return correct / total if total > 0 else 0.0


def buy_and_hold_return(mid_price_returns):
    """Cumulative return from holding long throughout."""
    if len(mid_price_returns) == 0:
        return 0.0
    return float(np.prod(1.0 + np.array(mid_price_returns)) - 1.0)


def profit_factor(trade_returns):
    """Gross Profit / Gross Loss."""
    tr = np.array(trade_returns)
    if len(tr) == 0:
        return 0.0
    gross_profit = np.sum(tr[tr > 0])
    gross_loss = np.abs(np.sum(tr[tr < 0]))
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def turnover(actions):
    """Fraction of time steps with entry/exit actions."""
    if len(actions) == 0:
        return 0.0
    trades = sum(1 for a in actions if a in ['Buy', 'Sell', 'Flat'])
    return trades / len(actions)


def average_holding_time(hold_durations):
    """Average ticks a position was held."""
    if not hold_durations:
        return 0.0
    return float(np.mean(hold_durations))


def average_trade_return(trade_returns):
    """Average return per completed trade."""
    if not trade_returns:
        return 0.0
    return float(np.mean(trade_returns))
