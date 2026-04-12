"""
Financial Performance Metrics for the Backtesting Engine.

Computes cumulative return, return-to-volatility ratio, Sortino ratio, 
maximum drawdown, win rate, decision accuracy, and benchmark returns.
"""

import numpy as np


def cumulative_return(values):
    """
    Compute cumulative return from a portfolio value series.

    Parameters
    ----------
    values : list or np.ndarray
        Portfolio value history (starting from initial capital).

    Returns
    -------
    rcum : float
        (V_T - V_0) / V_0
    """
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / values[0]


def return_to_volatility_ratio(returns, rf=0.0):
    """
    Compute raw time-series return-to-volatility ratio (non-annualized Sharpe).

    Parameters
    ----------
    returns : list or np.ndarray
        Per-step strategy returns.
    rf : float
        Risk-free rate per period. Default: 0.

    Returns
    -------
    ratio : float
        Signal-to-noise ratio within the dataset timeframe.
    """
    returns = np.array(returns)
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess = returns - rf
    return float(np.mean(excess) / np.std(excess))


def sortino_ratio(returns, rf=0.0):
    """
    Compute Sortino ratio evaluating strictly downside volatility.
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0
    excess = returns - rf
    downside = np.minimum(excess, 0)
    downside_std = np.std(downside)
    
    if downside_std == 0:
        print("    [Warning] Sortino undefined due to zero downside volatility")
        return 0.0
    return float(np.mean(excess) / downside_std)


def max_drawdown(values):
    """
    Compute maximum drawdown from a portfolio value series.

    Parameters
    ----------
    values : list or np.ndarray
        Portfolio value history.

    Returns
    -------
    mdd : float
        Maximum drawdown as a positive fraction (0 to 1).
    """
    values = np.array(values)
    if len(values) < 2:
        return 0.0
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return float(np.max(drawdown))


def win_rate(trade_returns):
    """
    Compute the fraction of profitable trades.

    Parameters
    ----------
    trade_returns : list or np.ndarray
        Returns of individual completed trades.

    Returns
    -------
    wr : float
        Fraction of trades with positive return. 0 if no trades.
    """
    if len(trade_returns) == 0:
        return 0.0
    trade_returns = np.array(trade_returns)
    return float(np.sum(trade_returns > 0)) / len(trade_returns)


def decision_accuracy(actions, y_true):
    """
    Compute directional accuracy of non-Hold actions.

    A Buy is correct if the true label is Up (0).
    A Sell is correct if the true label is Down (2).

    Parameters
    ----------
    actions : list of str
        List of actions ('Buy', 'Hold', 'Sell').
    y_true : np.ndarray
        True labels in {0, 1, 2}.

    Returns
    -------
    da : float
        Fraction of correct directional calls among Buy/Sell actions.
    """
    correct = 0
    total = 0
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
    """
    Compute cumulative return of a buy-and-hold strategy.

    Parameters
    ----------
    mid_price_returns : np.ndarray
        Per-step mid-price returns.

    Returns
    -------
    bh_return : float
        Cumulative return from holding throughout.
    """
    if len(mid_price_returns) == 0:
        return 0.0
    cumulative = np.prod(1.0 + np.array(mid_price_returns)) - 1.0
    return float(cumulative)


def passive_signal_return(y_true, mid_price_returns):
    """
    Compute the oracle upper-bound return using perfect directional knowledge,
    using geometric compounding to prevent numerical instability.
    """
    # 0=Up, 1=Stationary, 2=Down 
    passive_pos = np.where(y_true == 0, 1, np.where(y_true == 2, -1, 0))
    oracle_returns = passive_pos[:-1] * mid_price_returns[:-1]
    
    wealth = 1.0
    for r in oracle_returns:
        wealth *= (1.0 + r)
    return float(wealth - 1.0)


def profit_factor(trade_returns):
    """
    Compute Profit Factor: Gross Profit / Gross Loss.
    """
    trade_returns = np.array(trade_returns)
    if len(trade_returns) == 0:
        return 0.0
    gross_profit = np.sum(trade_returns[trade_returns > 0])
    gross_loss = np.abs(np.sum(trade_returns[trade_returns < 0]))
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def turnover(actions):
    """
    Compute total transaction count turnover roughly based on action switches.
    Number of entry/exit actions over total dataset length.
    """
    if len(actions) == 0:
        return 0.0
    trades = sum(1 for a in actions if a in ['Buy', 'Sell'])
    return trades / len(actions)


def average_holding_time(hold_durations):
    """
    Compute the average number of ticks a position was held.
    """
    if not hold_durations:
        return 0.0
    return float(np.mean(hold_durations))


def average_trade_return(trade_returns):
    """
    Compute the average return per completed trade.
    """
    if not trade_returns:
        return 0.0
    return float(np.mean(trade_returns))
