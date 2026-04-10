"""
Financial Performance Metrics for the Backtesting Engine.

Computes cumulative return, annualized Sharpe ratio, maximum drawdown,
win rate, decision accuracy, and buy-and-hold benchmark return.
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


def sharpe_ratio(returns, n_ann=252, rf=0.0):
    """
    Compute annualized Sharpe ratio.

    The annualization factor is ``sqrt(n_ann)``, so ``n_ann`` **must**
    match the observation frequency of *returns*:

        - Daily returns   → n_ann = 252  (default)
        - Hourly returns  → n_ann = 252 * 6.5 ≈ 1638
        - Minute returns  → n_ann = 252 * 390 = 98 280

    Using the wrong frequency inflates / deflates the ratio by
    ``sqrt(wrong / correct)``.

    Parameters
    ----------
    returns : list or np.ndarray
        Per-step strategy returns.
    n_ann : int
        Number of return observations per trading year.
        Default: 252 (daily returns — 252 trading days per year).
    rf : float
        Risk-free rate **per period** (same frequency as *returns*).
        Default: 0.

    Returns
    -------
    sharpe : float
        Annualized Sharpe ratio. Returns 0 if std is 0.
    """
    returns = np.array(returns)
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess = returns - rf
    return (np.mean(excess) / np.std(excess)) * np.sqrt(n_ann)


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
