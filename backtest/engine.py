"""
Backtesting Engine for the MoE framework.

Simulates a hypothetical trading account using the sequence of strategy
actions and realized mid-price returns. Implements long-only trading
with proportional transaction costs.
"""

import numpy as np


class BacktestEngine:
    """
    Portfolio simulation engine.

    Simulates trading with Buy/Hold/Sell signals, tracking portfolio value
    over time. Long-only: Buy opens a position, Sell closes it.

    Attributes
    ----------
    initial_capital : float
        Starting portfolio value.
    transaction_cost : float
        Proportional cost per trade side (e.g., 0.0001 = 1 basis point).
    portfolio_value : float
        Current portfolio value.
    position_open : bool
        Whether a long position is currently open.
    values : list of float
        Portfolio value history.
    trade_returns : list of float
        Returns of completed trades.
    actions_taken : list of str
        History of actions taken.
    """

    def __init__(self, initial_capital=10000.0, transaction_cost=0.0001):
        """
        Initialize the backtesting engine.

        Parameters
        ----------
        initial_capital : float
            Starting capital.
        transaction_cost : float
            Proportional transaction cost per side.
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        """Reset the engine to initial state."""
        self.portfolio_value = self.initial_capital
        self.position_open = False
        self.entry_value = 0.0
        self.values = [self.initial_capital]
        self.trade_returns = []
        self.actions_taken = []
        self.step_returns = []

    def step(self, action, mid_price_return):
        """
        Execute one time step of the portfolio simulation.

        Parameters
        ----------
        action : str
            Trading action: 'Buy', 'Sell', or 'Hold'.
        mid_price_return : float
            Realized one-step return r_t = (m_{t+1} - m_t) / m_t.
        """
        self.actions_taken.append(action)
        step_ret = 0.0

        if action == 'Buy' and not self.position_open:
            # Open long position, pay transaction cost
            self.portfolio_value *= (1.0 - self.transaction_cost)
            self.position_open = True
            self.entry_value = self.portfolio_value

        elif action == 'Sell' and self.position_open:
            # Apply the return first, then close with transaction cost
            self.portfolio_value *= (1.0 + mid_price_return)
            self.portfolio_value *= (1.0 - self.transaction_cost)
            self.position_open = False
            # Record trade return
            if self.entry_value > 0:
                trade_ret = (self.portfolio_value - self.entry_value) / self.entry_value
                self.trade_returns.append(trade_ret)
            step_ret = mid_price_return - self.transaction_cost

        elif self.position_open:
            # Position is open and action is Hold (or Buy while already in)
            self.portfolio_value *= (1.0 + mid_price_return)
            step_ret = mid_price_return

        # If no position and Hold or Sell, no change
        self.values.append(self.portfolio_value)
        self.step_returns.append(step_ret)

    def close_position(self, mid_price_return=0.0):
        """
        Force-close any open position at end of day.

        Parameters
        ----------
        mid_price_return : float
            Final return for closing.
        """
        if self.position_open:
            self.portfolio_value *= (1.0 + mid_price_return)
            self.portfolio_value *= (1.0 - self.transaction_cost)
            self.position_open = False
            if self.entry_value > 0:
                trade_ret = (self.portfolio_value - self.entry_value) / self.entry_value
                self.trade_returns.append(trade_ret)

    def get_portfolio_value(self):
        """
        Get the current portfolio value.

        Returns
        -------
        value : float
        """
        return self.portfolio_value

    def get_values_history(self):
        """
        Get the full portfolio value history.

        Returns
        -------
        values : list of float
        """
        return self.values

    def get_step_returns(self):
        """
        Get per-step returns.

        Returns
        -------
        returns : list of float
        """
        return self.step_returns
