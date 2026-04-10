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

    def __init__(self, initial_capital=10000.0, transaction_cost=0.0001, c_spread=0.00015, hold_k=25):
        """
        Initialize the backtesting engine.

        Parameters
        ----------
        initial_capital : float
            Starting capital.
        transaction_cost : float
            Proportional transaction cost per side.
        c_spread : float
            Fixed assumed spread cost for transaction realism.
        hold_k : int
            Minimum number of ticks a position must be held once opened.
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.c_spread = c_spread
        self.hold_k = hold_k
        self.reset()

    def reset(self):
        """Reset the engine to initial state."""
        self.portfolio_value = self.initial_capital
        self.position = None  # None, 'Long', or 'Short'
        self.entry_value = 0.0
        self.values = [self.initial_capital]
        self.trade_returns = []
        self.actions_taken = []
        self.step_returns = []
        self.hold_k_counter = 0
        self.hold_durations = []
        self.current_trade_duration = 0

    def step(self, action, mid_price_return):
        """
        Execute one time step of the portfolio simulation.

        Parameters
        ----------
        action : str
            Trading action: 'Buy', 'Sell', 'Flat', or 'Hold'.
        mid_price_return : float
            Realized one-step return r_t = (m_{t+1} - m_t) / m_t.
        """
        # --- 1. Minimum holding lockout (hold_k) ---
        if self.hold_k_counter > 0:
            self.hold_k_counter -= 1
            action = 'Hold'   # Override to maintain state
            
        self.actions_taken.append(action)
        
        # Decide Target Position
        if action == 'Buy':
            target_pos = 'Long'
        elif action == 'Sell':
            target_pos = 'Short'
        elif action == 'Flat':
            target_pos = None
        else: # 'Hold' -> maintain position
            target_pos = self.position

        c_total = self.transaction_cost + self.c_spread

        # Execute transitions (pay swap/crossing costs)
        if self.position != target_pos:
            # We must close current position if we have one
            if self.position is not None:
                self.portfolio_value *= (1.0 - c_total)
                if self.entry_value > 0:
                    trade_ret = (self.portfolio_value - self.entry_value) / self.entry_value
                    self.trade_returns.append(trade_ret)
                    self.hold_durations.append(self.current_trade_duration)

            # We open the new position if target is not None
            if target_pos is not None:
                self.portfolio_value *= (1.0 - c_total)
                self.entry_value = self.portfolio_value
                self.current_trade_duration = 0

            # Cooldown on ALL transitions (including exit to Flat)
            self.hold_k_counter = self.hold_k
            
            self.position = target_pos

        # Apply return corresponding to the CURRENT held position during t -> t+1
        step_ret = 0.0
        if self.position == 'Long':
            self.portfolio_value *= (1.0 + mid_price_return)
            step_ret = mid_price_return
        elif self.position == 'Short':
            self.portfolio_value *= (1.0 - mid_price_return)
            step_ret = -mid_price_return

        # Track active hold period length
        if self.position is not None:
            self.current_trade_duration += 1

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
        c_total = self.transaction_cost + self.c_spread
        if self.position == 'Long':
            self.portfolio_value *= (1.0 + mid_price_return)
            self.portfolio_value *= (1.0 - c_total)
            self.position = None
            if self.entry_value > 0:
                trade_ret = (self.portfolio_value - self.entry_value) / self.entry_value
                self.trade_returns.append(trade_ret)
        elif self.position == 'Short':
            self.portfolio_value *= (1.0 - mid_price_return)
            self.portfolio_value *= (1.0 - c_total)
            self.position = None
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
