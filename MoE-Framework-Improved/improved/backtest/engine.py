"""
Backtesting Engine for the MoE framework.

Simulates a hypothetical trading account using strategy actions and
realised mid-price returns.  Supports Long + Short positions.

Improvements over original:
  - Reduced hold_k from 25 to 15 for more responsive position management
  - Configurable spread cost
  - hold_durations tracked for close_position as well
  - Cleaner position-transition logic
"""

import numpy as np


class BacktestEngine:
    """
    Portfolio simulation engine.

    Supports Long and Short positions.
    Buy  → open Long  (or close Short → open Long)
    Sell → open Short (or close Long  → open Short)
    Flat → close any position
    Hold → maintain current position
    """

    def __init__(self, initial_capital=10000.0, transaction_cost=0.0001,
                 c_spread=0.00010, hold_k=15):
        """
        Parameters
        ----------
        initial_capital : float
        transaction_cost : float
            Proportional cost per trade side.
        c_spread : float
            Assumed bid-ask spread cost per side.
        hold_k : int
            Minimum number of ticks a position must be held.
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.c_spread = c_spread
        self.hold_k = hold_k
        self.reset()

    def reset(self):
        self.portfolio_value = self.initial_capital
        self.position = None          # None, 'Long', or 'Short'
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
        Execute one time step.

        Parameters
        ----------
        action : str  — 'Buy', 'Sell', 'Flat', or 'Hold'
        mid_price_return : float  — r_t = (m_{t+1} − m_t) / m_t
        """
        # --- Minimum-holding lockout ---
        if self.hold_k_counter > 0:
            self.hold_k_counter -= 1
            action = 'Hold'

        self.actions_taken.append(action)

        # Determine target position
        if action == 'Buy':
            target_pos = 'Long'
        elif action == 'Sell':
            target_pos = 'Short'
        elif action == 'Flat':
            target_pos = None
        else:
            target_pos = self.position   # Hold

        c_total = self.transaction_cost + self.c_spread

        # Position transition
        if self.position != target_pos:
            # Close current position if exists
            if self.position is not None:
                self.portfolio_value *= (1.0 - c_total)
                if self.entry_value > 0:
                    trade_ret = ((self.portfolio_value - self.entry_value)
                                 / self.entry_value)
                    self.trade_returns.append(trade_ret)
                    self.hold_durations.append(self.current_trade_duration)

            # Open new position if target is not flat
            if target_pos is not None:
                self.portfolio_value *= (1.0 - c_total)
                self.entry_value = self.portfolio_value
                self.current_trade_duration = 0

            self.hold_k_counter = self.hold_k
            self.position = target_pos

        # Apply return for held position
        step_ret = 0.0
        if self.position == 'Long':
            self.portfolio_value *= (1.0 + mid_price_return)
            step_ret = mid_price_return
        elif self.position == 'Short':
            self.portfolio_value *= (1.0 - mid_price_return)
            step_ret = -mid_price_return

        if self.position is not None:
            self.current_trade_duration += 1

        self.values.append(self.portfolio_value)
        self.step_returns.append(step_ret)

    def close_position(self, mid_price_return=0.0):
        """Force-close any open position."""
        c_total = self.transaction_cost + self.c_spread
        if self.position == 'Long':
            self.portfolio_value *= (1.0 + mid_price_return)
            self.portfolio_value *= (1.0 - c_total)
            if self.entry_value > 0:
                trade_ret = ((self.portfolio_value - self.entry_value)
                             / self.entry_value)
                self.trade_returns.append(trade_ret)
                self.hold_durations.append(self.current_trade_duration)
            self.position = None
        elif self.position == 'Short':
            self.portfolio_value *= (1.0 - mid_price_return)
            self.portfolio_value *= (1.0 - c_total)
            if self.entry_value > 0:
                trade_ret = ((self.portfolio_value - self.entry_value)
                             / self.entry_value)
                self.trade_returns.append(trade_ret)
                self.hold_durations.append(self.current_trade_duration)
            self.position = None

    def get_portfolio_value(self):
        return self.portfolio_value

    def get_values_history(self):
        return self.values

    def get_step_returns(self):
        return self.step_returns
