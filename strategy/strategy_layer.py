"""
Strategy Layer for the MoE framework.

Probabilistic Magnitude Signal Model with cost-aware execution.

Key design: MINIMIZE round-trip trades. Only change position when the
signal is strong enough to overcome transaction costs. The hysteresis
band prevents churn. No aggressive Flat exits — only exit when the
opposite signal is strong.

State Machine:
    signal = P(up) - P(down)

    1. If signal > tau_entry AND P(up) > theta_buy     → Buy (Long)
    2. If signal < -tau_entry AND P(down) > theta_sell  → Sell (Short)
    3. If |signal| < tau_exit AND in position            → Flat (exit)
    4. Otherwise                                         → Hold (maintain)

    Flip penalty: reversals require |signal| > tau_entry + flip_penalty
"""

import numpy as np


from collections import deque

class StrategyLayer:
    """
    Cost-aware magnitude signal strategy with hysteresis, flip penalty, and regime filter.
    """

    def __init__(self, tau_entry=0.60, tau_exit=0.15,
                 theta_buy=0.75, theta_sell=0.75,
                 flip_penalty=0.10, s_lookback=20, s_threshold=0.55):
        self.tau_entry = tau_entry
        self.tau_exit = tau_exit
        self.theta_buy = theta_buy
        self.theta_sell = theta_sell
        self.flip_penalty = flip_penalty
        
        self.s_lookback = s_lookback
        self.s_threshold = s_threshold
        self.s_history = deque(maxlen=s_lookback)

    def decide(self, pfinal, current_position=None):
        """
        Convert a probability vector to a trading action.

        Parameters
        ----------
        pfinal : np.ndarray, shape (3,)
            Final class probabilities [p_up, p_stationary, p_down].
        current_position : str or None
            Current engine position ('Long', 'Short', None).

        Returns
        -------
        action : str
            One of 'Buy', 'Sell', 'Flat', or 'Hold'.
        """
        signal = pfinal[0] - pfinal[2]
        abs_signal = abs(signal)

        # Update rolling signal strength state
        s_t = max(pfinal[0], pfinal[2])
        self.s_history.append(s_t)

        # Check regime filter
        if len(self.s_history) < self.s_lookback:
            is_regime_valid = False  # Ensure window is warm before trading
        else:
            avg_s = sum(self.s_history) / self.s_lookback
            is_regime_valid = (avg_s > self.s_threshold)

        # --- 1. Strong entry/reversal zone ---
        if abs_signal > self.tau_entry and is_regime_valid:
            is_flip = (
                (signal > 0 and current_position == 'Short') or
                (signal < 0 and current_position == 'Long')
            )
            required = self.tau_entry + self.flip_penalty if is_flip else self.tau_entry

            if abs_signal > required:
                if signal > 0 and pfinal[0] > self.theta_buy:
                    return 'Buy'
                if signal < 0 and pfinal[2] > self.theta_sell:
                    return 'Sell'

        # --- 2. Weak signal exit (only if in a position) ---
        # Use a TIGHT exit threshold to avoid churning
        if abs_signal < self.tau_exit and current_position is not None:
            return 'Flat'

        # --- 3. Hysteresis: maintain current state ---
        return 'Hold'

    def decide_batch(self, pfinal_batch, positions=None):
        """Convert a batch of probability vectors to trading actions."""
        if positions is None:
            positions = [None] * len(pfinal_batch)
        return [self.decide(pfinal_batch[i], positions[i])
                for i in range(len(pfinal_batch))]

    def update_thresholds(self, theta_buy=None, theta_sell=None, delta=None):
        """Update thresholds (kept for backtracking compatibility)."""
        if theta_buy is not None:
            self.theta_buy = np.clip(theta_buy, 0.50, 0.90)
        if theta_sell is not None:
            self.theta_sell = np.clip(theta_sell, 0.50, 0.90)
