"""
Strategy Layer for the MoE framework.

Probabilistic Magnitude Signal Model with cost-aware execution.

Improvements over original:
  - Lowered default tau_entry (0.35) for more trading opportunities
  - Lowered theta_buy/theta_sell (0.55) so calibrated probabilities can
    trigger trades more often
  - Reduced regime filter warmup from 20 to 10 ticks
  - Added confidence smoothing via EMA of signal to reduce whipsaw
"""

import numpy as np
from collections import deque


class StrategyLayer:
    """
    Cost-aware magnitude signal strategy with hysteresis, flip penalty,
    and regime filter.

    State Machine
    -------------
    signal = P(up) − P(down)

    1. |signal| > tau_entry AND directional prob > theta  → Buy / Sell
    2. |signal| < tau_exit  AND in position                → Flat (exit)
    3. Otherwise                                           → Hold
    """

    def __init__(self, tau_entry=0.35, tau_exit=0.10,
                 theta_buy=0.55, theta_sell=0.55,
                 flip_penalty=0.10,
                 s_lookback=10, s_threshold=0.45,
                 ema_alpha=0.3):
        self.tau_entry = tau_entry
        self.tau_exit = tau_exit
        self.theta_buy = theta_buy
        self.theta_sell = theta_sell
        self.flip_penalty = flip_penalty

        self.s_lookback = s_lookback
        self.s_threshold = s_threshold
        self.s_history = deque(maxlen=s_lookback)

        # EMA smoothing of the signal to reduce whipsaw trades
        self.ema_alpha = ema_alpha
        self.ema_signal = 0.0
        self.ema_initialised = False

    def decide(self, pfinal, current_position=None):
        """
        Convert a probability vector to a trading action.

        Parameters
        ----------
        pfinal : np.ndarray, shape (3,)
            [P(up), P(stationary), P(down)]
        current_position : str or None
            'Long', 'Short', or None

        Returns
        -------
        action : str
            'Buy', 'Sell', 'Flat', or 'Hold'
        """
        raw_signal = pfinal[0] - pfinal[2]

        # EMA smoothing
        if not self.ema_initialised:
            self.ema_signal = raw_signal
            self.ema_initialised = True
        else:
            self.ema_signal = (self.ema_alpha * raw_signal +
                               (1 - self.ema_alpha) * self.ema_signal)

        signal = self.ema_signal
        abs_signal = abs(signal)

        # Rolling strength for regime filter
        s_t = max(pfinal[0], pfinal[2])
        self.s_history.append(s_t)

        if len(self.s_history) < self.s_lookback:
            is_regime_valid = False
        else:
            avg_s = sum(self.s_history) / self.s_lookback
            is_regime_valid = (avg_s > self.s_threshold)

        # --- 1. Entry / reversal zone ---
        if abs_signal > self.tau_entry and is_regime_valid:
            is_flip = (
                (signal > 0 and current_position == 'Short') or
                (signal < 0 and current_position == 'Long')
            )
            required = (self.tau_entry + self.flip_penalty
                        if is_flip else self.tau_entry)

            if abs_signal > required:
                if signal > 0 and pfinal[0] > self.theta_buy:
                    return 'Buy'
                if signal < 0 and pfinal[2] > self.theta_sell:
                    return 'Sell'

        # --- 2. Weak-signal exit ---
        if abs_signal < self.tau_exit and current_position is not None:
            return 'Flat'

        # --- 3. Hysteresis: maintain current state ---
        return 'Hold'

    def decide_batch(self, pfinal_batch, positions=None):
        if positions is None:
            positions = [None] * len(pfinal_batch)
        return [self.decide(pfinal_batch[i], positions[i])
                for i in range(len(pfinal_batch))]

    def update_thresholds(self, theta_buy=None, theta_sell=None, delta=None):
        """Update thresholds (kept for backtracking compatibility)."""
        if theta_buy is not None:
            self.theta_buy = np.clip(theta_buy, 0.40, 0.80)
        if theta_sell is not None:
            self.theta_sell = np.clip(theta_sell, 0.40, 0.80)
