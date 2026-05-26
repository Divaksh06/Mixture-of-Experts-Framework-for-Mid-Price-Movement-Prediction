"""
Backtracking Module for the MoE framework.

Online-adaptive module that maintains a rolling buffer of recent
predictions and outcomes, periodically adjusting:
  1. Expert gating weights (performance-based exponential update)
  2. Strategy thresholds (FP/FN-rate driven)
  3. Action correction (reversal override)

Improvements over original:
  - Increased buffer size (200) and update frequency (100)
    for more statistically stable estimates
  - Softened reversal override threshold (0.65 → was 0.70)
  - Added directional accuracy tracking for smarter weight updates
  - Clamped weight ratios to prevent any single expert from dominating
"""

import numpy as np
from collections import deque


class BacktrackingModule:
    """
    Adaptive backtracking module for online weight and threshold adjustment.
    """

    def __init__(self, initial_weights=None,
                 theta_buy=0.55, theta_sell=0.55, delta=0.15,
                 N_buf=200, N_upd=100,
                 epsilon_theta=0.02, tau_FP=0.15, tau_FN=0.10,
                 theta_reverse=0.65, lambda_ema=0.85, eta_lr=0.1):
        """
        Parameters
        ----------
        initial_weights : array-like, shape (3,)
            Starting expert weights.  Default: uniform.
        theta_buy, theta_sell : float
            Initial strategy thresholds.
        delta : float
            Margin threshold.
        N_buf : int
            Rolling buffer size.
        N_upd : int
            Update every N_upd observations.
        epsilon_theta : float
            Step size for threshold adjustment.
        tau_FP, tau_FN : float
            Rate triggers for threshold adjustment.
        theta_reverse : float
            Probability threshold for action reversal override.
        lambda_ema : float
            EMA smoothing for accuracy tracking.
        eta_lr : float
            Learning rate for exponential weight update.
        """
        if initial_weights is not None:
            self.expert_weights = np.array(initial_weights, dtype=np.float64)
        else:
            self.expert_weights = np.ones(3, dtype=np.float64) / 3.0

        self.theta_buy = theta_buy
        self.theta_sell = theta_sell
        self.delta = delta
        self.N_buf = N_buf
        self.N_upd = N_upd
        self.epsilon_theta = epsilon_theta
        self.tau_FP = tau_FP
        self.tau_FN = tau_FN
        self.theta_reverse = theta_reverse
        self.lambda_ema = lambda_ema
        self.eta_lr = eta_lr

        self.historical_alpha = np.ones(3, dtype=np.float64) / 3.0

        self.buffer = deque(maxlen=N_buf)
        self.step_count = 0
        self.prev_action = 'Hold'
        self.prev_y_pred = 1

    # -----------------------------------------------------------------
    def add_observation(self, y_pred, pfinal, action, y_true,
                        expert_preds=None):
        self.buffer.append({
            'y_pred': y_pred,
            'pfinal': pfinal.copy(),
            'action': action,
            'y_true': y_true,
            'expert_preds': (expert_preds.copy()
                             if expert_preds is not None else None),
        })
        self.step_count += 1

    def should_update(self):
        return (self.step_count > 0
                and self.step_count % self.N_upd == 0
                and len(self.buffer) >= self.N_upd)

    # -----------------------------------------------------------------
    def update(self):
        """Perform weight and threshold update based on buffer."""
        if len(self.buffer) < 10:
            return

        buf_list = list(self.buffer)
        n = len(buf_list)

        # --- Expert weight adjustment ---
        if buf_list[0]['expert_preds'] is not None:
            expert_correct = np.zeros(3)
            expert_total = np.zeros(3)

            for obs in buf_list:
                if obs['expert_preds'] is not None:
                    for e in range(3):
                        expert_total[e] += 1
                        if obs['expert_preds'][e] == obs['y_true']:
                            expert_correct[e] += 1

            alpha_hat = np.where(
                expert_total > 0,
                expert_correct / expert_total,
                1.0 / 3
            )

            # EMA smoothing
            self.historical_alpha = (
                self.lambda_ema * self.historical_alpha
                + (1 - self.lambda_ema) * alpha_hat
            )

            # Exponential update
            numerator = self.expert_weights * np.exp(
                self.eta_lr * self.historical_alpha)
            denom = numerator.sum()
            if denom > 0:
                new_w = numerator / denom
                # Clamp: no expert below 10% or above 60%
                new_w = np.clip(new_w, 0.10, 0.60)
                new_w /= new_w.sum()
                self.expert_weights = new_w
            else:
                self.expert_weights = np.ones(3) / 3.0

        # --- Threshold adjustment ---
        buy_fp, sell_fp, buy_fn, sell_fn = 0, 0, 0, 0
        buy_actions, sell_actions = 0, 0

        for obs in buf_list:
            if obs['action'] == 'Buy':
                buy_actions += 1
                if obs['y_true'] != 0:
                    buy_fp += 1
            if obs['action'] == 'Sell':
                sell_actions += 1
                if obs['y_true'] != 2:
                    sell_fp += 1
            if obs['y_true'] == 0 and obs['action'] != 'Buy':
                buy_fn += 1
            if obs['y_true'] == 2 and obs['action'] != 'Sell':
                sell_fn += 1

        fp_rate_buy = buy_fp / max(buy_actions, 1)
        fn_rate_buy = buy_fn / n if n > 0 else 0
        fp_rate_sell = sell_fp / max(sell_actions, 1)
        fn_rate_sell = sell_fn / n if n > 0 else 0

        if fp_rate_buy > self.tau_FP:
            self.theta_buy = min(self.theta_buy + self.epsilon_theta, 0.80)
        elif fn_rate_buy > self.tau_FN:
            self.theta_buy = max(self.theta_buy - self.epsilon_theta, 0.40)

        if fp_rate_sell > self.tau_FP:
            self.theta_sell = min(self.theta_sell + self.epsilon_theta, 0.80)
        elif fn_rate_sell > self.tau_FN:
            self.theta_sell = max(self.theta_sell - self.epsilon_theta, 0.40)

    # -----------------------------------------------------------------
    def check_action_correction(self, pfinal, action):
        """Override to Hold if strong reversal detected."""
        if self.prev_action == 'Buy' and pfinal[2] > self.theta_reverse:
            return 'Hold'
        if self.prev_action == 'Sell' and pfinal[0] > self.theta_reverse:
            return 'Hold'
        return action

    def get_weights(self):
        return self.expert_weights.copy()

    def get_thresholds(self):
        return {
            'theta_buy': self.theta_buy,
            'theta_sell': self.theta_sell,
            'delta': self.delta,
        }

    def set_prev_action(self, action, y_pred):
        self.prev_action = action
        self.prev_y_pred = y_pred
