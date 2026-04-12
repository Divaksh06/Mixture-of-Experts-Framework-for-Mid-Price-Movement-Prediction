"""
Backtracking Module for the MoE framework.

Operates online during inference. Maintains a rolling memory buffer of
recent predictions, actions, and outcomes. Periodically computes
per-expert accuracy and adjusts:
  1. Expert gating weights (performance-based scaling)
  2. Strategy thresholds (adaptive based on false positive / false negative rates)
  3. Action correction (override to Hold if strong reversal detected)
"""

import numpy as np
from collections import deque


class BacktrackingModule:
    """
    Adaptive backtracking module for online weight and threshold adjustment.

    Attributes
    ----------
    N_buf : int
        Rolling buffer size.
    N_upd : int
        Update frequency (every N_upd observations).
    epsilon_theta : float
        Threshold adjustment step size.
    tau_FP : float
        False positive rate threshold for increasing thresholds.
    tau_FN : float
        False negative rate threshold for decreasing thresholds.
    theta_reverse : float
        Probability threshold for action reversal.
    error_margin : float
        Tolerance for treating a prediction as correct (default 0.05).
    expert_weights : np.ndarray, shape (3,)
        Current expert weights.
    theta_buy : float
        Current buy threshold.
    theta_sell : float
        Current sell threshold.
    delta : float
        Current margin threshold.
    """

    def __init__(self, initial_weights=None, theta_buy=0.70, theta_sell=0.70,
                 delta=0.15, N_buf=100, N_upd=50, epsilon_theta=0.02,
                 tau_FP=0.15, tau_FN=0.10, theta_reverse=0.70,
                 error_margin=0.05, lambda_ema=0.9, eta_lr=0.1,
                 tau_entry=0.60, tau_exit=0.15):
        """
        Initialize the backtracking module.

        Parameters
        ----------
        initial_weights : np.ndarray, shape (3,) or None
            Initial expert combination weights. Defaults to uniform.
        theta_buy : float
            Initial buy threshold.
        theta_sell : float
            Initial sell threshold.
        delta : float
            Initial margin threshold.
        N_buf : int
            Buffer size.
        N_upd : int
            Update frequency.
        epsilon_theta : float
            Step size for threshold adjustment.
        tau_FP : float
            False positive rate trigger threshold.
        tau_FN : float
            False negative rate trigger threshold.
        theta_reverse : float
            Reversal signal threshold.
        error_margin : float
            Tolerance for treating a prediction as correct (default 0.05).
            A prediction is considered correct if
            |prediction - true_label| <= error_margin.
        lambda_ema : float
            EMA smoothing factor for accuracy tracking.
        eta_lr : float
            Learning rate for weight exponential update.
        """
        if initial_weights is not None:
            self.expert_weights = np.array(initial_weights, dtype=np.float64)
        else:
            self.expert_weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3],
                                           dtype=np.float64)

        self.theta_buy = theta_buy
        self.theta_sell = theta_sell
        self.delta = delta
        self.tau_entry = tau_entry
        self.tau_exit = tau_exit
        self.N_buf = N_buf
        self.N_upd = N_upd
        self.epsilon_theta = epsilon_theta
        self.tau_FP = tau_FP
        self.tau_FN = tau_FN
        self.theta_reverse = theta_reverse
        self.error_margin = error_margin
        self.lambda_ema = lambda_ema
        self.eta_lr = eta_lr
        self.historical_alpha = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float64)

        # Buffer stores tuples: (y_pred, pfinal, action, y_true, expert_preds)
        self.buffer = deque(maxlen=N_buf)
        self.step_count = 0
        self.prev_action = 'Hold'
        self.prev_y_pred = 1  # default: stationary

    def add_observation(self, y_pred, pfinal, action, y_true,
                        expert_preds=None):
        """
        Add a new observation to the memory buffer.

        Parameters
        ----------
        y_pred : int
            Predicted class (0, 1, or 2).
        pfinal : np.ndarray, shape (3,)
            Final combined probabilities.
        action : str
            Trading action taken ('Buy', 'Hold', 'Sell').
        y_true : int
            True class label.
        expert_preds : np.ndarray, shape (3,) or None
            Individual expert predictions (one per expert).
        """
        self.buffer.append({
            'y_pred': y_pred,
            'pfinal': pfinal.copy(),
            'action': action,
            'y_true': y_true,
            'expert_preds': expert_preds.copy() if expert_preds is not None else None
        })
        self.step_count += 1

    def should_update(self):
        """Check if it's time to run an update step."""
        return (self.step_count > 0 and
                self.step_count % self.N_upd == 0 and
                len(self.buffer) >= self.N_upd)

    def update(self):
        """
        Perform weight and threshold update based on buffer contents.

        Updates expert_weights and theta_buy, theta_sell, delta based
        on per-expert accuracy and false positive/negative rates.
        """
        if len(self.buffer) < 10:
            return

        buf_list = list(self.buffer)
        n = len(buf_list)

        # --- Expert Weight Adjustment ---
        if buf_list[0]['expert_preds'] is not None:
            # Compute per-expert accuracy
            expert_correct = np.zeros(3)
            expert_total = np.zeros(3)

            for obs in buf_list:
                if obs['expert_preds'] is not None:
                    for e_idx in range(3):
                        expert_total[e_idx] += 1
                        if abs(obs['expert_preds'][e_idx] - obs['y_true']) <= self.error_margin:
                            expert_correct[e_idx] += 1

            alpha_hat = np.zeros(3)
            for e_idx in range(3):
                if expert_total[e_idx] > 0:
                    alpha_hat[e_idx] = expert_correct[e_idx] / expert_total[e_idx]
                else:
                    alpha_hat[e_idx] = 1.0 / 3
            
            # EMA of accuracy
            self.historical_alpha = (self.lambda_ema * self.historical_alpha 
                                     + (1 - self.lambda_ema) * alpha_hat)

            # Exponential update: w'_i = w_i * exp(eta * alpha_i) / sum
            numerator = self.expert_weights * np.exp(self.eta_lr * self.historical_alpha)
            denom = numerator.sum()
            if denom > 0:
                self.expert_weights = numerator / denom
            else:
                self.expert_weights = np.ones(3) / 3.0

        # --- Threshold Adjustment ---
        # Count false positives and false negatives for Buy/Sell
        buy_fp = 0  # Predicted Buy (action='Buy'), but true was not Up
        buy_fn = 0  # True was Up, but action was not Buy (missed opportunity)
        sell_fp = 0
        sell_fn = 0
        buy_actions = 0
        sell_actions = 0
        true_up = 0
        true_down = 0

        for obs in buf_list:
            if obs['action'] == 'Buy':
                buy_actions += 1
                if obs['y_true'] != 0:  # Not actually Up
                    buy_fp += 1
            if obs['action'] == 'Sell':
                sell_actions += 1
                if obs['y_true'] != 2:  # Not actually Down
                    sell_fp += 1
            if obs['y_true'] == 0:
                true_up += 1
                if obs['action'] != 'Buy':
                    buy_fn += 1
            if obs['y_true'] == 2:
                true_down += 1
                if obs['action'] != 'Sell':
                    sell_fn += 1

        # Compute rates
        fp_rate_buy = buy_fp / n if n > 0 else 0
        fn_rate_buy = buy_fn / n if n > 0 else 0
        fp_rate_sell = sell_fp / n if n > 0 else 0
        fn_rate_sell = sell_fn / n if n > 0 else 0

        # Adjust theta_buy
        if fp_rate_buy > self.tau_FP:
            self.theta_buy = min(self.theta_buy + self.epsilon_theta, 0.85)
        elif fn_rate_buy > self.tau_FN:
            self.theta_buy = max(self.theta_buy - self.epsilon_theta, 0.45)

        # Adjust theta_sell
        if fp_rate_sell > self.tau_FP:
            self.theta_sell = min(self.theta_sell + self.epsilon_theta, 0.85)
        elif fn_rate_sell > self.tau_FN:
            self.theta_sell = max(self.theta_sell - self.epsilon_theta, 0.45)

        # Adjust tau_entry — the primary execution gate
        # High combined FP → tighten entry (fewer, higher-quality trades)
        # High combined FN → loosen entry (capture missed opportunities)
        combined_fp = (fp_rate_buy + fp_rate_sell)
        combined_fn = (fn_rate_buy + fn_rate_sell)
        if combined_fp > 2 * self.tau_FP:
            self.tau_entry = min(self.tau_entry + self.epsilon_theta, 0.80)
            self.tau_exit = max(self.tau_exit - 0.01, 0.05)
        elif combined_fn > 2 * self.tau_FN:
            self.tau_entry = max(self.tau_entry - self.epsilon_theta, 0.40)
            self.tau_exit = min(self.tau_exit + 0.01, 0.25)

    def check_action_correction(self, pfinal, action):
        """
        Check if the current action should be overridden due to reversal.

        If the new prediction strongly contradicts a recent action,
        override to Hold.

        Parameters
        ----------
        pfinal : np.ndarray, shape (3,)
            Current final probabilities.
        action : str
            Current proposed action.

        Returns
        -------
        corrected_action : str
            Possibly corrected action.
        """
        # If previous action was Buy (prev predicted Up=0),
        # and now P(Down=2) > theta_reverse => override to Hold
        if self.prev_action == 'Buy' and pfinal[2] > self.theta_reverse:
            return 'Hold'
        # If previous action was Sell (prev predicted Down=2),
        # and now P(Up=0) > theta_reverse => override to Hold
        if self.prev_action == 'Sell' and pfinal[0] > self.theta_reverse:
            return 'Hold'
        return action

    def get_weights(self):
        """
        Get current expert weights.

        Returns
        -------
        weights : np.ndarray, shape (3,)
        """
        return self.expert_weights.copy()

    def get_thresholds(self):
        """
        Get current strategy thresholds.

        Returns
        -------
        thresholds : dict
            Dictionary with 'theta_buy', 'theta_sell', 'delta'.
        """
        return {
            'theta_buy': self.theta_buy,
            'theta_sell': self.theta_sell,
            'delta': self.delta,
            'tau_entry': self.tau_entry,
            'tau_exit': self.tau_exit,
        }

    def set_prev_action(self, action, y_pred):
        """
        Store the previous action and prediction for reversal detection.

        Parameters
        ----------
        action : str
        y_pred : int
        """
        self.prev_action = action
        self.prev_y_pred = y_pred
