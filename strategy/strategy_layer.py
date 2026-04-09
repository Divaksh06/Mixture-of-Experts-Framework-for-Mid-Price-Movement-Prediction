"""
Strategy Layer for the MoE framework.

Converts the soft probability output P_final = [p_up, p_stat, p_down]
into a discrete trading action using confidence margin filtering and
class-specific thresholds.

Decision Rules:
1. Confidence margin filter: margin = max(P) - second_max(P) > delta
   If not satisfied, action = Hold.
2. If margin passes:
   - Buy  if y_hat == 0 (Up) and p_up > theta_buy
   - Sell if y_hat == 2 (Down) and p_down > theta_sell
   - Hold otherwise
"""

import numpy as np


class StrategyLayer:
    """
    Strategy decision module with configurable thresholds.

    Attributes
    ----------
    theta_buy : float
        Minimum probability threshold for Buy actions.
    theta_sell : float
        Minimum probability threshold for Sell actions.
    delta : float
        Minimum confidence margin between top-two class probabilities.
    """

    def __init__(self, theta_buy=0.55, theta_sell=0.55, delta=0.10):
        """
        Initialize the strategy layer.

        Parameters
        ----------
        theta_buy : float
            Buy confidence threshold.
        theta_sell : float
            Sell confidence threshold.
        delta : float
            Minimum margin for directional trades.
        """
        self.theta_buy = theta_buy
        self.theta_sell = theta_sell
        self.delta = delta

    def decide(self, pfinal):
        """
        Convert a single probability vector to a trading action.

        Parameters
        ----------
        pfinal : np.ndarray, shape (3,)
            Final class probabilities [p_up, p_stationary, p_down].

        Returns
        -------
        action : str
            One of 'Buy', 'Sell', or 'Hold'.
        """
        # Predicted class
        y_hat = np.argmax(pfinal)

        # Confidence margin filter
        sorted_probs = np.sort(pfinal)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]

        if margin <= self.delta:
            return 'Hold'

        # Class-specific threshold check
        # Class 0 = Up -> Buy
        if y_hat == 0 and pfinal[0] > self.theta_buy:
            return 'Buy'
        # Class 2 = Down -> Sell
        elif y_hat == 2 and pfinal[2] > self.theta_sell:
            return 'Sell'
        else:
            return 'Hold'

    def decide_batch(self, pfinal_batch):
        """
        Convert a batch of probability vectors to trading actions.

        Parameters
        ----------
        pfinal_batch : np.ndarray, shape (n_samples, 3)

        Returns
        -------
        actions : list of str
            List of 'Buy', 'Sell', or 'Hold' for each sample.
        """
        return [self.decide(pfinal_batch[i]) for i in range(len(pfinal_batch))]

    def update_thresholds(self, theta_buy=None, theta_sell=None, delta=None):
        """
        Update the strategy thresholds.

        Parameters
        ----------
        theta_buy : float or None
        theta_sell : float or None
        delta : float or None
        """
        if theta_buy is not None:
            self.theta_buy = np.clip(theta_buy, 0.45, 0.85)
        if theta_sell is not None:
            self.theta_sell = np.clip(theta_sell, 0.45, 0.85)
        if delta is not None:
            self.delta = np.clip(delta, 0.01, 0.40)
