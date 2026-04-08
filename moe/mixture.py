"""
Mixture-of-Experts Forward Pass.

Provides utility functions to stack expert probabilities and compute
the final MoE prediction as the gating-weighted combination of expert outputs.
"""

import numpy as np


def get_stacked_probs(probs_lr, probs_xgb, probs_mlp):
    """
    Concatenate three expert probability vectors into a 9-dim vector.

    Parameters
    ----------
    probs_lr : np.ndarray, shape (n_samples, 3)
        Logistic Regression expert probabilities.
    probs_xgb : np.ndarray, shape (n_samples, 3)
        XGBoost expert probabilities.
    probs_mlp : np.ndarray, shape (n_samples, 3)
        MLP expert probabilities.

    Returns
    -------
    stacked : np.ndarray, shape (n_samples, 9)
        Concatenated probability vectors.
    """
    return np.concatenate([probs_lr, probs_xgb, probs_mlp], axis=1)


def compute_pfinal(weights, probs_lr, probs_xgb, probs_mlp):
    """
    Compute the final MoE probability prediction.

    P_final = w1 * P_LR + w2 * P_XGB + w3 * P_MLP

    Parameters
    ----------
    weights : np.ndarray, shape (n_samples, 3)
        Expert combination weights from the gating network.
    probs_lr : np.ndarray, shape (n_samples, 3)
        LR expert probabilities.
    probs_xgb : np.ndarray, shape (n_samples, 3)
        XGBoost expert probabilities.
    probs_mlp : np.ndarray, shape (n_samples, 3)
        MLP expert probabilities.

    Returns
    -------
    pfinal : np.ndarray, shape (n_samples, 3)
        Final combined class probabilities.
    """
    pfinal = (
        weights[:, 0:1] * probs_lr +
        weights[:, 1:2] * probs_xgb +
        weights[:, 2:3] * probs_mlp
    )
    return pfinal
