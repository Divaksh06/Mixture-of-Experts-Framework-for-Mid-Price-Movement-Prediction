"""
Anchored Forward Cross-Validation Iterator.

The FI-2010 dataset ships with 9 pre-split folds (CF_1 … CF_9).
Each fold is an anchored-forward split: training grows cumulatively
while the test set is always the next unseen day.

This module simply yields fold indices 1…N_FOLDS.
"""


def anchored_forward_cv(n_folds=9):
    """
    Yield fold indices for the anchored forward CV scheme.

    Parameters
    ----------
    n_folds : int
        Total number of folds (default 9, matching FI-2010 benchmark).

    Yields
    ------
    fold_idx : int
        Fold index from 1 to n_folds.
    """
    for fold_idx in range(1, n_folds + 1):
        yield fold_idx
