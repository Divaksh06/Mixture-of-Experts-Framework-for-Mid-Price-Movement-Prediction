"""
Anchored Forward Cross-Validation Fold Iterator for FI-2010.

The FI-2010 dataset specifies 9 cross-validation folds (1 through 9).
In each fold t, training uses days 1..t and testing uses day t+1.
The pre-split files already encode this protocol, so this module
simply yields fold indices from 1 to 9.
"""


def anchored_forward_cv(n_folds=9):
    """
    Generator that yields fold indices for the anchored forward
    cross-validation protocol used in FI-2010.

    Parameters
    ----------
    n_folds : int
        Number of folds. Default is 9 (as per the FI-2010 benchmark).

    Yields
    ------
    fold_idx : int
        Fold index from 1 to n_folds (inclusive).
    """
    for fold_idx in range(1, n_folds + 1):
        yield fold_idx
