"""
FI-2010 Dataset Loader.

Loads a single anchored-forward cross-validation fold from the
FI-2010 benchmark dataset.  Each .txt file is (features+labels × samples),
so it is transposed before slicing.

Feature layout (after transpose):
    Columns 0–143   : 144 LOB features (already normalised for Zscore variant)
    Column  144      : label at horizon k=1
    Column  145      : label at horizon k=2
    Column  146      : label at horizon k=3
    Column  147      : label at horizon k=5
    Column  148      : label at horizon k=10   ← primary target

Raw labels {1, 2, 3} are mapped to {0, 1, 2}  (Up, Stationary, Down).
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def load_fold(fold_idx, dataset_root, normalization='NoAuction_Zscore'):
    """
    Load train/test data for one anchored-forward fold.

    Parameters
    ----------
    fold_idx : int
        Fold number (1 through 9).
    dataset_root : str
        Root path to BenchmarkDatasets/.
    normalization : str
        One of 'NoAuction_Zscore', 'NoAuction_MinMax', 'NoAuction_DecPre',
        'Auction_Zscore', etc.

    Returns
    -------
    X_train : np.ndarray, shape (N_train, 144), float32
    y_train : np.ndarray, shape (N_train,), int64   — labels for k=10
    X_test  : np.ndarray, shape (N_test, 144), float32
    y_test  : np.ndarray, shape (N_test,), int64    — labels for k=10
    class_weights : np.ndarray, shape (3,), float32
    y_train_k5 : np.ndarray, shape (N_train,), int64 — labels for k=5
    y_test_k5  : np.ndarray, shape (N_test,), int64  — labels for k=5
    """
    # --- resolve directory structure ---
    parts = normalization.split('_', 1)
    category, norm_name = parts                         # e.g. 'NoAuction', 'Zscore'

    norm_prefix_map = {'Zscore': '1', 'MinMax': '2', 'DecPre': '3'}
    prefix = norm_prefix_map[norm_name]

    # file-level name capitalisation quirk in the dataset
    file_norm_name_map = {'Zscore': 'ZScore', 'MinMax': 'MinMax', 'DecPre': 'DecPre'}
    file_norm_name = file_norm_name_map[norm_name]

    norm_dir   = os.path.join(dataset_root, category,
                              f"{prefix}.{category}_{norm_name}")
    train_dir  = os.path.join(norm_dir, f"{category}_{norm_name}_Training")
    test_dir   = os.path.join(norm_dir, f"{category}_{norm_name}_Testing")

    train_file = os.path.join(
        train_dir, f"Train_Dst_{category}_{file_norm_name}_CF_{fold_idx}.txt")
    test_file  = os.path.join(
        test_dir,  f"Test_Dst_{category}_{file_norm_name}_CF_{fold_idx}.txt")

    # --- load & transpose (raw shape: rows=features+labels, cols=samples) ---
    train_raw = np.loadtxt(train_file).T        # → (N_train, 149)
    test_raw  = np.loadtxt(test_file).T          # → (N_test,  149)

    # --- split features / labels ---
    X_train = train_raw[:, :144].astype(np.float32)
    X_test  = test_raw[:,  :144].astype(np.float32)

    y_train_raw_k10 = train_raw[:, 148].astype(np.int64)
    y_test_raw_k10  = test_raw[:,  148].astype(np.int64)
    y_train_raw_k5  = train_raw[:, 147].astype(np.int64)
    y_test_raw_k5   = test_raw[:,  147].astype(np.int64)

    # --- optional re-scaling for non-Zscore variants ---
    if 'Zscore' not in normalization:
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

    # --- remap labels: {1,2,3} → {0,1,2} ---
    y_train    = y_train_raw_k10 - 1
    y_test     = y_test_raw_k10  - 1
    y_train_k5 = y_train_raw_k5  - 1
    y_test_k5  = y_test_raw_k5   - 1

    # --- balanced class weights ---
    classes = np.array([0, 1, 2])
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = cw.astype(np.float32)

    return X_train, y_train, X_test, y_test, class_weights, y_train_k5, y_test_k5


def load_mid_prices(fold_idx, dataset_root, split='test'):
    """
    Load raw (un-normalised) mid-prices for return computation.

    The FI-2010 Zscore-normalised data does not contain raw prices directly.
    We reconstruct a proxy mid-price from the *NoAuction_DecPre* variant
    (decimal-precision normalised) where ask/bid prices retain scale.

    If the DecPre variant is unavailable, fall back to the Zscore features
    and derive a *relative* mid-price proxy  (F0 + F2) / 2.

    Parameters
    ----------
    fold_idx : int
    dataset_root : str
    split : str, 'train' or 'test'

    Returns
    -------
    mid_prices : np.ndarray, shape (N,)
    """
    # Try DecPre first for realistic prices
    decpre_dir = os.path.join(dataset_root, 'NoAuction',
                              '3.NoAuction_DecPre')
    if split == 'test':
        sub = 'NoAuction_DecPre_Testing'
        prefix = 'Test_Dst_NoAuction_DecPre'
    else:
        sub = 'NoAuction_DecPre_Training'
        prefix = 'Train_Dst_NoAuction_DecPre'

    fpath = os.path.join(decpre_dir, sub, f"{prefix}_CF_{fold_idx}.txt")

    if os.path.isfile(fpath):
        raw = np.loadtxt(fpath).T
        best_ask = raw[:, 0]
        best_bid = raw[:, 2]
        mid_prices = (best_ask + best_bid) / 2.0
        return mid_prices

    # Fallback: Zscore variant — ask=F0, bid=F2 (normalised, but relative
    # changes are still directionally correct for return sign)
    parts = 'NoAuction_Zscore'.split('_', 1)
    category, norm_name = parts
    norm_dir  = os.path.join(dataset_root, category,
                             f"1.{category}_{norm_name}")
    if split == 'test':
        sub = f"{category}_{norm_name}_Testing"
        prefix = f"Test_Dst_{category}_ZScore"
    else:
        sub = f"{category}_{norm_name}_Training"
        prefix = f"Train_Dst_{category}_ZScore"

    fpath = os.path.join(norm_dir, sub, f"{prefix}_CF_{fold_idx}.txt")
    raw = np.loadtxt(fpath).T
    best_ask = raw[:, 0]
    best_bid = raw[:, 2]
    mid_prices = (best_ask + best_bid) / 2.0
    return mid_prices
