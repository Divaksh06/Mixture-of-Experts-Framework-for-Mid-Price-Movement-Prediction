"""
FI-2010 Benchmark Dataset Loader.

Loads the FI-2010 .txt files, extracts features and labels, applies
Z-score normalization, converts labels to 0-indexed, and computes
class weights for balanced training.

IMPORTANT NOTE ON FILE FORMAT:
The FI-2010 .txt files are TRANSPOSED relative to the usual convention.
Each file has shape (149, N_samples):
  - Rows 0–143:   144 features (each row is one feature across all samples)
  - Rows 144–148:  5 label rows for horizons k=1,2,3,5,10
So we must transpose after loading to get (N_samples, 149).
The k=10 horizon label is the LAST row, i.e., row index 148 (0-indexed).
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def load_fold(fold_idx, dataset_root, normalization='NoAuction_Zscore'):
    """
    Load a single cross-validation fold from the FI-2010 dataset.

    Parameters
    ----------
    fold_idx : int
        Fold index (1 through 9).
    dataset_root : str
        Root directory containing the BenchmarkDatasets folder structure.
    normalization : str
        Normalization variant to use. Default: 'NoAuction_Zscore'.
        Options: 'NoAuction_Zscore', 'NoAuction_MinMax', 'NoAuction_DecPre',
                 'Auction_Zscore', 'Auction_MinMax', 'Auction_DecPre'.

    Returns
    -------
    X_train : np.ndarray, shape (n_train, 144)
        Training features (Z-score normalized).
    y_train : np.ndarray, shape (n_train,)
        Training labels in {0, 1, 2} for k=10 horizon.
    X_test : np.ndarray, shape (n_test, 144)
        Test features (Z-score normalized using training statistics).
    y_test : np.ndarray, shape (n_test,)
        Test labels in {0, 1, 2} for k=10 horizon.
    class_weights : np.ndarray, shape (3,)
        Balanced class weights computed from training labels.
    y_train_k5 : np.ndarray, shape (n_train,)
        Training labels in {0, 1, 2} for k=5 horizon.
    y_test_k5 : np.ndarray, shape (n_test,)
        Test labels in {0, 1, 2} for k=5 horizon.
    """
    # Parse normalization string: we ignore it internally to FORCE DecPre for feature math
    # but still parse it to maintain API compatibility.
    parts = normalization.split('_', 1)
    if len(parts) != 2:
        raise ValueError(
            f"normalization must be like 'NoAuction_Zscore', got '{normalization}'"
        )
    category = "NoAuction"  # Force
    norm_name = "DecPre"    # Force for correct Imbalance math

    # Determine the numeric prefix for the normalization subfolder
    norm_prefix_map = {'Zscore': '1', 'MinMax': '2', 'DecPre': '3'}
    if norm_name not in norm_prefix_map:
        raise ValueError(
            f"Unknown normalization '{norm_name}'. Must be one of {list(norm_prefix_map.keys())}"
        )
    prefix = norm_prefix_map[norm_name]

    # Map norm_name to the casing used in actual filenames
    file_norm_name_map = {'Zscore': 'ZScore', 'MinMax': 'MinMax', 'DecPre': 'DecPre'}
    file_norm_name = file_norm_name_map[norm_name]

    # Build directory paths
    norm_dir = os.path.join(
        dataset_root, category,
        f"{prefix}.{category}_{norm_name}"
    )

    train_dir = os.path.join(norm_dir, f"{category}_{norm_name}_Training")
    test_dir = os.path.join(norm_dir, f"{category}_{norm_name}_Testing")

    # Build file paths (filenames use ZScore not Zscore)
    train_file = os.path.join(
        train_dir, f"Train_Dst_{category}_{file_norm_name}_CF_{fold_idx}.txt"
    )
    test_file = os.path.join(
        test_dir, f"Test_Dst_{category}_{file_norm_name}_CF_{fold_idx}.txt"
    )

    # Load data files
    try:
        # FI-2010 .txt files are TRANSPOSED: shape (149, N_samples)
        # Rows 0-143 = features, rows 144-148 = labels for k=1,2,3,5,10
        train_raw = np.loadtxt(train_file)
        test_raw = np.loadtxt(test_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find dataset file. Expected:\n"
            f"  Train: {train_file}\n"
            f"  Test:  {test_file}\n"
            f"Please ensure the FI-2010 dataset is placed at '{dataset_root}' "
            f"with the correct directory structure.\n"
            f"Original error: {e}"
        )

    # Transpose to get (N_samples, 149)
    train_raw = train_raw.T  # (N_train, 149)
    test_raw = test_raw.T    # (N_test, 149)

    # Extract features: columns 0-143 (144 features)
    X_train = train_raw[:, :144].astype(np.float32)
    X_test = test_raw[:, :144].astype(np.float32)

    # Compute additional features for Train and Test before Z-score
    def compute_features(X):
        # LOB column mapping (level i -> ask_p, ask_v, bid_p, bid_v)
        # Levels 1 to 5 correspond to 0-3, 4-7, 8-11, 12-15, 16-19
        ask_p1 = X[:, 0]
        ask_v1 = X[:, 1]
        bid_p1 = X[:, 2]
        bid_v1 = X[:, 3]

        ask_v_L5 = X[:, [1, 5, 9, 13, 17]].sum(axis=1)
        bid_v_L5 = X[:, [3, 7, 11, 15, 19]].sum(axis=1)

        imbalance_l1 = (bid_v1 - ask_v1) / (bid_v1 + ask_v1 + 1e-8)
        imbalance_l5 = (bid_v_L5 - ask_v_L5) / (bid_v_L5 + ask_v_L5 + 1e-8)
        spread = ask_p1 - bid_p1

        return np.column_stack([imbalance_l1, imbalance_l5, spread])

    # Append features (144 -> 147 features)
    X_train = np.hstack([X_train, compute_features(X_train)])
    X_test = np.hstack([X_test, compute_features(X_test)])

    # Extract k=10 horizon labels: column index 148
    # Row indices 144,145,146,147,148 correspond to k=1,2,3,5,10
    y_train_raw = train_raw[:, 148].astype(np.int64)
    y_test_raw = test_raw[:, 148].astype(np.int64)

    # Extract k=5 horizon labels: column index 147
    y_train_k5_raw = train_raw[:, 147].astype(np.int64)
    y_test_k5_raw = test_raw[:, 147].astype(np.int64)

    # Re-apply Z-score normalization: fit on training fold, transform both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert labels from {1, 2, 3} to {0, 1, 2}
    y_train = y_train_raw - 1
    y_test = y_test_raw - 1
    y_train_k5 = y_train_k5_raw - 1
    y_test_k5 = y_test_k5_raw - 1

    # Validate label values
    assert set(np.unique(y_train)).issubset({0, 1, 2}), \
        f"Unexpected training labels: {np.unique(y_train)}"
    assert set(np.unique(y_test)).issubset({0, 1, 2}), \
        f"Unexpected test labels: {np.unique(y_test)}"

    # Compute balanced class weights
    classes = np.array([0, 1, 2])
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = cw.astype(np.float32)

    return X_train, y_train, X_test, y_test, class_weights, y_train_k5, y_test_k5


def load_mid_prices(fold_idx, dataset_root, split='test'):
    """
    Load raw unscaled mid-prices directly from NoAuction_DecPre.
    Features: 0=AskPrice1, 2=BidPrice1.
    """
    category = 'NoAuction'
    norm_name = 'DecPre'
    prefix = '3'
    
    norm_dir = os.path.join(
        dataset_root, category,
        f"{prefix}.{category}_{norm_name}"
    )
    
    if split == 'train':
        dir_path = os.path.join(norm_dir, f"{category}_{norm_name}_Training")
        file_path = os.path.join(dir_path, f"Train_Dst_{category}_DecPre_CF_{fold_idx}.txt")
    else:
        dir_path = os.path.join(norm_dir, f"{category}_{norm_name}_Testing")
        file_path = os.path.join(dir_path, f"Test_Dst_{category}_DecPre_CF_{fold_idx}.txt")
        
    raw_data = np.loadtxt(file_path).T  # Transpose to (N_samples, 149)
    ask1 = raw_data[:, 0]
    bid1 = raw_data[:, 2]
    mid_price = (ask1 + bid1) / 2.0
    return mid_price

