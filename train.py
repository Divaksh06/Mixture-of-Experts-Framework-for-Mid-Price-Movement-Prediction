"""
Training Pipeline (Stages 1–3) for the MoE framework.

Stage 1: Train each expert independently on the training data.
Stage 2: Generate cross-validated stacked probabilities from experts.
Stage 3: Train the gating network on the stacked probabilities.

Currently configured to train on a single file (NoAuction_ZScore_CF_1).
All trained models and probabilities are saved to results/fold_{t}/.
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold

from data.load_fi2010 import load_fold
from experts.logistic_regression import LRExpert
from experts.xgboost_expert import XGBExpert
from experts.mlp_expert import MLPExpert
from moe.gating_network import GatingTrainer
from moe.mixture import get_stacked_probs


def train_fold(fold_idx, dataset_root, normalization='NoAuction_Zscore',
               results_dir='results'):
    """
    Execute Stages 1–3 for a single cross-validation fold.

    Parameters
    ----------
    fold_idx : int
        Fold index (1 through 9).
    dataset_root : str
        Root path to BenchmarkDatasets.
    normalization : str
        Normalization variant.
    results_dir : str
        Directory to save results.

    Returns
    -------
    fold_results : dict
        Dictionary containing trained models, expert metrics,
        and generated probabilities.
    """
    # ----------------------------------------------------------------
    # ORIGINAL: Multi-fold loading header
    # ----------------------------------------------------------------
    # print(f"\n{'='*60}")
    # print(f"  Fold {fold_idx}/9: Loading data...")
    # print(f"{'='*60}")
    # ----------------------------------------------------------------

    # SINGLE FILE: Loading header
    print(f"\n{'='*60}")
    print(f"  NoAuction_ZScore_CF_1: Loading data...")
    print(f"{'='*60}")

    # Load data
    X_train, y_train, X_test, y_test, class_weights = load_fold(
        fold_idx, dataset_root, normalization
    )
    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"  Class distribution (train): {np.bincount(y_train, minlength=3)}")
    print(f"  Class weights: {class_weights}")

    # Create output directory
    fold_dir = os.path.join(results_dir, f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)

    # ========== Stage 1: Independent Expert Training ==========
    # ----------------------------------------------------------------
    # ORIGINAL: print(f"\n  Fold {fold_idx}/9: Stage 1 — Training experts...")
    # ----------------------------------------------------------------
    print(f"\n  NoAuction_ZScore_CF_1: Stage 1 — Training experts...")

    # Train LR
    print(f"    Training Logistic Regression...")
    lr_expert = LRExpert()
    lr_expert.fit(X_train, y_train, class_weights)
    lr_metrics = lr_expert.evaluate(X_test, y_test)
    print(f"    LR — Accuracy: {lr_metrics['accuracy']:.4f}, "
          f"Macro-F1: {lr_metrics['macro_f1']:.4f}, Best C: {lr_expert.best_C}")

    # Train XGBoost
    print(f"    Training XGBoost...")
    xgb_expert = XGBExpert()
    xgb_expert.fit(X_train, y_train, class_weights)
    xgb_metrics = xgb_expert.evaluate(X_test, y_test)
    print(f"    XGB — Accuracy: {xgb_metrics['accuracy']:.4f}, "
          f"Macro-F1: {xgb_metrics['macro_f1']:.4f}, "
          f"Best params: {xgb_expert.best_params}")

    # Train MLP
    print(f"    Training MLP...")
    mlp_expert = MLPExpert()
    mlp_expert.fit(X_train, y_train, class_weights)
    mlp_metrics = mlp_expert.evaluate(X_test, y_test)
    print(f"    MLP — Accuracy: {mlp_metrics['accuracy']:.4f}, "
          f"Macro-F1: {mlp_metrics['macro_f1']:.4f}")

    # ========== Stage 2: Expert Probability Generation ==========
    # ----------------------------------------------------------------
    # ORIGINAL: print(f"\n  Fold {fold_idx}/9: Stage 2 — Generating stacked probabilities...")
    # ----------------------------------------------------------------
    print(f"\n  NoAuction_ZScore_CF_1: Stage 2 — Generating stacked probabilities...")

    # Generate cross-validated probabilities on training data
    # to avoid overfitting when training the gating network
    n_inner_folds = 3
    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)

    train_probs_lr = np.zeros((len(y_train), 3))
    train_probs_xgb = np.zeros((len(y_train), 3))
    train_probs_mlp = np.zeros((len(y_train), 3))

    for inner_fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr = y_train[tr_idx]

        # Compute inner class weights
        from sklearn.utils.class_weight import compute_class_weight
        inner_cw = compute_class_weight('balanced', classes=np.array([0, 1, 2]),
                                        y=y_tr).astype(np.float32)

        # LR
        lr_inner = LRExpert()
        lr_inner.fit(X_tr, y_tr, inner_cw)
        train_probs_lr[val_idx] = lr_inner.predict_proba(X_val)

        # XGB
        xgb_inner = XGBExpert()
        xgb_inner.fit(X_tr, y_tr, inner_cw)
        train_probs_xgb[val_idx] = xgb_inner.predict_proba(X_val)

        # MLP
        mlp_inner = MLPExpert()
        mlp_inner.fit(X_tr, y_tr, inner_cw)
        train_probs_mlp[val_idx] = mlp_inner.predict_proba(X_val)

    stacked_train = get_stacked_probs(train_probs_lr, train_probs_xgb, train_probs_mlp)

    # ========== Stage 3: Gating Network Training ==========
    # ----------------------------------------------------------------
    # ORIGINAL: print(f"\n  Fold {fold_idx}/9: Stage 3 — Training gating network...")
    # ----------------------------------------------------------------
    print(f"\n  NoAuction_ZScore_CF_1: Stage 3 — Training gating network...")

    gating_trainer = GatingTrainer()
    gating_trainer.fit(
        stacked_probs=stacked_train,
        expert_probs_list=[train_probs_lr, train_probs_xgb, train_probs_mlp],
        y=y_train,
        batch_size=256,
        epochs=30
    )

    # Generate test-set probabilities from full-data experts
    test_probs_lr = lr_expert.predict_proba(X_test)
    test_probs_xgb = xgb_expert.predict_proba(X_test)
    test_probs_mlp = mlp_expert.predict_proba(X_test)
    stacked_test = get_stacked_probs(test_probs_lr, test_probs_xgb, test_probs_mlp)

    # Get gating weights for test data
    test_weights = gating_trainer.get_weights(stacked_test)

    # Compute MoE final predictions on test
    from moe.mixture import compute_pfinal
    pfinal_test = compute_pfinal(test_weights, test_probs_lr, test_probs_xgb,
                                 test_probs_mlp)
    moe_preds = np.argmax(pfinal_test, axis=1)

    from sklearn.metrics import f1_score, accuracy_score
    moe_acc = accuracy_score(y_test, moe_preds)
    moe_f1 = f1_score(y_test, moe_preds, average='macro')
    print(f"    MoE — Accuracy: {moe_acc:.4f}, Macro-F1: {moe_f1:.4f}")

    # Save models and data
    fold_results = {
        'lr_expert': lr_expert,
        'xgb_expert': xgb_expert,
        'mlp_expert': mlp_expert,
        'gating_trainer': gating_trainer,
        'X_test': X_test,
        'y_test': y_test,
        'test_probs_lr': test_probs_lr,
        'test_probs_xgb': test_probs_xgb,
        'test_probs_mlp': test_probs_mlp,
        'pfinal_test': pfinal_test,
        'lr_metrics': lr_metrics,
        'xgb_metrics': xgb_metrics,
        'mlp_metrics': mlp_metrics,
        'moe_accuracy': moe_acc,
        'moe_macro_f1': moe_f1,
        'class_weights': class_weights,
    }

    # Save to disk
    save_path = os.path.join(fold_dir, 'fold_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(fold_results, f)
    print(f"  Results saved to {save_path}")

    return fold_results
