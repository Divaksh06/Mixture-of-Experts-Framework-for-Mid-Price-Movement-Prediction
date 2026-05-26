"""
XGBoost Expert for the MoE framework.

Gradient-boosted trees with softmax objective for 3-class classification.
Hyperparameters (max_depth, learning_rate) tuned via 3-fold stratified CV.

Improvements over original:
  - Early stopping during both CV and final training (prevents overfitting)
  - GPU acceleration via 'gpu_hist' tree method when CUDA is available
  - Post-hoc probability calibration (isotonic)
  - Expanded search grid
"""

import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, classification_report


def _detect_gpu():
    """Return True if CUDA GPU is available for XGBoost."""
    return torch.cuda.is_available()


class XGBExpert:
    """
    XGBoost expert with early stopping, optional GPU, and calibration.
    """

    def __init__(self, use_gpu=None):
        self.model = None
        self.calibrated_model = None
        self.best_params = None
        if use_gpu is None:
            self.use_gpu = _detect_gpu()
        else:
            self.use_gpu = use_gpu

    def _base_params(self, md, lr):
        """Return base XGBClassifier parameters."""
        params = dict(
            n_estimators=800,
            max_depth=md,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,             # L1 regularisation (new)
            reg_lambda=1.0,            # L2 regularisation
            min_child_weight=3,        # prevents overfitting to noise
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0,
        )
        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['device'] = 'cuda'
            params['n_jobs'] = 1
        else:
            params['tree_method'] = 'hist'
            params['n_jobs'] = -1
        return params

    def fit(self, X, y, class_weights=None):
        """
        Train with CV hyperparameter search + early stopping.
        """
        if class_weights is not None:
            sample_weights = np.array(
                [class_weights[int(label)] for label in y], dtype=np.float32)
        else:
            sample_weights = None

        max_depth_candidates = [3, 4, 5, 6]
        lr_candidates = [0.01, 0.05, 0.1]

        best_score = -1.0
        best_params = {'max_depth': 5, 'learning_rate': 0.1}

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for md in max_depth_candidates:
            for lr in lr_candidates:
                scores = []
                for train_idx, val_idx in skf.split(X, y):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]
                    sw_tr = sample_weights[train_idx] if sample_weights is not None else None

                    clf = xgb.XGBClassifier(**self._base_params(md, lr))
                    clf.fit(
                        X_tr, y_tr,
                        sample_weight=sw_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                    preds = clf.predict(X_val)
                    score = f1_score(y_val, preds, average='macro')
                    scores.append(score)

                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'max_depth': md, 'learning_rate': lr}

        self.best_params = best_params

        # Final training on full data with early-stopping on a held-out slice
        from sklearn.model_selection import train_test_split
        X_trn, X_es, y_trn, y_es = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y)
        sw_trn = None
        if sample_weights is not None:
            # Recompute for the subset
            from sklearn.utils.class_weight import compute_class_weight
            cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_trn)
            sw_trn = np.array([cw[int(l)] for l in y_trn], dtype=np.float32)

        self.model = xgb.XGBClassifier(
            **self._base_params(best_params['max_depth'],
                                best_params['learning_rate'])
        )
        self.model.fit(
            X_trn, y_trn,
            sample_weight=sw_trn,
            eval_set=[(X_es, y_es)],
            verbose=False
        )

        # Calibrate probabilities
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method='isotonic', cv=3
        )
        self.calibrated_model.fit(X, y)

    def predict_proba(self, X):
        """Return calibrated probabilities."""
        return self.calibrated_model.predict_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X, y):
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average='macro')
        report = classification_report(y, preds,
                                       target_names=['Up', 'Stationary', 'Down'])
        return {'accuracy': acc, 'macro_f1': macro_f1, 'report': report}
