"""
XGBoost Expert for the MoE framework.

Uses gradient-boosted decision trees with softmax objective for
three-class classification. Hyperparameters max_depth and learning_rate
are tuned via 3-fold stratified cross-validation on training data.
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report


class XGBExpert:
    """
    XGBoost expert wrapper with hyperparameter tuning.

    Attributes
    ----------
    model : xgb.XGBClassifier or None
        The fitted XGBoost model.
    best_params : dict or None
        The best hyperparameters found during tuning.
    """

    def __init__(self):
        """Initialize the XGBoost expert."""
        self.model = None
        self.best_params = None

    def fit(self, X, y, class_weights=None):
        """
        Train the XGBoost model with hyperparameter search.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 144)
            Training feature matrix.
        y : np.ndarray, shape (n_samples,)
            Training labels in {0, 1, 2}.
        class_weights : np.ndarray, shape (3,) or None
            Balanced class weights used to compute sample weights.
        """
        # Compute per-sample weights from class weights
        if class_weights is not None:
            sample_weights = np.array([class_weights[int(label)] for label in y],
                                      dtype=np.float32)
        else:
            sample_weights = None

        max_depth_candidates = [3, 5, 7]
        lr_candidates = [0.05, 0.1, 0.2]

        best_score = -1.0
        best_params = {'max_depth': 5, 'learning_rate': 0.1}

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for md in max_depth_candidates:
            for lr in lr_candidates:
                scores = []
                for train_idx, val_idx in skf.split(X, y):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    if sample_weights is not None:
                        sw_tr = sample_weights[train_idx]
                    else:
                        sw_tr = None

                    clf = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=md,
                        learning_rate=lr,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='multi:softprob',
                        num_class=3,
                        eval_metric='mlogloss',
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0
                    )
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

        # Retrain on full training data with best params
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        self.model.fit(X, y, sample_weight=sample_weights, verbose=False)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 144)

        Returns
        -------
        probs : np.ndarray, shape (n_samples, 3)
        """
        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 144)

        Returns
        -------
        preds : np.ndarray, shape (n_samples,)
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model on given data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 144)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        metrics : dict
            Dictionary with 'accuracy', 'macro_f1', and 'report'.
        """
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average='macro')
        report = classification_report(y, preds, target_names=['Up', 'Stationary', 'Down'])
        return {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'report': report
        }
