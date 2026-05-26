"""
Logistic Regression Expert for the MoE framework.

Uses multinomial logistic regression with L2 regularisation.
Hyperparameter C is selected via 3-fold stratified CV optimising macro-F1.

Improvements over original:
  - Expanded C search grid (added 0.001 and 100.0)
  - Post-hoc probability calibration via CalibratedClassifierCV (isotonic)
  - n_jobs=-1 for parallel CV scoring
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score, classification_report


class LRExpert:
    """
    Logistic Regression expert with hyperparameter tuning and calibration.
    """

    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.best_C = None

    def fit(self, X, y, class_weights=None):
        """
        Train with CV hyperparameter search, then calibrate probabilities.
        """
        C_candidates = [0.001, 0.01, 0.1, 1.0, 10.0]
        best_score = -1.0
        best_C = 1.0

        if class_weights is not None:
            cw_dict = {i: float(class_weights[i]) for i in range(len(class_weights))}
        else:
            cw_dict = 'balanced'

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for C in C_candidates:
            lr = LogisticRegression(
                C=C, solver='lbfgs', max_iter=1000,
                class_weight=cw_dict, random_state=42, n_jobs=None
            )
            scores = cross_val_score(
                lr, X, y, cv=skf, scoring='f1_macro', n_jobs=-1
            )
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_C = C

        self.best_C = best_C

        # Train on full data
        self.model = LogisticRegression(
            C=self.best_C, solver='lbfgs', max_iter=1000,
            class_weight=cw_dict, random_state=42, n_jobs=None
        )
        self.model.fit(X, y)

        # Probability calibration via isotonic regression (3-fold)
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
