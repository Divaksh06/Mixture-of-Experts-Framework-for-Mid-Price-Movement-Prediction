"""
Logistic Regression Expert for the MoE framework.

Uses multinomial logistic regression with L2 regularization.
Hyperparameter C is selected via 3-fold stratified cross-validation
on the training data, optimizing macro-averaged F1.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report


class LRExpert:
    """
    Logistic Regression expert wrapper with hyperparameter tuning.

    Attributes
    ----------
    model : LogisticRegression or None
        The fitted sklearn model.
    best_C : float or None
        The best regularization parameter found during tuning.
    """

    def __init__(self):
        """Initialize the LR expert."""
        self.model = None
        self.best_C = None

    def fit(self, X, y, class_weights=None):
        """
        Train the Logistic Regression model with hyperparameter search.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 144)
            Training feature matrix.
        y : np.ndarray, shape (n_samples,)
            Training labels in {0, 1, 2}.
        class_weights : np.ndarray, shape (3,) or None
            Balanced class weights. If provided, converted to a dict
            for sklearn's class_weight parameter.
        """
        C_candidates = [0.01, 0.1, 1.0, 10.0]
        best_score = -1.0
        best_C = 1.0

        # Build class_weight dict for sklearn
        if class_weights is not None:
            cw_dict = {i: float(class_weights[i]) for i in range(len(class_weights))}
        else:
            cw_dict = 'balanced'

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for C in C_candidates:
            lr = LogisticRegression(
                C=C,
                solver='lbfgs',
                max_iter=1000,
                class_weight=cw_dict,
                random_state=42,
                n_jobs=-1
            )
            scores = cross_val_score(
                lr, X, y, cv=skf, scoring='f1_macro', n_jobs=-1
            )
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_C = C

        self.best_C = best_C

        # Retrain on full training data with best C
        self.model = LogisticRegression(
            C=self.best_C,
            solver='lbfgs',
            max_iter=1000,
            class_weight=cw_dict,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 144)
            Feature matrix.

        Returns
        -------
        probs : np.ndarray, shape (n_samples, 3)
            Predicted class probabilities over {0, 1, 2}.
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
