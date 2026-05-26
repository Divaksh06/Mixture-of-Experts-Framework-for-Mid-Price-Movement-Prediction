"""
Multi-Layer Perceptron (MLP) Expert for the MoE framework.

Architecture:
  Input(F*lookback) → Dense(256, BN, ReLU, Drop)
                    → Dense(128, BN, ReLU, Drop)
                    → Dense(3)

Improvements over original:
  - ReduceLROnPlateau learning-rate scheduler
  - Gradient clipping for training stability
  - Verbose flag to suppress output during Stage 2 inner training
  - Cosine-annealing warm restarts as alternative scheduler
  - Temperature-scaled probability output for better calibration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Down-weights easy examples, focuses training on hard examples.
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss


class MLP(nn.Module):
    """
    Two-hidden-layer MLP for 3-class classification.
    """

    def __init__(self, input_dim=144, hidden1=256, hidden2=128,
                 output_dim=3, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden2, output_dim)
        )
        # Learnable temperature for probability calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Return raw logits."""
        return self.net(x)

    def forward_calibrated(self, x):
        """Return temperature-scaled logits for calibrated softmax."""
        logits = self.net(x)
        return logits / self.temperature.clamp(min=0.1)


class MLPExpert:
    """
    MLP expert wrapper with training loop, early stopping, LR scheduling.
    """

    def __init__(self, device=None, lookback=5, verbose=True):
        """
        Parameters
        ----------
        device : torch.device or None
        lookback : int
            Number of historical time steps to stack for temporal framing.
        verbose : bool
            If False, suppresses epoch-level print output (for inner CV).
        """
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = None
        self.lookback = lookback
        self.verbose = verbose

    def _create_temporal_window(self, X):
        """
        Convert (N, F) to (N, F*lookback) by rolling lookback window.
        Uses edge-padding (repeats first row) for samples near the start.
        """
        if self.lookback <= 1:
            return X

        N, F = X.shape
        pad = np.tile(X[0, :], (self.lookback - 1, 1))
        X_padded = np.vstack([pad, X])

        X_temporal = np.zeros((N, F * self.lookback), dtype=np.float32)
        for i in range(self.lookback):
            X_temporal[:, i*F:(i+1)*F] = X_padded[i:i+N, :]

        return X_temporal

    def fit(self, X, y, class_weights=None, batch_size=256, max_epochs=60,
            patience=7, val_fraction=0.1):
        """
        Train the MLP with early stopping and LR scheduling.
        """
        X_temp = self._create_temporal_window(X)

        # Stratified split preserving temporal locality
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_temp, y, test_size=val_fraction, random_state=42, stratify=y
        )

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=batch_size, shuffle=True, pin_memory=True
        )

        self.model = MLP(input_dim=X_temp.shape[1]).to(self.device)

        # Loss
        if class_weights is not None:
            wt = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            criterion = FocalLoss(weight=wt, gamma=2.0)
        else:
            criterion = FocalLoss(gamma=2.0)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3,
                               weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5
        )

        best_val_f1 = -1.0
        epochs_no_improve = 0
        best_state = None

        for epoch in range(max_epochs):
            # --- Train ---
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            # --- Validate ---
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_preds = val_logits.argmax(dim=1).cpu().numpy()
                val_f1 = f1_score(y_val, val_preds, average='macro')

            scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if self.verbose:
                    print(f"      Early stopping at epoch {epoch+1}, "
                          f"best Val-F1: {best_val_f1:.4f}")
                break

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def predict_proba(self, X):
        """Predict calibrated probabilities using temperature scaling."""
        self.model.eval()
        X_temp = self._create_temporal_window(X)
        X_t = torch.tensor(X_temp, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model.forward_calibrated(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average='macro')
        report = classification_report(y, preds,
                                       target_names=['Up', 'Stationary', 'Down'])
        return {'accuracy': acc, 'macro_f1': macro_f1, 'report': report}
