"""
Multi-Layer Perceptron (MLP) Expert for the MoE framework.

Implements a two-hidden-layer feedforward network in PyTorch:
  Input(735) -> Dense(256, ReLU) -> Dropout(0.3)
             -> Dense(128, ReLU) -> Dropout(0.3)
             -> Dense(3, Softmax)

Trained with Adam optimizer, cross-entropy loss with class weights,
batch size 256, up to 50 epochs with early stopping on validation macro-F1.
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
    Focuses training on hard examples by down-weighting easy examples.
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MLP(nn.Module):
    """
    Two-hidden-layer MLP for 3-class classification.

    Architecture:
        Input(735) -> Linear(256) -> ReLU -> Dropout(0.3)
                   -> Linear(128) -> ReLU -> Dropout(0.3)
                   -> Linear(3)
    """

    def __init__(self, input_dim=735, hidden1=256, hidden2=128, output_dim=3,
                 dropout_rate=0.3):
        """
        Initialize the MLP.

        Parameters
        ----------
        input_dim : int
            Dimension of input features.
        hidden1 : int
            Number of units in first hidden layer.
        hidden2 : int
            Number of units in second hidden layer.
        output_dim : int
            Number of output classes.
        dropout_rate : float
            Dropout probability.
        """
        super(MLP, self).__init__()
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

    def forward(self, x):
        """Forward pass (returns raw logits, softmax applied externally)."""
        return self.net(x)


class MLPExpert:
    """
    MLP expert wrapper with training loop and early stopping.

    Attributes
    ----------
    model : MLP or None
        The trained PyTorch MLP model.
    device : torch.device
        Device to use for training and inference.
    """

    def __init__(self, device=None, lookback=5):
        """
        Initialize the MLP expert.

        Parameters
        ----------
        device : torch.device or None
            If None, auto-detects GPU/CPU.
        lookback : int
            Number of historical steps to stack (temporal framing).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = None
        self.lookback = lookback

    def _create_temporal_window(self, X):
        """
        Convert (N, F) to (N, F*lookback) by rolling lookback window.
        Oldest timestep first.
        """
        if self.lookback <= 1:
            return X
            
        N, F = X.shape
        pad = np.tile(X[0, :], (self.lookback - 1, 1))
        X_padded = np.vstack([pad, X])
        
        X_temporal = np.zeros((N, F * self.lookback), dtype=np.float32)
        for i in range(self.lookback):
            # i=0: t-4, i=1: t-3 ... i=4: t
            X_temporal[:, i*F:(i+1)*F] = X_padded[i:i+N, :]
            
        return X_temporal

    def fit(self, X, y, class_weights=None, batch_size=256, max_epochs=50,
            patience=5, val_fraction=0.1):
        """
        Train the MLP model with early stopping.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 144)
            Training features.
        y : np.ndarray, shape (n_samples,)
            Training labels in {0, 1, 2}.
        class_weights : np.ndarray, shape (3,) or None
            Balanced class weights for the loss function.
        batch_size : int
            Training batch size.
        max_epochs : int
            Maximum number of training epochs.
        patience : int
            Early stopping patience (epochs without improvement).
        val_fraction : float
            Fraction of training data used for validation.
        """
        # Apply Temporal Lookback directly inside MLP expert
        X_temp = self._create_temporal_window(X)

        # Split into train and validation for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_temp, y, test_size=val_fraction, random_state=42, stratify=y
        )

        # Convert to tensors
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Build model with potentially expanded dimension
        self.model = MLP(input_dim=X_temp.shape[1]).to(self.device)

        # Loss with class weights and Focal Loss
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            criterion = FocalLoss(weight=weight_tensor, gamma=3.0)
        else:
            criterion = FocalLoss(gamma=3.0)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        best_val_f1 = -1.0
        epochs_no_improve = 0
        best_state = None

        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t.to(self.device))
                val_preds = val_logits.argmax(dim=1).cpu().numpy()
                val_f1 = f1_score(y_val, val_preds, average='macro')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"      Epoch {epoch+1:02d}/{max_epochs} - Val F1: {val_f1:.4f} (Improved)")
            else:
                epochs_no_improve += 1
                print(f"      Epoch {epoch+1:02d}/{max_epochs} - Val F1: {val_f1:.4f}")

            if epochs_no_improve >= patience:
                print(f"      Early stopping triggered at epoch {epoch+1:02d}.")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def predict_proba(self, X):
        """
        Predict class probabilities using softmax.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 144)

        Returns
        -------
        probs : np.ndarray, shape (n_samples, 3)
        """
        self.model.eval()
        X_temp = self._create_temporal_window(X)
        X_t = torch.tensor(X_temp, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

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
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

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
