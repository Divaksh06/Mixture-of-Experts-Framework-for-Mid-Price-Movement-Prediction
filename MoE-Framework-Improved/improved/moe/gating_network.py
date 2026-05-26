"""
Gating Network for the Mixture-of-Experts framework.

Architecture:  Input(9 + meta) → Dense(64, ReLU, Dropout) → Dense(3, Softmax)

Improvements over original:
  - Dropout regularisation in gating network
  - Early stopping on a validation split to prevent overfitting
  - Entropy of expert agreement as additional meta-feature (dim 10 or 12)
  - Class-weighted NLLLoss for better handling of class imbalance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class GatingNetwork(nn.Module):
    """
    Neural network gating module with dropout.
    """

    def __init__(self, input_dim=9, hidden_dim=64, num_experts=3,
                 dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=1)


def _compute_meta_features(stacked_probs):
    """
    Compute additional meta-features from stacked expert probabilities.

    Adds:
      - Max-class agreement across experts (3 values, one per class)
        = std of [p_LR(c), p_XGB(c), p_MLP(c)] for c in {0,1,2}
      → total: 9 (stacked) + 3 (agreement std) = 12

    Parameters
    ----------
    stacked_probs : np.ndarray, shape (N, 9)

    Returns
    -------
    enriched : np.ndarray, shape (N, 12)
    """
    N = stacked_probs.shape[0]
    # Reshape to (N, 3 experts, 3 classes)
    reshaped = stacked_probs.reshape(N, 3, 3)
    # Std across experts for each class → (N, 3)
    agreement_std = reshaped.std(axis=1)

    return np.concatenate([stacked_probs, agreement_std], axis=1).astype(np.float32)


class GatingTrainer:
    """
    Trainer for the gating network with early stopping and meta-features.
    """

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = None
        self.use_meta = True

    def fit(self, stacked_probs, expert_probs_list, y, batch_size=256,
            epochs=50, lr=1e-3, patience=8):
        """
        Train the gating network with early stopping.
        """
        # Compute enriched input
        if self.use_meta:
            gating_input = _compute_meta_features(stacked_probs)
        else:
            gating_input = stacked_probs

        input_dim = gating_input.shape[1]
        self.model = GatingNetwork(input_dim=input_dim).to(self.device)

        # Train/val split for early stopping
        (gi_tr, gi_val, y_tr, y_val,
         p0_tr, p0_val, p1_tr, p1_val, p2_tr, p2_val) = train_test_split(
            gating_input, y,
            expert_probs_list[0], expert_probs_list[1], expert_probs_list[2],
            test_size=0.15, random_state=42, stratify=y
        )

        # Tensors
        gi_tr_t = torch.tensor(gi_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        p0_tr_t = torch.tensor(p0_tr, dtype=torch.float32)
        p1_tr_t = torch.tensor(p1_tr, dtype=torch.float32)
        p2_tr_t = torch.tensor(p2_tr, dtype=torch.float32)

        gi_val_t = torch.tensor(gi_val, dtype=torch.float32).to(self.device)
        p0_val_t = torch.tensor(p0_val, dtype=torch.float32).to(self.device)
        p1_val_t = torch.tensor(p1_val, dtype=torch.float32).to(self.device)
        p2_val_t = torch.tensor(p2_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_ds = TensorDataset(gi_tr_t, p0_tr_t, p1_tr_t, p2_tr_t, y_tr_t)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr,
                               weight_decay=1e-4)

        best_val_f1 = -1.0
        epochs_no_improve = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            for batch_gi, batch_p0, batch_p1, batch_p2, batch_y in loader:
                batch_gi = batch_gi.to(self.device)
                batch_p0 = batch_p0.to(self.device)
                batch_p1 = batch_p1.to(self.device)
                batch_p2 = batch_p2.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                weights = self.model(batch_gi)   # (B, 3)

                p_final = (weights[:, 0:1] * batch_p0 +
                           weights[:, 1:2] * batch_p1 +
                           weights[:, 2:3] * batch_p2)

                log_p = torch.log(p_final + 1e-10)
                loss = criterion(log_p, batch_y)
                loss.backward()
                optimizer.step()

            # --- Validation ---
            self.model.eval()
            with torch.no_grad():
                w_val = self.model(gi_val_t)
                pf_val = (w_val[:, 0:1] * p0_val_t +
                          w_val[:, 1:2] * p1_val_t +
                          w_val[:, 2:3] * p2_val_t)
                val_preds = pf_val.argmax(dim=1).cpu().numpy()
                val_f1 = f1_score(y_val, val_preds, average='macro')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def get_weights(self, stacked_probs):
        """
        Get expert weights for given stacked probabilities.
        """
        self.model.eval()
        if self.use_meta:
            gating_input = _compute_meta_features(stacked_probs)
        else:
            gating_input = stacked_probs

        gi_t = torch.tensor(gating_input, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            weights = self.model(gi_t).cpu().numpy()
        return weights
