"""
Gating Network for the Mixture-of-Experts framework.

A lightweight two-layer neural network that takes the 9-dimensional
concatenation of three expert probability outputs and produces a
3-dimensional weight vector (w1, w2, w3) via softmax, determining
each expert's contribution to the final prediction.

Architecture: Input(9) -> Dense(64, ReLU) -> Dense(3, Softmax)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class GatingNetwork(nn.Module):
    """
    Neural network gating module.

    Takes the stacked expert probabilities (9-dim) and outputs
    expert combination weights (3-dim softmax).
    """

    def __init__(self, input_dim=9, hidden_dim=64, num_experts=3):
        """
        Initialize the gating network.

        Parameters
        ----------
        input_dim : int
            Dimension of stacked expert probabilities.
        hidden_dim : int
            Number of hidden units.
        num_experts : int
            Number of experts (output dimension).
        """
        super(GatingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 9)

        Returns
        -------
        weights : torch.Tensor, shape (batch, 3)
            Softmax-normalized expert weights.
        """
        logits = self.net(x)
        return torch.softmax(logits, dim=1)


class GatingTrainer:
    """
    Trainer for the gating network.

    Trains the gating network to minimize cross-entropy loss between
    the gating-weighted expert combination and the true labels.

    Attributes
    ----------
    model : GatingNetwork
        The gating network.
    device : torch.device
        Computation device.
    """

    def __init__(self, device=None):
        """
        Initialize the gating trainer.

        Parameters
        ----------
        device : torch.device or None
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = None

    def fit(self, stacked_probs, expert_probs_list, y, batch_size=256,
            epochs=30, lr=1e-3):
        """
        Train the gating network.

        The gating network learns weights w = g(stacked_probs) such that
        P_final = w1*P_LR + w2*P_XGB + w3*P_MLP minimizes cross-entropy.

        Parameters
        ----------
        stacked_probs : np.ndarray, shape (n_samples, 9)
            Concatenated expert probability vectors.
        expert_probs_list : list of np.ndarray
            List of 3 arrays, each shape (n_samples, 3), being the
            individual expert probability outputs.
        y : np.ndarray, shape (n_samples,)
            True labels in {0, 1, 2}.
        batch_size : int
            Training batch size.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        """
        self.model = GatingNetwork(input_dim=stacked_probs.shape[1]).to(self.device)

        # Convert to tensors
        stacked_t = torch.tensor(stacked_probs, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        # Expert probs as tensors
        expert_tensors = [
            torch.tensor(p, dtype=torch.float32) for p in expert_probs_list
        ]

        dataset = TensorDataset(
            stacked_t, expert_tensors[0], expert_tensors[1], expert_tensors[2], y_t
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for batch_stacked, batch_p1, batch_p2, batch_p3, batch_y in loader:
                batch_stacked = batch_stacked.to(self.device)
                batch_p1 = batch_p1.to(self.device)
                batch_p2 = batch_p2.to(self.device)
                batch_p3 = batch_p3.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Get gating weights: (batch, 3)
                weights = self.model(batch_stacked)

                # Compute weighted combination: P_final = sum(wi * Pi)
                p_final = (
                    weights[:, 0:1] * batch_p1 +
                    weights[:, 1:2] * batch_p2 +
                    weights[:, 2:3] * batch_p3
                )

                # Cross-entropy loss on the combined probabilities
                # Since p_final is already probabilities, use log for NLLLoss
                log_p = torch.log(p_final + 1e-10)
                loss = nn.NLLLoss()(log_p, batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

    def get_weights(self, stacked_probs):
        """
        Get expert weights for given stacked probabilities.

        Parameters
        ----------
        stacked_probs : np.ndarray, shape (n_samples, 9)

        Returns
        -------
        weights : np.ndarray, shape (n_samples, 3)
            Expert combination weights per sample.
        """
        self.model.eval()
        stacked_t = torch.tensor(stacked_probs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            weights = self.model(stacked_t).cpu().numpy()
        return weights
