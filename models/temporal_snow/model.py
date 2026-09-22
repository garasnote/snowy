"""
GRU-based temporal snow refinement model.

Takes a sequence of per-timestep feature vectors (snow predictions + meteo + time)
and outputs refined per-timestep snow probability.

Architecture:
    Input: (batch, seq_len, n_features) → GRU → Linear → sigmoid
    Output: (batch, seq_len, 1) per-timestep snow probability
"""

import torch
import torch.nn as nn


class TemporalSnowGRU(nn.Module):
    """GRU-based temporal refinement for snow presence detection."""

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, seq_len, 1)
        """
        out, _ = self.gru(x)        # (batch, seq_len, hidden)
        logits = self.head(out)      # (batch, seq_len, 1)
        return logits


class TemporalSnowLSTM(nn.Module):
    """LSTM variant (alternative to GRU)."""

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)
