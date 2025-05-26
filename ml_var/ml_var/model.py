"""Modulo conteniendo nuestro modelo."""

import torch
import torch.nn as nn
import numpy as np


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    def forward(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden_size]
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)  # [batch, seq_len]
        # Weighted sum of LSTM outputs
        context = (attn_weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # [batch, hidden_size]
        return context, attn_weights


class VaRModel(nn.Module):
    def __init__(self, input_size, num_lstm_layers, hidden_size, mdn_size, n_components, dropout, bidirectional_lstm):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            num_layers=num_lstm_layers, 
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional_lstm,
        )
        lstm_output_size = hidden_size * (2 if bidirectional_lstm else 1)
        self.attention = Attention(lstm_output_size)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, mdn_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mdn = nn.ModuleDict({
            'pi': nn.Linear(mdn_size, n_components),
            'mu': nn.Linear(mdn_size, n_components),
            'sigma': nn.Linear(mdn_size, n_components)
        })
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_output_size]
        context, attn_weights = self.attention(lstm_out)  # [batch, lstm_output_size], [batch, seq_len]
        features = self.fc(context)
        pi = torch.softmax(self.mdn['pi'](features), dim=-1)
        mu = self.mdn['mu'](features)
        sigma = torch.exp(self.mdn['sigma'](features))
        sigma = torch.clamp(sigma, min=1e-3, max=1e3)
        return pi, mu, sigma, attn_weights  # Optionally return attn_weights for visualization


def compute_var_from_mdn(pi, mu, sigma, alpha=0.01, n_samples=5000):
    """
    Estimate the VaR at level alpha (95/99 VAR) from the MDN output.
    Returns an array of VaR estimates, one per sample in the batch.
    """
    pi_np = pi.detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()
    sigma_np = sigma.detach().cpu().numpy()
    batch_size, n_components = pi_np.shape
    var_estimates = []
    for i in range(batch_size):
        # Sample which component to use for each sample
        component = np.random.choice(n_components, size=n_samples, p=pi_np[i])
        samples = np.random.normal(mu_np[i, component], sigma_np[i, component])
        var = np.quantile(samples, alpha)
        var_estimates.append(var)
    return np.array(var_estimates)
