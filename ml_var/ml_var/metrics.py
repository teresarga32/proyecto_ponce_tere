"""This module includes functions to compute metrics for our model training."""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from ml_var.model import compute_var_from_mdn
from ml_var.backtesting_metrics import var_backtesting

def mdn_loss(y, pi, mu, sigma, eps=1e-6):
    """La loss que usamos en nuestro entrenamiento.
    """
    y = y.unsqueeze(1).expand_as(mu)
    sigma = torch.clamp(sigma, min=eps, max=1e3)
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(y)
    log_prob = log_prob + torch.log(pi + eps)
    loss = -torch.logsumexp(log_prob, dim=1).mean()
    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: NaN or Inf loss encountered.")
    return loss


def evaluate_model_with_backtesting(model, X, y, batch_size=32, alpha=0.01):
    """Evaluamos nuestro modelo con referencia a la loss y a las backtesting metrics."""
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_losses = []
        all_var = []
        all_y_pred = []
        all_y_true = []
        for batch_X, batch_y in loader:
            pi, mu, sigma, _ = model(batch_X)
            loss = mdn_loss(batch_y, pi, mu, sigma)
            all_losses.append(loss.item())
            y_pred = (pi * mu).sum(dim=1)
            
            all_y_pred.append(y_pred.cpu().numpy())
            all_y_true.append(batch_y.cpu().numpy())
            
            # Compute VaR for each sample in batch
            var_batch = compute_var_from_mdn(pi, mu, sigma, alpha=alpha)
            all_var.extend(var_batch)
            
        all_y_pred = np.concatenate(all_y_pred)
        all_y_true = np.concatenate(all_y_true)
        mean_mae = np.mean(np.abs(all_y_true - all_y_pred))
        mean_mse = np.mean((all_y_true - all_y_pred) ** 2)
            
        mean_loss = np.mean(all_losses)
        std_loss = np.std(all_losses)

        # Backtesting metrics
        y_np = y.squeeze() if isinstance(y, np.ndarray) else y.numpy().squeeze()
        all_var = np.array(all_var)
        
        backtest = var_backtesting(y_np, all_var, alpha=alpha)
    return mean_loss, std_loss, mean_mae, mean_mse, all_var, backtest
