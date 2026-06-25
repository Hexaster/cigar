"""Training helpers for CIGAR."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import regression_metrics
from .reporting import format_physical_params


def to_same_shape(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if targets.ndim == 2 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)
    if predictions.ndim == 2 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)
    return predictions, targets


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        y_pred = model(batch)
        y_true = batch["target"]
        y_pred, y_true = to_same_shape(y_pred, y_true)
        preds.extend(y_pred.detach().cpu().view(-1).numpy())
        targets.extend(y_true.detach().cpu().view(-1).numpy())
    return np.asarray(preds), np.asarray(targets)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 2000,
    lr: float = 1e-3,
    patience: int = 30,
    min_delta: float = 0.0,
    verbose: bool = False,
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_weight = 0
        for batch in train_loader:
            optimizer.zero_grad()
            predictions = model(batch)
            targets = batch["target"]
            predictions, targets = to_same_shape(predictions, targets)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(targets.numel())
            train_weight += int(targets.numel())

        val_predictions, val_targets = collect_predictions(model, val_loader)
        val_loss = regression_metrics(val_targets, val_predictions)["MSE"]
        if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0):
            metrics = regression_metrics(val_targets, val_predictions)
            train_loss = train_loss_sum / max(train_weight, 1)
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                f"Test Loss: {val_loss:.4f}, MAE: {metrics['MAE']:.4f}, "
                f"MAPE: {metrics['MAPE'] * 100:.2f}%, R2: {metrics['R2']:.4f}"
            )
            params = format_physical_params(model)
            if params:
                print(params)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}. Best Test Loss: {best_val_loss:.4f}")
                break

    model.load_state_dict(best_state)
    return model
