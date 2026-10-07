"""
Training utilities for the residual CO2 Transformer.

The Transformer predicts a scale-normalized correction to the causal
persistence forecast.

This module handles:
- reproducible random seeds,
- PyTorch DataLoader construction,
- residual MSE loss,
- one training epoch,
- loss evaluation,
- prediction of scaled residual corrections.

Model-selection logic such as inner-run early stopping is intentionally
kept separate. The outer leave-one-run-out validation run must not be
used to select the stopping epoch.
"""

import random

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def set_seed(seed=42):
    """
    Set random seeds for reproducible model initialization,
    mini-batch ordering, and training.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    X,
    y,
    batch_size=32,
    shuffle=False,
    seed=42,
):
    """
    Convert NumPy input and target arrays into a PyTorch DataLoader.

    Parameters
    ----------
    X : np.ndarray
        Model inputs with shape
        (n_samples, context_steps, n_features).

    y : np.ndarray
        Scale-normalized residual targets with shape
        (n_samples, 6).

    batch_size : int
        Number of samples per mini-batch.

    shuffle : bool
        Whether to shuffle samples between epochs.

    seed : int
        Random seed used by the DataLoader generator.

    Returns
    -------
    torch.utils.data.DataLoader
    """

    X = np.asarray(
        X,
        dtype="float32",
    )

    y = np.asarray(
        y,
        dtype="float32",
    )

    if X.ndim != 3:
        raise ValueError(
            "X must have shape "
            "(n_samples, context_steps, n_features)."
        )

    if y.ndim != 2:
        raise ValueError(
            "y must have shape (n_samples, n_targets)."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values found in X."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "Non-finite values found in y."
        )

    X_tensor = torch.from_numpy(
        X
    )

    y_tensor = torch.from_numpy(
        y
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )

    return loader


def residual_mse(
    prediction,
    target,
):
    """
    Mean squared error on RMS-scaled residual targets.

    Each of the six target residuals has already been divided by
    its own training-fold RMS scale. Ordinary MSE therefore gives
    the six sampling points comparable influence during training.

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted scaled residuals, shape (batch, 6).

    target : torch.Tensor
        True scaled residuals, shape (batch, 6).

    Returns
    -------
    torch.Tensor
        Scalar mean squared error.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes must match. "
            f"Got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}."
        )

    return torch.mean(
        (prediction - target) ** 2
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    gradient_clip=1.0,
):
    """
    Train the model for one complete epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Residual CO2 Transformer.

    loader : DataLoader
        Training DataLoader.

    optimizer : torch.optim.Optimizer
        PyTorch optimizer.

    device : torch.device
        Device used for training.

    gradient_clip : float or None
        Maximum global gradient norm. Use None to disable clipping.

    Returns
    -------
    float
        Sample-weighted mean training loss for the epoch.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            X_batch
        )

        loss = residual_mse(
            prediction,
            y_batch,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss encountered."
            )

        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            )

        optimizer.step()

        batch_samples = X_batch.size(
            0
        )

        total_loss += (
            float(loss.item())
            * batch_samples
        )

        total_samples += batch_samples

    if total_samples == 0:
        raise RuntimeError(
            "Training DataLoader contained no samples."
        )

    return (
        total_loss
        / total_samples
    )


@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    device,
):
    """
    Evaluate residual MSE without updating model parameters.

    Returns
    -------
    float
        Sample-weighted mean loss.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )

        prediction = model(
            X_batch
        )

        loss = residual_mse(
            prediction,
            y_batch,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite evaluation loss encountered."
            )

        batch_samples = X_batch.size(
            0
        )

        total_loss += (
            float(loss.item())
            * batch_samples
        )

        total_samples += batch_samples

    if total_samples == 0:
        raise RuntimeError(
            "Evaluation DataLoader contained no samples."
        )

    return (
        total_loss
        / total_samples
    )


@torch.no_grad()
def predict_scaled_correction(
    model,
    X,
    device,
    batch_size=64,
):
    """
    Predict scale-normalized residual corrections.

    Parameters
    ----------
    model : torch.nn.Module
        Trained residual Transformer.

    X : np.ndarray
        Model inputs with shape
        (n_samples, context_steps, n_features).

    device : torch.device
        Device used for inference.

    batch_size : int
        Inference batch size.

    Returns
    -------
    np.ndarray
        Scaled residual predictions with shape (n_samples, 6).
    """

    X = np.asarray(
        X,
        dtype="float32",
    )

    if X.ndim != 3:
        raise ValueError(
            "X must have shape "
            "(n_samples, context_steps, n_features)."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values found in prediction inputs."
        )

    X_tensor = torch.from_numpy(
        X
    )

    loader = DataLoader(
        X_tensor,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.eval()

    predictions = []

    for X_batch in loader:

        X_batch = X_batch.to(
            device
        )

        prediction = model(
            X_batch
        )

        predictions.append(
            prediction
            .detach()
            .cpu()
            .numpy()
        )

    if len(predictions) == 0:
        return np.empty(
            (0, 6),
            dtype="float32",
        )

    return np.concatenate(
        predictions,
        axis=0,
    )