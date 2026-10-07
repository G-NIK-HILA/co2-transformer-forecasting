"""
Learned baseline models for the CO2 forecasting benchmark.

This module implements a multi-output Ridge baseline using the same
leave-one-run-out development folds and causal forecasting windows used
for the Transformer benchmark.
"""

import numpy as np

from sklearn.linear_model import Ridge

from src import data as D


def flatten_windows(X):
    """
    Flatten time-series windows from (N, L, F) to (N, L * F).

    Ridge has no temporal architecture, so each lag-feature combination
    becomes an independent linear predictor.
    """
    X = np.asarray(
        X,
        dtype="float32",
    )

    if X.ndim != 3:
        raise ValueError(
            "X must have shape (n_samples, context_steps, n_features)."
        )

    return X.reshape(
        X.shape[0],
        -1,
    )


def build_per_run_datasets(
    runs,
    development_runs,
    context_steps,
    horizon_steps,
    reference_context=18,
):
    """
    Build forecasting windows independently for every development run.
    """
    per_run = {}

    for run_name in development_runs:

        (
            X_run,
            y_run,
            sample_run_names,
            target_rows,
        ) = D.build_forecast_dataset(
            runs=runs,
            run_names=[run_name],
            context_steps=context_steps,
            horizon_steps=horizon_steps,
            reference_context=reference_context,
        )

        per_run[run_name] = {
            "X": np.asarray(
                X_run,
                dtype="float64",
            ),
            "y": np.asarray(
                y_run,
                dtype="float64",
            ),
            "run_names": sample_run_names,
            "target_rows": target_rows,
        }

    return per_run


def fit_window_feature_scaler(
    X_train,
):
    """
    Fit feature scaling using training windows only.

    The scaler is estimated independently for every lag-feature position
    after flattening. NaNs correspond only to intentionally unseen analyzer
    history.

    Scaling is applied before Ridge fitting. Undefined analyzer-history
    entries become zero after standardization, representing the corresponding
    training mean.
    """
    X_flat = flatten_windows(
        X_train
    ).astype(
        "float64"
    )

    mean = np.nanmean(
        X_flat,
        axis=0,
    )

    std = np.nanstd(
        X_flat,
        axis=0,
        ddof=0,
    )

    mean = np.where(
        np.isfinite(mean),
        mean,
        0.0,
    )

    std = np.where(
        (
            ~np.isfinite(std)
        )
        | (
            std < 1e-8
        ),
        1.0,
        std,
    )

    return {
        "mean": mean,
        "std": std,
    }


def transform_window_features(
    X,
    scaler,
):
    """
    Flatten and standardize Ridge inputs.

    Intentional NaNs from unseen analyzer history are replaced with zero only
    after standardization.
    """
    X_flat = flatten_windows(
        X
    ).astype(
        "float64"
    )

    X_scaled = (
        X_flat
        - scaler["mean"][None, :]
    ) / scaler["std"][None, :]

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X_scaled.astype(
        "float32"
    )


def fit_target_scaler(
    y_train,
):
    """
    Fit independent training-only standardization for the six CO2 targets.
    """
    y_train = np.asarray(
        y_train,
        dtype="float64",
    )

    mean = np.mean(
        y_train,
        axis=0,
    )

    std = np.std(
        y_train,
        axis=0,
        ddof=0,
    )

    std = np.where(
        (
            ~np.isfinite(std)
        )
        | (
            std < 1e-8
        ),
        1.0,
        std,
    )

    return {
        "mean": mean,
        "std": std,
    }


def transform_targets(
    y,
    scaler,
):
    """
    Standardize six-dimensional CO2 targets.
    """
    y = np.asarray(
        y,
        dtype="float64",
    )

    return (
        (
            y
            - scaler["mean"][None, :]
        )
        / scaler["std"][None, :]
    ).astype(
        "float32"
    )


def inverse_transform_targets(
    y_scaled,
    scaler,
):
    """
    Convert standardized predictions back to physical CO2 units.
    """
    y_scaled = np.asarray(
        y_scaled,
        dtype="float64",
    )

    return (
        y_scaled
        * scaler["std"][None, :]
        + scaler["mean"][None, :]
    )


def loro_ridge_predictions(
    runs,
    development_runs,
    context_steps,
    horizon_steps,
    alpha,
    reference_context=18,
):
    """
    Generate pooled leave-one-run-out Ridge predictions.

    Every validation fold fits:
    - feature scaling from training folds only;
    - target scaling from training folds only;
    - one multi-output Ridge model.

    Returns pooled predictions in physical CO2 units.
    """
    per_run = build_per_run_datasets(
        runs=runs,
        development_runs=development_runs,
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        reference_context=reference_context,
    )

    pooled_targets = []
    pooled_predictions = []
    pooled_run_names = []
    pooled_target_rows = []

    for validation_run in development_runs:

        validation_data = per_run[
            validation_run
        ]

        X_validation = validation_data[
            "X"
        ]

        y_validation = validation_data[
            "y"
        ]

        if len(y_validation) == 0:
            continue

        X_train = np.concatenate(
            [
                per_run[run_name]["X"]
                for run_name in development_runs
                if (
                    run_name != validation_run
                    and len(
                        per_run[run_name]["y"]
                    ) > 0
                )
            ],
            axis=0,
        )

        y_train = np.concatenate(
            [
                per_run[run_name]["y"]
                for run_name in development_runs
                if (
                    run_name != validation_run
                    and len(
                        per_run[run_name]["y"]
                    ) > 0
                )
            ],
            axis=0,
        )

        feature_scaler = (
            fit_window_feature_scaler(
                X_train
            )
        )

        target_scaler = (
            fit_target_scaler(
                y_train
            )
        )

        X_train_scaled = (
            transform_window_features(
                X_train,
                feature_scaler,
            )
        )

        X_validation_scaled = (
            transform_window_features(
                X_validation,
                feature_scaler,
            )
        )

        y_train_scaled = (
            transform_targets(
                y_train,
                target_scaler,
            )
        )

        model = Ridge(
            alpha=alpha,
            fit_intercept=True,
        )

        model.fit(
            X_train_scaled,
            y_train_scaled,
        )

        validation_prediction_scaled = (
            model.predict(
                X_validation_scaled
            )
        )

        validation_prediction = (
            inverse_transform_targets(
                validation_prediction_scaled,
                target_scaler,
            )
        )

        pooled_targets.append(
            y_validation
        )

        pooled_predictions.append(
            validation_prediction
        )

        pooled_run_names.extend(
            validation_data[
                "run_names"
            ]
        )

        pooled_target_rows.extend(
            validation_data[
                "target_rows"
            ]
        )

    return (
        np.concatenate(
            pooled_targets,
            axis=0,
        ),
        np.concatenate(
            pooled_predictions,
            axis=0,
        ),
        pooled_run_names,
        pooled_target_rows,
    )

def loro_ridge_residual_predictions(
    runs,
    development_runs,
    context_steps,
    horizon_steps,
    alpha,
    reference_context=18,
    last_row_only=False,
):
    """
    Leave-one-run-out Ridge that predicts the CHANGE from persistence.

    prediction = last measured CO2 per point + Ridge(window features)

    Where a point has not been measured yet, the training-fold mean is used
    as its persistence value. Scalers and fallback means use training folds
    only. If last_row_only is True, only the final input row is used.
    """
    per_run = build_per_run_datasets(
        runs=runs,
        development_runs=development_runs,
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        reference_context=reference_context,
    )

    columns = list(
        D.make_model_features(D.load_run(runs[development_runs[0]])).columns
    )
    memory_idx = [columns.index(f"last_co2_point_{p}") for p in range(1, 7)]

    pooled_targets, pooled_predictions = [], []
    pooled_run_names, pooled_target_rows = [], []

    for validation_run in development_runs:
        val = per_run[validation_run]
        if len(val["y"]) == 0:
            continue

        train_runs = [
            r for r in development_runs
            if r != validation_run and len(per_run[r]["y"]) > 0
        ]
        X_train = np.concatenate([per_run[r]["X"] for r in train_runs], axis=0)
        y_train = np.concatenate([per_run[r]["y"] for r in train_runs], axis=0)
        X_val, y_val = val["X"], val["y"]

        fallback = y_train.mean(axis=0)

        def persistence(X):
            last = X[:, -1, :][:, memory_idx]
            return np.where(np.isnan(last), fallback[None, :], last)

        base_train = persistence(X_train)
        base_val = persistence(X_val)
        residual_train = y_train - base_train

        if last_row_only:
            X_train, X_val = X_train[:, -1:, :], X_val[:, -1:, :]

        feature_scaler = fit_window_feature_scaler(X_train)
        target_scaler = fit_target_scaler(residual_train)

        X_train_scaled = transform_window_features(X_train, feature_scaler)
        X_val_scaled = transform_window_features(X_val, feature_scaler)
        residual_scaled = transform_targets(residual_train, target_scaler)

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train_scaled, residual_scaled)

        residual_pred = inverse_transform_targets(
            model.predict(X_val_scaled), target_scaler
        )

        pooled_targets.append(y_val)
        pooled_predictions.append(base_val + residual_pred)
        pooled_run_names.extend(val["run_names"])
        pooled_target_rows.extend(val["target_rows"])

    return (
        np.concatenate(pooled_targets, axis=0),
        np.concatenate(pooled_predictions, axis=0),
        pooled_run_names,
        pooled_target_rows,
    )