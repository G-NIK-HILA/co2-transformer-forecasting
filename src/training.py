"""
Training-side data preparation for the residual Transformer.

All fitted quantities are computed from the training runs only:
- feature scaling statistics,
- persistence fallback values,
- residual RMS scaling.

The held-out validation run does not influence any fitted quantity.

Residual-forecast formulation:

    forecast = persistence + residual_scale * model_output

The residual target is scaled but NOT mean-centered. Therefore, because
the Transformer's prediction head is initialized to zero:

    model_output = 0
    forecast = persistence

exactly before training.
"""

import numpy as np

from src import data as D


def last_measured_persistence(
    X_raw,
    columns,
    fallback,
):
    """
    Construct the causal persistence forecast.

    For each of the six CO2 sampling points, use the most recently
    measured CO2 value available at the final row of the input window.

    If a sampling point has not yet been measured in the current run,
    use the training-fold mean target for that point as a fallback.

    Parameters
    ----------
    X_raw : array-like
        Raw, unscaled input windows with shape
        (n_samples, context_steps, n_features).

    columns : list of str
        Feature names corresponding to the final dimension of X_raw.

    fallback : array-like
        Training-fold fallback values with shape (6,).

    Returns
    -------
    np.ndarray
        Persistence forecasts with shape (n_samples, 6).
    """

    memory_indices = [
        columns.index(
            f"last_co2_point_{p}"
        )
        for p in range(1, 7)
    ]

    X_raw = np.asarray(
        X_raw,
        dtype="float64",
    )

    fallback = np.asarray(
        fallback,
        dtype="float64",
    )

    # Final historical row of each input window.
    last_row = X_raw[:, -1, :]

    # Last directly measured CO2 value remembered for each point.
    last_measured = last_row[
        :,
        memory_indices,
    ]

    # Before a sampling point has been observed for the first time,
    # its memory feature is NaN. Only those unavailable values use
    # the training-fold fallback.
    persistence = np.where(
        np.isnan(last_measured),
        fallback[None, :],
        last_measured,
    )

    return persistence


def prepare_fold(
    runs,
    cache,
    train_runs,
    val_runs,
    context_steps,
    horizon_steps,
    reference_context=18,
):
    """
    Prepare one train/validation fold for residual Transformer training.

    Everything that must be learned or estimated from data is fitted
    using the training runs only.

    Returns
    -------
    dict
        X_train, X_val
            Scaled Transformer inputs with shape
            (n_samples, context_steps, n_features).

        base_train, base_val
            Causal persistence forecasts in physical CO2 units.

        y_train, y_val
            True six-point CO2 profiles in physical units.

        resid_train, resid_val
            Scale-normalized residual targets.

        resid_mean
            Always zero. Retained explicitly for compatibility and
            transparent inversion of the residual transformation.

        resid_std
            Training-fold RMS residual scale for each target. The name
            is retained for compatibility with the earlier interface,
            although the quantity is now RMS rather than standard
            deviation.

        persistence_fallback
            Training-fold target means used only when a point has not
            yet been measured in a run.

        feature_scaler
            Feature scaler fitted using training input rows only.

        train_run_names, train_target_rows
            Training-sample identities.

        val_run_names, val_target_rows
            Validation-sample identities.
    """

    # ---------------------------------------------------------
    # Build raw forecasting windows
    # ---------------------------------------------------------

    (
        X_train_raw,
        y_train,
        train_run_names,
        train_target_rows,
    ) = D.build_forecast_dataset(
        runs=runs,
        run_names=train_runs,
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        reference_context=reference_context,
    )

    (
        X_val_raw,
        y_val,
        val_run_names,
        val_target_rows,
    ) = D.build_forecast_dataset(
        runs=runs,
        run_names=val_runs,
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        reference_context=reference_context,
    )

    if len(y_train) == 0:
        raise ValueError(
            "Training fold has no valid forecast windows."
        )

    if len(y_val) == 0:
        raise ValueError(
            "Validation fold has no valid forecast windows."
        )

    X_train_raw = np.asarray(
        X_train_raw,
        dtype="float64",
    )

    X_val_raw = np.asarray(
        X_val_raw,
        dtype="float64",
    )

    y_train = np.asarray(
        y_train,
        dtype="float64",
    )

    y_val = np.asarray(
        y_val,
        dtype="float64",
    )

    # ---------------------------------------------------------
    # Fit feature scaler using TRAINING input rows only
    # ---------------------------------------------------------

    train_target_rows_by_run = {}

    for run_name, target_row in zip(
        train_run_names,
        train_target_rows,
    ):
        train_target_rows_by_run.setdefault(
            run_name,
            [],
        ).append(
            target_row
        )

    feature_scaler = D.fit_feature_scaler(
        cache=cache,
        train_target_rows=train_target_rows_by_run,
        context_steps=context_steps,
        horizon_steps=horizon_steps,
    )

    columns = feature_scaler[
        "columns"
    ]

    # ---------------------------------------------------------
    # Construct persistence baseline
    # ---------------------------------------------------------

    # Training-only fallback for a sampling point that has not yet
    # been directly measured in a particular run.
    persistence_fallback = y_train.mean(
        axis=0
    )

    base_train = last_measured_persistence(
        X_raw=X_train_raw,
        columns=columns,
        fallback=persistence_fallback,
    )

    base_val = last_measured_persistence(
        X_raw=X_val_raw,
        columns=columns,
        fallback=persistence_fallback,
    )

    # ---------------------------------------------------------
    # Raw residual targets
    #
    # residual = future truth - persistence
    # ---------------------------------------------------------

    residual_train_raw = (
        y_train
        - base_train
    )

    residual_val_raw = (
        y_val
        - base_val
    )

    # ---------------------------------------------------------
    # Scale residuals WITHOUT mean-centering
    # ---------------------------------------------------------
    #
    # We deliberately keep the residual center at zero.
    #
    # This guarantees:
    #
    # model output = 0
    # physical correction = 0
    # final forecast = persistence
    #
    # Because we do not subtract the empirical residual mean,
    # RMS is the appropriate scale:
    #
    # RMS = sqrt(mean(residual^2))
    #
    # After division by this value, each target has unit RMS on
    # the training fold.
    # ---------------------------------------------------------

    residual_mean = np.zeros(
        residual_train_raw.shape[1],
        dtype="float64",
    )

    residual_std = np.sqrt(
        np.mean(
            residual_train_raw ** 2,
            axis=0,
        )
    )

    # Numerical safety for any target whose residual variation is
    # effectively zero.
    residual_std = np.where(
        (~np.isfinite(residual_std))
        | (residual_std < 1e-8),
        1.0,
        residual_std,
    )

    residual_train_scaled = (
        residual_train_raw
        / residual_std[None, :]
    )

    residual_val_scaled = (
        residual_val_raw
        / residual_std[None, :]
    )

    # ---------------------------------------------------------
    # Scale Transformer inputs
    # ---------------------------------------------------------

    X_train_scaled = D.transform_features(
        X_train_raw,
        feature_scaler,
    )

    X_val_scaled = D.transform_features(
        X_val_raw,
        feature_scaler,
    )

    # ---------------------------------------------------------
    # Final numerical safety checks
    # ---------------------------------------------------------

    if not np.isfinite(
        X_train_scaled
    ).all():
        raise ValueError(
            "Non-finite values found in scaled training inputs."
        )

    if not np.isfinite(
        X_val_scaled
    ).all():
        raise ValueError(
            "Non-finite values found in scaled validation inputs."
        )

    if not np.isfinite(
        residual_train_scaled
    ).all():
        raise ValueError(
            "Non-finite values found in training residual targets."
        )

    if not np.isfinite(
        residual_val_scaled
    ).all():
        raise ValueError(
            "Non-finite values found in validation residual targets."
        )

    # ---------------------------------------------------------
    # Return fold
    # ---------------------------------------------------------

    return {
        "X_train": X_train_scaled.astype(
            "float32"
        ),

        "X_val": X_val_scaled.astype(
            "float32"
        ),

        "base_train": base_train,
        "base_val": base_val,

        "y_train": y_train,
        "y_val": y_val,

        "resid_train": residual_train_scaled.astype(
            "float32"
        ),

        "resid_val": residual_val_scaled.astype(
            "float32"
        ),

        # Kept explicitly so the inverse transformation is clear.
        # This is intentionally zero.
        "resid_mean": residual_mean,

        # Despite the historical key name, this is now the
        # training residual RMS scale.
        "resid_std": residual_std,

        "persistence_fallback": persistence_fallback,

        "feature_scaler": feature_scaler,

        "train_run_names": train_run_names,
        "train_target_rows": train_target_rows,

        "val_run_names": val_run_names,
        "val_target_rows": val_target_rows,
    }


def correction_to_forecast(
    correction_scaled,
    base,
    resid_mean,
    resid_std,
):
    """
    Convert scaled Transformer output into a physical CO2 forecast.

    With the current scale-only residual formulation:

        resid_mean = 0

    so this is effectively:

        physical_correction
            = scaled_correction * residual_RMS

        forecast
            = persistence + physical_correction

    The resid_mean argument is retained so that the transformation
    remains explicit and the interface stays compatible with earlier
    code.
    """

    correction_scaled = np.asarray(
        correction_scaled,
        dtype="float64",
    )

    base = np.asarray(
        base,
        dtype="float64",
    )

    resid_mean = np.asarray(
        resid_mean,
        dtype="float64",
    )

    resid_std = np.asarray(
        resid_std,
        dtype="float64",
    )

    physical_correction = (
        correction_scaled
        * resid_std[None, :]
        + resid_mean[None, :]
    )

    forecast = (
        base
        + physical_correction
    )

    return forecast