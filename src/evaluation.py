"""
Evaluation utilities and non-learned baselines for the CO2 forecasting task.

The baselines are evaluated with leave-one-run-out validation over the
development runs. The locked final test run is not used.
"""

import numpy as np

from src import data as D


def get_feature_columns(runs):
    """
    Return model-feature names in the same order used by forecasting windows.
    """
    first_run_path = next(iter(runs.values()))

    df = D.load_run(first_run_path)

    return list(
        D.make_model_features(df).columns
    )


def make_measured_mask(
    runs,
    sample_run_names,
    target_rows,
):
    """
    Identify which target component is directly measured at each target row.

    Each target vector contains six CO2 concentrations. At a given timestamp,
    however, AT400 directly measures only the sampling point indicated by the
    label. The other target components are interpolation-supported.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array with shape (n_samples, 6). Exactly one element per row
        should be True.
    """
    mask = np.zeros(
        (
            len(target_rows),
            D.N_SAMPLING_POINTS,
        ),
        dtype=bool,
    )

    labels_by_run = {}

    for sample_index, (
        run_name,
        target_row,
    ) in enumerate(
        zip(
            sample_run_names,
            target_rows,
        )
    ):
        if run_name not in labels_by_run:
            df = D.load_run(
                runs[run_name]
            )

            labels_by_run[
                run_name
            ] = df["label"].to_numpy()

        sampling_point = int(
            labels_by_run[
                run_name
            ][target_row]
        )

        mask[
            sample_index,
            sampling_point - 1,
        ] = True

    return mask


def persistence_predict(
    X_windows,
    feature_columns,
    fallback,
):
    """
    Predict the future six-point profile using last observed CO2 values.

    For each sampling point, the prediction is the most recent directly
    measured CO2 concentration available at the final input timestamp.

    If a point has not yet been measured by the forecast origin, its
    training-fold target mean is used as a leakage-safe fallback.
    """
    X_windows = np.asarray(
        X_windows,
        dtype="float64",
    )

    fallback = np.asarray(
        fallback,
        dtype="float64",
    )

    memory_indices = [
        feature_columns.index(
            f"last_co2_point_{point}"
        )
        for point in range(
            1,
            D.N_SAMPLING_POINTS + 1,
        )
    ]

    predictions = X_windows[
        :,
        -1,
        memory_indices,
    ].copy()

    predictions = np.where(
        np.isnan(predictions),
        fallback[None, :],
        predictions,
    )

    return predictions


def mean_predict(
    n_samples,
    fallback,
):
    """
    Predict the training-fold mean CO2 profile for every validation sample.
    """
    fallback = np.asarray(
        fallback,
        dtype="float64",
    )

    return np.tile(
        fallback[None, :],
        (
            n_samples,
            1,
        ),
    )


def regression_metrics(
    y_true,
    y_pred,
    mask=None,
):
    """
    Compute physical-unit MAE and RMSE overall and independently per point.

    Parameters
    ----------
    y_true : array-like
        Ground-truth targets with shape (n_samples, 6).

    y_pred : array-like
        Predictions with shape (n_samples, 6).

    mask : array-like of bool, optional
        If provided, metrics are calculated only where mask is True.
        This is used for directly measured-only evaluation.

    Returns
    -------
    metrics : dict
        Overall and per-point MAE/RMSE plus number of evaluated values.
    """
    y_true = np.asarray(
        y_true,
        dtype="float64",
    )

    y_pred = np.asarray(
        y_pred,
        dtype="float64",
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    errors = (
        y_pred
        - y_true
    )

    if mask is None:
        mask = np.ones_like(
            errors,
            dtype=bool,
        )
    else:
        mask = np.asarray(
            mask,
            dtype=bool,
        )

        if mask.shape != errors.shape:
            raise ValueError(
                "mask must have the same shape as y_true."
            )

    metrics = {
        "n": int(
            mask.sum()
        )
    }

    for point_index in range(
        errors.shape[1]
    ):
        point_errors = errors[
            mask[:, point_index],
            point_index,
        ]

        if len(point_errors) == 0:
            metrics[
                f"mae_p{point_index + 1}"
            ] = np.nan

            metrics[
                f"rmse_p{point_index + 1}"
            ] = np.nan

            continue

        metrics[
            f"mae_p{point_index + 1}"
        ] = float(
            np.mean(
                np.abs(
                    point_errors
                )
            )
        )

        metrics[
            f"rmse_p{point_index + 1}"
        ] = float(
            np.sqrt(
                np.mean(
                    point_errors ** 2
                )
            )
        )

    all_errors = errors[
        mask
    ]

    if len(all_errors) == 0:
        metrics["mae"] = np.nan
        metrics["rmse"] = np.nan
    else:
        metrics["mae"] = float(
            np.mean(
                np.abs(
                    all_errors
                )
            )
        )

        metrics["rmse"] = float(
            np.sqrt(
                np.mean(
                    all_errors ** 2
                )
            )
        )

    return metrics

def skill_vs_reference(
    y_true,
    y_pred,
    y_reference,
):
    """
    Per-point error ratio relative to a reference prediction.

    Skill < 1.0:
        model has lower RMSE than the reference.

    Skill = 1.0:
        model has the same RMSE as the reference.

    Skill > 1.0:
        model has higher RMSE than the reference.

    The calculation is performed independently for each
    CO2 sampling point.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    y_reference = np.asarray(
        y_reference,
        dtype="float64",
    )

    model_error = y_pred - y_true
    reference_error = y_reference - y_true

    model_rmse = np.sqrt(
        np.mean(
            model_error ** 2,
            axis=0,
        )
    )

    reference_rmse = np.sqrt(
        np.mean(
            reference_error ** 2,
            axis=0,
        )
    )

    return np.divide(
        model_rmse,
        reference_rmse,
        out=np.full_like(
            model_rmse,
            np.nan,
        ),
        where=reference_rmse > 0,
    )


def loro_baselines(
    runs,
    development_runs,
    context_steps,
    horizon_steps,
    reference_context=18,
):
    """
    Generate pooled leave-one-run-out predictions for simple baselines.

    For every fold:
    - one development run is validation;
    - all remaining development runs provide training information;
    - the locked final test run is never used;
    - training-target means are computed only from the training runs.

    Returns
    -------
    y_true : numpy.ndarray
        Pooled validation targets.

    predictions : dict
        Pooled predictions for persistence and per-point-mean baselines.

    measured_mask : numpy.ndarray
        Boolean mask identifying directly measured target components.

    sample_run_names : list
        Run associated with each pooled validation sample.

    target_rows : list
        Target row associated with each pooled validation sample.
    """
    feature_columns = get_feature_columns(
        runs
    )

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

        per_run[
            run_name
        ] = {
            "X": np.asarray(
                X_run,
                dtype="float32",
            ),
            "y": np.asarray(
                y_run,
                dtype="float32",
            ),
            "run_names": sample_run_names,
            "target_rows": target_rows,
        }

    pooled_targets = []
    pooled_persistence = []
    pooled_mean = []

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

        training_targets = [
            per_run[run_name]["y"]
            for run_name in development_runs
            if (
                run_name != validation_run
                and len(
                    per_run[run_name]["y"]
                ) > 0
            )
        ]

        if not training_targets:
            raise ValueError(
                "No training targets available "
                f"for validation run {validation_run}."
            )

        training_targets = np.concatenate(
            training_targets,
            axis=0,
        )

        training_mean = np.mean(
            training_targets,
            axis=0,
        )

        persistence_predictions = (
            persistence_predict(
                X_windows=X_validation,
                feature_columns=feature_columns,
                fallback=training_mean,
            )
        )

        mean_predictions = mean_predict(
            n_samples=len(
                y_validation
            ),
            fallback=training_mean,
        )

        pooled_targets.append(
            y_validation
        )

        pooled_persistence.append(
            persistence_predictions
        )

        pooled_mean.append(
            mean_predictions
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

    y_true = np.concatenate(
        pooled_targets,
        axis=0,
    )

    predictions = {
        "persistence": np.concatenate(
            pooled_persistence,
            axis=0,
        ),
        "per_point_mean": np.concatenate(
            pooled_mean,
            axis=0,
        ),
    }

    direct_measurement_mask = (
        make_measured_mask(
            runs=runs,
            sample_run_names=pooled_run_names,
            target_rows=pooled_target_rows,
        )
    )

    return (
        y_true,
        predictions,
        direct_measurement_mask,
        pooled_run_names,
        pooled_target_rows,
    )

def normalized_mae_per_point(
    y_true,
    y_pred,
    scale,
    mask=None,
):
    """
    Compute MAE independently for each sampling point after normalizing
    absolute errors by a supplied per-point scale.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, 6)
        Ground-truth CO2 concentrations.

    y_pred : array-like, shape (n_samples, 6)
        Predicted CO2 concentrations.

    scale : array-like, shape (6,)
        Per-point normalization scale. For cross-validation this should come
        from the corresponding training fold only.

    mask : array-like of bool, optional
        If supplied, evaluate only entries where mask is True.

    Returns
    -------
    dict
        Normalized MAE independently for each sampling point.
    """
    y_true = np.asarray(
        y_true,
        dtype="float64",
    )

    y_pred = np.asarray(
        y_pred,
        dtype="float64",
    )

    scale = np.asarray(
        scale,
        dtype="float64",
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if scale.shape != (y_true.shape[1],):
        raise ValueError(
            "scale must contain one value per target point."
        )

    safe_scale = np.where(
        (~np.isfinite(scale)) | (scale < 1e-8),
        1.0,
        scale,
    )

    absolute_normalized_error = (
        np.abs(y_pred - y_true)
        / safe_scale[None, :]
    )

    if mask is None:
        mask = np.ones_like(
            absolute_normalized_error,
            dtype=bool,
        )
    else:
        mask = np.asarray(
            mask,
            dtype=bool,
        )

        if mask.shape != y_true.shape:
            raise ValueError(
                "mask must have the same shape as y_true."
            )

    result = {}

    for point_index in range(
        y_true.shape[1]
    ):
        values = absolute_normalized_error[
            mask[:, point_index],
            point_index,
        ]

        result[
            f"nmae_p{point_index + 1}"
        ] = (
            float(np.mean(values))
            if len(values)
            else np.nan
        )

    return result


def skill_vs_persistence(
    y_true,
    y_pred,
    persistence_pred,
    mask=None,
):
    """
    Compute per-point MAE skill ratio relative to persistence.

    skill < 1 : model is better than persistence
    skill = 1 : model matches persistence
    skill > 1 : model is worse than persistence

    The model and persistence predictions must be evaluated on exactly the
    same target samples.
    """
    model_metrics = regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
    )

    persistence_metrics = regression_metrics(
        y_true=y_true,
        y_pred=persistence_pred,
        mask=mask,
    )

    result = {}

    for point in range(
        1,
        D.N_SAMPLING_POINTS + 1,
    ):
        model_mae = model_metrics[
            f"mae_p{point}"
        ]

        persistence_mae = persistence_metrics[
            f"mae_p{point}"
        ]

        if (
            not np.isfinite(model_mae)
            or not np.isfinite(persistence_mae)
            or persistence_mae < 1e-12
        ):
            skill = np.nan
        else:
            skill = (
                model_mae
                / persistence_mae
            )

        result[
            f"skill_p{point}"
        ] = float(skill)

    return result

def loro_normalized_baseline_metrics(
    runs,
    development_runs,
    context_steps,
    horizon_steps,
    reference_context=18,
):
    """
    Compute leave-one-run-out normalized MAE for the non-learned baselines.

    For each validation fold:
    - target standard deviations are estimated only from the other
      development runs;
    - validation errors are normalized using those training-only scales;
    - normalized absolute errors are then pooled across folds.

    Returns
    -------
    dict
        Per-point normalized MAE for persistence and per-point-mean baselines.
    """
    feature_columns = get_feature_columns(
        runs
    )

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
        }

    normalized_errors = {
        "persistence": [],
        "per_point_mean": [],
    }

    for validation_run in development_runs:

        X_validation = per_run[
            validation_run
        ]["X"]

        y_validation = per_run[
            validation_run
        ]["y"]

        if len(y_validation) == 0:
            continue

        training_targets = np.concatenate(
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

        training_mean = np.mean(
            training_targets,
            axis=0,
        )

        training_std = np.std(
            training_targets,
            axis=0,
            ddof=0,
        )

        safe_training_std = np.where(
            (
                ~np.isfinite(
                    training_std
                )
            )
            | (
                training_std
                < 1e-8
            ),
            1.0,
            training_std,
        )

        persistence_predictions = (
            persistence_predict(
                X_windows=X_validation,
                feature_columns=feature_columns,
                fallback=training_mean,
            )
        )

        mean_predictions = mean_predict(
            n_samples=len(
                y_validation
            ),
            fallback=training_mean,
        )

        for (
            baseline_name,
            predictions,
        ) in {
            "persistence":
                persistence_predictions,
            "per_point_mean":
                mean_predictions,
        }.items():

            fold_normalized_errors = (
                np.abs(
                    predictions
                    - y_validation
                )
                / safe_training_std[
                    None,
                    :
                ]
            )

            normalized_errors[
                baseline_name
            ].append(
                fold_normalized_errors
            )

    results = {}

    for (
        baseline_name,
        error_parts,
    ) in normalized_errors.items():

        pooled_errors = np.concatenate(
            error_parts,
            axis=0,
        )

        results[
            baseline_name
        ] = {
            f"nmae_p{point + 1}":
                float(
                    np.mean(
                        pooled_errors[
                            :,
                            point,
                        ]
                    )
                )
            for point in range(
                D.N_SAMPLING_POINTS
            )
        }

    return results