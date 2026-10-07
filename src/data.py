from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TEST_RUN = "140207_1"
N_SAMPLING_POINTS = 6

def list_runs(raw_dir=RAW_DATA_DIR):
    """Return the valid experimental Excel runs in the raw-data directory."""
    files = sorted(
        path
        for path in Path(raw_dir).glob("*.xlsx")
        if not path.name.startswith(("_$", "~$"))
    )

    return {path.stem: path for path in files}

def load_run(path):
    """Load one experimental run and standardize its column names."""
    df = pd.read_excel(
        path,
        sheet_name=0,
        header=[0, 1],
    )

    column_names = [column[0] for column in df.columns]
    column_names[0] = "timestamp"
    column_names[-1] = "label"

    if len(column_names) != len(set(column_names)):
        raise ValueError("Duplicate sensor names found after flattening columns.")

    df.columns = column_names

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["label"] = df["label"].astype(int)

    return df

def make_sparse_targets(df):
    """Create six sparse CO₂ target series from AT400 and the sampling label."""
    targets = pd.DataFrame(
        index=df.index,
        columns=[f"co2_point_{point}" for point in range(1, N_SAMPLING_POINTS + 1)],
        dtype=float,
    )

    for point in range(1, N_SAMPLING_POINTS + 1):
        mask = df["label"] == point
        targets.loc[mask, f"co2_point_{point}"] = df.loc[mask, "AT400"]

    return targets

def interpolate_targets(sparse_targets):
    """Interpolate each CO₂ target only between directly observed values."""
    return sparse_targets.interpolate(
        method="linear",
        limit_area="inside",
    )

def get_supported_range(interpolated_targets):
    """Return the first and last rows where all six CO₂ targets are supported."""
    fully_supported = interpolated_targets.notna().all(axis=1)

    if not fully_supported.any():
        raise ValueError("No rows contain a fully supported six-point CO₂ profile.")

    supported_indices = interpolated_targets.index[fully_supported]

    return supported_indices[0], supported_indices[-1]

def make_analyzer_memory(sparse_targets):
    """Carry the most recent measured CO₂ value forward for each sampling point."""
    return sparse_targets.ffill()

def make_analyzer_age(df):
    """Return the number of rows since each sampling point was last measured."""
    ages = pd.DataFrame(
        index=df.index,
        columns=[f"age_point_{point}" for point in range(1, N_SAMPLING_POINTS + 1)],
        dtype=float,
    )

    for point in range(1, N_SAMPLING_POINTS + 1):
        last_measurement_row = None

        for row_position, label in enumerate(df["label"]):
            if label == point:
                last_measurement_row = row_position
                ages.iloc[row_position, point - 1] = 0
            elif last_measurement_row is not None:
                ages.iloc[row_position, point - 1] = (
                    row_position - last_measurement_row
                )

    return ages

def make_label_features(df):
    """One-hot encode the active analyzer sampling location."""
    labels = pd.get_dummies(
        df["label"],
        prefix="sampling_point",
        dtype=float,
    )

    expected_columns = [
        f"sampling_point_{point}"
        for point in range(1, N_SAMPLING_POINTS + 1)
    ]

    return labels.reindex(
        columns=expected_columns,
        fill_value=0.0,
    )

CONSTANT_PROCESS_COLUMNS = {
    "FT101",
    "FT107",
    "FT300",
}


def make_process_features(df):
    """Return process-sensor features used as model inputs."""
    excluded_columns = {
        "timestamp",
        "label",
        "AT400",
        *CONSTANT_PROCESS_COLUMNS,
    }

    process_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    return df[process_columns].copy()

def make_model_features(df):
    """
    Causal feature set for the main model (110 features):
    86 process sensors + 6 last-measured CO2 values + 6 ages
    + 6 seen flags + 6 sampling-point one-hots.

    last_co2_point_k and age_point_k are intentionally NaN before point k is
    first measured. They are filled later, after scaling, and the seen flag
    tells the model that the filled value is a placeholder.
    """
    process_features = make_process_features(df)
    sparse_targets = make_sparse_targets(df)

    analyzer_memory = make_analyzer_memory(sparse_targets)
    analyzer_memory.columns = [
        f"last_co2_point_{p}" for p in range(1, N_SAMPLING_POINTS + 1)
    ]

    analyzer_age = make_analyzer_age(df)

    # 0 until point k has been measured for the first time, then 1 forever.
    # Only looks backward, so there is no leakage.
    seen_features = pd.DataFrame(index=df.index)
    for p in range(1, N_SAMPLING_POINTS + 1):
        seen_features[f"seen_point_{p}"] = (
            sparse_targets[f"co2_point_{p}"].notna().astype(float).cummax()
        )

    label_features = make_label_features(df)

    return pd.concat(
        [process_features, analyzer_memory, analyzer_age,
         seen_features, label_features],
        axis=1,
    )


def make_forecast_windows(
    df,
    context_steps,
    horizon_steps,
    allowed_target_rows=None,
):
    """
    Create causal input windows and six-point CO₂ forecasting targets.

    Missing values are allowed only in analyzer-history features before a
    sampling point has been observed for the first time. The corresponding
    seen_point feature identifies these intentional missing-history states.

    Any missing value in another input feature invalidates the window.
    """
    features = make_model_features(df)

    sparse_targets = make_sparse_targets(df)
    targets = interpolate_targets(sparse_targets)

    X_windows = []
    y_targets = []
    target_rows = []

    if allowed_target_rows is not None:
        allowed_target_rows = set(
            allowed_target_rows
        )

    intentional_nan_columns = {
        *[
            f"last_co2_point_{point}"
            for point in range(
                1,
                N_SAMPLING_POINTS + 1,
            )
        ],
        *[
            f"age_point_{point}"
            for point in range(
                1,
                N_SAMPLING_POINTS + 1,
            )
        ],
    }

    required_complete_columns = [
        column
        for column in features.columns
        if column not in intentional_nan_columns
    ]

    for target_row in targets.index:

        if (
            allowed_target_rows is not None
            and target_row not in allowed_target_rows
        ):
            continue

        input_end = (
            target_row
            - horizon_steps
        )

        input_start = (
            input_end
            - context_steps
            + 1
        )

        if input_start < 0:
            continue

        X_window = features.loc[
            input_start:input_end
        ]

        y_target = targets.loc[
            target_row
        ]

        if len(X_window) != context_steps:
            continue

        # Analyzer-memory and age NaNs are intentional before the first
        # observation. All other model inputs must be complete.
        if not (
            X_window[
                required_complete_columns
            ]
            .notna()
            .all()
            .all()
        ):
            continue

        if not y_target.notna().all():
            continue

        X_windows.append(
            X_window.to_numpy(
                dtype="float32"
            )
        )

        y_targets.append(
            y_target.to_numpy(
                dtype="float32"
            )
        )

        target_rows.append(
            target_row
        )

    return (
        X_windows,
        y_targets,
        target_rows,
    )

def split_run_names(runs):
    """Separate development runs from the locked final test run."""
    development_runs = [
        run_name
        for run_name in runs
        if run_name != TEST_RUN
    ]

    return development_runs, TEST_RUN

def build_forecast_dataset(
    runs,
    run_names,
    context_steps,
    horizon_steps,
    reference_context=18,
):
    """
    Assemble forecasting windows from multiple experimental runs.

    The longest reference context determines the allowed target timestamps so
    that different context lengths can be compared on identical targets.

    Parameters
    ----------
    runs : dict
        Mapping from run name to Excel-file path.

    run_names : iterable
        Experimental runs to include.

    context_steps : int
        Number of historical timestamps in each input sequence.

    horizon_steps : int
        Forecast horizon in rows.

    reference_context : int, optional
        Context length used to define the common target timestamps.

    Returns
    -------
    X_windows : list
        Input arrays with shape (context_steps, n_features).

    y_targets : list
        Six-dimensional CO₂ target arrays.

    sample_run_names : list
        Experimental run associated with every sample.

    target_rows : list
        Target-row index associated with every sample.
    """
    X_windows = []
    y_targets = []
    sample_run_names = []
    target_rows = []

    for run_name in run_names:
        df = load_run(runs[run_name])

        # Define target timestamps using the longest context so that all
        # candidate context lengths are evaluated on identical targets.
        _, _, common_target_rows = make_forecast_windows(
            df,
            context_steps=reference_context,
            horizon_steps=horizon_steps,
        )

        X_run, y_run, rows_run = make_forecast_windows(
            df,
            context_steps=context_steps,
            horizon_steps=horizon_steps,
            allowed_target_rows=common_target_rows,
        )

        X_windows.extend(X_run)
        y_targets.extend(y_run)
        sample_run_names.extend(
            [run_name] * len(X_run)
        )
        target_rows.extend(rows_run)

    return (
        X_windows,
        y_targets,
        sample_run_names,
        target_rows,
    )

UNSCALED_PREFIXES = (
    "seen_point_",
    "sampling_point_",
)


def build_run_cache(run_paths):
    """
    Load each experimental run once and cache its model features and targets.

    Returns
    -------
    cache : dict
        Mapping from run name to a tuple:
        (causal feature DataFrame, interpolated target DataFrame).
    """
    cache = {}

    for run_name, path in run_paths.items():
        df = load_run(path)

        features = make_model_features(df)

        sparse_targets = make_sparse_targets(df)
        targets = interpolate_targets(sparse_targets)

        cache[run_name] = (
            features,
            targets,
        )

    return cache


def input_rows_for_targets(
    target_rows,
    context_steps,
    horizon_steps,
):
    """
    Return the unique raw rows used as model inputs for a set of targets.

    Each underlying observation is returned only once even though overlapping
    forecasting windows may reuse it many times.
    """
    input_rows = set()

    for target_row in target_rows:
        input_end = (
            target_row
            - horizon_steps
        )

        input_start = (
            input_end
            - context_steps
            + 1
        )

        input_rows.update(
            range(
                input_start,
                input_end + 1,
            )
        )

    return sorted(input_rows)


def fit_feature_scaler(
    cache,
    train_target_rows,
    context_steps,
    horizon_steps,
):
    """
    Fit feature scaling statistics on unique training input rows only.

    Parameters
    ----------
    cache : dict
        Output of build_run_cache().

    train_target_rows : dict
        Mapping from training run name to the target-row indices used by that
        run's forecasting windows.

    context_steps : int
        Number of historical rows in each forecasting window.

    horizon_steps : int
        Number of rows between the final input observation and target.

    Returns
    -------
    scaler : dict
        Feature names plus training-only mean and standard deviation.

    Notes
    -----
    Intentional analyzer-history NaNs are ignored when estimating statistics.

    Binary seen-point and sampling-location indicators are left unscaled.

    Features with zero or near-zero training variance receive a scale of 1.
    """
    training_parts = []

    for run_name, target_rows in train_target_rows.items():

        input_rows = input_rows_for_targets(
            target_rows=target_rows,
            context_steps=context_steps,
            horizon_steps=horizon_steps,
        )

        if not input_rows:
            continue

        training_parts.append(
            cache[run_name][0].loc[
                input_rows
            ]
        )

    if not training_parts:
        raise ValueError(
            "No valid training feature rows were found."
        )

    training_features = pd.concat(
        training_parts,
        axis=0,
    )

    columns = list(
        training_features.columns
    )

    feature_mean = (
        training_features
        .mean(skipna=True)
        .to_numpy(dtype="float64")
    )

    feature_std = (
        training_features
        .std(
            ddof=0,
            skipna=True,
        )
        .to_numpy(dtype="float64")
    )

    feature_mean = np.where(
        np.isfinite(feature_mean),
        feature_mean,
        0.0,
    )

    feature_std = np.where(
        (~np.isfinite(feature_std))
        | (feature_std < 1e-8),
        1.0,
        feature_std,
    )

    # Binary indicators already have meaningful 0/1 representations.
    for index, column in enumerate(columns):
        if column.startswith(
            UNSCALED_PREFIXES
        ):
            feature_mean[index] = 0.0
            feature_std[index] = 1.0

    return {
        "columns": columns,
        "mean": feature_mean,
        "std": feature_std,
    }


def transform_features(
    X_windows,
    scaler,
):
    """
    Standardize forecasting inputs using training-only statistics.

    Intentional analyzer-history NaNs remain NaN during standardization and
    are then replaced by zero. In standardized space, zero corresponds to the
    training mean; the accompanying seen-point flag identifies whether the
    value is an actual observation or a placeholder.
    """
    X_windows = np.asarray(
        X_windows,
        dtype="float64",
    )

    feature_mean = np.asarray(
        scaler["mean"],
        dtype="float64",
    )

    feature_std = np.asarray(
        scaler["std"],
        dtype="float64",
    )

    if X_windows.ndim != 3:
        raise ValueError(
            "X_windows must have shape "
            "(n_samples, context_steps, n_features)."
        )

    if X_windows.shape[-1] != len(
        scaler["columns"]
    ):
        raise ValueError(
            "Number of input features does not match "
            "the fitted scaler."
        )

    if (
        len(feature_mean)
        != len(feature_std)
    ):
        raise ValueError(
            "Feature mean and standard deviation "
            "must have the same length."
        )

    X_scaled = (
        X_windows
        - feature_mean[None, None, :]
    ) / feature_std[None, None, :]

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
    )

    return X_scaled.astype(
        "float32"
    )

def fit_target_scaler(y_targets):
    """
    Fit independent standardization statistics for the six CO₂ targets.

    Parameters
    ----------
    y_targets : array-like
        Target values with shape (n_samples, 6).

    Returns
    -------
    target_mean : numpy.ndarray
        Mean of each CO₂ target.

    target_std : numpy.ndarray
        Standard deviation of each CO₂ target, with zero-variance
        targets assigned a scale of 1.
    """
    import numpy as np

    y_targets = np.asarray(
        y_targets,
        dtype="float32",
    )

    if y_targets.ndim != 2:
        raise ValueError(
            "y_targets must have shape "
            "(n_samples, n_targets)."
        )

    target_mean = y_targets.mean(
        axis=0
    )

    target_std = y_targets.std(
        axis=0
    )

    target_std = np.where(
        target_std < 1e-8,
        1.0,
        target_std,
    )

    return target_mean, target_std


def transform_targets(
    y_targets,
    target_mean,
    target_std,
):
    """
    Standardize CO₂ targets using pre-fitted training statistics.
    """
    import numpy as np

    y_targets = np.asarray(
        y_targets,
        dtype="float32",
    )

    target_mean = np.asarray(
        target_mean,
        dtype="float32",
    )

    target_std = np.asarray(
        target_std,
        dtype="float32",
    )

    if y_targets.ndim != 2:
        raise ValueError(
            "y_targets must have shape "
            "(n_samples, n_targets)."
        )

    if y_targets.shape[-1] != len(target_mean):
        raise ValueError(
            "Number of targets does not match "
            "the fitted target mean."
        )

    if len(target_mean) != len(target_std):
        raise ValueError(
            "target_mean and target_std must "
            "have the same length."
        )

    y_scaled = (
        y_targets
        - target_mean[None, :]
    ) / target_std[None, :]

    return y_scaled.astype(
        "float32"
    )


def inverse_transform_targets(
    y_scaled,
    target_mean,
    target_std,
):
    """
    Convert standardized CO₂ values back to their original units.
    """
    import numpy as np

    y_scaled = np.asarray(
        y_scaled,
        dtype="float32",
    )

    target_mean = np.asarray(
        target_mean,
        dtype="float32",
    )

    target_std = np.asarray(
        target_std,
        dtype="float32",
    )

    if y_scaled.ndim != 2:
        raise ValueError(
            "y_scaled must have shape "
            "(n_samples, n_targets)."
        )

    if y_scaled.shape[-1] != len(target_mean):
        raise ValueError(
            "Number of targets does not match "
            "the fitted target mean."
        )

    if len(target_mean) != len(target_std):
        raise ValueError(
            "target_mean and target_std must "
            "have the same length."
        )

    y_original = (
        y_scaled
        * target_std[None, :]
        + target_mean[None, :]
    )

    return y_original.astype(
        "float32"
    )