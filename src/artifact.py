"""
Build and load the deployable CO2 forecasting artifact.

The artifact is fitted using development runs only. The locked test run is
never loaded while building the artifact.

The final development-only model-selection procedure selected epoch 0.
Because the residual Transformer's prediction head is zero-initialized,
epoch 0 produces zero Transformer correction and the selected forecast is
therefore exactly the causal persistence forecast.

Production inference still instantiates the same Transformer architecture so
the deployed pipeline remains consistent with the research implementation.
"""

import inspect
from pathlib import Path

import numpy as np
import torch

from src import data as D
from src import nested as N
from src import trainer as TR
from src import training as T
from src.model import CO2Transformer


ARTIFACT_PATH = D.PROJECT_ROOT / "artifacts" / "model_v1.pt"

MODEL_VERSION = "co2-transformer-v1-epoch0"

CONTEXT_STEPS = 6
HORIZON_STEPS = 6
REFERENCE_CONTEXT = 18

SELECTED_EPOCH = 0
SEED = 42
EXPECTED_DEVELOPMENT_RUNS = 7
EXPECTED_FEATURES = 110


def architecture_defaults():
    """
    Return CO2Transformer constructor defaults for artifact metadata.

    Required dimensions such as n_features, n_targets, and max_length are
    stored separately from the prepared final fold. Optional architecture
    parameters are read directly from the model class rather than duplicated
    here.
    """
    parameters = inspect.signature(CO2Transformer.__init__).parameters

    return {
        name: parameter.default
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.default is not inspect.Parameter.empty
    }


def build_artifact():
    """
    Build the deployment artifact using development data only.

    The existing research preprocessing is reused through prepare_fold().
    A development run is supplied as the validation run solely because the
    fold-preparation interface constructs both train and validation arrays.
    All fitted deployment quantities come from the training side of the fold.

    The locked test run is excluded before any run is loaded.
    """
    runs = D.list_runs()
    development_runs, test_run = D.split_run_names(runs)

    if len(development_runs) != EXPECTED_DEVELOPMENT_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_DEVELOPMENT_RUNS} development runs, "
            f"found {len(development_runs)}: {development_runs}"
        )

    if test_run in development_runs:
        raise RuntimeError(
            "Locked test run unexpectedly appears in development runs."
        )

    dev_paths = {
        run_name: runs[run_name]
        for run_name in development_runs
    }

    # Only development files are loaded into this cache.
    cache = D.build_run_cache(dev_paths)

    validation_run = development_runs[-1]

    fold = T.prepare_fold(
        runs=dev_paths,
        cache=cache,
        train_runs=development_runs,
        val_runs=[validation_run],
        context_steps=CONTEXT_STEPS,
        horizon_steps=HORIZON_STEPS,
        reference_context=REFERENCE_CONTEXT,
    )

    n_features = int(fold["X_train"].shape[-1])
    n_targets = int(fold["y_train"].shape[-1])
    max_length = int(fold["X_train"].shape[1])

    if n_features != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURES} model features, got {n_features}."
        )

    if max_length != CONTEXT_STEPS:
        raise RuntimeError(
            f"Expected model context length {CONTEXT_STEPS}, "
            f"got {max_length}."
        )

    final_seed = SEED + len(development_runs) + 1000
    TR.set_seed(final_seed)

    # This is deliberately the same construction used by the research
    # pipeline. No model hyperparameters are redefined here.
    model = N.new_model(
        fold=fold,
        model_kwargs=None,
    ).cpu()

    # Epoch 0 was selected, so this fresh model must remain untrained.
    # Verify the residual prediction head is exactly zero-initialized.
    head = model.prediction_head

    max_head_weight = float(
        head.weight.detach().abs().max().item()
    )
    max_head_bias = float(
        head.bias.detach().abs().max().item()
    )

    if max_head_weight != 0.0 or max_head_bias != 0.0:
        raise RuntimeError(
            "Epoch-0 model does not have a zero-initialized "
            "prediction head."
        )

    model_kwargs = {
        "n_features": n_features,
        "n_targets": n_targets,
        "max_length": max_length,
    }

    feature_scaler = fold["feature_scaler"]

    if len(feature_scaler["columns"]) != EXPECTED_FEATURES:
        raise RuntimeError(
            "Feature scaler does not contain the expected "
            f"{EXPECTED_FEATURES} columns."
        )

    first_development_df = D.load_run(
        dev_paths[development_runs[0]]
    )

    raw_columns = list(first_development_df.columns)

    artifact = {
        "model_version": MODEL_VERSION,
        "selected_epoch": SELECTED_EPOCH,
        "context_steps": CONTEXT_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "reference_context": REFERENCE_CONTEXT,
        "development_runs": list(development_runs),
        "locked_test_run": test_run,
        "final_seed": final_seed,
        "model_kwargs": model_kwargs,
        "architecture_defaults": architecture_defaults(),
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "feature_scaler": feature_scaler,
        "resid_mean": np.asarray(
            fold["resid_mean"],
            dtype="float64",
        ),
        "resid_std": np.asarray(
            fold["resid_std"],
            dtype="float64",
        ),
        "persistence_fallback": np.asarray(
            fold["persistence_fallback"],
            dtype="float64",
        ),
        "raw_columns": raw_columns,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }

    return artifact, fold


def save_artifact(
    artifact,
    path=ARTIFACT_PATH,
):
    """Serialize the deployment artifact to disk."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        artifact,
        path,
    )

    return path


def load_artifact(
    path=ARTIFACT_PATH,
):
    """Load a deployment artifact on CPU."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def build_model(artifact):
    """
    Reconstruct the exact Transformer represented by the artifact.
    """
    model = CO2Transformer(
        **artifact["model_kwargs"]
    )

    model.load_state_dict(
        artifact["state_dict"],
        strict=True,
    )

    model.eval()

    return model


def predict_from_raw_windows(
    artifact,
    model,
    X_raw,
):
    """
    Forecast from unscaled 110-feature windows.

    Parameters
    ----------
    artifact : dict
        Loaded deployment artifact.

    model : CO2Transformer
        Reconstructed Transformer in evaluation mode.

    X_raw : array-like
        Shape:
            (n_samples, context_steps, n_features)

        Feature order must match:
            artifact["feature_scaler"]["columns"]

    Returns
    -------
    forecast : numpy.ndarray
        Final six-point CO2 forecast in physical units.

    persistence : numpy.ndarray
        Causal persistence component in physical units.

    transformer_correction : numpy.ndarray
        Transformer residual correction in physical units.
    """
    X_raw = np.asarray(
        X_raw,
        dtype="float32",
    )

    expected_shape_tail = (
        artifact["context_steps"],
        len(
            artifact["feature_scaler"]["columns"]
        ),
    )

    if X_raw.ndim != 3:
        raise ValueError(
            "X_raw must have shape "
            "(n_samples, context_steps, n_features)."
        )

    if X_raw.shape[1:] != expected_shape_tail:
        raise ValueError(
            "Unexpected input-window shape. "
            f"Expected (*, {expected_shape_tail[0]}, "
            f"{expected_shape_tail[1]}), "
            f"got {X_raw.shape}."
        )

    scaler = artifact["feature_scaler"]

    persistence = T.last_measured_persistence(
        X_raw=X_raw,
        columns=list(scaler["columns"]),
        fallback=artifact["persistence_fallback"],
    )

    X_scaled = D.transform_features(
        X_windows=X_raw,
        scaler=scaler,
    )

    correction_scaled = TR.predict_scaled_correction(
        model=model,
        X=X_scaled,
        device=torch.device("cpu"),
        batch_size=64,
    )

    forecast = T.correction_to_forecast(
        correction_scaled=correction_scaled,
        base=persistence,
        resid_mean=artifact["resid_mean"],
        resid_std=artifact["resid_std"],
    )

    transformer_correction = (
        forecast - persistence
    )

    return (
        forecast,
        persistence,
        transformer_correction,
    )


def self_check(
    artifact,
    fold,
):
    """
    Reload the model and verify the epoch-0 deployment invariant.

    This check uses development-fold arrays only. It does not open or
    evaluate the locked test run.
    """
    model = build_model(artifact)

    correction_scaled = TR.predict_scaled_correction(
        model=model,
        X=fold["X_val"],
        device=torch.device("cpu"),
        batch_size=64,
    )

    forecast = T.correction_to_forecast(
        correction_scaled=correction_scaled,
        base=fold["base_val"],
        resid_mean=artifact["resid_mean"],
        resid_std=artifact["resid_std"],
    )

    if not np.isfinite(forecast).all():
        raise RuntimeError(
            "Self-check produced non-finite forecasts."
        )

    if not np.allclose(
        correction_scaled,
        0.0,
    ):
        raise RuntimeError(
            "Epoch-0 Transformer produced a non-zero "
            "scaled correction."
        )

    if not np.allclose(
        forecast,
        fold["base_val"],
    ):
        raise RuntimeError(
            "Epoch-0 forecast does not equal persistence."
        )

    return {
        "windows_checked": int(len(forecast)),
        "max_abs_scaled_correction": float(
            np.abs(correction_scaled).max()
        ),
        "forecast_equals_persistence": bool(
            np.allclose(
                forecast,
                fold["base_val"],
            )
        ),
    }


def main():
    artifact, fold = build_artifact()

    path = save_artifact(
        artifact=artifact,
    )

    # Reload the serialized file rather than checking only the
    # in-memory dictionary. This verifies serialization as well.
    loaded = load_artifact(
        path=path,
    )

    checks = self_check(
        artifact=loaded,
        fold=fold,
    )

    print(
        f"Saved {path} "
        f"({path.stat().st_size / 1024:.1f} KB)"
    )

    print(
        "Self-check passed "
        "(development data only)."
    )

    print(
        "  windows checked             :",
        checks["windows_checked"],
    )

    print(
        "  max |scaled correction|     :",
        checks["max_abs_scaled_correction"],
    )

    print(
        "  forecast == persistence     :",
        checks["forecast_equals_persistence"],
    )

    print(
        "  selected_epoch              :",
        loaded["selected_epoch"],
    )

    print(
        "  context_steps               :",
        loaded["context_steps"],
    )

    print(
        "  horizon_steps               :",
        loaded["horizon_steps"],
    )

    print(
        "  reference_context           :",
        loaded["reference_context"],
    )

    print(
        "  model_kwargs                :",
        loaded["model_kwargs"],
    )

    print(
        "  architecture_defaults       :",
        loaded["architecture_defaults"],
    )

    print(
        "  scaler columns              :",
        len(
            loaded["feature_scaler"]["columns"]
        ),
    )

    print(
        "  development runs            :",
        loaded["development_runs"],
    )

    print(
        "  locked test loaded for fit  :",
        loaded["locked_test_run"]
        in loaded["development_runs"],
    )

    print(
        "  resid_std                   :",
        np.round(
            loaded["resid_std"],
            4,
        ),
    )

    print(
        "  persistence_fallback        :",
        np.round(
            loaded["persistence_fallback"],
            4,
        ),
    )

    print(
        "  raw columns                 :",
        len(
            loaded["raw_columns"]
        ),
    )


if __name__ == "__main__":
    main()
