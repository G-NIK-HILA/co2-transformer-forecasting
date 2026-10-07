"""Forecasting service: PostgreSQL rows -> 110 causal features -> Transformer.

For each requested origin row the service:

1. reads the run's rows from row 0 up to the latest requested origin
   and never reads later rows;
2. builds the research features with src.data.make_model_features;
3. takes the final context_steps feature rows ending at the origin;
4. applies the stored scaler and runs the real CO2Transformer;
5. returns forecast = persistence + Transformer correction for the six
   sampling points, horizon_steps rows after the origin.

The service never reads actual future CO2 targets and never computes an
error. The deployed model currently has selected_epoch == 0, so the
Transformer correction is exactly zero and the selected forecast equals
the causal persistence baseline.
"""

import os
from pathlib import Path

import numpy as np

from app import database as db
from src import artifact as A
from src import data as D


class ServiceError(Exception):
    code = "service_error"


class RunNotFound(ServiceError):
    code = "run_not_found"


class RowOutOfRange(ServiceError):
    code = "row_out_of_range"


class InsufficientHistory(ServiceError):
    code = "insufficient_history"


class IncompleteWindow(ServiceError):
    code = "incomplete_window"


def _point_dict(values):
    values = np.asarray(values, dtype=float)

    if values.shape != (6,):
        raise ValueError(
            f"Expected six sampling-point values, found {values.shape}."
        )

    return {
        f"point_{k + 1}": float(value)
        for k, value in enumerate(values)
    }


class ForecastService:
    def __init__(self, artifact_path=None):
        path = (
            artifact_path
            or os.environ.get("MODEL_ARTIFACT_PATH")
            or A.ARTIFACT_PATH
        )

        self.artifact_path = Path(path)
        self.artifact = A.load_artifact(self.artifact_path)
        self.model = A.build_model(self.artifact)

        self.columns = list(
            self.artifact["feature_scaler"]["columns"]
        )
        self.raw_columns = list(
            self.artifact["raw_columns"]
        )

        self.context_steps = int(
            self.artifact["context_steps"]
        )
        self.horizon_steps = int(
            self.artifact["horizon_steps"]
        )
        self.reference_context = int(
            self.artifact["reference_context"]
        )
        self.model_version = str(
            self.artifact["model_version"]
        )
        self.selected_epoch = int(
            self.artifact["selected_epoch"]
        )

        if self.context_steps != int(
            self.artifact["model_kwargs"]["max_length"]
        ):
            raise RuntimeError(
                "Artifact context_steps does not match model max_length."
            )

        if len(self.columns) != 110:
            raise RuntimeError(
                f"Expected 110 model features, found {len(self.columns)}."
            )

        if len(self.raw_columns) != 92:
            raise RuntimeError(
                f"Expected 92 raw columns, found {len(self.raw_columns)}."
            )

        intentional = {
            *[
                f"last_co2_point_{p}"
                for p in range(1, 7)
            ],
            *[
                f"age_point_{p}"
                for p in range(1, 7)
            ],
        }

        self.required_complete = [
            column
            for column in self.columns
            if column not in intentional
        ]

        self.memory_indices = [
            self.columns.index(
                f"last_co2_point_{p}"
            )
            for p in range(1, 7)
        ]

    def model_info(self):
        return {
            "model_version": self.model_version,
            "selected_epoch": self.selected_epoch,
            "context_steps": self.context_steps,
            "horizon_steps": self.horizon_steps,
            "reference_context": self.reference_context,
            "n_features": len(self.columns),
            "n_targets": 6,
            "development_runs": list(
                self.artifact["development_runs"]
            ),
        }

    def _features(self, frame):
        features = D.make_model_features(frame)

        actual_columns = list(features.columns)

        if set(actual_columns) != set(self.columns):
            missing = sorted(
                set(self.columns).difference(actual_columns)
            )
            unexpected = sorted(
                set(actual_columns).difference(self.columns)
            )

            raise RuntimeError(
                "Service features do not match artifact columns. "
                f"Missing={missing}, unexpected={unexpected}."
            )

        return features[self.columns]

    def predict_origins(
        self,
        connection,
        run_id,
        origins,
        strict=True,
    ):
        """
        Forecast from each origin row, where the origin is the last
        observation available to the forecast.

        strict=True raises on the first unusable origin.
        strict=False skips unusable origins and records the reason.
        """
        run = db.get_run(connection, run_id)

        if run is None:
            raise RunNotFound(
                f"Run {run_id!r} was not found."
            )

        try:
            origins = sorted(
                {
                    int(origin)
                    for origin in origins
                }
            )
        except (TypeError, ValueError) as exc:
            raise RowOutOfRange(
                "Origin rows must be integers."
            ) from exc

        if not origins:
            raise RowOutOfRange(
                "No origin rows were requested."
            )

        for origin in origins:
            if origin < 0 or origin >= run["n_rows"]:
                raise RowOutOfRange(
                    f"Origin row {origin} is outside "
                    f"0..{run['n_rows'] - 1} for run {run_id!r}."
                )

        frame = db.fetch_frame(
            connection,
            run_id,
            origins[-1],
            self.raw_columns,
        )

        features = self._features(frame)

        windows = []
        kept = []
        skipped = []

        for origin in origins:
            start = (
                origin
                - self.context_steps
                + 1
            )

            try:
                if start < 0:
                    raise InsufficientHistory(
                        f"Origin row {origin} needs "
                        f"{self.context_steps} rows of history; "
                        f"only {origin + 1} are available."
                    )

                window = features.iloc[
                    start: origin + 1
                ]

                if len(window) != self.context_steps:
                    raise InsufficientHistory(
                        f"Origin row {origin} did not produce "
                        f"{self.context_steps} context rows."
                    )

                bad = (
                    window[self.required_complete]
                    .isna()
                    .any()
                )

                if bool(bad.any()):
                    bad_columns = list(
                        bad.index[bad]
                    )

                    raise IncompleteWindow(
                        "Missing required sensor values in: "
                        + ", ".join(
                            bad_columns[:10]
                        )
                    )

            except ServiceError as exc:
                if strict:
                    raise

                skipped.append(
                    {
                        "origin_row": origin,
                        "reason": exc.code,
                        "detail": str(exc),
                    }
                )
                continue

            windows.append(
                window.to_numpy(
                    dtype="float32"
                )
            )
            kept.append(origin)

        predictions = []

        if windows:
            X_raw = np.stack(
                windows,
                axis=0,
            )

            (
                forecast,
                persistence,
                correction,
            ) = A.predict_from_raw_windows(
                self.artifact,
                self.model,
                X_raw,
            )

            if not np.isfinite(forecast).all():
                raise RuntimeError(
                    "Model produced a non-finite forecast."
                )

            if not np.isfinite(persistence).all():
                raise RuntimeError(
                    "Persistence baseline contains non-finite values."
                )

            if not np.isfinite(correction).all():
                raise RuntimeError(
                    "Transformer correction contains non-finite values."
                )

            if (
                self.selected_epoch == 0
                and not np.allclose(
                    correction,
                    0.0,
                    rtol=0.0,
                    atol=0.0,
                )
            ):
                raise RuntimeError(
                    "Epoch-0 artifact produced a non-zero "
                    "Transformer correction."
                )

            if (
                self.selected_epoch == 0
                and not np.allclose(
                    forecast,
                    persistence,
                    rtol=1e-7,
                    atol=1e-8,
                )
            ):
                raise RuntimeError(
                    "Epoch-0 forecast does not equal persistence."
                )

            timestamps = frame["timestamp"]

            for i, origin in enumerate(kept):
                fallback_points = [
                    point + 1
                    for point, feature_index
                    in enumerate(
                        self.memory_indices
                    )
                    if np.isnan(
                        X_raw[
                            i,
                            -1,
                            feature_index,
                        ]
                    )
                ]

                predictions.append(
                    {
                        "origin_row": origin,
                        "origin_timestamp": (
                            timestamps
                            .iloc[origin]
                            .isoformat()
                        ),
                        "target_row": (
                            origin
                            + self.horizon_steps
                        ),
                        "forecast": _point_dict(
                            forecast[i]
                        ),
                        "persistence": _point_dict(
                            persistence[i]
                        ),
                        "transformer_correction": _point_dict(
                            correction[i]
                        ),
                        "persistence_fallback_points": (
                            fallback_points
                        ),
                    }
                )

        timestamp_deltas = (
            frame["timestamp"]
            .diff()
            .dt.total_seconds()
            .dropna()
        )

        if timestamp_deltas.empty:
            median_step_seconds = None
        else:
            median_step_seconds = float(
                timestamp_deltas.median()
            )

        return {
            "run_id": run_id,
            "split": run["split"],
            **self.model_info(),
            "median_step_seconds": median_step_seconds,
            "approx_horizon_seconds": (
                None
                if median_step_seconds is None
                else (
                    median_step_seconds
                    * self.horizon_steps
                )
            ),
            "predictions": predictions,
            "skipped": skipped,
        }

    def predict_at_timestamp(
        self,
        connection,
        run_id,
        timestamp,
    ):
        """Forecast using the latest stored row at or before timestamp."""
        if db.get_run(
            connection,
            run_id,
        ) is None:
            raise RunNotFound(
                f"Run {run_id!r} was not found."
            )

        origin = db.row_for_timestamp(
            connection,
            run_id,
            timestamp,
        )

        if origin is None:
            raise RowOutOfRange(
                f"No observation at or before "
                f"{timestamp!r} in run {run_id!r}."
            )

        return self.predict_origins(
            connection,
            run_id,
            [origin],
            strict=True,
        )

    def predict_window(
        self,
        connection,
        run_id,
        start_timestamp,
        end_timestamp,
    ):
        """Forecast from every stored observation in a timestamp interval."""
        if db.get_run(
            connection,
            run_id,
        ) is None:
            raise RunNotFound(
                f"Run {run_id!r} was not found."
            )

        origins = db.rows_in_window(
            connection,
            run_id,
            start_timestamp,
            end_timestamp,
        )

        if not origins:
            raise RowOutOfRange(
                f"No observations between "
                f"{start_timestamp!r} and "
                f"{end_timestamp!r} in run {run_id!r}."
            )

        return self.predict_origins(
            connection,
            run_id,
            origins,
            strict=False,
        )
