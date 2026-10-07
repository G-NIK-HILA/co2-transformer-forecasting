"""Request and response models for the forecasting API."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApiModel(BaseModel):
    """Base API model with project-wide Pydantic configuration."""

    model_config = ConfigDict(protected_namespaces=())


class PredictRequest(ApiModel):
    """Request one forecasting mode.

    Supply exactly one of:
      1. origin_row
      2. timestamp
      3. start_timestamp together with end_timestamp

    Source timestamps have no verified timezone, so timestamps supplied to
    this API must also be naive.
    """

    run_id: str = Field(
        ...,
        min_length=1,
        description="Stored experiment run, for example 140313_1.",
    )

    origin_row: Optional[int] = Field(
        None,
        ge=0,
        description="Last observed row available to the forecast.",
    )

    timestamp: Optional[datetime] = Field(
        None,
        description=(
            "Use the latest stored observation at or before this timestamp."
        ),
    )

    start_timestamp: Optional[datetime] = Field(
        None,
        description="Start of a closed timestamp interval of forecast origins.",
    )

    end_timestamp: Optional[datetime] = Field(
        None,
        description="End of a closed timestamp interval of forecast origins.",
    )

    log_predictions: bool = Field(
        True,
        description="Write returned predictions to forecast_log.",
    )

    @model_validator(mode="after")
    def validate_mode(self):
        window_given = (
            self.start_timestamp is not None
            or self.end_timestamp is not None
        )

        modes = [
            self.origin_row is not None,
            self.timestamp is not None,
            window_given,
        ]

        if sum(modes) != 1:
            raise ValueError(
                "Provide exactly one of origin_row, timestamp, or "
                "start_timestamp together with end_timestamp."
            )

        if window_given:
            if (
                self.start_timestamp is None
                or self.end_timestamp is None
            ):
                raise ValueError(
                    "start_timestamp and end_timestamp must be "
                    "provided together."
                )

            if self.start_timestamp > self.end_timestamp:
                raise ValueError(
                    "start_timestamp must not be after end_timestamp."
                )

        for name in (
            "timestamp",
            "start_timestamp",
            "end_timestamp",
        ):
            value = getattr(self, name)

            if value is not None and value.tzinfo is not None:
                raise ValueError(
                    f"{name} must not include a timezone: "
                    "sensor timestamps are naive and the source data "
                    "has no verified timezone."
                )

        return self


class PredictionItem(ApiModel):
    origin_row: int
    origin_timestamp: str
    target_row: int
    forecast: Dict[str, float]
    persistence: Dict[str, float]
    transformer_correction: Dict[str, float]
    persistence_fallback_points: List[int]


class SkippedItem(ApiModel):
    origin_row: int
    reason: str
    detail: str


class PredictResponse(ApiModel):
    run_id: str
    split: str

    model_version: str
    selected_epoch: int

    context_steps: int
    horizon_steps: int
    reference_context: int

    n_features: int
    n_targets: int

    development_runs: List[str]

    median_step_seconds: Optional[float]
    approx_horizon_seconds: Optional[float]

    predictions: List[PredictionItem]
    skipped: List[SkippedItem]

    logged: int = 0
    notice: str


class RunInfo(ApiModel):
    run_id: str
    split: str
    source_file: str
    n_rows: int
    started_at: datetime
    ended_at: datetime


class ModelInfo(ApiModel):
    model_version: str
    selected_epoch: int

    context_steps: int
    horizon_steps: int
    reference_context: int

    n_features: int
    n_targets: int

    development_runs: List[str]

    notice: str


class HealthResponse(ApiModel):
    status: str
    database: str
    model_version: str
    selected_epoch: int
