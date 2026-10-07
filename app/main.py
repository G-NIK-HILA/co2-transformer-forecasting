"""FastAPI application for the CO2 forecasting service.

Run locally with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000

The API delegates forecasting to the parity-verified ForecastService.
It returns forecasts only: it does not read future CO2 targets or compute
forecast errors.
"""

from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app import audit
from app import database as db
from app.schemas import (
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    RunInfo,
)
from app.service import ForecastService, RunNotFound, ServiceError


def build_notice(service):
    """Describe the selected production procedure without hiding epoch 0."""

    base = (
        "This API returns forecasts only and computes no error against "
        "actual measurements."
    )

    if service.selected_epoch == 0:
        return (
            "The deployed Transformer was selected at epoch 0, so its "
            "learned correction is zero and the forecast equals the "
            "persistence baseline. "
            + base
        )

    return base


@asynccontextmanager
async def lifespan(app):
    """Load the model artifact once when the API starts."""

    app.state.service = ForecastService()
    yield


app = FastAPI(
    title="CO2 Pilot-Plant Forecasting Service",
    version="1.0.0",
    description=(
        "Six-point CO2 forecasts from the production-selected forecasting "
        "pipeline using PostgreSQL sensor observations."
    ),
    lifespan=lifespan,
)


@app.exception_handler(ServiceError)
async def service_error_handler(
    request: Request,
    exc: ServiceError,
):
    """Map service-domain failures to stable HTTP responses."""

    status_code = (
        404
        if isinstance(exc, RunNotFound)
        else 422
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": {
                "code": exc.code,
                "message": str(exc),
            }
        },
    )


def get_connection():
    """Provide one PostgreSQL connection per API request."""

    try:
        connection = db.connect()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "database_unavailable",
                "message": "PostgreSQL is not reachable.",
            },
        ) from exc

    try:
        yield connection
    finally:
        connection.close()


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health(request: Request):
    """Check model availability and PostgreSQL connectivity."""

    service = request.app.state.service

    try:
        connection = db.connect()

        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        finally:
            connection.close()

    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "database": "unreachable",
                "model_version": service.model_version,
                "selected_epoch": service.selected_epoch,
            },
        )

    return {
        "status": "ok",
        "database": "ok",
        "model_version": service.model_version,
        "selected_epoch": service.selected_epoch,
    }


@app.get(
    "/model",
    response_model=ModelInfo,
)
def model_info(request: Request):
    """Return the frozen production model configuration."""

    service = request.app.state.service

    return {
        **service.model_info(),
        "notice": build_notice(service),
    }


@app.get(
    "/runs",
    response_model=List[RunInfo],
)
def list_runs(
    connection=Depends(get_connection),
):
    """List runs currently stored in PostgreSQL."""

    return db.list_run_records(connection)


@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(
    body: PredictRequest,
    request: Request,
    connection=Depends(get_connection),
):
    """Forecast by origin row, timestamp, or timestamp interval."""

    service = request.app.state.service

    if body.origin_row is not None:
        result = service.predict_origins(
            connection,
            body.run_id,
            [body.origin_row],
            strict=True,
        )

    elif body.timestamp is not None:
        result = service.predict_at_timestamp(
            connection,
            body.run_id,
            body.timestamp,
        )

    else:
        result = service.predict_window(
            connection,
            body.run_id,
            body.start_timestamp,
            body.end_timestamp,
        )

    logged = 0

    if body.log_predictions:
        logged = audit.log_predictions(
            connection,
            result,
        )

    return {
        **result,
        "logged": logged,
        "notice": build_notice(service),
    }
