"""End-to-end API verification against live PostgreSQL.

The check uses FastAPI TestClient, so no separate Uvicorn server is needed.

Only development run 140313_1 is used for prediction checks. The locked test
run is not requested for forecasting. No future target is read and no error
metric is computed.

Run:

    python -m app.api_check
"""

import sys
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from app import database as db
from app.main import app


RUN = "140313_1"

FORBIDDEN_KEYS = {
    "error",
    "mae",
    "rmse",
    "actual",
    "actuals",
    "y_true",
}

results = []


def check(name, condition, detail=""):
    """Record and print one API verification result."""

    passed = bool(condition)
    results.append((name, passed))

    status = "PASS" if passed else "FAIL"

    if detail and not passed:
        print(f"[{status}] {name}  {detail}")
    else:
        print(f"[{status}] {name}")


def keys_anywhere(value):
    """Recursively collect dictionary keys from a JSON-like object."""

    found = set()

    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key).lower())
            found |= keys_anywhere(item)

    elif isinstance(value, list):
        for item in value:
            found |= keys_anywhere(item)

    return found


def main():
    connection = db.connect()

    try:
        with TestClient(app) as client:

            # ----------------------------------------------------------
            # Health
            # ----------------------------------------------------------

            response = client.get("/health")

            check(
                "GET /health returns 200 and database ok",
                (
                    response.status_code == 200
                    and response.json().get("database") == "ok"
                ),
                response.text,
            )

            # ----------------------------------------------------------
            # Model metadata
            # ----------------------------------------------------------

            response = client.get("/model")
            model_body = response.json()

            check(
                "GET /model reports epoch 0, L=6, h=6, 110 features",
                (
                    response.status_code == 200
                    and model_body.get("selected_epoch") == 0
                    and model_body.get("context_steps") == 6
                    and model_body.get("horizon_steps") == 6
                    and model_body.get("n_features") == 110
                ),
                response.text,
            )

            # ----------------------------------------------------------
            # Stored runs
            # ----------------------------------------------------------

            response = client.get("/runs")
            runs = response.json()

            if isinstance(runs, list):
                splits = [
                    item.get("split")
                    for item in runs
                ]
            else:
                splits = []

            check(
                "GET /runs lists 8 runs: 7 development and 1 test",
                (
                    response.status_code == 200
                    and isinstance(runs, list)
                    and len(runs) == 8
                    and splits.count("development") == 7
                    and splits.count("test") == 1
                ),
                response.text,
            )

            # ----------------------------------------------------------
            # Prediction by origin row
            # ----------------------------------------------------------

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "origin_row": 100,
                    "log_predictions": False,
                },
            )

            body = response.json()

            predictions = (
                body.get("predictions", [])
                if isinstance(body, dict)
                else []
            )

            prediction = (
                predictions[0]
                if predictions
                else {}
            )

            check(
                "POST /predict origin_row=100 returns one prediction",
                (
                    response.status_code == 200
                    and len(predictions) == 1
                ),
                response.text,
            )

            check(
                "target_row is origin_row + 6",
                prediction.get("target_row") == 106,
            )

            check(
                "six sampling points are returned",
                sorted(
                    prediction.get(
                        "forecast",
                        {},
                    )
                )
                == [
                    f"point_{point}"
                    for point in range(1, 7)
                ],
            )

            check(
                "forecast equals persistence for selected epoch 0",
                (
                    prediction.get("forecast")
                    == prediction.get("persistence")
                ),
            )

            correction = prediction.get(
                "transformer_correction",
                {},
            )

            check(
                "Transformer correction is exactly zero",
                (
                    len(correction) == 6
                    and all(
                        value == 0.0
                        for value in correction.values()
                    )
                ),
            )

            check(
                "log_predictions=false writes no audit row",
                body.get("logged") == 0,
            )

            check(
                "prediction response has no actual/error/metric fields",
                not (
                    keys_anywhere(body)
                    & FORBIDDEN_KEYS
                ),
            )

            notice = body.get("notice", "")

            check(
                "response notice discloses persistence and no error computation",
                (
                    "persistence" in notice
                    and "no error" in notice
                ),
            )

            if not prediction:
                raise RuntimeError(
                    "Cannot continue API check because the first "
                    "prediction request failed."
                )

            timestamp_100 = prediction[
                "origin_timestamp"
            ]

            # ----------------------------------------------------------
            # Prediction by timestamp
            # ----------------------------------------------------------

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "timestamp": timestamp_100,
                    "log_predictions": False,
                },
            )

            timestamp_body = response.json()

            timestamp_predictions = (
                timestamp_body.get(
                    "predictions",
                    [],
                )
                if isinstance(
                    timestamp_body,
                    dict,
                )
                else []
            )

            check(
                "timestamp at row 100 resolves to row 100 with identical forecast",
                (
                    response.status_code == 200
                    and len(timestamp_predictions) == 1
                    and timestamp_predictions[0].get(
                        "origin_row"
                    )
                    == 100
                    and timestamp_predictions[0].get(
                        "forecast"
                    )
                    == prediction.get("forecast")
                ),
                response.text,
            )

            later = (
                datetime.fromisoformat(
                    timestamp_100
                )
                + timedelta(seconds=10)
            ).isoformat()

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "timestamp": later,
                    "log_predictions": False,
                },
            )

            later_body = response.json()

            later_predictions = (
                later_body.get(
                    "predictions",
                    [],
                )
                if isinstance(
                    later_body,
                    dict,
                )
                else []
            )

            check(
                "timestamp 10 seconds later uses latest row at or before it",
                (
                    response.status_code == 200
                    and len(later_predictions) == 1
                    and later_predictions[0].get(
                        "origin_row"
                    )
                    == 100
                ),
                response.text,
            )

            # ----------------------------------------------------------
            # Prediction over timestamp interval
            # ----------------------------------------------------------

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "origin_row": 110,
                    "log_predictions": False,
                },
            )

            row_110_body = response.json()

            row_110_predictions = (
                row_110_body.get(
                    "predictions",
                    [],
                )
                if isinstance(
                    row_110_body,
                    dict,
                )
                else []
            )

            if not row_110_predictions:
                raise RuntimeError(
                    "Could not obtain timestamp for row 110."
                )

            timestamp_110 = row_110_predictions[
                0
            ]["origin_timestamp"]

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "start_timestamp": timestamp_100,
                    "end_timestamp": timestamp_110,
                    "log_predictions": False,
                },
            )

            window_body = response.json()

            window_predictions = (
                window_body.get(
                    "predictions",
                    [],
                )
                if isinstance(
                    window_body,
                    dict,
                )
                else []
            )

            origins = [
                item.get("origin_row")
                for item in window_predictions
            ]

            check(
                "window from rows 100 through 110 returns 11 ordered predictions",
                (
                    response.status_code == 200
                    and origins
                    == list(
                        range(
                            100,
                            111,
                        )
                    )
                ),
                response.text,
            )

            # ----------------------------------------------------------
            # Error handling
            # ----------------------------------------------------------

            response = client.post(
                "/predict",
                json={
                    "run_id": "no_such_run",
                    "origin_row": 50,
                },
            )

            body_error = response.json()

            check(
                "unknown run returns 404 run_not_found",
                (
                    response.status_code == 404
                    and body_error.get(
                        "detail",
                        {},
                    ).get("code")
                    == "run_not_found"
                ),
                response.text,
            )

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "origin_row": 3,
                },
            )

            body_error = response.json()

            check(
                "origin_row=3 returns 422 insufficient_history",
                (
                    response.status_code == 422
                    and body_error.get(
                        "detail",
                        {},
                    ).get("code")
                    == "insufficient_history"
                ),
                response.text,
            )

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "origin_row": 99999,
                },
            )

            body_error = response.json()

            check(
                "out-of-range origin returns 422 row_out_of_range",
                (
                    response.status_code == 422
                    and body_error.get(
                        "detail",
                        {},
                    ).get("code")
                    == "row_out_of_range"
                ),
                response.text,
            )

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "timestamp": (
                        "2000-01-01T00:00:00"
                    ),
                },
            )

            body_error = response.json()

            check(
                "timestamp before run returns 422 row_out_of_range",
                (
                    response.status_code == 422
                    and body_error.get(
                        "detail",
                        {},
                    ).get("code")
                    == "row_out_of_range"
                ),
                response.text,
            )

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "timestamp": (
                        "2014-03-13T12:00:00+00:00"
                    ),
                },
            )

            check(
                "timezone-aware timestamp returns 422",
                response.status_code == 422,
                response.text,
            )

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "origin_row": 50,
                    "timestamp": timestamp_100,
                },
            )

            check(
                "two prediction modes at once return 422",
                response.status_code == 422,
                response.text,
            )

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                },
            )

            check(
                "missing prediction mode returns 422",
                response.status_code == 422,
                response.text,
            )

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "start_timestamp": timestamp_110,
                    "end_timestamp": timestamp_100,
                },
            )

            check(
                "reversed timestamp window returns 422",
                response.status_code == 422,
                response.text,
            )

            # ----------------------------------------------------------
            # Audit log
            # ----------------------------------------------------------

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COALESCE(MAX(id), 0)
                    FROM forecast_log
                    """
                )

                before_id = int(
                    cursor.fetchone()[0]
                )

            response = client.post(
                "/predict",
                json={
                    "run_id": RUN,
                    "origin_row": 100,
                },
            )

            logged_body = response.json()

            check(
                "default prediction logging writes one audit row",
                (
                    response.status_code == 200
                    and logged_body.get(
                        "logged"
                    )
                    == 1
                ),
                response.text,
            )

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        origin_row,
                        target_row,
                        context_steps,
                        horizon_steps,
                        model_version,
                        selected_epoch,
                        forecast,
                        persistence,
                        correction
                    FROM forecast_log
                    WHERE id > %s
                      AND run_id = %s
                    ORDER BY id
                    """,
                    (
                        before_id,
                        RUN,
                    ),
                )

                logged_rows = cursor.fetchall()

            audit_ok = (
                len(logged_rows) == 1
                and logged_rows[0][0] == 100
                and logged_rows[0][1] == 106
                and logged_rows[0][2] == 6
                and logged_rows[0][3] == 6
                and logged_rows[0][4]
                == logged_body.get(
                    "model_version"
                )
                and logged_rows[0][5] == 0
                and logged_rows[0][6]
                == prediction.get(
                    "forecast"
                )
                and logged_rows[0][7]
                == prediction.get(
                    "persistence"
                )
                and all(
                    value == 0.0
                    for value in logged_rows[
                        0
                    ][8].values()
                )
            )

            check(
                "forecast_log row matches API response",
                audit_ok,
                str(logged_rows),
            )

            # Remove only rows created after this check's marker.
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM forecast_log
                    WHERE id > %s
                      AND run_id = %s
                    """,
                    (
                        before_id,
                        RUN,
                    ),
                )

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM forecast_log
                    WHERE id > %s
                      AND run_id = %s
                    """,
                    (
                        before_id,
                        RUN,
                    ),
                )

                remaining = int(
                    cursor.fetchone()[0]
                )

            check(
                "API-check audit rows are cleaned up",
                remaining == 0,
            )

    finally:
        connection.close()

    failed = [
        name
        for name, passed in results
        if not passed
    ]

    print()
    print(
        f"{len(results) - len(failed)} "
        f"of {len(results)} checks passed."
    )

    if failed:
        print("API CHECK FAILED:")

        for name in failed:
            print(f"  {name}")

        sys.exit(1)

    print(
        "API CHECK PASSED "
        "(development run only; "
        "locked test run not requested)."
    )


if __name__ == "__main__":
    main()
