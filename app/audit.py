"""Prediction audit logging.

Each returned forecast can be written to forecast_log. The audit record
contains the forecast, persistence baseline, and Transformer correction.

No future target measurement is read here and no forecast error is computed.
"""

from psycopg2.extras import Json, execute_values


def log_predictions(connection, result):
    """Insert one forecast_log row per returned prediction.

    Parameters
    ----------
    connection
        Open psycopg2 connection.
    result
        Dictionary returned by ForecastService.

    Returns
    -------
    int
        Number of inserted audit rows.
    """

    rows = [
        (
            result["run_id"],
            int(prediction["origin_row"]),
            int(prediction["target_row"]),
            int(result["context_steps"]),
            int(result["horizon_steps"]),
            result["model_version"],
            int(result["selected_epoch"]),
            Json(prediction["forecast"]),
            Json(prediction["persistence"]),
            Json(prediction["transformer_correction"]),
        )
        for prediction in result["predictions"]
    ]

    if not rows:
        return 0

    with connection.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO forecast_log (
                run_id,
                origin_row,
                target_row,
                context_steps,
                horizon_steps,
                model_version,
                selected_epoch,
                forecast,
                persistence,
                correction
            )
            VALUES %s
            """,
            rows,
        )

    return len(rows)
