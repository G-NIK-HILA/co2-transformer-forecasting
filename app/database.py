"""PostgreSQL access for the forecasting service.

Read access to runs and sensor readings. Rows are mapped back to the
original Excel column names using the model artifact's raw_columns, so the
research feature code (src.data) can be reused unchanged.
"""

import os
import time

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql


DEFAULT_DATABASE_URL = "postgresql://co2:co2@localhost:5432/co2"


def get_database_url():
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def connect(wait_seconds=0, database_url=None):
    """Open an autocommit connection, retrying for up to wait_seconds."""
    url = database_url or get_database_url()
    deadline = time.monotonic() + max(0, wait_seconds)

    while True:
        try:
            connection = psycopg2.connect(url)
            connection.autocommit = True
            return connection
        except psycopg2.OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


def _run_record(row):
    return {
        "run_id": row[0],
        "split": row[1],
        "source_file": row[2],
        "n_rows": int(row[3]),
        "started_at": row[4],
        "ended_at": row[5],
    }


def get_run(connection, run_id):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT run_id, split, source_file, n_rows, started_at, ended_at
            FROM runs
            WHERE run_id = %s
            """,
            (run_id,),
        )
        row = cursor.fetchone()

    return None if row is None else _run_record(row)


def list_run_records(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT run_id, split, source_file, n_rows, started_at, ended_at
            FROM runs
            ORDER BY run_id
            """
        )
        rows = cursor.fetchall()

    return [_run_record(row) for row in rows]


def row_for_timestamp(connection, run_id, timestamp):
    """Latest stored row at or before the timestamp, or None."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT MAX(row_idx)
            FROM sensor_readings
            WHERE run_id = %s AND ts <= %s
            """,
            (run_id, timestamp),
        )
        value = cursor.fetchone()[0]

    return None if value is None else int(value)


def rows_in_window(connection, run_id, start_timestamp, end_timestamp):
    """Stored row indices with start <= ts <= end, in order."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT row_idx
            FROM sensor_readings
            WHERE run_id = %s AND ts BETWEEN %s AND %s
            ORDER BY row_idx
            """,
            (run_id, start_timestamp, end_timestamp),
        )
        rows = cursor.fetchall()

    return [int(row[0]) for row in rows]


def fetch_frame(connection, run_id, last_row, raw_columns):
    """
    Rows 0..last_row of a run as a dataframe with the original column names
    and order (timestamp, 90 sensors, label) and a 0..n-1 index.
    """
    if raw_columns[0] != "timestamp" or raw_columns[-1] != "label":
        raise ValueError(
            "raw_columns must start with timestamp and end with label."
        )

    sensors = list(raw_columns[1:-1])

    select_columns = [
        sql.Identifier("row_idx"),
        sql.Identifier("ts"),
        sql.Identifier("sampling_label"),
        *[sql.Identifier(name.lower()) for name in sensors],
    ]

    query = sql.SQL(
        """
        SELECT {columns}
        FROM sensor_readings
        WHERE run_id = %s AND row_idx <= %s
        ORDER BY row_idx
        """
    ).format(columns=sql.SQL(", ").join(select_columns))

    with connection.cursor() as cursor:
        cursor.execute(query, (run_id, int(last_row)))
        rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(
            f"No sensor readings found for run {run_id!r} through row {last_row}."
        )

    stored = pd.DataFrame(
        rows,
        columns=["row_idx", "timestamp", "label", *sensors],
    )

    if not np.array_equal(
        stored["row_idx"].to_numpy(dtype=int),
        np.arange(len(stored)),
    ):
        raise RuntimeError(
            f"Stored rows for {run_id} are not contiguous from 0."
        )

    if int(stored["row_idx"].iloc[-1]) != int(last_row):
        raise RuntimeError(
            f"Requested row {last_row}, but database returned through "
            f"row {int(stored['row_idx'].iloc[-1])}."
        )

    data = {
        "timestamp": pd.to_datetime(
            stored["timestamp"]
        ).to_numpy()
    }

    for name in sensors:
        data[name] = pd.to_numeric(
            stored[name],
            errors="coerce",
        ).to_numpy(dtype="float64")

    data["label"] = stored["label"].to_numpy(dtype=int)

    frame = pd.DataFrame(
        data,
        index=pd.RangeIndex(len(stored)),
    )

    return frame[list(raw_columns)]
