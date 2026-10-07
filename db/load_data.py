"""Load the CO2 experiment data into PostgreSQL.

This module deliberately reuses the research pipeline's
src.data.list_runs() and src.data.load_run() functions so that database
ingestion uses the same raw-data interpretation as model development.

The loader:
    * loads all experiment runs;
    * labels the locked test run as "test";
    * preserves row order, timestamps, sampling labels, and all 90 sensors;
    * supports safe/idempotent reloads;
    * optionally waits for PostgreSQL to become ready;
    * validates stored values against the source Excel data;
    * reports observed timestamp spacing for each run.

Loading the locked test run into PostgreSQL is a deployment operation.
This script does not train on it, tune on it, or calculate test metrics.

Default connection:
    postgresql://co2:co2@localhost:5432/co2

The DATABASE_URL environment variable can override the default.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import TEST_RUN, list_runs, load_run


DEFAULT_DATABASE_URL = "postgresql://co2:co2@localhost:5432/co2"

EXPECTED_SENSOR_COUNT = 90
EXPECTED_TEST_RUN_COUNT = 1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Load the raw CO2 experiment runs into PostgreSQL and "
            "validate the stored data against the source files."
        )
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help=(
            "Reload every discovered run. Existing sensor rows for each "
            "run are replaced transactionally."
        ),
    )

    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        metavar="SECONDS",
        help=(
            "Wait up to SECONDS for PostgreSQL to become available. "
            "Useful when the loader starts with Docker Compose."
        ),
    )

    return parser.parse_args()


def get_database_url():
    """Return the PostgreSQL connection URL."""
    return os.environ.get(
        "DATABASE_URL",
        DEFAULT_DATABASE_URL,
    )


def connect_with_retry(database_url, wait_seconds):
    """Connect to PostgreSQL, optionally retrying until the timeout."""
    deadline = time.monotonic() + max(0, wait_seconds)
    last_error = None

    while True:
        try:
            return psycopg2.connect(database_url)

        except psycopg2.OperationalError as exc:
            last_error = exc

            if time.monotonic() >= deadline:
                break

            remaining = max(
                0,
                int(round(deadline - time.monotonic())),
            )

            print(
                "PostgreSQL is not ready yet; "
                f"retrying... ({remaining}s remaining)"
            )

            time.sleep(2)

    raise RuntimeError(
        "Could not connect to PostgreSQL. "
        f"DATABASE_URL={database_url!r}"
    ) from last_error


def apply_schema(connection):
    """Create the database schema from db/init.sql."""
    schema_path = ROOT / "db" / "init.sql"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}"
        )

    schema_text = schema_path.read_text(encoding="utf-8")

    with connection.cursor() as cursor:
        cursor.execute(schema_text)

    connection.commit()


def get_sensor_columns(df):
    """Return source and SQL names for the 90 raw sensor columns."""
    sensor_columns = [
        column
        for column in df.columns
        if column not in {"timestamp", "label"}
    ]

    if len(sensor_columns) != EXPECTED_SENSOR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SENSOR_COUNT} sensor columns, "
            f"found {len(sensor_columns)}."
        )

    sql_sensor_columns = [
        column.lower()
        for column in sensor_columns
    ]

    if len(set(sql_sensor_columns)) != len(sql_sensor_columns):
        raise RuntimeError(
            "Sensor names are not unique after lowercasing."
        )

    return sensor_columns, sql_sensor_columns


def validate_source_run(run_name, df):
    """Validate basic structural assumptions before database insertion."""
    if df.empty:
        raise RuntimeError(
            f"Run {run_name} contains no observations."
        )

    required_columns = {"timestamp", "label"}

    missing = required_columns.difference(df.columns)

    if missing:
        raise RuntimeError(
            f"Run {run_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if df["timestamp"].isna().any():
        raise RuntimeError(
            f"Run {run_name} contains missing timestamps."
        )

    if df["timestamp"].duplicated().any():
        raise RuntimeError(
            f"Run {run_name} contains duplicate timestamps."
        )

    if not df["timestamp"].is_monotonic_increasing:
        raise RuntimeError(
            f"Run {run_name} timestamps are not increasing."
        )

    labels = set(
        int(value)
        for value in df["label"].dropna().unique()
    )

    if not labels.issubset({1, 2, 3, 4, 5, 6}):
        raise RuntimeError(
            f"Run {run_name} contains invalid sampling labels: "
            f"{sorted(labels)}"
        )

    get_sensor_columns(df)


def to_database_value(value):
    """Convert pandas/NumPy scalar values to DB-compatible values."""
    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


def print_time_step_report(run_name, df):
    """Print observed timestamp-spacing diagnostics for one run."""
    deltas = (
        df["timestamp"]
        .sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    if deltas.empty:
        print(
            f"  Time step {run_name}: "
            "not available (fewer than 2 observations)"
        )
        return

    print(
        f"  Time step {run_name}: "
        f"min={deltas.min():.1f}s, "
        f"median={deltas.median():.1f}s, "
        f"max={deltas.max():.1f}s"
    )


def run_exists(connection, run_name):
    """Return True when a run is already present and complete."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                r.n_rows,
                COUNT(s.row_idx)
            FROM runs AS r
            LEFT JOIN sensor_readings AS s
                ON s.run_id = r.run_id
            WHERE r.run_id = %s
            GROUP BY r.n_rows
            """,
            (run_name,),
        )

        result = cursor.fetchone()

    if result is None:
        return False

    expected_rows, stored_rows = result

    return int(expected_rows) == int(stored_rows)


def build_records(run_name, df, sensor_columns):
    """Convert one source dataframe to sensor_readings records."""
    records = []

    for row_idx, row in df.iterrows():
        record = [
            run_name,
            int(row_idx),
            row["timestamp"].to_pydatetime(),
            int(row["label"]),
        ]

        record.extend(
            to_database_value(row[column])
            for column in sensor_columns
        )

        records.append(tuple(record))

    return records


def load_run_into_database(
    connection,
    run_name,
    source_path,
):
    """Replace one run atomically in PostgreSQL."""
    df = load_run(source_path)

    validate_source_run(run_name, df)

    sensor_columns, sql_sensor_columns = get_sensor_columns(df)

    split = (
        "test"
        if run_name == TEST_RUN
        else "development"
    )

    run_metadata = (
        run_name,
        split,
        Path(source_path).name,
        int(len(df)),
        df["timestamp"].min().to_pydatetime(),
        df["timestamp"].max().to_pydatetime(),
    )

    insert_columns = [
        "run_id",
        "row_idx",
        "ts",
        "sampling_label",
        *sql_sensor_columns,
    ]

    records = build_records(
        run_name,
        df,
        sensor_columns,
    )

    insert_query = sql.SQL(
        "INSERT INTO sensor_readings ({columns}) VALUES %s"
    ).format(
        columns=sql.SQL(", ").join(
            sql.Identifier(column)
            for column in insert_columns
        )
    )

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO runs (
                    run_id,
                    split,
                    source_file,
                    n_rows,
                    started_at,
                    ended_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id)
                DO UPDATE SET
                    split = EXCLUDED.split,
                    source_file = EXCLUDED.source_file,
                    n_rows = EXCLUDED.n_rows,
                    started_at = EXCLUDED.started_at,
                    ended_at = EXCLUDED.ended_at,
                    loaded_at = CURRENT_TIMESTAMP
                """,
                run_metadata,
            )

            cursor.execute(
                """
                DELETE FROM sensor_readings
                WHERE run_id = %s
                """,
                (run_name,),
            )

            execute_values(
                cursor,
                insert_query.as_string(connection),
                records,
                page_size=500,
            )

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM sensor_readings
                WHERE run_id = %s
                """,
                (run_name,),
            )

            stored_count = int(cursor.fetchone()[0])

            if stored_count != len(df):
                raise RuntimeError(
                    f"Row-count validation failed for {run_name}: "
                    f"source={len(df)}, "
                    f"database={stored_count}."
                )

    return df


def fetch_run_from_database(
    connection,
    run_name,
    sensor_columns,
):
    """Read one stored run back in source-data column order."""
    sql_sensor_columns = [
        column.lower()
        for column in sensor_columns
    ]

    select_columns = [
        sql.Identifier("row_idx"),
        sql.Identifier("ts"),
        sql.Identifier("sampling_label"),
        *[
            sql.Identifier(column)
            for column in sql_sensor_columns
        ],
    ]

    query = sql.SQL(
        """
        SELECT {columns}
        FROM sensor_readings
        WHERE run_id = %s
        ORDER BY row_idx
        """
    ).format(
        columns=sql.SQL(", ").join(select_columns)
    )

    with connection.cursor() as cursor:
        cursor.execute(query, (run_name,))
        rows = cursor.fetchall()

    columns = [
        "row_idx",
        "timestamp",
        "label",
        *sensor_columns,
    ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def verify_run_round_trip(
    connection,
    run_name,
    source_df,
):
    """Compare PostgreSQL values with the original loaded dataframe."""
    sensor_columns, _ = get_sensor_columns(source_df)

    stored_df = fetch_run_from_database(
        connection,
        run_name,
        sensor_columns,
    )

    if len(stored_df) != len(source_df):
        raise RuntimeError(
            f"Round-trip row-count mismatch for {run_name}: "
            f"source={len(source_df)}, "
            f"database={len(stored_df)}."
        )

    expected_row_idx = np.arange(
        len(source_df),
        dtype=int,
    )

    stored_row_idx = stored_df["row_idx"].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        stored_row_idx,
        expected_row_idx,
    ):
        raise RuntimeError(
            f"Row-index mismatch for {run_name}."
        )

    source_timestamps = pd.to_datetime(
        source_df["timestamp"]
    ).reset_index(drop=True)

    stored_timestamps = pd.to_datetime(
        stored_df["timestamp"]
    ).reset_index(drop=True)

    if not source_timestamps.equals(stored_timestamps):
        raise RuntimeError(
            f"Timestamp mismatch for {run_name}."
        )

    source_labels = source_df["label"].to_numpy(
        dtype=int
    )

    stored_labels = stored_df["label"].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        source_labels,
        stored_labels,
    ):
        raise RuntimeError(
            f"Sampling-label mismatch for {run_name}."
        )

    for column in sensor_columns:
        source_values = pd.to_numeric(
            source_df[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        stored_values = pd.to_numeric(
            stored_df[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.allclose(
            source_values,
            stored_values,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            mismatch = ~np.isclose(
                source_values,
                stored_values,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )

            first_bad = int(
                np.flatnonzero(mismatch)[0]
            )

            raise RuntimeError(
                f"Sensor mismatch for run={run_name}, "
                f"column={column}, "
                f"row={first_bad}: "
                f"source={source_values[first_bad]!r}, "
                f"database={stored_values[first_bad]!r}."
            )


def validate_database_summary(
    connection,
    runs,
):
    """Validate database-wide run, split, and observation counts."""
    expected_run_count = len(runs)

    expected_test_count = sum(
        run_name == TEST_RUN
        for run_name in runs
    )

    if expected_test_count != EXPECTED_TEST_RUN_COUNT:
        raise RuntimeError(
            "Expected exactly one locked test run in source data."
        )

    expected_development_count = (
        expected_run_count - expected_test_count
    )

    expected_total_rows = sum(
        len(load_run(path))
        for path in runs.values()
    )

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM runs"
        )
        stored_run_count = int(
            cursor.fetchone()[0]
        )

        cursor.execute(
            """
            SELECT split, COUNT(*)
            FROM runs
            GROUP BY split
            ORDER BY split
            """
        )
        split_counts = {
            split: int(count)
            for split, count in cursor.fetchall()
        }

        cursor.execute(
            "SELECT COUNT(*) FROM sensor_readings"
        )
        stored_total_rows = int(
            cursor.fetchone()[0]
        )

    if stored_run_count != expected_run_count:
        raise RuntimeError(
            "Database run-count validation failed: "
            f"expected={expected_run_count}, "
            f"database={stored_run_count}."
        )

    if (
        split_counts.get("development", 0)
        != expected_development_count
    ):
        raise RuntimeError(
            "Development-run count validation failed: "
            f"expected={expected_development_count}, "
            "database="
            f"{split_counts.get('development', 0)}."
        )

    if (
        split_counts.get("test", 0)
        != expected_test_count
    ):
        raise RuntimeError(
            "Test-run count validation failed: "
            f"expected={expected_test_count}, "
            f"database={split_counts.get('test', 0)}."
        )

    if stored_total_rows != expected_total_rows:
        raise RuntimeError(
            "Database observation-count validation failed: "
            f"expected={expected_total_rows}, "
            f"database={stored_total_rows}."
        )

    return {
        "runs": stored_run_count,
        "development_runs": split_counts.get(
            "development",
            0,
        ),
        "test_runs": split_counts.get("test", 0),
        "sensor_readings": stored_total_rows,
    }


def main():
    """Load and verify all experiment runs."""
    args = parse_args()

    runs = list_runs()

    if TEST_RUN not in runs:
        raise RuntimeError(
            f"Locked test run {TEST_RUN!r} "
            "was not discovered."
        )

    if len(runs) < 2:
        raise RuntimeError(
            "Expected development runs plus a test run."
        )

    database_url = get_database_url()

    print(f"Discovered {len(runs)} experiment runs.")
    print(f"Locked test run: {TEST_RUN}")
    print(f"Database: {database_url}")
    print()

    connection = connect_with_retry(
        database_url,
        args.wait,
    )

    try:
        apply_schema(connection)

        source_frames = {}

        for run_name in sorted(runs):
            source_path = runs[run_name]

            source_df = load_run(source_path)
            validate_source_run(
                run_name,
                source_df,
            )

            print_time_step_report(
                run_name,
                source_df,
            )

            already_loaded = run_exists(
                connection,
                run_name,
            )

            if already_loaded and not args.reload:
                print(
                    f"  {run_name}: already loaded; "
                    "verifying existing data."
                )

            else:
                source_df = load_run_into_database(
                    connection,
                    run_name,
                    source_path,
                )

                split = (
                    "test"
                    if run_name == TEST_RUN
                    else "development"
                )

                print(
                    f"  {run_name}: loaded "
                    f"{len(source_df)} rows "
                    f"[{split}]."
                )

            source_frames[run_name] = source_df

        print()
        print("Verifying database values against source data...")

        for run_name in sorted(runs):
            verify_run_round_trip(
                connection,
                run_name,
                source_frames[run_name],
            )

            print(
                f"  {run_name}: round-trip verification passed."
            )

        summary = validate_database_summary(
            connection,
            runs,
        )

        print()
        print("Database validation passed.")
        print(f"  Runs: {summary['runs']}")
        print(
            "  Development runs: "
            f"{summary['development_runs']}"
        )
        print(
            f"  Test runs: {summary['test_runs']}"
        )
        print(
            "  Sensor readings: "
            f"{summary['sensor_readings']}"
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
