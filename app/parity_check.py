"""Check that the database-driven service reproduces the research pipeline.

DEVELOPMENT RUNS ONLY.

Only development file paths are ever handed to the xlsx loaders, so the
locked test run is never read from Excel here and no actual target is
compared against any forecast.

Per development run this verifies:

1. Native research feature order equals artifact feature order.
2. Database-derived features equal xlsx-derived features.
3. Truncated-history features equal full-run features at the same row,
   demonstrating that production feature construction does not depend on
   future rows.
4. For the exact final research-pipeline windows
   (L=6, h=6, reference_context=18):
       - the service serves the same origins;
       - service persistence equals research persistence;
       - epoch-0 forecast equals persistence.
5. For every window independently admissible at L=6, h=6:
       - the service serves the same origins;
       - service persistence equals research persistence;
       - epoch-0 forecast equals persistence.

Run:
    python -m app.parity_check
"""

import sys

import numpy as np

from app import database as db
from app.service import ForecastService
from src import data as D
from src import training as T


EXPECTED_FINAL_RULE_WINDOWS = 523


def close(a, b):
    return np.allclose(
        np.asarray(
            a,
            dtype="float64",
        ),
        np.asarray(
            b,
            dtype="float64",
        ),
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True,
    )


def compare_windows(
    by_origin,
    origins,
    X_windows,
    columns,
    fallback,
):
    """
    Compare service persistence against research persistence.

    Returns:
        all_origins_served,
        persistence_matches,
        forecast_is_persistence,
        correction_is_zero
    """
    origins = [
        int(origin)
        for origin in origins
    ]

    if not origins:
        return True, True, True, True

    if not all(
        origin in by_origin
        for origin in origins
    ):
        return False, False, False, False

    reference = (
        T.last_measured_persistence(
            np.asarray(
                X_windows,
                dtype="float64",
            ),
            columns,
            fallback,
        )
    )

    service_persistence = np.asarray(
        [
            [
                by_origin[origin][
                    "persistence"
                ][f"point_{point}"]
                for point in range(1, 7)
            ]
            for origin in origins
        ],
        dtype="float64",
    )

    service_forecast = np.asarray(
        [
            [
                by_origin[origin][
                    "forecast"
                ][f"point_{point}"]
                for point in range(1, 7)
            ]
            for origin in origins
        ],
        dtype="float64",
    )

    service_correction = np.asarray(
        [
            [
                by_origin[origin][
                    "transformer_correction"
                ][f"point_{point}"]
                for point in range(1, 7)
            ]
            for origin in origins
        ],
        dtype="float64",
    )

    return (
        True,
        close(
            service_persistence,
            reference,
        ),
        close(
            service_forecast,
            service_persistence,
        ),
        close(
            service_correction,
            np.zeros_like(
                service_correction
            ),
        ),
    )


def main():
    service = ForecastService()

    all_runs = D.list_runs()

    (
        development_runs,
        test_run,
    ) = D.split_run_names(
        all_runs
    )

    dev_paths = {
        name: all_runs[name]
        for name in development_runs
    }

    if test_run != D.TEST_RUN:
        raise RuntimeError(
            "Unexpected locked test run."
        )

    if test_run in dev_paths:
        raise RuntimeError(
            "Locked test run is present in development paths."
        )

    artifact_development_runs = set(
        service.artifact[
            "development_runs"
        ]
    )

    if (
        set(development_runs)
        != artifact_development_runs
    ):
        raise RuntimeError(
            "Development runs do not match "
            "the runs stored in the artifact."
        )

    L = service.context_steps
    h = service.horizon_steps

    reference_context = int(
        service.artifact[
            "reference_context"
        ]
    )

    fallback = np.asarray(
        service.artifact[
            "persistence_fallback"
        ],
        dtype="float64",
    )

    print(
        f"Model: {service.model_version} "
        f"(epoch {service.selected_epoch})"
    )

    print(
        f"Configuration: "
        f"L={L}, h={h}, "
        f"reference_context={reference_context}"
    )

    print(
        "Development runs checked: "
        f"{development_runs}"
    )

    print(
        f"Locked test run {test_run}: "
        "not read from Excel and not evaluated."
    )

    print()

    connection = db.connect()

    failures = []
    total_final_rule = 0
    total_wide_rule = 0

    try:
        for run_id in development_runs:
            try:
                record = db.get_run(
                    connection,
                    run_id,
                )

                if (
                    record is None
                    or record["split"]
                    != "development"
                ):
                    raise AssertionError(
                        "Run is missing from PostgreSQL "
                        "or is not marked development."
                    )

                # Only a development path is ever passed here.
                df = D.load_run(
                    dev_paths[run_id]
                )

                n_rows = len(df)

                native = (
                    D.make_model_features(
                        df
                    )
                )

                order_ok = (
                    list(native.columns)
                    == service.columns
                )

                if not order_ok:
                    raise AssertionError(
                        "Native research feature order "
                        "does not match artifact feature order."
                    )

                reference = native[
                    service.columns
                ]

                frame = db.fetch_frame(
                    connection,
                    run_id,
                    n_rows - 1,
                    service.raw_columns,
                )

                from_db_native = (
                    D.make_model_features(
                        frame
                    )
                )

                db_order_ok = (
                    list(
                        from_db_native.columns
                    )
                    == service.columns
                )

                if not db_order_ok:
                    raise AssertionError(
                        "Database-derived feature order "
                        "does not match artifact feature order."
                    )

                from_db = (
                    from_db_native[
                        service.columns
                    ]
                )

                features_ok = (
                    reference.shape
                    == from_db.shape
                    and close(
                        reference.to_numpy(
                            dtype="float64"
                        ),
                        from_db.to_numpy(
                            dtype="float64"
                        ),
                    )
                )

                if not features_ok:
                    raise AssertionError(
                        "Database-derived features "
                        "do not match xlsx-derived features."
                    )

                causal_ok = True

                causal_rows = sorted(
                    {
                        min(
                            L - 1,
                            n_rows - 1,
                        ),
                        n_rows // 2,
                        n_rows - 1,
                    }
                )

                for row in causal_rows:
                    truncated = (
                        D.make_model_features(
                            frame.iloc[
                                : row + 1
                            ]
                        )[
                            service.columns
                        ]
                    )

                    if not close(
                        truncated.iloc[-1]
                        .to_numpy(
                            dtype="float64"
                        ),
                        from_db.iloc[row]
                        .to_numpy(
                            dtype="float64"
                        ),
                    ):
                        causal_ok = False
                        break

                if not causal_ok:
                    raise AssertionError(
                        "Feature construction depends "
                        "on future rows."
                    )

                # ------------------------------------------------------
                # Operational service predictions
                # ------------------------------------------------------
                output = (
                    service.predict_origins(
                        connection,
                        run_id,
                        list(
                            range(n_rows)
                        ),
                        strict=False,
                    )
                )

                by_origin = {
                    prediction[
                        "origin_row"
                    ]: prediction
                    for prediction
                    in output[
                        "predictions"
                    ]
                }

                # ------------------------------------------------------
                # A. Exact final research-pipeline windows.
                #
                # build_forecast_dataset applies the frozen
                # reference_context=18 eligibility rule while the actual
                # deployed model still uses L=6.
                # ------------------------------------------------------
                (
                    X_final,
                    _y_final_not_used,
                    names_final,
                    rows_final,
                ) = D.build_forecast_dataset(
                    runs=dev_paths,
                    run_names=[
                        run_id
                    ],
                    context_steps=L,
                    horizon_steps=h,
                    reference_context=(
                        reference_context
                    ),
                )

                if not all(
                    name == run_id
                    for name in names_final
                ):
                    raise AssertionError(
                        "Unexpected run identity "
                        "in final-rule windows."
                    )

                origins_final = [
                    int(row) - h
                    for row in rows_final
                ]

                total_final_rule += len(
                    origins_final
                )

                (
                    served_final,
                    persistence_final,
                    forecast_final,
                    correction_final,
                ) = compare_windows(
                    by_origin,
                    origins_final,
                    X_final,
                    service.columns,
                    fallback,
                )

                # ------------------------------------------------------
                # B. Every L=6/h=6 window the operational service can
                # legitimately serve, independent of reference_context.
                # ------------------------------------------------------
                (
                    X_wide,
                    _y_wide_not_used,
                    rows_wide,
                ) = D.make_forecast_windows(
                    df,
                    L,
                    h,
                )

                origins_wide = [
                    int(row) - h
                    for row in rows_wide
                ]

                total_wide_rule += len(
                    origins_wide
                )

                (
                    served_wide,
                    persistence_wide,
                    forecast_wide,
                    correction_wide,
                ) = compare_windows(
                    by_origin,
                    origins_wide,
                    X_wide,
                    service.columns,
                    fallback,
                )

                ok = all(
                    [
                        order_ok,
                        db_order_ok,
                        features_ok,
                        causal_ok,
                        served_final,
                        persistence_final,
                        forecast_final,
                        correction_final,
                        served_wide,
                        persistence_wide,
                        forecast_wide,
                        correction_wide,
                    ]
                )

                print(
                    f"{run_id}: "
                    f"rows={n_rows} "
                    f"order={order_ok} "
                    f"db_order={db_order_ok} "
                    f"features_match={features_ok} "
                    f"no_future_dependence={causal_ok}"
                )

                print(
                    "    final-rule "
                    f"windows={len(origins_final)} "
                    f"served={served_final} "
                    f"persistence_match="
                    f"{persistence_final} "
                    f"forecast_is_persistence="
                    f"{forecast_final} "
                    f"correction_is_zero="
                    f"{correction_final}"
                )

                print(
                    "    L=6 operational "
                    f"windows={len(origins_wide)} "
                    f"served={served_wide} "
                    f"persistence_match="
                    f"{persistence_wide} "
                    f"forecast_is_persistence="
                    f"{forecast_wide} "
                    f"correction_is_zero="
                    f"{correction_wide}"
                )

                print(
                    "    service "
                    f"predictions="
                    f"{len(output['predictions'])} "
                    f"skipped="
                    f"{len(output['skipped'])}"
                )

                if not ok:
                    failures.append(
                        run_id
                    )

            except Exception as exc:
                failures.append(
                    f"{run_id}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    f"{run_id}: FAILED "
                    f"[{type(exc).__name__}] "
                    f"{exc}"
                )

    finally:
        connection.close()

    print()

    print(
        "Final-rule windows across "
        "development runs: "
        f"{total_final_rule} "
        f"(artifact fit used "
        f"{EXPECTED_FINAL_RULE_WINDOWS})"
    )

    print(
        "Operational L=6 windows across "
        "development runs: "
        f"{total_wide_rule}"
    )

    if (
        total_final_rule
        != EXPECTED_FINAL_RULE_WINDOWS
    ):
        failures.append(
            "final-rule window total "
            f"{total_final_rule} != "
            f"{EXPECTED_FINAL_RULE_WINDOWS}"
        )

    if failures:
        print()
        print(
            "PARITY CHECK FAILED:"
        )

        for failure in failures:
            print(
                f"  {failure}"
            )

        sys.exit(1)

    print()
    print(
        "PARITY CHECK PASSED "
        "(development runs only; "
        "test run not read)."
    )


if __name__ == "__main__":
    main()
