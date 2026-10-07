
"""
Final model training utilities.

This module separates final model preparation from locked-test evaluation.

Development procedure:
1. Select the final epoch using leave-one-run-out cross-validation across
   all development runs.
2. Fit preprocessing using all development runs.
3. Train a fresh model on all development runs for exactly the selected
   number of epochs.

The locked test run is not evaluated by prepare_final_model().
"""

import numpy as np
import torch

from src import nested as N
from src import trainer as TR
from src import training as T


def prepare_final_model(
    runs,
    cache,
    development_runs,
    locked_test_run,
    context_steps,
    horizon_steps,
    reference_context=18,
    model_kwargs=None,
    max_epochs=100,
    learning_rate=3e-4,
    weight_decay=1e-4,
    batch_size=32,
    gradient_clip=1.0,
    seed=42,
    device=None,
    verbose=True,
):
    """
    Select the final epoch using development runs only and train a fresh
    model on all development runs.

    IMPORTANT
    ---------
    This function does NOT generate predictions for the locked test run.

    The locked test run is supplied to prepare_fold only so that the final
    fold-specific preprocessing objects and test inputs can be constructed
    using statistics fitted from the development runs. No test target is
    used for epoch selection, optimization, or model selection.
    """

    if device is None:
        device = torch.device("cpu")

    development_runs = list(development_runs)

    if locked_test_run in development_runs:
        raise ValueError(
            "Locked test run must not be included in development_runs."
        )

    if locked_test_run not in runs:
        raise KeyError(
            f"Locked test run {locked_test_run!r} was not found in runs."
        )

    if len(development_runs) < 2:
        raise ValueError(
            "At least two development runs are required."
        )

    # ========================================================
    # Stage 1
    # Select the final epoch using ONLY development runs.
    #
    # Every development run becomes an inner validation run
    # once. The locked test run is not involved.
    # ========================================================

    selection = N.select_epoch_by_inner_cv(
        runs=runs,
        cache=cache,
        training_runs=development_runs,
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        reference_context=reference_context,
        model_kwargs=model_kwargs,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        gradient_clip=gradient_clip,
        seed=seed,
        device=device,
        verbose=verbose,
    )

    best_epoch = int(
        selection["best_epoch"]
    )

    if best_epoch < 0 or best_epoch > max_epochs:
        raise RuntimeError(
            f"Invalid selected epoch: {best_epoch}"
        )

    # Verify that epoch selection used exactly the development
    # runs as its inner validation runs.
    inner_val_runs = list(
        selection["inner_val_runs"]
    )

    if set(inner_val_runs) != set(development_runs):
        raise RuntimeError(
            "Final epoch selection did not use exactly the "
            "development runs as inner validation runs."
        )

    if locked_test_run in inner_val_runs:
        raise RuntimeError(
            "Locked test run leaked into epoch selection."
        )

    # ========================================================
    # Stage 2
    # Prepare the final train/test fold.
    #
    # All seven development runs are training runs.
    #
    # prepare_fold must fit feature scaling, residual scaling,
    # persistence fallbacks, etc. from train_runs only.
    #
    # IMPORTANT:
    # At this stage we prepare X_val but DO NOT call prediction
    # and DO NOT calculate any locked-test metric.
    # ========================================================

    fold = T.prepare_fold(
        runs=runs,
        cache=cache,
        train_runs=development_runs,
        val_runs=[locked_test_run],
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        reference_context=reference_context,
    )

    # Basic structural checks.
    if len(fold["X_train"]) == 0:
        raise RuntimeError(
            "Final development training set is empty."
        )

    if len(fold["X_val"]) == 0:
        raise RuntimeError(
            "Prepared locked-test input set is empty."
        )

    if not np.isfinite(
        fold["X_train"]
    ).all():
        raise RuntimeError(
            "Non-finite values found in final training inputs."
        )

    if not np.isfinite(
        fold["resid_train"]
    ).all():
        raise RuntimeError(
            "Non-finite values found in final training residual targets."
        )

    if not np.isfinite(
        fold["X_val"]
    ).all():
        raise RuntimeError(
            "Non-finite values found in prepared locked-test inputs."
        )

    # ========================================================
    # Stage 3
    # Train a fresh final model on ALL development runs.
    # ========================================================

    final_seed = (
        seed
        + len(development_runs)
        + 1000
    )

    TR.set_seed(
        final_seed
    )

    model = N.new_model(
        fold=fold,
        model_kwargs=model_kwargs,
    ).to(
        device
    )

    if best_epoch == 0:

        # Zero-initialized residual head means the untrained
        # model represents zero correction / persistence.
        final_training = {
            "train_loss": np.asarray(
                [],
                dtype="float64",
            ),
            "val_loss": None,
        }

    else:

        train_loader = TR.make_loader(
            X=fold["X_train"],
            y=fold["resid_train"],
            batch_size=batch_size,
            shuffle=True,
            seed=final_seed,
        )

        final_training = N.train_fixed_epochs(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=best_epoch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            val_loader=None,
        )

    # ========================================================
    # IMPORTANT:
    # No predict_scaled_correction() call here.
    # No test MAE/RMSE here.
    # No inspection of y_val here.
    # ========================================================

    return {
        "model": model,
        "fold": fold,
        "selection": selection,
        "best_epoch": best_epoch,
        "final_training": final_training,
        "final_seed": final_seed,
        "development_runs": development_runs,
        "locked_test_run": locked_test_run,
        "context_steps": context_steps,
        "horizon_steps": horizon_steps,
        "reference_context": reference_context,
    }


def evaluate_locked_test(
    prepared,
    device=None,
    batch_size=64,
):
    """
    Evaluate an already prepared/frozen final model on the locked test.

    This function should be called exactly once after the final pipeline
    has been verified.
    """

    if device is None:
        device = torch.device("cpu")

    model = prepared["model"]

    fold = prepared["fold"]

    best_epoch = int(
        prepared["best_epoch"]
    )

    scaled_correction = TR.predict_scaled_correction(
        model=model,
        X=fold["X_val"],
        device=device,
        batch_size=batch_size,
    )

    forecast = T.correction_to_forecast(
        correction_scaled=scaled_correction,
        base=fold["base_val"],
        resid_mean=fold["resid_mean"],
        resid_std=fold["resid_std"],
    )

    if not np.isfinite(
        forecast
    ).all():
        raise RuntimeError(
            "Non-finite locked-test forecasts."
        )

    if best_epoch == 0:

        if not np.allclose(
            forecast,
            fold["base_val"],
        ):
            raise RuntimeError(
                "Epoch 0 was selected, but the final forecast "
                "does not reproduce persistence."
            )

    return {
        "forecast": forecast,
        "y_true": fold["y_val"],
        "persistence": fold["base_val"],
        "scaled_correction": scaled_correction,
        "best_epoch": best_epoch,
        "selection": prepared["selection"],
        "final_training": prepared["final_training"],
        "val_target_rows": fold["val_target_rows"],
        "val_run_names": fold["val_run_names"],
    }
