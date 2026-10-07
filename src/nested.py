"""
Nested model-selection procedure for the residual CO2 Transformer.

For one outer leave-one-run-out fold:

Stage 1
-------
Choose the training epoch count using inner leave-one-run-out
cross-validation across the outer-training runs only.

For each inner validation run, validation loss is expressed relative
to that run's zero-correction loss:

    ratio_k(epoch) = validation_loss_k(epoch) / zero_loss_k

Because the Transformer prediction head is initialized to zero,
epoch 0 corresponds exactly to the causal persistence forecast.
Therefore epoch 0 is explicitly included as a candidate with
ratio = 1 for every inner fold.

The primary epoch-selection score is the arithmetic mean of these
ratios across inner runs, with each run receiving equal weight.

Stage 2
-------
Discard all inner-CV models. Initialize a fresh Transformer and,
if the selected epoch is greater than zero, train it on ALL
outer-training runs for exactly the selected number of epochs.

If epoch 0 is selected, no correction training is performed and
the zero-initialized model reproduces persistence exactly.

Stage 3
-------
Evaluate the resulting model once on the outer held-out run.

The locked test run must never participate in development/model
selection.
"""

import numpy as np
import torch

from src import trainer as TR
from src import training as T
from src.model import CO2Transformer


def new_model(
    fold,
    model_kwargs=None,
):
    """
    Construct a fresh residual CO2 Transformer.

    Model dimensions are inferred from the prepared fold wherever
    possible so the nested procedure remains compatible with different
    context lengths and feature sets.
    """

    kwargs = {
        "n_features": fold["X_train"].shape[-1],
        "n_targets": fold["y_train"].shape[-1],
        "max_length": fold["X_train"].shape[1],
    }

    if model_kwargs is not None:
        kwargs.update(
            model_kwargs
        )

    return CO2Transformer(
        **kwargs
    )


def train_fixed_epochs(
    model,
    train_loader,
    device,
    epochs,
    learning_rate=3e-4,
    weight_decay=1e-4,
    gradient_clip=1.0,
    val_loader=None,
):
    """
    Train a model for exactly `epochs` epochs.

    If val_loader is supplied, validation loss is recorded after every
    epoch. Validation does not stop training and does not restore model
    weights.

    This function requires epochs >= 1. Epoch 0 is handled explicitly
    by run_outer_fold by leaving a fresh zero-initialized model
    untrained.

    Returns
    -------
    dict
        train_loss : np.ndarray
            Training loss after each epoch.

        val_loss : np.ndarray or None
            Validation loss after each epoch if val_loader was supplied.
    """

    if epochs < 1:
        raise ValueError(
            "epochs must be at least 1."
        )

    model = model.to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_curve = []

    if val_loader is not None:
        val_curve = []
    else:
        val_curve = None

    for _ in range(epochs):

        train_loss = TR.train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            gradient_clip=gradient_clip,
        )

        train_curve.append(
            train_loss
        )

        if val_loader is not None:

            val_loss = TR.evaluate_loss(
                model=model,
                loader=val_loader,
                device=device,
            )

            val_curve.append(
                val_loss
            )

    result = {
        "train_loss": np.asarray(
            train_curve,
            dtype="float64",
        ),
        "val_loss": None,
    }

    if val_curve is not None:
        result["val_loss"] = np.asarray(
            val_curve,
            dtype="float64",
        )

    return result


def select_epoch_by_inner_cv(
    runs,
    cache,
    training_runs,
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
    Select the epoch count using inner leave-one-run-out CV.

    Each outer-training run serves as the inner validation run exactly
    once.

    For every inner fold:

        1. Preprocessing is fit using the remaining runs only.
        2. The zero-correction validation loss is calculated before
           training. This is the persistence reference for that fold.
        3. A fresh Transformer is trained for the full max_epochs.
        4. Validation MSE is recorded after every trained epoch.
        5. The trained validation curve is divided by that fold's
           zero-correction loss.

    Epoch 0 is then added explicitly:

        ratio_k(0) = 1

    for every inner fold.

    The primary selection score is:

        mean_ratio(epoch)
            = mean_k ratio_k(epoch)

    where every physical run receives equal weight.

    The selected epoch is the earliest epoch with minimum arithmetic
    mean persistence-relative loss.

    Returns
    -------
    dict
        best_epoch
            Selected epoch in [0, max_epochs].

        mean_ratio
            Equal-run arithmetic mean ratio for epochs 0..max_epochs.

        median_ratio
            Median ratio across runs for epochs 0..max_epochs.
            Stored as a sensitivity diagnostic only.

        fold_ratios
            Per-inner-run ratios for epochs 0..max_epochs.

        zero_losses
            Zero-correction validation loss for every inner fold.

        fold_val_losses
            Raw validation losses for trained epochs 1..max_epochs.

        fold_train_losses
            Training losses for epochs 1..max_epochs.

        fold_sizes
            Number of validation windows in each inner fold.

        inner_val_runs
            Inner validation run names in fold order.
    """

    if len(training_runs) < 2:
        raise ValueError(
            "At least two training runs are required for inner CV."
        )

    if max_epochs < 1:
        raise ValueError(
            "max_epochs must be at least 1."
        )

    if device is None:
        device = torch.device(
            "cpu"
        )

    fold_val_losses = []
    fold_train_losses = []
    fold_sizes = []
    zero_losses = []
    inner_val_runs = []

    for fold_index, inner_val_run in enumerate(
        training_runs
    ):

        inner_train_runs = [
            run_name
            for run_name in training_runs
            if run_name != inner_val_run
        ]

        if verbose:

            print(
                "\n"
                f"Inner fold {fold_index + 1}/"
                f"{len(training_runs)}"
            )

            print(
                "  Inner validation:",
                inner_val_run,
            )

            print(
                "  Inner training:",
                inner_train_runs,
            )

        # -----------------------------------------------------
        # Fit preprocessing using INNER-TRAINING runs only
        # -----------------------------------------------------

        fold = T.prepare_fold(
            runs=runs,
            cache=cache,
            train_runs=inner_train_runs,
            val_runs=[inner_val_run],
            context_steps=context_steps,
            horizon_steps=horizon_steps,
            reference_context=reference_context,
        )

        # -----------------------------------------------------
        # Epoch-0 reference
        #
        # The zero-initialized residual model predicts a scaled
        # correction of exactly zero. Its MSE is therefore simply
        # mean(resid_val ** 2).
        # -----------------------------------------------------

        zero_loss = float(
            np.mean(
                np.asarray(
                    fold["resid_val"],
                    dtype="float64",
                )
                ** 2
            )
        )

        if (
            not np.isfinite(zero_loss)
            or zero_loss <= 0.0
        ):
            raise RuntimeError(
                "Invalid zero-correction loss for inner fold "
                f"{inner_val_run}: {zero_loss}"
            )

        zero_losses.append(
            zero_loss
        )

        # -----------------------------------------------------
        # Deterministic but fold-specific seed
        # -----------------------------------------------------

        fold_seed = (
            seed
            + fold_index
        )

        TR.set_seed(
            fold_seed
        )

        # -----------------------------------------------------
        # Fresh model for this inner fold
        # -----------------------------------------------------

        model = new_model(
            fold=fold,
            model_kwargs=model_kwargs,
        )

        # -----------------------------------------------------
        # Inner-training and inner-validation loaders
        # -----------------------------------------------------

        train_loader = TR.make_loader(
            X=fold["X_train"],
            y=fold["resid_train"],
            batch_size=batch_size,
            shuffle=True,
            seed=fold_seed,
        )

        val_loader = TR.make_loader(
            X=fold["X_val"],
            y=fold["resid_val"],
            batch_size=64,
            shuffle=False,
            seed=fold_seed,
        )

        # -----------------------------------------------------
        # Full training trajectory
        #
        # No early stopping inside an individual inner fold.
        # Every fold generates the same candidate epoch range.
        # -----------------------------------------------------

        curves = train_fixed_epochs(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=max_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            val_loader=val_loader,
        )

        fold_train_losses.append(
            curves["train_loss"]
        )

        fold_val_losses.append(
            curves["val_loss"]
        )

        fold_size = len(
            fold["y_val"]
        )

        fold_sizes.append(
            fold_size
        )

        inner_val_runs.append(
            inner_val_run
        )

        if verbose:

            trained_ratio = (
                curves["val_loss"]
                / zero_loss
            )

            individual_best_epoch = (
                int(
                    np.argmin(
                        trained_ratio
                    )
                )
                + 1
            )

            individual_best_ratio = float(
                np.min(
                    trained_ratio
                )
            )

            print(
                "  Validation windows:",
                fold_size,
            )

            print(
                "  Zero-correction loss:",
                round(
                    zero_loss,
                    6,
                ),
            )

            print(
                "  Best trained epoch:",
                individual_best_epoch,
            )

            print(
                "  Best trained ratio:",
                round(
                    individual_best_ratio,
                    6,
                ),
            )

    # ---------------------------------------------------------
    # Stack inner-fold results
    # ---------------------------------------------------------

    fold_val_losses = np.stack(
        fold_val_losses,
        axis=0,
    )

    fold_train_losses = np.stack(
        fold_train_losses,
        axis=0,
    )

    fold_sizes = np.asarray(
        fold_sizes,
        dtype="float64",
    )

    zero_losses = np.asarray(
        zero_losses,
        dtype="float64",
    )

    # ---------------------------------------------------------
    # Convert each fold to persistence-relative loss
    #
    # Shape before epoch 0:
    #     (n_inner_folds, max_epochs)
    #
    # Each fold is normalized by its own zero-correction loss.
    # ---------------------------------------------------------

    trained_ratios = (
        fold_val_losses
        / zero_losses[:, None]
    )

    # ---------------------------------------------------------
    # Add epoch 0 explicitly.
    #
    # Epoch 0 is persistence, therefore ratio = 1 for every run.
    #
    # Final shape:
    #     (n_inner_folds, max_epochs + 1)
    # ---------------------------------------------------------

    epoch_zero = np.ones(
        (
            len(training_runs),
            1,
        ),
        dtype="float64",
    )

    fold_ratios = np.concatenate(
        [
            epoch_zero,
            trained_ratios,
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # PRIMARY selection criterion:
    # arithmetic mean across physical runs with EQUAL run weight.
    # ---------------------------------------------------------

    mean_ratio = np.mean(
        fold_ratios,
        axis=0,
    )

    # Diagnostic only.
    median_ratio = np.median(
        fold_ratios,
        axis=0,
    )

    # np.argmin returns the first minimum, so an exact tie is
    # resolved in favor of the earlier / simpler epoch.
    best_epoch = int(
        np.argmin(
            mean_ratio
        )
    )

    if verbose:

        print(
            "\n"
            "Inner-CV persistence-relative epoch selection"
        )

        print(
            "  Inner validation runs:",
            inner_val_runs,
        )

        print(
            "  Validation window counts:",
            fold_sizes.astype(int).tolist(),
        )

        print(
            "  Zero-correction losses:",
            np.round(
                zero_losses,
                6,
            ).tolist(),
        )

        print(
            "  Selection rule:",
            "equal-run arithmetic mean of "
            "persistence-relative validation loss",
        )

        print(
            "  Epoch 0 score:",
            round(
                float(
                    mean_ratio[0]
                ),
                6,
            ),
        )

        print(
            "  Best trained score:",
            round(
                float(
                    np.min(
                        mean_ratio[1:]
                    )
                ),
                6,
            ),
        )

        print(
            "  Best trained epoch:",
            int(
                np.argmin(
                    mean_ratio[1:]
                )
                + 1
            ),
        )

        print(
            "  Selected epoch:",
            best_epoch,
        )

        print(
            "  Selected mean ratio:",
            round(
                float(
                    mean_ratio[
                        best_epoch
                    ]
                ),
                6,
            ),
        )

        print(
            "  Runs beating persistence "
            "at selected epoch:",
            int(
                np.sum(
                    fold_ratios[
                        :,
                        best_epoch
                    ]
                    < 1.0
                )
            ),
            "/",
            len(training_runs),
        )

    return {
        "best_epoch": best_epoch,
        "mean_ratio": mean_ratio,
        "median_ratio": median_ratio,
        "fold_ratios": fold_ratios,
        "zero_losses": zero_losses,
        "fold_val_losses": fold_val_losses,
        "fold_train_losses": fold_train_losses,
        "fold_sizes": fold_sizes,
        "inner_val_runs": inner_val_runs,
    }


def run_outer_fold(
    runs,
    cache,
    development_runs,
    outer_val_run,
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
    Run the complete nested procedure for one outer LORO fold.

    Stage 1:
        Select epoch count using persistence-relative inner LORO across
        outer-training runs.

    Stage 2:
        Initialize a fresh model.

        If best_epoch > 0, train it on ALL outer-training runs for
        exactly best_epoch epochs.

        If best_epoch == 0, leave the fresh zero-initialized model
        untrained. Its correction is exactly zero, so the resulting
        physical forecast is exactly persistence.

    Stage 3:
        Evaluate once on the outer held-out run.

    The locked test run is explicitly rejected from the development set.

    Returns
    -------
    dict
        forecast
        y_true
        persistence
        scaled_correction
        best_epoch
        selection
        final_training
        model
        fold
        val_run_names
        val_target_rows
    """

    if locked_test_run in development_runs:
        raise ValueError(
            "Locked test run must not appear in development_runs."
        )

    if outer_val_run == locked_test_run:
        raise ValueError(
            "Locked test run cannot be used as an outer "
            "development-validation run."
        )

    if outer_val_run not in development_runs:
        raise ValueError(
            "outer_val_run must appear in development_runs."
        )

    if device is None:
        device = torch.device(
            "cpu"
        )

    outer_train_runs = [
        run_name
        for run_name in development_runs
        if run_name != outer_val_run
    ]

    if verbose:

        print(
            "=" * 70
        )

        print(
            "OUTER FOLD"
        )

        print(
            "Outer validation run:",
            outer_val_run,
        )

        print(
            "Outer training runs:",
            outer_train_runs,
        )

        print(
            "Locked test run:",
            locked_test_run,
        )

        print(
            "=" * 70
        )

    # =========================================================
    # STAGE 1
    # Select epoch count using INNER CV only
    # =========================================================

    selection = select_epoch_by_inner_cv(
        runs=runs,
        cache=cache,
        training_runs=outer_train_runs,
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

    best_epoch = selection[
        "best_epoch"
    ]

    # =========================================================
    # STAGE 2
    # Prepare ALL outer-training runs and initialize a fresh model
    # =========================================================

    if verbose:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINAL OUTER-FOLD REFIT"
        )

    fold = T.prepare_fold(
        runs=runs,
        cache=cache,
        train_runs=outer_train_runs,
        val_runs=[outer_val_run],
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        reference_context=reference_context,
    )

    # Separate deterministic seed for the final refit.
    final_seed = (
        seed
        + len(outer_train_runs)
        + 1000
    )

    TR.set_seed(
        final_seed
    )

    model = new_model(
        fold=fold,
        model_kwargs=model_kwargs,
    )

    model = model.to(
        device
    )

    if best_epoch == 0:

        # -----------------------------------------------------
        # Epoch 0 selected.
        #
        # Do not train. The prediction head remains exactly zero,
        # so the final physical forecast is exactly persistence.
        # -----------------------------------------------------

        final_training = {
            "train_loss": np.asarray(
                [],
                dtype="float64",
            ),
            "val_loss": None,
        }

        if verbose:

            print(
                "Epoch 0 selected by inner CV."
            )

            print(
                "No residual correction training performed."
            )

            print(
                "Final model remains at exact persistence."
            )

    else:

        if verbose:

            print(
                "Training on all outer-training runs for",
                best_epoch,
                "epochs."
            )

        train_loader = TR.make_loader(
            X=fold["X_train"],
            y=fold["resid_train"],
            batch_size=batch_size,
            shuffle=True,
            seed=final_seed,
        )

        final_training = train_fixed_epochs(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=best_epoch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            val_loader=None,
        )

    # =========================================================
    # STAGE 3
    # Evaluate outer held-out run ONCE
    # =========================================================

    scaled_correction = TR.predict_scaled_correction(
        model=model,
        X=fold["X_val"],
        device=device,
        batch_size=64,
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
            "Non-finite outer-fold forecasts encountered."
        )

    # If epoch 0 was selected, verify the implementation really
    # reproduces persistence exactly.
    if best_epoch == 0:

        if not np.allclose(
            scaled_correction,
            0.0,
        ):
            raise RuntimeError(
                "Epoch 0 was selected, but the model produced "
                "a nonzero scaled correction."
            )

        if not np.allclose(
            forecast,
            fold["base_val"],
        ):
            raise RuntimeError(
                "Epoch 0 was selected, but the physical forecast "
                "does not equal persistence."
            )

    if verbose:

        model_mae = float(
            np.abs(
                forecast
                - fold["y_val"]
            ).mean()
        )

        persistence_mae = float(
            np.abs(
                fold["base_val"]
                - fold["y_val"]
            ).mean()
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "OUTER EVALUATION"
        )

        print(
            "Outer validation run:",
            outer_val_run,
        )

        print(
            "Selected epoch:",
            best_epoch,
        )

        print(
            "Selected inner-CV mean ratio:",
            round(
                float(
                    selection["mean_ratio"][
                        best_epoch
                    ]
                ),
                6,
            ),
        )

        print(
            "Transformer/residual-model MAE:",
            round(
                model_mae,
                4,
            ),
        )

        print(
            "Persistence MAE:",
            round(
                persistence_mae,
                4,
            ),
        )

        if best_epoch == 0:

            print(
                "Selected model:",
                "persistence (zero correction)",
            )

        else:

            print(
                "Selected model:",
                f"Transformer correction after {best_epoch} epochs",
            )

        print(
            "=" * 70
        )

    return {
        "forecast": forecast,
        "y_true": fold["y_val"],
        "persistence": fold["base_val"],
        "scaled_correction": scaled_correction,
        "best_epoch": best_epoch,
        "selection": selection,
        "final_training": final_training,
        "model": model,
        "fold": fold,
        "val_run_names": fold["val_run_names"],
        "val_target_rows": fold["val_target_rows"],
    }