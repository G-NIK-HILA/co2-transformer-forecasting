# CO2 Concentration Forecasting with a Transformer

This repository contains my solution to the Applied Computing time-series forecasting challenge using data from Imperial College London's Carbon Capture Pilot Plant.

The task is to forecast the CO2 concentration profile at six absorber sampling points. I implemented the forecasting model directly in PyTorch, evaluated it using run-wise validation, analyzed the main sources of forecast error, and built a deployable PostgreSQL + FastAPI + Docker pipeline.

The full modeling and root-cause analysis is in:

`notebooks/co2_forecasting_analysis.ipynb`

## Approach

The eight experimental runs are kept separate throughout model development. Following the challenge recommendation, `140207_1.xlsx` is reserved as the locked test run and the other seven runs are used for development.

The final forecasting representation contains 110 causal input features:

- 86 process-sensor variables
- 6 most recently measured CO2 values
- 6 analyzer-age variables
- 6 indicators showing whether each sampling point has been observed
- 6 sampling-point indicators

The CO2 analyzer rotates across the six sampling locations, so only one point is directly measured at many timestamps. The model therefore receives both the most recent available CO2 information and the age of that information rather than treating interpolated values as equivalent to new measurements.

The Transformer is implemented directly from PyTorch components. The implementation includes input projection, sinusoidal positional encoding, multi-head self-attention, feed-forward encoder blocks, layer normalization, residual connections, and a six-output prediction head. `nn.Transformer`, `nn.TransformerEncoder`, and `nn.MultiheadAttention` are not used.

The forecasting model is formulated as a residual correction to a causal persistence forecast:

\[
\hat{y}_{t+h} =
\hat{y}^{\mathrm{persistence}}_{t+h}
+
\Delta_{\theta}(X_{t-L+1:t})
\]

This makes persistence an explicit benchmark rather than an implicit comparison made after training.

## Validation and model selection

Model development uses the seven development runs only.

Because neighboring rows within a plant run are strongly dependent, validation is performed by holding out complete experimental runs rather than randomly splitting overlapping windows. Transformer epoch selection is nested inside each outer leave-one-run-out evaluation.

The following forecast horizons were benchmarked:

| Horizon | Approximate future time |
|---:|---:|
| 1 step | 0.72 min |
| 3 steps | 2.16 min |
| 6 steps | 4.32 min |
| 12 steps | 8.64 min |
| 18 steps | 12.96 min |

At the selected six-step horizon, context lengths of 6, 12, and 18 observations were compared on identical target rows.

The final configuration uses:

| Setting | Value |
|---|---:|
| Context length | 6 steps |
| Forecast horizon | 6 steps |
| Sampling interval | ~43 s |
| Context duration | ~4.3 min |
| Forecast lead time | ~4.3 min |
| Transformer dimension | 64 |
| Attention heads | 4 |
| Encoder blocks | 2 |
| Feed-forward dimension | 128 |
| Dropout | 0.1 |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |

The six-step horizon was retained as the primary operating point for the detailed comparison. Raw MAE is not used to rank horizons directly because changing the horizon also changes the number and composition of feasible forecast windows. Performance is instead interpreted relative to persistence at each horizon.

The three tested context lengths produced nearly identical persistence-relative performance. The six-step context was retained as the parsimonious choice because it requires the shortest input history and least computation, not because it demonstrated a meaningful accuracy advantage.

## Results

Persistence was a strong benchmark on this dataset.

Across the forecast-horizon comparison, the selected Transformer procedure produced measured-only MAE ratios relative to persistence between approximately 0.999 and 1.004. Changing the context length or testing the bounded learning-rate and model-size variants also did not produce a practically meaningful improvement.

The final epoch-selection rule was then applied using all seven development runs. It selected epoch 0.

The residual prediction head is zero-initialized, so epoch 0 corresponds exactly to a zero Transformer correction. The development-selected predictor carried to the locked test run was therefore the causal persistence forecast.

The held-out `140207_1` run was evaluated once after the development decisions were frozen:

| Evaluation | MAE | RMSE |
|---|---:|---:|
| Full six-point profile | 0.065725 | 0.139041 |
| Directly measured analyzer targets | 0.088761 | 0.170400 |

These numbers are the locked-test performance of the **development-selected predictor**, which is persistence. They are not reported as the test performance of a trained Transformer.

The trained Transformer was evaluated through the nested held-out development experiments. Those experiments showed that the learned residual corrections did not transfer reliably enough across runs to justify replacing persistence.

No additional model configuration was evaluated on the locked run after this result was obtained.

## Root-cause analysis

The analysis in the notebook examines forecast failures using out-of-fold predictions from the development runs rather than tuning explanations on the locked test run.

The main findings are:

- Forecast error is strongly heterogeneous across sampling points and experimental runs.
- Points 5 and 6 contribute disproportionately to the overall error.
- Rare CO2 excursions dominate squared error; the largest 5% of directly measured observations account for about 69% of the raw squared error in the development out-of-fold analysis.
- The rotating analyzer schedule creates sparse direct CO2 observations, so most reconstructed target components between measurements are interpolated.
- Analyzer age alone does not show a consistent monotonic relationship with error across runs.
- Transformer residual corrections are generally small relative to the persistence forecast and do not provide a stable cross-run improvement.
- Process-state movement is associated with some difficult periods, but the available evidence does not support assigning a single process sensor as a causal root cause.

The results point to a combination of sparse rotating measurements, strong short-term persistence, run-to-run heterogeneity, and rare CO2 excursions rather than one isolated failure mechanism.

The notebook contains the complete supporting figures and diagnostics.

## Repository structure

```text
.
├── app/
│   ├── api_check.py
│   ├── audit.py
│   ├── database.py
│   ├── main.py
│   ├── parity_check.py
│   ├── schemas.py
│   └── service.py
├── artifacts/
│   └── model_v1.pt
├── data/
│   └── raw/
├── db/
│   ├── init.sql
│   ├── load_data.py
│   └── make_schema.py
├── notebooks/
│   ├── co2_forecasting_analysis.ipynb
│   ├── figures/
│   └── results/
├── src/
│   ├── artifact.py
│   ├── baselines.py
│   ├── data.py
│   ├── evaluation.py
│   ├── final.py
│   ├── model.py
│   ├── nested.py
│   ├── trainer.py
│   └── training.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-notebook.txt
```

## Running the analysis locally

Python 3.9 was used for development.

Create and activate a virtual environment, then install the notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-notebook.txt
```

Place the eight source Excel files in:

```text
data/raw/
```

Then start Jupyter and open:

```text
notebooks/co2_forecasting_analysis.ipynb
```

The notebook contains the complete sequence from data inspection and preprocessing through benchmarking, Transformer evaluation, locked-test reporting, and root-cause analysis.

## Production-style deployment

Part 3 of the solution uses three components:

1. **PostgreSQL** stores the experimental runs as time-series sensor records and stores a forecast audit log.
2. **FastAPI** reconstructs the same causal preprocessing used during development and serves forecasts.
3. **Docker Compose** starts PostgreSQL, loads the raw data, and starts the forecasting API.

The deployment artifact is:

```text
artifacts/model_v1.pt
```

It stores the selected model configuration and preprocessing state required by the service.

The serving code retains the residual Transformer architecture. For this submitted artifact, development-only model selection chose epoch 0, so the selected Transformer correction is zero and the operational forecast equals the persistence component. This is intentional: the deployment serves the predictor selected by the development protocol rather than substituting a post-hoc trained checkpoint after inspecting the locked-test result.

### Start the complete stack

Docker and Docker Compose are required.

Make sure the eight Excel files are available under `data/raw/`, then run from the repository root:

```bash
docker compose up --build
```

This starts:

- PostgreSQL inside the Compose network on port `5432`
- PostgreSQL on host port `5433`
- the one-time data loader
- FastAPI on host port `8000`

The loader waits for PostgreSQL to become healthy before importing the Excel runs, and the API starts after the loader completes successfully.

### API

Once the stack is running:

```text
http://localhost:8000
```

Available endpoints include:

```text
GET  /health
GET  /model
GET  /runs
POST /predict
```

`/health` can be used to confirm that the service is running, `/model` reports the loaded forecasting configuration, `/runs` lists the available experimental runs, and `/predict` returns forecasts for a requested run and forecast origin.

The prediction response separates:

- the final forecast
- the persistence component
- the Transformer residual correction

so the selected model behavior remains auditable.

Prediction requests are also written to the PostgreSQL forecast log.

### Stop the stack

```bash
docker compose down
```

To also remove the PostgreSQL volume and rebuild the database from scratch:

```bash
docker compose down -v
```

## Reproducibility notes

The modeling protocol is designed to keep development decisions separate from the final held-out evaluation:

- `140207_1` is the locked test run.
- The remaining seven runs are used for model development.
- Feature normalization is fitted using training data only within each validation fold.
- Validation holds out complete runs.
- Transformer epoch selection is nested within the development procedure.
- Horizon, context, and bounded model comparisons are performed on development data.
- The locked run is evaluated only after the configuration and selection rule are frozen.
- Root-cause conclusions are based primarily on development out-of-fold predictions rather than post-hoc test-set tuning.

The saved locked-test result is retained under `notebooks/results/` so the notebook can report the frozen result without rerunning the held-out evaluation.

## Main implementation files

`src/model.py` contains the Transformer implementation written directly with PyTorch building blocks.

`src/data.py` contains the causal feature construction and preprocessing logic.

`src/nested.py` and `src/trainer.py` contain the run-wise training and nested model-selection procedure.

`src/final.py` contains final development-only model preparation and locked-test evaluation utilities.

`src/artifact.py` builds the deployment artifact without using the locked test run.

`app/service.py` reconstructs the forecasting pipeline used by the API.

`db/load_data.py` loads and verifies the experimental runs in PostgreSQL.

## Model interpretation

The Transformer was developed and evaluated as the forecasting model for this task, with persistence used as the benchmark for determining whether the learned multivariate temporal representation provided a reproducible forecasting improvement.

Across the run-wise validation experiments, the Transformer learned residual corrections to the persistence forecast, but the improvement was not consistent across held-out experimental runs. The development-only selection procedure therefore selected the zero-correction checkpoint for the final locked evaluation. This result reflects the strong short-term persistence of the CO2 profiles and the limited cross-run transfer of the learned corrections, rather than replacing the Transformer analysis with a separate post-hoc model.