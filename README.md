# Hyb-STEX

Official PyTorch implementation of **Hyb-STEX**, a spatiotemporal traffic-flow
forecasting model designed to reduce error during rare high-flow events.

The submission model is the same across all four datasets:

1. train the STE-Base predictor with ordinary MAE;
2. freeze the base predictor;
3. train one always-on residual head with event-weighted MAE;
4. define events independently for every horizon, node, and flow direction
   using the 90th percentile of training targets.

In the experiment code this configuration is named
`D_frozen_residual_event_weighted`. The event-loss weight is `0.25`. The
classifier-gated and dual-residual models remain available as ablations but
are not the default Hyb-STEX model reported by the submission.

## Datasets

Experiments use `NYCBike1`, `NYCBike2`, `NYCTaxi`, and `BJTaxi`, following the
data layout of [ST-SSL](https://github.com/Echo-Ji/ST-SSL):

```text
preprocessed_data/
  NYCBike1/
    train.npz
    val.npz
    test.npz
    adj_mx.npz
  NYCBike2/
  NYCTaxi/
  BJTaxi/
```

Each split contains `x`, `y`, and `evs_90`. Event labels are fitted from
training targets only, separately for every forecast horizon, node, and flow
direction. Valid high-flow events satisfy `y > 5` and `y > training p90`.

Download the original datasets from the
[ST-SSL dataset repository](https://github.com/Echo-Ji/ST-SSL_Dataset), then
generate the verified event-label copies:

```bash
python scripts/create_ev_labels.py \
  --data-dir external/ST-SSL_Dataset \
  --output-dir preprocessed_data
```

Data, checkpoints, and generated results are intentionally excluded from Git.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the appropriate CUDA-enabled PyTorch wheel separately if the default
package index does not match your CUDA runtime.

## Run the final Hyb-STEX model

The canonical runner trains both stages, evaluates the test split after model
selection on validation data, resumes compatible interrupted runs, and writes
Hyb-STEX-only result tables.

```bash
python scripts/run_hybstex.py \
  --data-dir preprocessed_data \
  --output-dir results/hybstex \
  --datasets NYCBike1 NYCBike2 NYCTaxi BJTaxi \
  --seeds 1 2 3 \
  --device cuda
```

Equivalent main entry point:

```bash
python main.py --submission-run --data-dir preprocessed_data --device cuda
```

On Windows, `run_experiments.bat` forwards its arguments to the same runner:

```powershell
.\run_experiments.bat --data-dir preprocessed_data --device cuda
```

Before a full run, use the CPU smoke test:

```bash
python scripts/run_hybstex.py \
  --datasets NYCTaxi \
  --seeds 1 \
  --device cpu \
  --smoke-test \
  --output-dir results/smoke
```

Use `--dry-run` to inspect the exact two commands without creating results.
Re-running the same command resumes completed seeds. Use `--restart` only when
you intentionally want to replace the selected output run.

## Outputs

The canonical output root contains:

```text
results/hybstex/
  base/                 # reusable STE-Base checkpoints and metrics
  final/                # residual-stage checkpoints and complete metrics
  per_seed_metrics.csv  # final Hyb-STEX rows only
  summary_metrics.csv   # mean and standard deviation across seeds
  run_info.json         # public configuration and protocol
```

The result tables report inflow, outflow, and mean values for ordinary MAE and
extreme-event error (EEE). Checkpoint paths and detailed training artifacts are
recorded by the underlying resumable runners.

## Evaluation protocol

- Event threshold: training-target p90, fitted per horizon/node/flow.
- Valid-flow rule: `target > 5`.
- Event comparison: strict `target > p90`.
- Model selection: validation loss only.
- Reported metrics: chronological test split only.
- Seeds: 1, 2, and 3.
- Final prediction: frozen residual-stage output `pred_2`.
- Residual: always on; no classifier gate at inference.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for exact commands,
stage definitions, output interpretation, and plotting instructions.

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Repository structure

```text
configs/       Dataset-specific architecture and optimization settings
lib/           Data loading, event masks, metrics, and utilities
model/         Hyb-STEX architecture and phase-wise trainer
scripts/       Canonical runner, ablations, evaluation, and plotting tools
tests/         Protocol and evaluation regression tests
```

## Acknowledgment

Hyb-STEX builds on the ST-SSL data pipeline and base implementation:

> J. Ji, J. Wang, C. Huang, et al. “Spatio-Temporal Self-Supervised Learning
> for Traffic Flow Prediction.” AAAI, 2023.

Please cite the accompanying Hyb-STEX manuscript when using this repository.
