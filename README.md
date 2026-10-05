# xrd_geometry

`xrd_geometry` infers five identifiable detector-geometry parameters directly from a two-dimensional powder X-ray diffraction image and reports a full predictive covariance. The detector and source spectrum define a trained model's fixed validity domain; the network itself receives only the image. In-plane rotation (`rot3`) is fixed by the experiment specification.

This repository is the XRD prototype for the future domain-agnostic PRISMS package. The reusable boundaries are exercised here first, while pyFAI-specific simulation and calibration semantics remain in the XRD application.

## Package structure

```text
src/xrd_geometry/
  contracts.py       parameter and fixed-experiment definitions
  preprocessing.py   shared training/inference image transform
  data.py            legacy HDF5 adapter and measurement randomization
  models.py          backbone wrapper and regression heads
  losses.py          deterministic and Gaussian NLL objectives
  training.py        configurable deterministic/probabilistic stages
  prediction.py      MC-dropout covariance decomposition
  evaluation.py      point error, NLL, and coverage metrics
  diagnostics.py     per-target Jacobians and uncertainty sensitivity
  checkpoints.py     versioned, self-describing artifacts
  factory.py         configured Hugging Face backbone construction
  cli/               training and image-only calibration commands
```

## Installation

Python 3.11 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

## Data contract

Existing HDF5 files remain supported. They contain `images` and six-column `labels` datasets under named groups, a `normalization` group with centers and scale factors, and optionally a `background_model` group. The adapter selects the five active targets and supplies fixed `rot3` from the experiment specification. Cached training images may remain clean expected-count signals; the adapter retains the current empirical-background and Poisson randomization behavior.

Named datasets may reside in separate HDF5 files. Every dataset supplies its own `hdf5_path` and group. `data.normalization_source` identifies the dataset whose normalization group defines the one target transform used throughout the curriculum; other files still provide labels in physical units.

Randomization is deterministic for a given seed, epoch, and sample index, while changing between epochs. Persistent data-loader workers are therefore disabled for randomized datasets so updated epoch state reaches worker processes.

Reference experimental data are one optional way to characterize a simulation. They are not a framework requirement.

## Training

The YAML file defines named datasets and ordered stages. Each stage selects its data, deterministic or probabilistic regime, objective, active loss weights, trainable scope, optimizer settings, duration, and optional early stopping. Early stopping monitors that stage's validation loss independently; `patience` counts consecutive epochs without an improvement of at least `min_delta`, and `restore_best` reloads the best stage weights before the next stage.

CUDA automatic mixed precision is enabled by default for compatibility with the original training loop. `float16` autocast covers model execution in deterministic and probabilistic stages, with gradient scaling during optimization. Full-covariance Gaussian likelihood calculations remain in FP32 for numerical stability. Mixed precision is disabled automatically on non-CUDA devices.

```yaml
training:
  mixed_precision:
    enabled: true
    dtype: float16  # float16 or bfloat16
```

```yaml
early_stopping:
  patience: 20
  min_delta: 0.0
  restore_best: true
```

```bash
xrd-geometry-train configs/train_swin.yaml
```

A dry run executes one batch and one epoch for every configured stage and writes a normal versioned artifact:

```bash
xrd-geometry-train configs/train_swin.yaml --dry-run
```

Resume a versioned artifact at a named stage with `--load-checkpoint` and `--start-stage`. If a probabilistic artifact is supplied without a stage, deterministic stages are skipped automatically.

New artifacts include weights, model construction settings, preprocessing, parameter normalization, units, fixed experimental assumptions, and training-stage provenance. Raw deterministic state dictionaries remain loadable with their original YAML configuration.

## Docker

The Compose service mounts the repository at `/app`, a host data directory at `/app/data` read-only, a writable model directory at `/app/models`, and a persistent Hugging Face cache. Defaults are `./data` and `./models`; override them with `XRD_DATA_DIR` and `XRD_MODEL_DIR`.

Build and start the Jupyter container without training:

```bash
docker compose build
docker compose up -d
```

After placing both HDF5 files in the configured data directory and editing the curriculum, training can be started explicitly:

```bash
docker compose run --rm xrd-geometry \
  xrd-geometry-train configs/train_swin.yaml
```

Starting the Compose service itself only launches JupyterLab on port 8888; it does not start training.

Training prints train and validation loss after every epoch and writes TensorBoard events under the configured output directory's `logs` subdirectory. Start TensorBoard in the running service with:

```bash
docker compose exec xrd-geometry \
  tensorboard --logdir models/PRISMS_XRD_2026.10/logs --host 0.0.0.0 --port 6006
```

TensorBoard is then available at `http://localhost:6006`. The Compose service publishes this port but does not start TensorBoard automatically.

## Inference

```bash
xrd-geometry-calibrate \
  --checkpoint models/xrd_geometry/xrd_geometry_swin.pt \
  --image path/to/pattern.tif \
  --samples 32 \
  --output prediction.json
```

Probabilistic inference returns the five-parameter mean together with conditional, head-model, and total covariance matrices in physical units. Head-only MC dropout caches the backbone representation. A broader stochastic scope is available through the Python API for later estimator comparisons.

Legacy deterministic checkpoints require their original configuration:

```bash
xrd-geometry-calibrate \
  --checkpoint models/legacy.pth \
  --config configs/legacy.yaml \
  --image path/to/pattern.tif
```

Network inference is the current endpoint. Iterative refinement and downstream uncertainty propagation are outside this phase.

## Diagnostics

`xrd_geometry.diagnostics.parameter_jacobians` returns one input Jacobian per target. `uncertainty_sensitivity` returns the gradient of the total covariance trace in normalized target coordinates. These are sensitivity diagnostics, not literal attention maps or causal explanations.

`xrd_geometry.evaluation` provides per-parameter MAE/RMSE, Gaussian NLL, marginal coverage and interval width, and joint covariance-ellipsoid coverage. These functions support later scientific evaluation; the dry run does not establish calibration.

## Tests

```bash
python -m pytest
```

The suite covers parameter and unit transforms, legacy HDF5 behavior, deterministic randomization, covariance validity and decomposition, deterministic-to-probabilistic dry-run training, checkpoint compatibility, and the two pixel-space diagnostics.

## License

MIT. See `LICENSE`.
