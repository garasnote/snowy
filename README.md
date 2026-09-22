# Alpine Webcam Ground & Snow Segmentation

Code for "Multi-Modal Machine Learning for Image-Based Snow Coverage Estimation"
(submitted to Computers & Geosciences). Automated pipeline for estimating
Presence of Snow on Ground (PSG) from fixed webcam imagery: QA gating, ground
segmentation, snow segmentation, and a temporal GRU smoother, validated against
independent ARSO meteorological station snow-presence records.

## What's here

- `eval_all.py` — unified evaluation script for all ground/snow/QA methods
  (teachers, ensembles, students, baselines), in-distribution and OOD.
- `eval_ood_per_camera.py` — per-camera OOD breakdown (ground/snow/QA).
- `models/student_ground/`, `models/student_snow/` — knowledge-distillation
  student training and inference (pseudo-label ensembles + GT fine-tuning,
  MobileNetV3 / EfficientNet-B0 / DINOv2 / DINOv3 / ConvNeXt / ViT-LoRA
  backbones).
- `models/temporal_snow/` — GRU/LSTM temporal smoother (uni- and
  bi-directional), training, inference, and visualization scripts.
- `models/multi_head/` — DINOv3MultiHead: single-backbone multi-task model
  (QA + ground + snow), two-phase pseudo-label pretraining then GT fine-tuning.
- `analysis/` — timeseries construction, meteorological station validation,
  SHAP analysis, Kalman filter baseline, error analysis.
- `preprocessing/qa/` — QA gate training (TinyQACNN, gradient-boosted trees)
  and inference.
- `preprocessing/segmentation/camera_valid_regions/` — hand-drawn per-camera
  masks excluding static obstructions (parking lots, distant landscape) from
  ground-truth computation.
- `tools/viewer/app.py` — PyQt desktop viewer for inspecting predictions,
  running batch inference, and comparing model checkpoints against the
  timeseries.
- `scripts/` — SLURM sbatch job scripts used to run training/evaluation on
  the FRIDA HPC cluster.

## What's not here

- **Raw webcam imagery**: 13 ARSO station camera feeds, not ours to
  redistribute. Available on request to the corresponding author, subject to
  ARSO's data-sharing terms.
- **Model checkpoints**: several hundred MB to multiple GB each. Available on
  request to the corresponding author.
- **Manual annotations**: Label Studio exports (ground/snow brush masks, QA
  taxonomy tags). These are our own work product, not ARSO's, and are shared
  freely, no request needed. A persistent link (Zenodo or similar) will be
  added here once deposited. `[TODO: deposit + link]`

## Configuration

Paths to data, large intermediate outputs, and cached model weights are
centralized in `paths.py` and overridable via environment variables
(`SNOW_DATA_DIR`, `SNOW_WORKSPACE_DIR`, `SNOW_MODEL_CACHE_DIR`), falling back
to local relative directories if unset. No SLURM or specific cluster/account
layout is required to run the code; the `scripts/*.sbatch` files are provided
for reference but are not necessary outside that HPC environment.

## Requirements

`pip install -r requirements.txt`, then separately install PyTorch (pinned to
`torch==2.7.0` from the cu126 index — see `scripts/setup_environment.sbatch`
for the exact command and CUDA 12.6 context this was developed against).
`requirements.txt` is the real cluster environment file, trimmed to drop a
few packages this particular repo doesn't need (see comments in the file for
what and why) and with five genuinely-used packages added back in that were
missing from the original. `mlflow` is optional (experiment tracking) and
`PyQt6` is only needed for `tools/viewer/app.py`.

## Reproducing results

Each `models/*/train.py` and `preprocessing/qa/train_*.py` script is
self-contained and documents its own CLI arguments in its module docstring.
`scripts/*.sbatch` show the exact invocations (variants, hyperparameters,
checkpoint paths) used to produce the results reported in the paper.
`eval_all.py --task {ground,snow,qa,smoother} --types <...>` reproduces the
evaluation tables; run with `--help` for the full list of `--types` per task.

## Citation

If you use this code, please cite the paper (details to be added on
publication).
