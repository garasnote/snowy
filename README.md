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
  taxonomy tags). Available on request.

## Requirements

PyTorch, PyTorch Lightning, `transformers` (HuggingFace), `torchmetrics`,
`scikit-learn`, `torchvision`, `mlflow` (optional, for experiment tracking),
`PyQt` (for the viewer only). No `requirements.txt` is pinned here yet — see
individual script headers for the specific libraries each depends on.

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
