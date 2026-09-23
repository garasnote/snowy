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
- `qa/` — the image-quality gate (Sec. "QA gate" in the paper, Table `tab:qa`):
  filters degraded/unusable frames before segmentation runs. This is a
  reported, evaluated pipeline stage, not preprocessing hygiene.
  - `train_qa_cnn.py` — TinyQACNN, the QA gate actually reported in the
    paper (3-class: clean/snow/non-usable).
  - `train_qa.py` — HistGradientBoosting classifier on handcrafted features;
    one of the comparison baselines TinyQACNN is measured against. Also
    produces `qa_features.csv` (with the `label3` column) that
    `train_qa_cnn.py` and `train_qa_dino.py` train from. `--plot-mistakes`
    plots its CV false positives/negatives (was a separate
    `plot_mistakes.py` script; folded in since nothing else used it).
  - `train_qa_dino.py` — DINOv2/DINOv3-backbone comparison baseline.
  - `predict_qa.py` — inference wrapper for the GradBoost model; imported
    as a library by `analysis/build_features.py`'s `timeseries` subcommand.
- `segmentation/` — zero-shot teacher / pseudo-label generation, the pseudo-
  labels the students in `models/student_ground/` and `models/student_snow/`
  are distilled from:
  - `ground_teachers.py` — DeepLabV3+, SegFormer, SAM3 (text / DINOv2-guided),
    DINOv2-PCA, Depth Anything V2 (one subcommand each; was 5 separate scripts).
  - `ground_utils.py` — the Sobel/fixed-fraction baseline, the majority-vote
    ensemble that combines the teachers above, and a depth-map re-threshold
    utility to re-run ground segmentation from cached Depth Anything V2 maps
    without a GPU (`resegment-depth` subcommand; was 3 separate scripts).
  - `snow_teachers.py` — CLIPSeg, Florence-2, GroundedSAM2, SAM3, YOLOE-26,
    SegGPT (one subcommand each; was 6 separate scripts).
  - `snow_utils.py` — the HSV baseline, its EER threshold-tuning helper, the
    majority-vote ensemble across the snow teachers, and a post-processing
    utility that ANDs snow masks with a ground mask (`clip-to-ground`
    subcommand; was 4 separate scripts; note in the file: the HSV baseline
    is not the PCA Blue-Band baseline reported in the paper, which never
    made it back from the cluster into this repo).
  - `SAMWrapper`, the SAM3 wrapper class, lives inside `snow_teachers.py`
    (used only by its `sam3` subcommand; ground_teachers.py's own `sam3`
    subcommand is self-contained).
- `models/student_ground/`, `models/student_snow/` — knowledge-distillation
  student training and inference (pseudo-label ensembles + GT fine-tuning,
  MobileNetV3 / EfficientNet-B0 / DINOv2 / DINOv3 / ConvNeXt / ViT-LoRA
  backbones).
- `models/temporal_snow/` — GRU/LSTM temporal smoother (uni- and
  bi-directional), training, inference, and visualization scripts.
- `models/multi_head/` — DINOv3MultiHead: single-backbone multi-task model
  (QA + ground + snow), two-phase pseudo-label pretraining then GT fine-tuning.
- `analysis/`:
  - `build_features.py` — `timeseries` and `image-features` subcommands.
    Not analysis, just the ETL that produces `snow_timeseries.csv` and
    `image_features.csv`, the inputs everything else here (and the GRU
    smoother, and the multi-head QA fallback) reads.
  - `temporal_diagnostics.py` — `meteo-validate` (the paper's headline ARSO
    F1 number), `melt-patterns` (transition/melt-sequence analysis),
    `grid` (qualitative snow-coverage comparison grids), `daily-agg-check`
    (sanity check for meteo-validate's daily aggregation). Was 4 separate
    scripts (`validate_meteo.py`, `temporal_analysis.py`,
    `make_snow_grid.py`, `check_daily_agg.py`).
  - SHAP analysis on the GRU smoother now lives at
    `models/temporal_snow/shap_gru.py` (moved out of `analysis/`, since it
    only ever imported the smoother model/dataset code).
  - Dropped: `kalman_baseline.py` (not part of the paper), and
    `snow_patchiness.py` / `crop_evaluation.py` / `error_analysis.py`
    (orphaned exploratory scripts — zero downstream consumers, output
    never referenced in the manuscript).
- `preprocessing/segmentation/camera_valid_regions/` — hand-drawn per-camera
  masks excluding static obstructions (parking lots, distant landscape) from
  ground-truth computation.
- `tools/viewer/app.py` — PyQt desktop viewer for inspecting predictions,
  running batch inference, and comparing model checkpoints against the
  timeseries.
- `tools/app_segmentation_comparison.py` — earlier Streamlit dashboard for
  comparing ground/snow segmentation methods side by side, from before the
  PyQt viewer above replaced it for day-to-day use. Run with
  `streamlit run tools/app_segmentation_comparison.py`.
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

Each `models/*/train.py` and `qa/train_*.py` script is
self-contained and documents its own CLI arguments in its module docstring.
`scripts/*.sbatch` show the exact invocations (variants, hyperparameters,
checkpoint paths) used to produce the results reported in the paper.
`eval_all.py --task {ground,snow,qa,smoother} --types <...>` reproduces the
evaluation tables; run with `--help` for the full list of `--types` per task.

## Citation

If you use this code, please cite the paper (details to be added on
publication).
