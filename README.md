# Alpine Webcam Ground & Snow Segmentation

Code for "Multi-Modal Machine Learning for Image-Based Snow Coverage Estimation"
(submitted to Computers & Geosciences). Automated pipeline for estimating
Presence of Snow on Ground (PSG) from fixed webcam imagery: QA gating, ground
segmentation, snow segmentation, and a temporal GRU smoother, validated against
independent ARSO meteorological station snow-presence records.

## What's here

- `eval_all.py` — unified evaluation script for all ground/snow/QA methods
  (teachers, ensembles, students, baselines), in-distribution and OOD.
  `--task ood-per-camera` (was a separate `eval_ood_per_camera.py`, folded
  in since it already imported straight from this file) gives the true
  per-camera OOD breakdown — ground IoU, snow IoU, QA AUROC/F1 for each of
  the 4 OOD cameras individually, not pooled into one n=495 block like
  `--task ground/snow --types ood`. Backs the paper's claim that CM00067
  is consistently the hardest camera in the cohort (Table
  `tab:ood-per-camera`).
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
  bi-directional):
  - `dataset.py`, `model.py`, `train_smoother.py`, `animate_periods.py` —
    library modules, imported by name throughout the repo (`eval_all.py`,
    `analysis/temporal_diagnostics.py`, `tools/viewer/app.py`,
    `tools/make_smoother_video.py`, and siblings in this directory).
  - `train.py` — trains the smoother itself.
  - `gru_ood_snow_freq.py` — GRU-smoothed snow frequency for a camera set
    (defaults to the 4 OOD cameras; matches Table `tab:cameras`'
    methodology).
  - `visualize_smoother.py` — `calendar`/`timeline`/`timelapse`
    subcommands (was 3 separate scripts: `calendar_heatmap.py`,
    `plot_full_timeline.py`, `timelapse.py`).
  - `figures.py` — one-off manuscript figure generators:
    `ground-comparison`/`teacher-comparison`/`best-results`/
    `smoother-week`/`shap` subcommands (was 5 separate scripts, including
    `shap_gru.py` which had moved here from `analysis/` in an earlier
    pass). `best-results` is the actual generator behind the paper's
    `grid_best_ground`/`grid_worst_ground`/`grid_best_snow`/
    `grid_worst_snow` appendix figures.
  - Dropped: `plot_ood_difficulty.py` (hardcoded the old, removed OOD
    trio — CM00006/CM00024/CM00068 — zero references, zero mentions in
    the manuscript).
- `models/multi_head/` — DINOv3MultiHead: single-backbone multi-task model
  (QA + ground + snow), two-phase pseudo-label pretraining then GT fine-tuning.
- `models/fusion_cnn/` — FusionCNN: shallow 3-layer conv net that learns a
  per-pixel weighting over the `segmentation/` teacher masks (replaces the
  hand-crafted IoU-weighted ensemble in `segmentation/ground_utils.py` /
  `snow_utils.py`), trained on GT-annotated images only. `inference_fusion.py`
  runs the trained model over all images to produce
  `preprocessing/segmentation/fusion_{ground,snow}_output/`, the pseudo-label
  source `models/multi_head/train_multihead.py --pseudo-snow` defaults to for
  Multihead's Phase 1 pretraining.
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
    `models/temporal_snow/figures.py`'s `shap` subcommand (moved out of
    `analysis/`, since it only ever imported the smoother model/dataset
    code).
  - Dropped: `kalman_baseline.py` (not part of the paper), and
    `snow_patchiness.py` / `crop_evaluation.py` / `error_analysis.py`
    (orphaned exploratory scripts — zero downstream consumers, output
    never referenced in the manuscript).
- `preprocessing/remove_bad_images.py` — first pass on any new camera's raw
  dump: moves night/dark frames (sunrise/sunset margin, optional brightness
  threshold) and corrupt/truncated files to a side folder before anything
  else touches the data. Corrupt-image detection used to be scattered,
  inconsistent try/except skips wherever a script happened to hit a broken
  file; this is the one upfront pass (was `remove_night_images.py`).
- `preprocessing/segmentation/camera_valid_regions/` — hand-drawn per-camera
  masks excluding static obstructions (parking lots, distant landscape) from
  ground-truth computation. `tools/draw_valid_region.py` creates them
  (click-to-draw polygon). One mask per camera, no sub-view support — see
  the CM00067 note below.
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
  freely, no request needed. Zenodo deposit reserved
  (10.5281/zenodo.22926962) but not yet published — DOI does not resolve
  until the record is published, so it's left unlinked here for now.

  Layout of `data/annotations/` — one canonical revision per camera; nothing is
  merged at load time, and no file here is generated on the fly:

  | file(s) | contents |
  |---|---|
  | `01, 09, 14, 70, 75, 80, 102, 108, 112.json` | snow annotations for the nine in-distribution cameras, **merged** export, one file per camera |
  | `54, 67, 71, 115.json` | the four held-out cameras: ground (`Ground Mano`) **and** snow (`Snow Mano`) |
  | `ground_annotations.json` | pixel-level ground masks, older label spelling `Ground`, all cameras |
  | `premerge/` | provenance only — nothing loads from here. The raw single-session Label Studio exports the merged files were built from. `san.json` and `heroku.json` are not cameras: they hold extra snow frames for CF00001 (70 and 12), CM00009 (5) and CM00014 (6), which is why the merged CF00001 file is much larger than the pre-merge one. |

  Content of the merged snow files matches the per-camera files deposited on
  Zenodo, so repository and archive agree. Rebuild both together if you
  re-export from Label Studio, or they will drift again.

  Counts to check after any re-export: 212 ground masks and 918 snow-task
  records (678 snow-covered, 240 snow-free) across the nine in-distribution
  cameras, matching Table 2 of the paper. `verify_tables.py` recomputes these
  and fails loudly on a mismatch.

## Camera image naming

Filenames follow `<CAMERA_ID>_<YYYYMMDDHHMM>_<NN>.jpg`. `NN` is not a frame
sequence number — it's a per-camera view/feed identifier from the source
station's own capture system. For every camera in the current cohort except
one, `NN` is a single fixed value across that camera's entire history (e.g.
always `_00` for CF00001) — effectively a constant, since the camera only
has one view.

**CM00067 is the exception**: 8 distinct `NN` values (05, 09, 14, 18, 23,
27, 32, 36), confirmed by inspecting the actual images — same rough
timestamp, genuinely different physical scenes (open field vs.
rooftop-heavy foreground vs. mountain panorama), not duplicate frames or a
scraping artifact. This lines up with the paper's own text calling out
CM00067 for "multiple views, houses, and drying racks that confuse
segmentation" (Sec. "Per-Camera Out-of-Distribution Performance") — it's
plausibly why CM00067 is the hardest camera in the cohort.

**Nothing in the pipeline currently treats these views separately.** Every
image under `CM00067/` is processed as one continuous timeline regardless
of which physical view it came from — ground truth, valid-region masks
(themselves camera-keyed, one file per camera, no per-view support), and
every teacher/student/eval script all key on `(camera, stem)` alone. If a
new multi-view camera gets added, know that this is unbuilt, not just
undocumented — the `NN` suffix is available in every filename, but nothing
reads it as a view identifier today.

## Configuration

Paths to data, large intermediate outputs, and cached model weights are
centralized in `paths.py` and overridable via environment variables
(`SNOW_DATA_DIR`, `SNOW_WORKSPACE_DIR`, `SNOW_MODEL_CACHE_DIR`), falling back
to local relative directories if unset. No SLURM or specific cluster/account
layout is required to run the code; the `scripts/*.sbatch` files are provided
for reference but are not necessary outside that HPC environment.

## Requirements

**Docker (recommended):** `docker build -t snow-detector .` — mirrors
`scripts/setup_environment.sbatch` exactly (CUDA 12.6 devel / Ubuntu 24.04,
Python 3.12, pinned `torch==2.7.0`, `requirements.txt`, SAM3). Heavy build
(mmcv/mmengine/mmsegmentation + SAM3 + notebook extras) — expect a long
build and a multi-GB image. Run with
`docker run --gpus all -it --rm -v "$(pwd)/data:/workspace/data" snow-detector`,
then `hf auth login` inside the container for SAM3 checkpoint downloads.

**Manual:** `pip install -r requirements.txt`, then separately install
PyTorch (pinned to `torch==2.7.0` from the cu126 index — see
`scripts/setup_environment.sbatch` for the exact command and CUDA 12.6
context this was developed against; that script is written for the FRIDA
cluster's Slurm + Pyxis/enroot setup specifically, not a generic recipe —
the Dockerfile is the portable equivalent). `requirements.txt` is the real
cluster environment file, trimmed to drop a few packages this particular
repo doesn't need (see comments in the file for what and why) and with five
genuinely-used packages added back in that were missing from the original.
`mlflow` is optional (experiment tracking) and `PyQt6` is only needed for
`tools/viewer/app.py`.

## Reproducing results

Each `models/*/train.py` and `qa/train_*.py` script is
self-contained and documents its own CLI arguments in its module docstring.
`scripts/*.sbatch` show the exact invocations (variants, hyperparameters,
checkpoint paths) used to produce the results reported in the paper.
`eval_all.py --task {ground,snow,qa,smoother,ood-per-camera} --types <...>`
reproduces the evaluation tables; run with `--help` for the full list of
`--types` per task.

## Citation

If you use this code, please cite the paper (details to be added on
publication).
