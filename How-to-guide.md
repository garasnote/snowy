# How-To Guide

## Environment

```bash
docker build -t snow-detector:full .
# without SAM3 / the OpenMMLab teacher stack (mmcv, mmengine, mmsegmentation, anomalib):
docker build --build-arg INSTALL_SAM3=false --build-arg INSTALL_TEACHERS=false -t snow-detector:slim .

docker run --gpus all -it --rm \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/model_cache:/workspace/model_cache" \
  -v "$(pwd)/workspace:/workspace/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  snow-detector:full \
  bash
```

The `~/.cache/huggingface` mount is what makes downloaded checkpoints
(teachers, student backbones) persist across container restarts — without
it, every fresh container re-downloads them from scratch (Florence-2 alone
is 1.55GB).

Two things about this image worth knowing before you trust a number from it:

- **The repo is copied into the image at build time** (`COPY . /workspace`), so
  code edited after the build is *not* inside the container. Check before relying
  on it:
  ```bash
  grep -c pick_best_ckpt /workspace/eval_all.py   # 0 = image predates the fix
  ```
  Rebuild (`docker build -t snow-detector:full .`) after changing any code, or
  bind-mount the files you are working on.
- **`data/` must contain real directories, not symlinks pointing outside the
  mount.** A symlinked camera directory resolves to a path that does not exist
  inside the container, and the evaluation then scores zero frames for that camera
  without an error. If your working copy uses symlinks into a larger tree, mount
  that tree too (or avoid symlinks inside `data/`).

`data/` needs `<CAMERA_ID>/<stem>.jpg` images, `data/annotations/*.json`
(Label Studio exports), and optionally `data/meteo_hourly.csv` /
`meteo_daily.csv`. None of this ships in the repo — `/data/` is
gitignored, populate it yourself.

**All data/output paths come from `paths.py`** (`DATA_DIR`, `WORKSPACE_DIR`,
`MODEL_CACHE_DIR`), each overridable via an env var
(`SNOW_DATA_DIR`/`SNOW_WORKSPACE_DIR`/`SNOW_MODEL_CACHE_DIR`) and otherwise
defaulting to `data/` / `workspace/` / `model_cache/` under the repo root —
exactly the three directories the `docker run` command above mounts. Any
script that writes real output imports from `paths.py` rather than
hardcoding a path; if you ever see a script default to somewhere else, that
script has a bug — it should be pointed at `WORKSPACE_DIR`, not patched with
a one-off `--output-dir` every time.

**`scripts/*.sbatch` files require Slurm + NVIDIA's Pyxis plugin** and
won't run without it. They're kept for provenance, not as something to
execute locally — the Docker setup above is the portable equivalent. To
see what a given `.sbatch` actually does, find the
`python <script>.py --flags` line inside its `bash -c "..."` block and
run that directly.

---

# New Camera Onboarding

### Before Anything

- **Remove night + corrupt/truncated images**:
  ```bash
  python preprocessing/remove_bad_images.py --data-dir /workspace/data --margin 60
  ```
  1-hour sunrise/sunset margin by default (`--margin`). Also opens and
  fully decodes every image first, moving anything that fails to decode.

- **Valid-region masks** (hand-drawn cutouts for parking lots / distant
  landscape / other static obstructions) — see the Tools section for
  `draw_valid_region.py`. One mask per camera — no sub-view support (see
  below). The camera allowlist that actually applies these masks is
  hardcoded in two places (`analysis/temporal_diagnostics.py`'s
  `GRID_VALID_REGION_CAMERAS`, `tools/viewer/app.py`'s
  `CAMERA_VALID_REGIONS`) — add the new camera to both or the mask is
  silently ignored.

- **Multi-view cameras**: filenames are
  `<CAMERA_ID>_<YYYYMMDDHHMM>_<NN>.jpg`. `NN` identifies the physical
  view. If a camera has multiple views, nothing in this pipeline
  separates them — ground truth, valid-region masks, and every
  teacher/student/eval script key on `(camera, stem)` only. Onboarding a
  multi-view camera means either annotating/training per-view manually,
  or building real sub-view support first.

- **Annotation targets**: roughly 20 ground / 200 snow per camera.

---

### QA Gate

Two classifiers: GradBoost is a required upstream step (produces
`qa_features.csv`) and a baseline; TinyQACNN is
the model actually reported in the paper.

```bash
# 1. GradBoost — produces qa_features.csv + qa_gradboost.pkl
python qa/train_qa.py \
  --img-dir /workspace/data \
  --ann-dir /workspace/data/annotations \
  --output-dir /workspace/workspace/qa_run \
  --plot-mistakes

# 2. TinyQACNN — trains from GradBoost's qa_features.csv
python qa/train_qa_cnn.py \
  --features-csv /workspace/workspace/qa_run/qa_features.csv \
  --img-dir /workspace/data \
  --output-dir /workspace/workspace/qa_run
```

`--img-dir` must point at the actual image root (default assumes
`data/balanced_snow/`). Frames tagged `Non-Usable + No Snow` are
recovered as valid usable negatives per the taxonomy fix described in the
paper's Methods — don't read a low "no-snow rejected" number as a
problem, that's correct behavior.

---

### Teachers → Pseudo-Labels

**Authenticate with HuggingFace first** — some teachers (SAM3 in
particular) pull gated checkpoints:

```bash
hf auth login
```

```bash
python segmentation/ground_teachers.py <method> --data-dir /workspace/data
python segmentation/snow_teachers.py <method> --data-dir /workspace/data
```

To run all methods:
```bash
for g in deeplabv3 segformer dinov2_pca depth sam3; do python segmentation/ground_teachers.py $g --data-dir /workspace/data; done
for s in clipseg florence2 groundedsam2 yoloe sam3; do python segmentation/snow_teachers.py $s --data-dir /workspace/data; done
python segmentation/snow_teachers.py seggpt --data-dir /workspace/data --ann-dir /workspace/data/annotations
```

`<method>`: `deeplabv3`, `segformer`, `sam3`, `dinov2_pca`, `depth` for
ground; `clipseg`, `florence2`, `groundedsam2`, `sam3`, `yoloe`, `seggpt`
for snow. Output defaults to `WORKSPACE_DIR/segmentation/{ground,snow}/<method>`
(see `paths.py` note above) — override with `--output-dir` only if you need
a non-default location.

`seggpt` is few-shot (in-context) rather than zero-shot — it needs
`--ann-dir` (or `--prompt-dir` for manually curated prompt images) to pull
prompt examples from, unlike the other five methods.

Two ways to combine teacher outputs into pseudo-labels:

```bash
# Rule-based ensemble (fixed majority/weighted vote)
python segmentation/ground_utils.py ensemble --weighted --min-votes N
python segmentation/snow_utils.py ensemble ...

# FusionCNN (learns a per-pixel weighting instead of a fixed vote rule)
python models/fusion_cnn/fusion_cnn.py train --task ground --epochs 50
python models/fusion_cnn/fusion_cnn.py train --task snow   --epochs 50
python models/fusion_cnn/fusion_cnn.py infer --task ground
python models/fusion_cnn/fusion_cnn.py infer --task snow
```

FusionCNN reads teacher predictions from a fixed, ordered list
(`GROUND_TEACHERS`/`SNOW_TEACHERS` in `fusion_cnn.py`) — that order defines
the model's input channels, so it must stay identical between the run that
trained a checkpoint and any later inference run using it. `infer` checks
this automatically and raises an error on mismatch rather than silently
producing misaligned predictions; if you add, remove, or reorder a teacher
in that list, retrain — an old checkpoint will no longer load against it.
A missing teacher output folder or per-image file is zero-filled and
reported as a `WARNING`, not a silent gap — check for these before trusting
a low-quality run.

---

### Students

```bash
python models/student_ground/train.py ...
python models/student_snow/train.py ...
```

Checkpoints are always full model saves (no heads/decoder-only mode).

`models/multi_head/train_multihead.py` (DINOv3MultiHead) loads its
backbone from `MODEL_CACHE_DIR` if populated, otherwise from
HuggingFace — see the comment at the top of that file for the one-time
population command. Mount `~/.cache/huggingface` as a persistent Docker
volume to avoid re-downloading it every container restart.

---

### Evaluate

```bash
python eval_all.py --task {ground,snow,qa,smoother,ood-per-camera} --types ...
```

`--task snow` alone defaults to `--types vlms baselines` — it does **not**
evaluate the student models, and prints a single-line summary table. To
reproduce the student tables add the type explicitly, and add `--val-only`
to use the temporally held-out validation split the reported in-distribution
numbers come from:

```bash
python eval_all.py --task snow --types students --val-only --device cpu
python eval_all.py --task ground --types students --breakdown --device cpu
```

Building a student model instantiates its pretrained backbone from the hub
(e.g. ConvNeXt downloads ~109 MB) before the checkpoint is loaded over it, so
the first evaluation of a given architecture needs network access even when
every checkpoint is already on disk. Mount `~/.cache/huggingface` as a
persistent volume, as with the teachers.

Visualize in `tools/viewer/app.py` (PyQt6).

---

# Checkpoints

No trained weights ship in the repo (see README, "What's not here"). The
evaluation scripts reference them by *relative* path, so they must be placed
inside the repo root, in exactly these directories:

```
models/student_ground/checkpoints/<run>/best-*.ckpt
models/student_snow/checkpoints/snow_gt_mobilenet_20260522_052052/best-*.ckpt
models/student_snow/checkpoints/snow_gt_convnext_unfrozen/best-*.ckpt
models/student_snow/checkpoints/snow_gt_vit_lora_unfrozen/best-*.ckpt
models/student_snow/checkpoints/snow_{convnext,mobilenet,vit_lora}_maj{2,3,4,5}_*/best-*.ckpt
models/temporal_snow/checkpoints/<run>/{best-*.ckpt,final.ckpt}
models/multi_head/checkpoints/<run>/best-epoch=*-avgval/.../snow_iou=*.ckpt
models/fusion_cnn/checkpoints/{fusion_ground,fusion_snow}/best-*.ckpt
models/best/{ground_mobilenet_*.ckpt,snow_mobilenet_*.ckpt,qa_cnn.pt,smoother_*.ckpt}
preprocessing/qa/checkpoints/qa_cnn.pt
```

Every one of these loads sits behind a `Path(...).exists()` or an empty-glob
check, so **a missing or misplaced checkpoint does not raise**. The student
blocks now say so in the log (`mobilenet: no checkpoint found under
models/student_snow/checkpoints/ - row skipped`) and the checkpoint block prints
the file it loaded; the teacher, smoother and QA blocks still skip quietly. If a
table comes back empty or short, check the directory layout before suspecting the
code — an empty summary table is the classic symptom of a broken path, not a bug
in the evaluation. Symlinking a checkpoint tree kept elsewhere into these paths
works and is the least invasive option, and symlinks should be **relative** so the
tree keeps working when it is moved, cloned or extracted from an archive:

```bash
ln -s /path/to/weights/student_snow/checkpoints models/student_snow/checkpoints
```

## Which checkpoint gets evaluated

Two different rules, and they are not interchangeable.

**Between runs.** Blocks that glob run directories (`snow_{variant}_maj{k}_*`,
`snow_gt_{variant}_kfold_{fold}_*`, and the same for OOD) sort them by
modification time and take the newest. That is deliberate: training runs from
before the annotation-leakage fix report inflated validation IoU (~0.99), so
recency is the cheap proxy for "trained after the fix". The cost is that the
evaluated run depends on the filesystem — a fresh `git clone`, an `rsync` or a
Zenodo download rewrites mtimes. To pin one, name the run directory explicitly
instead of globbing it.

**Within a run.** `pick_best_ckpt()` takes the `best-*.ckpt` with the highest
validation IoU encoded in its filename, never the newest file: several epochs of
one run sit side by side in the same directory, and mtime among them means
nothing. The eval prints what it loaded, so a log is enough to tell which epoch
produced a number:

    mobilenet: checkpoint snow_gt_mobilenet_20260522_052052/best-epoch=65-iouval_iou=0.821.ckpt

If the chosen filename reports a validation IoU above 0.95, the run predates the
leakage fix and a warning is printed.

## Loads are strict

Student loads check key parity before scoring and raise when anything is missing:

    checkpoint … does not fit build_model('vit_lora'): 172/256 parameters matched,
    84 absent (e.g. ['encoder.blocks.0.attn.qkv.down.weight', …]).
    Refusing to score a partially initialised model.

This matters. The earlier behaviour — filtering the `state_dict` and loading with
`strict=False` — evaluated partially initialised models silently, printing a
score for layers that were never loaded. One published row is affected:
`snow_gt_vit_lora_unfrozen` was trained when the LoRA adapter targeted
`attn.proj`, while `student_ground/models.py` now wraps the fused `attn.qkv`, so
the released code cannot load those checkpoints and refuses instead of guessing.
Every other student and ground checkpoint has complete parity, so for them the
strict check changes nothing.

---

## Channel completeness (FusionCNN)

Every teacher channel must have a real mask for a frame. If one is missing,
FusionCNN zero-fills that channel for that sample — and the network then learns
that an all-zero channel is a legitimate input for that teacher. Check before
training, and after any re-wiring:

    python tools/verify_fusion_inputs.py

It prints have/exp per channel and fails if any channel is short. The five snow
teachers other than Florence-2 share a 44-frame hole (CF00001, from 2022-12-11)
and seggpt has 40 more of its own; those masks were never produced and cannot be
recovered by renaming (checked: no `<stem> (1)_snow.png` variant exists).

If the masks cannot be regenerated yet, train on the frames that are complete
rather than zero-filling:

    python models/fusion_cnn/fusion_cnn.py train --task snow --complete-only

It drops exactly the frames lacking a mask from any teacher, prints what it
dropped and why, and leaves every channel a genuine prediction — 594 of 678
frames for snow, 504 train / 90 val, zero zero-fill warnings.

# Deployment / Usage

Not built yet. `models/student_ground/inference.py` and
`models/student_snow/inference.py` take `--data-dir` (batch directories),
no single-image CLI, no end-to-end `predict.py` chaining
QA → ground → snow → temporal-smoother for one new frame.
