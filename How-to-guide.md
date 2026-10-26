# How to run this code

## Environment

The Dockerfile defines two images.

The full image runs the teachers, mixes their outputs into pseudo-labels, and
trains students or backbones. It installs the OpenMMLab stack (mmengine, mmcv,
mmsegmentation), anomalib, SAM3 and GroundedSAM:

```bash
docker build -t snow-detector:full .
```

The slim image only loads released checkpoints, runs `eval_all.py` and reproduces
the tables. It skips the teacher stack, so it builds faster and takes less disk:

```bash
docker build --build-arg INSTALL_SAM3=false --build-arg INSTALL_TEACHERS=false \
  -t snow-detector:slim .
```

Start a container from either tag:

```bash
docker run --gpus all -it --rm \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/model_cache:/workspace/model_cache" \
  -v "$(pwd)/workspace:/workspace/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  snow-detector:full \
  bash
```

The slim image does not need a GPU. Evaluation and the deployment path run on
CPU, so `--gpus all` can go if the host has no NVIDIA card.

## Data

The images and annotations are in the Zenodo record,
https://doi.org/10.5281/zenodo.23386293. Unpack each station archive into
`data/<CAMERA_ID>/` and the Label Studio exports into `data/annotations/`. Put
`meteo_hourly.csv` and `meteo_daily.csv` in `data/` too; only the smoother reads
them. Nothing under `data/` is tracked by git.

Every input and output path comes from `paths.py`: `DATA_DIR`, `WORKSPACE_DIR`
and `MODEL_CACHE_DIR`. Each has an environment override
(`SNOW_DATA_DIR`, `SNOW_WORKSPACE_DIR`, `SNOW_MODEL_CACHE_DIR`) and otherwise
points at `data/`, `workspace/` and `model_cache/` under the repo root, the same
three directories the `docker run` command mounts. Scripts import from
`paths.py` instead of hardcoding a location. A script that defaults elsewhere
has a bug; fix it to use `WORKSPACE_DIR` instead of papering over it with
`--output-dir`.

## The Slurm scripts

`scripts/*.sbatch` are the jobs the experiments were submitted with. They record
the settings behind each table and are not meant to run locally, since that
needs a Slurm cluster with NVIDIA's Pyxis plugin. Each script wraps one
`python ...` command, and that command is what you run under Docker.

---

# Onboarding a new camera

### Before anything else

Remove night images and files that fail to decode:

```bash
python preprocessing/remove_bad_images.py --data-dir /workspace/data --margin 60
```

`--margin` sets the hours either side of sunrise and sunset to keep, one hour by
default. The script opens and decodes every image first and moves the ones that
fail.

Draw a valid-region mask for anything static that should not count, such as a
parking lot or a distant hillside. Tools for it are in the Tools section
(`draw_valid_region.py`). Save one per camera as
`camera_valid_regions/<CAMERA_ID>.png`. Masks are found by filename, so adding
one is all it takes. A camera with several views can carry a mask per view as
`camera_valid_regions/<CAMERA_ID>__<NN>.png`, and that one wins over the
camera-wide mask.

Frames are named `<CAMERA_ID>_<YYYYMMDDHHMM>_<NN>.jpg`, where `NN` is the view.
Because the view is part of the stem, masks, annotations, teachers and students
are already kept apart per view. The smoother is the exception: build and run it
per view, not on a timestamp that mixes views together.

For annotation volume, around 20 ground and 200 snow annotations per camera
proved enough in practice.

---

### Quality gate

Both classifiers are reported in Table 4. GradBoost runs first and is not
optional, because it writes `qa_features.csv` and TinyQACNN trains on that file.
GradBoost is the stronger gate in distribution (AUROC 0.982 against 0.975), while
TinyQACNN holds up better on unseen cameras (0.951 against 0.760 OOD) and is the
one `chain.py` loads.

```bash
# 1. GradBoost writes qa_features.csv and qa_gradboost.pkl
python qa/train_qa.py \
  --img-dir /workspace/data \
  --ann-dir /workspace/data/annotations \
  --output-dir /workspace/workspace/qa_run \
  --plot-mistakes

# 2. TinyQACNN trains on GradBoost's qa_features.csv
python qa/train_qa_cnn.py \
  --features-csv /workspace/workspace/qa_run/qa_features.csv \
  --img-dir /workspace/data \
  --output-dir /workspace/workspace/qa_run
```

---

### Teachers to pseudo-labels

Some teachers download gated checkpoints, SAM3 above all. Request access on
each model page, accept the terms and wait for approval, then log in:

```bash
hf auth login
```

```bash
python segmentation/ground_teachers.py <method> --data-dir /workspace/data
python segmentation/snow_teachers.py <method> --data-dir /workspace/data
```

```bash
for g in deeplabv3 segformer dinov2_pca depth sam3; do python segmentation/ground_teachers.py $g --data-dir /workspace/data; done
for s in clipseg florence2 groundedsam2 yoloe sam3; do python segmentation/snow_teachers.py $s --data-dir /workspace/data; done
python segmentation/snow_teachers.py seggpt --data-dir /workspace/data --ann-dir /workspace/data/annotations
```

`<method>` is `deeplabv3`, `segformer`, `sam3`, `dinov2_pca` or `depth` for
ground, and `clipseg`, `florence2`, `groundedsam2`, `sam3`, `yoloe` or `seggpt`
for snow. Output goes to `WORKSPACE_DIR/segmentation/{ground,snow}/<method>`.
Override with `--output-dir` only when you need another location.

`seggpt` is few-shot rather than zero-shot. It needs `--ann-dir`, or
`--prompt-dir` for hand-picked prompt images, to draw its examples from.

Where the weights come from, all fetched at run time and none redistributed here:

| Teacher | Weights |
|---|---|
| `deeplabv3` | `deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512`, via MMSegmentation |
| `segformer` | `nvidia/segformer-b5-finetuned-ade-640-640` |
| `dinov2_pca` | `facebook/dinov2-base`, PCA and Otsu in this repo |
| `depth` | `depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf` |
| `clipseg` | `CIDAS/clipseg-rd64-refined` |
| `florence2` | `microsoft/Florence-2-large` |
| `groundedsam2` | `IDEA-Research/grounding-dino-tiny`, masks refined by `facebook/sam2-hiera-large` |
| `yoloe` | `yoloe-26*-seg.pt` from Ultralytics |
| `sam3` | the `facebookresearch/sam3` package, weights pulled from the Hub on first use |
| `seggpt` | `BAAI/seggpt-vit-large` |

Combine the teacher outputs into pseudo-labels with a fixed vote rule:

```bash
python segmentation/ground_utils.py ensemble --weighted --min-votes N
python segmentation/snow_utils.py ensemble ...
```

---

## Training the students

Two trainers, one per task.

```
MODEL=vit_lora sbatch scripts/run_snow_gt_training.sbatch
ENSEMBLE=majority_3 MODEL=dinov2_large sbatch scripts/train_student_ground.sbatch
```

Runs land in `models/student_snow/checkpoints/<run>/` and
`models/student_ground/checkpoints/<run>/`.

# Checkpoints

No trained weights ship in the repository. The record carries them flat under
`models/` with `models/CHECKPOINTS.csv`, which records the path each file was
produced at. One command rebuilds the layout the evaluation scripts expect:

```bash
python tools/link_deposit_models.py
```

Run it from the repo root after unpacking the record. It creates relative
symlinks, so the tree survives a move or a clone. `--copy` copies instead, about
10 GB, and `--dry-run` reports what it would do.

Extract the smoother scalers from `models/smoothers/scalers.zip` next to the
smoother checkpoints before evaluating the smoother.

---

# Reproducing the paper's numbers

One command from the repo root:

```bash
bash tools/reproduce_paper.sh                       # CPU; 16 cores takes hours on the snow sweep
DEVICE=cuda bash tools/reproduce_paper.sh           # the setup the paper used
TASKS="snow ground" bash tools/reproduce_paper.sh   # a subset of tasks
```

It checks the prerequisites and stops if one is missing. Then it runs snow, ground, smoother, qa and ood-per-camera, and compares each logged number with the manuscript value. Logs go to `workspace/repro/`.

---

# Deployment

`deploy/` runs QA, ground, snow, `psg_raw` and the temporal smoother for one
frame, from the CLI, an HTTP API or the Gradio app:

```bash
python deploy/chain.py --weights /path/to/deploy-set --camera CM00014 --image frame.jpg   # CLI
uvicorn deploy.api:app --port 8077                                                # HTTP API
python deploy/app.py                                                              # Gradio UI
```

`psg_smoothed` is a percentage from 0 to 100, the same unit as `psg_raw`. It
stays `null` until 72 frames from that camera have been seen, and
`meta.psg_smoothed_reason` says which condition failed.

