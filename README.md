# Snowy: webcam snow cover for Alpine weather stations

Code for *Multi-Modal Machine Learning for Image-Based Snow Coverage Estimation*
(Osorio Rivera and Faganeli Pucer, submitted to *Computers & Geosciences*).

The pipeline estimates snow on the ground from fixed webcam images. A quality gate drops frames that cannot be used, then a ground student segments the terrain the camera can see and a snow student segments snow inside that terrain. A recurrent smoother turns the per-frame fractions into a daily series, which is compared against manual snow records from the Slovenian Environment Agency (ARSO).

Data, annotations and checkpoints: https://doi.org/10.5281/zenodo.23386293

Running instructions: [How-to-guide.md](How-to-guide.md)

## What is where

| Path | What it is |
|---|---|
| `preprocessing/` | Frame selection: daytime margin, decode checks, valid-region masks |
| `qa/` | Quality gate. `train_qa.py` fits HistGradientBoosting on handcrafted features and writes `qa_features.csv`; `train_qa_cnn.py` trains TinyQACNN. Both are reported in Table 4 |
| `segmentation/` | Teacher models (`ground_teachers.py`, `snow_teachers.py`) and pseudo-label assembly (`ground_utils.py`, `snow_utils.py`) |
| `models/student_ground/`, `models/student_snow/` | Distilled segmentation students, fine-tuning and inference |
| `models/temporal_snow/` | Recurrent smoother (GRU/LSTM, one or two directions) and its training data. The scalers for the eight released smoothers are in the record, `models/smoothers/scalers.zip` |
| `analysis/` | Feature building, temporal diagnostics, and `meteo_perview.py`, which recomputes the four-station table. `snow_timeseries.csv` ships in the record |
| `eval_all.py` | Entry point for evaluation: `--task ground\|snow\|qa\|smoother\|ood-per-camera` |
| `deploy/` | `chain.py` runs QA, ground, snow, `psg_raw` and the smoother for one frame, through a CLI, an HTTP API (`api.py`) or a Gradio app (`app.py`) |
| `tools/` | `reproduce_paper.sh`, valid-region masks, viewer, figure and video helpers |
| `scripts/` | The Slurm jobs the experiments were submitted with, kept for provenance. Account, partition and user paths are generic |

## Install

Python 3.11 or later.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126
```

`requirements.txt` covers the analysis code and the teacher stack, including
mmengine, mmcv and mmsegmentation for the DeepLabV3+ teacher, Streamlit for one
comparison app and PyQt6 for the frame viewer. The `Dockerfile` builds the same
environment. 

SAM3 needs Python 3.12, and `--build-arg INSTALL_SAM3=false` skips
it.

`paths.py` resolves the data root, the working directory and the checkpoint directory:

```bash
export SNOW_DATA_DIR=/path/to/data
export SNOW_WORKSPACE_DIR=/path/to/workspace
```

## Hardware

Training and evaluation ran on NVIDIA A100 and L4 GPUs with PyTorch 2.7.0 and CUDA 12.6. Tthe `Dockerfile` pins the same versions.

## Data

ARSO owns the imagery and the meteorological records. They are redistributed in
the Zenodo record under CC BY 4.0 with attribution, as ARSO permits. The record
also holds the annotations, the derived tables and the checkpoints behind every
result table.

The code reads them from these paths:

```
data/                          station images, unpacked as <CAMERA_ID>/*.jpg
data/annotations/              Label Studio exports (ground masks, snow masks, QA tags)
analysis/snow_timeseries.csv   per-frame model output
qa/checkpoints/qa_features.csv written by qa/train_qa.py
models/                        checkpoints, named as in models/CHECKPOINTS.csv
```

Filenames look like `<CAMERA_ID>_<YYYYMMDDHHMM>_<NN>.jpg`, where `NN` is the view. Onboarding a new camera is covered in [How-to-guide.md](How-to-guide.md).

## Reproducing the paper

`tools/reproduce_paper.sh` runs the evaluations and compares every number it
logs with the one printed in the manuscript. Missing data, annotations,
checkpoints or derived CSVs stop it with an error instead of producing an empty
table.

```bash
bash tools/reproduce_paper.sh                 # CPU, logs to workspace/repro/
DEVICE=cuda bash tools/reproduce_paper.sh     # GPU
TASKS="snow ground" bash tools/reproduce_paper.sh
```

Single tasks go through `eval_all.py`:

```bash
python eval_all.py --task ground --types students --breakdown            # Table 5
python eval_all.py --task snow   --types students                        # Tables 6, 7
python eval_all.py --task qa     --types cnn                             # Table 4
python eval_all.py --task smoother                                       # Table 10
python eval_all.py --task ood-per-camera                                 # per-camera OOD
```

Snow is scored on the held-out split, the last 20% of each camera's dates, which no student saw in training. The fine-tuned out-of-distribution row uses fold 4.

## Deployment

`deploy/chain.py` runs one frame, or a window of frames when `psg_smoothed` is wanted:

```bash
python deploy/chain.py --weights /path/to/best --camera CM00014 \
    --timestamp 2026-01-15T12:00 --frames frames/*.jpg
```

The smoother needs three meteorological values: `t2m` in °C, `visibility` in metres and `weather_code` in WMO codes. Pass them as `--t2m`, `--visibility` and `--weather-code`, or let `--meteo-csv` (default `data/meteo_hourly.csv`) look\ them up from `--camera` and `--timestamp`. If none are supplied, the smoother uses its training defaults (2.8 °C, 10000 m, code 0) and `psg_smoothed` means nothing. The HTTP API takes the same three keys in its `meteo` object and returns 422 for anything else.

Inference runs on CPU. The deploy set is about 500 MB and peaks near 1.2 GB of RAM, so 2 vCPU and 2 GB are enough.

## Inputs, outputs and options

Frame files are JPEGs named `<CAMERA_ID>_<YYYYMMDDHHMM>_<NN>.jpg`. The camera id selects the valid-region mask, and the timestamp is read from the name when you do not pass one. A single frame produces a quality score, a ground mask, a snow mask and `psg_raw`, which is snow pixels over ground pixels in percent. A window of 72 frames from one camera also produces `psg_smoothed`, in the same percent unit; below 72 frames it stays `null` and the response says how many were seen. The smoother needs the three meteorological values described under Deployment, and falls back to fixed defaults without them.

`eval_all.py` reads the images under `data/`, the Label Studio exports under `data/annotations/` and the checkpoints under `models/`. It writes tables to stdout, and `tools/reproduce_paper.sh` adds a log per task under `workspace/repro/`. `--types` chooses which model families are scored, `--breakdown` adds the per-camera table for ground and `--device` picks cpu or cuda.

## Models

Checkpoints are not in git (`*.ckpt` is ignored). Every model behind a result
table is in the record under `models/`, indexed by `models/CHECKPOINTS.csv`,
which maps each short name to the run and epoch file it came from. Names read
`task_regime_architecture`: `pl` for pseudo-label initialisation, `ft` for
fine-tuning, `gt` and `gtf` for ground-truth initialisation and fine-tuning,
`_f0` to `_f4` for cross-validation folds, `_k2` to `_k5` for the pseudo-label
consensus threshold.

Unpack the record and rebuild the tree the scripts expect before evaluating anything:

```bash
python tools/link_deposit_models.py
```

It reads `models/CHECKPOINTS.csv` and creates relative symlinks, so the tree survives a
move or a clone. `--dry-run` lists what it would link, `--copy` copies instead of linking.

The deployment path expects three files in one directory, passed as `--weights`:

| File | Model |
|---|---|
| `ground_pl_dinov2b.ckpt` | DINOv2-Base, pseudo-label |
| `snow_pl_convnextt_k3.ckpt` | ConvNeXt-Tiny, pseudo-label, k=3 |
| `qa_cnn.pt` | TinyQACNN quality gate |

`psg_smoothed` additionally needs `smoother_uni/smoother_gru_uni_best.ckpt` with its `scaler.json`, both extracted from `models/smoothers/scalers.zip` in the record. Without them the chain returns `psg_raw` and no smoothed value.

The snow stage uses the ConvNeXt-Tiny pseudo-label student, the best OOD snow IoU in the paper at 0.827. The ground stage uses DINOv2-Base on pseudo-labels: 0.821 OOD against 0.829 for the fine-tuned DINOv2-Large, at a third of the size.

## Results

`tools/reproduce_paper.sh` produces these numbers from the released checkpoints
and the record's `snow_timeseries.csv`.

| Quantity | Value |
|---|---|
| Ground segmentation, best OOD (four held-out cameras) | DINOv2-Large FT, IoU 0.829 |
| Snow segmentation, best OOD | ConvNeXt-Tiny PL, IoU 0.827 |
| QA gate, best OOD | TinyQACNN, AUROC 0.951, F1 0.757 |
| Daily snow presence F1, raw output | 0.903 |
| Daily snow presence F1, best smoother | 0.910 |
| Daily snow presence F1, smoother trained without the site | 0.784 to 0.798 |

The first three rows are out-of-distribution numbers. In distribution, IoU is 0.88 to 0.90 for ground and 0.82 to 0.91 for snow.

## Not included

The intermediate teacher outputs, which are several hundred GB, and the weights of the
pretrained teachers backbones, which come from their own projects.
[How-to-guide.md](How-to-guide.md) lists where each teacher comes from.

## Citation

```bibtex
@article{osorio2026snowy,
  title   = {Multi-Modal Machine Learning for Image-Based Snow Coverage Estimation},
  author  = {Osorio Rivera, Juan Pablo and Faganeli Pucer, Jana},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {Submitted}
}
```

The dataset and the checkpoints have their own DOI,
https://doi.org/10.5281/zenodo.23386293. Cite both when you reuse the models.

## License and acknowledgements

The code is under the GNU General Public License v3.0, see `LICENSE`. The imagery and the meteorological records are © Agencija Republike Slovenije za okolje (ARSO), https://www.arso.gov.si/, redistributed under CC BY 4.0 with attribution.

We thank the FRIDA computing cluster at the University of Ljubljana, and its technical and organizational team, for the computational infrastructure used for training, testing, and inference in this work. We are grateful to Miha Demšar (ARSO) for providing webcam recordings and advice during development, and to Asst. Prof. Dr. Jana Faganeli Pucer for guidance throughout the project.

## Contact

Juan Pablo Osorio Rivera. Open an issue in this repository.
