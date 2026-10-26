# deploy/ — the inference artefact

One frame in, `qa_score` + `psg_raw` + masks out. Add a window of frames of the
same camera and `psg_smoothed` becomes a real number instead of `null`.

Nothing outside this directory is changed by these tools, and no paper-facing code
is touched.

## The deploy set (one checkpoint per stage, ~33 MiB total)

| file | stage | what it is |
|---|---|---|
| `ground_mobilenet_0.916.ckpt` | ground mask | MobileNet-UNet student, 3-channel RGB, 298 keys |
| `snow_mobilenet_0.821.ckpt` | snow mask | same contract |
| `qa_cnn.pt` | QA gate | TinyQACNN, TorchScript, 128×128, 3-class |
| `smoother_uni/` | temporal | causal GRU, 21 features → 1, `bidirectional=False` |

Weights are **not** in this repo (`data/` is gitignored, and 128 curated
checkpoints is 16.5 GiB). Put them on the Hub and set `SNOWY_WEIGHTS_HUB`;
`deploy/weights.py` fetches them and refuses to continue if any is missing.

## Contract

```
POST /predict
in : {"image" | "image_url", "camera_id", "timestamp", "meteo": {...},
      "frames": [{"image"|"image_url", "timestamp"}, ...]}   # optional, same camera
out: {"qa_score", "psg_raw", "psg_smoothed"|null, "ground_mask", "snow_mask",
      "meta": {...}}
```

* `qa_score` — P(usable), `1 - P(non_usable)` from the 3-class gate
* `psg_raw` — snow pixels inside the ground mask / ground pixels × 100
* `psg_smoothed` — smoother output in **percent (0–100), the same scale as `psg_raw`**;
  **null** unless a full `seq_len` window of that camera has been seen, and
  `meta.psg_smoothed_reason` says why when it is null. The GRU itself emits a fraction in
  [0,1] (`train_smoother.py` normalises its target to [0,1]); the pipeline scales it ×100,
  matching the paper's figures (`visualize_smoother.py`: `smoothed_snow * 100`, "Snow %")
* masks — base64 8-bit PNG

## Commands

```bash
# audit every checkpoint before believing any number (torch-free for .ckpt)
python deploy/audit.py ../models/best/*.ckpt ../models/best/qa_cnn.pt

# one frame: qa_score + psg_raw + masks
python deploy/chain.py --image ../data/CM00014/CM00014_202212111000_27.jpg --camera CM00014

# a window: psg_smoothed becomes real (seq-len is provisional, see below)
python deploy/chain.py --smoother --seq-len 4 --camera CM00014 \
  --frames ../data/CM00014/CM00014_202212111000_27.jpg \
           ../data/CM00014/CM00014_202212121000_27.jpg \
           ../data/CM00014/CM00014_202212131300_27.jpg \
           ../data/CM00014/CM00014_202212141100_27.jpg

# acceptance: chain + deltas vs the published numbers
python deploy/acceptance.py --cameras CM00014 CM00108 --frames 6
python deploy/acceptance.py --qa-only --img-root /local/staging     # AUROC, resumable

# the API's logic without FastAPI installed
python deploy/selftest_api.py --camera CM00014 --seq-len 4

# serve it, or assemble a Space bundle (weights stay on the Hub)
uvicorn deploy.api:app --port 8000
python deploy/pack_space.py --dest /tmp/space_bundle
```

## Loads are audited, and partial loads fail

`eval_all.py` loads student checkpoints with a filtered `state_dict` under
`strict=False`, so missing keys initialise randomly and the run still prints an
IoU (vit_lora left 84/256 layer groups random). The deploy loader reports every
load and **raises** on any absent or mis-shaped key. Two things it learned the
hard way, both now explicit in the report:

* the checkpoints wrap the network as `model.<...>` while `build_model()` yields
  `<...>`, so it strips one leading component and **records that it did**
* `student_ground.models` builds its encoder with `weights="IMAGENET1K_V1"`, which
  downloads from the internet; `no_pretrained_download()` suppresses that, which is
  only safe *because* the audit then proves every layer came from the checkpoint

## Verified on real frames (2026-10-01, CPU, torch 2.4.0)

* loads: `ground 298/298 keys, absent 0, random 0`, `snow 298/298 keys, absent 0,
  random 0`, QA TorchScript loaded
* one dark non-usable frame (CM00054): `qa_score 0.0043`, `psg_raw 0.0`,
  `psg_smoothed null` — the gate rejects it, as the annotation says
* 21 annotated frames across 5 cameras: snow IoU **0.8301** / Dice **0.8829**
  (per-frame 0.798–0.944), QA gate **21/21** agreement with the annotation's
  usable / non-usable label under the repo's own `_taxonomy_to_label3`
* **QA gate over the whole annotated corpus** (1,515 frames with a local jpg;
  364 clean_no_snow / 934 snow / 217 non_usable), scored with `--qa-three-class`:

  | metric | this deploy set | paper (`tab:qa`) |
  |---|---|---|
  | 3-class macro AUROC, OOD cameras | **0.9514** | 0.9514 |
  | 3-class macro AUROC, ID cameras | 0.9877 | 0.9746 |
  | 3-class macro AUROC, all frames | 0.9760 | — |
  | binary AUROC (usable vs not) | 0.9683 | — |
  | 3-class macro F1 | 0.8549 | 0.8975 |

  The OOD number matching to four decimals identifies `qa_cnn.pt` as the checkpoint
  behind the paper's row, and identifies the paper's AUROC as a 3-class macro
  one-vs-rest average — the binary AUROC (0.9910 ID / 0.9115 OOD) is simply a
  different metric and must not be compared to it. Two open items: the ID split is
  my reconstruction (956 frames vs the paper's 961) and reads +0.013 higher, and the
  macro F1 is 0.043 below the quoted 0.8975 for reasons not yet explained — the
  per-class breakdown puts it in `non_usable` precision (0.694, 82 false positives),
  with snow→non_usable confusion accounting for most of the error.
* 4-frame window on CM00014: `psg_raw 97.96` → `psg_smoothed 96.0` (percent, as above)
* **HTTP API executed** (uvicorn, real requests, CPU): `GET /version` reports
  `smoother_loaded=True` and md5s per artefact; `POST /predict` with one frame returns
  `qa 0.7592 / psg_raw 86.17 / psg_smoothed null` with the reason
  `window not full: 1/4`; with four frames of CM00014 it returns `psg_smoothed 96.0`
  (the 0.96 fraction ×100 — see the unit note above)
  at `window 4/4` — the same numbers the CLI and `selftest_api.py` produce, so all
  three entry points agree. Malformed requests return **422** with a reason: missing
  `camera_id`, an unknown field (`extra_forbidden`), and an unreadable image path.
  Two bugs were found and fixed by doing this: neither `api.py` nor `app.py` called
  `load_smoother()`, so `psg_smoothed` could only ever be null; and unknown fields
  were silently dropped by pydantic, turning a bad request into a 500.

## Hosting

* The deploy set lives on the Hub: `garasnote/snowy-weights` (private, ~34 MiB of deploy
  files). `deploy/weights.py` pulls it at boot via `SNOWY_WEIGHTS_HUB`; nothing downloads
  silently, and a missing checkpoint fails instead of running random weights.
* A Hugging Face **Gradio** Space is no longer free: creating one on `cpu-basic` returns
  `402 Payment Required`, and only Static Spaces — which cannot run Python — are free
  without PRO. `deploy/publish_hub.py` therefore uploads the weights, then reports this
  instead of half-failing.
* The app itself runs anywhere torch does:
  `GRADIO_SHARE=1 python deploy/app.py` serves it locally and publishes a temporary
  public `*.gradio.live` URL. `deploy/api.py` exposes the same contract as
  `POST /predict` for a host that can run uvicorn.

## Verified, corrected, and still open

* **Ground IoU is scoreable.** An earlier revision of this file claimed otherwise and was
  wrong. Real ground GT (RLE brushlabels) lives in `ground_annotations.json` (267 ID
  frames, 43.3 M px) and the four OOD files `54/67/71/115` (157 / 184 / 63 / 89) — **760
  annotated frames**, every one with its image present under `data/`. The per-camera ID
  JSONs (01/09/14/70/75/80/102/108/112) do carry `Ground` / `Ground Mano` brush entries, but
  those decode to **zero pixels**; the earlier acceptance run sampled only those cameras and
  the zero result was mistaken for "the release has no ground GT". Independent
  confirmation that the metric was always available:
  `paper/eval_reproduction_findings.md` has ground students matching the published table to
  three decimals (0.8369 vs 0.837 across all five rows). Deploy-chain ground IoU on the 760
  annotated frames: `deploy/acceptance_report_ground.json`.
* **The smoother window is 72 hourly frames — three days of data**, which is how the paper
  describes it; it is not TPE-tuned. The causal GRU is window-free, so a shorter window only
  means less warm-up history. `models/temporal_snow/dataset.py` now defaults to 72, matching
  `deploy/contract.py`. Provenance is the one thing still open: the packaged uni smoother
  stores no training `seq_len` and its weights match none of the 69 shipped run checkpoints,
  so which run produced it is unconfirmed.
* **The fusion stage is not wired into the chain**, by design: `fusion_mask()` reads the six
  snow / five ground teacher masks from `workspace/segmentation/...`, which are not in the
  repo, and refuses to zero-fill a missing channel. The paper's deployed chain does not use
  it.
* API-layer wiring **was** executed here (uvicorn, real HTTP requests, 422s on bad input) as
  was the Gradio app and its callbacks — see above; `selftest_api.py` covers the same path
  without FastAPI installed.
* CPU inference on full-resolution frames takes a few seconds each: this is a demo, not a
  service.
