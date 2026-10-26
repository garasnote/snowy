# deploy

One frame in, a snow fraction out. `chain.py` runs the four stages in order: quality
gate, ground student, snow student, optional smoother. The CLI, the HTTP API and the
Gradio app all call that same path.

## Weights

Three files, from the Zenodo record, placed in one directory:

```
ground_pl_dinov2b.ckpt
snow_pl_convnextt_k3.ckpt
qa_cnn.pt
```

`--weights` points at that directory. `weights.py` holds the list, `audit.py` checks
each file against the model before anything is scored, and a partial load raises
instead of guessing.

## Contract

Input: an image, a camera id, an optional view id, and three meteorological values
(`t2m` in Celsius, `visibility` in metres, `weather_code` as WMO). `--meteo-csv`
looks the last three up from `data/meteo_hourly.csv` instead of taking them on the
command line.

Output: `qa_score`, `psg_raw`, `psg_smoothed`, the ground and snow masks as base64
PNG, and a `meta` block. `psg_smoothed` is null until 72 hourly frames exist for that
camera, and `meta.psg_smoothed_reason` says why.

Valid-region masks live under `preprocessing/segmentation/camera_valid_regions/`,
one per camera as `<CAMERA_ID>.png`, or one per view as `<CAMERA_ID>_v<NN>.png`.

## Commands

```
python deploy/chain.py --weights /path/to/weights --camera CF00001 --image frame.jpg
python deploy/chain.py --weights /path/to/weights --camera CF00001 --image frame.jpg --meteo-csv data/meteo_hourly.csv
uvicorn deploy.api:app --host 0.0.0.0 --port 8099
python deploy/app.py
python deploy/selftest_api.py
```

`chain.py` prints JSON to stdout. `api.py` adds `/healthz`, `/version` and `/predict`.
`app.py` is the Gradio interface, on port 7860 unless `PORT` says otherwise.

## Hosting

`pack_space.py` assembles a Hugging Face Space bundle; the weights are fetched at
startup from the Zenodo record and never committed. `publish_hub.py` pushes the
bundle. The Space and the API expose the same `chain.py` path.
