#!/usr/bin/env python3
"""HTTP API for the snowy PSG pipeline.  POST /predict.

Contract (deploy/contract.py is the normative definition; deploy/api.py is the
transport):

  in : {"image" | "image_url", "camera_id", "timestamp", "meteo": {...},
        "frames": [{"image"|"image_url", "timestamp"}, ...]}   # optional series
  out: {"qa_score", "psg_raw", "psg_smoothed"|null, "ground_mask", "snow_mask",
        "meta": {...}}

psg_smoothed is null until SnowyPipeline has seen seq_len frames of that camera:
the smoother is a causal GRU over a fixed window, and a one-shot request cannot
produce it. Faking it from a single image is the failure mode this release is
built to avoid.

psg_smoothed is in percent (0-100), the same unit as psg_raw. The GRU itself emits
a fraction in [0,1] and the pipeline scales it x100, exactly as the paper's figures
do (visualize_smoother.py: `smoothed_snow * 100`, ylabel "Snow %").

Run locally:
    uvicorn deploy.api:app --host 0.0.0.0 --port 8000

Env:
    SNOWY_WEIGHTS   directory with the deploy checkpoints (default ../models/best)
    SNOWY_SEQ_LEN   smoother window (default 72)
    SNOWY_DEVICE    cpu | cuda (default cpu)
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO), str(_REPO / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException                      # noqa: E402
from pydantic import BaseModel, Field                           # noqa: E402

from deploy import contract                                     # noqa: E402
from deploy.chain import SnowyPipeline                          # noqa: E402

app = FastAPI(title="snowy PSG inference", version="0.1.0")

_STATE: dict = {"pipe": None, "error": None}
CKPT_FILES = {"ground_student": "ground_pl_dinov2b.ckpt",
              "snow_student": "snow_pl_convnextt_k3.ckpt",
              "qa": "qa_cnn.pt"}


def get_pipeline() -> SnowyPipeline:
    """Build once. A failed checkpoint audit surfaces as 503, never as a number."""
    if _STATE["pipe"] is None:
        if _STATE["error"]:
            raise HTTPException(status_code=503, detail=_STATE["error"])
        try:
            weights = os.environ.get("SNOWY_WEIGHTS", str(_REPO.parent / "models" / "best"))
            _STATE["pipe"] = SnowyPipeline(
                weights,
                device=os.environ.get("SNOWY_DEVICE", "cpu"),
                seq_len=int(os.environ.get("SNOWY_SEQ_LEN", str(contract.DEFAULT_SEQ_LEN))),
            )
            # without this the smoother is never loaded and psg_smoothed can only ever
            # be null, which looks like a contract violation rather than a missing step
            try:
                _STATE["pipe"].load_smoother()
            except Exception as exc:
                _STATE["smoother_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:                       # loud, never silent
            _STATE["error"] = f"{type(exc).__name__}: {exc}"
            raise HTTPException(status_code=503, detail=_STATE["error"])
    return _STATE["pipe"]


def _md5(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class FrameIn(BaseModel):
    model_config = {"extra": "forbid"}       # an unknown field must be rejected, not dropped
    image: str | None = None
    image_url: str | None = None
    timestamp: str | None = None


class MeteoIn(BaseModel):
    """The three meteorological columns the smoother consumes."""
    model_config = {"extra": "forbid"}
    t2m: float | None = None
    visibility: float | None = None
    weather_code: float | None = None


class PredictIn(BaseModel):
    model_config = {"extra": "forbid"}
    image: str | None = None
    image_url: str | None = None
    camera_id: str
    timestamp: str | None = None
    meteo: MeteoIn | None = None
    frames: list[FrameIn] | None = None


@app.get("/healthz")
def healthz():
    """Liveness without building the models."""
    return {"ok": True, "built": _STATE["pipe"] is not None,
            "error": _STATE["error"], "repo": str(_REPO)}


@app.get("/version")
def version():
    """Exactly which artefacts are serving, so a number can be traced to a file."""
    pipe = get_pipeline()
    return {
        "weights_dir": str(pipe.weights),
        "seq_len": pipe.seq_len, "device": str(pipe.device),
        "checkpoints": {k: _md5(pipe.weights / v) for k, v in CKPT_FILES.items()},
        "load_reports": pipe.load_reports,
        "qa_is_torchscript": getattr(pipe, "qa_torchscript", None) is not None,
        "smoother_loaded": getattr(pipe, "smoother", None) is not None,
        "smoother_error": _STATE.get("smoother_error"),
    }



def _materialise(image: str) -> tuple[str, str | None]:
    """Turn a path, data URL, raw base64 blob or http(s) URL into a local file.

    Returns (path, temp_path); the temp file is deleted by the caller.
    """
    try:
        if Path(image).exists():
            return image, None
    except OSError:                      # base64 blobs are too long to be a filename
        pass
    payload = image
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    if payload.startswith("http://") or payload.startswith("https://"):
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        with os.fdopen(fd, "wb") as fh:
            with urllib.request.urlopen(payload, timeout=30) as r:
                fh.write(r.read())
        return tmp, tmp
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return image, None            # let the pipeline report the bad path
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as fh:
        fh.write(raw)
    return tmp, tmp


@app.post("/predict")
def predict(body: PredictIn):
    pipe = get_pipeline()

    request = body.model_dump()
    if body.frames:
        request["frames"] = [f.model_dump() for f in body.frames]
    try:
        req = contract.PredictRequest.from_dict(
            {k: v for k, v in request.items() if v is not None})
    except ValueError as exc:                            # contract violations -> 422
        raise HTTPException(status_code=422, detail=str(exc))

    frames = req.ordered_frames()
    if req.meteo is None:
        meteo = {}
    elif hasattr(req.meteo, "model_dump"):
        meteo = req.meteo.model_dump(exclude_none=True)
    else:
        meteo = dict(req.meteo)
    resp = None
    temps: list[str] = []
    try:
        for frame in frames:                             # oldest first, per camera
            path, tmp = _materialise(frame.image)
            if tmp:
                temps.append(tmp)
            resp, _raw = pipe.predict_image(path, req.camera_id,
                                            frame.resolved_timestamp(), meteo)
    except (FileNotFoundError, IsADirectoryError, OSError, ValueError) as exc:
        # an unreadable image is a bad request, not a server fault: say which frame
        raise HTTPException(status_code=422,
                            detail=f"could not read the frame: {type(exc).__name__}: {exc}")
    finally:
        for tmp in temps:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if resp is None:
        raise HTTPException(status_code=422, detail="no frames in request")

    resp.setdefault("meta", {})["frames_used"] = len(frames)
    if len(frames) < pipe.seq_len:
        resp["meta"]["psg_smoothed_note"] = (
            f"null: {len(frames)} frame(s) supplied, window is {pipe.seq_len}")
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
