#!/usr/bin/env python3
"""Gradio demo for the snowy PSG pipeline (free CPU tier friendly).

Two tabs, because the contract has two honest cases:
  * one frame   -> qa_score and psg_raw are real; psg_smoothed is null
  * a series    -> after seq_len frames of the same camera, psg_smoothed is real

Nothing here invents a smoothed value for a single image. The checkpoint audit is
shown in the UI: every stage reports how many keys it loaded and how many it left
random, and the app refuses to start if any of them is incomplete.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO), str(_REPO / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr                                              # noqa: E402
import numpy as np                                               # noqa: E402
from PIL import Image                                            # noqa: E402

from deploy import contract                                      # noqa: E402
from deploy.chain import SnowyPipeline                           # noqa: E402
from deploy.weights import ensure_weights                        # noqa: E402

WEIGHTS = Path(os.environ.get("SNOWY_WEIGHTS",
                              str(_REPO.parent / "models" / "best")))
SEQ_LEN = int(os.environ.get("SNOWY_SEQ_LEN", str(contract.DEFAULT_SEQ_LEN)))
DEVICE = os.environ.get("SNOWY_DEVICE", "cpu")

print("weights:", WEIGHTS)
ensure_weights(WEIGHTS, quiet=True)          # raises rather than run random weights
PIPE = SnowyPipeline(WEIGHTS, device=DEVICE, seq_len=SEQ_LEN)
try:
    PIPE.load_smoother()                     # else psg_smoothed can only ever be null
    SMOOTHER_ERROR = None
except Exception as exc:
    SMOOTHER_ERROR = f"{type(exc).__name__}: {exc}"
print("loaded:", {k: v.get("loaded") for k, v in PIPE.load_reports.items()})

def _report_line(name, rep):
    """The load reports have two shapes: audited state_dicts, and the QA/smoother
    loaders, which report differently. Render both honestly rather than printing
    loaded=None/None for the second kind."""
    if isinstance(rep, dict) and "loaded" in rep:
        line = (f"{name:14s} loaded={rep['loaded']}/{rep.get('model_keys')} "
                f"absent={len(rep.get('absent_from_model', []))} "
                f"random={len(rep.get('left_at_random_init', []))}")
        if rep.get("key_normalisation"):
            line += f"  keys={rep['key_normalisation']}"
        return line
    if isinstance(rep, dict):
        bits = " ".join(f"{k}={v}" for k, v in rep.items() if v is not None)
        return f"{name:14s} {bits}"
    return f"{name:14s} {rep}"


AUDIT = "\n".join(_report_line(k, v) for k, v in PIPE.load_reports.items())
if SMOOTHER_ERROR is not None:
    AUDIT += (f"\nsmoother       NOT LOADED: {SMOOTHER_ERROR} -> psg_smoothed stays null")


def _mask_image(mask):
    return Image.fromarray((np.asarray(mask) > 127).astype(np.uint8) * 255)


def _meteo_dict(t2m, visibility, weather_code):
    """Only the three columns the smoother consumes (temporal_snow.METEO_COLS)."""
    meteo = {}
    if t2m is not None:
        meteo["t2m"] = float(t2m)
    if visibility is not None:
        meteo["visibility"] = float(visibility)
    if weather_code is not None:
        meteo["weather_code"] = float(weather_code)
    return meteo


def one_frame(image, camera_id, timestamp, t2m, visibility, weather_code):
    if image is None:
        raise gr.Error("upload a frame")
    if not camera_id:
        raise gr.Error("camera_id is required — it selects the valid-region mask")
    meteo = _meteo_dict(t2m, visibility, weather_code)
    resp, raw = PIPE.predict_image(image, camera_id.strip(), timestamp or None, meteo)
    note = (f"psg_smoothed is null: the smoother is a causal GRU over "
            f"{SEQ_LEN} frames of one camera, and one image is not a window.")
    return (resp["qa_score"], resp["psg_raw"], None, note,
            _mask_image(raw["ground_mask"]), _mask_image(raw["snow_mask"]))


def series(files, camera_id, t2m, visibility, weather_code):
    if not files:
        raise gr.Error("upload the frames of one camera, oldest first")
    if not camera_id:
        raise gr.Error("camera_id is required")
    meteo = _meteo_dict(t2m, visibility, weather_code)
    last, n = None, 0
    for f in files:
        path = getattr(f, "name", f)
        last, _raw = PIPE.predict_image(path, camera_id.strip(), None, meteo)
        n += 1
    note = (f"{n} frames supplied; window {min(n, SEQ_LEN)}/{SEQ_LEN}. "
            + ("psg_smoothed is real." if last["psg_smoothed"] is not None
               else "psg_smoothed stays null until the window is full."))
    return last["qa_score"], last["psg_raw"], last["psg_smoothed"], note, \
        _mask_image(_raw["ground_mask"]), _mask_image(_raw["snow_mask"])


with gr.Blocks(title="snowy — presence of snow on ground") as demo:
    gr.Markdown(
        "## snowy — presence of snow on ground\n"
        "QA gate → ground mask → snow mask → PSG, from one webcam frame. "
        "`psg_raw` = snow pixels inside the ground mask / ground pixels × 100. "
        "`psg_smoothed` needs a window of frames of the same camera, so it is "
        "**null** for a single image — that value is not fabricated here.")
    with gr.Tab("one frame"):
        with gr.Row():
            img = gr.Image(type="pil", label="webcam frame")
            with gr.Column():
                cam = gr.Textbox(label="camera_id", placeholder="CM00014")
                ts = gr.Textbox(label="timestamp (optional; read from the filename)",
                                placeholder="2023-02-01T10:00:00")
                temp = gr.Number(label="t2m (°C, optional)")
                vis = gr.Number(label="visibility (m, optional)")
                wc = gr.Number(label="weather_code (WMO, optional)")
                go1 = gr.Button("run", variant="primary")
        with gr.Row():
            qa1 = gr.Number(label="qa_score (P usable)")
            psg1 = gr.Number(label="psg_raw (%)")
            sm1 = gr.Number(label="psg_smoothed")
        note1 = gr.Textbox(label="note", interactive=False)
        with gr.Row():
            gm1 = gr.Image(label="ground mask", type="pil")
            smask1 = gr.Image(label="snow mask", type="pil")
        go1.click(one_frame, [img, cam, ts, temp, vis, wc], [qa1, psg1, sm1, note1, gm1, smask1])

    with gr.Tab(f"a series ({SEQ_LEN}+ frames)"):
        with gr.Row():
            files = gr.Files(label="frames of one camera, oldest first")
            with gr.Column():
                cam2 = gr.Textbox(label="camera_id", placeholder="CM00014")
                temp2 = gr.Number(label="t2m (°C, optional)")
                vis2 = gr.Number(label="visibility (m, optional)")
                wc2 = gr.Number(label="weather_code (WMO, optional)")
                go2 = gr.Button("run series", variant="primary")
        with gr.Row():
            qa2 = gr.Number(label="qa_score (last frame)")
            psg2 = gr.Number(label="psg_raw (%) (last frame)")
            sm2 = gr.Number(label="psg_smoothed")
        note2 = gr.Textbox(label="note", interactive=False)
        go2.click(series, [files, cam2, temp2, vis2, wc2], [qa2, psg2, sm2, note2, gm1, smask1])

    with gr.Accordion("checkpoint audit (why these weights can be trusted)", open=False):
        gr.Textbox(value=AUDIT, lines=6, interactive=False)
    gr.Markdown(f"device `{DEVICE}`, seq_len `{SEQ_LEN}`, weights `{WEIGHTS}`. "
                f"CPU inference on full-resolution frames takes a few seconds each.")


if __name__ == "__main__":
    # GRADIO_SHARE=1 publishes a temporary public *.gradio.live URL (no account needed,
    # valid while this process runs). Useful where a hosted Space is not available.
    share = os.environ.get("GRADIO_SHARE") == "1"
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
                share=share)
