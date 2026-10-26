#!/usr/bin/env python3
"""The /predict contract, as data (no torch, no I/O).

Request
-------
    {
      "image":      "<path or base64>",     # or "image_url": "https://..."
      "camera_id":  "CM00054",
      "timestamp":  "2024-01-15T09:30:00",  # or infer from the filename
      "meteo":      {"t2m": -1.4, "visibility": 12000, "weather_code": 71},   # optional
      "frames":     [{"image": ..., "timestamp": ...}, ...]                   # optional
    }

Response
--------
    {
      "qa_score":     0.9821,      # P(frame usable) from TinyQACNN, in [0,1]
      "psg_raw":      47.3,        # snow pixels inside ground / ground pixels * 100
      "psg_smoothed": null,        # percent, 0-100, same scale as psg_raw; ONLY when
                                   # a full window is available; never faked. The GRU's
                                   # own output is a fraction in [0,1] and the pipeline
                                   # scales it x100, as the paper's figures do.
      "ground_mask":  "<base64 PNG>",
      "snow_mask":    "<base64 PNG>",
      "meta":         {"seq_len": 72, "window_seen": 5, "stages": {...}}
    }

Why psg_smoothed is nullable
----------------------------
The smoother is a GRU over a window of per-frame feature vectors (seq_len frames,
default 72 = ~36 h at 30-min sampling). A single image carries one timestep: there
is no window, so a smoothed value cannot exist. Returning the raw value in its
place, or averaging one frame, would be a fabricated number. The server therefore
returns null until it has seen seq_len frames for that camera_id, and the demo
shows the raw value alone. Feed `frames` (or call repeatedly with the same
camera_id) to accumulate a window.

Mask encoding: base64 of an 8-bit grayscale PNG at the input frame's resolution,
255 = positive. For the fusion stage the masks are 512x512 signatures, but the
masks returned here are always at the frame's own size.

Stage provenance (what each value is, and is not):
  qa_score      TinyQACNN, 3-class (clean_no_snow / snow / non_usable);
                qa_score = 1 - P(non_usable).
  ground_mask   ground student, mobilenet_v3_small + U-Net decoder, threshold 0.5.
  snow_mask     snow student, same architecture family, threshold 0.5.
  psg_raw       literal port of analysis/build_features.py:97-99.
  psg_smoothed  uni-directional GRU (causal) + the scaler shipped beside it.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from typing import Any, Optional

CLASS_PREFIX = re.compile(r"^(?P<cam>[A-Z]{2}\d{5})_(?P<stamp>\d{12})_(?P<idx>\d+)$")
CLASS_NAMES = ["clean_no_snow", "snow", "non_usable"]
DEFAULT_SEQ_LEN = 72          # packaged uni/bi smoother pairs; the paper runs use 96
MASK_THRESHOLD = 0.5
VALID_REGION_RULE = "pixels where camera_valid_regions/<cam>.png <= 127 are forced to 0"


@dataclass
class Frame:
    """One image plus what we know about it."""
    image: Any                      # path | URL | bytes | PIL.Image | np.ndarray
    timestamp: Optional[str] = None
    camera_id: Optional[str] = None

    def resolved_camera(self) -> Optional[str]:
        if self.camera_id:
            return self.camera_id
        if isinstance(self.image, str):
            m = CLASS_PREFIX.match(_stem(self.image))
            if m:
                return m.group("cam")
        return None

    def resolved_view(self) -> Optional[str]:
        """The view index NN of <cam>_<stamp>_<NN>.jpg, or None if unparseable."""
        if isinstance(self.image, str):
            m = CLASS_PREFIX.match(_stem(self.image))
            if m:
                return m.group("idx")
        return None

    def resolved_timestamp(self) -> Optional[str]:
        if self.timestamp:
            return self.timestamp
        if isinstance(self.image, str):
            m = CLASS_PREFIX.match(_stem(self.image))
            if m:
                s = m.group("stamp")           # YYYYMMDDHHMM
                return f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}:00"
        return None


def _stem(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


@dataclass
class PredictRequest:
    image: Any = None
    image_url: Optional[str] = None
    camera_id: Optional[str] = None
    timestamp: Optional[str] = None
    meteo: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "PredictRequest":
        unknown = set(d) - {"image", "image_url", "camera_id", "timestamp", "meteo", "frames"}
        if unknown:
            raise ValueError(f"unknown request fields: {sorted(unknown)}")
        if not d.get("image") and not d.get("image_url") and not d.get("frames"):
            raise ValueError("request needs one of: image, image_url, frames")
        if not d.get("camera_id"):
            raise ValueError("camera_id is required (it selects the valid-region mask)")
        return cls(image=d.get("image"), image_url=d.get("image_url"),
                   camera_id=d.get("camera_id"), timestamp=d.get("timestamp"),
                   meteo=d.get("meteo") or {}, frames=d.get("frames") or [])

    def ordered_frames(self) -> list[Frame]:
        """The frames this request covers, oldest first, ending at `image`."""
        out = [Frame(image=f.get("image"), timestamp=f.get("timestamp"),
                     camera_id=self.camera_id)
               for f in self.frames]
        if self.image is not None or self.image_url is not None:
            out.append(Frame(image=self.image or self.image_url,
                             timestamp=self.timestamp, camera_id=self.camera_id))
        return out


def encode_mask(mask, fmt: str = "base64") -> str:
    """8-bit grayscale PNG -> base64 str."""
    from PIL import Image
    import numpy as np
    if hasattr(mask, "convert") is False and not isinstance(mask, Image.Image):
        mask = Image.fromarray(np.asarray(mask).astype("uint8"))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_response(*, qa_score: float, psg_raw: float, psg_smoothed: Optional[float],
                   ground_mask, snow_mask, meta: dict) -> dict:
    """Assemble the documented response. Deliberately None-safe on psg_smoothed."""
    return {
        "qa_score": round(float(qa_score), 4),
        "psg_raw": round(float(psg_raw), 2),
        "psg_smoothed": None if psg_smoothed is None else round(float(psg_smoothed), 2),
        "ground_mask": encode_mask(ground_mask),
        "snow_mask": encode_mask(snow_mask),
        "meta": meta,
    }
