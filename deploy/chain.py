#!/usr/bin/env python3
"""One-frame → the /predict dict: QA -> ground -> snow -> psg_raw -> (smoother).

Design rules this file follows, all of them consequences of things that went
wrong earlier in this project:

* Every checkpoint load is audited. A filtered state_dict under strict=False let
  an earlier ViT-LoRA evaluation report an IoU with 84 of 256 layers still at
  random init. deploy/audit.py raises instead; this module uses it on every load.
* Missing teacher masks are never zero-filled. fusion_cnn.py silently zero-fills
  (models/fusion_cnn/fusion_cnn.py:495-500), which turns a missing channel into a
  plausible-looking mask. The fusion helper here refuses instead.
* psg_smoothed is null unless a real window exists. The smoother is a causal GRU
  over seq_len consecutive frames; one image is one timestep.
* Features are computed by the repo's own functions, not reimplemented:
  analysis.build_features.ts_compute_snow_features (:60-117),
  analysis.build_features.imgfeat_extract_features (:435-498),
  temporal_snow.dataset.build_feature_columns (:147-153) and FeatureScaler
  (:171-180). Only the two single-frame preprocessing steps that live inside a
  DataFrame path in dataset.py (:74-93) are mirrored here, and they are marked.

Weights expected (the deploy set, ~1.3 GB, dominated by DINOv2-Large):
    models/best/ground_dinov2l_ft.ckpt         ground student, DINOv2-Large FT (fold 4)
    models/best/snow_convnextt_pl.ckpt         snow student,   ConvNeXt-Tiny PL (k=3)
    models/best/qa_cnn.pt                      TinyQACNN, TorchScript (or qa_cnn.ckpt)
    models/best/smoother_uni/smoother_gru_uni_best.ckpt + scaler.json

CLI:
    python deploy/chain.py --image data/CM00054/CM00054_202401150930_00.jpg \
                           --camera CM00054 --weights /path/to/models/best
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for p in (str(_REPO), str(_REPO / "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

import deploy.audit as audit_mod                     # noqa: E402
from deploy import contract                          # noqa: E402
from deploy.contract import (CLASS_NAMES, DEFAULT_SEQ_LEN, MASK_THRESHOLD,  # noqa: E402
                             Frame, PredictRequest, build_response)

# ── paths inside the repo ────────────────────────────────────────────────────
VALID_REGION_DIR = _REPO / "preprocessing" / "segmentation" / "camera_valid_regions"
CAMERA_MASK_BOTTOM = {"CM00009": 130}          # mirrors student_*/inference.py

METEO_CSV_DEFAULT = _REPO / "data" / "meteo_hourly.csv"


def lookup_meteo(camera: str, timestamp: str | None, csv_path=None) -> dict:
    """Nearest ARSO hourly meteorological row for a camera and timestamp.

    The smoother consumes exactly three meteorological columns, t2m (degC),
    visibility (m) and weather_code (WMO); see temporal_snow.dataset.METEO_COLS.
    Returns an empty dict when the timestamp is unknown or the camera is absent
    from the table, in which case the caller falls back to the dataset defaults.
    """
    import pandas as pd

    path = Path(csv_path) if csv_path else METEO_CSV_DEFAULT
    if not timestamp or not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["camera", "datetime", "t2m", "weather_code", "visibility"])
    sub = df[df["camera"] == camera]
    if sub.empty:
        return {}
    delta = (pd.to_datetime(sub["datetime"]) - pd.to_datetime(timestamp)).abs()
    row = sub.loc[delta.idxmin()]
    return {"t2m": float(row["t2m"]), "weather_code": float(row["weather_code"]),
            "visibility": float(row["visibility"])}



@contextlib.contextmanager
def no_pretrained_download():
    """Build backbones without fetching ImageNet weights.

    student_ground.models builds its encoder with weights="IMAGENET1K_V1", which
    downloads ~10 MB from download.pytorch.org. On a machine with no egress (a
    locked-down Space, this sandbox) that raises URLError and the stage cannot
    start. Suppressing it is only safe because the checkpoint then supplies every
    layer — which is exactly what audit_against_model proves on every load
    (absent 0 / random 0). If the audit ever fails, this suppression would leave
    random weights, so the two must be used together.
    """
    import torchvision.models as tvm
    names = ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0",
             "convnext_tiny", "resnet50")
    saved = {}
    for n in names:
        fn = getattr(tvm, n, None)
        if fn is None:
            continue
        def make(orig):
            def inner(*a, **kw):
                kw["weights"] = None
                kw.pop("pretrained", None)
                return orig(*a, **kw)
            return inner
        saved[n] = fn
        setattr(tvm, n, make(fn))
    try:
        yield
    finally:
        for n, fn in saved.items():
            setattr(tvm, n, fn)


def _valid_region(camera: str, h: int, w: int, view: str | None = None):
    """Mask for one camera, or for one physical view of it when one exists.

    A multi-view camera can carry a mask per view as <camera>__<NN>.png; the
    camera-wide <camera>.png is the fallback.
    """
    names = []
    if view not in (None, ""):
        names += [f"{camera}__{view}.png", f"{camera}_{view}.png"]
    names.append(f"{camera}.png")
    for n in names:
        p = VALID_REGION_DIR / n
        if p.exists():
            return np.array(Image.open(p).convert("L").resize((w, h), Image.NEAREST))
    return None


class SnowyPipeline:
    """Loads the deploy set once; answers many frames."""

    def __init__(self, weights_dir, *, device: str = "cpu", seq_len: int = DEFAULT_SEQ_LEN,
                 workdir=None, qa_ckpt=None):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.weights = Path(weights_dir)
        self.seq_len = int(seq_len)
        self.workdir = Path(workdir or (_REPO / "workspace" / "deploy_tmp"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.load_reports = {}
        self._windows = defaultdict(lambda: deque(maxlen=self.seq_len))
        self._feature_cols = None
        self._scaler = None
        self._load_ground(seq_len)
        self._load_snow()
        self._load_qa(qa_ckpt)

    # ── loading, each one audited ────────────────────────────────────────────
    def _load_student(self, ckpt: Path, name: str):
        import torch
        from student_ground.configs import get_config
        from student_ground.models import build_model

        if not ckpt.exists():
            raise FileNotFoundError(f"{name} checkpoint missing: {ckpt}")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = blob.get("state_dict", blob)
        variant = (blob.get("hyper_parameters") or {}).get("model_variant", "mobilenet")
        with no_pretrained_download():
            model = build_model(variant)
        report = audit_mod.audit_against_model(model, sd, name=f"{name} ({ckpt.name})",
                                               allow_partial=False)
        self.load_reports[name] = report
        cfg = get_config(variant)
        model = model.to(self.device).eval()
        return model, cfg, variant

    def _first_existing(self, names, what):
        for n in names:
            p = self.weights / n
            if p.exists():
                if n != names[0]:
                    print(f"[{what}] {names[0]} not found, falling back to {n}", flush=True)
                return p
        raise FileNotFoundError(
            f"{what} checkpoint missing: looked for {', '.join(names)} in {self.weights}")

    def _load_ground(self, seq_len):
        # DINOv2-Base, pseudo-label: 0.821 OOD IoU against 0.829 for DINOv2-Large
        # FT, at a third of the file size. The default for deployment.
        ckpt = self._first_existing(
            ["ground_pl_dinov2b.ckpt", "ground_dinov2l_ft.ckpt",
             "ground_ft_dinov2l_f4.ckpt", "ground_mobilenet_0.916.ckpt"],
            "ground_student")
        self.ground_model, self.ground_cfg, self.ground_variant = self._load_student(ckpt, "ground_student")

    def _load_snow(self):
        # ConvNeXt-Tiny, pseudo-label initialisation: the best out-of-distribution
        # snow student in the paper (0.827 OOD IoU).
        ckpt = self._first_existing(
            ["snow_pl_convnextt_k3.ckpt", "snow_convnextt_pl.ckpt", "snow_mobilenet_0.821.ckpt"],
            "snow_student")
        self.snow_model, self.snow_cfg, self.snow_variant = self._load_student(ckpt, "snow_student")

    def _load_qa(self, qa_ckpt=None):
        import torch
        self.qa_torchscript = None
        self.qa_model = None
        pt = Path(qa_ckpt) if qa_ckpt else self.weights / "qa_cnn.pt"
        ck = self.weights / "qa_cnn.ckpt"
        if pt.exists():
            try:
                self.qa_torchscript = torch.jit.load(str(pt), map_location="cpu").eval()
                self.load_reports["qa"] = {"name": f"TinyQACNN TorchScript ({pt.name})",
                                           "format": "torchscript", "loaded_ok": True}
                return
            except Exception as e:
                print(f"[qa] TorchScript load failed ({e}); trying {ck.name}", flush=True)
        if not ck.exists():
            raise FileNotFoundError(f"no QA checkpoint at {pt} or {ck}")
        from qa.train_qa_cnn import TinyQACNN
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        sd = blob.get("state_dict", blob)
        names = blob.get("class_names")
        if names and list(names) != CLASS_NAMES:
            raise RuntimeError(f"QA class order differs from the contract: {names} != {CLASS_NAMES}")
        model = TinyQACNN()
        report = audit_mod.audit_against_model(model, sd, name=f"TinyQACNN ({ck.name})",
                                               allow_partial=False)
        report["class_names"] = names or CLASS_NAMES
        self.load_reports["qa"] = report
        self.qa_model = model.to(self.device).eval()

    def load_smoother(self, ckpt=None, scaler=None):
        """Opt-in: the smoother needs the scaler that shipped beside its checkpoint."""
        import json as _json
        import torch
        from temporal_snow.dataset import FeatureScaler, build_feature_columns
        import importlib

        mod = importlib.import_module("temporal_snow.train_smoother")
        ckpt = Path(ckpt or (self.weights / "smoother_uni" / "smoother_gru_uni_best.ckpt"))
        scaler = Path(scaler or (self.weights / "smoother_uni" / "scaler.json"))
        if not ckpt.exists() or not scaler.exists():
            raise FileNotFoundError(f"smoother needs {ckpt.name} and {scaler.name}")
        module = mod.TemporalSmootherModule.load_from_checkpoint(str(ckpt), map_location="cpu")
        module = module.to(self.device).eval()
        hp = getattr(module, "hyper_parameters", {}) or {}
        if hp.get("bidirectional"):
            raise RuntimeError("this smoother is bidirectional: it is NOT causal and cannot "
                               "produce a single-camera streaming value (see the paper's note)")
        self.smoother = module
        raw = _json.loads(Path(scaler).read_text())
        self._feature_cols = raw["feature_cols"]
        self._scaler = raw          # {'feature_cols', 'mean', 'std'}-style
        self.load_reports["smoother"] = {"name": ckpt.name, "seq_len": self.seq_len,
                                         "features": len(self._feature_cols),
                                         "bidirectional": bool(hp.get("bidirectional"))}
        print(f"[smoother] {ckpt.name} seq_len={self.seq_len} "
              f"features={len(self._feature_cols)} (causal)", flush=True)

    # ── per-stage inference ──────────────────────────────────────────────────
    def _student_prob(self, model, cfg, image: Image.Image):
        """Mirror student_*/inference.py: resize 512, ImageNet norm, sigmoid."""
        import torch
        from student_ground.configs import IMAGENET_MEAN, IMAGENET_STD
        size = cfg["input_size"]
        arr = np.array(image.convert("RGB").resize((size, size), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
        return prob

    def _mask_at_frame_size(self, prob, camera: str, image: Image.Image, view: str | None = None):
        """Threshold, resize to the frame, apply bottom crop and valid region."""
        w, h = image.size
        small = (prob >= MASK_THRESHOLD).astype(np.uint8) * 255
        mask = np.array(Image.fromarray(small).resize((w, h), Image.NEAREST))
        bottom = CAMERA_MASK_BOTTOM.get(camera, 0)
        if bottom > 0:
            mask[-bottom:, :] = 0
        vr = _valid_region(camera, mask.shape[0], mask.shape[1], view)
        if vr is not None:
            mask[vr <= 127] = 0
        return mask

    def qa_probs(self, image: Image.Image) -> list[float]:
        """Per-class probabilities, in CLASS_NAMES order (clean_no_snow, snow, non_usable).

        qa_score is derived from this so the two can never drift: the binary metric
        uses 1 - P(non_usable), while a 3-class macro AUROC needs all three.
        """
        import torch
        from student_ground.configs import IMAGENET_MEAN, IMAGENET_STD
        # qa/train_qa_cnn.py:53-60 -> resize 128, /255, then ImageNet normalisation.
        # Skipping the normalisation yields uniformly low probabilities (observed:
        # 0.001-0.07 on frames with 86-98% snow), which looks like a broken gate.
        size = 128                                   # qa/train_qa_cnn.py:28
        arr = np.array(image.convert("RGB").resize((size, size), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.qa_torchscript is not None:
                out = self.qa_torchscript(x)
            else:
                out = self.qa_model(x)
            return [float(v) for v in torch.softmax(out.float(), dim=1)[0]]

    def qa_score(self, image: Image.Image) -> float:
        return 1.0 - self.qa_probs(image)[CLASS_NAMES.index("non_usable")]

    # ── the smoother's input vector ─────────────────────────────────────────
    def _feature_vector(self, image: Image.Image, camera: str, timestamp: str,
                        ground_mask, snow_mask, meteo: dict, view: str | None = None):
        """21 numbers, using the repo's own extractors wherever they exist."""
        from analysis.build_features import imgfeat_extract_features, ts_compute_snow_features

        cam_dir = self.workdir / camera
        cam_dir.mkdir(parents=True, exist_ok=True)
        g_path = cam_dir / "tmp_ground.png"
        s_path = cam_dir / "tmp_snow.png"
        Image.fromarray(ground_mask).save(g_path)
        Image.fromarray(snow_mask).save(s_path)
        vr = VALID_REGION_DIR / f"{camera}.png"

        morph = ts_compute_snow_features((s_path, g_path, vr if vr.exists() else None))
        # image descriptors: the extractor reads from disk
        img_path = cam_dir / "tmp_frame.jpg"
        image.convert("RGB").save(img_path, quality=95)
        img_feats = imgfeat_extract_features(img_path) or {}

        f = {
            "student_snow_pct": morph["snow_pct"],
            "snow_n_components": morph["n_components"],
            "snow_largest_ratio": morph["largest_ratio"],
            "snow_centroid_y": morph["snow_centroid_y"],
            "ground_coverage": morph["ground_coverage"],
        }
        for k in ("brightness", "contrast", "edge_density", "entropy", "white_frac",
                  "low_sat_frac", "blue_ratio", "bottom_brightness", "bottom_white_frac"):
            f[k] = float(img_feats.get(k, np.nan))

        ts = np.datetime64(timestamp)
        hour = float(str(timestamp)[11:13]) + float(str(timestamp)[14:16]) / 60.0
        doy = int((ts - np.datetime64(f"{str(timestamp)[0:4]}-01-01")).astype(int)) + 1
        f.update({
            "hour": hour,
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "doy_sin": float(np.sin(2 * np.pi * doy / 365)),
            "doy_cos": float(np.cos(2 * np.pi * doy / 365)),
            "t2m": meteo.get("t2m", np.nan),
            "visibility": meteo.get("visibility", np.nan),
            "weather_code": meteo.get("weather_code", np.nan),
        })
        # ── the two steps mirrored from temporal_snow/dataset.py:74-93 ──
        # defaults for missing meteo, then log1p on the two heavy-tailed columns,
        # THEN the scaler (which was fitted on the log1p'd values).
        f.setdefault("t2m", 2.8)
        f["t2m"] = 2.8 if not np.isfinite(f.get("t2m", np.nan)) else f["t2m"]
        f["visibility"] = 10000.0 if not np.isfinite(f.get("visibility", np.nan)) else f["visibility"]
        f["weather_code"] = 0.0 if not np.isfinite(f.get("weather_code", np.nan)) else f["weather_code"]
        f["snow_n_components"] = np.log1p(f["snow_n_components"])
        f["visibility"] = np.log1p(f["visibility"])

        cols = self._feature_cols
        x = np.array([f.get(c, np.nan) for c in cols], dtype=np.float32)
        mean = np.asarray(self._scaler.get("mean"), dtype=np.float32)
        std = np.asarray(self._scaler.get("std"), dtype=np.float32)
        std = np.where(std == 0, 1.0, std)
        x = (x - mean) / std
        return np.nan_to_num(x, nan=0.0)

    def _smoother_step(self, camera: str, vec) -> float | None:
        """Causal: only the frames seen so far. Null until the window is full."""
        import torch
        buf = self._windows[camera]
        buf.append(vec)
        if len(buf) < self.seq_len:
            return None
        x = torch.tensor(np.stack(list(buf))[None, ...], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.smoother(x) if hasattr(self, "smoother") else None
        return float(out[0, -1].cpu().item()) if out is not None else None

    # ── public entry point ──────────────────────────────────────────────────
    def predict_image(self, image, camera_id: str, timestamp: str | None = None,
                      meteo: dict | None = None, want_smoothed: bool = True,
                      view: str | None = None) -> tuple[dict, dict]:
        src_path = image if isinstance(image, (str, Path)) else None
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGB")
        if timestamp is None and src_path is not None:
            # the filename carries the stamp: CM00014_202212111000_27 -> 2022-12-11T10:00
            timestamp = contract.Frame(image=str(src_path)).resolved_timestamp()
        if view is None and src_path is not None:
            # CM00067_202301151200_03 -> view "03"; cameras with one view use "00"
            view = contract.Frame(image=str(src_path)).resolved_view()
        meta = {"camera_id": camera_id, "size": list(image.size), "stages": {},
                "timestamp": timestamp, "view": view}

        ground_prob = self._student_prob(self.ground_model, self.ground_cfg, image)
        snow_prob = self._student_prob(self.snow_model, self.snow_cfg, image)
        ground_mask = self._mask_at_frame_size(ground_prob, camera_id, image, view)
        snow_mask = self._mask_at_frame_size(snow_prob, camera_id, image, view)

        g = ground_mask > 127
        s = (snow_mask > 127) & g                       # snow inside ground
        gpx = int(g.sum())
        psg_raw = (float(s.sum()) / gpx * 100.0) if gpx > 0 else 0.0
        meta["stages"] = {"ground_px": gpx, "snow_px_in_ground": int(s.sum()),
                          "valid_region": "applied" if _valid_region(camera_id, 1, 1) is not None
                                          or (VALID_REGION_DIR / f"{camera_id}.png").exists()
                                          else "none for this camera"}

        qa = self.qa_score(image)
        meta["stages"]["qa"] = self.load_reports.get("qa", {}).get("name", "?")

        psg_smoothed = None
        meta["window_seen"] = len(self._windows[camera_id])
        meta["seq_len"] = self.seq_len
        if not want_smoothed:
            meta["psg_smoothed_reason"] = "caller asked for psg_raw only"
        elif self._scaler is None:
            meta["psg_smoothed_reason"] = "no smoother loaded (call load_smoother())"
        elif not timestamp:
            meta["psg_smoothed_reason"] = ("no timestamp: the smoother's features need one "
                                           "(give --timestamp or a filename that carries it)")
        else:
            vec = self._feature_vector(image, camera_id, timestamp, ground_mask, snow_mask,
                                       meteo or {}, view)
            frac = self._smoother_step(camera_id, vec)
            # The GRU's native output is a fraction in [0,1] -- train_smoother.py takes
            # the target as snow_pct "normalized to [0, 1]". Every consumer, and the
            # paper's own figures (visualize_smoother.py:186 `smoothed_snow * 100`,
            # ylabel "Snow %"), reports percent. Convert here, once, so a 0.91 can never
            # sit next to a psg_raw of 91 and read as "the smoother collapsed".
            psg_smoothed = None if frac is None else round(float(frac) * 100.0, 2)
            meta["window_seen"] = len(self._windows[camera_id])
            if psg_smoothed is None:
                meta["psg_smoothed_reason"] = (
                    f"window not full: {meta['window_seen']}/{self.seq_len} frames of "
                    f"{camera_id} seen (the smoother is causal, it cannot use the future)")

        response = build_response(qa_score=qa, psg_raw=psg_raw, psg_smoothed=psg_smoothed,
                                  ground_mask=ground_mask, snow_mask=snow_mask, meta=meta)
        return response, {"ground_mask": ground_mask, "snow_mask": snow_mask, "psg_raw": psg_raw}


# ── fusion helper: refuses missing channels instead of zero-filling ──────────
FUSION_FOLDERS = {
    "ground": OrderedDict([("dinov2_pca", "dinov2_pca"), ("segformer", "segformer"),
                           ("deeplabv3", "deeplabv3"), ("sam3_text", "sam3/text"),
                           ("depth", "depth/N0.05_F0.95")]),
    "snow": OrderedDict([("clipseg", "clipseg"), ("florence2", "florence2"),
                         ("groundedsam2", "groundedsam2"), ("sam3", "sam3"),
                         ("yoloe", "yoloe"), ("seggpt", "seggpt")]),
}


def fusion_mask(ckpt, camera: str, stem: str, seg_root, size: int = 512):
    """FusionCNN output for one frame — but only with EVERY teacher present.

    fusion_cnn.py:495-500 zero-fills absent teacher masks and still writes a
    plausible mask; that is exactly how a 0.4479 IoU row happened. Here a missing
    channel is a hard error, and the channel order comes from the checkpoint's own
    `teachers` list, not from a hardcoded constant.
    """
    import torch
    # Loaded by file path: models/fusion_cnn/ has no __init__.py and nothing else in
    # the repo imports it, so a plain `import fusion_cnn` would resolve to the
    # *directory* and fail on the class name.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fusion_cnn_module", _REPO / "models" / "fusion_cnn" / "fusion_cnn.py")
    fusion_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion_mod)
    FusionCNN = fusion_mod.FusionCNN

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    teachers = list(blob["teachers"])
    task = blob["task"]
    folder_of = FUSION_FOLDERS[task]
    missing = []
    planes = []
    for t in teachers:
        p = Path(seg_root) / task / folder_of[t] / camera / f"{stem}_{task}.png"
        if not p.exists():
            missing.append(str(p))
            planes.append(None)
        else:
            planes.append(np.array(Image.open(p).convert("L").resize((size, size), Image.NEAREST),
                                   dtype=np.float32) / 255.0)
    if missing:
        raise FileNotFoundError(
            f"refusing to fuse {camera}/{stem}: {len(missing)} of {len(teachers)} teacher "
            f"channels are absent (fusion_cnn.py would zero-fill them silently): {missing[:3]}")

    model = FusionCNN(n_teachers=len(teachers))
    audit_mod.audit_against_model(model, blob["state_dict"],
                                 name=f"fusion_{task} ({Path(ckpt).name})", allow_partial=False)
    x = torch.from_numpy(np.stack(planes)[None, ...].astype(np.float32))
    with torch.no_grad():
        return model.eval()(x)[0, 0].numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None)
    ap.add_argument("--frames", nargs="+", default=None,
                    help="frames of ONE camera, oldest first. Supplying at least "
                         "seq_len of them is what makes psg_smoothed a real number.")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--timestamp", default=None)
    ap.add_argument("--weights", default=str(_REPO.parent / "models" / "best"),
                    help="directory holding the deploy set (default: <thesis>/models/best)")
    ap.add_argument("--t2m", type=float, default=None, help="air temperature, degC")
    ap.add_argument("--visibility", type=float, default=None, help="visibility, m")
    ap.add_argument("--weather-code", type=float, default=None, help="WMO weather code")
    ap.add_argument("--meteo-csv", default=None,
                    help="hourly meteorological table; fills the three meteo values "
                         "from --camera and --timestamp when the flags are omitted")
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--smoother", action="store_true", help="also load the GRU smoother")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    pipe = SnowyPipeline(args.weights, device=args.device, seq_len=args.seq_len)
    if args.smoother:
        pipe.load_smoother()
    images = args.frames or ([args.image] if args.image else [])
    if not images:
        print("need --image or --frames")
        return 2
    meteo = {k: v for k, v in (("t2m", args.t2m),
                               ("visibility", args.visibility),
                               ("weather_code", args.weather_code)) if v is not None}
    if not meteo:
        meteo = lookup_meteo(args.camera, args.timestamp, args.meteo_csv)
    resp = None
    for i, im in enumerate(images):
        ts = args.timestamp if len(images) == 1 else None
        resp, _ = pipe.predict_image(im, args.camera, ts, meteo)
    if resp.get("psg_smoothed") is None and len(images) < args.seq_len:
        resp.setdefault("meta", {})["psg_smoothed_note"] = (
            f"null: {len(images)} frame(s) supplied, the smoother window is {args.seq_len}")
    print(json.dumps(resp, indent=2, default=str)[:1500])
    print("\nload reports:")
    for k, v in pipe.load_reports.items():
        if isinstance(v, dict) and "loaded" in v:
            print(f"  {k}: {v['loaded']}/{v['checkpoint_keys']} keys, "
                  f"absent {len(v.get('absent_from_model', []))}, "
                  f"random {len(v.get('left_at_random_init', []))}")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
