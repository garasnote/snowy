#!/usr/bin/env python3
"""
Generate a video showing camera frames + temporal smoothing plot below.

Layout:
  Top:    camera image (original)
  Bottom: thesis-style temporal smoothing plot, vertical line at current frame

Usage:
    python make_smoother_video.py --camera CM00070 --start 2024-12-20 --end 2025-01-03
    python make_smoother_video.py --camera CM00112 --start 2025-11-17 --end 2025-11-25
    python make_smoother_video.py --all-models --camera CM00070 --start 2024-12-20 --end 2025-01-03
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "models"))

from temporal_snow.dataset import build_feature_columns, SNOW_PCT_COLS, FeatureScaler
from temporal_snow.train_smoother import TemporalSmootherModule

# ── Paths ──────────────────────────────────────────────────────────────────────
BEST_DIR    = _ROOT / "models" / "best"
CSV_PATH    = _ROOT / "analysis" / "snow_timeseries.csv"
DATA_DIR    = _ROOT / "data"

CONVNEXT_CKPT = (_ROOT / "models" / "student_snow" / "checkpoints" /
                 "snow_gt_convnext_20260507_230426" /
                 "best-epoch=95-iouval_iou=0.900.ckpt")

CHECKPOINTS = {
    "GRU (uni)":  (BEST_DIR / "smoother_gru_uni_best.ckpt",
                   BEST_DIR / "smoother_gru_uni_best_scaler.json"),
    "GRU (bi)":   (BEST_DIR / "smoother_gru_bi_best.ckpt",
                   BEST_DIR / "smoother_gru_bi_best_scaler.json"),
}

# Paper-style colors matching the thesis figures
COLORS = {
    "Raw student output": "#aaaaaa",
    "Kalman Filter":      "#2980b9",
    "GRU (uni)":          "#8e44ad",
    "GRU (bi)":           "#27ae60",
    "LSTM (uni)":         "#c0392b",
    "LSTM (bi)":          "#e67e22",
    "Snow threshold":     "#333333",
}
METEO_COLOR  = "#4a90d9"
VLINE_COLOR  = "#e74c3c"

# Video params
FPS          = 10
VIDEO_W      = 1280
VIDEO_H      = 960   # top 480 image, bottom 480 plot
PLOT_DPI     = 120


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * math.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * math.pi * df["datetime"].dt.hour / 24)
    doy = df["datetime"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * math.pi * doy / 365)
    return df


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    FILL = {
        "t2m": 0, "visibility": 1000, "weather_code": 0,
        "brightness": 128, "contrast": 50, "edge_density": 10, "entropy": 4,
        "white_frac": 0, "low_sat_frac": 0, "blue_ratio": 0,
        "bottom_white_frac": 0, "bottom_brightness": 128,
        "snow_n_components": 0, "snow_largest_ratio": 0,
        "snow_centroid_y": 0.5, "ground_coverage": 0,
    }
    df = df.copy()
    for snow_col in SNOW_PCT_COLS:
        if snow_col in df.columns:
            df[snow_col] = df[snow_col].fillna(0)
    for col, val in FILL.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    if "visibility" in df.columns:
        df["visibility"] = np.log1p(df["visibility"])
    if "snow_n_components" in df.columns:
        df["snow_n_components"] = np.log1p(df["snow_n_components"])
    return df


def predict_camera(module, cam_df: pd.DataFrame, scaler=None) -> np.ndarray:
    """Run smoother on full camera series, return per-row predictions (0-100)."""
    snow_cols = [c for c in SNOW_PCT_COLS if c in cam_df.columns]
    tmp = _add_time_features(cam_df)
    tmp = _fill_defaults(tmp)

    input_size = module.hparams.get("input_size", None)
    feat_cols_full  = build_feature_columns(snow_cols, use_image_features=True)
    feat_cols_noimg = build_feature_columns(snow_cols, use_image_features=False)
    if input_size:
        feat_cols = feat_cols_noimg if len(feat_cols_noimg) == input_size else feat_cols_full
        if len(feat_cols) != input_size:
            feat_cols = feat_cols_full[:input_size]
    else:
        feat_cols = feat_cols_full

    for col in feat_cols:
        if col not in tmp.columns:
            tmp[col] = 0.0

    if scaler is not None:
        tmp = scaler.transform(tmp)
    else:
        if "t2m" in tmp.columns:         tmp["t2m"]          = tmp["t2m"] / 30.0
        if "visibility" in tmp.columns:  tmp["visibility"]   = tmp["visibility"] / np.log1p(65000)
        if "weather_code" in tmp.columns: tmp["weather_code"] = tmp["weather_code"] / 100.0

    feats = np.nan_to_num(tmp[feat_cols].values.astype(np.float32), nan=0.0)
    x = torch.tensor(feats).unsqueeze(0)
    with torch.no_grad():
        pred = module(x).squeeze(0).cpu().numpy() * 100
    return pred.flatten()


def _kalman(series, Q=25, R=100):
    x, P = float(series[0]), 500.0
    out = []
    for z in series:
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x = x + K * (float(z) - x)
        P = (1 - K) * P_pred
        out.append(np.clip(x, 0, 100))
    return np.array(out)


def load_convnext(ckpt_path: Path, device):
    """Load ConvNeXt snow segmentation model."""
    from student_ground.train import StudentSegmentationModule
    print(f"Loading ConvNeXt: {ckpt_path.name}")
    module = StudentSegmentationModule.load_from_checkpoint(
        str(ckpt_path), model_variant="convnext"
    )
    model = module.model.eval().to(device)
    return model


@torch.no_grad()
def predict_snow_mask(model, img_path: Path, device, input_size=512, threshold=0.5):
    """Run ConvNeXt on one image, return binary mask (H_orig, W_orig) uint8 0/255."""
    from student_ground.configs import IMAGENET_MEAN, IMAGENET_STD
    from PIL import Image as PILImage

    pil = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = pil.size
    resized = pil.resize((input_size, input_size), PILImage.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    logits = model(tensor)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    binary = (prob >= threshold).astype(np.uint8) * 255
    # Resize back to original image dimensions
    mask_pil = PILImage.fromarray(binary).resize((orig_w, orig_h), PILImage.NEAREST)
    return np.array(mask_pil)


def apply_snow_overlay(img_bgr: np.ndarray, mask: np.ndarray,
                       color_bgr=(255, 255, 0), alpha=0.45) -> np.ndarray:
    """Blend snow mask as semi-transparent overlay on BGR image."""
    out = img_bgr.copy()
    snow_px = mask > 0
    colored = np.zeros_like(img_bgr)
    colored[:] = color_bgr
    out[snow_px] = cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)[snow_px]
    return out


def load_image(path: Path, w: int, h: int, mask: np.ndarray = None) -> np.ndarray:
    """Load and resize image to (h, w, 3) BGR. Optionally overlay snow mask."""
    img = cv2.imread(str(path))
    if img is None:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "Image not found", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        return img
    if mask is not None:
        img = apply_snow_overlay(img, mask)
    return cv2.resize(img, (w, h))


def render_plot_frame(
    datetimes, raw, kalman, smooths, meteo,
    current_dt, w: int, h: int,
    camera: str
) -> np.ndarray:
    """Render the thesis-style plot as an RGB numpy array (h, w, 3)."""
    fig, ax = plt.subplots(figsize=(w / PLOT_DPI, h / PLOT_DPI), dpi=PLOT_DPI)
    fig.patch.set_facecolor("white")

    # Meteo background
    if meteo is not None:
        snow_present = np.array(meteo).astype(bool)
        for i in range(len(datetimes) - 1):
            if snow_present[i]:
                ax.axvspan(datetimes[i], datetimes[i + 1],
                           alpha=0.12, color=METEO_COLOR, linewidth=0)

    # Raw predictions
    ax.plot(datetimes, raw, color=COLORS["Raw student output"],
            linewidth=0.9, alpha=0.7, label="Raw student output", zorder=2)

    # Kalman
    ax.plot(datetimes, kalman, color=COLORS["Kalman Filter"],
            linewidth=1.5, linestyle="--", alpha=0.85, label="Kalman Filter", zorder=3)

    # Smoothers
    smooth_order = ["GRU (bi)", "GRU (uni)", "LSTM (uni)", "LSTM (bi)"]
    for name in smooth_order:
        if name in smooths:
            ax.plot(datetimes, smooths[name],
                    color=COLORS.get(name, "#555555"),
                    linewidth=1.6, alpha=0.88, label=name, zorder=4)

    # Snow threshold
    ax.axhline(50, color=COLORS["Snow threshold"], linewidth=1.0,
               linestyle=":", alpha=0.7, label="Snow threshold (50%)", zorder=1)

    # Current-frame vertical line
    ax.axvline(current_dt, color=VLINE_COLOR, linewidth=2.0, alpha=0.9, zorder=6)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Snow Coverage (%)", fontsize=10)
    ax.set_title(f"Temporal Smoothing — Camera {camera}", fontsize=11, fontweight="bold")
    ax.set_ylim(-2, 102)
    ax.set_xlim(datetimes[0], datetimes[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if meteo is not None:
        handles.append(mpatches.Patch(facecolor=METEO_COLOR, alpha=0.3, label="Meteo: snow present"))
        labels.append("Meteo: snow present")
    ax.legend_ = None
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, fontsize=7.5, framealpha=0.9, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    # Render to numpy
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # drop alpha
    plt.close(fig)

    # Resize to target
    return cv2.resize(buf, (w, h))  # RGB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",   default="CM00070")
    parser.add_argument("--start",    default="2024-12-20", help="Window start YYYY-MM-DD")
    parser.add_argument("--end",      default="2025-01-03", help="Window end YYYY-MM-DD (exclusive)")
    parser.add_argument("--csv",      default=str(CSV_PATH))
    parser.add_argument("--fps",      type=int, default=FPS)
    parser.add_argument("--out",      default="smoother_video.mp4")
    parser.add_argument("--all-models", action="store_true",
                        help="Load all available checkpoint variants")
    parser.add_argument("--extra-checkpoint", nargs="+", default=[],
                        help="Extra checkpoint paths")
    parser.add_argument("--extra-label", nargs="+", default=[],
                        help="Labels for extra checkpoints")
    parser.add_argument("--snow-mask", action="store_true",
                        help="Overlay ConvNeXt snow mask on each frame")
    parser.add_argument("--snow-mask-ckpt", default=str(CONVNEXT_CKPT),
                        help="ConvNeXt checkpoint path (default: best IoU=0.900)")
    args = parser.parse_args()

    print("Loading CSV...")
    df = pd.read_csv(args.csv, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["camera", "datetime"]).reset_index(drop=True)

    cam_df = df[df["camera"] == args.camera].sort_values("datetime").reset_index(drop=True)
    if len(cam_df) == 0:
        print(f"ERROR: camera {args.camera} not found."); return

    # Window for display
    t_start = pd.Timestamp(args.start)
    t_end   = pd.Timestamp(args.end)
    mask    = (cam_df["datetime"] >= t_start) & (cam_df["datetime"] < t_end)
    win_df  = cam_df[mask].reset_index(drop=True)
    if len(win_df) == 0:
        print("ERROR: no rows in window"); return
    print(f"Window: {t_start.date()} → {t_end.date()}, {len(win_df)} frames")

    # Raw snow %
    snow_col = next((c for c in SNOW_PCT_COLS if c in cam_df.columns), None)
    raw_full = cam_df[snow_col].values.astype(np.float32)
    if raw_full.max() <= 1.01:
        raw_full *= 100

    # Kalman on full series, slice window
    kal_full   = _kalman(raw_full)
    raw_window = raw_full[mask.values]
    kal_window = kal_full[mask.values]
    dt_window  = win_df["datetime"].tolist()

    # Meteo
    meteo_window = None
    if "snow_present" in win_df.columns:
        meteo_window = win_df["snow_present"].fillna(False).astype(bool).tolist()

    # Load smoothers
    smooths = {}

    ckpts_to_load = dict(CHECKPOINTS) if args.all_models else {}
    # Always load at least GRU uni
    if not ckpts_to_load:
        ckpts_to_load["GRU (uni)"] = CHECKPOINTS["GRU (uni)"]

    for label, (ckpt_path, scaler_path) in ckpts_to_load.items():
        if not ckpt_path.exists():
            print(f"  SKIP {label}: {ckpt_path} not found"); continue
        print(f"  Loading {label}: {ckpt_path.name}")
        m = TemporalSmootherModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
        m.eval()
        scaler = FeatureScaler.load(scaler_path) if scaler_path.exists() else None
        pred_full = predict_camera(m, cam_df, scaler=scaler)
        smooths[label] = pred_full[mask.values]
        del m

    # Extra checkpoints
    for i, ec in enumerate(args.extra_checkpoint):
        lbl = args.extra_label[i] if i < len(args.extra_label) else f"Model {i+1}"
        print(f"  Loading extra {lbl}: {ec}")
        m = TemporalSmootherModule.load_from_checkpoint(ec, map_location="cpu")
        m.eval()
        sp = Path(ec).parent / "scaler.json"
        scaler = FeatureScaler.load(sp) if sp.exists() else None
        pred_full = predict_camera(m, cam_df, scaler=scaler)
        smooths[lbl] = pred_full[mask.values]
        del m

    print(f"Smoothers loaded: {list(smooths.keys())}")

    # Load ConvNeXt if requested
    seg_model = None
    seg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.snow_mask:
        seg_model = load_convnext(Path(args.snow_mask_ckpt), seg_device)
        print(f"  ConvNeXt on {seg_device}")

    # Image dir
    img_dir = DATA_DIR / args.camera

    # Video writer
    out_path = Path(args.out)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    img_h = VIDEO_H // 2
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (VIDEO_W, VIDEO_H))

    print(f"Rendering {len(win_df)} frames...")
    for fi, (_, row) in enumerate(win_df.iterrows()):
        stem     = row["stem"]
        img_path = img_dir / f"{stem}.jpg"
        current_dt = row["datetime"]

        # ConvNeXt mask (on original image before resize)
        snow_mask = None
        if seg_model is not None and img_path.exists():
            snow_mask = predict_snow_mask(seg_model, img_path, seg_device)

        # Top panel: camera image
        frame_img = load_image(img_path, VIDEO_W, img_h, mask=snow_mask)  # BGR

        # Overlay: datetime + snow% in top-left
        snow_raw_val = float(raw_window[fi])
        label_txt = f"{current_dt.strftime('%Y-%m-%d %H:%M')}  |  Raw: {snow_raw_val:.1f}%"
        if smooths:
            first_name = next(iter(smooths))
            smooth_val = float(smooths[first_name][fi])
            label_txt += f"  |  {first_name}: {smooth_val:.1f}%"
        if seg_model is not None:
            label_txt += "  |  ConvNeXt mask (cyan)"
        # Semi-transparent bar
        overlay = frame_img.copy()
        cv2.rectangle(overlay, (0, 0), (VIDEO_W, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame_img, 0.45, 0, frame_img)
        cv2.putText(frame_img, label_txt, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom panel: plot
        plot_rgb = render_plot_frame(
            dt_window, raw_window, kal_window, smooths, meteo_window,
            current_dt, VIDEO_W, img_h, args.camera
        )
        plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)

        # Stack vertically
        combined = np.vstack([frame_img, plot_bgr])
        writer.write(combined)

        if (fi + 1) % 10 == 0:
            print(f"  {fi + 1}/{len(win_df)}", end="\r")

    writer.release()
    print(f"\nSaved: {out_path}  ({len(win_df)} frames @ {args.fps} fps)")
    print(f"Duration: {len(win_df) / args.fps:.1f}s")


if __name__ == "__main__":
    main()
