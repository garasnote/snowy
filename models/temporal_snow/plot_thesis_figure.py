#!/usr/bin/env python3
"""
Thesis figure: raw vs smoothed snow_pct for a representative snowy week.
Picks best camera/week automatically (highest snow variance), or use --camera/--week.

Usage:
    python plot_thesis_figure.py --checkpoint <path>
    python plot_thesis_figure.py --checkpoint <path> --camera CM00108 --week 2024-01-15
    python plot_thesis_figure.py --checkpoint <path> --all-models
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.rcParams["font.family"] = "DejaVu Sans"

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "models"))

from temporal_snow.dataset import build_feature_columns, SNOW_PCT_COLS, FeatureScaler
from temporal_snow.train_smoother import TemporalSmootherModule


def predict_camera(module, cam_df, scaler=None):
    import math
    snow_cols = [c for c in SNOW_PCT_COLS if c in cam_df.columns]
    tmp = cam_df.copy()
    tmp["hour_sin"] = np.sin(2*math.pi*tmp["datetime"].dt.hour/24)
    tmp["hour_cos"] = np.cos(2*math.pi*tmp["datetime"].dt.hour/24)
    doy = tmp["datetime"].dt.dayofyear
    tmp["doy_sin"] = np.sin(2*math.pi*doy/365)
    tmp["doy_cos"] = np.cos(2*math.pi*doy/365)
    FILL = {"t2m":0,"visibility":1000,"weather_code":0,"brightness":128,"contrast":50,
            "edge_density":10,"entropy":4,"white_frac":0,"low_sat_frac":0,"blue_ratio":0,
            "bottom_white_frac":0,"bottom_brightness":128,"snow_n_components":0,
            "snow_largest_ratio":0,"snow_centroid_y":0.5,"ground_coverage":0}
    for col in snow_cols: tmp[col] = tmp[col].fillna(0)
    for col, val in FILL.items():
        if col in tmp.columns: tmp[col] = tmp[col].fillna(val)
    # Log-transform before scaling (same as training pipeline)
    if "visibility" in tmp.columns: tmp["visibility"] = np.log1p(tmp["visibility"])
    if "snow_n_components" in tmp.columns: tmp["snow_n_components"] = np.log1p(tmp["snow_n_components"])

    input_size = module.hparams.get("input_size", None)
    feat_cols_full = build_feature_columns(snow_cols, use_image_features=True)
    feat_cols_noimg = build_feature_columns(snow_cols, use_image_features=False)
    if input_size:
        if len(feat_cols_noimg) == input_size:
            feat_cols = feat_cols_noimg
        elif len(feat_cols_full) == input_size:
            feat_cols = feat_cols_full
        else:
            feat_cols = feat_cols_full[:input_size]
    else:
        feat_cols = feat_cols_full
    for col in feat_cols:
        if col not in tmp.columns: tmp[col] = 0.0

    if scaler is not None:
        # Use saved StandardScaler (correct)
        tmp = scaler.transform(tmp)
    else:
        # Fallback: manual approximate scaling (matches eval_all.py)
        if "t2m" in tmp.columns: tmp["t2m"] = tmp["t2m"] / 30.0
        if "visibility" in tmp.columns: tmp["visibility"] = tmp["visibility"] / np.log1p(65000)
        if "weather_code" in tmp.columns: tmp["weather_code"] = tmp["weather_code"] / 100.0
        if "brightness" in tmp.columns:
            tmp["brightness"] /= 255.0; tmp["contrast"] /= 128.0
            tmp["edge_density"] /= 50.0; tmp["entropy"] /= 6.0; tmp["bottom_brightness"] /= 255.0
        if "snow_n_components" in tmp.columns: tmp["snow_n_components"] /= np.log1p(100)

    print(f"  [debug] input_size={input_size} feat_cols={len(feat_cols)} scaler={'yes' if scaler else 'approx'} model={type(module.model).__name__}")

    snow_raw = cam_df[snow_cols[0]].values.astype(np.float32) if snow_cols else np.zeros(len(cam_df))
    feats = np.nan_to_num(tmp[feat_cols].values.astype(np.float32), nan=0.0)
    x = torch.tensor(feats).unsqueeze(0)
    with torch.no_grad():
        pred = module(x).squeeze(0).cpu().numpy() * 100

    # student_snow_pct is 0-100 in CSV; only multiply if stored as 0-1
    raw_out = snow_raw if snow_raw.max() > 1.01 else snow_raw * 100
    return raw_out, pred


def find_best_week(df, camera=None, week_start=None):
    """Find camera + week with highest snow variance for visual interest."""
    if camera and week_start:
        return camera, pd.Timestamp(week_start)

    best_cam, best_week, best_var = None, None, -1
    for cam, cam_df in df.groupby("camera"):
        if camera and cam != camera:
            continue
        cam_df = cam_df.sort_values("datetime").copy()
        cam_df["week"] = cam_df["datetime"].dt.to_period("W")
        for week, week_df in cam_df.groupby("week"):
            if len(week_df) < 10:
                continue
            snow_col = next((c for c in SNOW_PCT_COLS if c in week_df.columns), None)
            if snow_col is None:
                continue
            var = week_df[snow_col].std()
            # Prefer weeks with both snow and no-snow (transition)
            has_snow = (week_df[snow_col] > 0.05).any()
            has_clear = (week_df[snow_col] < 0.05).any()
            if has_snow and has_clear and var > best_var:
                best_var = var
                best_cam = cam
                best_week = week.start_time
    return best_cam, best_week


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to smoother checkpoint (.ckpt)")
    parser.add_argument("--extra-checkpoints", nargs="+", default=[], help="Additional checkpoints to overlay")
    parser.add_argument("--labels", nargs="+", default=[], help="Labels for extra checkpoints")
    parser.add_argument("--csv", default=str(_ROOT / "analysis/snow_timeseries.csv"))
    parser.add_argument("--camera", default=None)
    parser.add_argument("--week", default=None, help="Week start date YYYY-MM-DD")
    parser.add_argument("--window-days", type=int, default=14, help="Days to show (default 14)")
    parser.add_argument("--out", default="thesis_smoother_figure.pdf")
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_csv(args.csv, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    # Do NOT filter by qa_usable — we want to show raw spikes that smoothing suppresses
    df = df.sort_values(["camera", "datetime"]).reset_index(drop=True)

    def _load_scaler(ckpt_path):
        """Load scaler.json from checkpoint directory if available."""
        scaler_path = Path(ckpt_path).parent / "scaler.json"
        if scaler_path.exists():
            return FeatureScaler.load(scaler_path)
        return None

    # Load primary model
    print(f"Loading checkpoint: {args.checkpoint}")
    module = TemporalSmootherModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    module.eval()
    primary_scaler = _load_scaler(args.checkpoint)

    # Find best week
    cam, week_start = find_best_week(df, args.camera, args.week)
    if cam is None or week_start is None:
        print(f"ERROR: No suitable window found for camera={args.camera}"); return
    week_end = week_start + pd.Timedelta(days=args.window_days)
    print(f"Camera: {cam}  |  Window: {week_start.date()} → {week_end.date()}")

    cam_df = df[df["camera"] == cam].sort_values("datetime").reset_index(drop=True)
    snow_col = next((c for c in SNOW_PCT_COLS if c in cam_df.columns), None)

    # Predict on full camera series (uni models need warmup context)
    snow_raw, snow_smooth = predict_camera(module, cam_df, scaler=primary_scaler)

    # Display window
    mask = (cam_df["datetime"] >= week_start) & (cam_df["datetime"] < week_end)
    dt_window = cam_df.loc[mask, "datetime"]
    raw_window = snow_raw[mask.values]
    smooth_window = snow_smooth[mask.values]

    # Load extra checkpoints — also predict on full series, clip to window
    extra_smooths = []
    for ec in args.extra_checkpoints:
        m = TemporalSmootherModule.load_from_checkpoint(ec, map_location="cpu")
        m.eval()
        scaler_ec = _load_scaler(ec)
        _, s = predict_camera(m, cam_df, scaler=scaler_ec)
        extra_smooths.append(s[mask.values])

    # Kalman filter (Q=25, R=100 — best from eval)
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

    snow_col_raw = next((c for c in SNOW_PCT_COLS if c in cam_df.columns), None)
    raw_vals = cam_df[snow_col_raw].values if snow_col_raw else np.zeros(len(cam_df))
    # student_snow_pct is 0-100 scale in CSV
    if raw_vals.max() <= 1.01:
        raw_vals = raw_vals * 100
    raw_pct_full = raw_vals
    kalman_full = _kalman(raw_pct_full)
    kalman_window = kalman_full[mask.values]

    # Meteo snow presence
    meteo_col = "snow_present" if "snow_present" in cam_df.columns else None
    meteo_window = cam_df.loc[mask, meteo_col] if meteo_col else None

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))

    # Meteo background shading
    if meteo_window is not None:
        snow_present = meteo_window.values.astype(bool)
        for i in range(len(dt_window) - 1):
            if snow_present[i]:
                ax.axvspan(dt_window.iloc[i], dt_window.iloc[i+1],
                           alpha=0.12, color="#4a90d9", linewidth=0)

    # Raw predictions
    ax.plot(dt_window, raw_window, color="#aaaaaa", linewidth=0.8,
            alpha=0.7, label="Raw student output", zorder=2)

    # Kalman filter
    ax.plot(dt_window, kalman_window, color="#2980b9", linewidth=1.5,
            linestyle="--", alpha=0.85, label="Kalman Filter", zorder=3)

    # Extra model smooths
    colors = ["#e67e22", "#27ae60", "#8e44ad"]
    labels = args.labels if args.labels else [f"Model {i+1}" for i in range(len(extra_smooths))]
    for i, (es, lbl) in enumerate(zip(extra_smooths, labels)):
        ax.plot(dt_window, es, linewidth=1.5, color=colors[i % len(colors)],
                alpha=0.85, label=lbl, zorder=3)

    # Primary smooth (bi-LSTM)
    ax.plot(dt_window, smooth_window, color="#c0392b", linewidth=2.0,
            label="LSTM (bi)", zorder=4)

    ax.axhline(50, color="#333333", linewidth=1.0, linestyle=":", alpha=0.7, label="Snow threshold (50%)", zorder=1)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Snow Coverage (%)", fontsize=11)
    ax.set_title(f"Temporal Smoothing — Camera {cam}", fontsize=12, fontweight="bold")
    ax.set_ylim(-2, 102)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Collect all handles for fig legend
    handles, labels_leg = ax.get_legend_handles_labels()
    if meteo_window is not None:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor="#4a90d9", alpha=0.3, label="Meteo: snow present"))
        labels_leg.append("Meteo: snow present")
    ax.legend_ = None  # remove axes legend

    fig.legend(handles, labels_leg, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, fontsize=8.5, framealpha=0.9, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = Path(args.out)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
