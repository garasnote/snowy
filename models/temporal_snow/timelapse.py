#!/usr/bin/env python3
"""
Full-camera timelapse: webcam images + smoothed snow % curve below.

Subsamples to ~1 frame per hour for manageable file size.
Image on top, snow_pct curve on bottom with moving marker.

Usage:
    python timelapse.py --checkpoint path/to/best-smoother.ckpt
    python timelapse.py --checkpoint path/to/best.ckpt --cameras CF00001 --fps 12
    python timelapse.py --checkpoint path/to/best.ckpt --hours-per-frame 2
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
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from temporal_snow.dataset import load_timeseries, build_feature_columns, IMAGE_FEAT_COLS
from temporal_snow.animate_periods import predict_timeseries, load_gru


def make_timelapse(camera, df_cam, raw_snow, smoothed_snow, img_dir, out_path,
                   fps=8, hours_per_frame=1, max_frames=2000):
    dt = pd.to_datetime(df_cam["dt"]).values
    n = len(df_cam)

    # Subsample: pick 1 frame per hours_per_frame
    if hours_per_frame > 0:
        step = max(1, int(hours_per_frame * 60 / 10))  # assume ~10min intervals
        # More robust: use actual time gaps
        target_gap = np.timedelta64(int(hours_per_frame * 60), "m")
        picks = [0]
        for i in range(1, n):
            if dt[i] - dt[picks[-1]] >= target_gap:
                picks.append(i)
        picks = np.array(picks)
    else:
        picks = np.arange(n)

    if len(picks) > max_frames:
        step = len(picks) // max_frames
        picks = picks[::step]

    n_frames = len(picks)
    print(f"    {n} images → {n_frames} frames ({hours_per_frame}h/frame, {n_frames/fps:.0f}s video)")

    dt_pick = dt[picks]
    raw_pick = raw_snow[picks] * 100
    smooth_pick = smoothed_snow[picks] * 100

    # Preload image paths
    img_paths = []
    for idx in picks:
        row = df_cam.iloc[idx]
        p = Path(img_dir) / row["camera"] / f"{row['stem']}.jpg"
        img_paths.append(p if p.exists() else None)

    # Layout: image top, curve bottom
    fig = plt.figure(figsize=(12, 9))
    ax_img = fig.add_axes([0.02, 0.32, 0.96, 0.65])
    ax_curve = fig.add_axes([0.08, 0.05, 0.88, 0.24])

    # Static curve background (full raw in light blue)
    ax_curve.fill_between(dt_pick, raw_pick, alpha=0.15, color="steelblue")
    ax_curve.plot(dt_pick, smooth_pick, "-", color="steelblue", lw=0.5, alpha=0.3)
    ax_curve.axhline(50, color="gray", ls="--", lw=0.5, alpha=0.4)
    ax_curve.set_ylim(-2, 105)
    ax_curve.set_ylabel("Snow %", fontsize=9)
    ax_curve.set_xlabel("")
    ax_curve.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_curve.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_curve.tick_params(axis="x", rotation=30, labelsize=7)
    ax_curve.tick_params(axis="y", labelsize=8)
    ax_curve.grid(axis="x", alpha=0.2)

    # Dynamic elements
    img_handle = ax_img.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
    ax_img.axis("off")
    title_txt = ax_img.set_title("", fontsize=11, fontweight="bold", loc="left")

    # Revealed smoothed line + marker
    revealed_line, = ax_curve.plot([], [], "-", color="crimson", lw=1.5)
    marker, = ax_curve.plot([], [], "o", color="crimson", markersize=8, zorder=5)

    # Vertical line on curve
    vline = ax_curve.axvline(dt_pick[0], color="crimson", lw=0.8, alpha=0.5)

    def update(i):
        # Image
        if img_paths[i] is not None:
            try:
                img = np.array(Image.open(img_paths[i]).convert("RGB"))
                img_handle.set_data(img)
                img_handle.set_extent([0, img.shape[1], img.shape[0], 0])
            except Exception:
                pass

        # Title
        ts = pd.Timestamp(dt_pick[i]).strftime("%Y-%m-%d %H:%M")
        snow_val = smooth_pick[i]
        tag = "SNOW" if snow_val >= 50 else ""
        title_txt.set_text(f"{camera}  |  {ts}  |  snow: {snow_val:.0f}%  {tag}")

        # Reveal curve up to current frame
        revealed_line.set_data(dt_pick[:i+1], smooth_pick[:i+1])
        marker.set_data([dt_pick[i]], [smooth_pick[i]])
        vline.set_xdata([dt_pick[i]])

        return img_handle, title_txt, revealed_line, marker, vline

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(str(out_path), writer=writer)
    except Exception as e:
        print(f"    FFMpeg failed ({e}), falling back to GIF")
        out_path = out_path.with_suffix(".gif")
        anim.save(str(out_path), writer="pillow", fps=fps)

    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Full-camera timelapse with snow curve")
    parser.add_argument("--csv", default=str(_HERE.parent.parent / "analysis" / "snow_timeseries.csv"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img-dir", default=str(_HERE.parent.parent / "data"))
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE.parent.parent / "analysis" / "figures" / "gru_timelapses"))
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--hours-per-frame", type=float, default=1.0,
                        help="Subsample to 1 frame per N hours (0=all frames)")
    parser.add_argument("--max-frames", type=int, default=2000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df, snow_cols = load_timeseries(args.csv, require_target=False)
    has_img_feats = all(c in df.columns for c in IMAGE_FEAT_COLS)
    feature_cols = build_feature_columns(snow_cols, use_image_features=has_img_feats)
    snow_col = snow_cols[0] if snow_cols else "student_snow_pct"

    cameras = args.cameras or sorted(df["camera"].unique())
    out_root = Path(args.output_dir)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_gru(args.checkpoint, device)

    for camera in cameras:
        print(f"\n=== {camera} ===")
        df_cam = df[df["camera"] == camera].sort_values("dt").reset_index(drop=True)
        if len(df_cam) == 0:
            print("  No data")
            continue

        raw_snow = df_cam[snow_col].values.astype(np.float32)

        print(f"  Predicting {len(df_cam)} timesteps...")
        smoothed_snow = predict_timeseries(
            model, df_cam, feature_cols,
            seq_len=args.seq_len, device=device,
        )

        out_path = out_root / f"{camera}_timelapse.mp4"
        saved = make_timelapse(
            camera, df_cam, raw_snow, smoothed_snow,
            img_dir=args.img_dir,
            out_path=out_path,
            fps=args.fps,
            hours_per_frame=args.hours_per_frame,
            max_frames=args.max_frames,
        )
        print(f"  Saved: {saved}")

    print(f"\nDone! Output: {out_root}")


if __name__ == "__main__":
    main()
