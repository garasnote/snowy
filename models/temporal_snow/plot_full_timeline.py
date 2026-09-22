#!/usr/bin/env python3
"""
Full-station snow timeline: raw student snow_pct vs GRU-smoothed vs meteo ground truth.

One multi-panel figure per camera spanning entire observation period.
Top panel: snow_pct timeline (raw + smoothed + meteo snow_cm).
Bottom panel: daily aggregation for cleaner view.

Usage:
    python plot_full_timeline.py --checkpoint path/to/best-smoother.ckpt
    python plot_full_timeline.py --checkpoint path/to/best.ckpt --cameras CF00001 CM00108
    python plot_full_timeline.py --checkpoint path/to/best.ckpt --daily-only
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
from matplotlib.patches import Patch

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from temporal_snow.dataset import load_timeseries, build_feature_columns, IMAGE_FEAT_COLS
from temporal_snow.animate_periods import predict_timeseries, load_gru


def plot_camera_timeline(camera, df_cam, raw_snow, smoothed_snow, out_path,
                         meteo_snow=None, daily_only=False):
    dt = pd.to_datetime(df_cam["dt"])
    date_range = f"{dt.min().strftime('%Y-%m-%d')} → {dt.max().strftime('%Y-%m-%d')}"
    n_days = (dt.max() - dt.min()).days

    if daily_only:
        fig, ax_daily = plt.subplots(1, 1, figsize=(max(16, n_days * 0.04), 5))
        axes_list = [ax_daily]
    else:
        fig, (ax_full, ax_daily) = plt.subplots(
            2, 1, figsize=(max(16, n_days * 0.04), 10),
            gridspec_kw={"height_ratios": [1.2, 1]},
        )
        axes_list = [ax_full, ax_daily]

        # --- Top panel: full resolution ---
        ax_full.fill_between(dt, raw_snow * 100, alpha=0.15, color="steelblue",
                             label="Raw student snow%")
        ax_full.plot(dt, smoothed_snow * 100, "-", color="crimson", lw=0.8,
                     alpha=0.9, label="GRU smoothed")
        ax_full.axhline(50, color="gray", ls="--", lw=0.5, alpha=0.5)

        if meteo_snow is not None:
            snow_mask = meteo_snow.astype(bool)
            for i in range(len(snow_mask)):
                if snow_mask.iloc[i]:
                    ax_full.axvspan(dt.iloc[i] - pd.Timedelta(minutes=30),
                                    dt.iloc[i] + pd.Timedelta(minutes=30),
                                    alpha=0.08, color="dodgerblue", lw=0)

        ax_full.set_ylabel("Snow %")
        ax_full.set_ylim(-2, 105)
        ax_full.set_title(f"{camera} — Full resolution ({len(df_cam):,} images, {date_range})",
                          fontsize=11, fontweight="bold")
        ax_full.legend(loc="upper right", fontsize=8)
        ax_full.xaxis.set_major_locator(mdates.MonthLocator())
        ax_full.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_full.tick_params(axis="x", rotation=45)
        ax_full.grid(axis="x", alpha=0.3)

    # --- Bottom panel: daily aggregation ---
    daily_df = pd.DataFrame({
        "dt": dt,
        "raw": raw_snow * 100,
        "smoothed": smoothed_snow * 100,
    })
    daily_df["date"] = daily_df["dt"].dt.date
    daily_agg = daily_df.groupby("date").agg(
        raw_mean=("raw", "mean"),
        raw_max=("raw", "max"),
        smoothed_mean=("smoothed", "mean"),
        smoothed_max=("smoothed", "max"),
    ).reset_index()
    daily_agg["date"] = pd.to_datetime(daily_agg["date"])

    ax_daily.fill_between(daily_agg["date"], daily_agg["raw_max"],
                          alpha=0.12, color="steelblue", label="Raw daily max")
    ax_daily.plot(daily_agg["date"], daily_agg["raw_mean"], "-",
                  color="steelblue", lw=0.6, alpha=0.5, label="Raw daily mean")
    ax_daily.plot(daily_agg["date"], daily_agg["smoothed_mean"], "-",
                  color="crimson", lw=1.2, label="Smoothed daily mean")
    ax_daily.fill_between(daily_agg["date"], daily_agg["smoothed_max"],
                          alpha=0.15, color="crimson")



    ax_daily.axhline(50, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax_daily.set_ylabel("Snow %")
    ax_daily.set_ylim(-2, 105)
    title = f"{camera} — Daily aggregation" if not daily_only else \
            f"{camera} — Daily snow timeline ({date_range})"
    ax_daily.set_title(title, fontsize=11, fontweight="bold")
    ax_daily.legend(loc="upper right", fontsize=8)
    ax_daily.xaxis.set_major_locator(mdates.MonthLocator())
    ax_daily.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_daily.tick_params(axis="x", rotation=45)
    ax_daily.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_cameras_summary(all_results, out_path):
    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(18, 3 * n + 1), squeeze=False)

    for i, (camera, dt, raw, smoothed) in enumerate(all_results):
        ax = axes[i, 0]

        daily_df = pd.DataFrame({"dt": dt, "raw": raw * 100, "smoothed": smoothed * 100})
        daily_df["date"] = daily_df["dt"].dt.date
        daily_agg = daily_df.groupby("date").agg(
            raw_mean=("raw", "mean"),
            smoothed_mean=("smoothed", "mean"),
        ).reset_index()
        daily_agg["date"] = pd.to_datetime(daily_agg["date"])

        ax.fill_between(daily_agg["date"], daily_agg["raw_mean"],
                        alpha=0.2, color="steelblue")
        ax.plot(daily_agg["date"], daily_agg["smoothed_mean"], "-",
                color="crimson", lw=1.0)
        ax.axhline(50, color="gray", ls="--", lw=0.4, alpha=0.4)
        ax.set_ylim(-2, 105)
        ax.set_ylabel("Snow %", fontsize=8)
        ax.set_title(f"{camera} ({len(dt):,} imgs)", fontsize=9,
                      fontweight="bold", loc="left")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.2)

    fig.suptitle("All stations — Raw (blue) vs GRU-smoothed (red) snow %",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot full station snow timelines with GRU smoother"
    )
    parser.add_argument("--csv", default=str(_HERE.parent.parent / "analysis" / "snow_timeseries.csv"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE.parent.parent / "analysis" / "figures" / "gru_timelines"))
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--daily-only", action="store_true",
                        help="Only plot daily aggregation (smaller files)")
    parser.add_argument("--summary", action="store_true",
                        help="Also generate all-cameras summary figure")
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
    print(f"Features ({len(feature_cols)}): {feature_cols[:5]}...")
    print(f"Cameras: {cameras}\n")

    all_results = []

    for camera in cameras:
        print(f"=== {camera} ===")
        df_cam = df[df["camera"] == camera].sort_values("dt").reset_index(drop=True)
        if len(df_cam) == 0:
            print("  No data — skipping")
            continue

        raw_snow = df_cam[snow_col].values.astype(np.float32)

        print(f"  Predicting {len(df_cam)} timesteps...")
        smoothed_snow = predict_timeseries(
            model, df_cam, feature_cols,
            seq_len=args.seq_len, device=device,
        )

        meteo_snow = df_cam["snow_present"] if "snow_present" in df_cam.columns else None

        out_path = out_root / f"{camera}_timeline.png"
        saved = plot_camera_timeline(
            camera, df_cam, raw_snow, smoothed_snow, out_path,
            meteo_snow=meteo_snow,
            daily_only=args.daily_only,
        )
        print(f"  Saved: {saved}")

        dt = pd.to_datetime(df_cam["dt"])
        raw_pct = (raw_snow > 0.05).mean() * 100
        smooth_pct = (smoothed_snow > 0.5).mean() * 100
        print(f"  Raw snow>5%: {raw_pct:.1f}%, Smoothed>50%: {smooth_pct:.1f}%")

        all_results.append((camera, dt, raw_snow, smoothed_snow))

    if args.summary and len(all_results) > 1:
        summary_path = out_root / "all_cameras_summary.png"
        plot_all_cameras_summary(all_results, summary_path)
        print(f"\nSummary: {summary_path}")

    print(f"\nDone! Output: {out_root}")


if __name__ == "__main__":
    main()
