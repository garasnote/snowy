#!/usr/bin/env python3
"""
GRU/LSTM smoother visualizations — calendar heatmaps, full-station
timelines, and per-camera timelapse videos, all driven by a trained
smoother checkpoint.

Consolidated from calendar_heatmap.py, plot_full_timeline.py, timelapse.py.
Each keeps its original CLI flags, now behind a subcommand:

    python visualize_smoother.py calendar  --checkpoint path/to/best.ckpt
    python visualize_smoother.py timeline  --checkpoint path/to/best.ckpt --summary
    python visualize_smoother.py timelapse --checkpoint path/to/best.ckpt --fps 12

Run `python visualize_smoother.py <subcommand> --help` for the full option
list. animate_periods.py (period-animation subcommand of the same sbatch
job, run_gru_visualizations.sbatch) stays a separate file — it's a shared
library imported by calendar_heatmap/plot_full_timeline/timelapse's
underlying logic (predict_timeseries, load_gru) as well as by
gru_ood_snow_freq.py and shap_gru.py.
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
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from matplotlib.animation import FuncAnimation, FFMpegWriter

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from temporal_snow.dataset import load_timeseries, build_feature_columns, IMAGE_FEAT_COLS
from temporal_snow.animate_periods import predict_timeseries, load_gru


# =============================================================================
# CALENDAR — GitHub-style per-camera calendar heatmap of smoothed snow prob
# =============================================================================

CALENDAR_CMAP = LinearSegmentedColormap.from_list(
    "snowprob", ["#2b2b2b", "#3a6ea5", "#9fc5e8", "#ffffff"]
)


def plot_year(ax, year, daily_prob, title):
    """Draw one year as a weeks × weekdays grid."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    n_days = (end - start).days + 1

    cells = np.full((7, 54), np.nan)  # 54 weeks max
    for d in range(n_days):
        date = start + pd.Timedelta(days=d)
        weekday = date.weekday()  # 0=Mon
        week_col = int((date - start).days + start.weekday()) // 7
        prob = daily_prob.get(date.date(), np.nan)
        cells[weekday, week_col] = prob

    im = ax.imshow(cells, aspect="equal", cmap=CALENDAR_CMAP, vmin=0, vmax=1,
                   interpolation="nearest")

    month_starts = []
    for m in range(1, 13):
        d = pd.Timestamp(year=year, month=m, day=1)
        week_col = int((d - start).days + start.weekday()) // 7
        month_starts.append((week_col, d.strftime("%b")))
    ax.set_xticks([mc for mc, _ in month_starts])
    ax.set_xticklabels([lbl for _, lbl in month_starts], fontsize=7)

    ax.set_yticks(range(7))
    ax.set_yticklabels(["M", "T", "W", "T", "F", "S", "S"], fontsize=7)
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    return im


def calendar_for_camera(camera, df_cam, probs, out_path):
    df_cam = df_cam.copy()
    df_cam["dt"] = pd.to_datetime(df_cam["dt"])
    df_cam["prob"] = probs
    df_cam["date"] = df_cam["dt"].dt.date
    daily = df_cam.groupby("date")["prob"].mean()

    years = sorted({d.year for d in daily.index})
    fig, axes = plt.subplots(
        len(years), 1, figsize=(14, 1.8 * len(years) + 0.8),
        squeeze=False,
    )

    im = None
    for i, y in enumerate(years):
        year_daily = {d: v for d, v in daily.items() if d.year == y}
        im = plot_year(axes[i, 0], y, year_daily, f"{camera} — {y}")

    fig.suptitle(f"{camera}: Daily mean smoothed snow probability",
                 fontsize=12, y=1.0)
    fig.tight_layout()

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                        pad=0.02, aspect=30)
    cbar.set_label("Snow probability", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def add_calendar_subparser(subparsers):
    p = subparsers.add_parser("calendar", help="GitHub-style per-camera calendar heatmap")
    p.add_argument("--csv", default=str(_HERE.parent.parent / "analysis" / "snow_timeseries.csv"))
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Smoother checkpoint path")
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--output-dir", type=str,
                    default=str(_HERE.parent.parent / "analysis" / "figures" / "gru_calendars"))
    p.add_argument("--seq-len", type=int, default=72)
    return p


def run_calendar(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df, snow_cols = load_timeseries(args.csv, require_target=False)
    has_img_feats = all(c in df.columns for c in IMAGE_FEAT_COLS)
    feature_cols = build_feature_columns(snow_cols, use_image_features=has_img_feats)

    cameras = args.cameras or sorted(df["camera"].unique())
    out_root = Path(args.output_dir)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_gru(args.checkpoint, device)

    for cam in cameras:
        print(f"\n=== {cam} ===")
        df_cam = df[df["camera"] == cam].sort_values("dt").reset_index(drop=True)
        if len(df_cam) == 0:
            continue

        print(f"  Predicting {len(df_cam)} timesteps...")
        probs = predict_timeseries(
            model, df_cam, feature_cols,
            seq_len=args.seq_len, device=device,
        )
        out_path = out_root / f"{cam}_calendar.png"
        calendar_for_camera(cam, df_cam, probs, out_path)
        print(f"    saved {out_path.name}")

    print(f"\nDone! Output: {out_root}")


# =============================================================================
# TIMELINE — full-station snow timeline: raw vs smoothed vs meteo ground truth
# =============================================================================

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


def add_timeline_subparser(subparsers):
    p = subparsers.add_parser("timeline", help="Full station snow timelines with GRU smoother")
    p.add_argument("--csv", default=str(_HERE.parent.parent / "analysis" / "snow_timeseries.csv"))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--output-dir", type=str,
                    default=str(_HERE.parent.parent / "analysis" / "figures" / "gru_timelines"))
    p.add_argument("--seq-len", type=int, default=72)
    p.add_argument("--daily-only", action="store_true",
                    help="Only plot daily aggregation (smaller files)")
    p.add_argument("--summary", action="store_true",
                    help="Also generate all-cameras summary figure")
    return p


def run_timeline(args):
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


# =============================================================================
# TIMELAPSE — full-camera timelapse video with snow % curve
# =============================================================================

def make_timelapse(camera, df_cam, raw_snow, smoothed_snow, img_dir, out_path,
                   fps=8, hours_per_frame=1, max_frames=2000):
    dt = pd.to_datetime(df_cam["dt"]).values
    n = len(df_cam)

    if hours_per_frame > 0:
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

    img_paths = []
    for idx in picks:
        row = df_cam.iloc[idx]
        p = Path(img_dir) / row["camera"] / f"{row['stem']}.jpg"
        img_paths.append(p if p.exists() else None)

    fig = plt.figure(figsize=(12, 9))
    ax_img = fig.add_axes([0.02, 0.32, 0.96, 0.65])
    ax_curve = fig.add_axes([0.08, 0.05, 0.88, 0.24])

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

    img_handle = ax_img.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
    ax_img.axis("off")
    title_txt = ax_img.set_title("", fontsize=11, fontweight="bold", loc="left")

    revealed_line, = ax_curve.plot([], [], "-", color="crimson", lw=1.5)
    marker, = ax_curve.plot([], [], "o", color="crimson", markersize=8, zorder=5)

    vline = ax_curve.axvline(dt_pick[0], color="crimson", lw=0.8, alpha=0.5)

    def update(i):
        if img_paths[i] is not None:
            try:
                img = np.array(Image.open(img_paths[i]).convert("RGB"))
                img_handle.set_data(img)
                img_handle.set_extent([0, img.shape[1], img.shape[0], 0])
            except Exception:
                pass

        ts = pd.Timestamp(dt_pick[i]).strftime("%Y-%m-%d %H:%M")
        snow_val = smooth_pick[i]
        tag = "SNOW" if snow_val >= 50 else ""
        title_txt.set_text(f"{camera}  |  {ts}  |  snow: {snow_val:.0f}%  {tag}")

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


def add_timelapse_subparser(subparsers):
    p = subparsers.add_parser("timelapse", help="Full-camera timelapse video with snow-pct curve")
    p.add_argument("--csv", default=str(_HERE.parent.parent / "analysis" / "snow_timeseries.csv"))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--img-dir", default=str(_HERE.parent.parent / "data"))
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--output-dir", type=str,
                    default=str(_HERE.parent.parent / "analysis" / "figures" / "gru_timelapses"))
    p.add_argument("--seq-len", type=int, default=72)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--hours-per-frame", type=float, default=1.0,
                    help="Subsample to 1 frame per N hours (0=all frames)")
    p.add_argument("--max-frames", type=int, default=2000)
    return p


def run_timelapse(args):
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GRU/LSTM smoother visualizations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_calendar_subparser(subparsers)
    add_timeline_subparser(subparsers)
    add_timelapse_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "calendar":
        run_calendar(args)
    elif args.command == "timeline":
        run_timeline(args)
    elif args.command == "timelapse":
        run_timelapse(args)


if __name__ == "__main__":
    main()
