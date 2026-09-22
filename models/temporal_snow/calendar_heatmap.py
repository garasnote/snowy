#!/usr/bin/env python3
"""GitHub-style per-camera calendar heatmap of smoothed snow probability.

Each camera gets its own figure: one cell per day, colored by mean
smoothed snow_pct that day. Grid is weeks (x) × weekdays (y), one row per year.

Usage:
    python calendar_heatmap.py --checkpoint path/to/best-smoother.ckpt
    python calendar_heatmap.py --checkpoint path/to/best.ckpt --cameras CM00014
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
from matplotlib.colors import LinearSegmentedColormap

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from temporal_snow.dataset import load_timeseries, build_feature_columns, IMAGE_FEAT_COLS
from temporal_snow.animate_periods import predict_timeseries, load_gru


CMAP = LinearSegmentedColormap.from_list(
    "snowprob", ["#2b2b2b", "#3a6ea5", "#9fc5e8", "#ffffff"]
)


def plot_year(ax, year, daily_prob, title):
    """Draw one year as a weeks × weekdays grid."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    n_days = (end - start).days + 1

    # For every day of the year, compute (week_col, weekday_row)
    cells = np.full((7, 54), np.nan)  # 54 weeks max
    for d in range(n_days):
        date = start + pd.Timedelta(days=d)
        weekday = date.weekday()  # 0=Mon
        week_col = int((date - start).days + start.weekday()) // 7
        prob = daily_prob.get(date.date(), np.nan)
        cells[weekday, week_col] = prob

    im = ax.imshow(cells, aspect="equal", cmap=CMAP, vmin=0, vmax=1,
                   interpolation="nearest")

    # Month labels along x
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(_HERE.parent.parent / "analysis" / "snow_timeseries.csv"))
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Smoother checkpoint path")
    ap.add_argument("--cameras", nargs="+", default=None)
    ap.add_argument("--output-dir", type=str,
                    default=str(_HERE.parent.parent / "analysis" / "figures" / "gru_calendars"))
    ap.add_argument("--seq-len", type=int, default=72)
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
