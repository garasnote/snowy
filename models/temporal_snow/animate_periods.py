#!/usr/bin/env python3
"""
Animate snow/no-snow periods using the temporal GRU smoother.

Renders side-by-side: webcam image (left) + smoothed snow probability curve (right).
Periods = runs of ≥6 real hours with smoothed prob ≥ 0.5 (snow) or < 0.5 (no-snow).

Usage:
    python animate_periods.py --checkpoint path/to/best-smoother.ckpt
    python animate_periods.py --checkpoint path/to/best.ckpt --cameras CM00014 CF00001
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
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from temporal_snow.dataset import load_timeseries, build_feature_columns, IMAGE_FEAT_COLS
from temporal_snow.train_smoother import TemporalSmootherModule


def predict_timeseries(model, df_cam, feature_cols, seq_len=72, device="cuda"):
    """Run smoother over full camera timeseries with sliding window, return per-timestep prob.

    The smoother's forward() already applies sigmoid — output is [0,1] directly.
    """
    df_cam = df_cam.sort_values("dt").reset_index(drop=True)
    n = len(df_cam)
    feats = df_cam[feature_cols].fillna(0).values.astype(np.float32)

    if n < seq_len:
        pad = np.zeros((seq_len - n, feats.shape[1]), dtype=np.float32)
        x = np.concatenate([pad, feats], axis=0)
        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            out = model(x_t).squeeze().cpu().numpy()
        return out[-n:]

    starts = list(range(0, n - seq_len + 1))

    batch_size = 64
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i + batch_size]
            batch_x = np.stack([feats[s:s + seq_len] for s in batch_starts])
            x_t = torch.from_numpy(batch_x).to(device)
            out = model(x_t).cpu().numpy()  # (B, T) — already sigmoid'd
            if out.ndim == 3:
                out = out.squeeze(-1)
            all_preds.append(out)

    # Average per-timestep predictions across overlapping windows
    acc = np.zeros(n)
    cnt = np.zeros(n)
    all_preds = np.concatenate(all_preds, axis=0)
    for idx, start in enumerate(starts):
        acc[start:start + seq_len] += all_preds[idx]
        cnt[start:start + seq_len] += 1

    cnt[cnt == 0] = 1
    return acc / cnt


def find_periods(df_cam, probs, min_hours=6.0, threshold=0.5, kind="snow",
                 max_gap_hours=3.0):
    """Find continuous runs of p >= threshold (snow) or p < threshold (no-snow)
    spanning at least min_hours. Splits on time gaps > max_gap_hours so
    disjoint observation windows don't fuse into one fake 'period'."""
    dt = pd.to_datetime(df_cam["dt"]).values
    if kind == "snow":
        mask = probs >= threshold
    else:
        mask = probs < threshold

    periods = []
    run_start = None

    def close_run(s, e):
        span_hours = (dt[e] - dt[s]) / np.timedelta64(1, "h")
        if span_hours >= min_hours:
            periods.append((s, e, span_hours))

    for i in range(len(mask)):
        gap_broken = False
        if run_start is not None and i > 0:
            gap_h = (dt[i] - dt[i - 1]) / np.timedelta64(1, "h")
            if gap_h > max_gap_hours:
                close_run(run_start, i - 1)
                run_start = None
                gap_broken = True

        if mask[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and not gap_broken:
                close_run(run_start, i - 1)
            run_start = None

    if run_start is not None:
        close_run(run_start, len(mask) - 1)

    return periods


def animate_period(df_cam, probs, start_idx, end_idx, img_dir, out_path,
                   fps=4, max_frames=200, pad_hours=24.0,
                   max_duration_hours=72.0, max_gap_hours=72.0):
    """Render side-by-side animation: image + probability curve with marker.

    Strategy:
      1. Pick a time window around the detected period:
           [period_start - pad, period_end + pad], capped at max_duration_hours
           centered on the period's nearest threshold crossing.
      2. Resample frames UNIFORMLY IN TIME (not by index). We pick the
         real sample nearest to each target time. This keeps the x-axis
         even instead of compressing 3 months into half the frames."""
    dt_all = pd.to_datetime(df_cam["dt"]).values
    hour = np.timedelta64(60, "m")

    t_period_start = dt_all[start_idx]
    t_period_end = dt_all[end_idx]

    t_clip_start = t_period_start - np.timedelta64(int(pad_hours * 60), "m")
    t_clip_end = t_period_end + np.timedelta64(int(pad_hours * 60), "m")

    duration_h = (t_clip_end - t_clip_start) / hour
    if duration_h > max_duration_hours:
        # Recenter on period midpoint, cap total span
        mid = t_period_start + (t_period_end - t_period_start) / 2
        half = np.timedelta64(int(max_duration_hours * 30), "m")  # half in minutes
        t_clip_start = mid - half
        t_clip_end = mid + half

    clip_mask = (dt_all >= t_clip_start) & (dt_all <= t_clip_end)
    if not clip_mask.any():
        clip_start, clip_end = start_idx, end_idx
    else:
        clip_idxs = np.where(clip_mask)[0]
        clip_start, clip_end = clip_idxs[0], clip_idxs[-1]

    # Uniformly-spaced target times across the clip span
    t0 = dt_all[clip_start]
    t1 = dt_all[clip_end]
    n_frames = min(max_frames, clip_end - clip_start + 1)
    if n_frames < 2:
        n_frames = 2
    span_ns = (t1 - t0).astype("timedelta64[ns]").astype(np.int64)
    target_ns = np.linspace(0, span_ns, n_frames).astype(np.int64)
    # Pick nearest real sample to each target
    clip_dt = dt_all[clip_start:clip_end + 1]
    clip_ns = (clip_dt - t0).astype("timedelta64[ns]").astype(np.int64)
    picks = np.unique(np.array([
        clip_start + int(np.argmin(np.abs(clip_ns - tns))) for tns in target_ns
    ]))

    df_slice = df_cam.iloc[picks].reset_index(drop=True)
    prob_slice = probs[picks]

    # Locate the period highlight in sampled-frame space
    period_rel_start = int(np.searchsorted(picks, start_idx))
    period_rel_end = int(np.searchsorted(picks, end_idx))
    period_rel_start = min(period_rel_start, len(picks) - 1)
    period_rel_end = min(period_rel_end, len(picks) - 1)

    n = len(df_slice)

    # Precompute image paths
    img_paths = []
    for _, row in df_slice.iterrows():
        stem = row["stem"]
        p = Path(img_dir) / row["camera"] / f"{stem}.jpg"
        img_paths.append(p if p.exists() else None)

    dt = pd.to_datetime(df_slice["dt"]).values

    fig, (ax_img, ax_prob) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [1.3, 1]})

    # Static probability curve
    ax_prob.plot(dt, prob_slice, "-", color="steelblue", lw=1.5, alpha=0.5)
    ax_prob.axhline(0.5, color="gray", ls="--", lw=0.8)
    # Shade the detected period (the pre/post padding is outside this band)
    if 0 <= period_rel_start < n and 0 <= period_rel_end < n:
        ax_prob.axvspan(dt[period_rel_start], dt[period_rel_end],
                        color="gold", alpha=0.15, zorder=0)
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel("Snow probability (GRU)")
    ax_prob.set_xlabel("Time")
    ax_prob.set_title(f"{df_slice.iloc[0]['camera']}")
    ax_prob.tick_params(axis="x", rotation=30)
    fig.autofmt_xdate()

    # Dynamic elements
    img_handle = ax_img.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
    ax_img.axis("off")
    title_txt = ax_img.set_title("")
    marker, = ax_prob.plot([], [], "o", color="crimson", markersize=10, zorder=5)
    filled, = ax_prob.plot([], [], "-", color="crimson", lw=2.5)

    def update(i):
        # Update image
        if img_paths[i] is not None:
            img = np.array(Image.open(img_paths[i]).convert("RGB"))
            img_handle.set_data(img)
            img_handle.set_extent([0, img.shape[1], img.shape[0], 0])
        else:
            img_handle.set_data(np.zeros((100, 100, 3), dtype=np.uint8))

        # Title with timestamp + prob
        ts = pd.Timestamp(dt[i]).strftime("%Y-%m-%d %H:%M")
        p = prob_slice[i]
        tag = "SNOW" if p >= 0.5 else "no snow"
        title_txt.set_text(f"{ts}  |  p={p:.2f}  [{tag}]")

        # Reveal curve up to i
        marker.set_data([dt[i]], [prob_slice[i]])
        filled.set_data(dt[:i + 1], prob_slice[:i + 1])

        return img_handle, title_txt, marker, filled

    anim = FuncAnimation(fig, update, frames=n, interval=1000 // fps, blit=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(out_path), writer=writer)
    except Exception as e:
        print(f"  FFMpeg failed ({e}), falling back to GIF")
        out_path = out_path.with_suffix(".gif")
        anim.save(str(out_path), writer="pillow", fps=fps)

    plt.close(fig)
    return out_path


def load_gru(ckpt_path, device):
    model = TemporalSmootherModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(_HERE.parent.parent / "analysis" / "snow_timeseries.csv"))
    parser.add_argument("--img-dir", default=str(_HERE.parent.parent / "data"))
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Smoother checkpoint path")
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE.parent.parent / "analysis" / "figures" / "gru_animations"))
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--min-hours", type=float, default=6.0)
    parser.add_argument("--max-gap-hours", type=float, default=3.0,
                        help="Split periods on time gaps larger than this")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-periods-per-camera", type=int, default=3)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--pad-hours", type=float, default=24.0,
                        help="Lead-in/lead-out around the detected period so "
                             "the threshold-crossing transition is visible")
    parser.add_argument("--max-duration-hours", type=float, default=72.0,
                        help="Cap total clip duration. If period+pad exceeds "
                             "this, clip is recentered on period midpoint.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df, snow_cols = load_timeseries(args.csv, require_target=False)
    has_img_feats = all(c in df.columns for c in IMAGE_FEAT_COLS)
    feature_cols = build_feature_columns(snow_cols, use_image_features=has_img_feats)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    cameras = args.cameras or sorted(df["camera"].unique())
    out_root = Path(args.output_dir)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_gru(args.checkpoint, device)

    for camera in cameras:
        print(f"\n=== {camera} ===")
        df_cam = df[df["camera"] == camera].sort_values("dt").reset_index(drop=True)
        if len(df_cam) == 0:
            print(f"  No data — skipping")
            continue

        print(f"  Predicting {len(df_cam)} timesteps...")
        probs = predict_timeseries(
            model, df_cam, feature_cols,
            seq_len=args.seq_len, device=device,
        )

        for kind in ["snow", "no_snow"]:
            periods = find_periods(
                df_cam, probs,
                min_hours=args.min_hours,
                threshold=args.threshold,
                kind=kind,
                max_gap_hours=args.max_gap_hours,
            )
            periods = sorted(periods, key=lambda p: -p[2])[:args.max_periods_per_camera]
            print(f"    {kind}: {len(periods)} top periods "
                  f"(min {args.min_hours}h)")

            for pi, (s, e, hrs) in enumerate(periods):
                start_ts = pd.to_datetime(df_cam.iloc[s]["dt"]).strftime("%Y%m%d_%H%M")
                out_name = f"{camera}_{kind}_{pi:02d}_{start_ts}_{hrs:.0f}h.mp4"
                out_path = out_root / camera / out_name
                print(f"      rendering {out_name} ({e - s + 1} frames, {hrs:.1f}h)")
                try:
                    saved = animate_period(
                        df_cam, probs, s, e,
                        img_dir=args.img_dir,
                        out_path=out_path,
                        fps=args.fps,
                        max_frames=args.max_frames,
                        pad_hours=args.pad_hours,
                        max_duration_hours=args.max_duration_hours,
                        max_gap_hours=max(args.max_gap_hours * 24, 72.0),
                    )
                    print(f"        saved: {saved.name}")
                except Exception as err:
                    print(f"        FAILED: {err}")

    print(f"\nDone! Output: {out_root}")


if __name__ == "__main__":
    main()
