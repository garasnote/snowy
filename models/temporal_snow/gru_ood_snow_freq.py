#!/usr/bin/env python3
"""
Compute GRU-smoothed snow-presence frequency for the OOD cameras, matching
Table tab:cameras' methodology (student-network predictions run through the
production GRU smoother, % of frames with smoothed prob > 5%).

Usage:
    python gru_ood_snow_freq.py --checkpoint-dir models/temporal_snow/checkpoints/smoother_gru_h128_l2_k7_seq72_20260920_024134
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_MASTER = Path("/shared/home/juan.osorio/master")
sys.path.insert(0, str(_MASTER / "models"))

from temporal_snow.dataset import load_timeseries, FeatureScaler
from temporal_snow.train_smoother import TemporalSmootherModule
from temporal_snow.animate_periods import predict_timeseries as _predict_unscaled


def predict_timeseries_scaled(model, df_cam, feature_cols, scaler, seq_len=72, device="cuda"):
    """Same sliding-window logic as animate_periods.predict_timeseries, but
    applies the FeatureScaler before inference (required for checkpoints
    trained with standardized features)."""
    df_cam = df_cam.sort_values("dt").reset_index(drop=True)
    n = len(df_cam)
    feats = df_cam[feature_cols].fillna(0).values.astype(np.float32)
    feats = scaler.transform_array(feats)

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
            out = model(x_t).cpu().numpy()
            if out.ndim == 3:
                out = out.squeeze(-1)
            all_preds.append(out)

    acc = np.zeros(n)
    cnt = np.zeros(n)
    all_preds = np.concatenate(all_preds, axis=0)
    for idx, start in enumerate(starts):
        acc[start:start + seq_len] += all_preds[idx]
        cnt[start:start + seq_len] += 1
    return acc / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True,
                     help="Dir containing best-*.ckpt and scaler.json")
    ap.add_argument("--csv", default=str(_MASTER / "analysis" / "snow_timeseries.csv"))
    ap.add_argument("--cameras", nargs="+",
                     default=["CM00054", "CM00067", "CM00071", "CM00115"])
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpts = sorted(ckpt_dir.glob("best-*.ckpt"))
    if not ckpts:
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    assert ckpts, f"No .ckpt found in {ckpt_dir}"
    ckpt_path = ckpts[-1]
    scaler_path = ckpt_dir / "scaler.json"
    assert scaler_path.exists(), f"No scaler.json in {ckpt_dir}"

    print(f"Checkpoint: {ckpt_path}")
    print(f"Scaler:     {scaler_path}")

    scaler = FeatureScaler.load(scaler_path)
    feature_cols = scaler.feature_cols
    print(f"Feature cols ({len(feature_cols)}): {feature_cols}")

    device = torch.device(args.device)
    model = TemporalSmootherModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)

    df, snow_pct_cols = load_timeseries(args.csv, require_target=False, qa_filter=False)
    df["dt"] = pd.to_datetime(df["datetime"])

    print(f"\n{'Camera':<10} {'Total':>8} {'Snow>5%':>10} {'Freq%':>8}")
    print("-" * 40)
    results = {}
    for cam in args.cameras:
        df_cam = df[df["camera"] == cam].copy()
        if df_cam.empty:
            print(f"{cam:<10} {'NO DATA':>8}")
            continue
        probs = predict_timeseries_scaled(model, df_cam, feature_cols, scaler, device=device)
        n = len(probs)
        n_snow = int((probs > args.threshold).sum())
        freq = 100.0 * n_snow / n if n > 0 else 0.0
        results[cam] = (n, n_snow, freq)
        print(f"{cam:<10} {n:>8} {n_snow:>10} {freq:>7.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
