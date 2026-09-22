#!/usr/bin/env python3
"""
Kalman filter baseline for temporal snow percentage smoothing.

State: [snow_pct]  (1D)
Measurement: student_snow_pct (noisy visual prediction)
Process model: random walk (snow_pct changes slowly)

Compares against GRU/LSTM smoothers on the same val split used in validate_meteo.py.
"""
import math
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

CSV = Path(__file__).parent / "snow_timeseries.csv"
OUT = Path(__file__).parent / "kalman_results"
OUT.mkdir(exist_ok=True)


# ── Scalar Kalman filter ──────────────────────────────────────────────────────

class KalmanSnow:
    """
    1D Kalman filter for snow percentage.

    State transition: x_t = x_{t-1} + w,  w ~ N(0, Q)
    Measurement:      z_t = x_t + v,       v ~ N(0, R)

    Q (process noise): how much snow_pct can change between frames
    R (measurement noise): how noisy the visual snow_pct estimate is
    """

    def __init__(self, Q=25.0, R=200.0):
        self.Q = Q    # process noise variance (in pct^2)
        self.R = R    # measurement noise variance (in pct^2)
        self.x = None  # state estimate
        self.P = None  # estimate covariance

    def reset(self, x0=0.0, P0=500.0):
        self.x = x0
        self.P = P0

    def update(self, z):
        """One Kalman step. z = measurement (raw snow_pct)."""
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (z - x_pred)
        self.P = (1 - K) * P_pred

        return self.x


def smooth_camera(snow_pct_series, Q, R):
    """Apply Kalman filter to one camera's snow_pct series. Returns smoothed array."""
    kf = KalmanSnow(Q=Q, R=R)
    kf.reset(x0=float(snow_pct_series.iloc[0]), P0=500.0)
    smoothed = []
    for z in snow_pct_series:
        smoothed.append(kf.update(float(z)))
    return np.array(smoothed)


# ── Hyperparameter search on train split ─────────────────────────────────────

def tune_kalman(df_train, Q_grid, R_grid):
    """Find best Q, R by minimizing MAE on training cameras."""
    best_mae, best_Q, best_R = float("inf"), None, None

    for Q in Q_grid:
        for R in R_grid:
            maes = []
            for cam, cam_df in df_train.groupby("camera"):
                cam_df = cam_df.sort_values("datetime").reset_index(drop=True)
                if len(cam_df) < 10:
                    continue
                smoothed = smooth_camera(cam_df["student_snow_pct"], Q, R)
                # Target: gaussian-smoothed snow_pct (kernel=7, same as GRU training)
                from scipy.ndimage import gaussian_filter1d
                target = gaussian_filter1d(cam_df["student_snow_pct"].values.astype(float), sigma=3.5)
                mae = np.abs(smoothed - target).mean()
                maes.append(mae)
            if maes:
                avg_mae = np.mean(maes)
                if avg_mae < best_mae:
                    best_mae, best_Q, best_R = avg_mae, Q, R

    print(f"Best Q={best_Q}, R={best_R}, train MAE={best_mae:.4f}")
    return best_Q, best_R


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    print("Loading timeseries...")
    df = pd.read_csv(CSV, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[df["qa_usable"] == 1].copy()
    df = df.sort_values(["camera", "datetime"]).reset_index(drop=True)
    print(f"  {len(df)} QA-usable frames, {df['camera'].nunique()} cameras")

    # Reproduce temporal train/val split (same as validate_meteo.py)
    import re
    def extract_date(stem):
        m = re.search(r"(\d{4})(\d{2})(\d{2})", str(stem))
        from datetime import date
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

    val_set = set()
    from datetime import date
    for cam, cam_df in df.groupby("camera"):
        stems = cam_df["stem"].tolist()
        dates = [extract_date(s) for s in stems]
        unique_dates = sorted(set(d for d in dates if d))
        if len(unique_dates) < 3:
            n_val = max(1, int(len(stems) * 0.2))
            for s in stems[-n_val:]:
                val_set.add((cam, s))
            continue
        n_val_days = max(1, int(len(unique_dates) * 0.2))
        gap = min(1, len(unique_dates) - n_val_days - 1)
        cutoff = unique_dates[-(n_val_days + gap)]
        for stem, d in zip(stems, dates):
            if d and d >= cutoff and (d - cutoff).days >= gap:
                val_set.add((cam, stem))

    df_train = df[~df.apply(lambda r: (r["camera"], r["stem"]) in val_set, axis=1)]
    df_val   = df[ df.apply(lambda r: (r["camera"], r["stem"]) in val_set, axis=1)]
    print(f"  Train: {len(df_train)}, Val: {len(df_val)}")

    # Tune on train
    print("\nTuning Kalman hyperparameters on train split...")
    Q_grid = [1, 5, 10, 25, 50, 100]
    R_grid = [25, 50, 100, 200, 400, 800]
    best_Q, best_R = tune_kalman(df_train, Q_grid, R_grid)

    # Evaluate on val split
    print("\nEvaluating on val split...")
    from scipy.ndimage import gaussian_filter1d

    all_smoothed, all_target, all_raw = [], [], []

    for cam, cam_df in df.groupby("camera"):
        cam_df = cam_df.sort_values("datetime").reset_index(drop=True)

        # Smooth full camera timeline (Kalman needs context from train period)
        smoothed_full = smooth_camera(cam_df["student_snow_pct"], best_Q, best_R)
        target_full = gaussian_filter1d(cam_df["student_snow_pct"].values.astype(float), sigma=3.5)

        # Extract val indices
        val_mask = cam_df.apply(lambda r: (cam, r["stem"]) in val_set, axis=1)
        if val_mask.sum() == 0:
            continue

        smoothed_val = smoothed_full[val_mask.values]
        target_val   = target_full[val_mask.values]
        raw_val      = cam_df.loc[val_mask, "student_snow_pct"].values

        all_smoothed.extend(smoothed_val.tolist())
        all_target.extend(target_val.tolist())
        all_raw.extend(raw_val.tolist())

        cam_mae_raw  = np.abs(raw_val - target_val).mean()
        cam_mae_kalman = np.abs(smoothed_val - target_val).mean()
        print(f"  {cam}: raw MAE={cam_mae_raw:.3f}  Kalman MAE={cam_mae_kalman:.3f}  (n={val_mask.sum()})")

    all_smoothed = np.array(all_smoothed)
    all_target   = np.array(all_target)
    all_raw      = np.array(all_raw)

    mae_raw    = np.abs(all_raw - all_target).mean()
    mae_kalman = np.abs(all_smoothed - all_target).mean()

    # Snow-present frames only (target > 5%)
    snow_mask = all_target > 5.0
    mae_raw_snow    = np.abs(all_raw - all_target)[snow_mask].mean() if snow_mask.sum() > 0 else float("nan")
    mae_kalman_snow = np.abs(all_smoothed - all_target)[snow_mask].mean() if snow_mask.sum() > 0 else float("nan")

    # Hard frames (large raw error > 10%)
    hard_mask = np.abs(all_raw - all_target) > 10.0
    mae_raw_hard    = np.abs(all_raw - all_target)[hard_mask].mean() if hard_mask.sum() > 0 else float("nan")
    mae_kalman_hard = np.abs(all_smoothed - all_target)[hard_mask].mean() if hard_mask.sum() > 0 else float("nan")

    print(f"\n{'='*50}")
    print(f"Overall val MAE (raw):    {mae_raw:.4f}")
    print(f"Overall val MAE (Kalman): {mae_kalman:.4f}")
    print(f"Improvement:              {mae_raw - mae_kalman:.4f} ({(mae_raw-mae_kalman)/mae_raw*100:.1f}%)")
    print(f"\nSnow-present frames (target >5%): {snow_mask.sum()}/{len(snow_mask)}")
    print(f"  Raw MAE:    {mae_raw_snow:.4f}")
    print(f"  Kalman MAE: {mae_kalman_snow:.4f}")
    print(f"  Improvement: {(mae_raw_snow-mae_kalman_snow)/mae_raw_snow*100:.1f}%")
    print(f"\nHard frames (|raw error| >10%): {hard_mask.sum()}/{len(hard_mask)}")
    print(f"  Raw MAE:    {mae_raw_hard:.4f}")
    print(f"  Kalman MAE: {mae_kalman_hard:.4f}")
    print(f"  Improvement: {(mae_raw_hard-mae_kalman_hard)/mae_raw_hard*100:.1f}%")
    print(f"\nBest params: Q={best_Q}, R={best_R}")

    # Save results
    results = {
        "Q": best_Q, "R": best_R,
        "mae_raw": float(mae_raw),
        "mae_kalman": float(mae_kalman),
        "mae_raw_snow": float(mae_raw_snow),
        "mae_kalman_snow": float(mae_kalman_snow),
        "mae_raw_hard": float(mae_raw_hard),
        "mae_kalman_hard": float(mae_kalman_hard),
        "n_val": len(all_smoothed),
        "n_snow": int(snow_mask.sum()),
        "n_hard": int(hard_mask.sum()),
    }
    import json
    with open(OUT / "kalman_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/kalman_results.json")


if __name__ == "__main__":
    main()
