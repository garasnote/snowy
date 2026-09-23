"""
Temporal snow dataset — sliding windows over per-camera time series.

Loads the timeseries CSV produced by analysis/build_features.py's
"timeseries" subcommand and creates windowed sequences of features for
GRU-based temporal refinement.

Features per timestep:
  - student_snow_pct (from student snow model)
  - t2m, visibility, weather_code (meteo)
  - hour_sin, hour_cos, doy_sin, doy_cos (cyclical time encoding)

Target:
  - snow_present (binary, from meteo)
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


SNOW_PCT_COLS = ["student_snow_pct"]
METEO_COLS = ["t2m", "visibility", "weather_code"]
MASK_FEAT_COLS = [
    "snow_n_components", "snow_largest_ratio",
    "snow_centroid_y", "ground_coverage",
]
IMAGE_FEAT_COLS = [
    "brightness", "contrast", "edge_density", "entropy",
    "white_frac", "low_sat_frac", "blue_ratio",
    "bottom_brightness", "bottom_white_frac",
]


def load_timeseries(csv_path, snow_pct_cols=None, require_target=True, qa_filter=False):
    """Load and preprocess the timeseries CSV.

    Returns DataFrame with feature columns and binary target.
    qa_filter: if True, drop QA-rejected images (qa_usable==0).
    """
    df = pd.read_csv(csv_path)

    if snow_pct_cols is None:
        snow_pct_cols = [c for c in SNOW_PCT_COLS if c in df.columns]

    # Optionally drop QA-rejected images
    if qa_filter and "qa_usable" in df.columns:
        n_before = len(df)
        df = df[df["qa_usable"] == 1].reset_index(drop=True)
        n_rejected = n_before - len(df)
        if n_rejected > 0:
            print(f"  QA filter: dropped {n_rejected}/{n_before} rejected images")
    elif "qa_usable" in df.columns:
        print(f"  QA filter: disabled — using all {len(df)} frames")

    # Parse datetime
    df["dt"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["camera", "dt"]).reset_index(drop=True)

    # Cyclical time encoding
    df["hour_sin"] = np.sin(2 * math.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * math.pi * df["hour"] / 24)
    doy = df["dt"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * math.pi * doy / 365)

    # Fill missing values with sensible defaults (no scaling here — done after split)
    FILL_DEFAULTS = {
        "t2m": 0, "visibility": 1000, "weather_code": 0,
        "brightness": 128, "contrast": 50, "edge_density": 10, "entropy": 4,
        "white_frac": 0, "low_sat_frac": 0, "blue_ratio": 0,
        "bottom_white_frac": 0, "bottom_brightness": 128,
        "snow_n_components": 0, "snow_largest_ratio": 0,
        "snow_centroid_y": 0.5, "ground_coverage": 0,
    }
    for col in snow_pct_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col, default in FILL_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # Log-transform skewed features before standardization
    if "visibility" in df.columns:
        df["visibility"] = np.log1p(df["visibility"])
    if "snow_n_components" in df.columns:
        df["snow_n_components"] = np.log1p(df["snow_n_components"])

    # Target: binary snow presence from meteo
    if "snow_present" in df.columns:
        df["target"] = df["snow_present"].fillna(False).astype(float)
    elif require_target:
        raise ValueError("No 'snow_present' column in CSV — cannot create target")

    return df, snow_pct_cols


def aggregate_daily(df, snow_pct_cols):
    """Aggregate hourly timeseries to daily per camera.

    Takes mean of continuous features, max of snow_pct (daily peak),
    and keeps the target as-is (already daily).
    """
    agg_rules = {}
    # Snow pct: take daily max (peak coverage that day)
    for col in snow_pct_cols:
        if col in df.columns:
            agg_rules[col] = "max"
    # Image features: take daily mean
    for col in IMAGE_FEAT_COLS:
        if col in df.columns:
            agg_rules[col] = "mean"
    # Meteo: take first (same all day for daily meteo)
    for col in METEO_COLS:
        if col in df.columns:
            agg_rules[col] = "first"
    # Target: take max (if any hour is snow, day is snow)
    if "target" in df.columns:
        agg_rules["target"] = "max"
    # Time encoding: use midday values (mean of cyclical is wrong, use doy only)
    agg_rules["doy_sin"] = "first"
    agg_rules["doy_cos"] = "first"

    # Group by camera + date
    df["date_key"] = df["dt"].dt.date
    daily = df.groupby(["camera", "date_key"]).agg(agg_rules).reset_index()
    daily["dt"] = pd.to_datetime(daily["date_key"])
    daily = daily.sort_values(["camera", "dt"]).reset_index(drop=True)

    # Drop hour encoding (meaningless for daily)
    daily["hour_sin"] = 0.0
    daily["hour_cos"] = 1.0

    print(f"  Daily aggregation: {len(df)} hourly → {len(daily)} daily rows")
    n_pos = (daily["target"] == 1).sum() if "target" in daily.columns else 0
    print(f"  Daily positive rate: {n_pos}/{len(daily)} ({n_pos/len(daily)*100:.1f}%)")

    return daily


def build_feature_columns(snow_pct_cols, use_image_features=True):
    """Return ordered list of feature column names."""
    meteo = [c for c in METEO_COLS]
    time_enc = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    mask_feats = [c for c in MASK_FEAT_COLS]
    img_feats = [c for c in IMAGE_FEAT_COLS] if use_image_features else []
    return snow_pct_cols + mask_feats + img_feats + meteo + time_enc


class FeatureScaler:
    """StandardScaler that serializes to/from JSON for checkpoint portability."""

    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or []
        self.mean_ = None
        self.std_ = None

    def fit(self, df):
        vals = df[self.feature_cols].values.astype(np.float64)
        self.mean_ = np.nanmean(vals, axis=0)
        self.std_ = np.nanstd(vals, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, df):
        df = df.copy()
        for i, col in enumerate(self.feature_cols):
            if col in df.columns:
                df[col] = (df[col] - self.mean_[i]) / self.std_[i]
        return df

    def transform_array(self, x):
        """Transform a numpy array (n_samples, n_features) in-place."""
        return (x - self.mean_.astype(np.float32)) / self.std_.astype(np.float32)

    def to_dict(self):
        return {
            "feature_cols": self.feature_cols,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        scaler = cls(d["feature_cols"])
        scaler.mean_ = np.array(d["mean"], dtype=np.float64)
        scaler.std_ = np.array(d["std"], dtype=np.float64)
        return scaler

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


class TemporalSnowDataset(Dataset):
    """Sliding window sequences from per-camera time series.

    Each sample is:
      x: (seq_len, n_features) float32
      y: (seq_len,) float32 binary target
      mask: (seq_len,) bool — True where data exists (for variable-length seqs)
    """

    def __init__(self, df, feature_cols, seq_len=24, stride=6,
                 cameras=None, min_coverage=0.5):
        """
        Args:
            df:            Preprocessed DataFrame from load_timeseries
            feature_cols:  List of feature column names
            seq_len:       Window length (number of timesteps)
            stride:        Step between consecutive windows
            cameras:       Restrict to these cameras (None = all)
            min_coverage:  Minimum fraction of non-NaN features in window to include
        """
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.windows = []

        if cameras is not None:
            df = df[df["camera"].isin(cameras)]

        for camera, cam_df in df.groupby("camera"):
            cam_df = cam_df.sort_values("dt").reset_index(drop=True)
            n = len(cam_df)

            for start in range(0, n - seq_len + 1, stride):
                window = cam_df.iloc[start : start + seq_len]

                # Check minimum feature coverage
                feat_matrix = window[feature_cols].values
                coverage = np.isfinite(feat_matrix).mean()
                if coverage < min_coverage:
                    continue

                # Extract features and target
                x = feat_matrix.astype(np.float32)
                x = np.nan_to_num(x, nan=0.0)

                y = window["target"].values.astype(np.float32)
                valid = np.isfinite(window[feature_cols].values).all(axis=1)

                self.windows.append({
                    "x": x,
                    "y": y,
                    "mask": valid,
                    "camera": camera,
                    "start_dt": str(window.iloc[0]["dt"]),
                })

    def has_snow(self, idx):
        """Check if a window contains any positive (snow) timesteps."""
        return self.windows[idx]["y"].sum() > 0

    def balance(self, seed=42):
        """Undersample negative windows to achieve ~50/50 class balance."""
        rng = random.Random(seed)
        pos = [i for i in range(len(self.windows)) if self.has_snow(i)]
        neg = [i for i in range(len(self.windows)) if not self.has_snow(i)]

        if not pos or not neg:
            print(f"  [balance] Cannot balance: {len(pos)} pos, {len(neg)} neg windows")
            return

        # Undersample negatives to match positives
        n_keep = len(pos)
        neg_sampled = rng.sample(neg, min(n_keep, len(neg)))

        keep = set(pos) | set(neg_sampled)
        self.windows = [self.windows[i] for i in sorted(keep)]
        print(f"  [balance] Kept {len(pos)} pos + {len(neg_sampled)} neg = {len(self.windows)} windows")

    def oversample_positive(self, factor=3, seed=42):
        """Repeat positive windows to counteract class imbalance."""
        rng = random.Random(seed)
        pos = [self.windows[i] for i in range(len(self.windows)) if self.has_snow(i)]
        neg_count = len(self.windows) - len(pos)

        if not pos:
            print("  [oversample] No positive windows found")
            return

        # Repeat positives factor times
        extra = pos * (factor - 1)
        self.windows.extend(extra)
        rng.shuffle(self.windows)
        print(f"  [oversample] {len(pos)} pos × {factor} = {len(pos)*factor} "
              f"(+ {neg_count} neg = {len(self.windows)} total)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return {
            "x": torch.from_numpy(w["x"]),             # (seq_len, n_features)
            "y": torch.from_numpy(w["y"]),              # (seq_len,)
            "mask": torch.from_numpy(w["mask"]),        # (seq_len,)
        }


def build_dataloaders(
    csv_path,
    seq_len=24,
    stride=6,
    batch_size=32,
    val_cameras=None,
    num_workers=4,
    seed=42,
    balance_train=False,
    use_image_features=True,
    split="temporal",
    val_fraction=0.2,
    daily=False,
    oversample=0,
):
    """Build train/val DataLoaders.

    Args:
        csv_path:      Path to snow_timeseries.csv
        seq_len:       Window length
        stride:        Stride between windows
        batch_size:    Batch size
        val_cameras:   Cameras to hold out (only used if split="camera")
        num_workers:   DataLoader workers
        seed:          Random seed
        balance_train: Undersample negative windows in training set for ~50/50
        use_image_features: Include image-level features (brightness, etc.)
        split:         "temporal" (per-camera chronological) or "camera" (hold-out cameras)
        val_fraction:  Fraction for val in temporal split (default 0.2)
        daily:         Aggregate to daily resolution before windowing
        oversample:    Repeat positive windows N times (0 = disabled)

    Returns:
        train_loader, val_loader, feature_cols, (n_train, n_val), scaler
    """
    df, snow_pct_cols = load_timeseries(csv_path)

    # Check if image features are available in the CSV
    has_img_feats = all(c in df.columns for c in IMAGE_FEAT_COLS)
    if use_image_features and not has_img_feats:
        print("  [WARN] Image features not in CSV — run "
              "'python analysis/build_features.py image-features' + rebuild timeseries")
        use_image_features = False

    feature_cols = build_feature_columns(snow_pct_cols, use_image_features=use_image_features)

    # Daily aggregation
    if daily:
        df = aggregate_daily(df, snow_pct_cols)

    all_cameras = sorted(df["camera"].unique())

    # Only use cameras that have target labels (snow_present matched from meteo)
    labeled_cameras = []
    for cam in all_cameras:
        cam_df = df[df["camera"] == cam]
        n_labeled = cam_df["target"].notna().sum() if "target" in cam_df.columns else 0
        n_pos = (cam_df["target"] == 1).sum() if "target" in cam_df.columns else 0
        if n_labeled > 0:
            labeled_cameras.append(cam)
            print(f"  {cam}: {len(cam_df)} rows, {int(n_labeled)} labeled, "
                  f"{int(n_pos)} snow ({n_pos/n_labeled*100:.1f}%)")
        else:
            print(f"  {cam}: {len(cam_df)} rows, NO meteo labels — skipping")

    if not labeled_cameras:
        raise ValueError("No cameras with meteo labels found.")

    print(f"\n  Labeled cameras: {labeled_cameras}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Split mode: {split}, daily: {daily}")

    # Split data into train/val DataFrames, then fit scaler on train
    train_dfs = []
    val_dfs = []

    if split == "temporal":
        gap_days = 1
        for cam in labeled_cameras:
            cam_df = df[df["camera"] == cam].sort_values("dt").reset_index(drop=True)
            cam_df["_date"] = cam_df["dt"].dt.date
            unique_dates = sorted(cam_df["_date"].unique())
            n_dates = len(unique_dates)

            if n_dates < 3:
                split_idx = int(len(cam_df) * (1 - val_fraction))
                train_dfs.append(cam_df.iloc[:split_idx])
                val_dfs.append(cam_df.iloc[split_idx:])
            else:
                n_val_days = max(1, int(n_dates * val_fraction))
                gap = min(gap_days, n_dates - n_val_days - 1)
                gap = max(0, gap)
                val_start_idx = n_dates - n_val_days
                train_end_idx = val_start_idx - gap

                train_dates = set(unique_dates[:train_end_idx])
                val_dates = set(unique_dates[val_start_idx:])

                train_dfs.append(cam_df[cam_df["_date"].isin(train_dates)])
                val_dfs.append(cam_df[cam_df["_date"].isin(val_dates)])

        print(f"  Temporal split: day-boundary ({val_fraction:.0%} val, {gap_days}-day gap)")

    else:
        if val_cameras is None:
            rng = random.Random(seed)
            n_val = max(1, len(labeled_cameras) // 3)
            val_cameras = rng.sample(labeled_cameras, n_val)

        train_cameras = [c for c in labeled_cameras if c not in val_cameras]
        print(f"  Train cameras: {train_cameras}")
        print(f"  Val cameras:   {val_cameras}")

        train_dfs.append(df[df["camera"].isin(train_cameras)])
        val_dfs.append(df[df["camera"].isin(val_cameras)])

    all_train_df = pd.concat(train_dfs, ignore_index=True)
    all_val_df = pd.concat(val_dfs, ignore_index=True)

    # Fit StandardScaler on training data only
    scaler = FeatureScaler(feature_cols)
    scaler.fit(all_train_df)
    all_train_df = scaler.transform(all_train_df)
    all_val_df = scaler.transform(all_val_df)
    print(f"  Scaler fit on {len(all_train_df)} train rows")

    # Build windowed datasets from scaled data
    if split == "temporal":
        train_windows = []
        val_windows = []
        for cam in labeled_cameras:
            t_df = all_train_df[all_train_df["camera"] == cam]
            v_df = all_val_df[all_val_df["camera"] == cam]
            if len(t_df) > 0:
                ds = TemporalSnowDataset(t_df, feature_cols, seq_len=seq_len, stride=stride)
                train_windows.extend(ds.windows)
            if len(v_df) > 0:
                ds = TemporalSnowDataset(v_df, feature_cols, seq_len=seq_len, stride=stride)
                val_windows.extend(ds.windows)

        train_ds = TemporalSnowDataset.__new__(TemporalSnowDataset)
        train_ds.feature_cols = feature_cols
        train_ds.seq_len = seq_len
        train_ds.windows = train_windows

        val_ds = TemporalSnowDataset.__new__(TemporalSnowDataset)
        val_ds.feature_cols = feature_cols
        val_ds.seq_len = seq_len
        val_ds.windows = val_windows
    else:
        train_ds = TemporalSnowDataset(
            all_train_df, feature_cols, seq_len=seq_len, stride=stride,
        )
        val_ds = TemporalSnowDataset(
            all_val_df, feature_cols, seq_len=seq_len, stride=stride,
        )

    if balance_train:
        train_ds.balance(seed=seed)
    if oversample > 0:
        train_ds.oversample_positive(factor=oversample, seed=seed)

    print(f"  Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, feature_cols, (len(train_ds), len(val_ds)), scaler
