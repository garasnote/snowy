#!/usr/bin/env python3
"""
Temporal consistency filter for snow_pct — GRU-based smoother.

NOT a classifier. Takes noisy per-image snow_pct sequences and outputs
temporally smoothed snow_pct. Target = Gaussian-weighted moving average
of the input snow_pct itself (self-supervised).

Snow doesn't appear/disappear within minutes. If 5 consecutive images
say snow and 1 says no-snow, that 1 is wrong. The GRU learns this.

Evaluation against meteo is done separately after inference.

Usage:
    python train_smoother.py
    python train_smoother.py --seq-len 72 --kernel-size 7
    python train_smoother.py --bidirectional
"""

import argparse
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import CSVLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

_HERE = Path(__file__).resolve().parent

if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from temporal_snow.model import TemporalSnowGRU, TemporalSnowLSTM
from temporal_snow.dataset import (
    load_timeseries, build_feature_columns, IMAGE_FEAT_COLS, SNOW_PCT_COLS, FeatureScaler
)


def gaussian_kernel(size, sigma=None):
    """1D Gaussian kernel for smoothing."""
    if sigma is None:
        sigma = size / 4.0
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smooth_signal(signal, kernel_size=7, sigma=None):
    """Apply Gaussian smoothing to a 1D signal."""
    kernel = gaussian_kernel(kernel_size, sigma)
    # Pad with edge values
    pad = kernel_size // 2
    padded = np.pad(signal, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


class SmootherDataset(Dataset):
    """Sliding windows for temporal smoothing.

    Input: raw features including snow_pct
    Target: Gaussian-smoothed snow_pct
    """

    def __init__(self, df, feature_cols, snow_col="student_snow_pct",
                 seq_len=72, stride=12, kernel_size=7, sigma=None,
                 cameras=None, min_coverage=0.5):
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.windows = []

        if cameras is not None:
            df = df[df["camera"].isin(cameras)]

        for camera, cam_df in df.groupby("camera"):
            cam_df = cam_df.sort_values("dt").reset_index(drop=True)
            n = len(cam_df)

            if n < seq_len:
                continue

            # Use raw (unscaled) snow_pct for target, normalized to [0, 1]
            raw_col = snow_col + "_raw"
            if raw_col in cam_df.columns:
                raw_snow = cam_df[raw_col].values.astype(np.float32) / 100.0
            else:
                raw_snow = cam_df[snow_col].values.astype(np.float32)
            # Compute smoothed target for entire camera timeline
            smoothed_snow = smooth_signal(raw_snow, kernel_size=kernel_size, sigma=sigma)

            for start in range(0, n - seq_len + 1, stride):
                end = start + seq_len
                window = cam_df.iloc[start:end]

                feat_matrix = window[feature_cols].values
                coverage = np.isfinite(feat_matrix).mean()
                if coverage < min_coverage:
                    continue

                x = feat_matrix.astype(np.float32)
                x = np.nan_to_num(x, nan=0.0)

                # Target: smoothed snow_pct for this window
                y = smoothed_snow[start:end].astype(np.float32)

                self.windows.append({
                    "x": x,
                    "y": y,
                    "camera": camera,
                    "start_dt": str(window.iloc[0]["dt"]),
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return {
            "x": torch.from_numpy(w["x"]),
            "y": torch.from_numpy(w["y"]),
        }


class TemporalSmootherModule(L.LightningModule):
    """GRU-based temporal smoother — regression, not classification."""

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
        arch="gru",
        learning_rate=1e-3,
        weight_decay=1e-4,
        t_max=100,
    ):
        super().__init__()
        self.save_hyperparameters()

        ModelClass = TemporalSnowGRU if arch == "gru" else TemporalSnowLSTM
        self.model = ModelClass(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x).squeeze(-1))

    def _shared_step(self, batch):
        x = batch["x"]  # (B, T, F)
        y = batch["y"]  # (B, T) — smoothed snow_pct [0,1]

        pred = self(x)  # (B, T) — predicted smoothed snow_pct [0,1]

        # MSE loss
        loss = F.mse_loss(pred, y)

        # Also track MAE
        mae = F.l1_loss(pred, y)

        return loss, mae, pred, y

    def training_step(self, batch, batch_idx):
        loss, mae, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae, pred, y = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True)

        # Also compute binary metrics at threshold 0.5 for interpretability
        pred_bin = (pred > 0.5).long()
        target_bin = (y > 0.5).long()
        correct = (pred_bin == target_bin).float().mean()
        self.log("val_bin_acc", correct, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.t_max, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal snow smoother (GRU regression)"
    )

    parser.add_argument("--csv", type=str,
                        default=str(_HERE.parent.parent / "analysis" / "snow_timeseries.csv"))
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE / "checkpoints"))

    # Architecture
    parser.add_argument("--arch", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")

    # Sequence
    parser.add_argument("--seq-len", type=int, default=72,
                        help="Window length in timesteps")
    parser.add_argument("--stride", type=int, default=12)

    # Smoothing target
    parser.add_argument("--kernel-size", type=int, default=7,
                        help="Gaussian smoothing kernel size (in timesteps)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Gaussian sigma (default: kernel_size/4)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Data
    parser.add_argument("--qa-filter", action="store_true",
                        help="Filter QA-rejected images (default: use all frames)")
    parser.add_argument("--no-image-features", action="store_true")
    parser.add_argument("--split", type=str, default="temporal",
                        choices=["temporal", "camera"])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-cameras", nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Hardware
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32",
                        choices=["32", "16-mixed"])

    args = parser.parse_args()
    L.seed_everything(args.seed)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    # Load data
    df, snow_pct_cols = load_timeseries(csv_path, require_target=False,
                                        qa_filter=args.qa_filter)

    has_img_feats = all(c in df.columns for c in IMAGE_FEAT_COLS)
    use_image_features = not args.no_image_features and has_img_feats
    feature_cols = build_feature_columns(snow_pct_cols, use_image_features=use_image_features)

    snow_col = snow_pct_cols[0] if snow_pct_cols else "student_snow_pct"

    # Get cameras with enough data
    all_cameras = sorted(df["camera"].unique())
    valid_cameras = []
    for cam in all_cameras:
        cam_df = df[df["camera"] == cam]
        if len(cam_df) >= args.seq_len:
            n_snow = (cam_df[snow_col] > 0.05).sum()
            valid_cameras.append(cam)
            print(f"  {cam}: {len(cam_df)} rows, {n_snow} with snow_pct>5%")

    print(f"\n  Valid cameras: {valid_cameras}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Smoothing kernel: {args.kernel_size} (sigma={args.sigma or args.kernel_size/4:.1f})")

    # Split into train/val DataFrames, then fit scaler on train
    train_dfs = []
    val_dfs = []

    if args.split == "temporal":
        gap_days = 1
        for cam in valid_cameras:
            cam_df = df[df["camera"] == cam].sort_values("dt").reset_index(drop=True)
            cam_df["_date"] = cam_df["dt"].dt.date
            unique_dates = sorted(cam_df["_date"].unique())
            n_dates = len(unique_dates)

            if n_dates < 3:
                split_idx = int(len(cam_df) * (1 - args.val_fraction))
                train_dfs.append(cam_df.iloc[:split_idx])
                val_dfs.append(cam_df.iloc[split_idx:])
            else:
                n_val_days = max(1, int(n_dates * args.val_fraction))
                gap = min(gap_days, n_dates - n_val_days - 1)
                gap = max(0, gap)
                val_start_idx = n_dates - n_val_days
                train_end_idx = val_start_idx - gap

                train_dates = set(unique_dates[:train_end_idx])
                val_dates = set(unique_dates[val_start_idx:])

                t_df = cam_df[cam_df["_date"].isin(train_dates)]
                v_df = cam_df[cam_df["_date"].isin(val_dates)]
                train_dfs.append(t_df)
                val_dfs.append(v_df)
                print(f"    {cam}: train {len(t_df)} rows "
                      f"({t_df['_date'].min()} → {t_df['_date'].max()}), "
                      f"val {len(v_df)} rows "
                      f"({v_df['_date'].min()} → {v_df['_date'].max()}), "
                      f"gap={gap} days")

        print(f"  Split: temporal day-boundary ({args.val_fraction:.0%} val, {gap_days}-day gap)")
    else:
        if args.val_cameras:
            val_cams = args.val_cameras
        else:
            rng = random.Random(args.seed)
            n_val = max(1, len(valid_cameras) // 3)
            val_cams = rng.sample(valid_cameras, n_val)
        train_cams = [c for c in valid_cameras if c not in val_cams]
        train_dfs.append(df[df["camera"].isin(train_cams)])
        val_dfs.append(df[df["camera"].isin(val_cams)])
        print(f"  Split: camera (train={train_cams}, val={val_cams})")

    all_train_df = pd.concat(train_dfs, ignore_index=True)
    all_val_df = pd.concat(val_dfs, ignore_index=True)

    # Save raw snow_pct before scaling (target must stay in original [0, max] range)
    all_train_df["_raw_snow"] = all_train_df[snow_col].values.copy()
    all_val_df["_raw_snow"] = all_val_df[snow_col].values.copy()

    # Fit StandardScaler on training data
    scaler = FeatureScaler(feature_cols)
    scaler.fit(all_train_df)
    all_train_df = scaler.transform(all_train_df)
    all_val_df = scaler.transform(all_val_df)

    # Restore raw snow for target computation
    all_train_df[snow_col + "_raw"] = all_train_df["_raw_snow"]
    all_val_df[snow_col + "_raw"] = all_val_df["_raw_snow"]
    print(f"  Scaler fit on {len(all_train_df)} train rows")

    # Build windowed datasets from scaled data
    if args.split == "temporal":
        train_windows = []
        val_windows = []
        for cam in valid_cameras:
            t_df = all_train_df[all_train_df["camera"] == cam]
            v_df = all_val_df[all_val_df["camera"] == cam]
            if len(t_df) > 0:
                ds = SmootherDataset(
                    t_df, feature_cols, snow_col=snow_col,
                    seq_len=args.seq_len, stride=args.stride,
                    kernel_size=args.kernel_size, sigma=args.sigma,
                )
                train_windows.extend(ds.windows)
            if len(v_df) > 0:
                ds = SmootherDataset(
                    v_df, feature_cols, snow_col=snow_col,
                    seq_len=args.seq_len, stride=args.stride,
                    kernel_size=args.kernel_size, sigma=args.sigma,
                )
                val_windows.extend(ds.windows)

        train_dataset = SmootherDataset.__new__(SmootherDataset)
        train_dataset.feature_cols = feature_cols
        train_dataset.seq_len = args.seq_len
        train_dataset.windows = train_windows

        val_dataset = SmootherDataset.__new__(SmootherDataset)
        val_dataset.feature_cols = feature_cols
        val_dataset.seq_len = args.seq_len
        val_dataset.windows = val_windows
    else:
        train_dataset = SmootherDataset(
            all_train_df, feature_cols, snow_col=snow_col,
            seq_len=args.seq_len, stride=args.stride,
            kernel_size=args.kernel_size, sigma=args.sigma,
        )
        val_dataset = SmootherDataset(
            all_val_df, feature_cols, snow_col=snow_col,
            seq_len=args.seq_len, stride=args.stride,
            kernel_size=args.kernel_size, sigma=args.sigma,
        )

    print(f"  Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    input_size = len(feature_cols)
    bidir_str = "bi_" if args.bidirectional else ""
    run_name = (
        f"smoother_{args.arch}_h{args.hidden_size}_l{args.num_layers}_{bidir_str}"
        f"k{args.kernel_size}_seq{args.seq_len}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Arch:         {args.arch}")
    print(f"  Hidden:       {args.hidden_size}")
    print(f"  Layers:       {args.num_layers}")
    print(f"  Bidirectional: {args.bidirectional}")
    print(f"  Dropout:      {args.dropout}")
    print(f"  Seq len:      {args.seq_len}")
    print(f"  Kernel size:  {args.kernel_size}")
    print(f"  Input size:   {input_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Output:       {output_dir}")

    module = TemporalSmootherModule(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        arch=args.arch,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        t_max=args.epochs,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best-epoch={epoch:02d}-val_mae={val_mae:.4f}",
        monitor="val_mae",
        mode="min",
        save_top_k=3,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_mae", patience=20, mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = CSVLogger(save_dir=str(output_dir), name="csv")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        precision=args.precision,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=logger,
        default_root_dir=str(output_dir),
        log_every_n_steps=10,
    )

    trainer.fit(module, train_loader, val_loader)

    scaler.save(output_dir / "scaler.json")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best val MAE:  {checkpoint_cb.best_model_score:.6f}")
    print(f"  Scaler:        {output_dir / 'scaler.json'}")
    print(f"  Checkpoints:   {output_dir}")


if __name__ == "__main__":
    main()
