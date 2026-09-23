#!/usr/bin/env python3
"""
Train GRU temporal snow refinement model.

Uses the timeseries CSV from analysis/build_features.py's "timeseries"
subcommand as input.
Camera-based train/val split. Outputs per-timestep snow probability.

Usage:
    python train.py --csv analysis/snow_timeseries.csv
    python train.py --csv analysis/snow_timeseries.csv --arch lstm --bidirectional
    python train.py --csv analysis/snow_timeseries.csv --val-cameras CF00001 CM00009
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import CSVLogger
try:
    from lightning.pytorch.loggers import TensorBoardLogger
    HAS_TB = True
except ImportError:
    TensorBoardLogger = None
    HAS_TB = False
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy

_HERE = Path(__file__).resolve().parent

# Allow imports from parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from temporal_snow.model import TemporalSnowGRU, TemporalSnowLSTM
from temporal_snow.dataset import build_dataloaders


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal loss for binary classification (operates on logits)."""
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce
    return loss.mean()


class TemporalSnowModule(L.LightningModule):
    """Lightning module for temporal snow refinement."""

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
        arch="gru",
        learning_rate=1e-3,
        weight_decay=1e-4,
        pos_weight=1.0,
        t_max=100,
        loss_type="focal",
        focal_alpha=0.25,
        focal_gamma=2.0,
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

        self.loss_type = loss_type
        if loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )

        self.train_f1 = BinaryF1Score()
        self.train_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_acc = BinaryAccuracy()

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, logits, targets):
        if self.loss_type == "focal":
            return sigmoid_focal_loss(
                logits, targets,
                alpha=self.hparams.focal_alpha,
                gamma=self.hparams.focal_gamma,
            )
        return self.loss_fn(logits, targets)

    def _shared_step(self, batch):
        x = batch["x"]         # (B, T, F)
        y = batch["y"]         # (B, T)
        mask = batch["mask"]   # (B, T)

        logits = self(x).squeeze(-1)  # (B, T)

        # Masked loss — only on valid timesteps
        loss = self._compute_loss(logits[mask], y[mask])

        preds = (torch.sigmoid(logits[mask]) > 0.5).long()
        targets = y[mask].long()

        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.train_f1(preds, targets)
        self.train_acc(preds, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.val_f1(preds, targets)
        self.val_acc(preds, targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
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
        description="Train temporal snow refinement model (GRU/LSTM)"
    )

    parser.add_argument("--csv", type=str,
                        default=str(_HERE.parent.parent / "analysis" / "snow_timeseries.csv"),
                        help="Path to snow_timeseries.csv")
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE / "checkpoints"))

    # Architecture
    parser.add_argument("--arch", type=str, default="gru",
                        choices=["gru", "lstm"])
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true")

    # Sequence
    parser.add_argument("--seq-len", type=int, default=24,
                        help="Window length in timesteps (~1 day of hourly data)")
    parser.add_argument("--stride", type=int, default=6,
                        help="Stride between windows")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="focal",
                        choices=["focal", "bce"],
                        help="Loss function (default: focal)")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Focal loss alpha (default: 0.25)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="Positive class weight for BCE loss (only with --loss bce)")

    # Data
    parser.add_argument("--balance", action="store_true",
                        help="Undersample negative windows for ~50/50 class balance")
    parser.add_argument("--oversample", type=int, default=0,
                        help="Oversample positive windows by this factor (0=disabled)")
    parser.add_argument("--daily", action="store_true",
                        help="Aggregate to daily resolution before windowing")
    parser.add_argument("--no-image-features", action="store_true",
                        help="Disable image-level features (use only snow_pct + meteo + time)")
    parser.add_argument("--split", type=str, default="temporal",
                        choices=["temporal", "camera"],
                        help="Split strategy: temporal (chronological per camera) or camera (hold-out)")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Val fraction for temporal split (default: 0.2)")
    parser.add_argument("--val-cameras", nargs="+", default=None,
                        help="Cameras for validation (only with --split camera)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay for optimizer (L2 regularization)")

    # Hardware
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32",
                        choices=["32", "16-mixed"])

    args = parser.parse_args()
    L.seed_everything(args.seed)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run "
              f"'python analysis/build_features.py timeseries' first.")
        sys.exit(1)

    print("=" * 60)
    print("TEMPORAL SNOW REFINEMENT — TRAINING")
    print("=" * 60)

    train_loader, val_loader, feature_cols, (n_train, n_val), scaler = build_dataloaders(
        csv_path=csv_path,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        val_cameras=args.val_cameras,
        num_workers=args.num_workers,
        seed=args.seed,
        balance_train=args.balance,
        use_image_features=not args.no_image_features,
        split=args.split,
        val_fraction=args.val_fraction,
        daily=args.daily,
        oversample=args.oversample,
    )

    input_size = len(feature_cols)

    # Auto-compute positive weight from training data (only for BCE)
    if args.loss == "bce" and args.pos_weight == 1.0:
        all_targets = []
        for batch in train_loader:
            all_targets.append(batch["y"][batch["mask"]].numpy())
        all_targets = np.concatenate(all_targets)
        n_pos = all_targets.sum()
        n_neg = len(all_targets) - n_pos
        if n_pos > 0:
            auto_pw = n_neg / n_pos
            args.pos_weight = min(auto_pw, 10.0)
            print(f"  Auto pos_weight: {args.pos_weight:.2f} "
                  f"(pos={int(n_pos)}, neg={int(n_neg)})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.arch}_h{args.hidden_size}_l{args.num_layers}"
    if args.bidirectional:
        run_name += "_bi"
    run_name += f"_{args.loss}"
    if args.balance:
        run_name += "_bal"
    run_name += f"_seq{args.seq_len}_{ts}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Arch:         {args.arch}")
    print(f"  Hidden:       {args.hidden_size}")
    print(f"  Layers:       {args.num_layers}")
    print(f"  Bidirectional: {args.bidirectional}")
    print(f"  Seq len:      {args.seq_len}")
    print(f"  Input size:   {input_size}")
    print(f"  Loss:         {args.loss}")
    if args.loss == "focal":
        print(f"  Focal alpha:  {args.focal_alpha}")
        print(f"  Focal gamma:  {args.focal_gamma}")
    else:
        print(f"  Pos weight:   {args.pos_weight:.2f}")
    print(f"  Balanced:     {args.balance}")
    print(f"  LR:           {args.lr}")
    print(f"  Output:       {output_dir}")
    print()

    model = TemporalSnowModule(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        arch=args.arch,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        t_max=args.epochs,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params:,}\n")

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch:02d}-f1{val_f1:.3f}",
            monitor="val_f1",
            mode="max",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val_f1", patience=15, mode="max"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    loggers = [CSVLogger(str(output_dir), name="csv")]
    if HAS_TB:
        loggers.insert(0, TensorBoardLogger(str(output_dir), name="tb", version=""))

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    final_ckpt = output_dir / "final.ckpt"
    trainer.save_checkpoint(str(final_ckpt))

    scaler.save(output_dir / "scaler.json")
    print(f"  Scaler saved: {output_dir / 'scaler.json'}")

    best_f1 = trainer.callback_metrics.get("val_f1", "N/A")
    best_acc = trainer.callback_metrics.get("val_acc", "N/A")
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best val F1:  {best_f1}")
    print(f"  Best val Acc: {best_acc}")
    print(f"  Checkpoints:  {output_dir}")


if __name__ == "__main__":
    main()
