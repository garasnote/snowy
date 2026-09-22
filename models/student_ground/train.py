#!/usr/bin/env python3
"""
Student segmentation network — knowledge distillation from ensemble.

Trains a lightweight segmentation model on ensemble pseudo-labels
(soft confidence maps from ensemble_voting.py).

Three model variants:
    dinov2_base   — frozen DINOv2-Base + L-Decoder (~1.1M trainable params)
    dinov2_large  — frozen DINOv2-Large + L-Decoder (~1.3M trainable params)
    mobilenet     — frozen MobileNetV3-Small + UNet decoder (~3M total)

Usage:
    # Quick smoke test (1 epoch, 200 images)
    python train.py --model-variant dinov2_base --ensemble-threshold 4 --epochs 1 --sample 200

    # Full training run (early stopping patience=8 will terminate before 100 if converged)
    python train.py --model-variant dinov2_base --ensemble-threshold 5 --ensemble-weighted

    # Larger backbone
    python train.py --model-variant dinov2_large --ensemble-threshold 5 --ensemble-weighted --batch-size 8

    # CNN baseline
    python train.py --model-variant mobilenet --ensemble-threshold 5 --ensemble-weighted --batch-size 32

    # Fine-tune on real GT annotations (from Label Studio export)
    python train.py --finetune --checkpoint checkpoints/best-epoch=07-iouval_iou=0.908.ckpt

    # Fine-tune from scratch (no pretrained checkpoint)
    python train.py --finetune --model-variant dinov2_base --lr 1e-4

TensorBoard:
    tensorboard --logdir=checkpoints/<run_name>/tb
    # Then SSH port-forward: ssh -L 6006:localhost:6006 <cluster>
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import CSVLogger
try:
    import tensorboard  # noqa: F401 — check the actual package exists
    from lightning.pytorch.loggers import TensorBoardLogger
    HAS_TENSORBOARD = True
except (ImportError, ModuleNotFoundError):
    TensorBoardLogger = None
    HAS_TENSORBOARD = False
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import JaccardIndex
from torchmetrics.classification import BinaryF1Score
from torchvision.utils import make_grid

# Allow running as script: add master/models to sys.path so `student_ground` package resolves
_HERE = Path(__file__).resolve().parent          # master/models/student_ground/
_MODELS_DIR = _HERE.parent                       # master/models/
_SEG_OUTPUT = _HERE.parent.parent / "preprocessing" / "segmentation"  # where output data lives

if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from student_ground.configs import MODEL_REGISTRY, get_config, IMAGENET_MEAN, IMAGENET_STD
from student_ground.dataset import build_dataloaders, build_gt_dataloaders
from student_ground.losses import combined_loss
from student_ground.models import build_model


# ── Lightning Module ──────────────────────────────────────────────────────────

class StudentSegmentationModule(L.LightningModule):
    """Knowledge distillation student for binary ground segmentation."""

    def __init__(
        self,
        model_variant="dinov2_base",
        learning_rate=None,
        weight_decay=0.01,
        bce_weight=1.0,
        dice_weight=0.5,
        t_max=100,
        log_images=True,
        use_focal=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        unfreeze=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        try:
            cfg = get_config(model_variant)
        except KeyError:
            cfg = {}
        lr = learning_rate if learning_rate is not None else cfg.get("lr", 1e-3)
        self.hparams.learning_rate = lr

        self.model = build_model(model_variant)
        if unfreeze:
            for param in self.model.parameters():
                param.requires_grad = True
        self.input_size = cfg.get("input_size", 518)

        # Metrics (compared to hard binary mask, not soft targets)
        self.train_iou = JaccardIndex(task="binary")
        self.val_iou = JaccardIndex(task="binary")
        self.val_dice = BinaryF1Score()  # F1 == Dice for binary segmentation

        # Storage for image logging (first val batch)
        self._first_val_batch = None

    def forward(self, images):
        return self.model(images, output_size=(self.input_size, self.input_size))

    def _shared_step(self, batch):
        images = batch["image"]
        confidence = batch["confidence"]           # (B, 1, H, W) soft target
        binary_mask = batch["binary_mask"]         # (B, 1, H, W) hard target

        logits = self(images)                      # (B, 1, H, W)

        # Resize targets to match logits if needed
        if logits.shape != confidence.shape:
            confidence = F.interpolate(
                confidence, size=logits.shape[2:], mode="bilinear", align_corners=False
            )
            binary_mask = F.interpolate(
                binary_mask, size=logits.shape[2:], mode="nearest"
            )

        loss, bce, dice = combined_loss(
            logits, confidence,
            bce_weight=self.hparams.bce_weight,
            dice_weight=self.hparams.dice_weight,
            use_focal=self.hparams.get("use_focal", False),
            focal_alpha=self.hparams.get("focal_alpha", 0.25),
            focal_gamma=self.hparams.get("focal_gamma", 2.0),
        )

        # Predicted binary mask (for metric computation)
        preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
        targets = binary_mask.long().squeeze(1)

        return loss, bce, dice, preds, targets

    def training_step(self, batch, batch_idx):
        loss, bce, dice, preds, targets = self._shared_step(batch)

        self.train_iou(preds, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_bce", bce, on_step=False, on_epoch=True)
        self.log("train_dice_loss", dice, on_step=False, on_epoch=True)
        self.log("train_iou", self.train_iou, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, bce, dice, preds, targets = self._shared_step(batch)

        self.val_iou(preds, targets)
        self.val_dice(preds, targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_iou", self.val_iou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_dice, prog_bar=True, on_step=False, on_epoch=True)

        # Stash the first batch for image logging at epoch end
        if batch_idx == 0 and self.hparams.log_images:
            self._first_val_batch = {k: v.detach() if isinstance(v, torch.Tensor) else v
                                     for k, v in batch.items()}
        return loss

    def on_validation_epoch_end(self):
        if not self.hparams.log_images or self._first_val_batch is None:
            return

        tb = self._get_tb_writer()
        if tb is None:
            return

        batch = self._first_val_batch
        n = min(4, batch["image"].shape[0])

        images     = batch["image"][:n]
        confidence = batch["confidence"][:n]
        binary_mask = batch["binary_mask"][:n]

        with torch.no_grad():
            logits = self(images)
            preds = torch.sigmoid(logits)

        # Resize masks/preds to image resolution for display
        h, w = images.shape[2:]
        if preds.shape[2:] != (h, w):
            preds       = F.interpolate(preds,        size=(h, w), mode="bilinear", align_corners=False)
            confidence  = F.interpolate(confidence,   size=(h, w), mode="bilinear", align_corners=False)
            binary_mask = F.interpolate(binary_mask,  size=(h, w), mode="nearest")

        # Unnormalize images for display
        mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD,  device=images.device).view(1, 3, 1, 1)
        imgs = (images * std + mean).clamp(0, 1)

        # Expand single-channel masks to 3-channel for grid
        def to3(t): return t.expand(3, -1, -1).clamp(0, 1)  # (1,H,W) → (3,H,W)

        panels = []
        for i in range(n):
            panels += [imgs[i], to3(confidence[i]), to3(binary_mask[i]), to3(preds[i])]

        grid = make_grid(panels, nrow=4, pad_value=0.3)
        tb.add_image("val/image | soft_label | binary_mask | prediction",
                     grid, self.current_epoch)

    def _get_tb_writer(self):
        """Return the underlying SummaryWriter from the TensorBoardLogger, or None."""
        if TensorBoardLogger is None:
            return None
        loggers = self.trainer.loggers if hasattr(self.trainer, "loggers") else [self.trainer.logger]
        for lg in loggers:
            if isinstance(lg, TensorBoardLogger):
                return lg.experiment
        return None

    def configure_optimizers(self):
        # Only optimize trainable (decoder) parameters
        trainable = list(self.model.trainable_parameters())
        optimizer = AdamW(
            trainable,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train student segmentation network on ensemble pseudo-labels"
    )

    # Fine-tuning on real ground truth annotations
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune on real GT annotations instead of pseudo-labels")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint to load before fine-tuning")
    parser.add_argument("--annotations", type=str,
                        default="/shared/home/juan.osorio/master/data/annotations/ground_annotations.json",
                        help="Path to ground_annotations.json (used with --finetune)")
    parser.add_argument("--depth-dir", type=str, default=None,
                        help="Path to depth maps dir (enables RGBD 4-channel input)")

    # Paths
    parser.add_argument("--data-dir", type=str,
                        default="/shared/home/juan.osorio/master/data",
                        help="Root data directory (contains camera subdirs)")
    parser.add_argument("--ensemble-dir", type=str, default=None,
                        help="Path to specific ensemble dir (overrides --ensemble-threshold)")
    parser.add_argument("--ensemble-threshold", type=int, default=4,
                        help="Which majority_N ensemble to use (default: 4)")
    parser.add_argument("--ensemble-weighted", action="store_true",
                        help="Use weighted_N instead of majority_N ensemble")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "checkpoints"),
                        help="Directory for checkpoints and logs (default: models/student_ground/checkpoints/)")

    # Model
    parser.add_argument("--model-variant", type=str, default="dinov2_base",
                        help="Model variant (append '_rgbd' for depth, e.g. dinov2_base_rgbd)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs — early stopping (patience=8) will usually terminate earlier")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: from model config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: from model config)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bce-weight", type=float, default=1.0,
                        help="Weight for soft BCE loss term (default: 1.0)")
    parser.add_argument("--dice-weight", type=float, default=0.5,
                        help="Weight for Dice loss term (default: 0.5)")
    parser.add_argument("--val-split", type=float, default=0.2)

    # Data loading
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Restrict to specific cameras (default: all)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Use only N samples total (0 = all, useful for smoke tests)")

    parser.add_argument("--unfreeze", action="store_true",
                        help="Unfreeze encoder for full model fine-tuning")

    # Hardware
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--no-log-images", action="store_true",
                        help="Disable TensorBoard image logging (faster on slow storage)")

    # K-fold CV
    parser.add_argument("--kfold-idx", type=int, default=-1,
                        help="K-fold index (0-4 for 5-fold CV, -1 to disable)")
    parser.add_argument("--kfold-k", type=int, default=5,
                        help="Total number of folds")

    args = parser.parse_args()

    L.seed_everything(args.seed)

    cfg = get_config(args.model_variant)
    batch_size = args.batch_size or cfg["batch_size"]
    input_size = cfg["input_size"]
    log_images = not args.no_log_images

    # Default LR: lower for fine-tuning (1e-4), otherwise from config
    if args.lr is not None:
        lr = args.lr
    elif args.finetune:
        lr = 1e-4
    else:
        lr = cfg["lr"]

    # Timestamp for this run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.finetune:
        # ── Fine-tuning on real GT annotations ──
        if args.kfold_idx >= 0:
            run_name = f"finetune_{args.model_variant}_kfold_{args.kfold_idx}_{ts}"
        else:
            run_name = f"finetune_{args.model_variant}_{ts}"
        output_dir = Path(args.output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        annotations_path = Path(args.annotations)
        if not annotations_path.exists():
            print(f"ERROR: Annotations file not found: {annotations_path}")
            sys.exit(1)

        print("=" * 60)
        print("STUDENT SEGMENTATION — FINE-TUNING ON GT ANNOTATIONS")
        print("=" * 60)
        print(f"  Model:       {args.model_variant}")
        print(f"  Checkpoint:  {args.checkpoint or '(from scratch)'}")
        print(f"  Annotations: {annotations_path}")
        if args.kfold_idx >= 0:
            print(f"  K-Fold:      Fold {args.kfold_idx} of {args.kfold_k}")
        print(f"  Epochs:      {args.epochs}")
        print(f"  Batch size:  {batch_size}")
        print(f"  Input size:  {input_size}x{input_size}")
        print(f"  LR:          {lr}")
        print(f"  Depth:       {args.depth_dir or 'disabled (RGB only)'}")
        print(f"  Precision:   {args.precision}")
        print(f"  Output:      {output_dir}")
        if HAS_TENSORBOARD:
            print(f"  TensorBoard: {output_dir / 'tb'}")
        print()

        train_loader, val_loader, (n_train, n_val) = build_gt_dataloaders(
            annotations_path=str(annotations_path),
            data_dir=args.data_dir,
            input_size=input_size,
            batch_size=batch_size,
            val_fraction=args.val_split,
            num_workers=args.num_workers,
            cameras=args.cameras,
            seed=args.seed,
            depth_dir=args.depth_dir,
            kfold_idx=args.kfold_idx,
            kfold_k=args.kfold_k,
        )
    else:
        # ── Standard pseudo-label training ──
        if args.ensemble_dir:
            ensemble_dir = Path(args.ensemble_dir)
        else:
            prefix = "weighted" if args.ensemble_weighted else "majority"
            ensemble_dir = _SEG_OUTPUT / "ensemble_output" / f"{prefix}_{args.ensemble_threshold}"

        if not ensemble_dir.exists():
            print(f"ERROR: Ensemble directory not found: {ensemble_dir}")
            print("Run ensemble_voting.py first.")
            sys.exit(1)

        ens_tag = ("weighted" if args.ensemble_weighted else "majority") + f"_{args.ensemble_threshold}"
        run_name = f"{args.model_variant}_{ens_tag}_{ts}"
        output_dir = Path(args.output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("STUDENT SEGMENTATION NETWORK — TRAINING")
        print("=" * 60)
        print(f"  Model:       {args.model_variant}")
        print(f"  Ensemble:    {ensemble_dir}")
        print(f"  Epochs:      {args.epochs}")
        print(f"  Batch size:  {batch_size}")
        print(f"  Input size:  {input_size}x{input_size}")
        print(f"  LR:          {lr}")
        print(f"  Depth:       {args.depth_dir or 'disabled (RGB only)'}")
        print(f"  Precision:   {args.precision}")
        print(f"  Output:      {output_dir}")
        if HAS_TENSORBOARD:
            print(f"  TensorBoard: {output_dir / 'tb'}")
            print(f"  Monitor:     tensorboard --logdir={output_dir / 'tb'}")
            print(f"               ssh -L 6006:localhost:6006 <node>")
        else:
            print("  TensorBoard: NOT available (pip install tensorboard) — using CSV only")
        print()

        train_loader, val_loader, (n_train, n_val) = build_dataloaders(
            ensemble_dir=ensemble_dir,
            data_dir=args.data_dir,
            input_size=input_size,
            batch_size=batch_size,
            val_fraction=args.val_split,
            num_workers=args.num_workers,
            cameras=args.cameras,
            seed=args.seed,
            depth_dir=args.depth_dir,
        )

    # Optional sample limit for smoke tests
    if args.sample > 0:
        from torch.utils.data import Subset
        n_train_sub = min(args.sample, len(train_loader.dataset))
        n_val_sub = min(args.sample // 5, len(val_loader.dataset))
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(n_train_sub)),
            batch_size=batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(n_val_sub)),
            batch_size=batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        print(f"  [smoke test] Using {n_train_sub} train / {n_val_sub} val samples")

    # Model — load from checkpoint if fine-tuning
    if args.finetune and args.checkpoint:
        print(f"  Loading pretrained weights from: {args.checkpoint}")
        model = StudentSegmentationModule.load_from_checkpoint(
            args.checkpoint,
            model_variant=args.model_variant,
            learning_rate=lr,
            weight_decay=args.weight_decay,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            t_max=args.epochs,
            log_images=log_images,
            unfreeze=args.unfreeze,
        )
    else:
        model = StudentSegmentationModule(
            model_variant=args.model_variant,
            learning_rate=lr,
            weight_decay=args.weight_decay,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            t_max=args.epochs,
            log_images=log_images,
        )

    # Count trainable params
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_trainable:,} / {n_total:,} total")
    print()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch:02d}-iou{val_iou:.3f}",
            monitor="val_iou",
            mode="max",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val_iou",
            patience=8,
            mode="max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    loggers = [CSVLogger(str(output_dir), name="csv")]
    if HAS_TENSORBOARD:
        loggers.insert(0, TensorBoardLogger(str(output_dir), name="tb", version=""))

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=20,
        deterministic=False,  # faster with frozen backbone + AMP
    )

    trainer.fit(model, train_loader, val_loader)

    # Save final checkpoint
    final_ckpt = output_dir / "final.ckpt"
    trainer.save_checkpoint(str(final_ckpt))

    best_iou = trainer.callback_metrics.get("val_iou", "N/A")
    best_f1 = trainer.callback_metrics.get("val_f1", "N/A")
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best val IoU: {best_iou}")
    print(f"  Best val F1:  {best_f1}")
    print(f"  Checkpoints:  {output_dir}")
    print(f"  Final model:  {final_ckpt}")
    if HAS_TENSORBOARD:
        print(f"  TensorBoard:  tensorboard --logdir={output_dir / 'tb'}")


# Late import used only for --sample mode
from torch.utils.data import DataLoader  # noqa: E402

if __name__ == "__main__":
    main()
