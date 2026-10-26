#!/usr/bin/env python3
"""
Student snow segmentation — knowledge distillation from snow ensemble.

Reuses the student_ground training pipeline but with snow-specific
dataset discovery (ensemble_snow_output with *_snow.png masks).

Usage:
    # Pseudo-label training from snow ensemble
    python train.py --model-variant mobilenet --ensemble-threshold 2

    # Smoke test
    python train.py --model-variant mobilenet --epochs 1 --sample 200

    # Fine-tune on GT snow annotations
    python train.py --finetune --model-variant mobilenet --lr 1e-4
"""

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import CSVLogger
try:
    import tensorboard  # noqa: F401
    from lightning.pytorch.loggers import TensorBoardLogger
    HAS_TENSORBOARD = True
except (ImportError, ModuleNotFoundError):
    TensorBoardLogger = None
    HAS_TENSORBOARD = False

_HERE = Path(__file__).resolve().parent
_MODELS_DIR = _HERE.parent
_SEG_OUTPUT = _HERE.parent.parent / "preprocessing" / "segmentation"

if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

# Reuse everything from student_ground
from student_ground.configs import get_config, IMAGENET_MEAN, IMAGENET_STD
from student_ground.dataset import (
    build_train_transform, build_val_transform,
    SegmentationDataset, train_val_split,
)
from student_ground.train import StudentSegmentationModule
from torch.utils.data import DataLoader


# ── Snow-specific sample discovery ──────────────────────────────────────────

def discover_snow_samples(ensemble_dir, data_dir, cameras=None, depth_dir=None):
    """Scan snow ensemble output and match against original images.

    Same as student_ground's discover_samples but looks for
    *_snow.png (binary) and *_confidence.png (soft) masks.
    """
    ensemble_dir = Path(ensemble_dir)
    data_dir = Path(data_dir)
    if depth_dir is not None:
        depth_dir = Path(depth_dir)

    if cameras:
        cam_dirs = [ensemble_dir / c for c in cameras if (ensemble_dir / c).exists()]
    else:
        cam_dirs = sorted([
            d for d in ensemble_dir.iterdir()
            if d.is_dir() and not d.name.endswith(".csv")
        ])

    samples = []
    missing_images = 0

    for cam_dir in cam_dirs:
        camera = cam_dir.name
        for conf_path in sorted(cam_dir.glob("*_confidence.png")):
            stem = conf_path.stem.replace("_confidence", "")
            # Snow masks use _snow.png suffix (not _ground.png)
            mask_path = cam_dir / f"{stem}_snow.png"
            img_path = data_dir / camera / f"{stem}.jpg"

            if not img_path.exists():
                missing_images += 1
                continue
            if not mask_path.exists():
                continue

            depth_path = None
            if depth_dir is not None:
                dp = depth_dir / camera / f"{stem}_depth.png"
                if dp.exists():
                    depth_path = dp

            samples.append((img_path, conf_path, mask_path, depth_path))

    if missing_images:
        print(f"  [snow_dataset] Warning: {missing_images} images not found in {data_dir}")

    return samples


_REPO_ROOT = Path(__file__).resolve().parents[2]
# Ground denominator for the PSG fraction: the segmented ground, clipped to the camera's
# trusted region when one exists. Both live under preprocessing/segmentation/.
GROUND_DIR = _REPO_ROOT / "preprocessing" / "segmentation" / "student_ground_output" / "mobilenet_ft"
VALID_REGION_DIR = _REPO_ROOT / "preprocessing" / "segmentation" / "camera_valid_regions"


def snow_pct_on_ground(mask_arr, mask_path):
    """Snow percentage on the ground the camera can see, not on the whole frame.

    Falls back to the frame-wide fraction when no ground mask exists for that frame,
    so cameras that never went through the ground pass keep working.
    """
    from PIL import Image
    snow = mask_arr > 127
    try:
        camera = Path(mask_path).parent.name
        stem = Path(mask_path).name.replace("_snow.png", "").replace(".png", "")
        gpath = GROUND_DIR / camera / f"{stem}_ground.png"
        if not gpath.exists():
            return float(snow.sum()) / snow.size * 100
        g = np.array(Image.open(gpath).convert("L"))
        if g.shape[:2] != snow.shape:
            g = np.array(Image.fromarray(g).resize((snow.shape[1], snow.shape[0]), Image.NEAREST))
        region = g > 127
        vr = VALID_REGION_DIR / f"{camera}.png"
        if vr.exists():
            v = np.array(Image.open(vr).convert("L"))
            if v.shape[:2] != snow.shape:
                v = np.array(Image.fromarray(v).resize((snow.shape[1], snow.shape[0]), Image.NEAREST))
            region = region & (v > 127)
        n = int(region.sum())
        if n == 0:
            return float(snow.sum()) / snow.size * 100
        return float((snow & region).sum()) / n * 100
    except Exception:
        return float(snow.sum()) / snow.size * 100


def filter_by_snow_coverage(samples, min_snow_pct=1.0, seed=42):
    """Split samples into snow/no-snow based on mask pixel coverage.

    Returns (snow_samples, nosnow_samples).
    """
    from PIL import Image

    snow, nosnow = [], []
    for s in samples:
        mask_path = s[2]  # (img, conf, mask, depth)
        try:
            m = np.array(Image.open(mask_path).convert("L"))
            pct = snow_pct_on_ground(m, mask_path)
        except Exception:
            continue
        if pct >= min_snow_pct:
            snow.append(s)
        else:
            nosnow.append(s)
    return snow, nosnow


def build_snow_dataloaders(
    ensemble_dir, data_dir, input_size=518, batch_size=16,
    val_fraction=0.2, num_workers=8, cameras=None, seed=42, depth_dir=None,
    min_snow_pct=0.0,
):
    """Build train/val DataLoaders from snow ensemble pseudo-labels."""
    all_samples = discover_snow_samples(
        ensemble_dir, data_dir, cameras=cameras, depth_dir=depth_dir
    )
    if not all_samples:
        raise ValueError(
            f"No snow samples found!\n"
            f"  ensemble_dir: {ensemble_dir}\n"
            f"  data_dir:     {data_dir}\n"
            "Check that ensemble_snow_output exists and images are present."
        )

    if min_snow_pct > 0:
        snow_samples, nosnow_samples = filter_by_snow_coverage(
            all_samples, min_snow_pct=min_snow_pct, seed=seed
        )
        rng = random.Random(seed)
        n_neg = len(snow_samples)
        if len(nosnow_samples) > n_neg:
            nosnow_samples = rng.sample(nosnow_samples, n_neg)
        all_samples = snow_samples + nosnow_samples
        rng.shuffle(all_samples)
        print(f"  Filtered: {len(snow_samples)} snow + {len(nosnow_samples)} no-snow "
              f"(min {min_snow_pct}% coverage)")

    train_samples, val_samples = train_val_split(
        all_samples, val_fraction=val_fraction, seed=seed
    )

    print(f"  Snow dataset: {len(all_samples)} total, "
          f"{len(train_samples)} train, {len(val_samples)} val")

    train_tf = build_train_transform(input_size)
    val_tf = build_val_transform(input_size)

    train_ds = SegmentationDataset(train_samples, transform=train_tf, input_size=input_size)
    val_ds = SegmentationDataset(val_samples, transform=val_tf, input_size=input_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, (len(train_samples), len(val_samples))


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train student snow segmentation on ensemble pseudo-labels"
    )

    parser.add_argument("--data-dir", type=str,
                        default=str(_HERE.parent.parent / "data"),
                        help="Root data directory with camera subdirs")
    parser.add_argument("--ensemble-dir", type=str, default=None,
                        help="Path to specific snow ensemble dir")
    parser.add_argument("--ensemble-threshold", type=int, default=2,
                        help="Which majority_N snow ensemble to use (default: 2)")
    parser.add_argument("--gt", action="store_true",
                        help="Train on GT annotations instead of pseudo-labels")
    parser.add_argument("--ann-dir", type=str,
                        default=str(_HERE.parent.parent / "data" / "annotations"),
                        help="GT annotation dir (Label Studio JSON exports)")
    parser.add_argument("--fallback-img-dirs", nargs="*", default=None,
                        help="Extra dirs to search for images (e.g. night_images/)")
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE / "checkpoints"),
                        help="Directory for checkpoints")

    parser.add_argument("--model-variant", type=str, default="mobilenet",
                        help="Model variant (mobilenet, dinov2_base, etc.)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth-dir", type=str, default=None)
    parser.add_argument("--no-log-images", action="store_true")
    parser.add_argument("--focal", action="store_true",
                        help="Use Focal Loss instead of BCE")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--unfreeze", action="store_true",
                        help="Unfreeze encoder for full fine-tuning")
    parser.add_argument("--min-snow-pct", type=float, default=0.0,
                        help="Filter: only keep images with >=N%% snow pixels, "
                             "then balance with equal no-snow samples (0=disabled)")
    parser.add_argument("--pretrain-ckpt", type=str, default=None,
                        help="Initialize model weights from a pseudo-label checkpoint before GT finetuning")
    parser.add_argument("--kfold-idx", type=int, default=-1,
                        help="K-fold index (0-4 for 5-fold CV, -1 to disable)")
    parser.add_argument("--kfold-k", type=int, default=5,
                        help="Total number of folds")

    args = parser.parse_args()
    L.seed_everything(args.seed)

    cfg = get_config(args.model_variant)
    batch_size = args.batch_size or cfg["batch_size"]
    input_size = cfg["input_size"]
    lr = args.lr or cfg["lr"]
    log_images = not args.no_log_images

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.gt:
        # GT annotation mode
        from student_snow.gt_dataset import load_gt_snow_samples
        gt_mask_dir = _SEG_OUTPUT / "gt_snow_masks"
        all_samples = load_gt_snow_samples(
            ann_dir=args.ann_dir,
            img_dir=args.data_dir,
            mask_out_dir=gt_mask_dir,
            depth_dir=args.depth_dir,
            fallback_img_dirs=args.fallback_img_dirs,
        )
        if not all_samples:
            print("ERROR: No GT snow samples found!")
            sys.exit(1)

        if args.kfold_idx >= 0:
            import random as _random
            _rng = _random.Random(args.seed)
            _by_cam = {}
            for s in all_samples:
                _cam = s[0].parent.name
                _by_cam.setdefault(_cam, []).append(s)
            _folds = [[] for _ in range(args.kfold_k)]
            for _cam, _csamples in sorted(_by_cam.items()):
                _shuffled = list(_csamples)
                _rng.shuffle(_shuffled)
                _fold_size = len(_shuffled) // args.kfold_k
                for _fi in range(args.kfold_k):
                    _start = _fi * _fold_size
                    _end = _start + _fold_size if _fi < args.kfold_k - 1 else len(_shuffled)
                    _folds[_fi].extend(_shuffled[_start:_end])
            val_samples = _folds[args.kfold_idx]
            train_samples = [s for _i in range(args.kfold_k) if _i != args.kfold_idx for s in _folds[_i]]
            run_name = f"snow_gt_{args.model_variant}_kfold_{args.kfold_idx}_{ts}"
        else:
            train_samples, val_samples = train_val_split(
                all_samples, val_fraction=args.val_split, seed=args.seed
            )
        print(f"  GT dataset: {len(all_samples)} total, "
              f"{len(train_samples)} train, {len(val_samples)} val")

        train_tf = build_train_transform(input_size)
        val_tf = build_val_transform(input_size)
        train_ds = SegmentationDataset(train_samples, transform=train_tf, input_size=input_size)
        val_ds = SegmentationDataset(val_samples, transform=val_tf, input_size=input_size)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        if args.kfold_idx < 0:
            run_name = f"snow_gt_{args.model_variant}_{ts}"
    else:
        # Pseudo-label mode
        if args.ensemble_dir:
            ensemble_dir = Path(args.ensemble_dir)
        else:
            ensemble_dir = _SEG_OUTPUT / "ensemble_snow_output" / f"majority_{args.ensemble_threshold}"
            if not ensemble_dir.exists():
                # Handle legacy float-suffixed dirs (majority_3.0)
                alt = _SEG_OUTPUT / "ensemble_snow_output" / f"majority_{args.ensemble_threshold}.0"
                if alt.exists():
                    ensemble_dir = alt

        if not ensemble_dir.exists():
            print(f"ERROR: Snow ensemble directory not found: {ensemble_dir}")
            print("Run `python segmentation/snow_utils.py ensemble` first.")
            sys.exit(1)
        run_name = f"snow_{args.model_variant}_maj{args.ensemble_threshold}_{ts}"

        train_loader, val_loader, (n_train, n_val) = build_snow_dataloaders(
            ensemble_dir=ensemble_dir,
            data_dir=args.data_dir,
            input_size=input_size,
            batch_size=batch_size,
            val_fraction=args.val_split,
            num_workers=args.num_workers,
            cameras=args.cameras,
            seed=args.seed,
            depth_dir=args.depth_dir,
            min_snow_pct=args.min_snow_pct,
        )

    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    data_source = "GT annotations" if args.gt else str(ensemble_dir)
    print("=" * 60)
    print("STUDENT SNOW SEGMENTATION — TRAINING")
    print("=" * 60)
    print(f"  Model:       {args.model_variant}")
    print(f"  Data source: {data_source}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Input size:  {input_size}x{input_size}")
    print(f"  LR:          {lr}")
    print(f"  Precision:   {args.precision}")
    print(f"  Output:      {output_dir}")
    print()

    # Optional sample limit
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

    model = StudentSegmentationModule(
        model_variant=args.model_variant,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        t_max=args.epochs,
        log_images=log_images,
        use_focal=args.focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    if args.pretrain_ckpt:
        import torch as _torch
        _ckpt = _torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=False)
        _sd = {k.replace("model.", ""): v for k, v in _ckpt["state_dict"].items() if k.startswith("model.")}
        _model_sd = model.model.state_dict()
        _filtered = {k: v for k, v in _sd.items() if k in _model_sd and v.shape == _model_sd[k].shape}
        model.model.load_state_dict(_filtered, strict=False)
        print(f"  [PRETRAIN] Loaded {len(_filtered)}/{len(_model_sd)} weights from {args.pretrain_ckpt}")

    if args.unfreeze:
        inner = model.model
        if hasattr(inner, "lora_layers"):
            # ViT+LoRA: unfreeze LoRA adapters + decoder (not full encoder)
            for p in inner.trainable_parameters():
                p.requires_grad = True
            print("  [UNFROZEN] LoRA adapters + decoder trainable")
        else:
            for p in model.parameters():
                p.requires_grad = True
            if hasattr(inner, "freeze_encoder"):
                inner.freeze_encoder = False
                if hasattr(inner, "encoder"):
                    inner.encoder.train()
            print("  [UNFROZEN] All parameters trainable (full fine-tune)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_trainable:,} / {n_total:,} total\n")

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch:02d}-iou{val_iou:.3f}",
            monitor="val_iou",
            mode="max",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val_iou", patience=20, mode="max"),
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
        deterministic=False,
    )

    trainer.fit(model, train_loader, val_loader)

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


if __name__ == "__main__":
    main()