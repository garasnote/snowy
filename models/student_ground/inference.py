#!/usr/bin/env python3
"""
Student segmentation inference — run trained model on new images.

Loads a checkpoint and runs the student model on a directory of images,
producing binary ground masks and optionally probability maps.

Follows the same output structure as other segmentation methods:
    output_dir/{camera}/{stem}_ground.png       (binary mask, 0/255)
    output_dir/{camera}/{stem}_probability.png  (probability 0-255, if --save-prob)

Usage:
    # Single camera
    python inference.py --checkpoint checkpoints/dinov2_base.../best.ckpt
                        --data-dir /shared/home/juan.osorio/master/data
                        --cameras CM00006

    # All cameras
    python inference.py --checkpoint checkpoints/dinov2_base.../best.ckpt
                        --data-dir /shared/home/juan.osorio/master/data

    # Save probability maps too (useful for further ensembling)
    python inference.py --checkpoint ... --save-prob

    # Adjust prediction threshold
    python inference.py --checkpoint ... --threshold 0.4
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from student_ground.configs import MODEL_REGISTRY, get_config, IMAGENET_MEAN, IMAGENET_STD
from student_ground.dataset import load_depth_luminance
from student_ground.models import build_model

# Pixels to zero out at bottom of output mask (camera watermarks, timestamps, etc.)
CAMERA_MASK_BOTTOM = {
    "CM00009": 130,
}

# Static valid-region masks for cameras with ground misplacement
VALID_REGION_DIR = _HERE.parent.parent / "preprocessing" / "segmentation" / "camera_valid_regions"
_valid_region_cache = {}

def load_valid_region(camera, h, w):
    key = (camera, h, w)
    if key in _valid_region_cache:
        return _valid_region_cache[key]
    vr_path = VALID_REGION_DIR / f"{camera}.png"
    if not vr_path.exists():
        return None
    vr = np.array(Image.open(vr_path).convert("L").resize((w, h), Image.NEAREST))
    _valid_region_cache[key] = vr
    return vr


# ── Single-image inference ────────────────────────────────────────────────────

def load_and_preprocess(img_path, input_size, depth_path=None):
    """Load image, apply ImageNet normalization, optionally append depth channel."""
    img = np.array(Image.open(img_path).convert("RGB"))

    # Resize
    pil = Image.fromarray(img).resize((input_size, input_size), Image.BILINEAR)
    img = np.array(pil).astype(np.float32) / 255.0

    # Normalize
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std

    # (H, W, C) → (1, C, H, W)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    # Append depth as 4th channel if available
    if depth_path is not None:
        depth = load_depth_luminance(depth_path)
        depth_resized = np.array(
            Image.fromarray((depth * 255).astype(np.uint8)).resize(
                (input_size, input_size), Image.BILINEAR)
        ).astype(np.float32) / 255.0
        depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        tensor = torch.cat([tensor, depth_tensor], dim=1)  # (1, 4, H, W)

    return tensor


@torch.no_grad()
def run_model_batch(model, img_paths, input_size, device, threshold=0.5, depth_dir=None):
    """Run model on a list of image paths, return (prob, binary) arrays or None for bad files."""
    results = []
    for img_path in img_paths:
        try:
            depth_path = None
            if depth_dir is not None:
                camera = img_path.parent.name
                dp = Path(depth_dir) / camera / f"{img_path.stem}_depth.png"
                if dp.exists():
                    depth_path = dp
            tensor = load_and_preprocess(img_path, input_size, depth_path=depth_path).to(device)
            logits = model(tensor)                          # (1, 1, H, W)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)
            binary = (prob >= threshold).astype(np.uint8) * 255
            results.append((prob, binary))
        except Exception as e:
            print(f"  SKIP {img_path.name}: {e}", flush=True)
            results.append(None)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Student segmentation inference (GPU accelerated)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt file from training")
    parser.add_argument("--data-dir", type=str,
                        default="/shared/home/juan.osorio/master/data",
                        help="Root data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: student_output/ next to data-dir)")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Specific cameras (default: all in data-dir)")
    parser.add_argument("--model-variant", type=str, default=None,
                        help="Override model variant (inferred from checkpoint name by default)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binary mask (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Images per GPU batch (default: 8)")
    parser.add_argument("--save-prob", action="store_true",
                        help="Also save probability maps as _probability.png")
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only N images per camera (0 = all)")
    parser.add_argument("--depth-dir", type=str, default=None,
                        help="Path to depth maps dir (for RGBD models)")
    parser.add_argument("--usable-list", type=str, default=None,
                        help="Text file with usable stems (one per line). Skip images not listed.")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Infer output dir from checkpoint parent
        output_dir = ckpt_path.parent.parent.parent / "student_output" / ckpt_path.parent.name

    # Infer model variant from checkpoint path name if not specified
    variant = args.model_variant
    if variant is None:
        for v in MODEL_REGISTRY:
            if v in str(ckpt_path):
                variant = v
                break
        if variant is None:
            variant = "dinov2_base"
            print(f"  Warning: could not infer variant from checkpoint path, using {variant}")

    cfg = get_config(variant)
    input_size = cfg["input_size"]

    # Load model from checkpoint
    from student_ground.train import StudentSegmentationModule
    print(f"  Loading checkpoint: {ckpt_path}")
    module = StudentSegmentationModule.load_from_checkpoint(
        str(ckpt_path), model_variant=variant
    )
    model = module.model
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Model variant: {variant}")
    print(f"  Input size: {input_size}x{input_size}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Depth:     {args.depth_dir or 'disabled (RGB only)'}")

    # Discover cameras
    if args.cameras:
        cameras = args.cameras
    else:
        import re
        cameras = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)
        ])

    usable_stems = None
    if args.usable_list:
        with open(args.usable_list) as f:
            usable_stems = set(line.strip() for line in f if line.strip())
        print(f"  QA filter: {len(usable_stems)} usable stems loaded")

    print(f"  Cameras: {cameras}")
    print(f"  Output: {output_dir}")
    print()

    total_processed = 0

    for camera in cameras:
        cam_dir = data_dir / camera
        if not cam_dir.exists():
            print(f"  Skipping {camera}: not found in {data_dir}")
            continue

        img_paths = sorted(cam_dir.glob("*.jpg"))
        if usable_stems is not None:
            img_paths = [p for p in img_paths if p.stem in usable_stems]
        if args.sample > 0:
            img_paths = img_paths[:args.sample]

        out_cam = output_dir / camera
        out_cam.mkdir(parents=True, exist_ok=True)

        # Process in batches
        batches = [
            img_paths[i:i + args.batch_size]
            for i in range(0, len(img_paths), args.batch_size)
        ]

        cam_count = 0
        for batch_paths in tqdm(batches, desc=camera, unit="batch"):
            results = run_model_batch(
                model, batch_paths, input_size, device, threshold=args.threshold,
                depth_dir=args.depth_dir,
            )
            for img_path, result in zip(batch_paths, results):
                if result is None:
                    continue  # corrupted/unreadable image — already printed warning
                prob, binary = result
                stem = img_path.stem

                # Resize outputs to original image dimensions so masks are
                # compatible with all other segmentation methods in the dashboard.
                orig_size = Image.open(img_path).size  # (W, H)
                mask_arr = np.array(Image.fromarray(binary).resize(orig_size, Image.NEAREST))
                bottom_px = CAMERA_MASK_BOTTOM.get(camera, 0)
                if bottom_px > 0:
                    mask_arr[-bottom_px:, :] = 0
                    prob[-int(bottom_px * prob.shape[0] / mask_arr.shape[0]):, :] = 0
                vr = load_valid_region(camera, mask_arr.shape[0], mask_arr.shape[1])
                if vr is not None:
                    mask_arr[vr <= 127] = 0
                    prob_vr = load_valid_region(camera, prob.shape[0], prob.shape[1])
                    if prob_vr is not None:
                        prob[prob_vr <= 127] = 0
                Image.fromarray(mask_arr).save(out_cam / f"{stem}_ground.png")
                if args.save_prob:
                    prob_uint8 = (prob * 255).astype(np.uint8)
                    prob_img = Image.fromarray(prob_uint8).resize(orig_size, Image.BILINEAR)
                    prob_img.save(out_cam / f"{stem}_probability.png")
                cam_count += 1

        print(f"  {camera}: {cam_count} images → {out_cam}")
        total_processed += cam_count

    print()
    print(f"Done! Processed {total_processed} images total.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
