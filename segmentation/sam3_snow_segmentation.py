#!/usr/bin/env python3
"""
SAM3 text-prompted snow segmentation.

Uses SAMWrapper with text prompt "snow" to segment snow cover in webcam images.
Optionally masks out sky regions first.

Usage:
    # All cameras
    python sam3_snow_segmentation.py \
        --data-dir data/balanced_snow \
        --output-dir preprocessing/segmentation/sam3_snow_output

    # Specific cameras, with debug overlays
    python sam3_snow_segmentation.py \
        --data-dir data/balanced_snow \
        --cameras CF00001 CM00006 --debug

    # With ground mask filtering (snow only on ground)
    python sam3_snow_segmentation.py \
        --data-dir data/balanced_snow \
        --ground-mask-dir preprocessing/segmentation/student_ground_output/mobilenet_ft

    # Custom threshold + sample
    python sam3_snow_segmentation.py \
        --data-dir data/balanced_snow \
        --threshold 0.3 --sample 50
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sam_wrapper import SAMWrapper


# ── Processing ───────────────────────────────────────────────────────────────

def process_image(image_path, sam, threshold, output_dir,
                  ground_mask_dir=None, mask_sky=True, debug=False):
    """Process a single image. Returns result dict or None."""
    try:
        img = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    # Optionally mask sky first
    sky_mask = None
    if mask_sky:
        sky_mask = sam.segment_sky(img)

    # Segment snow (already handles sky exclusion if sky_mask provided)
    sam.processor.confidence_threshold = threshold
    snow_mask = sam.segment_snow(img, sky_mask=sky_mask)

    # Optionally restrict to ground region
    if ground_mask_dir is not None:
        ground_path = ground_mask_dir / camera / f"{stem}_ground.png"
        if ground_path.exists():
            ground = np.array(Image.open(ground_path).convert("L"))
            if ground.shape != snow_mask.shape:
                ground = np.array(
                    Image.fromarray(ground).resize(
                        (snow_mask.shape[1], snow_mask.shape[0]), Image.NEAREST
                    )
                )
            snow_mask = (snow_mask & (ground > 127)).astype(np.uint8)

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    # Save snow mask
    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        visualize(img, snow_mask, sky_mask, cam_out / f"{stem}_overview.png",
                  title=f"{camera} — {stem}")

    snow_pct = (snow_mask > 0).sum() / snow_mask.size * 100

    return {
        "image": str(image_path),
        "camera": camera,
        "filename": Path(image_path).name,
        "snow_pct": round(snow_pct, 2),
    }


def visualize(image, snow_mask, sky_mask, output_path, title=""):
    """Save overview figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 4 if sky_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    col = 1
    if sky_mask is not None:
        axes[col].imshow(sky_mask, cmap="gray")
        axes[col].set_title(f"Sky Mask ({sky_mask.sum() / sky_mask.size * 100:.1f}%)")
        axes[col].axis("off")
        col += 1

    snow_pct = snow_mask.sum() / snow_mask.size * 100
    axes[col].imshow(snow_mask, cmap="gray")
    axes[col].set_title(f"Snow Mask ({snow_pct:.1f}%)")
    axes[col].axis("off")
    col += 1

    overlay = image.copy().astype(np.float32)
    mask_bool = snow_mask > 0
    overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 150, 255]) * 0.5
    overlay[~mask_bool] = overlay[~mask_bool] * 0.5 + np.array([100, 100, 100]) * 0.5
    axes[col].imshow(overlay.clip(0, 255).astype(np.uint8))
    axes[col].set_title("Overlay (blue=snow)")
    axes[col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 text-prompted snow segmentation"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory with camera folders")
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific cameras (default: all)")
    parser.add_argument("--output-dir", type=str,
                        default="preprocessing/segmentation/sam3_snow_output",
                        help="Output directory")
    parser.add_argument("--ground-mask-dir", type=str, default=None,
                        help="Ground mask dir to restrict snow to ground region")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="SAM3 confidence threshold (default: 0.4)")
    parser.add_argument("--no-sky-mask", action="store_true",
                        help="Disable sky masking")
    parser.add_argument("--sample", type=int, default=0,
                        help="Images to sample per camera (0=all)")
    parser.add_argument("--debug", action="store_true",
                        help="Save overview figures")

    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    ground_mask_dir = Path(args.ground_mask_dir) if args.ground_mask_dir else None
    mask_sky = not args.no_sky_mask

    print("=" * 70)
    print("SAM3 — SNOW SEGMENTATION (text-prompted)")
    print("=" * 70)
    print(f"  Threshold:    {args.threshold}")
    print(f"  Sky masking:  {mask_sky}")
    print(f"  Ground mask:  {ground_mask_dir or 'disabled'}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_dir}")

    sam = SAMWrapper(device=device)

    cameras = args.cameras
    if not cameras:
        cameras = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)
        ])

    print(f"\nCameras: {cameras}")

    results = []
    for cam in cameras:
        cam_dir = data_dir / cam
        if not cam_dir.exists():
            print(f"  WARNING: {cam_dir} does not exist, skipping")
            continue

        jpgs = sorted(cam_dir.glob("*.jpg"))
        if args.sample > 0 and len(jpgs) > args.sample:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(jpgs), size=args.sample, replace=False)
            jpgs = [jpgs[i] for i in sorted(indices)]

        print(f"\n  {cam}: processing {len(jpgs)} images")

        for img_path in tqdm(jpgs, desc=f"  {cam}", leave=True):
            result = process_image(
                img_path, sam, args.threshold, output_dir,
                ground_mask_dir=ground_mask_dir, mask_sky=mask_sky,
                debug=args.debug
            )
            if result:
                results.append(result)

    # Save summary CSV
    if results:
        summary_path = output_dir / "sam3_snow_summary.csv"
        fieldnames = ["image", "camera", "filename", "snow_pct"]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary: {summary_path}")

    print(f"\nProcessed {len(results)} images.")
    print("Done!")


if __name__ == "__main__":
    main()
