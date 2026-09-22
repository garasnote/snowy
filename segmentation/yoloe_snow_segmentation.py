#!/usr/bin/env python3
"""
YOLOe-26 open-vocabulary snow segmentation.

Uses YOLOe-26 with text prompt "snow" for instance segmentation,
then merges all detected snow instances into a single binary mask.

Usage:
    # All cameras
    python yoloe_snow_segmentation.py \
        --data-dir /d/hpc/home/jo83525/master/data/balanced_snow

    # Specific cameras, debug overlays
    python yoloe_snow_segmentation.py \
        --data-dir /d/hpc/home/jo83525/master/data/balanced_snow \
        --cameras CF00001 CM00006 --debug

    # Custom model size (s/m/l)
    python yoloe_snow_segmentation.py \
        --data-dir /d/hpc/home/jo83525/master/data/balanced_snow \
        --model-size l
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ── Model loading ────────────────────────────────────────────────────────────

def load_yoloe(model_size="l", device="cuda"):
    """Load YOLOe-26 segmentation model and set snow classes."""
    from ultralytics import YOLO

    model_name = f"yoloe-26{model_size}-seg.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    model.to(device)

    # Set open-vocabulary classes for snow detection
    model.set_classes(["snow"])
    print(f"  YOLOe-26{model_size}-seg loaded, classes set to ['snow']")
    return model


# ── Segmentation ─────────────────────────────────────────────────────────────

def segment_snow(model, image_path, conf=0.25):
    """Run YOLOe segmentation and merge all snow masks into one binary mask.

    Returns (binary_mask, snow_pct) or (None, 0) on failure.
    """
    try:
        results = model.predict(str(image_path), conf=conf, verbose=False)
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None, 0.0

    if not results or len(results) == 0:
        return None, 0.0

    result = results[0]
    h, w = result.orig_shape

    # Merge all instance masks into a single binary mask
    combined = np.zeros((h, w), dtype=np.uint8)

    if result.masks is not None and len(result.masks) > 0:
        for mask_data in result.masks.data:
            # masks.data is (N, H, W) tensor — resize to original image size
            mask_np = mask_data.cpu().numpy()
            if mask_np.shape != (h, w):
                mask_resized = np.array(
                    Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                        (w, h), Image.NEAREST
                    )
                ) > 127
            else:
                mask_resized = mask_np > 0.5
            combined = np.maximum(combined, mask_resized.astype(np.uint8))

    snow_pct = combined.sum() / combined.size * 100
    return combined, snow_pct


# ── Processing ───────────────────────────────────────────────────────────────

def process_image(image_path, model, conf, output_dir, debug=False):
    """Process a single image. Returns result dict or None."""
    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    snow_mask, snow_pct = segment_snow(model, image_path, conf=conf)

    if snow_mask is None:
        return None

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    # Save snow mask
    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        visualize(image_path, snow_mask, cam_out / f"{stem}_overview.png",
                  title=f"{camera} — {stem}")

    return {
        "image": str(image_path),
        "camera": camera,
        "filename": Path(image_path).name,
        "snow_pct": round(snow_pct, 2),
    }


def visualize(image_path, mask, output_path, title=""):
    """Save overview figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.array(Image.open(image_path).convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    snow_pct = mask.sum() / mask.size * 100
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Snow Mask ({snow_pct:.1f}%)")
    axes[1].axis("off")

    overlay = image.copy().astype(np.float32)
    mask_bool = mask > 0
    overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 150, 255]) * 0.5
    overlay[~mask_bool] = overlay[~mask_bool] * 0.5 + np.array([100, 100, 100]) * 0.5
    axes[2].imshow(overlay.clip(0, 255).astype(np.uint8))
    axes[2].set_title("Overlay (blue=snow)")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLOe-26 open-vocabulary snow segmentation"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory with camera folders")
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific cameras (default: all)")
    parser.add_argument("--output-dir", type=str,
                        default="/d/hpc/home/jo83525/master/preprocessing/segmentation/yoloe_snow_output",
                        help="Output directory")
    parser.add_argument("--model-size", choices=["s", "m", "l"], default="l",
                        help="YOLOe-26 model size (default: l)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
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

    print("=" * 70)
    print("YOLOe-26 — SNOW SEGMENTATION (open-vocabulary)")
    print("=" * 70)
    print(f"  Model:      yoloe-26{args.model_size}-seg")
    print(f"  Confidence: {args.conf}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")

    model = load_yoloe(args.model_size, device=device)

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
                img_path, model, args.conf, output_dir, debug=args.debug
            )
            if result:
                results.append(result)

    # Save summary CSV
    if results:
        summary_path = output_dir / "yoloe_snow_summary.csv"
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
