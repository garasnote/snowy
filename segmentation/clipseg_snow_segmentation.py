#!/usr/bin/env python3
"""
CLIPSeg text-prompted snow segmentation.

Uses CLIPSeg (CLIP + decoder) for dense per-pixel prediction with
text prompt "snow". Produces a heatmap thresholded into a binary mask.

Lightweight model (~150M params), very different feature space from
SAM/YOLO/Florence — good for ensemble diversity.

Requires: transformers (CLIPSeg is built-in, no extra deps).

Usage:
    # All cameras
    python clipseg_snow_segmentation.py \
        --data-dir /shared/home/juan.osorio/master/data

    # Specific cameras, debug overlays
    python clipseg_snow_segmentation.py \
        --data-dir /shared/home/juan.osorio/master/data \
        --cameras CF00001 CM00009 --debug

    # Custom threshold + prompt
    python clipseg_snow_segmentation.py \
        --data-dir /shared/home/juan.osorio/master/data \
        --threshold 0.4 --text-prompt "snow on the ground"
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ── Model loading ────────────────────────────────────────────────────────────

def load_clipseg(device="cuda"):
    """Load CLIPSeg model and processor."""
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    model_id = "CIDAS/clipseg-rd64-refined"
    print(f"Loading {model_id}...")
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(
        model_id
    ).eval().to(device)
    print(f"  CLIPSeg loaded on {device}")
    return model, processor


# ── Segmentation ─────────────────────────────────────────────────────────────

def segment_snow(image, model, processor, text_prompt, device="cuda",
                 threshold=0.5):
    """Run CLIPSeg and threshold logits into binary mask.

    CLIPSeg outputs a single-channel logit map at 352x352 (fixed).
    We sigmoid → threshold → resize to original image dimensions.
    """
    inputs = processor(
        text=[text_prompt], images=[image],
        return_tensors="pt", padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits: (1, 352, 352) — raw logits
    logits = outputs.logits.squeeze(0)
    probs = torch.sigmoid(logits).cpu().numpy()

    # Resize probability map to original image size
    h, w = image.size[1], image.size[0]
    probs_resized = np.array(
        Image.fromarray(probs).resize((w, h), Image.BILINEAR)
    )

    binary_mask = (probs_resized >= threshold).astype(np.uint8)
    return binary_mask, probs_resized


# ── Processing ───────────────────────────────────────────────────────────────

def process_image(image_path, model, processor, text_prompt, output_dir,
                  device="cuda", threshold=0.5, debug=False):
    """Process a single image. Returns result dict or None."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    snow_mask, probs = segment_snow(
        image, model, processor, text_prompt,
        device=device, threshold=threshold,
    )

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        visualize(np.array(image), snow_mask, probs,
                  cam_out / f"{stem}_overview.png",
                  title=f"{camera} — {stem}")

    snow_pct = snow_mask.sum() / snow_mask.size * 100

    return {
        "image": str(image_path),
        "camera": camera,
        "filename": Path(image_path).name,
        "snow_pct": round(snow_pct, 2),
    }


def visualize(image, mask, probs, output_path, title=""):
    """Save overview figure with heatmap and mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(probs, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("CLIPSeg Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    snow_pct = mask.sum() / mask.size * 100
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Snow Mask ({snow_pct:.1f}%)")
    axes[2].axis("off")

    overlay = image.copy().astype(np.float32)
    mask_bool = mask > 0
    overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 150, 255]) * 0.5
    overlay[~mask_bool] = overlay[~mask_bool] * 0.5 + np.array([100, 100, 100]) * 0.5
    axes[3].imshow(overlay.clip(0, 255).astype(np.uint8))
    axes[3].set_title("Overlay (blue=snow)")
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CLIPSeg text-prompted snow segmentation"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory with camera folders")
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific cameras (default: all)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent
                                    / "preprocessing" / "segmentation"
                                    / "clipseg_snow_output"),
                        help="Output directory")
    parser.add_argument("--text-prompt", type=str, default="snow",
                        help="Text prompt for segmentation (default: snow)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binary mask (default: 0.5)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Images to sample per camera (0=all)")
    parser.add_argument("--debug", action="store_true",
                        help="Save overview figures")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("CLIPSEG — SNOW SEGMENTATION (text-prompted dense prediction)")
    print("=" * 70)
    print(f"  Model:      CIDAS/clipseg-rd64-refined")
    print(f"  Prompt:     {args.text_prompt}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")

    model, processor = load_clipseg(device=device)

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
                img_path, model, processor, args.text_prompt,
                output_dir, device=device, threshold=args.threshold,
                debug=args.debug,
            )
            if result:
                results.append(result)

    if results:
        summary_path = output_dir / "clipseg_snow_summary.csv"
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
