#!/usr/bin/env python3
"""
Florence-2 referring expression segmentation for snow detection.

Uses Florence-2-large with the <REFERRING_EXPRESSION_SEGMENTATION> task
and text prompt "snow" to produce snow masks.

Usage:
    # All cameras
    python florence2_snow_segmentation.py \
        --data-dir /d/hpc/home/jo83525/master/data/balanced_snow

    # Specific cameras, debug overlays
    python florence2_snow_segmentation.py \
        --data-dir /d/hpc/home/jo83525/master/data/balanced_snow \
        --cameras CF00001 CM00006 --debug

    # Use base model (faster, less accurate)
    python florence2_snow_segmentation.py \
        --data-dir /d/hpc/home/jo83525/master/data/balanced_snow \
        --model-size base
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


# ── Model loading ────────────────────────────────────────────────────────────

def load_florence2(model_size="large", device="cuda"):
    """Load Florence-2 model and processor."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = f"microsoft/Florence-2-{model_size}"
    print(f"Loading {model_id}...")

    # Clear stale cached custom code from prior runs
    import shutil
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "microsoft"
    for d in cache_dir.glob("Florence*"):
        shutil.rmtree(d, ignore_errors=True)
        print(f"  Cleared stale cache: {d.name}")

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, trust_remote_code=True,
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True,
    )

    print(f"  Florence-2-{model_size} loaded on {device}")
    return model, processor


# ── Inference ────────────────────────────────────────────────────────────────

def run_florence2(model, processor, image, text_prompt, device="cuda"):
    """Run Florence-2 referring expression segmentation.

    Returns parsed answer dict with polygons.
    """
    task = "<REFERRING_EXPRESSION_SEGMENTATION>"
    prompt = task + text_prompt

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    inputs = processor(
        text=prompt, images=image, return_tensors="pt"
    ).to(device, torch_dtype)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height),
    )
    return parsed


def polygons_to_mask(parsed_answer, width, height):
    """Convert Florence-2 polygon output to binary mask.

    The parsed answer for REFERRING_EXPRESSION_SEGMENTATION contains
    polygons as list of [x1,y1,x2,y2,...] coordinate lists.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Extract polygons from the parsed output
    key = "<REFERRING_EXPRESSION_SEGMENTATION>"
    if key not in parsed_answer:
        return np.zeros((height, width), dtype=np.uint8)

    result = parsed_answer[key]

    # Result contains 'polygons' key with list of polygon coordinate lists
    polygons = result.get("polygons", [])

    for polygon in polygons:
        # Each polygon is a list of [x1, y1, x2, y2, ...] coordinates
        if isinstance(polygon, list) and len(polygon) > 0:
            # Handle nested list format: [[x1,y1,x2,y2,...]]
            coords = polygon[0] if isinstance(polygon[0], list) else polygon
            if len(coords) >= 6:  # Need at least 3 points
                # Convert flat list to list of tuples
                points = [(coords[i], coords[i + 1])
                          for i in range(0, len(coords) - 1, 2)]
                draw.polygon(points, fill=255)

    return (np.array(mask) > 127).astype(np.uint8)


# ── Processing ───────────────────────────────────────────────────────────────

def process_image(image_path, model, processor, text_prompt,
                  output_dir, device="cuda", debug=False):
    """Process a single image. Returns result dict or None."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    parsed = run_florence2(model, processor, image, text_prompt, device=device)
    snow_mask = polygons_to_mask(parsed, image.width, image.height)

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    # Save snow mask
    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        visualize(image, snow_mask, cam_out / f"{stem}_overview.png",
                  title=f"{camera} — {stem}")

    snow_pct = snow_mask.sum() / snow_mask.size * 100

    return {
        "image": str(image_path),
        "camera": camera,
        "filename": Path(image_path).name,
        "snow_pct": round(snow_pct, 2),
    }


def visualize(image, mask, output_path, title=""):
    """Save overview figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_arr = np.array(image)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_arr)
    axes[0].set_title("Original")
    axes[0].axis("off")

    snow_pct = mask.sum() / mask.size * 100
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Snow Mask ({snow_pct:.1f}%)")
    axes[1].axis("off")

    overlay = img_arr.copy().astype(np.float32)
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
        description="Florence-2 referring expression snow segmentation"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory with camera folders")
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific cameras (default: all)")
    parser.add_argument("--output-dir", type=str,
                        default="/d/hpc/home/jo83525/master/preprocessing/segmentation/florence2_snow_output",
                        help="Output directory")
    parser.add_argument("--model-size", choices=["base", "large"],
                        default="large",
                        help="Florence-2 model size (default: large)")
    parser.add_argument("--text-prompt", type=str, default="snow",
                        help="Text prompt for segmentation (default: snow)")
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
    print("FLORENCE-2 — SNOW SEGMENTATION (referring expression)")
    print("=" * 70)
    print(f"  Model:       Florence-2-{args.model_size}")
    print(f"  Text prompt: {args.text_prompt}")
    print(f"  Device:      {device}")
    print(f"  Output:      {output_dir}")

    model, processor = load_florence2(args.model_size, device=device)

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
                output_dir, device=device, debug=args.debug
            )
            if result:
                results.append(result)

    # Save summary CSV
    if results:
        summary_path = output_dir / "florence2_snow_summary.csv"
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
