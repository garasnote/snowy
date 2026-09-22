#!/usr/bin/env python3
"""
Grounded SAM2 snow segmentation.

Uses GroundingDINO for open-vocabulary detection ("snow") and SAM2
for mask refinement. Merges all detected snow instances into a single
binary mask per image.

Requires: grounding-dino-tiny, sam2-hiera-large (auto-downloaded from HF).

Usage:
    # All cameras
    python grounded_sam2_snow_segmentation.py \
        --data-dir data

    # Specific cameras, debug overlays
    python grounded_sam2_snow_segmentation.py \
        --data-dir data \
        --cameras CF00001 CM00009 --debug

    # Custom thresholds + sample
    python grounded_sam2_snow_segmentation.py \
        --data-dir data \
        --box-threshold 0.25 --text-threshold 0.2 --sample 50
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

def load_grounding_dino(device="cuda"):
    """Load GroundingDINO model for open-set detection."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id
    ).eval().to(device)
    print(f"  GroundingDINO-tiny loaded on {device}")
    return model, processor


def load_sam2(device="cuda"):
    """Load SAM2 model for mask refinement."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("Loading SAM2 (hiera-large)...")
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large", device=device
    )
    print(f"  SAM2 loaded on {device}")
    return predictor


# ── Detection + segmentation ────────────────────────────────────────────────

def detect_snow_boxes(image, model, processor, device="cuda",
                      box_threshold=0.3, text_threshold=0.25):
    """Detect snow regions with GroundingDINO. Returns boxes in xyxy format."""
    text = "snow."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image.size[::-1]],
        )[0]

    boxes = results["boxes"].cpu()
    scores = results["scores"].cpu()
    keep = scores >= box_threshold
    boxes = boxes[keep].numpy()
    scores = scores[keep].numpy()
    return boxes, scores


def refine_masks_sam2(image_np, boxes, predictor):
    """Refine detection boxes into pixel-level masks using SAM2."""
    if len(boxes) == 0:
        return np.zeros(image_np.shape[:2], dtype=np.uint8)

    predictor.set_image(image_np)

    input_boxes = torch.tensor(boxes, dtype=torch.float32, device=predictor.device)

    masks, scores, _ = predictor.predict(
        box=input_boxes,
        multimask_output=False,
    )

    # masks shape: (N, 1, H, W) or (N, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    combined = np.zeros(image_np.shape[:2], dtype=np.uint8)
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        combined = np.maximum(combined, (mask > 0.5).astype(np.uint8))

    return combined


# ── Processing ───────────────────────────────────────────────────────────────

def process_image(image_path, gdino_model, gdino_processor, sam2_predictor,
                  output_dir, device="cuda", box_threshold=0.3,
                  text_threshold=0.25, debug=False):
    """Process a single image. Returns result dict or None."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    boxes, scores = detect_snow_boxes(
        image, gdino_model, gdino_processor,
        device=device, box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    image_np = np.array(image)

    if len(boxes) > 0:
        snow_mask = refine_masks_sam2(image_np, boxes, sam2_predictor)
    else:
        snow_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        visualize(image_np, snow_mask, boxes, scores,
                  cam_out / f"{stem}_overview.png",
                  title=f"{camera} — {stem}")

    snow_pct = snow_mask.sum() / snow_mask.size * 100

    return {
        "image": str(image_path),
        "camera": camera,
        "filename": Path(image_path).name,
        "snow_pct": round(snow_pct, 2),
        "n_detections": len(boxes),
    }


def visualize(image, mask, boxes, scores, output_path, title=""):
    """Save overview figure with detections and mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original + detection boxes
    axes[0].imshow(image)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 4, f"{score:.2f}", color="lime", fontsize=8)
    axes[0].set_title(f"Detections ({len(boxes)})")
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
        description="Grounded SAM2 snow segmentation"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory with camera folders")
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific cameras (default: all)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent
                                    / "preprocessing" / "segmentation"
                                    / "groundedsam2_snow_output"),
                        help="Output directory")
    parser.add_argument("--box-threshold", type=float, default=0.3,
                        help="GroundingDINO box confidence threshold (default: 0.3)")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="GroundingDINO text confidence threshold (default: 0.25)")
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
    print("GROUNDED SAM2 — SNOW SEGMENTATION")
    print("=" * 70)
    print(f"  Box threshold:  {args.box_threshold}")
    print(f"  Text threshold: {args.text_threshold}")
    print(f"  Device:         {device}")
    print(f"  Output:         {output_dir}")

    gdino_model, gdino_processor = load_grounding_dino(device=device)
    sam2_predictor = load_sam2(device=device)

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
                img_path, gdino_model, gdino_processor, sam2_predictor,
                output_dir, device=device,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                debug=args.debug,
            )
            if result:
                results.append(result)

    if results:
        summary_path = output_dir / "groundedsam2_snow_summary.csv"
        fieldnames = ["image", "camera", "filename", "snow_pct", "n_detections"]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary: {summary_path}")

    print(f"\nProcessed {len(results)} images.")
    print("Done!")


if __name__ == "__main__":
    main()
