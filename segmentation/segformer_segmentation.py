#!/usr/bin/env python3
"""
SegFormer-based ground segmentation using ADE20K pretrained model.

Uses nvidia/segformer-b5-finetuned-ade-640-640 to classify pixels into
ADE20K semantic categories, then merges ground-related classes into a
binary ground mask. No training needed — fully pretrained.

Usage:
    # Single image
    python segformer_segmentation.py --image /path/to/image.jpg --debug

    # All cameras
    python segformer_segmentation.py --data-dir data

    # Specific cameras, subsample
    python segformer_segmentation.py --data-dir data \
        --cameras CM00006 CM00024 --sample 50

    # Run on canonical views from DINOv3 pipeline
    python segformer_segmentation.py \
        --canonical-dir dinov3/canonical_views
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ADE20K ground-related class IDs (0-indexed)
# See: https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8
DEFAULT_GROUND_LABELS = [
    13,  # earth, ground
    6,   # road, route
    11,  # sidewalk, pavement
    29,  # field
    9,   # grass
    52,  # path
    91,  # dirt track
]

ADE20K_LABEL_NAMES = {
    0: "wall", 1: "building", 2: "sky", 3: "floor", 4: "tree",
    5: "ceiling", 6: "road", 7: "bed", 8: "windowpane", 9: "grass",
    10: "cabinet", 11: "sidewalk", 12: "person", 13: "earth",
    14: "door", 15: "table", 16: "mountain", 17: "plant", 18: "curtain",
    19: "chair", 20: "car", 21: "water", 22: "painting", 23: "sofa",
    24: "shelf", 25: "house", 26: "sea", 27: "mirror", 28: "rug",
    29: "field", 30: "armchair", 31: "seat", 32: "fence", 33: "desk",
    34: "rock", 35: "wardrobe", 36: "lamp", 37: "bathtub", 38: "railing",
    39: "cushion", 40: "base", 41: "box", 42: "column", 43: "signboard",
    44: "chest", 45: "counter", 46: "sand", 47: "sink", 48: "skyscraper",
    49: "fireplace", 50: "refrigerator", 51: "grandstand", 52: "path",
    53: "stairs", 54: "runway", 55: "case", 56: "pool_table", 57: "pillow",
    58: "screen_door", 59: "stairway", 60: "river", 61: "bridge",
    62: "bookcase", 63: "blind", 64: "coffee_table", 65: "toilet",
    66: "flower", 67: "book", 68: "hill", 69: "bench", 70: "countertop",
    71: "stove", 72: "palm", 73: "kitchen_island", 74: "computer",
    75: "swivel_chair", 76: "boat", 77: "bar", 78: "arcade_machine",
    79: "hovel", 80: "bus", 81: "towel", 82: "light", 83: "truck",
    84: "tower", 85: "chandelier", 86: "awning", 87: "streetlight",
    88: "booth", 89: "television", 90: "airplane", 91: "dirt_track",
    92: "apparel", 93: "pole", 94: "land", 95: "bannister",
    96: "escalator", 97: "ottoman", 98: "bottle", 99: "buffet",
}


# ── model loading ────────────────────────────────────────────────────────────

def load_segformer(model_name="nvidia/segformer-b5-finetuned-ade-640-640", device="cuda"):
    """Load SegFormer model and processor from HuggingFace."""
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    print(f"Loading {model_name}...")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model = model.to(device).eval()
    print(f"  Model loaded on {device}")
    return model, processor


# ── prediction ───────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_segmentation(model, processor, image, device="cuda"):
    """
    Predict semantic segmentation for a PIL image.

    Returns:
        label_map: np.ndarray (H, W) int, ADE20K class indices
        probs: np.ndarray (C, H, W) float32, softmax probabilities
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    logits = outputs.logits  # (1, num_classes, h, w) — lower resolution
    # Upsample to original image size
    h, w = image.size[1], image.size[0]  # PIL is (W, H)
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )

    probs = torch.softmax(upsampled, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)
    label_map = probs.argmax(axis=0)  # (H, W)

    return label_map, probs


# ── ground mask extraction ───────────────────────────────────────────────────

def extract_ground_mask(label_map, ground_label_ids, min_confidence=0.0, probs=None):
    """
    Merge selected ADE20K labels into a binary ground mask.

    Args:
        label_map: (H, W) int array of class indices
        ground_label_ids: list of ADE20K class IDs to consider ground
        min_confidence: if > 0, only include pixels where softmax prob
                        for the predicted class exceeds this threshold
        probs: (C, H, W) softmax probabilities (needed if min_confidence > 0)

    Returns:
        ground_mask: (H, W) bool, True = ground pixel
    """
    ground_mask = np.isin(label_map, ground_label_ids)

    if min_confidence > 0 and probs is not None:
        max_probs = probs.max(axis=0)  # (H, W)
        ground_mask = ground_mask & (max_probs >= min_confidence)

    return ground_mask


# ── visualization ────────────────────────────────────────────────────────────

def visualize_segformer(image, label_map, ground_mask, output_path, title="",
                        ground_label_ids=None):
    """Save a 2x2 overview: original, full segmentation, ground mask, overlay."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # Full segmentation map (colored by class)
    n_classes = label_map.max() + 1
    cmap = plt.cm.get_cmap("tab20", max(n_classes, 20))
    axes[0, 1].imshow(label_map, cmap=cmap, interpolation="nearest")
    axes[0, 1].set_title(f"Semantic Segmentation ({n_classes} classes present)")
    axes[0, 1].axis("off")

    # Ground mask
    axes[1, 0].imshow(ground_mask, cmap="gray")
    ground_pct = ground_mask.sum() / ground_mask.size * 100
    label_str = ""
    if ground_label_ids:
        names = [ADE20K_LABEL_NAMES.get(i, str(i)) for i in ground_label_ids
                 if np.any(label_map == i)]
        if names:
            label_str = f"\nActive: {', '.join(names)}"
    axes[1, 0].set_title(f"Ground Mask ({ground_pct:.1f}%){label_str}")
    axes[1, 0].axis("off")

    # Overlay
    img_array = np.array(image)
    overlay = img_array.copy()
    overlay[ground_mask] = (
        overlay[ground_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)
    overlay[~ground_mask] = (
        overlay[~ground_mask] * 0.5 + np.array([100, 100, 100]) * 0.5
    ).astype(np.uint8)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Overlay (green=ground)")
    axes[1, 1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── processing functions ─────────────────────────────────────────────────────

def process_image(image_path, model, processor, output_dir, ground_label_ids,
                  min_confidence=0.0, device="cuda", debug=False):
    """
    Process a single image: predict segmentation, extract ground mask, save outputs.

    Returns:
        dict with image path, ground percentage, etc., or None on failure.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    label_map, probs = predict_segmentation(model, processor, image, device=device)
    ground_mask = extract_ground_mask(label_map, ground_label_ids,
                                       min_confidence=min_confidence, probs=probs)

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    # Save ground mask
    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    if debug:
        # Save full segmentation map
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap("tab20", max(label_map.max() + 1, 20))
        plt.imsave(str(cam_out / f"{stem}_segmap.png"), label_map,
                    cmap=cmap)

        # Save overview figure
        visualize_segformer(image, label_map, ground_mask,
                            cam_out / f"{stem}_overview.png",
                            title=f"{camera} — {stem}",
                            ground_label_ids=ground_label_ids)

    ground_pct = ground_mask.sum() / ground_mask.size * 100

    return {
        "image": str(image_path),
        "camera": camera,
        "filename": Path(image_path).name,
        "ground_pct": round(ground_pct, 2),
        "method": "segformer",
    }


def process_folder(data_dir, cameras, model, processor, output_dir,
                   ground_label_ids, min_confidence, sample,
                   device="cuda", debug=False):
    """Process all images in the given camera folders."""
    data_dir = Path(data_dir)
    results = []

    if not cameras:
        cameras = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)
        ])

    print(f"\nCameras to process: {cameras}")

    for cam in cameras:
        cam_dir = data_dir / cam
        if not cam_dir.exists():
            print(f"  WARNING: {cam_dir} does not exist, skipping")
            continue

        jpgs = sorted(cam_dir.glob("*.jpg"))
        if sample > 0 and len(jpgs) > sample:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(jpgs), size=sample, replace=False)
            jpgs = [jpgs[i] for i in sorted(indices)]

        print(f"\n  {cam}: processing {len(jpgs)} images")

        for img_path in tqdm(jpgs, desc=f"  {cam}", leave=True):
            result = process_image(img_path, model, processor, output_dir,
                                   ground_label_ids, min_confidence=min_confidence,
                                   device=device, debug=debug)
            if result:
                results.append(result)

    return results


def process_canonical_views(canonical_dir, model, processor, output_dir,
                            ground_label_ids, min_confidence=0.0,
                            device="cuda", debug=True):
    """Process canonical view images from the DINOv3 pipeline."""
    canonical_dir = Path(canonical_dir)
    results = []

    locations = sorted([
        d for d in canonical_dir.iterdir() if d.is_dir()
    ])

    print(f"\nCanonical view locations: {len(locations)}")

    for loc_dir in locations:
        views = sorted(loc_dir.glob("canonical_view_*.jpg"))
        if not views:
            continue

        print(f"\n  {loc_dir.name}: {len(views)} canonical views")

        for view_path in views:
            result = process_image(view_path, model, processor, output_dir,
                                   ground_label_ids, min_confidence=min_confidence,
                                   device=device, debug=debug)
            if result:
                results.append(result)

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SegFormer-based ground segmentation (ADE20K pretrained)"
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str,
                             help="Single image path")
    input_group.add_argument("--data-dir", type=str,
                             help="Data directory with camera folders")
    input_group.add_argument("--canonical-dir", type=str,
                             help="Canonical views directory from DINOv3 pipeline")

    # Options
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific camera folders (default: all)")
    parser.add_argument("--output-dir", type=str,
                        default="preprocessing/segmentation/segformer_output",
                        help="Output directory")
    parser.add_argument("--model", type=str,
                        default="nvidia/segformer-b5-finetuned-ade-640-640",
                        help="HuggingFace model ID")
    parser.add_argument("--ground-labels", nargs="+", type=int,
                        default=DEFAULT_GROUND_LABELS,
                        help="ADE20K class IDs to merge as ground "
                             "(default: 13 6 11 29 9 52 91)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Min softmax probability for a pixel to be "
                             "included in the ground mask (default: 0.0 = no filter)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Images to sample per camera (0=all)")
    parser.add_argument("--debug", action="store_true",
                        help="Save segmentation maps and overview figures")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_labels = args.ground_labels
    ground_names = [ADE20K_LABEL_NAMES.get(i, str(i)) for i in ground_labels]

    print("=" * 70)
    print("SEGFORMER — GROUND SEGMENTATION (ADE20K)")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  Ground labels: {ground_labels}")
    print(f"                 {ground_names}")
    print(f"  Min conf:     {args.min_confidence}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_dir}")

    model, processor = load_segformer(args.model, device)

    # Dispatch
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = process_image(args.image, model, processor, output_dir,
                               ground_labels, min_confidence=args.min_confidence,
                               device=device, debug=True)
        if result:
            print(f"\n  Ground: {result['ground_pct']:.1f}%")
            print(f"  Saved to: {output_dir}")
        results = [result] if result else []

    elif args.data_dir:
        results = process_folder(args.data_dir, args.cameras, model, processor,
                                 output_dir, ground_labels, args.min_confidence,
                                 args.sample, device=device, debug=args.debug)

    elif args.canonical_dir:
        results = process_canonical_views(args.canonical_dir, model, processor,
                                          output_dir, ground_labels,
                                          min_confidence=args.min_confidence,
                                          device=device, debug=True)

    # Save summary
    if results:
        summary_path = output_dir / "segformer_summary.csv"
        fieldnames = ["image", "camera", "filename", "ground_pct", "method"]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary saved to: {summary_path}")

    print(f"\nProcessed {len(results)} images.")
    print("Done!")


if __name__ == "__main__":
    main()
