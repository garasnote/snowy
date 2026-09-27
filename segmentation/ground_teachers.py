#!/usr/bin/env python3
"""
Ground segmentation teachers — five zero-shot / pretrained methods whose
outputs are combined by majority vote (ground_utils.py: ensemble) into the
ground pseudo-labels used to train the ground student models (Methods,
"Ground Segmentation").

Consolidated from deeplabv3_segmentation.py, segformer_segmentation.py,
sam3_dino_segmentation.py, dinov2_pca_segmentation.py, depth_segmentation.py.
Each teacher keeps its original CLI flags, now behind a subcommand:

    python ground_teachers.py deeplabv3  --data-dir ./data
    python ground_teachers.py segformer  --data-dir ./data
    python ground_teachers.py sam3       --data-dir ./data --method text
    python ground_teachers.py dinov2_pca --data-dir ./data
    python ground_teachers.py depth      --data-dir ./data --model-size large

Run `python ground_teachers.py <teacher> --help` for that teacher's full
option list (e.g. depth's --grid/--compare modes, sam3's --method dino+sam).
All accept --image <path> for a single image or --canonical-dir <dir> for
the DINOv3 canonical-views pipeline, instead of --data-dir.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import WORKSPACE_DIR

# ── shared: ADE20K ground-related classes (DeepLabV3+, SegFormer) ──────────
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

# ── shared: input-mode argument group, reused by every teacher's subparser ──

def _add_input_and_common_args(p, default_output_dir):
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Single image path")
    input_group.add_argument("--data-dir", type=str,
                              help="Data directory with camera folders")
    input_group.add_argument("--canonical-dir", type=str,
                              help="Canonical views directory from DINOv3 pipeline")
    p.add_argument("--cameras", nargs="+", default=[],
                    help="Specific camera folders (default: all)")
    p.add_argument("--output-dir", type=str, default=default_output_dir,
                    help="Output directory")
    p.add_argument("--sample", type=int, default=0,
                    help="Images to sample per camera (0=all)")
    p.add_argument("--debug", action="store_true",
                    help="Save debug visualizations")


def _write_summary_csv(results, output_dir, name, fieldnames):
    if not results:
        return
    summary_path = output_dir / name
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary saved to: {summary_path}")


# =============================================================================
# DEEPLABV3+ — ADE20K pretrained (OpenMMLab)
# =============================================================================

def deeplabv3_load_model(model_name="deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512",
                          device="cuda"):
    """Load DeepLabV3+ model via mmsegmentation's MMSegInferencer.

    Config and checkpoint are downloaded automatically on first use.
    """
    # Bypass mmseg strict MMCV version check (mmcv 2.2.0 works but is rejected)
    try:
        import mmcv
        from packaging.version import parse as _parse_ver
        _real_ver = mmcv.__version__
        if _parse_ver(_real_ver) >= _parse_ver("2.2.0"):
            mmcv.__version__ = "2.1.0"
            mmcv.digit_version = (2, 1, 0, 0)
            import mmseg  # noqa: F401 -- triggers __init__ with patched version
            mmcv.__version__ = _real_ver
    except ImportError:
        pass

    from mmseg.apis import MMSegInferencer

    print(f"Loading {model_name}...")
    inferencer = MMSegInferencer(model=model_name, device=device)
    print(f"  Model loaded on {device}")
    return inferencer


def deeplabv3_predict(inferencer, image_path):
    """Predict semantic segmentation for an image.

    Returns:
        label_map: np.ndarray (H, W) int, ADE20K class indices
    """
    result = inferencer(str(image_path), show=False, return_datasamples=True)
    if isinstance(result, dict) and "predictions" in result:
        datasample = result["predictions"][0]
    elif isinstance(result, (list, tuple)):
        datasample = result[0]
    else:
        datasample = result
    label_map = datasample.pred_sem_seg.data.squeeze(0).cpu().numpy()
    return label_map


def _extract_ground_mask_ade20k(label_map, ground_label_ids):
    """Merge selected ADE20K labels into a binary ground mask (shared by
    DeepLabV3+ and SegFormer, which use the same class taxonomy)."""
    return np.isin(label_map, ground_label_ids)


def deeplabv3_visualize(image, label_map, ground_mask, output_path, title="",
                         ground_label_ids=None):
    """Save a 2x2 overview: original, full segmentation, ground mask, overlay."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    n_classes = label_map.max() + 1
    cmap = plt.cm.get_cmap("tab20", max(n_classes, 20))
    axes[0, 1].imshow(label_map, cmap=cmap, interpolation="nearest")
    axes[0, 1].set_title(f"DeepLabV3+ Segmentation ({n_classes} classes present)")
    axes[0, 1].axis("off")

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


def deeplabv3_process_image(image_path, inferencer, output_dir, ground_label_ids,
                             debug=False):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    label_map = deeplabv3_predict(inferencer, image_path)
    ground_mask = _extract_ground_mask_ade20k(label_map, ground_label_ids)

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name
    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    if debug:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap("tab20", max(label_map.max() + 1, 20))
        plt.imsave(str(cam_out / f"{stem}_segmap.png"), label_map, cmap=cmap)
        deeplabv3_visualize(image, label_map, ground_mask,
                            cam_out / f"{stem}_overview.png",
                            title=f"{camera} - {stem}",
                            ground_label_ids=ground_label_ids)

    ground_pct = ground_mask.sum() / ground_mask.size * 100
    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name,
        "ground_pct": round(ground_pct, 2), "method": "deeplabv3plus",
    }


def deeplabv3_process_folder(data_dir, cameras, inferencer, output_dir,
                              ground_label_ids, sample, debug=False):
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
            result = deeplabv3_process_image(img_path, inferencer, output_dir,
                                              ground_label_ids, debug=debug)
            if result:
                results.append(result)
    return results


def deeplabv3_process_canonical_views(canonical_dir, inferencer, output_dir,
                                       ground_label_ids, debug=True):
    canonical_dir = Path(canonical_dir)
    results = []
    locations = sorted([d for d in canonical_dir.iterdir() if d.is_dir()])
    print(f"\nCanonical view locations: {len(locations)}")
    for loc_dir in locations:
        views = sorted(loc_dir.glob("canonical_view_*.jpg"))
        if not views:
            continue
        print(f"\n  {loc_dir.name}: {len(views)} canonical views")
        for view_path in views:
            result = deeplabv3_process_image(view_path, inferencer, output_dir,
                                              ground_label_ids, debug=debug)
            if result:
                results.append(result)
    return results


def add_deeplabv3_subparser(subparsers):
    p = subparsers.add_parser("deeplabv3", help="DeepLabV3+ (ADE20K, OpenMMLab)")
    _add_input_and_common_args(p, str(WORKSPACE_DIR / "segmentation" / "ground" / "deeplabv3"))
    p.add_argument("--model", type=str,
                    default="deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512",
                    help="MMSeg model name (config name, checkpoint auto-downloaded)")
    p.add_argument("--ground-labels", nargs="+", type=int,
                    default=DEFAULT_GROUND_LABELS,
                    help="ADE20K class IDs to merge as ground (default: 13 6 11 29 9 52 91)")
    return p


def run_deeplabv3(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_labels = args.ground_labels
    ground_names = [ADE20K_LABEL_NAMES.get(i, str(i)) for i in ground_labels]

    print("=" * 70)
    print("DEEPLABV3+ - GROUND SEGMENTATION (ADE20K, OpenMMLab)")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  Ground labels: {ground_labels}")
    print(f"                 {ground_names}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_dir}")

    inferencer = deeplabv3_load_model(args.model, device)

    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = deeplabv3_process_image(args.image, inferencer, output_dir,
                                         ground_labels, debug=True)
        if result:
            print(f"\n  Ground: {result['ground_pct']:.1f}%")
            print(f"  Saved to: {output_dir}")
        results = [result] if result else []
    elif args.data_dir:
        results = deeplabv3_process_folder(args.data_dir, args.cameras, inferencer,
                                           output_dir, ground_labels, args.sample,
                                           debug=args.debug)
    elif args.canonical_dir:
        results = deeplabv3_process_canonical_views(args.canonical_dir, inferencer,
                                                     output_dir, ground_labels,
                                                     debug=True)

    _write_summary_csv(results, output_dir, "deeplabv3_summary.csv",
                        ["image", "camera", "filename", "ground_pct", "method"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# SEGFORMER — ADE20K pretrained (nvidia/segformer-b5-finetuned-ade-640-640)
# =============================================================================

def segformer_load_model(model_name="nvidia/segformer-b5-finetuned-ade-640-640", device="cuda"):
    """Load SegFormer model and processor from HuggingFace."""
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    print(f"Loading {model_name}...")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model = model.to(device).eval()
    print(f"  Model loaded on {device}")
    return model, processor


@torch.no_grad()
def segformer_predict(model, processor, image, device="cuda"):
    """Predict semantic segmentation for a PIL image.

    Returns:
        label_map: np.ndarray (H, W) int, ADE20K class indices
        probs: np.ndarray (C, H, W) float32, softmax probabilities
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    logits = outputs.logits
    h, w = image.size[1], image.size[0]
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )

    probs = torch.softmax(upsampled, dim=1).squeeze(0).cpu().numpy()
    label_map = probs.argmax(axis=0)
    return label_map, probs


def segformer_extract_ground_mask(label_map, ground_label_ids, min_confidence=0.0, probs=None):
    ground_mask = _extract_ground_mask_ade20k(label_map, ground_label_ids)
    if min_confidence > 0 and probs is not None:
        max_probs = probs.max(axis=0)
        ground_mask = ground_mask & (max_probs >= min_confidence)
    return ground_mask


def segformer_visualize(image, label_map, ground_mask, output_path, title="",
                         ground_label_ids=None):
    """Save a 2x2 overview: original, full segmentation, ground mask, overlay."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    n_classes = label_map.max() + 1
    cmap = plt.cm.get_cmap("tab20", max(n_classes, 20))
    axes[0, 1].imshow(label_map, cmap=cmap, interpolation="nearest")
    axes[0, 1].set_title(f"Semantic Segmentation ({n_classes} classes present)")
    axes[0, 1].axis("off")

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


def segformer_process_image(image_path, model, processor, output_dir, ground_label_ids,
                             min_confidence=0.0, device="cuda", debug=False):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    label_map, probs = segformer_predict(model, processor, image, device=device)
    ground_mask = segformer_extract_ground_mask(label_map, ground_label_ids,
                                                 min_confidence=min_confidence, probs=probs)

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name
    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    if debug:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap("tab20", max(label_map.max() + 1, 20))
        plt.imsave(str(cam_out / f"{stem}_segmap.png"), label_map, cmap=cmap)
        segformer_visualize(image, label_map, ground_mask,
                            cam_out / f"{stem}_overview.png",
                            title=f"{camera} - {stem}",
                            ground_label_ids=ground_label_ids)

    ground_pct = ground_mask.sum() / ground_mask.size * 100
    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name,
        "ground_pct": round(ground_pct, 2), "method": "segformer",
    }


def segformer_process_folder(data_dir, cameras, model, processor, output_dir,
                              ground_label_ids, min_confidence, sample,
                              device="cuda", debug=False):
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
            result = segformer_process_image(img_path, model, processor, output_dir,
                                              ground_label_ids, min_confidence=min_confidence,
                                              device=device, debug=debug)
            if result:
                results.append(result)
    return results


def segformer_process_canonical_views(canonical_dir, model, processor, output_dir,
                                       ground_label_ids, min_confidence=0.0,
                                       device="cuda", debug=True):
    canonical_dir = Path(canonical_dir)
    results = []
    locations = sorted([d for d in canonical_dir.iterdir() if d.is_dir()])
    print(f"\nCanonical view locations: {len(locations)}")
    for loc_dir in locations:
        views = sorted(loc_dir.glob("canonical_view_*.jpg"))
        if not views:
            continue
        print(f"\n  {loc_dir.name}: {len(views)} canonical views")
        for view_path in views:
            result = segformer_process_image(view_path, model, processor, output_dir,
                                              ground_label_ids, min_confidence=min_confidence,
                                              device=device, debug=debug)
            if result:
                results.append(result)
    return results


def add_segformer_subparser(subparsers):
    p = subparsers.add_parser("segformer", help="SegFormer (ADE20K pretrained)")
    _add_input_and_common_args(p, str(WORKSPACE_DIR / "segmentation" / "ground" / "segformer"))
    p.add_argument("--model", type=str, default="nvidia/segformer-b5-finetuned-ade-640-640",
                    help="HuggingFace model ID")
    p.add_argument("--ground-labels", nargs="+", type=int, default=DEFAULT_GROUND_LABELS,
                    help="ADE20K class IDs to merge as ground (default: 13 6 11 29 9 52 91)")
    p.add_argument("--min-confidence", type=float, default=0.0,
                    help="Min softmax probability for a pixel to be included "
                         "in the ground mask (default: 0.0 = no filter)")
    return p


def run_segformer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_labels = args.ground_labels
    ground_names = [ADE20K_LABEL_NAMES.get(i, str(i)) for i in ground_labels]

    print("=" * 70)
    print("SEGFORMER - GROUND SEGMENTATION (ADE20K)")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  Ground labels: {ground_labels}")
    print(f"                 {ground_names}")
    print(f"  Min conf:     {args.min_confidence}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_dir}")

    model, processor = segformer_load_model(args.model, device)

    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = segformer_process_image(args.image, model, processor, output_dir,
                                         ground_labels, min_confidence=args.min_confidence,
                                         device=device, debug=True)
        if result:
            print(f"\n  Ground: {result['ground_pct']:.1f}%")
            print(f"  Saved to: {output_dir}")
        results = [result] if result else []
    elif args.data_dir:
        results = segformer_process_folder(args.data_dir, args.cameras, model, processor,
                                           output_dir, ground_labels, args.min_confidence,
                                           args.sample, device=device, debug=args.debug)
    elif args.canonical_dir:
        results = segformer_process_canonical_views(args.canonical_dir, model, processor,
                                                     output_dir, ground_labels,
                                                     min_confidence=args.min_confidence,
                                                     device=device, debug=True)

    _write_summary_csv(results, output_dir, "segformer_summary.csv",
                        ["image", "camera", "filename", "ground_pct", "method"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# SAM3 (+ optional DINOv2-guided box prompt)
# =============================================================================
# Two modes:
#   text:     SAM3 text prompt ("ground, terrain, earth, ...") -- this is the
#             mode reported in the paper as the "SAM3" ground teacher.
#   dino+sam: DINOv2 PCA rough mask -> bounding box -> SAM3 box refinement.
#             Exploratory; not the configuration used in the paper.

def sam3_load_model(device="cuda"):
    """Load SAM3 model and processor."""
    sam3_dir = _REPO_ROOT.parent / "sam3"
    dinov3_dir = _REPO_ROOT.parent / "dinov3"
    sys.path.insert(0, str(sam3_dir))
    sys.path.insert(0, str(dinov3_dir))

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3 model from HuggingFace...")
    model = build_sam3_image_model(
        checkpoint_path=None, load_from_HF=True, device=device
    )
    processor = Sam3Processor(model, device=device)
    print(f"  SAM3 loaded on {device}")
    return processor


def sam3_load_dino(model_name="facebook/dinov2-base", device="cuda"):
    """Load DINOv2 model and processor from HuggingFace (dino+sam mode only)."""
    from transformers import AutoImageProcessor, AutoModel

    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"  DINOv2 loaded on {device}")
    return model, processor


def sam3_dino_pca_ground_mask(dino_model, dino_processor, image, device="cuda",
                               input_size=518):
    """Extract rough ground mask from DINOv2 PCA features.

    Returns binary mask at original image resolution + bounding box
    (cx, cy, w, h) normalized [0,1] for use as a SAM3 box prompt.
    """
    from sklearn.decomposition import PCA
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    resized = image.resize((input_size, input_size), Image.BILINEAR)
    inputs = dino_processor(images=resized, return_tensors="pt", do_resize=False).to(device)
    pixel_values = inputs["pixel_values"]
    _, _, input_h, input_w = pixel_values.shape
    patch_size = dino_model.config.patch_size
    grid_h = input_h // patch_size
    grid_w = input_w // patch_size

    with torch.inference_mode():
        outputs = dino_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        patch_features = hidden_states[:, 1:, :]
        features = patch_features.squeeze(0).reshape(grid_h, grid_w, -1)

    h, w, d = features.shape
    features_flat = features.cpu().numpy().reshape(-1, d)
    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(features_flat)

    pc1 = pca_feats[:, 0].reshape(h, w)
    thresh = threshold_otsu(pc1)
    binary = (pc1 > thresh).astype(int)

    bottom = binary[int(h * 0.66):, :]
    unique, counts = np.unique(bottom, return_counts=True)
    ground_label = unique[np.argmax(counts)]
    ground_mask_small = (binary == ground_label).astype(np.uint8)

    labeled, n_components = ndimage.label(ground_mask_small)
    if n_components > 1:
        bottom_labels = labeled[int(h * 0.66):, :]
        bottom_labels_flat = bottom_labels[bottom_labels > 0]
        if len(bottom_labels_flat) > 0:
            best_comp = np.bincount(bottom_labels_flat).argmax()
            ground_mask_small = (labeled == best_comp).astype(np.uint8)

    ys, xs = np.where(ground_mask_small > 0)
    if len(ys) == 0:
        return None, None

    y0, y1 = ys.min() / h, ys.max() / h
    x0, x1 = xs.min() / w, xs.max() / w
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = x1 - x0, y1 - y0
    box_cxcywh = [cx, cy, bw, bh]

    orig_h, orig_w = image.size[1], image.size[0]
    mask_pil = Image.fromarray(ground_mask_small * 255)
    mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
    rough_mask = np.array(mask_pil) > 127

    return rough_mask, box_cxcywh


def sam3_segment_text_only(sam_processor, pil_image, text_prompts, threshold=0.3):
    """Segment ground using SAM3 text prompts, combining all prompt masks."""
    combined_mask = np.zeros(
        (pil_image.size[1], pil_image.size[0]), dtype=np.uint8
    )
    for prompt in text_prompts:
        prompt = prompt.strip()
        if not prompt:
            continue
        sam_processor.confidence_threshold = threshold
        state = sam_processor.set_image(pil_image)
        state = sam_processor.set_text_prompt(prompt, state)
        if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
            masks = state["masks"]
            mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask)
    return combined_mask


def sam3_segment_dino_guided(sam_processor, dino_model, dino_processor, pil_image,
                              input_size=518, threshold=0.3, device="cuda"):
    """DINOv2 PCA rough mask -> bounding box -> SAM3 box prompt refinement."""
    rough_mask, box_cxcywh = sam3_dino_pca_ground_mask(
        dino_model, dino_processor, pil_image, device=device, input_size=input_size
    )
    if box_cxcywh is None:
        return sam3_segment_text_only(
            sam_processor, pil_image, ["ground"], threshold=threshold
        ), rough_mask

    sam_processor.confidence_threshold = threshold
    state = sam_processor.set_image(pil_image)
    state = sam_processor.set_text_prompt("ground", state)
    state = sam_processor.add_geometric_prompt(box_cxcywh, label=True, state=state)

    if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
        masks = state["masks"]
        refined_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8)
    else:
        refined_mask = rough_mask.astype(np.uint8) if rough_mask is not None else np.zeros(
            (pil_image.size[1], pil_image.size[0]), dtype=np.uint8
        )
    return refined_mask, rough_mask


def sam3_visualize(image, ground_mask, output_path, title="", rough_mask=None,
                    method="text"):
    """Save overview figure."""
    import matplotlib.pyplot as plt

    n_cols = 3 if rough_mask is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    col = 1
    if rough_mask is not None:
        axes[col].imshow(rough_mask, cmap="gray")
        axes[col].set_title("DINOv2 PCA rough mask")
        axes[col].axis("off")
        col += 1

    ground_pct = ground_mask.sum() / ground_mask.size * 100
    axes[col].imshow(ground_mask, cmap="gray")
    axes[col].set_title(f"SAM3 Ground Mask ({ground_pct:.1f}%)")
    axes[col].axis("off")
    col += 1

    img_array = np.array(image)
    overlay = img_array.copy()
    mask_bool = ground_mask > 0
    overlay[mask_bool] = (
        overlay[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)
    overlay[~mask_bool] = (
        overlay[~mask_bool] * 0.5 + np.array([100, 100, 100]) * 0.5
    ).astype(np.uint8)
    axes[col].imshow(overlay)
    axes[col].set_title("Overlay (green=ground)")
    axes[col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def sam3_process_image(image_path, sam_processor, method, text_prompts, threshold,
                        output_dir, dino_model=None, dino_processor=None,
                        input_size=518, device="cuda", debug=False):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    rough_mask = None
    if method == "dino+sam" and dino_model is not None:
        ground_mask, rough_mask = sam3_segment_dino_guided(
            sam_processor, dino_model, dino_processor, image,
            input_size=input_size, threshold=threshold, device=device
        )
    else:
        ground_mask = sam3_segment_text_only(
            sam_processor, image, text_prompts, threshold=threshold
        )

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name
    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    if debug:
        sam3_visualize(image, ground_mask, cam_out / f"{stem}_overview.png",
                       title=f"{camera} - {stem} ({method})",
                       rough_mask=rough_mask, method=method)

    ground_pct = (ground_mask > 0).sum() / ground_mask.size * 100
    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name,
        "ground_pct": round(ground_pct, 2), "method": f"sam3_{method}",
    }


def sam3_process_folder(data_dir, cameras, sam_processor, method, text_prompts,
                         threshold, output_dir, sample, dino_model=None,
                         dino_processor=None, input_size=518, device="cuda",
                         debug=False):
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
            result = sam3_process_image(
                img_path, sam_processor, method, text_prompts, threshold,
                output_dir, dino_model=dino_model, dino_processor=dino_processor,
                input_size=input_size, device=device, debug=debug
            )
            if result:
                results.append(result)
    return results


def sam3_process_canonical_views(canonical_dir, sam_processor, method, text_prompts,
                                  threshold, output_dir, dino_model=None,
                                  dino_processor=None, input_size=518,
                                  device="cuda", debug=True):
    canonical_dir = Path(canonical_dir)
    results = []
    locations = sorted([d for d in canonical_dir.iterdir() if d.is_dir()])
    print(f"\nCanonical view locations: {len(locations)}")
    for loc_dir in locations:
        views = sorted(loc_dir.glob("canonical_view_*.jpg"))
        if not views:
            continue
        print(f"\n  {loc_dir.name}: {len(views)} canonical views")
        for view_path in views:
            result = sam3_process_image(
                view_path, sam_processor, method, text_prompts, threshold,
                output_dir, dino_model=dino_model, dino_processor=dino_processor,
                input_size=input_size, device=device, debug=debug
            )
            if result:
                results.append(result)
    return results


def add_sam3_subparser(subparsers):
    p = subparsers.add_parser("sam3", help="SAM3 text-prompted (+ optional DINOv2-guided)")
    _add_input_and_common_args(p, str(WORKSPACE_DIR / "segmentation" / "ground" / "sam3"))
    p.add_argument("--method", choices=["text", "dino+sam"], default="text",
                    help="Segmentation method (default: text -- the one used in the paper)")
    p.add_argument("--text-prompts", type=str, default="ground,terrain,earth,road,grass",
                    help="Comma-separated text prompts for SAM3")
    p.add_argument("--threshold", type=float, default=0.3,
                    help="SAM3 confidence threshold (default: 0.3)")
    p.add_argument("--dino-model", type=str, default="facebook/dinov2-base",
                    help="DINOv2 model for dino+sam mode")
    p.add_argument("--input-size", type=int, default=518,
                    help="DINOv2 input resolution (default: 518)")
    return p


def run_sam3(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    text_prompts = [p.strip() for p in args.text_prompts.split(",")]

    print("=" * 70)
    print("SAM3 (+ DINOv2) - GROUND SEGMENTATION")
    print("=" * 70)
    print(f"  Method:       {args.method}")
    print(f"  Text prompts: {text_prompts}")
    print(f"  Threshold:    {args.threshold}")
    if args.method == "dino+sam":
        print(f"  DINO model:   {args.dino_model}")
        print(f"  Input size:   {args.input_size}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_dir}")

    sam_processor = sam3_load_model(device)

    dino_model, dino_processor = None, None
    if args.method == "dino+sam":
        dino_model, dino_processor = sam3_load_dino(args.dino_model, device)

    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = sam3_process_image(
            args.image, sam_processor, args.method, text_prompts, args.threshold,
            output_dir, dino_model=dino_model, dino_processor=dino_processor,
            input_size=args.input_size, device=device, debug=True
        )
        if result:
            print(f"\n  Ground: {result['ground_pct']:.1f}%")
            print(f"  Saved to: {output_dir}")
        results = [result] if result else []
    elif args.data_dir:
        results = sam3_process_folder(
            args.data_dir, args.cameras, sam_processor, args.method, text_prompts,
            args.threshold, output_dir, args.sample, dino_model=dino_model,
            dino_processor=dino_processor, input_size=args.input_size,
            device=device, debug=args.debug
        )
    elif args.canonical_dir:
        results = sam3_process_canonical_views(
            args.canonical_dir, sam_processor, args.method, text_prompts,
            args.threshold, output_dir, dino_model=dino_model,
            dino_processor=dino_processor, input_size=args.input_size,
            device=device, debug=True
        )

    _write_summary_csv(results, output_dir, "sam3_dino_summary.csv",
                        ["image", "camera", "filename", "ground_pct", "method"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# DINOv2 PCA — zero-shot, no training
# =============================================================================
# Extracts DINOv2 patch features, applies PCA to reduce to 3 components, and
# uses the first principal component to separate ground from background.
# The ground usually dominates the bottom third of the frame in these
# downward-tilted webcam shots, which is used to resolve the sign ambiguity.

def dinov2_pca_load_model(model_name="facebook/dinov2-base", device="cuda"):
    """Load DINOv2 model and processor from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModel

    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"  DINOv2 loaded on {device}")
    return model, processor


@torch.inference_mode()
def dinov2_pca_extract_features(model, processor, image, device="cuda", input_size=518):
    """Extract DINOv2 patch features and apply PCA.

    Returns:
        pca_rgb:  (H_patch, W_patch, 3) float32, PCA components as RGB [0,1]
        pc1:      (H_patch, W_patch) float32, first principal component
        grid_h, grid_w: patch grid dimensions
    """
    from sklearn.decomposition import PCA

    resized = image.resize((input_size, input_size), Image.BILINEAR)
    inputs = processor(images=resized, return_tensors="pt", do_resize=False).to(device)
    pixel_values = inputs["pixel_values"]
    _, _, input_h, input_w = pixel_values.shape
    patch_size = model.config.patch_size
    grid_h = input_h // patch_size
    grid_w = input_w // patch_size

    outputs = model(**inputs, output_hidden_states=True)
    patch_features = outputs.last_hidden_state[:, 1:, :]
    features = patch_features.squeeze(0).reshape(grid_h, grid_w, -1)

    h, w, d = features.shape
    features_flat = features.cpu().numpy().reshape(-1, d)
    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(features_flat)

    pca_rgb = pca_feats.reshape(h, w, 3)
    for c in range(3):
        cmin, cmax = pca_rgb[:, :, c].min(), pca_rgb[:, :, c].max()
        if cmax - cmin > 0:
            pca_rgb[:, :, c] = (pca_rgb[:, :, c] - cmin) / (cmax - cmin)

    pc1 = pca_feats[:, 0].reshape(h, w)
    return pca_rgb.astype(np.float32), pc1.astype(np.float32), grid_h, grid_w


def dinov2_pca_extract_ground_mask(pc1, orig_size):
    """Extract ground mask from first principal component using Otsu +
    connected components. Assumes ground dominates the bottom third of the
    image (typical for these downward-tilted webcams).

    Args:
        pc1: (H_patch, W_patch) first principal component
        orig_size: (width, height) of original image
    """
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    h, w = pc1.shape
    thresh = threshold_otsu(pc1)
    binary = (pc1 > thresh).astype(int)

    bottom = binary[int(h * 0.66):, :]
    unique, counts = np.unique(bottom, return_counts=True)
    ground_label = unique[np.argmax(counts)]
    ground_mask_small = (binary == ground_label).astype(np.uint8)

    labeled, n_components = ndimage.label(ground_mask_small)
    if n_components > 1:
        bottom_labels = labeled[int(h * 0.66):, :]
        bottom_labels_flat = bottom_labels[bottom_labels > 0]
        if len(bottom_labels_flat) > 0:
            best_comp = np.bincount(bottom_labels_flat).argmax()
            ground_mask_small = (labeled == best_comp).astype(np.uint8)

    orig_w, orig_h = orig_size
    mask_pil = Image.fromarray(ground_mask_small * 255)
    mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
    ground_mask = np.array(mask_pil) > 127
    return ground_mask


def dinov2_pca_visualize(image, pca_rgb, ground_mask, output_path, title=""):
    """Save overview: original, PCA RGB visualization, ground mask, overlay."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    pca_display = np.array(Image.fromarray(
        (pca_rgb * 255).astype(np.uint8)
    ).resize(image.size, Image.BILINEAR))
    axes[0, 1].imshow(pca_display)
    axes[0, 1].set_title("DINOv2 PCA (PC1=R, PC2=G, PC3=B)")
    axes[0, 1].axis("off")

    ground_pct = ground_mask.sum() / ground_mask.size * 100
    axes[1, 0].imshow(ground_mask, cmap="gray")
    axes[1, 0].set_title(f"Ground Mask ({ground_pct:.1f}%)")
    axes[1, 0].axis("off")

    img_array = np.array(image)
    overlay = img_array.copy().astype(np.float32)
    overlay[ground_mask] = overlay[ground_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    overlay[~ground_mask] = overlay[~ground_mask] * 0.5 + np.array([100, 100, 100]) * 0.5
    axes[1, 1].imshow(overlay.astype(np.uint8))
    axes[1, 1].set_title("Overlay (green=ground)")
    axes[1, 1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def dinov2_pca_process_image(image_path, model, processor, output_dir, input_size=518,
                              device="cuda", debug=False):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    pca_rgb, pc1, grid_h, grid_w = dinov2_pca_extract_features(
        model, processor, image, device=device, input_size=input_size
    )
    ground_mask = dinov2_pca_extract_ground_mask(pc1, image.size)

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name
    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    # Save PCA visualization (always -- it's the key output of this method)
    pca_img = Image.fromarray((pca_rgb * 255).astype(np.uint8))
    pca_upsampled = pca_img.resize(image.size, Image.BILINEAR)
    pca_upsampled.save(cam_out / f"{stem}_pca.png")

    if debug:
        dinov2_pca_visualize(image, pca_rgb, ground_mask,
                             cam_out / f"{stem}_overview.png",
                             title=f"{camera} - {stem}")

    ground_pct = ground_mask.sum() / ground_mask.size * 100
    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name,
        "ground_pct": round(ground_pct, 2), "method": "dinov2_pca",
    }


def dinov2_pca_process_folder(data_dir, cameras, model, processor, output_dir,
                               input_size, sample, device="cuda", debug=False):
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
            result = dinov2_pca_process_image(img_path, model, processor, output_dir,
                                              input_size=input_size, device=device,
                                              debug=debug)
            if result:
                results.append(result)
    return results


def dinov2_pca_process_canonical_views(canonical_dir, model, processor, output_dir,
                                        input_size, device="cuda", debug=True):
    canonical_dir = Path(canonical_dir)
    results = []
    locations = sorted([d for d in canonical_dir.iterdir() if d.is_dir()])
    print(f"\nCanonical view locations: {len(locations)}")
    for loc_dir in locations:
        views = sorted(loc_dir.glob("canonical_view_*.jpg"))
        if not views:
            continue
        print(f"\n  {loc_dir.name}: {len(views)} canonical views")
        for view_path in views:
            result = dinov2_pca_process_image(view_path, model, processor, output_dir,
                                              input_size=input_size, device=device,
                                              debug=debug)
            if result:
                results.append(result)
    return results


def add_dinov2_pca_subparser(subparsers):
    p = subparsers.add_parser("dinov2_pca", help="DINOv2 PCA (zero-shot, no training)")
    _add_input_and_common_args(p, str(WORKSPACE_DIR / "segmentation" / "ground" / "dinov2_pca"))
    p.add_argument("--model", type=str, default="facebook/dinov2-base",
                    help="DINOv2 model (base, large, giant)")
    p.add_argument("--input-size", type=int, default=518,
                    help="Input resolution for DINOv2 (default: 518)")
    return p


def run_dinov2_pca(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DINOv2 PCA - ZERO-SHOT GROUND SEGMENTATION")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Input size: {args.input_size}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")

    model, processor = dinov2_pca_load_model(args.model, device)

    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = dinov2_pca_process_image(args.image, model, processor, output_dir,
                                          input_size=args.input_size, device=device,
                                          debug=True)
        if result:
            print(f"\n  Ground: {result['ground_pct']:.1f}%")
            print(f"  Saved to: {output_dir}")
        results = [result] if result else []
    elif args.data_dir:
        results = dinov2_pca_process_folder(args.data_dir, args.cameras, model, processor,
                                            output_dir, args.input_size, args.sample,
                                            device=device, debug=args.debug)
    elif args.canonical_dir:
        results = dinov2_pca_process_canonical_views(args.canonical_dir, model, processor,
                                                      output_dir, args.input_size,
                                                      device=device, debug=True)

    _write_summary_csv(results, output_dir, "dinov2_pca_summary.csv",
                        ["image", "camera", "filename", "ground_pct", "method"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# DEPTH ANYTHING V2 — depth-threshold ground segmentation
# =============================================================================

DEPTH_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def depth_load_model(model_size="large", device="cuda"):
    """Load Depth Anything V2 model and processor from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = DEPTH_MODELS[model_size]
    print(f"Loading {model_id}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model = model.to(device).eval()
    print(f"  Model loaded on {device}")
    return model, processor


@torch.no_grad()
def depth_predict(model, processor, image, device="cuda"):
    """Predict depth map for a PIL image.

    Returns:
        depth_map: np.ndarray (H, W) float32, normalized to [0, 1].
                   0 = nearest, 1 = farthest.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

    h, w = image.size[1], image.size[0]
    if predicted_depth.shape != (h, w):
        from scipy.ndimage import zoom
        scale_h = h / predicted_depth.shape[0]
        scale_w = w / predicted_depth.shape[1]
        predicted_depth = zoom(predicted_depth, (scale_h, scale_w), order=1)

    # Depth Anything V2 outputs disparity (near=large values); normalize and
    # invert so that 0=near, 1=far (more intuitive for thresholding).
    d_min, d_max = predicted_depth.min(), predicted_depth.max()
    if d_max - d_min > 0:
        depth_norm = (predicted_depth - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(predicted_depth)
    depth_norm = 1.0 - depth_norm
    return depth_norm.astype(np.float32)


def depth_segment_ground(depth_map, method="otsu", threshold=0.4,
                          near_clip=0.05, far_clip=0.95):
    """Separate ground (foreground) from background using depth, excluding
    very near occlusions (watermarks, borders) and far background (sky).

    Args:
        depth_map: (H, W) float32, 0=near, 1=far
        method: "threshold" (percentile cutoff) or "otsu" (automatic bimodal)
        threshold: for "threshold" method, fraction of depth range considered ground
        near_clip: depth percentile below which pixels are rejected as occlusions
        far_clip:  depth percentile above which pixels are rejected as background
    """
    near_val = np.percentile(depth_map, near_clip * 100) if near_clip > 0 else -1
    far_val = np.percentile(depth_map, far_clip * 100) if far_clip < 1 else 2

    valid_mask = (depth_map > near_val) & (depth_map < far_val)
    valid_depths = depth_map[valid_mask]

    if method == "threshold":
        cutoff = np.percentile(valid_depths, threshold * 100)
        ground_mask = (depth_map <= cutoff) & valid_mask
    elif method == "otsu":
        from skimage.filters import threshold_otsu
        try:
            otsu_val = threshold_otsu(valid_depths)
            ground_mask = (depth_map <= otsu_val) & valid_mask
        except ValueError:
            ground_mask = (depth_map <= 0.5) & valid_mask
    else:
        raise ValueError(f"Unknown method: {method}. Use 'threshold' or 'otsu'.")

    return ground_mask


def depth_visualize(image, depth_map, ground_mask, output_path, title=""):
    """Save a 2x2 overview: original, depth colormap, ground mask, overlay."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(depth_map, cmap="inferno")
    axes[0, 1].set_title("Depth Map (dark=near, bright=far)")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    axes[1, 0].imshow(ground_mask, cmap="gray")
    ground_pct = ground_mask.sum() / ground_mask.size * 100
    axes[1, 0].set_title(f"Ground Mask ({ground_pct:.1f}% of image)")
    axes[1, 0].axis("off")

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


def depth_process_image(image_path, model, processor, output_dir, method, threshold,
                         near_clip=0.05, far_clip=0.95, device="cuda", debug=False):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    depth_map = depth_predict(model, processor, image, device=device)
    ground_mask = depth_segment_ground(depth_map, method=method, threshold=threshold,
                                       near_clip=near_clip, far_clip=far_clip)

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name
    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    if debug:
        import matplotlib.pyplot as plt
        plt.imsave(str(cam_out / f"{stem}_depth.png"), depth_map, cmap="inferno")
        depth_visualize(image, depth_map, ground_mask,
                        cam_out / f"{stem}_overview.png",
                        title=f"{camera} - {stem}")

    ground_pct = ground_mask.sum() / ground_mask.size * 100
    mean_depth = float(depth_map.mean())
    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name,
        "ground_pct": round(ground_pct, 2),
        "mean_depth": round(mean_depth, 4), "method": method,
    }


def depth_process_folder(data_dir, cameras, model, processor, output_dir, method,
                          threshold, sample, near_clip=0.05, far_clip=0.95,
                          device="cuda", debug=False):
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
            result = depth_process_image(img_path, model, processor, output_dir,
                                         method, threshold, near_clip=near_clip,
                                         far_clip=far_clip, device=device, debug=debug)
            if result:
                results.append(result)
    return results


def depth_process_canonical_views(canonical_dir, model, processor, output_dir,
                                   method, threshold, near_clip=0.05, far_clip=0.95,
                                   device="cuda", debug=True):
    canonical_dir = Path(canonical_dir)
    results = []
    locations = sorted([d for d in canonical_dir.iterdir() if d.is_dir()])
    print(f"\nCanonical view locations: {len(locations)}")
    for loc_dir in locations:
        views = sorted(loc_dir.glob("canonical_view_*.jpg"))
        if not views:
            continue
        print(f"\n  {loc_dir.name}: {len(views)} canonical views")
        for view_path in views:
            result = depth_process_image(view_path, model, processor, output_dir,
                                         method, threshold, near_clip=near_clip,
                                         far_clip=far_clip, device=device, debug=debug)
            if result:
                results.append(result)
    return results


def _depth_collect_image_paths(args, cameras):
    """Collect image paths based on input mode and sampling (grid-search mode)."""
    paths = []
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not cameras:
            cameras = sorted([
                d.name for d in data_dir.iterdir()
                if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)
            ])
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
            print(f"  {cam}: {len(jpgs)} images")
            paths.extend(jpgs)
    elif args.canonical_dir:
        canonical_dir = Path(args.canonical_dir)
        for loc_dir in sorted(d for d in canonical_dir.iterdir() if d.is_dir()):
            views = sorted(loc_dir.glob("canonical_view_*.jpg"))
            if views:
                print(f"  {loc_dir.name}: {len(views)} canonical views")
                paths.extend(views)
    elif args.image:
        paths = [Path(args.image)]
    return paths


def depth_run_grid(args, model, processor, device):
    """Predict depth once per image, then re-segment with every
    (near_clip, far_clip) combination from the grid."""
    import matplotlib.pyplot as plt

    base_output_dir = Path(args.output_dir)
    near_values = sorted(args.grid_near)
    far_values = sorted(args.grid_far)
    combos = [(n, f) for n in near_values for f in far_values]

    print(f"\nGrid: {len(combos)} combinations")
    print(f"  Near clips: {near_values}")
    print(f"  Far clips:  {far_values}")
    for n, f in combos:
        print(f"    N{n}_F{f}")

    combo_dirs = {}
    for n, f in combos:
        tag = f"N{n}_F{f}"
        d = base_output_dir / tag
        d.mkdir(parents=True, exist_ok=True)
        combo_dirs[(n, f)] = d

    print("\nCollecting images...")
    image_paths = _depth_collect_image_paths(args, args.cameras)
    if not image_paths:
        print("No images found.")
        return

    print(f"\nTotal images: {len(image_paths)}")
    print(f"Depth predicted once, segmented {len(combos)}x each\n")

    all_results = {combo: [] for combo in combos}

    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Warning: skipping {img_path}: {e}")
            continue

        depth_map = depth_predict(model, processor, image, device=device)

        stem = img_path.stem
        camera = img_path.parent.name
        mean_depth = float(depth_map.mean())

        shared_depth_dir = base_output_dir / "shared_depth" / camera
        shared_depth_dir.mkdir(parents=True, exist_ok=True)
        depth_path = shared_depth_dir / f"{stem}_depth.png"
        if not depth_path.exists():
            plt.imsave(str(depth_path), depth_map, cmap="inferno")

        for (n, f) in combos:
            ground_mask = depth_segment_ground(
                depth_map, method=args.method, threshold=args.threshold,
                near_clip=n, far_clip=f
            )

            cam_out = combo_dirs[(n, f)] / camera
            cam_out.mkdir(parents=True, exist_ok=True)

            mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
            mask_img.save(cam_out / f"{stem}_ground.png")

            if args.debug:
                depth_visualize(image, depth_map, ground_mask,
                                cam_out / f"{stem}_overview.png",
                                title=f"{camera} - {stem} (N{n}_F{f})")

            ground_pct = ground_mask.sum() / ground_mask.size * 100
            all_results[(n, f)].append({
                "image": str(img_path), "camera": camera,
                "filename": img_path.name,
                "ground_pct": round(ground_pct, 2),
                "mean_depth": round(mean_depth, 4), "method": args.method,
            })

    for (n, f), results in all_results.items():
        if not results:
            continue
        tag = f"N{n}_F{f}"
        summary_path = combo_dirs[(n, f)] / "depth_summary.csv"
        fieldnames = ["image", "camera", "filename", "ground_pct", "mean_depth", "method"]
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  {tag}: {len(results)} images, summary saved")

    print(f"\nRunning comparison across {len(combos)} parameter sets...")
    depth_compare_runs(base_output_dir, args.cameras if args.cameras else None)

    print(f"\nGrid search complete! {len(image_paths)} images x {len(combos)} combos")


def depth_compare_runs(base_output_dir, cameras_filter=None):
    """Find all N*_F* parameter run folders, and for each image that appears
    across runs, produce a side-by-side comparison figure.

    Output structure: base_output_dir/comparisons/<camera>/<stem>_compare.png
    """
    import matplotlib.pyplot as plt

    base_output_dir = Path(base_output_dir)
    run_dirs = sorted([
        d for d in base_output_dir.iterdir()
        if d.is_dir() and re.match(r"N[\d.]+_F[\d.]+", d.name)
    ])

    if len(run_dirs) < 2:
        print(f"Need at least 2 parameter runs to compare, found {len(run_dirs)}:")
        for d in run_dirs:
            print(f"  {d.name}")
        return

    run_labels = [d.name for d in run_dirs]
    print(f"Found {len(run_dirs)} runs: {run_labels}")

    first_run = run_dirs[0]
    all_cameras = sorted([
        d.name for d in first_run.iterdir()
        if d.is_dir() and re.match(r"C[A-Z]", d.name)
    ])

    if cameras_filter:
        all_cameras = [c for c in all_cameras if c in cameras_filter]

    if not all_cameras:
        all_cameras = sorted([d.name for d in first_run.iterdir() if d.is_dir()])

    print(f"Cameras: {all_cameras}")

    compare_dir = base_output_dir / "comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)
    total_figures = 0

    for camera in all_cameras:
        cam_compare_dir = compare_dir / camera
        cam_compare_dir.mkdir(parents=True, exist_ok=True)

        first_cam_dir = first_run / camera
        if not first_cam_dir.exists():
            continue

        ground_masks = sorted(first_cam_dir.glob("*_ground.png"))
        if not ground_masks:
            continue

        print(f"\n  {camera}: {len(ground_masks)} images to compare")

        for mask_path in ground_masks:
            stem = mask_path.stem.replace("_ground", "")

            all_exist = True
            for rd in run_dirs:
                if not (rd / camera / f"{stem}_ground.png").exists():
                    all_exist = False
                    break
            if not all_exist:
                continue

            depth_img = None
            for rd in run_dirs:
                depth_path = rd / camera / f"{stem}_depth.png"
                if depth_path.exists() and depth_img is None:
                    depth_img = np.array(Image.open(depth_path))
                break

            n_runs = len(run_dirs)
            fig, axes = plt.subplots(1, n_runs + 1, figsize=(4 * (n_runs + 1), 4))

            if depth_img is not None:
                axes[0].imshow(depth_img)
                axes[0].set_title("Depth Map", fontsize=10)
            else:
                axes[0].text(0.5, 0.5, "No depth\navailable",
                             ha="center", va="center", transform=axes[0].transAxes)
                axes[0].set_title("Depth Map", fontsize=10)
            axes[0].axis("off")

            for i, (rd, label) in enumerate(zip(run_dirs, run_labels)):
                mask_file = rd / camera / f"{stem}_ground.png"
                mask = np.array(Image.open(mask_file))
                ground_pct = (mask > 127).sum() / mask.size * 100

                axes[i + 1].imshow(mask, cmap="gray")
                axes[i + 1].set_title(f"{label}\n{ground_pct:.1f}% ground", fontsize=9)
                axes[i + 1].axis("off")

            fig.suptitle(f"{camera} - {stem}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            out_path = cam_compare_dir / f"{stem}_compare.png"
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close()
            total_figures += 1

    if total_figures > 0:
        _depth_create_summary_grid(base_output_dir, run_dirs, run_labels,
                                   all_cameras, compare_dir)

    print(f"\nSaved {total_figures} comparison figures to {compare_dir}/")


def _depth_create_summary_grid(base_output_dir, run_dirs, run_labels, cameras,
                               compare_dir):
    """One big grid: rows=cameras, columns=parameter runs. Uses the first
    available ground mask per camera as the representative."""
    import matplotlib.pyplot as plt

    representatives = []
    for camera in cameras:
        cam_dir = run_dirs[0] / camera
        if not cam_dir.exists():
            continue
        masks = sorted(cam_dir.glob("*_ground.png"))
        if masks:
            stem = masks[0].stem.replace("_ground", "")
            if all((rd / camera / f"{stem}_ground.png").exists() for rd in run_dirs):
                representatives.append((camera, stem))

    if not representatives:
        return

    n_cams = len(representatives)
    n_runs = len(run_dirs)
    fig, axes = plt.subplots(n_cams, n_runs, figsize=(3.5 * n_runs, 3 * n_cams))

    if n_cams == 1:
        axes = axes[np.newaxis, :]
    if n_runs == 1:
        axes = axes[:, np.newaxis]

    for row, (camera, stem) in enumerate(representatives):
        for col, (rd, label) in enumerate(zip(run_dirs, run_labels)):
            mask_file = rd / camera / f"{stem}_ground.png"
            mask = np.array(Image.open(mask_file))
            ground_pct = (mask > 127).sum() / mask.size * 100

            axes[row, col].imshow(mask, cmap="gray")
            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(label, fontsize=10, fontweight="bold")
            if col == 0:
                axes[row, col].set_ylabel(camera, fontsize=10, fontweight="bold",
                                          rotation=0, labelpad=60, va="center")

            axes[row, col].text(0.98, 0.02, f"{ground_pct:.0f}%",
                                transform=axes[row, col].transAxes,
                                ha="right", va="bottom", fontsize=9, color="white",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor="black", alpha=0.6))

    fig.suptitle("Parameter Comparison - Ground Masks", fontsize=14, fontweight="bold")
    plt.tight_layout()
    summary_path = compare_dir / "summary_grid.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSummary grid saved to: {summary_path}")


def add_depth_subparser(subparsers):
    p = subparsers.add_parser("depth", help="Depth Anything V2 (depth-threshold ground)")
    _add_input_and_common_args(p, str(WORKSPACE_DIR / "segmentation" / "ground" / "depth"))
    p.add_argument("--model-size", choices=["small", "base", "large"], default="large",
                    help="Depth Anything V2 model size")
    p.add_argument("--method", choices=["threshold", "otsu"], default="otsu",
                    help="Ground segmentation method")
    p.add_argument("--threshold", type=float, default=0.4,
                    help="Depth percentile for threshold method (default 0.4)")
    p.add_argument("--near-clip", type=float, default=0.05,
                    help="Depth percentile below which pixels are rejected as "
                         "occlusions/watermarks (default 0.05 = bottom 5%%)")
    p.add_argument("--far-clip", type=float, default=0.95,
                    help="Depth percentile above which pixels are rejected as "
                         "sky/far background (default 0.95 = top 5%%)")
    p.add_argument("--grid", action="store_true",
                    help="Run a grid of near/far clip values. Depth is predicted "
                         "once, segmentation repeated for each combo. Overrides "
                         "--near-clip/--far-clip.")
    p.add_argument("--grid-near", nargs="+", type=float, default=[0.05, 0.10, 0.15],
                    help="Near clip values for grid (default: 0.05 0.10 0.15)")
    p.add_argument("--grid-far", nargs="+", type=float, default=[0.85, 0.90, 0.95],
                    help="Far clip values for grid (default: 0.85 0.90 0.95)")
    p.add_argument("--compare", action="store_true",
                    help="Compare all parameter runs side-by-side "
                         "(looks for N*_F* folders in output-dir)")
    p.add_argument("--compare-cameras", nargs="+", default=[],
                    help="Limit comparison to specific cameras")
    return p


def run_depth(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_output_dir = Path(args.output_dir)

    if args.compare:
        print("=" * 70)
        print("COMPARISON MODE - Side-by-side parameter runs")
        print("=" * 70)
        depth_compare_runs(base_output_dir, args.compare_cameras)
        return

    if args.grid:
        print("=" * 70)
        print("DEPTH ANYTHING V2 - GRID SEARCH")
        print("=" * 70)
        print(f"  Model:   {args.model_size} ({DEPTH_MODELS[args.model_size]})")
        print(f"  Method:  {args.method}")
        print(f"  Device:  {device}")
        print(f"  Output:  {base_output_dir}")
        model, processor = depth_load_model(args.model_size, device)
        depth_run_grid(args, model, processor, device)
        return

    param_tag = f"N{args.near_clip}_F{args.far_clip}"
    output_dir = base_output_dir / param_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DEPTH ANYTHING V2 - GROUND SEGMENTATION")
    print("=" * 70)
    print(f"  Model:   {args.model_size} ({DEPTH_MODELS[args.model_size]})")
    print(f"  Method:  {args.method}" + (f" (percentile={args.threshold})"
          if args.method == "threshold" else ""))
    print(f"  Clips:   near={args.near_clip} far={args.far_clip}")
    print(f"  Tag:     {param_tag}")
    print(f"  Device:  {device}")
    print(f"  Output:  {output_dir}")

    model, processor = depth_load_model(args.model_size, device)

    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = depth_process_image(args.image, model, processor, output_dir,
                                     args.method, args.threshold,
                                     near_clip=args.near_clip, far_clip=args.far_clip,
                                     device=device, debug=True)
        if result:
            print(f"\n  Ground: {result['ground_pct']:.1f}%")
            print(f"  Mean depth: {result['mean_depth']:.4f}")
            print(f"  Saved to: {output_dir}")
        results = [result] if result else []
    elif args.data_dir:
        results = depth_process_folder(args.data_dir, args.cameras, model, processor,
                                       output_dir, args.method, args.threshold,
                                       args.sample, near_clip=args.near_clip,
                                       far_clip=args.far_clip, device=device,
                                       debug=args.debug)
    elif args.canonical_dir:
        results = depth_process_canonical_views(args.canonical_dir, model, processor,
                                                 output_dir, args.method, args.threshold,
                                                 near_clip=args.near_clip,
                                                 far_clip=args.far_clip,
                                                 device=device, debug=True)

    _write_summary_csv(results, output_dir, "depth_summary.csv",
                        ["image", "camera", "filename", "ground_pct", "mean_depth", "method"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ground segmentation teachers")
    subparsers = parser.add_subparsers(dest="teacher", required=True)

    add_deeplabv3_subparser(subparsers)
    add_segformer_subparser(subparsers)
    add_sam3_subparser(subparsers)
    add_dinov2_pca_subparser(subparsers)
    add_depth_subparser(subparsers)

    args = parser.parse_args()

    {
        "deeplabv3": run_deeplabv3,
        "segformer": run_segformer,
        "sam3": run_sam3,
        "dinov2_pca": run_dinov2_pca,
        "depth": run_depth,
    }[args.teacher](args)


if __name__ == "__main__":
    main()
