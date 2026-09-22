#!/usr/bin/env python3
"""
SAM3 + DINOv2 ground segmentation.

Two modes:
  - text:     SAM3 text prompt ("ground, terrain, earth") — simplest, no DINOv2
  - dino+sam: DINOv2 PCA rough mask → bounding box → SAM3 box prompt refinement

Usage:
    # Text-only (default)
    python sam3_dino_segmentation.py --image /path/to/image.jpg --debug

    # DINO-guided SAM3
    python sam3_dino_segmentation.py --image /path/to/image.jpg --method dino+sam --debug

    # Batch
    python sam3_dino_segmentation.py --data-dir data \
        --cameras CM00006 CM00024 --sample 50

    # Canonical views
    python sam3_dino_segmentation.py \
        --canonical-dir dinov3/canonical_views
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

# Add sam3 and dinov3 to path
SAM3_DIR = Path(__file__).resolve().parent.parent.parent / "sam3"
DINOV3_DIR = Path(__file__).resolve().parent.parent.parent / "dinov3"
sys.path.insert(0, str(SAM3_DIR))
sys.path.insert(0, str(DINOV3_DIR))


# ── SAM3 loading ─────────────────────────────────────────────────────────────

def load_sam3(device="cuda"):
    """Load SAM3 model and processor."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3 model from HuggingFace...")
    model = build_sam3_image_model(
        checkpoint_path=None, load_from_HF=True, device=device
    )
    processor = Sam3Processor(model, device=device)
    print(f"  SAM3 loaded on {device}")
    return processor


# ── DINOv2 loading ───────────────────────────────────────────────────────────

def load_dino(model_name="facebook/dinov2-base", device="cuda"):
    """Load DINOv2 model and processor from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModel

    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"  DINOv2 loaded on {device}")
    return model, processor


# ── DINOv2 PCA ground mask ──────────────────────────────────────────────────

def dino_pca_ground_mask(dino_model, dino_processor, image, device="cuda",
                         input_size=518):
    """
    Extract rough ground mask from DINOv2 PCA features.
    Returns binary mask at original image resolution + bounding box (x0,y0,x1,y1) normalized [0,1].
    """
    from sklearn.decomposition import PCA
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # Extract patch features
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
        # Remove CLS token
        patch_features = hidden_states[:, 1:, :]
        features = patch_features.squeeze(0).reshape(grid_h, grid_w, -1)

    # PCA on patch features
    h, w, d = features.shape
    features_flat = features.cpu().numpy().reshape(-1, d)
    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(features_flat)

    # PC1 + Otsu threshold
    pc1 = pca_feats[:, 0].reshape(h, w)
    thresh = threshold_otsu(pc1)
    binary = (pc1 > thresh).astype(int)

    # Pick label dominating bottom third = ground
    bottom = binary[int(h * 0.66):, :]
    unique, counts = np.unique(bottom, return_counts=True)
    ground_label = unique[np.argmax(counts)]
    ground_mask_small = (binary == ground_label).astype(np.uint8)

    # Keep largest connected component touching bottom
    labeled, n_components = ndimage.label(ground_mask_small)
    if n_components > 1:
        bottom_labels = labeled[int(h * 0.66):, :]
        bottom_labels_flat = bottom_labels[bottom_labels > 0]
        if len(bottom_labels_flat) > 0:
            best_comp = np.bincount(bottom_labels_flat).argmax()
            ground_mask_small = (labeled == best_comp).astype(np.uint8)

    # Compute bounding box in normalized [0,1] coords (cx, cy, w, h format for SAM3)
    ys, xs = np.where(ground_mask_small > 0)
    if len(ys) == 0:
        return None, None

    y0, y1 = ys.min() / h, ys.max() / h
    x0, x1 = xs.min() / w, xs.max() / w
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    bw = x1 - x0
    bh = y1 - y0
    box_cxcywh = [cx, cy, bw, bh]

    # Upsample mask to original resolution
    orig_h, orig_w = image.size[1], image.size[0]
    from PIL import Image as PILImage
    mask_pil = PILImage.fromarray(ground_mask_small * 255)
    mask_pil = mask_pil.resize((orig_w, orig_h), PILImage.NEAREST)
    rough_mask = np.array(mask_pil) > 127

    return rough_mask, box_cxcywh


# ── Segmentation methods ────────────────────────────────────────────────────

def segment_text_only(sam_processor, pil_image, text_prompts, threshold=0.3):
    """
    Segment ground using SAM3 text prompts.
    Runs each prompt separately and combines all masks.
    """
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


def segment_dino_guided(sam_processor, dino_model, dino_processor, pil_image,
                        input_size=518, threshold=0.3, device="cuda"):
    """
    DINOv2 PCA rough mask → bounding box → SAM3 box prompt refinement.
    Also combines with text prompt "ground" for best results.
    """
    # Step 1: DINOv2 PCA rough mask + bounding box
    rough_mask, box_cxcywh = dino_pca_ground_mask(
        dino_model, dino_processor, pil_image, device=device,
        input_size=input_size
    )

    if box_cxcywh is None:
        # Fallback to text-only
        return segment_text_only(
            sam_processor, pil_image, ["ground"], threshold=threshold
        ), rough_mask

    # Step 2: SAM3 with text + box prompt
    sam_processor.confidence_threshold = threshold
    state = sam_processor.set_image(pil_image)

    # Set text context first
    state = sam_processor.set_text_prompt("ground", state)

    # Add box prompt from DINOv2 PCA bounding box
    state = sam_processor.add_geometric_prompt(box_cxcywh, label=True, state=state)

    if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
        masks = state["masks"]
        refined_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8)
    else:
        # Fallback to rough mask
        refined_mask = rough_mask.astype(np.uint8) if rough_mask is not None else np.zeros(
            (pil_image.size[1], pil_image.size[0]), dtype=np.uint8
        )

    return refined_mask, rough_mask


# ── Visualization ────────────────────────────────────────────────────────────

def visualize_sam3(image, ground_mask, output_path, title="",
                   rough_mask=None, method="text"):
    """Save overview figure."""
    import matplotlib.pyplot as plt

    n_cols = 3 if rough_mask is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    col = 1
    # Rough DINO mask (only in dino+sam mode)
    if rough_mask is not None:
        axes[col].imshow(rough_mask, cmap="gray")
        axes[col].set_title("DINOv2 PCA rough mask")
        axes[col].axis("off")
        col += 1

    # Ground mask
    ground_pct = ground_mask.sum() / ground_mask.size * 100
    axes[col].imshow(ground_mask, cmap="gray")
    axes[col].set_title(f"SAM3 Ground Mask ({ground_pct:.1f}%)")
    axes[col].axis("off")
    col += 1

    # Overlay
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


# ── Processing functions ─────────────────────────────────────────────────────

def process_image(image_path, sam_processor, method, text_prompts, threshold,
                  output_dir, dino_model=None, dino_processor=None,
                  input_size=518, device="cuda", debug=False):
    """Process a single image. Returns result dict or None."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    rough_mask = None
    if method == "dino+sam" and dino_model is not None:
        ground_mask, rough_mask = segment_dino_guided(
            sam_processor, dino_model, dino_processor, image,
            input_size=input_size, threshold=threshold, device=device
        )
    else:
        ground_mask = segment_text_only(
            sam_processor, image, text_prompts, threshold=threshold
        )

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    # Save ground mask
    mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    if debug:
        visualize_sam3(image, ground_mask,
                       cam_out / f"{stem}_overview.png",
                       title=f"{camera} — {stem} ({method})",
                       rough_mask=rough_mask, method=method)

    ground_pct = (ground_mask > 0).sum() / ground_mask.size * 100

    return {
        "image": str(image_path),
        "camera": camera,
        "filename": Path(image_path).name,
        "ground_pct": round(ground_pct, 2),
        "method": f"sam3_{method}",
    }


def process_folder(data_dir, cameras, sam_processor, method, text_prompts,
                   threshold, output_dir, sample, dino_model=None,
                   dino_processor=None, input_size=518, device="cuda",
                   debug=False):
    """Process all images in camera folders."""
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
            result = process_image(
                img_path, sam_processor, method, text_prompts, threshold,
                output_dir, dino_model=dino_model,
                dino_processor=dino_processor, input_size=input_size,
                device=device, debug=debug
            )
            if result:
                results.append(result)

    return results


def process_canonical_views(canonical_dir, sam_processor, method, text_prompts,
                            threshold, output_dir, dino_model=None,
                            dino_processor=None, input_size=518,
                            device="cuda", debug=True):
    """Process canonical view images from the DINOv3 pipeline."""
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
            result = process_image(
                view_path, sam_processor, method, text_prompts, threshold,
                output_dir, dino_model=dino_model,
                dino_processor=dino_processor, input_size=input_size,
                device=device, debug=debug
            )
            if result:
                results.append(result)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 + DINOv2 ground segmentation"
    )

    # Input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Single image path")
    input_group.add_argument("--data-dir", type=str,
                             help="Data directory with camera folders")
    input_group.add_argument("--canonical-dir", type=str,
                             help="Canonical views directory")

    # Options
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific camera folders (default: all)")
    parser.add_argument("--output-dir", type=str,
                        default="preprocessing/segmentation/sam3_dino_output",
                        help="Output directory")
    parser.add_argument("--method", choices=["text", "dino+sam"],
                        default="text",
                        help="Segmentation method (default: text)")
    parser.add_argument("--text-prompts", type=str,
                        default="ground,terrain,earth,road,grass",
                        help="Comma-separated text prompts for SAM3 "
                             "(default: ground,terrain,earth,road,grass)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="SAM3 confidence threshold (default: 0.3)")
    parser.add_argument("--dino-model", type=str,
                        default="facebook/dinov2-base",
                        help="DINOv2 model for dino+sam mode")
    parser.add_argument("--input-size", type=int, default=518,
                        help="DINOv2 input resolution (default: 518)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Images to sample per camera (0=all)")
    parser.add_argument("--debug", action="store_true",
                        help="Save overview figures")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    text_prompts = [p.strip() for p in args.text_prompts.split(",")]

    print("=" * 70)
    print("SAM3 + DINOv2 — GROUND SEGMENTATION")
    print("=" * 70)
    print(f"  Method:       {args.method}")
    print(f"  Text prompts: {text_prompts}")
    print(f"  Threshold:    {args.threshold}")
    if args.method == "dino+sam":
        print(f"  DINO model:   {args.dino_model}")
        print(f"  Input size:   {args.input_size}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_dir}")

    # Load models
    sam_processor = load_sam3(device)

    dino_model, dino_processor = None, None
    if args.method == "dino+sam":
        dino_model, dino_processor = load_dino(args.dino_model, device)

    # Dispatch
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = process_image(
            args.image, sam_processor, args.method, text_prompts,
            args.threshold, output_dir, dino_model=dino_model,
            dino_processor=dino_processor, input_size=args.input_size,
            device=device, debug=True
        )
        if result:
            print(f"\n  Ground: {result['ground_pct']:.1f}%")
            print(f"  Saved to: {output_dir}")
        results = [result] if result else []

    elif args.data_dir:
        results = process_folder(
            args.data_dir, args.cameras, sam_processor, args.method,
            text_prompts, args.threshold, output_dir, args.sample,
            dino_model=dino_model, dino_processor=dino_processor,
            input_size=args.input_size, device=device, debug=args.debug
        )

    elif args.canonical_dir:
        results = process_canonical_views(
            args.canonical_dir, sam_processor, args.method, text_prompts,
            args.threshold, output_dir, dino_model=dino_model,
            dino_processor=dino_processor, input_size=args.input_size,
            device=device, debug=True
        )

    # Save summary
    if results:
        summary_path = output_dir / "sam3_dino_summary.csv"
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
