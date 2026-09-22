#!/usr/bin/env python3
"""
DINOv2 PCA zero-shot ground segmentation.

Extracts DINOv2 patch features, applies PCA to reduce to 3 components,
and uses the first principal component to separate ground from background.
No training required — fully zero-shot.

The first 3 PCA components typically correspond to the most dominant regions
in the image. Since the ground is usually the largest contiguous area in a
webcam shot, it shows up as a solid, distinct color in PCA visualization.

Usage:
    # Single image (with debug visualizations)
    python dinov2_pca_segmentation.py --image /path/to/image.jpg --debug

    # All cameras
    python dinov2_pca_segmentation.py --data-dir data

    # Specific cameras
    python dinov2_pca_segmentation.py --data-dir data \
        --cameras CM00006 CM00024 --sample 50

    # Use larger DINOv2 model
    python dinov2_pca_segmentation.py --data-dir data \
        --model facebook/dinov2-large
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ── model loading ────────────────────────────────────────────────────────────

def load_dino(model_name="facebook/dinov2-base", device="cuda"):
    """Load DINOv2 model and processor from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModel

    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"  DINOv2 loaded on {device}")
    return model, processor


# ── feature extraction + PCA ─────────────────────────────────────────────────

@torch.inference_mode()
def extract_pca_features(model, processor, image, device="cuda", input_size=518):
    """
    Extract DINOv2 patch features and apply PCA.

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
    # Remove CLS token, reshape to spatial grid
    patch_features = outputs.last_hidden_state[:, 1:, :]
    features = patch_features.squeeze(0).reshape(grid_h, grid_w, -1)

    # PCA on patch features → 3 components
    h, w, d = features.shape
    features_flat = features.cpu().numpy().reshape(-1, d)
    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(features_flat)

    # Normalize each component to [0, 1] for RGB visualization
    pca_rgb = pca_feats.reshape(h, w, 3)
    for c in range(3):
        cmin, cmax = pca_rgb[:, :, c].min(), pca_rgb[:, :, c].max()
        if cmax - cmin > 0:
            pca_rgb[:, :, c] = (pca_rgb[:, :, c] - cmin) / (cmax - cmin)

    pc1 = pca_feats[:, 0].reshape(h, w)

    return pca_rgb.astype(np.float32), pc1.astype(np.float32), grid_h, grid_w


# ── ground mask extraction ───────────────────────────────────────────────────

def extract_ground_mask(pc1, orig_size):
    """
    Extract ground mask from first principal component using Otsu + connected components.

    Assumes ground dominates the bottom third of the image (typical for webcams).

    Args:
        pc1: (H_patch, W_patch) first principal component
        orig_size: (width, height) of original image

    Returns:
        ground_mask: (H, W) bool at original resolution
    """
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    h, w = pc1.shape

    # Otsu threshold on PC1
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

    # Upsample to original resolution
    orig_w, orig_h = orig_size
    mask_pil = Image.fromarray(ground_mask_small * 255)
    mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
    ground_mask = np.array(mask_pil) > 127

    return ground_mask


# ── visualization ────────────────────────────────────────────────────────────

def visualize_pca(image, pca_rgb, ground_mask, output_path, title=""):
    """Save overview: original, PCA RGB visualization, ground mask, overlay."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # PCA RGB (upsampled for display)
    pca_display = np.array(Image.fromarray(
        (pca_rgb * 255).astype(np.uint8)
    ).resize(image.size, Image.BILINEAR))
    axes[0, 1].imshow(pca_display)
    axes[0, 1].set_title("DINOv2 PCA (PC1=R, PC2=G, PC3=B)")
    axes[0, 1].axis("off")

    # Ground mask
    ground_pct = ground_mask.sum() / ground_mask.size * 100
    axes[1, 0].imshow(ground_mask, cmap="gray")
    axes[1, 0].set_title(f"Ground Mask ({ground_pct:.1f}%)")
    axes[1, 0].axis("off")

    # Overlay
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


# ── processing ───────────────────────────────────────────────────────────────

def process_image(image_path, model, processor, output_dir, input_size=518,
                  device="cuda", debug=False):
    """Process a single image: extract PCA, segment ground, save outputs."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    pca_rgb, pc1, grid_h, grid_w = extract_pca_features(
        model, processor, image, device=device, input_size=input_size
    )
    ground_mask = extract_ground_mask(pc1, image.size)

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    # Save ground mask
    mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    # Save PCA visualization (always — it's the key output of this method)
    pca_img = Image.fromarray((pca_rgb * 255).astype(np.uint8))
    pca_upsampled = pca_img.resize(image.size, Image.BILINEAR)
    pca_upsampled.save(cam_out / f"{stem}_pca.png")

    if debug:
        visualize_pca(image, pca_rgb, ground_mask,
                      cam_out / f"{stem}_overview.png",
                      title=f"{camera} — {stem}")

    ground_pct = ground_mask.sum() / ground_mask.size * 100

    return {
        "image": str(image_path),
        "camera": camera,
        "filename": Path(image_path).name,
        "ground_pct": round(ground_pct, 2),
        "method": "dinov2_pca",
    }


def process_folder(data_dir, cameras, model, processor, output_dir,
                   input_size, sample, device="cuda", debug=False):
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
            result = process_image(img_path, model, processor, output_dir,
                                   input_size=input_size, device=device,
                                   debug=debug)
            if result:
                results.append(result)

    return results


def process_canonical_views(canonical_dir, model, processor, output_dir,
                            input_size, device="cuda", debug=True):
    """Process canonical view images."""
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
            result = process_image(view_path, model, processor, output_dir,
                                   input_size=input_size, device=device,
                                   debug=debug)
            if result:
                results.append(result)

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 PCA zero-shot ground segmentation"
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
                        default="preprocessing/segmentation/dinov2_pca_output",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="facebook/dinov2-base",
                        help="DINOv2 model (base, large, giant)")
    parser.add_argument("--input-size", type=int, default=518,
                        help="Input resolution for DINOv2 (default: 518)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Images to sample per camera (0=all)")
    parser.add_argument("--debug", action="store_true",
                        help="Save PCA overview figures")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DINOv2 PCA — ZERO-SHOT GROUND SEGMENTATION")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Input size: {args.input_size}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")

    model, processor = load_dino(args.model, device)

    # Dispatch
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = process_image(args.image, model, processor, output_dir,
                               input_size=args.input_size, device=device,
                               debug=True)
        if result:
            print(f"\n  Ground: {result['ground_pct']:.1f}%")
            print(f"  Saved to: {output_dir}")
        results = [result] if result else []

    elif args.data_dir:
        results = process_folder(args.data_dir, args.cameras, model, processor,
                                 output_dir, args.input_size, args.sample,
                                 device=device, debug=args.debug)

    elif args.canonical_dir:
        results = process_canonical_views(args.canonical_dir, model, processor,
                                          output_dir, args.input_size,
                                          device=device, debug=True)

    # Save summary
    if results:
        summary_path = output_dir / "dinov2_pca_summary.csv"
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
