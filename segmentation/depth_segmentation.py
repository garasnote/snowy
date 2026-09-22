#!/usr/bin/env python3
"""
Depth-based ground segmentation using Depth Anything V2.

Predicts per-pixel depth maps and separates foreground ground from
background (mountains, sky) using depth thresholding or Otsu's method.

Usage:
    # Single image
    python depth_segmentation.py --image /path/to/image.jpg --debug

    # All cameras
    python depth_segmentation.py --data-dir data

    # Specific cameras, subsample
    python depth_segmentation.py --data-dir data \
        --cameras CM00006 CM00024 --sample 50

    # Run on canonical views from DINOv3 pipeline
    python depth_segmentation.py \
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

# ── model loading ────────────────────────────────────────────────────────────

DEPTH_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def load_depth_model(model_size="large", device="cuda"):
    """Load Depth Anything V2 model and processor from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = DEPTH_MODELS[model_size]
    print(f"Loading {model_id}...")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model = model.to(device).eval()

    print(f"  Model loaded on {device}")
    return model, processor


# ── depth prediction ─────────────────────────────────────────────────────────

@torch.no_grad()
def predict_depth(model, processor, image, device="cuda"):
    """
    Predict depth map for a PIL image.

    Returns:
        depth_map: np.ndarray (H, W) float32, normalized to [0, 1].
                   0 = nearest, 1 = farthest.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Get predicted depth and resize to original image size
    predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

    # Resize to original image dimensions
    h, w = image.size[1], image.size[0]  # PIL is (W, H)
    if predicted_depth.shape != (h, w):
        from scipy.ndimage import zoom
        scale_h = h / predicted_depth.shape[0]
        scale_w = w / predicted_depth.shape[1]
        predicted_depth = zoom(predicted_depth, (scale_h, scale_w), order=1)

    # Normalize to [0, 1] — some models output inverse depth (near=large)
    # Depth Anything V2 outputs disparity (near=large values)
    # Invert so that 0=near, 1=far
    d_min, d_max = predicted_depth.min(), predicted_depth.max()
    if d_max - d_min > 0:
        depth_norm = (predicted_depth - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(predicted_depth)

    # Depth Anything outputs disparity: high values = close.
    # Invert so 0=near, 1=far (more intuitive for thresholding).
    depth_norm = 1.0 - depth_norm

    return depth_norm.astype(np.float32)


# ── ground segmentation ─────────────────────────────────────────────────────

def segment_ground_from_depth(depth_map, method="otsu", threshold=0.4,
                              near_clip=0.05, far_clip=0.95):
    """
    Separate ground (foreground) from background using depth,
    excluding very near occlusions (watermarks, borders) and far background (sky).

    Args:
        depth_map: (H, W) float32, 0=near, 1=far
        method: "threshold" (percentile cutoff) or "otsu" (automatic bimodal)
        threshold: for "threshold" method, fraction of depth range considered ground
        near_clip: depth percentile below which pixels are rejected as occlusions
                   (watermarks, camera borders, timestamps). 0 to disable.
        far_clip:  depth percentile above which pixels are rejected as background
                   (sky, distant mountains). 1 to disable.

    Returns:
        ground_mask: (H, W) bool, True = ground pixel
    """
    # Compute clipping bounds from percentiles
    near_val = np.percentile(depth_map, near_clip * 100) if near_clip > 0 else -1
    far_val = np.percentile(depth_map, far_clip * 100) if far_clip < 1 else 2

    # Valid region: not too close (occlusions) and not too far (sky)
    valid_mask = (depth_map > near_val) & (depth_map < far_val)

    # Run segmentation only on valid pixels
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


# ── visualization ────────────────────────────────────────────────────────────

def visualize_depth(image, depth_map, ground_mask, output_path, title=""):
    """Save a 2x2 overview: original, depth colormap, ground mask, overlay."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # Depth map
    im = axes[0, 1].imshow(depth_map, cmap="inferno")
    axes[0, 1].set_title("Depth Map (dark=near, bright=far)")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # Ground mask
    axes[1, 0].imshow(ground_mask, cmap="gray")
    ground_pct = ground_mask.sum() / ground_mask.size * 100
    axes[1, 0].set_title(f"Ground Mask ({ground_pct:.1f}% of image)")
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

def process_image(image_path, model, processor, output_dir, method, threshold,
                  near_clip=0.05, far_clip=0.95, device="cuda", debug=False):
    """
    Process a single image: predict depth, segment ground, save outputs.

    Returns:
        dict with image path, ground percentage, mean depth, or None on failure.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    depth_map = predict_depth(model, processor, image, device=device)
    ground_mask = segment_ground_from_depth(depth_map, method=method,
                                            threshold=threshold,
                                            near_clip=near_clip,
                                            far_clip=far_clip)

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    # Save ground mask
    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    if debug:
        # Save depth colormap
        import matplotlib.pyplot as plt
        plt.imsave(str(cam_out / f"{stem}_depth.png"), depth_map, cmap="inferno")

        # Save overview figure
        visualize_depth(image, depth_map, ground_mask,
                        cam_out / f"{stem}_overview.png",
                        title=f"{camera} — {stem}")

    ground_pct = ground_mask.sum() / ground_mask.size * 100
    mean_depth = float(depth_map.mean())

    return {
        "image": str(image_path),
        "camera": camera,
        "filename": Path(image_path).name,
        "ground_pct": round(ground_pct, 2),
        "mean_depth": round(mean_depth, 4),
        "method": method,
    }


def process_folder(data_dir, cameras, model, processor, output_dir, method,
                   threshold, sample, near_clip=0.05, far_clip=0.95,
                   device="cuda", debug=False):
    """Process all images in the given camera folders."""
    data_dir = Path(data_dir)
    results = []

    # Auto-discover cameras if none specified
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
                                   method, threshold, near_clip=near_clip,
                                   far_clip=far_clip, device=device, debug=debug)
            if result:
                results.append(result)

    return results


def process_canonical_views(canonical_dir, model, processor, output_dir,
                            method, threshold, near_clip=0.05, far_clip=0.95,
                            device="cuda", debug=True):
    """Process canonical view images from the DINOv3 pipeline."""
    canonical_dir = Path(canonical_dir)
    results = []

    # Find all canonical_view_*.jpg files
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
                                   method, threshold, near_clip=near_clip,
                                   far_clip=far_clip, device=device, debug=debug)
            if result:
                results.append(result)

    return results


# ── grid search: predict once, segment many ──────────────────────────────────

def _collect_image_paths(args, cameras):
    """Collect image paths based on input mode and sampling."""
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


def run_grid(args, model, processor, device):
    """
    Predict depth once per image, then re-segment with every
    (near_clip, far_clip) combination from the grid.
    """
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

    # Prepare output dirs for each combo
    combo_dirs = {}
    for n, f in combos:
        tag = f"N{n}_F{f}"
        d = base_output_dir / tag
        d.mkdir(parents=True, exist_ok=True)
        combo_dirs[(n, f)] = d

    # Collect image paths
    print("\nCollecting images...")
    image_paths = _collect_image_paths(args, args.cameras)
    if not image_paths:
        print("No images found.")
        return

    print(f"\nTotal images: {len(image_paths)}")
    print(f"Depth predicted once, segmented {len(combos)}x each\n")

    # Per-combo results
    all_results = {combo: [] for combo in combos}

    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Warning: skipping {img_path}: {e}")
            continue

        # Predict depth ONCE
        depth_map = predict_depth(model, processor, image, device=device)

        stem = img_path.stem
        camera = img_path.parent.name
        mean_depth = float(depth_map.mean())

        # Save depth map once (in first combo dir, or a shared dir)
        shared_depth_dir = base_output_dir / "shared_depth" / camera
        shared_depth_dir.mkdir(parents=True, exist_ok=True)
        depth_path = shared_depth_dir / f"{stem}_depth.png"
        if not depth_path.exists():
            plt.imsave(str(depth_path), depth_map, cmap="inferno")

        # Segment with each combo
        for (n, f) in combos:
            ground_mask = segment_ground_from_depth(
                depth_map, method=args.method, threshold=args.threshold,
                near_clip=n, far_clip=f
            )

            cam_out = combo_dirs[(n, f)] / camera
            cam_out.mkdir(parents=True, exist_ok=True)

            mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
            mask_img.save(cam_out / f"{stem}_ground.png")

            if args.debug:
                visualize_depth(image, depth_map, ground_mask,
                                cam_out / f"{stem}_overview.png",
                                title=f"{camera} — {stem} (N{n}_F{f})")

            ground_pct = ground_mask.sum() / ground_mask.size * 100
            all_results[(n, f)].append({
                "image": str(img_path),
                "camera": camera,
                "filename": img_path.name,
                "ground_pct": round(ground_pct, 2),
                "mean_depth": round(mean_depth, 4),
                "method": args.method,
            })

    # Save per-combo summaries
    for (n, f), results in all_results.items():
        if not results:
            continue
        tag = f"N{n}_F{f}"
        summary_path = combo_dirs[(n, f)] / "depth_summary.csv"
        fieldnames = ["image", "camera", "filename", "ground_pct",
                      "mean_depth", "method"]
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  {tag}: {len(results)} images, summary saved")

    # Auto-run comparison
    print(f"\nRunning comparison across {len(combos)} parameter sets...")
    compare_runs(base_output_dir, args.cameras if args.cameras else None)

    print(f"\nGrid search complete! {len(image_paths)} images x {len(combos)} combos")


# ── comparison across parameter runs ─────────────────────────────────────────

def compare_runs(base_output_dir, cameras_filter=None):
    """
    Find all N*_F* parameter run folders, and for each image that appears
    across runs, produce a side-by-side comparison figure.

    Output structure:
        base_output_dir/comparisons/<camera>/<stem>_compare.png
    """
    import matplotlib.pyplot as plt

    base_output_dir = Path(base_output_dir)

    # Discover parameter runs (folders matching N*_F*)
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

    # Discover cameras present in the first run
    first_run = run_dirs[0]
    all_cameras = sorted([
        d.name for d in first_run.iterdir()
        if d.is_dir() and re.match(r"C[A-Z]", d.name)
    ])

    if cameras_filter:
        all_cameras = [c for c in all_cameras if c in cameras_filter]

    if not all_cameras:
        # Maybe canonical views — look for any subfolder
        all_cameras = sorted([
            d.name for d in first_run.iterdir() if d.is_dir()
        ])

    print(f"Cameras: {all_cameras}")

    compare_dir = base_output_dir / "comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)
    total_figures = 0

    for camera in all_cameras:
        cam_compare_dir = compare_dir / camera
        cam_compare_dir.mkdir(parents=True, exist_ok=True)

        # Find ground mask files in the first run for this camera
        first_cam_dir = first_run / camera
        if not first_cam_dir.exists():
            continue

        ground_masks = sorted(first_cam_dir.glob("*_ground.png"))
        if not ground_masks:
            continue

        print(f"\n  {camera}: {len(ground_masks)} images to compare")

        for mask_path in ground_masks:
            stem = mask_path.stem.replace("_ground", "")

            # Check that this image exists across all runs
            all_exist = True
            for rd in run_dirs:
                if not (rd / camera / f"{stem}_ground.png").exists():
                    all_exist = False
                    break

            if not all_exist:
                continue

            # Find the original image or depth map for reference
            # Try to find the original image from any overview
            original_img = None
            depth_img = None
            for rd in run_dirs:
                overview_path = rd / camera / f"{stem}_overview.png"
                depth_path = rd / camera / f"{stem}_depth.png"
                if depth_path.exists() and depth_img is None:
                    depth_img = np.array(Image.open(depth_path))
                break

            # Build comparison figure: 1 row per run
            n_runs = len(run_dirs)
            fig, axes = plt.subplots(1, n_runs + 1, figsize=(4 * (n_runs + 1), 4))

            # First column: depth map (shared across runs, computed once)
            if depth_img is not None:
                axes[0].imshow(depth_img)
                axes[0].set_title("Depth Map", fontsize=10)
            else:
                axes[0].text(0.5, 0.5, "No depth\navailable",
                             ha="center", va="center", transform=axes[0].transAxes)
                axes[0].set_title("Depth Map", fontsize=10)
            axes[0].axis("off")

            # One column per parameter run: ground mask overlay
            for i, (rd, label) in enumerate(zip(run_dirs, run_labels)):
                mask_file = rd / camera / f"{stem}_ground.png"
                mask = np.array(Image.open(mask_file))
                ground_pct = (mask > 127).sum() / mask.size * 100

                axes[i + 1].imshow(mask, cmap="gray")
                axes[i + 1].set_title(f"{label}\n{ground_pct:.1f}% ground",
                                      fontsize=9)
                axes[i + 1].axis("off")

            fig.suptitle(f"{camera} — {stem}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            out_path = cam_compare_dir / f"{stem}_compare.png"
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close()
            total_figures += 1

    # Also create a summary comparison: pick one representative image per camera
    # and show all runs in a single grid
    if total_figures > 0:
        _create_summary_grid(base_output_dir, run_dirs, run_labels,
                             all_cameras, compare_dir)

    print(f"\nSaved {total_figures} comparison figures to {compare_dir}/")


def _create_summary_grid(base_output_dir, run_dirs, run_labels, cameras,
                         compare_dir):
    """
    One big grid: rows=cameras, columns=parameter runs.
    Uses the first available ground mask per camera as the representative.
    """
    import matplotlib.pyplot as plt

    # Pick one representative image per camera
    representatives = []
    for camera in cameras:
        cam_dir = run_dirs[0] / camera
        if not cam_dir.exists():
            continue
        masks = sorted(cam_dir.glob("*_ground.png"))
        if masks:
            stem = masks[0].stem.replace("_ground", "")
            # Check all runs have it
            if all((rd / camera / f"{stem}_ground.png").exists()
                   for rd in run_dirs):
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
                                ha="right", va="bottom", fontsize=9,
                                color="white",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor="black", alpha=0.6))

    fig.suptitle("Parameter Comparison — Ground Masks", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    summary_path = compare_dir / "summary_grid.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSummary grid saved to: {summary_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Depth-based ground segmentation with Depth Anything V2"
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
                        default="preprocessing/segmentation/depth_output",
                        help="Output directory")
    parser.add_argument("--model-size", choices=["small", "base", "large"],
                        default="large", help="Depth Anything V2 model size")
    parser.add_argument("--method", choices=["threshold", "otsu"],
                        default="otsu", help="Ground segmentation method")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Depth percentile for threshold method (default 0.4)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Images to sample per camera (0=all)")
    parser.add_argument("--near-clip", type=float, default=0.05,
                        help="Depth percentile below which pixels are rejected "
                             "as occlusions/watermarks (default 0.05 = bottom 5%%)")
    parser.add_argument("--far-clip", type=float, default=0.95,
                        help="Depth percentile above which pixels are rejected "
                             "as sky/far background (default 0.95 = top 5%%)")
    parser.add_argument("--debug", action="store_true",
                        help="Save depth colormaps and overview figures")

    # Grid search mode
    parser.add_argument("--grid", action="store_true",
                        help="Run a grid of near/far clip values. "
                             "Depth is predicted once, segmentation is repeated "
                             "for each combo. Overrides --near-clip/--far-clip.")
    parser.add_argument("--grid-near", nargs="+", type=float,
                        default=[0.05, 0.10, 0.15],
                        help="Near clip values for grid (default: 0.05 0.10 0.15)")
    parser.add_argument("--grid-far", nargs="+", type=float,
                        default=[0.85, 0.90, 0.95],
                        help="Far clip values for grid (default: 0.85 0.90 0.95)")

    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                        help="Compare all parameter runs side-by-side "
                             "(looks for N*_F* folders in output-dir)")
    parser.add_argument("--compare-cameras", nargs="+", default=[],
                        help="Limit comparison to specific cameras")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_output_dir = Path(args.output_dir)

    # ── compare mode ──────────────────────────────────────────────────────
    if args.compare:
        print("=" * 70)
        print("COMPARISON MODE — Side-by-side parameter runs")
        print("=" * 70)
        compare_runs(base_output_dir, args.compare_cameras)
        return

    # ── grid mode ─────────────────────────────────────────────────────────
    if args.grid:
        print("=" * 70)
        print("DEPTH ANYTHING V2 — GRID SEARCH")
        print("=" * 70)
        print(f"  Model:   {args.model_size} ({DEPTH_MODELS[args.model_size]})")
        print(f"  Method:  {args.method}")
        print(f"  Device:  {device}")
        print(f"  Output:  {base_output_dir}")
        model, processor = load_depth_model(args.model_size, device)
        run_grid(args, model, processor, device)
        return

    # ── parameter-tagged subfolder ────────────────────────────────────────
    param_tag = f"N{args.near_clip}_F{args.far_clip}"
    output_dir = base_output_dir / param_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DEPTH ANYTHING V2 — GROUND SEGMENTATION")
    print("=" * 70)
    print(f"  Model:   {args.model_size} ({DEPTH_MODELS[args.model_size]})")
    print(f"  Method:  {args.method}" + (f" (percentile={args.threshold})"
          if args.method == "threshold" else ""))
    print(f"  Clips:   near={args.near_clip} far={args.far_clip}")
    print(f"  Tag:     {param_tag}")
    print(f"  Device:  {device}")
    print(f"  Output:  {output_dir}")

    model, processor = load_depth_model(args.model_size, device)

    # ── dispatch ──────────────────────────────────────────────────────────
    if args.image:
        print(f"\nProcessing single image: {args.image}")
        result = process_image(args.image, model, processor, output_dir,
                               args.method, args.threshold,
                               near_clip=args.near_clip, far_clip=args.far_clip,
                               device=device, debug=True)
        if result:
            print(f"\n  Ground: {result['ground_pct']:.1f}%")
            print(f"  Mean depth: {result['mean_depth']:.4f}")
            print(f"  Saved to: {output_dir}")
        results = [result] if result else []

    elif args.data_dir:
        results = process_folder(args.data_dir, args.cameras, model, processor,
                                 output_dir, args.method, args.threshold,
                                 args.sample, near_clip=args.near_clip,
                                 far_clip=args.far_clip, device=device,
                                 debug=args.debug)

    elif args.canonical_dir:
        results = process_canonical_views(args.canonical_dir, model, processor,
                                          output_dir, args.method, args.threshold,
                                          near_clip=args.near_clip,
                                          far_clip=args.far_clip,
                                          device=device, debug=True)

    # ── save summary ──────────────────────────────────────────────────────
    if results:
        summary_path = output_dir / "depth_summary.csv"
        fieldnames = ["image", "camera", "filename", "ground_pct",
                      "mean_depth", "method"]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary saved to: {summary_path}")

    print(f"\nProcessed {len(results)} images.")
    print("Done!")


if __name__ == "__main__":
    main()
