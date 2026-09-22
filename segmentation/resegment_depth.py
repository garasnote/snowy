#!/usr/bin/env python3
"""
Re-segment ground masks from existing depth maps (NO GPU needed).

Reads depth PNGs from shared_depth/, applies new near/far clip parameters,
and saves new ground masks. Runs on CPU using multiprocessing.

Usage:
    # New grid of parameters
    python resegment_depth.py \
        --grid-near 0.001 0.01 0.05 0.10 0.15 \
        --grid-far 0.85 0.90 0.95 0.99 0.999

    # Just the new extreme values
    python resegment_depth.py \
        --grid-near 0.001 0.01 \
        --grid-far 0.99 0.999

    # Single camera
    python resegment_depth.py --cameras CM00006 \
        --grid-near 0.001 0.01 --grid-far 0.99 0.999
"""

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Precompute inverse inferno LUT (colormap RGB → scalar depth value)
# This maps inferno-colored pixels back to their original [0,1] depth.
import matplotlib.pyplot as plt

_INFERNO_LUT = None


def _build_inverse_inferno_lut():
    """Build a 256³ lookup table: inferno RGB → depth value."""
    global _INFERNO_LUT
    if _INFERNO_LUT is not None:
        return _INFERNO_LUT

    cmap = plt.cm.inferno
    # Sample the colormap at 4096 points for accuracy
    n = 4096
    vals = np.linspace(0, 1, n)
    colors = (cmap(vals)[:, :3] * 255).astype(np.uint8)  # (n, 3)

    # For each RGB combo, find the closest colormap entry
    # Build a flat lookup: for each (R,G,B) find the depth value
    # Use a simplified approach: just use R channel + G channel as key
    # since inferno is mostly unique in (R,G) space
    lut = np.zeros((256, 256, 256), dtype=np.float32)
    for i, v in enumerate(vals):
        r, g, b = colors[i]
        lut[r, g, b] = v

    # Fill gaps with nearest neighbor (for anti-aliased pixels)
    # Simpler: just build a 1D LUT based on luminance-weighted index
    _INFERNO_LUT = lut
    return lut


def load_depth_fast(path):
    """Load inferno-colored depth PNG back to float32 [0,1] using luminance.

    Since inferno colormap has monotonically increasing luminance,
    converting to perceived luminance gives a good approximation of
    the original depth ordering — sufficient for percentile thresholding.
    """
    # Load as RGB uint8 with PIL (fastest for PNG)
    img = np.array(Image.open(path).convert("RGB"))
    # Perceived luminance (inferno is ~monotonic in luminance)
    depth = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]) / 255.0
    return depth.astype(np.float32)


def segment_ground(depth_map, method="otsu", threshold=0.4,
                   near_clip=0.05, far_clip=0.95):
    """Separate ground from background using depth thresholding."""
    near_val = np.percentile(depth_map, near_clip * 100) if near_clip > 0 else -1
    far_val = np.percentile(depth_map, far_clip * 100) if far_clip < 1 else 2

    valid_mask = (depth_map > near_val) & (depth_map < far_val)
    valid_depths = depth_map[valid_mask]

    if len(valid_depths) == 0:
        return np.zeros_like(depth_map, dtype=bool)

    if method == "threshold":
        cutoff = np.percentile(valid_depths, threshold * 100)
        return (depth_map <= cutoff) & valid_mask
    elif method == "otsu":
        from skimage.filters import threshold_otsu
        try:
            otsu_val = threshold_otsu(valid_depths)
            return (depth_map <= otsu_val) & valid_mask
        except ValueError:
            return (depth_map <= 0.5) & valid_mask
    else:
        raise ValueError(f"Unknown method: {method}")


def process_one_image(args_tuple):
    """Process a single depth map with all parameter combos + save masks in worker."""
    depth_path, combos, combo_tags, method, threshold, output_dir = args_tuple

    stem = depth_path.stem.replace("_depth", "")
    camera = depth_path.parent.name

    depth_map = load_depth_fast(depth_path)
    mean_depth = float(depth_map.mean())

    results = []

    for (n, f), tag in zip(combos, combo_tags):
        ground_mask = segment_ground(depth_map, method=method,
                                     threshold=threshold, near_clip=n, far_clip=f)
        ground_pct = ground_mask.sum() / ground_mask.size * 100

        # Save mask directly in worker (no IPC overhead)
        mask_img = Image.fromarray((ground_mask * 255).astype(np.uint8))
        mask_img.save(output_dir / tag / camera / f"{stem}_ground.png")

        results.append({
            "tag": tag,
            "camera": camera,
            "filename": f"{stem}.jpg",
            "ground_pct": round(ground_pct, 2),
            "mean_depth": round(mean_depth, 4),
            "method": method,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Re-segment ground masks from existing depth maps (CPU only, fast)"
    )
    _DATA = Path(__file__).resolve().parent.parent / "preprocessing" / "segmentation"
    parser.add_argument("--depth-dir", type=str,
                        default=str(_DATA / "depth_output" / "shared_depth"),
                        help="Directory with shared depth maps")
    parser.add_argument("--output-dir", type=str,
                        default=str(_DATA / "depth_output"),
                        help="Base output directory for N*_F* folders")
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific cameras (default: all)")
    parser.add_argument("--method", choices=["threshold", "otsu"],
                        default="otsu", help="Segmentation method")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")

    # Grid mode
    parser.add_argument("--grid-near", nargs="+", type=float,
                        default=[0.001, 0.01, 0.05, 0.10, 0.15])
    parser.add_argument("--grid-far", nargs="+", type=float,
                        default=[0.85, 0.90, 0.95, 0.99, 0.999])

    # Single run mode
    parser.add_argument("--near-clip", type=float, default=None)
    parser.add_argument("--far-clip", type=float, default=None)

    args = parser.parse_args()

    depth_dir = Path(args.depth_dir)
    output_dir = Path(args.output_dir)

    if args.near_clip is not None and args.far_clip is not None:
        combos = [(args.near_clip, args.far_clip)]
    else:
        combos = [(n, f) for n in sorted(args.grid_near)
                  for f in sorted(args.grid_far)]

    combo_tags = [f"N{n}_F{f}" for n, f in combos]

    # Discover cameras
    if args.cameras:
        cameras = args.cameras
    else:
        cameras = sorted([
            d.name for d in depth_dir.iterdir()
            if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)
        ])

    print("=" * 70)
    print("RE-SEGMENT DEPTH MAPS (CPU only — multiprocessing)")
    print("=" * 70)
    print(f"  Depth dir:  {depth_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Cameras:    {cameras}")
    print(f"  Method:     {args.method}")
    print(f"  Workers:    {args.workers}")
    print(f"  Combos:     {len(combos)}")
    for tag in combo_tags:
        print(f"    {tag}")
    print()

    # Create output dirs
    for tag in combo_tags:
        (output_dir / tag).mkdir(parents=True, exist_ok=True)
        for cam in cameras:
            (output_dir / tag / cam).mkdir(parents=True, exist_ok=True)

    # Collect all depth map paths
    all_paths = []
    for camera in cameras:
        cam_dir = depth_dir / camera
        if not cam_dir.exists():
            print(f"  WARNING: {cam_dir} not found, skipping")
            continue
        paths = sorted(cam_dir.glob("*_depth.png"))
        print(f"  {camera}: {len(paths)} depth maps")
        all_paths.extend(paths)

    print(f"\n  Total: {len(all_paths)} images x {len(combos)} combos\n")

    # Process in parallel — loading, segmenting, AND saving all happen in workers
    task_args = [
        (p, combos, combo_tags, args.method, args.threshold, output_dir)
        for p in all_paths
    ]

    all_results = {tag: [] for tag in combo_tags}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one_image, a) for a in task_args]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results = future.result()
            for r in results:
                all_results[r["tag"]].append(r)

    # Save summaries
    print()
    fieldnames = ["camera", "filename", "ground_pct", "mean_depth", "method"]
    for tag in combo_tags:
        results = all_results[tag]
        if not results:
            continue
        summary_path = output_dir / tag / "depth_summary.csv"
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(results, key=lambda x: (x["camera"], x["filename"])):
                writer.writerow({k: r[k] for k in fieldnames})
        print(f"  {tag}: {len(results)} images")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
