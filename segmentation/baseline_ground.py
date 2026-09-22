#!/usr/bin/env python3
"""
Baseline ground segmentation — horizon detection via Sobel edge + bottom fill.

Simple geometric prior: ground is below the horizon line, which is detected
as the strongest horizontal edge in the image. Falls back to a fixed percentage
of the image height if no clear horizon is found.

This serves as a non-learned baseline for comparison against VLM teachers
and student models.

Usage:
    # Single image
    python baseline_ground.py --image /path/to/image.jpg --debug

    # All cameras
    python baseline_ground.py --data-dir data

    # Custom ground fraction
    python baseline_ground.py --data-dir ... --fallback-fraction 0.35
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


def detect_horizon(img_gray, smooth_sigma=5, min_edge_strength=10):
    """Detect horizon line using horizontal Sobel edges.

    Computes horizontal edge response, averages across columns,
    picks the row with strongest horizontal edge as the horizon.
    """
    sobel_y = ndimage.sobel(img_gray, axis=0)
    row_profile = np.abs(sobel_y).mean(axis=1)
    if smooth_sigma > 0:
        row_profile = ndimage.gaussian_filter1d(row_profile, sigma=smooth_sigma)

    h = len(row_profile)
    search_start = int(h * 0.1)
    search_end = int(h * 0.9)
    search_region = row_profile[search_start:search_end]

    if search_region.max() < min_edge_strength:
        return None, row_profile

    horizon_row = search_start + np.argmax(search_region)
    return horizon_row, row_profile


def baseline_ground_mask(image_path, ground_fraction=0.32, method="both", debug=False):
    """Generate ground mask using geometric baselines.

    Methods:
        fixed: bottom ground_fraction of image labeled as ground
        sobel: horizon detected via Sobel edge, everything below = ground
        both: returns both masks

    Returns:
        masks: dict of {method_name: (mask, info)}
    """
    img = Image.open(image_path)
    w, h = img.size
    results = {}

    # Fixed baseline
    if method in ("fixed", "both"):
        horizon_row = int(h * (1.0 - ground_fraction))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[horizon_row:, :] = 255
        results["fixed"] = (mask, {
            "horizon_row": horizon_row,
            "method": "fixed",
            "ground_pct": ground_fraction * 100,
        })

    # Sobel baseline
    if method in ("sobel", "both"):
        img_gray = np.array(img.convert("L"), dtype=np.float32)
        horizon_row, row_profile = detect_horizon(img_gray)
        if horizon_row is None:
            horizon_row = int(h * (1.0 - ground_fraction))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[horizon_row:, :] = 255
        ground_pct = (h - horizon_row) / h * 100
        info = {
            "horizon_row": horizon_row,
            "method": "sobel",
            "ground_pct": ground_pct,
        }
        if debug:
            info["row_profile"] = row_profile
        results["sobel"] = (mask, info)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Baseline ground segmentation (Sobel horizon + bottom fill)"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root data directory with camera subdirs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: preprocessing/segmentation/baseline_output)")
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--ground-fraction", type=float, default=0.32,
                        help="Fraction of image height labeled as ground (default: 0.32)")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _HERE = Path(__file__).resolve().parent
    _MASTER = _HERE.parent

    if args.image:
        results = baseline_ground_mask(args.image, args.ground_fraction, method="both", debug=args.debug)
        for method_name, (mask, info) in results.items():
            print(f"  [{method_name}] Horizon: row {info['horizon_row']}, Ground: {info['ground_pct']:.1f}%")

        if args.debug:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            img = Image.open(args.image).convert("RGB")
            img_np = np.array(img)
            n = len(results)
            fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 5))

            axes[0].imshow(img)
            for method_name, (_, info) in results.items():
                color = "red" if method_name == "fixed" else "cyan"
                axes[0].axhline(info["horizon_row"], color=color, linewidth=2,
                                label=f"{method_name} (row {info['horizon_row']})")
            axes[0].legend()
            axes[0].set_title("Original + horizon lines")
            axes[0].axis("off")

            for i, (method_name, (mask, info)) in enumerate(results.items()):
                overlay = img_np.copy()
                overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                axes[i + 1].imshow(overlay)
                axes[i + 1].set_title(f"{method_name} ({info['ground_pct']:.1f}%)")
                axes[i + 1].axis("off")

            debug_dir = _MASTER / "preprocessing" / "segmentation" / "baseline_output" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_path = debug_dir / f"{Path(args.image).stem}_baseline_debug.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Debug: {out_path}")
            plt.close()

        for method_name, (mask, info) in results.items():
            out_dir = _MASTER / "preprocessing" / "segmentation" / f"baseline_{method_name}_output"
            camera = Path(args.image).parent.name
            (out_dir / camera).mkdir(parents=True, exist_ok=True)
            out_mask = out_dir / camera / f"{Path(args.image).stem}_ground.png"
            Image.fromarray(mask).save(out_mask)
            print(f"Saved: {out_mask}")
        return

    if args.data_dir is None:
        args.data_dir = str(_MASTER / "data")

    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _MASTER / "preprocessing" / "segmentation" / "baseline_output"

    if args.cameras:
        cameras = args.cameras
    else:
        cameras = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)
        ])

    base_output = _MASTER / "preprocessing" / "segmentation"

    print(f"Baseline ground segmentation")
    print(f"  Data: {data_dir}")
    print(f"  Output: {base_output}/baseline_{{fixed,sobel}}_output/")
    print(f"  Cameras: {cameras}")
    print(f"  Fixed fraction: {args.ground_fraction:.0%}")
    print()

    total = 0
    stats = []

    for camera in cameras:
        cam_dir = data_dir / camera
        if not cam_dir.exists():
            continue
        img_paths = sorted(cam_dir.glob("*.jpg"))
        if args.sample > 0:
            img_paths = img_paths[:args.sample]

        for method_name in ["fixed", "sobel"]:
            out_cam = base_output / f"baseline_{method_name}_output" / camera
            out_cam.mkdir(parents=True, exist_ok=True)

        cam_stats = {"fixed_pcts": [], "sobel_pcts": []}

        for img_path in tqdm(img_paths, desc=camera):
            stem = img_path.stem
            try:
                results = baseline_ground_mask(img_path, args.ground_fraction, method="both")
                for method_name, (mask, info) in results.items():
                    out_path = base_output / f"baseline_{method_name}_output" / camera / f"{stem}_ground.png"
                    if not out_path.exists():
                        Image.fromarray(mask).save(out_path)
                    cam_stats[f"{method_name}_pcts"].append(info["ground_pct"])
                total += 1
            except Exception as e:
                print(f"  SKIP {stem}: {e}")

        fixed_avg = np.mean(cam_stats["fixed_pcts"]) if cam_stats["fixed_pcts"] else 0
        sobel_avg = np.mean(cam_stats["sobel_pcts"]) if cam_stats["sobel_pcts"] else 0
        print(f"  {camera}: {len(img_paths)} images, "
              f"fixed avg={fixed_avg:.1f}%, sobel avg={sobel_avg:.1f}%")
        stats.append({
            "camera": camera,
            "n_images": len(img_paths),
            "fixed_avg_pct": round(fixed_avg, 1),
            "sobel_avg_pct": round(sobel_avg, 1),
        })

    summary_path = base_output / "baseline_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["camera", "n_images", "fixed_avg_pct", "sobel_avg_pct"])
        w.writeheader()
        w.writerows(stats)

    print(f"\nDone! {total} images")
    print(f"  Fixed: {base_output / 'baseline_fixed_output'}")
    print(f"  Sobel: {base_output / 'baseline_sobel_output'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
