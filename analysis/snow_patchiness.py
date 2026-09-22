#!/usr/bin/env python3
"""
Snow patchiness / disconnectedness analysis.

For each snow mask, computes connected-component metrics:
  - n_components: number of disconnected snow patches
  - largest_ratio: fraction of snow area in the largest component
  - mean_area: average component size in pixels
  - total_snow_pct: total snow coverage

Outputs CSV: analysis/snow_patchiness.csv

Usage:
    python analysis/snow_patchiness.py
    python analysis/snow_patchiness.py --methods sam3 yoloe
    python analysis/snow_patchiness.py --cameras CF00001 CM00009
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm

MASTER = Path(__file__).resolve().parent.parent
OUTPUT_BASE = MASTER / "preprocessing" / "segmentation"

METHOD_DIRS = {
    "hsv": OUTPUT_BASE / "hsv_snow_output_mobilenet_ft",
    "sam3": OUTPUT_BASE / "sam3_snow_output",
    "yoloe": OUTPUT_BASE / "yoloe_snow_output",
    "florence2": OUTPUT_BASE / "florence2_snow_output",
}

# Also check for ground-clipped variants
for name in list(METHOD_DIRS.keys()):
    clipped = METHOD_DIRS[name].parent / f"{METHOD_DIRS[name].name}_ground_clipped"
    if clipped.exists():
        METHOD_DIRS[f"{name}_clipped"] = clipped

# Ensemble
ensemble_base = OUTPUT_BASE / "ensemble_snow_output"
if ensemble_base.exists():
    for sub in sorted(ensemble_base.iterdir()):
        if sub.is_dir():
            METHOD_DIRS[f"ensemble_{sub.name}"] = sub


_TS_RE = re.compile(r"_(\d{12})_")


def parse_datetime_str(stem):
    m = _TS_RE.search(stem)
    if not m:
        return None
    s = m.group(1)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}:00"


def analyze_one(args_tuple):
    """Compute patchiness metrics for one mask."""
    method, camera, mask_path = args_tuple
    stem = mask_path.stem.replace("_snow", "")

    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        binary = mask > 127

        total_snow = int(binary.sum())
        total_pixels = binary.size
        snow_pct = total_snow / total_pixels * 100

        if total_snow == 0:
            return {
                "method": method, "camera": camera, "stem": stem,
                "datetime": parse_datetime_str(stem),
                "total_snow_pct": 0.0,
                "n_components": 0,
                "largest_ratio": 0.0,
                "mean_area_px": 0.0,
                "median_area_px": 0.0,
            }

        # Morphological opening to remove tiny noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        labeled, n_components = ndimage.label(cleaned)
        if n_components == 0:
            return {
                "method": method, "camera": camera, "stem": stem,
                "datetime": parse_datetime_str(stem),
                "total_snow_pct": round(snow_pct, 3),
                "n_components": 0,
                "largest_ratio": 0.0,
                "mean_area_px": 0.0,
                "median_area_px": 0.0,
            }

        component_sizes = ndimage.sum(
            cleaned, labeled, range(1, n_components + 1)
        )
        component_sizes = np.array(component_sizes)
        largest = component_sizes.max()
        cleaned_total = component_sizes.sum()

        return {
            "method": method,
            "camera": camera,
            "stem": stem,
            "datetime": parse_datetime_str(stem),
            "total_snow_pct": round(snow_pct, 3),
            "n_components": n_components,
            "largest_ratio": round(float(largest / cleaned_total), 4) if cleaned_total > 0 else 0.0,
            "mean_area_px": round(float(component_sizes.mean()), 1),
            "median_area_px": round(float(np.median(component_sizes)), 1),
        }
    except Exception as e:
        print(f"  Error: {mask_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Snow patchiness analysis")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to analyze (default: all available)")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Cameras to analyze (default: all)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    methods = args.methods or list(METHOD_DIRS.keys())
    available = {m: d for m, d in METHOD_DIRS.items()
                 if m in methods and d.exists()}

    print(f"Analyzing {len(available)} methods: {list(available.keys())}")
    print()

    # Collect tasks
    tasks = []
    for method, method_dir in available.items():
        cam_dirs = sorted([d for d in method_dir.iterdir()
                           if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)])
        for cam_dir in cam_dirs:
            if args.cameras and cam_dir.name not in args.cameras:
                continue
            for mask_path in cam_dir.glob("*_snow.png"):
                tasks.append((method, cam_dir.name, mask_path))

    print(f"Total masks to analyze: {len(tasks)}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze_one, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Analyzing"):
            r = future.result()
            if r is not None:
                results.append(r)

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df = df.sort_values(["method", "camera", "stem"]).reset_index(drop=True)

    out_path = MASTER / "analysis" / "snow_patchiness.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)")

    # Summary
    print("\n--- Patchiness Summary (snowy images only) ---")
    snowy = df[df["total_snow_pct"] > 0.5]
    if len(snowy) > 0:
        summary = snowy.groupby("method").agg({
            "n_components": ["mean", "median", "max"],
            "largest_ratio": "mean",
            "total_snow_pct": "mean",
        }).round(2)
        print(summary.to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()
