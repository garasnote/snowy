#!/usr/bin/env python3
"""
Extract lightweight image features for all webcam images.

Features are cheap CPU-only descriptors useful for both QA classification
and temporal GRU snow refinement. No deep model needed.

Output: analysis/image_features.csv
    Columns: camera, stem, brightness, contrast, edge_density, entropy,
             white_frac, low_sat_frac, blue_ratio, bottom_brightness,
             bottom_white_frac, r_mean, g_mean, b_mean, h_mean, s_mean, v_mean,
             r_std, g_std, b_std, h_std, s_std, v_std

Usage:
    python analysis/extract_image_features.py
    python analysis/extract_image_features.py --cameras CF00001 CM00009
    python analysis/extract_image_features.py --workers 16
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

MASTER = Path(__file__).resolve().parent.parent
DATA_DIR = MASTER / "data"

_CAM_RE = re.compile(r"C[A-Z]\d+")
_TS_RE = re.compile(r"_(\d{12})_")

FEATURE_NAMES = [
    "brightness", "contrast", "edge_density", "entropy",
    "white_frac", "low_sat_frac", "blue_ratio",
    "bottom_brightness", "bottom_white_frac",
    "r_mean", "g_mean", "b_mean",
    "h_mean", "s_mean", "v_mean",
    "r_std", "g_std", "b_std",
    "h_std", "s_std", "v_std",
]


def extract_features(img_path):
    """Extract lightweight image features (CPU, ~10ms per image)."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    # Downsample for speed (features are statistical, resolution doesn't matter much)
    if img.width > 640:
        ratio = 640 / img.width
        img = img.resize((640, int(img.height * ratio)), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Grayscale
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    brightness = gray.mean()
    contrast = gray.std()

    # Edge density
    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    edge_density = (np.abs(gy).mean() + np.abs(gx).mean()) / 2.0

    # Entropy
    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))

    # White pixel fraction
    white_frac = float(((r > 200) & (g > 200) & (b > 200)).mean())

    # HSV
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    sat = hsv[:, :, 1]
    low_sat_frac = float((sat < 30).mean())

    # Blue ratio
    total = r + g + b + 1e-6
    blue_ratio = float((b / total).mean())

    # Bottom 70% (ground region)
    h_img = arr.shape[0]
    crop_start = int(h_img * 0.3)
    bottom_gray = gray[crop_start:]
    bottom_brightness = float(bottom_gray.mean())
    bottom_white_frac = float(
        ((r[crop_start:] > 200) & (g[crop_start:] > 200) & (b[crop_start:] > 200)).mean()
    )

    # RGB/HSV channel stats
    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "entropy": entropy,
        "white_frac": white_frac,
        "low_sat_frac": low_sat_frac,
        "blue_ratio": blue_ratio,
        "bottom_brightness": bottom_brightness,
        "bottom_white_frac": bottom_white_frac,
        "r_mean": rgb_mean[0], "g_mean": rgb_mean[1], "b_mean": rgb_mean[2],
        "h_mean": hsv_mean[0], "s_mean": hsv_mean[1], "v_mean": hsv_mean[2],
        "r_std": rgb_std[0], "g_std": rgb_std[1], "b_std": rgb_std[2],
        "h_std": hsv_std[0], "s_std": hsv_std[1], "v_std": hsv_std[2],
    }


def process_one(img_path):
    """Worker function for parallel processing."""
    camera = img_path.parent.name
    stem = img_path.stem
    feats = extract_features(img_path)
    if feats is None:
        return None
    feats["camera"] = camera
    feats["stem"] = stem
    return feats


def main():
    parser = argparse.ArgumentParser(description="Extract image features for all webcam images")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=str,
                        default=str(MASTER / "analysis" / "image_features.csv"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Discover cameras
    if args.cameras:
        cameras = args.cameras
    else:
        cameras = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and _CAM_RE.match(d.name)
        ])

    # Collect all image paths
    all_images = []
    for cam in cameras:
        cam_dir = data_dir / cam
        if not cam_dir.exists():
            continue
        imgs = sorted(cam_dir.glob("*.jpg"))
        all_images.extend(imgs)
        print(f"  {cam}: {len(imgs)} images")

    print(f"\nTotal images: {len(all_images)}")
    print(f"Workers: {args.workers}")
    print("Extracting features...")

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, p): p for p in all_images}
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                results.append(row)
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(all_images)} done...")

    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df = df.sort_values(["camera", "stem"]).reset_index(drop=True)

    # Reorder columns
    cols = ["camera", "stem"] + FEATURE_NAMES
    df = df[cols]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"  {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
