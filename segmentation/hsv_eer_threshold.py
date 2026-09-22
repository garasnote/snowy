#!/usr/bin/env python3
"""Find the Equal Error Rate (EER) threshold for HSV snow coverage.

For each HSV mask, computes the percentage of white pixels (coverage).
Then sweeps integer thresholds (0-100%) and finds where FAR == FRR.

Uses meteo_hourly.csv to match each image to its specific hour for
accurate snow labels (instead of daily which is too coarse).

Usage:
    python segmentation/hsv_eer_threshold.py \
        --mask-dir preprocessing/segmentation/hsv_snow_output \
        --meteo data/meteo_hourly.csv \
        --data-dir data/balanced_snow
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CAMERAS = [
    "CF00001", "CM00006", "CM00009", "CM00014",
    "CM00068", "CM00070", "CM00075", "CM00080",
    "CM00102", "CM00108", "CM00112",
]


def load_meteo_hourly(csv_path):
    """Load meteo_hourly.csv → {camera: {datetime_str: snow_present}}.

    Keys are ISO datetimes like '2024-01-29T09:00:00'.
    """
    snow_map = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            snow_map[row["camera"]][row["datetime"]] = row["snow_present"] == "True"
    return snow_map


def extract_datetime(filename):
    """CM00108_202401290900_05_snow.png → 2024-01-29T09:00:00"""
    m = re.search(r"_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})_", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:00"
    return None


def main():
    parser = argparse.ArgumentParser(description="HSV EER threshold finder")
    parser.add_argument("--mask-dir", type=Path, required=True,
                        help="Dir with HSV snow masks: {camera}/{stem}_snow.png")
    parser.add_argument("--meteo", type=Path, required=True,
                        help="meteo_hourly.csv")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="If set, only evaluate images present in this dir")
    parser.add_argument("--cameras", nargs="*", default=None)
    args = parser.parse_args()

    snow_map = load_meteo_hourly(args.meteo)
    cameras = args.cameras or CAMERAS

    # Collect (coverage_pct, is_snow) pairs
    samples = []
    skipped = 0
    no_meteo = 0

    for cam in cameras:
        cam_mask_dir = args.mask_dir / cam
        if not cam_mask_dir.exists():
            print(f"[{cam}] No mask dir, skipping")
            continue

        cam_snow = snow_map.get(cam, {})
        masks = sorted(cam_mask_dir.glob("*_snow.png"))
        cam_count = 0
        cam_no_meteo = 0

        for mask_path in masks:
            # Check if image exists in data-dir (if specified)
            if args.data_dir:
                stem = mask_path.stem.replace("_snow", "")
                img_path = args.data_dir / cam / f"{stem}.jpg"
                if not img_path.exists():
                    skipped += 1
                    continue

            dt = extract_datetime(mask_path.name)
            if dt is None or dt not in cam_snow:
                cam_no_meteo += 1
                no_meteo += 1
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                skipped += 1
                continue

            total_pixels = mask.shape[0] * mask.shape[1]
            snow_pixels = np.count_nonzero(mask)
            coverage = snow_pixels / total_pixels * 100

            samples.append((coverage, cam_snow[dt]))
            cam_count += 1

        print(f"[{cam}] {cam_count} samples ({cam_no_meteo} no hourly meteo match)")

    print(f"\nTotal: {len(samples)} samples, {skipped} skipped, {no_meteo} no hourly meteo match")
    if not samples:
        print("No samples to evaluate!")
        return

    # Count positives/negatives
    coverages = np.array([s[0] for s in samples])
    labels = np.array([s[1] for s in samples])
    n_snow = labels.sum()
    n_nosnow = len(labels) - n_snow
    print(f"Snow: {n_snow}, No-snow: {n_nosnow}")
    print()

    # Sweep thresholds in 0.1% steps from 0 to 20%
    print(f"{'Thresh%':>7} {'FAR':>8} {'FRR':>8} {'Acc':>8}")
    print("-" * 35)

    best_acc = 0.0
    best_thresh = 0.0
    best_far = 0.0
    best_frr = 0.0

    for step in range(201):
        thresh = step / 10.0
        predicted_snow = coverages >= thresh

        # FAR: no-snow images predicted as snow
        if n_nosnow > 0:
            far = predicted_snow[~labels].sum() / n_nosnow
        else:
            far = 0.0

        # FRR: snow images predicted as no-snow
        if n_snow > 0:
            frr = (~predicted_snow[labels]).sum() / n_snow
        else:
            frr = 0.0

        acc = (predicted_snow == labels).sum() / len(labels)

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_far = far
            best_frr = frr

        print(f"{thresh:>7.1f} {far:>8.4f} {frr:>8.4f} {acc:>8.4f}")

    print()
    print("=" * 35)
    print(f"Best threshold: {best_thresh:.1f}%")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  FAR: {best_far:.4f}")
    print(f"  FRR: {best_frr:.4f}")


if __name__ == "__main__":
    main()
