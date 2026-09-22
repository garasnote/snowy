#!/usr/bin/env python3
"""HSV-based snow detection baseline.

Segments snow by thresholding white/bright pixels in HSV space,
then AND-ing with a precomputed ground mask to restrict detection
to the ground region only.

Usage:
    python hsv_snow_baseline.py \
        --data-dir data/ \
        --ground-mask-dir preprocessing/segmentation/student_ground_output/mobilenet_ft \
        --output-dir preprocessing/segmentation/student_ground_output/hsv_snow_baseline

    # Tune thresholds:
    python hsv_snow_baseline.py --sat-max 40 --val-min 180
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CAMERAS = [
    "CF00001", "CM00006", "CM00009", "CM00014",
    "CM00068", "CM00070", "CM00075", "CM00080",
    "CM00102", "CM00108", "CM00112",
]


def segment_snow_hsv(img_bgr, sat_max=30, val_min=200):
    """Detect white/bright pixels in HSV space.

    Returns binary mask: 1 = snow candidate, 0 = not snow.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] < sat_max) & (hsv[:, :, 2] > val_min)
    return mask.astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser(description="HSV snow detection baseline")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root data dir containing camera folders with JPEGs")
    parser.add_argument("--ground-mask-dir", type=Path, default=None,
                        help="Dir with ground masks: {camera}/{stem}_ground.png (optional)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output dir for snow masks")
    parser.add_argument("--sat-max", type=int, default=30,
                        help="Max saturation for white detection (default: 30)")
    parser.add_argument("--val-min", type=int, default=200,
                        help="Min value for white detection (default: 200)")
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Cameras to process (default: all)")
    args = parser.parse_args()

    cameras = args.cameras or CAMERAS
    total = 0
    skipped = 0

    use_ground = args.ground_mask_dir is not None

    for cam in cameras:
        cam_img_dir = args.data_dir / cam
        cam_mask_dir = args.ground_mask_dir / cam if use_ground else None
        cam_out_dir = args.output_dir / cam
        cam_out_dir.mkdir(parents=True, exist_ok=True)

        if not cam_img_dir.exists():
            print(f"[{cam}] No image dir, skipping")
            continue

        images = sorted(cam_img_dir.glob("*.jpg"))
        cam_count = 0

        for img_path in images:
            stem = img_path.stem  # e.g. CM00108_202401290900_05

            if use_ground:
                ground_mask_path = cam_mask_dir / f"{stem}_ground.png"
                if not ground_mask_path.exists():
                    skipped += 1
                    continue

            try:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    skipped += 1
                    continue

                # HSV white detection
                white_mask = segment_snow_hsv(img_bgr, args.sat_max, args.val_min)

                if use_ground:
                    # Load ground mask and resize to match image if needed
                    ground_mask = np.array(Image.open(ground_mask_path).convert("L"))
                    if ground_mask.shape[:2] != img_bgr.shape[:2]:
                        ground_mask = cv2.resize(ground_mask, (img_bgr.shape[1], img_bgr.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                    # AND with ground mask → snow only on ground
                    snow_mask = cv2.bitwise_and(white_mask, ground_mask)
                else:
                    snow_mask = white_mask

                # Save snow mask
                cv2.imwrite(str(cam_out_dir / f"{stem}_snow.png"), snow_mask)
                cam_count += 1

            except Exception as e:
                print(f"[{cam}] Error on {stem}: {e}")
                skipped += 1

        total += cam_count
        print(f"[{cam}] {cam_count} masks saved")

    print(f"\nDone: {total} masks, {skipped} skipped")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()