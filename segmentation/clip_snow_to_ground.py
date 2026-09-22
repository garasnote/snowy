#!/usr/bin/env python3
"""
Clip snow masks to ground masks.

Post-processes existing snow segmentation outputs by AND-ing them with
ground masks, restricting snow detection to ground regions only.

For each source snow directory (e.g. sam3_snow_output), creates a new
directory alongside it with suffix `_ground_clipped` containing the
same file layout but masked.

Usage:
    # Clip all snow method outputs using mobilenet_ft ground masks
    python clip_snow_to_ground.py

    # Custom source dirs
    python clip_snow_to_ground.py --methods sam3 yoloe florence2

    # Use a different ground mask source
    python clip_snow_to_ground.py --ground-source efficientnet_ft
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


METHOD_DIRS = {
    "hsv":       "hsv_snow_output_mobilenet_ft",
    "sam3":      "sam3_snow_output",
    "yoloe":     "yoloe_snow_output",
    "florence2": "florence2_snow_output",
}


def clip_one(args_tuple):
    """Load snow mask + ground mask, AND them, save to dest."""
    snow_path, ground_path, dest_path = args_tuple

    if not ground_path.exists():
        return ("no_ground", snow_path.name)

    try:
        snow = np.array(Image.open(snow_path).convert("L"))
        ground = np.array(Image.open(ground_path).convert("L"))

        if ground.shape != snow.shape:
            ground = np.array(Image.fromarray(ground).resize(
                (snow.shape[1], snow.shape[0]), Image.NEAREST))

        snow_bool = snow > 127
        ground_bool = ground > 127
        clipped = (snow_bool & ground_bool).astype(np.uint8) * 255

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(clipped).save(dest_path)
        return ("ok", snow_path.name)
    except Exception as e:
        return ("error", f"{snow_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Clip snow masks to ground masks (AND operation)"
    )
    parser.add_argument("--base-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "preprocessing" / "segmentation",
                        help="Base directory with segmentation outputs")
    parser.add_argument("--methods", nargs="+",
                        default=list(METHOD_DIRS.keys()),
                        choices=list(METHOD_DIRS.keys()),
                        help="Which snow methods to clip")
    parser.add_argument("--ground-source", default="mobilenet_ft",
                        help="Student model under student_ground_output/")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    base = args.base_dir
    ground_base = base / "student_ground_output" / args.ground_source
    if not ground_base.exists():
        print(f"ERROR: ground source not found: {ground_base}")
        return

    print(f"Ground source: {ground_base}")
    print(f"Methods:       {args.methods}")
    print()

    for method in args.methods:
        src_dir = base / METHOD_DIRS[method]
        if not src_dir.exists():
            print(f"  SKIP {method}: {src_dir} does not exist")
            continue

        dest_dir = src_dir.parent / f"{src_dir.name}_ground_clipped"
        print(f"=== {method}: {src_dir.name} → {dest_dir.name} ===")

        # Collect all (snow, ground, dest) tasks
        tasks = []
        cameras = sorted([d for d in src_dir.iterdir()
                          if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)])
        for cam_dir in cameras:
            for snow_path in cam_dir.glob("*_snow.png"):
                stem = snow_path.stem.replace("_snow", "")
                ground_path = ground_base / cam_dir.name / f"{stem}_ground.png"
                dest_path = dest_dir / cam_dir.name / snow_path.name
                tasks.append((snow_path, ground_path, dest_path))

        if not tasks:
            print(f"  No snow masks found in {src_dir}")
            continue

        print(f"  Processing {len(tasks)} masks with {args.workers} workers...")
        stats = {"ok": 0, "no_ground": 0, "error": 0}
        errors = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(clip_one, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  {method}"):
                status, info = future.result()
                stats[status] += 1
                if status == "error":
                    errors.append(info)

        print(f"  Done: {stats['ok']} ok, {stats['no_ground']} no ground, "
              f"{stats['error']} errors")
        if errors[:3]:
            for e in errors[:3]:
                print(f"    ! {e}")
        print()

    print("All done!")


if __name__ == "__main__":
    main()
