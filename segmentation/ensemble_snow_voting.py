#!/usr/bin/env python3
"""
Ensemble snow segmentation — majority voting across methods.

Combines binary snow masks from HSV baseline, SAM3, YOLOe-26,
Florence-2, Grounded SAM2, and CLIPSeg using per-pixel voting.
CPU-only, reads existing outputs.

Supports two modes:
  - Uniform voting: each method gets 1 vote
  - Weighted voting: methods get configurable vote weights

Outputs per image:
  - Binary ensemble mask ({stem}_snow.png)
  - Soft confidence map ({stem}_confidence.png)
  - Agreement map (debug only, color-coded by vote count)

Usage:
    # Uniform voting, majority 2/3 (or 2/4 with Florence-2)
    python ensemble_snow_voting.py --min-votes 2

    # Weighted voting
    python ensemble_snow_voting.py --weighted --min-votes 3 \\
        --weights hsv=1 sam3=2 yoloe=2 florence2=2

    # Single camera with debug
    python ensemble_snow_voting.py --cameras CF00001 --debug
"""

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ── Method definitions ───────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "hsv": 1,
    "sam3": 1,
    "yoloe": 1,
    "florence2": 1,
    "groundedsam2": 1,
    "clipseg": 1,
    "seggpt": 1,
}


def build_method_dirs(base_dir):
    """Build dict of method_name -> output directory."""
    return {
        "hsv": base_dir / "hsv_snow_output_mobilenet_ft",
        "sam3": base_dir / "sam3_snow_output",
        "yoloe": base_dir / "yoloe_snow_output",
        "florence2": base_dir / "florence2_snow_output",
        "groundedsam2": base_dir / "groundedsam2_snow_output",
        "clipseg": base_dir / "clipseg_snow_output",
        "seggpt": base_dir / "seggpt_snow_output",
    }


def parse_weights(weight_strings):
    """Parse weight arguments like 'sam3=2 yoloe=3' into a dict."""
    weights = {}
    for ws in weight_strings:
        if "=" not in ws:
            raise ValueError(f"Invalid weight format '{ws}', expected 'method=N'")
        name, val = ws.split("=", 1)
        weights[name] = float(val)
    return weights


# ── Worker function ──────────────────────────────────────────────────────────

def process_one_image(args_tuple):
    """Load masks from all methods, vote, save results. Runs in worker."""
    (stem, camera, method_paths, min_votes,
     output_dir, save_confidence, debug, method_weights) = args_tuple

    masks = []
    weights = []
    ref_shape = None

    for method_name, mask_path in method_paths.items():
        if not mask_path.exists():
            continue
        m = np.array(Image.open(mask_path).convert("L"))
        mask_bool = m > 127

        if ref_shape is None:
            ref_shape = mask_bool.shape
        elif mask_bool.shape != ref_shape:
            pil_m = Image.fromarray(m)
            pil_m = pil_m.resize((ref_shape[1], ref_shape[0]), Image.NEAREST)
            mask_bool = np.array(pil_m) > 127

        masks.append(mask_bool)
        weights.append(method_weights.get(method_name, 1))

    n_methods = len(masks)
    if n_methods == 0 or ref_shape is None:
        return None

    # Weighted vote count
    vote_count = np.zeros(ref_shape, dtype=np.float32)
    total_weight = 0
    for mask, w in zip(masks, weights):
        vote_count += mask.astype(np.float32) * w
        total_weight += w

    # Binary ensemble mask
    ensemble_mask = vote_count >= min_votes

    # Confidence map: weighted vote fraction [0, 1]
    confidence = vote_count / max(total_weight, 1)

    snow_pct = ensemble_mask.sum() / ensemble_mask.size * 100

    # Save binary mask
    cam_out = output_dir / camera
    mask_img = Image.fromarray((ensemble_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    # Save confidence map
    if save_confidence:
        conf_img = Image.fromarray((confidence * 255).astype(np.uint8))
        conf_img.save(cam_out / f"{stem}_confidence.png")

    # Debug: agreement map
    if debug:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colored = (plt.cm.RdYlGn(confidence)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(colored).save(cam_out / f"{stem}_agreement.png")

    return {
        "camera": camera,
        "filename": f"{stem}.jpg",
        "snow_pct": round(snow_pct, 2),
        "n_methods": n_methods,
        "total_weight": round(total_weight, 3),
        "unanimous_snow_pct": round(
            (vote_count == total_weight).sum() / vote_count.size * 100, 2
        ),
        "unanimous_nosnow_pct": round(
            (vote_count == 0).sum() / vote_count.size * 100, 2
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble snow segmentation — majority voting (CPU only)"
    )
    parser.add_argument("--base-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent
                                    / "preprocessing" / "segmentation"),
                        help="Base directory containing all method outputs")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent
                                    / "preprocessing" / "segmentation"
                                    / "ensemble_snow_output"),
                        help="Output directory")
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific cameras (default: auto-detect)")
    parser.add_argument("--min-votes", type=int, default=2,
                        help="Minimum votes for snow (default: 2)")
    parser.add_argument("--min-methods", type=int, default=2,
                        help="Skip images with fewer methods available")
    parser.add_argument("--weighted", action="store_true",
                        help="Use weighted voting")
    parser.add_argument("--weights", nargs="*", default=[],
                        help="Custom weights as method=N pairs")
    parser.add_argument("--save-confidence", action="store_true", default=True)
    parser.add_argument("--no-confidence", action="store_true",
                        help="Skip saving confidence maps")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--debug", action="store_true",
                        help="Save color-coded agreement maps")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_base = Path(args.output_dir)
    save_confidence = not args.no_confidence

    # Resolve method weights
    if args.weighted:
        method_weights = dict(DEFAULT_WEIGHTS)
        if args.weights:
            method_weights.update(parse_weights(args.weights))
        output_dir = output_base / f"weighted_{args.min_votes}"
    else:
        method_weights = {m: 1 for m in build_method_dirs(base_dir).keys()}
        output_dir = output_base / f"majority_{args.min_votes}"

    method_dirs = build_method_dirs(base_dir)

    # Check which methods actually exist
    available_methods = {}
    for name, d in method_dirs.items():
        if d.exists() and any(d.iterdir()):
            available_methods[name] = d
        else:
            print(f"  NOTE: {name} not found at {d}")

    if len(available_methods) < 2:
        print(f"ERROR: Need at least 2 methods, found {len(available_methods)}")
        return

    # Discover cameras (union across all available methods)
    if args.cameras:
        cameras = args.cameras
    else:
        cam_sets = []
        for name, d in available_methods.items():
            cams = {c.name for c in d.iterdir()
                    if c.is_dir() and re.match(r"C[A-Z]\d+", c.name)}
            cam_sets.append(cams)
        cameras = sorted(set.union(*cam_sets))

    mode_str = "WEIGHTED" if args.weighted else "UNIFORM"
    print("=" * 70)
    print(f"ENSEMBLE SNOW — {mode_str} VOTING (CPU only)")
    print("=" * 70)
    print(f"  Methods:    {list(available_methods.keys())} "
          f"({len(available_methods)} total)")
    if args.weighted:
        active_w = {m: method_weights.get(m, 1) for m in available_methods}
        print(f"  Weights:    {active_w}")
        print(f"  Min votes:  {args.min_votes} of {sum(active_w.values()):.2f}")
    else:
        print(f"  Min votes:  {args.min_votes} of {len(available_methods)}")
    print(f"  Cameras:    {cameras}")
    print(f"  Workers:    {args.workers}")
    print(f"  Output:     {output_dir}")
    print()

    # Pre-create output directories
    for cam in cameras:
        (output_dir / cam).mkdir(parents=True, exist_ok=True)

    # Discover per-image tasks
    task_args = []
    for camera in cameras:
        method_stems = {}
        for name, d in available_methods.items():
            cam_dir = d / camera
            if not cam_dir.exists():
                method_stems[name] = set()
                continue
            method_stems[name] = {
                p.stem.replace("_snow", "")
                for p in cam_dir.glob("*_snow.png")
            }

        # All stems across methods
        all_stems = set()
        for s in method_stems.values():
            all_stems |= s

        matched = 0
        for stem in sorted(all_stems):
            paths = {}
            for name, d in available_methods.items():
                if stem in method_stems.get(name, set()):
                    paths[name] = d / camera / f"{stem}_snow.png"

            if len(paths) >= args.min_methods:
                task_args.append((
                    stem, camera, paths, args.min_votes,
                    output_dir, save_confidence, args.debug, method_weights
                ))
                matched += 1

        print(f"  {camera}: {matched} images with >={args.min_methods} methods")

    if not task_args:
        print("\nNo images found with enough methods!")
        return

    print(f"\n  Total: {len(task_args)} images\n")

    # Process in parallel
    all_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one_image, a) for a in task_args]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Ensembling"):
            result = future.result()
            if result is not None:
                all_results.append(result)

    # Save summary
    if all_results:
        summary_path = output_dir / "ensemble_snow_summary.csv"
        fieldnames = ["camera", "filename", "snow_pct", "n_methods",
                      "total_weight", "unanimous_snow_pct",
                      "unanimous_nosnow_pct"]
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(all_results,
                            key=lambda x: (x["camera"], x["filename"])):
                writer.writerow(r)

        avg_pct = np.mean([r["snow_pct"] for r in all_results])
        avg_unanimous = np.mean([r["unanimous_snow_pct"] for r in all_results])
        label = (f"weighted {args.min_votes}" if args.weighted
                 else f"majority {args.min_votes}")
        print(f"\n  Results: {len(all_results)} images")
        print(f"  Avg snow ({label}): {avg_pct:.1f}%")
        print(f"  Avg unanimous snow: {avg_unanimous:.1f}%")
        print(f"\n  Summary: {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
