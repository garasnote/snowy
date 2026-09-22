#!/usr/bin/env python3
"""
Ensemble ground segmentation — majority voting across 6 methods.

Combines binary masks from SegFormer, DeepLabV3+, Depth Anything V2,
SAM3 (text), SAM3+DINO, and DINOv2 PCA using per-pixel voting.
CPU-only, reads existing outputs, no GPU needed.

Supports two modes:
  - Uniform voting: each method gets 1 vote (original behavior)
  - Weighted voting: methods get different vote counts based on quality

Default weights (total=10):
  sam3_dino=0.3, dinov2_pca=0.3, deeplabv3=3, segformer=3, depth=0.3, sam3_text=3

Outputs per image:
  - Binary ensemble mask (thresholded by min-votes)
  - Soft confidence map (vote fraction, for student training)
  - Agreement map (debug only, color-coded by vote count)

Usage:
    # Uniform voting (original): majority 4/6
    python ensemble_voting.py

    # Weighted voting with default weights (threshold 5/10)
    python ensemble_voting.py --weighted --min-votes 5

    # Custom weights
    python ensemble_voting.py --weighted --weights sam3_dino=3 dinov2_pca=2 deeplabv3=2

    # Single camera with debug
    python ensemble_voting.py --weighted --cameras CM00006 --debug

    # Different depth combo
    python ensemble_voting.py --depth-combo N0.1_F0.9
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

# Default weights based on visual quality analysis (total ≈ 10):
#   segformer (great) = 3, sam3_text (awesome) = 3, deeplabv3 (good) = 3,
#   sam3_dino (worse) = 1/3, dinov2_pca (bad) = 1/3, depth (weak) = 1/3
DEFAULT_WEIGHTS = {
    "sam3_dino": 1/3,
    "dinov2_pca": 1/3,
    "deeplabv3": 3,
    "segformer": 3,
    "depth": 1/3,
    "sam3_text": 3,
}


def build_method_dirs(base_dir, depth_combo="N0.05_F0.95"):
    """Build dict of method_name → output directory."""
    return {
        "segformer": base_dir / "segformer_output",
        "deeplabv3": base_dir / "deeplabv3_output",
        "depth": base_dir / "depth_output" / depth_combo,
        "sam3_text": base_dir / "sam3_dino_output" / "text",
        "sam3_dino": base_dir / "sam3_dino_output" / "dino+sam",
        
    }


def parse_weights(weight_strings):
    """Parse weight arguments like 'sam3_dino=3 dinov2_pca=0.5' into a dict."""
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

    # Load all available masks with their weights
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

    ground_pct = ensemble_mask.sum() / ensemble_mask.size * 100

    # Save binary mask
    cam_out = output_dir / camera
    mask_img = Image.fromarray((ensemble_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    # Save confidence map (always — needed for student training)
    if save_confidence:
        conf_img = Image.fromarray((confidence * 255).astype(np.uint8))
        conf_img.save(cam_out / f"{stem}_confidence.png")

    # Debug: agreement map (color-coded)
    if debug:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cmap = plt.cm.RdYlGn
        normalized = confidence  # already in [0, 1]
        colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(colored).save(cam_out / f"{stem}_agreement.png")

    return {
        "camera": camera,
        "filename": f"{stem}.jpg",
        "ground_pct": round(ground_pct, 2),
        "n_methods": n_methods,
        "total_weight": total_weight,
        "unanimous_ground_pct": round(
            (vote_count == total_weight).sum() / vote_count.size * 100, 2
        ),
        "unanimous_bg_pct": round(
            (vote_count == 0).sum() / vote_count.size * 100, 2
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble ground segmentation — majority voting (CPU only)"
    )
    parser.add_argument("--base-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'preprocessing' / 'segmentation'),
                        help="Base directory containing all method outputs")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "preprocessing" / "segmentation" / "ensemble_output"),
                        help="Output directory")
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific cameras (default: auto-detect)")
    parser.add_argument("--min-votes", type=int, default=4,
                        help="Minimum votes for ground (default: 4 uniform, 5 weighted)")
    parser.add_argument("--min-methods", type=int, default=3,
                        help="Skip images with fewer than this many methods available")
    parser.add_argument("--depth-combo", type=str, default="N0.05_F0.95",
                        help="Depth parameter combo to use (default: N0.05_F0.95)")
    parser.add_argument("--weighted", action="store_true",
                        help="Use weighted voting (default: segformer=3, sam3_text=3, "
                             "deeplabv3=3, sam3_dino=0.33, dinov2_pca=0.33, depth=0.33)")
    parser.add_argument("--weights", nargs="*", default=[],
                        help="Custom weights as method=N pairs "
                             "(e.g., --weights sam3_dino=3 dinov2_pca=2)")
    parser.add_argument("--save-confidence", action="store_true", default=True,
                        help="Save soft confidence maps (default: True)")
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
        total_w = sum(method_weights.values())
        output_dir = output_base / f"weighted_{args.min_votes}"
    else:
        # Uniform voting: weight = 1 for all methods
        method_weights = {m: 1 for m in build_method_dirs(base_dir).keys()}
        total_w = None  # computed dynamically per image
        output_dir = output_base / f"majority_{args.min_votes}"

    method_dirs = build_method_dirs(base_dir, args.depth_combo)

    # Check which methods actually exist
    available_methods = {}
    for name, d in method_dirs.items():
        if d.exists():
            available_methods[name] = d
        else:
            print(f"  WARNING: {name} not found at {d}")

    if len(available_methods) < 2:
        print(f"ERROR: Need at least 2 methods, found {len(available_methods)}")
        return

    # Excluded cameras
    EXCLUDED_CAMERAS = {"CM00024"}  # Zgornja Sorica — removed from pipeline

    # Discover cameras (intersection across all available methods)
    if args.cameras:
        cameras = [c for c in args.cameras if c not in EXCLUDED_CAMERAS]
    else:
        cam_sets = []
        for name, d in available_methods.items():
            cams = {c.name for c in d.iterdir()
                    if c.is_dir() and re.match(r"C[A-Z]\d+", c.name)
                    and c.name not in EXCLUDED_CAMERAS}
            cam_sets.append(cams)
        cameras = sorted(set.intersection(*cam_sets))

    mode_str = "WEIGHTED" if args.weighted else "UNIFORM"
    print("=" * 70)
    print(f"ENSEMBLE — {mode_str} VOTING (CPU only)")
    print("=" * 70)
    print(f"  Methods:      {list(available_methods.keys())} ({len(available_methods)} total)")
    if args.weighted:
        active_w = {m: method_weights.get(m, 1) for m in available_methods}
        print(f"  Weights:      {active_w}")
        print(f"  Total weight: {sum(active_w.values()):.2f}")
        print(f"  Min votes:    {args.min_votes} of {sum(active_w.values()):.2f}")
    else:
        print(f"  Min votes:    {args.min_votes} of {len(available_methods)}")
    print(f"  Min methods:  {args.min_methods}")
    print(f"  Depth combo:  {args.depth_combo}")
    print(f"  Cameras:      {cameras}")
    print(f"  Workers:      {args.workers}")
    print(f"  Confidence:   {save_confidence}")
    print(f"  Debug:        {args.debug}")
    print(f"  Output:       {output_dir}")
    print()

    # Pre-create output directories
    for cam in cameras:
        (output_dir / cam).mkdir(parents=True, exist_ok=True)

    # Discover per-image tasks: find stems common across methods
    task_args = []
    for camera in cameras:
        # Index each method's stems for this camera
        method_stems = {}
        for name, d in available_methods.items():
            cam_dir = d / camera
            if not cam_dir.exists():
                method_stems[name] = set()
                continue
            method_stems[name] = {
                p.stem.replace("_ground", "")
                for p in cam_dir.glob("*_ground.png")
            }

        # Find all stems that appear in at least min_methods
        all_stems = set()
        for s in method_stems.values():
            all_stems |= s

        matched = 0
        for stem in sorted(all_stems):
            # Build paths dict for this stem
            paths = {}
            for name, d in available_methods.items():
                mask_path = d / camera / f"{stem}_ground.png"
                if stem in method_stems[name]:
                    paths[name] = mask_path

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
        summary_path = output_dir / "ensemble_summary.csv"
        fieldnames = ["camera", "filename", "ground_pct", "n_methods",
                      "total_weight", "unanimous_ground_pct", "unanimous_bg_pct"]
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(all_results, key=lambda x: (x["camera"], x["filename"])):
                # Round total_weight for cleaner CSV
                r["total_weight"] = round(r["total_weight"], 3)
                writer.writerow(r)

        avg_pct = np.mean([r["ground_pct"] for r in all_results])
        avg_unanimous = np.mean([r["unanimous_ground_pct"] for r in all_results])
        avg_bg = np.mean([r["unanimous_bg_pct"] for r in all_results])
        label = f"weighted {args.min_votes}" if args.weighted else f"majority {args.min_votes}"
        print(f"\n  Results: {len(all_results)} images")
        print(f"  Avg ground ({label}): {avg_pct:.1f}%")
        print(f"  Avg unanimous ground: {avg_unanimous:.1f}%")
        print(f"  Avg unanimous background: {avg_bg:.1f}%")
        print(f"\n  Summary saved to: {summary_path}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
