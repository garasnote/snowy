#!/usr/bin/env python3
"""
Ground segmentation utilities: the deterministic baseline and the
majority-vote ensemble that combines the teacher outputs from
ground_teachers.py into the pseudo-labels used to train the ground
student models (Methods, "Ground Segmentation").

Consolidated from baseline_ground.py and ensemble_voting.py.

Usage:
    python ground_utils.py baseline --data-dir ./data
    python ground_utils.py ensemble --weighted --min-votes 5

Run `python ground_utils.py <subcommand> --help` for the full option list
of each (they were independent scripts and keep their original flags).
"""

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent


# =============================================================================
# BASELINE — Sobel horizon detection + fixed bottom-fraction fill
# =============================================================================
# Simple geometric prior: ground is below the horizon line, which is detected
# as the strongest horizontal edge in the image. Falls back to a fixed
# percentage of the image height if no clear horizon is found. Serves as a
# non-learned baseline for comparison against VLM teachers and student
# models (Methods, "Deterministic baseline").

def baseline_detect_horizon(img_gray, smooth_sigma=5, min_edge_strength=10):
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
        horizon_row, row_profile = baseline_detect_horizon(img_gray)
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


def add_baseline_subparser(subparsers):
    p = subparsers.add_parser(
        "baseline", help="Sobel horizon + fixed bottom-fraction baseline")
    p.add_argument("--image", type=str, default=None,
                    help="Single image path")
    p.add_argument("--data-dir", type=str, default=None,
                    help="Root data directory with camera subdirs")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: preprocessing/segmentation/baseline_output)")
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--ground-fraction", type=float, default=0.32,
                    help="Fraction of image height labeled as ground (default: 0.32)")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    return p


def run_baseline(args):
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

            debug_dir = _REPO_ROOT / "preprocessing" / "segmentation" / "baseline_output" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_path = debug_dir / f"{Path(args.image).stem}_baseline_debug.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Debug: {out_path}")
            plt.close()

        for method_name, (mask, info) in results.items():
            out_dir = _REPO_ROOT / "preprocessing" / "segmentation" / f"baseline_{method_name}_output"
            camera = Path(args.image).parent.name
            (out_dir / camera).mkdir(parents=True, exist_ok=True)
            out_mask = out_dir / camera / f"{Path(args.image).stem}_ground.png"
            Image.fromarray(mask).save(out_mask)
            print(f"Saved: {out_mask}")
        return

    if args.data_dir is None:
        args.data_dir = str(_REPO_ROOT / "data")

    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _REPO_ROOT / "preprocessing" / "segmentation" / "baseline_output"

    if args.cameras:
        cameras = args.cameras
    else:
        cameras = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)
        ])

    base_output = _REPO_ROOT / "preprocessing" / "segmentation"

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


# =============================================================================
# ENSEMBLE — majority-vote aggregation across teacher outputs
# =============================================================================
# Combines binary masks from SegFormer, DeepLabV3+, Depth Anything V2,
# SAM3 (text), SAM3+DINO, and DINOv2 PCA using per-pixel voting. CPU-only,
# reads existing teacher outputs, no GPU needed. This is the actual
# methodology contribution described in Methods, "Ground Segmentation" —
# the teacher scripts themselves are off-the-shelf pretrained models, this
# is what turns their outputs into a single set of pseudo-labels.
#
# Supports two modes:
#   - Uniform voting: each method gets 1 vote (paper's reported k-of-N)
#   - Weighted voting: methods get different vote counts based on quality
#     (exploratory, not the configuration reported in the paper)

# Default weights based on visual quality analysis (total ~= 10):
#   segformer (great) = 3, sam3_text (awesome) = 3, deeplabv3 (good) = 3,
#   sam3_dino (worse) = 1/3, dinov2_pca (bad) = 1/3, depth (weak) = 1/3
ENSEMBLE_DEFAULT_WEIGHTS = {
    "sam3_dino": 1 / 3,
    "dinov2_pca": 1 / 3,
    "deeplabv3": 3,
    "segformer": 3,
    "depth": 1 / 3,
    "sam3_text": 3,
}


def ensemble_build_method_dirs(base_dir, depth_combo="N0.05_F0.95"):
    """Build dict of method_name -> output directory."""
    return {
        "segformer": base_dir / "segformer_output",
        "deeplabv3": base_dir / "deeplabv3_output",
        "depth": base_dir / "depth_output" / depth_combo,
        "sam3_text": base_dir / "sam3_dino_output" / "text",
        "sam3_dino": base_dir / "sam3_dino_output" / "dino+sam",
    }


def ensemble_parse_weights(weight_strings):
    """Parse weight arguments like 'sam3_dino=3 dinov2_pca=0.5' into a dict."""
    weights = {}
    for ws in weight_strings:
        if "=" not in ws:
            raise ValueError(f"Invalid weight format '{ws}', expected 'method=N'")
        name, val = ws.split("=", 1)
        weights[name] = float(val)
    return weights


def ensemble_process_one_image(args_tuple):
    """Load masks from all methods, vote, save results. Runs in worker process."""
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

    vote_count = np.zeros(ref_shape, dtype=np.float32)
    total_weight = 0
    for mask, w in zip(masks, weights):
        vote_count += mask.astype(np.float32) * w
        total_weight += w

    ensemble_mask = vote_count >= min_votes
    confidence = vote_count / max(total_weight, 1)
    ground_pct = ensemble_mask.sum() / ensemble_mask.size * 100

    cam_out = output_dir / camera
    mask_img = Image.fromarray((ensemble_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_ground.png")

    if save_confidence:
        conf_img = Image.fromarray((confidence * 255).astype(np.uint8))
        conf_img.save(cam_out / f"{stem}_confidence.png")

    if debug:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cmap = plt.cm.RdYlGn
        normalized = confidence
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


def add_ensemble_subparser(subparsers):
    p = subparsers.add_parser(
        "ensemble", help="Majority-vote ensemble across ground_teachers.py outputs")
    p.add_argument("--base-dir", type=str,
                    default=str(_REPO_ROOT / "preprocessing" / "segmentation"),
                    help="Base directory containing all method outputs")
    p.add_argument("--output-dir", type=str,
                    default=str(_REPO_ROOT / "preprocessing" / "segmentation" / "ensemble_output"),
                    help="Output directory")
    p.add_argument("--cameras", nargs="+", default=[],
                    help="Specific cameras (default: auto-detect)")
    p.add_argument("--min-votes", type=int, default=4,
                    help="Minimum votes for ground (default: 4 uniform, 5 weighted)")
    p.add_argument("--min-methods", type=int, default=3,
                    help="Skip images with fewer than this many methods available")
    p.add_argument("--depth-combo", type=str, default="N0.05_F0.95",
                    help="Depth parameter combo to use (default: N0.05_F0.95)")
    p.add_argument("--weighted", action="store_true",
                    help="Use weighted voting (default: segformer=3, sam3_text=3, "
                         "deeplabv3=3, sam3_dino=0.33, dinov2_pca=0.33, depth=0.33)")
    p.add_argument("--weights", nargs="*", default=[],
                    help="Custom weights as method=N pairs "
                         "(e.g., --weights sam3_dino=3 dinov2_pca=2)")
    p.add_argument("--save-confidence", action="store_true", default=True,
                    help="Save soft confidence maps (default: True)")
    p.add_argument("--no-confidence", action="store_true",
                    help="Skip saving confidence maps")
    p.add_argument("--workers", type=int, default=8,
                    help="Number of parallel workers")
    p.add_argument("--debug", action="store_true",
                    help="Save color-coded agreement maps")
    return p


def run_ensemble(args):
    base_dir = Path(args.base_dir)
    output_base = Path(args.output_dir)
    save_confidence = not args.no_confidence

    if args.weighted:
        method_weights = dict(ENSEMBLE_DEFAULT_WEIGHTS)
        if args.weights:
            method_weights.update(ensemble_parse_weights(args.weights))
        output_dir = output_base / f"weighted_{args.min_votes}"
    else:
        method_weights = {m: 1 for m in ensemble_build_method_dirs(base_dir).keys()}
        output_dir = output_base / f"majority_{args.min_votes}"

    method_dirs = ensemble_build_method_dirs(base_dir, args.depth_combo)

    available_methods = {}
    for name, d in method_dirs.items():
        if d.exists():
            available_methods[name] = d
        else:
            print(f"  WARNING: {name} not found at {d}")

    if len(available_methods) < 2:
        print(f"ERROR: Need at least 2 methods, found {len(available_methods)}")
        return

    # CM00024 (Zgornja Sorica) removed from the in-distribution pipeline —
    # thin ground strip / poor annotation consistency (see paper Methods).
    EXCLUDED_CAMERAS = {"CM00024"}

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
    print(f"ENSEMBLE - {mode_str} VOTING (CPU only)")
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

    for cam in cameras:
        (output_dir / cam).mkdir(parents=True, exist_ok=True)

    task_args = []
    for camera in cameras:
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

        all_stems = set()
        for s in method_stems.values():
            all_stems |= s

        matched = 0
        for stem in sorted(all_stems):
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

    all_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(ensemble_process_one_image, a) for a in task_args]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Ensembling"):
            result = future.result()
            if result is not None:
                all_results.append(result)

    if all_results:
        summary_path = output_dir / "ensemble_summary.csv"
        fieldnames = ["camera", "filename", "ground_pct", "n_methods",
                      "total_weight", "unanimous_ground_pct", "unanimous_bg_pct"]
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(all_results, key=lambda x: (x["camera"], x["filename"])):
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


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ground segmentation utilities: baseline + ensemble")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_baseline_subparser(subparsers)
    add_ensemble_subparser(subparsers)
    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline(args)
    elif args.command == "ensemble":
        run_ensemble(args)


if __name__ == "__main__":
    main()
