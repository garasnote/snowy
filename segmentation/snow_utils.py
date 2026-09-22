#!/usr/bin/env python3
"""
Snow segmentation utilities: the deterministic HSV baseline, its
threshold-tuning helper, and the majority-vote ensemble that combines
teacher outputs from snow_teachers.py into the snow pseudo-labels used to
train the snow student models (Methods, "Snow Segmentation").

Consolidated from hsv_snow_baseline.py, hsv_eer_threshold.py, and
ensemble_snow_voting.py.

NOTE: the deterministic baseline reported in the paper is the PCA Blue-Band
method (Portenier et al.), not this HSV threshold -- that script was never
brought back from the cluster into this repo (see paper Methods for the
method description). This HSV baseline was an earlier/alternative approach,
kept here because it's what's actually available; do not cite it as "the"
paper baseline.

Usage:
    python snow_utils.py baseline --data-dir ./data --output-dir ./out
    python snow_utils.py tune-threshold --mask-dir ./out --meteo data/meteo_hourly.csv
    python snow_utils.py ensemble --min-votes 2
"""

import argparse
import csv
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

CAMERAS = [
    "CF00001", "CM00006", "CM00009", "CM00014",
    "CM00068", "CM00070", "CM00075", "CM00080",
    "CM00102", "CM00108", "CM00112",
]


# =============================================================================
# BASELINE — HSV white/bright-pixel threshold
# =============================================================================

def baseline_segment_snow_hsv(img_bgr, sat_max=30, val_min=200):
    """Detect white/bright pixels in HSV space.

    Returns binary mask: 1 = snow candidate, 0 = not snow.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] < sat_max) & (hsv[:, :, 2] > val_min)
    return mask.astype(np.uint8) * 255


def add_baseline_subparser(subparsers):
    p = subparsers.add_parser("baseline", help="HSV white/bright-pixel snow baseline")
    p.add_argument("--data-dir", type=Path, required=True,
                    help="Root data dir containing camera folders with JPEGs")
    p.add_argument("--ground-mask-dir", type=Path, default=None,
                    help="Dir with ground masks: {camera}/{stem}_ground.png (optional)")
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Output dir for snow masks")
    p.add_argument("--sat-max", type=int, default=30,
                    help="Max saturation for white detection (default: 30)")
    p.add_argument("--val-min", type=int, default=200,
                    help="Min value for white detection (default: 200)")
    p.add_argument("--cameras", nargs="*", default=None,
                    help="Cameras to process (default: all)")
    return p


def run_baseline(args):
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
            stem = img_path.stem

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

                white_mask = baseline_segment_snow_hsv(img_bgr, args.sat_max, args.val_min)

                if use_ground:
                    ground_mask = np.array(Image.open(ground_mask_path).convert("L"))
                    if ground_mask.shape[:2] != img_bgr.shape[:2]:
                        ground_mask = cv2.resize(ground_mask, (img_bgr.shape[1], img_bgr.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                    snow_mask = cv2.bitwise_and(white_mask, ground_mask)
                else:
                    snow_mask = white_mask

                cv2.imwrite(str(cam_out_dir / f"{stem}_snow.png"), snow_mask)
                cam_count += 1

            except Exception as e:
                print(f"[{cam}] Error on {stem}: {e}")
                skipped += 1

        total += cam_count
        print(f"[{cam}] {cam_count} masks saved")

    print(f"\nDone: {total} masks, {skipped} skipped")
    print(f"Output: {args.output_dir}")


# =============================================================================
# TUNE-THRESHOLD — Equal Error Rate sweep for the HSV baseline's coverage cutoff
# =============================================================================

def eer_load_meteo_hourly(csv_path):
    """Load meteo_hourly.csv -> {camera: {datetime_str: snow_present}}.

    Keys are ISO datetimes like '2024-01-29T09:00:00'.
    """
    snow_map = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            snow_map[row["camera"]][row["datetime"]] = row["snow_present"] == "True"
    return snow_map


def eer_extract_datetime(filename):
    """CM00108_202401290900_05_snow.png -> 2024-01-29T09:00:00"""
    m = re.search(r"_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})_", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:00"
    return None


def add_tune_threshold_subparser(subparsers):
    p = subparsers.add_parser(
        "tune-threshold",
        help="Find the Equal Error Rate coverage threshold for HSV baseline masks")
    p.add_argument("--mask-dir", type=Path, required=True,
                    help="Dir with HSV snow masks: {camera}/{stem}_snow.png")
    p.add_argument("--meteo", type=Path, required=True,
                    help="meteo_hourly.csv")
    p.add_argument("--data-dir", type=Path, default=None,
                    help="If set, only evaluate images present in this dir")
    p.add_argument("--cameras", nargs="*", default=None)
    return p


def run_tune_threshold(args):
    snow_map = eer_load_meteo_hourly(args.meteo)
    cameras = args.cameras or CAMERAS

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
            if args.data_dir:
                stem = mask_path.stem.replace("_snow", "")
                img_path = args.data_dir / cam / f"{stem}.jpg"
                if not img_path.exists():
                    skipped += 1
                    continue

            dt = eer_extract_datetime(mask_path.name)
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

    coverages = np.array([s[0] for s in samples])
    labels = np.array([s[1] for s in samples])
    n_snow = labels.sum()
    n_nosnow = len(labels) - n_snow
    print(f"Snow: {n_snow}, No-snow: {n_nosnow}")
    print()

    print(f"{'Thresh%':>7} {'FAR':>8} {'FRR':>8} {'Acc':>8}")
    print("-" * 35)

    best_acc = 0.0
    best_thresh = 0.0
    best_far = 0.0
    best_frr = 0.0

    for step in range(201):
        thresh = step / 10.0
        predicted_snow = coverages >= thresh

        if n_nosnow > 0:
            far = predicted_snow[~labels].sum() / n_nosnow
        else:
            far = 0.0

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


# =============================================================================
# ENSEMBLE — majority-vote aggregation across teacher outputs
# =============================================================================
# Combines binary snow masks from the HSV baseline, SAM3, YOLOe-26,
# Florence-2, GroundedSAM2, CLIPSeg, and SegGPT using per-pixel voting.
# CPU-only, reads existing teacher outputs. This is the actual methodology
# contribution described in Methods, "Snow Segmentation" -- the teacher
# scripts themselves are off-the-shelf pretrained models, this is what turns
# their outputs into a single set of pseudo-labels.

ENSEMBLE_DEFAULT_WEIGHTS = {
    "hsv": 1,
    "sam3": 1,
    "yoloe": 1,
    "florence2": 1,
    "groundedsam2": 1,
    "clipseg": 1,
    "seggpt": 1,
}


def ensemble_build_method_dirs(base_dir):
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


def ensemble_parse_weights(weight_strings):
    """Parse weight arguments like 'sam3=2 yoloe=3' into a dict."""
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
    snow_pct = ensemble_mask.sum() / ensemble_mask.size * 100

    cam_out = output_dir / camera
    mask_img = Image.fromarray((ensemble_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if save_confidence:
        conf_img = Image.fromarray((confidence * 255).astype(np.uint8))
        conf_img.save(cam_out / f"{stem}_confidence.png")

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


def add_ensemble_subparser(subparsers):
    p = subparsers.add_parser(
        "ensemble", help="Majority-vote ensemble across snow_teachers.py outputs")
    p.add_argument("--base-dir", type=str,
                    default=str(_REPO_ROOT / "preprocessing" / "segmentation"),
                    help="Base directory containing all method outputs")
    p.add_argument("--output-dir", type=str,
                    default=str(_REPO_ROOT / "preprocessing" / "segmentation" / "ensemble_snow_output"),
                    help="Output directory")
    p.add_argument("--cameras", nargs="+", default=[],
                    help="Specific cameras (default: auto-detect)")
    p.add_argument("--min-votes", type=int, default=2,
                    help="Minimum votes for snow (default: 2)")
    p.add_argument("--min-methods", type=int, default=2,
                    help="Skip images with fewer methods available")
    p.add_argument("--weighted", action="store_true",
                    help="Use weighted voting")
    p.add_argument("--weights", nargs="*", default=[],
                    help="Custom weights as method=N pairs")
    p.add_argument("--save-confidence", action="store_true", default=True)
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

    method_dirs = ensemble_build_method_dirs(base_dir)

    available_methods = {}
    for name, d in method_dirs.items():
        if d.exists() and any(d.iterdir()):
            available_methods[name] = d
        else:
            print(f"  NOTE: {name} not found at {d}")

    if len(available_methods) < 2:
        print(f"ERROR: Need at least 2 methods, found {len(available_methods)}")
        return

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
    print(f"ENSEMBLE SNOW - {mode_str} VOTING (CPU only)")
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
                p.stem.replace("_snow", "")
                for p in cam_dir.glob("*_snow.png")
            }

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

    all_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(ensemble_process_one_image, a) for a in task_args]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Ensembling"):
            result = future.result()
            if result is not None:
                all_results.append(result)

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


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Snow segmentation utilities: baseline + tune-threshold + ensemble")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_baseline_subparser(subparsers)
    add_tune_threshold_subparser(subparsers)
    add_ensemble_subparser(subparsers)
    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline(args)
    elif args.command == "tune-threshold":
        run_tune_threshold(args)
    elif args.command == "ensemble":
        run_ensemble(args)


if __name__ == "__main__":
    main()
