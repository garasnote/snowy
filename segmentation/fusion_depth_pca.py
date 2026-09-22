#!/usr/bin/env python3
"""
DINOv2 + Depth Anything V2 fusion — ground segmentation.

Combines pre-computed depth maps and DINOv2 PCA ground masks using
multiple fusion strategies. CPU-only, reads existing outputs, no GPU needed.

Fusion strategies:
  - intersect:        ground only if BOTH depth Otsu AND PCA agree (strictest)
  - union:            ground if EITHER method says so (most permissive)
  - pca_depth_filter: PCA mask, but remove pixels in the far/near depth zones
  - depth_pca_filter: depth Otsu mask, but keep only pixels near PCA regions (dilated)

Usage:
    # All cameras, all strategies
    python fusion_depth_pca.py

    # Single camera
    python fusion_depth_pca.py --cameras CM00006

    # Specific strategies only
    python fusion_depth_pca.py --strategies intersect pca_depth_filter

    # Custom depth thresholds
    python fusion_depth_pca.py --near-clip 0.05 --far-clip 0.95
"""

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ALL_STRATEGIES = ["intersect", "union", "pca_depth_filter", "depth_pca_filter"]


# ── I/O helpers (run in workers) ─────────────────────────────────────────────

def load_depth_fast(path):
    """Load inferno-colored depth PNG to float32 [0,1] via luminance.

    Inferno colormap has monotonically increasing luminance, so perceived
    luminance is a good proxy for the original depth ordering — sufficient
    for percentile thresholding.
    """
    img = np.array(Image.open(path).convert("RGB"))
    depth = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]) / 255.0
    return depth.astype(np.float32)


def load_mask(path):
    """Load binary ground mask PNG → boolean array."""
    return np.array(Image.open(path).convert("L")) > 127


def depth_to_ground(depth_map, method="otsu", near_clip=0.05, far_clip=0.95):
    """Segment ground from depth map using Otsu or fixed threshold."""
    near_val = np.percentile(depth_map, near_clip * 100) if near_clip > 0 else -1
    far_val = np.percentile(depth_map, far_clip * 100) if far_clip < 1 else 2

    valid = (depth_map > near_val) & (depth_map < far_val)
    valid_depths = depth_map[valid]

    if len(valid_depths) == 0:
        return np.zeros_like(depth_map, dtype=bool)

    if method == "otsu":
        from skimage.filters import threshold_otsu
        try:
            otsu_val = threshold_otsu(valid_depths)
            return (depth_map <= otsu_val) & valid
        except ValueError:
            return (depth_map <= 0.5) & valid
    else:
        cutoff = np.percentile(valid_depths, 0.4 * 100)
        return (depth_map <= cutoff) & valid


# ── Fusion strategies ────────────────────────────────────────────────────────

def fuse_masks(depth_ground, pca_ground, depth_map, strategy,
               near_clip=0.05, far_clip=0.95):
    """Apply a fusion strategy to combine depth and PCA ground masks.

    Args:
        depth_ground: bool (H,W) — depth-based ground mask (Otsu)
        pca_ground:   bool (H,W) — DINOv2 PCA ground mask
        depth_map:    float32 (H,W) — raw depth values [0,1]
        strategy:     str — fusion strategy name
        near_clip:    float — near percentile for valid depth range
        far_clip:     float — far percentile for valid depth range
    """
    if strategy == "intersect":
        # Strict: only pixels where BOTH methods agree
        return depth_ground & pca_ground

    elif strategy == "union":
        # Permissive: either method is enough
        return depth_ground | pca_ground

    elif strategy == "pca_depth_filter":
        # Start from PCA mask, but remove pixels that are in the far depth zone
        # (depth says "this is sky/mountain" → reject even if PCA says ground)
        # Uses a softer threshold: reject only the far_clip percentile, not Otsu
        far_val = np.percentile(depth_map, far_clip * 100) if far_clip < 1 else 2
        near_val = np.percentile(depth_map, near_clip * 100) if near_clip > 0 else -1
        not_too_far = depth_map < far_val
        not_too_near = depth_map > near_val
        return pca_ground & not_too_far & not_too_near

    elif strategy == "depth_pca_filter":
        # Start from depth mask, dilate PCA mask slightly before filtering
        # (PCA is coarse/blocky, so allow some tolerance around PCA boundary)
        from scipy.ndimage import binary_dilation
        pca_dilated = binary_dilation(pca_ground, iterations=5)
        return depth_ground & pca_dilated

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ── Worker function ──────────────────────────────────────────────────────────

def process_one_image(args_tuple):
    """Load depth + PCA, compute all fusion strategies, save masks. Runs in worker."""
    (depth_path, pca_path, camera, stem, strategies,
     near_clip, far_clip, method, output_dir) = args_tuple

    # Load inputs
    depth_map = load_depth_fast(depth_path)
    pca_ground = load_mask(pca_path)

    # PCA mask may be different resolution than depth — resize PCA to match depth
    dh, dw = depth_map.shape
    ph, pw = pca_ground.shape
    if (dh, dw) != (ph, pw):
        pca_pil = Image.fromarray(pca_ground.astype(np.uint8) * 255)
        pca_pil = pca_pil.resize((dw, dh), Image.NEAREST)
        pca_ground = np.array(pca_pil) > 127

    # Compute depth ground mask once
    depth_ground = depth_to_ground(depth_map, method=method,
                                   near_clip=near_clip, far_clip=far_clip)

    results = []

    for strategy in strategies:
        fused = fuse_masks(depth_ground, pca_ground, depth_map, strategy,
                           near_clip=near_clip, far_clip=far_clip)
        ground_pct = fused.sum() / fused.size * 100

        # Save directly in worker
        mask_img = Image.fromarray((fused * 255).astype(np.uint8))
        mask_img.save(output_dir / strategy / camera / f"{stem}_ground.png")

        results.append({
            "strategy": strategy,
            "camera": camera,
            "filename": f"{stem}.jpg",
            "ground_pct": round(ground_pct, 2),
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 + Depth Anything V2 fusion (CPU only, multiprocessing)"
    )
    _DATA = Path(__file__).resolve().parent.parent / "preprocessing" / "segmentation"
    parser.add_argument("--depth-dir", type=str,
                        default=str(_DATA / "depth_output" / "shared_depth"),
                        help="Directory with depth maps from Depth Anything V2")
    parser.add_argument("--pca-dir", type=str,
                        default=str(_DATA / "dinov2_pca_output"),
                        help="Directory with DINOv2 PCA ground masks")
    parser.add_argument("--output-dir", type=str,
                        default=str(_DATA / "fusion_output"),
                        help="Base output directory")
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific cameras (default: auto-detect)")
    parser.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES,
                        choices=ALL_STRATEGIES,
                        help=f"Fusion strategies (default: all)")
    parser.add_argument("--near-clip", type=float, default=0.05,
                        help="Depth near clip percentile (default: 0.05)")
    parser.add_argument("--far-clip", type=float, default=0.95,
                        help="Depth far clip percentile (default: 0.95)")
    parser.add_argument("--method", choices=["otsu", "threshold"],
                        default="otsu", help="Depth segmentation method")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")

    args = parser.parse_args()

    depth_dir = Path(args.depth_dir)
    pca_dir = Path(args.pca_dir)
    output_dir = Path(args.output_dir)
    strategies = args.strategies

    # Discover cameras present in BOTH depth and PCA dirs
    if args.cameras:
        cameras = args.cameras
    else:
        depth_cams = {d.name for d in depth_dir.iterdir()
                      if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)}
        pca_cams = {d.name for d in pca_dir.iterdir()
                    if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)}
        cameras = sorted(depth_cams & pca_cams)

    print("=" * 70)
    print("DINOv2 + DEPTH ANYTHING V2 — FUSION (CPU only)")
    print("=" * 70)
    print(f"  Depth dir:   {depth_dir}")
    print(f"  PCA dir:     {pca_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Cameras:     {cameras}")
    print(f"  Strategies:  {strategies}")
    print(f"  Depth clips: near={args.near_clip} far={args.far_clip}")
    print(f"  Method:      {args.method}")
    print(f"  Workers:     {args.workers}")
    print()

    # Pre-create ALL output directories (before submitting to workers)
    for strategy in strategies:
        for cam in cameras:
            (output_dir / strategy / cam).mkdir(parents=True, exist_ok=True)

    # Discover matching image pairs: depth {stem}_depth.png ↔ PCA {stem}_ground.png
    task_args = []
    for camera in cameras:
        depth_cam_dir = depth_dir / camera
        pca_cam_dir = pca_dir / camera

        if not depth_cam_dir.exists() or not pca_cam_dir.exists():
            print(f"  WARNING: {camera} missing in one of the dirs, skipping")
            continue

        # Index PCA masks by stem for fast lookup
        pca_stems = {p.stem.replace("_ground", ""): p
                     for p in pca_cam_dir.glob("*_ground.png")}

        depth_paths = sorted(depth_cam_dir.glob("*_depth.png"))
        matched = 0
        for dp in depth_paths:
            stem = dp.stem.replace("_depth", "")
            if stem in pca_stems:
                task_args.append((
                    dp, pca_stems[stem], camera, stem, strategies,
                    args.near_clip, args.far_clip, args.method, output_dir
                ))
                matched += 1

        print(f"  {camera}: {matched} matched pairs "
              f"(depth={len(depth_paths)}, pca={len(pca_stems)})")

    if not task_args:
        print("\nNo matching image pairs found!")
        return

    print(f"\n  Total: {len(task_args)} images x {len(strategies)} strategies "
          f"= {len(task_args) * len(strategies)} masks\n")

    # Process in parallel — all work happens in workers
    all_results = {s: [] for s in strategies}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one_image, a) for a in task_args]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing"):
            for r in future.result():
                all_results[r["strategy"]].append(r)

    # Save per-strategy summaries
    print()
    fieldnames = ["camera", "filename", "ground_pct"]
    for strategy in strategies:
        results = all_results[strategy]
        if not results:
            continue
        summary_path = output_dir / strategy / "fusion_summary.csv"
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(results, key=lambda x: (x["camera"], x["filename"])):
                writer.writerow({k: r[k] for k in fieldnames})

        avg_pct = np.mean([r["ground_pct"] for r in results])
        print(f"  {strategy}: {len(results)} images, avg ground={avg_pct:.1f}%")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
