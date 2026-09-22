#!/usr/bin/env python3
"""
Build a snow percentage time series per camera.

Iterates over all snow mask directories, computes snow_pct per image,
sorts chronologically, and joins with meteorological data.

Output: analysis/snow_timeseries.csv

Usage:
    python analysis/build_snow_timeseries.py
    python analysis/build_snow_timeseries.py --cameras CF00001 CM00009
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

MASTER = Path(__file__).resolve().parent.parent
OUTPUT_BASE = MASTER / "preprocessing" / "segmentation"
GROUND_MASK_DIR = OUTPUT_BASE / "student_ground_output" / "mobilenet_ft"
VALID_REGION_DIR = OUTPUT_BASE / "camera_valid_regions"
METEO_HOURLY = MASTER / "data" / "meteo_hourly.csv"
METEO_DAILY = MASTER / "data" / "meteo_daily.csv"
QA_CNN_PATH = MASTER / "preprocessing" / "qa" / "checkpoints" / "qa_cnn.pt"
QA_GB_PATH = MASTER / "preprocessing" / "qa" / "checkpoints" / "qa_gradboost.pkl"

METHOD_DIRS = {
    "student": OUTPUT_BASE / "student_snow_output" / "mobilenet_gt",
}

_TS_RE = re.compile(r"_(\d{12})_")
_CAM_RE = re.compile(r"C[A-Z]\d+")


def parse_timestamp(stem):
    m = _TS_RE.search(stem)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M")


def compute_snow_features(args):
    """Load snow mask + ground mask, return snow_pct and spatial features."""
    from scipy.ndimage import label

    mask_path, ground_path = args[:2]
    vr_path = args[2] if len(args) > 2 else None
    try:
        snow = np.array(Image.open(mask_path).convert("L"))
        h, w = snow.shape

        # Load valid region mask if available
        valid_region = None
        if vr_path is not None and vr_path.exists():
            valid_region = np.array(
                Image.open(vr_path).convert("L").resize((w, h), Image.NEAREST)
            )

        snow_bin = snow > 127
        if valid_region is not None:
            snow_bin[valid_region <= 127] = False
        snow_pixels = snow_bin.sum()
        total_pixels = snow.size

        # Ground-normalized snow_pct
        ground_pixels = total_pixels
        ground_coverage = 1.0
        if ground_path is not None and ground_path.exists():
            ground = np.array(Image.open(ground_path).convert("L"))
            if ground.shape != snow.shape:
                ground = np.array(
                    Image.open(ground_path).convert("L").resize(
                        (w, h), Image.NEAREST
                    )
                )
            ground_bin = ground > 127
            if valid_region is not None:
                ground_bin[valid_region <= 127] = False
            ground_pixels = ground_bin.sum()
            ground_coverage = ground_pixels / total_pixels if total_pixels > 0 else 0
            snow_pixels = int((snow_bin & ground_bin).sum())  # intersection: snow within ground only

        snow_pct = float(snow_pixels) / ground_pixels * 100 if ground_pixels > 0 else 0.0

        # Spatial features from connected components
        n_components = 0
        largest_ratio = 0.0
        snow_centroid_y = 0.5
        if snow_pixels > 0:
            labeled, n_components = label(snow_bin)
            component_sizes = np.bincount(labeled.ravel())[1:]  # skip background
            largest_ratio = float(component_sizes.max()) / snow_pixels if len(component_sizes) > 0 else 0
            ys = np.where(snow_bin)[0]
            snow_centroid_y = float(ys.mean()) / snow.shape[0]

        return {
            "snow_pct": snow_pct,
            "n_components": n_components,
            "largest_ratio": largest_ratio,
            "snow_centroid_y": snow_centroid_y,
            "ground_coverage": ground_coverage,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Build snow time series CSV")
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--data-dir", type=str, default=str(MASTER / "data"),
                        help="Path to image data directory")
    parser.add_argument("--ground-mask-dir", type=str, default=str(GROUND_MASK_DIR),
                        help="Ground segmentation masks (snow%% = snow/ground, not snow/total)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip QA filtering (mark all images usable)")
    parser.add_argument("--qa-threshold", type=float, default=0.5)
    args = parser.parse_args()
    ground_mask_dir = Path(args.ground_mask_dir)

    data_dir = Path(args.data_dir)

    # Discover all cameras from all methods (union)
    all_cameras = set()
    available_methods = {}
    for name, d in METHOD_DIRS.items():
        if d.exists():
            available_methods[name] = d
            for cam_dir in d.iterdir():
                if cam_dir.is_dir() and _CAM_RE.match(cam_dir.name):
                    all_cameras.add(cam_dir.name)

    # Also discover cameras from data directory (includes images without snow masks)
    for cam_dir in data_dir.iterdir():
        if cam_dir.is_dir() and _CAM_RE.match(cam_dir.name):
            all_cameras.add(cam_dir.name)

    if args.cameras:
        all_cameras = set(args.cameras) & all_cameras

    cameras = sorted(all_cameras)
    print(f"Methods: {list(available_methods.keys())}")
    print(f"Cameras: {cameras}")
    print()

    # Discover ALL images from data directory (complete timeline)
    all_stems = {}  # {(camera, stem): {method: mask_path}}
    n_images = 0
    n_ir_skipped = 0
    for cam in cameras:
        cam_img_dir = data_dir / cam
        if not cam_img_dir.exists():
            continue
        for img_path in cam_img_dir.glob("*.jpg"):
            if "_IR_" in img_path.stem:
                n_ir_skipped += 1
                continue
            key = (cam, img_path.stem)
            all_stems.setdefault(key, {})
            n_images += 1

    if n_ir_skipped:
        print(f"Skipped {n_ir_skipped} IR (infrared) images")
    print(f"Total images from data dir: {n_images}")

    # Match with snow teacher masks
    n_matched = 0
    for method, method_dir in available_methods.items():
        for cam in cameras:
            cam_dir = method_dir / cam
            if not cam_dir.exists():
                continue
            for mask_path in cam_dir.glob("*_snow.png"):
                stem = mask_path.stem.replace("_snow", "")
                key = (cam, stem)
                if key in all_stems:
                    all_stems[key][method] = mask_path
                    n_matched += 1

    n_with_masks = sum(1 for v in all_stems.values() if v)
    print(f"Images with snow masks: {n_with_masks}")
    print(f"Images without masks: {len(all_stems) - n_with_masks} (snow_pct = 0)")

    # Check ground mask availability
    has_ground = ground_mask_dir.exists()
    if has_ground:
        print(f"Ground masks: {ground_mask_dir}")
        print("  snow_pct = snow pixels / ground pixels")
    else:
        print(f"WARNING: No ground masks at {ground_mask_dir}")
        print("  snow_pct = snow pixels / total pixels (less accurate)")

    # Compute snow_pct for all masks in parallel
    # Build task list: (key, method, snow_path, ground_path)
    tasks = []
    for key, method_paths in all_stems.items():
        cam, stem = key
        ground_path = ground_mask_dir / cam / f"{stem}_ground.png" if has_ground else None
        vr_path = VALID_REGION_DIR / f"{cam}.png"
        if not vr_path.exists():
            vr_path = None
        for method, path in method_paths.items():
            tasks.append((key, method, path, ground_path, vr_path))

    print(f"Total masks to read: {len(tasks)}")
    print("Computing snow percentages...")

    results = {}  # {(cam, stem, method): {snow_pct, n_components, ...}}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(compute_snow_features, (path, ground_path, vr_path)): (key, method)
            for key, method, path, ground_path, vr_path in tasks
        }
        for future in tqdm(as_completed(future_map), total=len(future_map)):
            (cam, stem), method = future_map[future]
            feats = future.result()
            if feats is not None:
                results[(cam, stem, method)] = feats

    # Build rows (all images, snow_pct=0 where no mask exists)
    rows = []
    for (cam, stem), method_paths in sorted(all_stems.items()):
        dt = parse_timestamp(stem)
        if dt is None:
            continue
        row = {
            "camera": cam,
            "stem": stem,
            "datetime": dt.isoformat(),
            "date": dt.strftime("%Y-%m-%d"),
            "hour": dt.hour,
            "month": dt.month,
        }
        for method in available_methods:
            feats = results.get((cam, stem, method))
            if feats is not None:
                row[f"{method}_snow_pct"] = feats["snow_pct"]
                # Spatial features only for student method (primary)
                if method == "student":
                    row["snow_n_components"] = feats["n_components"]
                    row["snow_largest_ratio"] = feats["largest_ratio"]
                    row["snow_centroid_y"] = feats["snow_centroid_y"]
                    row["ground_coverage"] = feats["ground_coverage"]
            else:
                row[f"{method}_snow_pct"] = 0.0
                if method == "student":
                    row["snow_n_components"] = 0
                    row["snow_largest_ratio"] = 0.0
                    row["snow_centroid_y"] = 0.5
                    row["ground_coverage"] = 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["camera", "datetime"]).reset_index(drop=True)
    print(f"\nBase timeseries: {len(df)} rows")

    # Join with meteo hourly
    if METEO_HOURLY.exists():
        print("Joining with meteo_hourly.csv...")
        meteo = pd.read_csv(METEO_HOURLY)
        meteo["hour_key"] = pd.to_datetime(meteo["datetime"]).dt.strftime("%Y%m%d%H")
        # Deduplicate (take first per camera+hour)
        meteo = meteo.drop_duplicates(subset=["camera", "hour_key"], keep="first")

        df["hour_key"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y%m%d%H")
        df = df.merge(
            meteo[["camera", "hour_key", "t2m", "weather_code", "visibility", "snow_present"]],
            on=["camera", "hour_key"],
            how="left",
        )
        df = df.drop(columns=["hour_key"])
        matched = df["t2m"].notna().sum()
        print(f"  Matched {matched}/{len(df)} rows with hourly meteo")

    # Join with meteo daily (for snow_cm + fallback snow_present)
    if METEO_DAILY.exists():
        print("Joining with meteo_daily.csv...")
        daily = pd.read_csv(METEO_DAILY)
        daily = daily.drop_duplicates(subset=["camera", "date"], keep="first")

        # Columns to pull from daily
        daily_cols = ["camera", "date", "snow_cm"]
        # Also pull snow_present as fallback for cameras without hourly meteo
        if "snow_present" in daily.columns:
            daily_cols.append("snow_present")
            daily = daily.rename(columns={"snow_present": "snow_present_daily"})
            daily_cols = [c.replace("snow_present", "snow_present_daily") for c in daily_cols]

        df = df.merge(
            daily[daily_cols],
            on=["camera", "date"],
            how="left",
        )
        matched = df["snow_cm"].notna().sum()
        print(f"  Matched {matched}/{len(df)} rows with daily snow_cm")

        # Fill missing snow_present from daily data (covers cameras without hourly meteo)
        if "snow_present_daily" in df.columns:
            before = df["snow_present"].notna().sum()
            df["snow_present"] = df["snow_present"].fillna(df["snow_present_daily"])
            after = df["snow_present"].notna().sum()
            print(f"  snow_present coverage: {before} → {after}/{len(df)} "
                  f"(+{after - before} from daily fallback)")
            df = df.drop(columns=["snow_present_daily"])

    # Join with image features (from extract_image_features.py)
    img_feat_path = MASTER / "analysis" / "image_features.csv"
    if img_feat_path.exists():
        print("Joining with image_features.csv...")
        img_feats = pd.read_csv(img_feat_path)
        before_cols = len(df.columns)
        df = df.merge(img_feats, on=["camera", "stem"], how="left")
        n_feat_cols = len(df.columns) - before_cols
        matched = df["brightness"].notna().sum()
        print(f"  Matched {matched}/{len(df)} rows with {n_feat_cols} image features")
    else:
        print(f"  [SKIP] {img_feat_path} not found — run extract_image_features.py first")

    # QA classification — CNN (preferred) or GradBoost fallback
    if args.skip_qa:
        df["qa_usable"] = 1
        df["prob_usable"] = np.nan
        print("QA filtering skipped (--skip-qa)")
    elif QA_CNN_PATH.exists():
        import torch
        from PIL import ImageFile
        print(f"Running QA classifier (CNN: {QA_CNN_PATH.name})...")
        qa_model = torch.jit.load(str(QA_CNN_PATH), map_location="cpu")
        qa_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qa_model = qa_model.to(device)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_size = 128

        probs = [np.nan] * len(df)
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        batch_size = 64
        batch_tensors = []
        batch_indices = []

        def flush_batch():
            if not batch_tensors:
                return
            batch = torch.cat(batch_tensors).to(device)
            with torch.no_grad():
                # TinyQACNN outputs 3 logits (clean_no_snow, snow, non_usable) —
                # NOT a single sigmoid logit. "usable" = 1 - P(non_usable).
                logits = qa_model(batch)
                batch_probs = (1.0 - torch.softmax(logits, dim=1)[:, 2]).cpu().numpy()
            for idx, p in zip(batch_indices, batch_probs):
                probs[idx] = float(p)
            batch_tensors.clear()
            batch_indices.clear()

        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="QA")):
            cam, stem = row["camera"], row["stem"]
            img_path = data_dir / cam / f"{stem}.jpg"
            if not img_path.exists():
                continue
            try:
                with Image.open(img_path) as test_img:
                    test_img.verify()
                img = Image.open(img_path).convert("RGB").resize(
                    (input_size, input_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = (arr - mean) / std
                tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
                batch_tensors.append(tensor)
                batch_indices.append(i)
                if len(batch_tensors) >= batch_size:
                    flush_batch()
            except Exception:
                probs[i] = 0.0

        flush_batch()

        df["prob_usable"] = probs
        df["qa_usable"] = (df["prob_usable"] >= args.qa_threshold).astype(int)
        df.loc[df["prob_usable"].isna(), "qa_usable"] = 1
        n_rejected = (df["qa_usable"] == 0).sum()
        print(f"  QA: {n_rejected}/{len(df)} images rejected "
              f"({100*n_rejected/len(df):.1f}%)")
    elif QA_GB_PATH.exists():
        import sys
        qa_dir = str(MASTER / "preprocessing" / "qa")
        if qa_dir not in sys.path:
            sys.path.insert(0, qa_dir)
        from predict_qa import load_qa_model, classify_images

        print(f"Running QA classifier (GradBoost: {QA_GB_PATH.name})...")
        clf, feature_cols = load_qa_model(str(QA_GB_PATH))
        qa_results = classify_images(
            clf, feature_cols, data_dir,
            meteo_path=str(METEO_HOURLY) if METEO_HOURLY.exists() else None,
            cameras=args.cameras,
            threshold=args.qa_threshold,
        )
        df["prob_usable"] = df.apply(
            lambda r: qa_results.get((r["camera"], r["stem"]), np.nan), axis=1
        )
        df["qa_usable"] = (df["prob_usable"] >= args.qa_threshold).astype(int)
        df.loc[df["prob_usable"].isna(), "qa_usable"] = 1
        n_rejected = (df["qa_usable"] == 0).sum()
        print(f"  QA: {n_rejected}/{len(df)} images rejected "
              f"({100*n_rejected/len(df):.1f}%)")
    else:
        df["qa_usable"] = 1
        df["prob_usable"] = np.nan
        print(f"  [SKIP] No QA model found")

    # Save
    out_path = MASTER / "analysis" / "snow_timeseries.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"Columns: {list(df.columns)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
