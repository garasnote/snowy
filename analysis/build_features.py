#!/usr/bin/env python3
"""
Feature-building utilities — produce the input CSVs everything else in
analysis/ and models/ reads. Not analysis themselves, just ETL.

Consolidated from build_snow_timeseries.py and extract_image_features.py.
Each keeps its original CLI flags, now behind a subcommand:

    python analysis/build_features.py timeseries
    python analysis/build_features.py timeseries --cameras CF00001 CM00009
    python analysis/build_features.py image-features
    python analysis/build_features.py image-features --cameras CF00001 --workers 16

Run `python analysis/build_features.py <subcommand> --help` for the full
option list.
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
DATA_DIR = MASTER / "data"

_CAM_RE = re.compile(r"C[A-Z]\d+")
_TS_RE = re.compile(r"_(\d{12})_")


# =============================================================================
# TIMESERIES — snow_pct per camera, joined with meteo + image features + QA
# =============================================================================

TS_OUTPUT_BASE = MASTER / "preprocessing" / "segmentation"
TS_GROUND_MASK_DIR = TS_OUTPUT_BASE / "student_ground_output" / "mobilenet_ft"
TS_VALID_REGION_DIR = TS_OUTPUT_BASE / "camera_valid_regions"
TS_METEO_HOURLY = MASTER / "data" / "meteo_hourly.csv"
TS_METEO_DAILY = MASTER / "data" / "meteo_daily.csv"
TS_QA_CNN_PATH = MASTER / "preprocessing" / "qa" / "checkpoints" / "qa_cnn.pt"
TS_QA_GB_PATH = MASTER / "preprocessing" / "qa" / "checkpoints" / "qa_gradboost.pkl"

TS_METHOD_DIRS = {
    "student": TS_OUTPUT_BASE / "student_snow_output" / "mobilenet_gt",
}


def ts_parse_timestamp(stem):
    m = _TS_RE.search(stem)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M")


def ts_compute_snow_features(args):
    """Load snow mask + ground mask, return snow_pct and spatial features."""
    from scipy.ndimage import label

    mask_path, ground_path = args[:2]
    vr_path = args[2] if len(args) > 2 else None
    try:
        snow = np.array(Image.open(mask_path).convert("L"))
        h, w = snow.shape

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


def add_timeseries_subparser(subparsers):
    p = subparsers.add_parser("timeseries", help="Build analysis/snow_timeseries.csv")
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--data-dir", type=str, default=str(MASTER / "data"),
                    help="Path to image data directory")
    p.add_argument("--ground-mask-dir", type=str, default=str(TS_GROUND_MASK_DIR),
                    help="Ground segmentation masks (snow%% = snow/ground, not snow/total)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--skip-qa", action="store_true",
                    help="Skip QA filtering (mark all images usable)")
    p.add_argument("--qa-threshold", type=float, default=0.5)
    return p


def run_timeseries(args):
    ground_mask_dir = Path(args.ground_mask_dir)
    data_dir = Path(args.data_dir)

    all_cameras = set()
    available_methods = {}
    for name, d in TS_METHOD_DIRS.items():
        if d.exists():
            available_methods[name] = d
            for cam_dir in d.iterdir():
                if cam_dir.is_dir() and _CAM_RE.match(cam_dir.name):
                    all_cameras.add(cam_dir.name)

    for cam_dir in data_dir.iterdir():
        if cam_dir.is_dir() and _CAM_RE.match(cam_dir.name):
            all_cameras.add(cam_dir.name)

    if args.cameras:
        all_cameras = set(args.cameras) & all_cameras

    cameras = sorted(all_cameras)
    print(f"Methods: {list(available_methods.keys())}")
    print(f"Cameras: {cameras}")
    print()

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

    has_ground = ground_mask_dir.exists()
    if has_ground:
        print(f"Ground masks: {ground_mask_dir}")
        print("  snow_pct = snow pixels / ground pixels")
    else:
        print(f"WARNING: No ground masks at {ground_mask_dir}")
        print("  snow_pct = snow pixels / total pixels (less accurate)")

    tasks = []
    for key, method_paths in all_stems.items():
        cam, stem = key
        ground_path = ground_mask_dir / cam / f"{stem}_ground.png" if has_ground else None
        vr_path = TS_VALID_REGION_DIR / f"{cam}.png"
        if not vr_path.exists():
            vr_path = None
        for method, path in method_paths.items():
            tasks.append((key, method, path, ground_path, vr_path))

    print(f"Total masks to read: {len(tasks)}")
    print("Computing snow percentages...")

    results = {}  # {(cam, stem, method): {snow_pct, n_components, ...}}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(ts_compute_snow_features, (path, ground_path, vr_path)): (key, method)
            for key, method, path, ground_path, vr_path in tasks
        }
        for future in tqdm(as_completed(future_map), total=len(future_map)):
            (cam, stem), method = future_map[future]
            feats = future.result()
            if feats is not None:
                results[(cam, stem, method)] = feats

    rows = []
    for (cam, stem), method_paths in sorted(all_stems.items()):
        dt = ts_parse_timestamp(stem)
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

    if TS_METEO_HOURLY.exists():
        print("Joining with meteo_hourly.csv...")
        meteo = pd.read_csv(TS_METEO_HOURLY)
        meteo["hour_key"] = pd.to_datetime(meteo["datetime"]).dt.strftime("%Y%m%d%H")
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

    if TS_METEO_DAILY.exists():
        print("Joining with meteo_daily.csv...")
        daily = pd.read_csv(TS_METEO_DAILY)
        daily = daily.drop_duplicates(subset=["camera", "date"], keep="first")

        daily_cols = ["camera", "date", "snow_cm"]
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

        if "snow_present_daily" in df.columns:
            before = df["snow_present"].notna().sum()
            df["snow_present"] = df["snow_present"].fillna(df["snow_present_daily"])
            after = df["snow_present"].notna().sum()
            print(f"  snow_present coverage: {before} → {after}/{len(df)} "
                  f"(+{after - before} from daily fallback)")
            df = df.drop(columns=["snow_present_daily"])

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
        print(f"  [SKIP] {img_feat_path} not found — run "
              f"'python analysis/build_features.py image-features' first")

    if args.skip_qa:
        df["qa_usable"] = 1
        df["prob_usable"] = np.nan
        print("QA filtering skipped (--skip-qa)")
    elif TS_QA_CNN_PATH.exists():
        import torch
        from PIL import ImageFile
        print(f"Running QA classifier (CNN: {TS_QA_CNN_PATH.name})...")
        qa_model = torch.jit.load(str(TS_QA_CNN_PATH), map_location="cpu")
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
    elif TS_QA_GB_PATH.exists():
        import sys
        qa_dir = str(MASTER / "preprocessing" / "qa")
        if qa_dir not in sys.path:
            sys.path.insert(0, qa_dir)
        from predict_qa import load_qa_model, classify_images

        print(f"Running QA classifier (GradBoost: {TS_QA_GB_PATH.name})...")
        clf, feature_cols = load_qa_model(str(TS_QA_GB_PATH))
        qa_results = classify_images(
            clf, feature_cols, data_dir,
            meteo_path=str(TS_METEO_HOURLY) if TS_METEO_HOURLY.exists() else None,
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

    out_path = MASTER / "analysis" / "snow_timeseries.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"Columns: {list(df.columns)}")
    print("\nDone!")


# =============================================================================
# IMAGE-FEATURES — cheap CPU-only per-image descriptors (QA + GRU input)
# =============================================================================

IMGFEAT_FEATURE_NAMES = [
    "brightness", "contrast", "edge_density", "entropy",
    "white_frac", "low_sat_frac", "blue_ratio",
    "bottom_brightness", "bottom_white_frac",
    "r_mean", "g_mean", "b_mean",
    "h_mean", "s_mean", "v_mean",
    "r_std", "g_std", "b_std",
    "h_std", "s_std", "v_std",
]


def imgfeat_extract_features(img_path):
    """Extract lightweight image features (CPU, ~10ms per image)."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    if img.width > 640:
        ratio = 640 / img.width
        img = img.resize((640, int(img.height * ratio)), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    brightness = gray.mean()
    contrast = gray.std()

    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    edge_density = (np.abs(gy).mean() + np.abs(gx).mean()) / 2.0

    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))

    white_frac = float(((r > 200) & (g > 200) & (b > 200)).mean())

    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    sat = hsv[:, :, 1]
    low_sat_frac = float((sat < 30).mean())

    total = r + g + b + 1e-6
    blue_ratio = float((b / total).mean())

    h_img = arr.shape[0]
    crop_start = int(h_img * 0.3)
    bottom_gray = gray[crop_start:]
    bottom_brightness = float(bottom_gray.mean())
    bottom_white_frac = float(
        ((r[crop_start:] > 200) & (g[crop_start:] > 200) & (b[crop_start:] > 200)).mean()
    )

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


def imgfeat_process_one(img_path):
    """Worker function for parallel processing."""
    camera = img_path.parent.name
    stem = img_path.stem
    feats = imgfeat_extract_features(img_path)
    if feats is None:
        return None
    feats["camera"] = camera
    feats["stem"] = stem
    return feats


def add_image_features_subparser(subparsers):
    p = subparsers.add_parser("image-features", help="Build analysis/image_features.csv")
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", type=str,
                    default=str(MASTER / "analysis" / "image_features.csv"))
    return p


def run_image_features(args):
    data_dir = Path(args.data_dir)

    if args.cameras:
        cameras = args.cameras
    else:
        cameras = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and _CAM_RE.match(d.name)
        ])

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
        futures = {pool.submit(imgfeat_process_one, p): p for p in all_images}
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                results.append(row)
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(all_images)} done...")

    df = pd.DataFrame(results)
    df = df.sort_values(["camera", "stem"]).reset_index(drop=True)

    cols = ["camera", "stem"] + IMGFEAT_FEATURE_NAMES
    df = df[cols]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"  {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build feature CSVs for analysis/ and models/")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_timeseries_subparser(subparsers)
    add_image_features_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "timeseries":
        run_timeseries(args)
    elif args.command == "image-features":
        run_image_features(args)


if __name__ == "__main__":
    main()
