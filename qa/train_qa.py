#!/usr/bin/env python3
"""Train GradientBoosting QA classifier: Usable vs Non-Usable.

Extracts image-level features (color stats, edge density, temporal, meteo)
and trains a binary classifier. Key target: filtering "No Snow" images
at pipeline start before segmentation.

Usage:
    python train_qa.py
    python train_qa.py --output-dir /path/to/save
"""

import argparse
import json
import re
import sys
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


# ── Annotation parsing ────────────────────────────────────────────────────

def _taxonomy_to_label3(flat: set) -> int:
    """Map flat taxonomy set to 3-class label.
    0 = clean_no_snow (Non-Usable + No Snow only)
    1 = snow           (Usable)
    2 = non_usable     (Non-Usable + bad quality reason)
    """
    if "Usable" in flat:
        return 1
    if "Non-Usable" in flat:
        reasons = flat - {"Non-Usable"}
        if reasons == {"No Snow"} or reasons == set():
            return 0
        return 2
    return -1  # unknown


def parse_annotations(ann_dir):
    """Parse Label Studio JSON exports → list of {camera, stem, label, label3, reasons}."""
    ann_dir = Path(ann_dir)
    # CM00006 (06.json) and CM00068 (68.json) excluded per professor
    snow_files = [
        "san.json", "heroku.json",
        "01.json", "09.json", "14.json",
        "70.json", "75.json", "80.json", "102.json", "108.json", "112.json",
        "54.json", "67.json", "71.json", "115.json",
    ]

    records = []
    for fname in snow_files:
        ann_path = ann_dir / fname
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            tasks = json.load(f)

        for task in tasks:
            url = task["data"]["image"]
            parts = url.rstrip("/").split("/")
            filename = parts[-1]
            camera = parts[-2]
            stem = Path(filename).stem

            anns = task.get("annotations", [])
            if not anns or not anns[0].get("result"):
                continue
            result = anns[0]["result"]

            flat_all = set()
            reasons = set()
            label3 = -1
            for r in result:
                v = r.get("value", {})
                for key in ("taxonomy", "choices"):
                    tags = v.get(key)
                    if not tags:
                        continue
                    flat = set()
                    for entry in tags:
                        if isinstance(entry, list):
                            flat.update(entry)
                        else:
                            flat.add(entry)
                    flat_all.update(flat)
                    if "Non-Usable" in flat:
                        reasons.update(flat - {"Non-Usable"})

            if not flat_all:
                continue
            label3 = _taxonomy_to_label3(flat_all)
            if label3 < 0:
                continue
            # Binary label: clean_no_snow (0) and snow (1) are both "usable"
            is_usable = label3 in (0, 1)

            records.append({
                "camera": camera,
                "stem": stem,
                "filename": filename,
                "label": 1 if is_usable else 0,
                "label3": label3,
                "no_snow": 1 if "No Snow" in reasons else 0,
                "blurry": 1 if "Blurry" in reasons else 0,
                "bad_light": 1 if "Bad Light" in reasons else 0,
                "not_in_frame": 1 if "Not in Frame" in reasons else 0,
            })

    return pd.DataFrame(records)


# ── Feature extraction ────────────────────────────────────────────────────

def parse_timestamp(stem):
    """Extract datetime from filename like CF00001_202212111530_00."""
    m = re.search(r"(\d{12})", stem)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M")
    return None


def spatial_pyramid_cells(h, w, levels=(1, 2, 4)):
    """Yield (row_start, row_end, col_start, col_end, level, cell_idx) for SPM."""
    cell_idx = 0
    for level in levels:
        rh, rw = h // level, w // level
        for ri in range(level):
            for ci in range(level):
                r0, r1 = ri * rh, (ri + 1) * rh if ri < level - 1 else h
                c0, c1 = ci * rw, (ci + 1) * rw if ci < level - 1 else w
                yield r0, r1, c0, c1, level, cell_idx
                cell_idx += 1


def extract_hog_pyramid(gray, levels=(1, 2, 4), orientations=9):
    """HOG histograms per spatial pyramid cell."""
    h, w = gray.shape
    # Precompute gradients
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * (180.0 / np.pi) % 180.0

    features = {}
    for r0, r1, c0, c1, level, idx in spatial_pyramid_cells(h, w, levels):
        cell_mag = mag[r0:r1, c0:c1]
        cell_ang = ang[r0:r1, c0:c1]
        hist, _ = np.histogram(
            cell_ang.ravel(), bins=orientations, range=(0, 180),
            weights=cell_mag.ravel()
        )
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        for b in range(orientations):
            features[f"hog_L{level}_c{idx}_b{b}"] = hist[b]
    return features



def extract_sift_pyramid(gray, levels=(1, 2, 4)):
    """SIFT keypoint stats per spatial pyramid cell."""
    h, w = gray.shape
    sift = cv2.SIFT_create(nfeatures=500)
    kps, des = sift.detectAndCompute(gray, None)

    features = {}
    for r0, r1, c0, c1, level, idx in spatial_pyramid_cells(h, w, levels):
        cell_kps = [k for k in kps if r0 <= k.pt[1] < r1 and c0 <= k.pt[0] < c1]
        n_kp = len(cell_kps)
        features[f"sift_L{level}_c{idx}_n"] = n_kp
        if n_kp > 0:
            features[f"sift_L{level}_c{idx}_scale_mean"] = np.mean([k.size for k in cell_kps])
            features[f"sift_L{level}_c{idx}_response_mean"] = np.mean([k.response for k in cell_kps])
        else:
            features[f"sift_L{level}_c{idx}_scale_mean"] = 0.0
            features[f"sift_L{level}_c{idx}_response_mean"] = 0.0

    # Global SIFT stats
    features["sift_total_kp"] = len(kps)
    if des is not None and len(des) > 0:
        features["sift_desc_mean"] = des.mean()
        features["sift_desc_std"] = des.std()
    else:
        features["sift_desc_mean"] = 0.0
        features["sift_desc_std"] = 0.0

    return features


def extract_image_features(img_path):
    """Extract low-level image features (CPU, no model needed)."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    # RGB stats
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))

    # HSV stats
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    # Brightness and contrast
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    brightness = gray.mean()
    contrast = gray.std()

    # Edge density (Laplacian approx via simple gradient magnitude)
    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    edge_density = (np.abs(gy).mean() + np.abs(gx).mean()) / 2.0

    # Color uniformity (low entropy → uniform → possibly foggy/dark/no-snow)
    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))

    # Blue ratio (sky/snow indicator)
    total = r + g + b + 1e-6
    blue_ratio = (b / total).mean()

    # White pixel fraction (potential snow indicator)
    white_frac = ((r > 200) & (g > 200) & (b > 200)).mean()

    # Saturation stats (low sat → gray/foggy/snowy)
    sat = hsv[:, :, 1]
    low_sat_frac = (sat < 30).mean()

    # Bottom 70% stats (ground region more relevant)
    h_img = arr.shape[0]
    crop_start = int(h_img * 0.3)
    bottom = gray[crop_start:]
    bottom_brightness = bottom.mean()
    bottom_white_frac = ((r[crop_start:] > 200) & (g[crop_start:] > 200) & (b[crop_start:] > 200)).mean()

    # Spatial pyramid features (HOG + SIFT; LBP removed — negligible SHAP contribution)
    gray_uint8 = (gray.clip(0, 255)).astype(np.uint8)
    hog_feats = extract_hog_pyramid(gray_uint8)
    sift_feats = extract_sift_pyramid(gray_uint8)

    base = {
        "r_mean": rgb_mean[0], "g_mean": rgb_mean[1], "b_mean": rgb_mean[2],
        "r_std": rgb_std[0], "g_std": rgb_std[1], "b_std": rgb_std[2],
        "h_mean": hsv_mean[0], "s_mean": hsv_mean[1], "v_mean": hsv_mean[2],
        "h_std": hsv_std[0], "s_std": hsv_std[1], "v_std": hsv_std[2],
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "entropy": entropy,
        "blue_ratio": blue_ratio,
        "white_frac": white_frac,
        "low_sat_frac": low_sat_frac,
        "bottom_brightness": bottom_brightness,
        "bottom_white_frac": bottom_white_frac,
    }
    base.update(hog_feats)
    base.update(sift_feats)
    return base


def add_temporal_features(df):
    """Add time-based features from filename timestamps."""
    hours, doy_sin, doy_cos, hour_sin, hour_cos, months = [], [], [], [], [], []

    for stem in df["stem"]:
        ts = parse_timestamp(stem)
        if ts:
            h = ts.hour + ts.minute / 60.0
            doy = ts.timetuple().tm_yday
            hours.append(h)
            hour_sin.append(np.sin(2 * np.pi * h / 24))
            hour_cos.append(np.cos(2 * np.pi * h / 24))
            doy_sin.append(np.sin(2 * np.pi * doy / 365))
            doy_cos.append(np.cos(2 * np.pi * doy / 365))
            months.append(ts.month)
        else:
            hours.append(np.nan)
            hour_sin.append(np.nan)
            hour_cos.append(np.nan)
            doy_sin.append(np.nan)
            doy_cos.append(np.nan)
            months.append(np.nan)

    df["hour"] = hours
    df["hour_sin"] = hour_sin
    df["hour_cos"] = hour_cos
    df["doy_sin"] = doy_sin
    df["doy_cos"] = doy_cos
    df["month"] = months
    return df


def add_meteo_features(df, meteo_path):
    """Join meteo data on (camera, nearest hour)."""
    meteo = pd.read_csv(meteo_path, parse_dates=["datetime"])

    meteo_lookup = {}
    for _, row in meteo.iterrows():
        key = (row["camera"], row["datetime"])
        meteo_lookup[key] = row

    t2m, visibility, weather_code, snow_present_meteo = [], [], [], []
    for _, row in df.iterrows():
        ts = parse_timestamp(row["stem"])
        if ts is None:
            t2m.append(np.nan)
            visibility.append(np.nan)
            weather_code.append(np.nan)
            snow_present_meteo.append(np.nan)
            continue

        rounded = ts.replace(minute=0, second=0, microsecond=0)
        key = (row["camera"], pd.Timestamp(rounded))
        m = meteo_lookup.get(key)
        if m is not None:
            t2m.append(m["t2m"])
            visibility.append(m["visibility"])
            weather_code.append(m["weather_code"])
            snow_present_meteo.append(1 if m["snow_present"] else 0)
        else:
            t2m.append(np.nan)
            visibility.append(np.nan)
            weather_code.append(np.nan)
            snow_present_meteo.append(np.nan)

    df["t2m"] = t2m
    df["visibility"] = visibility
    df["weather_code"] = weather_code
    df["snow_present_meteo"] = snow_present_meteo
    return df


# ── Training ──────────────────────────────────────────────────────────────

BASE_FEATURE_COLS = [
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "h_mean", "s_mean", "v_mean", "h_std", "s_std", "v_std",
    "brightness", "contrast", "edge_density", "entropy",
    "blue_ratio", "white_frac", "low_sat_frac",
    "bottom_brightness", "bottom_white_frac",
]

# Timestamp-derived — always real, never NaN regardless of meteo availability.
TEMPORAL_COLS = ["hour", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month"]
# Actual ARSO meteo columns — only real when meteo_hourly.csv was found and
# joined. Kept separate from TEMPORAL_COLS so build_feature_cols() can drop
# them when meteo isn't available, instead of training on an all-NaN column
# (HistGradientBoostingClassifier's binning step crashes on those: zero
# distinct values makes its sliding_window_view(..., 2) call fail outright,
# not just perform badly).
METEO_COLS = ["t2m", "visibility", "weather_code", "snow_present_meteo"]
TEMPORAL_METEO_COLS = TEMPORAL_COLS + METEO_COLS  # back-compat for any external import


def build_feature_cols(sample_feats, include_meteo=True):
    """Build full feature col list from a sample feature dict."""
    pyramid_cols = sorted([
        k for k in sample_feats
        if k.startswith(("hog_", "sift_"))
    ])
    cols = BASE_FEATURE_COLS + pyramid_cols + TEMPORAL_COLS
    if include_meteo:
        cols += METEO_COLS
    return cols


# ── Mistake plotting (optional, --plot-mistakes) ───────────────────────────

def plot_mistake_grid(img_paths, titles, suptitle, save_path, ncols=5):
    """Plot a grid of images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(img_paths)
    if n == 0:
        print(f"  No images for: {suptitle}")
        return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    if nrows == 1:
        axes = [axes] if ncols == 1 else [axes]
    axes_flat = np.array(axes).flatten()

    for i, (img_path, title) in enumerate(zip(img_paths, titles)):
        ax = axes_flat[i]
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "NOT FOUND", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_mistakes(df, img_dir, out_dir):
    """Plot grids of GradBoost's CV false positives / false negatives.

    df must already have cv_pred and cv_prob_usable columns (see main()'s
    CV-predictions step).
    """
    fp = df[(df["label"] == 0) & (df["cv_pred"] == 1)].copy()
    fp = fp.sort_values("cv_prob_usable", ascending=False)

    fn = df[(df["label"] == 1) & (df["cv_pred"] == 0)].copy()
    fn = fn.sort_values("cv_prob_usable", ascending=True)

    print(f"False Positives (Non-Usable called Usable): {len(fp)}")
    print(f"False Negatives (Usable called Non-Usable): {len(fn)}")

    fp_paths = [img_dir / row["camera"] / f"{row['stem']}.jpg" for _, row in fp.iterrows()]
    fp_titles = []
    for _, row in fp.iterrows():
        reasons = []
        if row.get("no_snow") == 1:
            reasons.append("no_snow")
        if row.get("blurry") == 1:
            reasons.append("blurry")
        if row.get("bad_light") == 1:
            reasons.append("bad_light")
        if row.get("not_in_frame") == 1:
            reasons.append("not_in_frame")
        reason_str = ",".join(reasons) if reasons else "?"
        fp_titles.append(f"{row['camera']}\n{reason_str}\np={row['cv_prob_usable']:.2f}")

    fn_paths = [img_dir / row["camera"] / f"{row['stem']}.jpg" for _, row in fn.iterrows()]
    fn_titles = [f"{row['camera']}\np={row['cv_prob_usable']:.2f}" for _, row in fn.iterrows()]

    plot_mistake_grid(
        fp_paths, fp_titles,
        f"FALSE POSITIVES — Non-Usable predicted as Usable ({len(fp)})",
        out_dir / "mistakes_false_positives.png",
    )
    plot_mistake_grid(
        fn_paths, fn_titles,
        f"FALSE NEGATIVES — Usable predicted as Non-Usable ({len(fn)})",
        out_dir / "mistakes_false_negatives.png",
    )

    print("\nFP (Non-Usable but high prob_usable):")
    for _, row in fp.head(10).iterrows():
        print(f"  {row['camera']}/{row['stem']}.jpg  p={row['cv_prob_usable']:.3f}  "
              f"no_snow={int(row.get('no_snow',0))} blurry={int(row.get('blurry',0))}")

    print("\nFN (Usable but low prob_usable):")
    for _, row in fn.head(10).iterrows():
        print(f"  {row['camera']}/{row['stem']}.jpg  p={row['cv_prob_usable']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train QA GradBoost classifier")
    parser.add_argument("--ann-dir", type=str,
                        default=str(_ROOT / "data" / "annotations"))
    parser.add_argument("--img-dir", type=str,
                        default=str(_ROOT / "data" / "balanced_snow"))
    parser.add_argument("--meteo", type=str,
                        default=str(_ROOT / "data" / "meteo_hourly.csv"))
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE / "checkpoints"))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--hp-search", action="store_true",
                        help="Run randomized HP search instead of single training")
    parser.add_argument("--hp-iter", type=int, default=50,
                        help="Number of random HP combinations to try")
    parser.add_argument("--plot-mistakes", action="store_true",
                        help="Plot grids of CV false positives/negatives after training")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse annotations
    print("Parsing annotations...")
    df = parse_annotations(ann_dir)
    print(f"  {len(df)} labeled images: {(df['label']==1).sum()} usable, "
          f"{(df['label']==0).sum()} non-usable")
    print(f"  3-class: clean_no_snow={( df['label3']==0).sum()}  "
          f"snow={(df['label3']==1).sum()}  non_usable={(df['label3']==2).sum()}")
    print(f"  Reasons: no_snow={df['no_snow'].sum()}, blurry={df['blurry'].sum()}, "
          f"bad_light={df['bad_light'].sum()}, not_in_frame={df['not_in_frame'].sum()}")

    # 2. Extract image features (HOG + LBP + SIFT pyramids)
    print("\nExtracting image features (HOG/SIFT pyramid)...")
    img_features = []
    valid_idx = []
    for i, row in df.iterrows():
        img_path = img_dir / row["camera"] / f"{row['stem']}.jpg"
        if not img_path.exists():
            continue
        feats = extract_image_features(img_path)
        img_features.append(feats)
        valid_idx.append(i)
        if len(img_features) % 100 == 0:
            print(f"    {len(img_features)} images processed...")

    print(f"  {len(img_features)}/{len(df)} images found")
    df = df.loc[valid_idx].reset_index(drop=True)
    feat_df = pd.DataFrame(img_features)
    df = pd.concat([df, feat_df], axis=1)

    # 3. Temporal features
    print("Adding temporal features...")
    df = add_temporal_features(df)

    # 4. Meteo features
    meteo_path = Path(args.meteo)
    has_meteo = meteo_path.exists()
    if has_meteo:
        print("Adding meteo features...")
        df = add_meteo_features(df, meteo_path)
    else:
        print(f"  Warning: {meteo_path} not found, skipping meteo features "
              f"(dropped from FEATURE_COLS, not trained on as all-NaN)")
        for col in ["t2m", "visibility", "weather_code", "snow_present_meteo"]:
            df[col] = np.nan

    # Build feature column list dynamically (includes pyramid features)
    FEATURE_COLS = build_feature_cols(img_features[0], include_meteo=has_meteo)

    # Save feature CSV
    csv_path = output_dir / "qa_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Features saved to {csv_path}")

    # 5. Train
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Hyperparameter search mode ───────────────────────────────────
    if args.hp_search:
        from scipy.stats import uniform, randint
        from sklearn.model_selection import RandomizedSearchCV

        param_dist = {
            "max_iter": [100, 200, 300, 500, 800],
            "max_depth": [3, 4, 5, 6, 8, None],
            "learning_rate": uniform(0.01, 0.29),
            "min_samples_leaf": randint(1, 30),
            "max_leaf_nodes": [15, 31, 63, 127, None],
            "l2_regularization": uniform(0, 1.0),
        }
        base_clf = HistGradientBoostingClassifier(random_state=42)

        print(f"\n{'=' * 60}")
        print(f"HYPERPARAMETER SEARCH ({args.hp_iter} iterations, 5-fold CV)")
        print(f"{'=' * 60}")
        print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")

        search = RandomizedSearchCV(
            base_clf, param_dist, n_iter=args.hp_iter, cv=cv,
            scoring="roc_auc", random_state=42, n_jobs=-1, verbose=1,
        )
        search.fit(X, y)

        print(f"\nBest AUROC (CV): {search.best_score_:.4f}")
        print(f"Best params:")
        for k, v in sorted(search.best_params_.items()):
            print(f"  {k:<25s} {v}")

        # Show top 10 results
        results_df = pd.DataFrame(search.cv_results_)
        results_df = results_df.sort_values("rank_test_score")
        print(f"\nTop 10 configurations:")
        for i, (_, row) in enumerate(results_df.head(10).iterrows()):
            print(f"  #{i+1} AUROC={row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}  "
                  f"lr={row['param_learning_rate']:.3f} depth={row['param_max_depth']} "
                  f"iter={row['param_max_iter']} leaf={row['param_min_samples_leaf']}")

        # Save search results
        hp_path = output_dir / "hp_search_results.csv"
        results_df.to_csv(hp_path, index=False)
        print(f"\nFull results saved to {hp_path}")

        # Retrain best model on all data + evaluate
        clf = search.best_estimator_
        y_prob_cv = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred_cv = (y_prob_cv >= 0.5).astype(int)

        print(f"\n{'=' * 60}")
        print("BEST MODEL — 5-FOLD CV RESULTS")
        print(f"{'=' * 60}")
        print(classification_report(y, y_pred_cv, target_names=["Non-Usable", "Usable"]))
        print("Confusion matrix:")
        print(confusion_matrix(y, y_pred_cv))
        auroc = roc_auc_score(y, y_prob_cv)
        auprc = average_precision_score(y, y_prob_cv)
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")

        print("\nDone! Use best params to retrain: pass them as CLI args.")
        return

    # ── Normal training mode ─────────────────────────────────────────
    print(f"\nTraining HistGradBoost (max_iter={args.n_estimators}, "
          f"max_depth={args.max_depth})...")
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")

    clf = HistGradientBoostingClassifier(
        max_iter=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=42,
    )

    # Cross-validated predictions
    y_pred_cv = cross_val_predict(clf, X, y, cv=cv, method="predict")
    y_prob_cv = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print("\nUsable (1) vs Non-Usable (0):")
    print(classification_report(y, y_pred_cv, target_names=["Non-Usable", "Usable"]))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred_cv))
    auroc = roc_auc_score(y, y_prob_cv)
    auprc = average_precision_score(y, y_prob_cv)
    print(f"\nAUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    # No-snow specific analysis
    no_snow_mask = df["no_snow"].values == 1
    if no_snow_mask.sum() > 0:
        print(f"\n{'=' * 60}")
        print(f"NO-SNOW DETECTION (subset: {no_snow_mask.sum()} images)")
        print(f"{'=' * 60}")
        # Non-usable predicted as non-usable among no-snow images
        no_snow_pred = y_pred_cv[no_snow_mask]
        no_snow_true = y[no_snow_mask]  # all 0 (non-usable)
        detected = (no_snow_pred == 0).sum()
        print(f"  No-snow images correctly rejected: {detected}/{no_snow_mask.sum()} "
              f"({100*detected/no_snow_mask.sum():.1f}%)")

    # 6. Train final model on all data
    print("\nTraining final model on all data...")
    clf.fit(X, y)

    # Feature importance (permutation-based)
    from sklearn.inspection import permutation_importance
    print("Computing permutation importance...")
    perm = permutation_importance(clf, X, y, n_repeats=10, random_state=42)
    importances = perm.importances_mean
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\nTop 20 features (permutation importance):")
    for i in range(min(20, len(FEATURE_COLS))):
        idx = sorted_idx[i]
        print(f"  {FEATURE_COLS[idx]:<35s} {importances[idx]:.4f}")

    # Save model
    model_path = output_dir / "qa_gradboost.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "feature_cols": FEATURE_COLS,
            "class_names": ["Non-Usable", "Usable"],
        }, f)
    print(f"\nModel saved to {model_path}")

    # Save CV predictions for analysis
    df["cv_pred"] = y_pred_cv
    df["cv_prob_usable"] = y_prob_cv
    pred_path = output_dir / "qa_cv_predictions.csv"
    df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    if args.plot_mistakes:
        print("\nPlotting mistakes...")
        plot_mistakes(df, img_dir, output_dir)

    # SHAP analysis (optional — needs shap package)
    try:
        import shap
    except ImportError:
        print("\n[SKIP] shap not installed — skipping SHAP analysis")
        print("\nDone!")
        return
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_sorted = np.argsort(mean_shap)[::-1]
    print(f"\nTop 20 features (mean |SHAP|):")
    for i in range(min(20, len(FEATURE_COLS))):
        idx = shap_sorted[i]
        print(f"  {FEATURE_COLS[idx]:<35s} {mean_shap[idx]:.4f}")

    # Save SHAP summary plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = 30
    top_idx = shap_sorted[:top_n]
    top_names = [FEATURE_COLS[i] for i in top_idx]
    top_vals = mean_shap[top_idx]
    ax.barh(range(top_n), top_vals[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("QA Classifier — Top 30 Feature Importances (SHAP)")
    plt.tight_layout()
    shap_plot_path = output_dir / "shap_importance.png"
    fig.savefig(shap_plot_path, dpi=150)
    plt.close()
    print(f"SHAP bar plot saved to {shap_plot_path}")

    # SHAP beeswarm (top 20 features)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values[:, top_idx[:20]],
        X[:, top_idx[:20]],
        feature_names=[FEATURE_COLS[i] for i in top_idx[:20]],
        show=False,
    )
    beeswarm_path = output_dir / "shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"SHAP beeswarm saved to {beeswarm_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
