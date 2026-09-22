#!/usr/bin/env python3
"""Generate snow coverage comparison grid at 5 quantiles (0, 25, 50, 75, 100%) per camera."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
BASE_DIR = Path("/shared/home/juan.osorio/master")
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
SEG_DIR = BASE_DIR / "preprocessing/segmentation"
GROUND_DIR = SEG_DIR / "student_ground_output/mobilenet_majority3"
SNOW_DIR = SEG_DIR / "student_snow_output/mobilenet_gt"
SMOOTHER_CKPT = BASE_DIR / "models/temporal_snow/checkpoints/smoother_gru_h64_l3_k7_seq96_20260524_155800/best-epoch=19-val_mae=0.0218.ckpt"


def overlay_mask(img, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay binary mask on image with given color and alpha."""
    overlay = img.copy()
    overlay[mask > 127] = color
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


VALID_REGION_CAMERAS = {"CM00075", "CM00108"}
VALID_REGION_DIR = SEG_DIR / "camera_valid_regions"

def get_valid_region(camera, h, w):
    if camera not in VALID_REGION_CAMERAS:
        return None
    vr_path = VALID_REGION_DIR / f"{camera}.png"
    if not vr_path.exists():
        return None
    from PIL import Image as PILImage
    vr = np.array(PILImage.open(vr_path).convert("L").resize((w, h), PILImage.NEAREST))
    return vr


def load_triplet(camera, stem):
    """Load original image + ground mask + snow mask."""
    img_path = DATA_DIR / camera / f"{stem}.jpg"
    ground_path = GROUND_DIR / camera / f"{stem}_ground.png"
    snow_path = SNOW_DIR / camera / f"{stem}_snow.png"

    if not img_path.exists():
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    vr = get_valid_region(camera, h, w)

    # Ground overlay (green)
    ground_img = img.copy()
    if ground_path.exists():
        ground_mask = cv2.imread(str(ground_path), cv2.IMREAD_GRAYSCALE)
        if ground_mask is not None:
            if ground_mask.shape[:2] != (h, w):
                ground_mask = cv2.resize(ground_mask, (w, h))
            if vr is not None:
                ground_mask = np.where(vr > 127, ground_mask, 0).astype(np.uint8)
            ground_img = overlay_mask(img, ground_mask, color=(0, 200, 0), alpha=0.4)

    # Snow overlay (cyan)
    snow_img = img.copy()
    if snow_path.exists():
        snow_mask = cv2.imread(str(snow_path), cv2.IMREAD_GRAYSCALE)
        if snow_mask is not None:
            if snow_mask.shape[:2] != (h, w):
                snow_mask = cv2.resize(snow_mask, (w, h))
            if vr is not None:
                snow_mask = np.where(vr > 127, snow_mask, 0).astype(np.uint8)
            snow_img = overlay_mask(img, snow_mask, color=(255, 200, 0), alpha=0.5)

    return img, ground_img, snow_img


def add_label(img, text, font_scale=0.5):
    """Add text label to image."""
    img = img.copy()
    cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def run_smoother(cam_df, ckpt_path):
    """Run GRU smoother on camera dataframe, return smoothed snow_pct series."""
    try:
        import math
        import torch
        from models.temporal_snow.train_smoother import TemporalSmootherModule
        from models.temporal_snow.dataset import build_feature_columns, SNOW_PCT_COLS

        module = TemporalSmootherModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
        module.eval()

        df = cam_df.copy()
        snow_pct_cols = [c for c in SNOW_PCT_COLS if c in df.columns]
        if not snow_pct_cols:
            return None

        df["dt"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("dt").reset_index(drop=True)
        df["hour_sin"] = np.sin(2 * math.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * math.pi * df["hour"] / 24)
        doy = df["dt"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
        df["doy_cos"] = np.cos(2 * math.pi * doy / 365)

        FILL_DEFAULTS = {
            "t2m": 0, "visibility": 1000, "weather_code": 0,
            "brightness": 128, "contrast": 50, "edge_density": 10, "entropy": 4,
            "white_frac": 0, "low_sat_frac": 0, "blue_ratio": 0,
            "bottom_white_frac": 0, "bottom_brightness": 128,
            "snow_n_components": 0, "snow_largest_ratio": 0,
            "snow_centroid_y": 0.5, "ground_coverage": 0,
        }
        for col in snow_pct_cols:
            df[col] = df[col].fillna(0)
        for col, default in FILL_DEFAULTS.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

        if "visibility" in df.columns:
            df["visibility"] = np.log1p(df["visibility"])
        if "snow_n_components" in df.columns:
            df["snow_n_components"] = np.log1p(df["snow_n_components"])

        # Legacy normalization
        for col in snow_pct_cols:
            df[col] = df[col] / 100.0
        if "t2m" in df.columns:
            df["t2m"] = df["t2m"] / 30.0
        if "visibility" in df.columns:
            df["visibility"] = df["visibility"] / np.log1p(65000)
        if "weather_code" in df.columns:
            df["weather_code"] = df["weather_code"] / 100.0
        if "brightness" in df.columns:
            df["brightness"] = df["brightness"] / 255.0
            df["contrast"] = df["contrast"] / 128.0
            df["edge_density"] = df["edge_density"] / 50.0
            df["entropy"] = df["entropy"] / 6.0
            df["bottom_brightness"] = df["bottom_brightness"] / 255.0
        if "snow_n_components" in df.columns:
            df["snow_n_components"] = df["snow_n_components"] / np.log1p(100)

        input_size = module.hparams.get("input_size", None)
        feature_cols = build_feature_columns(snow_pct_cols, use_image_features=True)
        if input_size and len(feature_cols) != input_size:
            feature_cols = feature_cols[:input_size]

        x = df[feature_cols].values.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)

        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0)
            pred = module(x_t).squeeze(0).numpy()

        return pd.Series(pred * 100.0, index=df.index)

    except Exception as e:
        print(f"  Smoother failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", nargs="+", default=[
        "CF00001", "CM00009", "CM00014", "CM00070",
        "CM00075", "CM00080", "CM00102", "CM00108", "CM00112"
    ])
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0, 50, 100])
    parser.add_argument("--output-dir", default="analysis/grids")
    parser.add_argument("--combined", default=None, help="Save all cameras stacked into one PNG")
    parser.add_argument("--thumb-size", type=int, default=320)
    args = parser.parse_args()

    df = pd.read_csv(BASE_DIR / "analysis/snow_timeseries.csv")
    df = df[df["qa_usable"] == 1]

    thumb_w = args.thumb_size
    thumb_h = int(thumb_w * 0.75)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera in args.cameras:
        cam_df = df[df["camera"] == camera].copy()
        # Use last 20% of temporal split (evaluation set)
        cam_df = cam_df.sort_values("datetime").reset_index(drop=True)
        split_idx = int(len(cam_df) * 0.8)
        cam_df = cam_df.iloc[split_idx:].reset_index(drop=True)

        # Run GRU smoother for quantile selection
        smoothed = run_smoother(cam_df, SMOOTHER_CKPT)
        if smoothed is not None:
            cam_df["smoothed_snow_pct"] = smoothed.values
            sort_col = "smoothed_snow_pct"
            print(f"\n{camera}: using GRU smoothed snow%, range [{cam_df['smoothed_snow_pct'].min():.1f}, {cam_df['smoothed_snow_pct'].max():.1f}]%")
        else:
            cam_df["smoothed_snow_pct"] = cam_df["student_snow_pct"]
            sort_col = "student_snow_pct"
            print(f"\n{camera}: smoother unavailable, using raw snow_pct")

        cam_df = cam_df.sort_values(sort_col)

        row_imgs = []
        for q in args.quantiles:
            # Find image closest to this ACTUAL snow percentage value (not quantile)
            idx = (cam_df[sort_col] - q).abs().idxmin()
            row = cam_df.loc[idx]
            stem = row["stem"]
            actual_pct = row[sort_col]

            print(f"  Q{q:3.0f}%: {stem} (snow={actual_pct:.1f}%)")

            triplet = load_triplet(camera, stem)
            if triplet is None:
                print(f"  WARNING: images not found for {stem}")
                placeholder = np.zeros((thumb_h, thumb_w * 3, 3), dtype=np.uint8)
                row_imgs.append(placeholder)
                continue

            orig, ground_img, snow_img = triplet

            def resize(img):
                return cv2.resize(img, (thumb_w, thumb_h))

            orig_t = add_label(resize(orig), f"Snow {actual_pct:.1f}%")
            ground_t = add_label(resize(ground_img), "Ground")
            snow_t = add_label(resize(snow_img), "Snow")

            row_imgs.append(np.hstack([orig_t, ground_t, snow_t]))

        grid = np.vstack(row_imgs)
        output_path = output_dir / f"grid_{camera}.png"
        cv2.imwrite(str(output_path), grid)
        print(f"  Saved: {output_path} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
