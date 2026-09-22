#!/usr/bin/env python3
"""
Bottom-N% crop evaluation for ground segmentation.

Crops both GT and predicted ground masks to the bottom N% of the image
(proxy for sky removal) and recomputes IoU. Compares with full-image IoU.

This tests the advisor's hypothesis: removing the top portion of the image
(where sky lives) should improve ground segmentation by eliminating sky
confusion.

Usage:
    python analysis/crop_evaluation.py
    python analysis/crop_evaluation.py --crop-pct 70
    python analysis/crop_evaluation.py --crop-pct 80
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MASTER = Path(__file__).resolve().parent.parent
OUTPUT_BASE = MASTER / "preprocessing" / "segmentation"
ANNOTATIONS_DIR = MASTER / "data" / "annotations"

EXCLUDED_CAMERAS = {"CM00024"}


# ── Label Studio RLE decoding ───────────────────────────────────────────────

class _InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)


def _bytes2bit(data):
    return "".join(
        str((data[num // 8] >> (7 - num % 8)) & 1) for num in range(len(data) * 8)
    )


def decode_ls_rle(rle_list, width, height):
    inp = _InputStream(_bytes2bit(bytes(rle_list)))
    num = inp.read(32)
    word_size = inp.read(5) + 1
    rle_sizes = [inp.read(4) + 1 for _ in range(4)]
    flat = np.zeros(num, dtype=np.uint8)
    i = 0
    while i < num:
        is_repeated = inp.read(1)
        rle_size_idx = inp.read(2)
        run_length = inp.read(rle_sizes[rle_size_idx])
        j = i + 1 + run_length
        if is_repeated:
            val = inp.read(word_size)
            flat[i:j] = val
        else:
            while i < j:
                flat[i] = inp.read(word_size)
                i += 1
        i = j
    n_channels = num // (width * height)
    if n_channels >= 4:
        rgba = flat.reshape(height, width, n_channels)
        return rgba[:, :, 3] > 0
    return flat[:width * height].reshape(height, width) > 0


# ── Ground annotation loading ──────────────────────────────────────────────

def load_ground_annotations():
    """Load ground truth annotations from ground_annotations.json."""
    ann_path = ANNOTATIONS_DIR / "ground_annotations.json"
    if not ann_path.exists():
        print(f"ERROR: {ann_path} not found")
        return {}

    with open(ann_path) as f:
        tasks = json.load(f)

    gt = {}
    for task in tasks:
        url = task["data"]["image"]
        parts = url.rstrip("/").split("/")
        filename = parts[-1]
        camera = parts[-2]
        if camera in EXCLUDED_CAMERAS:
            continue
        stem = Path(filename).stem
        annotations = task.get("annotations", [])
        if not annotations or not annotations[0].get("result"):
            continue
        result = annotations[0]["result"]
        w = result[0].get("original_width", 1296)
        h = result[0].get("original_height", 960)
        merged_mask = np.zeros((h, w), dtype=bool)
        for r in result:
            if r.get("type") == "brushlabels" and r.get("value", {}).get("rle"):
                rle = r["value"]["rle"]
                mask = decode_ls_rle(rle, w, h)
                merged_mask |= mask
        gt.setdefault(camera, {})[stem] = merged_mask

    return gt


# ── Ground method discovery ─────────────────────────────────────────────────

def discover_ground_methods():
    """Discover all ground segmentation methods (same as Streamlit registry)."""
    methods = {}

    # Zero-shot / ensemble methods
    for name, dirname in [
        ("SegFormer", "segformer_output"),
        ("DeepLabV3+", "deeplabv3_output"),
        ("SAM3+DINO", "sam3_dino_output"),
        ("DINOv2 PCA", "dinov2_pca_output"),
        ("DINOv3 PCA", "dinov3_pca_output"),
    ]:
        d = OUTPUT_BASE / dirname
        if d.exists():
            methods[name] = d

    # Ensemble
    ensemble_base = OUTPUT_BASE / "ensemble_output"
    if ensemble_base.exists():
        for sub in sorted(ensemble_base.iterdir()):
            if sub.is_dir():
                methods[f"Ensemble ({sub.name})"] = sub

    # Student models
    student_base = OUTPUT_BASE / "student_ground_output"
    if student_base.exists():
        for d in sorted(student_base.iterdir()):
            if d.is_dir() and not re.match(r"C[A-Z]\d+", d.name):
                methods[f"Student {d.name}"] = d

    # Fusion
    for fusion_name, fusion_dir_name in [
        ("Fusion", "fusion_output"),
        ("Fusion v3", "fusion_dinov3_output"),
    ]:
        fusion_dir = OUTPUT_BASE / fusion_dir_name
        if fusion_dir.exists():
            for sub in sorted(fusion_dir.iterdir()):
                if sub.is_dir():
                    methods[f"{fusion_name} {sub.name}"] = sub

    return methods


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_iou(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Crop evaluation: compare full-image vs bottom-N% IoU for ground segmentation")
    parser.add_argument("--crop-pct", type=float, default=70,
                        help="Keep bottom N%% of image (default: 70)")
    args = parser.parse_args()

    crop_frac = args.crop_pct / 100.0
    methods = discover_ground_methods()
    print(f"Ground methods ({len(methods)}): {list(methods.keys())}")
    print(f"Crop: bottom {args.crop_pct}%\n")

    gt_all = load_ground_annotations()
    n_imgs = sum(len(v) for v in gt_all.values())
    print(f"Ground annotations: {n_imgs} images across {len(gt_all)} cameras\n")

    if not gt_all:
        return

    rows = []
    for method_name, method_dir in methods.items():
        for cam, stems in gt_all.items():
            cam_dir = method_dir / cam
            if not cam_dir.exists():
                continue
            for s, gt_mask in stems.items():
                mask_path = cam_dir / f"{s}_ground.png"
                if not mask_path.exists():
                    continue
                pred = np.array(Image.open(mask_path).convert("L")) > 127

                # Resize pred to gt shape
                if pred.shape != gt_mask.shape:
                    pred = np.array(
                        Image.fromarray(pred.astype(np.uint8) * 255).resize(
                            (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                    ) > 127

                gt_pct = float(gt_mask.sum()) / gt_mask.size * 100

                # Full-image IoU
                iou_full = compute_iou(pred, gt_mask)

                # Cropped IoU
                h = gt_mask.shape[0]
                start_row = int(h * (1 - crop_frac))
                gt_crop = gt_mask[start_row:, :]
                pred_crop = pred[start_row:, :]
                gt_crop_pct = float(gt_crop.sum()) / gt_crop.size * 100
                iou_crop = compute_iou(pred_crop, gt_crop)

                rows.append({
                    "Method": method_name,
                    "Camera": cam,
                    "Stem": s,
                    "GT_ground_pct_full": round(gt_pct, 2),
                    "GT_ground_pct_crop": round(gt_crop_pct, 2),
                    "IoU_full": round(iou_full, 4),
                    "IoU_crop": round(iou_crop, 4),
                    "IoU_delta": round(iou_crop - iou_full, 4),
                })

    df = pd.DataFrame(rows)
    out_csv = MASTER / "analysis" / f"crop_ground_evaluation_{int(args.crop_pct)}pct.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df)} rows)\n")

    print(f"=== Results: {len(df)} rows ===\n")

    summary = df.groupby("Method").agg({
        "IoU_full": "mean",
        "IoU_crop": "mean",
        "IoU_delta": "mean",
    }).sort_values("IoU_crop", ascending=False)
    summary.columns = ["IoU (full)", f"IoU (bottom {int(args.crop_pct)}%)", "Delta"]
    print(summary.round(4).to_string())

    print(f"\n  Mean delta across all methods: "
          f"{df['IoU_delta'].mean():.4f}")
    print(f"  Positive delta = crop helps, negative = crop hurts")
    print("\nDone!")


if __name__ == "__main__":
    main()
