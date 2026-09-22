#!/usr/bin/env python3
"""
Cherry-picked best results figure.
Finds images with highest IoU between student prediction and GT,
saves side-by-side: Original | GT overlay | Prediction overlay.

Usage:
    python plot_best_results.py --task snow --top 10 --out thesis_best_snow/
    python plot_best_results.py --task ground --top 10 --out thesis_best_ground/
    python plot_best_results.py --task both --top 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "DejaVu Sans"

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_DATA = _ROOT / "data"
_ANN  = _ROOT / "data" / "annotations"
_CKPT_GROUND = _ROOT / "models" / "student_ground" / "checkpoints"
_CKPT_SNOW   = _ROOT / "models" / "student_snow"   / "checkpoints"

sys.path.insert(0, str(_ROOT / "models"))

GT_COLOR   = np.array([60,  200,  60],  dtype=np.uint8)   # green  — GT
PRED_COLOR = np.array([255, 100,  50],  dtype=np.uint8)   # orange — prediction
ALPHA = 0.55

IN_DIST_CAMERAS = ["CF00001","CM00009","CM00014","CM00070","CM00075","CM00080","CM00102","CM00108","CM00112"]

SNOW_ANN_FILES = {
    "CF00001": _ANN / "01.json",
    "CM00009": _ANN / "09.json",
    "CM00014": _ANN / "14.json",
    "CM00070": _ANN / "70.json",
    "CM00075": _ANN / "75.json",
    "CM00080": _ANN / "80.json",
    "CM00102": _ANN / "102.json",
    "CM00108": _ANN / "108.json",
    "CM00112": _ANN / "112.json",
}
GROUND_ANN_FILES = [_ANN / "ground_annotations.json", _ANN / "san.json", _ANN / "heroku.json"]


def decode_rle(rle_list, w, h):
    from student_ground.dataset import decode_ls_rle
    return decode_ls_rle(rle_list, w, h).astype(bool)


def load_snow_gt(cameras):
    """Returns list of (camera, stem, mask_bool)."""
    results = []
    for cam in cameras:
        ann_file = SNOW_ANN_FILES.get(cam)
        if not ann_file or not ann_file.exists():
            continue
        with open(ann_file) as f:
            tasks = json.load(f)
        for task in tasks:
            url = task["data"]["image"]
            if cam not in url:
                continue
            stem = url.split("/")[-1].replace(".jpg","")
            for ann in task.get("annotations",[]):
                for r in ann.get("result",[]):
                    if r.get("type") != "brushlabels" or "rle" not in r.get("value",{}):
                        continue
                    w = r.get("original_width", 1296)
                    h = r.get("original_height", 960)
                    results.append((cam, stem, decode_rle(r["value"]["rle"], w, h)))
    return results


def load_ground_gt(cameras):
    results = []
    for ann_file in GROUND_ANN_FILES:
        if not ann_file.exists():
            continue
        with open(ann_file) as f:
            tasks = json.load(f)
        for task in tasks:
            url = task["data"]["image"]
            cam = next((c for c in cameras if c in url), None)
            if not cam:
                continue
            raw = url.split("/")[-1].replace(".jpg","")
            stem = raw.split("-",1)[1] if "-" in raw and raw.split("-")[1][:2] in ("CM","CF") else raw
            for ann in task.get("annotations",[]):
                for r in ann.get("result",[]):
                    if r.get("type") != "brushlabels" or "rle" not in r.get("value",{}):
                        continue
                    w = r.get("original_width", 1296)
                    h = r.get("original_height", 960)
                    results.append((cam, stem, decode_rle(r["value"]["rle"], w, h)))
    return results


def find_img(camera, stem):
    for d in [_DATA/camera, _DATA/f"_EXCLUDED_{camera}", _DATA/"balanced_snow"/camera]:
        p = d / f"{stem}.jpg"
        if p.exists():
            return p
    return None


def find_latest_ckpt(base, prefix):
    dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)])
    if not dirs: return None
    d = dirs[-1]
    final = d / "final.ckpt"
    if final.exists(): return final
    ckpts = sorted(d.glob("best-*.ckpt"))
    return ckpts[-1] if ckpts else None


@torch.no_grad()
def infer(model, img_path, arch):
    from student_ground.configs import get_config, IMAGENET_MEAN, IMAGENET_STD
    cfg = get_config(arch)
    sz = cfg["input_size"]
    img = np.array(Image.open(img_path).convert("RGB"))
    pil = Image.fromarray(img).resize((sz,sz), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32)/255.0
    arr = (arr - np.array(IMAGENET_MEAN,np.float32).reshape(1,1,3)) / \
               np.array(IMAGENET_STD, np.float32).reshape(1,1,3)
    t = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
    prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    ow, oh = Image.open(img_path).size
    return np.array(Image.fromarray((prob>=0.5).astype(np.uint8)*255)
                    .resize((ow,oh), Image.NEAREST)) > 127


def iou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def overlay(img_rgb, mask, color):
    out = img_rgb.copy().astype(np.float32)
    if mask is not None and mask.any():
        m = mask
        if m.shape != img_rgb.shape[:2]:
            m = np.array(Image.fromarray(m.astype(np.uint8)*255)
                         .resize((img_rgb.shape[1],img_rgb.shape[0]),Image.NEAREST))>127
        out[m] = out[m]*(1-ALPHA) + color*ALPHA
    return out.clip(0,255).astype(np.uint8)


def save_grid(rows, task, out_path, title):
    """rows: list of (camera, stem, iou, img_rgb, gt_mask, pred_mask)"""
    color = GT_COLOR if task == "ground" else np.array([100,160,255],dtype=np.uint8)
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.5 * n), squeeze=False)
    axes[0, 0].set_title("Original",     fontsize=11, fontweight="bold", pad=6)
    axes[0, 1].set_title("Ground Truth", fontsize=11, fontweight="bold", pad=6)
    axes[0, 2].set_title("Prediction",   fontsize=11, fontweight="bold", pad=6)
    for ri, (cam, stem, iou_val, img_rgb, gt_mask, pred_mask) in enumerate(rows):
        axes[ri, 0].imshow(img_rgb)
        axes[ri, 1].imshow(overlay(img_rgb, gt_mask, color))
        axes[ri, 2].imshow(overlay(img_rgb, pred_mask, PRED_COLOR))
        axes[ri, 0].set_ylabel(f"{cam}\nIoU={iou_val:.3f}", fontsize=8, rotation=0,
                                labelpad=60, va="center")
        for ci in range(3):
            axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])
            for sp in axes[ri, ci].spines.values(): sp.set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


GROUND_CKPTS = {
    "mobilenet":  sorted((_CKPT_GROUND).glob("mobilenet_majority_3_*/best-*.ckpt")),
    "dinov2_base": sorted((_CKPT_GROUND).glob("dinov2_base_majority_3_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
}
SNOW_CKPTS = {
    "mobilenet": sorted((_CKPT_SNOW / "snow_gt_mobilenet_20260522_052052").glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime),
    "convnext":  sorted((_CKPT_SNOW / "snow_gt_convnext_unfrozen").glob("best-*.ckpt"),        key=lambda p: p.stat().st_mtime),
    "vit_lora":  sorted((_CKPT_SNOW / "snow_gt_vit_lora_unfrozen").glob("best-*.ckpt"),        key=lambda p: p.stat().st_mtime),
}


def load_model(arch, ckpt_path):
    from student_ground.train import StudentSegmentationModule
    m = StudentSegmentationModule.load_from_checkpoint(str(ckpt_path), model_variant=arch, map_location="cpu")
    return m.model.eval()


def run(task, cameras, top_n, out_dir, arch=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    if task == "ground":
        ckpt_dict = GROUND_CKPTS
        gt_list = load_ground_gt(cameras)
        print(f"Ground GT: {len(gt_list)} annotated images")
    else:
        ckpt_dict = SNOW_CKPTS
        gt_list = load_snow_gt(cameras)
        print(f"Snow GT: {len(gt_list)} annotated images")

    # Pick arch: CLI override or first available
    use_arch = arch or next((a for a, ckpts in ckpt_dict.items() if ckpts), None)
    ckpt_list = ckpt_dict.get(use_arch, [])
    if not ckpt_list:
        print(f"ERROR: no checkpoint for {use_arch}"); return
    ckpt = ckpt_list[-1]

    print(f"Model: {use_arch}  ckpt: {ckpt.parent.name}/{ckpt.name}")
    model = load_model(use_arch, ckpt)
    arch = use_arch

    scored = []
    for cam, stem, gt_mask in gt_list:
        img_path = find_img(cam, stem)
        if img_path is None:
            continue
        pred = infer(model, img_path, arch)
        # resize gt to pred size if needed
        if gt_mask.shape != pred.shape:
            gt_mask = np.array(Image.fromarray(gt_mask.astype(np.uint8)*255)
                               .resize((pred.shape[1],pred.shape[0]),Image.NEAREST))>127
        score = iou(pred, gt_mask)
        scored.append((score, cam, stem, gt_mask, pred, img_path))
        print(f"  {cam} {stem}: IoU={score:.3f}")

    scored.sort(key=lambda x: x[0], reverse=True)

    # Always: per-camera best+worst → 2 grids (best grid, worst grid), 9 rows each
    from collections import defaultdict
    by_cam = defaultdict(list)
    for item in scored:
        by_cam[item[1]].append(item)

    # Cameras excluded from worst grid due to labelling errors
    WORST_EXCLUDE = {
        "ground": {"CF00001", "CM00014"},
        "snow":   {"CM00014", "CM00080", "CM00112"},
    }
    worst_exclude = WORST_EXCLUDE.get(task, set())

    best_rows, worst_rows = [], []
    for cam in sorted(by_cam):
        entries = by_cam[cam]
        for score, cam_, stem, gt_mask, pred, img_path in [entries[0]]:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            best_rows.append((cam_, stem, score, img_rgb, gt_mask, pred))
        pick = entries[-2] if cam in worst_exclude and len(entries) >= 2 else entries[-1]
        if cam in worst_exclude:
            print(f"  {cam}: using 2nd-worst (labelling error in worst)")
        for score, cam_, stem, gt_mask, pred, img_path in [pick]:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            worst_rows.append((cam_, stem, score, img_rgb, gt_mask, pred))

    task_label = task.capitalize()
    save_grid(best_rows,  task, out_dir / f"grid_best_{task}.png",
              f"Best {task_label} Segmentation per Camera")
    save_grid(worst_rows, task, out_dir / f"grid_worst_{task}.png",
              f"Worst {task_label} Segmentation per Camera")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["snow","ground","both"], default="both")
    parser.add_argument("--top", type=int, default=10, help="Top N globally; use -1 for per-camera best+worst")
    parser.add_argument("--out", default="thesis_best_results/")
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--arch", default=None, help="Model arch override (e.g. dinov2_base, convnext)")
    args = parser.parse_args()

    cameras = args.cameras or IN_DIST_CAMERAS
    tasks = ["snow","ground"] if args.task == "both" else [args.task]

    for task in tasks:
        out = Path(args.out) / task
        print(f"\n{'='*60}\n{task.upper()}\n{'='*60}")
        run(task, cameras, args.top, out, arch=args.arch)


if __name__ == "__main__":
    main()
