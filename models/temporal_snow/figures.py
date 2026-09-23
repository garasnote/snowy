#!/usr/bin/env python3
"""
One-off manuscript figure generators — no shared sbatch job, each run
manually to produce a specific paper figure.

Consolidated from plot_ground_comparison.py, plot_teacher_comparison.py,
plot_best_results.py, plot_thesis_figure.py, shap_gru.py. Each keeps its
original CLI flags, now behind a subcommand:

    python figures.py ground-comparison --stem CF00001_202302011000_00
    python figures.py teacher-comparison --stem CF00001_202302011000_00
    python figures.py best-results --task both --top 10
    python figures.py smoother-week --checkpoint path/to/best.ckpt
    python figures.py shap

Run `python figures.py <subcommand> --help` for the full option list.
best-results is the actual generator behind the paper's
grid_best_ground/grid_worst_ground/grid_best_snow/grid_worst_snow
appendix figures.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "DejaVu Sans"

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SEG  = _ROOT / "preprocessing" / "segmentation"
_DATA = _ROOT / "data"
_ANN  = _ROOT / "data" / "annotations"

sys.path.insert(0, str(_ROOT / "models"))


# =============================================================================
# GROUND-COMPARISON — ground segmentation: teachers, ensembles, baselines, students
# =============================================================================

GC_CKPT = _ROOT / "models" / "student_ground" / "checkpoints"

GC_GROUND_COLOR = np.array([60, 180, 60], dtype=np.uint8)   # green overlay
GC_ALPHA = 0.45

GC_TEACHERS = [
    ("DINOv2-PCA",     _SEG / "dinov2_pca_output"),
    ("DINOv3-PCA",     _SEG / "dinov3_pca_output"),
    ("SegFormer",      _SEG / "segformer_output"),
    ("DeepLabV3+",     _SEG / "deeplabv3_output"),
    ("SAM3 (text)",    _SEG / "sam3_dino_output" / "text"),
    ("SAM3+DINOv2",    _SEG / "sam3_dino_output" / "dino+sam"),
    ("Depth Anything", _SEG / "depth_output" / "N0.05_F0.95"),
]

GC_ENSEMBLES = [
    ("Majority-2",   _SEG / "ensemble_output" / "majority_2"),
    ("Majority-3",   _SEG / "ensemble_output" / "majority_3"),
    ("Majority-4",   _SEG / "ensemble_output" / "majority_4"),
    ("Majority-5",   _SEG / "ensemble_output" / "majority_5"),
    ("Weighted-3",   _SEG / "ensemble_output" / "weighted_3"),
]

GC_BASELINES = [
    ("Fixed-32%",    _SEG / "baseline_fixed_output"),
    ("Sobel",        _SEG / "baseline_sobel_output"),
]

GC_STUDENT_ARCHS  = ["mobilenet", "mobilenet_rgbd", "efficientnet", "dinov2_base", "dinov2_large", "dinov3_base", "dinov3_large"]
GC_STUDENT_LABELS = ["MobileNet", "MobileNet+D", "EfficientNet-B0", "DINOv2-Base", "DINOv2-Large", "DINOv3-Base", "DINOv3-Large"]


def gc_load_mask(folder: Path, cam: str, stem: str):
    path = folder / cam / f"{stem}_ground.png"
    if not path.exists():
        print(f"  WARNING: missing {path}")
        return None
    return np.array(Image.open(path).convert("L")) > 127


def gc_find_latest_ckpt(arch: str) -> Path:
    for prefix in [f"{arch}_majority_3_", f"{arch}_majority_", f"{arch}_weighted_", f"{arch}_"]:
        dirs = sorted([d for d in GC_CKPT.iterdir() if d.is_dir() and d.name.startswith(prefix)])
        if dirs:
            ckpts = sorted(dirs[-1].glob("*.ckpt"))
            if ckpts:
                return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found for {arch}")


@torch.no_grad()
def gc_infer_student(ckpt_path: Path, arch: str, img_path: Path) -> np.ndarray:
    from student_ground.configs import get_config, IMAGENET_MEAN, IMAGENET_STD
    from student_ground.train import StudentSegmentationModule
    from student_ground.dataset import load_depth_luminance

    cfg = get_config(arch)
    input_size = cfg["input_size"]
    use_depth = cfg.get("use_depth", False)

    print(f"  Loading {arch} from {ckpt_path.parent.name}/{ckpt_path.name}")
    module = StudentSegmentationModule.load_from_checkpoint(
        str(ckpt_path), model_variant=arch, map_location="cpu"
    )
    model = module.model.eval()

    img = np.array(Image.open(img_path).convert("RGB"))
    pil = Image.fromarray(img).resize((input_size, input_size), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    if use_depth:
        depth_dir = _ROOT / "preprocessing" / "segmentation" / "depth_output"
        cam = img_path.parent.name
        dp = depth_dir / cam / f"{img_path.stem}_depth.png"
        if dp.exists():
            depth = load_depth_luminance(str(dp))
            depth_t = torch.from_numpy(
                np.array(Image.fromarray((depth*255).astype(np.uint8))
                         .resize((input_size, input_size), Image.BILINEAR)
                         ).astype(np.float32) / 255.0
            ).unsqueeze(0).unsqueeze(0)
            tensor = torch.cat([tensor, depth_t], dim=1)

    logits = model(tensor)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    orig_w, orig_h = Image.open(img_path).size
    binary = (prob >= 0.5).astype(np.uint8) * 255
    return np.array(Image.fromarray(binary).resize((orig_w, orig_h), Image.NEAREST)) > 127


def gc_overlay(img_rgb: np.ndarray, mask) -> np.ndarray:
    out = img_rgb.copy().astype(np.float32)
    if mask is not None and mask.any():
        out[mask] = out[mask] * (1 - GC_ALPHA) + GC_GROUND_COLOR * GC_ALPHA
    return out.clip(0, 255).astype(np.uint8)


def add_ground_comparison_subparser(subparsers):
    p = subparsers.add_parser("ground-comparison",
                               help="Ground segmentation comparison: teachers, ensembles, baselines, students")
    p.add_argument("--stem", default="CF00001_202302011000_00")
    p.add_argument("--out", default="thesis_ground_panels/")
    p.add_argument("--no-students", action="store_true")
    return p


def run_ground_comparison(args):
    stem = args.stem
    cam  = stem[:7]

    img_path = _DATA / cam / f"{stem}.jpg"
    if not img_path.exists():
        img_path = _DATA / "balanced_snow" / cam / f"{stem}.jpg"
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    print(f"Image: {img_path}  ({img_rgb.shape[1]}x{img_rgb.shape[0]})")

    panels = [("Original", img_rgb)]

    print("\n--- Teachers ---")
    for label, folder in GC_TEACHERS:
        panels.append((label, gc_overlay(img_rgb, gc_load_mask(folder, cam, stem))))

    print("\n--- Ensembles ---")
    for label, folder in GC_ENSEMBLES:
        panels.append((label, gc_overlay(img_rgb, gc_load_mask(folder, cam, stem))))

    print("\n--- Baselines ---")
    for label, folder in GC_BASELINES:
        panels.append((label, gc_overlay(img_rgb, gc_load_mask(folder, cam, stem))))

    if not args.no_students:
        print("\n--- Students (live inference) ---")
        for arch, label in zip(GC_STUDENT_ARCHS, GC_STUDENT_LABELS):
            try:
                ckpt = gc_find_latest_ckpt(arch)
                panels.append((label, gc_overlay(img_rgb, gc_infer_student(ckpt, arch, img_path))))
            except Exception as e:
                print(f"  SKIP {label}: {e}")
                panels.append((label + " [N/A]", img_rgb))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for title, panel_img in panels:
        if panel_img is None or title == "":
            continue
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        ax.imshow(panel_img)
        ax.set_title(title, fontsize=11, fontweight="bold" if title == "Original" else "normal", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        fname = title.replace("+", "plus").replace(" ", "_").replace("/", "_") + ".png"
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_dir / fname}")


# =============================================================================
# TEACHER-COMPARISON — snow segmentation: teacher + student mask comparison
# =============================================================================

TC_CKPT = _ROOT / "models" / "student_snow" / "checkpoints"

TC_TEACHERS = [
    ("CLIPSeg",      _SEG / "clipseg_snow_output"),
    ("Florence-2",   _SEG / "florence2_snow_output"),
    ("GroundedSAM2", _SEG / "groundedsam2_snow_output"),
    ("SAM3",         _SEG / "sam3_snow_output"),
    ("YOLOE-26",     _SEG / "yoloe_snow_output"),
    ("SegGPT",       _SEG / "seggpt_snow_output"),
]

TC_STUDENT_ARCHS = ["mobilenet", "convnext", "vit_lora"]
TC_STUDENT_LABELS = ["MobileNetV3", "ConvNeXt", "ViT+LoRA"]

TC_SNOW_COLOR = np.array([100, 160, 255], dtype=np.uint8)
TC_ALPHA = 0.45
TC_SUFFIX_OVERRIDE = {}


def tc_load_teacher_mask(folder: Path, cam: str, stem: str):
    suffix = TC_SUFFIX_OVERRIDE.get(str(folder), "_snow.png")
    path = folder / cam / f"{stem}{suffix}"
    if not path.exists():
        print(f"  WARNING: missing {path}")
        return None
    m = np.array(Image.open(path).convert("L"))
    return m > 127


def tc_find_latest_maj3_ckpt(arch: str) -> Path:
    dirs = sorted([
        d for d in TC_CKPT.iterdir()
        if d.is_dir() and d.name.startswith(f"snow_{arch}_maj3_")
    ])
    if not dirs:
        raise FileNotFoundError(f"No maj3 checkpoint for {arch}")
    latest = dirs[-1]
    ckpts = sorted(latest.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt in {latest}")
    return ckpts[-1]


@torch.no_grad()
def tc_infer_student(ckpt_path: Path, arch: str, img_path: Path) -> np.ndarray:
    from student_ground.configs import get_config, IMAGENET_MEAN, IMAGENET_STD
    from student_ground.models import build_model
    from student_ground.train import StudentSegmentationModule

    cfg = get_config(arch)
    input_size = cfg["input_size"]

    print(f"  Loading {arch} from {ckpt_path.parent.name}/{ckpt_path.name}")
    module = StudentSegmentationModule.load_from_checkpoint(
        str(ckpt_path), model_variant=arch, map_location="cpu"
    )
    model = module.model.eval()

    img = np.array(Image.open(img_path).convert("RGB"))
    pil = Image.fromarray(img).resize((input_size, input_size), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    logits = model(tensor)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    orig_w, orig_h = Image.open(img_path).size
    binary = (prob >= 0.5).astype(np.uint8) * 255
    binary_full = np.array(
        Image.fromarray(binary).resize((orig_w, orig_h), Image.NEAREST)
    )
    return binary_full > 127


def tc_pca_blue_band(img_rgb: np.ndarray, dn_min=63, dn_max=200) -> np.ndarray:
    from sklearn.decomposition import PCA as _PCA
    rgb = img_rgb.astype(np.float32)
    h, w = rgb.shape[:2]
    blue_band = rgb[:, :, 2]
    blue_mask = (blue_band >= dn_min).astype(np.uint8)
    rgb_flat = rgb.reshape(-1, 3)
    rgb_std = (rgb_flat - rgb_flat.mean(axis=0)) / (rgb_flat.std(axis=0) + 1e-8)
    pca_obj = _PCA(n_components=3)
    pc_scores = pca_obj.fit_transform(rgb_std)
    pc2 = pc_scores[:, 1].reshape(h, w)
    pc3 = pc_scores[:, 2].reshape(h, w)
    pca_mask = (pc3 < pc2).astype(np.uint8)
    blue_range = ((blue_band >= dn_min) & (blue_band <= dn_max)).astype(np.uint8)
    return (blue_mask | (pca_mask & blue_range)).astype(bool)


def tc_overlay(img_rgb: np.ndarray, mask) -> np.ndarray:
    out = img_rgb.copy().astype(np.float32)
    if mask is not None and mask.any():
        out[mask] = out[mask] * (1 - TC_ALPHA) + TC_SNOW_COLOR * TC_ALPHA
    return out.clip(0, 255).astype(np.uint8)


def add_teacher_comparison_subparser(subparsers):
    p = subparsers.add_parser("teacher-comparison",
                               help="Snow segmentation comparison: teacher + student masks on one image")
    p.add_argument("--stem", default="CF00001_202302011000_00")
    p.add_argument("--out", default="thesis_teacher_comparison.pdf")
    p.add_argument("--no-students", action="store_true")
    return p


def run_teacher_comparison(args):
    stem = args.stem
    cam  = stem[:7]

    img_path = _DATA / cam / f"{stem}.jpg"
    if not img_path.exists():
        img_path = _DATA / "balanced_snow" / cam / f"{stem}.jpg"
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    print(f"Image: {img_path}  ({img_rgb.shape[1]}x{img_rgb.shape[0]})")

    panels = [("Original", img_rgb)]

    for label, folder in TC_TEACHERS:
        mask = tc_load_teacher_mask(folder, cam, stem)
        panels.append((label, tc_overlay(img_rgb, mask)))

    print("  Running PCA blue-band baseline...")
    pca_mask = tc_pca_blue_band(img_rgb)
    panels.append(("PCA Blue-Band", tc_overlay(img_rgb, pca_mask)))

    teacher_masks = [tc_load_teacher_mask(folder, cam, stem) for _, folder in TC_TEACHERS]
    valid = [m for m in teacher_masks if m is not None]
    if valid:
        stack = np.stack([m.astype(np.uint8) for m in valid], axis=0)
        for k in range(1, len(valid) + 1):
            panels.append((f"Ensemble (maj≥{k})", tc_overlay(img_rgb, stack.sum(axis=0) >= k)))

    if not args.no_students:
        print("\nRunning live student inference...")
        for arch, label in zip(TC_STUDENT_ARCHS, TC_STUDENT_LABELS):
            try:
                ckpt = tc_find_latest_maj3_ckpt(arch)
                mask = tc_infer_student(ckpt, arch, img_path)
                panels.append((label, tc_overlay(img_rgb, mask)))
            except Exception as e:
                print(f"  SKIP {label}: {e}")
                panels.append((label + " [N/A]", img_rgb))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for title, panel_img in panels:
        if panel_img is None or title == "":
            continue
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        ax.imshow(panel_img)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fname = title.replace("+", "plus").replace(" ", "_").replace("/", "_") + ".png"
        fpath = out_dir / fname
        plt.tight_layout()
        plt.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fpath}")


# =============================================================================
# BEST-RESULTS — cherry-picked best/worst results grids (paper appendix figs)
# =============================================================================

BR_CKPT_GROUND = _ROOT / "models" / "student_ground" / "checkpoints"
BR_CKPT_SNOW   = _ROOT / "models" / "student_snow"   / "checkpoints"

BR_GT_COLOR   = np.array([60,  200,  60],  dtype=np.uint8)   # green  — GT
BR_PRED_COLOR = np.array([255, 100,  50],  dtype=np.uint8)   # orange — prediction
BR_ALPHA = 0.55

BR_IN_DIST_CAMERAS = ["CF00001","CM00009","CM00014","CM00070","CM00075","CM00080","CM00102","CM00108","CM00112"]

BR_SNOW_ANN_FILES = {
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
BR_GROUND_ANN_FILES = [_ANN / "ground_annotations.json", _ANN / "san.json", _ANN / "heroku.json"]

BR_GROUND_CKPTS = {
    "mobilenet":  sorted((BR_CKPT_GROUND).glob("mobilenet_majority_3_*/best-*.ckpt")),
    "dinov2_base": sorted((BR_CKPT_GROUND).glob("dinov2_base_majority_3_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
}
BR_SNOW_CKPTS = {
    "mobilenet": sorted((BR_CKPT_SNOW / "snow_gt_mobilenet_20260522_052052").glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime),
    "convnext":  sorted((BR_CKPT_SNOW / "snow_gt_convnext_unfrozen").glob("best-*.ckpt"),        key=lambda p: p.stat().st_mtime),
    "vit_lora":  sorted((BR_CKPT_SNOW / "snow_gt_vit_lora_unfrozen").glob("best-*.ckpt"),        key=lambda p: p.stat().st_mtime),
}

BR_WORST_EXCLUDE = {
    "ground": {"CF00001", "CM00014"},
    "snow":   {"CM00014", "CM00080", "CM00112"},
}


def br_decode_rle(rle_list, w, h):
    from student_ground.dataset import decode_ls_rle
    return decode_ls_rle(rle_list, w, h).astype(bool)


def br_load_snow_gt(cameras):
    """Returns list of (camera, stem, mask_bool)."""
    results = []
    for cam in cameras:
        ann_file = BR_SNOW_ANN_FILES.get(cam)
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
                    results.append((cam, stem, br_decode_rle(r["value"]["rle"], w, h)))
    return results


def br_load_ground_gt(cameras):
    results = []
    for ann_file in BR_GROUND_ANN_FILES:
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
                    results.append((cam, stem, br_decode_rle(r["value"]["rle"], w, h)))
    return results


def br_find_img(camera, stem):
    for d in [_DATA/camera, _DATA/f"_EXCLUDED_{camera}", _DATA/"balanced_snow"/camera]:
        p = d / f"{stem}.jpg"
        if p.exists():
            return p
    return None


@torch.no_grad()
def br_infer(model, img_path, arch):
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


def br_iou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def br_overlay(img_rgb, mask, color):
    out = img_rgb.copy().astype(np.float32)
    if mask is not None and mask.any():
        m = mask
        if m.shape != img_rgb.shape[:2]:
            m = np.array(Image.fromarray(m.astype(np.uint8)*255)
                         .resize((img_rgb.shape[1],img_rgb.shape[0]),Image.NEAREST))>127
        out[m] = out[m]*(1-BR_ALPHA) + color*BR_ALPHA
    return out.clip(0,255).astype(np.uint8)


def br_save_grid(rows, task, out_path, title):
    """rows: list of (camera, stem, iou, img_rgb, gt_mask, pred_mask)"""
    color = BR_GT_COLOR if task == "ground" else np.array([100,160,255],dtype=np.uint8)
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.5 * n), squeeze=False)
    axes[0, 0].set_title("Original",     fontsize=11, fontweight="bold", pad=6)
    axes[0, 1].set_title("Ground Truth", fontsize=11, fontweight="bold", pad=6)
    axes[0, 2].set_title("Prediction",   fontsize=11, fontweight="bold", pad=6)
    for ri, (cam, stem, iou_val, img_rgb, gt_mask, pred_mask) in enumerate(rows):
        axes[ri, 0].imshow(img_rgb)
        axes[ri, 1].imshow(br_overlay(img_rgb, gt_mask, color))
        axes[ri, 2].imshow(br_overlay(img_rgb, pred_mask, BR_PRED_COLOR))
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


def br_load_model(arch, ckpt_path):
    from student_ground.train import StudentSegmentationModule
    m = StudentSegmentationModule.load_from_checkpoint(str(ckpt_path), model_variant=arch, map_location="cpu")
    return m.model.eval()


def br_run(task, cameras, top_n, out_dir, arch=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    if task == "ground":
        ckpt_dict = BR_GROUND_CKPTS
        gt_list = br_load_ground_gt(cameras)
        print(f"Ground GT: {len(gt_list)} annotated images")
    else:
        ckpt_dict = BR_SNOW_CKPTS
        gt_list = br_load_snow_gt(cameras)
        print(f"Snow GT: {len(gt_list)} annotated images")

    use_arch = arch or next((a for a, ckpts in ckpt_dict.items() if ckpts), None)
    ckpt_list = ckpt_dict.get(use_arch, [])
    if not ckpt_list:
        print(f"ERROR: no checkpoint for {use_arch}"); return
    ckpt = ckpt_list[-1]

    print(f"Model: {use_arch}  ckpt: {ckpt.parent.name}/{ckpt.name}")
    model = br_load_model(use_arch, ckpt)
    arch = use_arch

    scored = []
    for cam, stem, gt_mask in gt_list:
        img_path = br_find_img(cam, stem)
        if img_path is None:
            continue
        pred = br_infer(model, img_path, arch)
        if gt_mask.shape != pred.shape:
            gt_mask = np.array(Image.fromarray(gt_mask.astype(np.uint8)*255)
                               .resize((pred.shape[1],pred.shape[0]),Image.NEAREST))>127
        score = br_iou(pred, gt_mask)
        scored.append((score, cam, stem, gt_mask, pred, img_path))
        print(f"  {cam} {stem}: IoU={score:.3f}")

    scored.sort(key=lambda x: x[0], reverse=True)

    by_cam = defaultdict(list)
    for item in scored:
        by_cam[item[1]].append(item)

    worst_exclude = BR_WORST_EXCLUDE.get(task, set())

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
    br_save_grid(best_rows,  task, out_dir / f"grid_best_{task}.png",
              f"Best {task_label} Segmentation per Camera")
    br_save_grid(worst_rows, task, out_dir / f"grid_worst_{task}.png",
              f"Worst {task_label} Segmentation per Camera")


def add_best_results_subparser(subparsers):
    p = subparsers.add_parser("best-results",
                               help="Cherry-picked best/worst results grids (paper appendix figures)")
    p.add_argument("--task", choices=["snow","ground","both"], default="both")
    p.add_argument("--top", type=int, default=10, help="Top N globally; use -1 for per-camera best+worst")
    p.add_argument("--out", default="thesis_best_results/")
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--arch", default=None, help="Model arch override (e.g. dinov2_base, convnext)")
    return p


def run_best_results(args):
    cameras = args.cameras or BR_IN_DIST_CAMERAS
    tasks = ["snow","ground"] if args.task == "both" else [args.task]

    for task in tasks:
        out = Path(args.out) / task
        print(f"\n{'='*60}\n{task.upper()}\n{'='*60}")
        br_run(task, cameras, args.top, out, arch=args.arch)


# =============================================================================
# SMOOTHER-WEEK — raw vs smoothed snow_pct for a representative snowy week
# =============================================================================

def sw_predict_camera(module, cam_df, scaler=None):
    import math
    from temporal_snow.dataset import build_feature_columns, SNOW_PCT_COLS

    snow_cols = [c for c in SNOW_PCT_COLS if c in cam_df.columns]
    tmp = cam_df.copy()
    tmp["hour_sin"] = np.sin(2*math.pi*tmp["datetime"].dt.hour/24)
    tmp["hour_cos"] = np.cos(2*math.pi*tmp["datetime"].dt.hour/24)
    doy = tmp["datetime"].dt.dayofyear
    tmp["doy_sin"] = np.sin(2*math.pi*doy/365)
    tmp["doy_cos"] = np.cos(2*math.pi*doy/365)
    FILL = {"t2m":0,"visibility":1000,"weather_code":0,"brightness":128,"contrast":50,
            "edge_density":10,"entropy":4,"white_frac":0,"low_sat_frac":0,"blue_ratio":0,
            "bottom_white_frac":0,"bottom_brightness":128,"snow_n_components":0,
            "snow_largest_ratio":0,"snow_centroid_y":0.5,"ground_coverage":0}
    for col in snow_cols: tmp[col] = tmp[col].fillna(0)
    for col, val in FILL.items():
        if col in tmp.columns: tmp[col] = tmp[col].fillna(val)
    if "visibility" in tmp.columns: tmp["visibility"] = np.log1p(tmp["visibility"])
    if "snow_n_components" in tmp.columns: tmp["snow_n_components"] = np.log1p(tmp["snow_n_components"])

    input_size = module.hparams.get("input_size", None)
    feat_cols_full = build_feature_columns(snow_cols, use_image_features=True)
    feat_cols_noimg = build_feature_columns(snow_cols, use_image_features=False)
    if input_size:
        if len(feat_cols_noimg) == input_size:
            feat_cols = feat_cols_noimg
        elif len(feat_cols_full) == input_size:
            feat_cols = feat_cols_full
        else:
            feat_cols = feat_cols_full[:input_size]
    else:
        feat_cols = feat_cols_full
    for col in feat_cols:
        if col not in tmp.columns: tmp[col] = 0.0

    if scaler is not None:
        tmp = scaler.transform(tmp)
    else:
        if "t2m" in tmp.columns: tmp["t2m"] = tmp["t2m"] / 30.0
        if "visibility" in tmp.columns: tmp["visibility"] = tmp["visibility"] / np.log1p(65000)
        if "weather_code" in tmp.columns: tmp["weather_code"] = tmp["weather_code"] / 100.0
        if "brightness" in tmp.columns:
            tmp["brightness"] /= 255.0; tmp["contrast"] /= 128.0
            tmp["edge_density"] /= 50.0; tmp["entropy"] /= 6.0; tmp["bottom_brightness"] /= 255.0
        if "snow_n_components" in tmp.columns: tmp["snow_n_components"] /= np.log1p(100)

    print(f"  [debug] input_size={input_size} feat_cols={len(feat_cols)} scaler={'yes' if scaler else 'approx'} model={type(module.model).__name__}")

    snow_raw = cam_df[snow_cols[0]].values.astype(np.float32) if snow_cols else np.zeros(len(cam_df))
    feats = np.nan_to_num(tmp[feat_cols].values.astype(np.float32), nan=0.0)
    x = torch.tensor(feats).unsqueeze(0)
    with torch.no_grad():
        pred = module(x).squeeze(0).cpu().numpy() * 100

    raw_out = snow_raw if snow_raw.max() > 1.01 else snow_raw * 100
    return raw_out, pred


def sw_find_best_week(df, camera=None, week_start=None):
    """Find camera + week with highest snow variance for visual interest."""
    from temporal_snow.dataset import SNOW_PCT_COLS

    if camera and week_start:
        return camera, pd.Timestamp(week_start)

    best_cam, best_week, best_var = None, None, -1
    for cam, cam_df in df.groupby("camera"):
        if camera and cam != camera:
            continue
        cam_df = cam_df.sort_values("datetime").copy()
        cam_df["week"] = cam_df["datetime"].dt.to_period("W")
        for week, week_df in cam_df.groupby("week"):
            if len(week_df) < 10:
                continue
            snow_col = next((c for c in SNOW_PCT_COLS if c in week_df.columns), None)
            if snow_col is None:
                continue
            var = week_df[snow_col].std()
            has_snow = (week_df[snow_col] > 0.05).any()
            has_clear = (week_df[snow_col] < 0.05).any()
            if has_snow and has_clear and var > best_var:
                best_var = var
                best_cam = cam
                best_week = week.start_time
    return best_cam, best_week


def add_smoother_week_subparser(subparsers):
    p = subparsers.add_parser("smoother-week",
                               help="Raw vs smoothed snow_pct for a representative snowy week")
    p.add_argument("--checkpoint", required=True, help="Path to smoother checkpoint (.ckpt)")
    p.add_argument("--extra-checkpoints", nargs="+", default=[], help="Additional checkpoints to overlay")
    p.add_argument("--labels", nargs="+", default=[], help="Labels for extra checkpoints")
    p.add_argument("--csv", default=str(_ROOT / "analysis/snow_timeseries.csv"))
    p.add_argument("--camera", default=None)
    p.add_argument("--week", default=None, help="Week start date YYYY-MM-DD")
    p.add_argument("--window-days", type=int, default=14, help="Days to show (default 14)")
    p.add_argument("--out", default="thesis_smoother_figure.pdf")
    return p


def run_smoother_week(args):
    import matplotlib.dates as mdates
    from temporal_snow.dataset import SNOW_PCT_COLS, FeatureScaler
    from temporal_snow.train_smoother import TemporalSmootherModule

    print("Loading data...")
    df = pd.read_csv(args.csv, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    # Do NOT filter by qa_usable — we want to show raw spikes that smoothing suppresses
    df = df.sort_values(["camera", "datetime"]).reset_index(drop=True)

    def _load_scaler(ckpt_path):
        scaler_path = Path(ckpt_path).parent / "scaler.json"
        if scaler_path.exists():
            return FeatureScaler.load(scaler_path)
        return None

    print(f"Loading checkpoint: {args.checkpoint}")
    module = TemporalSmootherModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    module.eval()
    primary_scaler = _load_scaler(args.checkpoint)

    cam, week_start = sw_find_best_week(df, args.camera, args.week)
    if cam is None or week_start is None:
        print(f"ERROR: No suitable window found for camera={args.camera}"); return
    week_end = week_start + pd.Timedelta(days=args.window_days)
    print(f"Camera: {cam}  |  Window: {week_start.date()} → {week_end.date()}")

    cam_df = df[df["camera"] == cam].sort_values("datetime").reset_index(drop=True)
    snow_col = next((c for c in SNOW_PCT_COLS if c in cam_df.columns), None)

    snow_raw, snow_smooth = sw_predict_camera(module, cam_df, scaler=primary_scaler)

    mask = (cam_df["datetime"] >= week_start) & (cam_df["datetime"] < week_end)
    dt_window = cam_df.loc[mask, "datetime"]
    raw_window = snow_raw[mask.values]
    smooth_window = snow_smooth[mask.values]

    extra_smooths = []
    for ec in args.extra_checkpoints:
        m = TemporalSmootherModule.load_from_checkpoint(ec, map_location="cpu")
        m.eval()
        scaler_ec = _load_scaler(ec)
        _, s = sw_predict_camera(m, cam_df, scaler=scaler_ec)
        extra_smooths.append(s[mask.values])

    def _kalman(series, Q=25, R=100):
        x, P = float(series[0]), 500.0
        out = []
        for z in series:
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x = x + K * (float(z) - x)
            P = (1 - K) * P_pred
            out.append(np.clip(x, 0, 100))
        return np.array(out)

    snow_col_raw = next((c for c in SNOW_PCT_COLS if c in cam_df.columns), None)
    raw_vals = cam_df[snow_col_raw].values if snow_col_raw else np.zeros(len(cam_df))
    if raw_vals.max() <= 1.01:
        raw_vals = raw_vals * 100
    raw_pct_full = raw_vals
    kalman_full = _kalman(raw_pct_full)
    kalman_window = kalman_full[mask.values]

    meteo_col = "snow_present" if "snow_present" in cam_df.columns else None
    meteo_window = cam_df.loc[mask, meteo_col] if meteo_col else None

    fig, ax = plt.subplots(figsize=(12, 4))

    if meteo_window is not None:
        snow_present = meteo_window.values.astype(bool)
        for i in range(len(dt_window) - 1):
            if snow_present[i]:
                ax.axvspan(dt_window.iloc[i], dt_window.iloc[i+1],
                           alpha=0.12, color="#4a90d9", linewidth=0)

    ax.plot(dt_window, raw_window, color="#aaaaaa", linewidth=0.8,
            alpha=0.7, label="Raw student output", zorder=2)

    ax.plot(dt_window, kalman_window, color="#2980b9", linewidth=1.5,
            linestyle="--", alpha=0.85, label="Kalman Filter", zorder=3)

    colors = ["#e67e22", "#27ae60", "#8e44ad"]
    labels = args.labels if args.labels else [f"Model {i+1}" for i in range(len(extra_smooths))]
    for i, (es, lbl) in enumerate(zip(extra_smooths, labels)):
        ax.plot(dt_window, es, linewidth=1.5, color=colors[i % len(colors)],
                alpha=0.85, label=lbl, zorder=3)

    ax.plot(dt_window, smooth_window, color="#c0392b", linewidth=2.0,
            label="LSTM (bi)", zorder=4)

    ax.axhline(50, color="#333333", linewidth=1.0, linestyle=":", alpha=0.7, label="Snow threshold (50%)", zorder=1)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Snow Coverage (%)", fontsize=11)
    ax.set_title(f"Temporal Smoothing — Camera {cam}", fontsize=12, fontweight="bold")
    ax.set_ylim(-2, 102)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    handles, labels_leg = ax.get_legend_handles_labels()
    if meteo_window is not None:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor="#4a90d9", alpha=0.3, label="Meteo: snow present"))
        labels_leg.append("Meteo: snow present")
    ax.legend_ = None  # remove axes legend

    fig.legend(handles, labels_leg, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, fontsize=8.5, framealpha=0.9, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = Path(args.out)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


# =============================================================================
# SHAP — SHAP analysis on bi-directional GRU temporal smoother
# =============================================================================

SHAP_DISPLAY = {
    "student_snow_pct": "Snow % (raw)",
    "snow_n_components": "Snow components",
    "snow_largest_ratio": "Largest component ratio",
    "snow_centroid_y": "Snow centroid Y",
    "ground_coverage": "Ground coverage",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "edge_density": "Edge density",
    "entropy": "Entropy",
    "white_frac": "White fraction",
    "low_sat_frac": "Low saturation frac",
    "blue_ratio": "Blue ratio",
    "bottom_brightness": "Bottom brightness",
    "bottom_white_frac": "Bottom white frac",
    "t2m": "Temperature (t2m)",
    "visibility": "Visibility",
    "weather_code": "Weather code",
    "hour_sin": "Hour (sin)",
    "hour_cos": "Hour (cos)",
    "doy_sin": "Day of year (sin)",
    "doy_cos": "Day of year (cos)",
}


class ShapGRUWrapper(torch.nn.Module):
    """Wrap GRU to accept flat (N, F) and return scalar per sample."""
    def __init__(self, module, seq_len):
        super().__init__()
        self.module = module
        self.seq_len = seq_len

    def forward(self, x):
        # x: (N, F) — treat each row as last step of a window
        B = x.shape[0]
        pad = torch.zeros(B, self.seq_len - 1, x.shape[1])
        seq = torch.cat([pad, x.unsqueeze(1)], dim=1)  # (B, seq_len, F)
        out = self.module(seq)  # (B, seq_len, 1)
        out = out.squeeze(-1) if out.dim() == 3 else out
        return out[:, -1].unsqueeze(1)   # (B, 1) for GradientExplainer


def add_shap_subparser(subparsers):
    p = subparsers.add_parser("shap", help="SHAP feature-importance analysis on the bi-GRU smoother")
    p.add_argument("--checkpoint", default=str(
        _HERE / "checkpoints/smoother_gru_h128_l2_bi_k7_seq96_20260623_175446/best-epoch=114-val_mae=0.0037.ckpt"))
    p.add_argument("--csv", default=str(_ROOT / "analysis/snow_timeseries.csv"))
    p.add_argument("--out", default=str(_ROOT / "analysis/shap_gru"))
    p.add_argument("--seq-len", type=int, default=96)
    return p


def run_shap(args):
    import math
    import shap
    from temporal_snow.train_smoother import TemporalSmootherModule
    from temporal_snow.dataset import build_feature_columns, SNOW_PCT_COLS, FeatureScaler

    out = args.out
    Path(out).mkdir(exist_ok=True, parents=True)

    module = TemporalSmootherModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    module.eval()

    scaler_path = Path(args.checkpoint).parent / "scaler.json"
    scaler = FeatureScaler.load(scaler_path) if scaler_path.exists() else None
    print(f"Scaler: {'loaded' if scaler else 'NOT FOUND — will fail'}")

    # ── Preprocess — no QA filter, same as training ─────────────────────────
    df = pd.read_csv(args.csv, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    snow_pct_cols = [c for c in SNOW_PCT_COLS if c in df.columns]
    feature_cols = build_feature_columns(snow_pct_cols, use_image_features=True)

    df["hour_sin"] = np.sin(2 * math.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * math.pi * df["datetime"].dt.hour / 24)
    doy = df["datetime"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * math.pi * doy / 365)

    fill = {"t2m":0,"visibility":1000,"weather_code":0,"brightness":128,
            "contrast":50,"edge_density":10,"entropy":4,"white_frac":0,
            "low_sat_frac":0,"blue_ratio":0,"bottom_white_frac":0,
            "bottom_brightness":128,"snow_n_components":0,"snow_largest_ratio":0,
            "snow_centroid_y":0.5,"ground_coverage":0}
    for col, val in fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col in snow_pct_cols:
        df[col] = df[col].fillna(0)

    if "visibility" in df.columns: df["visibility"] = np.log1p(df["visibility"])
    if "snow_n_components" in df.columns: df["snow_n_components"] = np.log1p(df["snow_n_components"])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = scaler.transform(df)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    print(f"Data: {X.shape}  Features: {len(feature_cols)}")

    # ── Wrap model for SHAP: input (N, F) → output (N,) ─────────────────────
    wrapper = ShapGRUWrapper(module, args.seq_len)
    wrapper.eval()

    # ── SHAP GradientExplainer ───────────────────────────────────────────────
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X), size=200, replace=False)
    exp_idx = rng.choice(len(X), size=500, replace=False)

    print("Computing SHAP values (GradientExplainer)...")
    background = torch.from_numpy(X[bg_idx])
    X_exp_t = torch.from_numpy(X[exp_idx])
    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(X_exp_t)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # take first (only) output
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]  # (N, F, 1) → (N, F)
    elif shap_values.ndim == 2 and shap_values.shape[0] == len(feature_cols):
        shap_values = shap_values.T  # (F, N) → (N, F)
    print(f"SHAP values shape: {shap_values.shape}  (expected ({len(exp_idx)}, {len(feature_cols)}))")

    # ── Plot 1: Bar chart — mean |SHAP| ──────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    names = [SHAP_DISPLAY.get(f, f) for f in feature_cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([names[i] for i in order], mean_abs[order], color="#2196F3")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Bi-GRU Temporal Smoother — Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{out}/shap_bar.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out}/shap_bar.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out}/shap_bar.pdf")

    # ── Plot 2: Dependence plots for top 4 features ──────────────────────────
    top4 = order[:4]
    X_exp_np = X[exp_idx]

    raw_vals = {
        "t2m": X_exp_np[:, feature_cols.index("t2m")] * 30,
        "student_snow_pct": X_exp_np[:, feature_cols.index("student_snow_pct")] * 100,
        "visibility": np.expm1(X_exp_np[:, feature_cols.index("visibility")] * np.log1p(65000)),
        "weather_code": X_exp_np[:, feature_cols.index("weather_code")] * 100,
    }
    xlabels = {
        "t2m": "Temperature (°C)",
        "student_snow_pct": "Raw snow % (model input)",
        "visibility": "Visibility (m)",
        "weather_code": "WMO weather code",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, fi in zip(axes, top4):
        fname = feature_cols[fi]
        x_vals = raw_vals.get(fname, X_exp_np[:, fi])
        xlabel = xlabels.get(fname, SHAP_DISPLAY.get(fname, fname))
        ax.scatter(x_vals, shap_values[:, fi], alpha=0.3, s=8, c="#E53935")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SHAP value")
        ax.set_title(SHAP_DISPLAY.get(fname, fname))

    plt.suptitle("GRU Smoother — SHAP Dependence Plots (Top 4 Features)", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{out}/shap_dependence.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out}/shap_dependence.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out}/shap_dependence.pdf")

    print("\nTop 10 features by importance:")
    for i in order[:10]:
        print(f"  {names[i]:<30} {mean_abs[i]:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Manuscript figure generators")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_ground_comparison_subparser(subparsers)
    add_teacher_comparison_subparser(subparsers)
    add_best_results_subparser(subparsers)
    add_smoother_week_subparser(subparsers)
    add_shap_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "ground-comparison":
        run_ground_comparison(args)
    elif args.command == "teacher-comparison":
        run_teacher_comparison(args)
    elif args.command == "best-results":
        run_best_results(args)
    elif args.command == "smoother-week":
        run_smoother_week(args)
    elif args.command == "shap":
        run_shap(args)


if __name__ == "__main__":
    main()
