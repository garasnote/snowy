#!/usr/bin/env python3
"""
Thesis figure: ground segmentation comparison — teachers, ensembles, baselines, students.
All masks loaded from pre-saved outputs (suffix _ground.png).
Students: live inference from latest majority-3 checkpoints.

Default image: CF00001_202302011000_00 (max snow_n_components=107).

Usage:
    python plot_ground_comparison.py
    python plot_ground_comparison.py --stem CM00009_202212150900_00 --out ground_panels/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams["font.family"] = "DejaVu Sans"

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SEG  = _ROOT / "preprocessing" / "segmentation"
_DATA = _ROOT / "data"
_CKPT = _ROOT / "models" / "student_ground" / "checkpoints"

sys.path.insert(0, str(_ROOT / "models"))

GROUND_COLOR = np.array([60, 180, 60], dtype=np.uint8)   # green overlay
ALPHA = 0.45

# ── Saved mask sources (all use _ground.png) ──────────────────────────────────

TEACHERS = [
    ("DINOv2-PCA",     _SEG / "dinov2_pca_output"),
    ("DINOv3-PCA",     _SEG / "dinov3_pca_output"),
    ("SegFormer",      _SEG / "segformer_output"),
    ("DeepLabV3+",     _SEG / "deeplabv3_output"),
    ("SAM3 (text)",    _SEG / "sam3_dino_output" / "text"),
    ("SAM3+DINOv2",    _SEG / "sam3_dino_output" / "dino+sam"),
    ("Depth Anything", _SEG / "depth_output" / "N0.05_F0.95"),
]

ENSEMBLES = [
    ("Majority-2",   _SEG / "ensemble_output" / "majority_2"),
    ("Majority-3",   _SEG / "ensemble_output" / "majority_3"),
    ("Majority-4",   _SEG / "ensemble_output" / "majority_4"),
    ("Majority-5",   _SEG / "ensemble_output" / "majority_5"),
    ("Weighted-3",   _SEG / "ensemble_output" / "weighted_3"),
]

BASELINES = [
    ("Fixed-32%",    _SEG / "baseline_fixed_output"),
    ("Sobel",        _SEG / "baseline_sobel_output"),
]

STUDENT_ARCHS   = ["mobilenet", "mobilenet_rgbd", "efficientnet", "dinov2_base", "dinov2_large", "dinov3_base", "dinov3_large"]
STUDENT_LABELS  = ["MobileNet", "MobileNet+D", "EfficientNet-B0", "DINOv2-Base", "DINOv2-Large", "DINOv3-Base", "DINOv3-Large"]


def load_mask(folder: Path, cam: str, stem: str) -> np.ndarray | None:
    path = folder / cam / f"{stem}_ground.png"
    if not path.exists():
        print(f"  WARNING: missing {path}")
        return None
    return np.array(Image.open(path).convert("L")) > 127


def find_latest_ckpt(arch: str) -> Path:
    # Prefer majority_3, fall back to any arch dir
    for prefix in [f"{arch}_majority_3_", f"{arch}_majority_", f"{arch}_weighted_", f"{arch}_"]:
        dirs = sorted([d for d in _CKPT.iterdir() if d.is_dir() and d.name.startswith(prefix)])
        if dirs:
            ckpts = sorted(dirs[-1].glob("*.ckpt"))
            if ckpts:
                return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found for {arch}")


@torch.no_grad()
def infer_student(ckpt_path: Path, arch: str, img_path: Path) -> np.ndarray:
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


def overlay(img_rgb: np.ndarray, mask) -> np.ndarray:
    out = img_rgb.copy().astype(np.float32)
    if mask is not None and mask.any():
        out[mask] = out[mask] * (1 - ALPHA) + GROUND_COLOR * ALPHA
    return out.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default="CF00001_202302011000_00")
    parser.add_argument("--out", default="thesis_ground_panels/")
    parser.add_argument("--no-students", action="store_true")
    args = parser.parse_args()

    stem = args.stem
    cam  = stem[:7]

    img_path = _DATA / cam / f"{stem}.jpg"
    if not img_path.exists():
        img_path = _DATA / "balanced_snow" / cam / f"{stem}.jpg"
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    print(f"Image: {img_path}  ({img_rgb.shape[1]}x{img_rgb.shape[0]})")

    panels = [("Original", img_rgb)]

    print("\n--- Teachers ---")
    for label, folder in TEACHERS:
        panels.append((label, overlay(img_rgb, load_mask(folder, cam, stem))))

    print("\n--- Ensembles ---")
    for label, folder in ENSEMBLES:
        panels.append((label, overlay(img_rgb, load_mask(folder, cam, stem))))

    print("\n--- Baselines ---")
    for label, folder in BASELINES:
        panels.append((label, overlay(img_rgb, load_mask(folder, cam, stem))))

    if not args.no_students:
        print("\n--- Students (live inference) ---")
        for arch, label in zip(STUDENT_ARCHS, STUDENT_LABELS):
            try:
                ckpt = find_latest_ckpt(arch)
                panels.append((label, overlay(img_rgb, infer_student(ckpt, arch, img_path))))
            except Exception as e:
                print(f"  SKIP {label}: {e}")
                panels.append((label + " [N/A]", img_rgb))

    # Save each panel individually
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


if __name__ == "__main__":
    main()
