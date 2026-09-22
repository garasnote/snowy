#!/usr/bin/env python3
"""
Thesis figure: teacher + student mask comparison on a single image.
Teachers: loaded from pre-saved masks.
Students (maj3): live inference from checkpoint.

Default image: CF00001_202302011000_00 (max snow_n_components=107).

Usage:
    python plot_teacher_comparison.py
    python plot_teacher_comparison.py --stem CM00014_202209160900_27 --out fig_teachers_white.pdf
    python plot_teacher_comparison.py --no-students
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
_CKPT = _ROOT / "models" / "student_snow" / "checkpoints"

sys.path.insert(0, str(_ROOT / "models"))

TEACHERS = [
    ("CLIPSeg",      _SEG / "clipseg_snow_output"),
    ("Florence-2",   _SEG / "florence2_snow_output"),
    ("GroundedSAM2", _SEG / "groundedsam2_snow_output"),
    ("SAM3",         _SEG / "sam3_snow_output"),
    ("YOLOE-26",     _SEG / "yoloe_snow_output"),
    ("SegGPT",       _SEG / "seggpt_snow_output"),
]

SUFFIX_OVERRIDE = {}

STUDENT_ARCHS = ["mobilenet", "convnext", "vit_lora"]
STUDENT_LABELS = ["MobileNetV3", "ConvNeXt", "ViT+LoRA"]

SNOW_COLOR = np.array([100, 160, 255], dtype=np.uint8)
ALPHA = 0.45


def load_teacher_mask(folder: Path, cam: str, stem: str):
    suffix = SUFFIX_OVERRIDE.get(str(folder), "_snow.png")
    path = folder / cam / f"{stem}{suffix}"
    if not path.exists():
        print(f"  WARNING: missing {path}")
        return None
    m = np.array(Image.open(path).convert("L"))
    return m > 127


def find_latest_maj3_ckpt(arch: str) -> Path:
    dirs = sorted([
        d for d in _CKPT.iterdir()
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
def infer_student(ckpt_path: Path, arch: str, img_path: Path) -> np.ndarray:
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
    # resize back to original
    orig_w, orig_h = Image.open(img_path).size
    binary = (prob >= 0.5).astype(np.uint8) * 255
    binary_full = np.array(
        Image.fromarray(binary).resize((orig_w, orig_h), Image.NEAREST)
    )
    return binary_full > 127


def pca_blue_band(img_rgb: np.ndarray, dn_min=63, dn_max=200) -> np.ndarray:
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


def overlay(img_rgb: np.ndarray, mask) -> np.ndarray:
    out = img_rgb.copy().astype(np.float32)
    if mask is not None and mask.any():
        out[mask] = out[mask] * (1 - ALPHA) + SNOW_COLOR * ALPHA
    return out.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default="CF00001_202302011000_00")
    parser.add_argument("--out", default="thesis_teacher_comparison.pdf")
    parser.add_argument("--no-students", action="store_true")
    args = parser.parse_args()

    stem = args.stem
    cam  = stem[:7]

    img_path = _DATA / cam / f"{stem}.jpg"
    if not img_path.exists():
        img_path = _DATA / "balanced_snow" / cam / f"{stem}.jpg"
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    print(f"Image: {img_path}  ({img_rgb.shape[1]}x{img_rgb.shape[0]})")

    # --- Build panels ---
    panels = [("Original", img_rgb)]

    for label, folder in TEACHERS:
        mask = load_teacher_mask(folder, cam, stem)
        panels.append((label, overlay(img_rgb, mask)))

    # PCA blue-band baseline (Härer et al. 2016) — live inference
    print("  Running PCA blue-band baseline...")
    pca_mask = pca_blue_band(img_rgb)
    panels.append(("PCA Blue-Band", overlay(img_rgb, pca_mask)))

    # Majority vote ensemble k=1..6
    teacher_masks = [load_teacher_mask(folder, cam, stem) for _, folder in TEACHERS]
    valid = [m for m in teacher_masks if m is not None]
    if valid:
        stack = np.stack([m.astype(np.uint8) for m in valid], axis=0)
        for k in range(1, len(valid) + 1):
            panels.append((f"Ensemble (maj≥{k})", overlay(img_rgb, stack.sum(axis=0) >= k)))

    if not args.no_students:
        print("\nRunning live student inference...")
        for arch, label in zip(STUDENT_ARCHS, STUDENT_LABELS):
            try:
                ckpt = find_latest_maj3_ckpt(arch)
                mask = infer_student(ckpt, arch, img_path)
                panels.append((label, overlay(img_rgb, mask)))
            except Exception as e:
                print(f"  SKIP {label}: {e}")
                panels.append((label + " [N/A]", img_rgb))

    # --- Save each panel individually ---
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


if __name__ == "__main__":
    main()
