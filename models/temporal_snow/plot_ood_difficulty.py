#!/usr/bin/env python3
"""
Appendix figure: OOD camera difficulty visualization.
For each OOD camera (CM00006, CM00024, CM00068):
  - Shows original image
  - GT ground mask overlay (green)
  - GT snow mask overlay (blue) — where available
  - Student ground prediction overlay
  - Student snow prediction overlay (maj3)

Saves one PNG per image per camera into --out dir.

Usage:
    python plot_ood_difficulty.py --out thesis_ood_panels/
    python plot_ood_difficulty.py --camera CM00068 --out thesis_ood_panels/
"""

import argparse
import json
import math
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
_DATA = _ROOT / "data"
_ANN  = _ROOT / "data" / "annotations"
_CKPT_GROUND = _ROOT / "models" / "student_ground" / "checkpoints"
_CKPT_SNOW   = _ROOT / "models" / "student_snow"   / "checkpoints"

sys.path.insert(0, str(_ROOT / "models"))

GROUND_COLOR = np.array([60,  180,  60], dtype=np.uint8)
SNOW_COLOR   = np.array([100, 160, 255], dtype=np.uint8)
PRED_GROUND  = np.array([255, 200,   0], dtype=np.uint8)
PRED_SNOW    = np.array([255,  80,  80], dtype=np.uint8)
ALPHA = 0.50

OOD_CAMERAS = ["CM00006", "CM00024", "CM00068"]

SNOW_ANN_FILES  = {"CM00068": _ANN / "68.json"}
GROUND_ANN_FILE = _ANN / "ground_annotations.json"


# ── RLE decode ────────────────────────────────────────────────────────────────

def _decode_rle(rle_list, w, h):
    from student_ground.dataset import decode_ls_rle
    return decode_ls_rle(rle_list, w, h).astype(bool)


# ── Load all GT annotations for a camera ──────────────────────────────────────

def load_gt_snow(camera):
    """Returns dict stem → bool mask (decoded from RLE)."""
    ann_file = SNOW_ANN_FILES.get(camera)
    if ann_file is None or not ann_file.exists():
        return {}
    with open(ann_file) as f:
        tasks = json.load(f)
    masks = {}
    for task in tasks:
        url = task["data"]["image"]
        if camera not in url:
            continue
        stem = url.split("/")[-1].replace(".jpg", "")
        for ann in task.get("annotations", []):
            for r in ann.get("result", []):
                if r.get("type") != "brushlabels" or "rle" not in r.get("value", {}):
                    continue
                w = r.get("original_width", 1296)
                h = r.get("original_height", 960)
                masks[stem] = _decode_rle(r["value"]["rle"], w, h)
    return masks


def load_gt_ground(camera):
    """Returns dict stem → bool mask."""
    if not GROUND_ANN_FILE.exists():
        return {}
    with open(GROUND_ANN_FILE) as f:
        tasks = json.load(f)
    masks = {}
    for task in tasks:
        url = task["data"]["image"]
        if camera not in url:
            continue
        raw_stem = url.split("/")[-1].replace(".jpg", "")
        # Handle prefixed stems like "60092655-CM00024_..."
        if "-" in raw_stem and raw_stem.split("-")[1].startswith("CM"):
            stem = raw_stem.split("-", 1)[1]
        else:
            stem = raw_stem
        for ann in task.get("annotations", []):
            for r in ann.get("result", []):
                if r.get("type") != "brushlabels" or "rle" not in r.get("value", {}):
                    continue
                w = r.get("original_width", 1296)
                h = r.get("original_height", 960)
                masks[stem] = _decode_rle(r["value"]["rle"], w, h)
    return masks


# ── Find image path ────────────────────────────────────────────────────────────

def find_img(camera, stem):
    for d in [_DATA / camera,
              _DATA / f"_EXCLUDED_{camera}",
              _DATA / "balanced_snow" / camera]:
        p = d / f"{stem}.jpg"
        if p.exists():
            return p
    return None


# ── Live inference ─────────────────────────────────────────────────────────────

def _load_student(ckpt_path, arch):
    from student_ground.train import StudentSegmentationModule
    m = StudentSegmentationModule.load_from_checkpoint(
        str(ckpt_path), model_variant=arch, map_location="cpu"
    )
    return m.model.eval()


@torch.no_grad()
def infer_mask(model, img_path, arch, task="ground"):
    from student_ground.configs import get_config, IMAGENET_MEAN, IMAGENET_STD
    cfg = get_config(arch)
    sz = cfg["input_size"]
    img = np.array(Image.open(img_path).convert("RGB"))
    pil = Image.fromarray(img).resize((sz, sz), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, np.float32).reshape(1,1,3)) / \
               np.array(IMAGENET_STD,  np.float32).reshape(1,1,3)
    t = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
    logits = model(t)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    ow, oh = Image.open(img_path).size
    binary = (prob >= 0.5).astype(np.uint8) * 255
    return np.array(Image.fromarray(binary).resize((ow, oh), Image.NEAREST)) > 127


def find_latest_ckpt(base, prefix):
    dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)])
    if not dirs: return None
    d = dirs[-1]
    final = d / "final.ckpt"
    if final.exists(): return final
    ckpts = sorted(d.glob("best-*.ckpt"))
    return ckpts[-1] if ckpts else None


# ── Overlay ────────────────────────────────────────────────────────────────────

def overlay(img_rgb, mask, color):
    out = img_rgb.copy().astype(np.float32)
    if mask is not None and mask.any():
        m = mask
        if m.shape != img_rgb.shape[:2]:
            m = np.array(Image.fromarray(m.astype(np.uint8)*255)
                         .resize((img_rgb.shape[1], img_rgb.shape[0]), Image.NEAREST)) > 127
        out[m] = out[m] * (1 - ALPHA) + color * ALPHA
    return out.clip(0, 255).astype(np.uint8)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=None, help="Single camera, or all OOD if omitted")
    parser.add_argument("--out", default="thesis_ood_panels/")
    parser.add_argument("--max-per-camera", type=int, default=6)
    args = parser.parse_args()

    cameras = [args.camera] if args.camera else OOD_CAMERAS
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    N_COLS, N_ROWS = 5, 5
    N = N_COLS * N_ROWS

    for camera in cameras:
        print(f"\n=== {camera} ===")

        all_imgs = []
        for d in [_DATA / camera, _DATA / f"_EXCLUDED_{camera}"]:
            if d.exists():
                all_imgs += sorted(d.glob("*.jpg"))

        if not all_imgs:
            print(f"  No images found"); continue

        # Evenly spread across full timeline
        step = max(1, len(all_imgs) // N)
        sampled = all_imgs[::step][:N]
        print(f"  {len(all_imgs)} images, sampling {len(sampled)}")

        cam_dir = out_dir / camera
        cam_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 3, N_ROWS * 2.3), squeeze=False)
        for idx, img_path in enumerate(sampled):
            r, c = divmod(idx, N_COLS)
            try:
                img_rgb = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                axes[r, c].axis("off"); continue
            axes[r, c].imshow(img_rgb)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values(): sp.set_visible(False)
        # blank any unused cells
        for idx in range(len(sampled), N_ROWS * N_COLS):
            r, c = divmod(idx, N_COLS)
            axes[r, c].axis("off")

        fig.suptitle(f"{camera} — Unseen OOD Camera (Sample)", fontsize=13, fontweight="bold", y=1.005)
        plt.tight_layout()
        fname = cam_dir / f"{camera}_collage.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")

    # Legend reference card
    handles = [
        mpatches.Patch(color=tuple(GROUND_COLOR/255), alpha=0.7, label="GT Ground (green)"),
        mpatches.Patch(color=tuple(SNOW_COLOR/255),   alpha=0.7, label="GT Snow (blue)"),
        mpatches.Patch(color=tuple(PRED_GROUND/255),  alpha=0.7, label="Pred Ground (yellow)"),
        mpatches.Patch(color=tuple(PRED_SNOW/255),    alpha=0.7, label="Pred Snow (red)"),
    ]
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.legend(handles=handles, loc="center", fontsize=9, frameon=False)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "legend.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
