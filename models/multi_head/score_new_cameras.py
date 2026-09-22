#!/usr/bin/env python3
"""
Score new cameras with DINOv3MultiHead: QA score + snow fraction.
Outputs a CSV for sampling annotation candidates.

Usage:
    python score_new_cameras.py --ckpt checkpoints/hp_search/trial000_.../best.pt
    python score_new_cameras.py --ckpt ... --cameras CM00054 CM00067 --top-n 200
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_DATA = _ROOT / "data"

sys.path.insert(0, str(_ROOT / "models"))
from multi_head.train_multihead import DINOv3MultiHead
from student_ground.configs import MODEL_REGISTRY

NEW_CAMERAS = ["CM00054", "CM00067", "CM00071", "CM00115"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items()}
    if any(k.startswith("backbone.layer.") for k in state):
        state = {
            ("backbone.model." + k[len("backbone."):] if k.startswith("backbone.layer.") else k): v
            for k, v in state.items()
        }
    variant = ckpt.get("hyper_parameters", {}).get("variant", "dinov3_base")
    model   = DINOv3MultiHead(variant=variant)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), variant


def make_tf(input_size):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def score_cameras(args):
    device = torch.device(args.device)
    model, variant = load_model(args.ckpt, device)
    input_size = MODEL_REGISTRY[variant]["input_size"]
    tf = make_tf(input_size)

    cameras = args.cameras or NEW_CAMERAS
    all_rows = []

    for cam in cameras:
        cam_dir = _DATA / cam
        if not cam_dir.exists():
            print(f"  skip {cam} (not found at {cam_dir})")
            continue

        img_paths = sorted(cam_dir.glob("*.jpg"))
        print(f"\n{cam}: {len(img_paths)} images")

        rows = []
        for i in tqdm(range(0, len(img_paths), args.batch_size), desc=cam):
            batch = img_paths[i:i + args.batch_size]
            imgs, paths = [], []
            for p in batch:
                try:
                    imgs.append(tf(Image.open(p).convert("RGB")))
                    paths.append(p)
                except Exception:
                    continue
            if not imgs:
                continue

            x = torch.stack(imgs).to(device)
            qa_logits, _, sn_logits = model.forward_all(x)
            qa_scores  = torch.sigmoid(qa_logits).cpu().numpy()
            sn_probs   = torch.sigmoid(sn_logits).squeeze(1).cpu().numpy()
            sn_fracs   = (sn_probs > 0.5).mean(axis=(1, 2))

            for p, qa, sn in zip(paths, qa_scores, sn_fracs):
                rows.append({
                    "camera":        cam,
                    "filename":      p.name,
                    "qa_score":      float(qa),
                    "snow_fraction": float(sn),
                })

        # Sort by snow_fraction desc, then qa_score desc
        rows.sort(key=lambda r: (r["snow_fraction"], r["qa_score"]), reverse=True)

        if args.top_n:
            # Among top snow candidates, require qa_score > threshold
            candidates = [r for r in rows if r["qa_score"] >= args.qa_min]
            selected   = candidates[:args.top_n]
            print(f"  {cam}: {len(selected)} candidates selected (qa>={args.qa_min}, top snow)")
        else:
            selected = rows

        all_rows.extend(selected)

    # Write CSV
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["camera", "filename", "qa_score", "snow_fraction"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows → {out}")

    # Per-camera summary
    from collections import defaultdict
    by_cam = defaultdict(list)
    for r in all_rows:
        by_cam[r["camera"]].append(r)
    print("\nSummary:")
    for cam, rows in sorted(by_cam.items()):
        sn = [r["snow_fraction"] for r in rows]
        qa = [r["qa_score"] for r in rows]
        print(f"  {cam}: {len(rows)} imgs | snow_frac {min(sn):.2f}–{max(sn):.2f} | qa {min(qa):.2f}–{max(qa):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--cameras",    nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-n",      type=int, default=200, help="Top N snow candidates per camera (0=all)")
    parser.add_argument("--qa-min",     type=float, default=0.5, help="Min QA score to include")
    parser.add_argument("--out-csv",    default=str(_ROOT / "data" / "snow_candidates_new_cameras.csv"))
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    score_cameras(args)


if __name__ == "__main__":
    main()
