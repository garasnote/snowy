#!/usr/bin/env python3
"""
DINOv3 multi-head inference: one backbone pass → QA + ground + snow masks.

Saves per-image:
  {stem}_ground.png  — binary ground mask
  {stem}_snow.png    — binary snow mask
  {stem}_qa.txt      — usability score (0–1 float)

Usage:
    python inference_multihead.py --cameras CM00009 CM00014
    python inference_multihead.py --task qa      # only QA scores, fast
    python inference_multihead.py --task ground  # only ground masks
    python inference_multihead.py --task snow    # only snow masks
    python inference_multihead.py                # all three outputs
"""

import argparse
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
from multi_head.train_multihead import DINOv3MultiHead, make_transform, IN_DIST_CAMERAS
from student_ground.configs import MODEL_REGISTRY


def find_latest_ckpt(variant):
    ckpt_dir = _HERE / "checkpoints" / variant
    ckpts    = sorted(ckpt_dir.glob("best-*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    return ckpts[-1]


@torch.no_grad()
def run_inference(args):
    ckpt_path = find_latest_ckpt(args.variant)
    print(f"Loading: {ckpt_path.name}")
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    variant = ckpt.get("variant", args.variant)

    model = DINOv3MultiHead(variant=variant)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(args.device).eval()

    tf    = make_transform(MODEL_REGISTRY[variant]["input_size"], augment=False)
    tasks = {"qa", "ground", "snow"} if args.task == "all" else {args.task}

    cameras = args.cameras if args.cameras else IN_DIST_CAMERAS

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for cam in cameras:
        cam_dir = _DATA / cam
        if not cam_dir.exists():
            print(f"  skip {cam} (not found)")
            continue

        img_paths = sorted(cam_dir.glob("*.jpg"))
        if args.sample:
            img_paths = img_paths[:args.sample]

        cam_out = out_root / cam
        cam_out.mkdir(exist_ok=True)

        # Skip already done
        if not args.overwrite:
            def done(p):
                checks = []
                if "ground" in tasks:
                    checks.append((cam_out / f"{p.stem}_ground.png").exists())
                if "snow" in tasks:
                    checks.append((cam_out / f"{p.stem}_snow.png").exists())
                if "qa" in tasks:
                    checks.append((cam_out / f"{p.stem}_qa.txt").exists())
                return all(checks)
            img_paths = [p for p in img_paths if not done(p)]

        for i in tqdm(range(0, len(img_paths), args.batch_size), desc=cam):
            batch_paths = img_paths[i:i + args.batch_size]
            imgs, orig_sizes, valid_paths = [], [], []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    orig_sizes.append(img.size)
                    imgs.append(tf(img))
                    valid_paths.append(p)
                except Exception:
                    continue

            if not imgs:
                continue

            x = torch.stack(imgs).to(args.device)

            if tasks == {"qa"}:
                qa_logits = model.forward_qa(x)
                qa_scores = torch.sigmoid(qa_logits).cpu().numpy()
                for p, score in zip(valid_paths, qa_scores):
                    (cam_out / f"{p.stem}_qa.txt").write_text(f"{score:.6f}\n")

            elif tasks == {"ground"}:
                gr_logits = model.forward_seg(x, "ground")
                gr_preds  = (torch.sigmoid(gr_logits) > 0.5).squeeze(1).cpu().numpy()
                for p, pred, (ow, oh) in zip(valid_paths, gr_preds, orig_sizes):
                    binary = (pred * 255).astype(np.uint8)
                    Image.fromarray(binary).resize((ow, oh), Image.NEAREST).save(
                        cam_out / f"{p.stem}_ground.png")

            elif tasks == {"snow"}:
                sn_logits = model.forward_seg(x, "snow")
                sn_preds  = (torch.sigmoid(sn_logits) > 0.5).squeeze(1).cpu().numpy()
                for p, pred, (ow, oh) in zip(valid_paths, sn_preds, orig_sizes):
                    binary = (pred * 255).astype(np.uint8)
                    Image.fromarray(binary).resize((ow, oh), Image.NEAREST).save(
                        cam_out / f"{p.stem}_snow.png")

            else:
                # All three — single backbone pass
                qa_logits, gr_logits, sn_logits = model.forward_all(x)
                qa_scores = torch.sigmoid(qa_logits).cpu().numpy()
                gr_preds  = (torch.sigmoid(gr_logits) > 0.5).squeeze(1).cpu().numpy()
                sn_preds  = (torch.sigmoid(sn_logits) > 0.5).squeeze(1).cpu().numpy()

                for p, qa, gr, sn, (ow, oh) in zip(
                        valid_paths, qa_scores, gr_preds, sn_preds, orig_sizes):
                    if "qa" in tasks:
                        (cam_out / f"{p.stem}_qa.txt").write_text(f"{qa:.6f}\n")
                    if "ground" in tasks:
                        Image.fromarray((gr * 255).astype(np.uint8)).resize(
                            (ow, oh), Image.NEAREST).save(cam_out / f"{p.stem}_ground.png")
                    if "snow" in tasks:
                        Image.fromarray((sn * 255).astype(np.uint8)).resize(
                            (ow, oh), Image.NEAREST).save(cam_out / f"{p.stem}_snow.png")

    print(f"\nDone → {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",    choices=["dinov3_base", "dinov3_large"],
                        default="dinov3_base")
    parser.add_argument("--task",       choices=["all", "qa", "ground", "snow"],
                        default="all")
    parser.add_argument("--cameras",    nargs="+", default=None)
    parser.add_argument("--out-dir",    default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample",     type=int, default=0)
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = str(_ROOT / "preprocessing" / "segmentation" /
                           f"multihead_{args.variant}_output")

    print(f"Variant: {args.variant}  Task: {args.task}  Device: {args.device}")
    run_inference(args)


if __name__ == "__main__":
    main()
