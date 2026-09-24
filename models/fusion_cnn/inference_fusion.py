#!/usr/bin/env python3
"""
Run FusionCNN inference on all images → save pseudo-label masks.

Loads pre-saved teacher masks per image, stacks as channels, runs FusionCNN,
saves binary output mask as _snow.png or _ground.png.

Usage:
    python inference_fusion.py --task snow   --out-dir preprocessing/segmentation/fusion_snow_output
    python inference_fusion.py --task ground --out-dir preprocessing/segmentation/fusion_ground_output
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
_SEG  = _ROOT / "preprocessing" / "segmentation"
_DATA = _ROOT / "data"

sys.path.insert(0, str(_HERE))
from train_fusion import FusionCNN, GROUND_TEACHERS, SNOW_TEACHERS, IN_DIST_CAMERAS

SUFFIX = {"snow": "_snow.png", "ground": "_ground.png"}


def find_latest_ckpt(task):
    ckpt_dir = _HERE / "checkpoints" / f"fusion_{task}"
    ckpts = sorted(ckpt_dir.glob("best-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    return ckpts[-1]


@torch.no_grad()
def run_inference(args):
    teachers = GROUND_TEACHERS if args.task == "ground" else SNOW_TEACHERS
    n = len(teachers)
    suffix = SUFFIX[args.task]

    ckpt_path = find_latest_ckpt(args.task)
    print(f"Loading: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FusionCNN(n_teachers=ckpt.get("n_teachers", n))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(args.device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = IN_DIST_CAMERAS if not args.cameras else args.cameras

    total = 0
    for cam in cameras:
        cam_img_dir = _DATA / cam
        if not cam_img_dir.exists():
            continue
        img_paths = sorted(cam_img_dir.glob("*.jpg"))
        if args.sample:
            img_paths = img_paths[:args.sample]

        cam_out = out_dir / cam
        cam_out.mkdir(exist_ok=True)

        # Filter already done
        if not args.overwrite:
            img_paths = [p for p in img_paths
                         if not (cam_out / f"{p.stem}{suffix}").exists()]

        # Process in batches
        bs = args.batch_size
        for i in tqdm(range(0, len(img_paths), bs), desc=cam):
            batch_paths = img_paths[i:i+bs]

            # Load teacher masks for whole batch: (B, N, H, W)
            batch_masks = []
            orig_sizes = []
            valid_paths = []
            for img_path in batch_paths:
                try:
                    orig_sizes.append(Image.open(img_path).size)
                except Exception:
                    continue
                valid_paths.append(img_path)
                stem = img_path.stem
                masks = []
                for name, folder in teachers:
                    p = folder / cam / f"{stem}{suffix}"
                    if p.exists():
                        m = np.array(Image.open(p).convert("L").resize(
                            (args.size, args.size), Image.NEAREST)) / 255.0
                    else:
                        m = np.zeros((args.size, args.size), dtype=np.float32)
                    masks.append(m.astype(np.float32))
                batch_masks.append(np.stack(masks, axis=0))  # (N, H, W)

            if not batch_masks: continue
            x = torch.from_numpy(np.stack(batch_masks, axis=0)).to(args.device)  # (B, N, H, W)
            preds = model(x).squeeze(1).cpu().numpy()  # (B, H, W)
            for img_path, pred, (orig_w, orig_h) in zip(valid_paths, preds, orig_sizes):
                binary = (pred >= 0.5).astype(np.uint8) * 255
                Image.fromarray(binary).resize((orig_w, orig_h), Image.NEAREST).save(
                    cam_out / f"{img_path.stem}{suffix}"
                )
            total += len(batch_paths)

    print(f"\nDone. {total} masks saved → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",      choices=["snow", "ground"], required=True)
    parser.add_argument("--out-dir",   default=None)
    parser.add_argument("--cameras",   nargs="+", default=None)
    parser.add_argument("--size",      type=int, default=512)
    parser.add_argument("--sample",    type=int, default=0)
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = str(_SEG / f"fusion_{args.task}_output")

    print(f"Task:   {args.task}")
    print(f"Device: {args.device}")
    print(f"Output: {args.out_dir}")
    run_inference(args)


if __name__ == "__main__":
    main()
