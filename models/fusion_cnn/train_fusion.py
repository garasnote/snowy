#!/usr/bin/env python3
"""
FusionCNN: shallow CNN that fuses N teacher prediction masks → final segmentation.

Input:  (B, N_teachers, H, W) — stacked binary/soft teacher predictions
Output: (B, 1, H, W)          — fused binary mask

Trains on GT-annotated images only. Replaces hand-crafted IoU-weighted ensemble.
Run inference on unannotated images to generate pseudo-labels for student KD.

Usage:
    python train_fusion.py --task ground --epochs 50
    python train_fusion.py --task snow   --epochs 50
    python train_fusion.py --task ground --inference-only --out-dir /path/to/output
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

_HERE  = Path(__file__).resolve().parent
_ROOT  = _HERE.parent.parent
_SEG   = _ROOT / "preprocessing" / "segmentation"
_ANN   = _ROOT / "data" / "annotations"

sys.path.insert(0, str(_ROOT / "models"))


# ── Model ─────────────────────────────────────────────────────────────────────

class FusionCNN(nn.Module):
    """Shallow CNN that fuses N teacher predictions into a single segmentation mask.

    Architecture:
        Conv3(N→32) → BN → ReLU
        Conv3(32→16) → BN → ReLU
        Conv1(16→1) → Sigmoid          ← per-pixel weighted combination

    The 3×3 convs capture local agreement context ("when 3/4 teachers agree
    on an edge, trust them"). The final 1×1 conv is a learned per-pixel
    channel weighting — strictly more powerful than fixed IoU weights.
    """

    def __init__(self, n_teachers: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_teachers, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),   # 1×1: learned per-pixel weighting
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)   # (B, 1, H, W)


# ── Loss ──────────────────────────────────────────────────────────────────────

def dice_loss(pred, target, smooth=1.0):
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def combined_loss(pred, target, bce_w=1.0, dice_w=1.0):
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce_w * bce + dice_w * dice


# ── Dataset ───────────────────────────────────────────────────────────────────

GROUND_TEACHERS = [
    ("dinov2_pca",  _SEG / "dinov2_pca_output"),
    ("dinov3_pca",  _SEG / "dinov3_pca_output"),
    ("segformer",   _SEG / "segformer_output"),
    ("deeplabv3",   _SEG / "deeplabv3_output"),
    ("sam3_text",   _SEG / "sam3_dino_output" / "text"),
    ("sam3_dino",   _SEG / "sam3_dino_output" / "dino+sam"),
    ("depth",       _SEG / "depth_output" / "N0.05_F0.95"),
]

SNOW_TEACHERS = [
    ("clipseg",       _SEG / "clipseg_snow_output"),
    ("florence2",     _SEG / "florence2_snow_output"),
    ("groundedsam2",  _SEG / "groundedsam2_snow_output"),
    ("sam3",          _SEG / "sam3_snow_output"),
    ("yoloe",         _SEG / "yoloe_snow_output"),
    ("seggpt",        _SEG / "seggpt_snow_output"),
]

IN_DIST_CAMERAS = [
    "CF00001","CM00009","CM00014","CM00070",
    "CM00075","CM00080","CM00102","CM00108","CM00112"
]


def load_gt_stems(task: str) -> List[Tuple[str, str]]:
    """Returns list of (camera, stem) for all GT-annotated images."""
    from student_ground.dataset import decode_ls_rle
    stems = []
    if task == "ground":
        for fname in ["ground_annotations.json", "san.json", "heroku.json"]:
            p = _ANN / fname
            if not p.exists(): continue
            import json as _json
            tasks = _json.load(open(p))
            for t in tasks:
                url = t["data"]["image"]
                cam = next((c for c in IN_DIST_CAMERAS if c in url), None)
                if not cam: continue
                raw = url.split("/")[-1].replace(".jpg","")
                stem = raw.split("-",1)[1] if "-" in raw and raw.split("-")[1][:2] in ("CM","CF") else raw
                anns = t.get("annotations",[])
                if any(r.get("type")=="brushlabels" and "rle" in r.get("value",{})
                       for ann in anns for r in ann.get("result",[])):
                    stems.append((cam, stem))
    else:  # snow
        for cam in IN_DIST_CAMERAS:
            import re, json as _json
            num = re.search(r'\d+', cam).group().lstrip('0') or '0'
            p = _ANN / f"{num}.json"
            if not p.exists(): continue
            tasks = _json.load(open(p))
            for t in tasks:
                url = t["data"]["image"]
                if cam not in url: continue
                stem = url.split("/")[-1].replace(".jpg","")
                anns = t.get("annotations",[])
                if any(r.get("type")=="brushlabels" and "rle" in r.get("value",{})
                       for ann in anns for r in ann.get("result",[])):
                    stems.append((cam, stem))
    return stems


def decode_gt(task, cam, stem):
    import json as _json
    from student_ground.dataset import decode_ls_rle
    suffix = "_ground.png" if task == "ground" else "_snow.png"
    files = (["ground_annotations.json","san.json","heroku.json"]
             if task == "ground"
             else [f"{cam.lstrip('CM0').lstrip('0') or '0'}.json",
                   f"{int(cam[2:])}.json"])
    for fname in files:
        p = _ANN / fname
        if not p.exists(): continue
        tasks = _json.load(open(p))
        for t in tasks:
            url = t["data"]["image"]
            if cam not in url: continue
            raw = url.split("/")[-1].replace(".jpg","")
            s = raw.split("-",1)[1] if "-" in raw and raw.split("-")[1][:2] in ("CM","CF") else raw
            if s != stem: continue
            for ann in t.get("annotations",[]):
                for r in ann.get("result",[]):
                    if r.get("type") != "brushlabels" or "rle" not in r.get("value",{}): continue
                    w = r.get("original_width", 1296)
                    h = r.get("original_height", 960)
                    return decode_ls_rle(r["value"]["rle"], w, h).astype(np.float32)
    return None


class FusionDataset(Dataset):
    def __init__(self, samples, teachers, task, size=512):
        self.samples  = samples   # [(cam, stem), ...]
        self.teachers = teachers  # [(name, folder), ...]
        self.task     = task
        self.size     = size
        self.suffix   = "_ground.png" if task == "ground" else "_snow.png"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cam, stem = self.samples[idx]
        masks = []
        for name, folder in self.teachers:
            p = folder / cam / f"{stem}{self.suffix}"
            if p.exists():
                m = np.array(Image.open(p).convert("L").resize(
                    (self.size, self.size), Image.NEAREST)) / 255.0
            else:
                m = np.zeros((self.size, self.size), dtype=np.float32)
            masks.append(m.astype(np.float32))

        x = torch.from_numpy(np.stack(masks, axis=0))          # (N, H, W)

        gt = decode_gt(self.task, cam, stem)
        if gt is None:
            gt = np.zeros((self.size, self.size), dtype=np.float32)
        gt_img = Image.fromarray((gt * 255).astype(np.uint8)).resize(
            (self.size, self.size), Image.NEAREST)
        y = torch.from_numpy(np.array(gt_img).astype(np.float32) / 255.0).unsqueeze(0)

        return x, y


# ── Train ─────────────────────────────────────────────────────────────────────

def train(args):
    teachers = GROUND_TEACHERS if args.task == "ground" else SNOW_TEACHERS
    n_teachers = len(teachers)

    stems = load_gt_stems(args.task)
    print(f"GT samples: {len(stems)}")
    if len(stems) == 0:
        print("ERROR: no GT found"); return

    train_s, val_s = train_test_split(stems, test_size=0.15, random_state=42)
    train_ds = FusionDataset(train_s, teachers, args.task, size=args.size)
    val_ds   = FusionDataset(val_s,   teachers, args.task, size=args.size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = FusionCNN(n_teachers=n_teachers).to(args.device)
    opt   = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    out_dir = _ROOT / "models" / "fusion_cnn" / "checkpoints" / f"fusion_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_iou, best_path = 0.0, None
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            pred = model(x)
            loss = combined_loss(pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        sched.step()

        # Val IoU
        model.eval()
        inter_sum, union_sum = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(args.device), y.to(args.device)
                pred = (model(x) > 0.5).float()
                inter_sum += (pred * y).sum().item()
                union_sum += (pred + y - pred * y).sum().item()
        iou = inter_sum / (union_sum + 1e-6)

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss/len(train_dl):.4f}  val_IoU={iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            if best_path: best_path.unlink(missing_ok=True)
            best_path = out_dir / f"best-epoch={epoch:03d}-iou={iou:.4f}.ckpt"
            torch.save({"state_dict": model.state_dict(),
                        "n_teachers": n_teachers,
                        "task": args.task,
                        "teachers": [n for n, _ in teachers]}, best_path)
            print(f"  ✓ saved {best_path.name}")

    print(f"\nBest val IoU: {best_iou:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       choices=["ground","snow"], required=True)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--size",       type=int,   default=512)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(f"Task: {args.task}  Device: {args.device}")
    train(args)


if __name__ == "__main__":
    main()
