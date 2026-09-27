#!/usr/bin/env python3
"""
FusionCNN: shallow CNN that fuses N teacher prediction masks → final segmentation.

Input:  (B, N_teachers, H, W) — stacked binary/soft teacher predictions
Output: (B, 1, H, W)          — fused binary mask

Trains on GT-annotated images only. Replaces hand-crafted IoU-weighted ensemble.
Run inference on unannotated images to generate pseudo-labels for student KD.

Consolidated from train_fusion.py and inference_fusion.py. Each keeps its
original CLI flags, now behind a subcommand:

    python fusion_cnn.py train --task ground --epochs 50
    python fusion_cnn.py train --task snow   --epochs 50
    python fusion_cnn.py infer --task snow
    python fusion_cnn.py infer --task ground

--out-dir defaults to WORKSPACE_DIR/segmentation/fusion_<task> (paths.py);
pass --out-dir explicitly to override.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from paths import WORKSPACE_DIR
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
from tqdm import tqdm

_HERE  = Path(__file__).resolve().parent
_ROOT  = _HERE.parent.parent
_ANN   = _ROOT / "data" / "annotations"
_DATA  = _ROOT / "data"

sys.path.insert(0, str(_ROOT / "models"))


# =============================================================================
# Shared — model, teacher lists, camera/annotation maps
# =============================================================================

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


# Folders must match ground_teachers.py / snow_teachers.py's own default
# --output-dir (WORKSPACE_DIR/segmentation/<task>/<method>), including the
# two teachers that nest one level deeper themselves: sam3 (per --method:
# "text" or "dino+sam") and depth (per --near-clip/--far-clip, default tag
# "N0.05_F0.95"). Adding/removing/reordering a teacher here changes the
# model's input channel count and channel semantics — a checkpoint trained
# against a different list will fail the name/order check in
# run_infer_fusion rather than silently misaligning channels.
_GT_GROUND = WORKSPACE_DIR / "segmentation" / "ground"
_GT_SNOW   = WORKSPACE_DIR / "segmentation" / "snow"

GROUND_TEACHERS = [
    ("dinov2_pca",  _GT_GROUND / "dinov2_pca"),
    ("segformer",   _GT_GROUND / "segformer"),
    ("deeplabv3",   _GT_GROUND / "deeplabv3"),
    ("sam3_text",   _GT_GROUND / "sam3" / "text"),
    ("depth",       _GT_GROUND / "depth" / "N0.05_F0.95"),
]

SNOW_TEACHERS = [
    ("clipseg",       _GT_SNOW / "clipseg"),
    ("florence2",     _GT_SNOW / "florence2"),
    ("groundedsam2",  _GT_SNOW / "groundedsam2"),
    ("sam3",          _GT_SNOW / "sam3"),
    ("yoloe",         _GT_SNOW / "yoloe"),
    ("seggpt",        _GT_SNOW / "seggpt"),
]

IN_DIST_CAMERAS = [
    "CF00001","CM00009","CM00014","CM00070",
    "CM00075","CM00080","CM00102","CM00108","CM00112"
]

# Snow annotation filenames aren't a clean zero-padded numeric derivation
# from the camera ID (CF00001 -> "01.json", CM00009 -> "09.json", but
# CM00102 -> "102.json" with no padding) — same explicit map figures.py's
# BR_SNOW_ANN_FILES uses.
SNOW_ANN_FILES = {
    "CF00001": "01.json", "CM00009": "09.json", "CM00014": "14.json",
    "CM00070": "70.json", "CM00075": "75.json", "CM00080": "80.json",
    "CM00102": "102.json", "CM00108": "108.json", "CM00112": "112.json",
}


# =============================================================================
# TRAIN
# =============================================================================

def dice_loss(pred, target, smooth=1.0):
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def combined_loss(pred, target, bce_w=1.0, dice_w=1.0):
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce_w * bce + dice_w * dice


def load_gt_stems(task: str) -> List[Tuple[str, str]]:
    """Returns list of (camera, stem) for all GT-annotated images."""
    from student_ground.dataset import decode_ls_rle
    stems = []
    if task == "ground":
        # ground_annotations.json only. san.json/heroku.json look like they
        # belong here by name but are 100% brushlabels=["Snow Mano"] — snow
        # data, not ground (see eval_all.py's load_gt_masks for the full
        # taxonomy note). Including them here would train FusionCNN's
        # ground head against snow masks for CF00001/CM00009/CM00014.
        for fname in ["ground_annotations.json"]:
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
                       and ("Ground" in r.get("value",{}).get("brushlabels",[])
                            or "Ground Mano" in r.get("value",{}).get("brushlabels",[]))
                       for ann in anns for r in ann.get("result",[])):
                    stems.append((cam, stem))
    else:  # snow
        for cam in IN_DIST_CAMERAS:
            import json as _json
            p = _ANN / SNOW_ANN_FILES[cam]
            if not p.exists(): continue
            tasks = _json.load(open(p))
            for t in tasks:
                url = t["data"]["image"]
                if cam not in url: continue
                stem = url.split("/")[-1].replace(".jpg","")
                anns = t.get("annotations",[])
                if any(r.get("type")=="brushlabels" and "rle" in r.get("value",{})
                       and "Snow Mano" in r.get("value",{}).get("brushlabels",[])
                       for ann in anns for r in ann.get("result",[])):
                    stems.append((cam, stem))
    return stems


def decode_gt(task, cam, stem):
    import json as _json
    from student_ground.dataset import decode_ls_rle
    suffix = "_ground.png" if task == "ground" else "_snow.png"
    # See load_gt_stems for why these are explicit maps, not derived: ground
    # is ground_annotations.json only (san.json/heroku.json are snow data
    # despite the name), and the snow filenames aren't a clean numeric
    # derivation from the camera ID.
    files = ["ground_annotations.json"] if task == "ground" else [SNOW_ANN_FILES[cam]]
    want_labels = ("Ground", "Ground Mano") if task == "ground" else ("Snow Mano",)
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
                    if not any(l in r["value"].get("brushlabels",[]) for l in want_labels): continue
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

        for name, folder in teachers:
            if not folder.exists():
                print(f"  WARNING: teacher '{name}' output folder does not "
                      f"exist: {folder} — every sample will get an all-zero "
                      f"channel for it")
                continue
            missing = sum(
                1 for cam, stem in samples
                if not (folder / cam / f"{stem}{self._suffix_for(task)}").exists()
            )
            if missing:
                print(f"  WARNING: teacher '{name}' had no output for "
                      f"{missing}/{len(samples)} samples in this split "
                      f"(zero-filled)")

    @staticmethod
    def _suffix_for(task):
        return "_ground.png" if task == "ground" else "_snow.png"

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


def run_train_fusion(args):
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


def add_train_subparser(subparsers):
    p = subparsers.add_parser("train", help="Train FusionCNN on GT-annotated images")
    p.add_argument("--task",       choices=["ground","snow"], required=True)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--size",       type=int,   default=512)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p


# =============================================================================
# INFER
# =============================================================================

INFER_SUFFIX = {"snow": "_snow.png", "ground": "_ground.png"}


def find_latest_ckpt(task):
    """Return the checkpoint with the highest validation IoU.

    Name order is not an order. sorted() places best-epoch=030-iou=0.6506 after
    best-epoch=006-iou=0.7472, so the previous rule returned an older, worse
    checkpoint - and after the ground teacher list was pruned to the paper's
    five, a checkpoint trained on a different number of input channels.
    """
    ckpt_dir = _HERE / "checkpoints" / f"fusion_{task}"

    def key(p):
        try:
            iou = float(p.stem.split("iou=")[1])
        except (IndexError, ValueError):
            iou = -1.0
        try:
            epoch = int(p.stem.split("epoch=")[1].split("-")[0])
        except (IndexError, ValueError):
            epoch = -1
        return (iou, epoch)

    ckpts = sorted(ckpt_dir.glob("best-*.ckpt"), key=key)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    return ckpts[-1]


@torch.no_grad()
def run_infer_fusion(args):
    teachers = GROUND_TEACHERS if args.task == "ground" else SNOW_TEACHERS
    n = len(teachers)
    suffix = INFER_SUFFIX[args.task]

    ckpt_path = find_latest_ckpt(args.task)
    print(f"Loading: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_teachers = ckpt.get("teachers")
    current_teachers = [name for name, _ in teachers]
    if ckpt_teachers is not None and ckpt_teachers != current_teachers:
        raise ValueError(
            f"Checkpoint {ckpt_path.name} was trained with teachers "
            f"{ckpt_teachers}, but the current {'GROUND' if args.task == 'ground' else 'SNOW'}_TEACHERS "
            f"list is {current_teachers}. Channel order/membership must "
            f"match exactly, or predictions will be silently misaligned — "
            f"retrain the checkpoint or restore the matching teacher list."
        )

    model = FusionCNN(n_teachers=ckpt.get("n_teachers", n))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(args.device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, folder in teachers:
        if not folder.exists():
            print(f"  WARNING: teacher '{name}' output folder does not "
                  f"exist: {folder} — every image will get an all-zero "
                  f"channel for it")

    cameras = IN_DIST_CAMERAS if not args.cameras else args.cameras
    missing_counts = {name: 0 for name, _ in teachers}

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
                        missing_counts[name] += 1
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

    for name, count in missing_counts.items():
        if count:
            print(f"  WARNING: teacher '{name}' had no output for "
                  f"{count} images (zero-filled)")

    print(f"\nDone. {total} masks saved → {out_dir}")


def add_infer_subparser(subparsers):
    p = subparsers.add_parser("infer", help="Run FusionCNN inference, save pseudo-label masks")
    p.add_argument("--task",      choices=["snow", "ground"], required=True)
    p.add_argument("--out-dir",   default=None)
    p.add_argument("--cameras",   nargs="+", default=None)
    p.add_argument("--size",      type=int, default=512)
    p.add_argument("--sample",    type=int, default=0)
    p.add_argument("--overwrite",   action="store_true")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def run_infer(args):
    if args.out_dir is None:
        args.out_dir = str(WORKSPACE_DIR / "segmentation" / f"fusion_{args.task}")

    print(f"Task:   {args.task}")
    print(f"Device: {args.device}")
    print(f"Output: {args.out_dir}")
    run_infer_fusion(args)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FusionCNN: learned teacher-ensemble fusion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_train_subparser(subparsers)
    add_infer_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "train":
        print(f"Task: {args.task}  Device: {args.device}")
        run_train_fusion(args)
    elif args.command == "infer":
        run_infer(args)


if __name__ == "__main__":
    main()
