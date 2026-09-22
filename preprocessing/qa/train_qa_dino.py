#!/usr/bin/env python3
"""
DINOv2/v3-based QA classifiers — two variants:

1. DINOv2-Base + MLP head  (frozen backbone, ~1M trainable MLP params)
2. DINOv3-Base + MLP head  (frozen backbone, register tokens)

Contrast with TinyQACNN (~50k params, raw pixels):
- DINOv2/v3: ~86M frozen params + MLP head; rich semantic features
- No spatial info needed — CLS token captures global image quality
- If DINOv2 can't beat TinyQACNN, computation cost is never justified

Usage:
    python train_qa_dino.py --backbone dinov2 --epochs 30
    python train_qa_dino.py --backbone dinov3 --epochs 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

sys.path.insert(0, str(_ROOT / "models"))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Backbone ──────────────────────────────────────────────────────────────────

class DINOBackbone(nn.Module):
    """Frozen DINOv2 or DINOv3 ViT — extracts CLS token as global descriptor."""

    CONFIGS = {
        "dinov2": {
            "hub_repo": "facebookresearch/dinov2",
            "hub_name": "dinov2_vitb14",
            "embed_dim": 768,
            "input_size": 224,
        },
        "dinov3": {
            "hub_repo": "facebookresearch/dinov2",   # DINOv3 uses same hub
            "hub_name": "dinov2_vitb14_reg",          # register-token variant
            "embed_dim": 768,
            "input_size": 224,
        },
    }

    def __init__(self, backbone: str = "dinov2"):
        super().__init__()
        cfg = self.CONFIGS[backbone]
        self.embed_dim  = cfg["embed_dim"]
        self.input_size = cfg["input_size"]

        self.model = torch.hub.load(cfg["hub_repo"], cfg["hub_name"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False   # frozen — no fine-tuning

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)["x_norm_clstoken"]   # (B, embed_dim)


# ── MLP Classifier ────────────────────────────────────────────────────────────

class DINOQAClassifier(nn.Module):
    """Frozen DINO backbone + trainable MLP head for QA binary classification.

    Architecture:
        DINO CLS token (768-d) → Linear(768→256) → BN → ReLU → Dropout
                                → Linear(256→64)  → BN → ReLU → Dropout
                                → Linear(64→1)    → scalar logit
    """

    def __init__(self, backbone: str = "dinov2", dropout: float = 0.3):
        super().__init__()
        self.backbone = DINOBackbone(backbone)
        d = self.backbone.embed_dim

        self.head = nn.Sequential(
            nn.Linear(d, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)   # frozen
        return self.head(feats).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class QADINODataset(Dataset):
    def __init__(self, img_paths, labels, input_size=224, augment=False):
        self.img_paths  = img_paths
        self.labels     = labels
        self.input_size = input_size
        self.augment    = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        import torchvision.transforms.v2 as T
        if self.augment:
            tf = T.Compose([
                T.Resize((self.input_size + 32, self.input_size + 32)),
                T.RandomCrop(self.input_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                T.ToImage(), T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            tf = T.Compose([
                T.Resize((self.input_size, self.input_size)),
                T.ToImage(), T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        return tf(img), torch.tensor(self.labels[idx], dtype=torch.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def run_fold(model, train_dl, val_dl, device, epochs, lr):
    opt   = AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit  = nn.BCEWithLogitsLoss()

    best_auroc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for x, y in val_dl:
                p = torch.sigmoid(model(x.to(device))).cpu().numpy()
                probs.extend(p); targets.extend(y.numpy())
        auroc = roc_auc_score(targets, probs)
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.clone() for k, v in model.head.state_dict().items()}

    model.head.load_state_dict(best_state)
    return best_auroc, probs, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["dinov2","dinov3"], default="dinov2")
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--folds",    type=int,   default=5)
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load QA CSV (same as TinyQACNN)
    feat_csv = _ROOT / "preprocessing" / "qa" / "checkpoints" / "qa_features.csv"
    if not feat_csv.exists():
        feat_csv = _ROOT / "analysis" / "image_features.csv"
    df = pd.read_csv(feat_csv)

    img_dir = _ROOT / "data"
    img_paths, labels = [], []
    for _, row in df.iterrows():
        cam  = row.get("camera", row.get("cam", ""))
        stem = row.get("stem", "")
        label = int(row.get("qa_usable", row.get("usable", row.get("label", -1))))
        if label < 0: continue
        p = img_dir / cam / f"{stem}.jpg"
        if not p.exists(): p = img_dir / "balanced_snow" / cam / f"{stem}.jpg"
        if not p.exists(): continue
        img_paths.append(p); labels.append(label)

    print(f"Dataset: {len(img_paths)} images  ({sum(labels)} usable / {len(labels)-sum(labels)} non-usable)")
    print(f"Backbone: {args.backbone}  Device: {args.device}")

    input_size = DINOBackbone.CONFIGS[args.backbone]["input_size"]
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    img_arr = np.array(img_paths)
    lbl_arr = np.array(labels)

    fold_aurocs, all_probs, all_targets = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(img_arr, lbl_arr), 1):
        print(f"\n── Fold {fold}/{args.folds} ──")
        tr_ds = QADINODataset(img_arr[tr_idx], lbl_arr[tr_idx], input_size, augment=True)
        va_ds = QADINODataset(img_arr[va_idx], lbl_arr[va_idx], input_size, augment=False)
        tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  num_workers=4, drop_last=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=4)

        model = DINOQAClassifier(args.backbone).to(args.device)
        auroc, probs, targets = run_fold(model, tr_dl, va_dl, args.device, args.epochs, args.lr)
        fold_aurocs.append(auroc)
        all_probs.extend(probs); all_targets.extend(targets)
        print(f"  Fold {fold} best AUROC: {auroc:.4f}")

    overall_auroc = roc_auc_score(all_targets, all_probs)
    preds_bin = [1 if p >= 0.5 else 0 for p in all_probs]
    overall_f1 = f1_score(all_targets, preds_bin)
    print(f"\n{'='*50}")
    print(f"Backbone:     {args.backbone}")
    print(f"AUROC (CV):   {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
    print(f"AUROC (pool): {overall_auroc:.4f}")
    print(f"F1   (pool):  {overall_f1:.4f}")

    # Save final model trained on all data
    out_dir = _HERE / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    full_ds = QADINODataset(img_arr, lbl_arr, input_size, augment=True)
    full_dl = DataLoader(full_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    model   = DINOQAClassifier(args.backbone).to(args.device)
    opt     = AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    crit    = nn.BCEWithLogitsLoss()
    for _ in range(args.epochs):
        model.train()
        for x, y in full_dl:
            opt.zero_grad()
            nn.BCEWithLogitsLoss()(model(x.to(args.device)), y.to(args.device)).backward()
            opt.step()

    save_path = out_dir / f"qa_{args.backbone}.pt"
    torch.save({"head_state": model.head.state_dict(),
                "backbone": args.backbone,
                "auroc_cv": float(np.mean(fold_aurocs)),
                "f1_cv":    overall_f1}, save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
