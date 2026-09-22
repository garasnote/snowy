#!/usr/bin/env python3
"""
DINOv3 multi-head model: one backbone, three task heads.

  Head 1 — QA:     image-level binary (usable / non-usable), CLS token → MLP
  Head 2 — Ground: pixel-level segmentation, patch tokens → DINOUNetDecoder
  Head 3 — Snow:   pixel-level segmentation, patch tokens → DINOUNetDecoder

Separate DataLoaders per task; batches interleaved each step.
Loss balancing via learned uncertainty weights (Kendall et al., 2018).
Training: PyTorch Lightning + EarlyStopping(patience) + MLflow logging.

Usage:
    python train_multihead.py --variant dinov3_base
    python train_multihead.py --variant dinov3_large --batch-size 8
    python train_multihead.py --variant dinov3_base --mlflow-uri http://localhost:5000
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import roc_auc_score, f1_score

try:
    from lightning.pytorch.loggers import MLFlowLogger
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_ANN  = _ROOT / "data" / "annotations"
_DATA = _ROOT / "data"
_QA   = _ROOT / "preprocessing" / "qa"
_SEG  = _ROOT / "preprocessing" / "segmentation"

# Offline backbone cache — prefer local path over HuggingFace at runtime.
# Download once: python -c "from transformers import AutoModel, AutoImageProcessor;
#   AutoModel.from_pretrained('facebook/dinov2-base').save_pretrained('/shared/home/juan.osorio/models/dinov2-base');
#   AutoImageProcessor.from_pretrained('facebook/dinov2-base').save_pretrained('/shared/home/juan.osorio/models/dinov2-base')"
_MODEL_CACHE = {
    "dinov3_base":  Path("/shared/home/juan.osorio/models/dinov2-base"),
    "dinov3_large": Path("/shared/home/juan.osorio/models/dinov2-large"),
}

sys.path.insert(0, str(_ROOT / "models"))
from student_ground.configs import MODEL_REGISTRY, IMAGENET_MEAN, IMAGENET_STD


class DINOUNetDecoder(nn.Module):
    """Progressive UNet decoder for DINOv2 patch tokens.

    Takes multi-scale patch token lists (shallow→deep), reshapes to spatial
    maps, then progressively upsamples 2× with skip connections.

    Input:  list of (B, N_patches, hidden_dim) — shallow to deep order
    Output: (B, 1, grid_h*2^n_up, grid_w*2^n_up) logits
    """

    def __init__(self, hidden_dim: int = 768, proj_dim: int = 128, num_layers: int = 4):
        super().__init__()
        # Project each layer's tokens to proj_dim
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.LayerNorm(proj_dim))
            for _ in range(num_layers)
        ])
        # UNet up-blocks: each doubles resolution, takes current + skip
        # Deepest level has no skip; remaining levels get skip from shallower layer
        # channels: deep→shallow, skip channels = proj_dim each
        self.up_blocks = nn.ModuleList()
        in_ch = proj_dim  # deepest feature
        for i in range(num_layers - 1):
            skip_ch = proj_dim
            self.up_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, proj_dim, 3, padding=1),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(proj_dim, proj_dim, 3, padding=1),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            ))
            in_ch = proj_dim
        self.head = nn.Conv2d(proj_dim, 1, 1)

    def forward(self, multi_level_features, grid_h, grid_w):
        # Project tokens and reshape to spatial maps
        spatial = []
        for feat, proj in zip(multi_level_features, self.projs):
            # feat: (B, N_patches, hidden_dim)
            x = proj(feat)                                   # (B, N, proj_dim)
            B, N, C = x.shape
            x = x.permute(0, 2, 1).reshape(B, C, grid_h, grid_w)  # (B, proj_dim, H, W)
            spatial.append(x)

        # spatial[0] = shallowest, spatial[-1] = deepest
        # Start from deepest, progressively upsample and merge skips
        x = spatial[-1]
        for i, block in enumerate(self.up_blocks):
            skip = spatial[-(i + 2)]  # next shallower level
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.head(x)  # (B, 1, grid_h, grid_w) — caller upsamples to full res

IN_DIST_CAMERAS = [
    "CF00001", "CM00009", "CM00014", "CM00070",
    "CM00075", "CM00080", "CM00102", "CM00108", "CM00112",
]


# ── Model ─────────────────────────────────────────────────────────────────────

class DINOv3MultiHead(nn.Module):
    """Frozen DINOv3 backbone + 3 independent task heads.

    Backbone runs once per image; all three heads share the features.
    Segmentation heads use DINOUNetDecoder: multi-scale patch tokens →
    progressive UNet upsampling with skip connections for sharper masks.
    Uncertainty weighting (log_var) balances task losses automatically.
    """

    def __init__(self, variant: str = "dinov3_base"):
        super().__init__()
        cfg = MODEL_REGISTRY[variant]
        self.variant        = variant
        self.input_size     = cfg["input_size"]
        self.patch_size     = cfg["patch_size"]
        self.extract_layers = cfg["extract_layers"]
        hidden_dim          = cfg["hidden_dim"]

        self.grid_h         = self.input_size // self.patch_size
        self.grid_w         = self.input_size // self.patch_size
        self.n_patch_tokens = self.grid_h * self.grid_w

        # Frozen backbone — prefer local cache, fall back to HuggingFace
        from transformers import AutoModel, AutoImageProcessor
        _local = _MODEL_CACHE.get(variant)
        _backbone_src = str(_local) if (_local and _local.exists()) else cfg["backbone"]
        self.processor = AutoImageProcessor.from_pretrained(_backbone_src)
        self.backbone  = AutoModel.from_pretrained(_backbone_src)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Head 1: QA — CLS token → MLP → scalar logit
        self.qa_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # Heads 2 & 3: segmentation via UNet decoder with skip connections
        self.ground_decoder = DINOUNetDecoder(
            hidden_dim=hidden_dim, proj_dim=128,
            num_layers=len(self.extract_layers),
        )
        self.snow_decoder = DINOUNetDecoder(
            hidden_dim=hidden_dim, proj_dim=128,
            num_layers=len(self.extract_layers),
        )

        # Learned log-variance per task for uncertainty weighting
        self.log_var = nn.Parameter(torch.zeros(3))

    @torch.no_grad()
    def extract_features(self, pixel_values):
        outputs      = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        cls_token    = hidden_states[-1][:, 0, :]
        patch_feats  = []
        for layer_idx in self.extract_layers:
            h = hidden_states[layer_idx][:, -self.n_patch_tokens:, :]
            patch_feats.append(h)
        return cls_token, patch_feats

    def forward_qa(self, pixel_values):
        cls, _ = self.extract_features(pixel_values)
        return self.qa_head(cls).squeeze(-1)

    def forward_seg(self, pixel_values, task: str, output_size=None):
        _, patch_feats = self.extract_features(pixel_values)
        decoder = self.ground_decoder if task == "ground" else self.snow_decoder
        logits  = decoder(patch_feats, self.grid_h, self.grid_w)
        if output_size is None:
            output_size = (self.input_size, self.input_size)
        if logits.shape[2:] != output_size:
            logits = F.interpolate(logits, size=output_size,
                                   mode="bilinear", align_corners=False)
        return logits

    def forward_all(self, pixel_values, output_size=None):
        """Single backbone pass → all 3 heads. For inference."""
        cls, patch_feats = self.extract_features(pixel_values)
        qa_logit     = self.qa_head(cls).squeeze(-1)
        gr_logits    = self.ground_decoder([f.clone() for f in patch_feats], self.grid_h, self.grid_w)
        sn_logits    = self.snow_decoder(patch_feats, self.grid_h, self.grid_w)
        if output_size is None:
            output_size = (self.input_size, self.input_size)
        if gr_logits.shape[2:] != output_size:
            gr_logits = F.interpolate(gr_logits, size=output_size,
                                      mode="bilinear", align_corners=False)
            sn_logits = F.interpolate(sn_logits, size=output_size,
                                      mode="bilinear", align_corners=False)
        return qa_logit, gr_logits, sn_logits

    def trainable_parameters(self):
        return (
            list(self.qa_head.parameters()) +
            list(self.ground_decoder.parameters()) +
            list(self.snow_decoder.parameters()) +
            [self.log_var]
        )


# ── Loss ──────────────────────────────────────────────────────────────────────

def dice_loss(pred, target, smooth=1.0):
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def seg_loss(logits, target):
    pred = torch.sigmoid(logits)
    return F.binary_cross_entropy_with_logits(logits, target) + 0.5 * dice_loss(pred, target)


def uncertainty_loss(task_losses, log_var):
    """Kendall et al.: L_i/(2σ²) + log(σ)."""
    total = sum(torch.exp(-log_var[i]) * l + log_var[i]
                for i, l in enumerate(task_losses))
    return total


# ── Datasets ──────────────────────────────────────────────────────────────────

def make_transform(input_size, augment=False):
    import torchvision.transforms.v2 as T
    if augment:
        return T.Compose([
            T.Resize((input_size + 32, input_size + 32)),
            T.RandomCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class QADataset(Dataset):
    def __init__(self, records, input_size, augment=False):
        self.records = records
        self.tf = make_transform(input_size, augment)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        img_path, label = self.records[idx]
        return self.tf(Image.open(img_path).convert("RGB")), \
               torch.tensor(label, dtype=torch.float32)


class SegDataset(Dataset):
    def __init__(self, records, input_size, augment=False):
        self.records    = records
        self.input_size = input_size
        self.tf         = make_transform(input_size, augment)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        img_path, mask_np = self.records[idx]
        img  = Image.open(img_path).convert("RGB")
        mask = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
            (self.input_size, self.input_size), Image.NEAREST)
        mask_t = torch.from_numpy(
            np.array(mask).astype(np.float32) / 255.0).unsqueeze(0)
        return self.tf(img), mask_t


# ── Data loading ──────────────────────────────────────────────────────────────

def load_qa_records():
    feat_csv = _QA / "checkpoints" / "qa_features.csv"
    if not feat_csv.exists():
        feat_csv = _ROOT / "analysis" / "image_features.csv"
    df = pd.read_csv(feat_csv)
    records = []
    for _, row in df.iterrows():
        cam   = str(row.get("camera", row.get("cam", "")))
        stem  = str(row.get("stem", ""))
        label = int(row.get("qa_usable", row.get("label", -1)))
        if label < 0:
            continue
        p = _DATA / cam / f"{stem}.jpg"
        if p.exists():
            records.append((p, label))
    return records


def load_ground_records():
    from student_ground.dataset import decode_ls_rle
    records = []
    for fname in ["ground_annotations.json", "san.json", "heroku.json"]:
        p = _ANN / fname
        if not p.exists():
            continue
        for t in json.load(open(p)):
            url = t["data"]["image"]
            cam = next((c for c in IN_DIST_CAMERAS if c in url), None)
            if not cam:
                continue
            raw  = url.split("/")[-1].replace(".jpg", "")
            stem = raw.split("-", 1)[1] if "-" in raw and raw.split("-")[1][:2] in ("CM", "CF") else raw
            img_path = _DATA / cam / f"{stem}.jpg"
            if not img_path.exists():
                continue
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "brushlabels" or "rle" not in r.get("value", {}):
                        continue
                    w = r.get("original_width", 1296)
                    h = r.get("original_height", 960)
                    mask = decode_ls_rle(r["value"]["rle"], w, h).astype(np.float32)
                    records.append((img_path, mask))
                    break
    return records


def load_snow_records():
    from student_ground.dataset import decode_ls_rle
    records = []
    for cam in IN_DIST_CAMERAS:
        num = re.search(r'\d+', cam).group().lstrip('0') or '0'
        p   = _ANN / f"{num}.json"
        if not p.exists():
            continue
        for t in json.load(open(p)):
            url = t["data"]["image"]
            if cam not in url:
                continue
            stem     = url.split("/")[-1].replace(".jpg", "")
            img_path = _DATA / cam / f"{stem}.jpg"
            if not img_path.exists():
                continue
            for ann in t.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type") != "brushlabels" or "rle" not in r.get("value", {}):
                        continue
                    w = r.get("original_width", 1296)
                    h = r.get("original_height", 960)
                    mask = decode_ls_rle(r["value"]["rle"], w, h).astype(np.float32)
                    records.append((img_path, mask))
                    break
    return records


class PseudoCheckpointCallback(Callback):
    """Save decoder weights every N epochs so a job kill doesn't lose everything."""

    def __init__(self, out_dir: Path, every_n_epochs: int = 5):
        self.out_dir        = Path(out_dir)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0:
            path = self.out_dir / f"pseudo_epoch{epoch:03d}.pt"
            torch.save({
                "epoch":          epoch,
                "ground_decoder": pl_module.model.ground_decoder.state_dict(),
                "snow_decoder":   pl_module.model.snow_decoder.state_dict(),
            }, path)
            print(f"  [pseudo ckpt] epoch {epoch} → {path}")


# ── Pseudolabel data ──────────────────────────────────────────────────────────

def _img_ok(path) -> bool:
    try:
        img = Image.open(path)
        img.load()
        return True
    except Exception:
        return False


def _scan_pseudo_records(pseudo_dir: Path, suffix: str) -> list:
    """Scan pseudo_dir for valid (img, mask) pairs — slow first-time only."""
    records = []
    for cam_dir in sorted(pseudo_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        cam     = cam_dir.name
        img_dir = _DATA / cam
        if not img_dir.exists():
            continue
        for mask_path in sorted(cam_dir.glob(f"*_{suffix}.png")):
            stem     = mask_path.stem[: -len(suffix) - 1]  # strip "_ground" / "_snow"
            img_path = img_dir / f"{stem}.jpg"
            if img_path.exists() and _img_ok(img_path):
                records.append((img_path, mask_path))
    return records


def _load_cache(cache_path: Path) -> list:
    records = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_s, mask_s = line.split("\t")
            records.append((Path(img_s), Path(mask_s)))
    return records


def _save_cache(records: list, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for img_p, mask_p in records:
            f.write(f"{img_p}\t{mask_p}\n")


def load_pseudo_ground_records(pseudo_dir, cache_path=None) -> list:
    if pseudo_dir is None:
        return []
    pseudo_dir = Path(pseudo_dir)
    if not pseudo_dir.exists():
        return []
    if cache_path and Path(cache_path).exists():
        print(f"  [cache] loading ground records from {cache_path}")
        return _load_cache(Path(cache_path))
    records = _scan_pseudo_records(pseudo_dir, "ground")
    if cache_path:
        _save_cache(records, Path(cache_path))
        print(f"  [cache] saved {len(records)} ground records → {cache_path}")
    return records


def load_pseudo_snow_records(pseudo_dir, cache_path=None) -> list:
    if pseudo_dir is None:
        return []
    pseudo_dir = Path(pseudo_dir)
    if not pseudo_dir.exists():
        return []
    if cache_path and Path(cache_path).exists():
        print(f"  [cache] loading snow records from {cache_path}")
        return _load_cache(Path(cache_path))
    records = _scan_pseudo_records(pseudo_dir, "snow")
    if cache_path:
        _save_cache(records, Path(cache_path))
        print(f"  [cache] saved {len(records)} snow records → {cache_path}")
    return records


class PseudoSegDataset(Dataset):
    def __init__(self, records, input_size, augment=False):
        self.records    = records
        self.input_size = input_size
        self.tf         = make_transform(input_size, augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, mask_path = self.records[idx]
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").resize(
            (self.input_size, self.input_size), Image.NEAREST)
        mask_t = torch.from_numpy(
            np.array(mask).astype(np.float32) / 255.0).unsqueeze(0)
        return self.tf(img), mask_t


class PseudoDataModule(pl.LightningDataModule):
    def __init__(self, variant, batch_size, num_workers=4,
                 pseudo_ground_dir=None, pseudo_snow_dir=None,
                 ground_cache=None, snow_cache=None):
        super().__init__()
        self.variant           = variant
        self.batch_size        = batch_size
        self.num_workers       = num_workers
        self.input_size        = MODEL_REGISTRY[variant]["input_size"]
        self.pseudo_ground_dir = pseudo_ground_dir
        self.pseudo_snow_dir   = pseudo_snow_dir
        self.ground_cache      = ground_cache
        self.snow_cache        = snow_cache

    def setup(self, stage=None):
        gr_recs = load_pseudo_ground_records(self.pseudo_ground_dir, self.ground_cache)
        sn_recs = load_pseudo_snow_records(self.pseudo_snow_dir, self.snow_cache)
        print(f"  Pseudo ground: {len(gr_recs)}  Pseudo snow: {len(sn_recs)}")
        sz = self.input_size
        self.gr_train = PseudoSegDataset(gr_recs, sz, augment=True)
        self.sn_train = PseudoSegDataset(sn_recs, sz, augment=True)

    def _dl(self, ds):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          drop_last=True)

    def train_dataloader(self):
        from lightning.pytorch.utilities import CombinedLoader
        return CombinedLoader(
            {"ground": self._dl(self.gr_train), "snow": self._dl(self.sn_train)},
            mode="max_size_cycle",
        )


class PseudoModule(pl.LightningModule):
    def __init__(self, variant, lr, max_epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model      = DINOv3MultiHead(variant=variant)
        self.lr         = lr
        self.max_epochs = max_epochs

    def training_step(self, batch, batch_idx):
        l_gr  = seg_loss(self.model.forward_seg(batch["ground"][0], "ground"),
                         batch["ground"][1])
        l_sn  = seg_loss(self.model.forward_seg(batch["snow"][0],   "snow"),
                         batch["snow"][1])
        total = l_gr + l_sn
        self.log("pseudo/loss_ground", l_gr.item(),   prog_bar=True)
        self.log("pseudo/loss_snow",   l_sn.item(),   prog_bar=True)
        self.log("pseudo/loss_total",  total.item())
        return total

    def configure_optimizers(self):
        opt   = AdamW(self.model.trainable_parameters(), lr=self.lr, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ── Lightning Module ───────────────────────────────────────────────────────────

class MultiHeadModule(pl.LightningModule):
    def __init__(self, variant, lr, max_epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model      = DINOv3MultiHead(variant=variant)
        self.qa_crit    = nn.BCEWithLogitsLoss()
        self.lr         = lr
        self.max_epochs = max_epochs

        # Accumulated val outputs for epoch-level metrics
        self._val_qa_probs   = []
        self._val_qa_targets = []
        self._val_gr_inter   = 0.0
        self._val_gr_union   = 0.0
        self._val_sn_inter   = 0.0
        self._val_sn_union   = 0.0

    # ── Iterators stored in DataModule, accessed via trainer ──────────────────

    def training_step(self, batch, batch_idx):
        # batch is a dict with keys "qa", "ground", "snow"
        task_losses = []

        x_qa, y_qa = batch["qa"]
        l_qa = self.qa_crit(self.model.forward_qa(x_qa), y_qa)
        task_losses.append(l_qa)

        x_gr, y_gr = batch["ground"]
        l_gr = seg_loss(self.model.forward_seg(x_gr, "ground"), y_gr)
        task_losses.append(l_gr)

        x_sn, y_sn = batch["snow"]
        l_sn = seg_loss(self.model.forward_seg(x_sn, "snow"), y_sn)
        task_losses.append(l_sn)

        total = uncertainty_loss(task_losses, self.model.log_var)

        self.log("train/loss_total",  total.item(), prog_bar=True)
        self.log("train/loss_qa",     l_qa.item())
        self.log("train/loss_ground", l_gr.item())
        self.log("train/loss_snow",   l_sn.item())
        return total

    def on_train_epoch_end(self):
        sigmas = torch.exp(self.model.log_var * 0.5).detach()
        self.log("sigma/qa",     sigmas[0].item())
        self.log("sigma/ground", sigmas[1].item())
        self.log("sigma/snow",   sigmas[2].item())

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:  # QA
            x, y = batch
            prob = torch.sigmoid(self.model.forward_qa(x)).cpu()
            self._val_qa_probs.extend(prob.tolist())
            self._val_qa_targets.extend(y.cpu().tolist())

        elif dataloader_idx == 1:  # Ground
            x, y = batch
            logits = self.model.forward_seg(x, "ground")
            pred   = (torch.sigmoid(logits) > 0.5).float()
            self._val_gr_inter += (pred * y).sum().item()
            self._val_gr_union += (pred + y - pred * y).sum().item()

        else:  # Snow
            x, y = batch
            logits = self.model.forward_seg(x, "snow")
            pred   = (torch.sigmoid(logits) > 0.5).float()
            self._val_sn_inter += (pred * y).sum().item()
            self._val_sn_union += (pred + y - pred * y).sum().item()

    def on_validation_epoch_end(self):
        probs   = self._val_qa_probs
        targets = self._val_qa_targets
        if len(set(targets)) > 1:
            auroc = roc_auc_score(targets, probs)
            f1    = f1_score(targets, [1 if p >= 0.5 else 0 for p in probs],
                             zero_division=0)
        else:
            auroc, f1 = 0.0, 0.0

        gr_iou = self._val_gr_inter / (self._val_gr_union + 1e-6)
        sn_iou = self._val_sn_inter / (self._val_sn_union + 1e-6)
        avg    = (auroc + gr_iou + sn_iou) / 3.0

        self.log("val/qa_auroc",    auroc,  prog_bar=True)
        self.log("val/qa_f1",       f1)
        self.log("val/ground_iou",  gr_iou, prog_bar=True)
        self.log("val/snow_iou",    sn_iou, prog_bar=True)
        self.log("val/avg_score",   avg,    prog_bar=True)

        # Reset accumulators
        self._val_qa_probs   = []
        self._val_qa_targets = []
        self._val_gr_inter   = 0.0
        self._val_gr_union   = 0.0
        self._val_sn_inter   = 0.0
        self._val_sn_union   = 0.0

    def configure_optimizers(self):
        opt   = AdamW(self.model.trainable_parameters(), lr=self.lr, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=1e-5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ── DataModule ────────────────────────────────────────────────────────────────

class MultiHeadDataModule(pl.LightningDataModule):
    """Wraps three separate DataLoaders into a combined batch dict."""

    def __init__(self, variant, batch_size, num_workers=4):
        super().__init__()
        self.variant     = variant
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.input_size  = MODEL_REGISTRY[variant]["input_size"]

    def setup(self, stage=None):
        from sklearn.model_selection import train_test_split

        qa_recs     = load_qa_records()
        ground_recs = load_ground_records()
        snow_recs   = load_snow_records()

        print(f"  QA: {len(qa_recs)}  Ground GT: {len(ground_recs)}  Snow GT: {len(snow_recs)}")

        qa_tr, qa_va = train_test_split(
            qa_recs, test_size=0.15, random_state=42,
            stratify=[r[1] for r in qa_recs])
        gr_tr, gr_va = train_test_split(ground_recs, test_size=0.15, random_state=42)
        sn_tr, sn_va = train_test_split(snow_recs,   test_size=0.15, random_state=42)

        sz = self.input_size
        self.qa_train  = QADataset(qa_tr,  sz, augment=True)
        self.qa_val    = QADataset(qa_va,  sz, augment=False)
        self.gr_train  = SegDataset(gr_tr, sz, augment=True)
        self.gr_val    = SegDataset(gr_va, sz, augment=False)
        self.sn_train  = SegDataset(sn_tr, sz, augment=True)
        self.sn_val    = SegDataset(sn_va, sz, augment=False)

    def _dl(self, ds, shuffle, drop_last=False):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          drop_last=drop_last)

    def train_dataloader(self):
        # CombinedLoader: interleaves batches from all three tasks each step
        from lightning.pytorch.utilities import CombinedLoader
        loaders = {
            "qa":     self._dl(self.qa_train,  shuffle=True, drop_last=True),
            "ground": self._dl(self.gr_train,  shuffle=True),
            "snow":   self._dl(self.sn_train,  shuffle=True),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return [
            self._dl(self.qa_val,  shuffle=False),
            self._dl(self.gr_val,  shuffle=False),
            self._dl(self.sn_val,  shuffle=False),
        ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",      choices=["dinov3_base", "dinov3_large"],
                        default="dinov3_base")
    parser.add_argument("--run-tag",      type=str,   default=None,
                        help="Suffix for the checkpoint dir (checkpoints/{variant}_{run-tag}/) "
                             "so parallel runs (e.g. a hybrid-pseudolabel finetune) don't "
                             "clobber each other's checkpoints/best-epoch files.")
    parser.add_argument("--max-epochs",   type=int,   default=100)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--mlflow-uri",   type=str,   default=None,
                        help="MLflow tracking URI (e.g. http://localhost:5000 or file:./mlruns)")
    parser.add_argument("--experiment",   type=str,   default="multihead_dinov3")
    parser.add_argument("--device",        default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pseudo-ground", type=str, default=None,
                        help="Dir of SAM3-text ground pseudolabels (cam/stem_ground.png)")
    parser.add_argument("--pseudo-snow",   type=str, default=None,
                        help="Dir of fusion snow pseudolabels (cam/stem_snow.png)")
    parser.add_argument("--pseudo-epochs", type=int, default=50,
                        help="Epochs for Phase 1 pseudolabel warmup")
    parser.add_argument("--pseudo-ground-cache", type=str, default=None,
                        help="Path to cache file for valid ground records (skip _img_ok scan if exists)")
    parser.add_argument("--pseudo-snow-cache",   type=str, default=None,
                        help="Path to cache file for valid snow records (skip _img_ok scan if exists)")
    parser.add_argument("--skip-pseudo-ckpt", type=str, default=None,
                        help="Path to a saved pseudo_phase1.pt (ground_decoder/snow_decoder state "
                             "dicts). If given, skip Phase 1 training entirely and load decoder "
                             "weights straight from this checkpoint.")
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.variant]
    if args.batch_size is None:
        args.batch_size = cfg["batch_size"]
    if args.lr is None:
        args.lr = cfg["lr"]

    print(f"Variant:    {args.variant}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR:         {args.lr}")
    print(f"Patience:   {args.patience} / max {args.max_epochs} epochs")

    ckpt_dir_name = args.variant if not args.run_tag else f"{args.variant}_{args.run_tag}"
    out_dir = _HERE / "checkpoints" / ckpt_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loggers
    loggers = [CSVLogger(str(out_dir), name="csv")]
    if args.mlflow_uri and _MLFLOW_AVAILABLE:
        # Force a brand-new run. An inherited MLFLOW_RUN_ID (e.g. leaked via
        # `srun --export=ALL` from a prior job/shell) makes MLFlowLogger
        # resume that run instead of creating one — then log_hyperparams()
        # crashes if this phase's params (lr, etc.) differ from what's
        # already logged there.
        os.environ.pop("MLFLOW_RUN_ID", None)
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_name = f"{args.variant}_lr{args.lr:.0e}_bs{args.batch_size}_job{job_id}"
        mlf_logger = MLFlowLogger(
            experiment_name=args.experiment,
            tracking_uri=args.mlflow_uri,
            run_name=run_name,
            run_id=None,
            log_model=False,
        )
        # NOTE: key is "cli_lr", not "lr" — MultiHeadModule.save_hyperparameters()
        # captures its own "lr" (= args.lr * 0.1, the actual Phase-2 training lr)
        # and Trainer.fit() auto-logs that under the same key at fit-start.
        # mlflow's file store forbids two different values for one param key
        # within a run, so logging our own "lr" here (the raw CLI value) would
        # collide with Lightning's auto-log and crash log_hyperparams().
        mlf_logger.log_hyperparams({
            "variant":    args.variant,
            "batch_size": args.batch_size,
            "cli_lr":     args.lr,
            "patience":   args.patience,
            "max_epochs": args.max_epochs,
        })
        loggers.insert(0, mlf_logger)
        print(f"MLflow:     {args.mlflow_uri}  experiment={args.experiment}  run={run_name}")
    elif args.mlflow_uri:
        print("WARNING: MLflow requested but not installed (pip install mlflow)")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(out_dir),
            filename="best-{epoch:03d}-avg{val/avg_score:.4f}-qa{val/qa_auroc:.4f}"
                     "-gr{val/ground_iou:.4f}-sn{val/snow_iou:.4f}",
            monitor="val/avg_score",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val/avg_score",
            patience=args.patience,
            mode="max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    accelerator = "gpu" if args.device == "gpu" else "cpu"

    # ── Phase 1: pseudolabel warmup ──────────────────────────────────────────
    pseudo_module = None
    pseudo_state  = None
    if args.skip_pseudo_ckpt:
        print(f"\n── Phase 1: SKIPPED, loading decoder weights from {args.skip_pseudo_ckpt} ──\n")
        pseudo_state = torch.load(args.skip_pseudo_ckpt, map_location="cpu")
    elif args.pseudo_ground or args.pseudo_snow:
        print(f"\n── Phase 1: pseudolabel warmup ({args.pseudo_epochs} epochs) ──")
        pseudo_dm     = PseudoDataModule(
            args.variant, args.batch_size, args.num_workers,
            pseudo_ground_dir=args.pseudo_ground,
            pseudo_snow_dir=args.pseudo_snow,
            ground_cache=args.pseudo_ground_cache,
            snow_cache=args.pseudo_snow_cache,
        )
        pseudo_module = PseudoModule(args.variant, args.lr, args.pseudo_epochs)
        pseudo_trainer = pl.Trainer(
            max_epochs=args.pseudo_epochs,
            accelerator=accelerator,
            devices=1,
            logger=CSVLogger(str(out_dir), name="csv_pseudo"),
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_checkpointing=False,
            callbacks=[PseudoCheckpointCallback(out_dir, every_n_epochs=5)],
        )
        pseudo_trainer.fit(pseudo_module, datamodule=pseudo_dm)

        pseudo_ckpt = out_dir / "pseudo_phase1.pt"
        pseudo_state = {
            "ground_decoder": pseudo_module.model.ground_decoder.state_dict(),
            "snow_decoder":   pseudo_module.model.snow_decoder.state_dict(),
        }
        torch.save(pseudo_state, pseudo_ckpt)
        print(f"Phase 1 weights saved → {pseudo_ckpt}")
        print("── Phase 1 complete ──\n")

    # ── Phase 2: GT finetune ─────────────────────────────────────────────────
    dm     = MultiHeadDataModule(args.variant, args.batch_size, args.num_workers)
    module = MultiHeadModule(args.variant, args.lr * 0.1, args.max_epochs)

    if pseudo_state is not None:
        module.model.ground_decoder.load_state_dict(pseudo_state["ground_decoder"])
        module.model.snow_decoder.load_state_dict(pseudo_state["snow_decoder"])
        print("Transferred pseudolabel decoder weights → GT module.")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(module, datamodule=dm)

    best = trainer.callback_metrics.get("val/avg_score", "N/A")
    print(f"\nBest val/avg_score: {best}")
    print(f"Checkpoints: {out_dir}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
