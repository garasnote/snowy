"""Dataset for student segmentation training.

Loads original webcam images + ensemble confidence maps (soft pseudo-labels)
produced by segmentation/ground_utils.py's ensemble subcommand. Both image
and mask are jointly augmented.

Also supports fine-tuning on real ground truth annotations from Label Studio
(ground_annotations.json with RLE-encoded brush masks).

Directory convention:
    images:      master/data/{camera}/{stem}.jpg
    confidence:  ensemble_output/majority_{N}/{camera}/{stem}_confidence.png
    binary mask: ensemble_output/majority_{N}/{camera}/{stem}_ground.png
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from .configs import IMAGENET_MEAN, IMAGENET_STD

# Default depth directory (inferno-colored PNGs from monocular depth estimation)
_HERE = Path(__file__).resolve().parent
_DEFAULT_DEPTH_DIR = _HERE.parent.parent / "preprocessing" / "segmentation" / "depth_output" / "shared_depth"


def load_depth_luminance(path):
    """Load inferno-colored depth PNG → float32 [0,1] via luminance.

    Inferno colormap has monotonically increasing luminance, so perceived
    luminance (BT.601) recovers the correct depth ordering.
    """
    img = np.array(Image.open(path).convert("RGB"))
    depth = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]) / 255.0
    return depth.astype(np.float32)


# ── Sample discovery ─────────────────────────────────────────────────────────

def discover_samples(ensemble_dir, data_dir, cameras=None, depth_dir=None):
    """Scan ensemble output and match against original images.

    Args:
        ensemble_dir: Path to majority_N or weighted_N directory
        data_dir:     Path to master/data/
        cameras:      Optional list of camera IDs to restrict to
        depth_dir:    Path to depth maps (None = no depth). Each sample
                      tuple gets a 4th element: depth_path or None.

    Returns:
        List of (image_path, confidence_path, mask_path, depth_path_or_None) tuples
    """
    ensemble_dir = Path(ensemble_dir)
    data_dir = Path(data_dir)
    if depth_dir is not None:
        depth_dir = Path(depth_dir)

    if cameras:
        cam_dirs = [ensemble_dir / c for c in cameras if (ensemble_dir / c).exists()]
    else:
        cam_dirs = sorted([
            d for d in ensemble_dir.iterdir()
            if d.is_dir() and not d.name.endswith(".csv")
        ])

    samples = []
    missing_images = 0
    missing_depth = 0

    for cam_dir in cam_dirs:
        camera = cam_dir.name
        for conf_path in sorted(cam_dir.glob("*_confidence.png")):
            # Derive stem: strip _confidence suffix
            stem = conf_path.stem.replace("_confidence", "")
            mask_path = cam_dir / f"{stem}_ground.png"
            img_path = data_dir / camera / f"{stem}.jpg"

            if not img_path.exists():
                missing_images += 1
                continue
            if not mask_path.exists():
                continue

            depth_path = None
            if depth_dir is not None:
                dp = depth_dir / camera / f"{stem}_depth.png"
                if dp.exists():
                    depth_path = dp
                else:
                    missing_depth += 1

            samples.append((img_path, conf_path, mask_path, depth_path))

    if missing_images:
        print(f"  [dataset] Warning: {missing_images} images not found in {data_dir}")
    if missing_depth:
        print(f"  [dataset] Warning: {missing_depth} depth maps not found in {depth_dir}")

    return samples


def _extract_timestamp(path):
    """Extract timestamp from filename for temporal sorting."""
    import re
    m = re.search(r"(\d{12})", path.stem)
    if m:
        return m.group(1)
    return path.stem


def _extract_date(path):
    """Extract date string (YYYYMMDD) from filename timestamp."""
    import re
    m = re.search(r"(\d{8})\d{4}", path.stem)
    if m:
        return m.group(1)
    return ""


def train_val_split(samples, val_fraction=0.2, seed=42, gap_days=1):
    """Temporal split per camera on day boundaries with gap.

    Sorts by timestamp within each camera, finds the day boundary
    closest to the target split point, then inserts a gap of
    gap_days between train and val to prevent same-day leakage.

    Args:
        samples:      List of (img, conf, mask, ...) tuples
        val_fraction: Fraction to use for validation
        seed:         Unused (kept for API compat)
        gap_days:     Days to skip between train and val (default 1)

    Returns:
        train_samples, val_samples
    """
    by_camera = {}
    for s in samples:
        camera = s[0].parent.name
        by_camera.setdefault(camera, []).append(s)

    train, val = [], []
    for camera, cam_samples in sorted(by_camera.items()):
        sorted_samples = sorted(cam_samples, key=lambda s: _extract_timestamp(s[0]))

        # Group by date
        dates = [_extract_date(s[0]) for s in sorted_samples]
        unique_dates = sorted(set(dates))

        if len(unique_dates) < 3:
            # Too few days — fall back to image-level split
            n_val = max(1, int(len(sorted_samples) * val_fraction))
            split_idx = len(sorted_samples) - n_val
            train.extend(sorted_samples[:split_idx])
            val.extend(sorted_samples[split_idx:])
            continue

        # Find day boundary closest to target split
        n_val_days = max(1, int(len(unique_dates) * val_fraction))
        gap = min(gap_days, len(unique_dates) - n_val_days - 1)
        gap = max(0, gap)

        val_start_day_idx = len(unique_dates) - n_val_days
        train_end_day_idx = val_start_day_idx - gap

        train_dates = set(unique_dates[:train_end_day_idx])
        val_dates = set(unique_dates[val_start_day_idx:])

        for s, d in zip(sorted_samples, dates):
            if d in train_dates:
                train.append(s)
            elif d in val_dates:
                val.append(s)
            # else: in gap — dropped

    return train, val


# ── Augmentations ─────────────────────────────────────────────────────────────

def build_train_transform(input_size=518):
    """Joint image+mask augmentations for training.

    Includes ACDC-inspired augmentations (brightness/blur/contrast)
    to simulate fog, night, and adverse weather conditions.
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required. pip install albumentations")

    return A.Compose([
        A.RandomResizedCrop(
            size=(input_size, input_size),
            scale=(0.5, 1.0),
            ratio=(0.75, 1.33),
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.2, hue=0.05,
            p=0.8,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"confidence": "mask", "binary_mask": "mask"})


def build_val_transform(input_size=518):
    """Validation transform: resize + normalize only."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required. pip install albumentations")

    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"confidence": "mask", "binary_mask": "mask"})


# ── Dataset ──────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    """Webcam ground segmentation dataset with soft pseudo-labels.

    Each sample returns:
        image:       (3, H, W) or (4, H, W) float32, ImageNet-normalized RGB + optional depth
        confidence:  (1, H, W) float32 in [0, 1] — soft teacher label
        binary_mask: (1, H, W) float32, 0 or 1 — hard ensemble mask
        stem:        filename stem (for debugging / saving outputs)
    """

    def __init__(self, samples, transform=None, input_size=518):
        """
        Args:
            samples:    List of (image_path, confidence_path, mask_path, depth_path_or_None)
            transform:  albumentations Compose transform
            input_size: fallback resize if no transform given
        """
        self.samples = samples
        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, conf_path, mask_path, depth_path = self.samples[idx]

        # Load image (RGB)
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load confidence map [0, 255] → [0, 1] float
        try:
            conf = np.array(Image.open(conf_path).convert("L")).astype(np.float32) / 255.0
        except Exception:
            conf = np.ones(image.shape[:2], dtype=np.float32)

        # Load binary mask [0 or 255] → [0 or 1] float
        binary = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.float32)

        # Load depth map if available
        depth = None
        if depth_path is not None:
            depth = load_depth_luminance(depth_path)

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                confidence=conf,
                binary_mask=binary,
            )
            image = transformed["image"]           # (3, H, W) tensor
            conf = transformed["confidence"]       # (H, W) tensor
            binary = transformed["binary_mask"]    # (H, W) tensor
        else:
            # Minimal fallback: resize + to tensor
            from PIL import Image as PILImage
            image = PILImage.fromarray(image).resize(
                (self.input_size, self.input_size), PILImage.BILINEAR
            )
            image = torch.from_numpy(
                np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
            )
            conf = torch.from_numpy(
                np.array(PILImage.fromarray(
                    (conf * 255).astype(np.uint8)
                ).resize((self.input_size, self.input_size), PILImage.BILINEAR)
                ).astype(np.float32) / 255.0
            )
            binary = torch.from_numpy(
                (np.array(PILImage.fromarray(
                    (binary * 255).astype(np.uint8)
                ).resize((self.input_size, self.input_size), PILImage.NEAREST)
                ) > 127).astype(np.float32)
            )

        # Append depth as 4th channel if available
        if depth is not None:
            h, w = image.shape[1], image.shape[2]
            depth_resized = np.array(
                Image.fromarray((depth * 255).astype(np.uint8)).resize(
                    (w, h), Image.BILINEAR)
            ).astype(np.float32) / 255.0
            depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0)  # (1, H, W)
            image = torch.cat([image, depth_tensor], dim=0)  # (4, H, W)

        # Ensure (1, H, W) shape for mask tensors
        if conf.ndim == 2:
            conf = conf.unsqueeze(0)
        if binary.ndim == 2:
            binary = binary.unsqueeze(0)

        return {
            "image": image,
            "confidence": conf,
            "binary_mask": binary,
            "stem": img_path.stem,
        }


# ── Ground truth annotation loading (for fine-tuning) ────────────────────────

EXCLUDED_CAMERAS = {"CM00024"}  # Zgornja Sorica — removed from pipeline


class _InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)


def _bytes2bit(data):
    return "".join(
        str((data[num // 8] >> (7 - num % 8)) & 1) for num in range(len(data) * 8)
    )


def decode_ls_rle(rle_list, width, height):
    """Decode Label Studio brush RLE → binary mask (H, W)."""
    inp = _InputStream(_bytes2bit(bytes(rle_list)))
    num = inp.read(32)
    word_size = inp.read(5) + 1
    rle_sizes = [inp.read(4) + 1 for _ in range(4)]
    flat = np.zeros(num, dtype=np.uint8)
    i = 0
    while i < num:
        is_repeated = inp.read(1)
        rle_size_idx = inp.read(2)
        run_length = inp.read(rle_sizes[rle_size_idx])
        j = i + 1 + run_length
        if is_repeated:
            val = inp.read(word_size)
            flat[i:j] = val
        else:
            while i < j:
                flat[i] = inp.read(word_size)
                i += 1
        i = j
    n_channels = num // (width * height)
    if n_channels >= 4:
        rgba = flat.reshape(height, width, n_channels)
        return rgba[:, :, 3] > 0
    return flat[:width * height].reshape(height, width) > 0


def discover_gt_samples(annotations_path, data_dir, cameras=None):
    """Load ground truth annotations from Label Studio JSON export.

    Args:
        annotations_path: Path to ground_annotations.json
        data_dir: Path to master/data/
        cameras: Optional list of cameras to restrict to

    Returns:
        List of (image_path, gt_mask_ndarray) tuples
    """
    annotations_path = Path(annotations_path)
    data_dir = Path(data_dir)

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")

    with open(annotations_path) as f:
        tasks = json.load(f)

    samples = []
    skipped = 0

    for task in tasks:
        url = task["data"]["image"]
        parts = url.rstrip("/").split("/")
        filename = parts[-1]
        camera = parts[-2]

        if camera in EXCLUDED_CAMERAS:
            continue
        if cameras and camera not in cameras:
            continue

        img_path = data_dir / camera / filename
        if not img_path.exists():
            skipped += 1
            continue

        annotations = task.get("annotations", [])
        if not annotations or not annotations[0].get("result"):
            continue

        result = annotations[0]["result"]
        w = result[0].get("original_width", 1296)
        h = result[0].get("original_height", 960)
        merged_mask = np.zeros((h, w), dtype=bool)

        for r in result:
            if r.get("type") == "brushlabels" and r.get("value", {}).get("rle"):
                rle = r["value"]["rle"]
                try:
                    mask = decode_ls_rle(rle, w, h)
                    merged_mask |= mask
                except Exception:
                    continue

        samples.append((img_path, merged_mask.astype(np.float32)))

    if skipped:
        print(f"  [gt_dataset] Warning: {skipped} images not found in {data_dir}")
    print(f"  [gt_dataset] Loaded {len(samples)} ground truth annotations")

    return samples


class GTSegmentationDataset(Dataset):
    """Dataset for fine-tuning on real ground truth annotations.

    Returns the same dict format as SegmentationDataset so the Lightning
    module works unchanged. confidence == binary_mask (hard labels only).
    """

    def __init__(self, samples, transform=None, input_size=518, depth_dir=None):
        self.samples = samples  # list of (img_path, gt_mask_ndarray)
        self.transform = transform
        self.input_size = input_size
        self.depth_dir = Path(depth_dir) if depth_dir else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_mask = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        binary = gt_mask.copy()

        # Load depth if available
        depth = None
        if self.depth_dir is not None:
            camera = img_path.parent.name
            stem = img_path.stem
            dp = self.depth_dir / camera / f"{stem}_depth.png"
            if dp.exists():
                depth = load_depth_luminance(dp)

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                # Re-use confidence slot for the GT mask
                confidence=binary,
                binary_mask=binary,
            )
            image = transformed["image"]
            conf = transformed["confidence"]
            binary = transformed["binary_mask"]
        else:
            from PIL import Image as PILImage
            image = PILImage.fromarray(image).resize(
                (self.input_size, self.input_size), PILImage.BILINEAR
            )
            image = torch.from_numpy(
                np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
            )
            binary = torch.from_numpy(
                (np.array(PILImage.fromarray(
                    (binary * 255).astype(np.uint8)
                ).resize((self.input_size, self.input_size), PILImage.NEAREST)
                ) > 127).astype(np.float32)
            )
            conf = binary.clone()

        # Append depth as 4th channel if available
        if depth is not None:
            h, w = image.shape[1], image.shape[2]
            depth_resized = np.array(
                Image.fromarray((depth * 255).astype(np.uint8)).resize(
                    (w, h), Image.BILINEAR)
            ).astype(np.float32) / 255.0
            depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0)
            image = torch.cat([image, depth_tensor], dim=0)  # (4, H, W)

        if conf.ndim == 2:
            conf = conf.unsqueeze(0)
        if binary.ndim == 2:
            binary = binary.unsqueeze(0)

        return {
            "image": image,
            "confidence": conf,       # == binary (hard labels for GT)
            "binary_mask": binary,
            "stem": img_path.stem,
        }


def build_gt_dataloaders(
    annotations_path,
    data_dir,
    input_size=518,
    batch_size=8,
    val_fraction=0.2,
    num_workers=4,
    cameras=None,
    seed=42,
    depth_dir=None,
    kfold_idx=-1,
    kfold_k=5,
):
    """Build train/val DataLoaders from real ground truth annotations.

    Args:
        kfold_idx: K-fold index (0-4 for 5-fold CV, -1 to disable k-fold and use val_fraction)
        kfold_k: Total number of folds
    """
    all_samples = discover_gt_samples(annotations_path, data_dir, cameras=cameras)
    if not all_samples:
        raise ValueError(f"No GT samples found in {annotations_path}")

    # Stratified split by camera
    rng = random.Random(seed)
    by_camera = {}
    for s in all_samples:
        camera = s[0].parent.name
        by_camera.setdefault(camera, []).append(s)

    if kfold_idx >= 0:
        # K-fold stratified split
        folds = [[] for _ in range(kfold_k)]
        for camera, cam_samples in sorted(by_camera.items()):
            shuffled = list(cam_samples)
            rng.shuffle(shuffled)
            fold_size = len(shuffled) // kfold_k
            for fi in range(kfold_k):
                start = fi * fold_size
                end = start + fold_size if fi < kfold_k - 1 else len(shuffled)
                folds[fi].extend(shuffled[start:end])

        val_samples = folds[kfold_idx]
        train_samples = [s for i in range(kfold_k) if i != kfold_idx for s in folds[i]]
    else:
        # Standard 80/20 split
        train_samples, val_samples = [], []
        for camera, cam_samples in sorted(by_camera.items()):
            shuffled = list(cam_samples)
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_fraction))
            val_samples.extend(shuffled[:n_val])
            train_samples.extend(shuffled[n_val:])

    print(f"  GT Dataset: {len(all_samples)} total, "
          f"{len(train_samples)} train, {len(val_samples)} val")

    train_tf = build_train_transform(input_size)
    val_tf = build_val_transform(input_size)

    train_ds = GTSegmentationDataset(train_samples, transform=train_tf, input_size=input_size, depth_dir=depth_dir)
    val_ds = GTSegmentationDataset(val_samples, transform=val_tf, input_size=input_size, depth_dir=depth_dir)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=len(train_samples) > batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, (len(train_samples), len(val_samples))


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    ensemble_dir,
    data_dir,
    input_size=518,
    batch_size=16,
    val_fraction=0.2,
    num_workers=8,
    cameras=None,
    seed=42,
    depth_dir=None,
):
    """Build train and val DataLoaders.

    Args:
        ensemble_dir:  Path to majority_N or weighted_N directory
        data_dir:      Path to master/data/
        input_size:    Target image resolution
        batch_size:    Training batch size
        val_fraction:  Fraction for validation (default 0.2)
        num_workers:   DataLoader workers
        cameras:       Restrict to specific cameras (None = all)
        seed:          Random seed for reproducibility
        depth_dir:     Path to depth maps directory (None = RGB only)

    Returns:
        train_loader, val_loader, (n_train, n_val)
    """
    all_samples = discover_samples(ensemble_dir, data_dir, cameras=cameras, depth_dir=depth_dir)
    if not all_samples:
        raise ValueError(
            f"No samples found!\n"
            f"  ensemble_dir: {ensemble_dir}\n"
            f"  data_dir:     {data_dir}\n"
            "Check that ensemble_output exists and images are present."
        )

    train_samples, val_samples = train_val_split(
        all_samples, val_fraction=val_fraction, seed=seed
    )

    print(f"  Dataset: {len(all_samples)} total, "
          f"{len(train_samples)} train, {len(val_samples)} val")

    train_tf = build_train_transform(input_size)
    val_tf = build_val_transform(input_size)

    train_ds = SegmentationDataset(train_samples, transform=train_tf, input_size=input_size)
    val_ds = SegmentationDataset(val_samples, transform=val_tf, input_size=input_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, (len(train_samples), len(val_samples))
