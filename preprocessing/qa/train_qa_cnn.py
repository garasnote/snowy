#!/usr/bin/env python3
"""
Train a tiny CNN for QA image classification.
Input: 128x128 RGB → 3-class: 0=clean_no_snow, 1=snow, 2=non_usable.
~inference ~5ms on CPU, ~1ms on GPU.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class QADataset(Dataset):
    def __init__(self, img_paths, labels, input_size=128, augment=False):
        self.img_paths = img_paths
        self.labels = labels
        self.input_size = input_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")

        if self.augment:
            import torchvision.transforms.v2 as T
            transform = T.Compose([
                T.Resize((self.input_size + 16, self.input_size + 16)),
                T.RandomCrop(self.input_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.RandomGrayscale(p=0.1),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            import torchvision.transforms.v2 as T
            transform = T.Compose([
                T.Resize((self.input_size, self.input_size)),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        tensor = transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


NUM_CLASSES = 3
CLASS_NAMES = ["clean_no_snow", "snow", "non_usable"]


class TinyQACNN(nn.Module):
    """
    3 conv blocks → global avg pool → classifier.
    Input: 128x128x3 → output: 3 logits (clean_no_snow / snow / non_usable).
    """
    def __init__(self, channels=(32, 64, 128), dropout=0.3):
        super().__init__()
        layers = []
        in_ch = 3
        for ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_ch = ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(y.cpu().numpy())
    all_probs = np.vstack(all_probs)   # (N, 3)
    all_labels = np.array(all_labels)
    preds = all_probs.argmax(axis=1)
    acc = np.mean(preds == all_labels)
    try:
        auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.0
    return total_loss / len(all_labels), acc, auroc, all_probs, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train tiny QA CNN")
    parser.add_argument("--features-csv", type=str,
                        default=str(_HERE / "checkpoints" / "qa_features.csv"))
    parser.add_argument("--img-dir", type=str,
                        default=str(_ROOT / "data" / "balanced_snow"))
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE / "checkpoints"))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--channels", type=str, default="32,64,128")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    channels = tuple(int(c) for c in args.channels.split(","))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)

    # Load labels — use label3 column (0=clean_no_snow, 1=snow, 2=non_usable)
    df = pd.read_csv(args.features_csv)
    if "label3" not in df.columns:
        raise ValueError("CSV missing 'label3' column — re-run train_qa.py to regenerate qa_features.csv")
    counts = {c: int((df["label3"] == c).sum()) for c in range(NUM_CLASSES)}
    print(f"Samples: {len(df)}  " + "  ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in range(NUM_CLASSES)))

    # Resolve image paths
    img_paths = []
    labels = []
    missing = 0
    for _, row in df.iterrows():
        p = img_dir / row["camera"] / f"{row['stem']}.jpg"
        if p.exists():
            img_paths.append(str(p))
            labels.append(int(row["label3"]))
        else:
            missing += 1
    if missing:
        print(f"  Warning: {missing} images not found")
    labels = np.array(labels, dtype=np.int64)
    print(f"  Found: {len(labels)} images")

    def _class_weights(lbl_array):
        """Inverse-frequency weights per class, normalised to mean=1."""
        counts = np.bincount(lbl_array, minlength=NUM_CLASSES).astype(float)
        counts = np.maximum(counts, 1)
        w = len(lbl_array) / (NUM_CLASSES * counts)
        return torch.tensor(w, dtype=torch.float32)

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = np.zeros((len(labels), NUM_CLASSES))
    fold_aurocs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(img_paths, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*60}")

        train_paths = [img_paths[i] for i in train_idx]
        val_paths = [img_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        class_w = _class_weights(train_labels).to(device)

        train_ds = QADataset(train_paths, train_labels, args.input_size, augment=True)
        val_ds = QADataset(val_paths, val_labels, args.input_size, augment=False)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        model = TinyQACNN(channels=channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(weight=class_w)

        best_auroc = 0
        patience = 10
        wait = 0

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion, device)
            val_loss, val_acc, val_auroc, val_probs, val_true = evaluate(model, val_dl, criterion, device)
            scheduler.step()

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_probs = val_probs.copy()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} │ train loss={train_loss:.4f} acc={train_acc:.3f} │ "
                      f"val loss={val_loss:.4f} acc={val_acc:.3f} auroc={val_auroc:.4f}")

            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        all_probs[val_idx] = best_probs
        fold_aurocs.append(best_auroc)
        print(f"  Best val AUROC: {best_auroc:.4f}")

    # CV results
    all_preds = all_probs.argmax(axis=1)
    print(f"\n{'='*60}")
    print("5-FOLD CV RESULTS")
    print(f"{'='*60}")
    print(classification_report(labels, all_preds, target_names=CLASS_NAMES))
    auroc = roc_auc_score(labels, all_probs, multi_class="ovr", average="macro")
    f1 = f1_score(labels, all_preds, average="macro")
    print(f"AUROC (macro OvR): {auroc:.4f}")
    print(f"F1 (macro):        {f1:.4f}")
    print(f"Fold AUROCs: {[f'{a:.4f}' for a in fold_aurocs]}")
    print(f"Mean AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

    # Train final model on all data
    print(f"\nTraining final model on all data...")
    class_w = _class_weights(labels).to(device)

    train_ds = QADataset(img_paths, labels, args.input_size, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)

    model = TinyQACNN(channels=channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion, device)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} │ loss={train_loss:.4f} acc={train_acc:.3f}")

    # Save as TorchScript for easy loading
    model.eval()
    model.cpu()
    scripted = torch.jit.script(model)
    save_path = output_dir / "qa_cnn.pt"
    scripted.save(str(save_path))
    print(f"\nModel saved to {save_path}")

    # Also save as regular checkpoint with metadata
    ckpt_path = output_dir / "qa_cnn.ckpt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": args.input_size,
        "channels": channels,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "auroc_cv": auroc,
        "f1_cv": f1,
        "n_params": sum(p.numel() for p in model.parameters()),
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel size: {n_params:,} params ({n_params*4/1024:.1f} KB)")

    # Benchmark inference speed
    print("\nBenchmarking inference speed...")
    model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, args.input_size, args.input_size).to(device)
    # Warmup
    for _ in range(10):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    n_runs = 100
    for _ in range(n_runs):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_runs * 1000
    print(f"  {device}: {elapsed:.1f} ms/image ({1000/elapsed:.0f} FPS)")


if __name__ == "__main__":
    main()
