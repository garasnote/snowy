#!/usr/bin/env python3
"""
Optuna HP search for tiny QA CNN.
Searches: architecture (channels, depth), lr, dropout, input size, augmentation.
Uses 3-fold CV AUROC as objective (faster than 5-fold, enough for HP search).
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
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import optuna

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class QADataset(Dataset):
    def __init__(self, img_paths, labels, input_size=128, augment=False,
                 color_jitter=0.3, blur_p=0.3):
        self.img_paths = img_paths
        self.labels = labels
        self.input_size = input_size
        self.augment = augment
        self.color_jitter = color_jitter
        self.blur_p = blur_p

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")

        if self.augment:
            import torchvision.transforms.v2 as T
            transforms = [
                T.Resize((self.input_size + 16, self.input_size + 16)),
                T.RandomCrop(self.input_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=self.color_jitter, contrast=self.color_jitter,
                              saturation=self.color_jitter * 0.7, hue=0.05),
                T.RandomGrayscale(p=0.1),
            ]
            if self.blur_p > 0:
                transforms.append(T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))],
                                                p=self.blur_p))
            transforms.extend([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            transform = T.Compose(transforms)
        else:
            import torchvision.transforms.v2 as T
            transform = T.Compose([
                T.Resize((self.input_size, self.input_size)),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        tensor = transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


class TinyQACNN(nn.Module):
    def __init__(self, channels=(32, 64, 128), dropout=0.3, use_se=False):
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
            if use_se:
                layers.append(SEBlock(ch))
            in_ch = ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(-1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total += len(y)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.5
    return auroc, all_probs, all_labels


def objective(trial, img_paths, labels, device, n_folds=3):
    # Architecture
    n_blocks = trial.suggest_int("n_blocks", 2, 4)
    base_ch = trial.suggest_categorical("base_channels", [16, 32, 48])
    ch_mult = trial.suggest_categorical("ch_multiplier", [1.5, 2.0, 2.5])
    channels = tuple(int(base_ch * ch_mult ** i) for i in range(n_blocks))
    use_se = trial.suggest_categorical("use_se", [True, False])

    # Training
    input_size = trial.suggest_categorical("input_size", [96, 128, 160])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Augmentation
    color_jitter = trial.suggest_float("color_jitter", 0.1, 0.5)
    blur_p = trial.suggest_float("blur_p", 0.0, 0.5)

    epochs = 40
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aurocs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(img_paths, labels)):
        train_paths = [img_paths[i] for i in train_idx]
        val_paths = [img_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)

        train_ds = QADataset(train_paths, train_labels, input_size, augment=True,
                             color_jitter=color_jitter, blur_p=blur_p)
        val_ds = QADataset(val_paths, val_labels, input_size, augment=False)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=False)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=False)

        model = TinyQACNN(channels=channels, dropout=dropout, use_se=use_se).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auroc = 0
        patience_count = 0

        for epoch in range(epochs):
            train_one_epoch(model, train_dl, optimizer, criterion, device)
            scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                auroc, _, _ = evaluate(model, val_dl, device)
                if auroc > best_auroc:
                    best_auroc = auroc
                    patience_count = 0
                else:
                    patience_count += 1
                if patience_count >= 3:
                    break

        fold_aurocs.append(best_auroc)

        # Prune bad trials early
        trial.report(np.mean(fold_aurocs), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    mean_auroc = np.mean(fold_aurocs)
    n_params = sum(p.numel() for p in model.parameters())
    trial.set_user_attr("n_params", n_params)
    trial.set_user_attr("channels", str(channels))
    trial.set_user_attr("fold_aurocs", str(fold_aurocs))

    return mean_auroc


def train_best(best_params, img_paths, labels, device, output_dir):
    """Train final model with best HP on all data, then 5-fold CV for reporting."""
    n_blocks = best_params["n_blocks"]
    base_ch = best_params["base_channels"]
    ch_mult = best_params["ch_multiplier"]
    channels = tuple(int(base_ch * ch_mult ** i) for i in range(n_blocks))
    use_se = best_params["use_se"]
    input_size = best_params["input_size"]
    lr = best_params["lr"]
    dropout = best_params["dropout"]
    weight_decay = best_params["weight_decay"]
    batch_size = best_params["batch_size"]
    color_jitter = best_params["color_jitter"]
    blur_p = best_params["blur_p"]

    # 5-fold CV for final metrics
    print(f"\n{'='*60}")
    print("BEST MODEL — 5-FOLD CV")
    print(f"{'='*60}")
    print(f"  Channels: {channels}, SE: {use_se}, Input: {input_size}")
    print(f"  lr={lr:.4f}, dropout={dropout:.2f}, wd={weight_decay:.5f}, bs={batch_size}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = np.zeros(len(labels))

    for fold, (train_idx, val_idx) in enumerate(cv.split(img_paths, labels)):
        train_paths = [img_paths[i] for i in train_idx]
        val_paths = [img_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)

        train_ds = QADataset(train_paths, train_labels, input_size, augment=True,
                             color_jitter=color_jitter, blur_p=blur_p)
        val_ds = QADataset(val_paths, val_labels, input_size, augment=False)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=False)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=False)

        model = TinyQACNN(channels=channels, dropout=dropout, use_se=use_se).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(60):
            train_one_epoch(model, train_dl, optimizer, criterion, device)
            scheduler.step()

        auroc, probs, _ = evaluate(model, val_dl, device)
        all_probs[val_idx] = probs
        print(f"  Fold {fold+1}: AUROC={auroc:.4f}")

    preds = (all_probs >= 0.5).astype(int)
    print(f"\n{classification_report(labels, preds, target_names=['Non-Usable', 'Usable'])}")
    auroc = roc_auc_score(labels, all_probs)
    f1 = f1_score(labels, preds)
    print(f"AUROC: {auroc:.4f}, F1: {f1:.4f}")

    # Train final on all data
    print("\nTraining final model on all data...")
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)

    train_ds = QADataset(img_paths, labels, input_size, augment=True,
                         color_jitter=color_jitter, blur_p=blur_p)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=False)

    model = TinyQACNN(channels=channels, dropout=dropout, use_se=use_se).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(60):
        loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={loss:.4f}")

    # Save TorchScript
    model.eval()
    model.cpu()
    scripted = torch.jit.script(model)
    save_path = output_dir / "qa_cnn.pt"
    scripted.save(str(save_path))
    print(f"\nTorchScript saved to {save_path}")

    # Save checkpoint
    ckpt_path = output_dir / "qa_cnn.ckpt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": input_size,
        "channels": channels,
        "use_se": use_se,
        "dropout": dropout,
        "auroc_cv": auroc,
        "f1_cv": f1,
        "n_params": sum(p.numel() for p in model.parameters()),
        "best_params": best_params,
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params*4/1024:.1f} KB)")

    # Benchmark
    model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    for _ in range(10):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(200):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"Inference: {ms:.2f} ms/image on {device} ({1000/ms:.0f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Optuna HP search for QA CNN")
    parser.add_argument("--features-csv", type=str,
                        default=str(_HERE / "checkpoints" / "qa_features.csv"))
    parser.add_argument("--img-dir", type=str,
                        default=str(_ROOT / "data" / "balanced_snow"))
    parser.add_argument("--output-dir", type=str,
                        default=str(_HERE / "checkpoints"))
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--db", type=str, default=str(_HERE / "optuna_qa_cnn.db"))
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)

    df = pd.read_csv(args.features_csv)
    print(f"Samples: {len(df)} ({(df['label']==1).sum()} usable, {(df['label']==0).sum()} non-usable)")

    img_paths = []
    labels = []
    for _, row in df.iterrows():
        p = img_dir / row["camera"] / f"{row['stem']}.jpg"
        if p.exists():
            img_paths.append(str(p))
            labels.append(row["label"])
    labels = np.array(labels)
    print(f"Found: {len(labels)} images")

    study = optuna.create_study(
        study_name="qa_cnn",
        direction="maximize",
        storage=f"sqlite:///{args.db}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )

    study.optimize(
        lambda trial: objective(trial, img_paths, labels, device),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print(f"\n{'='*60}")
    print("OPTUNA RESULTS")
    print(f"{'='*60}")
    print(f"Best AUROC: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k:<20s} {v}")
    if "n_params" in study.best_trial.user_attrs:
        print(f"  {'n_params':<20s} {study.best_trial.user_attrs['n_params']:,}")

    # Save results CSV
    results_df = study.trials_dataframe()
    results_path = output_dir / "optuna_qa_cnn_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Train best model
    train_best(study.best_params, img_paths, labels, device, output_dir)


if __name__ == "__main__":
    main()
