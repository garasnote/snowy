#!/usr/bin/env python3
"""Per-camera OOD breakdown: ground IoU, snow IoU, QA AUROC/F1, for each of
the 4 OOD cameras individually (not merged into one pooled n=495 like the
other eval_all.py OOD blocks do). Backs the paper's claim that CM00067 is
consistently the hardest camera in the cohort — that claim currently has no
supporting per-camera table.

Uses the same checkpoints, GT loading (load_gt_masks with label_class="Ground
Mano"/"Snow Mano"), and QA taxonomy parsing already verified this session.
"""
import sys
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "models"))

from eval_all import load_gt_masks, Evaluator, img_to_tensor, IMAGENET_MEAN, IMAGENET_STD

OOD_CAMERAS = ["CM00054", "CM00067", "CM00071", "CM00115"]
OOD_JSON_MAP = {
    "CM00054": "data/annotations/54.json",
    "CM00067": "data/annotations/67.json",
    "CM00071": "data/annotations/71.json",
    "CM00115": "data/annotations/115.json",
}

GROUND_CKPT = "models/student_ground/checkpoints/finetune_mobilenet_kfold_1_20260610_230732/best-epoch=48-iouval_iou=0.916.ckpt"
SNOW_CKPT = "models/student_snow/checkpoints/snow_gt_mobilenet_20260522_052052/best-epoch=65-iouval_iou=0.821.ckpt"

DATA_DIR = Path("data")


def eval_seg_per_camera(task, ckpt_path, label_class, device):
    from student_ground.models import build_model
    from student_ground.configs import get_config
    from student_ground.train import StudentSegmentationModule

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    variant = checkpoint.get("hyper_parameters", {}).get("model_variant", "mobilenet")
    model = build_model(variant)
    sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    cfg = get_config(variant)
    input_size = cfg.get("input_size", 512)

    results = {}
    for cam, fname in OOD_JSON_MAP.items():
        p = Path(fname)
        if not p.exists():
            results[cam] = None
            continue
        gt_masks = load_gt_masks(p, [cam], label_class=label_class)
        evaluator = Evaluator(device)
        count = 0
        for (c, stem), target in gt_masks.items():
            img_path = DATA_DIR / c / f"{stem}.jpg"
            if not img_path.exists():
                img_path = DATA_DIR / f"_EXCLUDED_{c}" / f"{stem}.jpg"
            if not img_path.exists():
                continue
            img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
            img_t = img_to_tensor(img).to(device)
            with torch.no_grad():
                logits = model(img_t)
                pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]
            target_resized = np.array(
                Image.fromarray(target).resize((input_size, input_size), Image.NEAREST)
            )
            evaluator.update(pred.astype(np.float32), target_resized)
            count += 1
        results[cam] = (evaluator.compute(), count) if count > 0 else None
    return results


def parse_qa_label3(task):
    anns = task.get("annotations", [])
    if not anns or not anns[0].get("result"):
        return None
    flat_all = set()
    for r in anns[0]["result"]:
        v = r.get("value", {})
        for key in ("taxonomy", "choices"):
            tags = v.get(key)
            if not tags:
                continue
            for entry in tags:
                if isinstance(entry, list):
                    flat_all.update(entry)
                else:
                    flat_all.add(entry)
    if "Usable" in flat_all:
        return 1
    if "Non-Usable" in flat_all:
        reasons = flat_all - {"Non-Usable"}
        if reasons <= {"No Snow"}:
            return 0
        return 2
    return None


def eval_qa_per_camera(device):
    import torchvision.transforms.v2 as T

    cnn_ckpt = _ROOT / "preprocessing" / "qa" / "checkpoints" / "qa_cnn.pt"
    model = torch.jit.load(str(cnn_ckpt), map_location=device).eval()
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    from sklearn.metrics import roc_auc_score, f1_score

    results = {}
    for cam, fname in OOD_JSON_MAP.items():
        p = Path(fname)
        if not p.exists():
            results[cam] = None
            continue
        with open(p) as f:
            tasks = json.load(f)
        img_paths, labels = [], []
        for task in tasks:
            label = parse_qa_label3(task)
            if label is None:
                continue
            url = task["data"]["image"]
            stem = Path(url.rstrip("/").split("/")[-1]).stem
            for base in [DATA_DIR / cam, DATA_DIR / f"_EXCLUDED_{cam}"]:
                for ext in ("jpg", "png"):
                    ip = base / f"{stem}.{ext}"
                    if ip.exists():
                        img_paths.append(ip)
                        labels.append(label)
                        break
        labels = np.array(labels, dtype=np.int64)
        if len(labels) == 0 or len(set(labels.tolist())) < 2:
            results[cam] = None
            continue
        probs = []
        with torch.no_grad():
            for i in range(0, len(img_paths), 32):
                batch = [transform(Image.open(p).convert("RGB")) for p in img_paths[i:i + 32]]
                x = torch.stack(batch).to(device)
                probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
        probs = np.vstack(probs)
        preds = probs.argmax(axis=1)
        try:
            auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except ValueError:
            auroc = float("nan")
        f1 = f1_score(labels, preds, average="macro")
        results[cam] = (auroc, f1, len(labels))
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Ground (per-camera) ===")
    ground_results = eval_seg_per_camera("ground", GROUND_CKPT, "Ground Mano", device)
    for cam in OOD_CAMERAS:
        r = ground_results[cam]
        if r is None:
            print(f"  {cam}: no data")
        else:
            m, n = r
            print(f"  {cam}: IoU={m['iou']:.4f} Dice={m['dice']:.4f} n={n}")

    print("\n=== Snow (per-camera) ===")
    snow_results = eval_seg_per_camera("snow", SNOW_CKPT, "Snow Mano", device)
    for cam in OOD_CAMERAS:
        r = snow_results[cam]
        if r is None:
            print(f"  {cam}: no data")
        else:
            m, n = r
            print(f"  {cam}: IoU={m['iou']:.4f} Dice={m['dice']:.4f} n={n}")

    print("\n=== QA (per-camera) ===")
    qa_results = eval_qa_per_camera(device)
    for cam in OOD_CAMERAS:
        r = qa_results[cam]
        if r is None:
            print(f"  {cam}: insufficient class balance")
        else:
            auroc, f1, n = r
            print(f"  {cam}: AUROC={auroc:.4f} F1={f1:.4f} n={n}")


if __name__ == "__main__":
    main()
