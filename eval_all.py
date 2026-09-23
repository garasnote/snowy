#!/usr/bin/env python3
"""
Unified evaluation script for all ground/snow segmentation methods.

Evaluates:
- Ground: teachers, ensembles, baselines, students (when ready)
- Snow: VLMs, baselines, students (when ready)

Usage:
    python eval_all.py --task ground --types teachers ensembles baselines
    python eval_all.py --task snow --types vlms baselines students
    python eval_all.py --task ground  # eval all types
    python eval_all.py --task snow
"""

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WORKSPACE_DIR

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from torchmetrics import JaccardIndex
from torchmetrics.classification import BinaryF1Score

sys.path.insert(0, str(Path(__file__).parent / "models"))
from student_ground.dataset import decode_ls_rle

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def img_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB → 1CHW float32 ImageNet-normalized tensor."""
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


class Evaluator:
    def __init__(self, device="cuda"):
        self.device = device
        self.iou = JaccardIndex(task="binary").to(device)
        self.dice = BinaryF1Score().to(device)

    def reset(self):
        self.iou.reset()
        self.dice.reset()

    def update(self, pred, target):
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float().to(self.device)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float().to(self.device)

        if pred.shape != target.shape:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=target.shape,
                mode="nearest"
            ).squeeze(0).squeeze(0)

        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

        self.iou.update(pred, target)
        self.dice.update(pred, target)

    def compute(self):
        return {
            "iou": self.iou.compute().item(),
            "dice": self.dice.compute().item(),
        }


def load_gt_masks(gt_json_path, cameras, label_class=None):
    """Load GT masks from JSON.

    label_class: if set, only merge brushlabels whose 'brushlabels' list contains
                 this value (e.g. "Ground Mano" or "Snow Mano").  Pass None (default)
                 to merge all brushlabels — backward-compatible with old single-task files.
    """
    gt_masks = {}

    with open(gt_json_path) as f:
        tasks = json.load(f)

    for task in tasks:
        if "data" not in task or "image" not in task["data"]:
            continue

        image_url = task["data"]["image"]

        # Parse camera + filename (handle both URL formats)
        if "/data/upload/" in image_url:
            filename_raw = image_url.split("/")[-1]
            camera = None
            import re
            match = re.search(r'(CM|CF)\d{5}', filename_raw)
            if match:
                camera = match.group(0)
                idx = filename_raw.find(camera)
                filename = filename_raw[idx:]
            else:
                continue
            if not camera:
                continue
        else:
            parts = image_url.split("/")
            camera = parts[-2]
            filename = parts[-1]

        stem = filename.replace(".jpg", "")

        if camera not in cameras:
            continue

        if "annotations" not in task or len(task["annotations"]) == 0:
            continue

        annotations = task["annotations"][0]
        if "result" not in annotations or len(annotations["result"]) == 0:
            continue

        # Find dimensions from first brushlabels entry (taxonomy entries lack these)
        w, h = 1296, 960
        for _r in annotations["result"]:
            if _r.get("original_width"):
                w = _r["original_width"]
                h = _r["original_height"]
                break

        merged = np.zeros((h, w), dtype=bool)
        has_brush = False
        is_no_snow = False
        is_non_usable = False

        all_tags = set()
        for result in annotations["result"]:
            v = result.get("value", {})
            # Check Non-Usable / No Snow taxonomy tags
            for key in ("taxonomy", "choices"):
                tags = v.get(key, [])
                flat = set()
                for entry in tags:
                    if isinstance(entry, list):
                        flat.update(entry)
                    else:
                        flat.add(entry)
                all_tags |= flat
                if "Non-Usable" in flat:
                    is_non_usable = True
                if "No Snow" in flat:
                    is_no_snow = True

            # Brushlabels: filter by label_class when specified
            if result.get("type") == "brushlabels" and "rle" in v:
                bl = v.get("brushlabels", [])
                if label_class is None or label_class in bl:
                    merged |= decode_ls_rle(v["rle"], w, h)
                    has_brush = True

        if is_non_usable:
            # Taxonomy design bug (see Sec. "Annotation protocol"): "No Snow" was
            # implemented as a child of "Non-Usable", so a snow-free frame with no
            # other quality issue gets tagged {"Non-Usable", "No Snow"} even though
            # it's a perfectly valid negative example. Recover it — same rule
            # already applied to the in-distribution annotation counts and the QA
            # taxonomy parser (_parse_qa_label3) — rather than genuinely-excluded
            # frames (Blurry / Not in Frame / Bad Light, alone or combined).
            other_reasons = all_tags - {"Non-Usable", "No Snow"}
            if not (is_no_snow and not other_reasons):
                continue

        if has_brush:
            gt_masks[(camera, stem)] = merged.astype(np.uint8)
        elif is_no_snow:
            gt_masks[(camera, stem)] = np.zeros((h, w), dtype=np.uint8)

    return gt_masks


def load_gt_masks_all(cameras, label_class=None):
    """Load GT masks from all annotation files."""
    merged = {}
    fnames = [
        "ground_annotations.json", "san.json", "heroku.json",
        "54.json", "67.json", "71.json", "115.json",
    ]
    for fname in fnames:
        p = Path("data/annotations") / fname
        if p.exists():
            merged.update(load_gt_masks(p, cameras, label_class=label_class))
    return merged


def eval_multihead_task(gt_masks, task, args, ckpt_path=None, label=None):
    """Evaluate a DINOv3MultiHead checkpoint on one task (ground/snow).

    ckpt_path: a pseudo_phase1.pt (dict w/ 'ground_decoder'/'snow_decoder'
    state dicts only, backbone stays pretrained-frozen — Phase 1 output) or
    a full Lightning .ckpt (Phase 2 GT-finetuned). Both load fine since the
    decoder state dict keys are the same shape either way.
    Returns a metrics dict, or None if nothing could be evaluated.
    """
    from multi_head.train_multihead import DINOv3MultiHead
    from student_ground.configs import MODEL_REGISTRY

    if ckpt_path is None:
        ckpt_path = Path("models/multi_head/checkpoints/dinov3_base/pseudo_phase1.pt")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"  multihead ({task}): checkpoint not found: {ckpt_path}")
        return None

    variant = "dinov3_base"
    model = DINOv3MultiHead(variant).to(args.device)
    model.eval()

    state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    if "ground_decoder" in state and "snow_decoder" in state:
        # Phase-1-only pseudolabel checkpoint: just the two decoder heads.
        model.ground_decoder.load_state_dict(state["ground_decoder"])
        model.snow_decoder.load_state_dict(state["snow_decoder"])
    else:
        # Full Lightning checkpoint (Phase 2 GT finetune).
        sd = {k.replace("model.", ""): v for k, v in state["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(sd, strict=False)

    input_size = MODEL_REGISTRY[variant]["input_size"]
    data_dir = Path("data")

    evaluator = Evaluator(args.device)
    count = 0
    label = label or f"multihead_{task}"

    for (camera, stem), target in tqdm(gt_masks.items(), desc=f"  {label}", leave=False):
        img_path = data_dir / camera / f"{stem}.jpg"
        if not img_path.exists():
            img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
        if not img_path.exists():
            continue

        img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
        img_t = img_to_tensor(img)

        with torch.no_grad():
            logits = model.forward_seg(img_t.to(args.device), task=task)
            pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]

        target_resized = np.array(
            Image.fromarray(target).resize((input_size, input_size), Image.NEAREST)
        )
        evaluator.update(pred.astype(np.float32), target_resized)
        count += 1

    if count == 0:
        print(f"  multihead ({task}): no images matched GT masks")
        return None

    metrics = evaluator.compute()
    print(f"  {label}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}  (ckpt: {ckpt_path.name}, n={count})")
    return metrics


def eval_ground(args):
    """Evaluate ground segmentation."""
    print("\n" + "=" * 100)
    print("GROUND SEGMENTATION EVALUATION")
    print("=" * 100)

    cameras = ["CF00001", "CM00009", "CM00014", "CM00070", "CM00075", "CM00080", "CM00102", "CM00108", "CM00112"]
    output_dir = Path("preprocessing/segmentation")
    workspace_dir = WORKSPACE_DIR
    data_dir = Path("data")

    gt_masks = {}
    for gt_json in [
        Path("data/annotations/ground_annotations.json"),
        Path("data/annotations/san.json"),
        Path("data/annotations/heroku.json"),
    ]:
        if gt_json.exists():
            gt_masks.update(load_gt_masks(gt_json, cameras))
    print(f"\nLoaded {len(gt_masks)} GT masks (ground_annotations + san + heroku)")

    results = {}

    # Teachers
    if "teachers" in args.types:
        print("\n--- Teachers ---")
        teachers = {
            "segformer": output_dir / "segformer_output",
            "deeplabv3": output_dir / "deeplabv3_output",
            "depth": workspace_dir / "depth_output" / "N0.05_F0.95",
            "sam3_text": output_dir / "sam3_dino_output" / "text",
            "sam3_dino": output_dir / "sam3_dino_output" / "dino+sam",
            "dinov2_pca": output_dir / "dinov2_pca_output",
        }

        for method, method_dir in teachers.items():
            if not method_dir.exists():
                continue

            evaluator = Evaluator(args.device)
            count = 0

            for (camera, stem), target in tqdm(gt_masks.items(), desc=f"  {method}", leave=False):
                pred_path = method_dir / camera / f"{stem}_ground.png"
                if not pred_path.exists():
                    continue

                pred = np.array(Image.open(pred_path).convert("L")) > 127
                evaluator.update(pred, target)
                count += 1

            if count > 0:
                metrics = evaluator.compute()
                results[method] = metrics
                print(f"  {method}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

    # Ensembles
    if "ensembles" in args.types:
        print("\n--- Ensembles ---")
        for threshold in [2, 3, 4, 5]:
            ensemble_dir = output_dir / "ensemble_output" / f"majority_{threshold}"
            if not ensemble_dir.exists():
                continue

            evaluator = Evaluator(args.device)
            count = 0

            for (camera, stem), target in tqdm(gt_masks.items(), desc=f"  maj_{threshold}", leave=False):
                pred_path = ensemble_dir / camera / f"{stem}_ground.png"
                if not pred_path.exists():
                    continue

                pred = np.array(Image.open(pred_path).convert("L")) > 127
                evaluator.update(pred, target)
                count += 1

            if count > 0:
                metrics = evaluator.compute()
                results[f"majority_{threshold}"] = metrics
                print(f"  majority_{threshold}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

        for threshold in [3, 4, 5]:
            ensemble_dir = output_dir / "ensemble_output" / f"weighted_{threshold}"
            if not ensemble_dir.exists():
                continue

            evaluator = Evaluator(args.device)
            count = 0

            for (camera, stem), target in tqdm(gt_masks.items(), desc=f"  wtd_{threshold}", leave=False):
                pred_path = ensemble_dir / camera / f"{stem}_ground.png"
                if not pred_path.exists():
                    continue

                pred = np.array(Image.open(pred_path).convert("L")) > 127
                evaluator.update(pred, target)
                count += 1

            if count > 0:
                metrics = evaluator.compute()
                results[f"weighted_{threshold}"] = metrics
                print(f"  weighted_{threshold}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

    # Baselines
    if "baselines" in args.types:
        print("\n--- Baselines ---")

        # Fixed bottom
        evaluator = Evaluator(args.device)
        for (camera, stem), target in gt_masks.items():
            h, w = target.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(h * 0.68):] = 1
            evaluator.update(mask, target)

        metrics = evaluator.compute()
        results["baseline_fixed_32pct"] = metrics
        print(f"  Fixed bottom 32%: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

        # Sobel horizon
        evaluator = Evaluator(args.device)
        count = 0

        for (camera, stem), target in tqdm(gt_masks.items(), desc="  Sobel", leave=False):
            img_path = data_dir / camera / f"{stem}.jpg"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape
            sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            edge_strength = np.abs(sobel_y).mean(axis=1)
            edge_smooth = cv2.GaussianBlur(edge_strength.astype(np.float32).reshape(-1, 1), (11, 1), 5).flatten()

            central_start = int(h * 0.1)
            central_end = int(h * 0.9)
            horizon_row = central_start + np.argmax(edge_smooth[central_start:central_end])

            mask = np.zeros((h, w), dtype=np.uint8)
            mask[horizon_row:] = 1

            evaluator.update(mask, target)
            count += 1

        if count > 0:
            metrics = evaluator.compute()
            results["baseline_sobel"] = metrics
            print(f"  Sobel horizon: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

    # Baselines OOD (new OOD cohort: CM00054, CM00071, CM00115)
    if "baselines_ood" in args.types:
        print("\n--- Baselines OOD (CM00054, CM00071, CM00115) ---")
        ood_cameras = ["CM00054", "CM00067", "CM00071", "CM00115"]
        gt_masks_ood = load_gt_masks_all(ood_cameras, label_class="Ground Mano")
        print(f"  Loaded {len(gt_masks_ood)} OOD GT masks")

        # Fixed bottom
        evaluator = Evaluator(args.device)
        for (camera, stem), target in gt_masks_ood.items():
            h, w = target.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(h * 0.68):] = 1
            evaluator.update(mask, target)
        metrics = evaluator.compute()
        results["ood_baseline_fixed_32pct"] = metrics
        print(f"  Fixed bottom 32%: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

        # Sobel horizon
        evaluator = Evaluator(args.device)
        count = 0
        for (camera, stem), target in tqdm(gt_masks_ood.items(), desc="  Sobel OOD", leave=False):
            img_path = data_dir / camera / f"{stem}.jpg"
            if not img_path.exists():
                img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            edge_strength = np.abs(sobel_y).mean(axis=1)
            edge_smooth = cv2.GaussianBlur(edge_strength.astype(np.float32).reshape(-1, 1), (11, 1), 5).flatten()
            central_start = int(h * 0.1)
            central_end = int(h * 0.9)
            horizon_row = central_start + np.argmax(edge_smooth[central_start:central_end])
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[horizon_row:] = 1
            evaluator.update(mask, target)
            count += 1
        if count > 0:
            metrics = evaluator.compute()
            results["ood_baseline_sobel"] = metrics
            print(f"  Sobel horizon: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

    # Students
    if "students" in args.types:
        print("\n--- Student Models ---")

        ground_student_baselines = {
            "mobilenet": sorted(Path("models/student_ground/checkpoints").glob("mobilenet_majority_3_*/best-*.ckpt")),
            "efficientnet": sorted(Path("models/student_ground/checkpoints").glob("efficientnet_majority_3_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            "dinov2_base": sorted(Path("models/student_ground/checkpoints").glob("dinov2_base_majority_3_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            "dinov2_large": sorted(Path("models/student_ground/checkpoints").glob("dinov2_large_majority_3_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            "dinov3_base": sorted(Path("models/student_ground/checkpoints").glob("dinov3_base_majority_3_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
        }

        for model_name, student_ckpts in ground_student_baselines.items():
            if not student_ckpts:
                print(f"  {model_name}: no checkpoint found")
                continue
            try:
                from student_ground.models import build_model
                ckpt_path = student_ckpts[-1]
                checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                variant = checkpoint.get("hyper_parameters", {}).get("model_variant", model_name)
                model = build_model(variant)
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
                model.load_state_dict(state_dict, strict=False)
                model = model.to(args.device)
                model.eval()

                evaluator = Evaluator(args.device)
                count = 0

                evaluator = Evaluator(args.device)
                count = 0

                from student_ground.configs import get_config
                cfg = get_config(variant)
                input_size = cfg.get("input_size", 512)

                for (camera, stem), target in tqdm(gt_masks.items(), desc=f"  {model_name}", leave=False):
                    img_path = data_dir / camera / f"{stem}.jpg"
                    if not img_path.exists():
                        continue

                    img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
                    img_t = img_to_tensor(img)

                    with torch.no_grad():
                        logits = model(img_t.to(args.device))
                        pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]

                    evaluator.update(pred, target)
                    count += 1

                if count > 0:
                    metrics = evaluator.compute()
                    results[f"{model_name}_majority_3"] = metrics
                    ckpt_name = ckpt_path.name.split("iou=")[1].rstrip(".ckpt")
                    print(f"  {model_name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f} (val: {ckpt_name})")
            except Exception as e:
                import traceback
                print(f"  {model_name} eval failed: {e}")
                traceback.print_exc()

    # OOD (out-of-distribution) — held-out cameras, best kfold per model
    if "students_ood" in args.types:
        print("\n--- Student Models OOD (pseudo-label maj3, CM00054/CM00071/CM00115) ---")
        ood_cameras = ["CM00054", "CM00067", "CM00071", "CM00115"]
        ood_masks = load_gt_masks_all(ood_cameras, label_class="Ground Mano")
        print(f"  Loaded {len(ood_masks)} OOD GT masks")

        pseudo_ood_ckpts = {
            "mobilenet":    sorted(Path("models/student_ground/checkpoints").glob("mobilenet_majority_3_*/best-*.ckpt")),
            "efficientnet": sorted(Path("models/student_ground/checkpoints").glob("efficientnet_majority_3_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            "dinov2_base":  sorted(Path("models/student_ground/checkpoints").glob("dinov2_base_majority_3_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            "dinov2_large": sorted(Path("models/student_ground/checkpoints").glob("dinov2_large_majority_3_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            "dinov3_base":  sorted(Path("models/student_ground/checkpoints").glob("dinov3_base_majority_3_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
        }

        for model_name, ckpts in pseudo_ood_ckpts.items():
            if not ckpts:
                print(f"  {model_name}: no checkpoint found")
                continue
            try:
                from student_ground.models import build_model
                from student_ground.configs import get_config
                ckpt_path = ckpts[-1]
                checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                variant = checkpoint.get("hyper_parameters", {}).get("model_variant", model_name)
                model = build_model(variant)
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
                model.load_state_dict(state_dict, strict=False)
                model = model.to(args.device).eval()

                cfg = get_config(variant)
                input_size = cfg.get("input_size", 512)

                evaluator = Evaluator(args.device)
                count = 0
                for (camera, stem), target in tqdm(ood_masks.items(), desc=f"  {model_name}", leave=False):
                    img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
                    if not img_path.exists():
                        img_path = data_dir / camera / f"{stem}.jpg"
                    if not img_path.exists():
                        continue
                    img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
                    img_t = img_to_tensor(img)
                    with torch.no_grad():
                        logits = model(img_t.to(args.device))
                        pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]
                    evaluator.update(pred, target)
                    count += 1

                if count > 0:
                    metrics = evaluator.compute()
                    results[f"{model_name}_pseudo_ood"] = metrics
                    print(f"  {model_name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f} (n={count})")
            except Exception as e:
                import traceback
                print(f"  {model_name} failed: {e}")
                traceback.print_exc()

    if "ood" in args.types:
        print("\n--- OOD Evaluation (Held-out Cameras: CM00054, CM00071, CM00115) ---")
        ood_cameras = ["CM00054", "CM00067", "CM00071", "CM00115"]
        ood_masks = load_gt_masks_all(ood_cameras, label_class="Ground Mano")

        if ood_masks:
            print(f"Loaded {len(ood_masks)} OOD GT masks")

            ood_kfold_ckpts = {
                "mobilenet":    sorted(Path("models/student_ground/checkpoints").glob("finetune_mobilenet_kfold_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
                "efficientnet": sorted(Path("models/student_ground/checkpoints").glob("finetune_efficientnet_kfold_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
                "dinov2_base":  sorted(Path("models/student_ground/checkpoints").glob("finetune_dinov2_base_kfold_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
                "dinov2_large": sorted(Path("models/student_ground/checkpoints").glob("finetune_dinov2_large_kfold_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
                "dinov3_base":  sorted(Path("models/student_ground/checkpoints").glob("finetune_dinov3_base_kfold_*/best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            }

            for model_name, ood_ckpts in ood_kfold_ckpts.items():
                if not ood_ckpts:
                    print(f"  {model_name}: no kfold checkpoint")
                    continue
                try:
                    from student_ground.models import build_model
                    from student_ground.configs import get_config
                    ckpt_path = ood_ckpts[-1]
                    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                    variant = checkpoint.get("hyper_parameters", {}).get("model_variant", model_name)
                    model = build_model(variant)
                    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
                    model.load_state_dict(state_dict, strict=False)
                    model = model.to(args.device)
                    model.eval()

                    cfg = get_config(variant)
                    input_size = cfg.get("input_size", 512)

                    evaluator = Evaluator(args.device)
                    count = 0

                    for (camera, stem), target in tqdm(ood_masks.items(), desc=f"  {model_name}_ood", leave=False):
                        img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
                        if not img_path.exists():
                            img_path = data_dir / camera / f"{stem}.jpg"
                        if not img_path.exists():
                            continue

                        img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
                        img_t = img_to_tensor(img)

                        with torch.no_grad():
                            logits = model(img_t.to(args.device))
                            pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]

                        evaluator.update(pred, target)
                        count += 1

                    if count > 0:
                        metrics = evaluator.compute()
                        results[f"{model_name}_kfold_ood"] = metrics
                        ckpt_name = ckpt_path.name.split("iou=")[1].rstrip(".ckpt")
                        print(f"  {model_name}_ood: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f} (val: {ckpt_name})")
                except Exception as e:
                    import traceback
                    print(f"  {model_name} OOD failed: {e}")
                    traceback.print_exc()

    # DINOv3 multihead (Phase 1 pseudolabel decoder, no GT finetune)
    if "multihead" in args.types:
        print("\n--- Multihead (DINOv3MultiHead, Phase 1 pseudolabel) ---")
        try:
            metrics = eval_multihead_task(gt_masks, "ground", args, ckpt_path=args.multihead_ckpt, label="multihead_ground_pseudo")
            if metrics is not None:
                results["multihead_ground_pseudo"] = metrics
        except Exception as e:
            import traceback
            print(f"  multihead ground eval failed: {e}")
            traceback.print_exc()

        print("\n--- Multihead OOD (Held-out Cameras: CM00054, CM00071, CM00115) ---")
        ood_cameras = ["CM00054", "CM00067", "CM00071", "CM00115"]
        ood_masks = load_gt_masks_all(ood_cameras, label_class="Ground Mano")
        print(f"  Loaded {len(ood_masks)} OOD GT masks")
        if ood_masks:
            try:
                metrics = eval_multihead_task(ood_masks, "ground", args, ckpt_path=args.multihead_ckpt, label="multihead_ground_pseudo_ood")
                if metrics is not None:
                    results["multihead_ground_pseudo_ood"] = metrics
            except Exception as e:
                import traceback
                print(f"  multihead ground OOD eval failed: {e}")
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY (sorted by IoU)")
    print("=" * 100)
    print(f"{'Method':<40} {'IoU':<15} {'Dice':<15}")
    print("-" * 100)
    for method, metrics in sorted(results.items(), key=lambda x: x[1]["iou"], reverse=True):
        print(f"{method:<40} {metrics['iou']:<15.4f} {metrics['dice']:<15.4f}")


def eval_snow(args):
    """Evaluate snow segmentation."""
    import cv2
    print("\n" + "=" * 100)
    print("SNOW SEGMENTATION EVALUATION")
    print("=" * 100)

    cameras = ["CF00001", "CM00009", "CM00014", "CM00070", "CM00075", "CM00080", "CM00102", "CM00108", "CM00112"]
    output_dir = Path("preprocessing/segmentation")
    data_dir = Path("data")

    camera_json_map = {
        "CF00001": "data/annotations/01.json",
        "CM00009": "data/annotations/09.json",
        "CM00014": "data/annotations/14.json",
        "CM00070": "data/annotations/70.json",
        "CM00075": "data/annotations/75.json",
        "CM00080": "data/annotations/80.json",
        "CM00102": "data/annotations/102.json",
        "CM00108": "data/annotations/108.json",
        "CM00112": "data/annotations/112.json",
    }
    extra_jsons = [
        "data/annotations/heroku.json",
        "data/annotations/san.json",
    ]

    gt_masks = {}
    for camera, json_path in camera_json_map.items():
        if camera not in cameras or not Path(json_path).exists():
            continue
        masks = load_gt_masks(Path(json_path), [camera])
        gt_masks.update(masks)
    for json_path in extra_jsons:
        if Path(json_path).exists():
            masks = load_gt_masks(Path(json_path), cameras)
            gt_masks.update(masks)

    print(f"\nLoaded {len(gt_masks)} GT snow masks")

    # Optionally restrict to val split only (reproduces training val IoU)
    if getattr(args, "val_only", False):
        import re
        from datetime import date

        def _extract_date(stem):
            m = re.search(r"(\d{4})(\d{2})(\d{2})", stem)
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else date(2000, 1, 1)

        by_cam = {}
        for (cam, stem) in gt_masks:
            by_cam.setdefault(cam, []).append(stem)

        val_set = set()
        val_fraction, gap_days = 0.2, 1
        for cam, stems in sorted(by_cam.items()):
            sorted_s = sorted(stems, key=_extract_date)
            unique_dates = sorted(set(_extract_date(s) for s in sorted_s))
            if len(unique_dates) < 3:
                n_val = max(1, int(len(sorted_s) * val_fraction))
                for s in sorted_s[-n_val:]:
                    val_set.add((cam, s))
                continue
            n_val_days = max(1, int(len(unique_dates) * val_fraction))
            gap = min(gap_days, len(unique_dates) - n_val_days - 1)
            cutoff = unique_dates[-(n_val_days + gap)]
            for s in sorted_s:
                d = _extract_date(s)
                if d >= cutoff and (d - cutoff).days >= gap:
                    val_set.add((cam, s))

        gt_masks = {k: v for k, v in gt_masks.items() if k in val_set}
        print(f"Val-only split: {len(gt_masks)} GT masks")

    results = {}

    # VLMs
    if "vlms" in args.types:
        print("\n--- VLM Methods ---")
        vlms = {
            "florence2": output_dir / "florence2_snow_output",
            "yoloe": output_dir / "yoloe_snow_output",
            "sam3_text": output_dir / "sam3_snow_output",
            "groundedsam2": output_dir / "groundedsam2_snow_output",
            "clipseg": output_dir / "clipseg_snow_output",
            "seggpt": output_dir / "seggpt_snow_output",
            "hsv": output_dir / "hsv_snow_output",
        }

        for method, method_dir in vlms.items():
            if not method_dir.exists():
                continue

            evaluator = Evaluator(args.device)
            count = 0

            for (camera, stem), target in tqdm(gt_masks.items(), desc=f"  {method}", leave=False):
                pred_path = method_dir / camera / f"{stem}_snow.png"
                if not pred_path.exists():
                    continue

                pred = np.array(Image.open(pred_path).convert("L")) > 127
                evaluator.update(pred, target)
                count += 1

            if count > 0:
                metrics = evaluator.compute()
                results[method] = metrics
                print(f"  {method}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

    # Baselines
    if "baselines" in args.types:
        print("\n--- Baselines ---")

        # HSV white
        evaluator = Evaluator(args.device)
        count = 0

        for (camera, stem), target in tqdm(gt_masks.items(), desc="  HSV", leave=False):
            # Try both active camera path and excluded camera path
            img_path = data_dir / camera / f"{stem}.jpg"
            if not img_path.exists():
                img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = ((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 200)).astype(np.uint8)

            evaluator.update(mask, target)
            count += 1

        if count > 0:
            metrics = evaluator.compute()
            results["baseline_hsv_white"] = metrics
            print(f"  HSV white: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

        # Brightness
        evaluator = Evaluator(args.device)
        count = 0

        for (camera, stem), target in tqdm(gt_masks.items(), desc="  Brightness", leave=False):
            img_path = data_dir / camera / f"{stem}.jpg"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            mask = (img > 200).astype(np.uint8)
            evaluator.update(mask, target)
            count += 1

        if count > 0:
            metrics = evaluator.compute()
            results["baseline_brightness"] = metrics
            print(f"  Brightness (>200): IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

        # PCA blue-band (Härer et al. 2016)
        from sklearn.decomposition import PCA as _PCA
        def _pca_blue_band(img, dn_min=63, dn_max=200):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            h, w = img.shape[:2]
            blue_band = rgb[:, :, 2]
            blue_mask = (blue_band >= dn_min).astype(np.uint8)
            rgb_flat = rgb.reshape(-1, 3)
            rgb_std = (rgb_flat - rgb_flat.mean(axis=0)) / (rgb_flat.std(axis=0) + 1e-8)
            pca_obj = _PCA(n_components=3)
            pc_scores = pca_obj.fit_transform(rgb_std)
            pc2 = pc_scores[:, 1].reshape(h, w)
            pc3 = pc_scores[:, 2].reshape(h, w)
            pca_mask = (pc3 < pc2).astype(np.uint8)
            blue_range = ((blue_band >= dn_min) & (blue_band <= dn_max)).astype(np.uint8)
            return (blue_mask | (pca_mask & blue_range)).astype(np.uint8)

        evaluator = Evaluator(args.device)
        count = 0
        for (camera, stem), target in tqdm(gt_masks.items(), desc="  PCA blue", leave=False):
            img_path = data_dir / camera / f"{stem}.jpg"
            if not img_path.exists():
                img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            mask = _pca_blue_band(img)
            if mask.shape != target.shape:
                mask = cv2.resize(mask, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
            evaluator.update(mask, target)
            count += 1
        if count > 0:
            metrics = evaluator.compute()
            results["baseline_pca_blue"] = metrics
            print(f"  PCA blue-band (Härer): IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

    # Snow baselines OOD (CM00054, CM00071, CM00115)
    if "baselines_ood" in args.types:
        print("\n--- Snow Baselines OOD (CM00054, CM00071, CM00115) ---")
        from sklearn.decomposition import PCA as _PCA2
        def _pca_blue_band_ood(img, dn_min=63, dn_max=200):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            h, w = img.shape[:2]
            blue_band = rgb[:, :, 2]
            blue_mask = (blue_band >= dn_min).astype(np.uint8)
            rgb_flat = rgb.reshape(-1, 3)
            rgb_std = (rgb_flat - rgb_flat.mean(axis=0)) / (rgb_flat.std(axis=0) + 1e-8)
            pca_obj = _PCA2(n_components=3)
            pc_scores = pca_obj.fit_transform(rgb_std)
            pc2 = pc_scores[:, 1].reshape(h, w)
            pc3 = pc_scores[:, 2].reshape(h, w)
            pca_mask = (pc3 < pc2).astype(np.uint8)
            blue_range = ((blue_band >= dn_min) & (blue_band <= dn_max)).astype(np.uint8)
            return (blue_mask | (pca_mask & blue_range)).astype(np.uint8)

        ood_json_map_snow = {
            "CM00054": "data/annotations/54.json",
            "CM00067": "data/annotations/67.json",
            "CM00071": "data/annotations/71.json",
            "CM00115": "data/annotations/115.json",
        }
        ood_masks_snow = {}
        for camera, json_path in ood_json_map_snow.items():
            if Path(json_path).exists():
                ood_masks_snow.update(load_gt_masks(Path(json_path), [camera], label_class="Snow Mano"))
        print(f"  Loaded {len(ood_masks_snow)} OOD GT snow masks")

        # HSV white OOD
        evaluator = Evaluator(args.device)
        count = 0
        for (camera, stem), target in tqdm(ood_masks_snow.items(), desc="  HSV OOD", leave=False):
            for img_path in [data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg",
                             data_dir / camera / f"{stem}.jpg"]:
                if img_path.exists(): break
            else: continue
            img = cv2.imread(str(img_path))
            if img is None: continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = ((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 200)).astype(np.uint8)
            evaluator.update(mask, target); count += 1
        if count > 0:
            metrics = evaluator.compute()
            results["ood_baseline_hsv_white"] = metrics
            print(f"  HSV white: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

        # PCA blue-band OOD
        evaluator = Evaluator(args.device)
        count = 0
        for (camera, stem), target in tqdm(ood_masks_snow.items(), desc="  PCA blue OOD", leave=False):
            for img_path in [data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg",
                             data_dir / camera / f"{stem}.jpg"]:
                if img_path.exists(): break
            else: continue
            img = cv2.imread(str(img_path))
            if img is None: continue
            mask = _pca_blue_band_ood(img)
            if mask.shape != target.shape:
                mask = cv2.resize(mask, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
            evaluator.update(mask, target); count += 1
        if count > 0:
            metrics = evaluator.compute()
            results["ood_baseline_pca_blue"] = metrics
            print(f"  PCA blue-band: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")

    # Snow pseudo-label students (VLM ensemble distillation)
    if "students_pseudo" in args.types:
        print("\n--- Snow Pseudo-Label Student Models (VLM ensemble distillation) ---")

        ckpt_base = Path("models/student_snow/checkpoints")
        pseudo_ckpts = {}
        for threshold in [2, 3, 4, 5]:
            for variant in ["mobilenet", "convnext", "vit_lora"]:
                # New naming: snow_{variant}_maj{threshold}_{ts}/
                dirs = sorted(ckpt_base.glob(f"snow_{variant}_maj{threshold}_*"), key=lambda p: p.stat().st_mtime)
                # Legacy fallback for old runs without threshold in name (maj2 assumed)
                if not dirs and threshold == 2:
                    dirs = sorted(ckpt_base.glob(f"snow_{variant}_2026*"), key=lambda p: p.stat().st_mtime)
                if not dirs:
                    continue
                ckpts = sorted(dirs[-1].glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime)
                if ckpts:
                    key = f"{variant}_maj{threshold}"
                    pseudo_ckpts[key] = (ckpts, variant)

        for model_name, (ckpt_list, variant) in pseudo_ckpts.items():
            if not ckpt_list:
                print(f"  {model_name}: no checkpoint found")
                continue
            ckpt_path = ckpt_list[-1]
            try:
                from student_ground.models import build_model
                from student_ground.configs import get_config
                checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                variant = checkpoint.get("hyper_parameters", {}).get("model_variant", variant)
                model = build_model(variant)
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
                model_sd = model.state_dict()
                filtered = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
                model.load_state_dict(filtered, strict=False)
                model = model.to(args.device)
                model.eval()

                cfg = get_config(variant)
                input_size = cfg.get("input_size", 512)

                evaluator = Evaluator(args.device)
                count = 0

                for (camera, stem), target in tqdm(gt_masks.items(), desc=f"  {model_name}", leave=False):
                    img_path = data_dir / camera / f"{stem}.jpg"
                    if not img_path.exists():
                        img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
                    if not img_path.exists():
                        continue

                    img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
                    img_t = img_to_tensor(img)

                    with torch.no_grad():
                        logits = model(img_t.to(args.device))
                        pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]

                    target_resized = np.array(
                        Image.fromarray(target).resize((input_size, input_size), Image.NEAREST)
                    )
                    evaluator.update(pred.astype(np.float32), target_resized)
                    count += 1

                if count > 0:
                    metrics = evaluator.compute()
                    results[f"student_{model_name}"] = metrics
                    val_iou = ckpt_path.name.split("iou=")[1].rstrip(".ckpt")
                    print(f"  {model_name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f} (pseudo-label val: {val_iou})")

            except Exception as e:
                import traceback
                print(f"  {model_name} failed: {e}")
                traceback.print_exc()

    # Snow students
    if "students" in args.types:
        print("\n--- Snow Student Models ---")

        snow_ckpts = {
            "mobilenet": sorted(Path("models/student_snow/checkpoints/snow_gt_mobilenet_20260522_052052").glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            "convnext":  sorted(Path("models/student_snow/checkpoints/snow_gt_convnext_unfrozen").glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            "vit_lora":  sorted(Path("models/student_snow/checkpoints/snow_gt_vit_lora_unfrozen").glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime),
        }

        for model_name, ckpt_list in snow_ckpts.items():
            if not ckpt_list:
                continue

            ckpt_path = ckpt_list[-1]

            try:
                from student_ground.models import build_model
                checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                variant = checkpoint.get("hyper_parameters", {}).get("model_variant", model_name)
                model = build_model(variant)
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
                # Filter out size-mismatched keys
                model_sd = model.state_dict()
                filtered = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
                model.load_state_dict(filtered, strict=False)
                model = model.to(args.device)
                model.eval()

                from student_ground.configs import get_config
                cfg = get_config(variant)
                input_size = cfg.get("input_size", 512)

                evaluator = Evaluator(args.device)
                count = 0

                for (camera, stem), target in tqdm(gt_masks.items(), desc=f"  {model_name}", leave=False):
                    img_path = data_dir / camera / f"{stem}.jpg"
                    if not img_path.exists():
                        img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
                    if not img_path.exists():
                        continue

                    img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
                    img_t = img_to_tensor(img)

                    with torch.no_grad():
                        logits = model(img_t.to(args.device))
                        pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]

                    # Resize GT to model input size (matches training pipeline)
                    target_resized = np.array(
                        Image.fromarray(target).resize((input_size, input_size), Image.NEAREST)
                    )

                    evaluator.update(pred.astype(np.float32), target_resized)
                    count += 1

                if count > 0:
                    metrics = evaluator.compute()
                    results[f"student_{model_name}"] = metrics
                    val_iou = ckpt_path.name.split("iou=")[1].rstrip(".ckpt")
                    print(f"  {model_name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f} (val: {val_iou})")

            except Exception as e:
                import traceback
                print(f"  {model_name} failed: {e}")
                traceback.print_exc()

    # Snow pseudo-label students: all models × k=2,3,4,5
    if "students_pseudo" in args.types:
        print("\n--- Snow Pseudo-Label Students (majority k=2..5) ---")

        models = ["mobilenet", "convnext", "vit_lora"]
        ks = [2, 3, 4, 5]

        for model_name in models:
            for k in ks:
                ckpt_dirs = sorted(
                    Path("models/student_snow/checkpoints").glob(f"snow_{model_name}_maj{k}_*"),
                    key=lambda p: p.stat().st_mtime
                )
                if not ckpt_dirs:
                    print(f"  {model_name} k={k}: no checkpoint dir")
                    continue
                ckpts = sorted(ckpt_dirs[-1].glob("best-*.ckpt"))
                if not ckpts:
                    print(f"  {model_name} k={k}: no best ckpt")
                    continue
                ckpt_path = ckpts[-1]

                try:
                    from student_ground.models import build_model
                    from student_ground.configs import get_config
                    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                    variant = checkpoint.get("hyper_parameters", {}).get("model_variant", model_name)
                    model = build_model(variant)
                    state_dict = {k2.replace("model.", ""): v for k2, v in checkpoint["state_dict"].items() if k2.startswith("model.")}
                    model_sd = model.state_dict()
                    filtered = {k2: v for k2, v in state_dict.items() if k2 in model_sd and v.shape == model_sd[k2].shape}
                    model.load_state_dict(filtered, strict=False)
                    model = model.to(args.device).eval()

                    cfg = get_config(variant)
                    input_size = cfg.get("input_size", 512)

                    evaluator = Evaluator(args.device)
                    count = 0
                    for (camera, stem), target in tqdm(gt_masks.items(), desc=f"  {model_name} k={k}", leave=False):
                        img_path = data_dir / camera / f"{stem}.jpg"
                        if not img_path.exists():
                            continue
                        img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
                        img_t = img_to_tensor(img)
                        with torch.no_grad():
                            logits = model(img_t.to(args.device))
                            pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]
                        target_resized = np.array(Image.fromarray(target).resize((input_size, input_size), Image.NEAREST))
                        evaluator.update(pred.astype(np.float32), target_resized)
                        count += 1

                    if count > 0:
                        metrics = evaluator.compute()
                        results[f"pseudo_{model_name}_k{k}"] = metrics
                        print(f"  {model_name} k={k}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f} (n={count})")
                except Exception as e:
                    import traceback
                    print(f"  {model_name} k={k} failed: {e}")
                    traceback.print_exc()

    # Pseudo-label students OOD (maj3 checkpoints, no GT fine-tuning)
    if "students_pseudo_ood" in args.types:
        print("\n--- Snow Pseudo-Label Students OOD (maj3, CM00054/CM00071/CM00115) ---")
        ood_cameras = ["CM00054", "CM00067", "CM00071", "CM00115"]
        ood_json_map = {
            "CM00054": "data/annotations/54.json",
            "CM00067": "data/annotations/67.json",
            "CM00071": "data/annotations/71.json",
            "CM00115": "data/annotations/115.json",
        }
        ood_masks_pseudo = {}
        for camera, json_path in ood_json_map.items():
            if Path(json_path).exists():
                ood_masks_pseudo.update(load_gt_masks(Path(json_path), [camera], label_class="Snow Mano"))
        print(f"  Loaded {len(ood_masks_pseudo)} OOD GT snow masks")

        for variant in ["mobilenet", "convnext", "vit_lora"]:
            ckpt_dirs = sorted(
                Path("models/student_snow/checkpoints").glob(f"snow_{variant}_maj3_*"),
                key=lambda p: p.stat().st_mtime
            )
            if not ckpt_dirs:
                print(f"  {variant}: no maj3 checkpoint")
                continue
            ckpts = sorted(ckpt_dirs[-1].glob("best-*.ckpt"))
            if not ckpts:
                continue
            ckpt_path = ckpts[-1]
            try:
                from student_ground.models import build_model
                from student_ground.configs import get_config
                checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                v = checkpoint.get("hyper_parameters", {}).get("model_variant", variant)
                model = build_model(v)
                sd = {k.replace("model.", ""): v2 for k, v2 in checkpoint["state_dict"].items() if k.startswith("model.")}
                msd = model.state_dict()
                model.load_state_dict({k: v2 for k, v2 in sd.items() if k in msd and v2.shape == msd[k].shape}, strict=False)
                model = model.to(args.device).eval()
                cfg = get_config(v)
                input_size = cfg.get("input_size", 512)

                evaluator = Evaluator(args.device)
                count = 0
                for (camera, stem), target in tqdm(ood_masks_pseudo.items(), desc=f"  {variant}", leave=False):
                    for img_path in [data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg",
                                     data_dir / camera / f"{stem}.jpg"]:
                        if img_path.exists(): break
                    else:
                        continue
                    img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
                    img_t = img_to_tensor(img)
                    with torch.no_grad():
                        logits = model(img_t.to(args.device))
                        pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]
                    target_resized = np.array(Image.fromarray(target).resize((input_size, input_size), Image.NEAREST))
                    evaluator.update(pred.astype(np.float32), target_resized)
                    count += 1

                if count > 0:
                    metrics = evaluator.compute()
                    results[f"pseudo_ood_{variant}"] = metrics
                    print(f"  {variant}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f} (n={count})")
            except Exception as e:
                import traceback
                print(f"  {variant} failed: {e}"); traceback.print_exc()

    # Snow students OOD (new OOD cohort: CM00054, CM00071, CM00115)
    if "students_ood" in args.types:
        print("\n--- Snow Student Models OOD (CM00054, CM00071, CM00115) ---")

        ood_cameras = ["CM00054", "CM00067", "CM00071", "CM00115"]
        ood_json_map = {
            "CM00054": "data/annotations/54.json",
            "CM00067": "data/annotations/67.json",
            "CM00071": "data/annotations/71.json",
            "CM00115": "data/annotations/115.json",
        }
        ood_masks = {}
        for camera, json_path in ood_json_map.items():
            if Path(json_path).exists():
                masks = load_gt_masks(Path(json_path), [camera], label_class="Snow Mano")
                ood_masks.update(masks)
        print(f"  Loaded {len(ood_masks)} OOD GT snow masks")

        snow_ckpts = {
            "mobilenet": sorted(Path("models/student_snow/checkpoints/snow_gt_mobilenet_20260522_052052").glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            "convnext":  sorted(Path("models/student_snow/checkpoints/snow_gt_convnext_unfrozen").glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime),
            "vit_lora":  sorted(Path("models/student_snow/checkpoints/snow_gt_vit_lora_unfrozen").glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime),
        }

        for model_name, ckpt_list in snow_ckpts.items():
            if not ckpt_list:
                continue
            ckpt_path = ckpt_list[-1]
            try:
                from student_ground.models import build_model
                from student_ground.configs import get_config
                checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                variant = checkpoint.get("hyper_parameters", {}).get("model_variant", model_name)
                model = build_model(variant)
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
                model_sd = model.state_dict()
                filtered = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
                model.load_state_dict(filtered, strict=False)
                model = model.to(args.device)
                model.eval()

                cfg = get_config(variant)
                input_size = cfg.get("input_size", 512)

                evaluator = Evaluator(args.device)
                count = 0

                for (camera, stem), target in tqdm(ood_masks.items(), desc=f"  {model_name}_ood", leave=False):
                    img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
                    if not img_path.exists():
                        img_path = data_dir / camera / f"{stem}.jpg"
                    if not img_path.exists():
                        continue

                    img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
                    img_t = img_to_tensor(img)

                    with torch.no_grad():
                        logits = model(img_t.to(args.device))
                        pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]

                    target_resized = np.array(
                        Image.fromarray(target).resize((input_size, input_size), Image.NEAREST)
                    )

                    evaluator.update(pred.astype(np.float32), target_resized)
                    count += 1

                if count > 0:
                    metrics = evaluator.compute()
                    results[f"student_{model_name}_ood"] = metrics
                    print(f"  {model_name}_ood: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f} (n={count})")

            except Exception as e:
                import traceback
                print(f"  {model_name} OOD failed: {e}")
                traceback.print_exc()

    # Snow GT kfold in-dist eval
    if "students_gt_kfold" in args.types:
        print("\n--- Snow GT K-Fold CV (in-distribution) ---")
        ckpt_base = Path("models/student_snow/checkpoints")

        for variant in ["mobilenet", "convnext", "vit_lora"]:
            fold_ious, fold_dices = [], []
            for fold in range(5):
                dirs = sorted(ckpt_base.glob(f"snow_gt_{variant}_kfold_{fold}_*"), key=lambda p: p.stat().st_mtime)
                if not dirs:
                    continue
                ckpts = sorted(dirs[-1].glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime)
                if not ckpts:
                    continue
                ckpt_path = ckpts[-1]
                try:
                    from student_ground.models import build_model
                    from student_ground.configs import get_config
                    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                    v = checkpoint.get("hyper_parameters", {}).get("model_variant", variant)
                    model = build_model(v)
                    state_dict = {k.replace("model.", ""): v2 for k, v2 in checkpoint["state_dict"].items() if k.startswith("model.")}
                    model_sd = model.state_dict()
                    filtered = {k: v2 for k, v2 in state_dict.items() if k in model_sd and v2.shape == model_sd[k].shape}
                    model.load_state_dict(filtered, strict=False)
                    model = model.to(args.device)
                    model.eval()
                    cfg = get_config(v)
                    input_size = cfg.get("input_size", 512)

                    evaluator = Evaluator(args.device)
                    count = 0
                    for (camera, stem), target in tqdm(gt_masks.items(), desc=f"  {variant}_fold{fold}", leave=False):
                        img_path = data_dir / camera / f"{stem}.jpg"
                        if not img_path.exists():
                            img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
                        if not img_path.exists():
                            continue
                        img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
                        img_t = img_to_tensor(img)
                        with torch.no_grad():
                            logits = model(img_t.to(args.device))
                            pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]
                        target_resized = np.array(Image.fromarray(target).resize((input_size, input_size), Image.NEAREST))
                        evaluator.update(pred.astype(np.float32), target_resized)
                        count += 1

                    if count > 0:
                        m = evaluator.compute()
                        fold_ious.append(m["iou"]); fold_dices.append(m["dice"])
                        val_iou = ckpt_path.name.split("iou=")[1].rstrip(".ckpt")
                        print(f"  {variant} fold{fold}: IoU={m['iou']:.4f}, Dice={m['dice']:.4f} (val: {val_iou})")
                except Exception as e:
                    import traceback
                    print(f"  {variant} fold{fold} failed: {e}")
                    traceback.print_exc()

            if fold_ious:
                mean_iou = np.mean(fold_ious); std_iou = np.std(fold_ious)
                mean_dice = np.mean(fold_dices); std_dice = np.std(fold_dices)
                print(f"  {variant} MEAN: IoU={mean_iou:.4f}±{std_iou:.4f}, Dice={mean_dice:.4f}±{std_dice:.4f}")
                results[f"gt_kfold_{variant}"] = {"iou": mean_iou, "dice": mean_dice}

    # Snow GT kfold OOD eval (CM00054, CM00071, CM00115)
    if "students_gt_ood" in args.types:
        print("\n--- Snow GT K-Fold OOD (CM00054, CM00071, CM00115) ---")
        ckpt_base = Path("models/student_snow/checkpoints")
        ood_json_map = {
            "CM00054": "data/annotations/54.json",
            "CM00067": "data/annotations/67.json",
            "CM00071": "data/annotations/71.json",
            "CM00115": "data/annotations/115.json",
        }
        ood_masks_gt = {}
        for camera, json_path in ood_json_map.items():
            if Path(json_path).exists():
                ood_masks_gt.update(load_gt_masks(Path(json_path), [camera], label_class="Snow Mano"))
        print(f"  Loaded {len(ood_masks_gt)} OOD GT snow masks")

        for variant in ["mobilenet", "convnext", "vit_lora"]:
            best_ckpt, best_val = None, 0
            for fold in range(5):
                dirs = sorted(ckpt_base.glob(f"snow_gt_{variant}_kfold_{fold}_*"), key=lambda p: p.stat().st_mtime)
                if not dirs: continue
                ckpts = sorted(dirs[-1].glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime)
                if not ckpts: continue
                ckpt = ckpts[-1]
                try:
                    val = float(ckpt.name.split("iou=")[1].rstrip(".ckpt"))
                    if val > best_val: best_val, best_ckpt = val, ckpt
                except: pass

            if not best_ckpt:
                print(f"  {variant}: no kfold checkpoint found"); continue
            try:
                from student_ground.models import build_model
                from student_ground.configs import get_config
                checkpoint = torch.load(best_ckpt, map_location=args.device, weights_only=False)
                v = checkpoint.get("hyper_parameters", {}).get("model_variant", variant)
                model = build_model(v)
                state_dict = {k.replace("model.", ""): v2 for k, v2 in checkpoint["state_dict"].items() if k.startswith("model.")}
                model_sd = model.state_dict()
                filtered = {k: v2 for k, v2 in state_dict.items() if k in model_sd and v2.shape == model_sd[k].shape}
                model.load_state_dict(filtered, strict=False)
                model = model.to(args.device)
                model.eval()
                cfg = get_config(v)
                input_size = cfg.get("input_size", 512)

                evaluator = Evaluator(args.device)
                count = 0
                for (camera, stem), target in tqdm(ood_masks_gt.items(), desc=f"  {variant}_ood", leave=False):
                    img_path = data_dir / f"_EXCLUDED_{camera}" / f"{stem}.jpg"
                    if not img_path.exists(): img_path = data_dir / camera / f"{stem}.jpg"
                    if not img_path.exists(): continue
                    img = np.array(Image.open(img_path).convert("RGB").resize((input_size, input_size)))
                    img_t = img_to_tensor(img)
                    with torch.no_grad():
                        logits = model(img_t.to(args.device))
                        pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]
                    target_resized = np.array(Image.fromarray(target).resize((input_size, input_size), Image.NEAREST))
                    evaluator.update(pred.astype(np.float32), target_resized)
                    count += 1

                if count > 0:
                    m = evaluator.compute()
                    results[f"gt_ood_{variant}"] = m
                    print(f"  {variant}_ood: IoU={m['iou']:.4f}, Dice={m['dice']:.4f} (best fold val: {best_val:.3f}, n={count})")
            except Exception as e:
                import traceback
                print(f"  {variant} GT OOD failed: {e}")
                traceback.print_exc()

    # Avalanche OOD: Recall@polygon + IoU@ROI on UIBK avalanche dataset
    if "avalanche_ood" in args.types:
        import json, cv2
        print("\n--- Avalanche OOD (UIBK, Recall@polygon + IoU@ROI) ---")

        _AVA_DIR = Path("datasets/avalanche")
        if not _AVA_DIR.exists():
            print(f"  ERROR: {_AVA_DIR} not found.")
        else:
            ROI_DILATE = 50  # pixels to dilate bbox for IoU@ROI

            def load_avalanche_polygons(json_path, img_w, img_h):
                """Load VOTT JSON → list of (polygon_pts, tag) per region."""
                with open(json_path) as f:
                    data = json.load(f)
                polygons = []
                for region in data.get("regions", []):
                    if region.get("type") != "POLYGON":
                        continue
                    pts = np.array([[p["x"], p["y"]] for p in region["points"]], dtype=np.float32)
                    # Points are in original image pixel coords
                    pts = pts.clip([[0, 0]], [[img_w - 1, img_h - 1]])
                    polygons.append(pts.astype(np.int32))
                return polygons

            def rasterize_polygons(polygons, h, w):
                """Fill polygons → binary mask (H, W)."""
                mask = np.zeros((h, w), dtype=np.uint8)
                for pts in polygons:
                    cv2.fillPoly(mask, [pts], 1)
                return mask

            def roi_from_polygons(polygons, h, w, dilate=ROI_DILATE):
                """Dilated bounding box enclosing all polygons → slice."""
                all_pts = np.concatenate(polygons, axis=0)
                x0 = max(0, int(all_pts[:, 0].min()) - dilate)
                y0 = max(0, int(all_pts[:, 1].min()) - dilate)
                x1 = min(w, int(all_pts[:, 0].max()) + dilate)
                y1 = min(h, int(all_pts[:, 1].max()) + dilate)
                return y0, y1, x0, x1

            # Collect all annotated images
            ann_types = ["glide", "loose", "slab"]
            samples = []  # (img_path, json_path, ann_type)
            for atype in ann_types:
                ann_subdir = _AVA_DIR / "annotations" / atype
                if not ann_subdir.exists():
                    continue
                for json_path in sorted(ann_subdir.glob("*.json")):
                    img_name = json_path.stem
                    for ext in ("jpg", "JPG", "jpeg", "png"):
                        img_path = _AVA_DIR / "images" / atype / f"{img_name}.{ext}"
                        if img_path.exists():
                            samples.append((img_path, json_path, atype))
                            break
            print(f"  Found {len(samples)} annotated avalanche images")

            # Load snow student models (pseudo maj3 — best k=3)
            snow_models = {}
            for variant in ["mobilenet", "convnext", "vit_lora"]:
                ckpt_dirs = sorted(
                    Path("models/student_snow/checkpoints").glob(f"snow_{variant}_maj3_*"),
                    key=lambda p: p.stat().st_mtime
                )
                if not ckpt_dirs:
                    continue
                ckpts = sorted(ckpt_dirs[-1].glob("best-*.ckpt"))
                if not ckpts:
                    continue
                try:
                    from student_ground.models import build_model
                    from student_ground.configs import get_config
                    ckpt_path = ckpts[-1]
                    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
                    v = checkpoint.get("hyper_parameters", {}).get("model_variant", variant)
                    m = build_model(v)
                    sd = {k.replace("model.", ""): v2 for k, v2 in checkpoint["state_dict"].items() if k.startswith("model.")}
                    msd = m.state_dict()
                    m.load_state_dict({k: v2 for k, v2 in sd.items() if k in msd and v2.shape == msd[k].shape}, strict=False)
                    m = m.to(args.device).eval()
                    cfg = get_config(v)
                    input_size = cfg.get("input_size", 512)
                    snow_models[variant] = (m, input_size)
                    print(f"  Loaded {variant} (input={input_size})")
                except Exception as e:
                    print(f"  {variant} load failed: {e}")

            if not snow_models:
                print("  No snow models loaded — skip.")
            else:
                # Per-model metrics
                model_recalls = {v: [] for v in snow_models}
                model_iou_roi = {v: [] for v in snow_models}

                for img_path, json_path, atype in tqdm(samples, desc="  Avalanche OOD"):
                    img_pil = Image.open(img_path).convert("RGB")
                    W, H = img_pil.size

                    polygons = load_avalanche_polygons(json_path, W, H)
                    if not polygons:
                        continue

                    poly_mask = rasterize_polygons(polygons, H, W)  # (H, W) binary
                    y0, y1, x0, x1 = roi_from_polygons(polygons, H, W)

                    for variant, (model, input_size) in snow_models.items():
                        img_resized = np.array(img_pil.resize((input_size, input_size)))
                        img_t = img_to_tensor(img_resized)
                        with torch.no_grad():
                            logits = model(img_t.to(args.device))
                            pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().numpy()[0]

                        # Resize pred back to original image size
                        pred_full = np.array(
                            Image.fromarray(pred.astype(np.uint8)).resize((W, H), Image.NEAREST)
                        )

                        # Recall@polygon: TP / (TP+FN) within polygon mask
                        tp = (pred_full & poly_mask).sum()
                        fn = (~pred_full.astype(bool) & poly_mask.astype(bool)).sum()
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        model_recalls[variant].append(recall)

                        # IoU@ROI: IoU within dilated bounding box
                        pred_roi = pred_full[y0:y1, x0:x1]
                        gt_roi = poly_mask[y0:y1, x0:x1]
                        tp_r = (pred_roi & gt_roi).sum()
                        fp_r = (pred_roi & ~gt_roi.astype(bool)).sum()
                        fn_r = (~pred_roi.astype(bool) & gt_roi.astype(bool)).sum()
                        iou_roi = tp_r / (tp_r + fp_r + fn_r) if (tp_r + fp_r + fn_r) > 0 else 0.0
                        model_iou_roi[variant].append(iou_roi)

                print(f"\n  {'Model':<20} {'Recall@polygon':<20} {'IoU@ROI'}")
                print(f"  {'-'*55}")
                for variant in snow_models:
                    rec = np.mean(model_recalls[variant])
                    iou_r = np.mean(model_iou_roi[variant])
                    print(f"  {variant:<20} {rec:<20.4f} {iou_r:.4f}")
                    results[f"avalanche_ood_{variant}"] = {
                        "iou": iou_r, "dice": rec,  # store recall as dice for summary compat
                        "recall_polygon": rec, "iou_roi": iou_r,
                    }

    # DINOv3 multihead (Phase 1 pseudolabel decoder, no GT finetune)
    if "multihead" in args.types:
        print("\n--- Multihead (DINOv3MultiHead, Phase 1 pseudolabel) ---")
        try:
            metrics = eval_multihead_task(gt_masks, "snow", args, ckpt_path=args.multihead_ckpt, label="multihead_snow_pseudo")
            if metrics is not None:
                results["multihead_snow_pseudo"] = metrics
        except Exception as e:
            import traceback
            print(f"  multihead snow eval failed: {e}")
            traceback.print_exc()

        print("\n--- Multihead OOD (Held-out Cameras: CM00054, CM00071, CM00115) ---")
        ood_json_map = {
            "CM00054": "data/annotations/54.json",
            "CM00067": "data/annotations/67.json",
            "CM00071": "data/annotations/71.json",
            "CM00115": "data/annotations/115.json",
        }
        ood_masks = {}
        for camera, json_path in ood_json_map.items():
            if Path(json_path).exists():
                ood_masks.update(load_gt_masks(Path(json_path), [camera], label_class="Snow Mano"))
        print(f"  Loaded {len(ood_masks)} OOD GT snow masks")
        if ood_masks:
            try:
                metrics = eval_multihead_task(ood_masks, "snow", args, ckpt_path=args.multihead_ckpt, label="multihead_snow_pseudo_ood")
                if metrics is not None:
                    results["multihead_snow_pseudo_ood"] = metrics
            except Exception as e:
                import traceback
                print(f"  multihead snow OOD eval failed: {e}")
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY (sorted by IoU)")
    print("=" * 100)
    print(f"{'Method':<40} {'IoU':<15} {'Dice':<15}")
    print("-" * 100)
    for method, metrics in sorted(results.items(), key=lambda x: x[1]["iou"], reverse=True):
        print(f"{method:<40} {metrics['iou']:<15.4f} {metrics['dice']:<15.4f}")


def eval_per_camera(gt_masks, pred_dir_func, device="cpu", task="ground"):
    """Evaluate per-camera performance."""
    cameras = ["CF00001", "CM00009", "CM00014", "CM00070", "CM00075", "CM00080", "CM00102", "CM00108", "CM00112"]
    excluded = ["CM00054", "CM00067", "CM00071", "CM00115"]

    print("\n" + "=" * 120)
    print("PER-CAMERA BREAKDOWN")
    print("=" * 120)
    print(f"{'Camera':<15} {'In-Domain IoU':<20} {'In-Domain Dice':<20} {'OOD IoU':<20} {'OOD Dice':<20}")
    print("-" * 120)

    for camera in cameras:
        # In-domain
        evaluator = Evaluator(device)
        id_count = 0
        for (cam, stem), target in gt_masks.items():
            if cam != camera:
                continue
            pred_path = pred_dir_func(cam, stem)
            if not pred_path.exists():
                continue
            pred = np.array(Image.open(pred_path).convert("L")) > 127
            evaluator.update(pred, target)
            id_count += 1

        id_metrics = evaluator.compute() if id_count > 0 else None
        id_iou = f"{id_metrics['iou']:.4f}" if id_metrics else "—"
        id_dice = f"{id_metrics['dice']:.4f}" if id_metrics else "—"

        # OOD (this camera as OOD doesn't make sense, skip)
        ood_iou = "—"
        ood_dice = "—"

        print(f"{camera:<15} {id_iou:<20} {id_dice:<20} {ood_iou:<20} {ood_dice:<20}")

    # OOD: excluded cameras vs. ensemble/best model
    print("\n" + "=" * 120)
    print("OUT-OF-DOMAIN (3 Excluded Cameras)")
    print("=" * 120)
    print(f"{'Camera':<15} {'IoU':<20} {'Dice':<20} {'Images':<20}")
    print("-" * 120)

    # Load OOD GT (excluded cameras) if available
    if task == "ground":
        ood_gt_path = Path("data/annotations/ground_annotations.json")
    else:
        # Snow OOD would need annotations for excluded cameras (likely don't have)
        print("OOD GT not available for snow (excluded cameras not annotated)")
        return

    ood_gt = load_gt_masks(ood_gt_path, excluded)
    print(f"Found {len(ood_gt)} OOD images")

    for camera in excluded:
        evaluator = Evaluator(device)
        count = 0
        for (cam, stem), target in ood_gt.items():
            if cam != camera:
                continue
            pred_path = pred_dir_func(cam, stem)
            if not pred_path.exists():
                continue
            pred = np.array(Image.open(pred_path).convert("L")) > 127
            evaluator.update(pred, target)
            count += 1

        if count > 0:
            metrics = evaluator.compute()
            print(f"{camera:<15} {metrics['iou']:<20.4f} {metrics['dice']:<20.4f} {count:<20}")
        else:
            print(f"{camera:<15} {'—':<20} {'—':<20} {count:<20}")


def eval_smoother(args):
    """Evaluate temporal smoothers: Kalman baseline + GRU/LSTM variants."""
    import re, math
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d

    print("\n" + "=" * 100)
    print("TEMPORAL SMOOTHER EVALUATION")
    print("=" * 100)

    CSV = Path("analysis/snow_timeseries.csv")
    if not CSV.exists():
        print(f"ERROR: {CSV} not found"); return

    df = pd.read_csv(CSV, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    # No QA filter — smoother trained on all frames, eval must match
    df = df.sort_values(["camera", "datetime"]).reset_index(drop=True)
    print(f"Loaded {len(df)} frames (all, no QA filter), {df['camera'].nunique()} cameras")

    # Reproduce temporal val split
    from datetime import date as _date
    def _extract_date(stem):
        m = re.search(r"(\d{4})(\d{2})(\d{2})", str(stem))
        return _date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

    val_set = set()
    for cam, cam_df in df.groupby("camera"):
        stems = cam_df["stem"].tolist()
        dates = [_extract_date(s) for s in stems]
        unique_dates = sorted(set(d for d in dates if d))
        if len(unique_dates) < 3:
            n_val = max(1, int(len(stems) * 0.2))
            for s in stems[-n_val:]: val_set.add((cam, s))
            continue
        n_val_days = max(1, int(len(unique_dates) * 0.2))
        gap = min(1, len(unique_dates) - n_val_days - 1)
        cutoff = unique_dates[-(n_val_days + gap)]
        for stem, d in zip(stems, dates):
            if d and d >= cutoff and (d - cutoff).days >= gap:
                val_set.add((cam, stem))

    df_train = df[~df.apply(lambda r: (r["camera"], r["stem"]) in val_set, axis=1)]

    def compute_mae(pred, target, mask=None):
        err = np.abs(np.array(pred) - np.array(target))
        return float(err[mask].mean()) if mask is not None and mask.sum() > 0 else float(err.mean())

    results = {}

    # ── Kalman baseline ───────────────────────────────────────────────────────
    if "kalman" in args.types or "all" in args.types:
        print("\n--- Kalman Filter ---")

        def kalman_smooth(series, Q, R):
            x, P = float(series[0]), 500.0
            out = []
            for z in series:
                P_pred = P + Q
                K = P_pred / (P_pred + R)
                x = x + K * (float(z) - x)
                P = (1 - K) * P_pred
                out.append(np.clip(x, 0, 100))
            return np.array(out)

        best_mae, best_Q, best_R = float("inf"), 100, 400
        for Q in [1, 5, 10, 25, 50, 100]:
            for R in [25, 50, 100, 200, 400, 800]:
                maes = []
                for cam, cam_df in df_train.groupby("camera"):
                    cam_df = cam_df.sort_values("datetime").reset_index(drop=True)
                    if len(cam_df) < 10: continue
                    sm = kalman_smooth(cam_df["student_snow_pct"].values, Q, R)
                    tgt = gaussian_filter1d(cam_df["student_snow_pct"].values.astype(float), sigma=3.5)
                    maes.append(np.abs(sm - tgt).mean())
                avg = np.mean(maes) if maes else float("inf")
                if avg < best_mae: best_mae, best_Q, best_R = avg, Q, R
        print(f"  Best Q={best_Q}, R={best_R}, train MAE={best_mae:.4f}")

        all_raw, all_sm, all_tgt = [], [], []
        for cam, cam_df in df.groupby("camera"):
            cam_df = cam_df.sort_values("datetime").reset_index(drop=True)
            sm_full = kalman_smooth(cam_df["student_snow_pct"].values, best_Q, best_R)
            tgt_full = gaussian_filter1d(cam_df["student_snow_pct"].values.astype(float), sigma=3.5)
            val_mask = cam_df.apply(lambda r: (cam, r["stem"]) in val_set, axis=1).values
            if val_mask.sum() == 0: continue
            all_raw.extend(cam_df["student_snow_pct"].values[val_mask].tolist())
            all_sm.extend(sm_full[val_mask].tolist())
            all_tgt.extend(tgt_full[val_mask].tolist())

        all_raw = np.array(all_raw); all_sm = np.array(all_sm); all_tgt = np.array(all_tgt)
        snow_mask = all_tgt > 5.0
        hard_mask = np.abs(all_raw - all_tgt) > 10.0

        print(f"  MAE (all):        raw={compute_mae(all_raw,all_tgt):.4f}  kalman={compute_mae(all_sm,all_tgt):.4f}")
        print(f"  MAE (snow >5%):   raw={compute_mae(all_raw,all_tgt,snow_mask):.4f}  kalman={compute_mae(all_sm,all_tgt,snow_mask):.4f}  n={snow_mask.sum()}")
        print(f"  MAE (hard >10%):  raw={compute_mae(all_raw,all_tgt,hard_mask):.4f}  kalman={compute_mae(all_sm,all_tgt,hard_mask):.4f}  n={hard_mask.sum()}")
        results["raw"]    = {"mae_all": compute_mae(all_raw,all_tgt), "mae_snow": compute_mae(all_raw,all_tgt,snow_mask), "mae_hard": compute_mae(all_raw,all_tgt,hard_mask)}
        results["kalman"] = {"mae_all": compute_mae(all_sm,all_tgt),  "mae_snow": compute_mae(all_sm,all_tgt,snow_mask),  "mae_hard": compute_mae(all_sm,all_tgt,hard_mask)}

    # ── GRU/LSTM smoothers ────────────────────────────────────────────────────
    if any(x in args.types for x in ["gru", "lstm", "all"]):
        print("\n--- GRU/LSTM Smoothers (latest of each type) ---")
        ckpt_base = Path("models/temporal_snow/checkpoints")

        # Pick latest dir for each of 4 types: uni-GRU, bi-GRU, uni-LSTM, bi-LSTM
        type_patterns = {
            "GRU (uni)":   lambda n: "gru"  in n and "_bi_" not in n,
            "GRU (bi)":    lambda n: "gru"  in n and "_bi_" in n,
            "LSTM (uni)":  lambda n: "lstm" in n and "_bi_" not in n,
            "LSTM (bi)":   lambda n: "lstm" in n and "_bi_" in n,
        }
        smoother_ckpts = {}
        for label, matcher in type_patterns.items():
            # Pick latest dir by timestamp, then best ckpt within it
            matching_dirs = sorted([
                d for d in ckpt_base.iterdir()
                if d.is_dir() and d.name.startswith("smoother_")
                and matcher(d.name)
                and "seq96" in d.name and "h128" in d.name
                and "20260623" in d.name  # today's retrained models only
            ])
            if not matching_dirs: continue
            latest_dir = matching_dirs[-1]
            ckpts = list(latest_dir.glob("best-*val_mae=*.ckpt"))
            if not ckpts: continue
            best_ckpt = min(ckpts, key=lambda p: float(p.stem.split("val_mae=")[1]) if "val_mae=" in p.stem else 999)
            best_mae = float(best_ckpt.stem.split("val_mae=")[1]) if "val_mae=" in best_ckpt.stem else 0
            smoother_ckpts[label] = (str(best_ckpt), best_mae)

        for name, (ckpt_path, ckpt_mae) in sorted(smoother_ckpts.items(), key=lambda x: x[1][1]):
            try:
                sys.path.insert(0, str(Path(__file__).parent / "models"))
                from temporal_snow.train_smoother import TemporalSmootherModule
                from temporal_snow.dataset import build_feature_columns, SNOW_PCT_COLS, FeatureScaler
                print(f"  [{name}] ckpt: {Path(ckpt_path).parent.name}/{Path(ckpt_path).name}")
                module = TemporalSmootherModule.load_from_checkpoint(ckpt_path, map_location="cpu")
                module.eval()

                # Load scaler if available (new checkpoints have scaler.json)
                scaler_path = Path(ckpt_path).parent / "scaler.json"
                scaler = FeatureScaler.load(scaler_path) if scaler_path.exists() else None

                all_raw2, all_sm2, all_tgt2 = [], [], []
                for cam, cam_df in df.groupby("camera"):
                    cam_df = cam_df.sort_values("datetime").reset_index(drop=True)
                    snow_pct_cols = [c for c in SNOW_PCT_COLS if c in cam_df.columns]
                    if not snow_pct_cols: continue
                    tmp = cam_df.copy()
                    tmp["hour_sin"] = np.sin(2*math.pi*tmp["datetime"].dt.hour/24)
                    tmp["hour_cos"] = np.cos(2*math.pi*tmp["datetime"].dt.hour/24)
                    doy = tmp["datetime"].dt.dayofyear
                    tmp["doy_sin"] = np.sin(2*math.pi*doy/365)
                    tmp["doy_cos"] = np.cos(2*math.pi*doy/365)
                    FILL = {"t2m":0,"visibility":1000,"weather_code":0,"brightness":128,"contrast":50,
                            "edge_density":10,"entropy":4,"white_frac":0,"low_sat_frac":0,"blue_ratio":0,
                            "bottom_white_frac":0,"bottom_brightness":128,"snow_n_components":0,
                            "snow_largest_ratio":0,"snow_centroid_y":0.5,"ground_coverage":0}
                    for col in snow_pct_cols: tmp[col] = tmp[col].fillna(0)
                    for col, val in FILL.items():
                        if col in tmp.columns: tmp[col] = tmp[col].fillna(val)
                    if "visibility" in tmp.columns: tmp["visibility"] = np.log1p(tmp["visibility"])
                    if "snow_n_components" in tmp.columns: tmp["snow_n_components"] = np.log1p(tmp["snow_n_components"])
                    input_size = module.hparams.get("input_size", None)
                    feat_cols = build_feature_columns(snow_pct_cols, use_image_features=True)
                    if input_size and len(feat_cols) != input_size: feat_cols = feat_cols[:input_size]
                    for col in feat_cols:
                        if col not in tmp.columns: tmp[col] = 0.0

                    if scaler is not None:
                        tmp = scaler.transform(tmp)
                    else:
                        # Fallback: manual approximate scaling
                        for col in snow_pct_cols: tmp[col] = tmp[col] / 100.0
                        if "t2m" in tmp.columns: tmp["t2m"] = tmp["t2m"] / 30.0
                        if "visibility" in tmp.columns: tmp["visibility"] = tmp["visibility"] / np.log1p(65000)
                        if "weather_code" in tmp.columns: tmp["weather_code"] = tmp["weather_code"] / 100.0
                        if "brightness" in tmp.columns:
                            tmp["brightness"] /= 255.0; tmp["contrast"] /= 128.0
                            tmp["edge_density"] /= 50.0; tmp["entropy"] /= 6.0
                            tmp["bottom_brightness"] /= 255.0
                        if "snow_n_components" in tmp.columns: tmp["snow_n_components"] /= np.log1p(100)

                    x = np.nan_to_num(tmp[feat_cols].values.astype(np.float32), nan=0.0)
                    with torch.no_grad():
                        pred = module(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy() * 100.0
                    tgt_full = gaussian_filter1d(cam_df["student_snow_pct"].values.astype(float), sigma=3.5)
                    val_mask = cam_df.apply(lambda r: (cam, r["stem"]) in val_set, axis=1).values
                    if val_mask.sum() == 0: continue
                    all_raw2.extend(cam_df["student_snow_pct"].values[val_mask].tolist())
                    all_sm2.extend(pred[val_mask].tolist())
                    all_tgt2.extend(tgt_full[val_mask].tolist())

                all_raw2=np.array(all_raw2); all_sm2=np.array(all_sm2); all_tgt2=np.array(all_tgt2)
                snow_mask2 = all_tgt2 > 5.0
                hard_mask2 = np.abs(all_raw2 - all_tgt2) > 10.0
                mae_all  = compute_mae(all_sm2, all_tgt2)
                mae_snow = compute_mae(all_sm2, all_tgt2, snow_mask2)
                mae_hard = compute_mae(all_sm2, all_tgt2, hard_mask2)
                print(f"  {name:<55} all={mae_all:.4f}  snow={mae_snow:.4f}  hard={mae_hard:.4f}")
                results[name] = {"mae_all": mae_all, "mae_snow": mae_snow, "mae_hard": mae_hard}
            except Exception as e:
                print(f"  {name}: FAILED — {e}")

    # Summary
    print("\n" + "=" * 100)
    print(f"{'Method':<55} {'MAE(all)':<12} {'MAE(snow>5%)':<15} {'MAE(hard>10%)'}")
    print("-" * 100)
    for method, m in sorted(results.items(), key=lambda x: x[1]["mae_all"]):
        print(f"{method:<55} {m['mae_all']:<12.4f} {m['mae_snow']:<15.4f} {m['mae_hard']:.4f}")


def eval_qa(args):
    """Evaluate QA models: TinyQACNN vs HistGradBoost. AUROC, F1, inference timing."""
    import time, pickle, sys
    import pandas as pd
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    _QA_DIR = Path("qa/checkpoints")
    _IMG_DIR = Path("data/balanced_snow")

    # ── Load feature CSV (pre-extracted) ─────────────────────────────────
    feat_csv = _QA_DIR / "qa_features.csv"
    if not feat_csv.exists():
        print(f"ERROR: {feat_csv} not found. Run qa/train_qa.py first.")
        return
    df = pd.read_csv(feat_csv)
    labels = df["label"].values
    # 3-class labels for CNN (0=clean_no_snow, 1=snow, 2=non_usable)
    if "label3" in df.columns:
        labels3 = df["label3"].values.astype(np.int64)
    else:
        # Fallback: map binary label to 0/1 (no class-2 distinction)
        labels3 = labels.astype(np.int64)

    # Feature columns: everything except metadata
    skip = {"camera", "stem", "filename", "label", "label3", "no_snow", "blurry", "bad_light", "not_in_frame"}
    feat_cols = [c for c in df.columns if c not in skip]
    X = df[feat_cols].values.astype(np.float32)
    print(f"QA dataset: {len(labels)} samples ({labels.sum():.0f} usable / {(1-labels).sum():.0f} non-usable)")
    if "label3" in df.columns:
        for c, name in enumerate(["clean_no_snow", "snow", "non_usable"]):
            print(f"  {name}: {(labels3==c).sum()}")
    print(f"Features: {len(feat_cols)}")

    # ── Resolve image paths for CNN ───────────────────────────────────────
    img_paths = []
    cnn_labels = []
    for _, row in df.iterrows():
        p = _IMG_DIR / row["camera"] / f"{row['stem']}.jpg"
        if p.exists():
            img_paths.append(str(p))
            cnn_labels.append(int(row.get("label3", row["label"])))
    cnn_labels = np.array(cnn_labels, dtype=np.int64)
    print(f"CNN images found: {len(img_paths)}/{len(df)}")

    print("\n" + "=" * 80)
    print("IMAGE QUALITY ASSESSMENT EVALUATION")
    print("=" * 80)

    results = {}

    # ─────────────────────────────────────────────────────────────────────
    # 1. HistGradBoost
    # ─────────────────────────────────────────────────────────────────────
    if "gradboost" in [t.lower() for t in args.types]:
        print("\n--- HistGradientBoosting ---")
        from sklearn.ensemble import HistGradientBoostingClassifier

        # Use stored CV predictions if available (fast, unbiased)
        cv_pred_csv = _QA_DIR / "qa_cv_predictions.csv"
        if cv_pred_csv.exists():
            cv_df = pd.read_csv(cv_pred_csv)
            gb_probs = cv_df["cv_prob_usable"].values
            gb_labels = cv_df["label"].values
            print(f"  Using stored CV predictions ({len(gb_probs)} samples)")
            # Still need a fitted model for timing — refit on full data
            gb_model = HistGradientBoostingClassifier(random_state=42)
            gb_model.fit(X, labels)
        else:
            # Re-run 5-fold CV from scratch
            print("  Running 5-fold CV (no stored predictions found)...")
            gb_model = HistGradientBoostingClassifier(random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            gb_probs = np.zeros(len(labels))
            for tr, va in cv.split(X, labels):
                m = HistGradientBoostingClassifier(random_state=42)
                m.fit(X[tr], labels[tr])
                gb_probs[va] = m.predict_proba(X[va])[:, 1]
            gb_model.fit(X, labels)
            gb_labels = labels

        gb_preds = (gb_probs >= 0.5).astype(int)
        gb_auroc = roc_auc_score(gb_labels, gb_probs)
        gb_f1 = f1_score(gb_labels, gb_preds)
        print(f"  AUROC (5-fold CV): {gb_auroc:.4f}")
        print(f"  F1   (5-fold CV): {gb_f1:.4f}")

        x_single = X[:1]
        for _ in range(10): gb_model.predict_proba(x_single)
        t0 = time.perf_counter()
        for _ in range(200): gb_model.predict_proba(x_single)
        single_ms = (time.perf_counter() - t0) / 200 * 1000
        print(f"  Inference single: {single_ms:.1f} ms")

        x_batch = X[:100]
        for _ in range(5): gb_model.predict_proba(x_batch)
        t0 = time.perf_counter()
        for _ in range(50): gb_model.predict_proba(x_batch)
        batch_ms = (time.perf_counter() - t0) / 50 / 100 * 1000
        print(f"  Inference batch-100 (per image): {batch_ms:.1f} ms")

        results["HistGradBoost"] = {"auroc": gb_auroc, "f1": gb_f1,
                                     "single_ms": single_ms, "batch100_ms": batch_ms}

    # ─────────────────────────────────────────────────────────────────────
    # 2. TinyQACNN
    # ─────────────────────────────────────────────────────────────────────
    cnn_ckpt = _QA_DIR / "qa_cnn.ckpt"
    if cnn_ckpt.exists() and "cnn" in [t.lower() for t in args.types]:
        print("\n--- TinyQACNN ---")
        sys.path.insert(0, str(Path("qa")))
        from train_qa_cnn import QADataset

        ckpt = torch.load(cnn_ckpt, map_location="cpu", weights_only=False)
        input_size = ckpt.get("input_size", 128)
        n_params = ckpt.get("n_params", 0)
        stored_auroc = ckpt.get("auroc_cv")
        stored_f1 = ckpt.get("f1_cv")

        # Try full model file first (.pt), fall back to state_dict with dynamic class
        cnn_pt = _QA_DIR / "qa_cnn.pt"
        device = torch.device(args.device)
        if cnn_pt.exists():
            try:
                model = torch.jit.load(str(cnn_pt), map_location=device)
                model = model.eval()
                print(f"  Loaded TorchScript model from qa_cnn.pt")
            except Exception:
                try:
                    model = torch.load(cnn_pt, map_location=device, weights_only=False)
                    model = model.to(device).eval()
                    print(f"  Loaded full model from qa_cnn.pt")
                except Exception as e2:
                    print(f"  ERROR: Cannot load CNN model: {e2}")
                    model = None
        else:
            print(f"  ERROR: qa_cnn.pt not found.")
            model = None
        if model is None:
            print("  Skipping TinyQACNN timing — model not loadable.")
        else:
            n_params = sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else n_params

        # AUROC: use stored if available
        if stored_auroc is not None:
            print(f"  AUROC (5-fold CV, stored): {stored_auroc:.4f}")
            print(f"  F1   (5-fold CV, stored):  {stored_f1:.4f}")
            cnn_auroc, cnn_f1 = stored_auroc, stored_f1
        elif model is not None:
            from torch.utils.data import DataLoader
            _num_cls = ckpt.get("num_classes", 2)
            dataset = QADataset(img_paths, cnn_labels.tolist(), input_size=input_size, augment=False)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            all_probs = np.zeros((len(cnn_labels), _num_cls))
            for _, val_idx in cv.split(img_paths, cnn_labels):
                val_ds = torch.utils.data.Subset(dataset, val_idx)
                val_dl = DataLoader(val_ds, batch_size=64, num_workers=4)
                fold_probs = []
                with torch.no_grad():
                    for x, _ in val_dl:
                        logits = model(x.to(device))
                        if _num_cls > 1:
                            probs = torch.softmax(logits, dim=1).cpu().numpy()
                        else:
                            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
                        fold_probs.append(probs)
                all_probs[val_idx] = np.vstack(fold_probs)
            if _num_cls > 2:
                cnn_auroc = roc_auc_score(cnn_labels, all_probs, multi_class="ovr", average="macro")
                cnn_f1 = f1_score(cnn_labels, all_probs.argmax(axis=1), average="macro")
            else:
                cnn_auroc = roc_auc_score(cnn_labels, all_probs[:, -1])
                cnn_f1 = f1_score(cnn_labels, (all_probs[:, -1] >= 0.5).astype(int))
            print(f"  AUROC (5-fold CV): {cnn_auroc:.4f}")
            print(f"  F1   (5-fold CV): {cnn_f1:.4f}")
        else:
            cnn_auroc = cnn_f1 = float("nan")

        print(f"  Params: {n_params:,}")

        single_ms = batch_ms = float("nan")
        if model is not None:
            dummy = torch.randn(1, 3, input_size, input_size).to(device)
            with torch.no_grad():
                for _ in range(20): model(dummy)
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(200): model(dummy)
                if device.type == "cuda": torch.cuda.synchronize()
            single_ms = (time.perf_counter() - t0) / 200 * 1000
            print(f"  Inference single ({device}): {single_ms:.1f} ms")

            dummy_batch = torch.randn(100, 3, input_size, input_size).to(device)
            with torch.no_grad():
                for _ in range(5): model(dummy_batch)
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(50): model(dummy_batch)
                if device.type == "cuda": torch.cuda.synchronize()
            batch_ms = (time.perf_counter() - t0) / 50 / 100 * 1000
        print(f"  Inference batch-100 (per image, {device}): {batch_ms:.1f} ms")

        results["TinyQACNN"] = {"auroc": cnn_auroc, "f1": cnn_f1,
                                 "n_params": n_params,
                                 "single_ms": single_ms, "batch100_ms": batch_ms}

    # ─────────────────────────────────────────────────────────────────────
    # 3. DINOv2+MLP and DINOv3+MLP QA classifiers
    # ─────────────────────────────────────────────────────────────────────
    for dino_key in ["dinov2", "dinov3"]:
        if dino_key not in [t.lower() for t in args.types]:
            continue
        pt_path = _QA_DIR / f"qa_{dino_key}.pt"
        if not pt_path.exists():
            print(f"\n--- {dino_key.upper()}+MLP --- SKIPPED (no checkpoint)")
            continue
        print(f"\n--- {dino_key.upper()}+MLP (frozen backbone + MLP head) ---")
        sys.path.insert(0, str(_QA_DIR))
        from train_qa_dino import DINOQAClassifier, QADINODataset
        device = torch.device(args.device)

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        model = DINOQAClassifier(backbone=dino_key).to(device)
        model.head.load_state_dict(ckpt["head_state"])
        model.eval()

        head_params  = sum(p.numel() for p in model.head.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Head params (trainable): {head_params:,}")
        print(f"  Total params (backbone+head): {total_params:,}")
        print(f"  Stored AUROC (CV): {ckpt.get('auroc_cv', float('nan')):.4f}")
        print(f"  Stored F1   (CV):  {ckpt.get('f1_cv',   float('nan')):.4f}")

        input_size = 224
        dataset = QADINODataset(img_paths, cnn_labels, input_size=input_size, augment=False)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_probs = np.zeros(len(cnn_labels))
        from torch.utils.data import DataLoader as _DL
        for _, val_idx in cv.split(img_paths, cnn_labels):
            val_ds = torch.utils.data.Subset(dataset, val_idx)
            val_dl = _DL(val_ds, batch_size=32, num_workers=4)
            fold_probs = []
            with torch.no_grad():
                for x, _ in val_dl:
                    logits = model(x.to(device))
                    fold_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_probs[val_idx] = fold_probs
        auroc = roc_auc_score(cnn_labels, all_probs)
        f1    = f1_score(cnn_labels, (all_probs >= 0.5).astype(int))
        print(f"  AUROC (5-fold CV, re-eval): {auroc:.4f}")
        print(f"  F1   (5-fold CV, re-eval):  {f1:.4f}")

        # Timing
        dummy = torch.randn(1, 3, input_size, input_size).to(device)
        with torch.no_grad():
            for _ in range(10): model(dummy)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100): model(dummy)
            if device.type == "cuda": torch.cuda.synchronize()
        single_ms = (time.perf_counter() - t0) / 100 * 1000
        print(f"  Inference single ({device}): {single_ms:.1f} ms")

        dummy_batch = torch.randn(100, 3, input_size, input_size).to(device)
        with torch.no_grad():
            for _ in range(3): model(dummy_batch)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20): model(dummy_batch)
            if device.type == "cuda": torch.cuda.synchronize()
        batch_ms = (time.perf_counter() - t0) / 20 / 100 * 1000
        print(f"  Inference batch-100 (per image, {device}): {batch_ms:.1f} ms")

        results[f"{dino_key.upper()}+MLP"] = {"auroc": auroc, "f1": f1,
                                               "n_params": head_params,
                                               "single_ms": single_ms, "batch100_ms": batch_ms}

    # ─────────────────────────────────────────────────────────────────────
    # 3b. DINOv3MultiHead QA head (binary usable/non-usable, sigmoid — same
    #     shape as GradBoost/DINO+MLP, NOT the 3-class TinyQACNN)
    # ─────────────────────────────────────────────────────────────────────
    if "multihead" in [t.lower() for t in args.types]:
        mh_ckpt_path = getattr(args, "multihead_ckpt", None) or \
            "models/multi_head/checkpoints/dinov3_base/pseudo_phase1.pt"
        mh_ckpt_path = Path(mh_ckpt_path)
        if not mh_ckpt_path.exists():
            print(f"\n--- DINOv3MultiHead QA --- SKIPPED (checkpoint not found: {mh_ckpt_path})")
        else:
            print(f"\n--- DINOv3MultiHead QA (ckpt: {mh_ckpt_path.name}) ---")
            from multi_head.train_multihead import DINOv3MultiHead
            device = torch.device(args.device)
            mh_model = DINOv3MultiHead("dinov3_base").to(device)
            mh_model.eval()
            state = torch.load(mh_ckpt_path, map_location=device, weights_only=False)
            if "state_dict" in state:
                sd = {k.replace("model.", ""): v for k, v in state["state_dict"].items() if k.startswith("model.")}
                mh_model.load_state_dict(sd, strict=False)
            else:
                print("  NOTE: Phase-1-only checkpoint has no qa_head weights (untrained, random init) — "
                      "QA numbers below are meaningless until a Phase-2 (GT) checkpoint is used.")

            # binary labels: use df["label"] (or qa_usable), matched to img_paths already built above
            mh_labels, mh_paths = [], []
            for _, row in df.iterrows():
                p = _IMG_DIR / row["camera"] / f"{row['stem']}.jpg"
                if p.exists():
                    mh_paths.append(p)
                    mh_labels.append(int(row["label"]))
            mh_labels = np.array(mh_labels, dtype=np.int64)

            import torchvision.transforms.v2 as T_mh
            mh_size = 512
            transform = T_mh.Compose([
                T_mh.Resize((mh_size, mh_size)),
                T_mh.ToImage(), T_mh.ToDtype(torch.float32, scale=True),
                T_mh.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            mh_probs = []
            with torch.no_grad():
                for i in range(0, len(mh_paths), 32):
                    batch = [transform(Image.open(p).convert("RGB")) for p in mh_paths[i:i+32]]
                    x = torch.stack(batch).to(device)
                    logits = mh_model.forward_qa(x)
                    mh_probs.append(torch.sigmoid(logits).cpu().numpy())
            mh_probs = np.concatenate(mh_probs)
            try:
                mh_auroc = roc_auc_score(mh_labels, mh_probs)
            except ValueError:
                mh_auroc = float("nan")
            mh_f1 = f1_score(mh_labels, (mh_probs >= 0.5).astype(int))
            print(f"  AUROC: {mh_auroc:.4f}")
            print(f"  F1:    {mh_f1:.4f}")
            results["DINOv3MultiHead_QA"] = {"auroc": mh_auroc, "f1": mh_f1}

    # ─────────────────────────────────────────────────────────────────────
    # 4. PatchCore (per-camera anomaly models → image-level QA AUROC)
    # ─────────────────────────────────────────────────────────────────────
    _PC_RESULTS = Path("models/anomaly_snow/results/patchcore")
    _PC_IMG_DIR = Path("data/balanced_snow")
    if _PC_RESULTS.exists() and "patchcore" in [t.lower() for t in args.types]:
        print("\n--- PatchCore (per-camera, anomaly → QA) ---")

        # Patch torch.load for anomalib compatibility
        _orig_load = torch.load
        def _patched_load(*a, **kw):
            kw["weights_only"] = False
            return _orig_load(*a, **kw)
        torch.load = _patched_load

        try:
            from anomalib.models import Patchcore
            from anomalib.data import Folder
            from anomalib.engine import Engine
        except ImportError:
            print("  ERROR: anomalib not installed. Skip PatchCore eval.")
            torch.load = _orig_load
        else:
            import inspect

            # Load QA labels
            cv_pred_csv = _QA_DIR / "qa_cv_predictions.csv"
            if not cv_pred_csv.exists():
                print("  ERROR: qa_cv_predictions.csv not found.")
            else:
                qa_df = pd.read_csv(cv_pred_csv)[["camera", "stem", "label"]]
                # stem → label lookup per camera
                qa_lookup = {(r.camera, r.stem): r.label for r in qa_df.itertuples()}

                cameras = sorted([d.name for d in _PC_RESULTS.iterdir() if d.is_dir()])
                all_scores = []
                all_labels = []
                timing_done = False
                single_ms = batch_ms = None

                for cam in cameras:
                    ckpt = (_PC_RESULTS / cam / "Patchcore" / cam /
                            "latest" / "weights" / "lightning" / "model.ckpt")
                    if not ckpt.exists():
                        print(f"  {cam}: no checkpoint — skip")
                        continue

                    # Find QA images for this camera that exist on disk
                    cam_qa = qa_df[qa_df["camera"] == cam]
                    img_paths = []
                    img_labels = []
                    for _, row in cam_qa.iterrows():
                        for ext in ("jpg", "png"):
                            p = _PC_IMG_DIR / cam / f"{row['stem']}.{ext}"
                            if p.exists():
                                img_paths.append(p)
                                img_labels.append(row["label"])
                                break

                    if not img_paths:
                        print(f"  {cam}: no QA images found on disk — skip")
                        continue

                    print(f"  {cam}: {len(img_paths)} QA images")

                    model = Patchcore.load_from_checkpoint(str(ckpt))
                    model = model.eval().to(torch.device(args.device))

                    # Run inference image by image (anomalib model.forward needs tensor)
                    import torchvision.transforms.v2 as T
                    transform = T.Compose([
                        T.Resize((256, 256)),
                        T.ToImage(), T.ToDtype(torch.float32, scale=True),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ])

                    scores = []
                    with torch.no_grad():
                        for p in img_paths:
                            img = Image.open(p).convert("RGB")
                            x = transform(img).unsqueeze(0).to(torch.device(args.device))
                            out = model(x)
                            # anomalib v2 returns InferenceBatch; v1 returns dict
                            if hasattr(out, "pred_score"):
                                score = out.pred_score
                            elif hasattr(out, "anomaly_map"):
                                score = out.anomaly_map
                            elif isinstance(out, dict):
                                score = out.get("pred_score", out.get("anomaly_map", torch.tensor(0.0)))
                            else:
                                score = out
                            if hasattr(score, "mean"):
                                score = float(score.mean().cpu())
                            else:
                                score = float(torch.tensor(score).mean().cpu())
                            scores.append(score)

                    all_scores.extend(scores)
                    all_labels.extend(img_labels)

                    # Timing on first camera
                    if not timing_done and len(img_paths) >= 1:
                        dummy = torch.randn(1, 3, 256, 256).to(torch.device(args.device))
                        with torch.no_grad():
                            for _ in range(5): model(dummy)
                            t0 = time.perf_counter()
                            for _ in range(50): model(dummy)
                        single_ms = (time.perf_counter() - t0) / 50 * 1000

                        # Batch-100: PatchCore KNN scales with batch — measure total/100
                        dummy_b = torch.randn(100, 3, 256, 256).to(torch.device(args.device))
                        with torch.no_grad():
                            for _ in range(2): model(dummy_b)
                            t0 = time.perf_counter()
                            for _ in range(10): model(dummy_b)
                        batch_ms = (time.perf_counter() - t0) / 10 / 100 * 1000
                        print(f"  Inference single: {single_ms:.1f} ms")
                        print(f"  Inference batch-100 (per image): {batch_ms:.1f} ms")
                        timing_done = True

                torch.load = _orig_load

                if all_scores:
                    all_scores = np.array(all_scores)
                    all_labels = np.array(all_labels)
                    # PatchCore: high score = anomalous = non-usable (label=0)
                    # Invert: AUROC for "usable" = 1 - anomaly score
                    pc_auroc = roc_auc_score(all_labels, -all_scores)
                    pc_f1 = f1_score(all_labels, (all_scores < np.median(all_scores)).astype(int))
                    print(f"  AUROC (QA usable/non-usable): {pc_auroc:.4f}")
                    print(f"  F1:                           {pc_f1:.4f}")
                    print(f"  N images: {len(all_scores)}")
                    results["PatchCore"] = {
                        "auroc": pc_auroc, "f1": pc_f1,
                        "single_ms": single_ms or 0.0,
                        "batch100_ms": batch_ms or 0.0,
                    }

    # ─────────────────────────────────────────────────────────────────────
    # LakeICE OOD eval: TinyQACNN on LakeICE dataset
    # Label: snow_frac > 5% → usable=1, else usable=0
    # Sihl: RGB (255,0,0)=Snow; StMoritz: grayscale 127=Snow
    # ─────────────────────────────────────────────────────────────────────
    if "lakeice_ood" in [t.lower() for t in args.types]:
        print("\n--- LakeICE OOD Evaluation ---")
        import torchvision.transforms.v2 as T_v2

        _LAKEICE_DIR = Path("datasets/LakeICE")
        if not _LAKEICE_DIR.exists():
            print(f"  ERROR: {_LAKEICE_DIR} not found.")
        else:
            SNOW_THRESH = 0.05

            def lakeice_snow_label(label_path):
                """Return (snow_frac, usable) for a LakeICE label PNG."""
                arr = np.array(Image.open(label_path))
                if arr.ndim == 2:
                    # StMoritz grayscale: 127=snow
                    snow = (arr == 127)
                else:
                    arr = arr[:, :, :3]  # drop alpha
                    snow = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 0)
                frac = snow.mean()
                return float(frac), int(frac > SNOW_THRESH)

            # Collect all (img_path, label_path, usable) triplets
            lakeice_samples = []
            for label_path in sorted(_LAKEICE_DIR.rglob("Labels/*.png")):
                img_path = Path(str(label_path).replace("Labels", "Data"))
                if not img_path.exists():
                    # Try jpg
                    img_path = img_path.with_suffix(".jpg")
                if not img_path.exists():
                    continue
                _, usable = lakeice_snow_label(label_path)
                lakeice_samples.append((img_path, label_path, usable))

            li_labels = np.array([s[2] for s in lakeice_samples])
            print(f"  Loaded {len(lakeice_samples)} LakeICE images "
                  f"({li_labels.sum()} usable / {(1-li_labels).sum()} non-usable, "
                  f"thresh={SNOW_THRESH*100:.0f}%)")

            li_results = {}

            # ── TinyQACNN ──────────────────────────────────────────────
            cnn_ckpt = _QA_DIR / "qa_cnn.pt"
            if cnn_ckpt.exists():
                device = torch.device(args.device)
                cnn_model = torch.load(cnn_ckpt, map_location=device, weights_only=False)
                cnn_model = cnn_model.to(device).eval()
                input_size = 128

                transform = T_v2.Compose([
                    T_v2.Resize((input_size, input_size)),
                    T_v2.ToImage(), T_v2.ToDtype(torch.float32, scale=True),
                    T_v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])

                li_probs = []
                bs = 32
                img_paths_li = [s[0] for s in lakeice_samples]
                with torch.no_grad():
                    for i in range(0, len(img_paths_li), bs):
                        batch = []
                        for p in img_paths_li[i:i+bs]:
                            img = Image.open(p).convert("RGB")
                            batch.append(transform(img))
                        x = torch.stack(batch).to(device)
                        logits = cnn_model(x)
                        li_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

                li_probs = np.array(li_probs)
                li_auroc = roc_auc_score(li_labels, li_probs)
                li_f1 = f1_score(li_labels, (li_probs >= 0.5).astype(int))
                print(f"  TinyQACNN: AUROC={li_auroc:.4f}, F1={li_f1:.4f}")
                li_results["TinyQACNN_lakeice"] = {"auroc": li_auroc, "f1": li_f1}

            print("\n  LakeICE OOD Summary:")
            for method, m in li_results.items():
                f1_str = f"{m['f1']:.4f}" if not (isinstance(m['f1'], float) and m['f1'] != m['f1']) else "—"
                print(f"    {method:<30} AUROC={m['auroc']:.4f}  F1={f1_str}")

    # ─────────────────────────────────────────────────────────────────────
    # QA OOD eval: held-out cameras CM00054, CM00071, CM00115
    # ─────────────────────────────────────────────────────────────────────
    if "qa_ood" in [t.lower() for t in args.types]:
        print("\n--- QA OOD (CM00054 + CM00071 + CM00115, held-out cameras) ---")
        import json as _json
        import torchvision.transforms.v2 as T_qa

        _ANN_DIR = Path("data/annotations")
        _OOD_CAMS = {"CM00054": "54.json", "CM00067": "67.json", "CM00071": "71.json", "CM00115": "115.json"}
        _OOD_IMG_DIRS = [Path("data")]

        _QA_CLASS_NAMES = ["clean_no_snow", "snow", "non_usable"]

        def _parse_qa_label3(task):
            """Parse Label Studio task → 3-class label (0=clean_no_snow, 1=snow, 2=non_usable)."""
            anns = task.get("annotations", [])
            if not anns or not anns[0].get("result"): return None
            flat_all = set()
            for r in anns[0]["result"]:
                v = r.get("value", {})
                for key in ("taxonomy", "choices"):
                    tags = v.get(key)
                    if not tags: continue
                    for entry in tags:
                        if isinstance(entry, list): flat_all.update(entry)
                        else: flat_all.add(entry)
            if "Usable" in flat_all: return 1
            if "Non-Usable" in flat_all:
                reasons = flat_all - {"Non-Usable"}
                if reasons <= {"No Snow"}: return 0
                return 2
            return None

        ood_img_paths, ood_labels = [], []
        for cam, fname in _OOD_CAMS.items():
            ann_path = _ANN_DIR / fname
            if not ann_path.exists(): continue
            with open(ann_path) as f:
                tasks = _json.load(f)
            for task in tasks:
                label = _parse_qa_label3(task)
                if label is None: continue
                url = task["data"]["image"]
                stem = Path(url.rstrip("/").split("/")[-1]).stem
                found = False
                for base in [Path("data") / cam, Path("data") / f"_EXCLUDED_{cam}"]:
                    for ext in ("jpg", "png"):
                        p = base / f"{stem}.{ext}"
                        if p.exists():
                            ood_img_paths.append(p)
                            ood_labels.append(label)
                            found = True
                            break
                    if found: break

        ood_labels = np.array(ood_labels, dtype=np.int64)
        counts3 = {c: int((ood_labels == c).sum()) for c in range(3)}
        print(f"  Loaded {len(ood_labels)} OOD QA images — "
              + "  ".join(f"{_QA_CLASS_NAMES[c]}={counts3[c]}" for c in range(3)))

        n_classes_present = sum(1 for c in range(3) if counts3[c] > 0)
        if len(ood_labels) > 0 and n_classes_present >= 2:
            from sklearn.metrics import roc_auc_score as _roc, f1_score as _f1, classification_report as _cr
            # TinyQACNN
            cnn_ckpt = _QA_DIR / "qa_cnn.pt"
            if cnn_ckpt.exists():
                device = torch.device(args.device)
                try:
                    cnn_model = torch.jit.load(str(cnn_ckpt), map_location=device)
                except Exception:
                    cnn_model = torch.load(cnn_ckpt, map_location=device, weights_only=False)
                cnn_model = cnn_model.to(device).eval()
                transform = T_qa.Compose([
                    T_qa.Resize((128, 128)),
                    T_qa.ToImage(), T_qa.ToDtype(torch.float32, scale=True),
                    T_qa.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
                ood_probs = []
                with torch.no_grad():
                    for i in range(0, len(ood_img_paths), 32):
                        batch = [transform(Image.open(p).convert("RGB")) for p in ood_img_paths[i:i+32]]
                        x = torch.stack(batch).to(device)
                        probs = torch.softmax(cnn_model(x), dim=1).cpu().numpy()
                        ood_probs.append(probs)
                ood_probs = np.vstack(ood_probs)   # (N, 3)
                ood_preds = ood_probs.argmax(axis=1)
                try:
                    cnn_ood_auroc = _roc(ood_labels, ood_probs, multi_class="ovr", average="macro")
                except ValueError:
                    cnn_ood_auroc = float("nan")
                cnn_ood_f1 = _f1(ood_labels, ood_preds, average="macro")
                print(f"  TinyQACNN OOD: AUROC={cnn_ood_auroc:.4f}, F1={cnn_ood_f1:.4f}")
                print(_cr(ood_labels, ood_preds, target_names=_QA_CLASS_NAMES, zero_division=0))

            # DINOv3MultiHead QA OOD — binary (sigmoid), collapse 3-class GT:
            # usable = clean_no_snow(0) or snow(1), non_usable = 2.
            if "multihead" in [t.lower() for t in args.types]:
                mh_ckpt_path = Path(getattr(args, "multihead_ckpt", None) or
                    "models/multi_head/checkpoints/dinov3_base/pseudo_phase1.pt")
                if not mh_ckpt_path.exists():
                    print(f"  DINOv3MultiHead QA OOD: skipped (checkpoint not found: {mh_ckpt_path})")
                else:
                    from multi_head.train_multihead import DINOv3MultiHead
                    import torchvision.transforms.v2 as T_mh_ood
                    device = torch.device(args.device)
                    mh_model = DINOv3MultiHead("dinov3_base").to(device)
                    mh_model.eval()
                    state = torch.load(mh_ckpt_path, map_location=device, weights_only=False)
                    if "state_dict" in state:
                        sd = {k.replace("model.", ""): v for k, v in state["state_dict"].items() if k.startswith("model.")}
                        mh_model.load_state_dict(sd, strict=False)
                        ood_labels_bin = (ood_labels != 2).astype(np.int64)  # usable=1, non_usable=0
                        transform = T_mh_ood.Compose([
                            T_mh_ood.Resize((512, 512)),
                            T_mh_ood.ToImage(), T_mh_ood.ToDtype(torch.float32, scale=True),
                            T_mh_ood.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ])
                        mh_ood_probs = []
                        with torch.no_grad():
                            for i in range(0, len(ood_img_paths), 32):
                                batch = [transform(Image.open(p).convert("RGB")) for p in ood_img_paths[i:i+32]]
                                x = torch.stack(batch).to(device)
                                mh_ood_probs.append(torch.sigmoid(mh_model.forward_qa(x)).cpu().numpy())
                        mh_ood_probs = np.concatenate(mh_ood_probs)
                        try:
                            mh_ood_auroc = roc_auc_score(ood_labels_bin, mh_ood_probs)
                        except ValueError:
                            mh_ood_auroc = float("nan")
                        mh_ood_f1 = f1_score(ood_labels_bin, (mh_ood_probs >= 0.5).astype(int))
                        print(f"  DINOv3MultiHead QA OOD: AUROC={mh_ood_auroc:.4f}, F1={mh_ood_f1:.4f}")
                    else:
                        print("  DINOv3MultiHead QA OOD: skipped (Phase-1-only checkpoint, qa_head untrained)")

            # DINOv2+MLP and DINOv3+MLP OOD
            # These models are binary (sigmoid output); skip if labels are 3-class
            n_qa_classes = len(np.unique(ood_labels))
            if n_qa_classes > 2:
                print("  DINO+MLP OOD: skipped (binary models, 3-class labels — retrain qa_dino*.pt as 3-class)")
            else:
                for dino_key in ["dinov2", "dinov3"]:
                    dino_pt = _QA_DIR / f"qa_{dino_key}.pt"
                    if not dino_pt.exists(): continue
                    sys.path.insert(0, str(_QA_DIR))
                    from train_qa_dino import DINOQAClassifier, QADINODataset
                    from torch.utils.data import DataLoader as _DL2
                    device = torch.device(args.device)
                    dino_model = DINOQAClassifier(backbone=dino_key).to(device)
                    dino_ckpt = torch.load(dino_pt, map_location="cpu", weights_only=False)
                    dino_model.head.load_state_dict(dino_ckpt["head_state"])
                    dino_model.eval()
                    ood_ds = QADINODataset(ood_img_paths, ood_labels.tolist(), input_size=224, augment=False)
                    ood_dl = _DL2(ood_ds, batch_size=16, num_workers=2)
                    dino_probs = []
                    with torch.no_grad():
                        for x, _ in ood_dl:
                            dino_probs.extend(torch.sigmoid(dino_model(x.to(device))).cpu().numpy().tolist())
                    dino_probs = np.array(dino_probs)
                    d_auroc = roc_auc_score(ood_labels, dino_probs)
                    d_f1    = f1_score(ood_labels, (dino_probs >= 0.5).astype(int))
                    print(f"  {dino_key.upper()}+MLP OOD: AUROC={d_auroc:.4f}, F1={d_f1:.4f}")

        else:
            print("  Skipping — insufficient class balance in OOD set")

    # ─────────────────────────────────────────────────────────────────────
    # Summary table
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<25} {'AUROC':<10} {'F1':<10} {'Single (ms)':<15} {'Batch-100/img (ms)'}")
    print("-" * 80)
    for method, m in results.items():
        single = m.get("single_ms")
        batch100 = m.get("batch100_ms")
        single_s = f"{single:.2f}" if single is not None else "—"
        batch_s = f"{batch100:.2f}" if batch100 is not None else "—"
        print(f"{method:<25} {m['auroc']:<10.4f} {m['f1']:<10.4f} "
              f"{single_s:<15} {batch_s}")


# =============================================================================
# OOD-PER-CAMERA — ground IoU, snow IoU, QA AUROC/F1, per individual OOD
# camera (not pooled into one n=495 block like the other OOD types above).
# Backs the paper's claim that CM00067 is consistently the hardest camera
# in the cohort (Table tab:ood-per-camera).
# =============================================================================

OODCAM_CAMERAS = ["CM00054", "CM00067", "CM00071", "CM00115"]
OODCAM_JSON_MAP = {
    "CM00054": "data/annotations/54.json",
    "CM00067": "data/annotations/67.json",
    "CM00071": "data/annotations/71.json",
    "CM00115": "data/annotations/115.json",
}

OODCAM_GROUND_CKPT = "models/student_ground/checkpoints/finetune_mobilenet_kfold_1_20260610_230732/best-epoch=48-iouval_iou=0.916.ckpt"
OODCAM_SNOW_CKPT = "models/student_snow/checkpoints/snow_gt_mobilenet_20260522_052052/best-epoch=65-iouval_iou=0.821.ckpt"

OODCAM_DATA_DIR = Path("data")


def oodcam_eval_seg_per_camera(task, ckpt_path, label_class, device):
    from student_ground.models import build_model
    from student_ground.configs import get_config

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    variant = checkpoint.get("hyper_parameters", {}).get("model_variant", "mobilenet")
    model = build_model(variant)
    sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    cfg = get_config(variant)
    input_size = cfg.get("input_size", 512)

    results = {}
    for cam, fname in OODCAM_JSON_MAP.items():
        p = Path(fname)
        if not p.exists():
            results[cam] = None
            continue
        gt_masks = load_gt_masks(p, [cam], label_class=label_class)
        evaluator = Evaluator(device)
        count = 0
        for (c, stem), target in gt_masks.items():
            img_path = OODCAM_DATA_DIR / c / f"{stem}.jpg"
            if not img_path.exists():
                img_path = OODCAM_DATA_DIR / f"_EXCLUDED_{c}" / f"{stem}.jpg"
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


def oodcam_parse_qa_label3(task):
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


def oodcam_eval_qa_per_camera(device):
    import torchvision.transforms.v2 as T

    cnn_ckpt = Path("qa") / "checkpoints" / "qa_cnn.pt"
    model = torch.jit.load(str(cnn_ckpt), map_location=device).eval()
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    from sklearn.metrics import roc_auc_score, f1_score

    results = {}
    for cam, fname in OODCAM_JSON_MAP.items():
        p = Path(fname)
        if not p.exists():
            results[cam] = None
            continue
        with open(p) as f:
            tasks = json.load(f)
        img_paths, labels = [], []
        for task in tasks:
            label = oodcam_parse_qa_label3(task)
            if label is None:
                continue
            url = task["data"]["image"]
            stem = Path(url.rstrip("/").split("/")[-1]).stem
            for base in [OODCAM_DATA_DIR / cam, OODCAM_DATA_DIR / f"_EXCLUDED_{cam}"]:
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


def eval_oodcam(args):
    device = torch.device(args.device)

    print("=== Ground (per-camera) ===")
    ground_results = oodcam_eval_seg_per_camera("ground", OODCAM_GROUND_CKPT, "Ground Mano", device)
    for cam in OODCAM_CAMERAS:
        r = ground_results[cam]
        if r is None:
            print(f"  {cam}: no data")
        else:
            m, n = r
            print(f"  {cam}: IoU={m['iou']:.4f} Dice={m['dice']:.4f} n={n}")

    print("\n=== Snow (per-camera) ===")
    snow_results = oodcam_eval_seg_per_camera("snow", OODCAM_SNOW_CKPT, "Snow Mano", device)
    for cam in OODCAM_CAMERAS:
        r = snow_results[cam]
        if r is None:
            print(f"  {cam}: no data")
        else:
            m, n = r
            print(f"  {cam}: IoU={m['iou']:.4f} Dice={m['dice']:.4f} n={n}")

    print("\n=== QA (per-camera) ===")
    qa_results = oodcam_eval_qa_per_camera(device)
    for cam in OODCAM_CAMERAS:
        r = qa_results[cam]
        if r is None:
            print(f"  {cam}: insufficient class balance")
        else:
            auroc, f1, n = r
            print(f"  {cam}: AUROC={auroc:.4f} F1={f1:.4f} n={n}")


def main():
    parser = argparse.ArgumentParser(description="Unified segmentation evaluation")
    parser.add_argument("--task", type=str, required=True,
                        choices=["ground", "snow", "smoother", "qa", "ood-per-camera"])
    parser.add_argument("--types", nargs="+", default=None, help="Evaluation types")
    parser.add_argument("--breakdown", action="store_true", help="Show per-camera breakdown")
    parser.add_argument("--val-only", action="store_true", help="Snow: evaluate on val split only")
    parser.add_argument("--multihead-ckpt", type=str, default=None,
                        help="Override checkpoint for --types multihead (default: "
                             "checkpoints/dinov3_base/pseudo_phase1.pt). Accepts a Phase 1 "
                             "pseudo_epochNNN.pt / pseudo_phase1.pt or a Phase 2 Lightning .ckpt.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.types is None:
        if args.task == "ground": args.types = ["teachers", "ensembles", "baselines", "students"]
        elif args.task == "snow": args.types = ["vlms", "baselines"]
        elif args.task == "qa": args.types = ["gradboost", "cnn"]
        else: args.types = ["all"]

    if args.task == "qa":
        eval_qa(args)
    elif args.task == "ground":
        eval_ground(args)
        if args.breakdown:
            cameras = ["CF00001", "CM00009", "CM00014", "CM00070", "CM00075", "CM00080", "CM00102", "CM00108", "CM00112"]
            gt_json = Path("data/annotations/ground_annotations.json")
            gt_masks = load_gt_masks(gt_json, cameras + ["CM00054", "CM00067", "CM00071", "CM00115"], label_class="Ground Mano")
            best_method_dir = Path("preprocessing/segmentation/ensemble_output/majority_3")
            def pred_func(cam, stem):
                return best_method_dir / cam / f"{stem}_ground.png"
            eval_per_camera(gt_masks, pred_func, device=args.device, task="ground")
    elif args.task == "snow":
        eval_snow(args)
    elif args.task == "ood-per-camera":
        eval_oodcam(args)
    else:
        eval_smoother(args)


if __name__ == "__main__":
    main()
