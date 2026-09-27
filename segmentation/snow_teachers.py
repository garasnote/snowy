#!/usr/bin/env python3
"""
Snow segmentation teachers — six zero-shot VLM / promptable-segmentation
methods whose outputs are combined by majority vote (snow_utils.py:
ensemble) into the snow pseudo-labels used to train the snow student
models (Methods, "Snow Segmentation").

Consolidated from clipseg_snow_segmentation.py, florence2_snow_segmentation.py,
grounded_sam2_snow_segmentation.py, sam3_snow_segmentation.py,
yoloe_snow_segmentation.py, seggpt_snow_segmentation.py. Each teacher keeps
its original CLI flags, now behind a subcommand:

    python snow_teachers.py clipseg      --data-dir ./data
    python snow_teachers.py florence2    --data-dir ./data
    python snow_teachers.py groundedsam2 --data-dir ./data
    python snow_teachers.py sam3         --data-dir ./data
    python snow_teachers.py yoloe        --data-dir ./data
    python snow_teachers.py seggpt       --data-dir ./data --ann-dir ./data/annotations

Run `python snow_teachers.py <teacher> --help` for that teacher's full
option list (e.g. seggpt's few-shot prompt selection, sam3's ground-mask
clipping). SAMWrapper (formerly sam_wrapper.py) lives here too, used only
by the sam3 subcommand.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import Optional

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: sam3 not installed. SAMWrapper will fail if used. Error: {e}")
    SAM3_AVAILABLE = False

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import WORKSPACE_DIR


def _add_common_args(p, default_output_dir):
    p.add_argument("--data-dir", type=str, required=True,
                    help="Data directory with camera folders")
    p.add_argument("--cameras", nargs="+", default=[],
                    help="Specific cameras (default: all)")
    p.add_argument("--output-dir", type=str, default=default_output_dir,
                    help="Output directory")
    p.add_argument("--sample", type=int, default=0,
                    help="Images to sample per camera (0=all)")
    p.add_argument("--debug", action="store_true",
                    help="Save overview figures")


def _discover_cameras(data_dir, cameras):
    if cameras:
        return cameras
    return sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)
    ])


def _sample_jpgs(cam_dir, sample):
    jpgs = sorted(cam_dir.glob("*.jpg"))
    if sample > 0 and len(jpgs) > sample:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(jpgs), size=sample, replace=False)
        jpgs = [jpgs[i] for i in sorted(indices)]
    return jpgs


def _write_summary_csv(results, output_dir, name, fieldnames):
    if not results:
        return
    summary_path = output_dir / name
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary: {summary_path}")


def _overlay_blue(image_arr, mask):
    """Shared blue-on-snow / gray-on-background overlay used by every
    snow teacher's debug visualization."""
    overlay = image_arr.copy().astype(np.float32)
    mask_bool = mask > 0
    overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 150, 255]) * 0.5
    overlay[~mask_bool] = overlay[~mask_bool] * 0.5 + np.array([100, 100, 100]) * 0.5
    return overlay.clip(0, 255).astype(np.uint8)


# =============================================================================
# CLIPSEG — text-prompted dense prediction
# =============================================================================
# Lightweight (~150M params), very different feature space from SAM/YOLO/
# Florence -- kept in the ensemble mainly for diversity.

def clipseg_load_model(device="cuda"):
    """Load CLIPSeg model and processor."""
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    model_id = "CIDAS/clipseg-rd64-refined"
    print(f"Loading {model_id}...")
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id).eval().to(device)
    print(f"  CLIPSeg loaded on {device}")
    return model, processor


def clipseg_segment(image, model, processor, text_prompt, device="cuda", threshold=0.5):
    """Run CLIPSeg and threshold logits into binary mask.

    CLIPSeg outputs a single-channel logit map at 352x352 (fixed); we
    sigmoid, threshold, then resize to original image dimensions.
    """
    inputs = processor(
        text=[text_prompt], images=[image], return_tensors="pt", padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze(0)
    probs = torch.sigmoid(logits).cpu().numpy()

    h, w = image.size[1], image.size[0]
    probs_resized = np.array(
        Image.fromarray(probs).resize((w, h), Image.BILINEAR)
    )

    binary_mask = (probs_resized >= threshold).astype(np.uint8)
    return binary_mask, probs_resized


def clipseg_visualize(image, mask, probs, output_path, title=""):
    """Save overview figure with heatmap and mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(probs, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("CLIPSeg Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    snow_pct = mask.sum() / mask.size * 100
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Snow Mask ({snow_pct:.1f}%)")
    axes[2].axis("off")

    axes[3].imshow(_overlay_blue(np.array(image), mask))
    axes[3].set_title("Overlay (blue=snow)")
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def clipseg_process_image(image_path, model, processor, text_prompt, output_dir,
                           device="cuda", threshold=0.5, debug=False):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    snow_mask, probs = clipseg_segment(
        image, model, processor, text_prompt, device=device, threshold=threshold,
    )

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        clipseg_visualize(image, snow_mask, probs, cam_out / f"{stem}_overview.png",
                          title=f"{camera} - {stem}")

    snow_pct = snow_mask.sum() / snow_mask.size * 100
    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name, "snow_pct": round(snow_pct, 2),
    }


def add_clipseg_subparser(subparsers):
    p = subparsers.add_parser("clipseg", help="CLIPSeg text-prompted dense prediction")
    _add_common_args(p, str(WORKSPACE_DIR / "segmentation" / "snow" / "clipseg"))
    p.add_argument("--text-prompt", type=str, default="snow",
                    help="Text prompt for segmentation (default: snow)")
    p.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for binary mask (default: 0.5)")
    return p


def run_clipseg(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("CLIPSEG - SNOW SEGMENTATION (text-prompted dense prediction)")
    print("=" * 70)
    print(f"  Model:      CIDAS/clipseg-rd64-refined")
    print(f"  Prompt:     {args.text_prompt}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")

    model, processor = clipseg_load_model(device=device)
    cameras = _discover_cameras(data_dir, args.cameras)
    print(f"\nCameras: {cameras}")

    results = []
    for cam in cameras:
        cam_dir = data_dir / cam
        if not cam_dir.exists():
            print(f"  WARNING: {cam_dir} does not exist, skipping")
            continue
        jpgs = _sample_jpgs(cam_dir, args.sample)
        print(f"\n  {cam}: processing {len(jpgs)} images")
        for img_path in tqdm(jpgs, desc=f"  {cam}", leave=True):
            result = clipseg_process_image(
                img_path, model, processor, args.text_prompt, output_dir,
                device=device, threshold=args.threshold, debug=args.debug,
            )
            if result:
                results.append(result)

    _write_summary_csv(results, output_dir, "clipseg_snow_summary.csv",
                        ["image", "camera", "filename", "snow_pct"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# FLORENCE-2 — referring expression segmentation
# =============================================================================

def florence2_load_model(model_size="large", device="cuda"):
    """Load Florence-2 model and processor."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = f"microsoft/Florence-2-{model_size}"
    print(f"Loading {model_id}...")

    # Clear stale cached custom code from prior runs
    import shutil
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "microsoft"
    for d in cache_dir.glob("Florence*"):
        shutil.rmtree(d, ignore_errors=True)
        print(f"  Cleared stale cache: {d.name}")

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    # attn_implementation="eager" works around a transformers regression
    # (>=4.55, https://github.com/huggingface/transformers/issues/39974):
    # Florence-2's trust_remote_code model class predates transformers'
    # newer SDPA-capability auto-detection and doesn't define
    # `_supports_sdpa`, which that detection path requires. "eager" skips
    # it entirely. No known transformers version satisfies both this and
    # DINOv3's own >=4.56.0 floor, so this is the fix, not a version pin.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, trust_remote_code=True,
        attn_implementation="eager",
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print(f"  Florence-2-{model_size} loaded on {device}")
    return model, processor


def florence2_run(model, processor, image, text_prompt, device="cuda"):
    """Run Florence-2 referring expression segmentation. Returns parsed
    answer dict with polygons."""
    task = "<REFERRING_EXPRESSION_SEGMENTATION>"
    prompt = task + text_prompt

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    with torch.inference_mode():
        # use_cache=False works around a second transformers regression on
        # top of the attn_implementation one above: newer transformers
        # passes a Cache object instead of the legacy tuple format Florence-
        # 2's remote code's prepare_inputs_for_generation expects, hitting
        # `past_key_values[0][0].shape` on a None entry mid-generation.
        # Disabling the KV cache sidesteps that code path entirely — slower
        # (no caching across steps) but correct.
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, num_beams=3, do_sample=False, use_cache=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task, image_size=(image.width, image.height),
    )
    return parsed


def florence2_polygons_to_mask(parsed_answer, width, height):
    """Convert Florence-2 polygon output to binary mask."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    key = "<REFERRING_EXPRESSION_SEGMENTATION>"
    if key not in parsed_answer:
        return np.zeros((height, width), dtype=np.uint8)

    result = parsed_answer[key]
    polygons = result.get("polygons", [])

    for polygon in polygons:
        if isinstance(polygon, list) and len(polygon) > 0:
            coords = polygon[0] if isinstance(polygon[0], list) else polygon
            if len(coords) >= 6:
                points = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
                draw.polygon(points, fill=255)

    return (np.array(mask) > 127).astype(np.uint8)


def florence2_visualize(image, mask, output_path, title=""):
    """Save overview figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_arr = np.array(image)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_arr)
    axes[0].set_title("Original")
    axes[0].axis("off")

    snow_pct = mask.sum() / mask.size * 100
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Snow Mask ({snow_pct:.1f}%)")
    axes[1].axis("off")

    axes[2].imshow(_overlay_blue(img_arr, mask))
    axes[2].set_title("Overlay (blue=snow)")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def florence2_process_image(image_path, model, processor, text_prompt, output_dir,
                             device="cuda", debug=False):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    parsed = florence2_run(model, processor, image, text_prompt, device=device)
    snow_mask = florence2_polygons_to_mask(parsed, image.width, image.height)

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        florence2_visualize(image, snow_mask, cam_out / f"{stem}_overview.png",
                            title=f"{camera} - {stem}")

    snow_pct = snow_mask.sum() / snow_mask.size * 100
    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name, "snow_pct": round(snow_pct, 2),
    }


def add_florence2_subparser(subparsers):
    p = subparsers.add_parser("florence2", help="Florence-2 referring expression segmentation")
    _add_common_args(p, str(WORKSPACE_DIR / "segmentation" / "snow" / "florence2"))
    p.add_argument("--model-size", choices=["base", "large"], default="large",
                    help="Florence-2 model size (default: large)")
    p.add_argument("--text-prompt", type=str, default="snow",
                    help="Text prompt for segmentation (default: snow)")
    return p


def run_florence2(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("FLORENCE-2 - SNOW SEGMENTATION (referring expression)")
    print("=" * 70)
    print(f"  Model:       Florence-2-{args.model_size}")
    print(f"  Text prompt: {args.text_prompt}")
    print(f"  Device:      {device}")
    print(f"  Output:      {output_dir}")

    model, processor = florence2_load_model(args.model_size, device=device)
    cameras = _discover_cameras(data_dir, args.cameras)
    print(f"\nCameras: {cameras}")

    results = []
    for cam in cameras:
        cam_dir = data_dir / cam
        if not cam_dir.exists():
            print(f"  WARNING: {cam_dir} does not exist, skipping")
            continue
        jpgs = _sample_jpgs(cam_dir, args.sample)
        print(f"\n  {cam}: processing {len(jpgs)} images")
        for img_path in tqdm(jpgs, desc=f"  {cam}", leave=True):
            result = florence2_process_image(
                img_path, model, processor, args.text_prompt, output_dir,
                device=device, debug=args.debug,
            )
            if result:
                results.append(result)

    _write_summary_csv(results, output_dir, "florence2_snow_summary.csv",
                        ["image", "camera", "filename", "snow_pct"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# GROUNDED SAM2 — GroundingDINO detection + SAM2 mask refinement
# =============================================================================

def groundedsam2_load_grounding_dino(device="cuda"):
    """Load GroundingDINO model for open-set detection."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).eval().to(device)
    print(f"  GroundingDINO-tiny loaded on {device}")
    return model, processor


def groundedsam2_load_sam2(device="cuda"):
    """Load SAM2 model for mask refinement."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("Loading SAM2 (hiera-large)...")
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large", device=device
    )
    print(f"  SAM2 loaded on {device}")
    return predictor


def groundedsam2_detect_boxes(image, model, processor, device="cuda",
                               box_threshold=0.3, text_threshold=0.25):
    """Detect snow regions with GroundingDINO. Returns boxes in xyxy format."""
    text = "snow."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    try:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, target_sizes=[image.size[::-1]],
        )[0]

    boxes = results["boxes"].cpu()
    scores = results["scores"].cpu()
    keep = scores >= box_threshold
    boxes = boxes[keep].numpy()
    scores = scores[keep].numpy()
    return boxes, scores


def groundedsam2_refine_masks(image_np, boxes, predictor):
    """Refine detection boxes into pixel-level masks using SAM2."""
    if len(boxes) == 0:
        return np.zeros(image_np.shape[:2], dtype=np.uint8)

    predictor.set_image(image_np)
    input_boxes = torch.tensor(boxes, dtype=torch.float32, device=predictor.device)
    masks, scores, _ = predictor.predict(box=input_boxes, multimask_output=False)

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    combined = np.zeros(image_np.shape[:2], dtype=np.uint8)
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        combined = np.maximum(combined, (mask > 0.5).astype(np.uint8))
    return combined


def groundedsam2_visualize(image, mask, boxes, scores, output_path, title=""):
    """Save overview figure with detections and mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 4, f"{score:.2f}", color="lime", fontsize=8)
    axes[0].set_title(f"Detections ({len(boxes)})")
    axes[0].axis("off")

    snow_pct = mask.sum() / mask.size * 100
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Snow Mask ({snow_pct:.1f}%)")
    axes[1].axis("off")

    axes[2].imshow(_overlay_blue(image, mask))
    axes[2].set_title("Overlay (blue=snow)")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def groundedsam2_process_image(image_path, gdino_model, gdino_processor, sam2_predictor,
                                output_dir, device="cuda", box_threshold=0.3,
                                text_threshold=0.25, debug=False):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    boxes, scores = groundedsam2_detect_boxes(
        image, gdino_model, gdino_processor, device=device,
        box_threshold=box_threshold, text_threshold=text_threshold,
    )

    image_np = np.array(image)
    if len(boxes) > 0:
        snow_mask = groundedsam2_refine_masks(image_np, boxes, sam2_predictor)
    else:
        snow_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        groundedsam2_visualize(image_np, snow_mask, boxes, scores,
                               cam_out / f"{stem}_overview.png",
                               title=f"{camera} - {stem}")

    snow_pct = snow_mask.sum() / snow_mask.size * 100
    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name, "snow_pct": round(snow_pct, 2),
        "n_detections": len(boxes),
    }


def add_groundedsam2_subparser(subparsers):
    p = subparsers.add_parser("groundedsam2", help="GroundingDINO detection + SAM2 refinement")
    _add_common_args(p, str(WORKSPACE_DIR / "segmentation" / "snow" / "groundedsam2"))
    p.add_argument("--box-threshold", type=float, default=0.3,
                    help="GroundingDINO box confidence threshold (default: 0.3)")
    p.add_argument("--text-threshold", type=float, default=0.25,
                    help="GroundingDINO text confidence threshold (default: 0.25)")
    return p


def run_groundedsam2(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("GROUNDED SAM2 - SNOW SEGMENTATION")
    print("=" * 70)
    print(f"  Box threshold:  {args.box_threshold}")
    print(f"  Text threshold: {args.text_threshold}")
    print(f"  Device:         {device}")
    print(f"  Output:         {output_dir}")

    gdino_model, gdino_processor = groundedsam2_load_grounding_dino(device=device)
    sam2_predictor = groundedsam2_load_sam2(device=device)
    cameras = _discover_cameras(data_dir, args.cameras)
    print(f"\nCameras: {cameras}")

    results = []
    for cam in cameras:
        cam_dir = data_dir / cam
        if not cam_dir.exists():
            print(f"  WARNING: {cam_dir} does not exist, skipping")
            continue
        jpgs = _sample_jpgs(cam_dir, args.sample)
        print(f"\n  {cam}: processing {len(jpgs)} images")
        for img_path in tqdm(jpgs, desc=f"  {cam}", leave=True):
            result = groundedsam2_process_image(
                img_path, gdino_model, gdino_processor, sam2_predictor, output_dir,
                device=device, box_threshold=args.box_threshold,
                text_threshold=args.text_threshold, debug=args.debug,
            )
            if result:
                results.append(result)

    _write_summary_csv(results, output_dir, "groundedsam2_snow_summary.csv",
                        ["image", "camera", "filename", "snow_pct", "n_detections"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# SAM3 — text-prompted (via SAMWrapper below), optional ground-mask clip
# =============================================================================

class SAMWrapper:
    """
    Wrapper for the Segment Anything Model 3 (SAM3) using the GitHub package.
    Uses text prompts for segmentation.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        if not SAM3_AVAILABLE:
            raise ImportError("sam3 library is missing. Install from: https://github.com/facebookresearch/sam3")

        print(f"Loading SAM3 model from HuggingFace...")
        self.model = build_sam3_image_model(
            checkpoint_path=None,
            load_from_HF=True,
            device=device
        )
        self.processor = Sam3Processor(self.model)
        self.current_image = None
        print("SAM3 model loaded successfully.")

    def set_image(self, image: np.ndarray):
        if isinstance(image, np.ndarray):
            self.current_image = Image.fromarray(image)
        else:
            self.current_image = image

    def segment_with_text(self, image: np.ndarray, text_prompt: str, threshold: float = 0.5) -> np.ndarray:
        """
        Segment image using text prompt. Returns binary mask (H, W) with
        all detected instances combined.
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        self.processor.confidence_threshold = threshold

        state = self.processor.set_image(pil_image)
        state = self.processor.set_text_prompt(text_prompt, state)

        if 'masks' in state and state['masks'] is not None and len(state['masks']) > 0:
            masks = state['masks']  # Already thresholded at 0.5
            combined_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8)
        else:
            h, w = image.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)

        return combined_mask

    def segment_sky(self, image: np.ndarray) -> np.ndarray:
        return self.segment_with_text(image, "sky", threshold=0.3)

    def segment_snow(self, image: np.ndarray, sky_mask: Optional[np.ndarray] = None) -> np.ndarray:
        snow_mask = self.segment_with_text(image, "snow", threshold=0.4)

        if sky_mask is not None:
            snow_mask = np.logical_and(snow_mask, np.logical_not(sky_mask)).astype(np.uint8)

        return snow_mask


def sam3_process_image(image_path, sam, threshold, output_dir,
                        ground_mask_dir=None, mask_sky=True, debug=False):
    try:
        img = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    sky_mask = None
    if mask_sky:
        sky_mask = sam.segment_sky(img)

    sam.processor.confidence_threshold = threshold
    snow_mask = sam.segment_snow(img, sky_mask=sky_mask)

    if ground_mask_dir is not None:
        ground_path = ground_mask_dir / camera / f"{stem}_ground.png"
        if ground_path.exists():
            ground = np.array(Image.open(ground_path).convert("L"))
            if ground.shape != snow_mask.shape:
                ground = np.array(
                    Image.fromarray(ground).resize(
                        (snow_mask.shape[1], snow_mask.shape[0]), Image.NEAREST
                    )
                )
            snow_mask = (snow_mask & (ground > 127)).astype(np.uint8)

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        sam3_visualize(img, snow_mask, sky_mask, cam_out / f"{stem}_overview.png",
                       title=f"{camera} - {stem}")

    snow_pct = (snow_mask > 0).sum() / snow_mask.size * 100
    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name, "snow_pct": round(snow_pct, 2),
    }


def sam3_visualize(image, snow_mask, sky_mask, output_path, title=""):
    """Save overview figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 4 if sky_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    col = 1
    if sky_mask is not None:
        axes[col].imshow(sky_mask, cmap="gray")
        axes[col].set_title(f"Sky Mask ({sky_mask.sum() / sky_mask.size * 100:.1f}%)")
        axes[col].axis("off")
        col += 1

    snow_pct = snow_mask.sum() / snow_mask.size * 100
    axes[col].imshow(snow_mask, cmap="gray")
    axes[col].set_title(f"Snow Mask ({snow_pct:.1f}%)")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(_overlay_blue(image, snow_mask))
    axes[col].set_title("Overlay (blue=snow)")
    axes[col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def add_sam3_subparser(subparsers):
    p = subparsers.add_parser("sam3", help="SAM3 text-prompted (via SAMWrapper)")
    _add_common_args(p, str(WORKSPACE_DIR / "segmentation" / "snow" / "sam3"))
    p.add_argument("--ground-mask-dir", type=str, default=None,
                    help="Ground mask dir to restrict snow to ground region")
    p.add_argument("--threshold", type=float, default=0.4,
                    help="SAM3 confidence threshold (default: 0.4)")
    p.add_argument("--no-sky-mask", action="store_true",
                    help="Disable sky masking")
    return p


def run_sam3(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    ground_mask_dir = Path(args.ground_mask_dir) if args.ground_mask_dir else None
    mask_sky = not args.no_sky_mask

    print("=" * 70)
    print("SAM3 - SNOW SEGMENTATION (text-prompted)")
    print("=" * 70)
    print(f"  Threshold:    {args.threshold}")
    print(f"  Sky masking:  {mask_sky}")
    print(f"  Ground mask:  {ground_mask_dir or 'disabled'}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_dir}")

    sam = SAMWrapper(device=device)
    cameras = _discover_cameras(data_dir, args.cameras)
    print(f"\nCameras: {cameras}")

    results = []
    for cam in cameras:
        cam_dir = data_dir / cam
        if not cam_dir.exists():
            print(f"  WARNING: {cam_dir} does not exist, skipping")
            continue
        jpgs = _sample_jpgs(cam_dir, args.sample)
        print(f"\n  {cam}: processing {len(jpgs)} images")
        for img_path in tqdm(jpgs, desc=f"  {cam}", leave=True):
            result = sam3_process_image(
                img_path, sam, args.threshold, output_dir,
                ground_mask_dir=ground_mask_dir, mask_sky=mask_sky, debug=args.debug,
            )
            if result:
                results.append(result)

    _write_summary_csv(results, output_dir, "sam3_snow_summary.csv",
                        ["image", "camera", "filename", "snow_pct"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# YOLOE-26 — open-vocabulary instance segmentation
# =============================================================================

def yoloe_load_model(model_size="l", device="cuda"):
    """Load YOLOe-26 segmentation model and set snow classes."""
    from ultralytics import YOLO

    model_name = f"yoloe-26{model_size}-seg.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    model.to(device)
    model.set_classes(["snow"])
    print(f"  YOLOe-26{model_size}-seg loaded, classes set to ['snow']")
    return model


def yoloe_segment(model, image_path, conf=0.25):
    """Run YOLOe segmentation and merge all snow masks into one binary mask.

    Returns (binary_mask, snow_pct) or (None, 0) on failure.
    """
    try:
        results = model.predict(str(image_path), conf=conf, verbose=False)
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None, 0.0

    if not results or len(results) == 0:
        return None, 0.0

    result = results[0]
    h, w = result.orig_shape
    combined = np.zeros((h, w), dtype=np.uint8)

    if result.masks is not None and len(result.masks) > 0:
        for mask_data in result.masks.data:
            mask_np = mask_data.cpu().numpy()
            if mask_np.shape != (h, w):
                mask_resized = np.array(
                    Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                        (w, h), Image.NEAREST
                    )
                ) > 127
            else:
                mask_resized = mask_np > 0.5
            combined = np.maximum(combined, mask_resized.astype(np.uint8))

    snow_pct = combined.sum() / combined.size * 100
    return combined, snow_pct


def yoloe_visualize(image_path, mask, output_path, title=""):
    """Save overview figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.array(Image.open(image_path).convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    snow_pct = mask.sum() / mask.size * 100
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Snow Mask ({snow_pct:.1f}%)")
    axes[1].axis("off")

    axes[2].imshow(_overlay_blue(image, mask))
    axes[2].set_title("Overlay (blue=snow)")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def yoloe_process_image(image_path, model, conf, output_dir, debug=False):
    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    snow_mask, snow_pct = yoloe_segment(model, image_path, conf=conf)
    if snow_mask is None:
        return None

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        yoloe_visualize(image_path, snow_mask, cam_out / f"{stem}_overview.png",
                        title=f"{camera} - {stem}")

    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name, "snow_pct": round(snow_pct, 2),
    }


def add_yoloe_subparser(subparsers):
    p = subparsers.add_parser("yoloe", help="YOLOe-26 open-vocabulary instance segmentation")
    _add_common_args(p, str(WORKSPACE_DIR / "segmentation" / "snow" / "yoloe"))
    p.add_argument("--model-size", choices=["s", "m", "l"], default="l",
                    help="YOLOe-26 model size (default: l)")
    p.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold (default: 0.25)")
    return p


def run_yoloe(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("YOLOe-26 - SNOW SEGMENTATION (open-vocabulary)")
    print("=" * 70)
    print(f"  Model:      yoloe-26{args.model_size}-seg")
    print(f"  Confidence: {args.conf}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")

    model = yoloe_load_model(args.model_size, device=device)
    cameras = _discover_cameras(data_dir, args.cameras)
    print(f"\nCameras: {cameras}")

    results = []
    for cam in cameras:
        cam_dir = data_dir / cam
        if not cam_dir.exists():
            print(f"  WARNING: {cam_dir} does not exist, skipping")
            continue
        jpgs = _sample_jpgs(cam_dir, args.sample)
        print(f"\n  {cam}: processing {len(jpgs)} images")
        for img_path in tqdm(jpgs, desc=f"  {cam}", leave=True):
            result = yoloe_process_image(img_path, model, args.conf, output_dir, debug=args.debug)
            if result:
                results.append(result)

    _write_summary_csv(results, output_dir, "yoloe_snow_summary.csv",
                        ["image", "camera", "filename", "snow_pct"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# SEGGPT — in-context few-shot segmentation
# =============================================================================
# Given a few (image, snow_mask) prompt pairs per camera (sourced from GT
# annotations or a manual prompt directory), segments snow in the remaining
# images via in-context learning. No text prompts involved.

class _InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def _seggpt_bytes2bit(data):
    return ''.join(
        str((byte >> (7 - bit)) & 1)
        for byte in data
        for bit in range(8)
    )


def seggpt_decode_ls_rle(rle, width, height):
    """Decode Label Studio brush RLE (custom bitstream) to binary mask."""
    inp = _InputStream(_seggpt_bytes2bit(rle))
    num = inp.read(32)
    word_size = inp.read(5) + 1
    rle_sizes = [inp.read(4) + 1 for _ in range(4)]
    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = inp.read(1)
        j = i + 1 + inp.read(rle_sizes[inp.read(2)])
        if x:
            val = inp.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                out[i] = inp.read(word_size)
                i += 1
    mask = np.reshape(out, [height, width, 4])[:, :, 3]
    return mask > 0


def seggpt_load_gt_prompts(ann_dir, img_dir, mask_cache_dir, num_prompts=5,
                            fallback_img_dirs=None):
    """Load GT snow annotations and select diverse prompt examples per camera.

    Returns dict: {camera: [(img_path, mask_array), ...]}
    """
    ann_dir = Path(ann_dir)
    img_dir = Path(img_dir)
    mask_cache_dir = Path(mask_cache_dir)
    mask_cache_dir.mkdir(parents=True, exist_ok=True)

    search_dirs = [img_dir]
    if fallback_img_dirs:
        search_dirs.extend([Path(d) for d in fallback_img_dirs])

    def find_image(camera, stem):
        for d in search_dirs:
            p = d / camera / f"{stem}.jpg"
            if p.exists():
                return p
        return None

    snow_files = [
        "san.json", "heroku.json",
        "01.json", "06.json", "09.json", "14.json", "68.json",
        "70.json", "75.json", "80.json", "102.json", "108.json", "112.json",
    ]

    gt = {}  # {(camera, stem): binary_mask_array}

    for fname in snow_files:
        ann_path = ann_dir / fname
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            tasks = json.load(f)

        for task in tasks:
            url = task["data"]["image"]
            parts = url.rstrip("/").split("/")
            filename = parts[-1]
            camera = parts[-2]
            stem = Path(filename).stem

            annotations = task.get("annotations", [])
            if not annotations or not annotations[0].get("result"):
                continue
            result = annotations[0]["result"]

            is_non_usable = False
            for r in result:
                v = r.get("value", {})
                for key in ("taxonomy", "choices"):
                    tags = v.get(key)
                    if not tags:
                        continue
                    flat = set()
                    for entry in tags:
                        if isinstance(entry, list):
                            flat.update(entry)
                        else:
                            flat.add(entry)
                    if "Non-Usable" in flat:
                        is_non_usable = True
                if is_non_usable:
                    break
            if is_non_usable:
                continue

            w, h = 1296, 960
            for r in result:
                if r.get("original_width"):
                    w = r["original_width"]
                    h = r["original_height"]
                    break

            merged_mask = np.zeros((h, w), dtype=bool)
            has_brush = False
            for r in result:
                if r.get("type") == "brushlabels" and r.get("value", {}).get("rle"):
                    rle_data = r["value"]["rle"]
                    mask = seggpt_decode_ls_rle(rle_data, w, h)
                    merged_mask |= mask
                    has_brush = True

            if has_brush and merged_mask.any():
                gt[(camera, stem)] = merged_mask

    cam_samples = {}
    for (camera, stem), mask in gt.items():
        img_path = find_image(camera, stem)
        if img_path is None:
            continue
        snow_pct = mask.sum() / mask.size
        if camera not in cam_samples:
            cam_samples[camera] = []
        cam_samples[camera].append((img_path, mask, snow_pct))

    prompts = {}
    for camera, samples in cam_samples.items():
        samples.sort(key=lambda x: x[2])
        n = min(num_prompts, len(samples))
        if n == 0:
            continue
        indices = np.linspace(0, len(samples) - 1, n, dtype=int)
        prompts[camera] = [(s[0], s[1]) for s in [samples[i] for i in indices]]
        coverages = [samples[i][2] * 100 for i in indices]
        print(f"  {camera}: {n} prompts, coverage: "
              f"{', '.join(f'{c:.0f}%' for c in coverages)}")

    return prompts


def seggpt_load_manual_prompts(prompt_dir, num_prompts=5):
    """Load prompt examples from a directory.

    Expects: {prompt_dir}/{camera}/{stem}.jpg + {stem}_snow.png
    """
    prompt_dir = Path(prompt_dir)
    prompts = {}

    for cam_dir in sorted(prompt_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        camera = cam_dir.name
        pairs = []
        for mask_path in sorted(cam_dir.glob("*_snow.png")):
            stem = mask_path.stem.replace("_snow", "")
            img_path = cam_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 127).astype(bool)
            pairs.append((img_path, mask))

        if pairs:
            prompts[camera] = pairs[:num_prompts]
            print(f"  {camera}: {len(prompts[camera])} manual prompts")

    return prompts


def seggpt_load_model(device="cuda"):
    """Load SegGPT model and processor from HuggingFace."""
    from transformers import SegGptImageProcessor, SegGptForImageSegmentation

    model_id = "BAAI/seggpt-vit-large"
    print(f"Loading {model_id}...")
    processor = SegGptImageProcessor.from_pretrained(model_id)
    model = SegGptForImageSegmentation.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).eval().to(device)
    print(f"  SegGPT loaded on {device} (FP16)")
    return model, processor


def seggpt_segment(query_image, prompt_pairs, model, processor, device="cuda"):
    """Run SegGPT few-shot segmentation.

    Args:
        query_image: PIL Image (RGB)
        prompt_pairs: list of (PIL Image, np.array mask) tuples
    """
    NUM_LABELS = 1
    target_sizes = [query_image.size[::-1]]

    prompt_images = []
    prompt_masks_pil = []
    for img_path, mask_arr in prompt_pairs:
        prompt_images.append(Image.open(img_path).convert("RGB"))
        mask_pil = Image.fromarray(mask_arr.astype(np.uint8)).convert("L")
        prompt_masks_pil.append(mask_pil)

    def _to_device(inputs_dict):
        return {
            k: v.to(device, dtype=torch.float16)
            if v.dtype in (torch.float32, torch.float64)
            else v.to(device)
            for k, v in inputs_dict.items()
        }

    if len(prompt_pairs) == 1:
        inputs = processor(
            images=query_image, prompt_images=prompt_images[0],
            prompt_masks=prompt_masks_pil[0], num_labels=NUM_LABELS,
            return_tensors="pt",
        )
        inputs = _to_device(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
    else:
        inputs = processor(
            images=[query_image] * len(prompt_pairs), prompt_images=prompt_images,
            prompt_masks=prompt_masks_pil, num_labels=NUM_LABELS,
            return_tensors="pt",
        )
        inputs = _to_device(inputs)
        with torch.no_grad():
            outputs = model(**inputs, feature_ensemble=True)

    batch_size = outputs.pred_masks.shape[0]
    tgt = target_sizes * batch_size
    mask = processor.post_process_semantic_segmentation(
        outputs, target_sizes=tgt, num_labels=NUM_LABELS
    )[0]
    binary_mask = (mask == 1).cpu().numpy().astype(np.uint8)
    return binary_mask


def seggpt_visualize(image, mask, output_path, title=""):
    """Save overview figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    snow_pct = mask.sum() / mask.size * 100
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Snow Mask ({snow_pct:.1f}%)")
    axes[1].axis("off")

    axes[2].imshow(_overlay_blue(image, mask))
    axes[2].set_title("Overlay (blue=snow)")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def seggpt_process_image(image_path, prompt_pairs, model, processor, output_dir,
                          device="cuda", debug=False):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    snow_mask = seggpt_segment(image, prompt_pairs, model, processor, device=device)

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        seggpt_visualize(np.array(image), snow_mask, cam_out / f"{stem}_overview.png",
                         title=f"SegGPT - {camera}/{stem}")

    snow_pct = snow_mask.sum() / snow_mask.size * 100
    return {
        "image": str(image_path), "camera": camera,
        "filename": Path(image_path).name, "snow_pct": round(snow_pct, 2),
    }


def add_seggpt_subparser(subparsers):
    p = subparsers.add_parser("seggpt", help="SegGPT in-context few-shot segmentation")
    p.add_argument("--data-dir", type=str, required=True,
                    help="Data directory with camera folders")
    p.add_argument("--ann-dir", type=str, default=None,
                    help="GT annotation dir (Label Studio JSON exports)")
    p.add_argument("--prompt-dir", type=str, default=None,
                    help="Manual prompt dir (alternative to --ann-dir)")
    p.add_argument("--fallback-img-dirs", nargs="*", default=None,
                    help="Extra dirs to search for prompt images")
    p.add_argument("--cameras", nargs="+", default=[],
                    help="Specific cameras (default: all)")
    p.add_argument("--output-dir", type=str,
                    default=str(WORKSPACE_DIR / "segmentation" / "snow" / "seggpt"),
                    help="Output directory")
    p.add_argument("--num-prompts", type=int, default=5,
                    help="Number of prompt examples per camera (default: 5)")
    p.add_argument("--sample", type=int, default=0,
                    help="Images to sample per camera (0=all)")
    p.add_argument("--resume", action="store_true",
                    help="Skip images that already have output masks")
    p.add_argument("--debug", action="store_true",
                    help="Save overview figures")
    return p


def run_seggpt(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("SEGGPT - FEW-SHOT SNOW SEGMENTATION (in-context learning)")
    print("=" * 70)
    print(f"  Model:       BAAI/seggpt-vit-large")
    print(f"  Num prompts: {args.num_prompts} per camera")
    print(f"  Device:      {device}")
    print(f"  Output:      {output_dir}")

    if args.prompt_dir:
        prompts = seggpt_load_manual_prompts(args.prompt_dir, args.num_prompts)
    elif args.ann_dir:
        mask_cache = output_dir / "_prompt_cache"
        prompts = seggpt_load_gt_prompts(
            args.ann_dir, data_dir, mask_cache, num_prompts=args.num_prompts,
            fallback_img_dirs=args.fallback_img_dirs,
        )
    else:
        print("ERROR: Provide --ann-dir or --prompt-dir for prompt examples")
        return

    if not prompts:
        print("ERROR: No prompt examples found!")
        return

    print(f"\n  Prompts loaded for {len(prompts)} cameras")

    model, processor = seggpt_load_model(device=device)
    cameras = _discover_cameras(data_dir, args.cameras)

    all_prompt_pairs = []
    for cam in prompts:
        all_prompt_pairs.extend(prompts[cam])
    fallback_prompts = all_prompt_pairs[:args.num_prompts] if all_prompt_pairs else []

    print(f"\nCameras to process: {cameras}")

    results = []
    for cam in cameras:
        cam_dir = data_dir / cam
        if not cam_dir.exists():
            print(f"  WARNING: {cam_dir} does not exist, skipping")
            continue

        cam_prompts = prompts.get(cam, fallback_prompts)
        if not cam_prompts:
            print(f"  WARNING: No prompts for {cam}, skipping")
            continue

        jpgs = _sample_jpgs(cam_dir, args.sample)

        prompt_stems = {Path(img_path).stem for img_path, _ in cam_prompts}
        jpgs = [j for j in jpgs if j.stem not in prompt_stems]

        if args.resume:
            before = len(jpgs)
            jpgs = [j for j in jpgs
                    if not (output_dir / cam / f"{j.stem}_snow.png").exists()]
            skipped = before - len(jpgs)
            if skipped:
                print(f"  {cam}: resuming - skipped {skipped} already processed")

        print(f"\n  {cam}: {len(jpgs)} images, {len(cam_prompts)} prompts")

        for img_path in tqdm(jpgs, desc=f"  {cam}", leave=True):
            result = seggpt_process_image(
                img_path, cam_prompts, model, processor, output_dir,
                device=device, debug=args.debug,
            )
            if result:
                results.append(result)

    _write_summary_csv(results, output_dir, "seggpt_snow_summary.csv",
                        ["image", "camera", "filename", "snow_pct"])
    print(f"\nProcessed {len(results)} images.")
    print("Done!")


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Snow segmentation teachers")
    subparsers = parser.add_subparsers(dest="teacher", required=True)

    add_clipseg_subparser(subparsers)
    add_florence2_subparser(subparsers)
    add_groundedsam2_subparser(subparsers)
    add_sam3_subparser(subparsers)
    add_yoloe_subparser(subparsers)
    add_seggpt_subparser(subparsers)

    args = parser.parse_args()

    {
        "clipseg": run_clipseg,
        "florence2": run_florence2,
        "groundedsam2": run_groundedsam2,
        "sam3": run_sam3,
        "yoloe": run_yoloe,
        "seggpt": run_seggpt,
    }[args.teacher](args)


if __name__ == "__main__":
    main()
