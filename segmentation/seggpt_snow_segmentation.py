#!/usr/bin/env python3
"""
SegGPT in-context snow segmentation (7th teacher).

Uses BAAI/seggpt-vit-large for few-shot snow segmentation.
Given a few (image, snow_mask) prompt pairs per camera, SegGPT
segments snow in all remaining images via in-context learning.

Prompt examples are sourced from GT annotations (Label Studio exports)
or manually specified prompt directories.

Requires: transformers>=4.36.0, torch, torchvision, pillow.

Usage:
    # Auto-select prompts from GT annotations
    python seggpt_snow_segmentation.py \
        --data-dir /shared/home/juan.osorio/master/data \
        --ann-dir /shared/home/juan.osorio/master/data/annotations

    # Specific cameras
    python seggpt_snow_segmentation.py \
        --data-dir /shared/home/juan.osorio/master/data \
        --ann-dir /shared/home/juan.osorio/master/data/annotations \
        --cameras CF00001 CM00009

    # Manual prompt dir (expects {camera}/{stem}.jpg + {camera}/{stem}_snow.png)
    python seggpt_snow_segmentation.py \
        --data-dir /shared/home/juan.osorio/master/data \
        --prompt-dir /path/to/prompts --num-prompts 5
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ── Label Studio RLE decoder ──────────────────────────────────────────────────

class _InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0
    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def _bytes2bit(data):
    return ''.join(
        str((byte >> (7 - bit)) & 1)
        for byte in data
        for bit in range(8)
    )


def _decode_ls_rle(rle, width, height):
    """Decode Label Studio brush RLE (custom bitstream) to binary mask."""
    inp = _InputStream(_bytes2bit(rle))
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


# ── Prompt selection from GT annotations ─────────────────────────────────────

def load_gt_prompts(ann_dir, img_dir, mask_cache_dir, num_prompts=5,
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
                    mask = _decode_ls_rle(rle_data, w, h)
                    merged_mask |= mask
                    has_brush = True

            if has_brush and merged_mask.any():
                gt[(camera, stem)] = merged_mask

    # Group by camera, select diverse prompts (varied snow coverage)
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
        # Pick evenly spaced by snow coverage for diversity
        indices = np.linspace(0, len(samples) - 1, n, dtype=int)
        prompts[camera] = [(s[0], s[1]) for s in [samples[i] for i in indices]]
        coverages = [samples[i][2] * 100 for i in indices]
        print(f"  {camera}: {n} prompts, coverage: "
              f"{', '.join(f'{c:.0f}%' for c in coverages)}")

    return prompts


def load_manual_prompts(prompt_dir, num_prompts=5):
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


# ── Model loading ────────────────────────────────────────────────────────────

def load_seggpt(device="cuda"):
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


# ── Segmentation ─────────────────────────────────────────────────────────────

def segment_snow_seggpt(query_image, prompt_pairs, model, processor,
                        device="cuda"):
    """Run SegGPT few-shot segmentation.

    Args:
        query_image: PIL Image (RGB)
        prompt_pairs: list of (PIL Image, np.array mask) tuples
        model: SegGPT model
        processor: SegGPT processor

    Returns:
        binary_mask: np.array (H, W), uint8, 0 or 1
    """
    NUM_LABELS = 1
    target_sizes = [query_image.size[::-1]]  # (H, W)

    prompt_images = []
    prompt_masks_pil = []
    for img_path, mask_arr in prompt_pairs:
        prompt_images.append(Image.open(img_path).convert("RGB"))
        # Processor with num_labels=1 expects class indices: 0=bg, 1=snow
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
            images=query_image,
            prompt_images=prompt_images[0],
            prompt_masks=prompt_masks_pil[0],
            num_labels=NUM_LABELS,
            return_tensors="pt",
        )
        inputs = _to_device(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
    else:
        inputs = processor(
            images=[query_image] * len(prompt_pairs),
            prompt_images=prompt_images,
            prompt_masks=prompt_masks_pil,
            num_labels=NUM_LABELS,
            return_tensors="pt",
        )
        inputs = _to_device(inputs)
        with torch.no_grad():
            outputs = model(**inputs, feature_ensemble=True)

    # Match target_sizes to actual output batch dimension
    batch_size = outputs.pred_masks.shape[0]
    tgt = target_sizes * batch_size
    mask = processor.post_process_semantic_segmentation(
        outputs, target_sizes=tgt, num_labels=NUM_LABELS
    )[0]
    binary_mask = (mask == 1).cpu().numpy().astype(np.uint8)
    return binary_mask


# ── Processing ───────────────────────────────────────────────────────────────

def process_image(image_path, prompt_pairs, model, processor, output_dir,
                  device="cuda", debug=False):
    """Process a single image. Returns result dict or None."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: skipping {image_path}: {e}")
        return None

    stem = Path(image_path).stem
    camera = Path(image_path).parent.name

    snow_mask = segment_snow_seggpt(
        image, prompt_pairs, model, processor, device=device,
    )

    cam_out = output_dir / camera
    cam_out.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((snow_mask * 255).astype(np.uint8))
    mask_img.save(cam_out / f"{stem}_snow.png")

    if debug:
        visualize(np.array(image), snow_mask,
                  cam_out / f"{stem}_overview.png",
                  title=f"SegGPT — {camera}/{stem}")

    snow_pct = snow_mask.sum() / snow_mask.size * 100
    return {
        "image": str(image_path),
        "camera": camera,
        "filename": Path(image_path).name,
        "snow_pct": round(snow_pct, 2),
    }


def visualize(image, mask, output_path, title=""):
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

    overlay = image.copy().astype(np.float32)
    mask_bool = mask > 0
    overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 150, 255]) * 0.5
    overlay[~mask_bool] = overlay[~mask_bool] * 0.5 + np.array([100, 100, 100]) * 0.5
    axes[2].imshow(overlay.clip(0, 255).astype(np.uint8))
    axes[2].set_title("Overlay (blue=snow)")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SegGPT few-shot snow segmentation (7th teacher)"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory with camera folders")
    parser.add_argument("--ann-dir", type=str, default=None,
                        help="GT annotation dir (Label Studio JSON exports)")
    parser.add_argument("--prompt-dir", type=str, default=None,
                        help="Manual prompt dir (alternative to --ann-dir)")
    parser.add_argument("--fallback-img-dirs", nargs="*", default=None,
                        help="Extra dirs to search for prompt images")
    parser.add_argument("--cameras", nargs="+", default=[],
                        help="Specific cameras (default: all)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent
                                    / "preprocessing" / "segmentation"
                                    / "seggpt_snow_output"),
                        help="Output directory")
    parser.add_argument("--num-prompts", type=int, default=5,
                        help="Number of prompt examples per camera (default: 5)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Images to sample per camera (0=all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images that already have output masks")
    parser.add_argument("--debug", action="store_true",
                        help="Save overview figures")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("SEGGPT — FEW-SHOT SNOW SEGMENTATION (in-context learning)")
    print("=" * 70)
    print(f"  Model:       BAAI/seggpt-vit-large")
    print(f"  Num prompts: {args.num_prompts} per camera")
    print(f"  Device:      {device}")
    print(f"  Output:      {output_dir}")

    # Load prompt examples
    if args.prompt_dir:
        prompts = load_manual_prompts(args.prompt_dir, args.num_prompts)
    elif args.ann_dir:
        mask_cache = output_dir / "_prompt_cache"
        prompts = load_gt_prompts(
            args.ann_dir, data_dir, mask_cache,
            num_prompts=args.num_prompts,
            fallback_img_dirs=args.fallback_img_dirs,
        )
    else:
        print("ERROR: Provide --ann-dir or --prompt-dir for prompt examples")
        return

    if not prompts:
        print("ERROR: No prompt examples found!")
        return

    print(f"\n  Prompts loaded for {len(prompts)} cameras")

    model, processor = load_seggpt(device=device)

    cameras = args.cameras
    if not cameras:
        cameras = sorted([
            d.name for d in data_dir.iterdir()
            if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)
        ])

    # For cameras without GT prompts, use prompts from nearest camera
    # that has them (fallback to any available prompts)
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

        jpgs = sorted(cam_dir.glob("*.jpg"))
        if args.sample > 0 and len(jpgs) > args.sample:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(jpgs), size=args.sample, replace=False)
            jpgs = [jpgs[i] for i in sorted(indices)]

        # Skip images used as prompts
        prompt_stems = set()
        for img_path, _ in cam_prompts:
            prompt_stems.add(Path(img_path).stem)

        jpgs = [j for j in jpgs if j.stem not in prompt_stems]

        if args.resume:
            before = len(jpgs)
            jpgs = [j for j in jpgs
                    if not (output_dir / cam / f"{j.stem}_snow.png").exists()]
            skipped = before - len(jpgs)
            if skipped:
                print(f"  {cam}: resuming — skipped {skipped} already processed")

        print(f"\n  {cam}: {len(jpgs)} images, {len(cam_prompts)} prompts")

        for img_path in tqdm(jpgs, desc=f"  {cam}", leave=True):
            result = process_image(
                img_path, cam_prompts, model, processor,
                output_dir, device=device, debug=args.debug,
            )
            if result:
                results.append(result)

    if results:
        summary_path = output_dir / "seggpt_snow_summary.csv"
        fieldnames = ["image", "camera", "filename", "snow_pct"]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary: {summary_path}")

    print(f"\nProcessed {len(results)} images.")
    print("Done!")


if __name__ == "__main__":
    main()
