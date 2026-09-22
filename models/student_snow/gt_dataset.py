"""GT annotation dataset for supervised snow segmentation training.

Loads snow masks from Label Studio JSON exports (RLE-encoded brush labels)
and pairs them with source images. Returns samples in the same format as
the pseudo-label pipeline: (image_path, confidence_path, mask_path, depth_path).

Since GT masks are binary, confidence is set to 1.0 everywhere.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image


class _InputStream:
    """Bit-level reader for Label Studio's custom RLE bitstream."""
    def __init__(self, data):
        self.data = data
        self.i = 0
    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def _bytes2bit(data):
    """Convert byte list to bit string."""
    return ''.join(
        str((byte >> (7 - bit)) & 1)
        for byte in data
        for bit in range(8)
    )


def decode_ls_rle(rle, width, height):
    """Decode Label Studio brush RLE to binary mask.

    LS uses a custom bitstream RLE (from @thi.ng/rle-pack), NOT simple
    run-length pairs. Decodes to flat RGBA uint8 array, then extracts
    alpha channel as the mask.
    """
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


def load_gt_snow_samples(ann_dir, img_dir, mask_out_dir, depth_dir=None,
                         include_negatives=True, fallback_img_dirs=None):
    """Parse snow annotations and write mask PNGs for training.

    Writes binary _snow.png and all-white _confidence.png per annotated image
    into mask_out_dir, then returns list of (img, conf, mask, depth) tuples.

    If include_negatives=True, also adds "No Snow" images with all-zero masks
    as negative training examples.
    """
    ann_dir = Path(ann_dir)
    img_dir = Path(img_dir)
    mask_out_dir = Path(mask_out_dir)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # CM00006 (06.json) and CM00068 (68.json) excluded per professor
    snow_files = [
        "san.json", "heroku.json",
        "01.json", "09.json", "14.json",
        "70.json", "75.json", "80.json", "102.json", "108.json", "112.json",
    ]
    NUMBERED = {"01.json", "09.json", "14.json",
                "70.json", "75.json", "80.json", "102.json",
                "108.json", "112.json"}

    gt = {}  # {(camera, stem): binary_mask}
    no_snow_images = []  # [(camera, stem)] for negative samples

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

            # Check non-usable and track "No Snow" reason
            is_non_usable = False
            is_no_snow = False
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
                    if "No Snow" in flat:
                        is_no_snow = True
                if is_non_usable:
                    break
            if is_non_usable:
                if (camera, stem) in gt:
                    del gt[(camera, stem)]
                if is_no_snow:
                    no_snow_images.append((camera, stem))
                continue

            w, h = 1296, 960
            for r in result:
                if r.get("original_width"):
                    w = r["original_width"]
                    h = r["original_height"]
                    break

            merged_mask = np.zeros((h, w), dtype=bool)
            for r in result:
                if r.get("type") == "brushlabels" and r.get("value", {}).get("rle"):
                    rle = r["value"]["rle"]
                    mask = decode_ls_rle(rle, w, h)
                    merged_mask |= mask

            gt[(camera, stem)] = merged_mask

    # Build list of image directories to search (primary + fallbacks)
    search_dirs = [img_dir]
    if fallback_img_dirs:
        search_dirs.extend([Path(d) for d in fallback_img_dirs])

    def find_image(camera, stem):
        for d in search_dirs:
            p = d / camera / f"{stem}.jpg"
            if p.exists():
                return p
        return None

    # Write mask PNGs and build sample list
    samples = []
    n_fallback = 0
    for (camera, stem), mask in gt.items():
        img_path = find_image(camera, stem)
        if img_path is None:
            continue
        if img_path.parent.parent != img_dir:
            n_fallback += 1

        img = Image.open(img_path)
        img_w, img_h = img.size

        cam_out = mask_out_dir / camera
        cam_out.mkdir(parents=True, exist_ok=True)

        mask_path = cam_out / f"{stem}_snow.png"
        conf_path = cam_out / f"{stem}_confidence.png"

        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        if (mask_img.width, mask_img.height) != (img_w, img_h):
            mask_img = mask_img.resize((img_w, img_h), Image.NEAREST)
        mask_img.save(mask_path)

        conf = mask_img.copy()
        conf.save(conf_path)

        depth_path = None
        if depth_dir:
            dp = Path(depth_dir) / camera / f"{stem}_depth.png"
            if dp.exists():
                depth_path = dp

        samples.append((img_path, conf_path, mask_path, depth_path))

    # Add negative samples (No Snow images with all-zero masks)
    n_negatives = 0
    if include_negatives and no_snow_images:
        neg_out = mask_out_dir / "_negatives"
        neg_out.mkdir(parents=True, exist_ok=True)
        for camera, stem in no_snow_images:
            img_path = find_image(camera, stem)
            if img_path is None:
                continue
            img = Image.open(img_path)
            img_w, img_h = img.size

            mask_path = neg_out / f"{camera}_{stem}_snow.png"
            conf_path = neg_out / f"{camera}_{stem}_confidence.png"

            # All-zero mask and confidence (no snow)
            Image.new("L", (img_w, img_h), 0).save(mask_path)
            Image.new("L", (img_w, img_h), 0).save(conf_path)

            samples.append((img_path, conf_path, mask_path, None))
            n_negatives += 1

    print(f"  GT snow samples: {len(samples)} ({len(gt)} positive, {n_negatives} negative, {n_fallback} from fallback dirs)")
    return samples
