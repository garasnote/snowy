#!/usr/bin/env python3
"""
Streamlit dashboard for comparing ground segmentation methods.

Compares: SegFormer, Depth Anything V2 (parameter grid), DeepLabV3+, SAM3+DINO, DINOv2 PCA, Fusion.

Usage:
    streamlit run app_segmentation_comparison.py --server.port 8501
"""

import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── paths ────────────────────────────────────────────────────────────────────

# Scripts live in master/segmentation/, outputs stay in master/preprocessing/segmentation/
_MASTER = Path(__file__).resolve().parent.parent
OUTPUT_BASE = _MASTER / "preprocessing" / "segmentation"
DATA_DIR = _MASTER / "data"

DEPTH_OUTPUT = OUTPUT_BASE / "depth_output"
SEGFORMER_OUTPUT = OUTPUT_BASE / "segformer_output"
DEEPLABV3_OUTPUT = OUTPUT_BASE / "deeplabv3_output"
SAM3_DINO_OUTPUT = OUTPUT_BASE / "sam3_dino_output"
DINOV2_PCA_OUTPUT = OUTPUT_BASE / "dinov2_pca_output"
DINOV3_PCA_OUTPUT = OUTPUT_BASE / "dinov3_pca_output"
FUSION_OUTPUT = OUTPUT_BASE / "fusion_output"
FUSION_DINOV3_OUTPUT = OUTPUT_BASE / "fusion_dinov3_output"
ENSEMBLE_OUTPUT = OUTPUT_BASE / "ensemble_output"
STUDENT_OUTPUT = OUTPUT_BASE / "student_ground_output"

# Snow segmentation outputs
SNOW_OUTPUT_BASE = OUTPUT_BASE
HSV_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "hsv_snow_output_mobilenet_ft"
SAM3_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "sam3_snow_output"
YOLOE_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "yoloe_snow_output"
FLORENCE2_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "florence2_snow_output"
GROUNDEDSAM2_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "groundedsam2_snow_output"
CLIPSEG_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "clipseg_snow_output"
SEGGPT_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "seggpt_snow_output"
ENSEMBLE_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "ensemble_snow_output"
STUDENT_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "student_snow_output"
SUBSPACE_AD_OUTPUT = SNOW_OUTPUT_BASE / "subspace_ad_output"
ANOMALY_RESULTS = _MASTER / "models" / "anomaly_snow" / "results"

# Allow override via /shared paths if local data doesn't exist
if not DATA_DIR.exists():
    DATA_DIR = Path("data")
if not OUTPUT_BASE.exists():
    OUTPUT_BASE = Path("preprocessing/segmentation")
    DEPTH_OUTPUT = OUTPUT_BASE / "depth_output"
    SEGFORMER_OUTPUT = OUTPUT_BASE / "segformer_output"
    DEEPLABV3_OUTPUT = OUTPUT_BASE / "deeplabv3_output"
    SAM3_DINO_OUTPUT = OUTPUT_BASE / "sam3_dino_output"
    DINOV2_PCA_OUTPUT = OUTPUT_BASE / "dinov2_pca_output"
    DINOV3_PCA_OUTPUT = OUTPUT_BASE / "dinov3_pca_output"
    FUSION_OUTPUT = OUTPUT_BASE / "fusion_output"
    FUSION_DINOV3_OUTPUT = OUTPUT_BASE / "fusion_dinov3_output"
    ENSEMBLE_OUTPUT = OUTPUT_BASE / "ensemble_output"
    STUDENT_OUTPUT = OUTPUT_BASE / "student_ground_output"
    SNOW_OUTPUT_BASE = OUTPUT_BASE
    HSV_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "hsv_snow_output_mobilenet_ft"
    SAM3_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "sam3_snow_output"
    YOLOE_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "yoloe_snow_output"
    FLORENCE2_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "florence2_snow_output"
    GROUNDEDSAM2_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "groundedsam2_snow_output"
    CLIPSEG_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "clipseg_snow_output"
    SEGGPT_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "seggpt_snow_output"
    ENSEMBLE_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "ensemble_snow_output"
    STUDENT_SNOW_OUTPUT = SNOW_OUTPUT_BASE / "student_snow_output"
    SUBSPACE_AD_OUTPUT = SNOW_OUTPUT_BASE / "subspace_ad_output"
    ANOMALY_RESULTS = Path("models/anomaly_snow/results")


# ── excluded cameras ─────────────────────────────────────────────────────────

EXCLUDED_CAMERAS = {"CM00024", "CM00006", "CM00068"}  # removed from all evaluation

# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def discover_cameras():
    """Find all camera directories in the data folder."""
    if not DATA_DIR.exists():
        return []
    return sorted([
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and re.match(r"C[A-Z]\d+", d.name)
        and d.name not in EXCLUDED_CAMERAS
    ])


@st.cache_data
def list_images(camera: str):
    """List all JPG images for a camera, sorted by name."""
    cam_dir = DATA_DIR / camera
    if not cam_dir.exists():
        return []
    return sorted([p.name for p in cam_dir.glob("*.jpg")])


@st.cache_data
def discover_depth_combos():
    """Find all N*_F* parameter combinations in depth output."""
    if not DEPTH_OUTPUT.exists():
        return []
    combos = sorted([
        d.name for d in DEPTH_OUTPUT.iterdir()
        if d.is_dir() and re.match(r"N[\d.]+_F[\d.]+", d.name)
    ])
    return combos


@st.cache_data
def discover_methods():
    """Auto-detect which segmentation methods have output."""
    methods = {}
    if SEGFORMER_OUTPUT.exists():
        methods["SegFormer"] = SEGFORMER_OUTPUT
    if DEEPLABV3_OUTPUT.exists():
        methods["DeepLabV3+"] = DEEPLABV3_OUTPUT
    if SAM3_DINO_OUTPUT.exists():
        # Check for method subdirectories (text/, dino+sam/)
        text_dir = SAM3_DINO_OUTPUT / "text"
        dino_sam_dir = SAM3_DINO_OUTPUT / "dino+sam"
        if text_dir.exists():
            methods["SAM3 (text)"] = text_dir
        if dino_sam_dir.exists():
            methods["SAM3+DINO"] = dino_sam_dir
        # Fallback: old flat structure without method subdirs
        if not text_dir.exists() and not dino_sam_dir.exists():
            methods["SAM3+DINO"] = SAM3_DINO_OUTPUT
    if DINOV2_PCA_OUTPUT.exists():
        methods["DINOv2 PCA"] = DINOV2_PCA_OUTPUT
    if DINOV3_PCA_OUTPUT.exists():
        methods["DINOv3 PCA (SAT)"] = DINOV3_PCA_OUTPUT
    # Add best ensemble variant to comparison
    if ENSEMBLE_OUTPUT.exists():
        # Prefer majority_4, fallback to first available
        for variant in ["majority_4", "majority_3", "majority_5"]:
            d = ENSEMBLE_OUTPUT / variant
            if d.exists():
                methods[f"Ensemble ({variant})"] = d
                break
    return methods


@st.cache_data
def discover_fusion_strategies():
    """Auto-detect which fusion strategies have output."""
    strategies = {}
    if FUSION_OUTPUT.exists():
        for subdir in ["intersect", "union", "pca_depth_filter", "depth_pca_filter"]:
            d = FUSION_OUTPUT / subdir
            if d.exists():
                strategies[subdir] = d
    return strategies


@st.cache_data
def discover_dinov3_fusion_strategies():
    """Auto-detect which DINOv3+depth fusion strategies have output."""
    strategies = {}
    if FUSION_DINOV3_OUTPUT.exists():
        for subdir in ["intersect", "union", "pca_depth_filter", "depth_pca_filter"]:
            d = FUSION_DINOV3_OUTPUT / subdir
            if d.exists():
                strategies[subdir] = d
    return strategies


@st.cache_data
def discover_ensemble_variants():
    """Auto-detect ensemble voting variants (majority_3, majority_4, etc.)."""
    variants = {}
    if ENSEMBLE_OUTPUT.exists():
        for d in sorted(ENSEMBLE_OUTPUT.iterdir()):
            if d.is_dir() and d.name.startswith("majority_"):
                variants[d.name] = d
    return variants


@st.cache_data
def discover_weighted_variants():
    """Auto-detect weighted ensemble variants (weighted_4, weighted_5, etc.)."""
    variants = {}
    if ENSEMBLE_OUTPUT.exists():
        for d in sorted(ENSEMBLE_OUTPUT.iterdir()):
            if d.is_dir() and d.name.startswith("weighted_"):
                variants[d.name] = d
    return variants


@st.cache_data
def discover_student_variants():
    """Auto-detect student model variants in student_ground_output/.

    Handles two layouts:
      - Flat:   student_ground_output/{camera}/  → single model, key="student"
      - Nested: student_ground_output/{variant}/{camera}/  → one key per variant
    """
    variants = {}
    if not STUDENT_OUTPUT.exists():
        return variants
    for d in sorted(STUDENT_OUTPUT.iterdir()):
        if not d.is_dir():
            continue
        # Camera dirs match C[A-Z]\d+ → flat layout (single model run)
        if re.match(r"C[A-Z]\d+", d.name):
            variants["student"] = STUDENT_OUTPUT
            return variants
        # Otherwise treat as a named variant subdirectory
        variants[d.name] = d
    return variants


@st.cache_data
def discover_snow_teachers():
    """Auto-detect which snow teacher methods have output."""
    teachers = {}
    for name, d in [
        ("HSV", HSV_SNOW_OUTPUT),
        ("SAM3", SAM3_SNOW_OUTPUT),
        ("YOLOe", YOLOE_SNOW_OUTPUT),
        ("Florence-2", FLORENCE2_SNOW_OUTPUT),
        ("GroundedSAM2", GROUNDEDSAM2_SNOW_OUTPUT),
        ("CLIPSeg", CLIPSEG_SNOW_OUTPUT),
        ("SegGPT", SEGGPT_SNOW_OUTPUT),
    ]:
        if d.exists():
            teachers[name] = d
    return teachers


@st.cache_data
def discover_snow_ensemble_variants():
    variants = {}
    if ENSEMBLE_SNOW_OUTPUT.exists():
        for d in sorted(ENSEMBLE_SNOW_OUTPUT.iterdir()):
            if d.is_dir() and (d.name.startswith("majority_") or d.name.startswith("weighted_")):
                variants[d.name] = d
    return variants


@st.cache_data
def discover_student_snow_variants():
    variants = {}
    if not STUDENT_SNOW_OUTPUT.exists():
        return variants
    for d in sorted(STUDENT_SNOW_OUTPUT.iterdir()):
        if not d.is_dir():
            continue
        if re.match(r"C[A-Z]\d+", d.name):
            variants["student"] = STUDENT_SNOW_OUTPUT
            return variants
        variants[d.name] = d
    return variants


def load_anomalib_mask(method, camera, stem):
    """Extract predicted mask from anomalib 5-panel composite image.

    Anomalib saves 5-panel composites: [Image, GT Mask, Anomaly Map, Image+Map, Image+PredMask].
    Panel 5 has red contours for the predicted mask. Instead we threshold the
    anomaly heatmap (panel 3) to get a binary mask.
    """
    method_dir = ANOMALY_RESULTS / method
    if not method_dir.exists():
        return None
    model_class = {"patchcore": "Patchcore", "efficientad": "EfficientAd"}.get(method)
    if not model_class:
        return None
    composite_path = (method_dir / camera / model_class / camera /
                      "latest" / "images" / "test" / "bad" / f"{stem}.png")
    if not composite_path.exists():
        return None
    try:
        img = np.array(Image.open(composite_path).convert("RGB"))
        n_panels = img.shape[1] // img.shape[0]
        panel_w = img.shape[1] // n_panels
        # Panel 3 is the anomaly heatmap (jet colormap).
        # Threshold red channel — anomalous regions are warm-colored.
        heatmap = img[:, panel_w * 2:panel_w * 3, :]
        return heatmap[:, :, 0] > 127
    except Exception:
        return None


@st.cache_data
def discover_anomaly_methods():
    """Auto-detect available anomaly detection methods."""
    methods = {}
    if not ANOMALY_RESULTS.exists():
        return methods
    for name in ["patchcore", "efficientad"]:
        d = ANOMALY_RESULTS / name
        if d.exists() and any(d.iterdir()):
            methods[name] = d
    return methods


def parse_combo(combo_name: str):
    """Parse 'N0.05_F0.85' into (near_clip, far_clip)."""
    m = re.match(r"N([\d.]+)_F([\d.]+)", combo_name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def load_image(path: Path):
    """Load image as numpy array, return None if missing."""
    if path.exists():
        return np.array(Image.open(path).convert("RGB"))
    return None


def load_mask(path: Path):
    """Load ground mask as boolean array, return None if missing."""
    if path.exists():
        mask = np.array(Image.open(path).convert("L"))
        return mask > 127
    return None


def make_overlay(original, mask, color=(0, 200, 0), alpha=0.45):
    """Create green overlay where mask is True, dim where False."""
    overlay = original.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    gray = np.array([100, 100, 100], dtype=np.float32)

    overlay[mask] = overlay[mask] * (1 - alpha) + color_arr * alpha
    overlay[~mask] = overlay[~mask] * 0.5 + gray * 0.5

    return overlay.clip(0, 255).astype(np.uint8)


def ground_pct(mask):
    """Compute ground percentage from boolean mask."""
    if mask is None:
        return 0.0
    return mask.sum() / mask.size * 100


def stem_from_filename(filename: str):
    """Get stem (without extension) from filename."""
    return Path(filename).stem


# ── Label Studio RLE decoding ────────────────────────────────────────────────

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
    # RGBA → use alpha channel > 0 for binary mask
    n_channels = num // (width * height)
    if n_channels >= 4:
        rgba = flat.reshape(height, width, n_channels)
        return rgba[:, :, 3] > 0
    return flat[:width * height].reshape(height, width) > 0


@st.cache_data
def load_ground_truth_annotations():
    """Load ground truth annotations from Label Studio JSON export.

    Returns dict: {camera: {stem: binary_mask}}
    """
    ann_path = DATA_DIR / "annotations" / "ground_annotations.json"
    if not ann_path.exists():
        return {}

    with open(ann_path) as f:
        tasks = json.load(f)

    gt = {}
    for task in tasks:
        url = task["data"]["image"]
        # URL: .../camera/filename.jpg
        parts = url.rstrip("/").split("/")
        filename = parts[-1]
        camera = parts[-2]
        if camera in EXCLUDED_CAMERAS:
            continue
        stem = Path(filename).stem

        annotations = task.get("annotations", [])
        if not annotations or not annotations[0].get("result"):
            continue

        # Merge all brush results for this image (may have multiple strokes)
        result = annotations[0]["result"]
        w = result[0].get("original_width", 1296)
        h = result[0].get("original_height", 960)
        merged_mask = np.zeros((h, w), dtype=bool)

        for r in result:
            if r.get("type") == "brushlabels" and r.get("value", {}).get("rle"):
                rle = r["value"]["rle"]
                mask = decode_ls_rle(rle, w, h)
                merged_mask |= mask

        gt.setdefault(camera, {})[stem] = merged_mask

    return gt


@st.cache_data
def load_snow_gt_annotations():
    """Load snow GT annotations from Label Studio JSON exports."""
    snow_files = [
        "san.json", "heroku.json",
        "01.json", "06.json", "09.json", "14.json", "68.json",
        "70.json", "75.json", "80.json", "102.json", "108.json", "112.json",
    ]
    gt = {}
    ann_dir = DATA_DIR / "annotations"
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
            if camera in EXCLUDED_CAMERAS:
                continue
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
            for r in result:
                if r.get("type") == "brushlabels" and r.get("value", {}).get("rle"):
                    rle = r["value"]["rle"]
                    mask = decode_ls_rle(rle, w, h)
                    merged_mask |= mask
            gt.setdefault(camera, {})[stem] = merged_mask
    return gt


def compute_seg_metrics(pred, gt):
    """Compute segmentation metrics between predicted and ground truth masks.

    Returns dict with IoU, Dice, Pixel Accuracy, Precision, Recall, F1.
    """
    if pred is None or gt is None:
        return None
    # Resize pred to gt shape if needed
    if pred.shape != gt.shape:
        pred = np.array(
            Image.fromarray(pred.astype(np.uint8) * 255).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST
            )
        ) > 127

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    pixel_acc = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "IoU": iou,
        "Dice": dice,
        "Pixel Acc": pixel_acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ground Segmentation Comparison",
    page_icon="🏔️",
    layout="wide",
)

st.title("🏔️ Ground Segmentation Comparison")

# ── sidebar ──────────────────────────────────────────────────────────────────

cameras = discover_cameras()
depth_combos = discover_depth_combos()
methods = discover_methods()
fusion_strategies = discover_fusion_strategies()
dinov3_fusion_strategies = discover_dinov3_fusion_strategies()
ensemble_variants = discover_ensemble_variants()
weighted_variants = discover_weighted_variants()
student_variants = discover_student_variants()

if not cameras:
    st.error(f"No cameras found in {DATA_DIR}")
    st.stop()

with st.sidebar:
    st.header("Navigation")

    camera = st.selectbox("Camera", cameras)
    images = list_images(camera)

    if not images:
        st.warning(f"No images found for {camera}")
        st.stop()

    img_idx = st.slider("Image index", 0, len(images) - 1, 0)

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅ Prev") and img_idx > 0:
            img_idx -= 1
    with col_next:
        if st.button("Next ➡") and img_idx < len(images) - 1:
            img_idx += 1

    selected_image = images[img_idx]
    stem = stem_from_filename(selected_image)
    st.caption(f"**{selected_image}**")
    st.caption(f"Image {img_idx + 1} of {len(images)}")

    st.divider()
    st.header("Methods")

    available_methods = list(methods.keys())
    selected_methods = st.multiselect(
        "Segmentation methods",
        available_methods,
        default=available_methods,
    )

    if depth_combos:
        st.divider()
        st.header("Depth Parameters")
        selected_depth_combo = st.selectbox(
            "Depth combo (for comparison tab)",
            depth_combos,
            index=len(depth_combos) // 2,  # default to middle
        )


# ── load data for current image ─────────────────────────────────────────────

original = load_image(DATA_DIR / camera / selected_image)
depth_map = load_image(DEPTH_OUTPUT / "shared_depth" / camera / f"{stem}_depth.png")

# Load method masks
method_masks = {}
for method_name in selected_methods:
    mask_path = methods[method_name] / camera / f"{stem}_ground.png"
    method_masks[method_name] = load_mask(mask_path)

# Load depth combo masks
depth_masks = {}
for combo in depth_combos:
    mask_path = DEPTH_OUTPUT / combo / camera / f"{stem}_ground.png"
    depth_masks[combo] = load_mask(mask_path)

# Load fusion masks (DINOv2 + DINOv3)
fusion_masks = {}
for strategy_name, strategy_dir in fusion_strategies.items():
    mask_path = strategy_dir / camera / f"{stem}_ground.png"
    fusion_masks[strategy_name] = load_mask(mask_path)

dinov3_fusion_masks = {}
for strategy_name, strategy_dir in dinov3_fusion_strategies.items():
    mask_path = strategy_dir / camera / f"{stem}_ground.png"
    dinov3_fusion_masks[strategy_name] = load_mask(mask_path)

# Load PCA visualization (DINOv2 + DINOv3)
pca_viz = load_image(DINOV2_PCA_OUTPUT / camera / f"{stem}_pca.png")
pca_mask = load_mask(DINOV2_PCA_OUTPUT / camera / f"{stem}_ground.png")
dinov3_pca_viz = load_image(DINOV3_PCA_OUTPUT / camera / f"{stem}_pca.png")
dinov3_pca_mask = load_mask(DINOV3_PCA_OUTPUT / camera / f"{stem}_ground.png")

# Load depth ground mask (best combo or default N0.05_F0.95)
depth_default_combo = "N0.05_F0.95"
depth_ground_mask = depth_masks.get(depth_default_combo)

# Load ensemble masks and confidence maps
ensemble_data = {}
for variant_name, variant_dir in ensemble_variants.items():
    mask_path = variant_dir / camera / f"{stem}_ground.png"
    conf_path = variant_dir / camera / f"{stem}_confidence.png"
    agree_path = variant_dir / camera / f"{stem}_agreement.png"
    ensemble_data[variant_name] = {
        "mask": load_mask(mask_path),
        "confidence": load_image(conf_path),
        "agreement": load_image(agree_path),
    }

# Load weighted ensemble masks and confidence maps
weighted_data = {}
for variant_name, variant_dir in weighted_variants.items():
    mask_path = variant_dir / camera / f"{stem}_ground.png"
    conf_path = variant_dir / camera / f"{stem}_confidence.png"
    agree_path = variant_dir / camera / f"{stem}_agreement.png"
    weighted_data[variant_name] = {
        "mask": load_mask(mask_path),
        "confidence": load_image(conf_path),
        "agreement": load_image(agree_path),
    }

# Load student model masks and optional probability maps
student_masks = {}
for variant_name, variant_dir in student_variants.items():
    mask_path = variant_dir / camera / f"{stem}_ground.png"
    prob_path = variant_dir / camera / f"{stem}_probability.png"
    student_masks[variant_name] = {
        "mask": load_mask(mask_path),
        "probability": load_image(prob_path),
    }


# ── ground truth annotations ──────────────────────────────────────────────────

gt_annotations = load_ground_truth_annotations()

# ── tabs ─────────────────────────────────────────────────────────────────────

tab_compare, tab_ensemble, tab_weighted, tab_fusion, tab_grid, tab_stats, tab_student, tab_ground_eval, tab_snow_compare, tab_snow_ensemble, tab_snow_eval = st.tabs([
    "📷 Image Comparison",
    "🗳️ Ensemble",
    "⚖️ Weighted",
    "🔀 Fusion Grid",
    "🔲 Parameter Grid",
    "📊 Statistics",
    "🎓 Student",
    "📐 Ground Evaluation",
    "❄️ Snow Teachers",
    "❄️ Snow Ensemble",
    "❄️ Snow Evaluation",
])


# ── Tab 1: Image Comparison ─────────────────────────────────────────────────

with tab_compare:
    # Row 1: Original + Depth map
    col1, col2 = st.columns(2)
    with col1:
        if original is not None:
            st.image(original, caption="Original", width="stretch")
        else:
            st.warning("Original image not found")
    with col2:
        if depth_map is not None:
            st.image(depth_map, caption="Depth Map", width="stretch")
        else:
            st.info("No depth map available")

    st.divider()

    # Row 2: Method comparison (overlays)
    all_to_show = {}

    # Add selected segmentation methods
    for method_name in selected_methods:
        mask = method_masks.get(method_name)
        if mask is not None and original is not None:
            all_to_show[method_name] = mask

    # Add selected depth combo
    if depth_combos and original is not None:
        depth_mask = depth_masks.get(selected_depth_combo)
        if depth_mask is not None:
            label = f"Depth ({selected_depth_combo})"
            all_to_show[label] = depth_mask

    if all_to_show:
        st.subheader("Method Comparison")
        n_methods = len(all_to_show)
        cols = st.columns(min(n_methods, 4))
        for i, (name, mask) in enumerate(all_to_show.items()):
            with cols[i % min(n_methods, 4)]:
                overlay = make_overlay(original, mask)
                pct = ground_pct(mask)
                st.image(overlay, caption=f"{name} ({pct:.1f}%)",
                         width="stretch")
    elif not selected_methods and not depth_combos:
        st.info("No segmentation methods available yet")

    # Row 3: Expandable depth grid
    if depth_combos and original is not None:
        with st.expander("🔲 Full Depth Parameter Grid", expanded=False):
            near_vals = sorted(set(parse_combo(c)[0] for c in depth_combos))
            far_vals = sorted(set(parse_combo(c)[1] for c in depth_combos))

            # Header row
            header_cols = st.columns([1] + [1] * len(far_vals))
            with header_cols[0]:
                st.markdown("**near \\ far**")
            for j, fv in enumerate(far_vals):
                with header_cols[j + 1]:
                    st.markdown(f"**F={fv}**")

            # Grid rows
            for nv in near_vals:
                row_cols = st.columns([1] + [1] * len(far_vals))
                with row_cols[0]:
                    st.markdown(f"**N={nv}**")
                for j, fv in enumerate(far_vals):
                    combo_name = f"N{nv}_F{fv}"
                    mask = depth_masks.get(combo_name)
                    with row_cols[j + 1]:
                        if mask is not None:
                            overlay = make_overlay(original, mask)
                            pct = ground_pct(mask)
                            st.image(overlay, caption=f"{pct:.1f}%",
                                     width="stretch")
                        else:
                            st.caption("N/A")


# ── Tab 2: Ensemble ────────────────────────────────────────────────────────

with tab_ensemble:
    has_ensemble = any(d["mask"] is not None for d in ensemble_data.values())

    if not has_ensemble:
        st.info("No ensemble results available yet. Run `python segmentation/ground_utils.py ensemble` first.")
    elif original is None:
        st.warning("Original image not found")
    else:
        st.subheader(f"Ensemble Voting — {selected_image}")

        # Row 1: All 6 individual method masks side by side
        st.markdown("#### Individual Methods (6 teachers)")
        teacher_masks = {}
        teacher_labels = {
            "SegFormer": SEGFORMER_OUTPUT,
            "DeepLabV3+": DEEPLABV3_OUTPUT,
            "Depth": DEPTH_OUTPUT / depth_default_combo,
            "SAM3 (text)": SAM3_DINO_OUTPUT / "text",
            "SAM3+DINO": SAM3_DINO_OUTPUT / "dino+sam",
            "DINOv2 PCA": DINOV2_PCA_OUTPUT,
        }
        for label, d in teacher_labels.items():
            mp = d / camera / f"{stem}_ground.png"
            teacher_masks[label] = load_mask(mp)

        active_teachers = {k: v for k, v in teacher_masks.items() if v is not None}
        if active_teachers:
            t_cols = st.columns(len(active_teachers))
            for i, (name, mask) in enumerate(active_teachers.items()):
                with t_cols[i]:
                    overlay = make_overlay(original, mask)
                    pct = ground_pct(mask)
                    st.image(overlay, caption=f"{name}\n({pct:.1f}%)",
                             width="stretch")

        # Row 2: Ensemble results per variant
        st.divider()
        st.markdown("#### Ensemble Results")

        active_ensembles = {k: v for k, v in ensemble_data.items()
                           if v["mask"] is not None}

        if active_ensembles:
            # Show overlay for each variant
            ens_cols = st.columns(len(active_ensembles))
            for i, (variant, data) in enumerate(active_ensembles.items()):
                with ens_cols[i]:
                    overlay = make_overlay(original, data["mask"])
                    pct = ground_pct(data["mask"])
                    votes = variant.replace("majority_", "")
                    st.image(overlay,
                             caption=f"Majority {votes}/6\n({pct:.1f}%)",
                             width="stretch")

            # Row 3: Confidence maps
            st.divider()
            st.markdown("#### Confidence Maps")
            st.caption("Brighter = more methods agree this pixel is ground")

            conf_cols = st.columns(len(active_ensembles))
            for i, (variant, data) in enumerate(active_ensembles.items()):
                with conf_cols[i]:
                    if data["confidence"] is not None:
                        st.image(data["confidence"],
                                 caption=f"{variant} confidence",
                                 width="stretch")
                    else:
                        st.caption("N/A")

            # Row 4: Agreement maps (if debug was enabled)
            has_agreement = any(d["agreement"] is not None
                               for d in active_ensembles.values())
            if has_agreement:
                st.divider()
                st.markdown("#### Agreement Maps")
                st.caption("Red=no agreement, Yellow=partial, Green=unanimous")

                agr_cols = st.columns(len(active_ensembles))
                for i, (variant, data) in enumerate(active_ensembles.items()):
                    with agr_cols[i]:
                        if data["agreement"] is not None:
                            st.image(data["agreement"],
                                     caption=f"{variant} agreement",
                                     width="stretch")
                        else:
                            st.caption("N/A")


# ── Tab 3: Weighted Ensemble ─────────────────────────────────────────────────

# Method weights reference for display
DISPLAY_WEIGHTS = {
    "SegFormer": 3,
    "SAM3 (text)": 3,
    "DeepLabV3+": 3,
    "SAM3+DINO": "1/3",
    "DINOv2 PCA": "1/3",
    "Depth": "1/3",
}

with tab_weighted:
    if original is None:
        st.warning("Original image not found")
    else:
        st.subheader(f"Weighted vs Majority Ensemble — {selected_image}")

        # Row 1: 6 teachers with their weights annotated
        st.markdown("#### Teacher Methods — with Quality Weights")
        st.caption(
            "Weights: **SegFormer=3, SAM3(text)=3, DeepLabV3+=3** vs "
            "SAM3+DINO=⅓, DINOv2 PCA=⅓, Depth=⅓"
        )

        teacher_labels = {
            "SegFormer": (SEGFORMER_OUTPUT, 3),
            "DeepLabV3+": (DEEPLABV3_OUTPUT, 3),
            "SAM3 (text)": (SAM3_DINO_OUTPUT / "text", 3),
            "SAM3+DINO": (SAM3_DINO_OUTPUT / "dino+sam", "1/3"),
            "DINOv2 PCA": (DINOV2_PCA_OUTPUT, "1/3"),
            "Depth": (DEPTH_OUTPUT / depth_default_combo, "1/3"),
        }

        teacher_cols = st.columns(len(teacher_labels))
        for i, (label, (t_dir, weight)) in enumerate(teacher_labels.items()):
            mp = t_dir / camera / f"{stem}_ground.png"
            mask = load_mask(mp)
            with teacher_cols[i]:
                if mask is not None:
                    overlay = make_overlay(original, mask)
                    pct = ground_pct(mask)
                    # Color caption: high weight = green, low = orange
                    w_str = f"w={weight}"
                    st.image(overlay, caption=f"{label}\n{w_str} | {pct:.1f}%",
                             width="stretch")
                else:
                    st.caption(f"{label}\nN/A")

        # Row 2: Side-by-side — best majority vs weighted variants
        st.divider()
        st.markdown("#### Majority vs Weighted Comparison")

        # Pick best majority for comparison (prefer majority_4)
        best_majority = None
        for preferred in ["majority_4", "majority_3", "majority_5"]:
            if preferred in ensemble_data and ensemble_data[preferred]["mask"] is not None:
                best_majority = preferred
                break

        active_weighted = {k: v for k, v in weighted_data.items()
                           if v["mask"] is not None}

        if not best_majority and not active_weighted:
            st.info(
                "No weighted ensemble results yet. Run:\n\n"
                "```bash\nWEIGHTED=1 MIN_VOTES=5 sbatch run_ensemble_voting.sbatch\n```"
            )
        else:
            # Build comparison list: original + majority (if exists) + all weighted
            compare_items = []
            if best_majority:
                compare_items.append((best_majority, ensemble_data[best_majority]))
            compare_items.extend(active_weighted.items())

            n_compare = len(compare_items)
            cmp_cols = st.columns(n_compare + 1)

            with cmp_cols[0]:
                st.image(original, caption="Original", width="stretch")

            for i, (variant, data) in enumerate(compare_items):
                with cmp_cols[i + 1]:
                    overlay = make_overlay(original, data["mask"])
                    pct = ground_pct(data["mask"])
                    # Friendly label
                    if variant.startswith("majority_"):
                        votes = variant.replace("majority_", "")
                        caption = f"Majority {votes}/6\n({pct:.1f}%)"
                    else:
                        thr = variant.replace("weighted_", "")
                        caption = f"Weighted ≥{thr}/10\n({pct:.1f}%)"
                    st.image(overlay, caption=caption, width="stretch")

        # Row 3: Confidence map comparison
        st.divider()
        st.markdown("#### Confidence Maps")
        st.caption(
            "For majority: fraction of 6 methods agreeing. "
            "For weighted: fraction of total weight (10) agreeing. "
            "Brighter = higher confidence."
        )

        conf_items = []
        if best_majority and ensemble_data[best_majority]["confidence"] is not None:
            conf_items.append((best_majority, ensemble_data[best_majority]["confidence"]))
        for variant, data in active_weighted.items():
            if data["confidence"] is not None:
                conf_items.append((variant, data["confidence"]))

        if conf_items:
            conf_cols = st.columns(len(conf_items))
            for i, (variant, conf_img) in enumerate(conf_items):
                with conf_cols[i]:
                    if variant.startswith("majority_"):
                        thr = variant.replace("majority_", "")
                        label = f"Majority {thr}/6"
                    else:
                        thr = variant.replace("weighted_", "")
                        label = f"Weighted ≥{thr}/10"
                    st.image(conf_img, caption=label, width="stretch")
        else:
            st.info("No confidence maps available")

        # Row 4: Agreement maps if debug was run
        has_agree = (
            (best_majority and ensemble_data.get(best_majority, {}).get("agreement") is not None)
            or any(d["agreement"] is not None for d in active_weighted.values())
        )
        if has_agree:
            st.divider()
            st.markdown("#### Agreement Maps (debug)")
            st.caption("Red=no agreement, Yellow=partial, Green=high agreement")

            agr_items = []
            if best_majority and ensemble_data[best_majority]["agreement"] is not None:
                agr_items.append((best_majority, ensemble_data[best_majority]["agreement"]))
            for variant, data in active_weighted.items():
                if data["agreement"] is not None:
                    agr_items.append((variant, data["agreement"]))

            agr_cols = st.columns(len(agr_items))
            for i, (variant, agr_img) in enumerate(agr_items):
                with agr_cols[i]:
                    if variant.startswith("majority_"):
                        thr = variant.replace("majority_", "")
                        label = f"Majority {thr}/6"
                    else:
                        thr = variant.replace("weighted_", "")
                        label = f"Weighted ≥{thr}/10"
                    st.image(agr_img, caption=label, width="stretch")


# ── Tab 4: Fusion Grid ─────────────────────────────────────────────────────

with tab_fusion:
    has_fusion = any(m is not None for m in fusion_masks.values())
    has_sources = pca_mask is not None or depth_ground_mask is not None

    if not has_fusion and not has_sources:
        st.info("No fusion results available yet. Run `fusion_depth_pca.py` first.")
    elif original is None:
        st.warning("Original image not found")
    else:
        st.subheader(f"DINOv2 + Depth Fusion — {selected_image}")

        # Row 1: Source inputs (PCA and Depth)
        st.markdown("#### Source Inputs")
        src_cols = st.columns(4)

        with src_cols[0]:
            if pca_viz is not None:
                st.image(pca_viz, caption="DINOv2 PCA (RGB)",
                         width="stretch")
            else:
                st.caption("PCA viz N/A")

        with src_cols[1]:
            if pca_mask is not None:
                overlay = make_overlay(original, pca_mask)
                pct = ground_pct(pca_mask)
                st.image(overlay, caption=f"DINOv2 PCA Mask ({pct:.1f}%)",
                         width="stretch")
            else:
                st.caption("PCA mask N/A")

        with src_cols[2]:
            if depth_map is not None:
                st.image(depth_map, caption="Depth Map",
                         width="stretch")
            else:
                st.caption("Depth map N/A")

        with src_cols[3]:
            if depth_ground_mask is not None:
                overlay = make_overlay(original, depth_ground_mask)
                pct = ground_pct(depth_ground_mask)
                st.image(overlay,
                         caption=f"Depth Mask ({depth_default_combo}, {pct:.1f}%)",
                         width="stretch")
            else:
                st.caption("Depth mask N/A")

        # Row 2: Fusion strategies
        if has_fusion:
            st.divider()
            st.markdown("#### Fusion Strategies")

            strategy_labels = {
                "intersect": "Intersect (AND)",
                "union": "Union (OR)",
                "pca_depth_filter": "PCA + Depth Filter",
                "depth_pca_filter": "Depth + PCA Filter",
            }

            active_strategies = {k: v for k, v in fusion_masks.items()
                                 if v is not None}

            if active_strategies:
                fus_cols = st.columns(len(active_strategies))
                for i, (strat_name, mask) in enumerate(active_strategies.items()):
                    with fus_cols[i]:
                        overlay = make_overlay(original, mask)
                        pct = ground_pct(mask)
                        label = strategy_labels.get(strat_name, strat_name)
                        st.image(overlay,
                                 caption=f"{label}\n({pct:.1f}%)",
                                 width="stretch")
            else:
                st.info("No fusion masks found for this image")

            # Row 3: Side-by-side binary masks (no overlay, raw black/white)
            st.divider()
            st.markdown("#### Raw Binary Masks")

            all_masks = {}
            if pca_mask is not None:
                all_masks["DINOv2 PCA"] = pca_mask
            if depth_ground_mask is not None:
                all_masks[f"Depth ({depth_default_combo})"] = depth_ground_mask
            for strat_name, mask in active_strategies.items():
                label = strategy_labels.get(strat_name, strat_name)
                all_masks[label] = mask

            if all_masks:
                mask_cols = st.columns(len(all_masks))
                for i, (name, mask) in enumerate(all_masks.items()):
                    with mask_cols[i]:
                        mask_display = (mask * 255).astype(np.uint8)
                        pct = ground_pct(mask)
                        st.image(mask_display, caption=f"{name}\n({pct:.1f}%)",
                                 width="stretch")

        # ── DINOv3 + Depth fusion ─────────────────────────────────────────────
        active_dinov3 = {k: v for k, v in dinov3_fusion_masks.items()
                         if v is not None}
        if active_dinov3 or dinov3_pca_mask is not None:
            st.divider()
            st.subheader("DINOv3 PCA (SAT) + Depth Fusion")
            st.caption("Same strategies, cleaner satellite-pretrained features")

            # Source row: DINOv3 PCA viz + mask
            src_cols = st.columns(4)
            with src_cols[0]:
                if dinov3_pca_viz is not None:
                    st.image(dinov3_pca_viz, caption="DINOv3 PCA (RGB)",
                             width="stretch")
                else:
                    st.caption("DINOv3 PCA viz N/A")
            with src_cols[1]:
                if dinov3_pca_mask is not None:
                    overlay = make_overlay(original, dinov3_pca_mask)
                    pct = ground_pct(dinov3_pca_mask)
                    st.image(overlay, caption=f"DINOv3 PCA Mask ({pct:.1f}%)",
                             width="stretch")
                else:
                    st.caption("DINOv3 PCA mask N/A")
            with src_cols[2]:
                if depth_map is not None:
                    st.image(depth_map, caption="Depth Map",
                             width="stretch")
                else:
                    st.caption("Depth map N/A")
            with src_cols[3]:
                if depth_ground_mask is not None:
                    overlay = make_overlay(original, depth_ground_mask)
                    pct = ground_pct(depth_ground_mask)
                    st.image(overlay,
                             caption=f"Depth Mask ({depth_default_combo}, {pct:.1f}%)",
                             width="stretch")
                else:
                    st.caption("Depth mask N/A")

            if active_dinov3:
                st.markdown("#### DINOv3 Fusion Strategies")
                fus_cols = st.columns(len(active_dinov3))
                for i, (strat_name, mask) in enumerate(active_dinov3.items()):
                    with fus_cols[i]:
                        overlay = make_overlay(original, mask)
                        pct = ground_pct(mask)
                        label = strategy_labels.get(strat_name, strat_name)
                        st.image(overlay, caption=f"{label}\n({pct:.1f}%)",
                                 width="stretch")
            else:
                st.info(
                    "No DINOv3 fusion results yet. Once DINOv3 PCA jobs finish, run:\n\n"
                    "```bash\nsbatch segmentation/run_fusion_dinov3_depth.sbatch\n```"
                )


# ── Tab 5: Depth Parameter Grid ─────────────────────────────────────────────

with tab_grid:
    if not depth_combos:
        st.info("No depth parameter grid results available")
    elif original is None:
        st.warning("Original image not found")
    else:
        st.subheader(f"Parameter Grid — {selected_image}")

        near_vals = sorted(set(parse_combo(c)[0] for c in depth_combos))
        far_vals = sorted(set(parse_combo(c)[1] for c in depth_combos))

        # Build ground_pct matrix for heatmap
        pct_matrix = []
        for nv in near_vals:
            row = []
            for fv in far_vals:
                combo_name = f"N{nv}_F{fv}"
                mask = depth_masks.get(combo_name)
                row.append(ground_pct(mask))
            pct_matrix.append(row)

        # Overlays grid
        for i, nv in enumerate(near_vals):
            cols = st.columns(len(far_vals))
            for j, fv in enumerate(far_vals):
                combo_name = f"N{nv}_F{fv}"
                mask = depth_masks.get(combo_name)
                with cols[j]:
                    if mask is not None:
                        overlay = make_overlay(original, mask)
                        pct = ground_pct(mask)
                        st.image(overlay,
                                 caption=f"N={nv} F={fv}\n{pct:.1f}%",
                                 width="stretch")
                    else:
                        st.caption(f"N={nv} F={fv}\nN/A")

        # Heatmap
        st.divider()
        st.subheader("Ground % Heatmap")

        try:
            import plotly.graph_objects as go

            fig = go.Figure(data=go.Heatmap(
                z=pct_matrix,
                x=[f"F={fv}" for fv in far_vals],
                y=[f"N={nv}" for nv in near_vals],
                text=[[f"{v:.1f}%" for v in row] for row in pct_matrix],
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale="YlGn",
                colorbar_title="Ground %",
            ))
            fig.update_layout(
                title=f"Ground Coverage — {camera} / {stem}",
                xaxis_title="Far Clip",
                yaxis_title="Near Clip",
                height=400,
            )
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            st.warning("Install plotly for interactive heatmap: `pip install plotly`")
            st.table({
                f"F={fv}": [f"{pct_matrix[i][j]:.1f}%"
                            for i in range(len(near_vals))]
                for j, fv in enumerate(far_vals)
            })


# ── Tab 6: Statistics ────────────────────────────────────────────────────────

with tab_stats:
    st.subheader(f"Statistics — {camera}")

    try:
        import pandas as pd
        import plotly.express as px

        # Load all depth summary CSVs
        combo_stats = []
        for combo in depth_combos:
            csv_path = DEPTH_OUTPUT / combo / "depth_summary.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df_cam = df[df["camera"] == camera]
                if not df_cam.empty:
                    near, far = parse_combo(combo)
                    combo_stats.append({
                        "combo": combo,
                        "near_clip": near,
                        "far_clip": far,
                        "mean_ground_pct": df_cam["ground_pct"].mean(),
                        "std_ground_pct": df_cam["ground_pct"].std(),
                        "median_ground_pct": df_cam["ground_pct"].median(),
                        "n_images": len(df_cam),
                    })

        if combo_stats:
            stats_df = pd.DataFrame(combo_stats)

            # Heatmap: mean ground % across parameter grid
            st.markdown("### Mean Ground % Across Parameter Grid")
            pivot = stats_df.pivot(
                index="near_clip", columns="far_clip", values="mean_ground_pct"
            )
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=[f"F={c}" for c in pivot.columns],
                y=[f"N={r}" for r in pivot.index],
                text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale="YlGn",
                colorbar_title="Mean Ground %",
            ))
            fig.update_layout(
                title=f"Mean Ground Coverage — {camera} ({combo_stats[0]['n_images']} images)",
                xaxis_title="Far Clip",
                yaxis_title="Near Clip",
                height=400,
            )
            st.plotly_chart(fig, width="stretch")

            # Summary table
            st.markdown("### Summary Table")
            display_df = stats_df[[
                "combo", "mean_ground_pct", "std_ground_pct",
                "median_ground_pct", "n_images"
            ]].round(2)
            display_df.columns = ["Parameters", "Mean %", "Std %", "Median %", "Images"]
            st.dataframe(display_df, width="stretch", hide_index=True)

            # Distribution: box plot per combo
            st.markdown("### Ground % Distribution per Parameter Combo")
            all_dfs = []
            for combo in depth_combos:
                csv_path = DEPTH_OUTPUT / combo / "depth_summary.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df_cam = df[df["camera"] == camera].copy()
                    if not df_cam.empty:
                        df_cam["combo"] = combo
                        all_dfs.append(df_cam)

            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                fig = px.box(
                    combined, x="combo", y="ground_pct",
                    title=f"Ground % Distribution — {camera}",
                    labels={"combo": "Parameters", "ground_pct": "Ground %"},
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, width="stretch")

        # Cross-method comparison (if methods have data)
        st.divider()
        st.markdown("### Cross-Method Comparison")
        st.caption("Comparing ground % across different segmentation methods for this camera")

        method_summary = []

        # Segformer
        for method_name, method_dir in methods.items():
            cam_dir = method_dir / camera
            if cam_dir.exists():
                masks = list(cam_dir.glob("*_ground.png"))
                if masks:
                    pcts = []
                    for mp in masks[:200]:  # sample for speed
                        m = load_mask(mp)
                        if m is not None:
                            pcts.append(ground_pct(m))
                    if pcts:
                        method_summary.append({
                            "Method": method_name,
                            "Mean Ground %": round(np.mean(pcts), 2),
                            "Std %": round(np.std(pcts), 2),
                            "Median %": round(np.median(pcts), 2),
                            "Images": len(masks),
                            "Sampled": len(pcts),
                        })

        # Add best depth combo
        if combo_stats:
            best = max(combo_stats, key=lambda x: x["mean_ground_pct"])
            method_summary.append({
                "Method": f"Depth ({best['combo']})",
                "Mean Ground %": round(best["mean_ground_pct"], 2),
                "Std %": round(best["std_ground_pct"], 2),
                "Median %": round(best["median_ground_pct"], 2),
                "Images": best["n_images"],
                "Sampled": best["n_images"],
            })

        if method_summary:
            st.dataframe(
                pd.DataFrame(method_summary),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No method results available for this camera yet")

    except ImportError:
        st.warning("Install pandas and plotly for statistics: `pip install pandas plotly`")


# ── Tab 7: Student Model ──────────────────────────────────────────────────────

with tab_student:
    if not student_variants:
        st.info(
            "No student model results yet.\n\n"
            "Run inference first:\n"
            "```bash\n"
            "CHECKPOINT=models/student_ground/checkpoints/dinov2_large_.../best-epoch=15-....ckpt \\\n"
            "  sbatch models/student_ground/run_student_inference.sbatch\n"
            "```\n"
            f"Expected output: `{STUDENT_OUTPUT}/{{camera}}/{{stem}}_ground.png`"
        )
    elif original is None:
        st.warning("Original image not found")
    else:
        st.subheader(f"Student Ground Segmentation — {selected_image}")

        # Find best weighted ensemble for reference (the training label)
        best_weighted = None
        for preferred in ["weighted_5", "weighted_4", "weighted_3"]:
            if preferred in weighted_data and weighted_data[preferred]["mask"] is not None:
                best_weighted = preferred
                break

        # ── Row 1: Original + training label reference ────────────────────────
        st.markdown("#### Reference")
        ref_cols = st.columns(2)
        with ref_cols[0]:
            st.image(original, caption="Original", width="stretch")
        with ref_cols[1]:
            if best_weighted:
                wdata = weighted_data[best_weighted]
                overlay = make_overlay(original, wdata["mask"])
                pct = ground_pct(wdata["mask"])
                thr = best_weighted.replace("weighted_", "")
                st.image(
                    overlay,
                    caption=f"Weighted Ensemble ≥{thr}/10 (training label, {pct:.1f}%)",
                    width="stretch",
                )
            else:
                st.info("No weighted ensemble available for reference")

        # ── Row 2: Student predictions per variant ────────────────────────────
        st.divider()
        st.markdown("#### Student Predictions")

        active_student = {k: v for k, v in student_masks.items() if v["mask"] is not None}

        if not active_student:
            st.info(
                f"No predictions found for **{selected_image}**.\n\n"
                "The inference job may still be running, or the image wasn't processed."
            )
        else:
            s_cols = st.columns(len(active_student))
            for i, (variant, sdata) in enumerate(active_student.items()):
                with s_cols[i]:
                    overlay = make_overlay(original, sdata["mask"])
                    pct = ground_pct(sdata["mask"])
                    st.image(overlay, caption=f"{variant}\n({pct:.1f}%)",
                             width="stretch")

            # ── Row 3: Agreement with training label ──────────────────────────
            if best_weighted:
                st.divider()
                st.markdown("#### Agreement with Weighted Ensemble")
                st.caption("How well the student reproduces its training labels on this image")

                ens_mask = weighted_data[best_weighted]["mask"]
                thr = best_weighted.replace("weighted_", "")

                metrics_rows = []
                for variant, sdata in active_student.items():
                    s_mask = sdata["mask"]
                    intersection = (s_mask & ens_mask).sum()
                    union = (s_mask | ens_mask).sum()
                    iou = intersection / union if union > 0 else 0.0
                    agree_pct = (s_mask == ens_mask).mean() * 100
                    tp = (s_mask & ens_mask).sum()
                    fp = (s_mask & ~ens_mask).sum()
                    fn = (~s_mask & ens_mask).sum()
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    metrics_rows.append({
                        "Variant": variant,
                        f"vs Weighted ≥{thr}": "",
                        "IoU": f"{iou:.3f}",
                        "F1": f"{f1:.3f}",
                        "Precision": f"{prec:.3f}",
                        "Recall": f"{rec:.3f}",
                        "Pixel Agree %": f"{agree_pct:.1f}%",
                        "Student %": f"{ground_pct(s_mask):.1f}%",
                        f"Ensemble %": f"{ground_pct(ens_mask):.1f}%",
                    })

                try:
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame(metrics_rows).drop(columns=[f"vs Weighted ≥{thr}"]),
                        width="stretch",
                        hide_index=True,
                    )
                except ImportError:
                    for row in metrics_rows:
                        st.markdown(
                            f"**{row['Variant']}** — IoU: {row['IoU']} | "
                            f"F1: {row['F1']} | Agreement: {row['Pixel Agree %']}"
                        )

        # ── Row 4: Probability maps (if --save-prob was used) ─────────────────
        has_prob = any(v["probability"] is not None for v in student_masks.values())
        if has_prob:
            st.divider()
            st.markdown("#### Probability Maps")
            st.caption("Brighter = higher model confidence that the pixel is ground")

            prob_variants = {k: v for k, v in student_masks.items()
                             if v["probability"] is not None}
            p_cols = st.columns(len(prob_variants))
            for i, (variant, sdata) in enumerate(prob_variants.items()):
                with p_cols[i]:
                    st.image(sdata["probability"],
                             caption=f"{variant} probability",
                             width="stretch")


# ── Tab 8: Ground Evaluation ─────────────────────────────────────────────────

@st.cache_data
def build_all_methods_registry():
    """Build a dict of {method_name: output_dir} for ALL methods to evaluate."""
    registry = {}

    # Flat methods: {method}_output/{camera}/{stem}_ground.png
    for name, d in [
        ("SegFormer", SEGFORMER_OUTPUT),
        ("DeepLabV3+", DEEPLABV3_OUTPUT),
        ("DINOv2 PCA", DINOV2_PCA_OUTPUT),
        ("DINOv3 PCA", DINOV3_PCA_OUTPUT),
    ]:
        if d.exists():
            registry[name] = d

    # SAM3+DINO variants
    if SAM3_DINO_OUTPUT.exists():
        for sub in ["dino+sam", "text"]:
            d = SAM3_DINO_OUTPUT / sub
            if d.exists():
                registry[f"SAM3 ({sub})"] = d

    # Ensemble variants
    if ENSEMBLE_OUTPUT.exists():
        for d in sorted(ENSEMBLE_OUTPUT.iterdir()):
            if d.is_dir() and (d.name.startswith("majority_") or d.name.startswith("weighted_")):
                registry[f"Ensemble {d.name}"] = d

    # Fusion (DINOv2)
    if FUSION_OUTPUT.exists():
        for sub in ["intersect", "union", "pca_depth_filter", "depth_pca_filter"]:
            d = FUSION_OUTPUT / sub
            if d.exists():
                registry[f"Fusion v2 {sub}"] = d

    # Fusion (DINOv3)
    if FUSION_DINOV3_OUTPUT.exists():
        for sub in ["intersect", "union", "pca_depth_filter", "depth_pca_filter"]:
            d = FUSION_DINOV3_OUTPUT / sub
            if d.exists():
                registry[f"Fusion v3 {sub}"] = d

    # Student variants
    if STUDENT_OUTPUT.exists():
        for d in sorted(STUDENT_OUTPUT.iterdir()):
            if d.is_dir() and not re.match(r"C[A-Z]\d+", d.name):
                registry[f"Student {d.name}"] = d
            elif d.is_dir() and re.match(r"C[A-Z]\d+", d.name):
                # Flat layout — single student model
                registry["Student"] = STUDENT_OUTPUT
                break

    # Best depth combo (N0.05_F0.95 default)
    if DEPTH_OUTPUT.exists():
        for combo in ["N0.05_F0.95", "N0.05_F0.9", "N0.1_F0.95"]:
            d = DEPTH_OUTPUT / combo
            if d.exists():
                registry[f"Depth ({combo})"] = d
                break

    return registry


@st.cache_data
def compute_full_evaluation(_gt_annotations, _all_methods):
    """Compute metrics for all methods on all annotated images.

    Returns list of dicts: [{method, camera, stem, IoU, Dice, ...}]
    """
    rows = []
    for method_name, method_dir in _all_methods.items():
        for cam, stems in _gt_annotations.items():
            cam_dir = method_dir / cam
            if not cam_dir.exists():
                continue
            for s, gt_mask in stems.items():
                mask_path = cam_dir / f"{s}_ground.png"
                pred = load_mask(mask_path)
                metrics = compute_seg_metrics(pred, gt_mask)
                if metrics is not None:
                    rows.append({"Method": method_name, "Camera": cam, "Stem": s, **metrics})
    return rows


with tab_ground_eval:
    if not gt_annotations:
        st.warning(
            "No ground truth annotations found.\n\n"
            "Upload `ground_annotations.json` to `data/annotations/`."
        )
    else:
        n_cams = len(gt_annotations)
        n_imgs = sum(len(v) for v in gt_annotations.values())
        st.subheader(f"📐 Ground Evaluation — {n_imgs} annotated images across {n_cams} cameras")

        all_methods_reg = build_all_methods_registry()

        if not all_methods_reg:
            st.error("No segmentation method outputs found.")
        else:
            # ── Method filter ────────────────────────────────────────
            method_names = list(all_methods_reg.keys())
            student_methods = [m for m in method_names if m.startswith("Student")]
            default_selection = student_methods if student_methods else method_names
            selected_methods = st.multiselect(
                "Methods to evaluate",
                method_names,
                default=default_selection,
            )
            all_methods_reg = {k: v for k, v in all_methods_reg.items() if k in selected_methods}
            # ── Image selector: only annotated images ─────────────────────
            cam_gt = gt_annotations.get(camera, {})
            if cam_gt:
                annotated_stems = sorted(cam_gt.keys())
                eval_sel = st.selectbox(
                    f"Annotated image ({len(annotated_stems)} for {camera})",
                    annotated_stems,
                    format_func=lambda s: f"{s}.jpg",
                )
                eval_stem = eval_sel
                gt_mask_current = cam_gt[eval_stem]
                eval_img = load_image(DATA_DIR / camera / f"{eval_stem}.jpg")
            else:
                st.info(f"No ground truth annotations for **{camera}**.")
                eval_stem = None
                gt_mask_current = None
                eval_img = None

            if gt_mask_current is not None and eval_img is not None:
                st.markdown("#### Current Image vs Ground Truth")

                gt_overlay = make_overlay(eval_img, gt_mask_current, color=(255, 215, 0), alpha=0.5)

                # Use user-selected methods (limited to first 9 for viz speed)
                viz_methods = dict(list(all_methods_reg.items())[:9])

                current_metrics = []
                for m_name, m_dir in viz_methods.items():
                    pred = load_mask(m_dir / camera / f"{eval_stem}_ground.png")
                    m = compute_seg_metrics(pred, gt_mask_current)
                    if m is not None:
                        current_metrics.append({"Method": m_name, **m})

                # Top row: GT overlay + average metrics
                top_left, top_right = st.columns([1, 1])
                with top_left:
                    st.image(gt_overlay, caption=f"Ground Truth ({ground_pct(gt_mask_current):.1f}% ground)",
                             width="stretch")
                with top_right:
                    if current_metrics:
                        try:
                            import pandas as pd
                            df_cur = pd.DataFrame(current_metrics)
                            avg = df_cur.drop(columns=["Method"]).mean()
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("Avg IoU", f"{avg['IoU']:.3f}")
                            mc2.metric("Avg Dice", f"{avg['Dice']:.3f}")
                            mc3.metric("Avg Pixel Acc", f"{avg['Pixel Acc']:.3f}")
                        except ImportError:
                            pass
                    else:
                        st.info("No method predictions for this image")

                # Per-method overlays — top 6 only for speed
                if current_metrics:
                    st.markdown("##### Method Predictions (top methods)")
                    sorted_metrics = sorted(current_metrics, key=lambda x: x["IoU"], reverse=True)
                    show_top = sorted_metrics[:6]
                    n_cols = 3
                    for row_start in range(0, len(show_top), n_cols):
                        chunk = show_top[row_start:row_start + n_cols]
                        cols = st.columns(n_cols)
                        for ci, entry in enumerate(chunk):
                            m_name = entry["Method"]
                            pred = load_mask(viz_methods[m_name] / camera / f"{eval_stem}_ground.png")
                            if pred is not None and eval_img is not None:
                                ov = make_overlay(eval_img, pred)
                                cols[ci].image(
                                    ov,
                                    caption=f"{m_name}\nIoU={entry['IoU']:.3f}  Dice={entry['Dice']:.3f}",
                                    width="stretch",
                                )

                    # Per-image metrics table (all evaluated methods)
                    try:
                        import pandas as pd
                        st.markdown("##### Per-Method Metrics (this image)")
                        df_img = pd.DataFrame(sorted_metrics)
                        for col in ["IoU", "Dice", "Pixel Acc", "Precision", "Recall", "F1"]:
                            df_img[col] = df_img[col].map(lambda x: f"{x:.4f}")
                        st.dataframe(df_img, width="stretch", hide_index=True)
                    except ImportError:
                        pass

            # ── Full ranking (heavy — behind a button) ────────────────────
            st.divider()
            st.markdown("### Method Ranking (all annotated images)")

            if "eval_rows" not in st.session_state:
                st.session_state["eval_rows"] = None

            if st.button("Compute full ranking", type="primary"):
                with st.spinner("Computing metrics across all annotated images..."):
                    st.session_state["eval_rows"] = compute_full_evaluation(
                        gt_annotations, all_methods_reg
                    )

            eval_rows = st.session_state["eval_rows"]

            if eval_rows:
                try:
                    import pandas as pd
                except ImportError:
                    st.warning("Install pandas for the ranking table: `pip install pandas`")
                    eval_rows = None

            if eval_rows:
                try:
                    from scipy import stats as scipy_stats
                    _HAS_SCIPY = True
                except ImportError:
                    _HAS_SCIPY = False

                def _t_crit_95(df_val):
                    """95% CI t-critical value. Uses scipy if available, else z=1.96."""
                    if _HAS_SCIPY:
                        return scipy_stats.t.ppf(0.975, max(df_val, 1))
                    return 1.96

                df_all = pd.DataFrame(eval_rows)

                # Summary table: mean metrics per method
                summary = (
                    df_all
                    .groupby("Method")[["IoU", "Dice", "Pixel Acc", "Precision", "Recall", "F1"]]
                    .agg(["mean", "std", "count"])
                )
                # Flatten MultiIndex columns
                summary.columns = [f"{m}_{s}" for m, s in summary.columns]
                summary = summary.reset_index()
                summary = summary.sort_values("IoU_mean", ascending=False)

                # Compute 95% CI for IoU: mean ± t * (std / sqrt(n))
                n = summary["IoU_count"]
                se = summary["IoU_std"] / np.sqrt(n)
                t_crit = n.map(lambda x: _t_crit_95(x - 1))
                summary["IoU_ci_lo"] = summary["IoU_mean"] - t_crit * se
                summary["IoU_ci_hi"] = summary["IoU_mean"] + t_crit * se

                # Display summary metrics at the top
                st.markdown("#### Average Metrics")
                m1, m2, m3, m4 = st.columns(4)
                best = summary.iloc[0]
                m1.metric("Best Method", best["Method"])
                m2.metric("Best IoU", f"{best['IoU_mean']:.4f}")
                m3.metric("Best Dice", f"{best['Dice_mean']:.4f}")
                m4.metric("# Images Evaluated", f"{int(best['IoU_count'])}")

                # Ranking table
                ci_label = "95% CI" if _HAS_SCIPY else "~95% CI (z)"
                st.markdown("#### Ranking Table")
                display_df = pd.DataFrame({
                    "Rank": range(1, len(summary) + 1),
                    "Method": summary["Method"].values,
                    "IoU": summary["IoU_mean"].map(lambda x: f"{x:.4f}"),
                    ci_label: [
                        f"[{lo:.4f}, {hi:.4f}]"
                        for lo, hi in zip(summary["IoU_ci_lo"], summary["IoU_ci_hi"])
                    ],
                    "IoU std": summary["IoU_std"].map(lambda x: f"{x:.4f}"),
                    "Dice": summary["Dice_mean"].map(lambda x: f"{x:.4f}"),
                    "Pixel Acc": summary["Pixel Acc_mean"].map(lambda x: f"{x:.4f}"),
                    "Precision": summary["Precision_mean"].map(lambda x: f"{x:.4f}"),
                    "Recall": summary["Recall_mean"].map(lambda x: f"{x:.4f}"),
                    "F1": summary["F1_mean"].map(lambda x: f"{x:.4f}"),
                    "# Images": summary["IoU_count"].astype(int),
                })
                st.dataframe(display_df, width="stretch", hide_index=True)

                # Statistical significance: pairwise Welch's t-tests between top methods
                if _HAS_SCIPY:
                    st.divider()
                    st.markdown("#### Statistical Significance (pairwise t-tests)")
                    st.caption("p-values for IoU difference between methods (Welch's t-test, two-sided)")

                    method_order = summary["Method"].values
                    top_methods = method_order[:min(8, len(method_order))]
                    sig_rows = []
                    for i in range(len(top_methods)):
                        for j in range(i + 1, len(top_methods)):
                            a = df_all[df_all["Method"] == top_methods[i]]["IoU"]
                            b = df_all[df_all["Method"] == top_methods[j]]["IoU"]
                            t_stat, p_val = scipy_stats.ttest_ind(a, b, equal_var=False)
                            diff = a.mean() - b.mean()
                            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                            sig_rows.append({
                                "Method A": top_methods[i],
                                "Method B": top_methods[j],
                                "IoU diff": f"{diff:+.4f}",
                                "p-value": f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}",
                                "Sig.": sig,
                            })
                    if sig_rows:
                        st.dataframe(pd.DataFrame(sig_rows), width="stretch", hide_index=True)
                        st.caption("\\* p<0.05 &nbsp; \\*\\* p<0.01 &nbsp; \\*\\*\\* p<0.001 &nbsp; ns = not significant")

                # Per-camera breakdown with CI
                st.divider()
                st.markdown("#### Per-Camera IoU (mean ± 95% CI)")

                def _fmt_ci(group):
                    m = group.mean()
                    s = group.std()
                    nn = len(group)
                    if nn < 2:
                        return f"{m:.3f}"
                    t_c = _t_crit_95(nn - 1)
                    half = t_c * s / np.sqrt(nn)
                    return f"{m:.3f}±{half:.3f}"

                pivot_ci = df_all.groupby(["Method", "Camera"])["IoU"].apply(_fmt_ci).unstack("Camera")
                pivot_ci = pivot_ci.reindex(summary["Method"].values)
                pivot_ci = pivot_ci.fillna("—")
                st.dataframe(pivot_ci, width="stretch")
            else:
                st.warning("No method outputs overlap with annotated images.")


# ── Tab: Snow Teachers ──────────────────────────────────────────────────────

with tab_snow_compare:
    snow_teachers = discover_snow_teachers()
    if not snow_teachers:
        st.info("No snow teacher outputs found.")
    elif original is None:
        st.warning("Original image not found")
    else:
        st.subheader(f"❄️ Snow Teacher Comparison — {selected_image}")

        # Load all teacher masks for current image
        snow_teacher_masks = {}
        for name, d in snow_teachers.items():
            mask_path = d / camera / f"{stem}_snow.png"
            snow_teacher_masks[name] = load_mask(mask_path)

        active = {k: v for k, v in snow_teacher_masks.items() if v is not None}

        if not active:
            st.info(f"No snow teacher predictions for {selected_image}")
        else:
            # Show original
            st.image(original, caption="Original", width=400)
            st.divider()

            # Teacher overlays - use blue color for snow
            st.markdown("#### Teacher Predictions")
            n = len(active)
            cols = st.columns(min(n, 4))
            for i, (name, mask) in enumerate(active.items()):
                with cols[i % min(n, 4)]:
                    overlay = make_overlay(original, mask, color=(0, 120, 255), alpha=0.45)
                    pct = ground_pct(mask)  # reusing ground_pct since it just counts True pixels
                    st.image(overlay, caption=f"{name}\n({pct:.1f}% snow)", width="stretch")

            # Second row if >4 teachers
            if n > 4:
                cols2 = st.columns(min(n - 4, 4))
                for i, (name, mask) in enumerate(list(active.items())[4:]):
                    with cols2[i]:
                        overlay = make_overlay(original, mask, color=(0, 120, 255), alpha=0.45)
                        pct = ground_pct(mask)
                        st.image(overlay, caption=f"{name}\n({pct:.1f}% snow)", width="stretch")


# ── Tab: Snow Ensemble ──────────────────────────────────────────────────────

with tab_snow_ensemble:
    snow_ensemble_variants = discover_snow_ensemble_variants()
    snow_student_variants = discover_student_snow_variants()

    if not snow_ensemble_variants and not snow_student_variants:
        st.info("No snow ensemble or student results yet.")
    elif original is None:
        st.warning("Original image not found")
    else:
        st.subheader(f"❄️ Snow Ensemble & Student — {selected_image}")

        # Ensemble variants
        if snow_ensemble_variants:
            st.markdown("#### Ensemble Voting Variants")
            snow_ens_data = {}
            for variant_name, variant_dir in snow_ensemble_variants.items():
                mask_path = variant_dir / camera / f"{stem}_snow.png"
                conf_path = variant_dir / camera / f"{stem}_confidence.png"
                snow_ens_data[variant_name] = {
                    "mask": load_mask(mask_path),
                    "confidence": load_image(conf_path),
                }

            active_ens = {k: v for k, v in snow_ens_data.items() if v["mask"] is not None}
            if active_ens:
                ens_cols = st.columns(len(active_ens))
                for i, (variant, data) in enumerate(active_ens.items()):
                    with ens_cols[i]:
                        overlay = make_overlay(original, data["mask"], color=(0, 120, 255), alpha=0.45)
                        pct = ground_pct(data["mask"])
                        st.image(overlay, caption=f"{variant}\n({pct:.1f}% snow)", width="stretch")

                # Confidence maps
                st.divider()
                st.markdown("#### Confidence Maps")
                conf_cols = st.columns(len(active_ens))
                for i, (variant, data) in enumerate(active_ens.items()):
                    with conf_cols[i]:
                        if data["confidence"] is not None:
                            st.image(data["confidence"], caption=f"{variant}", width="stretch")
                        else:
                            st.caption("N/A")
            else:
                st.info(f"No snow ensemble predictions for {selected_image}")

        # Student snow variants
        if snow_student_variants:
            st.divider()
            st.markdown("#### Student Snow Predictions")
            snow_student_masks = {}
            for variant_name, variant_dir in snow_student_variants.items():
                mask_path = variant_dir / camera / f"{stem}_snow.png"
                prob_path = variant_dir / camera / f"{stem}_probability.png"
                snow_student_masks[variant_name] = {
                    "mask": load_mask(mask_path),
                    "probability": load_image(prob_path),
                }

            active_stud = {k: v for k, v in snow_student_masks.items() if v["mask"] is not None}
            if active_stud:
                s_cols = st.columns(len(active_stud))
                for i, (variant, sdata) in enumerate(active_stud.items()):
                    with s_cols[i]:
                        overlay = make_overlay(original, sdata["mask"], color=(0, 120, 255), alpha=0.45)
                        pct = ground_pct(sdata["mask"])
                        st.image(overlay, caption=f"Student {variant}\n({pct:.1f}% snow)", width="stretch")


# ── Tab: Snow Evaluation ────────────────────────────────────────────────────

with tab_snow_eval:
    snow_gt = load_snow_gt_annotations()

    if not snow_gt:
        st.warning("No snow GT annotations found.")
    else:
        n_cams = len(snow_gt)
        n_imgs = sum(len(v) for v in snow_gt.values())
        st.subheader(f"❄️ Snow Evaluation — {n_imgs} annotated images across {n_cams} cameras")

        # Build snow methods registry
        snow_methods_reg = {}
        # Teachers
        for name, d in discover_snow_teachers().items():
            snow_methods_reg[f"Teacher: {name}"] = d
        # Ensemble variants
        for name, d in discover_snow_ensemble_variants().items():
            snow_methods_reg[f"Ensemble: {name}"] = d
        # Student variants
        for name, d in discover_student_snow_variants().items():
            snow_methods_reg[f"Student: {name}"] = d
        # Anomaly detection
        if SUBSPACE_AD_OUTPUT.exists():
            snow_methods_reg["SubspaceAD"] = SUBSPACE_AD_OUTPUT
        for ad_name in ["patchcore", "efficientad"]:
            ad_dir = ANOMALY_RESULTS / ad_name
            if ad_dir.exists() and any(ad_dir.iterdir()):
                snow_methods_reg[f"Anomaly: {ad_name}"] = ad_dir

        if not snow_methods_reg:
            st.error("No snow method outputs found.")
        else:
            method_names = list(snow_methods_reg.keys())
            selected_snow_methods = st.multiselect(
                "Methods to evaluate",
                method_names,
                default=method_names[:6],
                key="snow_eval_methods",
            )
            selected_reg = {k: v for k, v in snow_methods_reg.items() if k in selected_snow_methods}

            # Current image view
            cam_snow_gt = snow_gt.get(camera, {})
            if cam_snow_gt:
                annotated_stems = sorted(cam_snow_gt.keys())
                snow_eval_sel = st.selectbox(
                    f"Annotated image ({len(annotated_stems)} for {camera})",
                    annotated_stems,
                    format_func=lambda s: f"{s}.jpg",
                    key="snow_eval_image",
                )
                snow_gt_mask = cam_snow_gt[snow_eval_sel]
                snow_eval_img = load_image(DATA_DIR / camera / f"{snow_eval_sel}.jpg")

                if snow_gt_mask is not None and snow_eval_img is not None:
                    st.markdown("#### Current Image vs Snow GT")
                    gt_overlay = make_overlay(snow_eval_img, snow_gt_mask, color=(0, 120, 255), alpha=0.5)

                    top_left, top_right = st.columns([1, 1])
                    with top_left:
                        st.image(gt_overlay, caption=f"Snow GT ({ground_pct(snow_gt_mask):.1f}% snow)", width="stretch")
                    with top_right:
                        st.image(snow_eval_img, caption="Original", width="stretch")

                    def _load_snow_pred(m_name, cam, s):
                        """Load snow prediction mask, handling anomalib composites."""
                        if m_name.startswith("Anomaly: "):
                            ad_method = m_name.replace("Anomaly: ", "")
                            return load_anomalib_mask(ad_method, cam, s)
                        return load_mask(selected_reg[m_name] / cam / f"{s}_snow.png")

                    # Method predictions
                    current_metrics = []
                    for m_name, m_dir in selected_reg.items():
                        pred = _load_snow_pred(m_name, camera, snow_eval_sel)
                        m = compute_seg_metrics(pred, snow_gt_mask)
                        if m is not None:
                            current_metrics.append({"Method": m_name, **m})

                    if current_metrics:
                        sorted_metrics = sorted(current_metrics, key=lambda x: x["IoU"], reverse=True)

                        st.markdown("##### Method Predictions (top methods)")
                        show_top = sorted_metrics[:6]
                        n_cols = 3
                        for row_start in range(0, len(show_top), n_cols):
                            chunk = show_top[row_start:row_start + n_cols]
                            cols = st.columns(n_cols)
                            for ci, entry in enumerate(chunk):
                                m_name = entry["Method"]
                                pred = _load_snow_pred(m_name, camera, snow_eval_sel)
                                if pred is not None:
                                    ov = make_overlay(snow_eval_img, pred, color=(0, 120, 255), alpha=0.45)
                                    cols[ci].image(ov, caption=f"{m_name}\nIoU={entry['IoU']:.3f}", width="stretch")

                        try:
                            import pandas as pd
                            st.markdown("##### Per-Method Metrics (this image)")
                            df_img = pd.DataFrame(sorted_metrics)
                            for col in ["IoU", "Dice", "Pixel Acc", "Precision", "Recall", "F1"]:
                                df_img[col] = df_img[col].map(lambda x: f"{x:.4f}")
                            st.dataframe(df_img, width="stretch", hide_index=True)
                        except ImportError:
                            pass
            else:
                st.info(f"No snow GT annotations for **{camera}**.")

            # Full ranking
            st.divider()
            st.markdown("### Snow Method Ranking (all annotated images)")

            if "snow_eval_rows" not in st.session_state:
                st.session_state["snow_eval_rows"] = None

            if st.button("Compute full snow ranking", type="primary", key="snow_rank_btn"):
                with st.spinner("Computing snow metrics across all annotated images..."):
                    snow_rows = []
                    for method_name, method_dir in selected_reg.items():
                        is_anomaly = method_name.startswith("Anomaly: ")
                        ad_method = method_name.replace("Anomaly: ", "") if is_anomaly else None
                        for cam, stems in snow_gt.items():
                            if not is_anomaly:
                                cam_dir = method_dir / cam
                                if not cam_dir.exists():
                                    continue
                            for s, gt_mask in stems.items():
                                if is_anomaly:
                                    pred = load_anomalib_mask(ad_method, cam, s)
                                else:
                                    pred = load_mask(method_dir / cam / f"{s}_snow.png")
                                metrics = compute_seg_metrics(pred, gt_mask)
                                if metrics is not None:
                                    snow_rows.append({"Method": method_name, "Camera": cam, "Stem": s, **metrics})
                    st.session_state["snow_eval_rows"] = snow_rows

            snow_eval_rows = st.session_state["snow_eval_rows"]
            if snow_eval_rows:
                try:
                    import pandas as pd
                    df_snow = pd.DataFrame(snow_eval_rows)
                    summary = (
                        df_snow
                        .groupby("Method")[["IoU", "Dice", "Precision", "Recall", "F1"]]
                        .agg(["mean", "std", "count"])
                    )
                    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
                    summary = summary.reset_index().sort_values("IoU_mean", ascending=False)

                    st.markdown("#### Ranking Table")
                    display_df = pd.DataFrame({
                        "Rank": range(1, len(summary) + 1),
                        "Method": summary["Method"].values,
                        "IoU": summary["IoU_mean"].map(lambda x: f"{x:.4f}"),
                        "IoU std": summary["IoU_std"].map(lambda x: f"{x:.4f}"),
                        "Dice": summary["Dice_mean"].map(lambda x: f"{x:.4f}"),
                        "Precision": summary["Precision_mean"].map(lambda x: f"{x:.4f}"),
                        "Recall": summary["Recall_mean"].map(lambda x: f"{x:.4f}"),
                        "F1": summary["F1_mean"].map(lambda x: f"{x:.4f}"),
                        "# Images": summary["IoU_count"].astype(int),
                    })
                    st.dataframe(display_df, width="stretch", hide_index=True)

                    # Per-camera breakdown
                    st.divider()
                    st.markdown("#### Per-Camera IoU")
                    pivot = df_snow.groupby(["Method", "Camera"])["IoU"].mean().unstack("Camera")
                    pivot = pivot.reindex(summary["Method"].values)
                    pivot = pivot.round(3).fillna("—")
                    st.dataframe(pivot, width="stretch")
                except ImportError:
                    st.warning("Install pandas for ranking tables.")
