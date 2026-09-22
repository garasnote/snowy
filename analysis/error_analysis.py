#!/usr/bin/env python3
"""
Snow segmentation error analysis.

Computes per-image snow segmentation metrics, joins with meteorological data,
and categorizes failures by time-of-day, weather, season, and snow coverage.

Produces:
  - Per-image metrics CSV (analysis/snow_per_image_metrics.csv)
  - Breakdown tables printed to stdout
  - Matplotlib figures saved to analysis/figures/

Usage:
    python analysis/error_analysis.py
    python analysis/error_analysis.py --top-n 20 --save-figures
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ── Paths ───────────────────────────────────────────────────────────────────

MASTER = Path(__file__).resolve().parent.parent
OUTPUT_BASE = MASTER / "preprocessing" / "segmentation"
ANNOTATIONS_DIR = MASTER / "data" / "annotations"
METEO_HOURLY = MASTER / "data" / "meteo_hourly.csv"
FIGURES_DIR = MASTER / "analysis" / "figures"

EXCLUDED_CAMERAS = {"CM00024"}

# Snow method directories (same as Streamlit registry)
SNOW_METHODS = {}


def _add(label, path):
    if path.exists():
        SNOW_METHODS[label] = path
        clipped = path.parent / f"{path.name}_ground_clipped"
        if clipped.exists():
            SNOW_METHODS[f"{label} (ground)"] = clipped


def discover_methods():
    _add("HSV+MobileNet-FT", OUTPUT_BASE / "hsv_snow_output_mobilenet_ft")
    _add("SAM3", OUTPUT_BASE / "sam3_snow_output")
    _add("YOLOe-26", OUTPUT_BASE / "yoloe_snow_output")
    _add("Florence-2", OUTPUT_BASE / "florence2_snow_output")

    ensemble_base = OUTPUT_BASE / "ensemble_snow_output"
    if ensemble_base.exists():
        for sub in sorted(ensemble_base.iterdir()):
            if sub.is_dir():
                SNOW_METHODS[f"Ensemble ({sub.name})"] = sub


# ── Label Studio RLE decoding (copied from app) ────────────────────────────

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


# ── Annotation loading ─────────────────────────────────────────────────────

def load_snow_annotations():
    """Load snow annotations. Returns (gt, sources) dicts."""
    snow_files = [
        "san.json", "heroku.json",
        "01.json", "06.json", "09.json", "70.json", "75.json",
        "108.json", "112.json",
    ]
    NUMBERED = {"01.json", "06.json", "09.json", "70.json", "75.json",
                "108.json", "112.json"}
    gt = {}
    sources = {}
    for fname in snow_files:
        ann_path = ANNOTATIONS_DIR / fname
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

            # Check taxonomy (numbered only)
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
                        break
                if is_non_usable:
                    break
            if is_non_usable:
                if camera in gt and stem in gt[camera]:
                    del gt[camera][stem]
                    del sources[camera][stem]
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

            src = "numbered" if fname in NUMBERED else "san_heroku"
            gt.setdefault(camera, {})[stem] = merged_mask
            sources.setdefault(camera, {})[stem] = src

    return gt, sources


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_seg_metrics(pred, gt):
    if pred is None or gt is None:
        return None
    if pred.shape != gt.shape:
        pred = np.array(
            Image.fromarray(pred.astype(np.uint8) * 255).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST
            )
        ) > 127
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    pixel_acc = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"IoU": iou, "Dice": dice, "Pixel Acc": pixel_acc,
            "Precision": precision, "Recall": recall, "F1": f1}


def load_mask(path):
    if not path.exists():
        return None
    m = np.array(Image.open(path).convert("L"))
    return m > 127


# ── Timestamp parsing ───────────────────────────────────────────────────────

_TS_RE = re.compile(r"_(\d{12})_")


def parse_timestamp(stem):
    """Extract datetime from image stem like CF00001_202210071530_00."""
    m = _TS_RE.search(stem)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M")


def get_season(dt):
    m = dt.month
    if m in (12, 1, 2):
        return "winter"
    elif m in (3, 4, 5):
        return "spring"
    elif m in (6, 7, 8):
        return "summer"
    else:
        return "autumn"


def get_hour_bucket(dt):
    h = dt.hour
    if h < 6:
        return "night (0-6)"
    elif h < 10:
        return "morning (6-10)"
    elif h < 14:
        return "midday (10-14)"
    elif h < 18:
        return "afternoon (14-18)"
    else:
        return "evening (18-24)"


def get_gt_snow_bucket(pct):
    if pct < 0.5:
        return "no snow (<0.5%)"
    elif pct < 5:
        return "trace (0.5-5%)"
    elif pct < 20:
        return "light (5-20%)"
    elif pct < 50:
        return "moderate (20-50%)"
    else:
        return "heavy (50%+)"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Snow segmentation error analysis")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of worst images to report per method")
    parser.add_argument("--save-figures", action="store_true",
                        help="Save matplotlib figures to analysis/figures/")
    parser.add_argument("--min-gt-pct", type=float, default=0.5,
                        help="Min GT snow %% to count as snowy image")
    args = parser.parse_args()

    discover_methods()
    print(f"Found {len(SNOW_METHODS)} snow methods:")
    for name, path in SNOW_METHODS.items():
        print(f"  {name}: {path}")
    print()

    # Load annotations
    print("Loading snow annotations...")
    gt, sources = load_snow_annotations()
    n_imgs = sum(len(v) for v in gt.values())
    print(f"  {n_imgs} annotated images across {len(gt)} cameras\n")

    # Load meteo
    meteo = None
    if METEO_HOURLY.exists():
        print(f"Loading meteo from {METEO_HOURLY}...")
        meteo = pd.read_csv(METEO_HOURLY, parse_dates=["datetime"])
        meteo["hour_key"] = meteo["datetime"].dt.strftime("%Y%m%d%H")
        meteo = meteo.set_index(["camera", "hour_key"])
        print(f"  {len(meteo)} meteo records\n")
    else:
        print("WARNING: meteo_hourly.csv not found, skipping weather analysis\n")

    # Compute per-image metrics
    print("Computing per-image metrics...")
    rows = []
    for method_name, method_dir in SNOW_METHODS.items():
        for cam, stems in gt.items():
            cam_dir = method_dir / cam
            if not cam_dir.exists():
                continue
            for s, gt_mask in stems.items():
                mask_path = cam_dir / f"{s}_snow.png"
                pred = load_mask(mask_path)
                metrics = compute_seg_metrics(pred, gt_mask)
                if metrics is None:
                    continue

                gt_pct = float(gt_mask.sum()) / gt_mask.size * 100
                if pred is not None and pred.shape != gt_mask.shape:
                    pred_r = np.array(
                        Image.fromarray(pred.astype(np.uint8) * 255).resize(
                            (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                    ) > 127
                else:
                    pred_r = pred
                pred_pct = float(pred_r.sum()) / pred_r.size * 100 if pred_r is not None else 0.0

                dt = parse_timestamp(s)
                row = {
                    "Method": method_name,
                    "Camera": cam,
                    "Stem": s,
                    "Source": sources.get(cam, {}).get(s, "unknown"),
                    "GT_snow_pct": round(gt_pct, 2),
                    "Pred_snow_pct": round(pred_pct, 2),
                    **metrics,
                }
                if dt:
                    row["datetime"] = dt.isoformat()
                    row["hour"] = dt.hour
                    row["month"] = dt.month
                    row["season"] = get_season(dt)
                    row["hour_bucket"] = get_hour_bucket(dt)
                    row["gt_snow_bucket"] = get_gt_snow_bucket(gt_pct)

                    # Join meteo
                    if meteo is not None:
                        hour_key = dt.strftime("%Y%m%d%H")
                        try:
                            met = meteo.loc[(cam, hour_key)]
                            if isinstance(met, pd.DataFrame):
                                met = met.iloc[0]
                            row["t2m"] = met.get("t2m")
                            row["weather_code"] = met.get("weather_code")
                            row["visibility"] = met.get("visibility")
                            row["meteo_snow_present"] = met.get("snow_present")
                        except KeyError:
                            pass

                rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {len(df)} rows ({df['Method'].nunique()} methods x images)\n")

    # Save CSV
    out_csv = MASTER / "analysis" / "snow_per_image_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}\n")

    # ── Breakdown tables ────────────────────────────────────────────────────

    # Only snowy images for segmentation analysis
    df_snow = df[df["GT_snow_pct"] >= args.min_gt_pct].copy()
    print(f"=== SEGMENTATION ANALYSIS (GT snow >= {args.min_gt_pct}%): "
          f"{len(df_snow)} rows ===\n")

    # 1. Mean IoU by method
    print("--- Mean IoU by Method ---")
    tbl = (df_snow.groupby("Method")["IoU"]
           .agg(["mean", "std", "count"])
           .sort_values("mean", ascending=False))
    print(tbl.to_string())
    print()

    # 2. Mean IoU by hour bucket
    if "hour_bucket" in df_snow.columns:
        print("--- Mean IoU by Hour Bucket (all methods) ---")
        tbl = (df_snow.groupby("hour_bucket")["IoU"]
               .agg(["mean", "count"])
               .sort_values("mean"))
        print(tbl.to_string())
        print()

    # 3. Mean IoU by season
    if "season" in df_snow.columns:
        print("--- Mean IoU by Season ---")
        tbl = (df_snow.groupby("season")["IoU"]
               .agg(["mean", "count"])
               .sort_values("mean"))
        print(tbl.to_string())
        print()

    # 4. Mean IoU by GT snow coverage bucket
    if "gt_snow_bucket" in df_snow.columns:
        print("--- Mean IoU by GT Snow Coverage ---")
        tbl = (df_snow.groupby("gt_snow_bucket")["IoU"]
               .agg(["mean", "count"])
               .sort_values("mean"))
        print(tbl.to_string())
        print()

    # 5. Mean IoU by method x season
    if "season" in df_snow.columns:
        print("--- Mean IoU: Method x Season ---")
        pivot = df_snow.pivot_table(values="IoU", index="Method",
                                    columns="season", aggfunc="mean")
        print(pivot.round(4).to_string())
        print()

    # 6. Mean IoU by method x hour bucket
    if "hour_bucket" in df_snow.columns:
        print("--- Mean IoU: Method x Hour Bucket ---")
        pivot = df_snow.pivot_table(values="IoU", index="Method",
                                    columns="hour_bucket", aggfunc="mean")
        print(pivot.round(4).to_string())
        print()

    # 7. Temperature correlation
    if "t2m" in df_snow.columns:
        valid = df_snow.dropna(subset=["t2m"])
        if len(valid) > 10:
            print("--- IoU vs Temperature (per method) ---")
            for method, grp in valid.groupby("Method"):
                corr = grp[["IoU", "t2m"]].corr().iloc[0, 1]
                print(f"  {method}: r={corr:.3f} (n={len(grp)})")
            print()

    # 8. Worst images per method
    print(f"--- Top {args.top_n} Worst Images per Method ---")
    for method, grp in df_snow.groupby("Method"):
        worst = grp.nsmallest(args.top_n, "IoU")
        print(f"\n  {method}:")
        for _, r in worst.iterrows():
            extra = ""
            if "season" in r and pd.notna(r.get("season")):
                extra += f" [{r['season']}]"
            if "t2m" in r and pd.notna(r.get("t2m")):
                extra += f" [t={r['t2m']:.1f}C]"
            print(f"    IoU={r['IoU']:.4f}  GT={r['GT_snow_pct']:.1f}%  "
                  f"Pred={r['Pred_snow_pct']:.1f}%  {r['Camera']}/{r['Stem']}{extra}")
    print()

    # 9. Detection analysis: false positives in summer / no-snow
    if "meteo_snow_present" in df.columns:
        no_snow_meteo = df[df["meteo_snow_present"] == False]
        if len(no_snow_meteo) > 0:
            print("--- False Positive Analysis (meteo says no snow) ---")
            fp_rates = (no_snow_meteo.groupby("Method")
                        .apply(lambda g: pd.Series({
                            "n_images": len(g),
                            "mean_pred_pct": g["Pred_snow_pct"].mean(),
                            "pred_>1%": (g["Pred_snow_pct"] > 1).sum(),
                            "pred_>5%": (g["Pred_snow_pct"] > 5).sum(),
                        })))
            print(fp_rates.to_string())
            print()

    # ── Figures ─────────────────────────────────────────────────────────────

    if args.save_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving figures to {FIGURES_DIR}/...")

        # Fig 1: IoU distribution per method (box plot)
        fig, ax = plt.subplots(figsize=(10, 5))
        methods = df_snow["Method"].unique()
        data = [df_snow[df_snow["Method"] == m]["IoU"].values for m in methods]
        ax.boxplot(data, labels=methods, vert=True)
        ax.set_ylabel("IoU")
        ax.set_title(f"Snow IoU Distribution (GT >= {args.min_gt_pct}%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "iou_boxplot_by_method.png", dpi=150)
        plt.close(fig)

        # Fig 2: IoU vs GT snow % scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        for method in methods:
            grp = df_snow[df_snow["Method"] == method]
            ax.scatter(grp["GT_snow_pct"], grp["IoU"], alpha=0.4, s=10, label=method)
        ax.set_xlabel("GT Snow %")
        ax.set_ylabel("IoU")
        ax.set_title("IoU vs Ground-Truth Snow Coverage")
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "iou_vs_gt_snow_pct.png", dpi=150)
        plt.close(fig)

        # Fig 3: Mean IoU by hour (line per method)
        if "hour" in df_snow.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            for method in methods:
                grp = df_snow[df_snow["Method"] == method]
                hourly = grp.groupby("hour")["IoU"].mean()
                ax.plot(hourly.index, hourly.values, marker="o", markersize=3, label=method)
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Mean IoU")
            ax.set_title("Mean IoU by Hour of Day")
            ax.legend(fontsize=7)
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / "iou_by_hour.png", dpi=150)
            plt.close(fig)

        # Fig 4: IoU vs temperature scatter
        if "t2m" in df_snow.columns:
            valid = df_snow.dropna(subset=["t2m"])
            if len(valid) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                for method in methods:
                    grp = valid[valid["Method"] == method]
                    if len(grp) > 0:
                        ax.scatter(grp["t2m"], grp["IoU"], alpha=0.4, s=10, label=method)
                ax.set_xlabel("Temperature (C)")
                ax.set_ylabel("IoU")
                ax.set_title("IoU vs Temperature")
                ax.legend(fontsize=7, loc="lower right")
                plt.tight_layout()
                fig.savefig(FIGURES_DIR / "iou_vs_temperature.png", dpi=150)
                plt.close(fig)

        print("Done saving figures.\n")

    print("Error analysis complete!")


if __name__ == "__main__":
    main()
