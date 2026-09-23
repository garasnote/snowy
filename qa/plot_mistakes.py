#!/usr/bin/env python3
"""Plot misclassified images from QA GradBoost CV predictions.

Saves grid images of:
  - False Positives: Non-Usable predicted as Usable (27)
  - False Negatives: Usable predicted as Non-Usable (22)

Usage:
    python plot_mistakes.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def plot_grid(img_paths, titles, suptitle, save_path, ncols=5):
    """Plot a grid of images."""
    n = len(img_paths)
    if n == 0:
        print(f"  No images for: {suptitle}")
        return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    if nrows == 1:
        axes = [axes] if ncols == 1 else [axes]
    axes_flat = np.array(axes).flatten()

    for i, (img_path, title) in enumerate(zip(img_paths, titles)):
        ax = axes_flat[i]
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "NOT FOUND", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    pred_path = _HERE / "checkpoints" / "qa_cv_predictions.csv"
    img_dir = _ROOT / "data" / "balanced_snow"
    out_dir = _HERE / "checkpoints"

    df = pd.read_csv(pred_path)

    # False Positives: label=0 (Non-Usable) but predicted 1 (Usable)
    fp = df[(df["label"] == 0) & (df["cv_pred"] == 1)].copy()
    fp = fp.sort_values("cv_prob_usable", ascending=False)

    # False Negatives: label=1 (Usable) but predicted 0 (Non-Usable)
    fn = df[(df["label"] == 1) & (df["cv_pred"] == 0)].copy()
    fn = fn.sort_values("cv_prob_usable", ascending=True)

    print(f"False Positives (Non-Usable called Usable): {len(fp)}")
    print(f"False Negatives (Usable called Non-Usable): {len(fn)}")

    # Build image paths and titles
    fp_paths = [img_dir / row["camera"] / f"{row['stem']}.jpg" for _, row in fp.iterrows()]
    fp_titles = []
    for _, row in fp.iterrows():
        reasons = []
        if row.get("no_snow") == 1:
            reasons.append("no_snow")
        if row.get("blurry") == 1:
            reasons.append("blurry")
        if row.get("bad_light") == 1:
            reasons.append("bad_light")
        if row.get("not_in_frame") == 1:
            reasons.append("not_in_frame")
        reason_str = ",".join(reasons) if reasons else "?"
        fp_titles.append(f"{row['camera']}\n{reason_str}\np={row['cv_prob_usable']:.2f}")

    fn_paths = [img_dir / row["camera"] / f"{row['stem']}.jpg" for _, row in fn.iterrows()]
    fn_titles = [f"{row['camera']}\np={row['cv_prob_usable']:.2f}" for _, row in fn.iterrows()]

    plot_grid(
        fp_paths, fp_titles,
        f"FALSE POSITIVES — Non-Usable predicted as Usable ({len(fp)})",
        out_dir / "mistakes_false_positives.png",
    )
    plot_grid(
        fn_paths, fn_titles,
        f"FALSE NEGATIVES — Usable predicted as Non-Usable ({len(fn)})",
        out_dir / "mistakes_false_negatives.png",
    )

    # Also save worst confident mistakes
    print("\nFP (Non-Usable but high prob_usable):")
    for _, row in fp.head(10).iterrows():
        print(f"  {row['camera']}/{row['stem']}.jpg  p={row['cv_prob_usable']:.3f}  "
              f"no_snow={int(row.get('no_snow',0))} blurry={int(row.get('blurry',0))}")

    print("\nFN (Usable but low prob_usable):")
    for _, row in fn.head(10).iterrows():
        print(f"  {row['camera']}/{row['stem']}.jpg  p={row['cv_prob_usable']:.3f}")


if __name__ == "__main__":
    main()
