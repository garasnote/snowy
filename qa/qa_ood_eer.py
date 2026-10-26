#!/usr/bin/env python3
"""Operating points of the QA gate on the held-out cameras.

Loads the same TorchScript TinyQACNN that eval_all.py --task qa uses and the same 558 OOD QA
frames (annotation exports 54/67/71/115), then reports:

  EER        the cut where rejected usable and admitted non-usable frames are equally frequent.
  TPR@FPR    class-conditional and therefore prevalence-independent: fix the fraction of
             non-usable frames admitted (5/10/20%), take the lowest cut that meets it, and read
             off how much snow survives. pAUC narrows AUROC to the same region.
  Cllr       cut-free measure of whether the score is calibrated at this site.
  recal      a one-parameter monotone remap of the score, fitted on a random half of each
             camera's frames and measured on the other half, to test whether a single shared
             cut would then work at every site.

    python qa/qa_ood_eer.py --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T_qa
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

QA_DIR = Path(__file__).resolve().parent
ROOT = QA_DIR.parent
ANN_DIR = ROOT / "data" / "annotations"
DATA_DIR = ROOT / "data"
CLASS_NAMES = ["clean_no_snow", "snow", "non_usable"]
OOD_CAMS = {"CM00054": "54.json", "CM00067": "67.json", "CM00071": "71.json", "CM00115": "115.json"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ALPHAS = (0.05, 0.10, 0.20)
MAIN_ALPHA = 0.10


def parse_label3(task):
    """Label Studio task -> 3-class label (0=clean_no_snow, 1=snow, 2=non_usable)."""
    anns = task.get("annotations", [])
    if not anns or not anns[0].get("result"):
        return None
    flat = set()
    for r in anns[0]["result"]:
        v = r.get("value", {})
        for key in ("taxonomy", "choices"):
            tags = v.get(key)
            if not tags:
                continue
            for entry in tags:
                if isinstance(entry, list):
                    flat.update(entry)
                else:
                    flat.add(entry)
    if "Usable" in flat:
        return 1
    if "Non-Usable" in flat:
        return 0 if (flat - {"Non-Usable"}) <= {"No Snow"} else 2
    return None


def collect(cam, fname):
    ann_path = ANN_DIR / fname
    if not ann_path.exists():
        print(f"  {cam}: no annotation export at {ann_path}")
        return [], []
    with open(ann_path) as f:
        tasks = json.load(f)
    paths, labels = [], []
    for task in tasks:
        label = parse_label3(task)
        if label is None:
            continue
        stem = Path(task["data"]["image"].rstrip("/").split("/")[-1]).stem
        for base in (DATA_DIR / cam, DATA_DIR / f"_EXCLUDED_{cam}"):
            hit = next((base / f"{stem}.{e}" for e in ("jpg", "png") if (base / f"{stem}.{e}").exists()), None)
            if hit is not None:
                paths.append(hit)
                labels.append(label)
                break
    return paths, labels


def logit(s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def cut_at_fpr(scores, labels, alpha):
    """Highest cut on the non-usable score that still keeps at least (1 - alpha) of the
    non-usable frames above it, i.e. admits at most `alpha` of them. The score is
    P(non-usable), so a frame is admitted when its score is below the cut."""
    nu = labels == 2
    if nu.sum() == 0:
        return None
    k = max(int(np.floor(alpha * nu.sum())), 1)      # count of non-usable frames allowed through
    return float(np.sort(scores[nu])[k - 1])         # cut sits just above that many of them


def snow_kept(scores, labels, cut):
    snow, nu = labels == 1, labels == 2
    kept = float((scores[snow] < cut).mean()) if snow.sum() else float("nan")
    adm = float((scores[nu] < cut).mean()) if nu.sum() else float("nan")
    return kept, adm


def eer(scores, labels):
    u = labels < 2
    if u.all() or not u.any():
        return None, None
    best, cut = 9.0, 0.5
    for c in np.unique(np.round(scores, 4)):
        d = abs((scores[u] >= c).mean() - (scores[~u] < c).mean())
        if d < best:
            best, cut = d, float(c)
    return ((scores[u] >= cut).mean() + (scores[~u] < cut).mean()) / 2, cut


def cllr(scores, labels):
    """Cllr of the usable/non-usable decision: no cut, no prior. Lower is better calibrated."""
    u, nu = labels < 2, labels == 2
    if u.sum() == 0 or nu.sum() == 0:
        return float("nan")
    lr = np.clip(scores, 1e-9, 1 - 1e-9) / (1 - np.clip(scores, 1e-9, 1 - 1e-9))
    return float((np.log2(1 + lr[u]).sum() + np.log2(1 + 1 / lr[nu]).sum()) / (2 * len(labels)))


def pauc(scores, labels, max_fpr=0.10):
    try:
        return float(roc_auc_score((labels == 2).astype(int), scores, max_fpr=max_fpr))
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=str(QA_DIR / "checkpoints" / "qa_cnn.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-alpha", type=float, default=0.10,
                    help="admission budget both arms are held to in the held-out check")
    ap.add_argument("--repeats", type=int, default=5,
                    help="random half-splits used for the held-out calibration check")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = torch.jit.load(args.ckpt, map_location=device).to(device).eval()
    transform = T_qa.Compose([
        T_qa.Resize((128, 128)), T_qa.ToImage(), T_qa.ToDtype(torch.float32, scale=True),
        T_qa.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    probs, labels, cams = [], [], []
    print("== held-out cameras ==")
    for cam, fname in OOD_CAMS.items():
        paths, lab = collect(cam, fname)
        if not paths:
            continue
        p = []
        with torch.no_grad():
            for i in range(0, len(paths), 32):
                batch = torch.stack([transform(Image.open(x).convert("RGB")) for x in paths[i:i + 32]])
                p.append(torch.softmax(model(batch.to(device)), dim=1).cpu().numpy())
        probs.append(np.vstack(p)); labels.extend(lab); cams.extend([cam] * len(lab))
        n3 = np.bincount(np.array(lab), minlength=3)
        print(f"  {cam}: {len(lab):3d} frames  " + "  ".join(f"{CLASS_NAMES[c]}={n3[c]}" for c in range(3)))

    scores = np.vstack(probs)[:, 2]
    labels = np.array(labels); cams = np.array(cams)

    print(f"\n== {len(labels)} OOD frames, pooled ==")
    e, c = eer(scores, labels)
    print(f"AUROC {roc_auc_score((labels == 2).astype(int), scores):.4f}   "
          f"pAUC[0,0.10] {pauc(scores, labels):.4f}   Cllr {cllr(scores, labels):.4f}   "
          f"EER {e * 100:.1f}% at cut {c:.3f}")

    shared_cut = cut_at_fpr(scores, labels, args.calib_alpha)
    print(f"one shared cut at FPR {int(args.calib_alpha * 100)}%: {shared_cut:.3f}")

    print("\n== per camera ==")
    print(f"{'camera':<9} {'EER':>6} {'Cllr':>6} {'pAUC':>6} | "
          + "  ".join(f"snow kept@FPR{int(a * 100):02d}" for a in ALPHAS)
          + f" | shared cut keeps snow")
    for cam in OOD_CAMS:
        m = cams == cam
        if not m.any():
            continue
        s, y = scores[m], labels[m]
        e2, _ = eer(s, y)
        row = f"{cam:<9} {(e2 * 100 if e2 is not None else float('nan')):5.1f}% {cllr(s, y):6.3f} {pauc(s, y):6.3f} | "
        for a in ALPHAS:
            cut = cut_at_fpr(s, y, a)
            kept, _ = snow_kept(s, y, cut) if cut is not None else (float("nan"), float("nan"))
            row += f"   {kept * 100:5.1f}% (cut {cut:.3f})" if cut is not None else "        n/a       "
        kept_shared, adm_shared = snow_kept(s, y, shared_cut)
        row += f" | {kept_shared * 100:5.1f}% (admits {adm_shared * 100:.1f}% of non-usable)"
        print(row)

    print("\n== does a small labelled slice recover the gate? (held-out calibration) ==")
    print(f"Per camera, the cut is fitted on a random half of that camera's frames at an admission")
    print(f"budget of {args.calib_alpha * 100:.0f}% and scored on the other half, averaged over the repeats.")
    print("The shared pooled cut is held to the same budget and scored on the same held-out halves.")
    rng = np.random.default_rng(args.seed)
    agg = {}
    for cam in OOD_CAMS:
        m = cams == cam
        if not m.any():
            continue
        s, y = scores[m], labels[m]
        kept_cal, kept_shared, adm_cal, adm_shared = [], [], [], []
        for _ in range(args.repeats):
            idx = rng.permutation(len(s))
            fit, test = idx[: len(s) // 2], idx[len(s) // 2:]
            if (y[fit] == 2).sum() < 1 or (y[fit] < 2).sum() < 1 or (y[test] == 2).sum() < 1:
                continue
            cut = cut_at_fpr(s[fit], y[fit], args.calib_alpha)
            if cut is None:
                continue
            k, a = snow_kept(s[test], y[test], cut)
            kept_cal.append(k); adm_cal.append(a)
            k2, a2 = snow_kept(s[test], y[test], shared_cut)
            kept_shared.append(k2); adm_shared.append(a2)
        if not kept_cal:
            print(f"  {cam}: half-split leaves a class empty ({int((y == 2).sum())} non-usable frames) — skipped")
            continue
        fmt = lambda v: f"{np.mean(v) * 100:5.1f}% [{np.min(v) * 100:.0f}-{np.max(v) * 100:.0f}]"
        print(f"  {cam}: own slice {fmt(kept_cal)} snow kept (admits {np.mean(adm_cal) * 100:4.1f}%) | "
              f"shared cut {fmt(kept_shared)} snow kept (admits {np.mean(adm_shared) * 100:4.1f}%)")
        agg[cam] = (float(np.mean(kept_cal)), float(np.mean(kept_shared)))
    if agg:
        print(f"  mean over cameras: own slice {np.mean([v[0] for v in agg.values()]) * 100:5.1f}% snow kept"
              f"  vs shared cut {np.mean([v[1] for v in agg.values()]) * 100:5.1f}%")


if __name__ == "__main__":
    main()