#!/usr/bin/env python3
"""Acceptance test: run the deployed chain on annotated frames and publish the deltas.

Order of operations (matches the brief):
  a) one camera, a few frames with known ground truth, the full chain -> the dict
  b) compare psg_raw / masks against the published numbers and state the delta
  c) only then build/point at the Gradio demo

    python deploy/acceptance.py --cameras CM00014 CM00075 --frames 12 \
        --weights ~/Documents/Uni/Thesis/models/best --out deploy/acceptance_report.json

What it reports, and what it deliberately does not:
  * It prints the label vocabulary found in the annotation files. GT is read with
    eval_all.load_gt_masks, the repo's own loader, so no second interpretation of
    the annotation format creeps in.
  * IoU/Dice are computed on the annotated frames it samples. The published
    ground/snow IoUs are validation-split numbers, so the delta is approximate by
    construction — the report says so per line instead of implying equality.
  * psg_raw has no published per-frame counterpart: the published aggregates come
    from a per-camera timeseries CSV that is gitignored and absent from a fresh
    clone. The report states the value and says "no published counterpart" rather
    than inventing a comparison.
  * psg_smoothed is only compared when a window exists, and the seq_len used is
    printed. The packaged uni smoother does not record its training seq_len, so a
    smoothed delta is marked PROVISIONAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for p in (str(_REPO), str(_REPO / "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from collections import defaultdict                    # noqa: E402
from deploy.chain import SnowyPipeline                        # noqa: E402
from deploy.contract import CLASS_NAMES                       # noqa: E402


def _stub_cv2_if_absent():
    """eval_all imports cv2 at module level for mask resizing; load_gt_masks does
    not need it. A free-tier env (and this sandbox) has no opencv, so stub the
    module for this import rather than adding a heavy dependency for one function.
    """
    import sys
    try:
        import cv2  # noqa: F401
    except Exception:
        import types
        sys.modules["cv2"] = types.ModuleType("cv2")


_stub_cv2_if_absent()
from eval_all import load_gt_masks                          # noqa: E402  repo's own loader

PUBLISHED = {
    # from the checkpoint filenames of the deploy set (val IoU recorded at train time)
    "ground_student_mobilenet_val_iou": 0.916,
    "snow_student_mobilenet_val_iou": 0.821,
    # from the manuscript's ground table (in-distribution, 5-fold mean)
    "ground_student_mobilenet_paper_iou": 0.904,
    "snow_student_mobilenet_paper_iou": 0.895,
}


def label_vocabulary(ann_dir: Path) -> dict:
    """What brush/choice labels the annotation files actually use (union over all files)."""
    seen = defaultdict(set)
    for jf in sorted(ann_dir.glob("*.json")):
        try:
            tasks = json.loads(jf.read_text())
        except Exception:
            continue
        for t in tasks:
            for a in t.get("annotations", []) or []:
                for r in a.get("result", []) or []:
                    key = f"{r.get('from_name')}/{r.get('type')}"
                    val = r.get("value", {})
                    if r.get("type") == "brushlabels":
                        seen[key].update(str(x) for x in val.get("brushlabels", []))
                    elif r.get("type") == "taxonomy":
                        flat = [x for sub in val.get("taxonomy", [])
                                for x in (sub if isinstance(sub, list) else [sub])]
                        seen[key].update(str(x) for x in flat)
    return {k: sorted(v) for k, v in seen.items()}


def qa_truth_from_task(task) -> int | None:
    """TinyQACNN class index from the Label Studio taxonomy, or None.

    Rationale: qa/train_qa.py:38 `_taxonomy_to_label3` is the canonical rule — imported
    when possible so the acceptance test cannot drift from the training labels.
    ('Usable' -> 1 snow; 'Non-Usable' + no reason or 'No Snow' -> 0 clean_no_snow;
    any other reason -> 2 non_usable.)
    """
    for a in task.get("annotations", []) or []:
        flat = set()
        for r in a.get("result", []) or []:
            v = r.get("value", {})
            for key in ("taxonomy", "choices"):
                for entry in (v.get(key) or []):
                    if isinstance(entry, list):
                        flat.update(entry)
                    else:
                        flat.add(entry)
        if flat:
            return _taxonomy_to_label3(flat)
    return None


def _taxonomy_to_label3(flat: set) -> int | None:
    """Delegates to qa/train_qa.py when importable; identical rule otherwise."""
    try:
        from qa.train_qa import _taxonomy_to_label3 as repo_rule
        return repo_rule(flat)
    except Exception:
        if "Usable" in flat:
            return 1
        if "Non-Usable" in flat:
            reasons = flat - {"Non-Usable"}
            return 0 if reasons in (set(), {"No Snow"}) else 2
        return None


def gt_map(gt: dict, cam: str) -> dict:
    """load_gt_masks keys are (camera, stem) tuples — index one camera by stem."""
    out = {}
    for k, v in (gt or {}).items():
        if isinstance(k, tuple) and len(k) == 2:
            if k[0] == cam:
                out[k[1]] = v
        else:
            out[str(k)] = v
    return out


def iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, bool]:
    """IoU/Dice after nearest-neighbour resizing the prediction to the GT shape.

    Mirrors eval_all.Evaluator, which resizes predictions to the ground-truth mask
    before scoring: the GT masks sit at three native resolutions (520x1023,
    960x1296, 600x800) and torchmetrics' JaccardIndex raises on ragged shapes.
    Returns (iou, dice, resampled).
    """
    resampled = tuple(pred.shape) != tuple(gt.shape)
    if resampled:
        p = np.array(Image.fromarray((pred > 127).astype(np.uint8) * 255).resize(
            (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
    else:
        p = pred > 127
    g = gt > 0
    inter = int((p & g).sum())
    union = int((p | g).sum())
    iou = inter / union if union else float("nan")
    dice = 2 * inter / (int(p.sum()) + int(g.sum())) if (p.sum() + g.sum()) else float("nan")
    return iou, dice, resampled


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", nargs="+", default=["CM00014", "CM00075"])
    ap.add_argument("--frames", type=int, default=8, help="frames per camera")
    ap.add_argument("--qa-only", action="store_true",
                    help="score every annotated frame through the QA gate only (fast) "
                         "and report AUROC/F1 against the published values")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap new frames scored in this run (0 = all); the cache resumes")
    ap.add_argument("--img-root", default=None,
                    help="read jpgs from here instead of <repo>/data (e.g. a local staging "
                         "dir: reading ~1,800 frames off a mounted volume is the bottleneck)")
    ap.add_argument("--cache", default=None,
                    help="per-frame score cache (default: beside --out)")
    ap.add_argument("--qa-ckpt", default=None,
                    help="use this QA checkpoint instead of the one in the weights dir "
                         "(the deploy set ships both qa_cnn.pt and qa_cnn.ckpt)")
    ap.add_argument("--qa-three-class", action="store_true",
                    help="also record all three class probabilities, so the binary AUROC "
                         "and the 3-class macro AUROC / F1 can both be reported")
    ap.add_argument("--weights", default=str(_REPO.parent / "models" / "best"))
    ap.add_argument("--seq-len", type=int, default=72)
    ap.add_argument("--smoother", action="store_true")
    ap.add_argument("--out", default=str(_HERE / "acceptance_report.json"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    data_dir = _REPO / "data"
    ann_dir = data_dir / "annotations"
    print("=== label vocabulary in the annotation files ===")
    for k, v in label_vocabulary(ann_dir).items():
        print(f"  {k}: {v}")

    pipe = SnowyPipeline(args.weights, device=args.device, seq_len=args.seq_len,
                         qa_ckpt=args.qa_ckpt)
    if args.smoother:
        pipe.load_smoother()

    if args.qa_only:
        return _qa_only(pipe, data_dir, args)

    return _chain_test(pipe, data_dir, ann_dir, args)


def _qa_only(pipe, data_dir: Path, args) -> int:
    """Score annotated frames through the QA gate alone, resuming across restarts.

    Per-frame scores are appended to a JSONL cache as they are produced: long runs
    in this environment get killed by container restarts, and re-opening 1,400
    full-resolution jpgs is the expensive part.
    """
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    cache = Path(args.cache) if args.cache \
        else Path(args.out).with_name("qa_scores_cache.jsonl")
    done: dict = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:
                rec = json.loads(line)
                done[rec["stem"]] = rec
            except Exception:
                continue
    candidates = []
    for jf in sorted((data_dir / "annotations").glob("*.json")):
        try:
            tasks = json.loads(jf.read_text())
        except Exception:
            continue
        for task in tasks:
            stem = Path((task.get("data") or {}).get("image", "")).name.replace(".jpg", "")
            cls = qa_truth_from_task(task)
            cam = stem.split("_")[0] if "_" in stem else None
            if not cam or cls is None:
                continue
            jpg = Path(args.img_root) / cam / f"{stem}.jpg" if args.img_root \
                else data_dir / cam / f"{stem}.jpg"
            if jpg.exists():
                candidates.append((stem, cam, cls, jpg))
    todo = [c for c in candidates if c[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"cache: {len(done)} scored, {len(candidates)} candidates, "
          f"{len(todo)} to do this run")
    with cache.open("a") as fh:
        for stem, cam, cls, jpg in todo:
            im = Image.open(jpg).convert("RGB")
            rec = {"camera": cam, "stem": stem, "qa_gt_class": cls}
            if args.qa_three_class:
                rec["probs"] = pipe.qa_probs(im)
                rec["qa_score"] = 1.0 - rec["probs"][2]
            else:
                rec["qa_score"] = pipe.qa_score(im)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()                                   # survive a kill
            done[stem] = rec
    rows = [done[c[0]] for c in candidates if c[0] in done]

    if not rows:
        print("no annotated frame with a local jpg found")
        return 1
    y = [int(r["qa_gt_class"] != 2) for r in rows]        # usable vs not
    s = [r["qa_score"] for r in rows]
    pred = [int(v >= 0.5) for v in s]
    dist = {c: sum(1 for r in rows if r["qa_gt_class"] == c) for c in (0, 1, 2)}
    print(f"\n=== QA gate over {len(rows)} annotated frames with a local jpg ===")
    print(f"  label3 distribution      clean_no_snow={dist[0]} snow={dist[1]} "
          f"non_usable={dist[2]}")
    print(f"  binary usable/not        {sum(y)} usable, {len(y) - sum(y)} not")
    print(f"  AUROC                    {roc_auc_score(y, s):.4f}")
    print(f"  F1 (threshold 0.5)       {f1_score(y, pred, zero_division=0):.4f}")
    print(f"  accuracy                 {accuracy_score(y, pred):.4f}")
    if any("probs" in r for r in rows):
        P = np.array([r["probs"] for r in rows if "probs" in r], dtype=np.float64)
        g3 = np.array([r["qa_gt_class"] for r in rows if "probs" in r])
        print(f"  3-class macro AUROC      "
              f"{roc_auc_score(g3, P, multi_class='ovr', average='macro'):.4f}")
        print(f"  3-class macro F1         "
              f"{f1_score(g3, P.argmax(1), average='macro', zero_division=0):.4f}"
              f"   (paper's TinyQACNN row quotes 0.8975)")
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(g3, P.argmax(1)):
            cm[t, p] += 1
        print(f"  confusion (rows=truth, cols=pred) {list(CLASS_NAMES)}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name:14s} {list(cm[i])}")
    per_cam = {}
    for r in rows:
        per_cam.setdefault(r["camera"], []).append(r)
    print("  per camera (n, usable, AUROC when both classes present):")
    for cam, rs in sorted(per_cam.items()):
        cs = {int(x["qa_gt_class"] != 2) for x in rs}
        auc = (f"{roc_auc_score([int(x['qa_gt_class'] != 2) for x in rs], [x['qa_score'] for x in rs]):.4f}"
               if len(cs) == 2 else "n/a")
        print(f"    {cam}  n={len(rs):4d}  usable={sum(int(x['qa_gt_class'] != 2) for x in rs):4d}  AUROC={auc}")
    print(f"\n  published reference points (paper): TinyQACNN AUROC 0.9746 ID / 0.9514 OOD, "
          f"3-class F1 0.8975 — a binary AUROC here is not the same metric as the "
          f"3-class F1, and these frames are not split into the paper's ID/OOD sets.")
    out = {"n": len(rows), "label3_distribution": dist, "auroc": roc_auc_score(y, s),
           "f1": f1_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred),
           "per_camera": {c: len(v) for c, v in per_cam.items()}, "rows": rows}
    dest = Path(args.out).with_name("qa_only_report.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")
    return 0


def _chain_test(pipe, data_dir: Path, ann_dir: Path, args) -> int:

    report = {"weights": str(args.weights), "seq_len": args.seq_len,
              "load_reports": {k: {kk: (list(vv) if isinstance(vv, (list, tuple)) else vv)
                                   for kk, vv in v.items()}
                               for k, v in pipe.load_reports.items()},
              "published": PUBLISHED, "frames": [], "aggregate": {}}

    per_cam = {}
    for cam in args.cameras:
        jpgs = sorted((data_dir / cam).glob("*.jpg"))[:args.frames]
        if not jpgs:
            print(f"[{cam}] no local jpgs; skipped")
            continue
        gt_snow = {}
        gt_ground = {}
        qa_gt = {}
        jf = ann_dir / f"{cam[-3:].lstrip('0')}.json"
        if jf.exists():
            gt_snow = gt_map(load_gt_masks(str(jf), {cam}, label_class="Snow Mano"), cam)
            gt_ground = gt_map(load_gt_masks(str(jf), {cam}, label_class="Ground Mano"), cam)
            for task in json.loads(jf.read_text()):
                image_url = (task.get("data") or {}).get("image", "")
                stem = Path(image_url).name.replace(".jpg", "")
                cls = qa_truth_from_task(task)
                if stem and cls is not None:
                    qa_gt[stem] = cls
        gaf = ann_dir / "ground_annotations.json"
        if gaf.exists() and not gt_ground:
            gt_ground = gt_map(load_gt_masks(str(gaf), {cam}, label_class="Ground Mano"), cam)

        print(f"\n=== {cam}: {len(jpgs)} frames, {len(gt_snow)} snow GT, "
              f"{len(gt_ground)} ground GT, {len(qa_gt)} QA labels")
        annotated = [p for p in jpgs if p.stem in gt_snow or p.stem in gt_ground
                     or p.stem in qa_gt]
        if annotated:
            local = {p.stem: p for p in jpgs}
            jpgs = [local[p.stem] for p in sorted(annotated)][:args.frames]
            print(f"  sampling {len(jpgs)} frames that carry ground truth")
        else:
            jpgs = jpgs[:args.frames]
            print(f"  WARNING: no annotated frame among local jpgs; sampling {len(jpgs)} "
                  f"without GT (matches will read n/a)")
        rows = []
        for jpg in jpgs:
            resp, raw = pipe.predict_image(jpg, cam)
            row = {"camera": cam, "stem": jpg.stem, "qa_score": resp["qa_score"],
                   "psg_raw": resp["psg_raw"], "psg_smoothed": resp["psg_smoothed"]}
            if jpg.stem in qa_gt:
                row["qa_gt_class"] = qa_gt[jpg.stem]
                row["qa_gt_usable"] = int(qa_gt[jpg.stem] != 2)
                row["qa_pred_usable"] = int(resp["qa_score"] >= 0.5)
            if jpg.stem in gt_snow:
                if int(np.asarray(gt_snow[jpg.stem]).sum()) == 0:
                    row["snow_gt_empty"] = True          # e.g. a "No Snow" frame
                else:
                    row["snow_iou"], row["snow_dice"], row["snow_resampled"] = iou_dice(
                        raw["snow_mask"], gt_snow[jpg.stem])
            if jpg.stem in gt_ground:
                if int(np.asarray(gt_ground[jpg.stem]).sum()) == 0:
                    row["ground_gt_empty"] = True
                else:
                    row["ground_iou"], row["ground_dice"], row["ground_resampled"] = iou_dice(
                        raw["ground_mask"], gt_ground[jpg.stem])
            rows.append(row)
            extra = ""
            if "snow_iou" in row:
                extra += f" snowIoU={row['snow_iou']:.3f}"
            if "ground_iou" in row:
                extra += f" groundIoU={row['ground_iou']:.3f}"
            if "qa_gt_class" in row:
                extra += f" qaGT=class{row['qa_gt_class']}"
            print(f"  {jpg.stem}: qa={row['qa_score']:.3f} psg_raw={row['psg_raw']:.1f}{extra}")
        report["frames"].extend(rows)
        for r in rows:
            if "qa_gt_usable" in r and r["qa_gt_usable"] != r["qa_pred_usable"]:
                print(f"    QA MISMATCH {r['stem']}: gt={'usable' if r['qa_gt_usable'] else 'non-usable'}"
                      f" (class {r['qa_gt_class']}) predicted={r['qa_score']:.3f}")
        per_cam[cam] = rows

    def mean_of(rows, key):
        vals = [r[key] for r in rows
                if key in r and r[key] is not None and isinstance(r[key], (int, float))
                and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else None

    skipped = {
        "snow_gt_empty": sum(1 for r in report["frames"] if r.get("snow_gt_empty")),
        "ground_gt_empty": sum(1 for r in report["frames"] if r.get("ground_gt_empty")),
    }
    report["skipped_empty_gt"] = skipped
    if skipped["ground_gt_empty"]:
        print(f"\n  NOTE {skipped['ground_gt_empty']} sampled frames have an all-zero "
              f"ground mask: the Ground/Ground Mano brushlabels in the per-camera ID "
              f"JSONs decode to zero pixels. Real ground GT lives in "
              f"ground_annotations.json and the OOD files 54/67/71/115 — sample those "
              f"cameras (e.g. --cameras CM00067 CM00054 CM00115 CM00071) to score it.")

    all_rows = report["frames"]
    agg = {}
    for key in ("qa_score", "psg_raw", "psg_smoothed", "snow_iou", "snow_dice",
                "ground_iou", "ground_dice"):
        agg[key] = mean_of(all_rows, key)
    report["aggregate"] = agg

    print("\n=== aggregate over sampled frames ===")
    for k, v in agg.items():
        print(f"  {k:14s} {('n/a' if v is None else round(v, 4))}")

    qa_pairs = [(r["qa_gt_usable"], r["qa_pred_usable"]) for r in all_rows
                if "qa_gt_usable" in r]
    if qa_pairs:
        agree = sum(1 for g, p in qa_pairs if g == p)
        print(f"  QA gate       {agree}/{len(qa_pairs)} frames match the annotation's "
              f"usable/non-usable label")
        report["qa_agreement"] = {"n": len(qa_pairs), "agree": agree}

    print("\n=== delta vs published ===")
    if agg.get("ground_iou") is not None:
        d = agg["ground_iou"] - PUBLISHED["ground_student_mobilenet_val_iou"]
        print(f"  ground IoU {agg['ground_iou']:.4f} vs {PUBLISHED['ground_student_mobilenet_val_iou']}"
              f" (checkpoint's own val IoU) delta {d:+.4f}   [annotated frames, not the val split]")
    if agg.get("snow_iou") is not None:
        d = agg["snow_iou"] - PUBLISHED["snow_student_mobilenet_val_iou"]
        print(f"  snow   IoU {agg['snow_iou']:.4f} vs {PUBLISHED['snow_student_mobilenet_val_iou']}"
              f" (checkpoint's own val IoU) delta {d:+.4f}   [annotated frames, not the val split]")
    print("  psg_raw        no published per-frame counterpart (see module docstring)")
    if agg.get("psg_smoothed") is None:
        print("  psg_smoothed   null: no full window in this run (expected for one-shot frames)")
    else:
        print(f"  psg_smoothed   {agg['psg_smoothed']:.4f}  PROVISIONAL: packaged uni smoother "
              f"does not record its training seq_len (ran with {args.seq_len})")

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
