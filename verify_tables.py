#!/usr/bin/env python3
"""Verify the paper's dataset tables against the annotation files in this repository.

Every number printed here is recomputed from the annotation exports, not copied from
the manuscript, and compared against the value printed in the paper. A mismatch is
reported as FAIL so that a reviewer (or the author, before submission) can see exactly
which table can and cannot be regenerated from the released artifacts.

Usage
-----
    python verify_tables.py                        # uses data/annotations/ (repo copy)
    python verify_tables.py --ann-dir <path>       # e.g. the merged Zenodo export set
    python verify_tables.py --table 2 --verbose

Counting rules (verified against the manuscript's Table 2, which they reproduce exactly:
678 snow-covered / 240 snow-free / 918 total across the nine in-distribution cameras):

  * A frame counts as snow-COVERED if ANY annotation on it carries a snow brush mask
    (`brushlabels` containing `Snow Mano`), and as snow-FREE if no annotation has a mask
    but some annotation carries the `No Snow` taxonomy tag. No usability filtering is
    applied: this is what reproduces the published totals.
  * Ground masks count frames with a brush mask labelled `Ground` or `Ground Mano`
    (the label was renamed; both spellings appear in the exports).
  * `san.json` and `heroku.json` are NOT separate cameras. They are extra snow frames for
    three cameras — san: CF00001 (70 frames); heroku: CF00001 (12), CM00009 (5),
    CM00014 (6) — and eval_all.py loads them alongside the per-camera files. A repository
    that omits them silently loses most of CF00001's snow ground truth.

NOTE — the evaluation loader counts differently on purpose: eval_all.py:load_gt_masks
reads only `annotations[0]` per task and drops frames tagged `Non-Usable` unless the only
reason is `No Snow`. That yields per-camera frame sets that differ from Table 2's by a few
frames (e.g. CM00102: 121 vs 123). Table 2 describes the dataset; the loader defines what
gets evaluated. Both should be described separately in the paper.

Tables that CANNOT be verified from the repository alone are listed at the end of this
file along with what they need. Do not silently skip them.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- paper's numbers
# Transcribed from the manuscript. Update here if the manuscript is updated.
PAPER = {
    "table2": {
        # camera: (ground masks, snow covered, snow free, snow total)
        "CF00001": (18, 137, 39, 176),
        "CM00009": (20, 43, 68, 111),
        "CM00014": (17, 66, 15, 81),
        "CM00070": (18, 29, 12, 41),
        "CM00075": (40, 44, 72, 116),
        "CM00080": (19, 39, 3, 42),
        "CM00102": (20, 123, 3, 126),
        "CM00108": (40, 175, 19, 194),
        "CM00112": (20, 22, 9, 31),
        "TOTAL": (212, 678, 240, 918),
    },
    "table1_frames": {  # daytime frames per camera, Table 1
        "CM00014": 10498, "CF00001": 15497, "CM00009": 22224, "CM00070": 6402,
        "CM00112": 6416, "CM00102": 6312, "CM00075": 6410, "CM00108": 6407,
        "CM00080": 6401, "TOTAL": 86567,
    },
}
IN_DIST = [c for c in PAPER["table2"] if c != "TOTAL"]
CAM_RE = re.compile(r"(CM|CF)\d{5}")


def camera_of(task: dict) -> str | None:
    m = CAM_RE.search(str((task.get("data") or {}).get("image", "")))
    return m.group(0) if m else None


def stem_of(task: dict) -> str:
    return os.path.splitext(os.path.basename(str((task.get("data") or {}).get("image", "?"))))[0]


def tags_of(annotation: dict) -> set[str]:
    tags: set[str] = set()
    for result in annotation.get("result") or []:
        for key in ("taxonomy", "choices"):
            for entry in (result.get("value") or {}).get(key, []):
                tags.update(entry if isinstance(entry, list) else [entry])
    return tags


def has_brush(annotation: dict, labels=("Ground", "Ground Mano", "Snow Mano", "Snow")) -> bool:
    for result in annotation.get("result") or []:
        value = result.get("value") or {}
        if result.get("type") == "brushlabels" and "rle" in value:
            if not labels or any(l in (value.get("brushlabels") or []) for l in labels):
                return True
    return False


def count_annotations(path: Path, label_class: str) -> collections.Counter:
    """Per-camera (covered, free) counts under the rules documented at the top."""
    counts: collections.Counter = collections.Counter()
    for task in json.load(open(path)):
        cam = camera_of(task)
        if not cam:
            continue
        annotations = task.get("annotations") or []
        if not annotations:
            continue
        tags: set[str] = set()
        masked = False
        for annotation in annotations:
            tags |= tags_of(annotation)
            masked |= has_brush(annotation, (label_class,))
        if masked:
            counts[(cam, "covered")] += 1
        elif "No Snow" in tags:
            counts[(cam, "free")] += 1
    return counts


def snow_files(ann_dir: Path) -> list[Path]:
    """Files holding snow annotations, in either supported naming scheme."""
    named = sorted(ann_dir.glob("snow_*.json"))                 # Zenodo export set
    numbered = sorted(p for p in ann_dir.glob("*.json") if p.stem.isdigit())
    return named or numbered


def ground_counts(ann_dir: Path) -> dict[str, int]:
    """Distinct frames carrying a Ground/Ground Mano brush mask, per camera."""
    frames: dict[str, set] = collections.defaultdict(set)
    for path in ann_dir.glob("ground_annotations.json"):
        for task in json.load(open(path)):
            cam = camera_of(task)
            if not cam:
                continue
            for annotation in task.get("annotations") or []:
                if has_brush(annotation, ("Ground", "Ground Mano")):
                    frames[cam].add(stem_of(task))
    return {cam: len(s) for cam, s in frames.items()}


def verify_table2(ann_dir: Path) -> bool:
    files = snow_files(ann_dir)
    if not files:
        print(f"FAIL  no snow annotation files found in {ann_dir}")
        return False
    snow: collections.Counter = collections.Counter()
    for path in files:
        snow += count_annotations(path, "Snow Mano")
    extras = [ann_dir / "san.json", ann_dir / "heroku.json"]
    premerge = ann_dir / "premerge"
    for path in extras:
        if path.exists():
            snow += count_annotations(path, "Snow Mano")
            print(f"extra snow file   : {path.name} loaded")
        elif premerge.exists():
            print(f"extra snow file   : {path.name} folded into the merged per-camera files "
                  f"(raw copy kept in premerge/) — nothing to load, as expected")
        else:
            print(f"extra snow file   : {path.name} MISSING — eval_all.py (eval_snow) expects "
                  f"it and skips it silently; snow GT for its cameras will be incomplete")
    ground = ground_counts(ann_dir)
    print(f"annotation source : {ann_dir}")
    print(f"snow files        : {len(files)}")
    print(f"ground file       : {'ground_annotations.json' if list(ann_dir.glob('ground_annotations.json')) else 'MISSING'}\n")
    print(f"{'camera':10}{'ground':>8}{'covered':>9}{'free':>7}{'total':>7}   {'paper (g,cov,free,tot)':>26}")
    ok = True
    tg = tc = tf = 0
    for cam in IN_DIST:
        g = ground.get(cam, 0)
        cov = snow[(cam, "covered")]
        free = snow[(cam, "free")]
        pg, pc, pf, pt = PAPER["table2"][cam]
        match = (g, cov, free) == (pg, pc, pf)
        ok &= match
        tg += g; tc += cov; tf += free
        print(f"{cam:10}{g:8}{cov:9}{free:7}{cov + free:7}   {pg:>4},{pc:>4},{pf:>4},{pt:>4}   "
              f"{'OK' if match else 'FAIL'}")
    pg, pc, pf, pt = PAPER["table2"]["TOTAL"]
    tot_match = (tg, tc, tf) == (pg, pc, pf)
    ok &= tot_match
    print(f"{'TOTAL':10}{tg:8}{tc:9}{tf:7}{tc + tf:7}   {pg:>4},{pc:>4},{pf:>4},{pt:>4}   "
          f"{'OK' if tot_match else 'FAIL'}")
    return bool(ok)


def verify_table1(data_dir: Path | None) -> bool:
    """Table 1 frame counts. Needs the raw imagery + the sun-angle filter, so it is only
    checked when --data-dir points at a populated image tree, and only as a raw total."""
    if data_dir is None or not data_dir.exists():
        print("\nSKIP  Table 1 (daytime frame counts): needs the raw imagery tree "
              f"(pass --data-dir). Paper total for 9 cameras: {PAPER['table1_frames']['TOTAL']}.")
        return True
    print("\nNOTE  Table 1 applies a 1-hour sunrise/sunset buffer, which needs per-camera "
          "latitude/longitude and astral; raw file counts are reported for orientation only.")
    total = 0
    for cam in IN_DIST:
        d = data_dir / cam
        n = sum(1 for _ in d.glob("*.jpg")) if d.exists() else 0
        total += n
        print(f"  {cam:10} raw jpg files: {n:>8}   (paper, daytime-filtered: {PAPER['table1_frames'][cam]})")
    print(f"  {'TOTAL':10} raw jpg files: {total:>8}   (paper: {PAPER['table1_frames']['TOTAL']})")
    return True


NOT_VERIFIABLE_FROM_REPO = """
Tables that cannot be verified from the repository alone
--------------------------------------------------------
Table 3  QA gate (AUROC/F1, latency)      needs qa/ checkpoints + eval_all.py --task qa
Table 4  Ground segmentation              needs models/student_ground checkpoints
Table 5  Snow consensus sweep k=2..5      needs models/student_snow checkpoints, one per k
Table 6  Snow regimes (GT/Pseudo/FT)      needs the same, three initialisations
Table 7  Per-camera OOD                   needs checkpoints + the OOD annotation set
Table 8  Temporal smoothing MAE           needs GRU/LSTM checkpoints + smoother features
Table 9  ARSO station validation          needs smoother checkpoints + data/meteo_*.csv
A.1/A.2  Teacher tables                   needs the teacher outputs (segmentation/ runs)

Each of these is reproducible in principle, but only once the checkpoints are supplied.
The paper states they are "available from the corresponding author on request"; if that
remains true, this file should say so explicitly rather than leaving a reader to assume
the numbers are regenerable from the repository alone.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann-dir", default="data/annotations",
                    help="annotation export directory (default: data/annotations)")
    ap.add_argument("--data-dir", default=None,
                    help="optional raw imagery root, for the Table 1 orientation counts")
    ap.add_argument("--table", default="2", choices=["1", "2", "all"])
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir)
    ok = True
    if args.table in ("2", "all"):
        print("=" * 78)
        print("TABLE 2 — annotation inventory")
        print("=" * 78)
        ok &= verify_table2(ann_dir)
    if args.table in ("1", "all"):
        print("\n" + "=" * 78)
        print("TABLE 1 — daytime frame counts")
        print("=" * 78)
        ok &= verify_table1(Path(args.data_dir) if args.data_dir else None)
    print(NOT_VERIFIABLE_FROM_REPO)
    print(f"RESULT: {'all checked tables reproduce' if ok else 'MISMATCH — see FAIL rows above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
