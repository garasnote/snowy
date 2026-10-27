#!/usr/bin/env python3
"""Lay the deposited checkpoints out in the tree the evaluation scripts expect.

The Zenodo record ships flat, renamed files (`models/ground/ground_ft_dinov2l_f4.ckpt`)
plus `models/CHECKPOINTS.csv`. `eval_all.py` looks for the original layout instead
(`models/student_ground/checkpoints/<run>/best-*.ckpt`). This script reads the CSV
and creates that layout, linking each flat file to the path it came from.

    python tools/link_deposit_models.py            # symlinks, the default
    python tools/link_deposit_models.py --copy     # real copies, ~10 GB
    python tools/link_deposit_models.py --check    # verify only, exit 1 on any mismatch

Run it once, after unpacking the record into the repository root.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path


def check(rows, root: Path) -> int:
    """Compare the laid-out tree against CHECKPOINTS.csv. Reads, never writes.

    Three ways a row can fail, reported separately because the fixes differ:
      absent     nothing at source_run_dir/source_epoch_file (never linked, or the
                 record did not contain it)
      dangling   a symlink whose target is gone (the record was moved or deleted)
      size       the file exists but its byte count is not the CSV's size_bytes,
                 so it is a different or truncated checkpoint than the paper used
    """
    ok = 0
    problems = {"absent": [], "dangling": [], "size": []}
    for row in rows:
        dst = root / row["source_run_dir"] / row["source_epoch_file"]
        label = f"{row['filename']}  ->  {dst.relative_to(root)}"
        if dst.is_symlink() and not dst.exists():
            problems["dangling"].append(label)
        elif not dst.exists():
            problems["absent"].append(label)
        else:
            want = int(row["size_bytes"]) if row.get("size_bytes") else None
            got = dst.stat().st_size  # follows the symlink to the real file
            if want is not None and got != want:
                problems["size"].append(f"{label}  (csv {want} B, on disk {got} B)")
            else:
                ok += 1
    bad = sum(len(v) for v in problems.values())
    print(f"checkpoints: {ok}/{len(rows)} match CHECKPOINTS.csv, {bad} do not")
    notes = {"absent": "not in place", "dangling": "symlink points at nothing",
             "size": "wrong size, not the checkpoint the paper used"}
    for kind, items in problems.items():
        if items:
            print(f"\n{kind.upper()} - {notes[kind]} ({len(items)}):")
            for item in items:
                print(f"  {item}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root holding models/ (default: .)")
    ap.add_argument("--csv", default=None, help="CHECKPOINTS.csv (default: models/CHECKPOINTS.csv)")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="change nothing; verify every CSV row is in place at the "
                         "expected path with the expected size, exit 1 otherwise")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    csv_path = Path(args.csv) if args.csv else root / "models" / "CHECKPOINTS.csv"
    if not csv_path.exists():
        print(f"no {csv_path}; the record ships it as models/CHECKPOINTS.csv", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(csv_path.open()))
    if args.check:
        return check(rows, root)
    made = skipped = missing = 0
    missing_list = []
    by_name = None

    def resolve(filename: str) -> Path | None:
        """The deposited file for a CSV row.

        Rows are written either with the full path (`models/qa/qa_gradboost.pkl`)
        or with the bare filename (`qa_gradboost.pkl`), so a bare name is looked
        up under models/ before giving up.
        """
        nonlocal by_name
        direct = root / filename
        if direct.exists():
            return direct
        if by_name is None:
            by_name = {}
            for p in (root / "models").rglob("*"):
                if p.is_file():
                    by_name.setdefault(p.name, p)
        name = Path(filename).name
        stem = Path(name).stem
        hit = by_name.get(name)
        if hit is not None:
            return hit
        # CHECKPOINTS.csv calls the QA gate qa_cnn.ckpt; the record ships it as
        # qa_cnn.pt (TorchScript). Accept that one substitution, nothing broader.
        alt = {".ckpt": ".pt", ".pt": ".ckpt"}.get(Path(name).suffix)
        return by_name.get(stem + alt) if alt else None

    for row in rows:
        src = resolve(row["filename"])
        dst = root / row["source_run_dir"] / row["source_epoch_file"]
        if dst.exists():
            skipped += 1
            continue
        if src is None:
            missing += 1
            missing_list.append(row["filename"])
            continue
        if args.dry_run:
            made += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.relpath(src, dst.parent), dst)
        made += 1

    verb = "would link" if args.dry_run else ("copied" if args.copy else "linked")
    print(f"{verb} {made} checkpoint(s); {skipped} already in place; {missing} missing locally")
    if missing:
        print("\nno file at the deposited path, so the layout stays incomplete for these:")
        for name in missing_list:
            print(f"  {name}")
        if any("smoother" in n for n in missing_list):
            print("Extract models/smoothers/scalers.zip beside the smoother checkpoints first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
