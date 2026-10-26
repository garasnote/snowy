#!/usr/bin/env python3
"""Assemble a Hugging Face Space bundle from this repo.

The Space needs more than deploy/: chain.py imports the repo's own model builders
(student_ground.configs / build_model), the student_snow wrapper, the temporal
snow model, the QA net and the valid-region masks. This copies exactly that import
closure — no data, no imagery, no checkpoints (weights come from the Hub at boot,
see deploy/weights.py).

    python deploy/pack_space.py --dest /tmp/space_bundle

Nothing is uploaded here: pushing is the user's call.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# directories copied wholesale (python only)
DIRS = ["deploy", "models/student_ground", "models/student_snow",
        "models/temporal_snow", "qa", "analysis"]
# single files the chain imports
FILES = ["paths.py"]
# valid-region masks: small PNGs, one per camera
MASKS = ("preprocessing/segmentation/camera_valid_regions", ".png")

SPACE_README = """---
title: snowy PSG inference
emoji: ❄️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# snowy — presence of snow on ground (deployment demo)

QA gate → ground mask → snow mask → PSG for one webcam frame.

* `psg_raw` = snow pixels inside the ground mask / ground pixels × 100
* `psg_smoothed` is **null** for a single frame: the smoother is a causal GRU over a
  window of frames of the same camera. Use the series tab with `seq_len`+ frames.

Every stage reports its checkpoint audit in the UI (keys loaded, keys left random).
The app refuses to start if a checkpoint is incomplete — that check exists because
an earlier evaluation silently scored 84/256 randomly-initialised layer groups.

Weights are fetched from a private Hub repo at boot (`SNOWY_WEIGHTS_HUB`).
"""


def copy_dir(src: Path, dest: Path) -> tuple[int, int]:
    n = 0
    size = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if "__pycache__" in rel.parts or p.suffix in (".pyc", ".ckpt", ".pt"):
            continue
        if p.suffix == ".jsonl" or rel.name.startswith(("acceptance_report",
                                                        "qa_only_report")):
            continue                                          # score caches and reports
        if rel.parts and rel.parts[0] == "space_bundle":
            continue
        if p.is_dir():
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
        size += p.stat().st_size
    return n, size


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(REPO / "deploy" / "space_bundle"))
    args = ap.parse_args(argv)
    dest = Path(args.dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    total_n = total_b = 0
    for d in DIRS:
        src = REPO / d
        if not src.exists():
            raise FileNotFoundError(f"missing source directory: {src}")
        n, b = copy_dir(src, dest / d)
        total_n += n
        total_b += b
        print(f"  {d:28s} {n:4d} files  {b/1024:8.1f} KiB")

    for f in FILES:
        src = REPO / f
        if not src.exists():
            raise FileNotFoundError(f"missing source file: {src}")
        shutil.copy2(src, dest / f)
        total_n += 1
        total_b += src.stat().st_size

    mask_src = REPO / MASKS[0]
    if not mask_src.exists():
        raise FileNotFoundError(
            f"{mask_src} is missing — the valid-region masks are required, they are "
            f"applied per camera before scoring")
    mdest = dest / MASKS[0]
    mdest.mkdir(parents=True, exist_ok=True)
    masks = sorted(mask_src.glob(f"*{MASKS[1]}"))
    for p in masks:
        shutil.copy2(p, mdest / p.name)
    total_n += len(masks)
    total_b += sum(p.stat().st_size for p in masks)
    print(f"  {'valid regions':28s} {len(masks):4d} files")

    # Space entry point and metadata
    shutil.copy2(REPO / "deploy" / "app.py", dest / "app.py")
    shutil.copy2(REPO / "deploy" / "requirements-space.txt", dest / "requirements.txt")
    (dest / "README.md").write_text(SPACE_README)
    (dest / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    print(f"\n{total_n} files, {total_b/1024/1024:.1f} MiB -> {dest}")
    print("weights are NOT included: set SNOWY_WEIGHTS_HUB so deploy/weights.py "
          "fetches them at boot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
