#!/usr/bin/env python3
"""Fill the missing Florence-2 snow masks for the annotated frames.

snow_teachers.py walks whole camera directories, so re-running `florence2`
there cannot target the specific frames the release needs and costs days per
camera. This driver loads the model once and runs only the frames that are
annotated and still have no mask, in the layout the pipeline reads:

    preprocessing/segmentation/florence2_snow_output/<camera>/<stem>_snow.png

That directory is also what workspace/segmentation/snow/florence2 points at
(see tools/fix_fusion_links.sh), so one run feeds both FusionCNN and the
evaluation. Already-present masks are skipped, so re-running resumes instead of
restarting.

    # gauge the rate on 10 frames
    python tools/fill_florence2_gaps.py --model-size base --limit 10

    # the real run
    python tools/fill_florence2_gaps.py --model-size base --device cuda

    # see what would be done, without loading the model
    python tools/fill_florence2_gaps.py --dry-run
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_CANDIDATES = [HERE.parent / "paper-code-release", HERE.parent, Path("/workspace")]
ROOT = next((c for c in _CANDIDATES if (c / "segmentation" / "snow_teachers.py").exists()),
            _CANDIDATES[0])


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=["base", "large"], default="base")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--out-dir", default=None,
                    help="default: preprocessing/segmentation/florence2_snow_output")
    ap.add_argument("--text-prompt", default="snow")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--data-dir", action="append", default=None,
                    help="tree holding <camera>/<stem>.jpg; repeatable, searched in order "
                         "(default: <repo>/data)")
    ap.add_argument("--release-dir", default=None,
                    help="tree holding segmentation/ and models/fusion_cnn/ "
                         "(default: the repo this script is run from)")
    ap.add_argument("--frames-file", default=None,
                    help="list of '<camera>/<stem>' lines to process, one per line. "
                         "Use this instead of the annotation loader when only the "
                         "teacher code is present on the machine.")
    ap.add_argument("--dry-run", action="store_true", help="list the work, load nothing")
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    rel = Path(args.release_dir) if args.release_dir else ROOT
    if not (rel / "segmentation" / "snow_teachers.py").exists():
        sys.exit(f"ERROR: {rel}/segmentation/snow_teachers.py not found "
                 f"(pass --release-dir pointing at a checkout)")
    st = load_module(rel / "segmentation" / "snow_teachers.py", "snow_teachers")

    if args.frames_file:
        # Explicit frame list. Preferred on a cluster where only segmentation/
        # was pushed: the annotation loader lives in the model packages, and
        # importing them drags in student_ground/student_snow as well.
        stems = []
        for line in Path(args.frames_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cam, _, stem = line.partition("/")
            stems.append((cam.strip(), stem.strip()))
    else:
        fc = load_module(rel / "models" / "fusion_cnn" / "fusion_cnn.py", "fusion_cnn")
        stems = fc.load_gt_stems("snow")

    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "preprocessing" / "segmentation" / "florence2_snow_output")
    roots = [Path(d) for d in args.data_dir] if args.data_dir else [ROOT / "data"]
    for r in roots:
        if not r.is_dir():
            print(f"WARNING: data root does not exist: {r}")

    # Frames are spread across roots: data/<camera>/ holds the earlier captures,
    # night_images/<camera>/ the later ones the annotations also reference.
    todo = []
    for cam, stem in stems:
        dst = out_dir / cam / f"{stem}_snow.png"
        if dst.exists():
            continue
        src = None
        for r in roots:
            cand = r / cam / f"{stem}.jpg"
            if cand.exists():
                src = cand
                break
        todo.append((cam, stem, src, dst))

    have = len(stems) - len(todo)
    print(f"{'frames from list' if args.frames_file else 'annotated snow frames'} : {len(stems)}")
    print(f"already have a mask   : {have}")
    print(f"to run                : {len(todo)}")
    print(f"output dir            : {out_dir}")
    print(f"input roots           : " + ", ".join(str(r) for r in roots))
    if todo:
        resolved = sum(1 for _, _, src, _ in todo if src is not None)
        missing = len(todo) - resolved
        for r in roots:
            n = sum(1 for _, _, src, _ in todo
                    if src is not None and str(src).startswith(str(r) + "/"))
            print(f"    {r}: {n} of {len(todo)} frames")
        if resolved == 0:
            print(f"\nFATAL: none of the {len(todo)} frames were found in any data root")
            print("  expected layout: <data-root>/<camera>/<stem>.jpg")
            print("  re-run with --data-dir pointing at the tree with the JPEGs (repeatable)")
            return 2
        if missing:
            print(f"  WARNING: {missing}/{len(todo)} frames not found in any root "
                  f"(they will be skipped)")
    if args.limit:
        todo = todo[:args.limit]
        print(f"limited to            : {len(todo)} frames")
    if args.dry_run or not todo:
        for cam, stem, src, _ in todo[:10]:
            print(f"   would run {cam}/{stem}  (image exists: {src.exists()})")
        return 0

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nloading Florence-2-{args.model_size} on {device} ...")
    t0 = time.time()
    model, processor = st.florence2_load_model(args.model_size, device)
    print(f"loaded in {time.time() - t0:.1f}s\n")

    done, failed, t_start = 0, 0, time.time()
    for i, (cam, stem, src, dst) in enumerate(todo, 1):
        if src is None:
            print(f"  [{i}/{len(todo)}] {cam}/{stem}: image missing, skipped")
            failed += 1
            continue
        try:
            image = Image.open(src).convert("RGB")
            parsed = st.florence2_run(model, processor, image, args.text_prompt, device=device)
            mask = st.florence2_polygons_to_mask(parsed, image.width, image.height)
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((np.asarray(mask) * 255).astype(np.uint8)).save(dst)
            done += 1
            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - t_start) / i
                left = (len(todo) - i) * rate
                print(f"  [{i}/{len(todo)}] {done} written, {failed} failed  "
                      f"({rate:.1f}s/frame, ~{left/3600:.1f}h remaining)")
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(todo)}] {cam}/{stem}: {type(e).__name__}: {str(e)[:90]}")

    print(f"\ndone: {done} written, {failed} failed in {(time.time()-t_start)/3600:.2f}h")
    print(f"-> {out_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
