#!/usr/bin/env python3
"""Fill the snow masks a single teacher is missing, for an explicit frame list.

`snow_teachers.py` walks whole camera directories, so re-running one of its
subcommands to recover a handful of un-annotated frames would re-segment every
frame in every camera. This driver takes an explicit list of frames and re-runs
ONE teacher over exactly those frames, loading the model once:

    # what would run, and where the frames resolve (imports nothing)
    python tools/fill_teacher_gaps.py --teacher clipseg \
        --frames-file missing_clipseg.txt --data-dir /workspace/data --dry-run

    # the real run (resumes: masks already on disk are skipped)
    python tools/fill_teacher_gaps.py --teacher clipseg \
        --frames-file missing_clipseg.txt \
        --data-dir /workspace/data --data-dir /workspace/night_images \
        --out-dir preprocessing/segmentation/clipseg_snow_output --device cuda

    # gauge the rate on the first 10 missing frames
    python tools/fill_teacher_gaps.py --teacher clipseg --limit 10 ...

The frames file is one frame per line, `CAMERA/STEM` (`CF00001/CF00001_202212111530_00`).
Each frame is resolved against the `--data-dir` roots in the order given, as
`<root>/<CAMERA>/<STEM>.jpg`, and the teacher writes its own mask layout,
`<OUT>/<CAMERA>/<STEM>_snow.png`.

Every per-teacher setting (prompt, threshold, confidence, model size, how the
SAM2 predictor / SAM3 wrapper / SegGPT prompt pairs are built) is the default
the teacher's own subcommand in `segmentation/snow_teachers.py` uses; the
source line numbers are cited at each call site below. Nothing is reimplemented
here - the model is built by and the frame is run through the teacher module's
own functions.

Safety: if the frame list resolves to zero images on disk the driver prints
FATAL naming every data root and exits 2 *before* torch, transformers or the
teacher module are imported. A previous job failed only after the model had
loaded, and another silently skipped every frame; the guard exists so a bad
path costs a second instead of a GPU hour.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_root():
    """Repo root = the tree holding segmentation/snow_teachers.py."""
    for cand in (HERE.parent, *HERE.parents):
        if (cand / "segmentation" / "snow_teachers.py").exists():
            return cand
    return HERE.parent


ROOT = _find_root()

TEACHERS = ["clipseg", "groundedsam2", "sam3", "yoloe", "seggpt"]

# Canonical mask directories for the release: <ROOT>/preprocessing/segmentation/
# <teacher>_snow_output, the layout fusion_cnn.py / student_snow read
# (models/fusion_cnn/fusion_cnn.py:104-111, models/temporal_snow/figures.py:219-224).
OUT_DIR_NAME = {t: f"{t}_snow_output" for t in TEACHERS}


def _load_teacher_module():
    """Import segmentation/snow_teachers.py lazily, by path.

    Called only from inside the per-teacher run branch: the module imports
    torch/transformers at module scope, so a missing sam3 (or a broken
    model stack) must not be able to break arg parsing, frame resolution or
    --dry-run.
    """
    path = ROOT / "segmentation" / "snow_teachers.py"
    if not path.exists():
        sys.exit(f"FATAL: {path} not found - run this from the code release")
    spec = importlib.util.spec_from_file_location("snow_teachers", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["snow_teachers"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Re-run ONE snow teacher over an explicit list of frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Safety:")[0].split("\n", 2)[2],
    )
    ap.add_argument("--teacher", required=True, choices=TEACHERS,
                    help="which teacher to re-run")
    ap.add_argument("--frames-file", required=True,
                    help="one 'CAMERA/STEM' per line (blank lines and # comments ignored)")
    ap.add_argument("--data-dir", action="append", default=None, metavar="DIR",
                    help="tree holding <DIR>/<CAMERA>/<STEM>.jpg; repeatable, "
                         "searched in the order given (default: <repo>/data)")
    ap.add_argument("--out-dir", default=None, metavar="DIR",
                    help="where masks go, <OUT>/<CAMERA>/<STEM>_snow.png "
                         "(default: <repo>/preprocessing/segmentation/<teacher>_snow_output)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N missing frames (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the frames and print the work, then exit "
                         "without importing torch or any model library")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    # seggpt only: its prompts are (image, mask) examples, not text prompts
    # (snow_teachers.py:1354-1377, 1394-1401).
    ap.add_argument("--ann-dir", default=None,
                    help="seggpt: Label Studio GT annotation dir for prompt examples")
    ap.add_argument("--prompt-dir", default=None,
                    help="seggpt: manual prompt dir (alternative to --ann-dir)")
    ap.add_argument("--num-prompts", type=int, default=5,
                    help="seggpt: prompt examples per camera (default: 5)")
    return ap.parse_args(argv)


def read_frames(frames_file):
    """Read 'CAMERA/STEM' lines, preserving order and dropping blanks/comments."""
    path = Path(frames_file)
    if not path.exists():
        print(f"FATAL: frames file not found: {path}")
        return None
    frames = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cam, _, stem = line.partition("/")
        cam, stem = cam.strip(), stem.strip()
        if not cam or not stem:
            print(f"WARNING: bad line (want CAMERA/STEM): {raw!r}")
            continue
        frames.append((cam, stem))
    return frames


def resolve_frames(frames, roots):
    """Map each frame to its source .jpg, against the roots in order."""
    resolved, unresolved, per_root = [], [], [0] * len(roots)
    for cam, stem in frames:
        for i, root in enumerate(roots):
            cand = root / cam / f"{stem}.jpg"
            if cand.exists():
                resolved.append((cam, stem, cand, i))
                per_root[i] += 1
                break
        else:
            unresolved.append((cam, stem))
    return resolved, unresolved, per_root


def main(argv=None):
    args = parse_args(argv)
    roots = [Path(d) for d in args.data_dir] if args.data_dir else [ROOT / "data"]
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "preprocessing" / "segmentation" / OUT_DIR_NAME[args.teacher])

    # seggpt cannot run without its (image, mask) prompt examples, and the
    # failure would only surface after the model had loaded -> refuse now.
    if args.teacher == "seggpt" and not (args.ann_dir or args.prompt_dir):
        print("FATAL: --teacher seggpt needs --ann-dir or --prompt-dir "
              "(its prompts are (image, mask) examples, not text)")
        return 2

    # ---- 1. resolve ------------------------------------------------------
    frames = read_frames(args.frames_file)
    if frames is None:
        return 2
    resolved, unresolved, per_root = resolve_frames(frames, roots)

    print(f"teacher               : {args.teacher}")
    print(f"frames file           : {args.frames_file}")
    print(f"frames in list        : {len(frames)}")
    print(f"resolved to an image  : {len(resolved)}")
    print(f"not found in any root : {len(unresolved)}")
    print("data roots (searched in order):")
    for i, root in enumerate(roots):
        state = "" if root.is_dir() else "   [root does not exist]"
        print(f"    [{i}] {root}: supplied {per_root[i]} frames{state}")

    # ---- 2. fatal guard, before any model import -------------------------
    # Zero resolved frames means the paths/flags are wrong; loading a model to
    # then skip every frame is the failure this guard exists to prevent.
    if not resolved:
        print()
        print(f"FATAL: none of the {len(frames)} frames in {args.frames_file} "
              f"resolved to an image; searched --data-dir roots: "
              f"{', '.join(str(r) for r in roots)}")
        for i, root in enumerate(roots):
            print(f"  FATAL data root [{i}] {root}  "
                  f"({'exists' if root.is_dir() else 'MISSING'})")
        print("  expected layout: <data-root>/<CAMERA>/<STEM>.jpg")
        print("  repeat --data-dir once per tree holding the JPEGs (searched in order)")
        return 2

    # ---- 3. skip frames that already have a mask (resume) ----------------
    todo, present = [], 0
    for cam, stem, src, _ in resolved:
        dst = out_dir / cam / f"{stem}_snow.png"
        if dst.exists():
            present += 1
            continue
        todo.append((cam, stem, src, dst))

    print(f"output dir            : {out_dir}")
    print(f"already present       : {present}  (skipped, resume-safe)")
    print(f"unresolved, skipped   : {len(unresolved)}  (no source .jpg in any root)")
    print(f"to run                : {len(todo)}")
    if unresolved:
        shown = ", ".join(f"{c}/{s}" for c, s in unresolved[:5])
        more = "" if len(unresolved) <= 5 else f" (+{len(unresolved) - 5} more)"
        print(f"  WARNING: {len(unresolved)} frames have no image in any data root: "
              f"{shown}{more}")
    if args.limit and len(todo) > args.limit:
        todo = todo[:args.limit]
        print(f"limited to            : {len(todo)} frames (--limit {args.limit})")

    # ---- 4. dry run: counts printed, nothing imported or written ---------
    if args.dry_run:
        print("\n(dry run: no model imported, no masks written)")
        for cam, stem, src, dst in todo[:10]:
            print(f"   would run {cam}/{stem}\n       image {src}\n       mask  {dst}")
        if len(todo) > 10:
            print(f"   ... and {len(todo) - 10} more")
        return 0

    if not todo:
        print("\nnothing to run - every resolved frame already has a mask.")
        return 0

    # ---- 5. load the model once, loop the frames -------------------------
    st = _load_teacher_module()
    t_load = time.time()

    if args.teacher == "clipseg":
        # snow_teachers.py:238 loads CLIPSeg; run_clipseg's defaults are
        # --text-prompt "snow" (l.216) and --threshold 0.5 (l.218).
        model, processor = st.clipseg_load_model(device=args.device)

        def run_frame(img_path, cam):
            return st.clipseg_process_image(
                img_path, model, processor, "snow", out_dir,
                device=args.device, threshold=0.5, debug=False)

    elif args.teacher == "groundedsam2":
        # snow_teachers.py:655-656 loads GroundingDINO-tiny then the SAM2
        # hiera-large predictor; defaults --box-threshold 0.3 (l.634) and
        # --text-threshold 0.25 (l.636). The text prompt "snow." is hardcoded
        # in groundedsam2_detect_boxes (l.509).
        # sam2 is a source checkout in this image rather than an installed
        # package, so make its directory importable before the loader runs
        # `from sam2.sam2_image_predictor import SAM2ImagePredictor`.
        for cand in ("/workspace/sam2", "/opt/sam2"):
            if Path(cand).is_dir() and cand not in sys.path:
                sys.path.insert(0, cand)
                print(f"  sam2 source dir on sys.path: {cand}")
        try:
            import sam2  # noqa: F401
        except ImportError as exc:
            print(f"  FATAL: 'sam2' is not importable ({exc}).")
            print("  Looked for a source checkout in /workspace/sam2 and /opt/sam2.")
            return 3
        gdino_model, gdino_processor = st.groundedsam2_load_grounding_dino(
            device=args.device)
        sam2_predictor = st.groundedsam2_load_sam2(device=args.device)

        def run_frame(img_path, cam):
            return st.groundedsam2_process_image(
                img_path, gdino_model, gdino_processor, sam2_predictor, out_dir,
                device=args.device, box_threshold=0.3, text_threshold=0.25,
                debug=False)

    elif args.teacher == "sam3":
        # snow_teachers.py:862 builds SAMWrapper(device); run_sam3 defaults are
        # --threshold 0.4 (l.838) and sky masking on (l.851, no --no-sky-mask).
        # The "sky"/"snow" prompts and their 0.3/0.4 thresholds are hardcoded
        # in SAMWrapper.segment_sky/segment_snow (l.738, l.741). No ground-mask
        # clipping here: the release loop runs sam3 without --ground-mask-dir
        # (How-to-guide.md:142); clipping is a separate post-pass
        # (snow_utils.py clip-to-ground).
        sam = st.SAMWrapper(device=args.device)

        def run_frame(img_path, cam):
            return st.sam3_process_image(
                img_path, sam, 0.4, out_dir,
                ground_mask_dir=None, mask_sky=True, debug=False)

    elif args.teacher == "yoloe":
        # snow_teachers.py:1016 loads yoloe-26 with run_yoloe's defaults:
        # --model-size "l" (l.995) and --conf 0.25 (l.997); yoloe_load_model
        # sets the class list to ["snow"] (l.900).
        model = st.yoloe_load_model("l", device=args.device)

        def run_frame(img_path, cam):
            return st.yoloe_process_image(img_path, model, 0.25, out_dir,
                                          debug=False)

    else:  # seggpt
        # snow_teachers.py:1394-1401 loads the prompt pairs (manual dir, or GT
        # annotations + the data root, with the extra roots as image fallbacks
        # and the mask cache next to the masks, l.1397); :1412 loads the model;
        # :1418 builds the cross-camera fallback prompts and :1429 uses
        # prompts[cam] when the camera has its own examples. --num-prompts 5
        # (l.1369).
        if args.prompt_dir:
            prompts = st.seggpt_load_manual_prompts(
                args.prompt_dir, num_prompts=args.num_prompts)
        else:
            prompts = st.seggpt_load_gt_prompts(
                args.ann_dir, roots[0], out_dir / "_prompt_cache",
                num_prompts=args.num_prompts,
                fallback_img_dirs=[str(r) for r in roots[1:]])
        if not prompts:
            print("\nFATAL: seggpt found no prompt examples "
                  "(check --ann-dir / --prompt-dir)")
            return 2
        all_pairs = [pair for cam in prompts for pair in prompts[cam]]
        fallback_prompts = all_pairs[:args.num_prompts]
        model, processor = st.seggpt_load_model(device=args.device)

        def run_frame(img_path, cam):
            cam_prompts = prompts.get(cam, fallback_prompts)
            return st.seggpt_process_image(
                img_path, cam_prompts, model, processor, out_dir,
                device=args.device, debug=False)

    print(f"model loaded in {time.time() - t_load:.1f}s\n")

    n = len(todo)
    done = failed = 0
    t_start = time.time()
    print(f"running {n} frame(s) on {args.device} ...")
    for i, (cam, stem, src, dst) in enumerate(todo, 1):
        try:
            result = run_frame(src, cam)
            # "written" means the file is actually on disk, not just that the
            # teacher returned a dict.
            if result is None or not dst.exists():
                raise RuntimeError(
                    "no mask written" if result is not None
                    else "teacher returned None (see message above)")
            done += 1
        except Exception as e:
            failed += 1
            print(f"  [{i}/{n}] {cam}/{stem}: {type(e).__name__}: {str(e)[:120]}")
        if i % 10 == 0 or i == n:
            rate = (time.time() - t_start) / i
            left = (n - i) * rate
            print(f"  [{i}/{n}] {done} written, {failed} failed  "
                  f"({rate:.1f}s/frame, ~{left / 3600:.1f}h remaining)")

    print(f"\ndone: {done} written, {failed} failed "
          f"in {(time.time() - t_start) / 3600:.2f}h")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
