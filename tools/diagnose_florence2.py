#!/usr/bin/env python3
"""Diagnose the Florence-2 teacher on this machine, through the release's own
code path (it imports snow_teachers.py, so a failure here is the failure you
would hit in a real run).

    python tools/diagnose_florence2.py                        # one image, large
    python tools/diagnose_florence2.py --model-size base      # smaller model
    python tools/diagnose_florence2.py --device cpu           # rule out the GPU

Prints the versions that matter, the GPU and its memory, then loads, runs one
frame, and reports peak VRAM and how many pixels the mask covers. On failure it
prints the full traceback and classifies it, which is what separates a weak GPU
from a code regression.
"""
import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Resolve the release root: next to this script's parent (local checkout), or
# the repo itself when tools/ is bind-mounted into the container at /workspace.
_CANDIDATES = [HERE.parent / "paper-code-release", HERE.parent, Path("/workspace")]
ROOT = next((c for c in _CANDIDATES if (c / "segmentation" / "snow_teachers.py").exists()),
            _CANDIDATES[0])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "segmentation"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="frame to test (default: any under data/CF00001)")
    ap.add_argument("--model-size", choices=["base", "large"], default="large")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--text-prompt", default="snow")
    args = ap.parse_args()

    import torch
    import transformers
    from PIL import Image

    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    print(f"  python        {sys.version.split()[0]}")
    print(f"  torch         {torch.__version__}")
    print(f"  transformers  {transformers.__version__}")
    print(f"  cuda          {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu           {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  vram total    {total:.1f} GiB")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device used   {device}")

    # pick an image
    img_path = args.image
    if img_path is None:
        cands = sorted((ROOT / "data" / "CF00001").glob("*.jpg"))
        if not cands:
            cands = sorted((ROOT / "data").glob("*/*.jpg"))
        if not cands:
            sys.exit("no input frames found under data/ - pass --image explicitly")
        img_path = cands[0]
    img_path = Path(img_path)
    print(f"  frame         {img_path}")
    image = Image.open(img_path).convert("RGB")
    print(f"  frame size    {image.width}x{image.height}")

    spec = importlib.util.spec_from_file_location(
        "snow_teachers", ROOT / "segmentation" / "snow_teachers.py")
    st = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(st)
    except SystemExit:
        pass

    print()
    print("=" * 78)
    print(f"LOAD microsoft/Florence-2-{args.model_size}  (release code path)")
    print("=" * 78)
    try:
        t0 = time.time()
        model, processor = st.florence2_load_model(args.model_size, device)
        print(f"  loaded in {time.time() - t0:.1f}s")
        if device == "cuda":
            print(f"  vram after load   {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    except Exception:
        print("\nLOAD FAILED\n")
        traceback.print_exc()
        return 2

    print()
    print("=" * 78)
    print(f"RUN referring-expression segmentation, prompt={args.text_prompt!r}")
    print("=" * 78)
    try:
        t0 = time.time()
        parsed = st.florence2_run(model, processor, image, args.text_prompt, device=device)
        dt = time.time() - t0
        print(f"  ran in {dt:.1f}s")
        if device == "cuda":
            print(f"  peak vram         {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
        mask = st.florence2_polygons_to_mask(parsed, image.width, image.height)
        import numpy as np
        covered = int(np.asarray(mask).sum())
        print(f"  mask pixels       {covered} / {image.width * image.height} "
              f"({100.0 * covered / (image.width * image.height):.2f}%)")
        if covered == 0:
            print("  NOTE: empty mask. The model loaded and ran, but found no snow")
            print("        here - that is a prompt/model-quality question, not a crash.")
    except torch.cuda.OutOfMemoryError:
        print("\nOUT OF MEMORY - this is the GPU, not the code.\n")
        traceback.print_exc()
        if device == "cuda":
            print("\nTry: --model-size base, then --device cpu to confirm.")
        return 3
    except Exception:
        print("\nRUN FAILED (not OOM) - this is a code/library failure.\n")
        traceback.print_exc()
        return 4

    print()
    print("OK: Florence-2 loads and produces a mask on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
