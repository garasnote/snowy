#!/usr/bin/env python3
"""Checkpoint auditor for the snowy release — loud, or it fails.

Why this exists
---------------
eval_all.py builds a model from a variant name, filters the checkpoint's
state_dict down to keys that exist with matching shapes, then calls
load_state_dict(..., strict=False). Anything dropped keeps its random
initialisation and the eval still prints an IoU. That is how a ViT-LoRA run
reported numbers with 84 of 256 layers untrained (see diag_keys.py and
paper/eval_reproductions_findings.md). A deployable loader must never do that.

Two halves, deliberately separable:

1. read_checkpoint(path) — torch-FREE. A .ckpt is a zip holding a pickled
   state_dict; we unpickle it with torch stubbed out, so this works even on a
   machine with no torch installed (useful for auditing weights you are about
   to upload, on any box).

2. audit_against_model(model, state_dict, ...) — needs torch. Builds the real
   module, compares key-by-key, and RAISES on any absent key or shape mismatch
   unless you explicitly pass allow_partial=True. Reports what it loaded, what
   it refused, and which layers would have been left random.

CLI:
    python deploy/audit.py path/to/a.ckpt [more.ckpt ...]        # metadata only
    python deploy/audit.py --json a.ckpt                          # machine-readable
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import zipfile
from pathlib import Path


# ── torch-free checkpoint reading ────────────────────────────────────────────

class _Stub:
    """Stand-in for any torch/numpy class encountered while unpickling."""

    def __init__(self, *a, **k):
        self.args = a
        self.kw = k

    def __repr__(self):
        return f"Stub({self.args[:2]})"

    def __call__(self, *a, **k):
        return _Stub(*a, **k)


def _tensor(storage, storage_offset, size, stride, *rest, **kw):
    return {"__tensor__": True, "shape": tuple(size), "stride": tuple(stride)}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("collections", "builtins", "__builtin__"):
            return super().find_class(module, name)
        if name in ("_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_tensor_v3"):
            return _tensor
        if name == "_rebuild_parameter":
            return lambda data, requires_grad, **k: data
        return _Stub

    def persistent_load(self, pid):
        try:
            return {"__storage__": str(pid[1]), "numel": pid[4]}
        except Exception:
            return {"__storage__": str(pid)}


def _pickle_bytes(path: Path) -> tuple[bytes, str]:
    """Return (pickle bytes, layout name) for a torch.save artefact."""
    if not zipfile.is_zipfile(path):
        return path.read_bytes(), "bare-pickle"
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for cand in [n for n in names if n == "data.pkl" or n.endswith("/data.pkl")]:
            return z.read(cand), "legacy-zip" if not cand.startswith("archive/") else "torch-zip"
        if any(n.endswith("constants.pkl") for n in names):
            return b"", "torchscript"          # weights live in constants.pkl
        return b"", f"unknown-zip:{names[:3]}"


def read_checkpoint(path) -> dict:
    """Describe a checkpoint without needing torch. Never raises for a
    well-formed file; returns {'format': 'unreadable', 'error': ...} otherwise."""
    path = Path(path)
    info = {"path": str(path), "bytes": path.stat().st_size if path.exists() else None}
    if not path.exists():
        return {**info, "format": "missing"}

    # TorchScript archives keep weights in constants.pkl, not data.pkl. Detect
    # them first, otherwise the data.pkl whose top object is a ScriptModule
    # reconstruction just looks like an opaque pickle.
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            scripted = [n for n in names if n.endswith("constants.pkl")]
            if scripted or any("/code/" in n for n in names):
                out = {**info, "layout": "torchscript-zip", "format": "torchscript",
                       "note": "weights in constants.pkl; load with torch.jit.load()"}
                if scripted:
                    try:
                        consts = _Unpickler(__import__("io").BytesIO(z.read(scripted[0]))).load()
                        if isinstance(consts, dict):
                            shapes = {k: tuple(v["shape"]) for k, v in consts.items()
                                      if isinstance(v, dict) and v.get("__tensor__")}
                            out["constant_tensors"] = len(shapes)
                            out["conv_shapes"] = {k: s for k, s in shapes.items() if len(s) == 4}
                            out["tail_shapes"] = {k: s for k, s in shapes.items()
                                                  if len(s) in (1, 2) and "weight" in str(k)}
                    except Exception as e:
                        out["constants_error"] = f"{type(e).__name__}: {e}"
                return out

    raw, layout = _pickle_bytes(path)
    info["layout"] = layout
    if layout == "torchscript":
        return {**info, "format": "torchscript",
                "note": "weights are in constants.pkl; key-level audit needs torch.jit.load"}
    if not raw:
        return {**info, "format": "unreadable", "error": layout}

    try:
        obj = _Unpickler(__import__("io").BytesIO(raw)).load()
    except Exception as e:
        return {**info, "format": "unreadable", "error": f"{type(e).__name__}: {e}"}

    if not isinstance(obj, dict):
        return {**info, "format": "opaque-object", "top_level": type(obj).__name__}

    sd = obj.get("state_dict") or obj.get("model_state_dict")
    if not isinstance(sd, dict):
        return {**info, "format": "dict-without-state_dict", "keys": sorted(map(str, obj))[:12]}

    shapes = {k: tuple(v["shape"]) for k, v in sd.items()
              if isinstance(v, dict) and v.get("__tensor__")}
    convs = {k: s for k, s in shapes.items() if len(s) == 4}
    contract = {k: obj[k] for k in ("task", "n_teachers", "teachers", "num_classes")
                if k in obj}
    hp = obj.get("hyper_parameters") or {}
    if isinstance(hp, dict):
        contract.update({k: hp[k] for k in ("model_variant", "input_size", "use_depth",
                                           "hidden_size", "num_layers", "bidirectional")
                         if k in hp})
    return {**info, "format": "state_dict", "state_dict_keys": len(sd), "shapes": shapes,
            "conv_shapes": convs, "contract": contract, "epoch": obj.get("epoch")}


# ── the loud loader (needs torch) ────────────────────────────────────────────

def audit_against_model(model, state_dict: dict, *, name: str = "model",
                        allow_partial: bool = False) -> dict:
    """Load state_dict into model, refusing to do it silently.

    Returns a report; raises RuntimeError if keys are absent/mis-shaped and
    allow_partial is False. This is the check whose absence let 84/256 layers
    stay random in an earlier ViT-LoRA evaluation.

    Key normalisation: LightningModule checkpoints wrap the network as
    `model.<...>`, while a bare `build_model()` yields `<...>`. We try the
    state_dict as given, then with one leading component stripped, and record
    which one matched so the report never hides the transform.
    """
    import torch  # noqa: F401  (imported here so read_checkpoint stays torch-free)

    msd = model.state_dict()
    normalisation = "as-is"
    if not any(k in msd for k in state_dict):
        stripped = {k.split(".", 1)[1]: v for k, v in state_dict.items() if "." in k}
        if any(k in msd for k in stripped):
            normalisation = f"stripped leading '{next(iter(state_dict)).split('.')[0]}.' (LightningModule wrapper)"
            state_dict = stripped

    present, absent, shape_bad = {}, [], []
    for k, v in state_dict.items():
        if k not in msd:
            absent.append(k)                      # in ckpt, not in model
        elif tuple(v.shape) != tuple(msd[k].shape):
            shape_bad.append((k, tuple(v.shape), tuple(msd[k].shape)))
        else:
            present[k] = v
    untouched = [k for k in msd if k not in present]   # would stay random

    report = {"name": name, "checkpoint_keys": len(state_dict), "model_keys": len(msd),
              "key_normalisation": normalisation,
              "loaded": len(present), "absent_from_model": absent,
              "shape_mismatch": shape_bad, "left_at_random_init": untouched}

    fatal = bool(absent or shape_bad or untouched)
    if fatal and not allow_partial:
        lines = [f"[{name}] REFUSING to load a partial checkpoint:"]
        if absent:
            lines.append(f"  {len(absent)} keys in checkpoint have no home in the model, e.g. {absent[:5]}")
        if shape_bad:
            lines.append(f"  {len(shape_bad)} shapes disagree, e.g. {shape_bad[:3]}")
        if untouched:
            lines.append(f"  {len(untouched)} model layers would keep RANDOM weights, e.g. {untouched[:5]}")
        lines.append("  Pass allow_partial=True only if you can explain every line above.")
        raise RuntimeError("\n".join(lines))

    model.load_state_dict({k: v for k, v in state_dict.items() if k in present}, strict=False)
    report["loaded_ok"] = True
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)

    reports, bad = [], 0
    for p in args.checkpoints:
        r = read_checkpoint(p)
        reports.append(r)
        if r.get("format") in ("missing", "unreadable"):
            bad += 1
    if args.json:
        print(json.dumps(reports, indent=2, default=str))
    else:
        for r in reports:
            print(f"\n=== {Path(r['path']).name} ===")
            print(f"  layout   : {r.get('layout')}   format: {r.get('format')}")
            if "state_dict_keys" in r:
                print(f"  keys     : {r['state_dict_keys']}   epoch: {r.get('epoch')}")
                if r.get("contract"):
                    print(f"  contract : {r['contract']}")
                for k, s in list(r.get("conv_shapes", {}).items())[:3]:
                    print(f"  conv     : {k} {s}")
            elif "error" in r or "note" in r:
                print(f"  note     : {r.get('error') or r.get('note')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
