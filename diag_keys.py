#!/usr/bin/env python3
"""Instant check: does eval_all's model-building path load the whole checkpoint,
and what input size did training actually use?

eval_all builds the model from a variant name, then filters the checkpoint's
state_dict down to keys that exist with matching shapes and loads with
strict=False. Anything dropped is silently left at its random init -- which
would lower the reported IoU without any error. Checkpoint hyper_parameters
also record the training-time settings.
"""
import sys
from pathlib import Path

sys.path.insert(0, "models")
sys.path.insert(0, ".")
import torch
from student_ground.models import build_model
from student_ground.configs import get_config

CKPTS = [
    ("mobilenet", "models/student_snow/checkpoints/snow_gt_mobilenet_20260522_052052"),
    ("convnext", "models/student_snow/checkpoints/snow_gt_convnext_unfrozen"),
    ("vit_lora", "models/student_snow/checkpoints/snow_gt_vit_lora_unfrozen"),
]

for fallback, d in CKPTS:
    ck = sorted(Path(d).glob("best-*.ckpt"))
    if not ck:
        print(f"{fallback}: no checkpoint found"); continue
    path = ck[-1]
    c = torch.load(path, map_location="cpu", weights_only=False)
    sd = c.get("state_dict", {})
    hp = c.get("hyper_parameters", {}) or {}
    variant = hp.get("model_variant", fallback)
    print(f"\n=== {fallback} ({path.name}) ===")
    print(f"  hyper_parameters keys : {sorted(hp)[:12]}")
    for k in ("model_variant", "input_size", "image_size", "lr", "batch_size", "use_depth", "lora_rank"):
        if k in hp:
            print(f"  hp[{k}]              = {hp[k]}")
    epoch = c.get("epoch")
    print(f"  epoch                 = {epoch}")

    try:
        model = build_model(variant)
    except Exception as e:
        print(f"  build_model({variant}) FAILED: {e}")
        continue

    msd = model.state_dict()
    with_model = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    no_prefix = {k: v for k, v in sd.items() if not k.startswith("model.")}
    matched, shape_bad, absent = {}, [], []
    for k, v in with_model.items():
        if k not in msd:
            absent.append(k)
        elif tuple(v.shape) != tuple(msd[k].shape):
            shape_bad.append((k, tuple(v.shape), tuple(msd[k].shape)))
        else:
            matched[k] = v
    print(f"  state_dict keys       = {len(sd)}  (prefixed 'model.': {len(with_model)}, "
          f"other prefixes: {len(no_prefix)})")
    if no_prefix:
        print(f"    non-'model.' keys   : {sorted(no_prefix)[:6]}")
    print(f"  built model keys      = {len(msd)}")
    print(f"  MATCHED               = {len(matched)}")
    print(f"  ABSENT from model     = {len(absent)}")
    print(f"  SHAPE MISMATCH        = {len(shape_bad)}")
    if absent:
        print(f"    absent examples     : {absent[:6]}")
    if shape_bad:
        for k, a, b in shape_bad[:6]:
            print(f"    {k}: ckpt{a} vs model{b}")
    model.load_state_dict({k: v for k, v in sd.items() if k in msd}, strict=False)
    leftovers = [k for k in msd if k not in matched]
    print(f"  layers left at random init: {len(leftovers)}")
    if leftovers:
        print(f"    e.g. {leftovers[:8]}")
    print(f"  eval_all would use input_size={get_config(variant).get('input_size')}")
