#!/usr/bin/env python3
"""Build a hybrid Phase 1 checkpoint: ground decoder from one epoch, snow
decoder from another. Both source files must have the same shapes (they
do — same architecture, just different training epoch).

Usage:
    python build_hybrid_pseudo_ckpt.py \
        --ground-ckpt checkpoints/dinov3_base/pseudo_epoch005.pt \
        --snow-ckpt   checkpoints/dinov3_base/pseudo_epoch015.pt \
        --out         checkpoints/dinov3_base/pseudo_phase1_hybrid.pt
"""
import argparse
import torch

p = argparse.ArgumentParser()
p.add_argument("--ground-ckpt", required=True)
p.add_argument("--snow-ckpt", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

ground_src = torch.load(args.ground_ckpt, map_location="cpu")
snow_src   = torch.load(args.snow_ckpt, map_location="cpu")

assert "ground_decoder" in ground_src, f"{args.ground_ckpt} missing 'ground_decoder' key"
assert "snow_decoder" in snow_src, f"{args.snow_ckpt} missing 'snow_decoder' key"

hybrid = {
    "ground_decoder": ground_src["ground_decoder"],
    "snow_decoder":   snow_src["snow_decoder"],
    "source": {
        "ground_decoder_from": args.ground_ckpt,
        "snow_decoder_from":   args.snow_ckpt,
    },
}

torch.save(hybrid, args.out)
print(f"Wrote hybrid checkpoint -> {args.out}")
print(f"  ground_decoder <- {args.ground_ckpt}")
print(f"  snow_decoder   <- {args.snow_ckpt}")
