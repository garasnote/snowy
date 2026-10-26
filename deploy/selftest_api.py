#!/usr/bin/env python3
"""Self-test for the /predict code path, without needing FastAPI installed.

api.py is deliberately thin: it validates the JSON body with
contract.PredictRequest.from_dict, walks the frames through SnowyPipeline, and
returns the built response. This script exercises exactly that path with a real
request, so the logic is testable wherever torch exists (a Space build, a CI box,
a reviewer's laptop) — and it asserts the two cases the contract turns on:

  * one frame  -> psg_smoothed is null and says why
  * a full window -> psg_smoothed is a number

    python deploy/selftest_api.py --camera CM00014 --seq-len 4

Exit code 0 means the path is sound; anything else prints what failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO), str(_REPO / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from deploy import contract                                    # noqa: E402
from deploy.chain import SnowyPipeline                         # noqa: E402


def run_request(pipe: SnowyPipeline, body: dict) -> dict:
    """Mirrors deploy/api.py:predict — the same three steps, no web layer."""
    req = contract.PredictRequest.from_dict({k: v for k, v in body.items() if v is not None})
    frames = req.ordered_frames()
    resp = None
    for frame in frames:
        resp, _raw = pipe.predict_image(frame.image, req.camera_id,
                                        frame.resolved_timestamp(), req.meteo)
    resp.setdefault("meta", {})["frames_used"] = len(frames)
    return resp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="CM00014")
    ap.add_argument("--seq-len", type=int, default=4)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--img-root", default=None, help="where the jpgs live")
    ap.add_argument("--weights", default=str(_REPO.parent / "models" / "best"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    root = Path(args.img_root) if args.img_root else _REPO / "data"
    jpgs = sorted((root / args.camera).glob("*.jpg"))[:args.frames]
    if not jpgs:
        print(f"no jpgs under {root/args.camera}")
        return 2

    pipe = SnowyPipeline(args.weights, device=args.device, seq_len=args.seq_len)
    pipe.load_smoother()
    failures = []

    print("=== case 1: one frame ===")
    one = run_request(pipe, {"image": str(jpgs[-1]), "camera_id": args.camera,
                             "meteo": {"t2m": -1.0, "visibility": 9000, "weather_code": 71}})
    for k in ("qa_score", "psg_raw", "psg_smoothed", "ground_mask", "snow_mask"):
        if k not in one:
            failures.append(f"response missing '{k}'")
    print(f"  keys={sorted(k for k in one if k != 'meta')}")
    print(f"  qa_score={one['qa_score']}  psg_raw={one['psg_raw']}  "
          f"psg_smoothed={one['psg_smoothed']}")
    if one["psg_smoothed"] is not None:
        failures.append("one frame produced a psg_smoothed value — the contract says null")
    if not one.get("meta", {}).get("psg_smoothed_reason") and one["psg_smoothed"] is None:
        failures.append("psg_smoothed is null without a reason in meta")

    print(f"=== case 2: {len(jpgs)} frames (window {args.seq_len}) ===")
    pipe2 = SnowyPipeline(args.weights, device=args.device, seq_len=args.seq_len)
    pipe2.load_smoother()
    series = run_request(pipe2, {"camera_id": args.camera,
                                 "frames": [{"image": str(p)} for p in jpgs],
                                 "meteo": {"t2m": -1.0, "visibility": 9000,
                                           "weather_code": 71}})
    print(f"  psg_raw={series['psg_raw']}  psg_smoothed={series['psg_smoothed']}  "
          f"window={series['meta'].get('window_seen')}/{series['meta'].get('seq_len')}")
    if len(jpgs) >= args.seq_len and series["psg_smoothed"] is None:
        failures.append(f"{len(jpgs)} frames >= seq_len {args.seq_len} but still null: "
                        f"{series['meta'].get('psg_smoothed_reason')}")

    print("=== contract rejections ===")
    for bad, why in (({"image": "x.jpg"}, "no camera_id"),
                     ({"camera_id": args.camera}, "no image"),
                     ({"camera_id": args.camera, "image": "x.jpg", "bogus": 1}, "unknown field")):
        try:
            contract.PredictRequest.from_dict(bad)
            failures.append(f"contract accepted an invalid request ({why})")
        except ValueError:
            print(f"  rejected as expected: {why}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
