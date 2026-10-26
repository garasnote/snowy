#!/usr/bin/env python3
"""
Build a data-dir mirror containing only the images with real GT snow
annotations (Label Studio "Snow Mano" brushlabels), for running a teacher
only on the labeled subset instead of the full dataset. Same camera/
annotation-file mapping fusion_cnn.py's SNOW_ANN_FILES/IN_DIST_CAMERAS uses.

    python scripts/filter_labeled_snow.py --data-dir ./data --out-dir ./data_snow_labeled
"""
import argparse
import json
from pathlib import Path

IN_DIST_CAMERAS = ["CF00001", "CM00009", "CM00014", "CM00070",
                    "CM00075", "CM00080", "CM00102", "CM00108", "CM00112"]
SNOW_ANN_FILES = {
    "CF00001": "01.json", "CM00009": "09.json", "CM00014": "14.json",
    "CM00070": "70.json", "CM00075": "75.json", "CM00080": "80.json",
    "CM00102": "102.json", "CM00108": "108.json", "CM00112": "112.json",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--ann-dir", type=str, default=None,
                   help="Default: <data-dir>/annotations")
    args = p.parse_args()

    src_root = Path(args.data_dir)
    ann_dir = Path(args.ann_dir) if args.ann_dir else src_root / "annotations"
    dst_root = Path(args.out_dir)
    dst_root.mkdir(exist_ok=True)

    total = 0
    for cam in IN_DIST_CAMERAS:
        p_ann = ann_dir / SNOW_ANN_FILES[cam]
        if not p_ann.exists():
            print(f"{cam}: MISSING {p_ann}")
            continue
        tasks = json.load(open(p_ann))
        stems = set()
        for t in tasks:
            url = t["data"]["image"]
            if cam not in url:
                continue
            stem = url.split("/")[-1].replace(".jpg", "")
            anns = t.get("annotations", [])
            if any(r.get("type") == "brushlabels" and "rle" in r.get("value", {})
                   and "Snow Mano" in r.get("value", {}).get("brushlabels", [])
                   for ann in anns for r in ann.get("result", [])):
                stems.add(stem)
        cam_dst = dst_root / cam
        cam_dst.mkdir(exist_ok=True)
        linked = 0
        for stem in stems:
            src_img = src_root / cam / f"{stem}.jpg"
            if not src_img.exists():
                continue
            dst_img = cam_dst / f"{stem}.jpg"
            if not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())
            linked += 1
        total += linked
        print(f"{cam}: {linked}/{len(stems)} annotated stems linked")

    print(f"TOTAL: {total} labeled images ready at {dst_root}")


if __name__ == "__main__":
    main()
