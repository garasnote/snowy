#!/usr/bin/env python3
"""Check that every FusionCNN input channel actually resolves.

FusionCNN reads workspace/segmentation/{ground,snow}/<teacher folder>/<camera>/
<stem>_{ground,snow}.png. A missing teacher folder - or the wrong folder, which
is easy because 'sam3_text' resolves to sam3/text and 'depth' to
depth/N0.05_F0.95 - does NOT fail: the channel is filled with zeros and the run
still completes, printing a lower but believable val IoU. Observed cost of
getting it wrong: a 1-epoch val IoU of 0.57 instead of 0.73.

    micromamba run -n snow-viewer python -u tools/verify_fusion_inputs.py

Coverage is compared against EXPECTED, which is the coverage measured on the
frame set pulled from the cluster. A channel below its expected count is a
wiring or data problem; exit code 1 means do not start training.
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "paper-code-release"
if not (ROOT / "models" / "fusion_cnn" / "fusion_cnn.py").exists():
    sys.exit(f"ERROR: release tree not found at {ROOT}")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "fusion_cnn"))
os.chdir(ROOT)

spec = importlib.util.spec_from_file_location(
    "fusion_cnn", ROOT / "models" / "fusion_cnn" / "fusion_cnn.py")
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
except SystemExit:
    pass

# Measured on the pulled frame set (212 ground / 678 snow annotated frames).
# Omitted teachers are expected to be complete. The listed ones are known data
# gaps on the cluster, not wiring bugs (verified against the pull, 2026-09-27):
# dinov2_pca uses the complete local copy; florence2 reached 678/678 after the
# cluster gap fill; seggpt tops out at 594;
# and clipseg/groundedsam2/sam3/yoloe share a 44-frame hole (CF00001:39,
# CM00009:3, CM00014:2). A count BELOW these means wiring, not data.
EXPECTED = {("ground", "dinov2_pca"): 212,   # cluster copy has 4 (CF00001 only);
                                             # the complete local copy has all 212
            ("snow", "clipseg"): 634,
            ("snow", "florence2"): 678,
            ("snow", "groundedsam2"): 634,
            ("snow", "sam3"): 634,
            ("snow", "yoloe"): 634,
            ("snow", "seggpt"): 594}

problems = []
print(f"{'task':<8}{'teacher':<14}{'have':>6}/{'exp':<7}{'shortfall':<10}worst cameras")
for task, teachers in (("ground", m.GROUND_TEACHERS), ("snow", m.SNOW_TEACHERS)):
    stems = m.load_gt_stems(task)
    for name, folder in teachers:
        exp = EXPECTED.get((task, name), len(stems))
        have, miss = 0, {}
        for cam, stem in stems:
            if (Path(folder) / cam / f"{stem}_{task}.png").exists():
                have += 1
            else:
                miss[cam] = miss.get(cam, 0) + 1
        worst = ", ".join(f"{c}:{n}" for c, n in
                          sorted(miss.items(), key=lambda kv: -kv[1])[:4]) or "-"
        note = ""
        if have < exp:
            note = f"  <-- {exp - have} below expected"
            problems.append(f"{task}/{name}: {have} of {exp} expected")
        print(f"{task:<8}{name:<14}{have:>6}/{exp:<7}{'':<10}{worst}{note}")

print()
if problems:
    print("NOT READY - these channels are short and would be zero-filled:")
    for p in problems:
        print(f"  - {p}")
    print("\nIf a ground channel is short, re-run tools/fix_fusion_links.sh.")
    sys.exit(1)
print("OK: every channel resolves as expected. Safe to train.")
