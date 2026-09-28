#!/usr/bin/env bash
# Wire the teacher-mask trees into the layout FusionCNN actually resolves.
#
# fusion_cnn.py reads every channel through WORKSPACE_DIR/segmentation/<task>/<rel>,
# and the "folder" of each teacher is NOT its name:
#
#   ground:  segformer, deeplabv3, dinov2_pca are flat,
#            'sam3_text' resolves to segmentation/ground/sam3/text
#            'sam3_dino' resolves to segmentation/ground/sam3/dino+sam
#            'depth'     resolves to segmentation/ground/depth/N0.05_F0.95
#   snow:    all six are flat (clipseg, florence2, groundedsam2, sam3, yoloe, seggpt)
#
# Get this wrong and the affected channels are silently all-zero: the run still
# completes and prints a believable (lower) val IoU. Verified: mis-wiring cost
# three ground channels and dropped a 1-epoch val IoU from 0.73 to 0.57.
#
# Usage:  bash fix_fusion_links.sh [repo_root]
#   local:    bash ~/Documents/Uni/Thesis/tools/fix_fusion_links.sh
#   cluster:  bash fix_fusion_links.sh ~/master
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$HERE/../paper-code-release}"
cd "$ROOT"
P=preprocessing/segmentation
[ -d "$P" ] || { echo "ERROR: $ROOT/$P not found (pass the repo root as \$1)" >&2; exit 1; }

mk() {  # mk <dest-relative> <source-relative>
  mkdir -p "$(dirname "$1")" 2>/dev/null || true
  if [ -L "$1" ]; then
    rm -f "$1"
  elif [ -e "$1" ]; then
    # A stale real directory (usually left by an early `docker run`, owned by
    # root and empty or partial). Move it aside rather than delete it; the
    # canonical masks live in preprocessing/segmentation/.
    mv "$1" "$1.stale-$(date +%s)" && echo "  moved aside: $1"
  fi
  ln -s "$(python3 -c "import os,sys;print(os.path.relpath(sys.argv[2], os.path.dirname(sys.argv[1])))" "$1" "$P/$2")" "$1" \
    || echo "  FAILED to link $1 (no write permission on $(dirname "$1")? see the header)"
}

# --- ground teachers (6) ------------------------------------------------------
mk workspace/segmentation/ground/segformer          segformer_output
mk workspace/segmentation/ground/deeplabv3          deeplabv3_output
mk workspace/segmentation/ground/depth/N0.05_F0.95  depth_output/N0.05_F0.95
# sam3_dino / dino+sam is intentionally not linked: it is an experimental variant
# that appears nowhere in the paper's teacher table, and fusion_cnn.py no longer
# lists it as a channel.
mk workspace/segmentation/ground/sam3/text          sam3_dino_output/text
# drop any leftover empty dino+sam directory from earlier wirings
rmdir workspace/segmentation/ground/sam3/dino+sam 2>/dev/null || true
# dinov2_pca: the cluster's dinov2_pca_output covers CF00001 only (4 of the 212
# annotated frames). A complete local copy can be left in the tree as
# dinov2_pca.stale-*; prefer it, because a partial channel is zero-filled for
# the rest and quietly lowers the result.
stale=$(ls -d workspace/segmentation/ground/dinov2_pca.stale-* 2>/dev/null | head -1)
if [ -n "$stale" ]; then
  rm -f workspace/segmentation/ground/dinov2_pca
  ln -s "$(basename "$stale")" workspace/segmentation/ground/dinov2_pca \
    && echo "  dinov2_pca -> $(basename "$stale")  (complete local copy; 212/212 frames)"
else
  mk workspace/segmentation/ground/dinov2_pca dinov2_pca_output
fi

# --- snow teachers (6) --------------------------------------------------------
mk workspace/segmentation/snow/clipseg              clipseg_snow_output
mk workspace/segmentation/snow/florence2            florence2_snow_output
mk workspace/segmentation/snow/groundedsam2         groundedsam2_snow_output
mk workspace/segmentation/snow/sam3                 sam3_snow_output
mk workspace/segmentation/snow/yoloe                yoloe_snow_output
mk workspace/segmentation/snow/seggpt               seggpt_snow_output

set +e
echo "== resolved (pngs reachable under each) =="
DEPS="workspace/segmentation/ground/segformer
workspace/segmentation/ground/deeplabv3
workspace/segmentation/ground/depth/N0.05_F0.95
workspace/segmentation/ground/sam3/text
workspace/segmentation/ground/dinov2_pca
workspace/segmentation/snow/clipseg
workspace/segmentation/snow/florence2
workspace/segmentation/snow/groundedsam2
workspace/segmentation/snow/sam3
workspace/segmentation/snow/yoloe
workspace/segmentation/snow/seggpt"
echo "$DEPS" | while read -r d; do
  [ -z "$d" ] && continue
  n=$(find -L "$d" -name '*.png' 2>/dev/null | wc -l)
  printf '  %-50s pngs=%s\n' "$d" "$n"
done
echo
echo "All lines should be >0. dinov2_pca uses the complete local copy when one is"
echo "present (see the note above it), otherwise the cluster's CF00001-only tree."
echo "A zero or a small count anywhere else means this machine's preprocessing/"
echo "tree differs from the one this script was written against."
