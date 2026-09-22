#!/bin/bash
#SBATCH --job-name=img-stats
#SBATCH --partition=amd
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/shared/home/juan.osorio/output/img_stats_%j.out

# Recompute image statistics + regenerate snow comparison plots.
# Only active 9 cameras (excludes CM00006, CM00024, CM00068).

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
STATS_DIR="${MASTER_DIR}/preprocessing/statistics"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/cpu_env.sqfs"

CAMERAS="CF00001 CM00009 CM00014 CM00070 CM00075 CM00080 CM00102 CM00108 CM00112"
PATTERN="{$(echo $CAMERAS | tr ' ' ',')}/*.jpg"

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "IMAGE STATISTICS — RECOMPUTE + PLOTS"
echo "============================================================"
echo "  Cameras: ${CAMERAS}"
echo "  Data:    ${DATA_DIR}"
echo "  Output:  ${STATS_DIR}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    pip install --break-system-packages --quiet pyarrow seaborn 2>/dev/null || true

    echo '=== Step 1: Compute image statistics ==='
    for CAM in ${CAMERAS}; do
        echo \"  Processing \${CAM}...\"
        python ${STATS_DIR}/compute_image_stats.py \
            --data-dir ${DATA_DIR} \
            --output ${STATS_DIR}/image_statistics_\${CAM}.parquet \
            --pattern \"\${CAM}/*.jpg\"
    done

    echo ''
    echo '=== Step 2: Merge parquet files ==='
    python -c \"
import pandas as pd
from pathlib import Path
parts = list(Path('${STATS_DIR}').glob('image_statistics_C*.parquet'))
print(f'Merging {len(parts)} camera parquet files...')
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
df.to_parquet('${STATS_DIR}/image_statistics.parquet')
print(f'Total: {len(df)} images')
for p in parts:
    p.unlink()
\"

    echo ''
    echo '=== Step 3: Regenerate plots ==='
    cd ${STATS_DIR}
    python snow_comparison.py

    echo ''
    echo 'Done!'
  "

echo ""
echo "Job finished!"
