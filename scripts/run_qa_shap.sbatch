#!/bin/bash
#SBATCH --job-name=qa-shap
#SBATCH --partition=amd
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/shared/home/juan.osorio/output/qa_shap_%j.out

# Load saved QA model + compute SHAP feature importances.
# No training — just analysis on existing pickle.

MASTER_DIR="/shared/home/juan.osorio/master"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/cpu_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    pip install --break-system-packages --quiet shap 2>/dev/null || true

    python -c \"
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output_dir = '${MASTER_DIR}/qa/checkpoints'

# Load model
with open(f'{output_dir}/qa_gradboost.pkl', 'rb') as f:
    bundle = pickle.load(f)
clf = bundle['model']
feature_cols = bundle['feature_cols']

# Load features from CV predictions
df = pd.read_csv(f'{output_dir}/qa_cv_predictions.csv')
X = df[feature_cols].values.astype(np.float32)

print('Computing SHAP values...')
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

mean_shap = np.abs(shap_values).mean(axis=0)
shap_sorted = np.argsort(mean_shap)[::-1]

print()
print('Top 30 features (mean |SHAP|):')
for i in range(min(30, len(feature_cols))):
    idx = shap_sorted[i]
    print(f'  {feature_cols[idx]:<40s} {mean_shap[idx]:.4f}')

# Bar plot
fig, ax = plt.subplots(figsize=(10, 8))
top_n = 30
top_idx = shap_sorted[:top_n]
top_names = [feature_cols[i] for i in top_idx]
top_vals = mean_shap[top_idx]
ax.barh(range(top_n), top_vals[::-1])
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_names[::-1], fontsize=8)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('QA Classifier — Top 30 Feature Importances (SHAP)')
plt.tight_layout()
fig.savefig(f'{output_dir}/shap_importance.png', dpi=150)
plt.close()
print(f'Saved: {output_dir}/shap_importance.png')

# Beeswarm
fig2, ax2 = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values[:, top_idx[:20]],
    features=X[:, top_idx[:20]],
    feature_names=[feature_cols[i] for i in top_idx[:20]],
    show=False,
)
fig2 = plt.gcf()
fig2.savefig(f'{output_dir}/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {output_dir}/shap_beeswarm.png')

print()
print('Done!')
\"
  "

echo ""
echo "Job finished!"
