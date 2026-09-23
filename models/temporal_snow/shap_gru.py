#!/usr/bin/env python3
"""SHAP analysis on bi-directional GRU temporal smoother.

Moved from analysis/shap_gru.py — lives here because it imports the
smoother model/dataset code directly, not because it needs anything
else in analysis/.
"""
import sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "models"))

CKPT = str(_HERE / "checkpoints/smoother_gru_h128_l2_bi_k7_seq96_20260623_175446/best-epoch=114-val_mae=0.0037.ckpt")
CSV  = str(_ROOT / "analysis/snow_timeseries.csv")
OUT  = str(_ROOT / "analysis/shap_gru")
Path(OUT).mkdir(exist_ok=True)

from temporal_snow.train_smoother import TemporalSmootherModule
from temporal_snow.dataset import build_feature_columns, SNOW_PCT_COLS, FeatureScaler
import torch

module = TemporalSmootherModule.load_from_checkpoint(CKPT, map_location="cpu")
module.eval()

scaler_path = Path(CKPT).parent / "scaler.json"
scaler = FeatureScaler.load(scaler_path) if scaler_path.exists() else None
print(f"Scaler: {'loaded' if scaler else 'NOT FOUND — will fail'}")

# ── Preprocess — no QA filter, same as training ───────────────────────────────
df = pd.read_csv(CSV, parse_dates=["datetime"])
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

snow_pct_cols = [c for c in SNOW_PCT_COLS if c in df.columns]
feature_cols = build_feature_columns(snow_pct_cols, use_image_features=True)

df["hour_sin"] = np.sin(2 * math.pi * df["datetime"].dt.hour / 24)
df["hour_cos"] = np.cos(2 * math.pi * df["datetime"].dt.hour / 24)
doy = df["datetime"].dt.dayofyear
df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
df["doy_cos"] = np.cos(2 * math.pi * doy / 365)

FILL = {"t2m":0,"visibility":1000,"weather_code":0,"brightness":128,
        "contrast":50,"edge_density":10,"entropy":4,"white_frac":0,
        "low_sat_frac":0,"blue_ratio":0,"bottom_white_frac":0,
        "bottom_brightness":128,"snow_n_components":0,"snow_largest_ratio":0,
        "snow_centroid_y":0.5,"ground_coverage":0}
for col, val in FILL.items():
    if col in df.columns:
        df[col] = df[col].fillna(val)
for col in snow_pct_cols:
    df[col] = df[col].fillna(0)

if "visibility" in df.columns: df["visibility"] = np.log1p(df["visibility"])
if "snow_n_components" in df.columns: df["snow_n_components"] = np.log1p(df["snow_n_components"])

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

df = scaler.transform(df)

X = df[feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0)
print(f"Data: {X.shape}  Features: {len(feature_cols)}")

# ── Wrap model for SHAP: input (N, F) → output (N,) ─────────────────────────
SEQ_LEN = 96

class GRUWrapper(torch.nn.Module):
    """Wrap GRU to accept flat (N, F) and return scalar per sample."""
    def __init__(self, module, seq_len):
        super().__init__()
        self.module = module
        self.seq_len = seq_len

    def forward(self, x):
        # x: (N, F) — treat each row as last step of a window
        # Pad with zeros for prior context
        B = x.shape[0]
        pad = torch.zeros(B, self.seq_len - 1, x.shape[1])
        seq = torch.cat([pad, x.unsqueeze(1)], dim=1)  # (B, seq_len, F)
        out = self.module(seq)  # (B, seq_len, 1)
        out = out.squeeze(-1) if out.dim() == 3 else out
        return out[:, -1].unsqueeze(1)   # (B, 1) for GradientExplainer

wrapper = GRUWrapper(module, SEQ_LEN)
wrapper.eval()

# ── SHAP GradientExplainer ───────────────────────────────────────────────────
import shap

# Background: random 200 samples
rng = np.random.default_rng(42)
bg_idx = rng.choice(len(X), size=200, replace=False)
background = torch.from_numpy(X[bg_idx])

# Explain 500 random samples
exp_idx = rng.choice(len(X), size=500, replace=False)
X_exp = torch.from_numpy(X[exp_idx])

print("Computing SHAP values (GradientExplainer)...")
background = torch.from_numpy(X[bg_idx])
X_exp_t = torch.from_numpy(X[exp_idx])
explainer = shap.GradientExplainer(wrapper, background)
shap_values = explainer.shap_values(X_exp_t)
# GradientExplainer returns list[ndarray] or ndarray depending on n_outputs
if isinstance(shap_values, list):
    shap_values = shap_values[0]  # take first (only) output
shap_values = np.array(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]  # (N, F, 1) → (N, F)
elif shap_values.ndim == 2 and shap_values.shape[0] == len(feature_cols):
    shap_values = shap_values.T  # (F, N) → (N, F)
print(f"SHAP values shape: {shap_values.shape}  (expected ({len(exp_idx)}, {len(feature_cols)}))")

# ── Plot 1: Bar chart — mean |SHAP| ─────────────────────────────────────────
mean_abs = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs)[::-1]

# Clean feature names for display
DISPLAY = {
    "student_snow_pct": "Snow % (raw)",
    "snow_n_components": "Snow components",
    "snow_largest_ratio": "Largest component ratio",
    "snow_centroid_y": "Snow centroid Y",
    "ground_coverage": "Ground coverage",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "edge_density": "Edge density",
    "entropy": "Entropy",
    "white_frac": "White fraction",
    "low_sat_frac": "Low saturation frac",
    "blue_ratio": "Blue ratio",
    "bottom_brightness": "Bottom brightness",
    "bottom_white_frac": "Bottom white frac",
    "t2m": "Temperature (t2m)",
    "visibility": "Visibility",
    "weather_code": "Weather code",
    "hour_sin": "Hour (sin)",
    "hour_cos": "Hour (cos)",
    "doy_sin": "Day of year (sin)",
    "doy_cos": "Day of year (cos)",
}
names = [DISPLAY.get(f, f) for f in feature_cols]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([names[i] for i in order], mean_abs[order], color="#2196F3")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Bi-GRU Temporal Smoother — Feature Importance")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/shap_bar.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}/shap_bar.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}/shap_bar.pdf")

# ── Plot 2: Dependence plots for top 4 features ───────────────────────────────
top4 = order[:4]
X_exp_np = X[exp_idx]

# Inverse-normalize t2m for readable axis (×30)
raw_vals = {
    "t2m": X_exp_np[:, feature_cols.index("t2m")] * 30,
    "student_snow_pct": X_exp_np[:, feature_cols.index("student_snow_pct")] * 100,
    "visibility": np.expm1(X_exp_np[:, feature_cols.index("visibility")] * np.log1p(65000)),
    "weather_code": X_exp_np[:, feature_cols.index("weather_code")] * 100,
}
xlabels = {
    "t2m": "Temperature (°C)",
    "student_snow_pct": "Raw snow % (model input)",
    "visibility": "Visibility (m)",
    "weather_code": "WMO weather code",
}

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()

for ax, fi in zip(axes, top4):
    fname = feature_cols[fi]
    x_vals = raw_vals.get(fname, X_exp_np[:, fi])
    xlabel = xlabels.get(fname, DISPLAY.get(fname, fname))
    ax.scatter(x_vals, shap_values[:, fi], alpha=0.3, s=8, c="#E53935")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SHAP value")
    ax.set_title(DISPLAY.get(fname, fname))

plt.suptitle("GRU Smoother — SHAP Dependence Plots (Top 4 Features)", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_dependence.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}/shap_dependence.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}/shap_dependence.pdf")

print("\nTop 10 features by importance:")
for i in order[:10]:
    print(f"  {names[i]:<30} {mean_abs[i]:.4f}")
