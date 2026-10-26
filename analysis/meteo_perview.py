#!/usr/bin/env python3
"""
Recompute the four-station meteorological validation (tab:meteo) on the
per-view, grey-filtered series.

Two variants are printed:
  pooled   — one series per camera, every QA-usable frame (the basis of the
             numbers currently in the paper)
  perview  — CM00067 split into its eight views, each smoothed separately,
             grey (IR/dusk) frames dropped before smoothing

Smoothers: the four 13-camera seq72 runs of 2026-10-12.
"""
import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

THESIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS / "models"))

CKPT_DIR = THESIS / "models" / "temporal_snow" / "checkpoints"

# Each entry: (training run directory, flat name in the Zenodo deposit).
# The training layout ships with the run directory; the deposit ships the flat
# checkpoint alone, so either can be used.
SMOOTHERS = {
    "Uni-GRU": ("smoother_gru_h128_l2_k7_seq72_20261012_023822", "models/smoothers/smoother_unigru_idood.ckpt"),
    "Bi-GRU": ("smoother_gru_h128_l2_bi_k7_seq72_20261012_024146", "models/smoothers/smoother_bigru_idood.ckpt"),
    "Uni-LSTM": ("smoother_lstm_h128_l2_k7_seq72_20261012_025346", "models/smoothers/smoother_unilstm_idood.ckpt"),
    "Bi-LSTM": ("smoother_lstm_h128_l2_bi_k7_seq72_20261012_025525", "models/smoothers/smoother_bilstm_idood.ckpt"),
}


def resolve_ckpt(run_dir, flat_name=None):
    if isinstance(run_dir, Path):
        if run_dir.exists():
            return run_dir
        raise FileNotFoundError(f"no checkpoint at {run_dir}")
    d = CKPT_DIR / run_dir
    if d.is_dir():
        cand = sorted(d.glob("best-epoch=*.ckpt"),
                      key=lambda x: float(x.stem.split("val_mae=")[-1]))
        if cand:
            return cand[0]
    flat = THESIS / flat_name
    if flat.exists():
        return flat
    raise FileNotFoundError(
        f"no checkpoint for {run_dir}: looked in {d}/best-epoch=*.ckpt and {flat}"
    )


SMOOTHERS_9CAM = {
    "Uni-GRU-9": CKPT_DIR / "smoother_gru_h128_l2_k7_seq96_20260623_175404/best-epoch=25-val_mae=0.0066.ckpt",
    "Bi-GRU-9": CKPT_DIR / "smoother_gru_h128_l2_bi_k7_seq96_20260623_175446/best-epoch=114-val_mae=0.0037.ckpt",
    "Uni-LSTM-9": CKPT_DIR / "smoother_lstm_h128_l2_k7_seq96_20260623_175408/best-epoch=25-val_mae=0.0069.ckpt",
    "Bi-LSTM-9": CKPT_DIR / "smoother_lstm_h128_l2_bi_k7_seq96_20260623_175446/best-epoch=24-val_mae=0.0058.ckpt",
}

SNOW_INTERVALS = {
    "CF00001": [
        ("202212111530", "202212121100"), ("202212170700", "202212171000"),
        ("202301160700", "202302021630"), ("202302080830", "202302081000"),
        ("202302260700", "202302270900"), ("202311300700", "202311300900"),
        ("202312030700", "202312051530"), ("202312090700", "202312090900"),
        ("202401190700", "202402051130"), ("202404161600", "202404161730"),
        ("202411220700", "202411241000"), ("202412140700", "202412140900"),
        ("202412230700", "202501021000"), ("202502140700", "202502141400"),
        ("202511180700", "202511180900"),
    ],
    "CM00009": [
        ("202212020600", "202212020900"), ("202212101800", "202212171100"),
        ("202301160600", "202302101600"), ("202312090600", "202312090900"),
        ("202401200700", "202402071300"), ("202404161700", "202404170700"),
        ("202411220600", "202411251600"), ("202412222200", "202412241500"),
        ("202502140600", "202502181000"),
    ],
    "CM00014": [
        ("202212110600", "202212181000"), ("202301160700", "202302111600"),
        ("202312060600", "202312070900"), ("202401190800", "202402021600"),
        ("202411220600", "202411241200"), ("202412140700", "202412140800"),
        ("202412230600", "202412231500"), ("202502140600", "202502141200"),
        ("202511230600", "202511230800"),
    ],
    "CM00067": [
        ("202411220700", "202411261900"), ("202412230700", "202412311900"),
        ("202502140700", "202502151900"), ("202511200700", "202511201900"),
        ("202512240700", "202512261900"), ("202601070700", "202601271900"),
        ("202602200700", "202602221900"), ("202603270700", "202603271900"),
    ],
}

STATION_NAME = {
    "CF00001": "Ljubljana Bežigrad",
    "CM00009": "Ljubljana Airport",
    "CM00014": "Celje-Medlog",
    "CM00067": "Šmartno pri SG",
}

FILL_DEFAULTS = {
    "t2m": 0, "visibility": 1000, "weather_code": 0,
    "brightness": 128, "contrast": 50, "edge_density": 10, "entropy": 4,
    "white_frac": 0, "low_sat_frac": 0, "blue_ratio": 0,
    "bottom_white_frac": 0, "bottom_brightness": 128,
    "snow_n_components": 0, "snow_largest_ratio": 0,
    "snow_centroid_y": 0.5, "ground_coverage": 0,
}


def parse_ts(s):
    return datetime.strptime(s, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def label_intervals(dt_series, intervals):
    labels = np.zeros(len(dt_series), dtype=int)
    for start_s, end_s in intervals:
        mask = (dt_series >= parse_ts(start_s)) & (dt_series <= parse_ts(end_s))
        labels[mask.values] = 1
    return labels


def load_module(path):
    from temporal_snow.train_smoother import TemporalSmootherModule
    from temporal_snow.dataset import FeatureScaler
    module = TemporalSmootherModule.load_from_checkpoint(str(path), map_location="cpu")
    module.eval()
    sc = Path(path).parent / "scaler.json"
    if sc.exists():
        module.scaler = FeatureScaler.load(sc)
    else:
        print(f"  WARNING: no scaler next to {Path(path).name}")
    return module


def run_smoother(cam_df, module):
    import torch
    from temporal_snow.dataset import build_feature_columns, SNOW_PCT_COLS

    df = cam_df.copy()
    snow_pct_cols = [c for c in SNOW_PCT_COLS if c in df.columns]
    if not snow_pct_cols:
        return np.zeros(len(df))

    df["dt"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("dt").reset_index(drop=True)
    df["hour_sin"] = np.sin(2 * math.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * math.pi * df["datetime"].dt.hour / 24)
    doy = df["dt"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * math.pi * doy / 365)

    for col in snow_pct_cols:
        df[col] = df[col].fillna(0)
    for col, default in FILL_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    if "visibility" in df.columns:
        df["visibility"] = np.log1p(df["visibility"])
    if "snow_n_components" in df.columns:
        df["snow_n_components"] = np.log1p(df["snow_n_components"])

    feature_cols = build_feature_columns(snow_pct_cols, use_image_features=True)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    x = np.nan_to_num(df[feature_cols].values.astype(np.float32), nan=0.0)
    if getattr(module, "scaler", None) is not None:
        x = module.scaler.transform_array(x)

    with torch.no_grad():
        pred = module(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
    return np.clip(pred * 100.0, 0, 100)


def daily_frame(df, score_cols):
    """Collapse frames to camera-days: label = any frame in an interval,
    score = maximum frame score that day (per convention in the paper)."""
    d = df.copy()
    d["_date"] = pd.to_datetime(d["datetime"]).dt.date
    agg = {"_label": ("_label", "max")}
    for c in score_cols:
        agg[c] = (c, "max")
    return d.groupby("_date").agg(**agg).reset_index()


def evaluate(daily, score_col, name, out):
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

    labels = daily["_label"].values
    scores = daily[score_col].values / 100.0
    pred = (scores >= 0.5).astype(int)
    if len(np.unique(labels)) < 2:
        out[name] = None
        return
    auc = roc_auc_score(labels, scores)
    f1 = f1_score(labels, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    out[name] = dict(auc=auc, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), n=len(labels))


def run_variant(df, per_view, drop_gray, label, sm= None):
    sm = sm or SMOOTHERS
    modules = {k: load_module(resolve_ckpt(*v)) for k, v in sm.items()}
    results = {}
    overall_labels, overall_preds = [], {k: [] for k in ["Raw"] + list(sm)}

    for cam, cam_all in df.groupby("camera"):
        series = {}
        if per_view and cam == "CM00067":
            views = cam_all["stem"].str.rsplit("_", n=1).str[-1]
            for v, sub in cam_all.groupby(views):
                series[f"{cam}:{v}"] = sub
        else:
            series[cam] = cam_all

        frames = []
        for _, sub in series.items():
            sub = sub.copy()
            sub["_label"] = label_intervals(sub["datetime"], SNOW_INTERVALS[cam])
            sub["pct_Raw"] = sub["student_snow_pct"].values
            for name, mod in modules.items():
                sub[f"pct_{name}"] = run_smoother(sub, mod)
            frames.append(sub)

        cam_frames = pd.concat(frames, ignore_index=True)
        # camera-day score is the max over every view of that day
        daily = daily_frame(cam_frames, [c for c in cam_frames.columns if c.startswith("pct_")])
        res = {}
        for name in ["Raw"] + list(sm):
            evaluate(daily, f"pct_{name}", name, res)
        results[cam] = res
        print(f"  {cam} ({STATION_NAME[cam]}): {len(daily)} camera-days, {int(daily['_label'].sum())} snow")
        for name, r in res.items():
            if r:
                print(f"    {name:9s} AUC {r['auc']:.3f}  F1 {r['f1']:.3f}  TP {r['tp']} FP {r['fp']} FN {r['fn']}")

        overall_labels.extend(daily["_label"].tolist())
        for name in ["Raw"] + list(sm):
            overall_preds[name].extend((daily[f"pct_{name}"].values / 100.0 >= 0.5).astype(int).tolist())

    print(f"\n  OVERALL ({label})")
    from sklearn.metrics import f1_score, confusion_matrix
    L = np.array(overall_labels)
    summary = {}
    for name, preds in overall_preds.items():
        P = np.array(preds)
        tn, fp, fn, tp = confusion_matrix(L, P).ravel()
        macro = np.mean([r[name]["f1"] for r in results.values() if r.get(name)])
        summary[name] = dict(f1=f1_score(L, P, zero_division=0), macro=macro,
                             tp=int(tp), fp=int(fp), fn=int(fn), n=len(L))
        print(f"    {name:9s} micro-F1 {summary[name]['f1']:.3f}  macro-F1 {macro:.3f}  FP {fp}")
    return results, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(THESIS / "analysis" / "snow_timeseries.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[df["camera"].isin(SNOW_INTERVALS)]
    df = df[df["qa_usable"] == 1]
    print(f"Loaded {len(df)} QA-usable frames across {df['camera'].nunique()} stations\n")

    gray = (df["r_mean"] - df["g_mean"]).abs() < 1
    print(f"greyscale frames: {int(gray.sum())} of {len(df)} ({gray.mean()*100:.1f}%)")

    print("\n=== variant A: pooled, all frames (paper's current basis) ===")
    run_variant(df, per_view=False, drop_gray=False, label="pooled, all frames")

    print("\n=== variant B: per-view CM00067, grey frames dropped ===")
    run_variant(df[~gray], per_view=True, drop_gray=True, label="per-view, grey dropped")

    print("\n=== variant C: nine-camera smoothers on the per-view, grey-dropped series ===")
    run_variant(df[~gray], per_view=True, drop_gray=True, label="9-cam smoothers", sm=SMOOTHERS_9CAM)


if __name__ == "__main__":
    main()
