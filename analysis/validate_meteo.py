#!/usr/bin/env python3
"""
Validate pipeline snow_pct against manual ARSO snow-presence annotations.
Reports: raw student_snow_pct, uni-GRU smoothed, bi-GRU smoothed.
Uses same GRU loading/preprocessing as make_snow_grid.py and viewer.
"""
import sys
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "models"))

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
    # Šmartno pri SG (OOD camera). Source: Smartno_manual_observation.xlsx,
    # daily year/month/day/snow_cover records (2024-10-01..2026-07-31, 669
    # days, 45 snow-covered), collapsed into runs of consecutive
    # snow-covered days. Unlike CF00001/CM00009/CM00014's per-observation
    # timestamps, this data has only daily resolution — intervals use a
    # fixed 07:00-19:00 daytime window per day, not real observation times.
    "CM00067": [
        ("202411220700", "202411261900"), ("202412230700", "202412311900"),
        ("202502140700", "202502151900"), ("202511200700", "202511201900"),
        ("202512240700", "202512261900"), ("202601070700", "202601271900"),
        ("202602200700", "202602221900"), ("202603270700", "202603271900"),
    ],
}

SMOOTHERS = {
    "Uni-GRU":  "models/temporal_snow/checkpoints/smoother_gru_h128_l2_k7_seq96_20260623_175404/best-epoch=37-val_mae=0.0066.ckpt",
    "Bi-GRU":   "models/temporal_snow/checkpoints/smoother_gru_h128_l2_bi_k7_seq96_20260623_175446/best-epoch=114-val_mae=0.0037.ckpt",
    "Uni-LSTM": "models/temporal_snow/checkpoints/smoother_lstm_h128_l2_k7_seq96_20260623_175408/best-epoch=29-val_mae=0.0069.ckpt",
    "Bi-LSTM":  "models/temporal_snow/checkpoints/smoother_lstm_h128_l2_bi_k7_seq96_20260623_175446/best-epoch=24-val_mae=0.0058.ckpt",
    # Retrained 2026-09-20 on the full 13-camera set (incl. CM00067) — the
    # 4 above predate CM00067 entirely and were trained on 9 cameras only.
    "Uni-GRU-v2": "models/temporal_snow/checkpoints/smoother_gru_h128_l2_k7_seq72_20260920_024134/best-epoch=23-val_mae=0.0052.ckpt",
}


def parse_ts(s):
    return datetime.strptime(s, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def label_intervals(dt_series, intervals):
    labels = np.zeros(len(dt_series), dtype=int)
    for start_s, end_s in intervals:
        start, end = parse_ts(start_s), parse_ts(end_s)
        mask = (dt_series >= start) & (dt_series <= end)
        labels[mask] = 1
    return labels


def load_module(ckpt_path):
    from temporal_snow.train_smoother import TemporalSmootherModule
    from temporal_snow.dataset import FeatureScaler
    module = TemporalSmootherModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    module.eval()
    # TemporalSmootherModule never stores the scaler on self — train_smoother.py
    # keeps it as a local var and only serializes it to scaler.json next to the
    # checkpoint. hasattr(module, "scaler") is therefore always False, silently
    # routing every checkpoint (scaled or not) through the legacy/unscaled
    # normalization path in run_smoother(). Load scaler.json explicitly here so
    # newer, standardized checkpoints actually get their scaler applied.
    scaler_path = Path(ckpt_path).parent / "scaler.json"
    module.scaler = FeatureScaler.load(scaler_path) if scaler_path.exists() else None
    # seq_len isn't a saved hparam either (see TemporalSmootherModule.__init__)
    # — parse it from the checkpoint dir name (..._seq72_... / ..._seq96_...),
    # matching the convention used throughout models/temporal_snow/*.py.
    m = re.search(r"_seq(\d+)", str(ckpt_path))
    module.seq_len = int(m.group(1)) if m else 72
    return module


def run_smoother(cam_df, module):
    """Run the smoother over one camera's timeseries.

    cam_df must already be the output of dataset.load_timeseries() filtered to
    one camera (see main()) — this function no longer hand-rebuilds features.
    Earlier versions duplicated load_timeseries()'s preprocessing here
    (cyclical time encoding, log-transforms, fill defaults) and drifted from
    it in multiple ways (hour_sin/cos derived from re-parsed datetime instead
    of the CSV's own `hour` column; visibility/snow_n_components missing their
    log1p transform; no scaler applied at all), each of which independently
    produced degenerate near-zero output. Delegating to load_timeseries()
    guarantees identical preprocessing to what the scaler was fit on and what
    gru_ood_snow_freq.py already verified against known-good numbers.
    """
    import torch
    from temporal_snow.dataset import build_feature_columns, SNOW_PCT_COLS

    df = cam_df.copy()
    snow_pct_cols = [c for c in SNOW_PCT_COLS if c in df.columns]
    if not snow_pct_cols:
        return np.zeros(len(df))

    df = df.sort_values("dt").reset_index(drop=True)

    hp = module.hparams
    has_scaler = hasattr(module, "scaler") and module.scaler is not None
    scale = 100.0

    feature_cols = (module.scaler.feature_cols if has_scaler
                     else build_feature_columns(snow_pct_cols, use_image_features=True))
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    x = df[feature_cols].fillna(0).values.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)
    if has_scaler:
        x = module.scaler.transform_array(x)

    # Sliding-window inference: the GRU was trained on seq_len windows (e.g.
    # 72 or 96 steps), not on whole-camera sequences (14,000+ steps for
    # CF00001). Feeding the full sequence in one shot previously produced
    # degenerate near-zero output on every camera, including in-distribution
    # ones the model was trained on — this matches the verified windowed
    # approach in temporal_snow/animate_periods.py::predict_timeseries and
    # gru_ood_snow_freq.py, both of which reproduce known-good IoU/frequency
    # numbers elsewhere in this project.
    seq_len = getattr(module, "seq_len", 72)
    n = len(x)
    device = torch.device("cpu")

    with torch.no_grad():
        if n < seq_len:
            pad = np.zeros((seq_len - n, x.shape[1]), dtype=np.float32)
            x_in = np.concatenate([pad, x], axis=0)
            x_t = torch.from_numpy(x_in).unsqueeze(0).to(device)
            out = module(x_t).squeeze().numpy()
            pred = out[-n:]
        else:
            starts = list(range(0, n - seq_len + 1))
            batch_size = 64
            all_preds = []
            for i in range(0, len(starts), batch_size):
                batch_starts = starts[i:i + batch_size]
                batch_x = np.stack([x[s:s + seq_len] for s in batch_starts])
                x_t = torch.from_numpy(batch_x).to(device)
                out = module(x_t).numpy()
                if out.ndim == 3:
                    out = out.squeeze(-1)
                all_preds.append(out)
            acc = np.zeros(n)
            cnt = np.zeros(n)
            all_preds = np.concatenate(all_preds, axis=0)
            for idx, start in enumerate(starts):
                acc[start:start + seq_len] += all_preds[idx]
                cnt[start:start + seq_len] += 1
            pred = acc / np.maximum(cnt, 1)

    pred_pct = np.clip(pred * scale, 0, 100)
    print(f"    output range: [{pred_pct.min():.1f}, {pred_pct.max():.1f}] mean={pred_pct.mean():.1f}")
    return pred_pct


def evaluate(labels, scores, name, threshold=None):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, accuracy_score, confusion_matrix)
    scores_norm = scores / 100.0 if scores.max() > 1.0 else scores
    if threshold is None:
        best_f1, best_t = 0, 0.5
        for t in np.linspace(scores_norm.min(), scores_norm.max(), 200):
            p = (scores_norm >= t).astype(int)
            f = f1_score(labels, p, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        threshold = best_t
    pred = (scores_norm >= threshold).astype(int)
    auc = roc_auc_score(labels, scores_norm)
    ap  = average_precision_score(labels, scores_norm)
    acc = accuracy_score(labels, pred)
    f1  = f1_score(labels, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    print(f"  [{name}@t={threshold:.2f}] AUC-ROC={auc:.3f}  AUC-PR={ap:.3f}  Acc={acc:.3f}  F1={f1:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")
    return dict(auc=auc, ap=ap, acc=acc, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), threshold=threshold)


def main():
    import argparse
    from temporal_snow.dataset import load_timeseries

    ap = argparse.ArgumentParser()
    ap.add_argument("--window-only", action="store_true",
                     help="Restrict each camera to its labeled-interval date span "
                          "instead of the full QA-usable history.")
    ap.add_argument("--daily", action="store_true",
                     help="Aggregate per-frame smoother output and interval labels "
                          "to one row per camera-day (max, i.e. any positive frame "
                          "that day makes the day positive) before scoring, instead "
                          "of scoring every frame independently.")
    args = ap.parse_args()

    ts_path = Path(__file__).parent / "snow_timeseries.csv"
    # qa_filter=True drops QA-rejected rows outright, matching this script's
    # previous behavior (df["qa_usable"] == 1). load_timeseries() also builds
    # hour_sin/cos from the CSV's own `hour` column (not a re-parsed
    # datetime), applies the log1p transform to visibility/snow_n_components,
    # and fills defaults — all identically to what train_smoother.py fit the
    # scaler on. See run_smoother()'s docstring for why this matters.
    df, _snow_pct_cols = load_timeseries(str(ts_path), require_target=False, qa_filter=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    cameras = list(SNOW_INTERVALS.keys())
    df = df[df["camera"].isin(cameras)].copy()
    df = df.sort_values(["camera", "datetime"]).reset_index(drop=True)
    print(f"Loaded {len(df)} QA-usable frames\n")

    print("Loading smoother modules...")
    modules = {}
    for name, ckpt in SMOOTHERS.items():
        print(f"  {name}...")
        modules[name] = load_module(ckpt)
    print("Done.\n")

    # Run all smoothers per camera
    for name in SMOOTHERS:
        df[f"pct_{name}"] = 0.0
    for cam in cameras:
        mask = df["camera"] == cam
        cam_df = df[mask].copy()
        for name, module in modules.items():
            print(f"  {name} on {cam}...")
            df.loc[mask, f"pct_{name}"] = run_smoother(cam_df, module)

    from sklearn.metrics import f1_score as f1_fn
    def best_pred(labels, scores):
        sn = scores / 100.0 if scores.max() > 1.0 else scores
        best_f1, best_t = 0, 0.5
        for t in np.linspace(sn.min(), sn.max(), 200):
            p = (sn >= t).astype(int)
            f = f1_fn(labels, p, zero_division=0)
            if f > best_f1: best_f1, best_t = f, t
        return (sn >= best_t).astype(int)

    all_labels = []
    all_preds = {name: [] for name in ["Raw"] + list(SMOOTHERS.keys())}

    for cam in cameras:
        cam_df = df[df["camera"] == cam]
        if args.window_only:
            # Restrict to each camera's labeled-interval date span (min start
            # to max end across its SNOW_INTERVALS tuples), not the full
            # multi-year QA-usable history. Approximates the original,
            # smaller evaluation scope ("3,042 frames" for 3 stations) whose
            # exact derivation is no longer known with certainty.
            starts = [parse_ts(s) for s, _ in SNOW_INTERVALS[cam]]
            ends = [parse_ts(e) for _, e in SNOW_INTERVALS[cam]]
            win_start, win_end = min(starts), max(ends)
            cam_df = cam_df[(cam_df["datetime"] >= win_start) & (cam_df["datetime"] <= win_end)]
        labels = label_intervals(cam_df["datetime"], SNOW_INTERVALS[cam])

        if args.daily:
            # Collapse to one row per camera-day: a day is labeled positive if
            # ANY frame that day falls in a labeled snow interval, and each
            # smoother's daily score is its per-frame max that day — the same
            # "max = daily peak" convention dataset.aggregate_daily() uses.
            # Approximates a plausible reading of the original, much smaller
            # evaluation scope ("3,042 frames" for 3 stations): daily
            # conclusions from frame-level evidence rather than every frame
            # scored independently (confirmed close: 3,360 camera-days for
            # the same 3 stations here, vs. 3,042 originally — exact
            # derivation still uncertain, but daily granularity is the right
            # ballpark, per-frame scale is not).
            daily_df = cam_df.copy()
            daily_df["_label"] = labels
            daily_df["_date"] = daily_df["dt"].dt.date
            agg = {"_label": "max", "student_snow_pct": "max"}
            for name in SMOOTHERS:
                agg[f"pct_{name}"] = "max"
            cam_df = daily_df.groupby("_date").agg(agg).reset_index()
            labels = cam_df["_label"].values.astype(int)

        n_snow = labels.sum()
        print(f"\n── {cam} ({len(cam_df)} frames, {n_snow} snow) ──")
        if n_snow == 0 or n_snow == len(labels):
            print("  Skipping (single class)"); continue

        evaluate(labels, cam_df["student_snow_pct"].values, "Raw")
        for name in SMOOTHERS:
            evaluate(labels, cam_df[f"pct_{name}"].values, name)

        all_labels.extend(labels.tolist())
        all_preds["Raw"].extend(best_pred(labels, cam_df["student_snow_pct"].values).tolist())
        for name in SMOOTHERS:
            all_preds[name].extend(best_pred(labels, cam_df[f"pct_{name}"].values).tolist())

    print("\n══ Overall (per-camera optimal threshold) ══")
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    L = np.array(all_labels)
    for name, preds in all_preds.items():
        P = np.array(preds)
        acc = accuracy_score(L, P)
        f1  = f1_score(L, P, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(L, P).ravel()
        print(f"  [{name}] Acc={acc:.3f}  F1={f1:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")


if __name__ == "__main__":
    main()
