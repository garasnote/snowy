#!/usr/bin/env python3
"""
Temporal-pipeline diagnostics — everything that validates or inspects the
GRU/LSTM smoother and its input timeseries (analysis/snow_timeseries.csv).

Consolidated from validate_meteo.py, temporal_analysis.py, make_snow_grid.py,
check_daily_agg.py. Each keeps its original CLI flags, now behind a
subcommand:

    python analysis/temporal_diagnostics.py meteo-validate
    python analysis/temporal_diagnostics.py meteo-validate --window-only --daily
    python analysis/temporal_diagnostics.py melt-patterns
    python analysis/temporal_diagnostics.py melt-patterns --cameras CF00001
    python analysis/temporal_diagnostics.py grid
    python analysis/temporal_diagnostics.py grid --cameras CM00108 --quantiles 0 50 100
    python analysis/temporal_diagnostics.py daily-agg-check

Run `python analysis/temporal_diagnostics.py <subcommand> --help` for the
full option list. meteo-validate produces the paper's headline ARSO
validation number (F1); the others are supporting diagnostics.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "models"))

MASTER = _ROOT


# =============================================================================
# METEO-VALIDATE — pipeline snow_pct vs manual ARSO snow-presence annotations
# =============================================================================

from datetime import datetime, timezone

METEO_SNOW_INTERVALS = {
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

METEO_SMOOTHERS = {
    "Uni-GRU":  "models/temporal_snow/checkpoints/smoother_gru_h128_l2_k7_seq96_20260623_175404/best-epoch=37-val_mae=0.0066.ckpt",
    "Bi-GRU":   "models/temporal_snow/checkpoints/smoother_gru_h128_l2_bi_k7_seq96_20260623_175446/best-epoch=114-val_mae=0.0037.ckpt",
    "Uni-LSTM": "models/temporal_snow/checkpoints/smoother_lstm_h128_l2_k7_seq96_20260623_175408/best-epoch=29-val_mae=0.0069.ckpt",
    "Bi-LSTM":  "models/temporal_snow/checkpoints/smoother_lstm_h128_l2_bi_k7_seq96_20260623_175446/best-epoch=24-val_mae=0.0058.ckpt",
    # Retrained 2026-09-20 on the full 13-camera set (incl. CM00067) — the
    # 4 above predate CM00067 entirely and were trained on 9 cameras only.
    "Uni-GRU-v2": "models/temporal_snow/checkpoints/smoother_gru_h128_l2_k7_seq72_20260920_024134/best-epoch=23-val_mae=0.0052.ckpt",
}


def meteo_parse_ts(s):
    return datetime.strptime(s, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def meteo_label_intervals(dt_series, intervals):
    labels = np.zeros(len(dt_series), dtype=int)
    for start_s, end_s in intervals:
        start, end = meteo_parse_ts(start_s), meteo_parse_ts(end_s)
        mask = (dt_series >= start) & (dt_series <= end)
        labels[mask] = 1
    return labels


def meteo_load_module(ckpt_path):
    from temporal_snow.train_smoother import TemporalSmootherModule
    from temporal_snow.dataset import FeatureScaler
    module = TemporalSmootherModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    module.eval()
    # TemporalSmootherModule never stores the scaler on self — train_smoother.py
    # keeps it as a local var and only serializes it to scaler.json next to the
    # checkpoint. hasattr(module, "scaler") is therefore always False, silently
    # routing every checkpoint (scaled or not) through the legacy/unscaled
    # normalization path in meteo_run_smoother(). Load scaler.json explicitly
    # here so newer, standardized checkpoints actually get their scaler applied.
    scaler_path = Path(ckpt_path).parent / "scaler.json"
    module.scaler = FeatureScaler.load(scaler_path) if scaler_path.exists() else None
    # seq_len isn't a saved hparam either (see TemporalSmootherModule.__init__)
    # — parse it from the checkpoint dir name (..._seq72_... / ..._seq96_...),
    # matching the convention used throughout models/temporal_snow/*.py.
    m = re.search(r"_seq(\d+)", str(ckpt_path))
    module.seq_len = int(m.group(1)) if m else 72
    return module


def meteo_run_smoother(cam_df, module):
    """Run the smoother over one camera's timeseries.

    cam_df must already be the output of dataset.load_timeseries() filtered to
    one camera (see run_meteo_validate()) — this function no longer
    hand-rebuilds features. Earlier versions duplicated load_timeseries()'s
    preprocessing here (cyclical time encoding, log-transforms, fill
    defaults) and drifted from it in multiple ways (hour_sin/cos derived from
    re-parsed datetime instead of the CSV's own `hour` column;
    visibility/snow_n_components missing their log1p transform; no scaler
    applied at all), each of which independently produced degenerate
    near-zero output. Delegating to load_timeseries() guarantees identical
    preprocessing to what the scaler was fit on and what
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


def meteo_evaluate(labels, scores, name, threshold=None):
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


def add_meteo_validate_subparser(subparsers):
    p = subparsers.add_parser("meteo-validate",
                               help="Validate pipeline snow_pct against ARSO snow-presence annotations")
    p.add_argument("--window-only", action="store_true",
                    help="Restrict each camera to its labeled-interval date span "
                         "instead of the full QA-usable history.")
    p.add_argument("--daily", action="store_true",
                    help="Aggregate per-frame smoother output and interval labels "
                         "to one row per camera-day (max, i.e. any positive frame "
                         "that day makes the day positive) before scoring, instead "
                         "of scoring every frame independently.")
    return p


def run_meteo_validate(args):
    from temporal_snow.dataset import load_timeseries

    ts_path = _HERE / "snow_timeseries.csv"
    # qa_filter=True drops QA-rejected rows outright, matching this script's
    # previous behavior (df["qa_usable"] == 1). load_timeseries() also builds
    # hour_sin/cos from the CSV's own `hour` column (not a re-parsed
    # datetime), applies the log1p transform to visibility/snow_n_components,
    # and fills defaults — all identically to what train_smoother.py fit the
    # scaler on. See meteo_run_smoother()'s docstring for why this matters.
    df, _snow_pct_cols = load_timeseries(str(ts_path), require_target=False, qa_filter=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    cameras = list(METEO_SNOW_INTERVALS.keys())
    df = df[df["camera"].isin(cameras)].copy()
    df = df.sort_values(["camera", "datetime"]).reset_index(drop=True)
    print(f"Loaded {len(df)} QA-usable frames\n")

    print("Loading smoother modules...")
    modules = {}
    for name, ckpt in METEO_SMOOTHERS.items():
        print(f"  {name}...")
        modules[name] = meteo_load_module(ckpt)
    print("Done.\n")

    for name in METEO_SMOOTHERS:
        df[f"pct_{name}"] = 0.0
    for cam in cameras:
        mask = df["camera"] == cam
        cam_df = df[mask].copy()
        for name, module in modules.items():
            print(f"  {name} on {cam}...")
            df.loc[mask, f"pct_{name}"] = meteo_run_smoother(cam_df, module)

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
    all_preds = {name: [] for name in ["Raw"] + list(METEO_SMOOTHERS.keys())}

    for cam in cameras:
        cam_df = df[df["camera"] == cam]
        if args.window_only:
            # Restrict to each camera's labeled-interval date span (min start
            # to max end across its METEO_SNOW_INTERVALS tuples), not the full
            # multi-year QA-usable history. Approximates the original,
            # smaller evaluation scope ("3,042 frames" for 3 stations) whose
            # exact derivation is no longer known with certainty.
            starts = [meteo_parse_ts(s) for s, _ in METEO_SNOW_INTERVALS[cam]]
            ends = [meteo_parse_ts(e) for _, e in METEO_SNOW_INTERVALS[cam]]
            win_start, win_end = min(starts), max(ends)
            cam_df = cam_df[(cam_df["datetime"] >= win_start) & (cam_df["datetime"] <= win_end)]
        labels = meteo_label_intervals(cam_df["datetime"], METEO_SNOW_INTERVALS[cam])

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
            for name in METEO_SMOOTHERS:
                agg[f"pct_{name}"] = "max"
            cam_df = daily_df.groupby("_date").agg(agg).reset_index()
            labels = cam_df["_label"].values.astype(int)

        n_snow = labels.sum()
        print(f"\n── {cam} ({len(cam_df)} frames, {n_snow} snow) ──")
        if n_snow == 0 or n_snow == len(labels):
            print("  Skipping (single class)"); continue

        meteo_evaluate(labels, cam_df["student_snow_pct"].values, "Raw")
        for name in METEO_SMOOTHERS:
            meteo_evaluate(labels, cam_df[f"pct_{name}"].values, name)

        all_labels.extend(labels.tolist())
        all_preds["Raw"].extend(best_pred(labels, cam_df["student_snow_pct"].values).tolist())
        for name in METEO_SMOOTHERS:
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


# =============================================================================
# MELT-PATTERNS — melt sequences, transition analysis, morning/evening, etc.
# =============================================================================

MELT_TIMESERIES_CSV = MASTER / "analysis" / "snow_timeseries.csv"
MELT_FIGURES_DIR = MASTER / "analysis" / "figures" / "temporal"


def melt_plot_snow_timeseries(df_cam, camera, method, out_dir):
    """Plot snow_pct over time for one camera, one method, overlaid with temperature."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    col = f"{method}_snow_pct"
    if col not in df_cam.columns:
        return
    valid = df_cam.dropna(subset=[col]).copy()
    if len(valid) < 5:
        return

    valid["dt"] = pd.to_datetime(valid["datetime"])

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(valid["dt"], valid[col], "b-", alpha=0.6, linewidth=0.5, label=f"{method} snow %")
    if len(valid) > 20:
        rolling = valid.set_index("dt")[col].rolling("7D").mean()
        ax1.plot(rolling.index, rolling.values, "b-", linewidth=1.5, label="7-day avg")
    ax1.set_ylabel("Snow %", color="blue")
    ax1.set_ylim(-2, 102)
    ax1.tick_params(axis="y", labelcolor="blue")

    if "t2m" in valid.columns:
        t2m = valid.dropna(subset=["t2m"])
        if len(t2m) > 0:
            ax2 = ax1.twinx()
            ax2.plot(t2m["dt"], t2m["t2m"], "r-", alpha=0.3, linewidth=0.5)
            ax2.axhline(0, color="red", linestyle="--", alpha=0.3)
            ax2.set_ylabel("Temperature (C)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_title(f"{camera} — {method} snow coverage over time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax1.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / f"timeseries_{camera}_{method}.png", dpi=150)
    plt.close(fig)


def melt_analyze_transitions(df, method, min_snow_pct=1.0):
    """Analyze day-to-day transitions. Flag 'impossible' ones."""
    col = f"{method}_snow_pct"
    if col not in df.columns:
        return None

    results = []
    for camera, cam_df in df.groupby("camera"):
        cam_df = cam_df.dropna(subset=[col]).sort_values("datetime").copy()
        cam_df["dt"] = pd.to_datetime(cam_df["datetime"])

        for i in range(1, len(cam_df)):
            prev = cam_df.iloc[i - 1]
            curr = cam_df.iloc[i]
            dt_diff = (curr["dt"] - prev["dt"]).total_seconds() / 3600

            if dt_diff > 24 or dt_diff < 0.1:
                continue

            prev_pct = prev[col]
            curr_pct = curr[col]
            delta = curr_pct - prev_pct

            prev_hour = prev["dt"].hour
            curr_hour = curr["dt"].hour

            is_overnight = prev_hour >= 18 and curr_hour <= 8
            is_suspicious = False
            reason = ""

            if is_overnight and delta < -20:
                is_suspicious = True
                reason = "overnight_melt"
            elif not is_overnight and delta > 40 and prev_pct < 5:
                is_suspicious = True
                reason = "sudden_appearance"

            if abs(delta) > 5:
                results.append({
                    "camera": camera,
                    "from_stem": prev["stem"],
                    "to_stem": curr["stem"],
                    "from_dt": prev["dt"],
                    "to_dt": curr["dt"],
                    "hours_gap": round(dt_diff, 1),
                    "from_pct": round(prev_pct, 1),
                    "to_pct": round(curr_pct, 1),
                    "delta": round(delta, 1),
                    "is_overnight": is_overnight,
                    "suspicious": is_suspicious,
                    "reason": reason,
                })

    return pd.DataFrame(results) if results else None


def melt_morning_evening_asymmetry(df, method):
    """Compare morning and evening snow_pct within same day."""
    col = f"{method}_snow_pct"
    if col not in df.columns:
        return None

    df = df.dropna(subset=[col]).copy()
    df["dt"] = pd.to_datetime(df["datetime"])
    df["date_str"] = df["dt"].dt.strftime("%Y-%m-%d")

    morning = df[(df["hour"] >= 6) & (df["hour"] <= 10)].groupby(
        ["camera", "date_str"])[col].mean().reset_index()
    morning.columns = ["camera", "date", "morning_pct"]

    evening = df[(df["hour"] >= 16) & (df["hour"] <= 20)].groupby(
        ["camera", "date_str"])[col].mean().reset_index()
    evening.columns = ["camera", "date", "evening_pct"]

    merged = morning.merge(evening, on=["camera", "date"])
    if len(merged) == 0:
        return None

    merged["delta"] = merged["evening_pct"] - merged["morning_pct"]
    return merged


def melt_summer_false_positives(df, method):
    """Analyze predicted snow % when meteo says no snow."""
    col = f"{method}_snow_pct"
    if col not in df.columns or "snow_present" not in df.columns:
        return None

    no_snow = df[df["snow_present"] == False].dropna(subset=[col])
    if len(no_snow) == 0:
        return None

    return {
        "method": method,
        "n_images": len(no_snow),
        "mean_pred_pct": round(no_snow[col].mean(), 2),
        "median_pred_pct": round(no_snow[col].median(), 2),
        "max_pred_pct": round(no_snow[col].max(), 2),
        "pct_above_1": round((no_snow[col] > 1).mean() * 100, 1),
        "pct_above_5": round((no_snow[col] > 5).mean() * 100, 1),
        "pct_above_10": round((no_snow[col] > 10).mean() * 100, 1),
    }


def add_melt_patterns_subparser(subparsers):
    p = subparsers.add_parser("melt-patterns", help="Melt sequences, transitions, morning/evening asymmetry")
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument("--method", type=str, default=None,
                    help="Single method for detailed analysis (default: all)")
    return p


def run_melt_patterns(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if not MELT_TIMESERIES_CSV.exists():
        print(f"ERROR: {MELT_TIMESERIES_CSV} not found. Run "
              f"'python analysis/build_features.py timeseries' first.")
        return

    df = pd.read_csv(MELT_TIMESERIES_CSV)
    print(f"Loaded: {len(df)} rows, {df['camera'].nunique()} cameras")

    if args.cameras:
        df = df[df["camera"].isin(args.cameras)]
        print(f"Filtered to cameras: {args.cameras}")

    methods = [c.replace("_snow_pct", "")
               for c in df.columns if c.endswith("_snow_pct")]
    if args.method:
        methods = [m for m in methods if m == args.method]
    print(f"Methods: {methods}\n")

    MELT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Generating time series plots ===")
    cameras = sorted(df["camera"].unique())
    for cam in cameras:
        cam_df = df[df["camera"] == cam]
        for method in methods:
            melt_plot_snow_timeseries(cam_df, cam, method, MELT_FIGURES_DIR)
    print(f"  Saved to {MELT_FIGURES_DIR}/\n")

    print("=== Transition Analysis ===")
    for method in methods:
        trans = melt_analyze_transitions(df, method)
        if trans is not None and len(trans) > 0:
            suspicious = trans[trans["suspicious"]]
            print(f"\n  {method}: {len(trans)} transitions with |delta| > 5%")
            if len(suspicious) > 0:
                print(f"  {len(suspicious)} suspicious transitions:")
                for _, r in suspicious.head(10).iterrows():
                    print(f"    {r['camera']} {r['from_dt']} -> {r['to_dt']}: "
                          f"{r['from_pct']}% -> {r['to_pct']}% "
                          f"({r['reason']}, {r['hours_gap']}h gap)")
            trans.to_csv(MELT_FIGURES_DIR / f"transitions_{method}.csv", index=False)
    print()

    print("=== Morning vs Evening Asymmetry ===")
    for method in methods:
        asym = melt_morning_evening_asymmetry(df, method)
        if asym is not None and len(asym) > 0:
            mean_delta = asym["delta"].mean()
            melt_days = (asym["delta"] < -5).sum()
            stable = (asym["delta"].abs() < 2).sum()
            print(f"  {method}: {len(asym)} day-pairs, mean delta={mean_delta:.2f}%")
            print(f"    Melt days (evening < morning by >5%): {melt_days}")
            print(f"    Stable days (|delta| < 2%): {stable}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(asym["delta"], bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(0, color="red", linestyle="--")
            ax.set_xlabel("Evening snow % - Morning snow %")
            ax.set_ylabel("Count")
            ax.set_title(f"{method}: Morning→Evening Snow Change Distribution")
            plt.tight_layout()
            fig.savefig(MELT_FIGURES_DIR / f"morning_evening_{method}.png", dpi=150)
            plt.close(fig)
    print()

    print("=== Summer False Positive Analysis ===")
    fp_rows = []
    for method in methods:
        fp = melt_summer_false_positives(df, method)
        if fp:
            fp_rows.append(fp)
            print(f"  {method}: mean pred {fp['mean_pred_pct']}% when no snow, "
                  f"{fp['pct_above_5']}% of images have pred > 5%")
    if fp_rows:
        fp_df = pd.DataFrame(fp_rows)
        print(f"\n{fp_df.to_string(index=False)}")
        fp_df.to_csv(MELT_FIGURES_DIR / "summer_false_positives.csv", index=False)
    print()

    if len(methods) >= 2:
        print("=== Multi-method Agreement Plot ===")
        for cam in cameras:
            cam_df = df[df["camera"] == cam].copy()
            cam_df["dt"] = pd.to_datetime(cam_df["datetime"])
            cam_df = cam_df.sort_values("dt")

            fig, ax = plt.subplots(figsize=(14, 4))
            for method in methods:
                col = f"{method}_snow_pct"
                if col in cam_df.columns:
                    valid = cam_df.dropna(subset=[col])
                    if len(valid) > 20:
                        rolling = valid.set_index("dt")[col].rolling("7D").mean()
                        ax.plot(rolling.index, rolling.values, linewidth=1.2, label=method)
            ax.set_ylabel("Snow % (7-day avg)")
            ax.set_title(f"{cam} — All methods snow coverage")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)
            ax.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(MELT_FIGURES_DIR / f"multi_method_{cam}.png", dpi=150)
            plt.close(fig)
        print(f"  Saved to {MELT_FIGURES_DIR}/\n")

    print("Temporal analysis complete!")


# =============================================================================
# GRID — snow coverage comparison grid at N quantiles per camera
# =============================================================================

from paths import DATA_DIR as GRID_DATA_DIR

GRID_SEG_DIR = MASTER / "preprocessing/segmentation"
GRID_GROUND_DIR = GRID_SEG_DIR / "student_ground_output/mobilenet_majority3"
GRID_SNOW_DIR = GRID_SEG_DIR / "student_snow_output/mobilenet_gt"
GRID_SMOOTHER_CKPT = MASTER / "models/temporal_snow/checkpoints/smoother_gru_h64_l3_k7_seq96_20260524_155800/best-epoch=19-val_mae=0.0218.ckpt"
GRID_VALID_REGION_CAMERAS = {"CM00075", "CM00108"}
GRID_VALID_REGION_DIR = GRID_SEG_DIR / "camera_valid_regions"


def grid_overlay_mask(img, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay binary mask on image with given color and alpha."""
    import cv2
    overlay = img.copy()
    overlay[mask > 127] = color
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def grid_get_valid_region(camera, h, w, view=None):
    """Mask for a camera, or for one view of it; None when the camera has none.

    Any camera with a mask file is masked, no allowlist. A multi-view camera can
    carry <camera>__<NN>.png per view, with <camera>.png as the fallback.
    """
    names = ([f"{camera}__{view}.png", f"{camera}_{view}.png"] if view else []) + [f"{camera}.png"]
    for name in names:
        vr_path = GRID_VALID_REGION_DIR / name
        if vr_path.exists():
            from PIL import Image as PILImage
            return np.array(PILImage.open(vr_path).convert("L").resize((w, h), PILImage.NEAREST))
    return None


def grid_load_triplet(camera, stem):
    """Load original image + ground mask + snow mask."""
    import cv2

    img_path = GRID_DATA_DIR / camera / f"{stem}.jpg"
    ground_path = GRID_GROUND_DIR / camera / f"{stem}_ground.png"
    snow_path = GRID_SNOW_DIR / camera / f"{stem}_snow.png"

    if not img_path.exists():
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    view = stem.rsplit("_", 1)[-1] if "_" in stem else None
    vr = grid_get_valid_region(camera, h, w, view)

    ground_img = img.copy()
    if ground_path.exists():
        ground_mask = cv2.imread(str(ground_path), cv2.IMREAD_GRAYSCALE)
        if ground_mask is not None:
            if ground_mask.shape[:2] != (h, w):
                ground_mask = cv2.resize(ground_mask, (w, h))
            if vr is not None:
                ground_mask = np.where(vr > 127, ground_mask, 0).astype(np.uint8)
            ground_img = grid_overlay_mask(img, ground_mask, color=(0, 200, 0), alpha=0.4)

    snow_img = img.copy()
    if snow_path.exists():
        snow_mask = cv2.imread(str(snow_path), cv2.IMREAD_GRAYSCALE)
        if snow_mask is not None:
            if snow_mask.shape[:2] != (h, w):
                snow_mask = cv2.resize(snow_mask, (w, h))
            if vr is not None:
                snow_mask = np.where(vr > 127, snow_mask, 0).astype(np.uint8)
            snow_img = grid_overlay_mask(img, snow_mask, color=(255, 200, 0), alpha=0.5)

    return img, ground_img, snow_img


def grid_add_label(img, text, font_scale=0.5):
    """Add text label to image."""
    import cv2
    img = img.copy()
    cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def grid_run_smoother(cam_df, ckpt_path):
    """Run GRU smoother on camera dataframe, return smoothed snow_pct series."""
    try:
        import math
        import torch
        from temporal_snow.train_smoother import TemporalSmootherModule
        from temporal_snow.dataset import build_feature_columns, SNOW_PCT_COLS

        module = TemporalSmootherModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
        module.eval()

        df = cam_df.copy()
        snow_pct_cols = [c for c in SNOW_PCT_COLS if c in df.columns]
        if not snow_pct_cols:
            return None

        df["dt"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("dt").reset_index(drop=True)
        df["hour_sin"] = np.sin(2 * math.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * math.pi * df["hour"] / 24)
        doy = df["dt"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
        df["doy_cos"] = np.cos(2 * math.pi * doy / 365)

        fill_defaults = {
            "t2m": 0, "visibility": 1000, "weather_code": 0,
            "brightness": 128, "contrast": 50, "edge_density": 10, "entropy": 4,
            "white_frac": 0, "low_sat_frac": 0, "blue_ratio": 0,
            "bottom_white_frac": 0, "bottom_brightness": 128,
            "snow_n_components": 0, "snow_largest_ratio": 0,
            "snow_centroid_y": 0.5, "ground_coverage": 0,
        }
        for col in snow_pct_cols:
            df[col] = df[col].fillna(0)
        for col, default in fill_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

        if "visibility" in df.columns:
            df["visibility"] = np.log1p(df["visibility"])
        if "snow_n_components" in df.columns:
            df["snow_n_components"] = np.log1p(df["snow_n_components"])

        # Legacy normalization
        for col in snow_pct_cols:
            df[col] = df[col] / 100.0
        if "t2m" in df.columns:
            df["t2m"] = df["t2m"] / 30.0
        if "visibility" in df.columns:
            df["visibility"] = df["visibility"] / np.log1p(65000)
        if "weather_code" in df.columns:
            df["weather_code"] = df["weather_code"] / 100.0
        if "brightness" in df.columns:
            df["brightness"] = df["brightness"] / 255.0
            df["contrast"] = df["contrast"] / 128.0
            df["edge_density"] = df["edge_density"] / 50.0
            df["entropy"] = df["entropy"] / 6.0
            df["bottom_brightness"] = df["bottom_brightness"] / 255.0
        if "snow_n_components" in df.columns:
            df["snow_n_components"] = df["snow_n_components"] / np.log1p(100)

        input_size = module.hparams.get("input_size", None)
        feature_cols = build_feature_columns(snow_pct_cols, use_image_features=True)
        if input_size and len(feature_cols) != input_size:
            feature_cols = feature_cols[:input_size]

        x = df[feature_cols].values.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)

        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0)
            pred = module(x_t).squeeze(0).numpy()

        return pd.Series(pred * 100.0, index=df.index)

    except Exception as e:
        print(f"  Smoother failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def add_grid_subparser(subparsers):
    p = subparsers.add_parser("grid", help="Snow coverage comparison grid at N quantiles per camera")
    p.add_argument("--cameras", nargs="+", default=[
        "CF00001", "CM00009", "CM00014", "CM00070",
        "CM00075", "CM00080", "CM00102", "CM00108", "CM00112"
    ])
    p.add_argument("--quantiles", nargs="+", type=float, default=[0, 50, 100])
    p.add_argument("--output-dir", default="analysis/grids")
    p.add_argument("--combined", default=None, help="Save all cameras stacked into one PNG")
    p.add_argument("--thumb-size", type=int, default=320)
    return p


def run_grid(args):
    import cv2

    df = pd.read_csv(MASTER / "analysis/snow_timeseries.csv")
    df = df[df["qa_usable"] == 1]

    thumb_w = args.thumb_size
    thumb_h = int(thumb_w * 0.75)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera in args.cameras:
        cam_df = df[df["camera"] == camera].copy()
        cam_df = cam_df.sort_values("datetime").reset_index(drop=True)
        split_idx = int(len(cam_df) * 0.8)
        cam_df = cam_df.iloc[split_idx:].reset_index(drop=True)

        smoothed = grid_run_smoother(cam_df, GRID_SMOOTHER_CKPT)
        if smoothed is not None:
            cam_df["smoothed_snow_pct"] = smoothed.values
            sort_col = "smoothed_snow_pct"
            print(f"\n{camera}: using GRU smoothed snow%, range [{cam_df['smoothed_snow_pct'].min():.1f}, {cam_df['smoothed_snow_pct'].max():.1f}]%")
        else:
            cam_df["smoothed_snow_pct"] = cam_df["student_snow_pct"]
            sort_col = "student_snow_pct"
            print(f"\n{camera}: smoother unavailable, using raw snow_pct")

        cam_df = cam_df.sort_values(sort_col)

        row_imgs = []
        for q in args.quantiles:
            idx = (cam_df[sort_col] - q).abs().idxmin()
            row = cam_df.loc[idx]
            stem = row["stem"]
            actual_pct = row[sort_col]

            print(f"  Q{q:3.0f}%: {stem} (snow={actual_pct:.1f}%)")

            triplet = grid_load_triplet(camera, stem)
            if triplet is None:
                print(f"  WARNING: images not found for {stem}")
                placeholder = np.zeros((thumb_h, thumb_w * 3, 3), dtype=np.uint8)
                row_imgs.append(placeholder)
                continue

            orig, ground_img, snow_img = triplet

            def resize(img):
                return cv2.resize(img, (thumb_w, thumb_h))

            orig_t = grid_add_label(resize(orig), f"Snow {actual_pct:.1f}%")
            ground_t = grid_add_label(resize(ground_img), "Ground")
            snow_t = grid_add_label(resize(snow_img), "Snow")

            row_imgs.append(np.hstack([orig_t, ground_t, snow_t]))

        grid = np.vstack(row_imgs)
        output_path = output_dir / f"grid_{camera}.png"
        cv2.imwrite(str(output_path), grid)
        print(f"  Saved: {output_path} ({grid.shape[1]}x{grid.shape[0]})")


# =============================================================================
# DAILY-AGG-CHECK — sanity check for validate_meteo's daily aggregation
# =============================================================================

def add_daily_agg_check_subparser(subparsers):
    p = subparsers.add_parser(
        "daily-agg-check",
        help="Check daily aggregation of the meteo-validation cameras lands near 3,042 rows")
    return p


def run_daily_agg_check(args):
    from paths import ANALYSIS_DIR
    from temporal_snow.dataset import load_timeseries, aggregate_daily

    df, snow_pct_cols = load_timeseries(
        str(ANALYSIS_DIR / "snow_timeseries.csv"),
        require_target=False, qa_filter=True,
    )

    for cameras, label in [
        (["CF00001", "CM00009", "CM00014"], "3 original stations"),
        (["CF00001", "CM00009", "CM00014", "CM00067"], "4 stations (+ Smartno)"),
    ]:
        sub = df[df["camera"].isin(cameras)].copy()
        daily = aggregate_daily(sub, snow_pct_cols)
        print(f"{label}: {len(sub)} raw frames -> {len(daily)} camera-days")
        for cam in cameras:
            n_raw = (sub["camera"] == cam).sum()
            n_daily = (daily["camera"] == cam).sum()
            print(f"    {cam}: {n_raw} frames -> {n_daily} days")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Temporal-pipeline diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_meteo_validate_subparser(subparsers)
    add_melt_patterns_subparser(subparsers)
    add_grid_subparser(subparsers)
    add_daily_agg_check_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "meteo-validate":
        run_meteo_validate(args)
    elif args.command == "melt-patterns":
        run_melt_patterns(args)
    elif args.command == "grid":
        run_grid(args)
    elif args.command == "daily-agg-check":
        run_daily_agg_check(args)


if __name__ == "__main__":
    main()
