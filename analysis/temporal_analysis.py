#!/usr/bin/env python3
"""
Temporal analysis of snow cover from timeseries data.

Reads analysis/snow_timeseries.csv and produces:
  - Snow melt sequence plots per camera (snow_pct over time + temperature)
  - Day-to-day transition analysis (impossible transitions)
  - Morning vs evening asymmetry
  - Summer false positive analysis
  - Plots saved to analysis/figures/temporal/

Usage:
    python analysis/temporal_analysis.py
    python analysis/temporal_analysis.py --cameras CF00001
    python analysis/temporal_analysis.py --method sam3
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

MASTER = Path(__file__).resolve().parent.parent
TIMESERIES_CSV = MASTER / "analysis" / "snow_timeseries.csv"
FIGURES_DIR = MASTER / "analysis" / "figures" / "temporal"


def plot_snow_timeseries(df_cam, camera, method, out_dir):
    """Plot snow_pct over time for one camera, one method, overlaid with temperature."""
    col = f"{method}_snow_pct"
    if col not in df_cam.columns:
        return
    valid = df_cam.dropna(subset=[col]).copy()
    if len(valid) < 5:
        return

    valid["dt"] = pd.to_datetime(valid["datetime"])

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(valid["dt"], valid[col], "b-", alpha=0.6, linewidth=0.5, label=f"{method} snow %")
    # Rolling average
    if len(valid) > 20:
        rolling = valid.set_index("dt")[col].rolling("7D").mean()
        ax1.plot(rolling.index, rolling.values, "b-", linewidth=1.5, label="7-day avg")
    ax1.set_ylabel("Snow %", color="blue")
    ax1.set_ylim(-2, 102)
    ax1.tick_params(axis="y", labelcolor="blue")

    # Temperature overlay
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


def analyze_transitions(df, method, min_snow_pct=1.0):
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

            # Only look at consecutive images within 24h
            if dt_diff > 24 or dt_diff < 0.1:
                continue

            prev_pct = prev[col]
            curr_pct = curr[col]
            delta = curr_pct - prev_pct

            prev_hour = prev["dt"].hour
            curr_hour = curr["dt"].hour

            # Flag "impossible" transitions
            is_overnight = prev_hour >= 18 and curr_hour <= 8
            is_suspicious = False
            reason = ""

            if is_overnight and delta < -20:
                # Large snow loss overnight — suspicious (no melt without sun)
                is_suspicious = True
                reason = "overnight_melt"
            elif not is_overnight and delta > 40 and prev_pct < 5:
                # Sudden appearance during day without prior snow — could be real snowfall
                # but flag for review
                is_suspicious = True
                reason = "sudden_appearance"

            if abs(delta) > 5:  # Only record meaningful transitions
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


def morning_evening_asymmetry(df, method):
    """Compare morning and evening snow_pct within same day."""
    col = f"{method}_snow_pct"
    if col not in df.columns:
        return None

    df = df.dropna(subset=[col]).copy()
    df["dt"] = pd.to_datetime(df["datetime"])
    df["date_str"] = df["dt"].dt.strftime("%Y-%m-%d")

    # Morning: 6-10, Evening: 16-20
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


def summer_false_positives(df, method):
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


def main():
    parser = argparse.ArgumentParser(description="Temporal snow analysis")
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--method", type=str, default=None,
                        help="Single method for detailed analysis (default: all)")
    args = parser.parse_args()

    if not TIMESERIES_CSV.exists():
        print(f"ERROR: {TIMESERIES_CSV} not found. Run build_snow_timeseries.py first.")
        return

    df = pd.read_csv(TIMESERIES_CSV)
    print(f"Loaded: {len(df)} rows, {df['camera'].nunique()} cameras")

    if args.cameras:
        df = df[df["camera"].isin(args.cameras)]
        print(f"Filtered to cameras: {args.cameras}")

    # Detect available methods from columns
    methods = [c.replace("_snow_pct", "")
               for c in df.columns if c.endswith("_snow_pct")]
    if args.method:
        methods = [m for m in methods if m == args.method]
    print(f"Methods: {methods}\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Time series plots ───────────────────────────────────────────────
    print("=== Generating time series plots ===")
    cameras = sorted(df["camera"].unique())
    for cam in cameras:
        cam_df = df[df["camera"] == cam]
        for method in methods:
            plot_snow_timeseries(cam_df, cam, method, FIGURES_DIR)
    print(f"  Saved to {FIGURES_DIR}/\n")

    # ── 2. Transition analysis ─────────────────────────────────────────────
    print("=== Transition Analysis ===")
    for method in methods:
        trans = analyze_transitions(df, method)
        if trans is not None and len(trans) > 0:
            suspicious = trans[trans["suspicious"]]
            print(f"\n  {method}: {len(trans)} transitions with |delta| > 5%")
            if len(suspicious) > 0:
                print(f"  {len(suspicious)} suspicious transitions:")
                for _, r in suspicious.head(10).iterrows():
                    print(f"    {r['camera']} {r['from_dt']} -> {r['to_dt']}: "
                          f"{r['from_pct']}% -> {r['to_pct']}% "
                          f"({r['reason']}, {r['hours_gap']}h gap)")
            # Save
            trans.to_csv(FIGURES_DIR / f"transitions_{method}.csv", index=False)
    print()

    # ── 3. Morning vs evening ──────────────────────────────────────────────
    print("=== Morning vs Evening Asymmetry ===")
    for method in methods:
        asym = morning_evening_asymmetry(df, method)
        if asym is not None and len(asym) > 0:
            mean_delta = asym["delta"].mean()
            melt_days = (asym["delta"] < -5).sum()
            stable = (asym["delta"].abs() < 2).sum()
            print(f"  {method}: {len(asym)} day-pairs, mean delta={mean_delta:.2f}%")
            print(f"    Melt days (evening < morning by >5%): {melt_days}")
            print(f"    Stable days (|delta| < 2%): {stable}")

            # Plot delta distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(asym["delta"], bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(0, color="red", linestyle="--")
            ax.set_xlabel("Evening snow % - Morning snow %")
            ax.set_ylabel("Count")
            ax.set_title(f"{method}: Morning→Evening Snow Change Distribution")
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f"morning_evening_{method}.png", dpi=150)
            plt.close(fig)
    print()

    # ── 4. Summer false positives ──────────────────────────────────────────
    print("=== Summer False Positive Analysis ===")
    fp_rows = []
    for method in methods:
        fp = summer_false_positives(df, method)
        if fp:
            fp_rows.append(fp)
            print(f"  {method}: mean pred {fp['mean_pred_pct']}% when no snow, "
                  f"{fp['pct_above_5']}% of images have pred > 5%")
    if fp_rows:
        fp_df = pd.DataFrame(fp_rows)
        print(f"\n{fp_df.to_string(index=False)}")
        fp_df.to_csv(FIGURES_DIR / "summer_false_positives.csv", index=False)
    print()

    # ── 5. Multi-method agreement over time ────────────────────────────────
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
            fig.savefig(FIGURES_DIR / f"multi_method_{cam}.png", dpi=150)
            plt.close(fig)
        print(f"  Saved to {FIGURES_DIR}/\n")

    print("Temporal analysis complete!")


if __name__ == "__main__":
    main()
