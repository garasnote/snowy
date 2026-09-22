#!/usr/bin/env python3
"""Check if daily aggregation of the 3 original meteo-validation cameras
lands near 3,042 rows."""
import sys
from pathlib import Path

_MASTER = Path("/shared/home/juan.osorio/master")
sys.path.insert(0, str(_MASTER / "models"))

from temporal_snow.dataset import load_timeseries, aggregate_daily

df, snow_pct_cols = load_timeseries(
    str(_MASTER / "analysis" / "snow_timeseries.csv"),
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
