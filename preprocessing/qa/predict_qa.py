#!/usr/bin/env python3
"""Run QA classifier on images to filter non-usable ones.

Usage as script:
    python predict_qa.py --img-dir /path/to/camera/images
    python predict_qa.py --img-dir /path/to/images --threshold 0.5

Usage as library:
    from preprocessing.qa.predict_qa import load_qa_model, classify_images
    clf, feature_cols = load_qa_model("path/to/qa_gradboost.pkl")
    results = classify_images(clf, feature_cols, img_dir, meteo_path)
    # results: dict of {(camera, stem): prob_usable}
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

from train_qa import extract_image_features, parse_timestamp


DEFAULT_MODEL = _HERE / "checkpoints" / "qa_gradboost.pkl"


def load_qa_model(model_path=None):
    if model_path is None:
        model_path = DEFAULT_MODEL
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"]


def classify_images(clf, feature_cols, img_dir, meteo_path=None,
                    cameras=None, threshold=0.5):
    """Classify all images in img_dir. Returns {(camera, stem): prob_usable}."""
    img_dir = Path(img_dir)
    images = sorted(img_dir.rglob("*.jpg"))
    if cameras:
        cameras_set = set(cameras)
        images = [p for p in images if p.parent.name in cameras_set]

    meteo_lookup = {}
    if meteo_path and Path(meteo_path).exists():
        meteo = pd.read_csv(meteo_path, parse_dates=["datetime"])
        for _, row in meteo.iterrows():
            meteo_lookup[(row["camera"], row["datetime"])] = row

    records = []
    n_corrupt = 0
    for img_path in images:
        camera = img_path.parent.name
        stem = img_path.stem

        try:
            feats = extract_image_features(img_path)
        except Exception:
            n_corrupt += 1
            continue

        ts = parse_timestamp(stem)
        if ts:
            h = ts.hour + ts.minute / 60.0
            doy = ts.timetuple().tm_yday
            feats["hour"] = h
            feats["hour_sin"] = np.sin(2 * np.pi * h / 24)
            feats["hour_cos"] = np.cos(2 * np.pi * h / 24)
            feats["doy_sin"] = np.sin(2 * np.pi * doy / 365)
            feats["doy_cos"] = np.cos(2 * np.pi * doy / 365)
            feats["month"] = ts.month

            rounded = ts.replace(minute=0, second=0, microsecond=0)
            m = meteo_lookup.get((camera, pd.Timestamp(rounded)))
            if m is not None:
                feats["t2m"] = m["t2m"]
                feats["visibility"] = m["visibility"]
                feats["weather_code"] = m["weather_code"]
                feats["snow_present_meteo"] = 1 if m["snow_present"] else 0

        for col in feature_cols:
            if col not in feats:
                feats[col] = np.nan

        feats["camera"] = camera
        feats["stem"] = stem
        records.append(feats)

    if n_corrupt:
        print(f"  QA: skipped {n_corrupt} corrupt images")
    if not records:
        return {}

    df = pd.DataFrame(records)
    X = df[feature_cols].values.astype(np.float32)
    probs = clf.predict_proba(X)[:, 1]

    n_usable = (probs >= threshold).sum()
    n_rejected = len(probs) - n_usable
    print(f"  QA: {n_usable} usable, {n_rejected} rejected (threshold={threshold})")

    return {
        (row["camera"], row["stem"]): float(prob)
        for (_, row), prob in zip(df.iterrows(), probs)
    }


def write_usable_list(results, threshold, output_path):
    """Write usable stems to text file (one per line)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    usable = [stem for (cam, stem), prob in results.items() if prob >= threshold]
    with open(output_path, "w") as f:
        for stem in sorted(usable):
            f.write(stem + "\n")
    return len(usable)


def main():
    parser = argparse.ArgumentParser(description="Run QA classifier on images")
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--meteo", type=str,
                        default=str(_ROOT / "data" / "meteo_hourly.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Write usable stems to this text file")
    args = parser.parse_args()

    clf, feature_cols = load_qa_model(args.model)
    results = classify_images(
        clf, feature_cols, args.img_dir,
        meteo_path=args.meteo,
        cameras=args.cameras,
        threshold=args.threshold,
    )

    if args.output:
        n = write_usable_list(results, args.threshold, args.output)
        print(f"  Wrote {n} usable stems to {args.output}")
    else:
        for (cam, stem), prob in sorted(results.items()):
            tag = "USABLE" if prob >= args.threshold else "REJECT"
            print(f"  {cam}/{stem}: {prob:.3f} {tag}")


if __name__ == "__main__":
    main()
