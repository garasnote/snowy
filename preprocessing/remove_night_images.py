#!/usr/bin/env python3
"""
Move night/dark images to a separate folder based on sunrise/sunset times.

Uses the astral library to compute sunrise/sunset for each image's date
at Slovenia's center coordinates (~46.15°N, 14.99°E).

Margin shrinks the daylight window: --margin 60 means daylight is
sunrise+60min to sunset-60min (removes twilight images).

Usage:
    python remove_night_images.py --data-dir /path/to/data --margin 60
    python remove_night_images.py --data-dir /path/to/data --margin 60 --brightness 30
    python remove_night_images.py --data-dir /path/to/data --dry-run
    python remove_night_images.py --data-dir /path/to/data --cameras CM00024 CF00001
"""

import argparse
import csv
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from astral import LocationInfo
from astral.sun import sun
from PIL import Image
from tqdm import tqdm


SLOVENIA_LOCATION = LocationInfo(
    name="Slovenia",
    region="Slovenia",
    timezone="Europe/Ljubljana",
    latitude=46.15,
    longitude=14.99,
)
TZ = ZoneInfo("Europe/Ljubljana")


def parse_image_datetime(filename: str) -> datetime | None:
    """Parse datetime from image filename.

    Handles both formats:
        CM00006_202401290000_36.jpg     → parts[1] is datetime
        CF00001_VIS_202509042100_00.jpg → parts[2] is datetime
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    for part in parts[1:]:
        if len(part) == 12:
            try:
                dt = datetime.strptime(part, "%Y%m%d%H%M")
                return dt.replace(tzinfo=TZ)
            except ValueError:
                continue
        elif len(part) == 14:
            try:
                dt = datetime.strptime(part, "%Y%m%d%H%M%S")
                return dt.replace(tzinfo=TZ)
            except ValueError:
                continue
    return None


def get_sun_times(date, margin_minutes: int = 0):
    """Get sunrise/sunset with margin shrinking the daylight window.

    margin > 0 means: sunrise + margin (later) and sunset - margin (earlier),
    so twilight images near dawn/dusk are also classified as night.
    """
    s = sun(SLOVENIA_LOCATION.observer, date=date, tzinfo=TZ)
    sunrise = s["sunrise"] + timedelta(minutes=margin_minutes)
    sunset = s["sunset"] - timedelta(minutes=margin_minutes)
    return sunrise, sunset


def check_brightness(img_path, threshold=30):
    """Return True if image is too dark (mean brightness < threshold)."""
    try:
        img = Image.open(img_path).convert("L")
        return np.array(img).mean() < threshold
    except Exception:
        return False


def process_camera(
    camera_dir: Path,
    output_dir: Path,
    margin: int,
    dry_run: bool,
    sun_cache: dict,
    brightness_threshold: int = 0,
) -> list[dict]:
    """Process all images in a camera folder. Returns log entries."""
    camera_id = camera_dir.name
    image_files = sorted(camera_dir.glob("*.jpg"))
    log_entries = []
    moved_count = 0

    for img_path in image_files:
        img_dt = parse_image_datetime(img_path.name)
        if img_dt is None:
            continue

        date_key = img_dt.date()
        if date_key not in sun_cache:
            try:
                sun_cache[date_key] = get_sun_times(date_key, margin)
            except Exception:
                continue
        sunrise, sunset = sun_cache[date_key]

        reason = None
        if img_dt < sunrise:
            reason = "before_sunrise"
        elif img_dt > sunset:
            reason = "after_sunset"
        elif brightness_threshold > 0 and check_brightness(img_path, brightness_threshold):
            reason = "too_dark"

        if reason:
            moved_count += 1
            log_entries.append({
                "camera": camera_id,
                "filename": img_path.name,
                "image_datetime": img_dt.strftime("%Y-%m-%d %H:%M"),
                "sunrise": sunrise.strftime("%H:%M"),
                "sunset": sunset.strftime("%H:%M"),
                "reason": reason,
            })

            if not dry_run:
                dest_dir = output_dir / camera_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(dest_dir / img_path.name))

    return log_entries


def main():
    parser = argparse.ArgumentParser(
        description="Move night images to a separate folder based on sunrise/sunset times."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/shared/home/juan.osorio/master/data",
        help="Path to data directory with camera folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to move night images (default: night_images/ next to data dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be moved without actually moving",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Minutes of twilight buffer around sunrise/sunset (default: 0)",
    )
    parser.add_argument(
        "--brightness",
        type=int,
        default=0,
        help="Mean brightness threshold (0-255). Images darker than this are moved regardless of sun position (default: 0 = disabled)",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Specific camera IDs to process (default: all)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir.parent / "night_images"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return

    # Find camera folders
    if args.cameras:
        camera_dirs = [data_dir / c for c in args.cameras if (data_dir / c).is_dir()]
    else:
        camera_dirs = sorted(
            [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(("CM", "CF"))]
        )

    if not camera_dirs:
        print("ERROR: No camera folders found.")
        return

    print("=" * 60)
    print("NIGHT IMAGE REMOVAL")
    print("=" * 60)
    print(f"Data dir:    {data_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Margin:      {args.margin} min (sunrise+{args.margin}, sunset-{args.margin})")
    print(f"Brightness:  {args.brightness} (0=disabled)")
    print(f"Dry run:     {args.dry_run}")
    print(f"Cameras:     {len(camera_dirs)}")
    print(f"Location:    Slovenia (46.15°N, 14.99°E)")
    print("=" * 60)
    print()

    sun_cache = {}
    all_log_entries = []
    camera_stats = {}

    for camera_dir in tqdm(camera_dirs, desc="Processing cameras"):
        total_images = len(list(camera_dir.glob("*.jpg")))
        entries = process_camera(camera_dir, output_dir, args.margin, args.dry_run, sun_cache, args.brightness)
        all_log_entries.extend(entries)
        camera_stats[camera_dir.name] = {
            "total": total_images,
            "night": len(entries),
            "day": total_images - len(entries),
        }

    # Write log CSV
    if all_log_entries and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "night_images_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["camera", "filename", "image_datetime", "sunrise", "sunset", "reason"]
            )
            writer.writeheader()
            writer.writerows(all_log_entries)
        print(f"\nLog saved to: {log_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_night = 0
    total_all = 0
    for camera in sorted(camera_stats.keys()):
        stats = camera_stats[camera]
        total_night += stats["night"]
        total_all += stats["total"]
        pct = (stats["night"] / stats["total"] * 100) if stats["total"] > 0 else 0
        action = "would move" if args.dry_run else "moved"
        print(f"  {camera}: {action} {stats['night']}/{stats['total']} ({pct:.1f}% night)")

    pct_total = (total_night / total_all * 100) if total_all > 0 else 0
    action = "Would move" if args.dry_run else "Moved"
    print(f"\n  TOTAL: {action} {total_night}/{total_all} images ({pct_total:.1f}% night)")
    print("=" * 60)


if __name__ == "__main__":
    main()
