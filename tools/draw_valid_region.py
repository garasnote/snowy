#!/usr/bin/env python3
"""Draw a valid region mask by clicking polygon points on an image.

Usage:
    python draw_valid_region.py /path/to/sample_image.jpg CM00075
    python draw_valid_region.py /path/to/sample_image.jpg CM00108

Left-click: add point
Right-click: undo last point
Middle-click (or press Enter): finish polygon and save
Press 'r' to reset all points
Press 'q' to quit without saving
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python draw_valid_region.py <image_path> <camera_name>")
    sys.exit(1)

img_path = sys.argv[1]
camera = sys.argv[2]

out_dir = Path(__file__).parent.parent / "preprocessing" / "segmentation" / "camera_valid_regions"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{camera}.png"

img = Image.open(img_path).convert("RGB")
W, H = img.size
print(f"Image: {W}x{H}")
print(f"Output: {out_path}")
print()
print("Left-click: add point")
print("Right-click: undo last point")
print("Enter: finish and save")
print("r: reset | q: quit")

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.imshow(img)
ax.set_title(f"{camera} — click to define valid ground region")
points = []
point_markers = []
line_obj = None


def redraw():
    global line_obj
    if line_obj is not None:
        line_obj.remove()
        line_obj = None
    if len(points) >= 2:
        xs = [p[0] for p in points] + [points[0][0]]
        ys = [p[1] for p in points] + [points[0][1]]
        line_obj, = ax.plot(xs, ys, "r-", linewidth=2, alpha=0.7)
    fig.canvas.draw_idle()


def on_click(event):
    if event.inaxes != ax:
        return
    if event.button == 1:  # left click
        points.append((event.xdata, event.ydata))
        m, = ax.plot(event.xdata, event.ydata, "ro", markersize=6)
        point_markers.append(m)
        redraw()
        print(f"  Point {len(points)}: ({event.xdata:.0f}, {event.ydata:.0f})")
    elif event.button == 3:  # right click — undo
        if points:
            points.pop()
            m = point_markers.pop()
            m.remove()
            redraw()
            print(f"  Undo → {len(points)} points")


def on_key(event):
    if event.key == "enter":
        if len(points) < 3:
            print("  Need at least 3 points!")
            return
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        int_points = [(int(x), int(y)) for x, y in points]
        draw.polygon(int_points, fill=255)
        mask.save(out_path)
        n_white = np.sum(np.array(mask) > 127)
        print(f"\nSaved: {out_path}")
        print(f"  {n_white} valid pixels / {W*H} total ({100*n_white/(W*H):.1f}%)")
        print(f"  Polygon: {int_points}")
        plt.close()
    elif event.key == "r":
        points.clear()
        for m in point_markers:
            m.remove()
        point_markers.clear()
        redraw()
        print("  Reset!")
    elif event.key == "q":
        print("  Quit without saving")
        plt.close()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
plt.tight_layout()
plt.show()
