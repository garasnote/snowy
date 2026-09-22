#!/usr/bin/env python3
"""
Snow Pipeline Viewer — PyQt6 desktop app for inspecting the full
ground+snow segmentation pipeline.

Tabs:
  - Viewer:     Image playback with ground/snow mask overlays
  - Timeseries: Snow % + temperature over time (matplotlib)
  - QA:         Image quality scores and rejection analysis

Features:
  - Camera selector, date range picker, timeline scrubbing
  - Playback with speed control (0.25x–8x)
  - Overlay toggles: ground mask, snow mask
  - Side panel: QA score, snow_pct, meteo data per frame
  - Live inference: run ground/snow models on any image on-the-fly
  - Add new camera data directories

Usage:
    python tools/viewer/app.py --data-dir ~/data --csv ~/snow_timeseries.csv
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QCheckBox,
    QSplitter, QFileDialog, QDateEdit,
    QStatusBar, QMessageBox, QFrame, QSizePolicy, QScrollArea,
    QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from PyQt6.QtCore import Qt, QTimer, QDate
from PyQt6.QtGui import QPixmap, QImage, QAction, QKeySequence, QColor

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MASTER = Path(__file__).resolve().parent.parent.parent

GROUND_MASK_DIR = MASTER / "preprocessing" / "segmentation" / "student_ground_output" / "mobilenet_ft"
SNOW_MASK_DIR = MASTER / "preprocessing" / "segmentation" / "student_snow_output" / "mobilenet_gt"
TIMESERIES_CSV = MASTER / "analysis" / "snow_timeseries.csv"


def load_image_as_qpixmap(path):
    from PIL import ImageFile
    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        data = np.array(img)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev
    h, w, ch = data.shape
    qimg = QImage(data.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def load_mask_as_array(path):
    if not path.exists():
        return None
    img = Image.open(path).convert("L")
    return np.array(img)


def overlay_mask_fast(base_pixmap, mask_array, color, alpha=0.4):
    if mask_array is None:
        return base_pixmap

    qimg = base_pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    stride = qimg.bytesPerLine()
    try:
        ptr = qimg.bits()
        ptr.setsize(h * stride)
        raw = np.frombuffer(ptr, dtype=np.uint8).reshape((h, stride))
        base_np = raw[:, :w * 3].reshape((h, w, 3)).copy()
    except Exception:
        from io import BytesIO
        buf = BytesIO()
        base_pixmap.save(buf, "PNG")
        buf.seek(0)
        base_np = np.array(Image.open(buf).convert("RGB"))
        h, w = base_np.shape[:2]

    mask_resized = np.array(
        Image.fromarray(mask_array).resize((w, h), Image.NEAREST)
    )
    mask_bool = mask_resized > 127

    r, g, b = color
    overlay_color = np.array([r, g, b], dtype=np.float32)
    base_float = base_np.astype(np.float32)
    base_float[mask_bool] = (1 - alpha) * base_float[mask_bool] + alpha * overlay_color
    composited = np.clip(base_float, 0, 255).astype(np.uint8)
    composited = np.ascontiguousarray(composited)
    bpl = w * 3
    qimg_out = QImage(composited.data, w, h, bpl, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg_out.copy())


# ── Info Panel ──────────────────────────────────────────────────────────────

class InfoPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.title = QLabel("Frame Info")
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.title)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self.info_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        self.snow_bar_label = QLabel("Snow %: —")
        self.snow_bar_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.snow_bar_label)

        self.snow_bar = QProgressBar()
        self.snow_bar.setRange(0, 100)
        self.snow_bar.setValue(0)
        self.snow_bar.setTextVisible(True)
        self.snow_bar.setFixedHeight(20)
        layout.addWidget(self.snow_bar)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)

        self.meteo_label = QLabel("Meteo: —")
        self.meteo_label.setWordWrap(True)
        self.meteo_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self.meteo_label)

        layout.addStretch()

    def update_info(self, row, smoothed_val=None):
        if row is None:
            self.info_label.setText("No data")
            self.snow_bar.setValue(0)
            self.snow_bar_label.setText("Snow %: —")
            self.meteo_label.setText("Meteo: —")
            return

        stem = row.get("stem", "?")
        cam = row.get("camera", "?")
        dt = row.get("datetime", "?")
        qa = row.get("prob_usable", float("nan"))
        qa_str = f"{qa:.2f}" if pd.notna(qa) else "N/A"
        qa_flag = row.get("qa_usable", 1)
        ground_cov = row.get("ground_coverage", float("nan"))
        ground_str = f"{ground_cov:.1%}" if pd.notna(ground_cov) else "N/A"

        lines = [
            f"Camera:   {cam}",
            f"Stem:     {stem}",
            f"DateTime: {dt}",
            f"QA score: {qa_str} ({'OK' if qa_flag == 1 else 'REJECT'})",
            f"Ground:   {ground_str}",
        ]
        self.info_label.setText("\n".join(lines))

        snow_pct = row.get("student_snow_pct", 0)
        if pd.isna(snow_pct):
            snow_pct = 0
        snow_label = f"Snow %: {snow_pct:.1f}%"
        if smoothed_val is not None:
            snow_label += f"  (smoothed: {smoothed_val:.1f}%)"
        self.snow_bar.setValue(min(int(snow_pct), 100))
        self.snow_bar_label.setText(snow_label)

        if snow_pct > 50:
            self.snow_bar.setStyleSheet("QProgressBar::chunk { background: #4488ff; }")
        elif snow_pct > 10:
            self.snow_bar.setStyleSheet("QProgressBar::chunk { background: #88bbff; }")
        else:
            self.snow_bar.setStyleSheet("QProgressBar::chunk { background: #44aa44; }")

        t2m = row.get("t2m", float("nan"))
        vis = row.get("visibility", float("nan"))
        wcode = row.get("weather_code", float("nan"))
        snow_present = row.get("snow_present", "?")
        snow_cm = row.get("snow_cm", float("nan"))

        meteo_lines = [
            f"Temp:     {t2m:.1f}°C" if pd.notna(t2m) else "Temp:     N/A",
            f"Visibility: {vis:.0f}" if pd.notna(vis) else "Visibility: N/A",
            f"Weather:  {int(wcode)}" if pd.notna(wcode) else "Weather:  N/A",
            f"Snow GT:  {snow_present}",
            f"Snow cm:  {snow_cm:.1f}" if pd.notna(snow_cm) else "Snow cm:  N/A",
        ]
        self.meteo_label.setText("\n".join(meteo_lines))


class ImageDisplay(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #1a1a1a;")
        self.setText("No image loaded")
        self._current_pixmap = None

    def set_pixmap_scaled(self, pixmap):
        if pixmap is None:
            self.setText("No image")
            self._current_pixmap = None
            return
        self._current_pixmap = pixmap
        self._rescale()

    def _rescale(self):
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


# ── Timeseries Tab ──────────────────────────────────────────────────────────

class TimeseriesTab(QWidget):
    frame_selected = None  # will be set as signal

    def __init__(self, parent_viewer):
        super().__init__()
        self.viewer = parent_viewer
        self.cam_df = None
        self.smoothed = None
        self.kalman = None
        self._cursor_line = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if not HAS_MPL:
            layout.addWidget(QLabel("matplotlib not installed — timeseries view unavailable"))
            return

        # Controls
        ctrl = QHBoxLayout()
        self.rolling_cb = QCheckBox("7-day rolling avg")
        self.rolling_cb.setChecked(True)
        self.rolling_cb.toggled.connect(self._replot)
        ctrl.addWidget(self.rolling_cb)

        self.temp_cb = QCheckBox("Temperature overlay")
        self.temp_cb.setChecked(True)
        self.temp_cb.toggled.connect(self._replot)
        ctrl.addWidget(self.temp_cb)

        self.meteo_cb = QCheckBox("Snow GT (meteo)")
        self.meteo_cb.setChecked(True)
        self.meteo_cb.toggled.connect(self._replot)
        ctrl.addWidget(self.meteo_cb)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Matplotlib canvas
        self.figure = Figure(figsize=(12, 4), dpi=100)
        self.figure.patch.set_facecolor("#1e1e1e")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        layout.addWidget(self.canvas, stretch=1)

    def update_camera(self, cam_df, smoothed=None, kalman=None):
        self.cam_df = cam_df
        self.smoothed = smoothed
        self.kalman = kalman
        self._replot()

    def update_cursor(self, idx):
        if not HAS_MPL or self.cam_df is None or len(self.cam_df) == 0:
            return
        if idx < 0 or idx >= len(self.cam_df):
            return
        dt = self.cam_df.iloc[idx].get("datetime")
        if pd.isna(dt):
            return
        ax = self.figure.axes[0] if self.figure.axes else None
        if ax is None:
            return
        if self._cursor_line is not None:
            self._cursor_line.set_xdata([dt, dt])
        else:
            self._cursor_line = ax.axvline(dt, color="#ff6666", linewidth=1.5, alpha=0.8, zorder=10)
        self.canvas.draw_idle()

    def _replot(self):
        if not HAS_MPL:
            return
        self.figure.clear()
        self._cursor_line = None
        if self.cam_df is None or len(self.cam_df) == 0:
            self.canvas.draw()
            return

        df = self.cam_df.copy()
        if "datetime" not in df.columns:
            self.canvas.draw()
            return
        df["dt"] = pd.to_datetime(df["datetime"])
        col = "student_snow_pct"
        if col not in df.columns:
            self.canvas.draw()
            return

        ax1 = self.figure.add_subplot(111)
        ax1.set_facecolor("#1e1e1e")

        # Raw snow_pct
        ax1.plot(df["dt"], df[col], color="#4488ff", alpha=0.3, linewidth=0.3, label="Snow %")

        # Rolling average
        if self.rolling_cb.isChecked() and len(df) > 20:
            rolling = df.set_index("dt")[col].rolling("7D").mean()
            ax1.plot(rolling.index, rolling.values, color="#4488ff", linewidth=2, label="7-day avg")

        # Smoother (GRU/LSTM) output
        if self.smoothed is not None and len(self.smoothed) == len(df):
            ax1.plot(df["dt"], self.smoothed.values, color="#44ff88", linewidth=1.8,
                     alpha=0.9, label="GRU/LSTM")

        # Kalman filter output
        if self.kalman is not None and len(self.kalman) == len(df):
            ax1.plot(df["dt"], self.kalman, color="#ff9933", linewidth=1.5,
                     alpha=0.85, label="Kalman", linestyle="--")

        ax1.set_ylabel("Snow %", color="#4488ff")
        ax1.set_ylim(-2, 102)
        ax1.tick_params(axis="y", labelcolor="#4488ff")
        ax1.tick_params(axis="x", colors="#aaaaaa")

        # Meteo snow_present shading
        if self.meteo_cb.isChecked() and "snow_present" in df.columns:
            snow_gt = df["snow_present"].fillna(False)
            snow_bool = (snow_gt == True) | (snow_gt == "True")  # noqa: E712
            if snow_bool.any():
                ax1.fill_between(
                    df["dt"], 0, 100,
                    where=snow_bool,
                    alpha=0.08, color="#88ccff", label="Meteo: snow",
                )

        # Temperature overlay
        if self.temp_cb.isChecked() and "t2m" in df.columns:
            t2m = df.dropna(subset=["t2m"])
            if len(t2m) > 0:
                ax2 = ax1.twinx()
                ax2.plot(t2m["dt"], t2m["t2m"], color="#ff6644", alpha=0.4, linewidth=0.5)
                ax2.axhline(0, color="#ff6644", linestyle="--", alpha=0.3)
                ax2.set_ylabel("Temp (°C)", color="#ff6644")
                ax2.tick_params(axis="y", labelcolor="#ff6644")
                ax2.spines["right"].set_color("#ff6644")

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        self.figure.autofmt_xdate()

        camera = df.iloc[0].get("camera", "")
        ax1.set_title(f"{camera} — Snow Timeline", color="#dddddd", fontsize=13)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_color("#555555")
        ax1.spines["left"].set_color("#4488ff")

        ax1.legend(loc="upper left", fontsize=8, facecolor="#2a2a2a", edgecolor="#555555",
                   labelcolor="#dddddd")

        self.figure.tight_layout()
        self.canvas.draw()

    def _on_click(self, event):
        if event.inaxes is None or self.cam_df is None:
            return
        try:
            click_dt = mdates.num2date(event.xdata).replace(tzinfo=None)
        except Exception:
            return
        df = self.cam_df.copy()
        df["dt"] = pd.to_datetime(df["datetime"])
        diffs = (df["dt"] - click_dt).abs()
        nearest_idx = diffs.idxmin()
        pos = df.index.get_loc(nearest_idx)
        self.viewer._goto_frame(pos)


# ── QA Tab ──────────────────────────────────────────────────────────────────

def _extract_qa_features(img_path, stem=None):
    """Extract QA features: base image stats + temporal. Matches qa_gradboost_lite model."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    brightness = gray.mean()
    contrast = gray.std()
    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    edge_density = (np.abs(gy).mean() + np.abs(gx).mean()) / 2.0
    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    total = r + g + b + 1e-6
    blue_ratio = (b / total).mean()
    white_frac = ((r > 200) & (g > 200) & (b > 200)).mean()
    sat = hsv[:, :, 1]
    low_sat_frac = (sat < 30).mean()
    h_img = arr.shape[0]
    crop_start = int(h_img * 0.3)
    bottom = gray[crop_start:]
    bottom_brightness = bottom.mean()
    bottom_white_frac = ((r[crop_start:] > 200) & (g[crop_start:] > 200) & (b[crop_start:] > 200)).mean()

    feats = {
        "r_mean": rgb_mean[0], "g_mean": rgb_mean[1], "b_mean": rgb_mean[2],
        "r_std": rgb_std[0], "g_std": rgb_std[1], "b_std": rgb_std[2],
        "h_mean": hsv_mean[0], "s_mean": hsv_mean[1], "v_mean": hsv_mean[2],
        "h_std": hsv_std[0], "s_std": hsv_std[1], "v_std": hsv_std[2],
        "brightness": brightness, "contrast": contrast,
        "edge_density": edge_density, "entropy": entropy,
        "blue_ratio": blue_ratio, "white_frac": white_frac,
        "low_sat_frac": low_sat_frac,
        "bottom_brightness": bottom_brightness, "bottom_white_frac": bottom_white_frac,
    }

    if stem:
        m = re.search(r"(\d{12})", stem)
        if m:
            from datetime import datetime as _dt
            ts = _dt.strptime(m.group(1), "%Y%m%d%H%M")
            h = ts.hour + ts.minute / 60.0
            doy = ts.timetuple().tm_yday
            feats["hour"] = h
            feats["hour_sin"] = np.sin(2 * np.pi * h / 24)
            feats["hour_cos"] = np.cos(2 * np.pi * h / 24)
            feats["doy_sin"] = np.sin(2 * np.pi * doy / 365)
            feats["doy_cos"] = np.cos(2 * np.pi * doy / 365)
            feats["month"] = ts.month

    return feats


def _load_qa_model(model_path):
    """Load QA model — pickle (GradBoost) or TorchScript (.pt) CNN."""
    model_path = Path(model_path)
    if model_path.suffix == ".pt":
        import torch
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        return ("cnn", model, 128), None
    elif model_path.suffix == ".ckpt" and "qa_cnn" in model_path.stem:
        import torch
        sys_path_parent = str(model_path.resolve().parent.parent.parent)
        if sys_path_parent not in sys.path:
            sys.path.insert(0, sys_path_parent)
        from preprocessing.qa.train_qa_cnn import TinyQACNN
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model = TinyQACNN(channels=ckpt["channels"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return ("cnn", model, ckpt["input_size"]), None
    else:
        import pickle
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        return ("gradboost", bundle["model"]), bundle["feature_cols"]


class QAWorker(QWidget):
    """Background QA processing — runs in main thread via QTimer chunks."""
    pass


class QATab(QWidget):
    def __init__(self, parent_viewer):
        super().__init__()
        self.viewer = parent_viewer
        self.qa_model = None
        self.qa_feature_cols = None
        self._qa_results = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Top controls
        ctrl = QHBoxLayout()
        self.summary_label = QLabel("QA Summary: —")
        self.summary_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        ctrl.addWidget(self.summary_label, stretch=1)

        self.run_qa_btn = QPushButton("Run QA on Camera")
        self.run_qa_btn.setStyleSheet("padding: 4px 12px; font-weight: bold;")
        self.run_qa_btn.clicked.connect(self._run_qa)
        ctrl.addWidget(self.run_qa_btn)

        self.stop_qa_btn = QPushButton("Stop")
        self.stop_qa_btn.setStyleSheet("padding: 4px 12px; color: #ff4444; font-weight: bold;")
        self.stop_qa_btn.clicked.connect(self._stop_qa)
        self.stop_qa_btn.setVisible(False)
        ctrl.addWidget(self.stop_qa_btn)

        self.test_img_btn = QPushButton("Test Image...")
        self.test_img_btn.setStyleSheet("padding: 4px 12px;")
        self.test_img_btn.clicked.connect(self._test_single_image)
        ctrl.addWidget(self.test_img_btn)

        layout.addLayout(ctrl)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        if HAS_MPL:
            self.figure = Figure(figsize=(12, 3), dpi=100)
            self.figure.patch.set_facecolor("#1e1e1e")
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas, stretch=1)

        # Best & Worst gallery
        gallery_header = QLabel("Worst & Best Samples:")
        gallery_header.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(gallery_header)

        self._gallery_scroll = QScrollArea()
        self._gallery_scroll.setWidgetResizable(True)
        self._gallery_scroll.setFixedHeight(180)
        self._gallery_scroll.setStyleSheet("QScrollArea { border: 1px solid #444; background: #1e1e1e; }")
        self._gallery_container = QWidget()
        self._gallery_layout = QHBoxLayout(self._gallery_container)
        self._gallery_layout.setContentsMargins(8, 4, 8, 4)
        self._gallery_layout.setSpacing(6)
        self._gallery_scroll.setWidget(self._gallery_container)

        placeholder = QLabel("Run QA to see best & worst images")
        placeholder.setStyleSheet("color: #888; font-size: 12px;")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gallery_layout.addWidget(placeholder)

        layout.addWidget(self._gallery_scroll)

        self.table_label = QLabel("Rejected images:")
        self.table_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(self.table_label)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Stem", "DateTime", "QA Score", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table, stretch=1)

    def _ensure_qa_model(self):
        if self.qa_model is not None:
            return True
        qa_path = None
        for name in ("qa_cnn.pt", "qa_gradboost_lite.pkl", "qa_gradboost.pkl"):
            for base in (
                self.viewer.models_dir.parent / "preprocessing" / "qa" / "checkpoints",
                Path(__file__).resolve().parent.parent.parent / "preprocessing" / "qa" / "checkpoints",
                self.viewer.models_dir / "best",
            ):
                candidate = base / name
                if candidate.exists():
                    qa_path = candidate
                    break
            if qa_path:
                break
        if not qa_path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select QA Model", str(self.viewer.models_dir),
                "QA Model (*.pkl *.pt *.ckpt)"
            )
            if not path:
                return False
            qa_path = Path(path)
        try:
            self.qa_model, self.qa_feature_cols = _load_qa_model(qa_path)
            if isinstance(self.qa_model, tuple) and self.qa_model[0] == "cnn":
                print(f"Loaded QA CNN: {qa_path.name} (input {self.qa_model[2]}×{self.qa_model[2]})")
            else:
                print(f"Loaded QA model: {qa_path.name} ({len(self.qa_feature_cols)} features)")
            return True
        except Exception as e:
            QMessageBox.warning(self, "QA Model Error", f"Failed to load QA model:\n{e}")
            return False

    def _run_qa(self):
        if not self._ensure_qa_model():
            return
        cam_df = self.viewer.cam_df
        if cam_df is None or len(cam_df) == 0:
            return

        camera = cam_df.iloc[0]["camera"]
        data_dir = self.viewer.data_dir / camera
        stems = cam_df["stem"].tolist()

        self.progress.setVisible(True)
        self.progress.setMaximum(len(stems))
        self.progress.setValue(0)
        self.progress.setFormat(f"QA: 0/{len(stems)}")
        self.run_qa_btn.setEnabled(False)
        self.stop_qa_btn.setVisible(True)

        self._qa_queue = stems
        self._qa_camera = camera
        self._qa_data_dir = data_dir
        self._qa_idx = 0
        self._qa_records = []

        self._qa_timer = QTimer()
        self._qa_timer.timeout.connect(self._qa_process_batch)
        self._qa_timer.start(0)

    def _qa_predict_one(self, img_path, stem):
        """Run QA prediction on a single image. Handles both CNN and GradBoost."""
        with Image.open(img_path) as test_img:
            test_img.verify()
        if isinstance(self.qa_model, tuple) and self.qa_model[0] == "cnn":
            import torch
            _, model, input_size = self.qa_model
            img = Image.open(img_path).convert("RGB").resize(
                (input_size, input_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD if 'IMAGENET_MEAN' in dir() else arr
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
            with torch.no_grad():
                logit = model(tensor)
                prob = torch.sigmoid(logit).item()
            return prob
        else:
            _, gb_model = self.qa_model
            feats = _extract_qa_features(img_path, stem=stem)
            feat_vec = np.array([[feats.get(c, 0.0) for c in self.qa_feature_cols]])
            return gb_model.predict_proba(feat_vec)[0, 1]

    def _qa_process_batch(self):
        batch_size = 20
        end = min(self._qa_idx + batch_size, len(self._qa_queue))

        for i in range(self._qa_idx, end):
            stem = self._qa_queue[i]
            img_path = self._qa_data_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue
            try:
                prob = self._qa_predict_one(img_path, stem)
            except Exception:
                prob = 0.0
            self._qa_records.append({
                "stem": stem, "prob_usable": prob,
                "qa_usable": 1 if prob >= 0.5 else 0,
            })

        self._qa_idx = end
        self.progress.setValue(end)
        n_rej = sum(1 for r in self._qa_records if r["qa_usable"] == 0)
        self.progress.setFormat(f"QA: {end}/{len(self._qa_queue)} ({n_rej} rejected)")
        self.summary_label.setText(
            f"QA: {end}/{len(self._qa_queue)} processed — {n_rej} rejected so far"
        )

        if end % 100 == 0 and self._qa_records:
            self._update_partial_plot()

        if self._qa_idx >= len(self._qa_queue):
            self._qa_timer.stop()
            self._qa_finish()

    def _update_partial_plot(self):
        if not HAS_MPL or not self._qa_records:
            return
        self.figure.clear()
        scores = [r["prob_usable"] for r in self._qa_records]
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1e1e1e")
        ax.hist(scores, bins=50, color="#4488ff", alpha=0.7, edgecolor="#2266cc")
        ax.axvline(0.5, color="#ff4444", linestyle="--", linewidth=1.5)
        ax.set_xlabel("QA Score", color="#dddddd")
        ax.set_ylabel("Count", color="#dddddd")
        n_rej = sum(1 for s in scores if s < 0.5)
        ax.set_title(f"QA Scores ({len(scores)} processed, {n_rej} rejected)",
                     color="#dddddd", fontsize=11)
        ax.tick_params(colors="#aaaaaa")
        self.figure.tight_layout()
        self.canvas.draw()

    def _stop_qa(self):
        if hasattr(self, "_qa_timer"):
            self._qa_timer.stop()
        self.stop_qa_btn.setVisible(False)
        self._qa_finish()

    def _qa_finish(self):
        self.progress.setVisible(False)
        self.run_qa_btn.setEnabled(True)
        self.stop_qa_btn.setVisible(False)

        if not self._qa_records:
            self.summary_label.setText("QA: No images processed")
            return

        qa_df = pd.DataFrame(self._qa_records)
        cam_df = self.viewer.cam_df
        cam_df = cam_df.drop(columns=["prob_usable", "qa_usable"], errors="ignore")
        cam_df = cam_df.merge(qa_df, on="stem", how="left")
        cam_df["prob_usable"] = cam_df["prob_usable"].fillna(1.0)
        cam_df["qa_usable"] = cam_df["qa_usable"].fillna(1).astype(int)

        self.viewer.cam_df = cam_df
        self._qa_results = qa_df
        self.update_camera(cam_df)

    def _build_gallery(self, cam_df):
        while self._gallery_layout.count():
            w = self._gallery_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

        if "prob_usable" not in cam_df.columns:
            return

        scored = cam_df.dropna(subset=["prob_usable"])
        if len(scored) == 0:
            return

        camera = cam_df.iloc[0]["camera"]
        data_dir = self.viewer.data_dir / camera

        exists_mask = scored["stem"].apply(lambda s: (data_dir / f"{s}.jpg").exists())
        scored = scored[exists_mask]
        if len(scored) == 0:
            return

        worst = scored.nsmallest(5, "prob_usable")
        best = scored.nlargest(5, "prob_usable")

        def add_section(label_text, rows, color):
            section = QVBoxLayout()
            lbl = QLabel(label_text)
            lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            section_w = QWidget()
            section_w.setLayout(section)
            section.addWidget(lbl)

            row_layout = QHBoxLayout()
            row_layout.setSpacing(4)
            for _, row in rows.iterrows():
                stem = row["stem"]
                score = row["prob_usable"]
                img_path = data_dir / f"{stem}.jpg"
                card = QVBoxLayout()
                card.setSpacing(1)
                thumb_label = QLabel()
                thumb_label.setFixedSize(120, 90)
                thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                thumb_label.setStyleSheet("border: 1px solid #555; background: #222;")
                thumb_label.setCursor(Qt.CursorShape.PointingHandCursor)

                if img_path.exists():
                    pix = QPixmap(str(img_path)).scaled(
                        120, 90, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    thumb_label.setPixmap(pix)
                else:
                    thumb_label.setText("N/A")

                thumb_label.mousePressEvent = lambda ev, s=stem: self._goto_stem(s)

                score_lbl = QLabel(f"{score:.3f}")
                score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                sc = "#ff4444" if score < 0.5 else "#44bb44"
                score_lbl.setStyleSheet(f"color: {sc}; font-size: 10px;")

                card_w = QWidget()
                card_w.setLayout(card)
                card.addWidget(thumb_label)
                card.addWidget(score_lbl)
                row_layout.addWidget(card_w)

            section.addLayout(row_layout)
            self._gallery_layout.addWidget(section_w)

        add_section("5 Worst", worst, "#ff4444")

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #555;")
        self._gallery_layout.addWidget(sep)

        add_section("5 Best", best, "#44bb44")
        self._gallery_layout.addStretch()

    def _test_single_image(self):
        if not self._ensure_qa_model():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image to Test", str(self.viewer.data_dir),
            "Images (*.jpg *.jpeg *.png)"
        )
        if not path:
            return
        try:
            stem = Path(path).stem
            prob = self._qa_predict_one(path, stem)
            verdict = "USABLE" if prob >= 0.5 else "REJECTED"

            msg = f"QA Score: {prob:.4f}\nVerdict: {verdict}\n\nFile: {Path(path).name}"
            QMessageBox.information(self, "QA Test Result", msg)

            self.viewer.tabs.setCurrentIndex(0)
            pixmap = load_image_as_qpixmap(path)

            if self.viewer.ground_model is not None or self.viewer.snow_model is not None:
                self.viewer._infer_on_pixmap(pixmap, path)
            else:
                self.viewer.image_display.set_pixmap_scaled(pixmap)
                self.viewer.info_panel.update_info({
                    "stem": Path(path).stem, "camera": "test",
                    "datetime": "—", "prob_usable": prob,
                    "qa_usable": 1 if prob >= 0.5 else 0,
                    "student_snow_pct": float("nan"),
                })
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to process image:\n{e}")

    def update_camera(self, cam_df):
        if cam_df is None or len(cam_df) == 0:
            self.summary_label.setText("QA Summary: No data")
            self.table.setRowCount(0)
            return

        total = len(cam_df)
        if "qa_usable" in cam_df.columns:
            n_ok = (cam_df["qa_usable"] == 1).sum()
            n_rej = total - n_ok
        else:
            n_ok = total
            n_rej = 0

        self.summary_label.setText(
            f"QA Summary: {total} images — {n_ok} usable, {n_rej} rejected "
            f"({n_rej/total*100:.1f}%)" if total > 0 else "QA Summary: No data"
        )

        if HAS_MPL:
            self.figure.clear()

            if "prob_usable" in cam_df.columns:
                scores = cam_df["prob_usable"].dropna()
                has_scores = len(scores) > 0 and not (scores == 1.0).all()
            else:
                has_scores = False

            if has_scores:
                ax1 = self.figure.add_subplot(121)
                ax1.set_facecolor("#1e1e1e")
                ax1.hist(scores, bins=50, color="#4488ff", alpha=0.7, edgecolor="#2266cc")
                ax1.axvline(0.5, color="#ff4444", linestyle="--", linewidth=1.5, label="Threshold")
                ax1.set_xlabel("QA Score", color="#dddddd")
                ax1.set_ylabel("Count", color="#dddddd")
                ax1.set_title("Score Distribution", color="#dddddd", fontsize=11)
                ax1.tick_params(colors="#aaaaaa")
                ax1.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="#dddddd", fontsize=8)

                if "datetime" in cam_df.columns:
                    ax2 = self.figure.add_subplot(122)
                    ax2.set_facecolor("#1e1e1e")
                    dt = pd.to_datetime(cam_df["datetime"])
                    colors = ["#ff4444" if s < 0.5 else "#4488ff" for s in cam_df["prob_usable"].fillna(1)]
                    ax2.scatter(dt, cam_df["prob_usable"], s=2, alpha=0.4, c=colors)
                    ax2.axhline(0.5, color="#ff4444", linestyle="--", linewidth=1, alpha=0.5)
                    ax2.set_xlabel("Date", color="#dddddd")
                    ax2.set_ylabel("QA Score", color="#dddddd")
                    ax2.set_title("QA Score over Time", color="#dddddd", fontsize=11)
                    ax2.tick_params(colors="#aaaaaa")
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                    ax2.set_ylim(-0.05, 1.05)
            else:
                ax = self.figure.add_subplot(111)
                ax.set_facecolor("#1e1e1e")
                ax.text(0.5, 0.5, "Click 'Run QA on Camera' to compute QA scores",
                        ha="center", va="center", color="#888888", fontsize=13,
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

            self.figure.tight_layout()
            self.canvas.draw()

        if "qa_usable" in cam_df.columns:
            rejected = cam_df[cam_df["qa_usable"] == 0].sort_values(
                "prob_usable", ascending=True
            ).head(200)
        else:
            rejected = pd.DataFrame()

        self.table.setRowCount(len(rejected))
        for i, (_, row) in enumerate(rejected.iterrows()):
            self.table.setItem(i, 0, QTableWidgetItem(str(row.get("stem", ""))))
            self.table.setItem(i, 1, QTableWidgetItem(str(row.get("datetime", ""))[:19]))
            qa = row.get("prob_usable", float("nan"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{qa:.3f}" if pd.notna(qa) else "N/A"))
            btn = QPushButton("View")
            btn.clicked.connect(lambda checked, stem=row.get("stem"): self._goto_stem(stem))
            self.table.setCellWidget(i, 3, btn)

        self.table_label.setText(f"Rejected images ({n_rej} total, showing {len(rejected)}):")
        self._build_gallery(cam_df)

    def _goto_stem(self, stem):
        if self.viewer.cam_df is None:
            return
        match = self.viewer.cam_df[self.viewer.cam_df["stem"] == stem]
        if len(match) > 0:
            idx = match.index[0]
            self.viewer.tabs.setCurrentIndex(0)
            self.viewer._goto_frame(idx)

    def _on_row_double_clicked(self, row, col):
        stem_item = self.table.item(row, 0)
        if stem_item:
            self._goto_stem(stem_item.text())


# ── Ground Tab ──────────────────────────────────────────────────────────────

class GroundTab(QWidget):
    def __init__(self, parent_viewer):
        super().__init__()
        self.viewer = parent_viewer
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.summary_label = QLabel("Ground Coverage Summary: —")
        self.summary_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(self.summary_label)

        if HAS_MPL:
            self.figure = Figure(figsize=(12, 4), dpi=100)
            self.figure.patch.set_facecolor("#1e1e1e")
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas, stretch=1)

    def update_camera(self, cam_df):
        if cam_df is None or len(cam_df) == 0 or "ground_coverage" not in cam_df.columns:
            self.summary_label.setText("Ground Coverage: No data")
            return

        gc = cam_df["ground_coverage"].dropna()
        self.summary_label.setText(
            f"Ground Coverage: mean={gc.mean():.1%}, median={gc.median():.1%}, "
            f"std={gc.std():.1%} ({len(gc)} images)"
        )

        if not HAS_MPL:
            return

        self.figure.clear()
        df = cam_df.copy()
        df["dt"] = pd.to_datetime(df["datetime"])

        # Subplot 1: ground coverage over time
        ax1 = self.figure.add_subplot(211)
        ax1.set_facecolor("#1e1e1e")
        ax1.plot(df["dt"], df["ground_coverage"] * 100, color="#44ff44", alpha=0.3, linewidth=0.3)
        if len(df) > 20:
            rolling = df.set_index("dt")["ground_coverage"].rolling("7D").mean() * 100
            ax1.plot(rolling.index, rolling.values, color="#44ff44", linewidth=2)
        ax1.set_ylabel("Ground %", color="#44ff44")
        ax1.set_ylim(-2, 102)
        ax1.tick_params(colors="#aaaaaa")
        camera = df.iloc[0].get("camera", "")
        ax1.set_title(f"{camera} — Ground Coverage", color="#dddddd", fontsize=12)

        # Subplot 2: histogram
        ax2 = self.figure.add_subplot(212)
        ax2.set_facecolor("#1e1e1e")
        ax2.hist(gc * 100, bins=50, color="#44ff44", alpha=0.7, edgecolor="#22aa22")
        ax2.set_xlabel("Ground Coverage %", color="#dddddd")
        ax2.set_ylabel("Count", color="#dddddd")
        ax2.tick_params(colors="#aaaaaa")

        self.figure.tight_layout()
        self.canvas.draw()


# ── Snow Tab ────────────────────────────────────────────────────────────────

class SnowTab(QWidget):
    def __init__(self, parent_viewer):
        super().__init__()
        self.viewer = parent_viewer
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.summary_label = QLabel("Snow Analysis: —")
        self.summary_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(self.summary_label)

        if HAS_MPL:
            self.figure = Figure(figsize=(12, 5), dpi=100)
            self.figure.patch.set_facecolor("#1e1e1e")
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas, stretch=1)

    def update_camera(self, cam_df):
        if cam_df is None or len(cam_df) == 0:
            self.summary_label.setText("Snow Analysis: No data")
            return

        col = "student_snow_pct"
        if col not in cam_df.columns:
            self.summary_label.setText("Snow Analysis: no snow_pct column")
            return

        snow = cam_df[col].dropna()
        n_snow = (snow > 5).sum()
        self.summary_label.setText(
            f"Snow: mean={snow.mean():.1f}%, max={snow.max():.1f}%, "
            f"{n_snow}/{len(snow)} images with >5% snow"
        )

        if not HAS_MPL:
            return

        self.figure.clear()
        df = cam_df.copy()
        df["dt"] = pd.to_datetime(df["datetime"])
        df["hour_float"] = df["dt"].dt.hour + df["dt"].dt.minute / 60.0

        # Subplot 1: snow vs hour of day (check IR/night contamination)
        ax1 = self.figure.add_subplot(221)
        ax1.set_facecolor("#1e1e1e")
        ax1.scatter(df["hour_float"], df[col], s=1, alpha=0.2, color="#4488ff")
        ax1.set_xlabel("Hour of day", color="#dddddd")
        ax1.set_ylabel("Snow %", color="#dddddd")
        ax1.set_title("Snow % vs Time of Day", color="#dddddd", fontsize=10)
        ax1.set_xlim(0, 24)
        ax1.set_ylim(-2, 102)
        ax1.tick_params(colors="#aaaaaa")

        # Subplot 2: snow vs temperature
        ax2 = self.figure.add_subplot(222)
        ax2.set_facecolor("#1e1e1e")
        if "t2m" in df.columns:
            valid = df.dropna(subset=["t2m", col])
            ax2.scatter(valid["t2m"], valid[col], s=1, alpha=0.2, color="#4488ff")
            ax2.axvline(0, color="#ff4444", linestyle="--", alpha=0.5)
            ax2.set_xlabel("Temperature (°C)", color="#dddddd")
        ax2.set_ylabel("Snow %", color="#dddddd")
        ax2.set_title("Snow % vs Temperature", color="#dddddd", fontsize=10)
        ax2.tick_params(colors="#aaaaaa")

        # Subplot 3: monthly mean snow %
        ax3 = self.figure.add_subplot(223)
        ax3.set_facecolor("#1e1e1e")
        df["month"] = df["dt"].dt.month
        monthly = df.groupby("month")[col].mean()
        ax3.bar(monthly.index, monthly.values, color="#4488ff", alpha=0.7, edgecolor="#2266cc")
        ax3.set_xlabel("Month", color="#dddddd")
        ax3.set_ylabel("Mean Snow %", color="#dddddd")
        ax3.set_title("Monthly Mean Snow %", color="#dddddd", fontsize=10)
        ax3.set_xticks(range(1, 13))
        ax3.tick_params(colors="#aaaaaa")

        # Subplot 4: consecutive frame delta (temporal consistency)
        ax4 = self.figure.add_subplot(224)
        ax4.set_facecolor("#1e1e1e")
        deltas = df[col].diff().dropna()
        ax4.hist(deltas, bins=100, color="#ffaa44", alpha=0.7, edgecolor="#cc8833")
        ax4.set_xlabel("Frame-to-frame Snow % change", color="#dddddd")
        ax4.set_ylabel("Count", color="#dddddd")
        ax4.set_title("Temporal Consistency", color="#dddddd", fontsize=10)
        ax4.tick_params(colors="#aaaaaa")

        camera = df.iloc[0].get("camera", "")
        self.figure.suptitle(f"{camera} — Snow Analysis", color="#dddddd", fontsize=13, y=1.02)
        self.figure.tight_layout()
        self.canvas.draw()


# ── Main Window ─────────────────────────────────────────────────────────────

class SnowPipelineViewer(QMainWindow):

    def __init__(self, data_dir, timeseries_csv=None, models_dir=None,
                 ground_mask_dir=None, snow_mask_dir=None, smoother_ckpt=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.csv_path = Path(timeseries_csv) if timeseries_csv else TIMESERIES_CSV
        self.ground_mask_dir = Path(ground_mask_dir) if ground_mask_dir else GROUND_MASK_DIR
        self.snow_mask_dir = Path(snow_mask_dir) if snow_mask_dir else SNOW_MASK_DIR
        self.models_dir = Path(models_dir) if models_dir else MASTER / "models"

        self.df = None
        self.cam_df = None
        self.current_idx = 0
        self.playing = False
        self.play_speed = 1.0
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._advance_frame)

        self.show_ground = False
        self.show_snow = False
        self.ground_model = None
        self.snow_model = None
        self.multihead_model = None   # DINOv3MultiHead — replaces ground+snow when loaded
        self.multihead_cfg = None
        self.live_inference = False
        self.use_gpu = False
        self.demo_mode = False
        self._demo_timer = QTimer()
        self._demo_timer.timeout.connect(self._demo_step)
        self._demo_fps_times = []

        self.smoother_module = None
        self.smoothed_snow = None
        self._smoother_tag = None
        if smoother_ckpt and HAS_TORCH:
            self._load_smoother(smoother_ckpt)

        self._init_ui()
        self._load_data()

    def _load_smoother(self, ckpt_path):
        try:
            import sys
            models_parent = str(self.models_dir.parent)
            if models_parent not in sys.path:
                sys.path.insert(0, models_parent)
            from models.temporal_snow.train_smoother import TemporalSmootherModule
            self.smoother_module = TemporalSmootherModule.load_from_checkpoint(
                ckpt_path, map_location="cpu"
            )
            self.smoother_module.eval()
            hp = self.smoother_module.hparams
            arch = hp.get("arch", "?")
            hidden = hp.get("hidden_size", "?")
            layers = hp.get("num_layers", "?")
            bidir = hp.get("bidirectional", False)
            tag = f"{arch} h{hidden} l{layers}" + (" bi" if bidir else " uni")
            self._smoother_tag = tag
            # Load scaler if available (new standardized models)
            scaler_path = Path(ckpt_path).parent / "scaler.json"
            if scaler_path.exists():
                from models.temporal_snow.dataset import FeatureScaler
                self._smoother_scaler = FeatureScaler.load(scaler_path)
                print(f"Loaded smoother: {Path(ckpt_path).name} (with scaler)")
            else:
                self._smoother_scaler = None
                print(f"Loaded smoother: {Path(ckpt_path).name} (legacy normalization)")
            print(f"  {tag}")
        except Exception as e:
            print(f"WARNING: Failed to load smoother: {e}")
            self.smoother_module = None
            self._smoother_tag = None
            self._smoother_scaler = None

    def _build_smoother_menu(self):
        menu = self.smoother_menu
        menu.clear()

        none_action = QAction("None (disable)", self)
        none_action.triggered.connect(self._disable_smoother)
        menu.addAction(none_action)
        menu.addSeparator()

        ckpt_dir = self.models_dir / "temporal_snow" / "checkpoints"
        if ckpt_dir.is_dir():
            entries = []
            for d in sorted(ckpt_dir.iterdir()):
                if not d.is_dir() or not d.name.startswith("smoother_"):
                    continue
                best = sorted(d.glob("best-*.ckpt"))
                if not best:
                    continue
                ckpt = best[0]
                for c in best:
                    try:
                        mae_a = float(c.stem.split("val_mae=")[1])
                        mae_b = float(ckpt.stem.split("val_mae=")[1])
                        if mae_a < mae_b:
                            ckpt = c
                    except (IndexError, ValueError):
                        pass
                name = d.name.replace("smoother_", "")
                mae_val = 1.0
                try:
                    mae_val = float(ckpt.stem.split("val_mae=")[1])
                    label = f"{name}  (MAE {mae_val:.4f})"
                except (IndexError, ValueError):
                    label = name
                bidir = "_bi_" in d.name
                entries.append((bidir, mae_val, label, str(ckpt)))

            uni = [(l, p) for bi, m, l, p in sorted(entries, key=lambda x: x[1]) if not bi]
            bi = [(l, p) for bi, m, l, p in sorted(entries, key=lambda x: x[1]) if bi]

            if uni:
                menu.addSection("Unidirectional (causal)")
                for label, path in uni[:8]:
                    act = QAction(label, self)
                    act.triggered.connect(lambda checked, p=path: self._switch_smoother(p))
                    menu.addAction(act)

            if bi:
                menu.addSection("Bidirectional")
                for label, path in bi[:8]:
                    act = QAction(label, self)
                    act.triggered.connect(lambda checked, p=path: self._switch_smoother(p))
                    menu.addAction(act)

        menu.addSeparator()
        load_action = QAction("Load from file...", self)
        load_action.triggered.connect(self._load_smoother_dialog)
        menu.addAction(load_action)

    def _switch_smoother(self, ckpt_path):
        self._load_smoother(ckpt_path)
        if self.smoother_module is None:
            self.status.showMessage("ERROR: Failed to load smoother (check terminal)", 5000)
            return
        if self.cam_df is None or len(self.cam_df) == 0:
            self.status.showMessage("Smoother loaded but no camera data", 5000)
            return
        self.status.showMessage("Running smoother...", 0)
        QApplication.processEvents()
        self.smoothed_snow = self._run_smoother(self.cam_df)
        if self.smoothed_snow is None:
            self.status.showMessage("Smoother returned no results (check terminal)", 5000)
            return
        self.timeseries_tab.update_camera(self.cam_df, smoothed=self.smoothed_snow, kalman=self._run_kalman(self.cam_df))
        self._plot_mini_timeline()
        tag = getattr(self, "_smoother_tag", None) or Path(ckpt_path).parent.name
        self.status.showMessage(f"Smoother: {tag}", 5000)

    def _disable_smoother(self):
        self.smoother_module = None
        self._smoother_tag = None
        self.smoothed_snow = None
        if self.cam_df is not None:
            self.timeseries_tab.update_camera(self.cam_df, smoothed=None, kalman=self._run_kalman(self.cam_df))
            self._plot_mini_timeline()
        self.status.showMessage("Smoother disabled", 3000)

    def _load_smoother_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Temporal Smoother", str(self.models_dir / "temporal_snow"),
            "Checkpoints (*.ckpt);;All Files (*)"
        )
        if path:
            self._switch_smoother(path)

    def _run_smoother(self, cam_df):
        """Run temporal smoother on camera dataframe, return smoothed snow_pct series."""
        if self.smoother_module is None or not HAS_TORCH:
            return None
        try:
            import math
            from models.temporal_snow.dataset import (
                build_feature_columns, IMAGE_FEAT_COLS, SNOW_PCT_COLS
            )

            df = cam_df.copy()
            snow_pct_cols = [c for c in SNOW_PCT_COLS if c in df.columns]
            if not snow_pct_cols:
                return None

            df["dt"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("dt").reset_index(drop=True)

            # Cyclical time encoding
            df["hour_sin"] = np.sin(2 * math.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * math.pi * df["hour"] / 24)
            doy = df["dt"].dt.dayofyear
            df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
            df["doy_cos"] = np.cos(2 * math.pi * doy / 365)

            # Fill missing values
            FILL_DEFAULTS = {
                "t2m": 0, "visibility": 1000, "weather_code": 0,
                "brightness": 128, "contrast": 50, "edge_density": 10, "entropy": 4,
                "white_frac": 0, "low_sat_frac": 0, "blue_ratio": 0,
                "bottom_white_frac": 0, "bottom_brightness": 128,
                "snow_n_components": 0, "snow_largest_ratio": 0,
                "snow_centroid_y": 0.5, "ground_coverage": 0,
            }
            for col in snow_pct_cols:
                df[col] = df[col].fillna(0)
            for col, default in FILL_DEFAULTS.items():
                if col in df.columns:
                    df[col] = df[col].fillna(default)

            # Log-transform skewed features (must match training preprocessing)
            if "visibility" in df.columns:
                df["visibility"] = np.log1p(df["visibility"])
            if "snow_n_components" in df.columns:
                df["snow_n_components"] = np.log1p(df["snow_n_components"])

            scaler = getattr(self, "_smoother_scaler", None)

            if scaler is not None:
                # New standardized model — use scaler's feature_cols
                feature_cols = scaler.feature_cols
                df = scaler.transform(df)
            else:
                # Legacy model — apply hardcoded normalization
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

                has_img = all(c in df.columns for c in IMAGE_FEAT_COLS)
                input_size = self.smoother_module.hparams.get("input_size", None)
                feature_cols = None
                if input_size is not None:
                    for use_img in ([True, False] if has_img else [False]):
                        candidate = build_feature_columns(snow_pct_cols, use_image_features=use_img)
                        if len(candidate) == input_size:
                            feature_cols = candidate
                            break
                if feature_cols is None:
                    feature_cols = build_feature_columns(snow_pct_cols, use_image_features=has_img)

            input_size = self.smoother_module.hparams.get("input_size", None)
            if input_size is not None and len(feature_cols) != input_size:
                print(f"  Smoother input_size mismatch: model wants {input_size}, "
                      f"have {len(feature_cols)} — padding/truncating")
                if len(feature_cols) > input_size:
                    feature_cols = feature_cols[:input_size]

            x = df[feature_cols].values.astype(np.float32)
            x = np.nan_to_num(x, nan=0.0)

            if input_size is not None and x.shape[1] < input_size:
                pad = np.zeros((x.shape[0], input_size - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)

            with torch.no_grad():
                x_t = torch.from_numpy(x).unsqueeze(0)
                pred = self.smoother_module(x_t).squeeze(0).numpy()

            return pd.Series(pred * 100.0, index=df.index)
        except Exception as e:
            print(f"  Smoother inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _run_kalman(self, cam_df, Q=100.0, R=400.0):
        """Run Kalman filter on camera snow_pct. Returns numpy array."""
        if "student_snow_pct" not in cam_df.columns:
            return None
        try:
            series = cam_df.sort_values("datetime")["student_snow_pct"].fillna(0).values.astype(float)
            x = float(series[0])
            P = 500.0
            smoothed = []
            for z in series:
                # Predict
                P_pred = P + Q
                # Update
                K = P_pred / (P_pred + R)
                x = x + K * (z - x)
                P = (1 - K) * P_pred
                smoothed.append(np.clip(x, 0, 100))
            return np.array(smoothed)
        except Exception as e:
            print(f"  Kalman failed: {e}")
            return None

    def _init_ui(self):
        self.setWindowTitle("Snow Pipeline Viewer")
        self.setMinimumSize(1200, 750)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        add_cam_action = QAction("&Add Camera Directory...", self)
        add_cam_action.triggered.connect(self._add_camera_dir)
        file_menu.addAction(add_cam_action)

        load_csv_action = QAction("&Load Timeseries CSV...", self)
        load_csv_action.triggered.connect(self._load_csv_dialog)
        file_menu.addAction(load_csv_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        inference_menu = menubar.addMenu("&Inference")
        load_multihead_action = QAction("Load &Multihead (DINO) Model...", self)
        load_multihead_action.triggered.connect(self._load_multihead_model)
        inference_menu.addAction(load_multihead_action)

        inference_menu.addSeparator()

        load_ground_action = QAction("Load &Ground Model...", self)
        load_ground_action.triggered.connect(lambda: self._load_model("ground"))
        inference_menu.addAction(load_ground_action)

        load_snow_action = QAction("Load &Snow Model...", self)
        load_snow_action.triggered.connect(lambda: self._load_model("snow"))
        inference_menu.addAction(load_snow_action)

        self.live_infer_action = QAction("&Live Inference", self)
        self.live_infer_action.setCheckable(True)
        self.live_infer_action.toggled.connect(self._toggle_live_inference)
        inference_menu.addAction(self.live_infer_action)

        infer_current_action = QAction("Infer &Current Frame", self)
        infer_current_action.setShortcut(QKeySequence("Ctrl+I"))
        infer_current_action.triggered.connect(self._infer_current)
        inference_menu.addAction(infer_current_action)

        inference_menu.addSeparator()
        self.gpu_action = QAction("Use &GPU", self)
        self.gpu_action.setCheckable(True)
        self.gpu_action.toggled.connect(self._toggle_gpu)
        inference_menu.addAction(self.gpu_action)

        self.demo_action = QAction("&Demo Mode (Live FPS)", self)
        self.demo_action.setCheckable(True)
        self.demo_action.setShortcut(QKeySequence("Ctrl+D"))
        self.demo_action.toggled.connect(self._toggle_demo_mode)
        inference_menu.addAction(self.demo_action)

        batch_infer_action = QAction("&Batch Inference (all frames)...", self)
        batch_infer_action.setShortcut(QKeySequence("Ctrl+B"))
        batch_infer_action.triggered.connect(self._batch_inference)
        inference_menu.addAction(batch_infer_action)

        # Smoother submenu
        inference_menu.addSeparator()
        self.smoother_menu = inference_menu.addMenu("&Temporal Smoother")
        self._build_smoother_menu()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Top bar: camera selector + date range
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(120)
        self.camera_combo.currentTextChanged.connect(self._on_camera_changed)
        top_bar.addWidget(self.camera_combo)

        top_bar.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.dateChanged.connect(self._on_date_filter_changed)
        top_bar.addWidget(self.date_from)

        top_bar.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.dateChanged.connect(self._on_date_filter_changed)
        top_bar.addWidget(self.date_to)

        top_bar.addStretch()

        self.ground_cb = QCheckBox("Ground")
        self.ground_cb.setStyleSheet("color: #44ff44;")
        self.ground_cb.toggled.connect(self._on_overlay_changed)
        top_bar.addWidget(self.ground_cb)

        self.snow_cb = QCheckBox("Snow")
        self.snow_cb.setStyleSheet("color: #4488ff;")
        self.snow_cb.toggled.connect(self._on_overlay_changed)
        top_bar.addWidget(self.snow_cb)

        self.skip_rejected_cb = QCheckBox("Skip rejected")
        self.skip_rejected_cb.setStyleSheet("color: #ff8844;")
        self.skip_rejected_cb.setToolTip("Skip QA-rejected frames during playback/navigation")
        top_bar.addWidget(self.skip_rejected_cb)

        main_layout.addLayout(top_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { padding: 6px 16px; font-size: 12px; }
            QTabBar::tab:selected { background: #3a3a3a; color: white; }
        """)

        # Tab 0: Viewer
        viewer_widget = QWidget()
        viewer_layout = QVBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        viewer_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.image_display = ImageDisplay()
        viewer_splitter.addWidget(self.image_display)
        self.info_panel = InfoPanel()
        viewer_splitter.addWidget(self.info_panel)
        viewer_splitter.setSizes([800, 300])
        viewer_layout.addWidget(viewer_splitter, stretch=3)

        # Mini timeline below image (raw + smoothed snow_pct)
        if HAS_MPL:
            self.mini_figure = Figure(figsize=(12, 1.8), dpi=100)
            self.mini_figure.patch.set_facecolor("#1e1e1e")
            self.mini_canvas = FigureCanvas(self.mini_figure)
            self.mini_canvas.setMaximumHeight(140)
            self.mini_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.mini_canvas.mpl_connect("button_press_event", self._on_mini_click)
            viewer_layout.addWidget(self.mini_canvas)
            self._mini_cursor = None

        self.tabs.addTab(viewer_widget, "Viewer")

        # Tab 1: Timeseries
        self.timeseries_tab = TimeseriesTab(self)
        self.tabs.addTab(self.timeseries_tab, "Timeseries")

        # Tab 2: Snow
        self.snow_tab = SnowTab(self)
        self.tabs.addTab(self.snow_tab, "Snow")

        # Tab 3: Ground
        self.ground_tab = GroundTab(self)
        self.tabs.addTab(self.ground_tab, "Ground")

        # Tab 4: QA
        self.qa_tab = QATab(self)
        self.tabs.addTab(self.qa_tab, "QA")

        main_layout.addWidget(self.tabs, stretch=1)

        # Timeline slider
        slider_layout = QHBoxLayout()
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(100)
        slider_layout.addWidget(self.frame_label)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.timeline_slider, stretch=1)

        self.timestamp_label = QLabel("")
        self.timestamp_label.setMinimumWidth(160)
        slider_layout.addWidget(self.timestamp_label)

        main_layout.addLayout(slider_layout)

        # Playback controls
        controls = QHBoxLayout()

        for text, width, slot in [
            ("|◀", 40, lambda: self._goto_frame(0)),
            ("◀◀", 40, lambda: self._step(-10)),
            ("◀", 40, lambda: self._step(-1)),
        ]:
            btn = QPushButton(text)
            btn.setFixedWidth(width)
            btn.clicked.connect(slot)
            controls.addWidget(btn)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(60)
        self.btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self.btn_play)

        for text, width, slot in [
            ("▶", 40, lambda: self._step(1)),
            ("▶▶", 40, lambda: self._step(10)),
            ("▶|", 40, lambda: self._goto_frame(len(self.cam_df) - 1 if self.cam_df is not None else 0)),
        ]:
            btn = QPushButton(text)
            btn.setFixedWidth(width)
            btn.clicked.connect(slot)
            controls.addWidget(btn)

        controls.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x", "8x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        controls.addWidget(self.speed_combo)

        controls.addStretch()
        main_layout.addLayout(controls)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._setup_shortcuts()

    def _setup_shortcuts(self):
        from PyQt6.QtGui import QShortcut
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(1))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-1))
        QShortcut(QKeySequence("Shift+Right"), self, lambda: self._step(10))
        QShortcut(QKeySequence("Shift+Left"), self, lambda: self._step(-10))
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, lambda: self._goto_frame(0))
        QShortcut(QKeySequence(Qt.Key.Key_End), self,
                  lambda: self._goto_frame(len(self.cam_df) - 1 if self.cam_df is not None else 0))
        QShortcut(QKeySequence(Qt.Key.Key_G), self, lambda: self.ground_cb.toggle())
        QShortcut(QKeySequence(Qt.Key.Key_S), self, lambda: self.snow_cb.toggle())
        QShortcut(QKeySequence("1"), self, lambda: self.tabs.setCurrentIndex(0))
        QShortcut(QKeySequence("2"), self, lambda: self.tabs.setCurrentIndex(1))
        QShortcut(QKeySequence("3"), self, lambda: self.tabs.setCurrentIndex(2))
        QShortcut(QKeySequence("4"), self, lambda: self.tabs.setCurrentIndex(3))
        QShortcut(QKeySequence("5"), self, lambda: self.tabs.setCurrentIndex(4))

    # ── Data Loading ────────────────────────────────────────────────────────

    def _load_data(self):
        if self.csv_path.exists():
            self.status.showMessage(f"Loading {self.csv_path.name}...")
            QApplication.processEvents()
            self.df = pd.read_csv(self.csv_path)
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])
            cameras = sorted(self.df["camera"].unique())
            self.camera_combo.blockSignals(True)
            self.camera_combo.clear()
            self.camera_combo.addItems(cameras)
            self.camera_combo.blockSignals(False)
            if cameras:
                self.camera_combo.setCurrentText(cameras[0])
                self._on_camera_changed(cameras[0])
            self.status.showMessage(f"Loaded {len(self.df)} rows, {len(cameras)} cameras")
        else:
            self._scan_data_dir()

    def _scan_data_dir(self):
        cameras = []
        if self.data_dir.exists():
            for d in sorted(self.data_dir.iterdir()):
                if d.is_dir() and list(d.glob("*.jpg"))[:1]:
                    cameras.append(d.name)
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItems(cameras)
        self.camera_combo.blockSignals(False)
        if cameras:
            self._on_camera_changed(cameras[0])
        self.status.showMessage(f"Scanned {len(cameras)} cameras (no CSV loaded)")

    def _on_camera_changed(self, camera):
        if not camera:
            return
        if self.df is not None and camera in self.df["camera"].values:
            mask = (self.df["camera"] == camera) & ~self.df["stem"].str.contains("_IR_|_VIS_", na=False)
            self.cam_df = self.df[mask].sort_values("datetime").reset_index(drop=True)
        else:
            cam_dir = self.data_dir / camera
            if not cam_dir.exists():
                self.cam_df = pd.DataFrame(columns=["camera", "stem", "datetime", "hour"])
            else:
                images = sorted(cam_dir.glob("*.jpg"))
                rows = []
                for img in images:
                    stem = img.stem
                    if "_IR_" in stem or "_VIS_" in stem:
                        continue
                    parts = stem.split("_")
                    if len(parts) >= 2 and len(parts[1]) >= 12:
                        try:
                            dt = pd.to_datetime(parts[1], format="%Y%m%d%H%M")
                        except ValueError:
                            dt = pd.NaT
                    else:
                        dt = pd.NaT
                    rows.append({"camera": camera, "stem": stem, "datetime": dt,
                                 "hour": dt.hour if pd.notna(dt) else 12})
                self.cam_df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

        n = len(self.cam_df)
        self.timeline_slider.setMaximum(max(0, n - 1))
        self.frame_label.setText(f"0 / {n}")

        if n > 0 and "datetime" in self.cam_df.columns:
            dates = self.cam_df["datetime"].dropna()
            if len(dates) > 0:
                d_min, d_max = dates.min(), dates.max()
                self.date_from.blockSignals(True)
                self.date_to.blockSignals(True)
                self.date_from.setDate(QDate(d_min.year, d_min.month, d_min.day))
                self.date_to.setDate(QDate(d_max.year, d_max.month, d_max.day))
                self.date_from.blockSignals(False)
                self.date_to.blockSignals(False)

        # Run smoother if available
        self.smoothed_snow = self._run_smoother(self.cam_df)

        # Update all tabs + mini timeline
        self.timeseries_tab.update_camera(self.cam_df, smoothed=self.smoothed_snow, kalman=self._run_kalman(self.cam_df))
        self.snow_tab.update_camera(self.cam_df)
        self.ground_tab.update_camera(self.cam_df)
        self.qa_tab.update_camera(self.cam_df)
        self._plot_mini_timeline()

        self._goto_frame(0)

    def _on_date_filter_changed(self):
        if self.df is None or self.cam_df is None:
            return
        camera = self.camera_combo.currentText()
        if not camera:
            return

        d_from = self.date_from.date().toPyDate()
        d_to = self.date_to.date().toPyDate()

        cam_all = self.df[(self.df["camera"] == camera) & ~self.df["stem"].str.contains("_IR_|_VIS_", na=False)].copy()
        cam_all["datetime"] = pd.to_datetime(cam_all["datetime"])
        mask = (cam_all["datetime"].dt.date >= d_from) & (cam_all["datetime"].dt.date <= d_to)
        self.cam_df = cam_all[mask].sort_values("datetime").reset_index(drop=True)

        n = len(self.cam_df)
        self.timeline_slider.setMaximum(max(0, n - 1))
        self.frame_label.setText(f"0 / {n}")

        self.smoothed_snow = self._run_smoother(self.cam_df)
        self.timeseries_tab.update_camera(self.cam_df, smoothed=self.smoothed_snow, kalman=self._run_kalman(self.cam_df))
        self.snow_tab.update_camera(self.cam_df)
        self.ground_tab.update_camera(self.cam_df)
        self.qa_tab.update_camera(self.cam_df)
        self._plot_mini_timeline()

        self._goto_frame(0)

    # ── Playback ────────────────────────────────────────────────────────────

    def _on_overlay_changed(self):
        self.show_ground = self.ground_cb.isChecked()
        self.show_snow = self.snow_cb.isChecked()
        self._display_current_frame()

    def _on_slider_changed(self, value):
        if self.cam_df is not None and 0 <= value < len(self.cam_df):
            self.current_idx = value
            self._display_current_frame()
            self.timeseries_tab.update_cursor(value)
            self._update_mini_cursor(value)

    def _on_speed_changed(self, text):
        self.play_speed = float(text.replace("x", ""))
        if self.playing:
            self.play_timer.setInterval(int(200 / self.play_speed))

    def _plot_mini_timeline(self):
        if not HAS_MPL or not hasattr(self, "mini_figure"):
            return
        self.mini_figure.clear()
        self._mini_cursor = None
        if self.cam_df is None or len(self.cam_df) == 0:
            self.mini_canvas.draw()
            return
        df = self.cam_df
        if "datetime" not in df.columns or "student_snow_pct" not in df.columns:
            self.mini_canvas.draw()
            return

        dt = pd.to_datetime(df["datetime"])
        ax = self.mini_figure.add_subplot(111)
        ax.set_facecolor("#1e1e1e")
        ax.fill_between(dt, df["student_snow_pct"], alpha=0.2, color="#4488ff")
        ax.plot(dt, df["student_snow_pct"], color="#4488ff", alpha=0.4, linewidth=0.3)

        if self.smoothed_snow is not None and len(self.smoothed_snow) == len(df):
            ax.plot(dt, self.smoothed_snow.values, color="#44ff88", linewidth=1.5,
                    alpha=0.9, label="Smoother")

        ax.set_ylim(-2, 102)
        ax.set_ylabel("Snow%", color="#aaaaaa", fontsize=8)
        ax.tick_params(axis="y", labelcolor="#777777", labelsize=7)
        ax.tick_params(axis="x", labelcolor="#777777", labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        for spine in ax.spines.values():
            spine.set_color("#333333")
        self.mini_figure.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.25)
        self.mini_canvas.draw()

    def _update_mini_cursor(self, idx):
        if not HAS_MPL or not hasattr(self, "mini_figure"):
            return
        if self.cam_df is None or idx >= len(self.cam_df):
            return
        dt = pd.to_datetime(self.cam_df.iloc[idx].get("datetime"))
        if pd.isna(dt):
            return
        ax = self.mini_figure.axes[0] if self.mini_figure.axes else None
        if ax is None:
            return
        if self._mini_cursor is not None:
            self._mini_cursor.set_xdata([dt, dt])
        else:
            self._mini_cursor = ax.axvline(dt, color="#ff6666", linewidth=1.5, alpha=0.8, zorder=10)
        self.mini_canvas.draw_idle()

    def _on_mini_click(self, event):
        if event.inaxes is None or self.cam_df is None:
            return
        try:
            click_dt = mdates.num2date(event.xdata).replace(tzinfo=None)
        except Exception:
            return
        df = self.cam_df.copy()
        df["dt"] = pd.to_datetime(df["datetime"])
        diffs = (df["dt"] - click_dt).abs()
        nearest_idx = diffs.idxmin()
        self._goto_frame(nearest_idx)

    def _find_valid_idx(self, idx, direction=1):
        if self.cam_df is None or not self.skip_rejected_cb.isChecked():
            return idx
        if "qa_usable" not in self.cam_df.columns:
            return idx
        n = len(self.cam_df)
        while 0 <= idx < n:
            val = self.cam_df.iloc[idx].get("qa_usable", 1)
            if pd.isna(val) or int(val) == 1:
                return idx
            idx += direction
        return -1

    def _goto_frame(self, idx):
        if self.cam_df is None:
            return
        idx = max(0, min(idx, len(self.cam_df) - 1))
        valid = self._find_valid_idx(idx, direction=1)
        if valid < 0:
            valid = self._find_valid_idx(idx, direction=-1)
        if valid >= 0:
            idx = valid
        self.current_idx = idx
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(idx)
        self.timeline_slider.blockSignals(False)
        self._display_current_frame()
        self.timeseries_tab.update_cursor(idx)
        self._update_mini_cursor(idx)

    def _step(self, delta):
        if self.cam_df is None:
            return
        candidate = self.current_idx + delta
        if self.skip_rejected_cb.isChecked() and "qa_usable" in self.cam_df.columns:
            candidate = self._find_valid_idx(candidate, direction=1 if delta > 0 else -1)
            if candidate < 0:
                return
        self._goto_frame(candidate)

    def _toggle_play(self):
        if self.playing:
            self.playing = False
            self.play_timer.stop()
            self.btn_play.setText("▶")
        else:
            self.playing = True
            self.btn_play.setText("⏸")
            self.play_timer.setInterval(int(200 / self.play_speed))
            self.play_timer.start()

    def _advance_frame(self):
        if self.cam_df is None:
            self._toggle_play()
            return
        nxt = self._find_valid_idx(self.current_idx + 1, direction=1)
        if nxt < 0 or nxt >= len(self.cam_df):
            self._toggle_play()
            return
        self._goto_frame(nxt)

    def _display_current_frame(self):
        if self.cam_df is None or len(self.cam_df) == 0:
            self.image_display.setText("No frames")
            return

        idx = self.current_idx
        row = self.cam_df.iloc[idx]
        camera = row["camera"]
        stem = row["stem"]

        self.frame_label.setText(f"{idx + 1} / {len(self.cam_df)}")
        dt = row.get("datetime", "")
        self.timestamp_label.setText(str(dt)[:19] if pd.notna(dt) else "")

        img_path = self.data_dir / camera / f"{stem}.jpg"
        if not img_path.exists():
            self.image_display.setText(f"Image not found:\n{img_path.name}")
            self.info_panel.update_info(row.to_dict() if hasattr(row, "to_dict") else None)
            return

        try:
            pixmap = load_image_as_qpixmap(img_path)
        except Exception as e:
            self.image_display.setText(f"Error loading image:\n{e}")
            return

        if self.live_inference and self.multihead_model is not None:
            out = self._multihead_infer(img_path)
            if out is not None:
                gr_mask, sn_mask, qa_prob = out
                if self.show_ground:
                    pixmap = overlay_mask_fast(pixmap, gr_mask, (0, 255, 0), alpha=0.35)
                if self.show_snow:
                    pixmap = overlay_mask_fast(pixmap, sn_mask, (68, 136, 255), alpha=0.45)
                verdict = "OK" if qa_prob >= 0.5 else "REJECT"
                self.status.showMessage(f"DINO live │ QA={qa_prob:.3f} ({verdict})")
        else:
            if self.show_ground:
                ground_path = self.ground_mask_dir / camera / f"{stem}_ground.png"
                ground_mask = load_mask_as_array(ground_path)
                if ground_mask is not None:
                    vr = self._get_valid_region(camera, ground_mask.shape[0], ground_mask.shape[1])
                    if vr is not None:
                        ground_mask[vr <= 127] = 0
                    pixmap = overlay_mask_fast(pixmap, ground_mask, (0, 255, 0), alpha=0.35)
                else:
                    self.status.showMessage(f"Ground mask not found: {ground_path}")

            if self.show_snow:
                snow_path = self.snow_mask_dir / camera / f"{stem}_snow.png"
                snow_mask = load_mask_as_array(snow_path)
                if snow_mask is not None:
                    vr = self._get_valid_region(camera, snow_mask.shape[0], snow_mask.shape[1])
                    if vr is not None:
                        snow_mask[vr <= 127] = 0
                    pixmap = overlay_mask_fast(pixmap, snow_mask, (68, 136, 255), alpha=0.45)
                else:
                    self.status.showMessage(f"Snow mask not found: {snow_path}")

        self.image_display.set_pixmap_scaled(pixmap)
        smoothed_val = None
        if self.smoothed_snow is not None and idx < len(self.smoothed_snow):
            smoothed_val = self.smoothed_snow.iloc[idx]
        self.info_panel.update_info(
            row.to_dict() if hasattr(row, "to_dict") else None,
            smoothed_val=smoothed_val,
        )

    # ── File Operations ─────────────────────────────────────────────────────

    def _add_camera_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Camera Image Directory", str(self.data_dir)
        )
        if not dir_path:
            return
        cam_name = Path(dir_path).name
        images = list(Path(dir_path).glob("*.jpg"))
        if not images:
            QMessageBox.warning(self, "No Images", f"No .jpg files found in {dir_path}")
            return

        target = self.data_dir / cam_name
        if not target.exists() and Path(dir_path) != target:
            try:
                target.symlink_to(dir_path)
            except OSError:
                pass

        if self.camera_combo.findText(cam_name) == -1:
            self.camera_combo.addItem(cam_name)
        self.camera_combo.setCurrentText(cam_name)
        self.status.showMessage(f"Added camera {cam_name} ({len(images)} images)")

    def _load_csv_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Timeseries CSV", str(self.csv_path.parent), "CSV (*.csv)"
        )
        if path:
            self.csv_path = Path(path)
            self._load_data()

    # ── Inference ───────────────────────────────────────────────────────────

    @property
    def infer_device(self):
        import torch
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _toggle_gpu(self, enabled):
        import torch
        if enabled and not torch.cuda.is_available():
            QMessageBox.warning(self, "No GPU", "CUDA not available. Staying on CPU.")
            self.gpu_action.setChecked(False)
            return
        self.use_gpu = enabled
        dev = self.infer_device
        for model in [self.ground_model, self.snow_model, self.multihead_model]:
            if model is not None:
                model.to(dev)
        self.status.showMessage(f"Inference device: {dev}")

    def _load_model(self, model_type):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {model_type.title()} Model Checkpoint",
            str(self.models_dir), "Checkpoint (*.ckpt)"
        )
        if not path:
            return

        self.status.showMessage(f"Loading {model_type} model...")
        QApplication.processEvents()

        try:
            import torch
            models_path = str(self.models_dir)
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
            from student_ground.train import StudentSegmentationModule

            module = StudentSegmentationModule.load_from_checkpoint(path, map_location="cpu")
            module.eval()
            module.to(self.infer_device)

            if model_type == "ground":
                self.ground_model = module
            else:
                self.snow_model = module

            self.status.showMessage(f"{model_type.title()} model loaded: {Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Model Load Error", str(e))
            self.status.showMessage(f"Failed to load {model_type} model")

    def _load_multihead_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Multihead (DINO) Checkpoint",
            str(self.models_dir / "multi_head"), "Checkpoint (*.pt *.ckpt)"
        )
        if not path:
            return

        self.status.showMessage("Loading multihead model (DINO backbone — may take 30s)...")
        QApplication.processEvents()

        try:
            import torch
            models_parent = str(self.models_dir.parent)
            if models_parent not in sys.path:
                sys.path.insert(0, models_parent)

            models_parent = str(self.models_dir.parent)
            models_dir_str = str(self.models_dir)
            for p in [models_parent, models_dir_str]:
                if p not in sys.path:
                    sys.path.insert(0, p)
            from models.multi_head.train_multihead import DINOv3MultiHead

            # Inline configs — avoids student_ground.configs import on local PC
            _MODEL_REGISTRY = {
                "dinov3_base": {
                    "backbone": "facebook/dinov2-base",
                    "hidden_dim": 768, "patch_size": 14, "input_size": 518,
                    "extract_layers": [3, 6, 9, 11],
                },
                "dinov3_large": {
                    "backbone": "facebook/dinov2-large",
                    "hidden_dim": 1024, "patch_size": 14, "input_size": 518,
                    "extract_layers": [6, 12, 18, 23],
                },
            }

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
                state = {k.removeprefix("model."): v for k, v in state.items()}
                variant = ckpt.get("hyper_parameters", {}).get("variant", "dinov3_base")
            else:
                state = ckpt.get("state_dict", ckpt)
                variant = "dinov3_base"

            cfg = _MODEL_REGISTRY[variant]
            model = DINOv3MultiHead(variant=variant)
            # TrialModel saves backbone.layer.* but DINOv3MultiHead expects backbone.model.layer.*
            # Remap unconditionally when checkpoint lacks the .model. level
            if any(k.startswith("backbone.layer.") for k in state):
                state = {
                    ("backbone.model." + k[len("backbone."):] if k.startswith("backbone.layer.") else k): v
                    for k, v in state.items()
                }
            model.load_state_dict(state, strict=True)
            model.to(self.infer_device)
            model.eval()

            self.multihead_model = model
            self.multihead_cfg = cfg
            self.ground_model = None
            self.snow_model = None

            self.status.showMessage(
                f"Multihead model loaded ({variant}): {Path(path).name} — "
                f"ground+snow+QA in one pass"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Multihead Load Error", str(e))
            self.status.showMessage("Failed to load multihead model")

    def _multihead_infer(self, img_path):
        """Run DINOv3MultiHead on one image. Returns (ground_mask, snow_mask, qa_prob) or None."""
        if self.multihead_model is None or not HAS_TORCH:
            return None
        try:
            import torch
            from PIL import Image as PILImage
            from torchvision import transforms

            input_size = self.multihead_cfg["input_size"]
            img = PILImage.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            tfm = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            tensor = tfm(img).unsqueeze(0).to(self.infer_device)

            with torch.no_grad():
                qa_logit, gr_logits, sn_logits = self.multihead_model.forward_all(
                    tensor, output_size=(orig_h, orig_w)
                )
                qa_prob = torch.sigmoid(qa_logit).item()
                gr_mask = (torch.sigmoid(gr_logits).squeeze().cpu().numpy() >= 0.5).astype(np.uint8) * 255
                sn_mask = (torch.sigmoid(sn_logits).squeeze().cpu().numpy() >= 0.5).astype(np.uint8) * 255

            return gr_mask, sn_mask, qa_prob
        except Exception as e:
            print(f"Multihead inference error: {e}")
            import traceback; traceback.print_exc()
            return None

    def _toggle_live_inference(self, enabled):
        self.live_inference = enabled
        if enabled:
            if self.ground_model is None and self.snow_model is None and self.multihead_model is None:
                QMessageBox.information(self, "No Models",
                                        "Load ground/snow models or multihead model first via Inference menu.")
                self.live_infer_action.setChecked(False)
                return
            self.status.showMessage("Live inference ON")
        else:
            self.status.showMessage("Live inference OFF")

    def _infer_current(self):
        if self.cam_df is None or len(self.cam_df) == 0:
            return

        row = self.cam_df.iloc[self.current_idx]
        camera = row["camera"]
        stem = row["stem"]
        img_path = self.data_dir / camera / f"{stem}.jpg"

        if not img_path.exists():
            self.status.showMessage("Image not found")
            return

        if self.ground_model is None and self.snow_model is None and self.multihead_model is None:
            QMessageBox.information(self, "No Models", "Load a model first.")
            return

        dev = self.infer_device
        self.status.showMessage(f"Running inference on {dev}...")
        QApplication.processEvents()

        try:
            import time
            t0 = time.perf_counter()
            pixmap = load_image_as_qpixmap(img_path)

            if self.multihead_model is not None:
                out = self._multihead_infer(img_path)
                if out is not None:
                    gr_mask, sn_mask, qa_prob = out
                    if self.show_ground:
                        pixmap = overlay_mask_fast(pixmap, gr_mask, (0, 255, 0), alpha=0.35)
                    if self.show_snow:
                        pixmap = overlay_mask_fast(pixmap, sn_mask, (68, 136, 255), alpha=0.45)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self.image_display.set_pixmap_scaled(pixmap)
                    verdict = "OK" if qa_prob >= 0.5 else "REJECT"
                    self.status.showMessage(
                        f"Multihead — {dev} — {elapsed_ms:.0f} ms │ QA={qa_prob:.3f} ({verdict})"
                    )
            else:
                import torch
                models_path = str(self.models_dir)
                if models_path not in sys.path:
                    sys.path.insert(0, models_path)
                from student_snow.inference import load_and_preprocess

                results = {}
                for name, model in [("ground", self.ground_model), ("snow", self.snow_model)]:
                    if model is None:
                        continue
                    tensor = load_and_preprocess(img_path, model.input_size).to(dev)
                    with torch.no_grad():
                        logits = model(tensor)
                        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                        results[name] = (prob >= 0.5).astype(np.uint8) * 255
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if "ground" in results:
                    pixmap = overlay_mask_fast(pixmap, results["ground"], (0, 255, 0), alpha=0.35)
                if "snow" in results:
                    pixmap = overlay_mask_fast(pixmap, results["snow"], (68, 136, 255), alpha=0.45)
                self.image_display.set_pixmap_scaled(pixmap)
                self.status.showMessage(f"Inference complete — {dev} — {elapsed_ms:.0f} ms")

        except Exception as e:
            self.status.showMessage(f"Inference error: {e}")
            QMessageBox.warning(self, "Inference Error", str(e))

    CAMERA_MASK_BOTTOM = {"CM00009": 130}
    CAMERA_VALID_REGIONS = {"CM00075", "CM00108"}
    _valid_region_cache = {}

    def _get_valid_region(self, camera, h, w):
        """Load and cache static valid-region mask, resized to (h, w)."""
        key = (camera, h, w)
        if key in self._valid_region_cache:
            return self._valid_region_cache[key]
        if camera not in self.CAMERA_VALID_REGIONS:
            return None
        vr_path = (self.models_dir.parent / "preprocessing" / "segmentation"
                    / "camera_valid_regions" / f"{camera}.png")
        if not vr_path.exists():
            return None
        from PIL import Image as PILImage
        vr = np.array(PILImage.open(vr_path).convert("L").resize((w, h), PILImage.NEAREST))
        self._valid_region_cache[key] = vr
        return vr

    def _infer_on_pixmap(self, pixmap, img_path):
        try:
            import torch, time
            models_path = str(self.models_dir)
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
            from student_snow.inference import load_and_preprocess

            camera = Path(img_path).parent.name
            dev = self.infer_device
            t0 = time.perf_counter()
            for name, model in [("ground", self.ground_model), ("snow", self.snow_model)]:
                if model is None:
                    continue
                tensor = load_and_preprocess(img_path, model.input_size).to(dev)
                with torch.no_grad():
                    logits = model(tensor)
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                    mask = (prob >= 0.5).astype(np.uint8) * 255
                bottom_px = self.CAMERA_MASK_BOTTOM.get(camera, 0)
                if bottom_px > 0:
                    scale = mask.shape[0] / pixmap.height()
                    mask[-int(bottom_px * scale):, :] = 0
                vr = self._get_valid_region(camera, mask.shape[0], mask.shape[1])
                if vr is not None:
                    mask[vr <= 127] = 0
                color = (0, 255, 0) if name == "ground" else (68, 136, 255)
                alpha = 0.35 if name == "ground" else 0.45
                pixmap = overlay_mask_fast(pixmap, mask, color, alpha)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self.image_display.set_pixmap_scaled(pixmap)
            self.status.showMessage(f"Inference — {dev} — {elapsed_ms:.0f} ms")
        except Exception as e:
            self.image_display.set_pixmap_scaled(pixmap)
            self.status.showMessage(f"Inference error: {e}")

    # ── Batch Inference ─────────────────────────────────────────────────────

    def _batch_inference(self):
        if self.ground_model is None and self.snow_model is None:
            QMessageBox.information(self, "No Models",
                                    "Load ground and/or snow models first via Inference menu.")
            return
        if self.cam_df is None or len(self.cam_df) == 0:
            QMessageBox.information(self, "No Data", "Load a camera first.")
            return

        import torch, time
        models_path = str(self.models_dir)
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        from student_snow.inference import load_and_preprocess
        from PIL import Image as PILImage

        n = len(self.cam_df)
        camera = self.cam_df.iloc[0]["camera"]
        dev = self.infer_device

        reply = QMessageBox.question(
            self, "Batch Inference",
            f"Run inference on {n} frames for {camera}?\n"
            f"Device: {dev}\nThis may take a while.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        snow_pcts = []
        ground_covs = []
        t0 = time.perf_counter()

        for i in range(n):
            row = self.cam_df.iloc[i]
            stem = row["stem"]
            img_path = self.data_dir / camera / f"{stem}.jpg"

            if not img_path.exists():
                snow_pcts.append(float("nan"))
                ground_covs.append(float("nan"))
                continue

            try:
                ground_mask = None
                snow_mask = None
                img_h, img_w = PILImage.open(img_path).size[::-1]

                for name, model in [("ground", self.ground_model), ("snow", self.snow_model)]:
                    if model is None:
                        continue
                    tensor = load_and_preprocess(img_path, model.input_size).to(dev)
                    with torch.no_grad():
                        logits = model(tensor)
                        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                    mask = (prob >= 0.5).astype(np.uint8)
                    mask_full = np.array(
                        PILImage.fromarray(mask * 255).resize((img_w, img_h), PILImage.NEAREST)
                    ) // 255
                    bottom_px = self.CAMERA_MASK_BOTTOM.get(camera, 0)
                    if bottom_px > 0:
                        mask_full[-bottom_px:, :] = 0
                    vr = self._get_valid_region(camera, img_h, img_w)
                    if vr is not None:
                        mask_full[vr <= 127] = 0
                    if name == "ground":
                        ground_mask = mask_full
                    else:
                        snow_mask = mask_full

                total_px = img_h * img_w
                ground_px = int(ground_mask.sum()) if ground_mask is not None else total_px
                ground_cov = ground_px / total_px if total_px > 0 else 0.0

                if snow_mask is not None and ground_px > 0:
                    snow_in_ground = (snow_mask & ground_mask).sum() if ground_mask is not None else snow_mask.sum()
                    snow_pct = 100.0 * snow_in_ground / ground_px
                else:
                    snow_pct = 0.0

                snow_pcts.append(snow_pct)
                ground_covs.append(ground_cov)

            except Exception as e:
                print(f"  Batch skip {stem}: {e}")
                snow_pcts.append(float("nan"))
                ground_covs.append(float("nan"))

            if (i + 1) % 50 == 0 or i == n - 1:
                elapsed = time.perf_counter() - t0
                fps = (i + 1) / elapsed
                eta = (n - i - 1) / fps if fps > 0 else 0
                self.status.showMessage(
                    f"Batch inference: {i+1}/{n} ({fps:.1f} fps, ETA {eta:.0f}s)")
                QApplication.processEvents()

        self.cam_df["student_snow_pct"] = snow_pcts
        self.cam_df["ground_coverage"] = ground_covs

        elapsed = time.perf_counter() - t0
        self.status.showMessage(f"Batch inference done: {n} frames in {elapsed:.1f}s")

        self.smoothed_snow = self._run_smoother(self.cam_df)
        self.timeseries_tab.update_camera(self.cam_df, smoothed=self.smoothed_snow, kalman=self._run_kalman(self.cam_df))
        self._plot_mini_timeline()
        self.snow_tab.update_camera(self.cam_df)
        self.ground_tab.update_camera(self.cam_df)

    # ── Demo Mode ──────────────────────────────────────────────────────────

    def _toggle_demo_mode(self, enabled):
        if enabled:
            if self.ground_model is None and self.snow_model is None and self.multihead_model is None:
                QMessageBox.information(self, "No Models",
                                        "Load ground/snow or multihead model first.")
                self.demo_action.setChecked(False)
                return
            if self.cam_df is None or len(self.cam_df) == 0:
                self.demo_action.setChecked(False)
                return
            self.demo_mode = True
            self._demo_fps_times.clear()
            self._demo_infer_times = []
            if self.playing:
                self._toggle_play()

            self._demo_preload = {}
            self._demo_preload_thread = None

            if self.multihead_model is None:
                import torch
                models_path = str(self.models_dir)
                if models_path not in sys.path:
                    sys.path.insert(0, models_path)
                from student_snow.inference import load_and_preprocess
                self._demo_preprocess = load_and_preprocess
                self._start_demo_preload()

            self.status.showMessage(f"Demo mode ON — {self.infer_device} — preloading...")
            self._demo_timer.start(0)
        else:
            self.demo_mode = False
            self._demo_timer.stop()
            self._demo_preload = {}
            self.status.showMessage("Demo mode OFF")

    def _start_demo_preload(self):
        import threading
        cam_df = self.cam_df
        start_idx = self.current_idx
        data_dir = self.data_dir
        buf = self._demo_preload
        input_sizes = {}
        for name, model in [("ground", self.ground_model), ("snow", self.snow_model)]:
            if model is not None:
                input_sizes[name] = model.input_size
        preprocess_fn = self._demo_preprocess

        def _worker():
            count = min(200, len(cam_df) - start_idx)
            for i in range(start_idx, start_idx + count):
                if not self.demo_mode:
                    return
                row = cam_df.iloc[i]
                camera = row["camera"]
                stem = row["stem"]
                img_path = data_dir / camera / f"{stem}.jpg"
                if not img_path.exists():
                    continue
                try:
                    img_np = np.array(Image.open(img_path).convert("RGB"))
                    tensors = {}
                    for name, sz in input_sizes.items():
                        tensors[name] = preprocess_fn(img_path, sz)
                    buf[i] = (img_np, tensors)
                except Exception:
                    continue

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        self._demo_preload_thread = t

    def _demo_step(self):
        import time, torch
        if self.cam_df is None or not self.demo_mode:
            self._demo_timer.stop()
            return

        idx = self.current_idx
        row = self.cam_df.iloc[idx]
        camera = row["camera"]
        stem = row["stem"]
        dev = self.infer_device

        t_total = time.perf_counter()

        cached = self._demo_preload.get(idx)
        if cached is not None:
            img_np, tensors = cached
            del self._demo_preload[idx]
        else:
            img_path = self.data_dir / camera / f"{stem}.jpg"
            if not img_path.exists():
                self.current_idx += 1
                return
            try:
                img_np = np.array(Image.open(img_path).convert("RGB"))
                tensors = {}
                for name, model in [("ground", self.ground_model), ("snow", self.snow_model)]:
                    if model is not None:
                        tensors[name] = self._demo_preprocess(img_path, model.input_size)
            except Exception as e:
                self.status.showMessage(f"Demo error: {e}")
                self.current_idx += 1
                return

        try:
            t_infer = time.perf_counter()
            h, w = img_np.shape[:2]
            qimg = QImage(img_np.data, w, h, w * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg.copy())

            img_path = self.data_dir / camera / f"{stem}.jpg"

            if self.multihead_model is not None:
                out = self._multihead_infer(img_path)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - t_infer) * 1000
                if out is not None:
                    gr_mask, sn_mask, qa_prob = out
                    if self.show_ground:
                        pixmap = overlay_mask_fast(pixmap, gr_mask, (0, 255, 0), alpha=0.35)
                    if self.show_snow:
                        pixmap = overlay_mask_fast(pixmap, sn_mask, (68, 136, 255), alpha=0.45)
            else:
                masks = {}
                for name, model in [("ground", self.ground_model), ("snow", self.snow_model)]:
                    if model is None or name not in tensors:
                        continue
                    tensor = tensors[name].to(dev)
                    with torch.no_grad():
                        logits = model(tensor)
                        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                        masks[name] = (prob >= 0.5).astype(np.uint8) * 255
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - t_infer) * 1000

                bottom_px = self.CAMERA_MASK_BOTTOM.get(camera, 0)
                if bottom_px > 0:
                    for name in masks:
                        scale = masks[name].shape[0] / h
                        masks[name][-int(bottom_px * scale):, :] = 0
                for name in masks:
                    vr = self._get_valid_region(camera, masks[name].shape[0], masks[name].shape[1])
                    if vr is not None:
                        masks[name][vr <= 127] = 0

                if "ground" in masks:
                    pixmap = overlay_mask_fast(pixmap, masks["ground"], (0, 255, 0), alpha=0.35)
                if "snow" in masks:
                    pixmap = overlay_mask_fast(pixmap, masks["snow"], (68, 136, 255), alpha=0.45)

            self.image_display.set_pixmap_scaled(pixmap)
        except Exception as e:
            self.status.showMessage(f"Demo error: {e}")
            self._demo_timer.stop()
            self.demo_action.setChecked(False)
            return

        total_ms = (time.perf_counter() - t_total) * 1000
        self._demo_fps_times.append(total_ms)
        self._demo_infer_times.append(infer_ms)
        if len(self._demo_fps_times) > 30:
            self._demo_fps_times = self._demo_fps_times[-30:]
            self._demo_infer_times = self._demo_infer_times[-30:]

        avg_total = sum(self._demo_fps_times) / len(self._demo_fps_times)
        avg_infer = sum(self._demo_infer_times) / len(self._demo_infer_times)
        total_fps = 1000.0 / avg_total if avg_total > 0 else 0
        infer_fps = 1000.0 / avg_infer if avg_infer > 0 else 0
        preloaded = len(self._demo_preload)

        self.frame_label.setText(f"{idx + 1} / {len(self.cam_df)}")
        dt = row.get("datetime", "")
        self.timestamp_label.setText(str(dt)[:19] if pd.notna(dt) else "")
        self.status.showMessage(
            f"DEMO {self.infer_device} │ infer {infer_ms:.0f}ms ({infer_fps:.0f} FPS) │ "
            f"total {total_ms:.0f}ms ({total_fps:.0f} FPS) │ buf {preloaded}"
        )

        nxt = self._find_valid_idx(idx + 1, direction=1)
        if nxt >= 0 and nxt < len(self.cam_df):
            self.current_idx = nxt
            self.timeline_slider.setValue(self.current_idx)
        else:
            self._demo_timer.stop()
            self.demo_action.setChecked(False)
            self.status.showMessage(
                f"Demo done │ {self.infer_device} │ avg infer {avg_infer:.0f}ms ({infer_fps:.0f} FPS) │ "
                f"avg total {avg_total:.0f}ms ({total_fps:.0f} FPS)"
            )


# ── Entry Point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Snow Pipeline Viewer")
    parser.add_argument("--data-dir", type=str, default=str(MASTER / "data"))
    parser.add_argument("--csv", type=str, default=str(TIMESERIES_CSV))
    parser.add_argument("--ground-masks", type=str, default=str(GROUND_MASK_DIR))
    parser.add_argument("--snow-masks", type=str, default=str(SNOW_MASK_DIR))
    parser.add_argument("--models-dir", type=str, default=str(MASTER / "models"))
    parser.add_argument("--smoother-ckpt", type=str, default=None,
                        help="Path to temporal smoother (GRU/LSTM) checkpoint")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 40, 40))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 50, 50))
    palette.setColor(QPalette.ColorRole.Link, QColor(70, 150, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(70, 130, 200))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    viewer = SnowPipelineViewer(
        data_dir=args.data_dir,
        timeseries_csv=args.csv,
        models_dir=args.models_dir,
        ground_mask_dir=args.ground_masks,
        snow_mask_dir=args.snow_masks,
        smoother_ckpt=args.smoother_ckpt,
    )
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
