"""Render the deploy chain's actual output to images a human can look at.

Runs the shipped chain over real annotated frames and writes what came out next to
the ground truth: input, prediction, GT outlines, and a TP/FP/FN agreement panel.
Also renders `psg_raw` vs `psg_smoothed` over a run of consecutive frames as two
stacked SVG panels (same 0-100 axis, so the damping is not hidden at the bottom of a
shared axis). Open `deploy/viz/index.html`.

    /workspace/thesis/.deployenv/bin/python deploy/viz_frames.py --series 80

Every panel is checked to differ from the input frame before it is written, so a
silently empty overlay cannot produce a page that looks plausible and shows nothing.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from deploy.chain import SnowyPipeline                     # noqa: E402
from deploy.acceptance import load_gt_masks, gt_map         # noqa: E402 (stubs cv2)

ANN = _REPO / "data" / "annotations"
IMG_ROOTS = [_REPO / "data",
             Path("/workspace/thesis/data"),
             Path("/workspace/thesis/night_images"),
             Path("/workspace/thesis/data/annotation_candidates_new"),
             Path("/workspace/thesis/data/annotation_samples")]
OUT = _HERE / "viz"
ID_CAMS = {"CF00001", "CM00006", "CM00009", "CM00014", "CM00070", "CM00075", "CM00080",
           "CM00102", "CM00108", "CM00112", "CM00024", "CM00068"}
OOD_CAMS = {"CM00054", "CM00067", "CM00071", "CM00115"}


# ── masks: the chain hands out uint8 with a 127 threshold (chain.py:356) ─────
def _bool_mask(mask):
    """Whatever the caller hands over -> a boolean array.

    Three encodings meet here: boolean masks (the GT loader), 0/1 uint8 (also the GT
    loader, via np.uint8) and 0/255 uint8 (the chain's predictions). A fixed >127
    threshold silently erases the 0/1 kind, which is how the GT panel ended up
    identical to the frame.
    """
    a = np.asarray(mask)
    if a.dtype == bool:
        return a
    return a > 0 if a.size and a.max() <= 1 else a > 127


def _mask_image(mask, size) -> Image.Image | None:
    if mask is None:
        return None
    b = _bool_mask(mask)
    if not b.any():
        return None
    return Image.fromarray((b.astype("uint8") * 255)).resize(size, Image.NEAREST)


def _paint(img: Image.Image, boolean_mask, colour, alpha=120) -> Image.Image:
    layer = Image.new("RGB", img.size, colour)
    m = Image.fromarray((boolean_mask.astype("uint8") * 255))
    return Image.composite(Image.blend(img, layer, alpha / 255), img, m)


def tint(img, mask, colour, alpha=120):
    m = _mask_image(mask, img.size)
    if m is None:
        return img
    return _paint(img, np.asarray(m) > 127, colour, alpha)


def outline(img, mask, colour, width=3):
    m = _mask_image(mask, img.size)
    if m is None:
        return img
    edge = m.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 40 else 0)
    edge = edge.filter(ImageFilter.MaxFilter(max(3, width)))
    return Image.composite(Image.new("RGB", img.size, colour), img, edge)


def agreement(img, pred, gt, colour_tp=(0, 225, 60), colour_fp=(255, 0, 200),
              colour_fn=(255, 70, 70)):
    """Where the prediction is right, invented, or missing, against GT."""
    p = _mask_image(pred, img.size)
    g = _mask_image(gt, img.size)
    if p is None or g is None:
        return None
    P, G = _bool_mask(p), _bool_mask(g)
    out = img.copy()
    out = _paint(out, G & ~P, colour_fn)
    out = _paint(out, P & ~G, colour_fp)
    out = _paint(out, P & G, colour_tp)
    return out


def coverage_pct(mask):
    """Share of the frame the mask lights up — the number that makes an empty
    overlay impossible to miss."""
    if mask is None:
        return None
    return round(float(_bool_mask(mask).mean()) * 100, 1)


def iou_dice(pred, gt):
    if pred is None or gt is None:
        return None, None
    P, G = _bool_mask(pred), _bool_mask(gt)
    inter, union = int((P & G).sum()), int((P | G).sum())
    iou = inter / union if union else float("nan")
    denom = int(P.sum()) + int(G.sum())
    dice = 2 * inter / denom if denom else float("nan")
    return iou, dice


# ── data lookup ──────────────────────────────────────────────────────────────
def find_image(cam: str, stem: str):
    for root in IMG_ROOTS:
        p = root / cam / f"{stem}.jpg"
        if p.exists():
            return p
    return None


def gt_for(cam: str) -> dict:
    """GT masks for one camera, from the repo's own loader.

    Annotation files are named by camera number: `14.json`, `67.json`, `115.json`,
    but `01.json`/`09.json` keep two digits. `CM00067.json` does not exist.
    """
    out: dict = {}
    digits = cam[2:].lstrip("0") or "0"
    if len(digits) < 2:
        digits = digits.zfill(2)
    for name in (f"{cam}.json", f"{digits}.json", "ground_annotations.json"):
        path = ANN / name
        if not path.exists():
            continue
        for lc in ("Snow Mano", "Ground Mano"):
            try:
                m = gt_map(load_gt_masks(str(path), {cam}, label_class=lc), cam)
            except Exception as exc:
                print(f"  [gt] {name} / {lc}: {type(exc).__name__}: {exc}")
                continue
            key = "snow" if lc.startswith("Snow") else "ground"
            for stem, mask in m.items():
                if mask is not None and np.asarray(mask).any():
                    out.setdefault(stem, {})[key] = mask
    return out


def pick_frames():
    picks = []
    for cam in ["CM00108", "CM00067", "CM00054", "CM00014", "CM00112", "CF00001"]:
        g = gt_for(cam)
        usable = sorted((s, p) for s, p in
                        ((s, find_image(cam, s)) for s in g) if p is not None)
        snow = [kv for kv in usable if "snow" in g[kv[0]]]
        ground = [kv for kv in usable if "ground" in g[kv[0]]]
        for cand in (snow[:1] + ground[:1] + usable[:1]):
            if cand and not any(c[1] == cand[0] for c in picks):
                picks.append((cam, cand[0], cand[1]))
    return picks


def series_frames(cam: str, n: int, anchor: str | None = None):
    seen = {}
    for root in IMG_ROOTS:
        for p in sorted((root / cam).glob("*.jpg")):
            seen.setdefault(p.name, p)
    files = [seen[k] for k in sorted(seen)]
    if anchor:
        for i, p in enumerate(files):
            if p.stem == anchor:
                files = files[i:]
                break
    return files[:n]


# ── SVG charts: two stacked panels, one shared 0-100 scale ───────────────────
def _panel(vals, colour, title, w, h, lo=0.0, hi=100.0, labels=None):
    n = max(1, len(vals))
    def xy(i, v):
        return 46 + (w - 70) * i / max(1, n - 1), h - 26 - (h - 46) * (v - lo) / (hi - lo)
    parts = [f'<text x="46" y="16" font-size="13" fill="#333">{html.escape(title)}</text>']
    for frac in (0.0, 0.5, 1.0):
        y = h - 26 - (h - 46) * frac
        parts.append(f'<line x1="46" y1="{y:.1f}" x2="{w-24}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="4" y="{y+4:.1f}" font-size="11" fill="#888">'
                     f'{lo + (hi-lo)*frac:.0f}</text>')
    seg, pen = [], False
    for i, v in enumerate(vals):
        if v is None or not isinstance(v, (int, float)) or v != v:
            pen = False
            continue
        x, y = xy(i, v)
        if not pen:
            seg.append([f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{x:.1f},{y:.1f}'])
            pen = True
        else:
            seg[-1][0] += f' {x:.1f},{y:.1f}'
        seg[-1].append(i)
    for item in seg:
        parts.append(item[0] + '"/>')
        pts = item[1:]
        parts.append(f'<circle cx="{xy(pts[0], 0)[0]:.1f}" cy="{xy(pts[0], vals[pts[0]])[1]:.1f}" r="3" fill="{colour}"/>')
    parts.append(f'<line x1="46" y1="{h-26}" x2="{w-24}" y2="{h-26}" stroke="#999"/>')
    if labels:
        parts.append(f'<text x="46" y="{h-8}" font-size="11" fill="#666">{html.escape(labels[0])}</text>')
        parts.append(f'<text x="{w-24}" y="{h-8}" font-size="11" fill="#666" text-anchor="end">{html.escape(labels[-1])}</text>')
    # just the panel's contents: two panels share one outer <svg>, because two root
    # elements in one file is invalid XML and browsers render nothing at all
    return f'<rect width="{w}" height="{h}" fill="#fbfbfb"/>{"".join(parts)}' 


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/workspace/thesis/models/best")
    ap.add_argument("--series", type=int, default=80)
    ap.add_argument("--series-cam", default="CM00108")
    ap.add_argument("--series-anchor", default=None)
    args = ap.parse_args(argv)

    OUT.mkdir(exist_ok=True)
    pipe = SnowyPipeline(args.weights, device="cpu", seq_len=72)
    smoother_error = None
    try:
        pipe.load_smoother()
    except Exception as exc:
        smoother_error = f"{type(exc).__name__}: {exc}"
    print("audit:", {k: v.get("loaded") for k, v in pipe.load_reports.items()})
    print("smoother:", "loaded" if smoother_error is None else smoother_error)

    rows, warnings = [], []
    for cam, stem, path in pick_frames():
        try:
            img = Image.open(path).convert("RGB")
            small = img.resize((760, int(img.height * 760 / img.width)), Image.LANCZOS)
            resp, raw = pipe.predict_image(str(path), cam, None, {"t2m": -1.0})
            g = gt_for(cam).get(stem, {})
            p_snow, p_ground = raw.get("snow_mask"), raw.get("ground_mask")

            # psg_raw counts snow *inside* the ground mask (chain.py:353-359), so that
            # is what the panel shows; the anywhere-in-frame count goes in the JSON
            s_in_g = (_bool_mask(p_snow) & _bool_mask(p_ground)) \
                if (p_snow is not None and p_ground is not None) else None
            panels = [("frame", small.copy()),
                      ("prediction: snow-in-ground+cyan, ground outline+yellow",
                       outline(tint(small.copy(), s_in_g, (0, 210, 255), 130),
                               p_ground, (255, 210, 0), 3))]
            if g.get("snow") is not None or g.get("ground") is not None:
                panels.append(("ground truth: snow+white, ground+red",
                               outline(outline(small.copy(), g.get("ground"), (255, 40, 40)),
                                       g.get("snow"), (255, 255, 255))))
            agree = agreement(small.copy(), p_snow, g.get("snow"))
            if agree is not None:
                siou, _ = iou_dice(p_snow, g.get("snow"))
                panels.append((f"vs snow GT: green=correct, magenta=invented, "
                               f"red=missed (IoU {siou:.3f})", agree))

            # a panel that equals the frame means the overlay silently painted nothing
            for title, im in panels[1:]:
                diff = np.asarray(im.convert("L"), dtype="int16") - \
                       np.asarray(panels[0][1].convert("L"), dtype="int16")
                changed = float((np.abs(diff) > 4).mean())
                if changed < 0.005:
                    warnings.append(f"{stem}: '{title[:38]}' changed {changed*100:.2f}% "
                                    f"of pixels — overlay looks empty")

            names = []
            for i, (title, im) in enumerate(panels):
                fn = f"{stem}_{i}.png"
                im.save(OUT / fn)
                names.append((title, fn))
            cover = {"snow_in_ground": coverage_pct(s_in_g),
                     "ground": coverage_pct(p_ground),
                     "snow_anywhere": coverage_pct(p_snow)}
            rows.append({"cam": cam, "stem": stem,
                         "cohort": "ID" if cam in ID_CAMS else ("OOD" if cam in OOD_CAMS else "?"),
                         "qa": resp.get("qa_score"), "qa_class": resp.get("qa_class"),
                         "psg_raw": resp.get("psg_raw"), "smoothed": resp.get("psg_smoothed"),
                         "panels": names,
                         "mask_px_pct": cover})
            print(f"  {stem}: qa={resp.get('qa_score')} psg_raw={resp.get('psg_raw')} "
                  f"gt={'yes' if g else 'no'} mask%={rows[-1]['mask_px_pct']}")
        except Exception as exc:
            print(f"  {stem}: SKIPPED {type(exc).__name__}: {exc}")

    seq = series_frames(args.series_cam, args.series, args.series_anchor)
    raw_vals, smooth_vals, labels = [], [], []
    for p in seq:
        try:
            resp, _ = pipe.predict_image(str(p), args.series_cam, None, {"t2m": -1.0})
            labels.append(p.stem.split("_")[1])
            raw_vals.append(resp.get("psg_raw"))
            smooth_vals.append(resp.get("psg_smoothed"))
        except Exception as exc:
            print(f"  series {p.stem}: skipped {type(exc).__name__}")
    (OUT / "series.json").write_text(json.dumps(
        {"camera": args.series_cam, "labels": labels, "psg_raw": raw_vals,
         "psg_smoothed": smooth_vals}, indent=1))

    def jitter(vals):
        v = [x for x in vals if isinstance(x, (int, float))]
        return (sum(abs(v[i] - v[i - 1]) for i in range(1, len(v))) / (len(v) - 1)) if len(v) > 1 else float("nan")

    chart = ('<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="400">'
             + _panel(raw_vals, "#1f6feb", "psg_raw  (per frame)", 1100, 200, labels=labels)
             + f'<g transform="translate(0,200)">'
             + _panel(smooth_vals, "#d1242f",
                      f"psg_smoothed  (causal GRU; first value at frame {pipe.seq_len})",
                      1100, 200, labels=labels)
             + '</g></svg>')
    (OUT / "series.svg").write_text(chart)

    cards = []
    for r in rows:
        panels = "".join(
            f'<figure><img src="{fn}" loading="lazy"><figcaption>{html.escape(t)}</figcaption></figure>'
            for t, fn in r["panels"])
        cards.append(f"""<section class="card">
  <h2>{r['stem']} <span class="tag">{r['cohort']} camera {r['cam']}</span></h2>
  <div class="panels">{panels}</div>
  <pre>{html.escape(json.dumps({k: r[k] for k in ('qa', 'qa_class', 'psg_raw', 'smoothed', 'mask_px_pct')}, indent=1))}</pre>
</section>""")

    warn_html = ""
    if warnings:
        warn_html = '<div class="note"><b>Overlay warnings</b><br>' + \
                    "<br>".join(html.escape(w) for w in warnings) + "</div>"

    page = f"""<!doctype html><meta charset="utf-8"><title>snowy deploy chain - actual output</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#111;background:#fff}}
 h1{{font-size:22px}} h2{{font-size:17px}} .card{{border:1px solid #ddd;border-radius:10px;padding:12px;margin:18px 0}}
 .panels{{display:flex;gap:12px;flex-wrap:wrap}} figure{{margin:0;max-width:480px}}
 img{{width:460px;max-width:100%;border:1px solid #eee;border-radius:6px;display:block}}
 figcaption{{font-size:12px;color:#555;margin-top:4px}} .tag{{font-size:12px;color:#666;font-weight:400}}
 pre{{background:#f6f8fa;padding:8px;border-radius:6px;font-size:13px;overflow:auto}}
 .audit{{background:#f6f8fa;padding:10px;border-radius:8px;white-space:pre-wrap;font-size:13px}}
 .note{{background:#fff8e1;border:1px solid #f0e0a0;padding:10px;border-radius:8px;font-size:13px;margin:12px 0}}
 .bad{{background:#ffecec;border:1px solid #f0b0b0}}
</style>
<h1>snowy deploy chain &mdash; output on real frames</h1>
<div class="note">Every image was produced by <code>deploy/viz_frames.py</code> with the shipped
checkpoints in <code>{html.escape(args.weights)}</code>, on CPU. The JSON under each frame is the
same dict <code>POST /predict</code> returns. Reproduce:
<code>micromamba run -p ~/Documents/Uni/Thesis/.deployenv python deploy/viz_frames.py</code><br>
Interactive: <code>micromamba run -p ~/Documents/Uni/Thesis/.deployenv python deploy/app.py</code>
then <code>http://127.0.0.1:7860</code>.</div>
{warn_html}
<h2>psg_raw vs psg_smoothed &mdash; {len(seq)} consecutive frames, {args.series_cam}</h2>
<p>Both series are <b>percent (0&#8211;100)</b>, the same unit as every other psg number.
The GRU's own output is a fraction in [0,1]; the pipeline scales it &times;100 once, the way
<code>visualize_smoother.py</code> plots it (<code>smoothed_snow * 100</code>, ylabel
"Snow %"). Smoothed appears only once the {pipe.seq_len}-frame window is full, so the smoothed
panel is empty before that. Expected behaviour: the causal filter tracks the <i>level</i> of the
recent raw signal, so it should sit near the trailing mean, not near zero.
Mean absolute frame-to-frame change: raw {jitter(raw_vals):.2f} over
{sum(1 for v in raw_vals if isinstance(v, (int,float)))} frames, smoothed {jitter(smooth_vals):.2f}
over {sum(1 for v in smooth_vals if isinstance(v, (int,float)))} frames.</p>
{chart}
<div class="audit"><b>Checkpoint audit as loaded by this run</b>{" (smoother NOT loaded: " + html.escape(str(smoother_error)) + ")" if smoother_error else ""}
{html.escape(json.dumps(pipe.load_reports, indent=1))}</div>
{''.join(cards)}
"""
    (OUT / "index.html").write_text(page)
    print(f"\nwrote {OUT/'index.html'}")
    print(f"overlay warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
