#!/usr/bin/env bash
#
# Reproduce the paper's evaluation numbers from the released checkpoints.
#
#   bash tools/reproduce_paper.sh            # everything, CPU, logs to workspace/repro/
#   DEVICE=cuda bash tools/reproduce_paper.sh
#   TASKS="snow ground" bash tools/reproduce_paper.sh
#
# Why this script exists: every failure mode below is silent in the individual
# scripts. A missing backbone download, a missing derived CSV, or a checkpoint that
# was renamed after the paper was written produces an *empty row* or a dashes-only
# table, not an error. This wrapper makes each of those loud, then runs the tasks
# and diffs the logged numbers against the values printed in the manuscript.
#
# Prerequisites it checks and refuses to fake: the checkpoint layout (it links the
# deposited files first, then verifies them against models/CHECKPOINTS.csv), data/, data/annotations/,
# checkpoints under models/, and two derived inputs eval_all.py hardcodes
# (analysis/snow_timeseries.csv, qa/checkpoints/qa_features.csv).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
DEVICE="${DEVICE:-cpu}"
# note: no ":-" — an explicitly empty TASKS must mean "no tasks", not "all of them"
TASKS="${TASKS-snow ground smoother qa ood-per-camera}"
OUT="$REPO/workspace/repro"
DATA="${SNOW_DATA_DIR:-$REPO/data}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT"
FAIL=0

say () { printf '%s\n' "$*"; }
ok  () { printf '  [ok]   %s\n' "$*"; }
bad () { printf '  [FAIL] %s\n' "$*"; FAIL=1; }

# ── 1. environment traps ─────────────────────────────────────────────────────
# micromamba/conda pythons look for a CA bundle inside the env prefix; when it is
# missing, every torchvision backbone download dies with CERTIFICATE_VERIFY_FAILED
# and the affected model is skipped with a score of nothing.
if [ -z "${SSL_CERT_FILE:-}" ]; then
  for c in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt; do
    [ -f "$c" ] && export SSL_CERT_FILE="$c" REQUESTS_CA_BUNDLE="$c" && \
      say "CA bundle: $c" && break
  done
fi

say "== prerequisites =="
# eval_all.py imports cv2 at module level; a venv without it dies before printing anything
if ! "$PY" -c 'import cv2' >/dev/null 2>&1; then
  bad "no cv2 for $PY -> pip install opencv-python-headless   (or run inside the Docker image)"
fi
[ -d "$DATA" ] && ok "data dir: $DATA" || bad "data dir missing: $DATA"
# an empty camera directory is how "0 images found" starts
n_img=$(ls "$DATA"/*/*.jpg 2>/dev/null | wc -l)
if [ "$n_img" -gt 0 ]; then ok "images under data/: $n_img jpgs"
else bad "no jpgs under $DATA/<camera>/ — populate it (see How-to-guide.md, 'New Camera Onboarding')"; fi
[ -d "$DATA/annotations" ] && ok "annotations: $(ls "$DATA/annotations" | wc -l) files" \
  || bad "annotations missing: $DATA/annotations"
# The Zenodo record ships flat, renamed checkpoints; eval_all.py reads the original
# run-directory layout. A first run on a fresh checkout needs this link step, and it
# is idempotent, so it runs every time (LINK=0 skips it). --check then compares every
# row of models/CHECKPOINTS.csv with what is on disk, including byte size, so a
# renamed, truncated or substituted checkpoint is named here and not discovered as an
# empty table row later.
say "== checkpoint layout =="
if [ "${LINK:-1}" = "0" ]; then
  say "  [skip] LINK=0"
elif [ ! -f models/CHECKPOINTS.csv ]; then
  bad "models/CHECKPOINTS.csv missing - unpack the Zenodo record into the repo root first"
else
  "$PY" tools/link_deposit_models.py | sed 's/^/  /'
  if "$PY" tools/link_deposit_models.py --check > "$OUT/checkpoint_check.txt" 2>&1; then
    ok "$(head -1 "$OUT/checkpoint_check.txt")"
  else
    bad "checkpoints do not match models/CHECKPOINTS.csv (full list: $OUT/checkpoint_check.txt)"
    sed 's/^/  /' "$OUT/checkpoint_check.txt" | head -40
  fi
fi

for ck in models/best/ground_mobilenet_*.ckpt models/best/snow_mobilenet_*.ckpt \
          models/best/qa_cnn.pt; do
  ls $ck >/dev/null 2>&1 && ok "checkpoint: $ck" || bad "checkpoint missing: $ck"
done
n_students=$(ls -d models/student_snow/checkpoints/*/ 2>/dev/null | wc -l)
[ "$n_students" -gt 0 ] && ok "student runs: $n_students dirs under models/student_snow/checkpoints" \
  || bad "no student runs under models/student_snow/checkpoints (tables will be empty)"

# ── 2. derived inputs eval_all.py expects at fixed relative paths ────────────
say "== derived inputs =="
# Neither file ships (both are *.csv / checkpoints/, i.e. gitignored) and both are
# slow to rebuild, so this script names the command instead of launching it behind
# your back. Set GEN=1 to have it run them. The QA task fails outright without its
# feature CSV; the smoother task fails without the timeseries.
if [ -f analysis/snow_timeseries.csv ]; then
  ok "analysis/snow_timeseries.csv ($(du -h analysis/snow_timeseries.csv | cut -f1))"
elif [ "${GEN:-0}" = "1" ]; then
  say "  [gen]  analysis/snow_timeseries.csv"
  "$PY" analysis/build_features.py --output analysis/snow_timeseries.csv || \
    bad "could not build analysis/snow_timeseries.csv"
else
  bad "analysis/snow_timeseries.csv missing -> run: $PY analysis/build_features.py --output analysis/snow_timeseries.csv"
fi
if [ -f qa/checkpoints/qa_features.csv ]; then
  ok "qa/checkpoints/qa_features.csv"
elif [ "${GEN:-0}" = "1" ]; then
  say "  [gen]  qa/checkpoints/qa_features.csv (slow: feature extraction over the corpus)"
  "$PY" qa/train_qa.py --img-dir "$DATA" --ann-dir "$DATA/annotations" \
        --output-dir qa/checkpoints || bad "could not build qa_features.csv"
else
  bad "qa/checkpoints/qa_features.csv missing -> run: $PY qa/train_qa.py --img-dir $DATA --ann-dir $DATA/annotations --output-dir qa/checkpoints"
fi

# ── 3. the evaluation tasks ──────────────────────────────────────────────────
run () {
  local name="$1"; shift
  say "== $name =="
  local start; start=$(date +%s)
  "$PY" -u eval_all.py "$@" > "$OUT/$name.log" 2>&1
  local rc=$? secs=$(( $(date +%s) - start ))
  if [ $rc -ne 0 ]; then bad "$name exited $rc after ${secs}s (see $OUT/$name.log)"; return; fi
  # an empty or dashes-only summary is the classic silent failure
  if grep -qE '^\s*[-—]\s' "$OUT/$name.log" 2>/dev/null || ! grep -qE '[0-9]\.[0-9]{3}' "$OUT/$name.log"; then
    bad "$name produced no scores in ${secs}s — check $OUT/$name.log (missing checkpoint or unreadable data)"
  else
    ok "$name scored in ${secs}s"
  fi
  sed -n '/SUMMARY/,/^$/p' "$OUT/$name.log" | head -20
}

for t in $TASKS; do
  case "$t" in
    # Snow is scored on the held-out split by eval_all.py.
    snow)           run snow           --task snow   --types students --device "$DEVICE" ;;
    ground)         run ground         --task ground --types students --breakdown --device "$DEVICE" ;;
    smoother)       run smoother       --task smoother --device "$DEVICE" ;;
    qa)             run qa             --task qa --device "$DEVICE" ;;
    ood-per-camera) run ood_per_camera --task ood-per-camera --device "$DEVICE" ;;
    *) bad "unknown task: $t" ;;
  esac
done

# ── 4. compare against the manuscript's tables ───────────────────────────────
say "== manuscript comparison =="
"$PY" - "$OUT" <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
# (log, row label in the eval output, column label, manuscript value, table)
# Manuscript values, as printed in the paper.
# vit_lora cannot load at all (LoRA target changed) — it is expected to refuse.
EXPECT = [
    ("snow_students.log", "mobilenet",   "IoU",  0.895, "tab:snow"),
    ("snow_students.log", "convnext",    "IoU",  0.905, "tab:snow"),
    ("ground_students.log", "mobilenet", "PL IoU",   0.837, "tab:ground"),
    ("ground_students.log", "efficientnet", "PL IoU", 0.836, "tab:ground"),
    ("ground_students.log", "dinov2_large", "PL IoU", 0.832, "tab:ground"),
    ("ground_students.log", "dinov2_base",  "PL IoU", 0.830, "tab:ground"),
    ("ground_students.log", "dinov3_base",  "PL IoU", 0.829, "tab:ground"),
    ("qa.log", "TinyQACNN", "ID AUROC", 0.975, "tab:qa"),
    ("qa.log", "TinyQACNN", "OOD AUROC", 0.951, "tab:qa"),
]
for fname, row, col, paper, table in EXPECT:
    p = out / fname
    if not p.exists():
        print(f"  {table:12s} {row:14s} {col:9s} paper {paper:.3f}  -> task not run")
        continue
    line = next((l for l in p.read_text(errors="ignore").splitlines()
                 if row.lower() in l.lower() and re.search(r"0\.\d{3}", l)), None)
    if not line:
        print(f"  {table:12s} {row:14s} {col:9s} paper {paper:.3f}  -> no number logged")
        continue
    got = float(re.search(r"0\.\d{3}", line).group(0))
    mark = "PASS" if abs(got - paper) <= 0.005 else "DIFF"
    print(f"  {table:12s} {row:14s} {col:9s} paper {paper:.3f}  got {got:.3f}  {mark}")
print("\nPASS = within 0.005 of the manuscript. DIFF means: check the checkpoint "
      "selection rule (see 'Which checkpoint gets evaluated' in How-to-guide.md) "
      "before assuming the code is wrong.")
PY

say ""
[ $FAIL -eq 0 ] && say "prerequisites and tasks: OK" || say "SOMETHING ABOVE FAILED — read the [FAIL] lines"
exit $FAIL
