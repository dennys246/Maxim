#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo "[basic_vision] tmp=$TMP_DIR"

export TMP_DIR

"$PYTHON" - <<'PY'
import os

import numpy as np

try:
    import cv2
except Exception as e:
    print("[basic_vision] SKIP: OpenCV not available:", e)
    raise SystemExit(0)

tmp_dir = os.environ.get("TMP_DIR", "")
if not tmp_dir:
    raise SystemExit("TMP_DIR not set by launcher.")

video_path = os.path.join(tmp_dir, "test.mp4")
fps = 20.0
height = 360
width = 640
num_frames = 60

writer = None
for codec in ("mp4v", "avc1"):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    w = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if w is not None and w.isOpened():
        writer = w
        break
    try:
        w.release()
    except Exception:
        pass

if writer is None:
    print("[basic_vision] SKIP: no working mp4 codec (mp4v/avc1)")
    raise SystemExit(0)

for i in range(num_frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x = int((i / max(1, num_frames - 1)) * (width - 80))
    cv2.rectangle(frame, (x, 80), (x + 80, 160), (0, 255, 0), thickness=-1)
    cv2.putText(frame, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    writer.write(frame)

writer.release()

size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
assert size > 0, f"expected video output, got size={size}"

cap = cv2.VideoCapture(video_path)
ok, first = cap.read()
cap.release()
assert ok and first is not None and first.size > 0, "failed to read back written video"

print("[basic_vision] OK: wrote mp4:", video_path, f"({size} bytes)")
PY

echo "[basic_vision] done"
