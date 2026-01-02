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

REQUIRE_ROBOT=0
if [[ "${1:-}" == "--require-robot" ]]; then
  REQUIRE_ROBOT=1
fi

export REQUIRE_ROBOT

"$PYTHON" - <<'PY'
import os

require_robot = os.environ.get("REQUIRE_ROBOT", "0").strip() in ("1", "true", "yes", "on")

try:
    from reachy_mini import ReachyMini
except Exception as e:
    print("[basic_move] SKIP: reachy_mini not installed:", e)
    raise SystemExit(0)

robot_name = os.environ.get("MAXIM_ROBOT_NAME", "reachy_mini")

try:
    mini = ReachyMini(robot_name=robot_name, localhost_only=False, spawn_daemon=False, use_sim=False, timeout=5.0)
except Exception as e:
    msg = f"[basic_move] {'FAIL' if require_robot else 'SKIP'}: could not connect to Reachy Mini '{robot_name}': {e}"
    print(msg)
    raise SystemExit(1 if require_robot else 0)

print("[basic_move] Connected to Reachy Mini:", robot_name)
print("[basic_move] Wiggling antennas...")
try:
    mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
    mini.goto_target(antennas=[0.0, 0.0], duration=0.5)
except Exception as e:
    print("[basic_move] FAIL: movement command failed:", e)
    raise SystemExit(1)
print("[basic_move] OK")
PY
