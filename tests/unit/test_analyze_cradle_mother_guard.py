"""analyze_cradle_mother's missing-control guard (verify-the-instrument class).

The pre-#508 analyzer compared MOTHER-TAUGHT against ``_mean([]) == 0.0`` when
the ``no_feed`` arm was absent — a single-arm run reported PASS against a
control that never ran (found during the Exp 48 heartbeat investigation). A
future simplification of ``pooled()`` could silently restore that vacuous
PASS with nothing red; this smoke test is the instrument check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_cradle_mother.py"

_ACTS = ("act1_early", "act2_warming", "act3_consolidating", "act4_autonomous")


def _row(arm: str, directedness: dict[str, float]) -> dict:
    return {"arm": arm, "fade": {a: {"directedness": directedness[a]} for a in _ACTS}}


def _run(rows: list[dict], tmp_path: Path) -> subprocess.CompletedProcess:
    inp = tmp_path / "runs.jsonl"
    inp.write_text("\n".join(json.dumps(r) for r in rows))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--in", str(inp)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_single_arm_run_cannot_pass_mother_taught(tmp_path):
    """taught clears LEARNED but no control ran → VOID gate + INCOMPLETE (exit 5)."""
    learned = {"act1_early": 0.3, "act2_warming": 0.8, "act3_consolidating": 0.9, "act4_autonomous": 0.9}
    proc = _run([_row("taught", learned)] * 3, tmp_path)
    assert proc.returncode == 5, proc.stdout + proc.stderr
    assert "VOID" in proc.stdout
    assert "INCOMPLETE" in proc.stdout
    assert "MOTHER-TAUGHT" in proc.stdout
    assert "PASS" not in proc.stdout.split("MOTHER-TAUGHT")[1].splitlines()[0]


def test_both_arms_present_computes_mother_taught_normally(tmp_path):
    learned = {"act1_early": 0.3, "act2_warming": 0.8, "act3_consolidating": 0.9, "act4_autonomous": 0.9}
    chance = {a: 0.5 for a in _ACTS}
    proc = _run([_row("taught", learned)] * 3 + [_row("no_feed", chance)] * 3, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "GRADUATE" in proc.stdout
    assert "VOID" not in proc.stdout
