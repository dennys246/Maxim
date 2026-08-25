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


def _row_turns(arm: str, directedness: dict[str, float], turns: int) -> dict:
    return {"arm": arm, "fade": {a: {"directedness": directedness[a], "turns": turns} for a in _ACTS}}


class TestGateV2:
    """Gate v2 (frozen 2026-08-14): act1-only EARLY bin, S7 ceiling clause,
    S5 exposure flag. 48_cradle_mother_seam.md §Gate v2."""

    def test_early_bin_is_act1_only(self, tmp_path):
        """act2 no longer pollutes the baseline: act1 low + act2 already-high
        must still show a full rise (v1's mean(act1,act2) would shrink it)."""
        d = {"act1_early": 0.30, "act2_warming": 0.90, "act3_consolidating": 0.90, "act4_autonomous": 0.90}
        chance = {a: 0.5 for a in _ACTS}
        proc = _run([_row("taught", d)] * 3 + [_row("no_feed", chance)] * 3, tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "early=0.300" in proc.stdout

    def test_ceiling_reports_learned_at_ceiling_not_fail(self, tmp_path):
        """early >= 0.65 makes rise unattainable — S7 says report the ceiling,
        never silently FAIL (the Exp 37 Mistral24B ceiling-void class)."""
        d = {a: 0.90 for a in _ACTS}
        chance = {a: 0.5 for a in _ACTS}
        proc = _run([_row("taught", d)] * 3 + [_row("no_feed", chance)] * 3, tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "LEARNED-AT-CEILING" in proc.stdout
        assert "GRADUATE" in proc.stdout

    def test_ceiling_with_degradation_fails(self, tmp_path):
        d = {"act1_early": 0.90, "act2_warming": 0.80, "act3_consolidating": 0.70, "act4_autonomous": 0.60}
        chance = {a: 0.5 for a in _ACTS}
        proc = _run([_row("taught", d)] * 3 + [_row("no_feed", chance)] * 3, tmp_path)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "LEARNED-AT-CEILING" in proc.stdout
        assert "FAIL" in proc.stdout

    def test_exposure_skew_flags_instead_of_graduating(self, tmp_path):
        """Both gates pass but taught got 2x the turns — S5 says flag, exit 7."""
        d = {"act1_early": 0.30, "act2_warming": 0.60, "act3_consolidating": 0.90, "act4_autonomous": 0.90}
        chance = {a: 0.5 for a in _ACTS}
        rows = [_row_turns("taught", d, 24)] * 3 + [_row_turns("no_feed", chance, 12)] * 3
        proc = _run(rows, tmp_path)
        assert proc.returncode == 7, proc.stdout + proc.stderr
        assert "EXPOSURE-FLAG" in proc.stdout

    def test_matched_exposure_no_flag(self, tmp_path):
        d = {"act1_early": 0.30, "act2_warming": 0.60, "act3_consolidating": 0.90, "act4_autonomous": 0.90}
        chance = {a: 0.5 for a in _ACTS}
        rows = [_row_turns("taught", d, 12)] * 3 + [_row_turns("no_feed", chance, 12)] * 3
        proc = _run(rows, tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "EXPOSURE-FLAG" not in proc.stdout


# ── Gate v3 (Exp 52, frozen 2026-08-25) ──────────────────────────────────────


def _run_v3(rows: list[dict], tmp_path: Path) -> subprocess.CompletedProcess:
    inp = tmp_path / "runs.jsonl"
    inp.write_text("\n".join(json.dumps(r) for r in rows))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--in", str(inp), "--gate", "v3"],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _jitter(base: dict[str, float], k: int) -> dict[str, float]:
    # per-seed spread so the L2 apparatus check sees variance
    return {a: max(0.0, min(1.0, v + 0.01 * k)) for a, v in base.items()}


_LEARNED = {"act1_early": 0.3, "act2_warming": 0.8, "act3_consolidating": 0.9, "act4_autonomous": 0.9}
_CHANCE = {"act1_early": 0.5, "act2_warming": 0.5, "act3_consolidating": 0.5, "act4_autonomous": 0.5}


class TestGateV3:
    def test_v3_passes_when_satiated_stays_flat(self, tmp_path):
        rows = (
            [_row("taught", _jitter(_LEARNED, k)) for k in range(3)]
            + [_row("no_feed", _jitter(_CHANCE, k)) for k in range(3)]
            + [_row("satiated", _jitter(_CHANCE, k)) for k in range(3)]
        )
        proc = _run_v3(rows, tmp_path)
        assert "HUNGER-NECESSARY" in proc.stdout and "PASS" in proc.stdout.split("HUNGER-NECESSARY")[1].splitlines()[0]
        assert proc.returncode == 0, proc.stdout

    def test_v3_fails_when_the_satiated_arm_learns(self, tmp_path):
        rows = (
            [_row("taught", _jitter(_LEARNED, k)) for k in range(3)]
            + [_row("no_feed", _jitter(_CHANCE, k)) for k in range(3)]
            + [_row("satiated", _jitter(_LEARNED, k)) for k in range(3)]
        )
        proc = _run_v3(rows, tmp_path)
        assert "HUNGER-NECESSARY" in proc.stdout
        assert "FAIL" in proc.stdout.split("HUNGER-NECESSARY")[1].splitlines()[0]
        # LEARNED + MOTHER-TAUGHT pass but the never-hungry infant learned: the
        # pre-registration's "plumbing leak" branch — apparatus, distinct exit.
        assert "HUNGER-LEAK" in proc.stdout and proc.returncode == 9

    def test_v3_cap_fails_a_slowly_rising_satiated_arm(self, tmp_path):
        # Amendment 2: satiated late 0.62 vs no_feed 0.17 rises < 0.15 and sits 0.28
        # below taught, but is NOT indistinguishable from the teacherless control.
        slow = {"act1_early": 0.55, "act2_warming": 0.58, "act3_consolidating": 0.62, "act4_autonomous": 0.62}
        low = {a: 0.17 for a in _ACTS}
        rows = (
            [_row("taught", _jitter(_LEARNED, k)) for k in range(3)]
            + [_row("no_feed", _jitter(low, k)) for k in range(3)]
            + [_row("satiated", _jitter(slow, k)) for k in range(3)]
        )
        proc = _run_v3(rows, tmp_path)
        line = proc.stdout.split("HUNGER-NECESSARY")[1].splitlines()[0]
        assert "FAIL" in line and proc.returncode == 9

    def test_v3_is_void_without_the_satiated_arm(self, tmp_path):
        rows = [_row("taught", _jitter(_LEARNED, k)) for k in range(3)] + [
            _row("no_feed", _jitter(_CHANCE, k)) for k in range(3)
        ]
        proc = _run_v3(rows, tmp_path)
        assert "HUNGER-NECESSARY" in proc.stdout and "VOID" in proc.stdout.split("HUNGER-NECESSARY")[1].splitlines()[0]
        assert "INCOMPLETE" in proc.stdout and proc.returncode == 5

    def test_v3_apparatus_gate_refuses_seed_invariant_fractions(self, tmp_path):
        # v2's phase-lock signature: every seed identical. No verdict; exit 8.
        rows = [_row("taught", _LEARNED)] * 3 + [_row("no_feed", _CHANCE)] * 3 + [_row("satiated", _CHANCE)] * 3
        proc = _run_v3(rows, tmp_path)
        assert "SEED-INVARIANT" in proc.stdout
        assert proc.returncode == 8

    def test_v3_apparatus_gate_refuses_fewer_than_three_seeds(self, tmp_path):
        # 2 identical rows/arm: the spread check cannot run → no verdict, exit 8
        # (a truncated / --resume'd file must not GRADUATE on the v2 signature).
        rows = [_row("taught", _LEARNED)] * 2 + [_row("no_feed", _CHANCE)] * 2 + [_row("satiated", _CHANCE)] * 2
        proc = _run_v3(rows, tmp_path)
        assert "SKIPPED" in proc.stdout and proc.returncode == 8

    def test_v3_s3_assertions_refuse_a_credited_satiated_arm(self, tmp_path):
        rows = (
            [_row("taught", _jitter(_LEARNED, k)) for k in range(3)]
            + [_row("no_feed", _jitter(_CHANCE, k)) for k in range(3)]
            + [_row("satiated", _jitter(_CHANCE, k)) for k in range(3)]
        )
        for r in rows:
            if r["arm"] == "satiated":
                for m in r["fade"].values():
                    m["credited_rate"] = 0.3  # a credit reached the never-hungry infant
        proc = _run_v3(rows, tmp_path)
        assert "S3 assertions VIOLATED" in proc.stdout and proc.returncode == 8

    def test_v2_default_is_unchanged_by_v3_rows(self, tmp_path):
        # Feeding v3-shaped data to the default gate must still be the v2 verdict.
        rows = [_row("taught", _LEARNED)] * 3 + [_row("no_feed", _CHANCE)] * 3 + [_row("satiated", _CHANCE)] * 3
        proc = _run(rows, tmp_path)
        assert "HUNGER-NECESSARY" not in proc.stdout
        assert proc.returncode == 0
