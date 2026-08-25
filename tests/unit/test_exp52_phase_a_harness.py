"""Exp 52 Phase A harness smoke (``scripts/orient_substrate/9_hunger_relief_orient.py``).

Runs the scripted harness end-to-end at toy size and pins the parts the
pre-registration relies on: the report carries the four gates + mechanism-sanity
telemetry; under ``--credit relief`` the satiated arm mints zero credits and every
taught credit is +1; ``--credit constant`` (the A/B against probe 4) still runs.
Not a science test — seeds/ticks are far below the frozen parameters."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "orient_substrate" / "9_hunger_relief_orient.py"


def _run(tmp_path: Path, *extra: str) -> dict:
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--seeds", "2", "--ticks", "80", "--bin", "20", "--json", str(out), *extra],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(out.read_text())


def test_relief_mode_report_shape_and_mechanism_sanity(tmp_path):
    rep = _run(tmp_path)
    assert rep["credit"] == "relief"
    assert set(rep["gates"]) == {"LEARNED", "HUNGER_NECESSARY", "MOTHER_NECESSARY", "MECHANISM_SANITY"}
    assert rep["gates"]["MECHANISM_SANITY"] is True
    sat = rep["telemetry"]["satiated"]
    assert all(t["credits"] == 0 for t in sat) and all(t["fed"] > 0 for t in sat)
    taught = rep["telemetry"]["taught"]
    assert all(t["credits"] == t["fed"] for t in taught)
    assert all(set(t["credit_rewards"]) <= {1.0} for t in taught)
    assert rep["provenance"]["executed_git_hash"]
    assert set(rep["curves"]) == {"taught", "satiated", "yoked", "no_feed"}
    assert all(t["hunger_at_feed_median"] is not None for t in taught)


def test_constant_mode_credits_every_feed_with_gain(tmp_path):
    rep = _run(tmp_path, "--credit", "constant")
    assert rep["credit"] == "constant"
    assert rep["gates"]["MECHANISM_SANITY"] is True
    # constant credit ignores need: the satiated arm IS credited on every feed
    assert all(t["credits"] == t["fed"] for t in rep["telemetry"]["satiated"])
