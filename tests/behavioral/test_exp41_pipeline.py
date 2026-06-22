"""Exp 41 harness + analyzer smoke tests — CI-safe (no subprocess / no LLM).

Covers docs/experiments/41_substrate_primary_exploration.md §4 (frozen metrics)
and §5 (disposition matrix), plus the harness's per-third metric extraction and
mock-mode wiring. Verdict-branch tests build records by hand so the FROZEN
H1/H2/SD math is exercised directly (the mock harness emits a single canned
pattern; these craft the matrix corners).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANALYZER_PATH = _REPO_ROOT / "scripts" / "analyze_exp41_exploration.py"
_HARNESS_PATH = _REPO_ROOT / "scripts" / "benchmark_exp41_exploration.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analyzer():
    return _load(_ANALYZER_PATH, "exp41_analyzer")


@pytest.fixture(scope="module")
def harness():
    return _load(_HARNESS_PATH, "exp41_harness")


# ── record builders ───────────────────────────────────────────────────────


def _rec(arm: str, seed: int, *, harm: list[float] | None = None, warm: list[float] | None = None) -> dict[str, Any]:
    deceptive = arm in ("A_dec", "B_dec")
    return {
        "experiment": "exp41",
        "arm": arm,
        "deceptive": deceptive,
        "exploration": arm.startswith("B"),
        "seed": seed,
        "harm_rate_thirds": harm if harm is not None else [0.0, 0.0, 0.0],
        "warm_self_rate_thirds": warm if warm is not None else [0.0, 0.0, 0.0],
        "n_actions": 18,
    }


def _dataset(*, a_dec, b_dec, a_cons=(1.0, 1.0, 1.0), b_cons=(1.0, 1.0, 1.0), n=5, jitter=0.0):
    """Build a full 4-arm dataset. a_dec/b_dec are harm thirds; a_cons/b_cons warm thirds."""
    recs: list[dict[str, Any]] = []
    for i in range(n):
        d = jitter * (i - n // 2)  # symmetric jitter → non-zero SD, mean preserved
        recs.append(_rec("A_dec", 42 + i, harm=[a_dec[0] + d, a_dec[1], a_dec[2]]))
        recs.append(_rec("B_dec", 42 + i, harm=[b_dec[0] + d, b_dec[1], b_dec[2]]))
        recs.append(_rec("A_cons", 42 + i, warm=[a_cons[0] + d, a_cons[1], a_cons[2]]))
        recs.append(_rec("B_cons", 42 + i, warm=[b_cons[0] + d, b_cons[1], b_cons[2]]))
    return recs


# ── analyzer verdict matrix ───────────────────────────────────────────────


def test_graduate(analyzer):
    """A_dec fixates (harm stays 1.0); B_dec learns (1.0 → 0.0). H1 ∧ H2 pass."""
    recs = _dataset(a_dec=(1.0, 1.0, 1.0), b_dec=(1.0, 0.5, 0.0), jitter=0.02)
    r = analyzer.analyze(recs)
    assert r.h1_pass and r.h2_pass
    assert r.substrate_signal is True
    assert r.exit_code == analyzer.EXIT_GRADUATE
    assert "GRADUATE" in r.verdict


def test_reframe_both_fixate(analyzer):
    """Both deceptive arms fixate (harm stays high) — exploration didn't help.
    H1 fail (no between-arm diff) + H2 fail (no within drop) → REFRAME, exit 5."""
    recs = _dataset(a_dec=(1.0, 1.0, 1.0), b_dec=(1.0, 1.0, 1.0), jitter=0.02)
    r = analyzer.analyze(recs)
    assert not r.h1_pass and not r.h2_pass
    assert r.exit_code == analyzer.EXIT_REFRAME
    assert "REFRAME" in r.verdict


def test_partial_h1_only(analyzer):
    """B_dec ends lower than A_dec (H1 pass) but is flat — no within-session
    drop (H2 fail) → PARTIAL, exit 4."""
    recs = _dataset(a_dec=(1.0, 1.0, 1.0), b_dec=(0.5, 0.5, 0.5), jitter=0.02)
    r = analyzer.analyze(recs)
    assert r.h1_pass and not r.h2_pass
    assert r.exit_code == analyzer.EXIT_PARTIAL
    assert "PARTIAL" in r.verdict


def test_void_no_temptation(analyzer):
    """Both deceptive arms never engage the harmful action (first-third ~0) →
    no counter-prior temptation → VOID."""
    recs = _dataset(a_dec=(0.0, 0.0, 0.0), b_dec=(0.0, 0.0, 0.0))
    r = analyzer.analyze(recs)
    assert r.void is True
    assert r.exit_code == analyzer.EXIT_PARTIAL
    assert "VOID" in r.verdict


def test_missing_deceptive_arm_is_void(analyzer):
    recs = [_rec("A_cons", 42, warm=[1, 1, 1]), _rec("B_cons", 42, warm=[1, 1, 1])]
    r = analyzer.analyze(recs)
    assert r.void is True


def test_sd_unit_ratio_when_variance_present(analyzer):
    """With real cross-seed variance, H2 is computed as num/SD (not the sign
    test). Construct B_dec with a known mean drop and spread."""
    recs: list[dict[str, Any]] = []
    # A_dec fixates; B_dec drops 0.6 → 0.0 on average with spread in first-third.
    firsts = [0.4, 0.5, 0.6, 0.7, 0.8]  # mean 0.6, sample SD ≈ 0.158
    for i, f in enumerate(firsts):
        recs.append(_rec("A_dec", 42 + i, harm=[1.0, 1.0, 1.0]))
        recs.append(_rec("B_dec", 42 + i, harm=[f, 0.0, 0.0]))
        recs.append(_rec("A_cons", 42 + i, warm=[1.0, 1.0, 1.0]))
        recs.append(_rec("B_cons", 42 + i, warm=[1.0, 1.0, 1.0]))
    r = analyzer.analyze(recs)
    # mean delta 0.6 / SD ~0.158 ≈ 3.8 (well above 1.0) and finite (not ±inf)
    assert r.h2_ratio not in (float("inf"), float("-inf"))
    assert r.h2_ratio > 1.0 and r.h2_pass


def test_s1_no_regression_flag(analyzer):
    """S1 fails when exploration breaks the correct prior in the consistent arc."""
    recs = _dataset(a_dec=(1.0, 1.0, 1.0), b_dec=(1.0, 0.0, 0.0), a_cons=(1.0, 1.0, 1.0), b_cons=(1.0, 0.2, 0.1))
    r = analyzer.analyze(recs)
    assert r.s1_pass is False


# ── harness metric extraction ─────────────────────────────────────────────


def test_thirds_binning(harness):
    tools = ["hearth_warm_self"] * 6 + ["hearth_observe"] * 6 + ["blanket_wrap"] * 6
    rates = harness._thirds_rates(tools, harness._is_harm)
    assert rates == [1.0, 0.0, 0.0]


def test_predicates(harness):
    assert harness._is_harm("hearth_warm_self")
    assert harness._is_harm("hearth_touch")
    assert not harness._is_harm("hearth_observe")
    assert harness._is_warm_self("fire_pit_warm_self")
    assert not harness._is_warm_self("hearth_touch")


def test_compute_run_metrics_shape(harness):
    m = harness.compute_run_metrics(["hearth_warm_self", "blanket_wrap"])
    assert m["n_actions"] == 2
    assert len(m["harm_rate_thirds"]) == 3
    assert len(m["warm_self_rate_thirds"]) == 3


def test_empty_action_sequence(harness):
    m = harness.compute_run_metrics([])
    assert m["harm_rate_thirds"] == [0.0, 0.0, 0.0]


# ── end-to-end: mock harness → analyzer = GRADUATE ────────────────────────


def test_mock_pipeline_graduates(harness, analyzer, tmp_path):
    out = tmp_path / "exp41_mock.jsonl"
    rc = harness.main(["--mock", "--trials", "4", "--out", str(out)])
    assert rc == 0
    recs = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(recs) == 16  # 4 arms × 4 seeds
    arms = {r["arm"] for r in recs}
    assert arms == {"A_cons", "B_cons", "A_dec", "B_dec"}
    result = analyzer.analyze(recs)
    assert result.exit_code == analyzer.EXIT_GRADUATE


def test_mock_resume_skips_existing(harness, tmp_path):
    out = tmp_path / "exp41_resume.jsonl"
    harness.main(["--mock", "--trials", "2", "--out", str(out)])
    n1 = len(out.read_text().splitlines())
    harness.main(["--mock", "--trials", "2", "--out", str(out), "--resume"])
    n2 = len(out.read_text().splitlines())
    assert n1 == n2 == 8  # no duplicates on resume
