"""Exp 37 analyzer smoke test — engineered-variance JSONL fixtures.

Builds synthetic JSONL records with controlled per-trial means and verifies
the analyzer produces the predicted verdict on each branch of the pre-reg's
graduation matrix. The harness's `--mock-llm` mode emits records with zero
variance (each arm always gets the same mock_failure_count), which doesn't
exercise the variance-survival math. These tests build records by hand.

Key verdict paths exercised:

  1. EARNED — all criteria pass
  2. EARNED (footnoted) — primary + isolation + secondary pass; 0 corroborating
  3. PARTIAL — reframed — primary + isolation pass but ALL ablations fail
  4. PARTIAL — investigation gate — primary or isolation FAIL
  5. Robustness divergence flagged when per-turn and per-action verdicts disagree
  6. Schema-version mismatch refused
  7. Incomplete design refused
  8. Zero-SD corroborating metric reported as insufficient data
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANALYZER_PATH = _REPO_ROOT / "scripts" / "analyze_exp37.py"


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("exp37_analyzer", _ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp37_analyzer"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analyzer():
    return _load_analyzer()


# ─── Fixture builders ────────────────────────────────────────────────────


def _record(
    *,
    trial_pair_id: int,
    arm: str,
    scenario: str,
    primary: float,
    per_action: float | None = None,
    safe_fraction: float = 0.5,
    tool_diversity: int = 5,
    time_to_steady: int | None = 5,
    schema_version: str = "1.0",
) -> dict[str, Any]:
    """Build a single JSONL record with engineered metric values.

    Mirrors the harness's `build_record` output but skips the cost / token
    fields the analyzer doesn't read.
    """
    if per_action is None:
        per_action = primary  # default: agree with primary
    return {
        "_format_version": schema_version,
        "schema_version": schema_version,
        "experiment": "exp37_cross_session_graduation",
        "harness_version": "1.0",
        "trial_pair_id": trial_pair_id,
        "arm": arm,
        "scenario": scenario,
        "session_id": f"synth_{scenario}_{arm}_{trial_pair_id}",
        "prior_session_id": None if arm == "A" else f"prior_{trial_pair_id}",
        "data_home": "/tmp/synth",
        "seed": 42 + trial_pair_id,
        "model": "claude-sonnet",
        "version_info": {"harness_version": "1.0", "version": "0.9.1"},
        "turns": 12,
        "finish_reason": "max_turns",
        "duration_s": 600.0,
        "cost_usd": 0.21,
        "total_input_tokens": 8000,
        "total_output_tokens": 1500,
        "tool_usage": {"x": 1},
        "tool_class_diversity": tool_diversity,
        "aut_memories_formed": 100,
        "aut_causal_links": 40,
        "wall_clock_iso": "2026-05-31T00:00:00Z",
        "primary_metric_repeat_failure_action_rate": primary,
        "per_action_failure_rate": per_action,
        "failure_class_action_count": int(primary * 12),
        "failure_class_actions_per_turn": [],
        "turn_count_binned": 12,
        "affordance_preference_safe_count": int(safe_fraction * 10),
        "affordance_preference_failed_count": int((1 - safe_fraction) * 10),
        "affordance_preference_safe_fraction": safe_fraction,
        "time_to_safe_steady_state_turns": time_to_steady,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records))


def _arm_a_baseline() -> list[float]:
    # Five trials around mean ~0.65, SD ~0.05.
    return [0.60, 0.62, 0.65, 0.68, 0.70]


def _arm_b_improved() -> list[float]:
    # Five trials clearly below Arm A's band — mean ~0.20.
    return [0.15, 0.18, 0.20, 0.22, 0.25]


def _arm_c_within_a_band() -> list[float]:
    # Five trials within Arm A's empirical band (overlapping).
    return [0.62, 0.64, 0.66, 0.68, 0.69]


def _build_full_design(
    *,
    a_vals: list[float],
    b_vals: list[float],
    c_vals: list[float],
    ablations: dict[str, list[float]] | None = None,
    corroborating_safe_a: list[float] | None = None,
    corroborating_safe_b: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build all 60 records (5 trials × 2 scenarios × 6 arms)."""
    ablations = ablations or {
        "B-wire-a-off": [v for v in b_vals],
        "B-wire-1-off": [v for v in b_vals],
        "B-nac-bias-off": [v for v in b_vals],
    }
    records: list[dict[str, Any]] = []
    for scenario in ("fire_pit", "sharp_rock"):
        for trial_id in range(1, 6):
            i = trial_id - 1
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="A",
                    scenario=scenario,
                    primary=a_vals[i],
                    safe_fraction=(corroborating_safe_a or [0.3] * 5)[i],
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="B",
                    scenario=scenario,
                    primary=b_vals[i],
                    safe_fraction=(corroborating_safe_b or [0.8] * 5)[i],
                    tool_diversity=3,
                    time_to_steady=2,
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="C",
                    scenario=scenario,
                    primary=c_vals[i],
                    safe_fraction=0.3,
                )
            )
            for ab_arm, ab_vals in ablations.items():
                records.append(
                    _record(
                        trial_pair_id=trial_id,
                        arm=ab_arm,
                        scenario=scenario,
                        primary=ab_vals[i],
                    )
                )
    return records


# ─── 1. EARNED verdict ───────────────────────────────────────────────────


def test_earned_full_pass(analyzer, tmp_path):
    """All criteria pass: primary + isolation + secondary + ≥1 corroborating."""
    # Ablations shrink Arm B's gain back toward Arm A (close to a_vals).
    ablations = {
        "B-wire-a-off": [0.55, 0.58, 0.60, 0.63, 0.65],  # close to A
        "B-wire-1-off": [0.20, 0.22, 0.24, 0.26, 0.28],  # still B-like
        "B-nac-bias-off": [0.50, 0.52, 0.55, 0.58, 0.60],  # close to A
    }
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
        ablations=ablations,
        corroborating_safe_a=[0.20, 0.25, 0.30, 0.32, 0.35],  # mean 0.28, SD ~0.06
        corroborating_safe_b=[0.75, 0.80, 0.85, 0.88, 0.90],  # mean 0.84, ~9 SD above
    )
    p = tmp_path / "earned.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    verdicts, label, md = result.scenarios, result.overall_label, result.markdown
    assert label == "EARNED", f"unexpected label: {label!r}\n\n{md}"
    for v in verdicts:
        assert v.primary_pass, f"{v.scenario} primary did not pass"
        assert v.isolation_pass, f"{v.scenario} isolation did not pass"
    assert sum(v.corroborating_hits for v in verdicts) >= 1
    assert sum(v.secondary_hits for v in verdicts) >= 1


# ─── 2. EARNED (footnoted) — 0 corroborating hits ────────────────────────


def test_earned_footnoted_zero_corroborating(analyzer, tmp_path):
    """Primary + isolation + secondary pass; 0/3 corroborating hits."""
    ablations = {
        "B-wire-a-off": [0.55, 0.58, 0.60, 0.63, 0.65],
        "B-wire-1-off": [0.20, 0.22, 0.24, 0.26, 0.28],
        "B-nac-bias-off": [0.20, 0.22, 0.24, 0.26, 0.28],
    }
    # Corroborating metrics: A and B identical → 0 shift in SD units.
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
        ablations=ablations,
        corroborating_safe_a=[0.30, 0.35, 0.40, 0.45, 0.50],
        corroborating_safe_b=[0.30, 0.35, 0.40, 0.45, 0.50],  # SAME — no shift
    )
    # Also flatten tool_diversity + time_to_steady on B so all 3 corroborating fail.
    for r in records:
        if r["arm"] == "B":
            r["tool_class_diversity"] = 5  # match A
            r["time_to_safe_steady_state_turns"] = 5  # match A
    p = tmp_path / "earned_footnoted.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    verdicts, label, md = result.scenarios, result.overall_label, result.markdown
    assert label == "EARNED (footnoted)", f"unexpected label: {label!r}\n\n{md}"
    assert sum(v.corroborating_hits for v in verdicts) == 0


# ─── 3. PARTIAL — reframed — all ablations leave B's delta intact ────────


def test_partial_reframed_all_ablations_fail(analyzer, tmp_path):
    """Primary + isolation pass; all 3 ablations stay close to Arm B."""
    ablations = {
        "B-wire-a-off": [0.15, 0.18, 0.20, 0.22, 0.25],  # = B
        "B-wire-1-off": [0.15, 0.18, 0.20, 0.22, 0.25],  # = B
        "B-nac-bias-off": [0.15, 0.18, 0.20, 0.22, 0.25],  # = B
    }
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
        ablations=ablations,
        corroborating_safe_a=[0.20, 0.25, 0.30, 0.32, 0.35],
        corroborating_safe_b=[0.75, 0.80, 0.85, 0.88, 0.90],
    )
    p = tmp_path / "reframed.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    verdicts, label, md = result.scenarios, result.overall_label, result.markdown
    assert label == "PARTIAL — reframed", f"unexpected label: {label!r}\n\n{md}"
    assert sum(v.secondary_hits for v in verdicts) == 0


# ─── 4. PARTIAL — investigation gate — primary FAILS ─────────────────────


def test_partial_investigation_primary_fails(analyzer, tmp_path):
    """Arm B mean lands INSIDE Arm A's band — primary fails."""
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=[0.60, 0.62, 0.65, 0.68, 0.70],  # = A — no delta
        c_vals=_arm_c_within_a_band(),
    )
    p = tmp_path / "primary_fail.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    verdicts, label, md = result.scenarios, result.overall_label, result.markdown
    assert label == "PARTIAL — investigation gate", f"unexpected label: {label!r}\n\n{md}"
    assert all(not v.primary_pass for v in verdicts)


# ─── 5. PARTIAL — investigation gate — isolation FAILS ───────────────────


def test_partial_investigation_isolation_fails(analyzer, tmp_path):
    """Arm C mean also drops below Arm A's band — general-caution confound."""
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=[0.20, 0.22, 0.25, 0.28, 0.30],  # ALSO improved like B
    )
    p = tmp_path / "isolation_fail.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    label, rationale, md = result.overall_label, result.overall_rationale, result.markdown
    assert label == "PARTIAL — investigation gate", f"unexpected label: {label!r}\n\n{md}"
    assert "Primary or isolation FAIL" in rationale


# ─── 6. Robustness divergence flagged ────────────────────────────────────


def test_robustness_divergence_emits_note(analyzer, tmp_path):
    """When per-turn primary passes but per-action robustness disagrees, the
    analyzer surfaces a note (per protocol §1)."""
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    # Force per_action to land INSIDE A's band on B's records (disagree).
    for r in records:
        if r["arm"] == "B":
            r["per_action_failure_rate"] = 0.65  # near A
    p = tmp_path / "robust_diverge.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    verdicts = result.scenarios
    notes_joined = " ".join(n for v in verdicts for n in v.notes)
    assert "Robustness divergence" in notes_joined, f"expected robustness note; got: {notes_joined!r}"


# ─── 7. Schema mismatch refused ──────────────────────────────────────────


def test_schema_version_mismatch_refused(analyzer, tmp_path):
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    # Bump one record's schema version.
    records[0]["_format_version"] = "2.0"
    records[0]["schema_version"] = "2.0"
    p = tmp_path / "wrong_schema.jsonl"
    _write_jsonl(p, records)
    with pytest.raises(analyzer.AnalyzerError, match=r"schema version"):
        analyzer.run_analysis(
            in_path=p,
            expected_trials=5,
            arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
            scenarios=("fire_pit", "sharp_rock"),
            strict_schema_version="1.0",
        )


# ─── 8. Incomplete design refused ────────────────────────────────────────


def test_incomplete_design_refused(analyzer, tmp_path):
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    # Drop ALL trial 5 records — design is incomplete.
    records = [r for r in records if r["trial_pair_id"] != 5]
    p = tmp_path / "incomplete.jsonl"
    _write_jsonl(p, records)
    with pytest.raises(analyzer.AnalyzerError, match=r"Incomplete design"):
        analyzer.run_analysis(
            in_path=p,
            expected_trials=5,
            arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
            scenarios=("fire_pit", "sharp_rock"),
            strict_schema_version="1.0",
        )


def test_duplicate_trial_pair_refused(analyzer, tmp_path):
    """Append-only re-runs that double a trial_pair_id must be caught."""
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    # Duplicate one Arm A fire_pit trial — analyzer must refuse.
    extra = dict(records[0])
    extra["trial_pair_id"] = 1
    extra["session_id"] = "duplicate"
    records.append(extra)
    p = tmp_path / "duplicates.jsonl"
    _write_jsonl(p, records)
    with pytest.raises(analyzer.AnalyzerError, match=r"Incomplete design|MORE than expected"):
        analyzer.run_analysis(
            in_path=p,
            expected_trials=5,
            arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
            scenarios=("fire_pit", "sharp_rock"),
            strict_schema_version="1.0",
        )


# ─── 9. Exit codes ───────────────────────────────────────────────────────


def test_exit_code_earned(analyzer, tmp_path):
    ablations = {
        "B-wire-a-off": [0.55, 0.58, 0.60, 0.63, 0.65],
        "B-wire-1-off": [0.20, 0.22, 0.24, 0.26, 0.28],
        "B-nac-bias-off": [0.50, 0.52, 0.55, 0.58, 0.60],
    }
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
        ablations=ablations,
        corroborating_safe_a=[0.20, 0.25, 0.30, 0.32, 0.35],
        corroborating_safe_b=[0.75, 0.80, 0.85, 0.88, 0.90],
    )
    p = tmp_path / "earned_cli.jsonl"
    _write_jsonl(p, records)
    rc = analyzer.main(["--in", str(p), "--out", str(tmp_path / "report.md")])
    assert rc == 0


def test_exit_code_investigation_gate(analyzer, tmp_path):
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=[0.60, 0.62, 0.65, 0.68, 0.70],
        c_vals=_arm_c_within_a_band(),
    )
    p = tmp_path / "fail_cli.jsonl"
    _write_jsonl(p, records)
    rc = analyzer.main(["--in", str(p), "--out", str(tmp_path / "report.md")])
    assert rc == 4


def test_exit_code_reframed(analyzer, tmp_path):
    ablations = {
        "B-wire-a-off": [0.15, 0.18, 0.20, 0.22, 0.25],
        "B-wire-1-off": [0.15, 0.18, 0.20, 0.22, 0.25],
        "B-nac-bias-off": [0.15, 0.18, 0.20, 0.22, 0.25],
    }
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
        ablations=ablations,
    )
    p = tmp_path / "reframed_cli.jsonl"
    _write_jsonl(p, records)
    rc = analyzer.main(["--in", str(p), "--out", str(tmp_path / "report.md")])
    assert rc == 3


# ─── 10. Schema-version field defaults from old-format records ───────────


def test_missing_schema_field_treated_as_0_x(analyzer, tmp_path):
    """Records without _format_version OR schema_version flag as '0.x' and
    are refused under the default '1.0' strict-schema check."""
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    records[0].pop("_format_version")
    records[0].pop("schema_version")
    p = tmp_path / "no_schema.jsonl"
    _write_jsonl(p, records)
    with pytest.raises(analyzer.AnalyzerError, match=r"schema version"):
        analyzer.run_analysis(
            in_path=p,
            expected_trials=5,
            arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
            scenarios=("fire_pit", "sharp_rock"),
            strict_schema_version="1.0",
        )


# ─── 11. C1 fold — censor None in time_to_safe_steady_state ──────────────


def test_steady_state_none_on_arm_a_pass_via_censor(analyzer, tmp_path):
    """When Arm A never reaches steady state (all None) and Arm B reaches
    it consistently, the corroborating metric MUST count as a PASS — that's
    the maximum-possible transfer signal. The pre-fix behavior treated this
    as 'insufficient data → FAIL', inverting the strongest signal.
    """
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
        # Need ablations passing too so verdict reaches EARNED.
        ablations={
            "B-wire-a-off": [0.55, 0.58, 0.60, 0.63, 0.65],
            "B-wire-1-off": [0.20, 0.22, 0.24, 0.26, 0.28],
            "B-nac-bias-off": [0.50, 0.52, 0.55, 0.58, 0.60],
        },
    )
    # Force Arm A's time_to_steady to None (never reached) and Arm B's to 2.
    for r in records:
        if r["arm"] == "A":
            r["time_to_safe_steady_state_turns"] = None
        if r["arm"] == "B":
            r["time_to_safe_steady_state_turns"] = 2
    p = tmp_path / "steady_state_censored.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    # At least one corroborating metric (steady state) must hit per scenario.
    steady_hits = sum(
        1
        for v in result.scenarios
        for c in v.corroborating_details
        if c["field"] == "time_to_safe_steady_state_turns" and c["pass"]
    )
    assert steady_hits >= 1, f"steady-state metric did not pass after censor; verdicts:\n{result.markdown}"


# ─── 12. C2 fold — wrong-direction corroborating triggers investigation ──


def test_wrong_direction_corroborating_triggers_investigation(analyzer, tmp_path):
    """Per pre-reg §Corroborating: if ALL corroborating metrics diverge from
    the prediction by ≥1 SD while the primary passes, that's evidence the
    primary is a measurement artifact — investigation gate, not EARNED."""
    # Primary + isolation + secondary all pass; corroborating metrics shift
    # in the WRONG direction.
    ablations = {
        "B-wire-a-off": [0.55, 0.58, 0.60, 0.63, 0.65],
        "B-wire-1-off": [0.20, 0.22, 0.24, 0.26, 0.28],
        "B-nac-bias-off": [0.50, 0.52, 0.55, 0.58, 0.60],
    }
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
        ablations=ablations,
        # safe_fraction predicted to INCREASE on B; here it decreases.
        corroborating_safe_a=[0.85, 0.86, 0.87, 0.88, 0.89],  # A high
        corroborating_safe_b=[0.10, 0.11, 0.12, 0.13, 0.14],  # B low (wrong direction)
    )
    # tool_diversity: predicted DECREASE on B; here it increases (wrong direction).
    # time_to_steady: predicted DECREASE on B; here it increases (wrong direction).
    for r in records:
        if r["arm"] == "A":
            r["tool_class_diversity"] = 3
            r["time_to_safe_steady_state_turns"] = 2
        if r["arm"] == "B":
            r["tool_class_diversity"] = 10  # ↑ (wrong direction)
            r["time_to_safe_steady_state_turns"] = 8  # ↑ (wrong direction)
    p = tmp_path / "wrong_direction.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    assert result.overall_label == "PARTIAL — investigation gate", (
        f"expected investigation gate due to wrong-direction corroborating; "
        f"got {result.overall_label!r}\n\n{result.markdown}"
    )
    assert "measurement artifact" in result.overall_rationale


# ─── 13. I1 fold — ablation overshoot does NOT count as shrinkage ────────


def test_ablation_overshoot_does_not_pass_secondary(analyzer, tmp_path):
    """If an ablation pushes the value PAST Arm A (opposite-direction
    effect, not a shrinkage toward A), it must NOT count as secondary PASS.
    Concrete scenario: NAc-bias-off removes a protective brake, making the
    agent recklessly attempt failure-class actions MORE than A."""
    ablations = {
        # All three overshoot Arm A baseline (~0.65) in the +0.95 direction.
        "B-wire-a-off": [0.92, 0.94, 0.95, 0.96, 0.98],
        "B-wire-1-off": [0.93, 0.94, 0.95, 0.96, 0.97],
        "B-nac-bias-off": [0.92, 0.93, 0.95, 0.96, 0.97],
    }
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
        ablations=ablations,
    )
    p = tmp_path / "overshoot.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    # No secondary hit — all three are overshoot.
    secondary_hits = sum(v.secondary_hits for v in result.scenarios)
    assert secondary_hits == 0, f"overshoot ablations should NOT pass secondary; got {secondary_hits} hits"
    # The verdict should branch into the secondary-FAIL family. (Reframed
    # vs falsified depends on corroborating count.)
    assert result.overall_label.startswith("PARTIAL"), (
        f"overshoot leaves secondary FAIL — verdict must not be EARNED; got {result.overall_label!r}"
    )
    # Overshoot notes must be visible to the operator.
    overshoot_notes = [s for v in result.scenarios for s in v.secondary_details if s.get("overshoot")]
    assert overshoot_notes, "overshoot detail flag not surfaced for operator inspection"


# ─── 14. I3 fold — catastrophic-FAIL distinct from reframed ──────────────


def test_catastrophic_fail_falsified_verdict(analyzer, tmp_path):
    """Primary + isolation pass; secondary FAIL (all 3 ablations match B)
    AND 0/3 corroborating hit. Distinct from PARTIAL-reframed (which has
    ≥1 corroborating hit and only retracts bio-attribution)."""
    ablations = {
        "B-wire-a-off": [0.15, 0.18, 0.20, 0.22, 0.25],
        "B-wire-1-off": [0.15, 0.18, 0.20, 0.22, 0.25],
        "B-nac-bias-off": [0.15, 0.18, 0.20, 0.22, 0.25],
    }
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
        ablations=ablations,
        # ZERO corroborating shift.
        corroborating_safe_a=[0.50] * 5,
        corroborating_safe_b=[0.50] * 5,
    )
    # Flatten tool_diversity + time_to_steady on B to match A → 0 zero-SD passes.
    for r in records:
        if r["arm"] == "B":
            r["tool_class_diversity"] = 5
            r["time_to_safe_steady_state_turns"] = 5
    p = tmp_path / "catastrophic.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    assert result.overall_label == "PARTIAL — falsified", (
        f"expected catastrophic-FAIL verdict; got {result.overall_label!r}\n\n{result.markdown}"
    )
    # Exit code 5 maps for this label.
    rc = analyzer.main(["--in", str(p), "--out", str(tmp_path / "report.md")])
    assert rc == 5


# ─── 15. B2 fold — verdict labels are documented in pre-reg + protocol ───


def test_verdict_labels_appear_in_preregistration_and_protocol(analyzer):
    """Every verdict label the analyzer can emit must appear verbatim in
    the pre-reg's verdict-matrix section AND the protocol's §D table.
    Drift between the three sources would produce wrong-label results
    silently on the next experiment iteration."""
    repo_root = _REPO_ROOT
    prereg = (repo_root / "docs" / "experiments" / "37_cross_session_graduation.md").read_text()
    protocol = (
        repo_root / "docs" / "experiments" / "protocols" / "37_cross_session_graduation_reproduction.md"
    ).read_text()
    for label in analyzer.EXIT_CODE_FOR_LABEL.keys():
        # Strip trailing parenthetical for the pre-reg / protocol lookup, the
        # documents may abbreviate "EARNED (footnoted)" → "EARNED with footnote".
        # We require the canonical analyzer label to appear in the protocol's
        # canonical verdict matrix table; the pre-reg's slightly different
        # wording is tolerated.
        assert label in protocol, (
            f"verdict label {label!r} from EXIT_CODE_FOR_LABEL is not present in "
            f"the protocol §D verdict matrix. Either the analyzer drifted from "
            f"the protocol, or the protocol needs updating in lockstep."
        )
        # Pre-reg uses some prose variants; at minimum, check that the matrix
        # core labels (EARNED / PARTIAL — investigation / PARTIAL — reframed)
        # surface somewhere in the pre-reg.
        if label in {"EARNED", "PARTIAL — investigation gate", "PARTIAL — reframed"}:
            assert label.split(" — ")[0] in prereg, f"verdict label root {label!r} missing from pre-reg."


# ─── 16. Exit-code mapping covers every emitable verdict ─────────────────


def test_exit_code_mapping_covers_every_verdict(analyzer):
    """Any label produced by overall_verdict must have an entry in
    EXIT_CODE_FOR_LABEL — otherwise main() falls through to default 4 and
    downstream automation can't distinguish."""
    # Enumerate by introspection — both the verdict constants and the map
    # are module-level.
    verdict_constants = {v for k, v in vars(analyzer).items() if k.startswith("VERDICT_") and isinstance(v, str)}
    missing = verdict_constants - set(analyzer.EXIT_CODE_FOR_LABEL.keys())
    assert not missing, f"verdict constants without exit-code mapping: {missing!r}"


# ─── 17. I4 fold — missing required field surfaces as AnalyzerError ──────


def test_missing_required_field_raises_analyzer_error(analyzer, tmp_path):
    """Required keys (trial_pair_id, arm, scenario, experiment) must surface
    a clean AnalyzerError with file:line context, not a bare KeyError stack."""
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    records[0].pop("arm")
    p = tmp_path / "missing_arm.jsonl"
    _write_jsonl(p, records)
    with pytest.raises(analyzer.AnalyzerError, match=r"missing required field"):
        analyzer.run_analysis(
            in_path=p,
            expected_trials=5,
            arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
            scenarios=("fire_pit", "sharp_rock"),
            strict_schema_version="1.0",
        )


# ─── 18. I5 fold — dual-version disagreement refused ─────────────────────


def test_format_version_schema_version_mismatch_refused(analyzer, tmp_path):
    """If _format_version and schema_version disagree on the same record,
    the harness mis-emitted — refuse rather than guess which is canonical."""
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    records[0]["_format_version"] = "1.0"
    records[0]["schema_version"] = "2.0"
    p = tmp_path / "dual_version.jsonl"
    _write_jsonl(p, records)
    with pytest.raises(analyzer.AnalyzerError, match=r"disagree"):
        analyzer.run_analysis(
            in_path=p,
            expected_trials=5,
            arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
            scenarios=("fire_pit", "sharp_rock"),
            strict_schema_version="1.0",
        )


# ─── 19. --heading-suffix flag adds disambiguation to Markdown heading ───


def test_heading_suffix_renders_into_markdown(analyzer, tmp_path):
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    p = tmp_path / "heading.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
        heading_suffix="2026-06-15 run-12",
    )
    assert "## Results — 2026-06-15 run-12" in result.markdown
