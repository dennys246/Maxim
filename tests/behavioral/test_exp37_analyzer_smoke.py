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
    fire_approach_count: int = 0,
    # exp37_metric_pivot.md (Path 2) new primary. Default mirrors
    # ``primary`` but inverted (1 - primary) so existing engineered-variance
    # fixtures keep their semantics: an "Arm B improved" fixture had
    # ``primary`` LOW (B < A.p2.5 → legacy PASS); under the pivot, NEW
    # primary is HIGH (B > A.p97.5 → pivot PASS). Both pass together
    # without divergence. Explicit caller override available for
    # divergence-engineering tests.
    positive_approach_fraction: float | None = None,
    engagement_count: int = 5,
    time_to_first_warm_self: int | None = 1,
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
        # cradle_activation_fixes.md P2: descriptive corroborating metric.
        # Default 0 keeps existing engineered-variance tests unchanged; new
        # tests below opt in by passing non-zero values.
        "fire_approach_action_count": fire_approach_count,
        # exp37_metric_pivot.md (Path 2): NEW primary + denominator + new
        # corroborating. Default positive_approach_fraction = 1 - primary
        # mirrors the inverse-direction relationship so existing
        # engineered fixtures (where Arm B's ``primary`` is lower than A's)
        # automatically encode "Arm B's positive-approach-fraction is
        # higher than A's" — both metrics PASS together for EARNED cases,
        # no spurious divergence flagged.
        "positive_approach_engagement_fraction": (
            positive_approach_fraction if positive_approach_fraction is not None else 1.0 - primary
        ),
        "fire_pit_engagement_count": engagement_count,
        "time_to_first_warm_self_action": time_to_first_warm_self,
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
    """exp37_metric_pivot.md (Path 2): when the NEW primary
    (positive_approach_engagement_fraction, increase direction) passes
    but the legacy ROBUSTNESS metric (per_action_failure_rate, decrease
    direction) disagrees, the analyzer surfaces a note. The two metrics
    measure different aspects of the substrate-transfer claim — a
    divergence indicates "B's warm_self preference shifted without
    reducing touch behavior" (or vice versa) and warrants investigation.
    """
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    # Force per_action to land NEAR A on B's records → robustness check
    # FAILS while the new primary still PASSes (positive_approach is
    # populated from 1 - primary by default, so it's still in the
    # "improved" zone for B).
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
    assert "Primary / robustness divergence" in notes_joined, (
        f"expected pivot-era divergence note; got: {notes_joined!r}"
    )


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
    # time_to_first_warm_self: predicted DECREASE on B; here it increases (wrong direction).
    for r in records:
        if r["arm"] == "A":
            r["tool_class_diversity"] = 3
            r["time_to_safe_steady_state_turns"] = 2
            r["time_to_first_warm_self_action"] = 2  # A reaches warm_self at action 2
        if r["arm"] == "B":
            r["tool_class_diversity"] = 10  # ↑ (wrong direction)
            r["time_to_safe_steady_state_turns"] = 8  # ↑ (wrong direction)
            r["time_to_first_warm_self_action"] = 10  # ↑ (wrong direction — substrate fails to bias earlier)
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
    # Exp 38 counter-prior labels live in the Exp 38 pre-registration doc, NOT
    # the Exp 37 protocol — the analyzer is shared but the labels are
    # experiment-scoped (counter_prior_substrate_experiment.md §5).
    exp38_prereg = (repo_root / "docs" / "experiments" / "38_counter_prior_substrate.md").read_text()
    for label in analyzer.EXIT_CODE_FOR_LABEL.keys():
        if label.startswith("COUNTER-PRIOR"):
            assert label in exp38_prereg, (
                f"counter-prior verdict label {label!r} from EXIT_CODE_FOR_LABEL is not "
                f"present in the Exp 38 pre-registration doc. Update the doc in lockstep."
            )
            continue
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


# ─── fire_approach_action_count descriptive corroborating metric ────────


def _build_design_with_approach(
    *,
    fire_pit_a: list[int],
    fire_pit_b: list[int],
) -> list[dict[str, Any]]:
    """Build a minimal full design where ONLY fire_pit's
    fire_approach_action_count varies; sharp_rock stays at the default 0.
    Other metrics keep the standard EARNED-shape values so the analyzer
    runs end-to-end without hitting design-completeness errors.
    """
    a_vals = _arm_a_baseline()
    b_vals = _arm_b_improved()
    c_vals = _arm_c_within_a_band()
    records: list[dict[str, Any]] = []
    for scenario in ("fire_pit", "sharp_rock"):
        for trial_id in range(1, 6):
            i = trial_id - 1
            approach_a = fire_pit_a[i] if scenario == "fire_pit" else 0
            approach_b = fire_pit_b[i] if scenario == "fire_pit" else 0
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="A",
                    scenario=scenario,
                    primary=a_vals[i],
                    safe_fraction=0.3,
                    fire_approach_count=approach_a,
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="B",
                    scenario=scenario,
                    primary=b_vals[i],
                    safe_fraction=0.8,
                    tool_diversity=3,
                    time_to_steady=2,
                    fire_approach_count=approach_b,
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
            for ab_arm in ("B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"):
                records.append(
                    _record(
                        trial_pair_id=trial_id,
                        arm=ab_arm,
                        scenario=scenario,
                        primary=b_vals[i],
                    )
                )
    return records


def test_approach_descriptive_is_none_when_all_zero(analyzer, tmp_path):
    """sharp_rock (and the default fire_pit case) should produce
    ``approach_descriptive=None`` so the analyzer doesn't render a
    degenerate row.
    """
    records = _build_design_with_approach(fire_pit_a=[0] * 5, fire_pit_b=[0] * 5)
    p = tmp_path / "approach_none.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    for v in result.scenarios:
        assert v.approach_descriptive is None
    # Markdown does NOT include the descriptive block when nothing to show.
    assert "Descriptive corroborating" not in result.markdown


def test_approach_descriptive_renders_when_nonzero(analyzer, tmp_path):
    """fire_pit with non-zero approach counts on Arm A and Arm B should
    produce a populated descriptive block in the markdown and on the
    ScenarioVerdict.
    """
    records = _build_design_with_approach(
        fire_pit_a=[2, 3, 2, 3, 4],  # mean 2.8
        fire_pit_b=[3, 4, 4, 5, 5],  # mean 4.2 — higher, predicted direction
    )
    p = tmp_path / "approach_nonzero.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    fire_v = next(v for v in result.scenarios if v.scenario == "fire_pit")
    rock_v = next(v for v in result.scenarios if v.scenario == "sharp_rock")
    assert fire_v.approach_descriptive is not None
    assert fire_v.approach_descriptive["a_mean"] == pytest.approx(2.8)
    assert fire_v.approach_descriptive["b_mean"] == pytest.approx(4.2)
    assert fire_v.approach_descriptive["delta"] == pytest.approx(1.4)
    assert fire_v.approach_descriptive["predicted_direction"] == "same_or_higher"
    # Negative-delta warning note should NOT appear (Δ > 0 here).
    assert fire_v.approach_descriptive["note"] is None
    # sharp_rock still suppressed.
    assert rock_v.approach_descriptive is None
    assert "Descriptive corroborating" in result.markdown
    assert "fire_approach_action_count" in result.markdown


def test_approach_descriptive_negative_delta_emits_warning_note(analyzer, tmp_path):
    """If Arm B's approach count is LOWER than A's (general-avoidance pattern
    suspected) the analyzer attaches an explanatory note to the
    descriptive block. The verdict itself is unchanged — the metric is NOT
    gated.
    """
    records = _build_design_with_approach(
        fire_pit_a=[4, 5, 4, 5, 4],  # mean 4.4
        fire_pit_b=[1, 2, 1, 2, 1],  # mean 1.4 — Arm B avoided the fire entirely
    )
    p = tmp_path / "approach_negative.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    fire_v = next(v for v in result.scenarios if v.scenario == "fire_pit")
    assert fire_v.approach_descriptive is not None
    assert fire_v.approach_descriptive["delta"] < 0
    assert fire_v.approach_descriptive["note"] is not None
    assert "general caution" in fire_v.approach_descriptive["note"]


def test_scenario_verdict_approach_descriptive_default_none(analyzer):
    """Backward-compat: ScenarioVerdict.approach_descriptive defaults to
    None so consumers constructed pre-PR-E continue to work.
    """
    v = analyzer.ScenarioVerdict(
        scenario="dummy",
        primary_pass=True,
        isolation_pass=True,
        primary_pass_robustness=True,
        corroborating_hits=1,
        corroborating_wrong_direction=0,
        corroborating_details=[],
        secondary_hits=1,
        secondary_details=[],
        a_mean=0.5,
        a_sd=0.05,
        a_band=(0.4, 0.6),
        b_mean=0.2,
        c_mean=0.5,
        notes=[],
    )
    assert v.approach_descriptive is None


# ─── exp37_metric_pivot.md (Path 2) — direction flip + pivot semantics ──


def test_primary_metric_is_positive_approach_engagement_fraction(analyzer):
    """Pin the pivot: the analyzer's PRIMARY_METRIC constant should be
    the new field, with direction=increase. Catches accidental revert
    to the pre-pivot constant in a future refactor.
    """
    assert analyzer.PRIMARY_METRIC == "positive_approach_engagement_fraction"
    assert analyzer.PRIMARY_METRIC_DIRECTION == "increase"
    assert analyzer.ROBUSTNESS_METRIC == "per_action_failure_rate"
    assert analyzer.ROBUSTNESS_METRIC_DIRECTION == "decrease"


def test_primary_pass_when_b_above_a_p97_band(analyzer, tmp_path):
    """Pre-2026-06-05 this test asserted ``B.mean > A.p97.5``. After the
    SD-shift swap the assertion shifts to ``(B - A) / A.sd ≥ +1`` SD in
    the predicted direction. _record's default ``positive_approach_fraction
    = 1 - primary`` maps the existing "Arm B improved" fixture (low
    primary on B) onto "high positive_approach on B" → A ~0.35 SD ~0.04,
    B ~0.80 → delta_sd ~11 SD → PASS.
    """
    ablations = {
        "B-wire-a-off": [0.55, 0.58, 0.60, 0.63, 0.65],
        "B-wire-1-off": [0.20, 0.22, 0.24, 0.26, 0.28],
        "B-nac-bias-off": [0.50, 0.52, 0.55, 0.58, 0.60],
    }
    records = _build_full_design(
        a_vals=_arm_a_baseline(),  # primary ~0.65 → positive_approach ~0.35
        b_vals=_arm_b_improved(),  # primary ~0.20 → positive_approach ~0.80
        c_vals=_arm_c_within_a_band(),
        ablations=ablations,
    )
    p = tmp_path / "pivot_earned.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    for v in result.scenarios:
        assert v.primary_pass, (
            f"{v.scenario}: expected primary PASS with positive_approach pivot — "
            f"a_mean={v.a_mean}, a_band={v.a_band}, b_mean={v.b_mean}"
        )


def test_primary_fail_when_b_below_a_band(analyzer, tmp_path):
    """Symmetric to the EARNED test: if Arm B's positive_approach equals
    A's mean (substrate transfer ABSENT or REVERSED), the SD-shift test
    yields ``delta_sd = 0`` → not ≥ +1 SD → FAIL.
    """
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_a_baseline(),  # B same as A → no improvement
        c_vals=_arm_c_within_a_band(),
    )
    # Default 1-primary maps both to ~0.35 → delta_sd = 0 → FAIL.
    p = tmp_path / "pivot_fail.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    for v in result.scenarios:
        assert not v.primary_pass, f"{v.scenario}: expected primary FAIL — a_band={v.a_band}, b_mean={v.b_mean}"


def test_pivot_markdown_uses_sd_shift_predicted_label(analyzer, tmp_path):
    """Post-2026-06-05 the markdown should display ``Δ = X SD`` and the
    SD-shift threshold for B's predicted side, NOT the legacy percentile-
    band reference. Direction-aware via PRIMARY_METRIC_DIRECTION.
    """
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    p = tmp_path / "pivot_md.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    assert "SD" in result.markdown
    assert "need" in result.markdown
    # Legacy percentile-band references must be gone.
    assert "A.p97.5" not in result.markdown
    assert "A.p2.5" not in result.markdown


def test_time_to_first_warm_self_corroborating_present(analyzer, tmp_path):
    """The new corroborating metric should appear in the verdict's
    corroborating_details. Substrate transfer predicts B reaches
    warm_self earlier than A → direction = decrease.
    """
    # Engineer the new metric specifically: A reaches warm_self at action 4,
    # B at action 1 (substrate-biased earlier).
    records = []
    a_vals = _arm_a_baseline()
    b_vals = _arm_b_improved()
    c_vals = _arm_c_within_a_band()
    for scenario in ("fire_pit", "sharp_rock"):
        for trial_id in range(1, 6):
            i = trial_id - 1
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="A",
                    scenario=scenario,
                    primary=a_vals[i],
                    time_to_first_warm_self=4 + i,  # 4..8
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="B",
                    scenario=scenario,
                    primary=b_vals[i],
                    time_to_first_warm_self=1,  # consistently early
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="C",
                    scenario=scenario,
                    primary=c_vals[i],
                    time_to_first_warm_self=4 + i,
                )
            )
            for ab in ("B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"):
                records.append(
                    _record(
                        trial_pair_id=trial_id,
                        arm=ab,
                        scenario=scenario,
                        primary=b_vals[i],
                    )
                )
    p = tmp_path / "ttfws.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    fire_v = next(v for v in result.scenarios if v.scenario == "fire_pit")
    fields = [c["field"] for c in fire_v.corroborating_details]
    assert "time_to_first_warm_self_action" in fields, (
        f"time_to_first_warm_self_action should appear in corroborating; got fields={fields}"
    )
    ttfws_record = next(c for c in fire_v.corroborating_details if c["field"] == "time_to_first_warm_self_action")
    # B's mean (1.0) should be < A's mean (~6.0) → predicted direction
    # (decrease) → PASS.
    assert ttfws_record["pass"], f"expected PASS; got {ttfws_record}"


# ─── exp37_sd_shift.md — SD-shift primary on bounded distributions ──────


def test_sd_shift_primary_pass_on_bounded_distribution(analyzer, tmp_path):
    """Validates the SD-shift swap on the empirical-style case the
    2026-06-05 validation smoke surfaced: Arm A piles up at the ceiling
    of a bounded metric. Engineered to have A SD low enough that
    `A.mean + 1.0 * A.sd` is still within [0, 1], so a B shift of ≥+1 SD
    is reachable.
    """
    # A: tightly clustered near the ceiling. mean=0.92, sd~0.04.
    a_engineered = [0.88, 0.90, 0.92, 0.94, 0.96]
    # B: shifted up by ~3 SD. Still within [0, 1].
    b_engineered = [0.97, 0.98, 0.99, 1.00, 1.00]
    records = []
    for scenario in ("fire_pit", "sharp_rock"):
        for trial_id in range(1, 6):
            i = trial_id - 1
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="A",
                    scenario=scenario,
                    primary=0.5,
                    positive_approach_fraction=a_engineered[i],
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="B",
                    scenario=scenario,
                    primary=0.5,
                    positive_approach_fraction=b_engineered[i],
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="C",
                    scenario=scenario,
                    primary=0.5,
                    positive_approach_fraction=a_engineered[i],  # within A's range
                )
            )
            for ab in ("B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"):
                records.append(
                    _record(
                        trial_pair_id=trial_id,
                        arm=ab,
                        scenario=scenario,
                        primary=0.5,
                        positive_approach_fraction=b_engineered[i],
                    )
                )
    p = tmp_path / "sd_shift_bounded.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    for v in result.scenarios:
        assert v.primary_pass, (
            f"{v.scenario}: SD-shift should PASS on bounded ceiling case — "
            f"a_mean={v.a_mean}, a_sd={v.a_sd}, b_mean={v.b_mean}"
        )


def test_sd_shift_primary_fail_when_b_shift_below_threshold(analyzer, tmp_path):
    """Validates the SD-shift FAIL path: B shifts in the correct
    direction but by less than 1 SD → FAIL. Distinguishes "small
    substrate effect, real but undetectable at this N" from "no
    substrate effect at all."
    """
    a_engineered = [0.40, 0.45, 0.50, 0.55, 0.60]  # mean 0.5, sd ~0.08
    b_engineered = [0.55, 0.55, 0.55, 0.55, 0.55]  # mean 0.55, delta=0.05, ~0.6 SD
    records = []
    for scenario in ("fire_pit", "sharp_rock"):
        for trial_id in range(1, 6):
            i = trial_id - 1
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="A",
                    scenario=scenario,
                    primary=0.5,
                    positive_approach_fraction=a_engineered[i],
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="B",
                    scenario=scenario,
                    primary=0.5,
                    positive_approach_fraction=b_engineered[i],
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="C",
                    scenario=scenario,
                    primary=0.5,
                    positive_approach_fraction=a_engineered[i],
                )
            )
            for ab in ("B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"):
                records.append(
                    _record(
                        trial_pair_id=trial_id,
                        arm=ab,
                        scenario=scenario,
                        primary=0.5,
                        positive_approach_fraction=b_engineered[i],
                    )
                )
    p = tmp_path / "sd_shift_subthreshold.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    for v in result.scenarios:
        assert not v.primary_pass, (
            f"{v.scenario}: SD-shift should FAIL when B shift <1 SD — "
            f"a_sd={v.a_sd}, delta={v.b_mean - v.a_mean if v.b_mean and v.a_mean else 'N/A'}"
        )


def test_sd_shift_zero_sd_fallback_pass_on_directional_shift(analyzer, tmp_path):
    """When Arm A's SD is 0 (all-identical trials), the zero-SD fallback
    fires: pass on directional sign + non-zero shift. This matches the
    I2 corroborating fallback and prevents the test from erroring when
    A has truly zero variance.
    """
    a_engineered = [1.0, 1.0, 1.0, 1.0, 1.0]  # SD = 0
    b_engineered = [1.0, 1.0, 1.0, 1.0, 1.0]  # No shift → should FAIL
    # B same as A → no directional shift → zero-SD fallback rejects.
    records = []
    for scenario in ("fire_pit", "sharp_rock"):
        for trial_id in range(1, 6):
            i = trial_id - 1
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="A",
                    scenario=scenario,
                    primary=0.5,
                    positive_approach_fraction=a_engineered[i],
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="B",
                    scenario=scenario,
                    primary=0.5,
                    positive_approach_fraction=b_engineered[i],
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="C",
                    scenario=scenario,
                    primary=0.5,
                    positive_approach_fraction=a_engineered[i],
                )
            )
            for ab in ("B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"):
                records.append(
                    _record(
                        trial_pair_id=trial_id,
                        arm=ab,
                        scenario=scenario,
                        primary=0.5,
                        positive_approach_fraction=b_engineered[i],
                    )
                )
    p = tmp_path / "sd_shift_zero_sd.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    # B == A (no shift) → fallback rejects → FAIL.
    for v in result.scenarios:
        assert not v.primary_pass, (
            f"{v.scenario}: zero-SD case with no shift should FAIL — "
            f"a_sd={v.a_sd}, delta={(v.b_mean or 0) - (v.a_mean or 0)}"
        )


# ─── Exp 38: structural-absence detection (folds sharp_rock artifact) ─────


def _build_absence_design(*, absent_scenario: str = "sharp_rock") -> list[dict[str, Any]]:
    """Full EARNED-shape design where ``absent_scenario``'s PRIMARY_METRIC
    (positive_approach_engagement_fraction) is identical (0.0) across EVERY arm
    — the real-data shape of sharp_rock (no positive-approach affordance). The
    other scenario keeps the standard EARNED variance.
    """
    a_vals = _arm_a_baseline()
    b_vals = _arm_b_improved()
    c_vals = _arm_c_within_a_band()
    records: list[dict[str, Any]] = []
    for scenario in ("fire_pit", "sharp_rock"):
        absent = scenario == absent_scenario
        pa = 0.0 if absent else None  # None → _record default (1 - primary, varies)
        for trial_id in range(1, 6):
            i = trial_id - 1
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="A",
                    scenario=scenario,
                    primary=a_vals[i],
                    safe_fraction=0.3,
                    positive_approach_fraction=pa,
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="B",
                    scenario=scenario,
                    primary=b_vals[i],
                    safe_fraction=0.8,
                    tool_diversity=3,
                    time_to_steady=2,
                    positive_approach_fraction=pa,
                )
            )
            records.append(
                _record(
                    trial_pair_id=trial_id,
                    arm="C",
                    scenario=scenario,
                    primary=c_vals[i],
                    safe_fraction=0.3,
                    positive_approach_fraction=pa,
                )
            )
            for ab in ("B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"):
                records.append(
                    _record(
                        trial_pair_id=trial_id,
                        arm=ab,
                        scenario=scenario,
                        primary=b_vals[i],
                        positive_approach_fraction=pa,
                    )
                )
    return records


def test_structural_absence_reports_na_not_fail(analyzer, tmp_path):
    """The Exp 37 sharp_rock artifact (counter_prior_substrate_experiment.md
    §6.3): a PRIMARY_METRIC structurally 0 across all arms must report N/A, NOT
    FAIL, and must NOT drag the overall verdict to investigation. fire_pit
    (with real variance) determines the verdict alone."""
    records = _build_absence_design(absent_scenario="sharp_rock")
    p = tmp_path / "absence.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off"),
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    by_scenario = {v.scenario: v for v in result.scenarios}
    assert by_scenario["sharp_rock"].primary_structural_absence is True
    assert by_scenario["fire_pit"].primary_structural_absence is False
    # The structurally-absent primary renders as N/A, never FAIL.
    assert "N/A" in result.markdown
    # The artifact is folded: sharp_rock no longer forces an investigation gate.
    # fire_pit (the only gated scenario) passes primary + isolation, so the
    # overall can never be the investigation verdict that the pre-fix sharp_rock
    # FAIL produced.
    assert result.overall_label != analyzer.VERDICT_PARTIAL_INVESTIGATION
    assert by_scenario["fire_pit"].primary_pass is True


def test_is_structurally_absent_unit(analyzer):
    """Direct unit: constant-across-all-arms → True; any variance → False."""
    arms = ("A", "B", "C")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for arm in arms:
        grouped[("s", arm)] = [{"m": 0.0}, {"m": 0.0}]
    assert analyzer._is_structurally_absent(grouped, "s", "m", arms=arms) is True
    # Introduce a single varying value.
    grouped[("s", "B")][0]["m"] = 0.5
    assert analyzer._is_structurally_absent(grouped, "s", "m", arms=arms) is False
    # A non-zero CONSTANT is still "absent" (no variance to discriminate on).
    for arm in arms:
        grouped[("s", arm)] = [{"m": 1.0}, {"m": 1.0}]
    assert analyzer._is_structurally_absent(grouped, "s", "m", arms=arms) is True


# ─── Exp 38: counter-prior interaction verdicts ──────────────────────────

_CP_ARMS = ("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off")


def _cp_record(*, trial_pair_id, arm, scenario, warm_self_frac, first_contact):
    """A record carrying the Exp 38 counter-prior fields on top of the base
    schema. positive_approach is forced to 0 for deceptive_fire (structural
    absence, real-data shape); warm_self_engagement_fraction is the live
    interaction channel."""
    r = _record(
        trial_pair_id=trial_pair_id,
        arm=arm,
        scenario=scenario,
        primary=0.3,
        positive_approach_fraction=(0.0 if scenario == "deceptive_fire" else warm_self_frac),
    )
    r["warm_self_engagement_fraction"] = warm_self_frac
    r["first_contact_warm_self"] = first_contact
    return r


def _cp_design(*, con_ws, dec_ws, con_fc, dec_fc) -> list[dict[str, Any]]:
    """Build a 60-record counter-prior design. Each of con_ws/dec_ws is a dict
    arm→[5 floats] (warm_self_frac); con_fc/dec_fc is arm→[5 bools]
    (first_contact_warm_self)."""
    records: list[dict[str, Any]] = []
    for scenario, ws, fc in (("fire_pit", con_ws, con_fc), ("deceptive_fire", dec_ws, dec_fc)):
        for arm in _CP_ARMS:
            for t in range(1, 6):
                records.append(
                    _cp_record(
                        trial_pair_id=t,
                        arm=arm,
                        scenario=scenario,
                        warm_self_frac=ws[arm][t - 1],
                        first_contact=fc[arm][t - 1],
                    )
                )
    return records


def _broadcast(value, arms=_CP_ARMS):
    """Helper: same 5-value list (or bool list) for every arm."""
    return {arm: list(value) for arm in arms}


def _run_cp(analyzer, tmp_path, records, name="cp.jsonl"):
    p = tmp_path / name
    _write_jsonl(p, records)
    return analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=_CP_ARMS,
        scenarios=("fire_pit", "deceptive_fire"),
        strict_schema_version="1.0",
    )


# Warm_self-fraction baselines (5 trials).
_WS_HIGH = [0.70, 0.75, 0.80, 0.85, 0.90]  # follows the fire→warm prior
_WS_LOW = [0.15, 0.20, 0.20, 0.25, 0.20]  # avoids warming
_WS_MID = [0.45, 0.50, 0.50, 0.55, 0.50]  # ablation: reverts PARTWAY toward A (still below)
_FC_ALL_TRUE = [True, True, True, True, True]
_FC_ALL_FALSE = [False, False, False, False, False]


def test_counter_prior_substrate_matters(analyzer, tmp_path):
    """B reduces warm_self specifically on the deceptive hearth (interaction +
    first-contact pass) AND ablations revert → SUBSTRATE_MATTERS."""
    con_ws = _broadcast(_WS_HIGH)  # consistent: everyone warms safely
    dec_ws = {
        "A": _WS_HIGH,
        "B": _WS_LOW,
        "C": _WS_HIGH,
        # Ablations revert PARTWAY toward A (re-approach the hearth) — staying on
        # B's side of A so _compute_secondary registers shrinkage.
        "B-wire-a-off": _WS_MID,
        "B-wire-1-off": _WS_MID,
        "B-nac-bias-off": _WS_MID,
    }
    con_fc = _broadcast(_FC_ALL_TRUE)
    dec_fc = {
        "A": _FC_ALL_TRUE,
        "B": _FC_ALL_FALSE,
        "C": _FC_ALL_TRUE,
        "B-wire-a-off": _FC_ALL_TRUE,
        "B-wire-1-off": _FC_ALL_TRUE,
        "B-nac-bias-off": _FC_ALL_TRUE,
    }
    result = _run_cp(analyzer, tmp_path, _cp_design(con_ws=con_ws, dec_ws=dec_ws, con_fc=con_fc, dec_fc=dec_fc))
    cp = result.counter_prior
    assert cp is not None
    assert cp.interaction_pass is True
    assert cp.interaction < 0  # B reduced warm_self specifically in deceptive
    assert cp.first_contact_pass is True
    assert cp.ablation_hits >= 1
    assert result.overall_label == analyzer.VERDICT_CP_SUBSTRATE_MATTERS
    assert analyzer.EXIT_CODE_FOR_LABEL[result.overall_label] == 0
    assert "Counter-prior interaction" in result.markdown


def test_counter_prior_dominance(analyzer, tmp_path):
    """B keeps warming the deceptive hearth (no specific reduction) → DOMINANCE
    (a strong result: there WAS a gap and the substrate didn't fill it)."""
    con_ws = _broadcast(_WS_HIGH)
    dec_ws = _broadcast(_WS_HIGH)  # B still warms despite carried pain
    con_fc = _broadcast(_FC_ALL_TRUE)
    dec_fc = _broadcast(_FC_ALL_TRUE)
    result = _run_cp(analyzer, tmp_path, _cp_design(con_ws=con_ws, dec_ws=dec_ws, con_fc=con_fc, dec_fc=dec_fc))
    cp = result.counter_prior
    assert cp.interaction_pass is False
    assert cp.avoids_both is False
    assert result.overall_label == analyzer.VERDICT_CP_DOMINANCE
    assert analyzer.EXIT_CODE_FOR_LABEL[result.overall_label] == 0


def test_counter_prior_void_general_caution(analyzer, tmp_path):
    """B reduces warm_self in BOTH worlds (general caution, not specific to the
    hearth) → VOID general caution."""
    con_ws = {arm: (_WS_LOW if arm.startswith("B") else _WS_HIGH) for arm in _CP_ARMS}
    dec_ws = {arm: (_WS_LOW if arm.startswith("B") else _WS_HIGH) for arm in _CP_ARMS}
    con_fc = {arm: (_FC_ALL_FALSE if arm.startswith("B") else _FC_ALL_TRUE) for arm in _CP_ARMS}
    dec_fc = {arm: (_FC_ALL_FALSE if arm.startswith("B") else _FC_ALL_TRUE) for arm in _CP_ARMS}
    result = _run_cp(analyzer, tmp_path, _cp_design(con_ws=con_ws, dec_ws=dec_ws, con_fc=con_fc, dec_fc=dec_fc))
    cp = result.counter_prior
    # Interaction ≈ 0 (both dropped equally), but both dropped ≥1 SD.
    assert cp.interaction_pass is False
    assert cp.avoids_both is True
    assert result.overall_label == analyzer.VERDICT_CP_VOID_GENERAL_CAUTION
    assert analyzer.EXIT_CODE_FOR_LABEL[result.overall_label] == 4


def test_counter_prior_void_not_attributable(analyzer, tmp_path):
    """B avoids the deceptive hearth specifically (interaction + first-contact
    pass) but NO ablation reverts → VOID not-substrate-attributable."""
    con_ws = _broadcast(_WS_HIGH)
    dec_ws = {arm: (_WS_HIGH if arm in ("A", "C") else _WS_LOW) for arm in _CP_ARMS}  # ablations STAY low
    con_fc = _broadcast(_FC_ALL_TRUE)
    dec_fc = {
        "A": _FC_ALL_TRUE,
        "B": _FC_ALL_FALSE,
        "C": _FC_ALL_TRUE,
        "B-wire-a-off": _FC_ALL_FALSE,
        "B-wire-1-off": _FC_ALL_FALSE,
        "B-nac-bias-off": _FC_ALL_FALSE,
    }
    result = _run_cp(analyzer, tmp_path, _cp_design(con_ws=con_ws, dec_ws=dec_ws, con_fc=con_fc, dec_fc=dec_fc))
    cp = result.counter_prior
    assert cp.interaction_pass is True
    assert cp.first_contact_pass is True
    assert cp.ablation_hits == 0
    assert result.overall_label == analyzer.VERDICT_CP_VOID_NOT_ATTRIBUTABLE
    assert analyzer.EXIT_CODE_FOR_LABEL[result.overall_label] == 4


def test_counter_prior_none_for_exp37_data(analyzer, tmp_path):
    """An Exp 37 run (no deceptive_fire scenario) yields counter_prior=None and
    falls through to the legacy per-scenario verdict path."""
    records = _build_full_design(
        a_vals=_arm_a_baseline(),
        b_vals=_arm_b_improved(),
        c_vals=_arm_c_within_a_band(),
    )
    p = tmp_path / "exp37.jsonl"
    _write_jsonl(p, records)
    result = analyzer.run_analysis(
        in_path=p,
        expected_trials=5,
        arms=_CP_ARMS,
        scenarios=("fire_pit", "sharp_rock"),
        strict_schema_version="1.0",
    )
    assert result.counter_prior is None
    assert "Counter-prior interaction" not in result.markdown
    assert not result.overall_label.startswith("COUNTER-PRIOR")


def test_counter_prior_first_contact_extraction(analyzer):
    """_extract_bool_proportion: proportion of True over non-None; all-None → None."""
    recs = [{"f": True}, {"f": False}, {"f": True}, {"f": None}]
    assert analyzer._extract_bool_proportion(recs, "f") == pytest.approx(2 / 3)
    assert analyzer._extract_bool_proportion([{"f": None}, {"f": None}], "f") is None
    assert analyzer._extract_bool_proportion([], "f") is None


def test_counter_prior_interaction_direction_sign(analyzer, tmp_path):
    """The interaction is signed: only a NEGATIVE interaction (B reduces
    warm_self in deceptive relative to consistent) passes. A positive
    interaction (B warms MORE in deceptive) must NOT pass."""
    con_ws = _broadcast(_WS_HIGH)
    # B warms MORE in deceptive than A (wrong direction): interaction positive.
    dec_ws = {arm: ([0.95, 0.96, 0.97, 0.98, 0.99] if arm == "B" else _WS_HIGH) for arm in _CP_ARMS}
    con_fc = _broadcast(_FC_ALL_TRUE)
    dec_fc = _broadcast(_FC_ALL_TRUE)
    result = _run_cp(analyzer, tmp_path, _cp_design(con_ws=con_ws, dec_ws=dec_ws, con_fc=con_fc, dec_fc=dec_fc))
    cp = result.counter_prior
    assert cp.interaction is not None and cp.interaction > 0
    assert cp.interaction_pass is False
