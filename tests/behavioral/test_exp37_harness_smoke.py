"""Exp 37 harness smoke test — mock-LLM end-to-end.

Verifies the apparatus works end-to-end on a synthetic backend WITHOUT
spawning ``maxim --sim`` subprocesses or burning LLM cost. The harness
itself drives real subprocesses for graduation runs; the mock mode here
exercises every code path the harness uses to bookkeep paired trials.

Key invariants pinned:

  1. Trial-pair-id is shared across arms within a trial (so the analyzer
     can compute per-pair deltas).
  2. JSONL records carry the locked schema fields.
  3. Per-trial sandbox isolation — no two arms share a data_home.
  4. Cost-cap aborts cleanly with non-zero exit code.
  5. Failure-class detection routes through the per-scenario rules
     (entity-name tools + body-level pick_up + entity arg).
  6. Turn-binning on say/respond boundaries (the operationalization of
     the pre-reg's "per-turn" language).
  7. Arm-set selection works (`--arms A,B` runs only those arms).
  8. Resumed runs identify their prior_session_id in the JSONL record.

The smoke test is fast (<1s) and CI-safe — no LLM, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HARNESS_PATH = _REPO_ROOT / "scripts" / "benchmark_cross_session.py"


def _load_harness():
    """Load the harness module by path (it lives in scripts/, not src/)."""
    spec = importlib.util.spec_from_file_location("exp37_harness", _HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp37_harness"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path / "work"


@pytest.fixture
def out_path(tmp_path: Path) -> Path:
    return tmp_path / "out.jsonl"


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ─── 1. End-to-end smoke: all arms, both scenarios, 2 trials ─────────────


def test_full_run_emits_expected_record_count(harness, workdir, out_path):
    """6 arms × 2 scenarios × 2 trials = 24 records."""
    summary = harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=harness.ARMS,
        scenarios=harness.SCENARIOS,
        trials=2,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=6,
        seed_base=42,
        mock=True,
    )
    records = _read_records(out_path)
    expected = 2 * 2 * len(harness.ARMS)
    assert len(records) == expected, f"expected {expected} records, got {len(records)}"
    assert summary["records_written"] == expected


# ─── 2. Trial-pair-id alignment across arms ──────────────────────────────


def test_trial_pair_id_aligns_across_arms(harness, workdir, out_path):
    harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=harness.ARMS,
        scenarios=harness.SCENARIOS,
        trials=3,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=6,
        seed_base=42,
        mock=True,
    )
    records = _read_records(out_path)
    pair_ids = sorted({r["trial_pair_id"] for r in records})
    assert pair_ids == [1, 2, 3]
    # Each trial pair must include all arms for both scenarios.
    for pid in pair_ids:
        for scenario in harness.SCENARIOS:
            arms_seen = {r["arm"] for r in records if r["trial_pair_id"] == pid and r["scenario"] == scenario}
            assert arms_seen == set(harness.ARMS), (
                f"trial {pid} scenario {scenario}: missing arms {set(harness.ARMS) - arms_seen}"
            )


# ─── 3. Schema completeness — every locked field present ─────────────────

_REQUIRED_FIELDS = (
    "schema_version",
    "experiment",
    "harness_version",
    "trial_pair_id",
    "arm",
    "scenario",
    "session_id",
    "prior_session_id",
    "data_home",
    "seed",
    "model",
    "version_info",
    "turns",
    "finish_reason",
    "duration_s",
    "cost_usd",
    "total_input_tokens",
    "total_output_tokens",
    "tool_usage",
    "tool_class_diversity",
    "aut_memories_formed",
    "aut_causal_links",
    "wall_clock_iso",
    "primary_metric_repeat_failure_action_rate",
    "per_action_failure_rate",
    "failure_class_action_count",
    "failure_class_actions_per_turn",
    "turn_count_binned",
    "affordance_preference_safe_count",
    "affordance_preference_failed_count",
    "affordance_preference_safe_fraction",
    "time_to_safe_steady_state_turns",
)


def test_record_schema_complete(harness, workdir, out_path):
    harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=("A", "B", "C"),
        scenarios=("fire_pit",),
        trials=1,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    records = _read_records(out_path)
    assert len(records) == 3
    for rec in records:
        missing = [f for f in _REQUIRED_FIELDS if f not in rec]
        assert not missing, f"arm={rec.get('arm')}: missing fields {missing}"
        assert rec["schema_version"] == harness.SCHEMA_VERSION
        assert rec["experiment"] == harness.EXPERIMENT_ID


# ─── 4. Sandbox isolation — no two arms share a data_home ────────────────


def test_sandbox_isolation_per_arm(harness, workdir, out_path):
    harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=harness.ARMS,
        scenarios=harness.SCENARIOS,
        trials=2,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    records = _read_records(out_path)
    homes = [r["data_home"] for r in records]
    assert len(homes) == len(set(homes)), (
        f"data_home collision detected: {len(homes)} records but only {len(set(homes))} unique homes"
    )


# ─── 5. Cost cap aborts ──────────────────────────────────────────────────


def test_cost_cap_per_record_safety_net_fires_during_first_pair(harness, workdir, out_path):
    """Per-record safety net catches cap breach when the FIRST pair (no
    history yet to project from) blows through the cap. Some records exist
    before the abort fires — partial-pair data is the price of catching
    runaway costs in flight when projection has no signal."""
    with pytest.raises(harness.CostCapExceeded, match=r"exceeds cap"):
        harness.run_benchmark(
            out_path=out_path,
            workdir=workdir,
            arms=harness.ARMS,
            scenarios=harness.SCENARIOS,
            trials=2,
            model="claude-sonnet",
            cost_cap=0.05,
            max_turns=4,
            seed_base=42,
            mock=True,
        )
    # First pair starts (projection seed=0 passes), per-record safety net
    # fires mid-pair when cumulative ($0.02 + $0.02 + ...) > $0.05.
    records = _read_records(out_path)
    assert 1 <= len(records) <= 6  # partial first-pair, never a full pair


def test_cost_cap_projection_aborts_cleanly_at_pair_boundary(harness, workdir, out_path):
    """Once a pair's worth of data has been observed, the projection fires
    BETWEEN pairs — the next pair never starts and the JSONL stays clean."""
    # Each mock arm reports $0.02. Pair 1 fire_pit ~ $0.14, pair 1 sharp_rock
    # ~ $0.12 (shared Arm C prior). After ~$0.26 cumulative, the start-of-
    # trial-2-fire_pit projection (observed_max=$0.02 × 6 + $0.02 = $0.14)
    # pushes total to $0.40 > $0.30 cap, so trial 2 fire_pit is rejected.
    with pytest.raises(harness.CostCapExceeded, match=r"Aborting cleanly between trial pairs"):
        harness.run_benchmark(
            out_path=out_path,
            workdir=workdir,
            arms=harness.ARMS,
            scenarios=harness.SCENARIOS,
            trials=2,
            model="claude-sonnet",
            cost_cap=0.30,
            max_turns=4,
            seed_base=42,
            mock=True,
        )
    records = _read_records(out_path)
    trial_ids = {r["trial_pair_id"] for r in records}
    assert trial_ids == {1}, f"Expected only trial 1 complete; got trials {trial_ids}"


def test_cost_cap_exit_code(harness, workdir, out_path):
    """CLI main() returns 3 on cost-cap abort."""
    argv = [
        "--out",
        str(out_path),
        "--workdir",
        str(workdir),
        "--trials",
        "2",
        "--cost-cap",
        "0.05",
        "--sim-max-turns",
        "4",
        "--mock-llm",
    ]
    rc = harness.main(argv)
    assert rc == 3


# ─── 6. Failure-class detection routes through the per-scenario rules ────


def test_failure_class_detection_fire_pit(harness):
    """Direct entity affordance + body pick_up with entity arg both fire."""
    actions = [
        {"tool": "fire_pit_touch", "params": {}},
        {"tool": "respond", "params": {"message": "ouch"}},
        {"tool": "infant_humanoid_pick_up", "params": {"object": "fire_pit"}},
        {"tool": "respond", "params": {"message": "double ouch"}},
        {"tool": "fire_pit_observe", "params": {}},
        {"tool": "respond", "params": {"message": "safer"}},
    ]
    m = harness.compute_metrics(actions, "fire_pit")
    assert m["failure_class_action_count"] == 2
    assert m["failure_class_actions_per_turn"] == [1, 1, 0]
    assert m["affordance_preference_safe_count"] == 1
    assert m["affordance_preference_failed_count"] == 2
    # 1 / 3 = ~0.333 — repeat failure on 2 of 3 turns.
    assert m["primary_metric_repeat_failure_action_rate"] == pytest.approx(2 / 3)


def test_failure_class_detection_sharp_rock(harness):
    """sharp_rock_touch and infant_humanoid_pick_up(sharp_rock) fire; examine is safe."""
    actions = [
        {"tool": "sharp_rock_touch", "params": {}},
        {"tool": "respond", "params": {}},
        {"tool": "infant_humanoid_pick_up", "params": {"object": "blanket"}},  # NOT failure
        {"tool": "sharp_rock_examine", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "sharp_rock")
    assert m["failure_class_action_count"] == 1
    assert m["affordance_preference_safe_count"] == 1
    assert m["affordance_preference_failed_count"] == 1


def test_per_action_rate_matches_count(harness):
    actions = [
        {"tool": "fire_pit_touch", "params": {}},
        {"tool": "fire_pit_observe", "params": {}},
        {"tool": "respond", "params": {}},
        {"tool": "fire_pit_touch", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "fire_pit")
    assert m["failure_class_action_count"] == 2
    # 2 failures / 5 actions = 0.4
    assert m["per_action_failure_rate"] == pytest.approx(0.4)


# ─── 7. Turn-binning on say/respond ──────────────────────────────────────


def test_turn_binning_splits_on_respond(harness):
    """Each respond/say closes a turn; failures within a turn deduplicate."""
    actions = [
        {"tool": "fire_pit_touch", "params": {}},
        {"tool": "fire_pit_touch", "params": {}},  # second hit in same turn — still 1
        {"tool": "respond", "params": {}},
        {"tool": "fire_pit_observe", "params": {}},
        {"tool": "say", "params": {}},
        {"tool": "fire_pit_touch", "params": {}},
        # No final respond — tail bucket flushed.
    ]
    m = harness.compute_metrics(actions, "fire_pit")
    assert m["failure_class_actions_per_turn"] == [1, 0, 1]
    assert m["turn_count_binned"] == 3


def test_time_to_safe_steady_state(harness):
    """First turn at which 3 consecutive zero-failure turns begin."""
    actions = []
    for _ in range(2):
        actions += [
            {"tool": "fire_pit_touch", "params": {}},
            {"tool": "respond", "params": {}},
        ]
    for _ in range(3):
        actions += [
            {"tool": "fire_pit_observe", "params": {}},
            {"tool": "respond", "params": {}},
        ]
    m = harness.compute_metrics(actions, "fire_pit")
    # per_turn_failure = [1, 1, 0, 0, 0] — steady state begins at index 2.
    assert m["time_to_safe_steady_state_turns"] == 2


def test_no_steady_state_when_failures_persist(harness):
    actions = []
    for _ in range(5):
        actions += [
            {"tool": "fire_pit_touch", "params": {}},
            {"tool": "respond", "params": {}},
        ]
    m = harness.compute_metrics(actions, "fire_pit")
    assert m["time_to_safe_steady_state_turns"] is None


# ─── 8. Arm subset selection ─────────────────────────────────────────────


def test_arm_subset_a_only(harness, workdir, out_path):
    harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=("A",),
        scenarios=("fire_pit",),
        trials=2,
        model="claude-sonnet",
        cost_cap=10.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    records = _read_records(out_path)
    assert len(records) == 2
    assert all(r["arm"] == "A" for r in records)
    assert all(r["prior_session_id"] is None for r in records)


def test_arm_subset_a_and_b_only(harness, workdir, out_path):
    """B requires A to be enabled — the harness shares Arm A as the prior."""
    harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=("A", "B"),
        scenarios=("fire_pit",),
        trials=2,
        model="claude-sonnet",
        cost_cap=10.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    records = _read_records(out_path)
    arms = sorted(r["arm"] for r in records)
    assert arms == ["A", "A", "B", "B"]
    b_records = [r for r in records if r["arm"] == "B"]
    a_records = [r for r in records if r["arm"] == "A"]
    # B's prior_session_id matches an A session id from the same trial pair.
    for b in b_records:
        a = next(r for r in a_records if r["trial_pair_id"] == b["trial_pair_id"])
        assert b["prior_session_id"] == a["session_id"]


def test_unknown_arm_raises(harness, workdir, out_path):
    with pytest.raises(ValueError, match="Unknown arm"):
        harness.run_benchmark(
            out_path=out_path,
            workdir=workdir,
            arms=("X",),
            scenarios=("fire_pit",),
            trials=1,
            model="claude-sonnet",
            cost_cap=10.0,
            max_turns=4,
            seed_base=42,
            mock=True,
        )


# ─── 9. Arm C prior is shared across scenarios within a trial ────────────


def test_arm_c_prior_shared_across_scenarios(harness, workdir, out_path):
    harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=("C",),
        scenarios=harness.SCENARIOS,
        trials=2,
        model="claude-sonnet",
        cost_cap=10.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    records = _read_records(out_path)
    by_trial: dict[int, set[str]] = {}
    for r in records:
        by_trial.setdefault(r["trial_pair_id"], set()).add(r["prior_session_id"])
    # Within a trial, both scenarios' Arm C records share the SAME prior.
    for trial_id, priors in by_trial.items():
        assert len(priors) == 1, f"trial {trial_id}: expected 1 shared C-prior, got {priors}"
    # Across trials, priors differ.
    all_priors = {r["prior_session_id"] for r in records}
    assert len(all_priors) == 2


# ─── 10. C1 fold: header skip + B2 fold: mock/real schema tripwire ─────


def test_load_latest_session_skips_stage_0b_header(harness, tmp_path):
    """The real simulation/report.py::save_action_log writes a header line
    as the FIRST entry of actions.jsonl. Without the skip, primary metric
    is off by one and the tail bucket flushes extra zero turns."""
    session_dir = tmp_path / "sim_reports" / "20260530_120000"
    session_dir.mkdir(parents=True)
    (session_dir / "report.json").write_text(
        '{"session_id": "20260530_120000", "turns": 1, "tool_usage": {}, "cost_usd": 0.0, '
        '"total_input_tokens": 0, "total_output_tokens": 0, "finish_reason": "max_turns", '
        '"duration_s": 1.0, "aut_memories_formed": 0, "aut_causal_links": 0}'
    )
    actions_jsonl = (
        '{"_format_version": "1.1", "_record_kind": "header", "session_id": "20260530_120000"}\n'
        '{"timestamp": 1.0, "tool": "fire_pit_touch", "params": {}, "success": true}\n'
        '{"timestamp": 2.0, "tool": "respond", "params": {}, "success": true}\n'
    )
    (session_dir / "actions.jsonl").write_text(actions_jsonl)
    result = harness._load_latest_session(tmp_path, exclude=None)
    # Header is skipped — only 2 actual actions visible to compute_metrics.
    assert len(result.actions) == 2
    assert result.actions[0]["tool"] == "fire_pit_touch"
    # Primary metric computes correctly: 1 failure-class turn out of 1 turn.
    m = harness.compute_metrics(result.actions, "fire_pit")
    assert m["failure_class_action_count"] == 1
    assert m["primary_metric_repeat_failure_action_rate"] == 1.0


def test_load_latest_session_uses_exact_session_id_match(harness, tmp_path):
    """Substring match would fail if one session_id contains another."""
    for sid in ("20260530_120000", "20260530_120000_continued"):
        sd = tmp_path / "sim_reports" / sid
        sd.mkdir(parents=True)
        (sd / "report.json").write_text('{"session_id": "' + sid + '", "turns": 1}')
    # Exclude the SHORTER id — substring match would mistakenly skip the longer.
    result = harness._load_latest_session(tmp_path, exclude="20260530_120000")
    assert result.session_id == "20260530_120000_continued"


def test_mock_writes_header_matching_real_format(harness, workdir):
    """B2 fold: mock's actions.jsonl MUST carry the Stage-0b header so the
    skip path is exercised on the smoke-test side too."""
    home = workdir / "test_mock_format"
    home.mkdir(parents=True)
    harness.run_one_sim(
        data_home=home,
        goal=harness.SCENARIO_GOAL["fire_pit"],
        model="x",
        max_turns=3,
        resume_session=None,
        mock=True,
        mock_failure_count=2,
    )
    actions_path = next((home / "sim_reports").glob("*/actions.jsonl"))
    first_line = actions_path.read_text().splitlines()[0]
    parsed = json.loads(first_line)
    assert parsed.get("_record_kind") == "header"
    assert "_format_version" in parsed


def test_record_schema_includes_format_version(harness, workdir, out_path):
    """B1 fold: _format_version is the project-wide persistence envelope."""
    harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=("A",),
        scenarios=("fire_pit",),
        trials=1,
        model="x",
        cost_cap=10.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    records = _read_records(out_path)
    assert records[0]["_format_version"] == harness.SCHEMA_VERSION


def test_mock_report_carries_fields_harness_reads(harness, workdir):
    """B2 tripwire: if a future src/maxim/simulation/report.py change drops
    or renames a field the harness reads, this test should catch the drift
    BEFORE a $14 real-LLM run discovers it the hard way."""
    home = workdir / "test_mock_fields"
    home.mkdir(parents=True)
    sim = harness.run_one_sim(
        data_home=home,
        goal=harness.SCENARIO_GOAL["fire_pit"],
        model="x",
        max_turns=3,
        resume_session=None,
        mock=True,
        mock_failure_count=2,
    )
    fields_read_by_harness = (
        "turns",
        "finish_reason",
        "duration_s",
        "cost_usd",
        "total_input_tokens",
        "total_output_tokens",
        "tool_usage",
        "aut_memories_formed",
        "aut_causal_links",
    )
    for f in fields_read_by_harness:
        assert f in sim.report, (
            f"mock report missing {f!r} that build_record expects. "
            "If you removed it from _mock_sim, also remove the reader; "
            "if you removed it from real simulation/report.py, update both."
        )


# ─── 10b. FAILURE_CLASS YAML cross-check (Architecture I2 fold) ──────────


def test_failure_class_yaml_cross_check_passes_on_current_yamls(harness):
    """Live YAMLs must declare every affordance the harness rules reference.
    If pyyaml or ComponentRegistry isn't importable in this env the check
    is a silent no-op (acceptable; the live-run pre-flight still asserts)."""
    # No raise = pass; if either YAML missing rules, this raises.
    harness._assert_failure_class_matches_yaml("fire_pit")
    harness._assert_failure_class_matches_yaml("sharp_rock")


def test_failure_class_yaml_cross_check_rejects_stale_rule(harness, monkeypatch):
    """Mutating FAILURE_CLASS to reference an affordance not in the YAML
    must produce a loud RuntimeError, not a silent zero-signal experiment."""
    bad_rules = dict(harness.FAILURE_CLASS["fire_pit"])
    bad_rules["direct_failure_tools"] = frozenset({"fire_pit_doesnotexist"})
    monkeypatch.setitem(harness.FAILURE_CLASS, "fire_pit", bad_rules)
    with pytest.raises(RuntimeError, match=r"FAILURE_CLASS"):
        harness._assert_failure_class_matches_yaml("fire_pit")


# ─── 11. I1 fold: append idempotency ─────────────────────────────────────


def test_append_refuses_existing_record_keys(harness, workdir, out_path):
    """Re-running --trials 5 twice into the same --out must not silently
    duplicate trial_pair_id 1..5. The harness refuses with a clear error."""
    harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=("A",),
        scenarios=("fire_pit",),
        trials=1,
        model="x",
        cost_cap=10.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    workdir2 = workdir.parent / "work2"
    workdir2.mkdir()
    with pytest.raises(RuntimeError, match="Refusing to re-emit existing record key"):
        harness.run_benchmark(
            out_path=out_path,
            workdir=workdir2,
            arms=("A",),
            scenarios=("fire_pit",),
            trials=1,
            model="x",
            cost_cap=10.0,
            max_turns=4,
            seed_base=42,
            mock=True,
        )


# ─── 12. Version info captured per record ────────────────────────────────


def test_version_info_present(harness, workdir, out_path):
    harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=("A",),
        scenarios=("fire_pit",),
        trials=1,
        model="claude-sonnet",
        cost_cap=10.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    records = _read_records(out_path)
    vi = records[0]["version_info"]
    assert "harness_version" in vi
    assert vi["harness_version"] == harness.HARNESS_VERSION
    # Either the maxim version came through or an error was captured —
    # either way, the field is present.
    assert "version" in vi or "version_info_error" in vi
