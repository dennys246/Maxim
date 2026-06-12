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


# ─── 0. --subsim-timeout-s plumbing (reasoning-model support) ────────────


def test_subsim_timeout_s_plumbs_through_to_run_one_sim(harness, workdir, out_path, monkeypatch):
    """``run_benchmark(subsim_timeout_s=N)`` must reach every ``run_one_sim``
    call's ``timeout_s``. Reasoning-model fires (DeepSeek-R1) need a longer
    per-sub-sim subprocess ceiling than the 1800s default; this pins the
    plumbing so the flag isn't silently dropped. Regression for the
    2026-06-11 R1 fire where 12-turn sub-sims ran 60-90 min and the 30-min
    default would have killed them."""
    captured: list[int] = []
    real_run_one_sim = harness.run_one_sim

    def spy(*args, **kwargs):
        captured.append(kwargs.get("timeout_s"))
        return real_run_one_sim(*args, **kwargs)

    monkeypatch.setattr(harness, "run_one_sim", spy)
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
        subsim_timeout_s=9999,
    )
    assert captured, "run_one_sim was never called"
    assert all(t == 9999 for t in captured), f"subsim_timeout_s did not propagate to every run_one_sim: {captured}"


def test_subsim_timeout_s_defaults_to_1800(harness, workdir, out_path, monkeypatch):
    """Without --subsim-timeout-s, the default 1800s is used (fast-model
    behavior unchanged)."""
    captured: list[int] = []
    real_run_one_sim = harness.run_one_sim

    def spy(*args, **kwargs):
        captured.append(kwargs.get("timeout_s"))
        return real_run_one_sim(*args, **kwargs)

    monkeypatch.setattr(harness, "run_one_sim", spy)
    harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=("A",),
        scenarios=("fire_pit",),
        trials=1,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    assert captured and all(t == 1800 for t in captured), captured


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
    # cradle_activation_fixes.md P2: positive-approach descriptive metric.
    "fire_approach_action_count",
    # exp37_metric_pivot.md (Path 2): NEW primary + denominator + new
    # corroborating.
    "positive_approach_engagement_fraction",
    "fire_pit_engagement_count",
    "time_to_first_warm_self_action",
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


# ─── 5b. Systemic-failure early-abort guard (credit/auth outage) ─────────


def test_systemic_failure_aborts_after_consecutive_zero_token(harness, workdir, out_path, monkeypatch):
    """N consecutive zero-input-token sub-sims (the credit-exhaustion / auth
    signature) abort the fire fast instead of grinding out garbage records."""
    monkeypatch.setenv("MAXIM_BENCH_MOCK_ZERO_TOKENS", "1")
    with pytest.raises(harness.SystemicLLMFailure):
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
            max_consecutive_failures=3,
        )
    # Aborted at the threshold, not after all 24 records.
    records = _read_records(out_path)
    assert len(records) == 3, f"expected abort at 3 records, got {len(records)}"


def test_systemic_failure_guard_disabled_with_zero_threshold(harness, workdir, out_path, monkeypatch):
    """--max-consecutive-failures 0 disables the guard — zero-token runs still
    complete (operator override)."""
    monkeypatch.setenv("MAXIM_BENCH_MOCK_ZERO_TOKENS", "1")
    summary = harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=harness.ARMS,
        scenarios=harness.SCENARIOS,
        trials=1,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=4,
        seed_base=42,
        mock=True,
        max_consecutive_failures=0,
    )
    expected = 1 * 2 * len(harness.ARMS)
    assert summary["records_written"] == expected


def test_systemic_failure_guard_skipped_for_local_models(harness, workdir, out_path, monkeypatch):
    """The total_input_tokens==0 signal is CLOUD-SPECIFIC — local llama-cpp
    runs do not populate the field even on successful inference. The guard
    must be gated on cloud-model-only or it fires as a false positive on
    every local-model fire.

    Regression for the 2026-06-10 Mistral24B fire crash: the fire produced 3
    records with in_tok=0 (normal for local), the guard tripped, the harness
    aborted as "systemic provider failure" when the LLM was actually working
    fine. Qwen14B and Qwen32B fires would also break on re-fire under the
    pre-fix code path.
    """
    monkeypatch.setenv("MAXIM_BENCH_MOCK_ZERO_TOKENS", "1")
    summary = harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=harness.ARMS,
        scenarios=harness.SCENARIOS,
        trials=1,
        # Local model name — gating on _is_cloud_model(model) must skip the
        # check, NOT abort after 3 records.
        model="mistral-small-24b",
        cost_cap=100.0,
        max_turns=4,
        seed_base=42,
        mock=True,
        max_consecutive_failures=3,
    )
    # Without the gate this would have raised SystemicLLMFailure after 3
    # records. With the gate, the local fire completes all 12.
    expected = 1 * 2 * len(harness.ARMS)
    assert summary["records_written"] == expected, (
        f"local-model fire should complete all {expected} records, got "
        f"{summary['records_written']} (regression: guard fired on local)"
    )


def test_systemic_failure_guard_still_fires_for_cloud(harness, workdir, out_path, monkeypatch):
    """Cloud-model fires (claude-*, gpt-*, etc.) MUST still trip the guard
    when token counts are zero — that's the real credit-exhaustion / auth
    cascade signature the guard was designed for. The local-model gate must
    not weaken cloud safety."""
    monkeypatch.setenv("MAXIM_BENCH_MOCK_ZERO_TOKENS", "1")
    for cloud_model in ("claude-sonnet", "gpt-4o", "deepseek-chat"):
        out = workdir.parent / f"out_{cloud_model.replace('-', '_')}.jsonl"
        with pytest.raises(harness.SystemicLLMFailure):
            harness.run_benchmark(
                out_path=out,
                workdir=workdir.parent / f"work_{cloud_model.replace('-', '_')}",
                arms=harness.ARMS,
                scenarios=harness.SCENARIOS,
                trials=2,
                model=cloud_model,
                cost_cap=100.0,
                max_turns=4,
                seed_base=42,
                mock=True,
                max_consecutive_failures=3,
            )


def test_systemic_failure_does_not_fire_on_healthy_run(harness, workdir, out_path):
    """Healthy (non-zero-token) runs never trip the guard."""
    summary = harness.run_benchmark(
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
        max_consecutive_failures=3,
    )
    assert summary["records_written"] == 2 * 2 * len(harness.ARMS)


def test_systemic_failure_exit_code(harness, workdir, out_path, monkeypatch):
    """CLI main() returns 5 on systemic-failure abort."""
    monkeypatch.setenv("MAXIM_BENCH_MOCK_ZERO_TOKENS", "1")
    argv = [
        "--out",
        str(out_path),
        "--workdir",
        str(workdir),
        "--trials",
        "2",
        "--cost-cap",
        "100.0",
        "--sim-max-turns",
        "4",
        "--max-consecutive-failures",
        "3",
        "--mock-llm",
    ]
    rc = harness.main(argv)
    assert rc == 5


# ─── 5c. Resume: skip complete groups, prune + rerun failed/missing ──────


def test_resume_skips_complete_groups(harness, workdir, out_path):
    """Resuming a fully-complete run adds nothing and preserves all records."""
    kw = dict(
        out_path=out_path,
        workdir=workdir,
        arms=harness.ARMS,
        scenarios=harness.SCENARIOS,
        trials=1,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    s1 = harness.run_benchmark(**kw)
    n1 = s1["records_written"]
    assert n1 == 1 * 2 * len(harness.ARMS)
    s2 = harness.run_benchmark(**kw, resume=True)
    assert s2["records_written"] == 0, "resume re-ran already-complete groups"
    assert len(_read_records(out_path)) == n1  # nothing duplicated


def test_resume_runs_missing_groups(harness, workdir, out_path):
    """Resume with a larger trial count keeps pair 1 and runs pair 2."""
    base = dict(
        out_path=out_path,
        workdir=workdir,
        arms=harness.ARMS,
        scenarios=harness.SCENARIOS,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    harness.run_benchmark(**base, trials=1)
    harness.run_benchmark(**base, trials=2, resume=True)
    records = _read_records(out_path)
    pairs = sorted({r["trial_pair_id"] for r in records})
    assert pairs == [1, 2]
    assert len(records) == 2 * 2 * len(harness.ARMS)  # no dupes, both pairs present


def test_resume_prunes_and_reruns_failed_groups(harness, workdir, out_path, monkeypatch):
    """A run that completed with zero-token (credit-fail) records is pruned and
    re-run cleanly on resume."""
    base = dict(
        out_path=out_path,
        workdir=workdir,
        arms=harness.ARMS,
        scenarios=harness.SCENARIOS,
        trials=1,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=4,
        seed_base=42,
        mock=True,
    )
    # First pass: simulate systemic failure (all zero-token), guard disabled so
    # the contaminated file is fully written.
    monkeypatch.setenv("MAXIM_BENCH_MOCK_ZERO_TOKENS", "1")
    harness.run_benchmark(**base, max_consecutive_failures=0)
    contaminated = _read_records(out_path)
    assert all(r["total_input_tokens"] == 0 for r in contaminated)
    # Resume with the provider healthy: prune the failed groups + re-run clean.
    monkeypatch.delenv("MAXIM_BENCH_MOCK_ZERO_TOKENS", raising=False)
    harness.run_benchmark(**base, resume=True)
    healed = _read_records(out_path)
    assert len(healed) == 1 * 2 * len(harness.ARMS)
    assert all(r["total_input_tokens"] > 0 for r in healed), "resume left failed records"
    assert (out_path.with_suffix(out_path.suffix + ".resume-bak")).exists()


def test_resume_no_existing_file_is_fresh(harness, workdir, out_path):
    """--resume with no prior file behaves like a fresh run."""
    summary = harness.run_benchmark(
        out_path=out_path,
        workdir=workdir,
        arms=harness.ARMS,
        scenarios=harness.SCENARIOS,
        trials=1,
        model="claude-sonnet",
        cost_cap=100.0,
        max_turns=4,
        seed_base=42,
        mock=True,
        resume=True,
    )
    assert summary["records_written"] == 1 * 2 * len(harness.ARMS)


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


def test_fire_pit_warm_self_counts_as_safe_and_approach(harness):
    """cradle_activation_fixes.md P2: ``fire_pit_warm_self`` is a positive
    proximity affordance. It counts toward the existing
    ``affordance_preference_safe_count`` AND drives the new descriptive
    ``fire_approach_action_count`` corroborating metric.
    """
    actions = [
        {"tool": "fire_pit_warm_self", "params": {}},
        {"tool": "respond", "params": {}},
        {"tool": "fire_pit_warm_self", "params": {}},
        {"tool": "fire_pit_observe", "params": {}},
        {"tool": "respond", "params": {}},
        {"tool": "fire_pit_touch", "params": {}},  # Aversive — counts toward failure_class
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "fire_pit")
    # Approach-specific count.
    assert m["fire_approach_action_count"] == 2
    # warm_self ALSO counts as a broad safe choice alongside observe.
    assert m["affordance_preference_safe_count"] == 3
    assert m["affordance_preference_failed_count"] == 1
    # The aversive edge still fires.
    assert m["failure_class_action_count"] == 1


def test_sharp_rock_fire_approach_count_is_zero(harness):
    """sharp_rock has empty ``direct_approach_tools`` by design — the
    metric is structurally 0 regardless of what tools the agent calls.
    cradle_activation_fixes.md P2: asymmetric metric is intentional, NOT
    a bug.
    """
    actions = [
        # Even a fire_pit_warm_self call doesn't count for the SHARP_ROCK
        # scenario because the metric is scoped to scenario-specific
        # approach tools.
        {"tool": "fire_pit_warm_self", "params": {}},
        {"tool": "sharp_rock_examine", "params": {}},
        {"tool": "sharp_rock_touch", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "sharp_rock")
    assert m["fire_approach_action_count"] == 0


# ─── exp37_metric_pivot.md (Path 2) — positive_approach_engagement_fraction ──


def test_positive_approach_engagement_fraction_pure_warm_self(harness):
    """Pure warm_self engagement → fraction = 1.0 (ceiling)."""
    actions = [
        {"tool": "fire_pit_warm_self", "params": {}},
        {"tool": "respond", "params": {}},
        {"tool": "fire_pit_warm_self", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "fire_pit")
    assert m["fire_pit_engagement_count"] == 2
    assert m["positive_approach_engagement_fraction"] == 1.0
    assert m["time_to_first_warm_self_action"] == 0


def test_positive_approach_engagement_fraction_mixed_engagement(harness):
    """Realistic mixed: 2 warm_self + 1 observe + 1 touch + 1
    pick_up(fire_pit) → fraction = 2 / 5 = 0.4.
    """
    actions = [
        {"tool": "fire_pit_observe", "params": {}},
        {"tool": "fire_pit_warm_self", "params": {}},
        {"tool": "infant_humanoid_pick_up", "params": {"object": "fire_pit"}},
        {"tool": "fire_pit_warm_self", "params": {}},
        {"tool": "fire_pit_touch", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "fire_pit")
    assert m["fire_pit_engagement_count"] == 5
    assert m["positive_approach_engagement_fraction"] == pytest.approx(2 / 5)
    # First warm_self is action index 1 (0-based).
    assert m["time_to_first_warm_self_action"] == 1


def test_positive_approach_engagement_fraction_zero_engagement(harness):
    """Sessions where the agent never engages with fire_pit at all →
    fraction = 0.0 (denominator clamped) and time_to_first_warm_self
    is None.
    """
    actions = [
        {"tool": "sense_cool_air", "params": {}},
        {"tool": "respond", "params": {}},
        {"tool": "cool_air_feel", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "fire_pit")
    assert m["fire_pit_engagement_count"] == 0
    assert m["positive_approach_engagement_fraction"] == 0.0
    assert m["time_to_first_warm_self_action"] is None


def test_positive_approach_engagement_fraction_only_aversive(harness):
    """Agent that ONLY touches the fire (no warm_self, no observe) →
    fraction = 0 / 1 = 0.0. The substrate-transferred preference signal
    indicates aversion, not engagement.
    """
    actions = [
        {"tool": "fire_pit_touch", "params": {}},
        {"tool": "respond", "params": {}},
        {"tool": "fire_pit_touch", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "fire_pit")
    assert m["fire_pit_engagement_count"] == 2
    assert m["positive_approach_engagement_fraction"] == 0.0
    assert m["time_to_first_warm_self_action"] is None


def test_sharp_rock_positive_approach_engagement_structurally_zero(harness):
    """sharp_rock has empty ``direct_approach_tools`` → numerator is
    structurally 0 → fraction is structurally 0 regardless of
    engagement count. Validates the scenario-asymmetric design.
    """
    actions = [
        {"tool": "sharp_rock_examine", "params": {}},
        {"tool": "sharp_rock_touch", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "sharp_rock")
    assert m["fire_pit_engagement_count"] == 2  # examine + touch on sharp_rock
    assert m["positive_approach_engagement_fraction"] == 0.0
    assert m["time_to_first_warm_self_action"] is None


def test_body_pick_up_counts_as_engagement(harness):
    """Body-level pick_up(fire_pit) counts toward the engagement
    denominator via ``body_engagement_rules`` (parallels
    body_failure_rules). Confirms the body channel is wired.
    """
    actions = [
        {"tool": "fire_pit_warm_self", "params": {}},
        # pick_up(blanket) — NOT engagement (object != fire_pit).
        {"tool": "infant_humanoid_pick_up", "params": {"object": "blanket"}},
        # pick_up(fire_pit) — engagement (matches body_engagement_rules).
        {"tool": "infant_humanoid_pick_up", "params": {"object": "fire_pit"}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "fire_pit")
    # 1 warm_self + 1 pick_up(fire_pit) = 2 engagement actions.
    # pick_up(blanket) excluded.
    assert m["fire_pit_engagement_count"] == 2
    assert m["positive_approach_engagement_fraction"] == 0.5


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


# ─── 10c. Exp 38 counter-prior — deceptive_fire / cradle_false_hearth ────


def test_failure_class_yaml_cross_check_passes_deceptive_fire(harness):
    """The deceptive_fire FAILURE_CLASS rules must validate against the live
    cradle_false_hearth.yaml — the entity name (``hearth``) ≠ scenario name, so
    this also exercises the SCENARIO_ENTITY generalization of the validator."""
    harness._assert_failure_class_matches_yaml("deceptive_fire")


def test_scenario_entity_maps_deceptive_to_hearth(harness):
    """deceptive_fire surfaces the ``hearth`` entity from cradle_false_hearth,
    NOT a ``deceptive_fire`` component (the legacy scenario==entity assumption)."""
    component_ref, entity_name = harness.SCENARIO_ENTITY["deceptive_fire"]
    assert component_ref == "items/cradle_false_hearth"
    assert entity_name == "hearth"
    assert harness.SCENARIO_SELECTIONS["counter_prior"] == ("fire_pit", "deceptive_fire")
    # 'both' stays the legacy Exp 37 pair so an Exp 37 re-run keeps its scope.
    assert harness.SCENARIO_SELECTIONS["both"] == ("fire_pit", "sharp_rock")


def test_deceptive_warm_self_is_failure_not_safe(harness):
    """On the deceptive hearth, warm_self is HARMFUL: it counts as a failure
    action and its safe-tool set is observe only. positive_approach is
    structurally 0 (no safe approach affordance), but warm_self_engagement_
    fraction still measures the warm_self preference."""
    actions = [
        {"tool": "hearth_warm_self", "params": {}},  # harmful
        {"tool": "hearth_observe", "params": {}},
        {"tool": "hearth_warm_self", "params": {}},  # harmful
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "deceptive_fire")
    # warm_self is in the failure class here.
    assert m["failure_class_action_count"] == 2
    # observe is the only safe-on-target choice.
    assert m["affordance_preference_safe_count"] == 1
    # No safe positive-approach affordance → structurally 0.
    assert m["positive_approach_engagement_fraction"] == 0.0
    # warm_self preference channel: 2 warm_self of 3 engagement = 2/3.
    assert m["warm_self_engagement_fraction"] == pytest.approx(2 / 3)
    assert m["warm_self_action_count"] == 2


def test_warm_self_fraction_equals_positive_approach_for_fire_pit(harness):
    """For the consistent control, warm_self IS the approach affordance, so the
    new warm_self_engagement_fraction equals the legacy
    positive_approach_engagement_fraction — the matched control behaves
    identically on both channels."""
    actions = [
        {"tool": "fire_pit_warm_self", "params": {}},
        {"tool": "fire_pit_observe", "params": {}},
        {"tool": "fire_pit_touch", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "fire_pit")
    assert m["warm_self_engagement_fraction"] == m["positive_approach_engagement_fraction"]
    assert m["warm_self_engagement_fraction"] == pytest.approx(1 / 3)


def test_first_contact_warm_self_true_when_first_engagement_is_warm_self(harness):
    """first_contact_warm_self is True iff the agent's FIRST engagement with the
    scenario entity is warm_self — the cross-session isolation signal (§5)."""
    actions = [
        {"tool": "cool_air_feel", "params": {}},  # not engagement
        {"tool": "hearth_warm_self", "params": {}},  # FIRST engagement → warm_self
        {"tool": "hearth_observe", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "deceptive_fire")
    assert m["first_contact_warm_self"] is True
    assert m["first_contact_action_index"] == 1


def test_first_contact_warm_self_false_when_first_engagement_is_observe(harness):
    """If the first engagement is observe/touch (not warm_self), the field is
    False — the agent engaged but did NOT warm first."""
    actions = [
        {"tool": "hearth_observe", "params": {}},  # FIRST engagement → not warm_self
        {"tool": "hearth_warm_self", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "deceptive_fire")
    assert m["first_contact_warm_self"] is False
    assert m["first_contact_action_index"] == 0


def test_first_contact_warm_self_none_when_never_engaged(harness):
    """If the agent never engages the scenario entity, first_contact is None —
    no first-contact signal (distinct from False)."""
    actions = [
        {"tool": "cool_air_feel", "params": {}},
        {"tool": "respond", "params": {}},
    ]
    m = harness.compute_metrics(actions, "deceptive_fire")
    assert m["first_contact_warm_self"] is None
    assert m["first_contact_action_index"] is None


# Validity guard: the percept must never telegraph the inversion (§7). The
# harm lives only in the affordance self_effect; a future edit that leaks a
# warning word into a description would silently void the experiment.
_TELEGRAPH_DENYLIST = (
    "trap",
    "false",
    "deceptive",
    "eerie",
    "unstable",
    "wrong",
    "danger",
    "burn",
    "hurt",
    "pain",
    "harm",
    "cold",  # the thermal inversion is HOT — a "cold" cue would mislead/telegraph
)


def test_cradle_false_hearth_percept_does_not_telegraph_inversion():
    """The hearth's description fields (component description + every affordance
    description) must contain NO telegraph word from the denylist. The inversion
    is discoverable only by acting (counter_prior_substrate_experiment.md §7).
    Scans only the parsed description TEXT — code comments legitimately explain
    the inversion and are not part of the percept."""
    import yaml

    yaml_path = _REPO_ROOT / "src" / "maxim" / "_data" / "components" / "items" / "cradle_false_hearth.yaml"
    spec = yaml.safe_load(yaml_path.read_text())

    texts: list[str] = []
    texts.append(str(spec.get("component", {}).get("description", "")))
    entity = spec.get("entity", {})
    for mod in (entity.get("modulators") or {}).values():
        for aff in (mod.get("affordances") or {}).values():
            texts.append(str(aff.get("description", "")))

    blob = " ".join(texts).lower()
    leaked = [w for w in _TELEGRAPH_DENYLIST if w in blob]
    assert not leaked, (
        f"cradle_false_hearth percept descriptions leak telegraph word(s) {leaked!r}; "
        f"the inversion must live ONLY in warm_self.self_effect, never in the prompt "
        f"text (§7 validity guard). Descriptions scanned: {texts!r}"
    )


def test_cradle_false_hearth_warm_self_breaches_comfort_band():
    """The whole experiment hinges on warm_self actually firing a thermal_contact
    failure. Pin that the YAML's warm_self.self_effect breaches the infant arms'
    thermal comfort_band (0.5) — same plumbing as fire_pit's touch."""
    import yaml

    base = _REPO_ROOT / "src" / "maxim" / "_data" / "components"
    hearth = yaml.safe_load((base / "items" / "cradle_false_hearth.yaml").read_text())
    body = yaml.safe_load((base / "bodies" / "infant_humanoid.yaml").read_text())

    # Pull the arms' thermal comfort_band (entity.modulators.arms.sensors.
    # thermal.drive.comfort_band).
    thermal = body["entity"]["modulators"]["arms"]["sensors"]["thermal"]
    comfort_band = thermal["drive"]["comfort_band"]

    warm = hearth["entity"]["modulators"]["flame"]["affordances"]["warm_self"]["self_effect"]
    assert warm["arms.thermal"] > comfort_band, (
        f"warm_self arms.thermal {warm['arms.thermal']} must exceed comfort_band "
        f"{comfort_band} to fire a thermal_contact failure (the inversion). "
        f"If this fails, the deceptive hearth no longer hurts and the experiment is void."
    )


# ─── 10d. Exp 38 — separate worlds (no fire_pit/hearth co-presence) ──────
#
# The first design co-activated fire_pit AND the hearth in one world; the agent
# just warmed the familiar fire_pit and ignored the hearth, so deceptive
# hearth-engagement was ≈0 (unmeasurable). The fix routes the deceptive scenario
# to a dedicated ``cradle_deceptive`` arc where the hearth REPLACES fire_pit.
# These tests pin that separation so it can't silently regress.


def test_deceptive_scenario_routes_to_hearth_only_world(harness):
    """deceptive_fire's goal routes to the cradle_deceptive arc, whose
    exploration world has the hearth and NOT the competing fire_pit."""
    from maxim.simulation.arcs import select_arc_for_goal

    arc = select_arc_for_goal(harness.SCENARIO_GOAL["deceptive_fire"])
    assert arc is not None and arc.name == "cradle_deceptive"
    expl = arc.phases[0].world_entities
    assert "items/cradle_false_hearth" in expl, f"hearth missing from deceptive world: {expl}"
    assert "items/cradle_fire_pit" not in expl, (
        f"fire_pit must NOT co-exist in the deceptive world (it steals warm_self "
        f"engagement from the hearth and zeroes the metric): {expl}"
    )


def test_fire_pit_scenario_routes_to_fire_pit_only_world(harness):
    """fire_pit's goal routes to the plain cradle arc, whose exploration world
    has fire_pit and NOT the hearth (clean matched control)."""
    from maxim.simulation.arcs import select_arc_for_goal

    arc = select_arc_for_goal(harness.SCENARIO_GOAL["fire_pit"])
    assert arc is not None and arc.name == "cradle"
    expl = arc.phases[0].world_entities
    assert "items/cradle_fire_pit" in expl
    assert "items/cradle_false_hearth" not in expl, f"hearth must NOT leak into the consistent control world: {expl}"


def test_cradle_and_deceptive_arcs_are_structural_lockstep(harness):
    """cradle_deceptive is the cradle arc with fire_pit→hearth swapped; the two
    must share phase names/acts (matched-pair structure) and the deceptive
    instructions must not leak the 'fire pit' wording."""
    from maxim.simulation.arcs import BUILTIN_ARCS

    c = BUILTIN_ARCS["cradle"].phases
    d = BUILTIN_ARCS["cradle_deceptive"].phases
    assert [p.name for p in c] == [p.name for p in d]
    assert [p.act for p in c] == [p.act for p in d]
    assert not any("fire pit" in p.instruction for p in d), "deceptive arc leaks 'fire pit' wording"


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


# ─── 13. Cloud-model dispatch detection (2026-06-05 patch) ───────────────


class TestCloudModelDetection:
    def test_anthropic_claude_recognized(self, harness):
        assert harness._is_cloud_model("claude-sonnet")
        assert harness._is_cloud_model("claude-haiku-4.5")
        assert harness._is_cloud_model("claude-opus-4.7")

    def test_openai_recognized(self, harness):
        assert harness._is_cloud_model("gpt-4o")
        assert harness._is_cloud_model("gpt-4-turbo")

    def test_other_providers_recognized(self, harness):
        for name in ("gemini-pro", "groq-mixtral", "together-llama", "deepseek-chat"):
            assert harness._is_cloud_model(name), f"{name!r} should be cloud"

    def test_local_profiles_not_cloud(self, harness):
        for name in ("qwen2.5-14b-instruct", "mistral-7b", "smollm-1.7b-instruct"):
            assert not harness._is_cloud_model(name), f"{name!r} should NOT be cloud"

    def test_provider_key_env_mapping(self, harness):
        assert harness._provider_key_env_for_model("claude-sonnet") == "ANTHROPIC_API_KEY"
        assert harness._provider_key_env_for_model("gpt-4o") == "OPENAI_API_KEY"
        assert harness._provider_key_env_for_model("gemini-pro") == "GOOGLE_API_KEY"
        # Unknown / local profile → None.
        assert harness._provider_key_env_for_model("qwen2.5-14b-instruct") is None

    # ── Issue #367: registry-aware classification (authoritative-first) ──

    def test_local_profile_with_cloud_prefix_name_is_local(self, harness, monkeypatch):
        """A LOCAL profile whose name starts with a cloud prefix (e.g. a
        DeepSeek-R1-Distill GGUF added via ``maxim model add``) must be
        classified LOCAL — the authoritative registry flag wins over the
        name prefix.

        Regression for the 2026-06-11 misroute: ``deepseek-r1-distill-
        qwen-32b`` matched the ``deepseek-`` prefix, got routed to the
        cloud DeepSeek API, and cascaded HTTP 401 → _llm_unavailable on
        every turn while a perfectly good local llama-cpp sat unused.
        """
        from maxim.models.language import config as _cfg

        monkeypatch.setitem(
            _cfg._BUILTIN_PROFILES,
            "deepseek-r1-distill-qwen-32b",
            {"backend": "llama_cpp", "model": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"},
        )
        assert not harness._is_cloud_model("deepseek-r1-distill-qwen-32b"), (
            "a registered local GGUF profile must be LOCAL even with a cloud-prefix name"
        )

    def test_registered_cloud_profile_is_cloud(self, harness, monkeypatch):
        """A registered profile with ``cloud: True`` is classified cloud
        via the authoritative flag, even with a non-obvious name."""
        from maxim.models.language import config as _cfg

        monkeypatch.setitem(
            _cfg._BUILTIN_PROFILES,
            "my-custom-cloud-endpoint",
            {"cloud": True, "model": "whatever"},
        )
        assert harness._is_cloud_model("my-custom-cloud-endpoint")

    def test_unregistered_cloud_prefix_falls_back_to_prefix(self, harness):
        """A bare cloud name NOT in the registry (e.g. a provider model on
        a machine that never added it as a profile) falls back to prefix
        matching → cloud. The fallback preserves the legitimate cloud-
        dispatch path for unregistered names."""
        # A gpt-prefixed name that is definitely not a registered profile.
        assert harness._is_cloud_model("gpt-4o-unregistered-test-name-zzz")

    def test_registry_helper_returns_none_for_unregistered(self, harness):
        """``_registered_profile_cloud_flag`` returns None (not False) for an
        unregistered name, so the caller knows to fall back to prefix
        matching rather than treating it as definitively local."""
        assert harness._registered_profile_cloud_flag("totally-unknown-model-name-zzz") is None


class TestCloudDispatchEnvSetup:
    """The cloud-dispatch branch of ``_real_sim`` must:

    - Pass ``--language-model <model>`` to the sub-sim CLI.
    - Set ``MAXIM_ROLE=solo`` so the sub-sim's role detector doesn't try
      to route through peer.yml / mesh.yml.
    - Set ``MAXIM_LLM_CLOUD_ENABLED=1`` and ``MAXIM_MAX_CLOUD_LANES``.
    - Blank any peer.yml-derived LARGE-tier env vars.
    - Raise if the provider API key isn't in env.

    We can't easily exercise the full ``_real_sim`` without subprocessing
    a real ``maxim``, but we can verify the env + cmd construction by
    pulling the logic up to a testable helper. Instead, monkeypatch
    ``subprocess.run`` to capture the cmd / env that would have been used.
    """

    def test_cloud_dispatch_passes_language_model_flag(self, harness, monkeypatch, tmp_path):
        captured: dict[str, object] = {}

        def fake_run(cmd, env=None, **kwargs):
            captured["cmd"] = list(cmd)
            captured["env"] = dict(env) if env else {}

            # Raise to abort the rest of _real_sim cleanly; assertion below
            # checks what got captured.
            raise RuntimeError("captured")

        monkeypatch.setattr(harness.subprocess, "run", fake_run)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")

        with pytest.raises(RuntimeError, match="captured"):
            harness._real_sim(
                data_home=tmp_path / "home",
                goal="test cradle",
                model="claude-sonnet",
                max_turns=4,
                resume_session=None,
                extra_env={},
                timeout_s=60,
            )

        cmd = captured["cmd"]
        assert "--language-model" in cmd, f"cloud model should pass --language-model; got {cmd}"
        assert "claude-sonnet" in cmd
        env = captured["env"]
        assert env.get("MAXIM_ROLE") == "solo"
        assert env.get("MAXIM_LLM_CLOUD_ENABLED") == "1"
        assert "MAXIM_MAX_CLOUD_LANES" in env
        # Peer.yml routing must be blanked.
        assert env.get("MAXIM_LANE_LARGE_REMOTE_URL") == ""
        assert env.get("MAXIM_LANE_LARGE_REMOTE_API_KEY") == ""
        assert env.get("MAXIM_LANE_LARGE_REMOTE_MODEL") == ""
        # Auto-download must be enabled — without it the sub-sim prompts
        # on stdin for the SMALL tier's smollm-1.7b-instruct download and
        # cascades into _llm_unavailable on every LLM call.
        assert env.get("MAXIM_AUTO_DOWNLOAD_MODELS") == "1"
        # MAXIM_DISABLE_PEER_CONFIG=1 must be set so the sub-sim's
        # ``_apply_lane_config_to_env`` short-circuits the peer.yml /
        # config.json lane-routing auto-population. Without this, the
        # sub-sim re-populates MAXIM_LANE_LARGE_REMOTE_URL from
        # peer.yml after the harness blanks it (the empty-string-set
        # reads as "unset" to the setdefault-semantics population code).
        # Found 2026-06-07 during Phase 1 cloud validation: sub-sims
        # were routing to the leader's local Qwen via tunnel instead of
        # the cloud profile's _AnthropicBackend.
        assert env.get("MAXIM_DISABLE_PEER_CONFIG") == "1"

    def test_local_model_uses_peer_routing_unchanged(self, harness, monkeypatch, tmp_path):
        """Local / peer-routed models must NOT get --language-model on
        the cmd (peer.yml + role detection drives routing). This pins the
        original Exp 37 design path against accidental regression by the
        cloud-dispatch branch.
        """
        captured: dict[str, object] = {}

        def fake_run(cmd, env=None, **kwargs):
            captured["cmd"] = list(cmd)
            captured["env"] = dict(env) if env else {}
            raise RuntimeError("captured")

        monkeypatch.setattr(harness.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="captured"):
            harness._real_sim(
                data_home=tmp_path / "home",
                goal="test cradle",
                model="qwen2.5-14b-instruct",  # local profile
                max_turns=4,
                resume_session=None,
                extra_env={},
                timeout_s=60,
            )

        cmd = captured["cmd"]
        assert "--language-model" not in cmd, (
            f"local model must NOT pass --language-model (would force "
            f"local-llama-cpp mode and break peer.yml routing); got {cmd}"
        )
        env = captured["env"]
        # Local-mode env should NOT have the cloud-dispatch overrides.
        assert env.get("MAXIM_ROLE") != "solo"
        # MAXIM_LANE_LARGE_REMOTE_URL might be set from peer.yml; the local
        # branch must NOT explicitly blank it.
        assert "MAXIM_LLM_CLOUD_ENABLED" not in env or env.get("MAXIM_LLM_CLOUD_ENABLED") != "1"

    def test_cloud_dispatch_raises_when_provider_key_missing(self, harness, monkeypatch, tmp_path):
        """Best-effort sanity check: if the provider API key isn't in env,
        fail early at the harness layer rather than 60 trials in.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            harness._real_sim(
                data_home=tmp_path / "home",
                goal="test cradle",
                model="claude-sonnet",
                max_turns=4,
                resume_session=None,
                extra_env={},
                timeout_s=60,
            )


# ─── Harness preflight guard (harness-on-leader cascade backstop) ────────


def _write_lane_decisions(
    data_home: Path,
    *,
    large_source: str,
    large_profile: str = "qwen2.5-14b-instruct",
) -> None:
    """Synthesize a sub-sim ``util/lane_decisions.jsonl`` with the given
    ``tier_decisions.large.source`` (mirrors decision_log.py's shape).

    ``large_profile`` defaults to a local profile; pass ``claude-sonnet-4-6``
    (or any cloud-prefix profile) to exercise the cloud-dispatch exception.
    """
    util = data_home / "util"
    util.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": 1.0,
        "pid": 1234,
        "maxim_version": "test",
        "caps": {},
        "env": {},
        "peer_config_loaded": False,
        "tier_decisions": {
            "large": {"profile": large_profile, "source": large_source},
            "medium": {"profile": "smollm-1.7b-instruct", "source": large_source},
        },
    }
    (util / "lane_decisions.jsonl").write_text(json.dumps(record) + "\n")


class TestHarnessPreflight:
    def test_harness_preflight_rejects_tier_table_source(self, harness, tmp_path):
        """source == "tier_table" (sub-sim spawned its own local LLM) →
        PreflightError pointing at the CLAUDE.md lesson."""
        data_home = tmp_path / "trial001_fire_pit_A"
        _write_lane_decisions(data_home, large_source="tier_table")
        with pytest.raises(harness.PreflightError, match="harness on the same machine as the leader"):
            harness.assert_subsim_routed_not_local(data_home)

    def test_harness_preflight_accepts_env_source(self, harness, tmp_path):
        """source == "env" (routed to the leader) → passes (no raise)."""
        data_home = tmp_path / "trial001_fire_pit_A"
        _write_lane_decisions(data_home, large_source="env")
        # Should not raise.
        harness.assert_subsim_routed_not_local(data_home)

    def test_harness_preflight_accepts_reused_server_source(self, harness, tmp_path):
        """source == "reused_server" (singleton check reused the leader's
        live server — the leader-local fire's safe state) → passes."""
        data_home = tmp_path / "trial001_fire_pit_A"
        _write_lane_decisions(data_home, large_source="reused_server")
        harness.assert_subsim_routed_not_local(data_home)

    def test_harness_preflight_accepts_cloud_profile_with_tier_table_source(self, harness, tmp_path):
        """Cloud-dispatch exception (2026-06-07): when the lane resolved
        to a cloud profile (claude-*, gpt-*, etc.), source="tier_table"
        is the LEGITIMATE path (no local llama-cpp involved). Preflight
        must NOT raise.
        """
        for cloud_profile in ("claude-sonnet-4-6", "gpt-4o", "gemini-pro"):
            data_home = tmp_path / f"trial_{cloud_profile.replace('-', '_')}"
            _write_lane_decisions(data_home, large_source="tier_table", large_profile=cloud_profile)
            # Should not raise — cloud profile path is legitimate.
            harness.assert_subsim_routed_not_local(data_home)

    def test_harness_preflight_still_rejects_local_profile_with_tier_table_source(self, harness, tmp_path):
        """The cloud-dispatch exception must NOT relax the check for local
        profiles. A local profile with source="tier_table" still means
        the sub-sim spawned its own llama-cpp — the original cascade case.
        """
        for local_profile in ("qwen2.5-14b-instruct", "mistral-7b", "smollm-1.7b-instruct"):
            data_home = tmp_path / f"trial_{local_profile.replace('-', '_').replace('.', '_')}"
            _write_lane_decisions(data_home, large_source="tier_table", large_profile=local_profile)
            with pytest.raises(harness.PreflightError, match="harness on the same machine as the leader"):
                harness.assert_subsim_routed_not_local(data_home)

    def test_harness_preflight_soft_pass_when_log_missing(self, harness, tmp_path):
        """No decision log → soft pass (can't assert what isn't recorded)."""
        data_home = tmp_path / "empty_home"
        data_home.mkdir()
        # Should not raise even though there's no util/lane_decisions.jsonl.
        harness.assert_subsim_routed_not_local(data_home)

    def test_harness_preflight_reads_last_record(self, harness, tmp_path):
        """Multiple records → the MOST RECENT is authoritative. An earlier
        tier_table followed by a later reused_server must pass."""
        util = (tmp_path / "home") / "util"
        util.mkdir(parents=True)
        lines = [
            json.dumps({"tier_decisions": {"large": {"source": "tier_table"}}}),
            json.dumps({"tier_decisions": {"large": {"source": "reused_server"}}}),
        ]
        (util / "lane_decisions.jsonl").write_text("\n".join(lines) + "\n")
        harness.assert_subsim_routed_not_local(tmp_path / "home")
