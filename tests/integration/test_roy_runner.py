"""Integration tests for the Roy three-arm iteration runner.

Spec: ``src/maxim/simulation/roy_runner.py``. The runner ties together
R1 (curriculum runner) and R2 (substrate diff library) into one
operator command that primes arm A, runs all three arms on the same
test scenario, and writes pairwise substrate divergence to
``~/.maxim/roy/<iteration_name>/``.

These tests use a fake ``sim_runner`` (similar to the curriculum
runner's fake) that writes realistic ``aut_nac.json`` /
``aut_hippocampus.json`` snapshots to a tmp ``MAXIM_DATA_HOME``.
``run_curriculum`` is invoked for real with the same fake injected,
so the priming → arm A resume handoff is exercised end-to-end with
no orchestrator / LLM dependency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from maxim.simulation.roy_runner import (
    ArmResult,
    RoyIterationResult,
    run_roy_iteration,
    validate_spec,
)
from maxim.simulation.sim_types import SimulationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_data_home(monkeypatch, tmp_path):
    """Point ~/.maxim at a tmp dir so artifact writes don't touch the
    developer's real ``~/.maxim/``.
    """
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    from maxim.utils import paths

    paths._reset_caches()
    yield tmp_path / "data"
    paths._reset_caches()


def _smoke_fixture(tmp_path: Path, name: str = "warmup.yaml") -> Path:
    """A minimal fixture so the runner's existence + path resolution
    passes. The fake sim_runner doesn't consume the fixture; we just
    need a real file to point at.
    """
    p = tmp_path / name
    p.write_text(
        "name: smoke\n"
        "timing: step_based\n"
        "percepts:\n"
        "  - at: 0\n"
        "    source: cli\n"
        "    cli_input: hello\n"
        "    salience: 0.5\n"
        "    novelty: 0.5\n"
    )
    return p


def _write_yaml(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Fake sim_runner: writes substrate_diff-compatible snapshots.
# ---------------------------------------------------------------------------


def _make_fake_sim_runner(
    *,
    growth_per_call: int = 5,
    fail_on_arm: str | None = None,
) -> Callable[..., SimulationResult]:
    """Build a fake ``start_simulation_mode`` substitute.

    Each call:

    * Generates a unique session_id (monotonic — resists same-second collisions).
    * Writes ``aut_nac.json`` / ``aut_hippocampus.json`` into the
      session dir under ``sim_reports()``. Snapshot shapes match what
      ``substrate_diff`` reads: ``reward_bias`` dict, ``links`` dict
      of lists, episodes/memories lists.
    * When ``resume_session`` is set, copies that session's substrate
      forward and adds ``growth_per_call`` new keys / episodes — so
      arm A (resume_session=priming) ends with strictly larger
      substrate than arms B and C (blank).

    ``fail_on_arm`` raises during the matching arm's sim so we can
    exercise the partial-success path through pairwise_diffs.
    """
    call_log: list[dict[str, Any]] = []

    def runner(**kwargs: Any) -> SimulationResult:
        from maxim.utils.paths import sim_reports as _sim_reports

        call_idx = len(call_log) + 1
        goal = kwargs.get("goal", "")

        if fail_on_arm is not None and goal.endswith(f":arm_{fail_on_arm}"):
            call_log.append({"raised": True, "kwargs": kwargs})
            raise RuntimeError(f"fake sim runner raised on arm {fail_on_arm}")

        session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_call{call_idx:03d}"
        session_dir = _sim_reports() / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Carry substrate forward if resuming.
        prev_reward_bias: dict[str, float] = {}
        prev_links: dict[str, list[Any]] = {}
        prev_episodes: list[dict[str, Any]] = []
        resume_id = kwargs.get("resume_session")
        if resume_id:
            prev_dir = _sim_reports() / resume_id
            prev_nac = prev_dir / "aut_nac.json"
            prev_hippo = prev_dir / "aut_hippocampus.json"
            if prev_nac.exists():
                with open(prev_nac) as f:
                    prev = json.load(f)
                prev_reward_bias = dict(prev.get("reward_bias", {}))
                prev_links = dict(prev.get("links", {}))
            if prev_hippo.exists():
                with open(prev_hippo) as f:
                    prev = json.load(f)
                prev_episodes = list(prev.get("episodes", []))

        # Grow substrate this call.
        new_reward_bias = dict(prev_reward_bias)
        for i in range(growth_per_call):
            key = f"call{call_idx:03d}_node{i}"
            new_reward_bias[key] = 0.1 * (i + 1)

        new_links = dict(prev_links)
        new_links[f"src_call{call_idx:03d}"] = [{"target": f"tgt_{i}", "strength": 0.5} for i in range(growth_per_call)]

        new_episodes = list(prev_episodes)
        for i in range(growth_per_call):
            new_episodes.append(
                {
                    "id": f"ep_call{call_idx:03d}_{i}",
                    "valence": 0.1 * (i + 1) - 0.5 * (1 if i % 2 else 0),
                }
            )

        # Write NAc snapshot (substrate_diff reads reward_bias + links).
        with open(session_dir / "aut_nac.json", "w") as f:
            json.dump(
                {
                    "_format_version": "1.0",
                    "reward_bias": new_reward_bias,
                    "goal_reward_bias": {},
                    "links": new_links,
                    "resumed_from": resume_id or "",
                },
                f,
            )
        # Write Hippocampus snapshot (substrate_diff reads episodes).
        with open(session_dir / "aut_hippocampus.json", "w") as f:
            json.dump(
                {
                    "_format_version": "1.0",
                    "episodes": new_episodes,
                    "memories": [],
                    "resumed_from": resume_id or "",
                },
                f,
            )

        call_log.append(
            {
                "kwargs": kwargs,
                "session_id": session_id,
                "session_dir": str(session_dir),
                "reward_bias_keys": len(new_reward_bias),
                "episodes": len(new_episodes),
            }
        )

        return SimulationResult(
            goal=goal,
            persona=kwargs.get("persona", ""),
            turns=int(kwargs.get("max_turns", 0)),
            total_actions=growth_per_call,
            blocked_actions=0,
            duration_s=0.01,
            finish_reason="complete",
            session_id=session_id,
            session_dir=str(session_dir),
        )

    runner.call_log = call_log  # type: ignore[attr-defined]
    return runner


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------


def _build_smoke_spec(tmp_path: Path, *, inline_priming: bool = True) -> Path:
    fixture = _smoke_fixture(tmp_path)
    if inline_priming:
        return _write_yaml(
            tmp_path / "iter.yaml",
            "name: roy-smoke\n"
            "description: tiny smoke\n"
            "embodiment: bodies/infant_humanoid\n"
            "aut_mode: substrate-primary\n"
            "priming:\n"
            "  name: roy-smoke-priming\n"
            "  stages:\n"
            "    - name: stage1\n"
            f"      fixture: {fixture.name}\n"
            "      turns: 2\n"
            "test_scenario:\n"
            f"  fixture: {fixture.name}\n"
            "  turns: 2\n"
            "arms:\n"
            "  a: { substrate: from_priming, system_prompt: neutral }\n"
            "  b: { substrate: blank,        system_prompt: adversarial }\n"
            "  c: { substrate: blank,        system_prompt: neutral }\n",
        )
    else:
        curriculum = _write_yaml(
            tmp_path / "priming.yaml",
            f"name: roy-smoke-priming\nstages:\n  - name: stage1\n    fixture: {fixture.name}\n    turns: 2\n",
        )
        return _write_yaml(
            tmp_path / "iter.yaml",
            "name: roy-smoke\n"
            "embodiment: bodies/infant_humanoid\n"
            "aut_mode: substrate-primary\n"
            f"priming: {curriculum.name}\n"
            "test_scenario:\n"
            f"  fixture: {fixture.name}\n"
            "  turns: 2\n"
            "arms:\n"
            "  a: { substrate: from_priming, system_prompt: neutral }\n"
            "  b: { substrate: blank,        system_prompt: adversarial }\n"
            "  c: { substrate: blank,        system_prompt: neutral }\n",
        )


# ---------------------------------------------------------------------------
# 1. End-to-end smoke: priming + 3 arms + pairwise diffs
# ---------------------------------------------------------------------------


class TestRoyRunnerInvariants:
    """Process-global invariants the Roy runner is responsible for."""

    def test_run_forces_interactive_off(self, isolated_data_home, tmp_path):
        """R5 Roy-0 finding: the orchestrator's AUTO interactive mode
        enables Rich Live + raw-terminal stdin reader on TTY launch,
        which deadlocks when the runner drives five+ sims back-to-back
        from a YAML spec. The runner must force ``interactive=off``
        process-globally at entry so all three arms (and the priming
        curriculum) inherit the non-interactive setting regardless of
        how the operator invoked the CLI.
        """
        from maxim.simulation.sim_logger import (
            InteractiveMode,
            get_interactive_mode,
            set_interactive_mode,
        )

        # Pre-arm to ON so we can detect the runner forcing it OFF.
        set_interactive_mode("on")
        assert get_interactive_mode() == InteractiveMode.ON

        spec_path = _build_smoke_spec(tmp_path)
        runner = _make_fake_sim_runner()
        run_roy_iteration(
            spec_path,
            sim_runner=runner,
            artifact_root=tmp_path / "roy",
        )
        # After the run, interactive mode is OFF — the runner does NOT
        # restore the prior mode (see roy_runner.py rationale).
        assert get_interactive_mode() == InteractiveMode.OFF


class TestRoyEndToEnd:
    def test_three_arm_smoke_runs_to_completion(self, isolated_data_home, tmp_path):
        """The full pipeline produces three distinct arm sessions and
        three available pairwise diffs.
        """
        spec_path = _build_smoke_spec(tmp_path)
        runner = _make_fake_sim_runner(growth_per_call=3)

        result = run_roy_iteration(
            spec_path,
            sim_runner=runner,
            artifact_root=tmp_path / "roy",
        )

        assert isinstance(result, RoyIterationResult)
        assert result.aborted_at is None
        assert result.name == "roy-smoke"
        assert result.description == "tiny smoke"

        # All three arms ran and produced session ids.
        assert set(result.arms.keys()) == {"a", "b", "c"}
        for key in ("a", "b", "c"):
            arm = result.arms[key]
            assert isinstance(arm, ArmResult)
            assert arm.error is None, f"arm {key} errored: {arm.error}"
            assert arm.session_id, f"arm {key} missing session_id"
            assert Path(arm.session_dir).is_dir()

        # All three session_ids are distinct.
        ids = {result.arms[k].session_id for k in ("a", "b", "c")}
        assert len(ids) == 3, f"expected distinct session_ids, got {ids}"

        # Arm A resumed from priming; B and C did not.
        assert result.arms["a"].resumed_from == result.priming["final_session_id"]
        assert result.arms["b"].resumed_from == ""
        assert result.arms["c"].resumed_from == ""

    def test_pairwise_diffs_populated_with_real_substrate(self, isolated_data_home, tmp_path):
        """All three pairwise diffs are available, NAc + EC + Hippocampus
        sections report concrete numbers, and A-vs-blank pairs show
        non-zero divergence (arm A carries priming substrate forward).
        """
        spec_path = _build_smoke_spec(tmp_path)
        runner = _make_fake_sim_runner(growth_per_call=4)

        result = run_roy_iteration(
            spec_path,
            sim_runner=runner,
            artifact_root=tmp_path / "roy",
        )
        assert result.aborted_at is None

        for key in ("a_vs_b", "a_vs_c", "b_vs_c"):
            d = result.pairwise_diffs[key]
            assert d.get("available") is True, f"{key} unavailable: {d}"
            assert d["nac"]["available"] is True
            # EC snapshot isn't written by the fake runner, so it
            # surfaces as available=false rather than crashing.
            assert "ec" in d
            assert d["hippocampus"]["available"] is True

        # A vs C: A has strictly more substrate than C because A
        # resumed from priming. The NAc reward_bias L2 between them
        # should be non-zero.
        a_vs_c = result.pairwise_diffs["a_vs_c"]
        assert a_vs_c["nac"]["reward_bias_l2"] > 0.0, (
            "A-vs-C NAc reward_bias L2 should be non-zero — A inherited priming substrate"
        )
        # Episodes are appended monotonically — A should have strictly
        # more than C.
        a_vs_c_hipp = a_vs_c["hippocampus"]
        assert a_vs_c_hipp["episode_count_delta"] > 0

        # B vs C: both blank, equal turns. Reward-bias keys are
        # call-indexed and B's call_idx != C's, so the keys differ —
        # the L2 is non-zero. The test just asserts the diff *exists*
        # and is computable, not direction.
        b_vs_c = result.pairwise_diffs["b_vs_c"]
        assert b_vs_c["nac"]["available"] is True

    def test_artifact_persistence(self, isolated_data_home, tmp_path):
        """result.json carries _format_version + arms + pairwise_diffs,
        and summary.md mentions every arm + diff key.
        """
        spec_path = _build_smoke_spec(tmp_path)
        runner = _make_fake_sim_runner()

        result = run_roy_iteration(
            spec_path,
            sim_runner=runner,
            artifact_root=tmp_path / "roy",
        )

        artifact_dir = Path(result.artifact_dir)
        result_path = artifact_dir / "result.json"
        summary_path = artifact_dir / "summary.md"

        assert result_path.is_file()
        assert summary_path.is_file()

        payload = json.loads(result_path.read_text())
        assert payload["_format_version"] == "1.0"
        assert payload["name"] == "roy-smoke"
        assert set(payload["arms"].keys()) == {"a", "b", "c"}
        assert set(payload["pairwise_diffs"].keys()) == {"a_vs_b", "a_vs_c", "b_vs_c"}
        assert payload["aborted_at"] is None

        summary = summary_path.read_text()
        for tag in ("roy-smoke", "Priming", "Arms", "a_vs_b", "a_vs_c", "b_vs_c"):
            assert tag in summary, f"missing {tag!r} in summary.md"

    def test_external_curriculum_path_also_works(self, isolated_data_home, tmp_path):
        """``priming: <path>`` (string) resolves relative to the spec
        directory and runs the same as an inline curriculum.
        """
        spec_path = _build_smoke_spec(tmp_path, inline_priming=False)
        runner = _make_fake_sim_runner()

        result = run_roy_iteration(
            spec_path,
            sim_runner=runner,
            artifact_root=tmp_path / "roy",
        )
        assert result.aborted_at is None
        assert all(a.session_id for a in result.arms.values())

    def test_inline_priming_file_is_cleaned_up(self, isolated_data_home, tmp_path):
        """The materialised inline-priming sibling file should be
        removed after the curriculum runs so the spec directory
        doesn't accumulate transient artifacts.
        """
        spec_path = _build_smoke_spec(tmp_path, inline_priming=True)
        before = set(tmp_path.iterdir())
        runner = _make_fake_sim_runner()

        result = run_roy_iteration(
            spec_path,
            sim_runner=runner,
            artifact_root=tmp_path / "roy",
        )
        assert result.aborted_at is None

        after = set(tmp_path.iterdir())
        # The only new entries should be the runner's artifact_root,
        # not a stray .roy_inline_priming_*.yaml left over.
        new_entries = after - before
        assert not any(p.name.startswith(".roy_inline_priming_") for p in new_entries), (
            f"inline priming file was not cleaned up: {[p.name for p in new_entries]}"
        )


# ---------------------------------------------------------------------------
# 2. Failure surfaces honestly
# ---------------------------------------------------------------------------


class TestRoyFailureHandling:
    def test_priming_failure_aborts_before_arms(self, isolated_data_home, tmp_path):
        """If priming aborts, no arms are invoked and aborted_at is
        ``priming``. The runner still persists result.json so the
        operator can inspect the partial state.
        """
        fixture = _smoke_fixture(tmp_path)
        # Spec with a missing priming fixture so run_curriculum fails
        # at parse time INSIDE run_roy_iteration's priming step.
        spec_path = _write_yaml(
            tmp_path / "iter.yaml",
            "name: roy-fail\n"
            "priming:\n"
            "  name: roy-fail-priming\n"
            "  stages:\n"
            "    - name: stage1\n"
            "      fixture: definitely_missing.yaml\n"
            "      turns: 1\n"
            "test_scenario:\n"
            f"  fixture: {fixture.name}\n"
            "  turns: 1\n"
            "arms:\n"
            "  a: { substrate: from_priming, system_prompt: neutral }\n"
            "  b: { substrate: blank,        system_prompt: adversarial }\n"
            "  c: { substrate: blank,        system_prompt: neutral }\n",
        )
        runner = _make_fake_sim_runner()

        result = run_roy_iteration(
            spec_path,
            sim_runner=runner,
            artifact_root=tmp_path / "roy",
        )

        assert result.aborted_at == "priming"
        assert result.arms == {}  # no arms invoked
        assert result.pairwise_diffs == {}
        assert (Path(result.artifact_dir) / "result.json").is_file()
        # The sim runner was never called.
        assert runner.call_log == []  # type: ignore[attr-defined]

    def test_failed_arm_marks_pair_unavailable(self, isolated_data_home, tmp_path):
        """When one arm errors, the iteration still completes the other
        two and computes the b_vs_c pair; pairs involving the failed
        arm surface ``available: false`` rather than crashing.
        """
        spec_path = _build_smoke_spec(tmp_path)
        runner = _make_fake_sim_runner(fail_on_arm="a")

        result = run_roy_iteration(
            spec_path,
            sim_runner=runner,
            artifact_root=tmp_path / "roy",
        )
        # Iteration completes (no priming abort); arm A is the one
        # that errored.
        assert result.aborted_at is None
        assert result.arms["a"].error is not None
        assert "fake sim runner raised" in result.arms["a"].error
        assert result.arms["b"].error is None
        assert result.arms["c"].error is None

        # Pairs involving A are unavailable; b_vs_c is fine.
        assert result.pairwise_diffs["a_vs_b"]["available"] is False
        assert result.pairwise_diffs["a_vs_c"]["available"] is False
        assert result.pairwise_diffs["b_vs_c"]["available"] is True


# ---------------------------------------------------------------------------
# 3. Spec validation — fail-fast on malformed specs
# ---------------------------------------------------------------------------


class TestSpecValidation:
    def test_missing_arms_rejected(self, tmp_path):
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\n"
            "priming:\n"
            "  name: p\n"
            "  stages:\n"
            f"    - {{name: s, fixture: {fixture.name}, turns: 1}}\n"
            "test_scenario:\n"
            f"  fixture: {fixture.name}\n"
            "  turns: 1\n",
        )
        with pytest.raises(ValueError, match="arms"):
            validate_spec(spec)

    def test_arm_a_must_be_from_priming(self, tmp_path):
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\n"
            "priming:\n"
            "  name: p\n"
            "  stages:\n"
            f"    - {{name: s, fixture: {fixture.name}, turns: 1}}\n"
            "test_scenario:\n"
            f"  fixture: {fixture.name}\n"
            "  turns: 1\n"
            "arms:\n"
            "  a: { substrate: blank,        system_prompt: neutral }\n"
            "  b: { substrate: blank,        system_prompt: adversarial }\n"
            "  c: { substrate: blank,        system_prompt: neutral }\n",
        )
        with pytest.raises(ValueError, match="arms.a.substrate"):
            validate_spec(spec)

    def test_invalid_substrate_value_rejected(self, tmp_path):
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\n"
            "priming:\n"
            "  name: p\n"
            "  stages:\n"
            f"    - {{name: s, fixture: {fixture.name}, turns: 1}}\n"
            "test_scenario:\n"
            f"  fixture: {fixture.name}\n"
            "  turns: 1\n"
            "arms:\n"
            "  a: { substrate: hybrid,       system_prompt: neutral }\n"
            "  b: { substrate: blank,        system_prompt: adversarial }\n"
            "  c: { substrate: blank,        system_prompt: neutral }\n",
        )
        with pytest.raises(ValueError, match="substrate"):
            validate_spec(spec)

    def test_missing_test_scenario_fixture_rejected(self, tmp_path):
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\n"
            "priming:\n"
            "  name: p\n"
            "  stages:\n"
            f"    - {{name: s, fixture: {fixture.name}, turns: 1}}\n"
            "test_scenario:\n"
            "  fixture: nope.yaml\n"
            "  turns: 1\n"
            "arms:\n"
            "  a: { substrate: from_priming, system_prompt: neutral }\n"
            "  b: { substrate: blank,        system_prompt: adversarial }\n"
            "  c: { substrate: blank,        system_prompt: neutral }\n",
        )
        with pytest.raises(ValueError, match="test_scenario fixture not found"):
            validate_spec(spec)

    def test_missing_priming_rejected(self, tmp_path):
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\n"
            "test_scenario:\n"
            f"  fixture: {fixture.name}\n"
            "  turns: 1\n"
            "arms:\n"
            "  a: { substrate: from_priming, system_prompt: neutral }\n"
            "  b: { substrate: blank,        system_prompt: adversarial }\n"
            "  c: { substrate: blank,        system_prompt: neutral }\n",
        )
        with pytest.raises(ValueError, match="priming"):
            validate_spec(spec)

    def test_unknown_arm_key_rejected(self, tmp_path):
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\n"
            "priming:\n"
            "  name: p\n"
            "  stages:\n"
            f"    - {{name: s, fixture: {fixture.name}, turns: 1}}\n"
            "test_scenario:\n"
            f"  fixture: {fixture.name}\n"
            "  turns: 1\n"
            "arms:\n"
            "  a: { substrate: from_priming, system_prompt: neutral }\n"
            "  b: { substrate: blank,        system_prompt: adversarial }\n"
            "  c: { substrate: blank,        system_prompt: neutral }\n"
            "  d: { substrate: blank,        system_prompt: neutral }\n",
        )
        with pytest.raises(ValueError, match="unknown key"):
            validate_spec(spec)


# ---------------------------------------------------------------------------
# 4. CLI dry-run + run path
# ---------------------------------------------------------------------------


class TestRoyCli:
    def test_dry_run_validates_without_invoking_sims(self, isolated_data_home, tmp_path):
        """``maxim roy run <spec> --dry-run`` parses + reports without
        creating any artifact dir or invoking sims.
        """
        spec_path = _build_smoke_spec(tmp_path)
        from maxim.roy.cli import run_roy_subcommand

        rc = run_roy_subcommand(["run", str(spec_path), "--dry-run"])
        assert rc == 0
        # No artifact dir was created — dry-run is read-only.
        assert not (isolated_data_home / "roy").exists()

    def test_dry_run_rejects_malformed_spec(self, isolated_data_home, tmp_path):
        """A malformed spec under --dry-run exits non-zero without
        running anything.
        """
        bad = _write_yaml(tmp_path / "bad.yaml", "not: a valid spec\n")
        from maxim.roy.cli import run_roy_subcommand

        rc = run_roy_subcommand(["run", str(bad), "--dry-run"])
        assert rc == 2

    def test_dry_run_missing_file(self, isolated_data_home, tmp_path):
        from maxim.roy.cli import run_roy_subcommand

        rc = run_roy_subcommand(["run", str(tmp_path / "nope.yaml"), "--dry-run"])
        assert rc == 2

    def test_bundled_smoke_spec_dry_runs(self):
        """The committed scenarios/roy/roy_0_smoke_iteration.yaml
        smoke spec validates against the parser. Catches drift
        between the spec format and the bundled example.
        """
        repo_root = Path(__file__).resolve().parents[2]
        spec = repo_root / "scenarios" / "roy" / "roy_0_smoke_iteration.yaml"
        assert spec.exists(), f"bundled smoke spec missing: {spec}"

        parsed = validate_spec(spec)
        assert parsed.name == "roy-0-smoke"
        assert set(parsed.arms.keys()) == {"a", "b", "c"}
        assert parsed.arms["a"].substrate == "from_priming"
        assert parsed.arms["b"].substrate == "blank"
        assert parsed.arms["c"].substrate == "blank"
