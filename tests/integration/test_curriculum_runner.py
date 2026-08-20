"""Integration tests for the Roy curriculum runner.

Spec: ``src/maxim/simulation/curriculum_runner.py``. The runner chains
substrate-only priming sessions by threading ``resume_session`` from
each stage's :class:`SimulationResult` into the next stage's
``start_simulation_mode`` call.

These tests use a fake ``sim_runner`` injected via the
``sim_runner=`` parameter so we don't have to spin up the full
orchestrator (LLM router, sandbox, AUT thread). The fake writes
realistic ``aut_hippocampus.json`` / ``aut_nac.json`` snapshots to a
tmp ``MAXIM_DATA_HOME`` so we can verify the substrate actually grows
across stages, mirroring the on-disk shape of a real run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from maxim.simulation.curriculum_runner import (
    CurriculumResult,
    StageResult,
    run_curriculum,
)
from maxim.simulation.sim_types import SimulationResult


# ---------------------------------------------------------------------------
# Fake sim_runner: writes realistic substrate snapshots to a tmp data dir.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_sim_reports(monkeypatch, tmp_path):
    """Point ~/.maxim at a tmp dir so save/load roundtrips don't touch
    the developer's real ``~/.maxim/sim_reports/``.
    """
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    from maxim.utils import paths

    paths._reset_caches()
    yield tmp_path / "data"
    paths._reset_caches()


def _make_fake_sim_runner(
    *,
    growth_per_call: int = 5,
    fail_on_call: int | None = None,
    return_empty_session_on_call: int | None = None,
    finish_reason_on_call: tuple[int, str] | None = None,
) -> Callable[..., SimulationResult]:
    """Build a fake ``start_simulation_mode`` substitute.

    Each call:
    * Generates a unique session_id (monotonic; resists same-second collisions).
    * Writes ``aut_hippocampus.json`` + ``aut_nac.json`` into the
      session dir under ``sim_reports()``. ``aut_nac.json``'s shape
      mirrors the real persisted format closely enough that loading
      it via ``NAc.load`` would round-trip — we record link counts in
      ``call_log[i]["nac_links"]`` so tests can assert growth.
    * If ``resume_session`` was passed, copies the previous session's
      substrate counts forward and adds ``growth_per_call``.

    ``fail_on_call`` raises on the Nth call (1-indexed) to simulate an
    orchestrator failure mid-curriculum.
    ``return_empty_session_on_call`` returns a SimulationResult with
    ``session_id=""`` to simulate an early-bail orchestrator (proves
    the runner refuses to chain into a missing session).
    ``finish_reason_on_call`` returns an otherwise complete session with the
    supplied terminal status, exercising cleanly unwound runtime aborts.
    """
    call_log: list[dict[str, Any]] = []

    def runner(**kwargs: Any) -> SimulationResult:
        from maxim.utils.paths import sim_reports as _sim_reports

        call_idx = len(call_log) + 1
        if fail_on_call is not None and call_idx == fail_on_call:
            call_log.append({"raised": True, "kwargs": kwargs})
            raise RuntimeError(f"fake sim runner raised on call {call_idx}")

        if return_empty_session_on_call is not None and call_idx == return_empty_session_on_call:
            call_log.append({"empty_session": True, "kwargs": kwargs})
            # finish_reason="complete" deliberately — we want the
            # session_id="" branch to fire, not the finish_reason=="error"
            # branch (which fires first when both signal a failure).
            return SimulationResult(
                goal=kwargs.get("goal", ""),
                mode=kwargs.get("mode", ""),
                turns=0,
                total_actions=0,
                blocked_actions=0,
                duration_s=0.0,
                finish_reason="complete",
                session_id="",
                session_dir="",
            )

        # Compose a stable, monotone session_id so two calls in the
        # same second don't collide (real start_simulation_mode uses
        # strftime, which can — that's a separate orchestrator bug we
        # don't want to import into the test surface).
        session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_call{call_idx:03d}"
        session_dir = _sim_reports() / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Carry substrate forward if resuming.
        prev_links = 0
        prev_episodes = 0
        resume_id = kwargs.get("resume_session")
        if resume_id:
            prev_dir = _sim_reports() / resume_id
            prev_nac = prev_dir / "aut_nac.json"
            prev_hippo = prev_dir / "aut_hippocampus.json"
            if prev_nac.exists():
                with open(prev_nac) as f:
                    prev_links = int(json.load(f).get("link_count", 0))
            if prev_hippo.exists():
                with open(prev_hippo) as f:
                    prev_episodes = int(json.load(f).get("episode_count", 0))

        new_links = prev_links + growth_per_call
        new_episodes = prev_episodes + growth_per_call

        with open(session_dir / "aut_nac.json", "w") as f:
            json.dump(
                {
                    "_format_version": "1.0",
                    "link_count": new_links,
                    "resumed_from": resume_id or "",
                },
                f,
            )
        with open(session_dir / "aut_hippocampus.json", "w") as f:
            json.dump(
                {
                    "_format_version": "1.0",
                    "episode_count": new_episodes,
                    "resumed_from": resume_id or "",
                },
                f,
            )

        call_log.append(
            {
                "kwargs": kwargs,
                "session_id": session_id,
                "session_dir": str(session_dir),
                "nac_links": new_links,
                "hippo_episodes": new_episodes,
            }
        )

        finish_reason = "complete"
        if finish_reason_on_call is not None and call_idx == finish_reason_on_call[0]:
            finish_reason = finish_reason_on_call[1]

        return SimulationResult(
            goal=kwargs.get("goal", ""),
            mode=kwargs.get("mode", ""),
            turns=int(kwargs.get("max_turns", 0)),
            total_actions=growth_per_call,
            blocked_actions=0,
            duration_s=0.01,
            finish_reason=finish_reason,
            session_id=session_id,
            session_dir=str(session_dir),
        )

    runner.call_log = call_log  # type: ignore[attr-defined]
    return runner


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _smoke_fixture(tmp_path: Path, name: str = "warmup.yaml") -> Path:
    """A minimal ScenarioSource-shaped fixture so the runner's path
    resolution + existence check passes. The fake sim_runner doesn't
    actually consume the fixture; we just need a real file to point at.
    """
    return _write_yaml(
        tmp_path / name,
        "name: smoke\n"
        "timing: step_based\n"
        "percepts:\n"
        "  - at: 0\n"
        "    source: cli\n"
        "    cli_input: hello\n"
        "    salience: 0.5\n"
        "    novelty: 0.5\n",
    )


# ---------------------------------------------------------------------------
# 1. End-to-end 2-stage curriculum runs with substrate persistence
# ---------------------------------------------------------------------------


class TestCurriculumEndToEnd:
    def test_two_stage_runs_to_completion_and_chains_substrate(self, isolated_sim_reports, tmp_path):
        """Both stages run, both write substrate snapshots, and the
        result identifies the final session dir.
        """
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "curriculum.yaml",
            "name: roy-0-smoke\n"
            "embodiment: bodies/infant_humanoid\n"
            "aut_mode: substrate-primary\n"
            "stages:\n"
            "  - name: act1_warmup\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n"
            "  - name: act2_main\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 5\n",
        )

        runner = _make_fake_sim_runner(growth_per_call=4)
        result = run_curriculum(spec, sim_runner=runner)

        assert isinstance(result, CurriculumResult)
        assert result.aborted_at is None
        assert len(result.stages) == 2
        assert all(isinstance(s, StageResult) for s in result.stages)
        assert all(s.error is None for s in result.stages)

        # Both stages produced session dirs; final pointers are populated.
        s1, s2 = result.stages
        assert s1.session_id and s2.session_id
        assert result.final_session_id == s2.session_id
        assert result.final_session_dir == s2.session_dir
        assert Path(result.final_session_dir).is_dir()

        # The fixture path was resolved to absolute.
        assert s1.source_value == str(fixture.resolve())
        assert s2.source_value == str(fixture.resolve())

        # Substrate snapshots exist on disk for the final stage.
        assert (Path(s2.session_dir) / "aut_nac.json").exists()
        assert (Path(s2.session_dir) / "aut_hippocampus.json").exists()

    def test_each_stage_receives_correct_kwargs(self, isolated_sim_reports, tmp_path):
        """Verifies the runner's kwarg construction: top-level defaults
        propagate to every stage, per-stage overrides win, and stages
        2+ get ``resume_session`` set to the previous stage's id.
        """
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "curriculum.yaml",
            "name: kwargs-check\n"
            "embodiment: bodies/infant_humanoid\n"
            "aut_mode: substrate-primary\n"
            "persona: neutral\n"  # legacy key — pins the pre-1.1 alias read
            "stages:\n"
            "  - name: act1\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n"
            "  - name: act2\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 7\n"
            "    persona: adversarial\n"  # per-stage override
            "    aut_mode: llm-primary\n",
        )

        runner = _make_fake_sim_runner()
        result = run_curriculum(spec, sim_runner=runner)
        assert result.aborted_at is None

        call1, call2 = runner.call_log  # type: ignore[attr-defined]
        kw1, kw2 = call1["kwargs"], call2["kwargs"]

        # Stage 1: top-level defaults; no resume.
        assert kw1["entity_ref"] == "bodies/infant_humanoid"
        assert kw1["aut_mode"] == "substrate-primary"
        assert kw1["mode"] == "neutral"  # read via the legacy persona: alias
        assert kw1["max_turns"] == 3
        assert kw1["fixture_path"] == str(fixture.resolve())
        assert "resume_session" not in kw1

        # Stage 2: per-stage overrides win; resume_session chains stage 1's id.
        assert kw2["entity_ref"] == "bodies/infant_humanoid"
        assert kw2["aut_mode"] == "llm-primary"  # overridden
        assert kw2["mode"] == "adversarial"  # overridden (legacy alias)
        assert kw2["max_turns"] == 7
        assert kw2["resume_session"] == call1["session_id"]

    def test_arc_stage_routes_through_generative(self, isolated_sim_reports, tmp_path):
        """``arc:`` stages should pass ``goal=arc_name`` and
        ``generative=True`` so the orchestrator's existing
        ``select_arc_for_goal`` lookup picks them up.
        """
        spec = _write_yaml(
            tmp_path / "curriculum.yaml",
            "name: arc-curr\n"
            "embodiment: bodies/infant_humanoid\n"
            "stages:\n"
            "  - name: pre\n"
            "    arc: cradle_prelinguistic\n"
            "    turns: 2\n",
        )

        runner = _make_fake_sim_runner()
        result = run_curriculum(spec, sim_runner=runner)
        assert result.aborted_at is None

        kw = runner.call_log[0]["kwargs"]  # type: ignore[attr-defined]
        assert kw["goal"] == "cradle_prelinguistic"
        assert kw["generative"] is True


# ---------------------------------------------------------------------------
# 2. Stage 2 starts with non-empty NAc (proves the resume path)
# ---------------------------------------------------------------------------


class TestSubstrateChain:
    def test_stage2_inherits_stage1_substrate_growth(self, isolated_sim_reports, tmp_path):
        """The fake sim_runner reads stage 1's persisted ``aut_nac.json``
        when ``resume_session`` is set, so stage 2's link count strictly
        exceeds stage 1's. This proves the curriculum runner is
        plumbing ``resume_session`` correctly — without it, the fake
        starts from zero and growth is identical across stages.
        """
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "curriculum.yaml",
            "name: chain-test\n"
            "embodiment: bodies/infant_humanoid\n"
            "aut_mode: substrate-primary\n"
            "stages:\n"
            "  - name: stage1\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n"
            "  - name: stage2\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n",
        )

        runner = _make_fake_sim_runner(growth_per_call=4)
        result = run_curriculum(spec, sim_runner=runner)
        assert result.aborted_at is None

        # Read the on-disk snapshots — stage 2 should hold strictly more
        # links than stage 1 because the fake added growth_per_call on
        # top of stage 1's count.
        s1_nac = json.loads((Path(result.stages[0].session_dir) / "aut_nac.json").read_text())
        s2_nac = json.loads((Path(result.stages[1].session_dir) / "aut_nac.json").read_text())

        assert s1_nac["link_count"] == 4
        assert s2_nac["link_count"] == 8
        assert s2_nac["resumed_from"] == result.stages[0].session_id

        # The final substrate path is identifiable from CurriculumResult.
        assert result.final_session_id == result.stages[1].session_id
        assert Path(result.final_session_dir).name == result.final_session_id


# ---------------------------------------------------------------------------
# 3. Failure surfaces cleanly without orphaning sessions
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_clean_runtime_abort_stops_chain_and_marks_stage_failed(self, isolated_sim_reports, tmp_path):
        """A typed D13 abort must not become the substrate for stage 2."""
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "curriculum.yaml",
            "name: typed-abort\n"
            "stages:\n"
            "  - name: stage1\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n"
            "  - name: stage2\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n",
        )

        runner = _make_fake_sim_runner(finish_reason_on_call=(1, "planning_failed"))
        result = run_curriculum(spec, sim_runner=runner)

        assert result.aborted_at == "stage1"
        assert len(runner.call_log) == 1  # type: ignore[attr-defined]
        assert result.stages[0].finish_reason == "planning_failed"
        assert "unusable simulation result" in (result.stages[0].error or "")
        assert result.final_session_id == ""

    def test_stage1_exception_aborts_chain_cleanly(self, isolated_sim_reports, tmp_path):
        """A sim_runner exception on stage 1 records ``aborted_at`` and
        prevents stage 2 from running — no orphan substrate from a
        half-built stage 2 should appear on disk.
        """
        from maxim.utils.paths import sim_reports as _sim_reports

        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "curriculum.yaml",
            "name: fail-test\n"
            "stages:\n"
            "  - name: stage1\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n"
            "  - name: stage2\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n",
        )

        runner = _make_fake_sim_runner(fail_on_call=1)
        result = run_curriculum(spec, sim_runner=runner)

        # Chain aborted at stage1; stage2 was never invoked.
        assert result.aborted_at == "stage1"
        assert len(result.stages) == 1
        assert result.stages[0].error is not None
        assert "fake sim runner raised" in result.stages[0].error
        assert result.stages[0].finish_reason == "error"
        assert result.stages[0].session_id == ""

        # No usable final pointer (nothing succeeded).
        assert result.final_session_id == ""
        assert result.final_session_dir == ""

        # Only the failed call was made; no on-disk state was created.
        assert len(runner.call_log) == 1  # type: ignore[attr-defined]
        assert not any(_sim_reports().iterdir()), (
            "Stage 2 should not have left orphan session dirs after stage 1 failed"
        )

    def test_stage2_exception_keeps_stage1_substrate_as_final(self, isolated_sim_reports, tmp_path):
        """When stage 2 fails after stage 1 succeeded, the result's
        final_session pointers should still reference stage 1 — that's
        the most-evolved substrate the operator can salvage.
        """
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "curriculum.yaml",
            "name: fail2\n"
            "stages:\n"
            "  - name: ok\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n"
            "  - name: boom\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n",
        )

        runner = _make_fake_sim_runner(fail_on_call=2)
        result = run_curriculum(spec, sim_runner=runner)

        assert result.aborted_at == "boom"
        assert len(result.stages) == 2
        assert result.stages[0].error is None
        assert result.stages[1].error is not None
        assert result.final_session_id == result.stages[0].session_id
        assert Path(result.final_session_dir).is_dir()

    def test_empty_session_id_aborts_chain(self, isolated_sim_reports, tmp_path):
        """If a stage returns ``session_id=""`` we refuse to chain into
        the next stage with a missing resume id — that would silently
        break the chain (orchestrator would treat ``""`` as fresh).
        """
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "curriculum.yaml",
            "name: empty-sess\n"
            "stages:\n"
            "  - name: stage1\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n"
            "  - name: stage2\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 3\n",
        )

        runner = _make_fake_sim_runner(return_empty_session_on_call=1)
        result = run_curriculum(spec, sim_runner=runner)

        assert result.aborted_at == "stage1"
        assert len(result.stages) == 1
        assert "empty session_id" in (result.stages[0].error or "")


# ---------------------------------------------------------------------------
# 4. Spec validation — fail-fast on malformed YAML
# ---------------------------------------------------------------------------


class TestSpecValidation:
    def test_missing_stages_rejected(self, tmp_path):
        spec = _write_yaml(tmp_path / "bad.yaml", "name: x\n")
        with pytest.raises(ValueError, match="non-empty list"):
            run_curriculum(spec, sim_runner=_make_fake_sim_runner())

    def test_missing_name_rejected(self, tmp_path):
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "stages:\n  - name: a\n    arc: cradle\n    turns: 1\n",
        )
        with pytest.raises(ValueError, match="'name'"):
            run_curriculum(spec, sim_runner=_make_fake_sim_runner())

    def test_stage_with_two_sources_rejected(self, tmp_path):
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\n"
            "stages:\n"
            "  - name: a\n"
            f"    fixture: {fixture.name}\n"
            "    arc: cradle_prelinguistic\n"
            "    turns: 1\n",
        )
        with pytest.raises(ValueError, match="exactly one of"):
            run_curriculum(spec, sim_runner=_make_fake_sim_runner())

    def test_stage_with_zero_sources_rejected(self, tmp_path):
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\nstages:\n  - name: a\n    turns: 1\n",
        )
        with pytest.raises(ValueError, match="exactly one of"):
            run_curriculum(spec, sim_runner=_make_fake_sim_runner())

    def test_missing_fixture_rejected(self, tmp_path):
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\nstages:\n  - name: a\n    fixture: nope.yaml\n    turns: 1\n",
        )
        with pytest.raises(ValueError, match="fixture not found"):
            run_curriculum(spec, sim_runner=_make_fake_sim_runner())

    def test_invalid_aut_mode_rejected(self, tmp_path):
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\n"
            "aut_mode: hybrid\n"  # not a real mode
            "stages:\n"
            "  - name: a\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 1\n",
        )
        with pytest.raises(ValueError, match="aut_mode"):
            run_curriculum(spec, sim_runner=_make_fake_sim_runner())

    def test_duplicate_stage_names_rejected(self, tmp_path):
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            "name: x\n"
            "stages:\n"
            "  - name: same\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 1\n"
            "  - name: same\n"
            f"    fixture: {fixture.name}\n"
            "    turns: 1\n",
        )
        with pytest.raises(ValueError, match="Duplicate stage name"):
            run_curriculum(spec, sim_runner=_make_fake_sim_runner())

    def test_nonpositive_turns_rejected(self, tmp_path):
        fixture = _smoke_fixture(tmp_path)
        spec = _write_yaml(
            tmp_path / "bad.yaml",
            f"name: x\nstages:\n  - name: a\n    fixture: {fixture.name}\n    turns: 0\n",
        )
        with pytest.raises(ValueError, match="positive int"):
            run_curriculum(spec, sim_runner=_make_fake_sim_runner())


# ---------------------------------------------------------------------------
# 5. Bundled scenarios/cradle/roy_0_smoke.yaml round-trips through the parser
# ---------------------------------------------------------------------------


class TestBundledSmokeCurriculum:
    def test_bundled_smoke_curriculum_parses_and_runs(self, isolated_sim_reports):
        """The committed ``scenarios/cradle/roy_0_smoke.yaml`` smoke
        curriculum loads, validates, and runs to completion against the
        fake sim runner. Catches drift between the parser and the
        bundled example.
        """
        repo_root = Path(__file__).resolve().parents[2]
        spec = repo_root / "scenarios" / "cradle" / "roy_0_smoke.yaml"
        assert spec.exists(), f"bundled smoke curriculum missing: {spec}"

        runner = _make_fake_sim_runner()
        result = run_curriculum(spec, sim_runner=runner)
        assert result.aborted_at is None
        assert len(result.stages) == 2
        assert all(s.source_kind == "fixture" for s in result.stages)
