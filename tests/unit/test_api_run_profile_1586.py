"""#1586: ``run()``, ``imagine()`` and ``campaign()`` hand their model to the run, never through ``MAXIM_LLM_PROFILE``.

Each wrote ``MAXIM_LLM_PROFILE = model`` for the whole run, so every other thread resolving a profile saw this run's
model, and ``imagine()``/``campaign()`` wrote it before their ``try``: a scenario or campaign that failed to load left
it set. The model now rides ``build_primary_router(profile=)`` (#1538) and ``start_simulation_mode(llm_profile=)``,
carried into a ``/new`` run. ``MAXIM_LLM_ENABLED`` keeps its setdefault, scoped to the run and restored on every exit
(owner decision 2026-11-03).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

OPERATOR = "operator-model"
RUN_MODEL = "run-model-1586"


class _Stop(Exception):
    """Ends a run at the build under test."""


@pytest.fixture
def operator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator's profile in the env (as ``maxim config`` projects it), MAXIM_LLM_ENABLED unset; the model check
    and the persisted-choice lookup are out of scope here."""
    monkeypatch.setenv("MAXIM_LLM_PROFILE", OPERATOR)
    monkeypatch.delenv("MAXIM_LLM_ENABLED", raising=False)
    monkeypatch.setattr("maxim.api._resolve_model", lambda model: model)
    monkeypatch.setattr("maxim.api._validate_model", lambda model: None)


def _env() -> dict[str, str | None]:
    import os

    return {k: os.environ.get(k) for k in ("MAXIM_LLM_PROFILE", "MAXIM_LLM_ENABLED")}


def _campaign_yaml(tmp_path: Path) -> str:
    path = tmp_path / "c.yaml"
    path.write_text(
        textwrap.dedent("""\
        campaign:
          name: p1586
          goal: test
          seed: 1
        acts:
          - name: act1
            encounters: [enc1]
        encounters:
          enc1:
            scene: "Test."
            choices: [go]
            branches: { go: __END__ }
    """)
    )
    return str(path)


def _recording_sim(seen: list[dict[str, Any]]) -> Any:
    def _sim(**kwargs: Any) -> Any:
        seen.append({"llm_profile": kwargs.get("llm_profile"), "env": _env()})
        raise _Stop

    return _sim


def test_imagine_hands_its_model_to_the_run_and_the_env_keeps_the_operators(
    operator_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maxim.api import imagine

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", _recording_sim(seen))
    with pytest.raises(_Stop):
        imagine("g", model=RUN_MODEL)
    assert seen == [{"llm_profile": RUN_MODEL, "env": {"MAXIM_LLM_PROFILE": OPERATOR, "MAXIM_LLM_ENABLED": "1"}}]
    assert _env() == {"MAXIM_LLM_PROFILE": OPERATOR, "MAXIM_LLM_ENABLED": None}


def test_campaign_hands_its_model_to_the_run_and_the_env_keeps_the_operators(
    operator_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from maxim.api import campaign

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", _recording_sim(seen))
    with pytest.raises(_Stop):
        campaign(_campaign_yaml(tmp_path), model=RUN_MODEL)
    assert seen == [{"llm_profile": RUN_MODEL, "env": {"MAXIM_LLM_PROFILE": OPERATOR, "MAXIM_LLM_ENABLED": "1"}}]
    assert _env() == {"MAXIM_LLM_PROFILE": OPERATOR, "MAXIM_LLM_ENABLED": None}


def test_a_scenario_that_fails_to_load_leaves_the_env_as_it_was(
    operator_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from maxim.api import imagine

    def _bad(_path: Any) -> Any:
        raise ValueError("bad scenario")

    monkeypatch.setattr("maxim.simulation.scenario_source.load_scenario", _bad)
    with pytest.raises(ValueError, match="bad scenario"):
        imagine("g", model=RUN_MODEL, scenario=str(tmp_path / "s.yaml"))
    assert _env() == {"MAXIM_LLM_PROFILE": OPERATOR, "MAXIM_LLM_ENABLED": None}


def test_a_campaign_that_fails_to_load_leaves_the_env_as_it_was(
    operator_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from maxim.api import campaign

    def _bad(*_a: Any, **_k: Any) -> Any:
        raise ValueError("bad campaign")

    monkeypatch.setattr("maxim.simulation.dm_schema.load_campaign", _bad)
    with pytest.raises(ValueError, match="bad campaign"):
        campaign(str(tmp_path / "c.yaml"), model=RUN_MODEL)
    assert _env() == {"MAXIM_LLM_PROFILE": OPERATOR, "MAXIM_LLM_ENABLED": None}


def test_run_hands_its_model_to_the_router_build_and_the_env_keeps_the_operators(
    operator_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from maxim.api import run

    seen: list[dict[str, Any]] = []

    def _build(*_a: Any, **kwargs: Any) -> Any:
        seen.append({"profile": kwargs.get("profile"), "env": _env()})
        raise _Stop

    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", _build)
    with pytest.raises(_Stop):
        run(model=RUN_MODEL, goal="g", home_dir=str(tmp_path))
    assert seen == [{"profile": RUN_MODEL, "env": {"MAXIM_LLM_PROFILE": OPERATOR, "MAXIM_LLM_ENABLED": "1"}}]
    assert _env() == {"MAXIM_LLM_PROFILE": OPERATOR, "MAXIM_LLM_ENABLED": None}


def test_the_run_profile_reaches_the_shared_build_and_the_config_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """``start_simulation_mode(llm_profile=)`` hands it to the shared router build and, when that build gives no
    router, to the fallback ``load_llm_config``; the ``/new`` run carries it."""
    from maxim.simulation import orchestrator

    builds: list[dict[str, Any]] = []
    configs: list[Any] = []

    def _build(*_a: Any, **kwargs: Any) -> Any:
        builds.append(kwargs.get("profile"))
        return None, None

    def _config(*_a: Any, **kwargs: Any) -> Any:
        configs.append(kwargs.get("profile_override"))
        raise _Stop

    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", _build)
    monkeypatch.setattr("maxim.models.language.router.load_llm_config", _config)
    with pytest.raises(_Stop):
        orchestrator._build_shared_router(RUN_MODEL)
    assert builds == [RUN_MODEL] and configs == [RUN_MODEL]
    assert "llm_profile" in orchestrator.NEW_RUN_CARRIES


def test_the_api_entry_points_no_longer_write_the_session_profile() -> None:
    from tests.unit.test_aut_router_profile_1538 import _profile_env_writers

    assert not {w for w in _profile_env_writers() if w.startswith("maxim/api.py::")}


def test_the_run_profile_drives_the_real_sim_build(tmp_path: Path) -> None:
    """Through the real ``start_simulation_mode`` (the harness's ``run_profile`` arm)."""
    from tests.unit._orch_harness import OPERATOR_PROFILE_ENV, assert_this_checkout, run_arm

    assert_this_checkout()
    trace = run_arm("run_profile", tmp_path)
    operator = OPERATOR_PROFILE_ENV["MAXIM_LLM_PROFILE"]
    assert trace["router_builds"] == [{"profile": "orch-run-model", "env": operator}]
    assert trace["maxim_env_after"].get("MAXIM_LLM_PROFILE") == operator
    assert trace["report"]["provenance"]["language_profile"] == "orch-run-model"


def test_research_hands_its_model_to_its_router_and_its_sim(
    operator_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``maxim.research(model=)`` reported ``orchestrator_model=<model>`` while its Writer, Reviewer and narrator ran on
    the operator's profile (review fold, owner decision 2026-11-03)."""
    from maxim.api import research

    builds: list[Any] = []
    seen: list[dict[str, Any]] = []

    def _build(*_a: Any, **kwargs: Any) -> Any:
        builds.append(kwargs.get("profile"))
        return None, None

    def _config(*_a: Any, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        builds.append(("config", kwargs.get("profile_override")))
        return SimpleNamespace(enabled=False)

    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", _build)
    monkeypatch.setattr("maxim.models.language.router.load_llm_config", _config)
    monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", _recording_sim(seen))
    with pytest.raises(_Stop):
        research("q", model=RUN_MODEL)
    assert builds == [RUN_MODEL, ("config", RUN_MODEL)]
    assert seen == [{"llm_profile": RUN_MODEL, "env": {"MAXIM_LLM_PROFILE": OPERATOR, "MAXIM_LLM_ENABLED": None}}]


@pytest.mark.parametrize("persisted", [None, "persisted-1586"])
def test_research_with_no_model_resolves_it_like_the_other_verbs(
    monkeypatch: pytest.MonkeyPatch, persisted: str | None
) -> None:
    """No ``model=``: the last chosen one, else ``mistral-7b``; the session's ``llm.profile`` does not decide (owner
    decision 2026-11-03: research matches run/imagine/campaign)."""
    from maxim import api

    monkeypatch.setenv("MAXIM_LLM_PROFILE", OPERATOR)
    monkeypatch.setattr("maxim.runtime.lane_backends._read_persisted_model", lambda: persisted)
    builds: list[Any] = []

    def _build(*_a: Any, **kwargs: Any) -> Any:
        builds.append(kwargs.get("profile"))
        raise _Stop

    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", _build)
    monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", _recording_sim([]))
    with pytest.raises(_Stop):
        api.research("q")
    assert builds == [persisted or api._API_DEFAULT_MODEL]
