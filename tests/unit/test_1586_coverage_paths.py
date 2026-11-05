"""#1586: the router-build and campaign-load helpers that ``run()``, ``start_simulation_mode`` and
``start_research_mode`` call, exercised on their own paths: the no-backend error, the global-config fallback and the
campaign YAML load (its turns, its status line and its failure warning)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from maxim.exceptions import ConfigurationError


class _FakeRouter:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.warmed = False

    def warmup(self) -> None:
        self.warmed = True


# --- api._build_run_router -------------------------------------------------------------------------------------------


def test_build_run_router_raises_when_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_build(*, profile: str | None = None, **_kw: Any) -> tuple[None, None]:
        seen["profile"] = profile
        return None, None

    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", fake_build)
    from maxim.api import _build_run_router

    with pytest.raises(ConfigurationError, match=r"No LLM backend available for model='m-1586'"):
        _build_run_router("m-1586")
    assert seen["profile"] == "m-1586"


def test_build_run_router_returns_the_built_router(monkeypatch: pytest.MonkeyPatch) -> None:
    router = object()
    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", lambda **kw: (router, None))
    from maxim.api import _build_run_router

    assert _build_run_router("m-1586") is router


# --- orchestrator._build_shared_router -------------------------------------------------------------------------------


def _patch_shared_router_deps(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", lambda **kw: (None, None))

    def fake_load(*, profile_override: str | None = None) -> SimpleNamespace:
        seen["profile_override"] = profile_override
        return SimpleNamespace(enabled=enabled)

    monkeypatch.setattr("maxim.models.language.router.load_llm_config", fake_load)
    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _FakeRouter)
    return seen


def test_shared_router_falls_back_to_the_global_config_on_the_run_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_shared_router_deps(monkeypatch, enabled=True)
    from maxim.simulation.orchestrator import _build_shared_router

    router = _build_shared_router("p-1586")
    assert isinstance(router, _FakeRouter)
    assert router.warmed
    assert router.config.enabled
    assert seen["profile_override"] == "p-1586"


def test_shared_router_is_none_when_the_fallback_config_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_shared_router_deps(monkeypatch, enabled=False)
    from maxim.simulation.orchestrator import _build_shared_router

    assert _build_shared_router(None) is None
    assert seen["profile_override"] is None


# --- research_orchestrator._load_campaign_turns ----------------------------------------------------------------------


def test_load_campaign_turns_reads_percepts_with_text(monkeypatch: pytest.MonkeyPatch) -> None:
    percepts = [
        {
            "cli_input": "  first turn \n",
            "metadata": {"phase": "p1", "experiment_role": "probe", "scenario_tag": "t1"},
            "salience": 0.5,
            "novelty": 0.2,
        },
        {"cli_input": ""},  # no text: skipped
        {"other": "x"},  # no cli_input: skipped
        {"cli_input": "second"},  # defaults
    ]
    loaded: dict[str, Any] = {}

    def fake_load(path: Any) -> SimpleNamespace:
        loaded["path"] = str(path)
        return SimpleNamespace(percepts=percepts)

    statuses: list[str] = []
    monkeypatch.setattr("maxim.simulation.scenario_source.load_scenario", fake_load)
    monkeypatch.setattr("maxim.simulation.sim_logger.display_status", statuses.append)
    from maxim.simulation.research_orchestrator import _load_campaign_turns

    turns = _load_campaign_turns("camp.yaml")
    assert loaded["path"] == "camp.yaml"
    assert turns == [
        {"text": "first turn", "phase": "p1", "role": "probe", "tag": "t1", "salience": 0.5, "novelty": 0.2},
        {"text": "second", "phase": "", "role": "", "tag": "", "salience": 0.8, "novelty": 0.7},
    ]
    assert statuses == ["Loaded 2 campaign turns from camp.yaml"]


def test_load_campaign_turns_warns_and_returns_empty_on_a_bad_file(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(path: Any) -> None:
        raise ValueError("bad yaml 1586")

    monkeypatch.setattr("maxim.simulation.scenario_source.load_scenario", boom)
    from maxim.simulation.research_orchestrator import _load_campaign_turns

    with caplog.at_level(logging.WARNING, logger="maxim.simulation.research_orchestrator"):
        assert _load_campaign_turns("missing.yaml") == []
    assert any(
        "Failed to load campaign YAML" in r.getMessage() and "bad yaml 1586" in r.getMessage() for r in caplog.records
    )


def test_load_campaign_turns_without_a_campaign_loads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def never(path: Any) -> None:
        raise AssertionError("load_scenario must not be called without a campaign")

    monkeypatch.setattr("maxim.simulation.scenario_source.load_scenario", never)
    from maxim.simulation.research_orchestrator import _load_campaign_turns

    assert _load_campaign_turns(None) == []
