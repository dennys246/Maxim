"""#863 step 2 at the three orchestrator SEM_TRACE sites (``scripts/release_gates.json`` items ``863.orch.*``).

Each site wrapped its ``sim_log`` emits in ``try: ... except Exception: pass``. Since #868 every ``sim_*`` emitter
is contained (it never raises), so the wrap protected nothing but the CALLER: a fault while building the trace's
arguments was swallowed in silence. With the wraps dropped, such a fault surfaces.

Each test drives the real ``start_simulation_mode`` through the characterization harness's ``imagination`` arm
(an embodied AUT, so all three sites are on the path) and injects a ONE-SHOT fault into exactly one site's
argument construction, raising ``_SiteFault``, which nothing else in the run raises:

- ``sem_trace_tools``: the trace's first read of an entity's ``name`` (``EntityMap.list_entities`` is armed only
  after ``mark_goal_selected``, the last call before the trace, so the tool selection above it reads real entities).
- ``sem_trace_imagination_active``: the trace's read of ``ImaginationTrigger._enabled`` (armed once the trigger's
  ``__init__`` has returned; nothing reads it between construction and the trace). That site sits inside the
  trigger's construction ``try``, so the fault surfaces as that handler's WARNING, not as a raise.
- ``sem_trace_imagination_failed``: the f-string ``{e}`` over the construction failure (the trigger's ``__init__``
  raises an exception whose ``__format__`` raises; its ``__str__``, which the handler's ``logger.warning`` uses,
  does not).

Before the fix each fault is swallowed and the run completes, so each test failed (its red gate).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from tests.unit._orch_harness import run_arm


class _SiteFault(Exception):
    """The injected caller-side fault (nothing else in a run raises it)."""


def test_a_fault_building_the_sem_tool_trace_surfaces_863(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.tools.discovery as discovery
    from maxim.embodiment.entity_map import EntityMap

    orig_mark = discovery.mark_goal_selected
    orig_list = EntityMap.list_entities
    fired: list[bool] = []

    class _Entity:
        @property
        def name(self) -> str:
            raise _SiteFault("sem_trace_tools: entity name")

    def list_once(self: Any) -> list[Any]:
        if fired:
            return orig_list(self)
        fired.append(True)
        return [_Entity()]

    def mark(*a: Any, **k: Any) -> Any:
        out = orig_mark(*a, **k)
        monkeypatch.setattr(EntityMap, "list_entities", list_once)
        return out

    monkeypatch.setattr(discovery, "mark_goal_selected", mark)
    with pytest.raises(_SiteFault, match="sem_trace_tools"):
        run_arm("imagination", tmp_path)
    assert fired == [True]


def test_a_fault_building_the_imagination_active_trace_surfaces_863(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from maxim.imagination.trigger import ImaginationTrigger

    state = {"armed": False, "fired": 0}

    class _OneShotEnabled:
        def __get__(self, obj: Any, typ: Any = None) -> Any:
            if obj is None:
                return self
            if state["armed"]:
                state["armed"] = False
                state["fired"] += 1
                raise _SiteFault("sem_trace_imagination_active: _enabled")
            return obj.__dict__["_enabled_863"]

        def __set__(self, obj: Any, value: Any) -> None:
            obj.__dict__["_enabled_863"] = value

    orig_init = ImaginationTrigger.__init__

    def init(self: Any, *a: Any, **k: Any) -> None:
        orig_init(self, *a, **k)
        state["armed"] = True

    monkeypatch.setattr(ImaginationTrigger, "_enabled", _OneShotEnabled(), raising=False)
    monkeypatch.setattr(ImaginationTrigger, "__init__", init)
    with caplog.at_level(logging.WARNING, logger="maxim.simulation.orchestrator"):
        run_arm("imagination", tmp_path)
    assert state["fired"] == 1
    assert any(
        r.name == "maxim.simulation.orchestrator"
        and r.getMessage().startswith("ImaginationTrigger construction failed")
        and "sem_trace_imagination_active" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_fault_building_the_imagination_failed_trace_surfaces_863(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maxim.imagination.trigger import ImaginationTrigger

    class _Unformattable(Exception):
        def __str__(self) -> str:
            return "construction refused (#863 probe)"

        def __format__(self, spec: str) -> str:
            raise _SiteFault("sem_trace_imagination_failed: {e}")

    def init(self: Any, *a: Any, **k: Any) -> None:
        raise _Unformattable()

    monkeypatch.setattr(ImaginationTrigger, "__init__", init)
    with pytest.raises(_SiteFault, match="sem_trace_imagination_failed"):
        run_arm("imagination", tmp_path)
