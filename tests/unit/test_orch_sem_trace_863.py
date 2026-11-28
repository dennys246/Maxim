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
  ``__init__`` has returned; nothing reads it between construction and the trace). The trace sits in the
  construction ``try``'s ``else:`` (a review fold), so the fault raises; it is not reported as a construction failure.
- ``sem_trace_imagination_failed``: the f-string ``{e}`` over the construction failure (the trigger's ``__init__``
  raises an exception whose ``__format__`` raises; its ``__str__``, which the handler's ``logger.warning`` uses,
  does not).

Before the fix each fault is swallowed and the run completes, so each test failed (its red gate, 22b126b8). Each
test also asserts the raise comes FROM its site's trace statement (``_raised_at``), so a later read of the faulted
object elsewhere cannot satisfy it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.unit._orch_harness import run_arm


class _SiteFault(Exception):
    """The injected caller-side fault (nothing else in a run raises it)."""


def _assert_sim_logging_restored() -> None:
    """The run raised in setup, before any teardown: the harness must still close the sim log it opened (#1839)."""
    from maxim.simulation import sim_logger as SL

    assert (SL._sim_active, SL._log_file) == (False, None)


def _raised_at(excinfo: pytest.ExceptionInfo[Any], marker: str) -> bool:
    """Whether a frame of the raise is the statement carrying ``marker`` (the site's own trace line)."""
    return any(marker in str(e.statement) for e in excinfo.traceback)


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
    with pytest.raises(_SiteFault, match="sem_trace_tools") as excinfo:
        run_arm("imagination", tmp_path)
    assert fired == [True]
    assert _raised_at(excinfo, "Entities:"), excinfo.traceback
    _assert_sim_logging_restored()


def test_a_fault_building_the_imagination_active_trace_surfaces_863(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    with pytest.raises(_SiteFault, match="sem_trace_imagination_active") as excinfo:
        run_arm("imagination", tmp_path)
    assert state["fired"] == 1
    assert _raised_at(excinfo, "ImaginationTrigger ACTIVE"), excinfo.traceback
    _assert_sim_logging_restored()


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
    with pytest.raises(_SiteFault, match="sem_trace_imagination_failed") as excinfo:
        run_arm("imagination", tmp_path)
    assert _raised_at(excinfo, "ImaginationTrigger FAILED"), excinfo.traceback
    _assert_sim_logging_restored()
