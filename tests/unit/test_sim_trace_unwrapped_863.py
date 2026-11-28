"""#863 step 2 across ``src/maxim/simulation/``: no broad silent ``try`` wraps a contained ``sim_*`` emitter.

Since #868 every emitter ``sim_logger`` marks ``@_contained`` reports its own faults and never raises. So a
``try: <emit> except Exception: pass`` around one can only swallow the CALLER's fault while building the emit's
arguments. The structural test below holds the package at zero such sites. It is the guard for the whole class:
at c3269d82 it finds nine (the three orchestrator sites the 1.4.0 gate names, ``test_orch_sem_trace_863.py``, plus
``bridge.py``, ``fixture_orchestrator.py``, ``generative_runner.py`` and three in ``tools.py``).

The behavioural tests then prove the drop is not vacuous at the sites that a unit test reaches cheaply: each
hands the tool an argument whose ``__format__`` (or ``__add__``) raises ``_SiteFault``, which only the trace's
f-string evaluates, and asserts the raise surfaces FROM the trace statement.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SIM = _ROOT / "src" / "maxim" / "simulation"


class _SiteFault(Exception):
    """The injected caller-side fault."""


def _contained_emitters() -> set[str]:
    tree = ast.parse((_SIM / "sim_logger.py").read_text(encoding="utf-8"))
    return {
        f.name
        for f in tree.body
        if isinstance(f, ast.FunctionDef)
        and any(isinstance(d, ast.Name) and d.id == "_contained" for d in f.decorator_list)
    }


def _broad_silent(h: ast.ExceptHandler) -> bool:
    broad = h.type is None or (isinstance(h.type, ast.Name) and h.type.id in ("Exception", "BaseException"))
    return broad and all(isinstance(s, (ast.Pass, ast.Continue)) for s in h.body)


def _emits(stmts: list[ast.stmt], names: set[str]) -> list[ast.Call]:
    out = []
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Call):
                f = n.func
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                if name in names:
                    out.append(n)
    return out


def _silent_emitter_wraps(root: Path) -> list[str]:
    """Each broad try with an all-silent handler that is the INNERMOST try owning a contained-emitter call."""
    names = _contained_emitters()
    hits = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Try) and any(_broad_silent(h) for h in node.handlers)):
                continue
            nested = {
                id(c)
                for s in node.body
                for inner in ast.walk(s)
                if isinstance(inner, ast.Try)
                for c in _emits(inner.body, names)
            }
            if any(id(c) not in nested for c in _emits(node.body, names)):
                hits.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    return hits


def test_the_emitter_set_is_read_from_sim_logger() -> None:
    names = _contained_emitters()
    assert {"sim_log", "sim_action", "sim_percept"} <= names, names


def test_no_silent_try_wraps_a_contained_emitter_in_simulation() -> None:
    assert _silent_emitter_wraps(_SIM) == []


def _raised_at(excinfo: pytest.ExceptionInfo[Any], marker: str) -> bool:
    return any(marker in str(e.statement) for e in excinfo.traceback)


class _Src(str):
    """A ``source`` that behaves as a str everywhere except in an f-string hole."""

    def __format__(self, spec: str) -> str:
        raise _SiteFault("trace argument")


@pytest.fixture
def _frozen_body_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import maxim.embodiment.body as body_module

    monkeypatch.setattr(body_module, "time", types.SimpleNamespace(time=lambda: 1_000_000.0))


class _Bus:
    def __init__(self) -> None:
        self.published: list[Any] = []

    def publish(self, signal: Any) -> None:
        self.published.append(signal)


def test_a_fault_building_the_damage_trace_surfaces_863(_frozen_body_clock: None) -> None:
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.simulation.tools import DamageComponentTool

    root = ComponentRegistry().instantiate("bodies/base_humanoid")
    part = next(n for n, m in root.modulators.items() if hasattr(m, "apply_damage") and m.vital_metrics)
    tool = DamageComponentTool(embodiment=Embodiment(root, pain_bus=_Bus()), entity_map=None)
    with pytest.raises(_SiteFault) as excinfo:
        tool.execute(component=part, amount=0.1, source=_Src("probe"))
    assert _raised_at(excinfo, "component damage:"), excinfo.traceback


def test_a_fault_building_the_sensor_trace_surfaces_863(_frozen_body_clock: None) -> None:
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.simulation.tools import SetEntitySensorTool

    root = ComponentRegistry().instantiate("bodies/infant_humanoid")
    tool = SetEntitySensorTool(embodiment=Embodiment(root, pain_bus=_Bus()), entity_map=None)
    with pytest.raises(_SiteFault) as excinfo:
        tool.execute(sensor="arms.thermal", value=0.8, source=_Src("probe"))
    assert _raised_at(excinfo, "sensor set:"), excinfo.traceback


def test_a_fault_building_the_bridge_entry_trace_surfaces_863() -> None:
    from maxim.simulation.bridge import SimulationBridge

    class _Turn(int):
        def __add__(self, other: Any) -> Any:
            raise _SiteFault("turn count")

    bridge = object.__new__(SimulationBridge)
    bridge.action_sink = SimpleNamespace(actions=[])
    bridge._turn_count = _Turn(0)
    with pytest.raises(_SiteFault) as excinfo:
        bridge._send_and_wait("hello", timeout=None, settle_s=None, salience=0.8, novelty=0.7)
    assert _raised_at(excinfo, "Bridge.send_and_wait ENTER"), excinfo.traceback


def test_the_mother_tick_trace_names_a_missing_telemetry_key_863() -> None:
    from maxim.simulation.generative_runner import _log_mother_tick

    with pytest.raises(KeyError, match="fed"):
        _log_mother_tick("act1", {"credited": 0})
