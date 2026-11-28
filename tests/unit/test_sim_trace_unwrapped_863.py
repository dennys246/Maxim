"""#863 step 2: no broad silent ``try`` swallows a contained ``sim_*`` emit, in ``src/maxim/`` (ratcheted).

Since #868 every emitter ``sim_logger`` marks ``@_contained`` reports its own faults and never raises. So a broad
``try`` whose handler is silent (``pass``/``continue``/``...``) around such an emit can only swallow the CALLER's
fault while building the emit's arguments. The structural test walks every emit up through its enclosing ``try``s
to the first one with a broad handler (an inner narrow handler does not catch a broad fault) and counts it when that
handler is silent. ``maxim.simulation`` is held at zero: at c3269d82 it had nine (the three orchestrator sites the
1.4.0 gate names, ``test_orch_sem_trace_863.py``, plus ``bridge.py``, ``fixture_orchestrator.py``,
``generative_runner.py`` and three in ``tools.py``), all dropped in #1838. Every other package is pinned per function
in ``_REMAINING`` (#863's burn-down outside ``simulation/``), which may only shrink: a new site fails, and a fixed
one must be removed from the pin.

The behavioural tests then prove the drop is not vacuous at the sites that a unit test reaches cheaply: each
hands the tool an argument whose ``__format__`` (or ``__add__``) raises ``_SiteFault``, which only the trace's
f-string evaluates, and asserts the raise surfaces FROM the trace statement.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "maxim"
_SIM = _SRC / "simulation"

# #863's remaining sites outside ``simulation/`` (2026-11-27, at #1838): (file, enclosing function) -> count. Shrink
# only; each fix removes its entry here in the same commit. Deferred because: too-large:~26 sites, about 130 src lines
# with per-site classification (several sit under outer explicit-form handlers, as the 863.loop.* sites do).
_REMAINING: dict[tuple[str, str], int] = {
    ("src/maxim/agents/llm_worker.py", "LLMWorker._process_request"): 1,
    ("src/maxim/agents/llm_worker.py", "_sim_log_proposal_built"): 1,
    ("src/maxim/agents/llm_worker.py", "_sim_log_raw_response"): 1,
    ("src/maxim/agents/memory_agent.py", "MemoryAgent._complete_forming_memory"): 1,
    ("src/maxim/agents/percept_factory.py", "_log_percept"): 1,
    ("src/maxim/embodiment/cerebellum.py", "Cerebellum.predict"): 1,
    ("src/maxim/imagination/trigger.py", "ImaginationTrigger._encode_entity_affordances"): 2,
    ("src/maxim/imagination/trigger.py", "ImaginationTrigger._resolve_phrase"): 2,
    ("src/maxim/imagination/trigger.py", "ImaginationTrigger.process_manifest"): 3,
    ("src/maxim/imagination/trigger.py", "ImaginationTrigger.process_percept"): 2,
    ("src/maxim/memory/concept_extractor.py", "ConceptExtractor._register_concept"): 2,
    ("src/maxim/memory/hippocampus_retrieval.py", "RetrievalMixin.recall"): 1,
    ("src/maxim/proprioception/perceived_pain.py", "PerceivedPainAssessor.assess"): 1,
    ("src/maxim/proprioception/perceived_pain.py", "PerceivedPainAssessor.assess_text"): 1,
    ("src/maxim/runtime/pain_interceptor.py", "PainInterceptorExecutor.execute"): 1,
    ("src/maxim/runtime/thought_gate.py", "ThoughtGate._log_decision"): 1,
    ("src/maxim/runtime/worker_pool.py", "JobRegistry.mark_completed"): 1,
    ("src/maxim/tools/discovery.py", "SenseToolsTool.execute"): 3,
}


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


def _broad(t: ast.expr | None) -> bool:
    if t is None:
        return True
    if isinstance(t, ast.Name):
        return t.id in ("Exception", "BaseException")
    return isinstance(t, ast.Tuple) and any(_broad(e) for e in t.elts)


def _silent(h: ast.ExceptHandler) -> bool:
    return all(
        isinstance(s, (ast.Pass, ast.Continue))
        or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis)
        for s in h.body
    )


def _silent_emitter_wraps(scope: Path, root: Path = _ROOT) -> Counter[tuple[str, str]]:
    """(file, enclosing function) -> the emits whose first broad-catching enclosing ``try`` has a silent handler.

    The emitter names are ``sim_logger``'s ``@_contained`` functions plus any alias a module imports one under. The
    walk stops at the enclosing function (a caller's ``try`` is out of reach of a static per-function scan), and an
    emit inside a handler, ``else`` or ``finally`` is not protected by that ``try``. Blind spots: a relative
    ``from .sim_logger import ...`` alias, ``try``/``except*`` (``ast.TryStar``), and emits through a local wrapper
    (none in ``src/maxim/`` at #1838)."""
    contained = _contained_emitters()
    hits: Counter[tuple[str, str]] = Counter()
    for path in sorted(scope.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parent = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        names = set(contained)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module == "maxim.simulation.sim_logger":
                names |= {a.asname or a.name for a in n.names if a.name in contained}
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            if (f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None) not in names:
                continue
            child: ast.AST = call
            cur = parent.get(call)
            swallowed = False
            while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if isinstance(cur, ast.Try) and child in cur.body:
                    catchers = [h for h in cur.handlers if _broad(h.type)]
                    if catchers:
                        swallowed = _silent(catchers[0])
                        break
                child, cur = cur, parent.get(cur)
            if not swallowed:
                continue
            qual = []
            node = cur
            while node is not None:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qual.append(node.name)
                node = parent.get(node)
            hits[(str(path.relative_to(root)), ".".join(reversed(qual)) or "<module>")] += 1
    return hits


def test_the_emitter_set_is_read_from_sim_logger() -> None:
    names = _contained_emitters()
    assert {"sim_log", "sim_action", "sim_percept"} <= names, names


def test_no_silent_try_wraps_a_contained_emitter_in_simulation() -> None:
    assert dict(_silent_emitter_wraps(_SIM)) == {}


def test_outside_simulation_the_silent_wraps_only_shrink() -> None:
    now = dict(_silent_emitter_wraps(_SRC))
    grown = {k: v for k, v in now.items() if v > _REMAINING.get(k, 0)}
    assert grown == {}, f"a new silent try around a contained emit (#863): {grown}"
    stale = {k: v for k, v in _REMAINING.items() if now.get(k, 0) < v}
    assert stale == {}, f"fixed: lower or remove these _REMAINING pins in the same commit: {stale}"


@pytest.mark.parametrize(
    "src",
    [
        "try:\n    sim_log('X', f'{a}')\nexcept (ValueError, Exception):\n    pass\n",
        "try:\n    sim_log('X', f'{a}')\nexcept Exception:\n    ...\n",
        "from maxim.simulation.sim_logger import sim_log as _sl\ntry:\n    _sl('X', f'{a}')\nexcept Exception:\n    pass\n",
        "try:\n    try:\n        sim_log('X', f'{a}')\n    except ValueError:\n        raise\nexcept Exception:\n    pass\n",
    ],
    ids=["tuple-handler", "ellipsis-body", "aliased-import", "narrow-inner-under-silent-outer"],
)
def test_the_scan_sees_the_shapes_that_hide_a_wrap(tmp_path: Path, src: str) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("def f(a):\n" + "".join("    " + line + "\n" for line in src.splitlines()))
    assert sum(_silent_emitter_wraps(tmp_path / "pkg", tmp_path).values()) == 1


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
