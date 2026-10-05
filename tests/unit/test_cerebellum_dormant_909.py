"""#909: the Cerebellum's read side has no production caller and says so.

Forward-model TRAINING is live (``embodiment/tool_bridge.py`` -> ``observe_from_action``) and, since #908,
persisted. Everything that READS it -- prediction, engram formation and recall, program crystallization,
the ``CerebellumModulator`` backend -- has no non-test caller in ``src/`` or ``scripts/``. Each carries a
``Dormant since`` docstring (CLAUDE.md: dormancy over deletion). This scan fails the moment one gains a
caller, so the PR that wires it must revisit the marker and engram_formation.md E7 in the same change.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]  # THIS checkout, never an installed shadow

# Called name -> files allowed to call it: only other dormant code (the chain below a dormant entry point).
DORMANT_CALLS: dict[str, set[str]] = {
    "query_engrams": set(),
    "form_engram": set(),
    "observe_action_sequence": set(),
    "cleanup_program": set(),
    # Cerebellum.observe_action_sequence delegates here; it is dormant itself.
    "observe_sequence": {"src/maxim/embodiment/cerebellum.py"},
    "cerebellum_modulator_factory": set(),
    # The dormant factory builds it.
    "CerebellumModulator": {"src/maxim/embodiment/backends/cerebellum_modulator.py"},
}


def _calls() -> list[tuple[str, int, str, str]]:
    """Every call in src/ + scripts/: (file, line, called name, receiver source or '')."""
    found = []
    for path in [*(REPO / "src").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]:
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)  # f(...) and obj.f(...)
            receiver = (ast.get_source_segment(text, func.value) or "") if isinstance(func, ast.Attribute) else ""
            found.append((str(path.relative_to(REPO)), node.lineno, name or "", receiver))
    return found


@pytest.fixture(scope="module")
def calls() -> list[tuple[str, int, str, str]]:
    return _calls()


@pytest.mark.parametrize("name", sorted(DORMANT_CALLS))
def test_a_dormant_symbol_has_no_production_caller(calls, name) -> None:
    callers = [f"{f}:{line}" for f, line, called, _ in calls if called == name and f not in DORMANT_CALLS[name]]
    assert not callers, f"{name} gained a caller: revisit its Dormant marker and engram_formation.md E7"


def test_cerebellum_predict_has_no_production_caller(calls) -> None:
    """``predict`` is a common name (NAc, harm predictors), so this one is judged by its receiver. A blind spot:
    a Cerebellum bound to a name without "cereb" in it would not be seen."""
    callers = [
        f"{f}:{line}"
        for f, line, called, receiver in calls
        if called == "predict"
        and "cereb" in receiver.lower()
        and f != "src/maxim/embodiment/backends/cerebellum_modulator.py"
    ]
    assert not callers, callers


def test_the_receiver_filter_sees_the_dormant_modulator_predict(calls) -> None:
    """Known answer for the receiver filter: the dormant backend's own ``self._cerebellum.predict`` call is
    seen (and then excused), so an empty receiver could not let the predict scan pass vacuously."""
    assert any(
        f == "src/maxim/embodiment/backends/cerebellum_modulator.py" and called == "predict" and "cereb" in r.lower()
        for f, _, called, r in calls
    )


def test_the_scan_sees_the_live_training_call(calls) -> None:
    """Known answer: the scan must find the one live Cerebellum call, or it proves nothing."""
    assert any(called == "observe_from_action" and f.startswith("src/") for f, _, called, _ in calls)


@pytest.mark.parametrize(
    ("obj", "attr"),
    [
        ("maxim.embodiment.cerebellum:Cerebellum", "predict"),
        ("maxim.embodiment.cerebellum:Cerebellum", "query_engrams"),
        ("maxim.embodiment.cerebellum:Cerebellum", "observe_action_sequence"),
        ("maxim.embodiment.cerebellum:Cerebellum", "form_engram"),
        ("maxim.embodiment.cerebellum:Cerebellum", "cleanup_program"),
        ("maxim.embodiment.backends.cerebellum_modulator", "CerebellumModulator"),
        ("maxim", "embodiment.engrams"),
        ("maxim.embodiment.motor:ProgramRegistry", "observe_sequence"),
        ("maxim.embodiment.backends.cerebellum_modulator", "cerebellum_modulator_factory"),
    ],
)
def test_each_dormant_symbol_says_so(obj, attr) -> None:
    import importlib

    module, _, cls = obj.partition(":")
    if module == "maxim":  # a whole module
        target = importlib.import_module(f"maxim.{attr}")
    else:
        owner = importlib.import_module(module)
        target = getattr(getattr(owner, cls) if cls else owner, attr)
    assert "Dormant since" in (target.__doc__ or ""), f"{obj}.{attr}"
