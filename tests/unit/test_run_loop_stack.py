"""``bootstrap.run_loop_stack``: the one sanctioned ``run_agentic_loop`` caller (docs/plans/loop_stack_builder.md).

Pins the classification of every loop parameter, the required-keyword TypeError for each STACK keyword, the three
checks (the hub both ways, the coach, embodiment, and the ``llm_worker=None`` rule), that nothing is mutated, and that
every argument reaches the loop unchanged.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from maxim.runtime import agent_loop, bootstrap
from maxim.runtime.bootstrap import LOOP_PARAMETERS, run_loop_stack

_EXTRA = ("acting_coach", "is_embodied")  # the two worker statements: run_loop_stack keywords, not loop parameters


def _stack_kwargs(**over: Any) -> dict[str, Any]:
    hub = over.pop("_hub", None)
    worker = SimpleNamespace(acting_coach=None, is_embodied=False)
    kw: dict[str, Any] = {name: object() for name, kind in LOOP_PARAMETERS.items() if kind == "stack"}
    kw.update(
        agent=SimpleNamespace(memory_hub=hub),
        memory_hub=hub,
        llm_worker=worker,
        aut_mode="llm-primary",
        consolidation="full",
        acting_coach=None,
        is_embodied=False,
    )
    kw.update(over)
    return kw


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(agent_loop, "run_agentic_loop", lambda *a, **k: calls.append((a, k)))
    return calls


def test_the_classification_covers_the_loops_signature_exactly() -> None:
    params = inspect.signature(agent_loop.run_agentic_loop).parameters
    assert set(LOOP_PARAMETERS) == set(params), (
        f"unclassified: {sorted(set(params) - set(LOOP_PARAMETERS))}; "
        f"stale: {sorted(set(LOOP_PARAMETERS) - set(params))}"
    )
    assert set(LOOP_PARAMETERS.values()) == {"stack", "run"}
    assert sum(v == "stack" for v in LOOP_PARAMETERS.values()) == 27
    assert sum(v == "run" for v in LOOP_PARAMETERS.values()) == 12


def test_the_signature_is_keyword_only_with_stack_required_and_run_defaulted_as_the_loop_defaults() -> None:
    mine = inspect.signature(run_loop_stack).parameters
    loop = inspect.signature(agent_loop.run_agentic_loop).parameters
    assert set(mine) == set(LOOP_PARAMETERS) | set(_EXTRA)
    for name, p in mine.items():
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, f"{name} is not keyword-only"
        if name in _EXTRA or LOOP_PARAMETERS[name] == "stack":
            assert p.default is inspect.Parameter.empty, f"STACK keyword {name} has a default"
        else:
            assert p.default == loop[name].default, f"RUN keyword {name}'s default differs from the loop's"
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in mine.values()), "a ** catch-all reopens the hole"


@pytest.mark.parametrize("name", sorted([n for n, k in LOOP_PARAMETERS.items() if k == "stack"] + list(_EXTRA)))
def test_omitting_any_stack_keyword_is_a_type_error(name: str, recorded: list[Any]) -> None:
    kw = _stack_kwargs()
    kw.pop(name)
    with pytest.raises(TypeError, match=name):
        run_loop_stack(**kw)
    assert recorded == []


def test_every_argument_reaches_the_loop_unchanged(recorded: list[Any]) -> None:
    kw = _stack_kwargs(max_steps=7, run_id="r", target_hz=2.0, end_bound=object(), planning_liveness=True)
    run_loop_stack(**kw)
    [(args, got)] = recorded
    assert args == ()  # every argument by keyword, the six positionals too (review round, wire SF)
    for name in LOOP_PARAMETERS:
        expected = kw.get(name, inspect.signature(run_loop_stack).parameters[name].default)
        assert got[name] is expected or got[name] == expected, name
    assert set(got) == set(LOOP_PARAMETERS)  # the worker statements are not forwarded


def test_a_keyword_only_loop_stub_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """``test_api_core``'s ``def run_agentic_loop(**kwargs)`` and the atonia lambda take keywords only: a migrated site
    must still reach them."""
    seen: list[dict[str, Any]] = []

    def run_agentic_loop(**kwargs: Any) -> None:
        seen.append(kwargs)

    monkeypatch.setattr(agent_loop, "run_agentic_loop", run_agentic_loop)
    kw = _stack_kwargs()
    run_loop_stack(**kw)
    [got] = seen
    assert got["agent"] is kw["agent"] and got["executor"] is kw["executor"]
    assert set(got) == set(LOOP_PARAMETERS)


def test_loop_parameters_is_typed_with_the_two_literals() -> None:
    # the stored annotation string is unparsed, so Python normalizes its quotes
    assert bootstrap.__annotations__["LOOP_PARAMETERS"].replace("'", '"') == 'dict[str, Literal["stack", "run"]]'


def test_a_hub_handed_to_the_loop_but_not_wired_raises(recorded: list[Any]) -> None:
    kw = _stack_kwargs()
    kw["memory_hub"] = object()  # the agent holds None
    with pytest.raises(ValueError, match="memory hub"):
        run_loop_stack(**kw)
    assert recorded == []


def test_a_wired_agent_handed_no_hub_raises(recorded: list[Any]) -> None:
    hub = object()
    kw = _stack_kwargs(_hub=hub)
    kw["memory_hub"] = None
    with pytest.raises(ValueError, match="memory hub"):
        run_loop_stack(**kw)
    assert recorded == []


def test_a_different_hub_raises_and_the_same_hub_passes(recorded: list[Any]) -> None:
    hub = object()
    kw = _stack_kwargs(_hub=hub)
    kw["memory_hub"] = object()
    with pytest.raises(ValueError, match="memory hub"):
        run_loop_stack(**kw)
    run_loop_stack(**_stack_kwargs(_hub=hub))
    assert len(recorded) == 1


def test_a_coach_the_worker_does_not_hold_raises(recorded: list[Any]) -> None:
    coach = object()
    kw = _stack_kwargs(acting_coach=coach)  # the worker's coach is None
    with pytest.raises(ValueError, match="acting_coach"):
        run_loop_stack(**kw)
    kw = _stack_kwargs()
    kw["llm_worker"] = SimpleNamespace(acting_coach=coach, is_embodied=False)
    with pytest.raises(ValueError, match="acting_coach"):
        run_loop_stack(**kw)  # stated None, the worker holds one
    kw["acting_coach"] = coach
    run_loop_stack(**kw)
    assert len(recorded) == 1


def test_the_coach_is_compared_by_identity_not_equality(recorded: list[Any]) -> None:
    class _Eq:
        def __eq__(self, other: object) -> bool:
            return True

        __hash__ = object.__hash__

    kw = _stack_kwargs(acting_coach=_Eq())
    kw["llm_worker"] = SimpleNamespace(acting_coach=_Eq(), is_embodied=False)
    with pytest.raises(ValueError, match="acting_coach"):
        run_loop_stack(**kw)


def test_an_embodiment_the_worker_does_not_say_raises(recorded: list[Any]) -> None:
    with pytest.raises(ValueError, match="is_embodied"):
        run_loop_stack(**_stack_kwargs(is_embodied=True))
    kw = _stack_kwargs(is_embodied=True)
    kw["llm_worker"] = SimpleNamespace(acting_coach=None, is_embodied=True)
    run_loop_stack(**kw)
    assert len(recorded) == 1


def test_without_a_worker_the_call_must_say_no_coach_and_not_embodied(recorded: list[Any]) -> None:
    for over in ({"acting_coach": object()}, {"is_embodied": True}):
        kw = _stack_kwargs(**over)
        kw["llm_worker"] = None
        with pytest.raises(ValueError, match="llm_worker=None"):
            run_loop_stack(**kw)
    kw = _stack_kwargs()
    kw["llm_worker"] = None
    run_loop_stack(**kw)
    assert len(recorded) == 1


def test_a_bare_mock_worker_fails_loudly(recorded: list[Any]) -> None:
    from unittest.mock import MagicMock

    kw = _stack_kwargs()
    kw["llm_worker"] = MagicMock()
    with pytest.raises(ValueError):
        run_loop_stack(**kw)
    kw["llm_worker"] = SimpleNamespace()
    with pytest.raises(AttributeError):
        run_loop_stack(**kw)
    assert recorded == []


def test_nothing_is_mutated(recorded: list[Any]) -> None:
    hub = object()
    kw = _stack_kwargs(_hub=hub)
    worker, agent = kw["llm_worker"], kw["agent"]
    before = (dict(vars(worker)), dict(vars(agent)))
    run_loop_stack(**kw)
    assert (dict(vars(worker)), dict(vars(agent))) == before


def test_the_loop_is_looked_up_when_called_not_bound_at_import() -> None:
    assert "run_agentic_loop" not in vars(bootstrap), "bootstrap binds the loop at import: a patch would miss it"


def test_the_agents_memory_hub_property_reads_the_wired_hub_and_is_read_only() -> None:
    from maxim.agents.maxim_agent import MaximAgent

    prop = inspect.getattr_static(MaximAgent, "memory_hub")
    assert isinstance(prop, property) and prop.fset is None
    fake = SimpleNamespace(memory=SimpleNamespace(_memory_hub=None))
    assert prop.fget(fake) is None
    hub = object()
    fake.memory._memory_hub = hub
    assert prop.fget(fake) is hub
