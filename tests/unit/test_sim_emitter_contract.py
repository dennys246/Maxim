"""The telemetry emitters never raise into their caller (#863, step 1).

Before this contract 31 of the 33 ``sim_*`` emitters leaked an exception on a bad argument or a
failing render, so 99 call sites wrapped them defensively — and 60 of those wraps also enclosed the
caller's own logic, which is how ``NAc.predict`` silently lost every prediction from the sim log
(#861). The guarantee now lives in ONE place, the emitters themselves, so call sites can drop their
``try`` and let their own mistakes surface.

Emitters are DISCOVERED, never listed: a new ``sim_*`` added without the contract fails here, not in
a caller months later.
"""

from __future__ import annotations

import inspect

import pytest

import maxim.simulation.sim_logger as sl

_NOT_EMITTERS = {"sim_agent_context"}  # a context manager for agent attribution, not an emitter


def _emitters() -> list[str]:
    return sorted(
        name
        for name, obj in vars(sl).items()
        if name.startswith("sim_")
        and name not in _NOT_EMITTERS
        and inspect.isfunction(obj)
        and getattr(obj, "__module__", None) == sl.__name__
    )


class _Hostile:
    """Every way an emitter could touch an argument raises."""

    def _boom(self, *args, **kwargs):
        raise RuntimeError("hostile argument")

    __str__ = __repr__ = __format__ = __len__ = __iter__ = __getitem__ = _boom
    __bool__ = __float__ = __int__ = __round__ = __lt__ = __gt__ = __eq__ = __hash__ = _boom

    def __getattr__(self, name):
        raise RuntimeError(f"hostile attribute {name}")


def _hostile_call(fn):
    kwargs = {
        p.name: _Hostile()
        for p in inspect.signature(fn).parameters.values()
        if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    return fn(**kwargs)


def test_discovery_found_the_emitter_surface():
    """The discovery is only as good as what it finds; a broken filter would pass everything."""
    found = _emitters()
    assert len(found) >= 30, found
    assert {"sim_log", "sim_nac_predict", "sim_pain", "sim_learn"} <= set(found)


@pytest.mark.parametrize("name", _emitters())
def test_every_emitter_carries_the_contract(name):
    assert getattr(getattr(sl, name), "__maxim_contained__", False), (
        f"{name} is a public emitter without @_contained — a caller would have to guard it again"
    )


@pytest.mark.parametrize("name", _emitters())
def test_hostile_arguments_never_escape_an_emitter(name, monkeypatch):
    """The behavioural half. Before the contract, 31 of these raised."""
    reported: list[str] = []
    monkeypatch.setattr(sl, "log_swallowed_exception", lambda e=None, **kw: reported.append(kw.get("operation")))
    _hostile_call(getattr(sl, name))  # must not raise


def test_a_contained_failure_is_REPORTED_not_dropped(monkeypatch):
    """Containment that loses the failure would just move the silence one frame down."""
    reported: list[str] = []
    monkeypatch.setattr(sl, "log_swallowed_exception", lambda e=None, **kw: reported.append(kw.get("operation")))
    _hostile_call(sl.sim_nac_predict)
    assert reported == ["sim_emit:sim_nac_predict"], reported


def test_a_failing_terminal_render_never_escapes_sim_log(monkeypatch):
    """The other leak: sim_log's own render path, exercised with a live sim and a broken display."""

    class _BrokenDisplay:
        def log(self, *args, **kwargs):
            raise OSError("terminal went away")

    monkeypatch.setattr(sl, "_sim_active", True)
    monkeypatch.setattr(sl, "_active_display", _BrokenDisplay())
    monkeypatch.setattr(sl, "log_swallowed_exception", lambda e=None, **kw: None)
    sl.sim_log("NAc", "a perfectly ordinary event")  # must not raise


def test_containment_does_not_swallow_success(monkeypatch):
    """Anti-vacuity: an emitter that silently dropped everything would pass every arm above."""
    seen: list[dict] = []
    sl.register_sim_sink(seen.append)
    try:
        sl.sim_learn("a new causal link", detail="rpe=+0.5")
    finally:
        sl.unregister_sim_sink(seen.append)
    assert any("a new causal link" in str(r) for r in seen), seen


def test_interrupts_still_propagate():
    """Only Exception is contained. Swallowing Ctrl-C inside a logger would be a new, worse bug."""

    @sl._contained
    def _interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _interrupted()
