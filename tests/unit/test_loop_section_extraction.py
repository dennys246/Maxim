"""Direct tests for the extracted numbered sections of ``run_agentic_loop``.

docs/plans/god_function_decomposition.md — the first extraction (1.1.2 Cluster
B). The point of the plan is auditability: a behavioural delta emitted from a
3,500-line function cannot be checked for contamination. A section that has
been moved out but is still only reachable by running the whole loop has not
bought that, so these tests call the extracted functions DIRECTLY.

What each test pins is the behaviour that was inline before the move, not a
new contract:

* section 7 must stay total — a host callback that raises must not escape into
  the loop;
* section 8.5 must run all six NAc decays, in order, every tick, and must
  tolerate a None NAc (the memory_hub-less loop).

The order assertion is deliberate. The plan calls section order load-bearing,
and the six decays inside 8.5 are a single unit under one handler: if an early
call raises, the later ones are skipped for that tick. That is pre-existing
behaviour, pinned here so a later semantic fix has to be a deliberate,
reviewed change rather than a silent one.
"""

from __future__ import annotations

from typing import Any

import pytest

from maxim.runtime.agent_loop import _loop_bio_tick_maintenance, _loop_step_callback


class _RecordingNAc:
    """Records the decay calls the per-tick maintenance makes, in order."""

    def __init__(self, raise_on: str | None = None) -> None:
        self.calls: list[str] = []
        self._raise_on = raise_on

    def _record(self, name: str) -> None:
        self.calls.append(name)
        if name == self._raise_on:
            raise RuntimeError(f"boom in {name}")

    def decay_eligibility(self) -> None:
        self._record("decay_eligibility")

    def decay_reward_biases(self) -> None:
        self._record("decay_reward_biases")

    def decay_goal_reward_biases(self) -> None:
        self._record("decay_goal_reward_biases")

    def decay_cluster_reward_biases(self) -> None:
        self._record("decay_cluster_reward_biases")

    def decay_percept_valences(self) -> None:
        self._record("decay_percept_valences")

    def decay_exploration_visits(self) -> None:
        self._record("decay_exploration_visits")


EXPECTED_DECAYS = [
    "decay_eligibility",
    "decay_reward_biases",
    "decay_goal_reward_biases",
    "decay_cluster_reward_biases",
    "decay_percept_valences",
    "decay_exploration_visits",
]


# ── Section 8.5 — BIO-SYSTEM PER-TICK MAINTENANCE ────────────────────────────


def test_bio_tick_runs_all_six_decays_in_order():
    nac = _RecordingNAc()
    _loop_bio_tick_maintenance(nac)
    assert nac.calls == EXPECTED_DECAYS


def test_bio_tick_is_a_noop_without_a_nac():
    """The memory_hub-less loop passes None; that must not raise."""
    _loop_bio_tick_maintenance(None)


def test_bio_tick_contains_a_raising_decay():
    """A decay blowing up must not escape into the loop body."""
    nac = _RecordingNAc(raise_on="decay_reward_biases")
    _loop_bio_tick_maintenance(nac)
    assert nac.calls == ["decay_eligibility", "decay_reward_biases"]


def test_bio_tick_decays_are_all_called_every_tick():
    """Three ticks => three full passes. Traces must not decay only once."""
    nac = _RecordingNAc()
    for _ in range(3):
        _loop_bio_tick_maintenance(nac)
    assert nac.calls == EXPECTED_DECAYS * 3


# ── Section 7 — CALL STEP CALLBACK ───────────────────────────────────────────


class _Level:
    def __init__(self, value: str) -> None:
        self.value = value


class _Autonomy:
    def __init__(self, value: str = "supervised") -> None:
        self.current_level = _Level(value)


class _Ctrl:
    """The fields section 7 reads off the LoopController context."""

    def __init__(self, on_step: Any, *, pending_proposal: Any = None) -> None:
        self.on_step = on_step
        self.state = object()
        self.memory = object()
        self.autonomy_controller = _Autonomy()
        self.pending_proposal = pending_proposal


def test_step_callback_passes_the_snapshot():
    seen: list[dict[str, Any]] = []
    ctrl = _Ctrl(seen.append, pending_proposal=object())
    _loop_step_callback(ctrl, step_num=7)

    assert len(seen) == 1
    payload = seen[0]
    assert payload["step"] == 7
    assert payload["state"] is ctrl.state
    assert payload["memory"] is ctrl.memory
    assert payload["autonomy_level"] == "supervised"
    assert payload["ctrl.pending_proposal"] is True


def test_step_callback_reports_absent_proposal_as_false():
    seen: list[dict[str, Any]] = []
    _loop_step_callback(_Ctrl(seen.append, pending_proposal=None), step_num=1)
    assert seen[0]["ctrl.pending_proposal"] is False


def test_step_callback_survives_a_raising_host_callback():
    """A host's callback must never take down the agent loop."""

    def explode(_payload: dict[str, Any]) -> None:
        raise RuntimeError("host callback blew up")

    _loop_step_callback(_Ctrl(explode), step_num=2)


def test_step_callback_tolerates_a_non_callable():
    _loop_step_callback(_Ctrl(None), step_num=3)
    _loop_step_callback(_Ctrl("not callable"), step_num=4)


# ── The extraction itself ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["_loop_step_callback", "_loop_bio_tick_maintenance"])
def test_sections_are_module_level(name):
    """The score card's condition is a MODULE-LEVEL function, callable without
    constructing a loop. If these become closures again, the extraction is
    undone and this fails."""
    import maxim.runtime.agent_loop as agent_loop

    fn = getattr(agent_loop, name)
    assert callable(fn)
    assert fn.__qualname__ == name, f"{name} is nested, not module-level: {fn.__qualname__}"
