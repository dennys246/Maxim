"""World clock A-pre: its red gates and WC5's characterization guards (docs/plans/world_clock_a_pre.md §Gates).

PR 0 lands what today's surface can express:

- **G-cost-month** (#1769, landed in PR 0, flipped in PR 1a): a ``CostTracker`` on the wall keeps its month, windows
  and EMAs whatever ``timestamp`` a caller passes.
- **WC5, robots unchanged** (green): today's persisted Hippocampus and NAc files load and save back unchanged; the
  prompt's date and time enter only through ``maxim.agents.prompt_builder.datetime``; and the shipped dispatch seam
  books no energy cost. That last pin replaces the planned G-tool-duration gate, which today's code could only pass
  vacuously (owner decision 2026-11-24; see ``test_wc5_execute_and_learn_books_no_energy_cost_today``).

Each gate is ``xfail(strict=True, raises=AssertionError)``: an ``ImportError``, ``TypeError`` or ``NameError`` is never
the expected failure. The facts a gate relies on (both records are priced) are asserted in a separate green test, so a
broken setup cannot pass as the expected red; a precondition of that kind is how PR 0 found the energy booking has no
caller. Fake wall clocks ADVANCE on every read, because a frozen clock hides cadence bugs. The one exception is the WC5
store round trips, which pin ``time.time`` to the fixture's own ``saved_at`` so the re-save can be compared byte for
byte (their content reads no other clock). Everything that writes state uses ``tmp_path``.
"""

from __future__ import annotations

import datetime as _dt
import functools
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "world_clock_wc5"
T0 = 1_764_000_000.25  # 2025-11-24 UTC, with unusual sub-seconds


class _AdvancingWall:
    """A ``time.time`` that advances ``step`` seconds per read, plus any ``jump`` a test adds."""

    def __init__(self, start: float, step: float = 0.001) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now

    def jump(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def wall(monkeypatch: pytest.MonkeyPatch) -> _AdvancingWall:
    fake = _AdvancingWall(T0)
    monkeypatch.setattr(time, "time", fake)
    return fake


# ── G-cost-month (#1769) ─────────────────────────────────────────────────────

_PRICED = "wc-model"


def _tracker(state_dir: Path) -> Any:
    from maxim.models.language.cost_tracker import CostTracker, CostTrackerConfig, ModelPricing

    state_dir.mkdir(parents=True, exist_ok=True)
    config = CostTrackerConfig(state_path=str(state_dir / "cost_state.json"))
    return CostTracker({_PRICED: ModelPricing(input_price=3.0, output_price=0.0)}, config)


def _record_wall_then_epoch(tracker: Any) -> tuple[float, float]:
    """One record on the wall, one with a caller ``timestamp`` of 0.0: the costs booked."""
    first = tracker.record(provider="cloud", model=_PRICED, input_tokens=1_000_000, output_tokens=0)
    second = tracker.record(provider="cloud", model=_PRICED, input_tokens=1_000_000, output_tokens=0, timestamp=0.0)
    return first, second


def test_cost_month_gate_setup_prices_both_records(tmp_path: Path, wall: _AdvancingWall) -> None:
    """G-cost-month's precondition: both records are priced at 3.0, so the gate's totals are meaningful."""
    assert _record_wall_then_epoch(_tracker(tmp_path / "wall")) == (3.0, 3.0)


# Flipped in PR 1a (#1769, D-WC5): the whole tracker runs on the wall, and a caller clock is ignored.
def test_g_cost_month_a_caller_timestamp_cannot_move_the_tracker_off_the_wall(
    tmp_path: Path, wall: _AdvancingWall
) -> None:
    # Window pruning is destructive, so the caller-clock reads go to their own unpruned tracker, FIRST, with a far
    # future ``now`` that would prune everything if it were honoured (a read at 0.0 prunes nothing and proves nothing).
    far_future = T0 + 40 * 86_400.0
    tracker, probe = _tracker(tmp_path / "wall"), _tracker(tmp_path / "caller")
    _record_wall_then_epoch(tracker)
    _record_wall_then_epoch(probe)
    wall.jump(5.0)
    caller_totals = probe.get_totals(now=far_future)
    caller_provider = probe.get_per_provider_totals("cloud", now=far_future)
    totals = tracker.get_totals()
    assert totals["monthly"] == 6.0, f"the month was reset by a caller timestamp: {totals}"
    assert totals["hourly"] == 6.0 and totals["daily"] == 6.0, f"a record fell out of the rolling windows: {totals}"
    assert tracker.get_per_provider_totals("cloud") == {"hourly": 6.0, "daily": 6.0, "monthly": 6.0}
    assert caller_totals == totals, f"a caller clock reached window pruning: {caller_totals} vs {totals}"
    assert caller_provider == {"hourly": 6.0, "daily": 6.0, "monthly": 6.0}, f"per provider: {caller_provider}"
    for name, rate in tracker.get_spend_rates().items():
        assert rate.last_update >= T0, f"{name} EMA was stamped with the caller's clock: {rate.last_update}"


# ── The tool duration (OD-A4): today it reaches no learning (#1788) ──────────


def _duration_reads(wall: _AdvancingWall, jump_s: float) -> dict[str, list[Any]]:
    """Run one tool through the real ``execute_and_learn`` while the wall jumps ``jump_s`` inside the tool.

    The tool does the same work either way: only the wall jump differs. Returns what NAc was shown of the tool's
    duration: each energy event as (valence, delta_seconds), and the ``delta_seconds`` of each ``tool_result`` link.
    Both consume ``record_outcome``'s ``elapsed_s`` (``_observe_energy_cost`` and ``_observe_tool_result``), so a
    wiring that fed only one of them still changes this result.
    """
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.proprioception.pain_bus import build_pain_bus
    from maxim.runtime import tool_dispatch
    from maxim.runtime.bootstrap import build_executor
    from maxim.runtime.tool_policy import ToolPolicy
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    class _Work(Tool):
        name = "work"
        description = "does a fixed amount of work"
        input_schema: dict = {}

        def execute(self, **kwargs: Any) -> Any:
            ran.append(jump_s)
            wall.jump(jump_s)
            return ToolOutput(success=True, output="done")

    ran: list[float] = []

    nac = NAc(NACConfig())
    seen: list[tuple[str, float]] = []
    tool_deltas: list[float] = []
    original = nac.observe

    def spy(**kwargs: Any) -> Any:
        if kwargs.get("event_type") == "energy":
            seen.append((kwargs["outcome_valence"].value, float(kwargs["delta_seconds"])))
        if kwargs.get("outcome_type") == "tool_result":
            tool_deltas.append(float(kwargs["delta_seconds"]))
        return original(**kwargs)

    nac.observe = spy  # type: ignore[method-assign]
    registry = ToolRegistry()
    registry.register(_Work())
    executor = build_executor(
        registry,
        pain_bus=build_pain_bus(hippocampus=None, nac=nac),
        policy=ToolPolicy.unrestricted(reason="world clock G-tool-duration"),
        nac=nac,
    )
    sim = MagicMock()
    sim.is_sim_mode = False
    proposal = MagicMock(reasoning="do the work", degraded=False, source=None, clusters=None, cluster_id=None)
    recent: list[dict[str, Any]] = []
    outcome = tool_dispatch.execute_and_learn(
        agent=MagicMock(_execution_tracker=None),
        agent_name="wc",
        agent_id="world_clock",
        executor=executor,
        sim=sim,
        state=MagicMock(data={"active_goal": "work"}),
        environment=MagicMock(),
        memory=MagicMock(),
        hippocampus=None,
        memory_hub=MagicMock(),
        result_cache=MagicMock(),
        autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
        rec_outcome=functools.partial(tool_dispatch.record_outcome, drive_relief_only=False),
        recent_outcomes=recent,
        max_recent=10,
        llm_worker=MagicMock(),
        context_pool=MagicMock(),
        nac=nac,
        run_id="world_clock_energy_cost",
        overlay={},
        action={"tool_name": "work", "params": {}},
        confidence=0.9,
        proposal=proposal,
        observation={},
        human_involved=False,
    )
    # An empty result must mean "booked nothing", never "never got there": the tool ran and its outcome was booked.
    assert not outcome.raised, outcome.error
    assert ran == [jump_s], "the tool never ran"
    assert len(recent) == 1 and recent[0].get("success") is True, recent
    return {"energy": seen, "tool_result_delta_s": tool_deltas}


def test_wc5_execute_and_learn_books_no_energy_cost_today(wall: _AdvancingWall) -> None:
    """Characterization (owner decision 2026-11-24): the shipped dispatch seam hands NAc no tool duration at all.

    ``record_outcome``'s ``elapsed_s`` feeds two sinks: the energy booking (``_observe_energy_cost``, NEGATIVE valence
    above 2.0 s) and the ``tool_result`` link's ``delta_seconds`` (persisted ``temporal_delta``, read by
    ``NAc.predict_delay``). No caller passes it: ``execute_and_learn`` measures ``exec_elapsed`` for its logs only
    (true since the booking landed in 54b77b3a). So no energy cost is booked and every tool link learns a 0.0 s
    delay, whatever the wall did, and G-tool-duration could only pass vacuously. Robots keep this (WC5). The PR that
    wires the duration (A-post with B2, fenced ``runtime/tool_dispatch.py``, #1788) replaces this pin with
    G-tool-duration, landed red, in the same PR.
    """
    for jump_s in (1.0, 3.0):
        assert _duration_reads(wall, jump_s) == {"energy": [], "tool_result_delta_s": [0.0]}, jump_s


def test_the_energy_booking_itself_works_when_given_a_duration() -> None:
    """The half that is live: given ``elapsed_s``, ``record_outcome`` books NEUTRAL at or below 2.0 s and NEGATIVE
    above. G-tool-duration's future surface, pinned so wiring the duration cannot also change the rule."""
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.runtime import tool_dispatch

    def valences(elapsed_s: float) -> list[str]:
        nac = NAc(NACConfig())
        seen: list[str] = []
        original = nac.observe

        def spy(**kwargs: Any) -> Any:
            if kwargs.get("event_type") == "energy":
                seen.append(kwargs["outcome_valence"].value)
            return original(**kwargs)

        nac.observe = spy  # type: ignore[method-assign]
        tool_dispatch._observe_energy_cost(nac, "work", elapsed_s, "world_clock")
        return seen

    assert valences(1.0) == ["neutral"]
    assert valences(3.0) == ["negative"]


# ── WC5: robots unchanged (characterization, green) ──────────────────────────


def _sorted_index(raw: bytes, index: str) -> bytes:
    """The file with one id index's lists sorted, everything else as written.

    Both writers keep that index as a ``set`` (``hippocampus_persistence``'s ``context_index``, NAc's
    ``outcome_index``). A store saved before #1789 holds it in PYTHONHASHSEED order, as the fixtures do; since #1789
    the writers sort it, so re-saving an old store sorts that one index and leaves every other byte as it was.
    """
    data = json.loads(raw)
    data[index] = {k: sorted(v) for k, v in data.get(index, {}).items()}
    return json.dumps(data, indent=2).encode()


def _resaved_equals_original(raw: bytes, resaved: bytes, index: str) -> None:
    # The writer's format is plain ``json.dumps(indent=2)``, so re-serialising the original reproduces it byte for
    # byte; ``_sorted_index(raw)`` is then the original with only that index changed.
    assert json.dumps(json.loads(raw), indent=2).encode() == raw.rstrip(b"\n"), "original: the canonical form is lossy"
    # Since #1789 the re-saved file is already sorted: compared as written, with nothing re-sorted on its side.
    assert resaved.rstrip(b"\n") == _sorted_index(raw, index)


def test_wc5_a_hippocampus_home_saved_before_a_pre_loads_and_saves_back_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    original = (FIXTURES / "hippocampus.json").read_bytes()
    monkeypatch.setattr(time, "time", lambda: json.loads(original)["saved_at"])
    hippocampus = Hippocampus(
        HippocampusConfig(persistence_path=None, enable_sleep_consolidation=False, auto_save_after_sleep=False)
    )
    hippocampus.load(str(FIXTURES / "hippocampus.json"))
    hippocampus.save(str(tmp_path / "hippocampus.json"))
    _resaved_equals_original(original, (tmp_path / "hippocampus.json").read_bytes(), "context_index")


def test_wc5_a_nac_store_saved_before_a_pre_loads_and_saves_back_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maxim.decisions.nac import NAc, NACConfig

    original = (FIXTURES / "nac.json").read_bytes()
    nac = NAc(NACConfig())
    nac.load(str(FIXTURES / "nac.json"))
    monkeypatch.setattr(time, "time", lambda: json.loads(original)["saved_at"])
    nac.save(str(tmp_path / "nac.json"))
    _resaved_equals_original(original, (tmp_path / "nac.json").read_bytes(), "outcome_index")


@pytest.fixture
def pinned_tz() -> Iterator[None]:
    """TZ pinned for the prompt characterization (POSIX ``tzset``; elsewhere the patched ``datetime`` carries it)."""
    old = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    yield
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    if hasattr(time, "tzset"):
        time.tzset()


# Two instants that share no date or time text: weekday, month, day, hour, minute and AM/PM all differ.
INSTANT_A = _dt.datetime(2025, 3, 7, 15, 4, 5)
INSTANT_B = _dt.datetime(2026, 8, 19, 9, 41, 27)
# Today's rendered text for INSTANT_A: the A-pre port's host default must reproduce exactly this (WC5).
DATE_A, TIME_A = "Friday, March 07, 2025", "03:04 PM"


def _prompts(monkeypatch: Any, instant: _dt.datetime | None, builder: Any = None) -> dict[str, str]:
    """The three prompt kinds that render the date and time, built with ``prompt_builder.datetime`` patched.

    ``builder`` is a ready ``PromptBuilder`` (G-render passes an ``LLMWorker``'s own); by default one is built here.
    ``instant=None`` leaves ``datetime`` unpatched (G-tz, whose children must read the real, timezone-dependent clock).
    """
    from maxim.agents import prompt_builder
    from maxim.agents.autonomy import AutonomyLevel
    from maxim.agents.bus import StructuredContext
    from maxim.agents.llm_fallback import ReasoningCarryover
    from maxim.agents.llm_types import LLMRequest, ModeInfo
    from maxim.models.language.token_counter import CharEstimateCounter

    class _At(_dt.datetime):
        @classmethod
        def now(cls, tz: Any = None) -> Any:  # noqa: ARG003 - the builder calls the naive form
            return instant

    # PR 2 moves the patch target to the perceived-time port (world_clock_a_pre.md WC5).
    if instant is not None:
        monkeypatch.setattr(prompt_builder, "datetime", _At)
    builder = builder or prompt_builder.PromptBuilder(
        llm=None,
        reasoning_carryover=ReasoningCarryover(),
        n_ctx=32000,
        token_counter=CharEstimateCounter(),
        tool_index=None,
    )

    def request(*, tools: set[str], use_tool_prompting: bool) -> LLMRequest:
        return LLMRequest(
            request_id="wc5",
            context=StructuredContext(timestamp=T0),
            mode=ModeInfo(name="active", goal="assist", context_prompt=""),
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            internet_access=False,
            internet_policy_summary="",
            available_tools=tools,
            tool_descriptions={t: f"the {t} tool" for t in tools},
            use_tool_prompting=use_tool_prompting,
            triggering_input="maxim, what should we do next?",
            deliberation_available=False,
        )

    return {
        "tool_aware": builder.build_prompt(request(tools={"respond", "look"}, use_tool_prompting=True)),
        "answer_only": builder.build_prompt(request(tools=set(), use_tool_prompting=False)),
        "engage": builder._build_engage_prompt("internet_search", "weather", "rain later", "observe"),
    }


def _date_time(instant: _dt.datetime) -> tuple[str, str]:
    return instant.strftime("%A, %B %d, %Y"), instant.strftime("%I:%M %p")


def test_wc5_prompt_date_and_time_render_exactly_as_today(monkeypatch: pytest.MonkeyPatch, pinned_tz: None) -> None:
    assert _date_time(INSTANT_A) == (DATE_A, TIME_A)
    prompts = _prompts(monkeypatch, INSTANT_A)
    assert f"Today is {DATE_A}\nCurrent time: {TIME_A}" in prompts["tool_aware"]
    assert prompts["answer_only"].startswith(f"ANSWER_ONLY|Today is {DATE_A}. The time is {TIME_A}.\n\n")
    assert DATE_A in prompts["engage"] and TIME_A in prompts["engage"]


def test_wc5_no_wall_derived_time_text_reaches_a_prompt_beside_the_patched_datetime(
    monkeypatch: pytest.MonkeyPatch, pinned_tz: None, wall: _AdvancingWall
) -> None:
    """Two instants, with the wall jumped between builds: the prompts differ only in the date and time text.

    Scope, stated: this rules out time text derived from ``time.time`` (the jump moves every clock field), for a
    builder with no cloud cost tracker (``llm=None``, so ``build_budget_context`` renders nothing). It cannot see an
    unpatched ``datetime.now()`` or argument-less ``time.strftime()`` elsewhere, which the two builds, milliseconds
    apart in real time, would render alike; the census shows ``prompt_builder.py``'s only clock sites are its two
    ``datetime.now`` calls. With a cost tracker attached, the budget section reads the wall through ``CostTracker``
    and ``EnergySignal`` (no date text; PR 1's D-WC5 keeps the tracker on the wall). ``request.context_pool_text``
    arrives pre-rendered; ``ContextPool``'s stamp render is PR 2's.
    """

    def masked(instant: _dt.datetime) -> dict[str, str]:
        date_s, time_s = _date_time(instant)
        return {
            k: v.replace(date_s, "<DATE>").replace(time_s, "<TIME>") for k, v in _prompts(monkeypatch, instant).items()
        }

    first = masked(INSTANT_A)
    wall.jump(86_400.0 * 400 + 3_723.5)  # not a whole number of days: every clock field moves
    second = masked(INSTANT_B)
    for kind in first:
        assert "<DATE>" in first[kind] and "<TIME>" in first[kind], f"{kind} renders no date or time"
        assert first[kind] == second[kind], f"{kind} reads time from somewhere other than prompt_builder.datetime"


# ── The two collaborators (PR 1a: time/agent_time.py) ────────────────────────


def test_step_clock_reaches_machinery_default() -> None:
    """I2: the machinery clock's host default reads ``time.*`` at call time, so a clock patched there reaches it.

    ``_StepClock`` patches the ``time`` module's attributes for a run. A default that bound ``time.time`` at import (a
    ``default_factory=time.time`` or ``from time import time``) would keep reading the real clock under it.
    """
    from maxim.time.agent_time import HOST_MACHINERY_CLOCK
    from tests.unit._loop_harness import EPOCH, _StepClock

    clock = _StepClock()
    clock.install()
    try:
        before_s, before_mono = HOST_MACHINERY_CLOCK.now_s(), HOST_MACHINERY_CLOCK.monotonic_s()
        time.sleep(2.5)  # the owner thread: advances the step clock by exactly this much
        after_s, after_mono = HOST_MACHINERY_CLOCK.now_s(), HOST_MACHINERY_CLOCK.monotonic_s()
    finally:
        clock.uninstall()
    assert before_s == EPOCH and after_s == EPOCH + 2.5, (before_s, after_s)
    assert after_mono - before_mono == 2.5, (before_mono, after_mono)


def test_the_host_port_is_todays_local_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    """The port's host default is ``datetime.now()`` / ``datetime.fromtimestamp()`` exactly, patched by name (I2)."""
    from maxim.time import agent_time

    pinned = _dt.datetime(2025, 3, 7, 15, 4, 5)

    class _PinnedDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz: Any = None) -> _dt.datetime:  # type: ignore[override]
            return pinned

    monkeypatch.setattr(agent_time, "datetime", _PinnedDatetime)
    assert agent_time.HOST_PERCEIVED_TIME.now() == pinned
    stamp = 1_741_359_845.75
    assert agent_time.HOST_PERCEIVED_TIME.perceive(stamp) == _dt.datetime.fromtimestamp(stamp)


def test_no_calendar_declares_no_time() -> None:
    from maxim.time.agent_time import NO_CALENDAR

    assert NO_CALENDAR.now() is None
    assert NO_CALENDAR.perceive(T0) is None
    assert NO_CALENDAR.perceive(0.0) is None  # 0.0 is a time (#1684), and still renders nothing


def test_the_collaborators_satisfy_their_protocols() -> None:
    from maxim.time.agent_time import (
        HOST_MACHINERY_CLOCK,
        HOST_PERCEIVED_TIME,
        NO_CALENDAR,
        MachineryClock,
        PerceivedTime,
    )

    assert isinstance(HOST_MACHINERY_CLOCK, MachineryClock)
    assert isinstance(HOST_PERCEIVED_TIME, PerceivedTime) and isinstance(NO_CALENDAR, PerceivedTime)
    assert not isinstance(NO_CALENDAR, MachineryClock), "a calendar port is not a machinery clock"


def test_a_caller_clock_given_to_the_cost_tracker_warns(tmp_path: Path, wall: _AdvancingWall) -> None:
    """D-WC5: the ignored keywords say so, every time, so a caller on another clock learns it is not honoured."""
    tracker = _tracker(tmp_path / "warn")
    with pytest.warns(DeprecationWarning, match=r"record\(timestamp=\.\.\.\) is deprecated and ignored"):
        tracker.record(provider="cloud", model=_PRICED, input_tokens=1, output_tokens=0, timestamp=1.0)
    with pytest.warns(DeprecationWarning, match=r"get_totals\(now=\.\.\.\)"):
        tracker.get_totals(now=1.0)
    with pytest.warns(DeprecationWarning, match=r"get_per_provider_totals\(now=\.\.\.\)"):
        tracker.get_per_provider_totals("cloud", now=1.0)


# ── Red gates whose surface is born in PR 1a (world_clock_a_pre.md §Gates) ──
#
# Each opens with its PRE-SURFACE assertion: the collaborator keyword it needs does not exist until the PR that
# flips it, so until then it fails by AssertionError (never TypeError). Each gate's host-side positive control (the
# same render or decision under today's host default DOES show the time) is a separate green test, so a gate whose
# body could not fail is caught now, not when it flips. The window gates' bodies are also proven in 1a against a
# scratch copy of PR 4's change; every gate is proven again by deletion in the PR that flips it.


def _takes(fn: Any, keyword: str) -> bool:
    return keyword in inspect.signature(fn).parameters


def _with_port(port: Any) -> dict[str, Any]:
    """The ``port=`` keyword for a consumer, or nothing for the host default (a consumer built without one IS host)."""
    from maxim.time.agent_time import HOST_PERCEIVED_TIME

    return {} if port is HOST_PERCEIVED_TIME else {"port": port}


class _StepMachinery:
    """A ``MachineryClock`` a test advances by hand, independent of the patched wall."""

    def __init__(self, start: float = 500.0) -> None:
        self.t = start

    def now_s(self) -> float:
        return self.t

    def monotonic_s(self) -> float:
        return self.t


# A sentinel instant with unusual seconds: none of its renderings occurs in a prompt by chance.
SENTINEL = _dt.datetime(2031, 10, 13, 13, 37, 42, 250000)


def _sentinel_forms(instant: _dt.datetime) -> list[str]:
    """Every form the sentinel would take if any clock reached the text: raw, clock and date."""
    return [
        f"{int(instant.timestamp())}",
        instant.strftime("%H:%M:%S"),
        f"[{instant.strftime('%H:%M:%S')}]",
        instant.strftime("%I:%M %p"),
        instant.strftime("%H:%M"),
        instant.strftime("%A, %B %d, %Y"),
        instant.strftime("%Y-%m-%d"),
        instant.strftime("%B %d"),
    ]


def _calendar_prompts(monkeypatch: Any, port: Any, *, plant: bool = True) -> list[str]:
    """Prompts and context-pool text under *port*, on an ``LLMWorker`` and ``ContextPool`` built directly (not
    through ``build_loop_run``, which keeps the host default in A-pre, D-WC7). The sentinel is planted in the wall
    clock, which is the machinery clock's host default and every ``ContextPool`` stamp, and in the patched
    ``prompt_builder.datetime``. ``plant=False`` plants nothing: every clock is the real one (G-tz)."""
    from maxim.agents.context_pool import ContextPool
    from maxim.agents.llm_worker import LLMWorker

    if plant:
        monkeypatch.setattr(time, "time", _AdvancingWall(SENTINEL.timestamp(), step=0.0))
    # spec=[]: a backend with no cost tracker or cloud policy, so the budget section renders nothing (a bare MagicMock
    # answers every probe and breaks build_budget_context).
    worker = LLMWorker(MagicMock(spec=[]), n_ctx=32000, **_with_port(port))
    prompts = _prompts(monkeypatch, SENTINEL if plant else None, builder=worker._prompt_builder)
    pool = ContextPool(**_with_port(port))
    pool.add_raw("observation", "the red mug moved to the shelf")
    pool.add_outcome(tool_name="look", success=True, result_summary="saw the mug")
    return [*prompts.values(), pool.get_context_text()]


def test_g_render_setup_the_host_port_renders_the_planted_time(
    monkeypatch: pytest.MonkeyPatch, pinned_tz: None
) -> None:
    """G-render's positive control: under today's host default the planted date and time DO reach the text."""
    from maxim.time.agent_time import HOST_PERCEIVED_TIME

    text = "\n".join(_calendar_prompts(monkeypatch, HOST_PERCEIVED_TIME))
    assert SENTINEL.strftime("%A, %B %d, %Y") in text and SENTINEL.strftime("%I:%M %p") in text


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="G-render (OD-A2): the port is born in PR 1a, threaded in PR 2"
)
def test_g_render_no_calendar_renders_no_planted_time(monkeypatch: pytest.MonkeyPatch, pinned_tz: None) -> None:
    """Prompt and context-pool half. The tool-result renders and the frozen-store offset/scale case land red in PR 2,
    with the surface they need (§Gates, amended 2026-11-25)."""
    from maxim.agents.context_pool import ContextPool
    from maxim.agents.llm_worker import LLMWorker
    from maxim.agents.prompt_builder import PromptBuilder
    from maxim.time.agent_time import NO_CALENDAR

    for fn in (LLMWorker.__init__, PromptBuilder.__init__, ContextPool.__init__):
        assert _takes(fn, "port"), f"{fn.__qualname__} takes no port yet (PR 2)"
    for text in _calendar_prompts(monkeypatch, NO_CALENDAR):
        leaked = [form for form in _sentinel_forms(SENTINEL) if form in text]
        assert not leaked, f"a planted time reached a NO_CALENDAR render: {leaked}"


_TZ_CHILD = """
import json, sys
sys.path.insert(0, sys.argv[1])
from maxim.time.agent_time import HOST_PERCEIVED_TIME, NO_CALENDAR
from tests.unit.test_world_clock_a_pre import _calendar_prompts
class _MP:
    def setattr(self, target, name, value):
        setattr(target, name, value)
port = {"host": HOST_PERCEIVED_TIME, "none": NO_CALENDAR}[sys.argv[2]]
print(json.dumps(_calendar_prompts(_MP(), port, plant=False)))
"""


def _prompts_in_two_timezones(tmp_path: Path, port_name: str) -> list[str]:
    """The same prompts built in two child processes whose ``TZ`` is 13 hours apart, on the REAL clock (nothing
    patched: a patched ``datetime`` would hide the very timezone dependence G-tz looks for)."""
    if not hasattr(time, "tzset"):
        pytest.skip("G-tz needs POSIX time.tzset; it runs in CI on Linux (a skip is never a pass)")
    from tests._child_env import isolation_env

    repo = Path(__file__).resolve().parents[2]
    outputs = []
    for tz in ("UTC", "Etc/GMT-13"):  # each child starts under its TZ (tzset at interpreter start)
        home = tmp_path / f"home-{port_name}-{tz.replace('/', '_')}"
        home.mkdir()
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(repo / "src"), "HOME": str(home), "TZ": tz}
        env.update(isolation_env())
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(_TZ_CHILD), str(repo), port_name],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        # A child that crashes is a broken gate, not the expected red: fail it with the child's own error.
        if proc.returncode != 0:
            pytest.fail(f"child under TZ={tz} failed:\n{proc.stderr}")
        outputs.append(proc.stdout)
    return outputs


def test_g_tz_setup_the_host_prompt_depends_on_the_timezone(tmp_path: Path) -> None:
    """G-tz's positive control: under today's host default the two timezones' prompts differ (13 hours apart)."""
    first, second = _prompts_in_two_timezones(tmp_path, "host")
    assert first != second, "the host prompt renders no local time: G-tz could not fail"


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="G-tz (OD-WC6): the port is born in PR 1a, threaded in PR 2"
)
def test_g_tz_a_no_calendar_prompt_is_the_same_in_every_timezone(tmp_path: Path) -> None:
    from maxim.agents.context_pool import ContextPool
    from maxim.agents.llm_worker import LLMWorker

    for fn in (LLMWorker.__init__, ContextPool.__init__):
        assert _takes(fn, "port"), f"{fn.__qualname__} takes no port yet (PR 2)"
    first, second = _prompts_in_two_timezones(tmp_path, "none")
    assert first == second, "a NO_CALENDAR prompt depends on the timezone"


def _scn_derived(port: Any, tmp_path: Path, name: str, monkeypatch: Any) -> tuple[bytes, dict[str, Any]]:
    """One SCN, registered identically, and every SCN-derived consumer's output under *port*.

    The wall and the ``datetime`` the relative-date parser reads are pinned at T0, so "yesterday" means the day the
    memories were registered. The Statistician gets a confirmed pattern with a high temporal anomaly (the render
    half) and one assessment against the oscillator (the decision half).
    """
    from maxim.agents.statistician_agent import PatternDetector, PatternState, StatisticianAgent
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.retrieval import temporal_signal
    from maxim.simulation.introspection import Observer
    from maxim.time.scn import SCN
    from maxim.time.temporal_signature import TemporalSignature
    from maxim.tools.introspection import TemporalPatternsTool

    class _AtT0(_dt.datetime):
        @classmethod
        def now(cls, tz: Any = None) -> Any:  # noqa: ARG003
            return _dt.datetime.fromtimestamp(T0)

    monkeypatch.setattr(time, "time", _AdvancingWall(T0, step=0.0))
    monkeypatch.setattr(temporal_signal, "datetime", _AtT0)
    scn = SCN()
    scn.enable_oscillator()
    for i in range(1, 7):  # the same hour on six consecutive days, the last one yesterday
        scn.register(f"m{i}", TemporalSignature.from_timestamp(T0 - 86_400.0 * i))
    stat = StatisticianAgent(MagicMock(), MagicMock(), scn=scn, **_with_port(port))
    confirmed = PatternDetector(metric_name="tool_latency", state=PatternState.CONFIRMED_PATTERN, temporal_anomaly=0.9)
    stat._detectors["tool_latency"] = confirmed
    assessed = PatternDetector(metric_name="assessed")
    stat._assess_with_temporal_context(assessed)
    # NAc anchor similarity (the temporal credit fallback) reads the host calendar in both runs: never the port.
    nac = NAc(NACConfig())
    nac.update_eligibility("a", "node", 1.0, temporal_sig=TemporalSignature.from_timestamp(T0 - 86_400.0))
    nac._eligibility[("a", "node")] = 0.0  # the fast trace has expired, so only the anchor can credit
    seen = {
        "patterns": TemporalPatternsTool(scn, **_with_port(port)).execute(discover_rhythms=True).output,
        "phase": Observer(memory_hub=MagicMock(scn=scn), **_with_port(port)).temporal_patterns(),
        "relative": temporal_signal.TemporalReferenceSignal(scn, **_with_port(port)).extract("what did I do yesterday"),
        "summary": stat.summarize_for_context(),
        "assessed_anomaly": assessed.temporal_anomaly,
        "nac_anchor_credit": nac.distribute_reward("a", 1.0),
    }
    path = tmp_path / f"scn-{name}.json"
    scn.save(str(path), overwrite=True)
    return path.read_bytes(), seen


def test_g_scn_setup_the_host_port_renders_every_scn_derived_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G-scn's positive control: under today's host default each SCN-derived consumer DOES emit or resolve time."""
    from maxim.time.agent_time import HOST_PERCEIVED_TIME

    _, seen = _scn_derived(HOST_PERCEIVED_TIME, tmp_path, "host", monkeypatch)
    assert (seen["patterns"] or {}).get("rhythmic_patterns"), seen["patterns"]
    assert seen["phase"].get("current_phase"), seen["phase"]
    assert seen["relative"], "the host resolves no relative date: G-scn's relative-date check could not fail"
    assert "[TEMPORAL]" in seen["summary"], seen["summary"]
    assert seen["assessed_anomaly"] != 0.0, "the host assessment set no temporal anomaly"
    assert seen["nac_anchor_credit"], "no anchor credit: G-scn's NAc check compares nothing"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="G-scn (D-WC2 revised): SCN-derived renders, PR 3")
def test_g_scn_no_calendar_emits_no_scn_derived_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.agents.statistician_agent import StatisticianAgent
    from maxim.retrieval.temporal_signal import TemporalReferenceSignal
    from maxim.simulation.introspection import Observer
    from maxim.time.agent_time import HOST_PERCEIVED_TIME, NO_CALENDAR
    from maxim.tools.introspection import TemporalPatternsTool

    for fn in (
        StatisticianAgent.__init__,
        TemporalReferenceSignal.__init__,
        Observer.__init__,
        TemporalPatternsTool.__init__,
    ):
        assert _takes(fn, "port"), f"{fn.__qualname__} takes no port yet (PR 3)"
    host_bins, host = _scn_derived(HOST_PERCEIVED_TIME, tmp_path, "host", monkeypatch)
    none_bins, seen = _scn_derived(NO_CALENDAR, tmp_path, "none", monkeypatch)
    assert not (seen["patterns"] or {}).get("rhythmic_patterns"), f"temporal_patterns binned: {seen['patterns']}"
    assert not seen["phase"].get("current_phase"), f"current_phase rendered: {seen['phase']}"
    assert seen["relative"] == [], f"a relative date resolved: {seen['relative']}"
    assert "[TEMPORAL]" not in seen["summary"], seen["summary"]
    assert seen["assessed_anomaly"] == 0.0, "the assessment read the calendar under NO_CALENDAR"
    # Registration and NAc are untouched (D-WC2 revised): the SCN's bins and NAc's anchor credit are the host run's.
    assert none_bins == host_bins
    assert seen["nac_anchor_credit"] == host["nac_anchor_credit"]


def _window_hippocampus(clock: _StepMachinery) -> Any:
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    config = HippocampusConfig(persistence_path=None, enable_sleep_consolidation=False, auto_save_after_sleep=False)
    return Hippocampus(config, machinery_clock=clock)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="G-window: Hippocampus takes the machinery clock in PR 4")
def test_g_window_the_dedup_window_counts_machinery_seconds(
    wall: _AdvancingWall, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maxim.memory.hippocampus import Hippocampus

    assert _takes(Hippocampus.__init__, "machinery_clock"), "Hippocampus takes no machinery_clock yet (PR 4)"
    obs = "the red mug is on the table"
    # 0 machinery seconds and 31 wall seconds apart: the same moment for the agent, so it de-duplicates.
    clock = _StepMachinery()
    hippocampus = _window_hippocampus(clock)
    assert hippocampus.store_observation(obs)
    wall.jump(31.0)
    assert hippocampus.store_observation(obs) == "", "a wall jump opened the window"
    # 31 machinery seconds and 0 wall seconds apart: past the 30 s window, so it is stored again.
    monkeypatch.setattr(time, "time", _AdvancingWall(wall.now, step=0.0))
    clock = _StepMachinery()
    hippocampus = _window_hippocampus(clock)
    assert hippocampus.store_observation(obs)
    clock.t += 31.0
    assert hippocampus.store_observation(obs), "the window did not count machinery seconds"


def _expected_window(script: list[tuple[float, str]], window_s: float = 30.0) -> list[bool]:
    """A reference model of the de-duplication window on the machinery clock: which steps are stored."""
    t, last_seen, stored = 0.0, {}, []
    for dt, obs in script:
        t += dt
        keep = obs not in last_seen or (t - last_seen[obs]) >= window_s
        if keep:
            last_seen[obs] = t
        stored.append(keep)
    return stored


# G-pace's scripted session: (machinery seconds since the last step, observation). Cumulative machinery times 0, 5,
# 10, 41, 43, 53, 78, 118, 119, 120, 150, 153: "a" repeats 5 s and 1 s after it was stored (de-duplicated, steps 1
# and 9); every other repeat is more than the 30 s window apart (stored).
_SCRIPT = list(zip([0.0, 5.0, 5.0, 31.0, 2.0, 10.0, 25.0, 40.0, 1.0, 1.0, 30.0, 3.0], "aababcacaabc"))
_SCRIPT_STORED = [True, False, True, True, True, True, True, True, True, False, True, True]


def _window_digest(wall_step: Any) -> list[bool]:
    """Run ``_SCRIPT`` through the window; ``wall_step(i, dt)`` moves the wall between steps. Ids are excluded (D-WC1):
    the digest is which observations were stored."""
    clock = _StepMachinery()
    hippocampus = _window_hippocampus(clock)
    stored = []
    for i, (dt, obs) in enumerate(_SCRIPT):
        clock.t += dt
        wall_step(i, dt)
        stored.append(bool(hippocampus.store_observation(obs)))
    return stored


def test_g_pace_setup_the_script_de_duplicates() -> None:
    """The window gates' script exercises both outcomes, and the reference model agrees with the hand count."""
    assert _expected_window(_SCRIPT) == _SCRIPT_STORED
    assert False in _SCRIPT_STORED and True in _SCRIPT_STORED


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="G-pace (OD-WC2, A-pre scope): the window, PR 4")
def test_g_pace_the_window_is_the_same_at_any_wall_pace(wall: _AdvancingWall) -> None:
    from maxim.memory.hippocampus import Hippocampus

    assert _takes(Hippocampus.__init__, "machinery_clock"), "Hippocampus takes no machinery_clock yet (PR 4)"
    jitter = [0.013, 0.37, 0.002, 0.91, 0.05, 0.6, 0.2, 0.033, 0.48, 0.07, 0.29, 0.011]
    at_1x = _window_digest(lambda i, dt: wall.jump(dt + jitter[i]))
    at_20x = _window_digest(lambda i, dt: wall.jump(dt / 20.0 + jitter[i]))
    assert at_1x == _SCRIPT_STORED, f"the window's decisions are not the machinery clock's: {at_1x}"
    assert at_1x == at_20x, "the window's decisions depend on the wall's pace"


class _JumpingWall:
    """A ``time.time`` that jumps a huge, varying amount on EVERY read, from a fixed schedule."""

    def __init__(self, start: float, schedule: list[float]) -> None:
        self.now, self._schedule, self._reads = start, schedule, 0

    def __call__(self) -> float:
        self.now += self._schedule[self._reads % len(self._schedule)]
        self._reads += 1
        return self.now


# G-long's long-horizon script: 120 observations over 3 labels, 0 to 22 machinery seconds apart, so a label recurs
# about every 33 s and its repeats straddle the 30 s window (90 stored, 30 de-duplicated by the reference model).
_LONG_SCRIPT = [(float((i * 7) % 23), "abc"[(i * 2) % 3]) for i in range(120)]


def test_g_long_setup_the_long_script_de_duplicates() -> None:
    expected = _expected_window(_LONG_SCRIPT)
    assert expected.count(False) >= 20 and expected.count(True) >= 20, "the long script must exercise both outcomes"


def _long_run(monkeypatch: Any, schedule: list[float]) -> list[bool]:
    """Observe, consolidate, sleep and decay through a memory hub (no ATL, no background workers), with the wall
    jumping per read. The digest is A-pre's families only: the window's decisions (§Gates: NAc, ATL, SCN and the stamps
    are excluded until their PRs)."""
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.integration.memory_hub import build_memory_hub
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.similarity.ec import ECConfig, EntorhinalCortex
    from maxim.time.scn import SCN

    monkeypatch.setattr(time, "time", _JumpingWall(T0, schedule))
    clock = _StepMachinery()
    config = HippocampusConfig(persistence_path=None, enable_sleep_consolidation=False, auto_save_after_sleep=False)
    hippocampus = Hippocampus(config, machinery_clock=clock)
    hub = build_memory_hub(
        agent_id="g_long",
        hippocampus=hippocampus,
        scn=SCN(),
        nac=NAc(NACConfig()),
        ec=EntorhinalCortex(ECConfig()),
        atl=None,
        load_persisted=False,
        start_background_workers=False,
    )
    stored = []
    for i, (dt, obs) in enumerate(_LONG_SCRIPT):
        clock.t += dt
        stored.append(bool(hippocampus.store_observation(obs)))
        if i % 20 == 19:
            hippocampus.consolidate()
            hippocampus.sleep(auto_save=False)
            hub.nac.decay_all()
    return stored


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="G-long (BD-14.2), A-pre half: the window, PR 4")
def test_g_long_huge_wall_jumps_change_nothing_in_a_pre_s_families(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only ``time.time`` jumps: ``monotonic`` jumps would trip ``Deadline`` and ``wait_session_ended``. The full-loop
    half lands red in A-post's first PR, with the loop surface it needs (§Gates, amended 2026-11-25)."""
    from maxim.memory.hippocampus import Hippocampus

    assert _takes(Hippocampus.__init__, "machinery_clock"), "Hippocampus takes no machinery_clock yet (PR 4)"
    expected = _expected_window(_LONG_SCRIPT)
    first = _long_run(monkeypatch, [10_000_000.25, 3_000_000.5, 77_777_777.75])
    second = _long_run(monkeypatch, [123_456.125, 9_999_999.0, 55_555_555.5, 1_000.75])
    assert first == expected, "the window's decisions are not the machinery clock's"
    assert first == second, "a wall jump reached A-pre's families"
