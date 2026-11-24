"""World clock A-pre: its red gates and WC5's characterization guards (docs/plans/world_clock_a_pre.md §Gates).

PR 0 lands what today's surface can express:

- **G-cost-month** (#1769, lands 0, flips in PR 1): a ``CostTracker`` on the wall keeps its month, windows and EMAs
  whatever ``timestamp`` a caller passes.
- **WC5, robots unchanged** (green): today's persisted Hippocampus and NAc files load and save back unchanged; the
  prompt's date and time enter only through ``maxim.agents.prompt_builder.datetime``; and the shipped dispatch seam
  books no energy cost. That last pin replaces the planned G-tool-duration gate, which today's code could only pass
  vacuously (owner decision 2026-11-24; see ``test_wc5_execute_and_learn_books_no_energy_cost_today``).

Each gate is ``xfail(strict=True, raises=AssertionError)``: an ``ImportError``, ``TypeError`` or ``NameError`` is never
the expected failure. The facts a gate relies on (the run booked an energy event; the second record was priced) are
asserted in a separate green test, so a broken setup cannot pass as the expected red. (That precondition is how
PR 0 found the energy booking has no caller.) Fake clocks ADVANCE on every
read: a frozen clock hides cadence bugs. Everything that writes state uses ``tmp_path``.
"""

from __future__ import annotations

import datetime as _dt
import functools
import json
import os
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


def _tracker(tmp_path: Path) -> Any:
    from maxim.models.language.cost_tracker import CostTracker, CostTrackerConfig, ModelPricing

    config = CostTrackerConfig(state_path=str(tmp_path / "cost_state.json"))
    return CostTracker({_PRICED: ModelPricing(input_price=3.0, output_price=0.0)}, config)


def _record_wall_then_epoch(tracker: Any) -> tuple[float, float]:
    """One record on the wall, one with a caller ``timestamp`` of 0.0: the costs booked."""
    first = tracker.record(provider="cloud", model=_PRICED, input_tokens=1_000_000, output_tokens=0)
    second = tracker.record(provider="cloud", model=_PRICED, input_tokens=1_000_000, output_tokens=0, timestamp=0.0)
    return first, second


def test_cost_month_gate_setup_prices_both_records(tmp_path: Path, wall: _AdvancingWall) -> None:
    """G-cost-month's precondition: both records are priced at 3.0, so the gate's totals are meaningful."""
    assert _record_wall_then_epoch(_tracker(tmp_path)) == (3.0, 3.0)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="G-cost-month (#1769): a caller timestamp of 0.0 resets the month and falls out of the windows; PR 1",
)
def test_g_cost_month_a_caller_timestamp_cannot_move_the_tracker_off_the_wall(
    tmp_path: Path, wall: _AdvancingWall
) -> None:
    tracker = _tracker(tmp_path)
    _record_wall_then_epoch(tracker)
    wall.jump(5.0)
    totals = tracker.get_totals()
    assert totals == tracker.get_totals(now=0.0), "a caller clock reached window pruning"
    assert totals["monthly"] == 6.0, f"the month was reset by a caller timestamp: {totals}"
    assert totals["hourly"] == 6.0 and totals["daily"] == 6.0, f"a record fell out of the rolling windows: {totals}"
    assert tracker.get_per_provider_totals("cloud") == {"hourly": 6.0, "daily": 6.0, "monthly": 6.0}
    for name, rate in tracker.get_spend_rates().items():
        assert rate.last_update >= T0, f"{name} EMA was stamped with the caller's clock: {rate.last_update}"


# ── G-tool-duration (OD-A4's root fix) ───────────────────────────────────────


def _energy_valences(wall: _AdvancingWall, jump_s: float) -> list[tuple[str, float]]:
    """Run one tool through the real ``execute_and_learn`` while the wall jumps ``jump_s`` inside the tool.

    The tool does the same work either way: only the wall jump differs. Returns each energy event NAc was shown,
    as (valence, delta_seconds).
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
    original = nac.observe

    def spy(**kwargs: Any) -> Any:
        if kwargs.get("event_type") == "energy":
            seen.append((kwargs["outcome_valence"].value, float(kwargs["delta_seconds"])))
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
    return seen


def test_wc5_execute_and_learn_books_no_energy_cost_today(wall: _AdvancingWall) -> None:
    """Characterization (owner decision 2026-11-24): the shipped dispatch seam books NO energy cost, at any duration.

    ``record_outcome``'s energy booking (NEGATIVE valence above 2.0 s, ``_observe_energy_cost``) runs only when a
    caller passes ``elapsed_s``, and none does: ``execute_and_learn`` measures ``exec_elapsed`` for its logs and
    never hands it on (true since the booking landed in 54b77b3a). So OD-A4's tool-duration wall read reaches no
    learning today, and G-tool-duration could not fail by assertion: it would pass vacuously. Robots keep this
    (WC5). The PR that wires the duration (A-post with B2, fenced ``runtime/tool_dispatch.py``) replaces this pin
    with G-tool-duration, landed red, in the same PR.
    """
    assert _energy_valences(wall, 1.0) == []
    assert _energy_valences(wall, 3.0) == []


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

    Both writers serialise a ``set`` as a list (``hippocampus_persistence``'s ``context_index``, NAc's
    ``outcome_index``), so that order follows PYTHONHASHSEED and two saves of one store differ only there. The writer
    fix (sorted output) is its own PR (owner decision 2026-11-24); until it lands, WC5 compares with that one index
    sorted and every other byte exact.
    """
    data = json.loads(raw)
    data[index] = {k: sorted(v) for k, v in data.get(index, {}).items()}
    return json.dumps(data, indent=2).encode()


def _resaved_equals_original(raw: bytes, resaved: bytes, index: str) -> None:
    # The writer's format is plain ``json.dumps(indent=2)``, so re-serialising the original reproduces it byte for
    # byte; the canonical compare below is then exact for every byte outside the one index.
    assert json.dumps(json.loads(raw), indent=2).encode() == raw.rstrip(b"\n"), "the canonical form is lossy"
    assert _sorted_index(resaved, index) == _sorted_index(raw, index)


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


def _prompts(monkeypatch: pytest.MonkeyPatch, instant: _dt.datetime) -> dict[str, str]:
    """The three prompt kinds that render the date and time, built with ``prompt_builder.datetime`` patched."""
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
    monkeypatch.setattr(prompt_builder, "datetime", _At)
    builder = prompt_builder.PromptBuilder(
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


def test_wc5_the_prompt_reads_time_only_through_the_patched_datetime(
    monkeypatch: pytest.MonkeyPatch, pinned_tz: None, wall: _AdvancingWall
) -> None:
    """Two instants, with the wall advancing between builds: the prompts differ only in the date and time text."""

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
