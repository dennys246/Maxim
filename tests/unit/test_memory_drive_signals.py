"""Per-drive pressure and relief on the encoding record (memory-strength plan, Phase 2b-ii).

Phase 2c gates drive importance by RELEVANCE -- a starving stretch must tag the traces whose
actions touched hunger, not everything that happened while hungry -- so a scalar is not enough:
the record names each drive. Both values are read from the executor's stamp, pressure from BEFORE
the action (afterwards an ``eat`` would show its own hunger already relieved). Still write-only:
nothing reads the record until Phase 2c.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from maxim.embodiment.sem import (
    EntropicDriveSpec,
    HomeostaticDriveSpec,
    drive_comfort_progress,
    drive_pressure,
    drive_span,
    relief_fraction_from_progress,
)
from maxim.memory.encoding import EncodingSignals
from maxim.tools.base import Tool, ToolOutput

# minecraft_player's shape: health homeostatic (set point 20, comfort band 6) and food entropic
# (drift down, deprivation 6 / satisfaction 16), both on the declared raw range [0, 40].
_HEALTH = HomeostaticDriveSpec(set_point=20.0, drift_rate=0.0, comfort_band=6.0)
_FOOD = EntropicDriveSpec(
    drift_direction="down",
    drift_rate=0.0,
    deprivation_threshold=6.0,
    deprivation_pain=0.5,
    satisfaction_threshold=16.0,
)
_RANGE = (0.0, 40.0)


def drive_relief_fraction(spec, before: float, after: float, lo: float, hi: float):
    """What the production pair computes together: difference, then normalise."""
    return relief_fraction_from_progress(spec, drive_comfort_progress(spec, before, after), lo, hi)


# ── the helpers ──────────────────────────────────────────────────────────────


def test_a_drives_span_is_the_most_it_can_move():
    assert drive_span(_HEALTH, *_RANGE) == 20.0  # the farthest health can sit from its set point
    assert drive_span(_FOOD, *_RANGE) == 10.0  # its OWN deprivation->satisfaction band (6..16),
    # NOT the declared [0, 40] range: that is widened for the encoder's neutral, so inheriting it
    # made a full satisfaction read 0.25 and put 1.0 out of reach (Phase 2b-ii review)
    assert drive_span(_HEALTH, 0.0, 0.0) is None  # no range, no denominator
    assert drive_span(_HEALTH, float("nan"), 40.0) is None


def test_relief_is_the_fraction_of_what_the_drive_could_give():
    assert drive_relief_fraction(_HEALTH, 10.0, 20.0, *_RANGE) == 0.5  # half of its 20 of headroom
    assert drive_relief_fraction(_FOOD, 6.0, 11.0, *_RANGE) == 0.5  # half of the band
    assert drive_relief_fraction(_HEALTH, 20.0, 10.0, *_RANGE) == 0.0  # harm is the pain channel's
    assert drive_relief_fraction(_HEALTH, 10.0, 20.0, 0.0, 0.0) is None


def test_relief_saturates_at_one_rather_than_running_past_it():
    assert drive_relief_fraction(_FOOD, 0.0, 40.0, *_RANGE) == 1.0  # more than satisfied is still 1


def test_relief_and_pressure_agree_on_what_a_full_relief_is():
    # the Phase 2b-ii review's finding: they were on different scales, so fully satisfying a
    # starving drive recorded 0.25 relief while its pressure went 1.0 -> 0.0
    assert drive_pressure(_FOOD, 6.0, *_RANGE) == 1.0
    assert drive_relief_fraction(_FOOD, 6.0, 16.0, *_RANGE) == 1.0
    assert drive_pressure(_FOOD, 16.0, *_RANGE) == 0.0
    assert drive_pressure(_HEALTH, 0.0, *_RANGE) == 1.0
    assert drive_relief_fraction(_HEALTH, 0.0, 20.0, *_RANGE) == 1.0


def test_pressure_reads_zero_inside_comfort_and_one_at_the_edge():
    assert drive_pressure(_HEALTH, 20.0, *_RANGE) == 0.0  # at the set point: measured, not absent
    assert drive_pressure(_HEALTH, 14.0, *_RANGE) == 0.0  # inside the comfort band
    assert drive_pressure(_HEALTH, 0.0, *_RANGE) == 1.0
    assert drive_pressure(_HEALTH, 7.0, *_RANGE) == pytest.approx(0.5)
    assert drive_pressure(_FOOD, 16.0, *_RANGE) == 0.0  # satisfied
    assert drive_pressure(_FOOD, 6.0, *_RANGE) == 1.0  # deprived
    assert drive_pressure(_FOOD, 11.0, *_RANGE) == pytest.approx(0.5)
    assert drive_pressure(_HEALTH, float("nan"), *_RANGE) is None  # unreadable, not "no pressure"


def test_pressure_covers_what_the_corrective_need_cannot():
    # corrective_need_intensity answers a different question and returns None for entropic "up"
    # drives and above-set-point deficits; it stays exactly as it is (Exp 58/60/62 fingerprints).
    from maxim.embodiment.sem import corrective_need_intensity

    hunger_up = EntropicDriveSpec(
        drift_direction="up",
        drift_rate=0.0,
        deprivation_threshold=0.8,
        deprivation_pain=0.5,
        satisfaction_threshold=0.2,
    )
    assert corrective_need_intensity(hunger_up, 0.9) is None
    assert drive_pressure(hunger_up, 0.9, 0.0, 1.0) == 1.0  # past deprivation: full pressure
    assert drive_pressure(hunger_up, 0.5, 0.0, 1.0) == pytest.approx(0.5)
    assert corrective_need_intensity(_HEALTH, 30.0) is None  # above the set point
    assert drive_pressure(_HEALTH, 30.0, *_RANGE) == pytest.approx(4 / 14)  # 4 past the band, of 14


# ── the producers keep their scalar and gain a per-drive record ─────────────


def _body(**metrics: float) -> Any:
    entity = SimpleNamespace(
        drive_specs={"health": _HEALTH, "food": _FOOD},
        vital_metrics=dict(metrics),
        modulators={},
        sensors={},
        full_path="agent.body",
    )
    entity.walk = lambda: [entity]
    return entity


def test_the_per_drive_record_always_sums_to_the_credit_scalar():
    from maxim.embodiment.tool_bridge import _drive_potential_diff, _drive_progress_by_drive

    body = _body(health=12.0, food=30.0)
    pre = {"health": 6.0, "food": 10.0}
    effect = {"health": 6.0, "food": 20.0}
    by_drive = _drive_progress_by_drive(body, effect, pre)
    assert by_drive == {
        "health": drive_comfort_progress(_HEALTH, 6.0, 12.0),
        "food": drive_comfort_progress(_FOOD, 10.0, 30.0),
    }
    assert sum(by_drive.values()) == pytest.approx(_drive_potential_diff(body, effect, pre))


# ── the executor stamps the body around each invocation ─────────────────────


class _Eat(Tool):
    name = "eat"
    description = "eats"
    input_schema: dict[str, Any] = {}

    def __init__(self, body: Any) -> None:
        super().__init__()
        self._body = body

    def execute(self, **kwargs: Any) -> ToolOutput:
        self._body.vital_metrics["food"] = 30.0  # the world moved while the tool ran
        return ToolOutput(
            success=True,
            side_effects={"drive_progress_by_drive": {"food": 5.0}, "drive_progress_body": self._body.full_path},
        )


def _executor_with(body: Any):
    from maxim.runtime.executor import Executor
    from maxim.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(_Eat(body))
    executor = Executor(tool_registry=registry)
    executor.embodiment = SimpleNamespace(root=body)
    body.sensors = {
        "health": SimpleNamespace(reading_schema={"range": [0.0, 40.0]}),
        "food": SimpleNamespace(reading_schema={"range": [0.0, 40.0]}),
    }
    return executor


def test_pressure_is_stamped_from_BEFORE_the_action():
    body = _body(health=8.0, food=10.0)  # hurt and hungry when the action is chosen
    result = _executor_with(body).execute({"tool_name": "eat", "params": {}})
    assert dict(result.drive_pressure_before) == {
        "health": pytest.approx(drive_pressure(_HEALTH, 8.0, *_RANGE)),
        "food": pytest.approx(drive_pressure(_FOOD, 10.0, *_RANGE)),
    }
    assert body.vital_metrics["food"] == 30.0  # ...and the tool did relieve it, afterwards


def test_relief_is_stamped_as_a_fraction_per_drive():
    result = _executor_with(_body(health=8.0, food=10.0)).execute({"tool_name": "eat", "params": {}})
    assert dict(result.drive_relief) == {"food": 0.5}  # 5 of the 10 its band could give


def test_no_body_means_no_drive_stamp():
    from maxim.runtime.executor import Executor
    from maxim.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(_Eat(_body(food=10.0)))
    result = Executor(tool_registry=registry).execute({"tool_name": "eat", "params": {}})
    assert result.drive_pressure_before is None and result.drive_relief is None


def test_a_tool_cannot_stamp_its_own_drive_signals():
    class Liar(Tool):
        name = "liar"
        description = "claims relief"
        input_schema: dict[str, Any] = {}

        def execute(self, **kwargs: Any) -> ToolOutput:
            return ToolOutput(success=True, drive_relief=(("food", 1.0),), drive_pressure_before=(("food", 1.0),))

    from maxim.runtime.executor import Executor
    from maxim.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(Liar())
    result = Executor(tool_registry=registry).execute({"tool_name": "liar", "params": {}})
    assert result.drive_relief is None and result.drive_pressure_before is None


# ── the loop capture records what the stamp carried ─────────────────────────


def test_the_loop_capture_records_both_per_drive_signals():
    from unittest.mock import MagicMock

    from maxim.runtime.bio_integration import capture_episodic_memory

    hippo = MagicMock()
    stamped = ToolOutput(
        success=True,
        rpe=0.4,
        drive_pressure_before=(("food", 0.8), ("health", 0.1)),
        drive_relief=(("food", 0.5),),
    )
    capture_episodic_memory(
        hippocampus=hippo,
        executor=None,
        observation={},
        state=None,
        intent={},
        action={"tool_name": "eat"},
        result=stamped,
        run_id="r",
    )
    encoding = hippo.capture_from_loop_async.call_args.kwargs["encoding"]
    assert encoding.drive_pressure == (("food", 0.8), ("health", 0.1))
    assert encoding.drive_relief == (("food", 0.5),)
    assert encoding.measured() == ("surprise", "drive_pressure", "drive_relief")


# ── the record's shape ───────────────────────────────────────────────────────


def _signals(**over: Any) -> EncodingSignals:
    base = dict(
        site="loop", salience=None, novelty=None, surprise=None, pain=None, drive_pressure=None, drive_relief=None
    )
    return EncodingSignals(**{**base, **over})


def test_per_drive_values_are_validated_like_every_other_signal():
    with pytest.raises(ValueError):
        _signals(drive_relief=(("food", 1.4),))
    with pytest.raises(TypeError):
        _signals(drive_relief={"food": 0.5})  # a mapping would be mutable inside a frozen record
    with pytest.raises(ValueError, match="sorted"):
        _signals(drive_pressure=(("health", 0.1), ("food", 0.8)))
    with pytest.raises(ValueError, match="once"):
        _signals(drive_relief=(("food", 0.1), ("food", 0.2)))
    with pytest.raises(ValueError, match="collide"):
        _signals(extra={"drive_relief": 1})


def test_the_per_drive_record_round_trips_through_the_trace(tmp_path, complete_memory_args):
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    signals = _signals(drive_pressure=(("food", 0.8),), drive_relief=(("food", 0.5),))
    hippo = Hippocampus(HippocampusConfig(persistence_path=None))
    mid = hippo.capture(**{**complete_memory_args, "encoding": signals})
    path = str(tmp_path / "h.json")
    hippo.save(path)
    restored = Hippocampus(HippocampusConfig(persistence_path=None))
    restored.load(path)
    assert restored.recall_by_ids([mid])[0].encoding == signals


# ── the measured branch, end to end through the shipped survival path ───────


def _world_eat_tool(food_initial: float, food_after: float):
    """The Exp 60 shape: a world-owned interoceptive drive whose relief is MEASURED, not modeled.

    On ``minecraft_player`` food is live-world-owned, so the modeled ``self_effect`` is stripped and
    the measured branch is the ONLY shipped producer of ``drive_progress_by_drive``. Mirrors
    tests/unit/test_survival_learns_break2.py's fixture.
    """
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.sem import AffordanceSchema, ModulatorResult
    from maxim.embodiment.spec import _parse_entity
    from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

    body = _parse_entity(
        {
            "name": "player",
            "entity_type": "body",
            "sensors": {
                "food": {
                    "unit": "points",
                    "range": [0, 40],
                    "initial": food_initial,
                    "modality": "world",
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "down",
                        "drift_rate": 0.0,
                        "deprivation_threshold": 6.0,
                        "deprivation_pain": 0.5,
                        "satisfaction_threshold": 16.0,
                    },
                }
            },
        }
    )
    embodiment = Embodiment(body)
    embodiment.live_world_set_sensors = {"food"}

    class _World:
        name = "world"

        def check_affordance_requires(self, _name):
            return (True, "")

        def execute(self, affordance, params):
            body.vital_metrics["food"] = food_after  # the world's post-action truth
            return ModulatorResult(
                modulator_name="world", entity_name="player", affordance=affordance, params=params, success=True
            )

    schema = AffordanceSchema(description="Eat", self_effect={"food": 4.0})
    tool = ModulatorAffordanceTool(body, _World(), "eat", schema, "minecraft_player_eat", embodiment=embodiment)
    return body, embodiment, tool


def test_the_measured_branch_records_per_drive_progress_and_names_its_body():
    _, _, tool = _world_eat_tool(food_initial=2.0, food_after=12.0)
    side = tool.execute().side_effects or {}
    assert side["drive_progress_by_drive"] == {"food": pytest.approx(10.0)}  # world-measured relief
    assert side["drive_progress_by_drive"]["food"] == pytest.approx(side["drive_potential_diff"])
    assert side["drive_progress_body"] == "player"


def test_the_whole_path_from_the_world_to_the_trace(complete_memory_args):
    """world relief -> side effect -> executor stamp -> EncodingSignals on a real trace."""
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.runtime.bio_integration import capture_episodic_memory
    from maxim.runtime.executor import Executor
    from maxim.tools.registry import ToolRegistry

    body, embodiment, tool = _world_eat_tool(food_initial=2.0, food_after=12.0)
    registry = ToolRegistry()
    registry.register(tool)
    executor = Executor(tool_registry=registry)
    executor.embodiment = embodiment

    result = executor.execute({"tool_name": tool.name, "params": {}})
    hippo = Hippocampus(HippocampusConfig(persistence_path=None))
    capture_episodic_memory(
        hippocampus=hippo,
        executor=executor,
        observation={},
        state=None,
        intent={},
        action={"tool_name": tool.name},
        result=result,
        run_id="r",
    )
    hippo._process_capture(hippo._capture_queue.get_nowait())
    [trace] = list(hippo)
    # starving when it acted (food 2, deprived at 6), and the world's relief was the whole band
    assert dict(trace.encoding.drive_pressure) == {"food": 1.0}
    assert dict(trace.encoding.drive_relief) == {"food": 1.0}


def test_progress_from_another_body_is_not_recorded_as_this_ones():
    from maxim.runtime.executor import Executor
    from maxim.tools.registry import ToolRegistry

    class _Foreign(Tool):
        name = "feed_other"
        description = "acts on another entity"
        input_schema: dict[str, Any] = {}

        def execute(self, **kwargs: Any) -> ToolOutput:
            return ToolOutput(
                success=True,
                side_effects={"drive_progress_by_drive": {"food": 20.0}, "drive_progress_body": "mother.body"},
            )

    registry = ToolRegistry()
    registry.register(_Foreign())
    executor = Executor(tool_registry=registry)
    executor.embodiment = SimpleNamespace(root=_body(food=10.0))
    assert executor.execute({"tool_name": "feed_other", "params": {}}).drive_relief is None


def test_a_body_read_glitch_never_costs_the_action():
    from maxim.runtime.executor import Executor
    from maxim.tools.registry import ToolRegistry

    body = _body(food="not-a-number")  # a sensor that cannot be read as a number
    del body.sensors  # ...and a body shape the range walk does not expect
    registry = ToolRegistry()
    registry.register(_Eat(body))
    executor = Executor(tool_registry=registry)
    executor.embodiment = SimpleNamespace(root=body)
    result = executor.execute({"tool_name": "eat", "params": {}})
    assert result.success and result.drive_pressure_before is None  # the record is skipped, not the action
