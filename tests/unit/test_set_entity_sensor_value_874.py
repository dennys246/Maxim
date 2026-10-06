"""#874: ``set_entity_sensor`` VALUE mode must resolve the sensor the way DELTA mode does.

Value mode wrote ``root.vital_metrics[sensor]`` for every name. On the real
``bodies/infant_humanoid`` that made ``arms.thermal`` an orphan ROOT key: the
arm's own ``thermal`` stayed 0.0 (and drifted on its own), while
``evaluate_failures`` read the orphan first, so drive pain latched at the
written value and never decayed. It also clamped every write to [0, 1], so a
cold write to a [-1, 1] sensor became 0, and it reported success for a sensor
the body does not have.

Every test runs the REAL tool against the REAL body (a synthetic body is how
#871's first fix nearly broke every startle).
"""

from __future__ import annotations

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.simulation.tools import SetEntitySensorTool


class _Bus:
    def __init__(self) -> None:
        self.signals: list = []

    def publish(self, signal) -> None:
        self.signals.append(signal)


def _infant():
    root = ComponentRegistry().instantiate("bodies/infant_humanoid")
    bus = _Bus()
    emb = Embodiment(root, pain_bus=bus)
    return root, bus, SetEntitySensorTool(embodiment=emb, entity_map=None)


def test_value_reaches_the_arm_sub_sensor_not_an_orphan_root_key():
    root, _, tool = _infant()
    out = tool.execute(sensor="arms.thermal", value=0.8, source="fire")
    assert out.success
    assert root.modulators["arms"].vital_metrics["thermal"] == pytest.approx(0.8)
    assert "arms.thermal" not in root.vital_metrics


def test_value_clamps_to_the_declared_range_not_unit():
    root, _, tool = _infant()
    assert tool.execute(sensor="core_temperature", value=-0.5, source="cold").success
    assert root.vital_metrics["core_temperature"] == pytest.approx(-0.5)
    assert tool.execute(sensor="arms.thermal", value=-2.0, source="ice").success
    assert root.modulators["arms"].vital_metrics["thermal"] == pytest.approx(-1.0)


def test_value_on_a_missing_sensor_fails_and_writes_nothing():
    root, _, tool = _infant()
    before = dict(root.vital_metrics)
    for name in ("temperature", "wings.thermal", "arms.wetness"):
        out = tool.execute(sensor=name, value=0.5)
        assert out.success is False and "not found" in out.error, name
    assert root.vital_metrics == before


def test_arm_heat_pain_follows_the_arm_as_it_cools():
    """The state-based channel (returned FailureEvents) must read the ARM's value.

    The PainBus latch holds peak severity during a breach by design, so this reads
    the per-call events. With the orphan key, evaluate_failures kept reading 0.8.
    """
    root, _, tool = _infant()
    assert tool.execute(sensor="arms.thermal", value=0.8, source="fire").success

    def arm_pain() -> float:
        events = tool._embodiment.evaluate_failures()
        return max((e.pain_intensity for e in events if e.failure_name == "drive:arms.thermal:discomfort"), default=0.0)

    hot = arm_pain()
    assert hot > 0.0
    root.modulators["arms"].vital_metrics["thermal"] = 0.6  # the arm cools (its homeostatic drift)
    assert arm_pain() < hot


def test_the_first_write_still_produces_arm_heat_pain():
    """Pinned before and after the fix: cradle Act 1's heat must hurt (#874 must not remove it)."""
    root, bus, tool = _infant()
    assert tool.execute(sensor="arms.thermal", value=0.8, source="fire").success
    assert root.drive_breach_severity["arms.thermal"] == pytest.approx(0.3)
    sources = [s.context.get("source") for s in bus.signals]
    assert "drive:arms.thermal" in sources


def test_value_still_sets_a_root_sensor():
    root, _, tool = _infant()
    assert tool.execute(sensor="hunger", value=0.0, source="food").success
    assert root.vital_metrics["hunger"] == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["value", "delta"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "lots", None, True])
def test_a_non_finite_or_non_numeric_value_is_rejected_and_writes_nothing(bad, mode):
    """NaN would pass the clamp as the range's MAXIMUM (min(1.0, nan) is 1.0) and report success."""
    if mode == "delta" and bad is None:
        pytest.skip("delta=None means value mode")
    root, _, tool = _infant()
    before = root.modulators["arms"].vital_metrics["thermal"]
    out = tool.execute(sensor="arms.thermal", **{mode: bad})
    assert out.success is False
    assert root.modulators["arms"].vital_metrics["thermal"] == before


def test_the_missing_sensor_error_names_only_sensors_the_tool_accepts():
    """A self-correcting orchestrator must never be told to use a name that then fails."""
    from maxim.embodiment.tool_bridge import _resolve_sensor_slot

    root, _, tool = _infant()
    tool._embodiment.evaluate_failures()  # writes the derived <mod>.integrity root keys
    out = tool.execute(sensor="wings.thermal", value=0.5)
    listed = out.error.split("it has: ", 1)[1].split(", ")
    assert "arms.thermal" in listed and "hunger" in listed
    assert "health" not in listed  # derived: listing it would steer the orchestrator into a no-op
    for name in listed:
        assert _resolve_sensor_slot(root, name) is not None, name


@pytest.mark.parametrize("kwargs", [{"value": 0.2}, {"delta": -0.3}])
def test_a_derived_health_write_fails_instead_of_succeeding_as_a_no_op(kwargs):
    """evaluate_failures re-derives ``health`` from component integrity on every call."""
    root, _, tool = _infant()
    tool._embodiment.evaluate_failures()  # health now sits on the root, derived
    derived = root.vital_metrics["health"]
    out = tool.execute(sensor="health", **kwargs)
    assert out.success is False and "derived" in out.error
    assert root.vital_metrics["health"] == pytest.approx(derived)


def test_a_derived_health_body_without_components_keeps_health_writable():
    """``derive_health()`` is None without component sensors, so evaluate_failures never overwrites it."""

    class _Body:
        name = full_path = "blob"
        metadata = {"health": "derived"}
        sensors: dict = {}
        modulators: dict = {}

        def __init__(self) -> None:
            self.vital_metrics = {"health": 0.5}

        def derive_health(self):
            return None

    class _Emb:
        def __init__(self, root) -> None:
            self.root = root

        def evaluate_failures(self):
            return []

    body = _Body()
    out = SetEntitySensorTool(embodiment=_Emb(body), entity_map=None).execute(sensor="health", value=0.9)
    assert out.success
    assert body.vital_metrics["health"] == pytest.approx(0.9)
