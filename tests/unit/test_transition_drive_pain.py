"""Transition-based drive-pain emission (docs/plans/deferred/transition_based_drive_pain.md).

CHANNEL SPLIT (settled in the pre-merge two-lens fold — both lenses
independently reproduced a sign inversion in the first design, which latched
BOTH channels):

- **Channel 1** — the ``FailureEvent`` list returned by ``evaluate_failures``
  (direct attribution via ``ToolOutput.side_effects``) stays **state-based**,
  one event per call while breached. ``tool_bridge``'s B8 delta filter already
  attributes this channel correctly and is state-INDEPENDENT, so it keeps
  working when a sensor saturates. Latching it starved that filter and flipped
  a *repeat* harmful affordance to POSITIVE credit.
- **Channel 2** — ``_publish_drive_pain`` (the unfiltered PainBus path this
  plan was written to fix) is latched on **severity**: it fires on band entry
  and again only when the breach materially DEEPENS (re-injury), with
  hysteresis on the recovery point. Steady-state and recovery are silent,
  which kills the per-tick flood.

The latch lives on the **Entity** (``drive_breach_severity``), not the
``Embodiment`` wrapper, so it survives reparenting and ephemeral
per-invocation wrappers and cannot collide between same-named siblings.

Motivation is untouched by either channel: it rides the drive VALUE
(``body_state_summary`` / ``_read_drive_states``), pinned below.

The channel-1 repeat-causer guard (the regression both lenses caught) lives in
tests/unit/test_substrate_primary_scene_harm.py, end-to-end through the real
``ModulatorAffordanceTool.execute`` on the Exp 42 fixtures.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from maxim.embodiment.body import Embodiment
from maxim.embodiment.spec import _parse_entity


def _drive_events(events) -> list:
    return [e for e in events if e.failure_name.startswith("drive:")]


def _make_homeostatic_body(initial: float = 0.0, *, drift_rate: float = 0.0, pain_bus=None) -> Embodiment:
    """Body with one homeostatic drive. drift_rate=0 keeps a breach lingering
    across calls (the world-set-sensor shape, e.g. reachy azimuth)."""
    data = {
        "name": "test_body",
        "entity_type": "body",
        "sensors": {
            "temperature": {
                "unit": "celsius_norm",
                "range": [-1, 1],
                "initial": initial,
                "drive": {
                    "drift_mode": "homeostatic",
                    "set_point": 0.0,
                    "drift_rate": drift_rate,
                    "comfort_band": 0.3,
                    "pain_scale": 0.5,
                },
            },
        },
    }
    return Embodiment(_parse_entity(data), pain_bus=pain_bus)


def _make_entropic_body(initial: float = 0.8, *, pain_bus=None) -> Embodiment:
    data = {
        "name": "test_body",
        "entity_type": "body",
        "sensors": {
            "hunger": {
                "unit": "ratio",
                "range": [0, 1],
                "initial": initial,
                "drive": {
                    "drift_mode": "entropic",
                    "drift_direction": "up",
                    "drift_rate": 0.0,
                    "deprivation_threshold": 0.7,
                    "deprivation_pain": 0.3,
                    "satisfaction_threshold": 0.3,
                },
            },
        },
    }
    return Embodiment(_parse_entity(data), pain_bus=pain_bus)


class TestChannelOneStaysStateBased:
    """The returned FailureEvent list must NOT be latched — B8 filters it."""

    def test_lingering_breach_still_returns_event_every_call(self):
        emb = _make_homeostatic_body(initial=0.9)
        for i in range(4):
            evs = _drive_events(emb.evaluate_failures())
            assert len(evs) == 1, f"call {i}: direct channel must stay state-based"
            assert evs[0].failure_name == "drive:temperature:discomfort"

    def test_entropic_lingering_deprivation_returns_event_every_call(self):
        emb = _make_entropic_body(initial=0.8)
        for i in range(3):
            evs = _drive_events(emb.evaluate_failures())
            assert len(evs) == 1, f"call {i}: direct channel must stay state-based"
            assert evs[0].failure_name == "drive:hunger:deprived"

    def test_saturated_sensor_still_returns_event(self):
        """At clamp, severity cannot deepen — the direct channel must still
        report, which is exactly why it is not severity-latched."""
        emb = _make_homeostatic_body(initial=1.0)
        assert len(_drive_events(emb.evaluate_failures())) == 1
        assert len(_drive_events(emb.evaluate_failures())) == 1


class TestChannelTwoSeverityLatch:
    """_publish_drive_pain fires on entry + material deepening only."""

    def test_lingering_breach_publishes_once(self):
        bus = MagicMock()
        emb = _make_homeostatic_body(initial=0.9, pain_bus=bus)
        for _ in range(4):
            emb.evaluate_failures()
        assert bus.publish.call_count == 1

    def test_deepening_republishes(self):
        """Re-injury (a materially worse breach) must be heard — a body that
        goes silent when injured further is anaesthesia, not onset-coding."""
        bus = MagicMock()
        emb = _make_homeostatic_body(initial=0.5, pain_bus=bus)
        emb.evaluate_failures()
        assert bus.publish.call_count == 1

        emb.root.vital_metrics["temperature"] = 0.95  # deepened well past eps
        emb.evaluate_failures()
        assert bus.publish.call_count == 2

        emb.evaluate_failures()  # unchanged — silent
        assert bus.publish.call_count == 2

    def test_marginal_deepening_does_not_republish(self):
        """Per-tick drift creep must not re-fire (eps = 5% of comfort_band)."""
        bus = MagicMock()
        emb = _make_homeostatic_body(initial=0.9, pain_bus=bus)
        emb.evaluate_failures()
        emb.root.vital_metrics["temperature"] = 0.9005
        emb.evaluate_failures()
        assert bus.publish.call_count == 1

    def test_recovery_then_rebreach_republishes(self):
        bus = MagicMock()
        emb = _make_homeostatic_body(initial=0.9, pain_bus=bus)
        emb.evaluate_failures()
        emb.root.vital_metrics["temperature"] = 0.0  # comfortably inside band
        emb.evaluate_failures()
        emb.root.vital_metrics["temperature"] = -0.9
        emb.evaluate_failures()
        assert bus.publish.call_count == 2

    def test_band_edge_jitter_does_not_chatter(self):
        """Hysteresis: a noisy world-set sensor sitting on the band edge (live
        DoA azimuth is exactly this) must not emit one 'onset' per jitter."""
        bus = MagicMock()
        emb = _make_homeostatic_body(initial=0.31, pain_bus=bus)
        emb.evaluate_failures()
        assert bus.publish.call_count == 1
        for v in (0.29, 0.31, 0.28, 0.32, 0.295, 0.305):
            emb.root.vital_metrics["temperature"] = v
            emb.evaluate_failures()
        assert bus.publish.call_count == 1, "band-edge jitter must not re-publish"

    def test_entropic_clears_on_satisfaction_threshold_not_deprivation(self):
        """The spec's own declared recovery point is the hysteresis."""
        bus = MagicMock()
        emb = _make_entropic_body(initial=0.8, pain_bus=bus)
        emb.evaluate_failures()
        assert bus.publish.call_count == 1

        # Between satisfaction (0.3) and deprivation (0.7): NOT yet recovered.
        emb.root.vital_metrics["hunger"] = 0.5
        emb.evaluate_failures()
        emb.root.vital_metrics["hunger"] = 0.75
        emb.evaluate_failures()
        assert bus.publish.call_count == 1, "must not re-fire without crossing satisfaction_threshold"

        # Fed past satisfaction → latch clears → next deprivation re-fires.
        emb.root.vital_metrics["hunger"] = 0.2
        emb.evaluate_failures()
        emb.root.vital_metrics["hunger"] = 0.8
        emb.evaluate_failures()
        assert bus.publish.call_count == 2

    def test_entropic_deepening_republishes(self):
        bus = MagicMock()
        emb = _make_entropic_body(initial=0.75, pain_bus=bus)
        emb.evaluate_failures()
        emb.root.vital_metrics["hunger"] = 1.0
        emb.evaluate_failures()
        assert bus.publish.call_count == 2

    def test_within_band_never_publishes(self):
        bus = MagicMock()
        emb = _make_homeostatic_body(initial=0.1, pain_bus=bus)
        emb.evaluate_failures()
        emb.evaluate_failures()
        assert bus.publish.call_count == 0
        assert _drive_events(emb.evaluate_failures()) == []


class TestLatchIdentityAndLifecycle:
    def test_latch_lives_on_entity_not_wrapper(self):
        """An ephemeral per-invocation Embodiment wrapper (simulation/tools.py
        constructs these) must not resurrect state-based publishing."""
        bus = MagicMock()
        entity = _parse_entity(
            {
                "name": "test_body",
                "entity_type": "body",
                "sensors": {
                    "temperature": {
                        "unit": "celsius_norm",
                        "range": [-1, 1],
                        "initial": 0.9,
                        "drive": {
                            "drift_mode": "homeostatic",
                            "set_point": 0.0,
                            "drift_rate": 0.0,
                            "comfort_band": 0.3,
                            "pain_scale": 0.5,
                        },
                    },
                },
            }
        )
        Embodiment(entity, pain_bus=bus).evaluate_failures()
        Embodiment(entity, pain_bus=bus).evaluate_failures()  # fresh wrapper, same body
        assert bus.publish.call_count == 1

    def test_same_name_siblings_do_not_collide(self):
        """Two entities sharing a name (and therefore a full_path) must keep
        independent latches — a path-keyed latch silently erased one another."""
        bus = MagicMock()
        root = _parse_entity({"name": "body", "entity_type": "body"})
        child_spec = {
            "name": "torch",
            "entity_type": "item",
            "sensors": {
                "heat": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "initial": 0.0,
                    "drive": {
                        "drift_mode": "homeostatic",
                        "set_point": 0.0,
                        "drift_rate": 0.0,
                        "comfort_band": 0.3,
                        "pain_scale": 0.5,
                    },
                },
            },
        }
        t1 = _parse_entity(child_spec)
        t2 = _parse_entity(child_spec)
        for t in (t1, t2):
            t.parent = root
            root.children.append(t)
        assert t1.full_path == t2.full_path  # the collision precondition

        emb = Embodiment(root, pain_bus=bus)
        t1.vital_metrics["heat"] = 0.9
        emb.evaluate_failures()
        assert bus.publish.call_count == 1

        # t2's genuine first breach must still be heard.
        t2.vital_metrics["heat"] = 0.9
        emb.evaluate_failures()
        assert bus.publish.call_count == 2

        # Neither re-fires while both merely persist.
        emb.evaluate_failures()
        assert bus.publish.call_count == 2

    def test_reparent_does_not_spuriously_refire(self):
        """Entity acquisition reparents entities (full_path changes); the latch
        must travel with the entity, not its path."""
        bus = MagicMock()
        root = _parse_entity({"name": "body", "entity_type": "body"})
        hand = _parse_entity({"name": "hand", "entity_type": "part"})
        hand.parent = root
        root.children.append(hand)
        torch = _parse_entity(
            {
                "name": "torch",
                "entity_type": "item",
                "sensors": {
                    "heat": {
                        "unit": "ratio",
                        "range": [0, 1],
                        "initial": 0.9,
                        "drive": {
                            "drift_mode": "homeostatic",
                            "set_point": 0.0,
                            "drift_rate": 0.0,
                            "comfort_band": 0.3,
                            "pain_scale": 0.5,
                        },
                    },
                },
            }
        )
        torch.parent = root
        root.children.append(torch)

        emb = Embodiment(root, pain_bus=bus)
        emb.evaluate_failures()
        assert bus.publish.call_count == 1

        torch.reparent(hand)
        emb.evaluate_failures()
        assert bus.publish.call_count == 1, "reparenting is not a new injury"

    def test_unreadable_sensor_clears_latch(self):
        """A drive whose sensor stops reporting must not strand its latch —
        otherwise the next genuine breach is swallowed forever."""
        bus = MagicMock()
        emb = _make_homeostatic_body(initial=0.9, pain_bus=bus)
        emb.evaluate_failures()
        assert bus.publish.call_count == 1

        # Sensor disappears (backend error / modulator removed).
        emb.root.sensors.pop("temperature", None)
        emb.root.vital_metrics.pop("temperature", None)
        emb.evaluate_failures()

        # Comes back, still breached → must be heard as a fresh breach.
        emb.root.vital_metrics["temperature"] = 0.95
        emb.evaluate_failures()
        assert bus.publish.call_count == 2

    def test_latch_not_serialized(self):
        """Session-runtime state only — never persisted (mirrors FailureMode.active)."""
        emb = _make_homeostatic_body(initial=0.9)
        emb.evaluate_failures()
        assert emb.root.drive_breach_severity
        assert "drive_breach_severity" not in emb.root.to_dict()

    def test_latch_is_per_drive(self):
        bus = MagicMock()
        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "temperature": {
                    "unit": "celsius_norm",
                    "range": [-1, 1],
                    "initial": 0.9,
                    "drive": {
                        "drift_mode": "homeostatic",
                        "set_point": 0.0,
                        "drift_rate": 0.0,
                        "comfort_band": 0.3,
                        "pain_scale": 0.5,
                    },
                },
                "hunger": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "initial": 0.1,
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "up",
                        "drift_rate": 0.0,
                        "deprivation_threshold": 0.7,
                        "deprivation_pain": 0.3,
                        "satisfaction_threshold": 0.3,
                    },
                },
            },
        }
        emb = Embodiment(_parse_entity(data), pain_bus=bus)
        emb.evaluate_failures()
        assert bus.publish.call_count == 1  # temperature only

        emb.root.vital_metrics["hunger"] = 0.9
        emb.evaluate_failures()
        assert bus.publish.call_count == 2  # hunger fires independently

        emb.evaluate_failures()
        assert bus.publish.call_count == 2


class TestMotivationPreserved:
    def test_motivation_value_unchanged_while_latched(self):
        """The drive VALUE keeps reporting discomfort every read — latching the
        pain channel must not touch the motivational signal."""
        emb = _make_homeostatic_body(initial=0.9)
        emb.evaluate_failures()
        emb.evaluate_failures()

        summary = emb.body_state_summary()
        temp = summary[0]["sensors"]["temperature"]
        assert "outside comfort band" in temp["drive"]
        assert temp["value"] == 0.9

    def test_standard_failure_modes_still_fire_per_call(self):
        """Non-drive failure modes are OUT of scope: their per-call trigger
        semantics are preserved (they have their own trigger conditions)."""
        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "integrity": {"unit": "ratio", "range": [0, 1], "initial": 0.1},
            },
            "failure_modes": [
                {
                    "name": "broken",
                    "trigger": {"field": "integrity", "op": "<", "value": 0.2, "pain": 0.6},
                },
            ],
        }
        emb = Embodiment(_parse_entity(data))
        first = [e for e in emb.evaluate_failures() if e.failure_name == "broken"]
        second = [e for e in emb.evaluate_failures() if e.failure_name == "broken"]
        assert len(first) == 1
        assert len(second) == 1
