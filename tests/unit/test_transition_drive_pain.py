"""Transition-based drive-pain emission (docs/plans/deferred/transition_based_drive_pain.md).

Drive-pain FailureEvents + PainBus publishes fire on band ENTRY (within-band →
out-of-band transition), latched per ``(entity_path, drive_name)`` on the Body,
and clear on the reverse transition. This makes attribution land on the action
that caused the crossing instead of blaming every bystander action that executes
while a breach lingers (the pre-B8 Exp 42 collapse, on BOTH channels).

What must NOT change (pinned here too):
- Motivation is carried by the drive VALUE (body_state_summary /
  _read_drive_states), not the FailureEvent — a latched-out sensor still
  reports its discomfort every read.
- Standard (non-drive) failure modes keep their per-call trigger semantics.
- Re-injury after recovery is a new transition and fires again.

These tests were verified to FAIL on the pre-fix state-based emitter
(re-fire every call) before the latch landed.
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


class TestHomeostaticTransitionLatch:
    def test_lingering_breach_fires_once(self):
        bus = MagicMock()
        emb = _make_homeostatic_body(initial=0.9, pain_bus=bus)

        first = _drive_events(emb.evaluate_failures())
        second = _drive_events(emb.evaluate_failures())
        third = _drive_events(emb.evaluate_failures())

        assert len(first) == 1
        assert first[0].failure_name == "drive:temperature:discomfort"
        assert second == []
        assert third == []
        assert bus.publish.call_count == 1

    def test_recovery_clears_latch_and_rebreach_fires_again(self):
        bus = MagicMock()
        emb = _make_homeostatic_body(initial=0.9, pain_bus=bus)

        assert len(_drive_events(emb.evaluate_failures())) == 1

        # Recover within band: no event, latch clears.
        emb.root.vital_metrics["temperature"] = 0.1
        assert _drive_events(emb.evaluate_failures()) == []

        # Genuine re-injury: a NEW transition fires again.
        emb.root.vital_metrics["temperature"] = -0.9
        rebreach = _drive_events(emb.evaluate_failures())
        assert len(rebreach) == 1
        assert bus.publish.call_count == 2

    def test_bystander_call_sees_no_drive_event(self):
        """A second evaluate (e.g. inside a bystander affordance's execute)
        during a lingering breach returns NO drive FailureEvent — the direct
        channel cannot blame the bystander, with no per-consumer filter."""
        emb = _make_homeostatic_body(initial=0.0)

        # Causing action pushes out of band, then evaluates (tool_bridge order).
        emb.root.vital_metrics["temperature"] = 0.9
        causing = _drive_events(emb.evaluate_failures())
        assert len(causing) == 1

        # Bystander executes while the breach lingers.
        bystander = _drive_events(emb.evaluate_failures())
        assert bystander == []

    def test_within_band_never_latches(self):
        emb = _make_homeostatic_body(initial=0.1)
        assert _drive_events(emb.evaluate_failures()) == []
        assert _drive_events(emb.evaluate_failures()) == []


class TestEntropicTransitionLatch:
    def test_lingering_deprivation_fires_once(self):
        bus = MagicMock()
        emb = _make_entropic_body(initial=0.8, pain_bus=bus)

        first = _drive_events(emb.evaluate_failures())
        second = _drive_events(emb.evaluate_failures())

        assert len(first) == 1
        assert first[0].failure_name == "drive:hunger:deprived"
        assert second == []
        assert bus.publish.call_count == 1

    def test_satisfaction_clears_latch_and_redeprivation_fires(self):
        emb = _make_entropic_body(initial=0.8)
        assert len(_drive_events(emb.evaluate_failures())) == 1

        # Fed: back under threshold → latch clears silently.
        emb.root.vital_metrics["hunger"] = 0.2
        assert _drive_events(emb.evaluate_failures()) == []

        # Deprived again → new transition fires.
        emb.root.vital_metrics["hunger"] = 0.95
        assert len(_drive_events(emb.evaluate_failures())) == 1

    def test_drift_direction_down(self):
        data = {
            "name": "test_body",
            "entity_type": "body",
            "sensors": {
                "warmth": {
                    "unit": "ratio",
                    "range": [0, 1],
                    "initial": 0.1,
                    "drive": {
                        "drift_mode": "entropic",
                        "drift_direction": "down",
                        "drift_rate": 0.0,
                        "deprivation_threshold": 0.3,
                        "deprivation_pain": 0.4,
                        "satisfaction_threshold": 0.8,
                    },
                },
            },
        }
        emb = Embodiment(_parse_entity(data))
        assert len(_drive_events(emb.evaluate_failures())) == 1
        assert _drive_events(emb.evaluate_failures()) == []
        emb.root.vital_metrics["warmth"] = 0.9
        assert _drive_events(emb.evaluate_failures()) == []
        emb.root.vital_metrics["warmth"] = 0.05
        assert len(_drive_events(emb.evaluate_failures())) == 1


class TestLatchScopeAndMotivation:
    def test_latch_is_per_drive(self):
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
        emb = Embodiment(_parse_entity(data))

        # Only temperature is breached at start.
        first = _drive_events(emb.evaluate_failures())
        assert [e.failure_name for e in first] == ["drive:temperature:discomfort"]

        # Hunger breaching later fires independently; temperature stays latched.
        emb.root.vital_metrics["hunger"] = 0.9
        second = _drive_events(emb.evaluate_failures())
        assert [e.failure_name for e in second] == ["drive:hunger:deprived"]

    def test_latch_resets_on_new_body_instance(self):
        emb1 = _make_homeostatic_body(initial=0.9)
        assert len(_drive_events(emb1.evaluate_failures())) == 1

        emb2 = _make_homeostatic_body(initial=0.9)
        assert len(_drive_events(emb2.evaluate_failures())) == 1

    def test_motivation_value_unchanged_while_latched(self):
        """The drive VALUE keeps reporting discomfort every read — onset-only
        emission must not touch the motivational signal."""
        emb = _make_homeostatic_body(initial=0.9)
        emb.evaluate_failures()
        emb.evaluate_failures()  # latched out — no new events

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
