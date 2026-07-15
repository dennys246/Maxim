"""Tests for Roy-5b interoceptive-vocalization scaffolding.

Pin the deterministic naming-event emitter, its YAML metadata parse,
the body loadability, the Roy-5b spec parseability, the modulator-aware
sensor collector, and the EmbodimentPerceptSource integration: when a
body declares ``naming_events:``, utterances appear in the body-state
percept text in the SAME tick the drive crosses threshold — this is the
within-tick co-firing guarantee Stage 3's Hebbian retest depends on.

See docs/plans/archive/roy_5_encoder_alignment_disambiguator.md Stage 3.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from maxim.embodiment.naming_events import (
    NamingPattern,
    collect_sensor_values,
    derive_naming_utterances,
    format_naming_section,
    parse_naming_patterns,
)


# ---------------------------------------------------------------------------
# parse_naming_patterns — YAML metadata parsing
# ---------------------------------------------------------------------------


class TestParseNamingPatterns:
    def test_empty_returns_empty_tuple(self):
        assert parse_naming_patterns(None) == ()
        assert parse_naming_patterns([]) == ()

    def test_valid_minimal_entry(self):
        raw = [{"sensor_key": "hunger", "utterance": "hungry", "threshold": 0.7}]
        patterns = parse_naming_patterns(raw)
        assert len(patterns) == 1
        assert patterns[0].sensor_key == "hunger"
        assert patterns[0].utterance == "hungry"
        assert patterns[0].threshold == 0.7
        assert patterns[0].direction == "above"  # default
        assert patterns[0].hysteresis_band == 0.2  # default

    def test_valid_full_entry(self):
        raw = [
            {
                "sensor_key": "arms.thermal",
                "utterance": "warm",
                "threshold": 0.5,
                "direction": "below",
                "hysteresis_band": 0.15,
            }
        ]
        patterns = parse_naming_patterns(raw)
        assert patterns[0].direction == "below"
        assert patterns[0].hysteresis_band == 0.15

    def test_multiple_entries_preserves_order(self):
        raw = [
            {"sensor_key": "hunger", "utterance": "hungry", "threshold": 0.7},
            {"sensor_key": "thirst", "utterance": "thirsty", "threshold": 0.6},
            {"sensor_key": "arms.thermal", "utterance": "warm", "threshold": 0.7},
        ]
        patterns = parse_naming_patterns(raw)
        assert [p.sensor_key for p in patterns] == ["hunger", "thirst", "arms.thermal"]

    def test_rejects_non_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            parse_naming_patterns({"sensor_key": "hunger"})

    def test_rejects_non_dict_entry(self):
        with pytest.raises(ValueError, match="must be a dict"):
            parse_naming_patterns(["not a dict"])

    def test_rejects_missing_required_field(self):
        with pytest.raises(ValueError, match="missing required field"):
            parse_naming_patterns([{"sensor_key": "hunger", "utterance": "hungry"}])

    def test_rejects_invalid_direction(self):
        raw = [
            {
                "sensor_key": "hunger",
                "utterance": "hungry",
                "threshold": 0.7,
                "direction": "sideways",
            }
        ]
        with pytest.raises(ValueError, match="direction must be"):
            parse_naming_patterns(raw)

    def test_rejects_negative_hysteresis_band(self):
        raw = [
            {
                "sensor_key": "hunger",
                "utterance": "hungry",
                "threshold": 0.7,
                "hysteresis_band": -0.1,
            }
        ]
        with pytest.raises(ValueError, match="hysteresis_band must be >= 0"):
            parse_naming_patterns(raw)

    def test_source_prefix_in_error_message(self):
        """F1 fold (executor review): parse errors mention which body
        produced them so operators don't see a context-free ValueError."""
        with pytest.raises(ValueError, match="my_body:"):
            parse_naming_patterns([{"sensor_key": "hunger", "utterance": "hungry"}], source="my_body")


# ---------------------------------------------------------------------------
# derive_naming_utterances — threshold crossing + hysteresis
# ---------------------------------------------------------------------------


class TestDeriveNamingUtterances:
    def test_no_patterns_returns_empty(self):
        utterances, fired = derive_naming_utterances({"hunger": 0.9}, (), set())
        assert utterances == ()
        assert fired == set()

    def test_fires_on_above_threshold_crossing(self):
        patterns = (NamingPattern("hunger", "hungry", 0.7),)
        utterances, fired = derive_naming_utterances({"hunger": 0.75}, patterns, set())
        assert utterances == ("hungry",)
        assert fired == {"hunger"}

    def test_silent_below_threshold(self):
        patterns = (NamingPattern("hunger", "hungry", 0.7),)
        utterances, fired = derive_naming_utterances({"hunger": 0.5}, patterns, set())
        assert utterances == ()
        assert fired == set()

    def test_hysteresis_suppresses_repeat(self):
        # Once fired, staying above threshold doesn't re-fire.
        patterns = (NamingPattern("hunger", "hungry", 0.7, hysteresis_band=0.2),)
        utterances, fired = derive_naming_utterances({"hunger": 0.75}, patterns, {"hunger"})
        assert utterances == ()
        assert fired == {"hunger"}

    def test_hysteresis_rearms_after_drop_below_band(self):
        patterns = (NamingPattern("hunger", "hungry", 0.7, hysteresis_band=0.2),)
        # Drop to 0.45 — below 0.7 - 0.2 = 0.5, re-arms.
        utterances, fired = derive_naming_utterances({"hunger": 0.45}, patterns, {"hunger"})
        assert utterances == ()
        assert fired == set()
        # Now crossing again fires.
        utterances, fired = derive_naming_utterances({"hunger": 0.75}, patterns, fired)
        assert utterances == ("hungry",)

    def test_does_not_rearm_inside_hysteresis_band(self):
        patterns = (NamingPattern("hunger", "hungry", 0.7, hysteresis_band=0.2),)
        # 0.55 is below threshold but inside the [0.5, 0.7] hysteresis band.
        utterances, fired = derive_naming_utterances({"hunger": 0.55}, patterns, {"hunger"})
        assert utterances == ()
        assert fired == {"hunger"}  # still latched

    def test_below_direction(self):
        patterns = (NamingPattern("temperature", "cold", 0.2, direction="below"),)
        utterances, fired = derive_naming_utterances({"temperature": 0.1}, patterns, set())
        assert utterances == ("cold",)
        assert fired == {"temperature"}

    def test_below_rearms_after_rise_above_band(self):
        patterns = (NamingPattern("temperature", "cold", 0.2, direction="below", hysteresis_band=0.2),)
        # Rise to 0.5 — above 0.2 + 0.2 = 0.4, re-arms.
        utterances, fired = derive_naming_utterances({"temperature": 0.5}, patterns, {"temperature"})
        assert fired == set()

    def test_dotted_sensor_key_matches_flat_dict(self):
        patterns = (NamingPattern("arms.thermal", "warm", 0.7),)
        utterances, fired = derive_naming_utterances({"arms.thermal": 0.85}, patterns, set())
        assert utterances == ("warm",)
        assert fired == {"arms.thermal"}

    def test_missing_sensor_silently_skipped(self):
        patterns = (NamingPattern("hunger", "hungry", 0.7),)
        utterances, fired = derive_naming_utterances({"thirst": 0.9}, patterns, set())
        assert utterances == ()
        assert fired == set()

    def test_multiple_patterns_fire_independently(self):
        patterns = (
            NamingPattern("hunger", "hungry", 0.7),
            NamingPattern("thirst", "thirsty", 0.6),
            NamingPattern("arms.thermal", "warm", 0.7),
        )
        # arms.thermal=0.4 didn't cross 0.7, so only hungry + thirsty fire.
        utterances, fired = derive_naming_utterances(
            {"hunger": 0.8, "thirst": 0.65, "arms.thermal": 0.4}, patterns, set()
        )
        assert set(utterances) == {"hungry", "thirsty"}
        assert fired == {"hunger", "thirst"}


# ---------------------------------------------------------------------------
# format_naming_section — text rendering
# ---------------------------------------------------------------------------


class TestFormatNamingSection:
    def test_empty_returns_empty_string(self):
        assert format_naming_section(()) == ""

    def test_single_utterance(self):
        result = format_naming_section(("hungry",))
        # Header is neutrally framed (review cross-confirmed: avoid
        # "Caregiver" semantics which would leak into the substrate).
        assert "=== Body Utterances ===" in result
        assert "Caregiver" not in result
        assert "- hungry" in result

    def test_multiple_utterances_one_per_line(self):
        result = format_naming_section(("hungry", "thirsty"))
        lines = result.split("\n")
        assert lines[0] == "=== Body Utterances ==="
        assert "- hungry" in lines
        assert "- thirsty" in lines


# ---------------------------------------------------------------------------
# collect_sensor_values — modulator-aware sensor walk
# (B1 fold: body_state_summary skips dotted modulator keys; the
#  collector must walk modulator vital_metrics directly.)
# ---------------------------------------------------------------------------


class TestCollectSensorValues:
    @pytest.fixture
    def real_body(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.component_registry import ComponentRegistry

        reg = ComponentRegistry()
        entity = reg.instantiate("bodies/infant_humanoid_naming_v1")
        return Embodiment(entity), entity

    def test_entity_level_drives_collected(self, real_body):
        body, entity = real_body
        entity.vital_metrics["hunger"] = 0.8
        entity.vital_metrics["thirst"] = 0.65
        values = collect_sensor_values(body)
        assert values["hunger"] == pytest.approx(0.8)
        assert values["thirst"] == pytest.approx(0.65)

    def test_modulator_sub_sensors_collected_via_dotted_key(self, real_body):
        """The load-bearing B1 fold: ``arms.thermal`` lives in
        ``entity.modulators['arms'].vital_metrics['thermal']``, NOT in
        ``body_state_summary``'s output. The collector MUST surface it
        or "warm" silently never fires."""
        body, entity = real_body
        entity.modulators["arms"].vital_metrics["thermal"] = 0.85
        values = collect_sensor_values(body)
        assert values["arms.thermal"] == pytest.approx(0.85)
        # Other modulator sub-sensors should also surface.
        assert "arms.pressure" in values
        assert "head.thermal" in values


# ---------------------------------------------------------------------------
# Body YAML loadability — pin the new component file
# ---------------------------------------------------------------------------


class TestInfantNamingBodyYaml:
    @pytest.fixture
    def body_path(self) -> Path:
        import maxim

        pkg_root = Path(maxim.__file__).parent
        return pkg_root / "_data" / "components" / "bodies" / "infant_humanoid_naming_v1.yaml"

    def test_yaml_parses(self, body_path):
        assert body_path.exists()
        with body_path.open() as f:
            data = yaml.safe_load(f)
        assert "component" in data
        assert "entity" in data
        assert data["component"]["name"] == "infant_humanoid_naming_v1"

    def test_extends_canonical_infant_body(self, body_path):
        """F6 fold (executor review): extend bodies/infant_humanoid not
        bodies/base_humanoid so drive-curve tuning on the canonical
        cradle body propagates here and the spec's "single-variable
        change vs Roy-4" promise holds."""
        with body_path.open() as f:
            data = yaml.safe_load(f)
        assert data["component"]["extends"] == "bodies/infant_humanoid"

    def test_declares_naming_events(self, body_path):
        with body_path.open() as f:
            data = yaml.safe_load(f)
        naming_events = data["entity"].get("naming_events")
        assert naming_events is not None
        assert len(naming_events) == 3
        keys = {e["sensor_key"] for e in naming_events}
        assert keys == {"hunger", "thirst", "arms.thermal"}

    def test_naming_events_parse_via_helper(self, body_path):
        with body_path.open() as f:
            data = yaml.safe_load(f)
        patterns = parse_naming_patterns(data["entity"]["naming_events"])
        assert len(patterns) == 3
        utterances = {p.utterance for p in patterns}
        assert utterances == {"hungry", "thirsty", "warm"}

    def test_resolves_via_component_registry(self, body_path):
        """Loadability through the real ComponentRegistry — catches
        ``extends:`` resolution + metadata inheritance failures the
        YAML-only parse misses."""
        from maxim.embodiment.component_registry import ComponentRegistry

        reg = ComponentRegistry()
        entity = reg.instantiate("bodies/infant_humanoid_naming_v1")
        assert entity.name == "infant_humanoid_naming_v1"
        # Inherited drive_specs from infant_humanoid.
        assert "hunger" in entity.drive_specs
        assert "arms.thermal" in entity.drive_specs
        # Own naming_events metadata.
        assert entity.metadata.get("naming_events") is not None


# ---------------------------------------------------------------------------
# Roy-5b spec loadability
# ---------------------------------------------------------------------------


class TestRoy5bSpec:
    @pytest.fixture
    def spec_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "scenarios" / "roy" / "roy_5b_iteration.yaml"

    def test_yaml_parses(self, spec_path):
        assert spec_path.exists()
        with spec_path.open() as f:
            data = yaml.safe_load(f)
        assert data["name"] == "roy-5b"

    def test_embodiment_is_naming_v1_body(self, spec_path):
        with spec_path.open() as f:
            data = yaml.safe_load(f)
        assert data["embodiment"] == "bodies/infant_humanoid_naming_v1"
        # Priming stage also references the same body, so substrate
        # encoding is consistent across priming arcs.
        assert data["priming"]["embodiment"] == "bodies/infant_humanoid_naming_v1"

    def test_substrate_primary_mode(self, spec_path):
        """Substrate-primary matches Roy-4. The naming utterances reach
        the AUT via the body_state percept path, NOT via the
        send_and_wait text channel that substrate-primary suppresses."""
        with spec_path.open() as f:
            data = yaml.safe_load(f)
        assert data["aut_mode"] == "substrate-primary"
        assert data["priming"]["aut_mode"] == "substrate-primary"

    def test_reuses_roy_2pc_holdout_fixture(self, spec_path):
        """Same engineered-overlap fixture as Roy-2c / Roy-4. Single-
        variable comparison requires every other input identical."""
        with spec_path.open() as f:
            data = yaml.safe_load(f)
        assert data["test_scenario"]["fixture"] == "roy_2pc_holdout.yaml"

    def test_three_arm_shape_matches_roy_4(self, spec_path):
        with spec_path.open() as f:
            data = yaml.safe_load(f)
        assert set(data["arms"].keys()) == {"a", "b", "c"}
        assert data["arms"]["a"]["substrate"] == "from_priming"
        assert data["arms"]["b"]["substrate"] == "blank"
        assert data["arms"]["c"]["substrate"] == "blank"


# ---------------------------------------------------------------------------
# EmbodimentPerceptSource integration — co-firing in body_state text
# ---------------------------------------------------------------------------


class TestPerceptSourceNamingHookMock:
    def test_no_naming_metadata_byte_identical_percept(self):
        """The pre-Stage-3 percept content shape is preserved when a
        body has no naming_events metadata."""
        from maxim.embodiment.percepts import EmbodimentPerceptSource

        mock = MagicMock()
        mock.root.name = "bare_body"
        mock.root.metadata = {}  # no naming_events
        mock.evaluate_failures.return_value = []
        mock.format_body_state_for_prompt.return_value = "BODY"
        mock.tick_vital_drift = MagicMock()

        source = EmbodimentPerceptSource(mock, poll_hz=1000.0)
        percept = source.next_percept()
        assert percept is not None
        assert "Body Utterances" not in percept.content
        assert "Caregiver" not in percept.content


# ---------------------------------------------------------------------------
# B2 fold: within-tick co-firing on a REAL Embodiment
# (the existing MagicMock test only string-checks; this test pins the
#  structural claim that body_state + utterance land in the SAME
#  percept content the agent loop sees in ONE next_percept call.)
# ---------------------------------------------------------------------------


class TestNamingEventsWithinTickCoFiringIntegration:
    @pytest.fixture
    def real_naming_body(self):
        """Build a real Embodiment + EmbodimentPerceptSource on the
        Roy-5b body. Returns the source and the entity ref so tests
        can mutate vital_metrics to force drive crossings without
        running real drift physics."""
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.embodiment.percepts import EmbodimentPerceptSource

        reg = ComponentRegistry()
        entity = reg.instantiate("bodies/infant_humanoid_naming_v1")
        body = Embodiment(entity)
        # Disable drift during the test so the value we write is the
        # value the next_percept call observes.
        body.tick_vital_drift = lambda dt=0.0: None  # type: ignore[assignment]
        # Poll at high rate so the poll-interval guard doesn't reject.
        source = EmbodimentPerceptSource(body, poll_hz=1000.0)
        return source, entity, body

    def test_no_utterance_below_thresholds(self, real_naming_body):
        source, entity, _ = real_naming_body
        # All drives start at 0; thresholds at 0.7/0.6/0.7.
        percept = source.next_percept()
        assert percept is not None
        assert "Body Utterances" not in percept.content

    def test_hunger_utterance_co_fires_within_same_percept(self, real_naming_body):
        """The load-bearing within-tick claim: the body-state snapshot
        AND the "hungry" utterance appear in the SAME percept content
        returned by ONE next_percept call. This is what makes
        LinguisticEncoder and SensorEncoder fire in the same agent-loop
        tick window — the precondition Stage 3's Hebbian retest depends
        on."""
        source, entity, _ = real_naming_body
        entity.vital_metrics["hunger"] = 0.8

        percept = source.next_percept()
        assert percept is not None
        # Co-firing guarantee: BOTH sections in ONE percept.
        assert "=== Body State ===" in percept.content
        assert "=== Body Utterances ===" in percept.content
        assert "hungry" in percept.content
        # And the drive value is in the same content so SensorEncoder
        # processes the same snapshot.
        assert "hunger" in percept.content

    def test_modulator_sub_sensor_utterance_fires(self, real_naming_body):
        """Pins B1: ``arms.thermal`` lives in modulator vital_metrics,
        NOT in body_state_summary. If the collector skips the modulator
        path, "warm" silently never fires."""
        source, entity, _ = real_naming_body
        entity.modulators["arms"].vital_metrics["thermal"] = 0.85

        percept = source.next_percept()
        assert percept is not None
        assert "warm" in percept.content

    def test_all_three_utterances_can_co_fire_in_one_tick(self, real_naming_body):
        source, entity, _ = real_naming_body
        entity.vital_metrics["hunger"] = 0.8
        entity.vital_metrics["thirst"] = 0.7
        entity.modulators["arms"].vital_metrics["thermal"] = 0.85

        percept = source.next_percept()
        assert percept is not None
        assert "hungry" in percept.content
        assert "thirsty" in percept.content
        assert "warm" in percept.content

    def test_utterance_does_not_repeat_on_consecutive_above_threshold_ticks(self, real_naming_body):
        """Hysteresis on a real source — "hungry" lands in the tick the
        drive crosses, NOT every subsequent tick."""
        source, entity, _ = real_naming_body
        entity.vital_metrics["hunger"] = 0.8

        p1 = source.next_percept()
        assert "hungry" in p1.content
        # Second tick: still above threshold, should NOT re-emit.
        source._last_poll = 0.0
        p2 = source.next_percept()
        assert p2 is not None
        assert "hungry" not in p2.content
        assert "Body Utterances" not in p2.content
