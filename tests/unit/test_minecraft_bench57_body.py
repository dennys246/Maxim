"""Guards for the Exp 57 bench57 body — minecraft_bench with the world channel
made DIRECTION-AWARE (offset_x/offset_z added; see the body YAML docstring).

Pins the apparatus change and everything it must NOT change: the body loads,
its world channel is OFFSETS-ONLY (offset_x/offset_z — signed horizontal
position), and it keeps the d1 modeled interoceptive drive and the eight opaque
aff_a..aff_h affordances (the L12 opacity mitigation) verbatim. The world
channel deliberately EXCLUDES the situation-constant Exp 56 sensors
(distance_from_spawn/y_altitude/speed/on_ground/time_of_day): across the four
frozen slots they carry zero situation info and, through the encoder, dilute
the offsets to a jitter-fragile margin (all-seven → 0.011; offsets-only →
0.478). Re-adding any of them to the world channel is a regression this pins.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BENCH57_YAML = REPO / "src/maxim/_data/components/bodies/minecraft_bench57.yaml"

WORLD_SENSORS = {"offset_x", "offset_z"}
#: Sensors that must NOT be in the world channel — situation-constant across the
#: frozen slots, so they only dilute the offsets (the 0.011-margin failure mode).
DILUTING_SENSORS = {"y_altitude", "distance_from_spawn", "speed", "on_ground", "time_of_day"}
OPAQUE_AFFORDANCES = {"aff_a", "aff_b", "aff_c", "aff_d", "aff_e", "aff_f", "aff_g", "aff_h"}


class TestBench57Spec:
    def test_entity_name_and_affordances(self):
        spec = yaml.safe_load(BENCH57_YAML.read_text())
        assert spec["entity"]["name"] == "minecraft_bench57"
        affs = spec["entity"]["modulators"]["act"]["affordances"]
        assert set(affs) == OPAQUE_AFFORDANCES, "the eight opaque affordances (L12 mitigation) must be preserved"
        assert all(a.startswith("aff_") for a in affs), "opaque affordance names are the L12 mitigation"
        assert spec["entity"]["modulators"]["act"].get("abstract") is True

    def test_d1_is_modeled_interoception_verbatim(self):
        spec = yaml.safe_load(BENCH57_YAML.read_text())
        d1 = spec["entity"]["sensors"]["d1"]
        assert d1.get("modality") != "world", (
            "d1 must be MODELED interoception — a world-owned drive short-circuits the operant path"
        )
        drive = d1["drive"]
        # The EXACT Exp 56 spec — a retune here would silently change the teacher's lever.
        assert drive["drift_mode"] == "entropic"
        assert drive["drift_direction"] == "up"
        assert drive["drift_rate"] == 0.006
        assert drive["deprivation_threshold"] == 0.7
        assert drive["deprivation_pain"] == 0.3
        assert drive["satisfaction_threshold"] == 0.3

    def test_world_channel_is_offsets_only(self):
        spec = yaml.safe_load(BENCH57_YAML.read_text())
        sensors = spec["entity"]["sensors"]
        world = {n for n, s in sensors.items() if isinstance(s, dict) and s.get("modality") == "world"}
        assert world == WORLD_SENSORS, "world channel is offsets-only (no situation-constant dilution)"
        assert not (world & DILUTING_SENSORS), "the diluting Exp 56 world sensors must NOT be in the world channel"
        for name in WORLD_SENSORS:
            s = sensors[name]
            assert s["modality"] == "world"
            assert s["unit"] == "blocks"
            assert s["range"] == [-128, 128]
            assert s["initial"] == 0
        assert len(world) <= 12, "L11 per-channel budget"


class TestBench57Instantiates:
    def test_body_loads_and_world_sensors_include_offsets(self):
        from maxim.embodiment.component_registry import ComponentRegistry

        e = ComponentRegistry().instantiate("bodies/minecraft_bench57")
        assert e.name == "minecraft_bench57"
        world = {n for n, s in e.sensors.items() if (s.reading_schema or {}).get("modality") == "world"}
        assert world == WORLD_SENSORS, "instantiated world channel must be offsets-only"
        assert not (world & DILUTING_SENSORS)
        assert "d1" in e.sensors
        assert set(e.modulators["act"].affordances) == OPAQUE_AFFORDANCES

    def test_world_ranges_derives_the_offsets_from_this_body(self):
        # common57.world_ranges() instantiates BODY_REF57 — its declared world
        # ranges must now carry the offset sensors (the encode path the B-phase
        # uses to separate the four FROZEN slots).
        import sys

        sys.path.insert(0, str(REPO / "scripts"))
        from exp57 import common57 as X

        ranges = X.world_ranges()
        assert set(ranges) == WORLD_SENSORS, "world_ranges must be offsets-only (no diluting sensors)"
        assert ranges["offset_x"] == (-128.0, 128.0)
        assert ranges["offset_z"] == (-128.0, 128.0)
