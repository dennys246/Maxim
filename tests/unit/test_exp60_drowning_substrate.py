"""Exp 60 (drowning-avoidance) substrate additions — structural guards.

Locks the three additions the four-lens review required BEFORE the water
apparatus / harness are built (docs/experiments/exp60_drowning_avoidance_prereg.md):

1. `is_in_water` — a binary world sensor (the stable pre-damage underwater cue).
2. an air-hunger `drive` on `oxygen` → failure mode ``drive:oxygen`` in the
   Wire-4 fear allowlist (so drowning books a drowning-SPECIFIC fear before
   tissue damage, not generic injury fear).
3. a `surface` affordance matched by ``_DRIVE_TOOL_AFFINITIES["threat"]`` (so
   learned drowning-fear actually scores the swim-up action — the Exp-58
   dead-read-path bug must not recur here).

These are cheap structural asserts; the behavioural claim is the (unbuilt) live
experiment, gated on the separability probe on the water pool.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from maxim.decisions.nac import _DRIVE_TOOL_AFFINITIES, NAc, NACConfig

_BODY = Path("src/maxim/_data/components/bodies/minecraft_player.yaml")
AGENT, WATER, SHORE = "agent-1", "cluster-underwater", "cluster-shore"


def _body_sensors() -> dict:
    data = yaml.safe_load(_BODY.read_text())

    def find(o):
        if isinstance(o, dict):
            if "sensors" in o:
                return o["sensors"]
            for v in o.values():
                r = find(v)
                if r:
                    return r
        return None

    return find(data)


class TestIsInWaterSensor:
    def test_declared_as_a_world_sensor(self):
        sensors = _body_sensors()
        assert "is_in_water" in sensors, "is_in_water world sensor missing (Exp 60)"
        assert sensors["is_in_water"]["modality"] == "world"

    def test_rest_sits_at_neutral(self):
        # range must put rest (dry, 0) at the A4-neutral midpoint so a dry bot is
        # SILENT and only submersion shouts (mirrors is_raining). Midpoint of the
        # declared range must equal the initial/rest value 0.
        s = _body_sensors()["is_in_water"]
        lo, hi = s["range"]
        assert (lo + hi) / 2 == s["initial"] == 0


class TestOxygenAirHungerDrive:
    def test_oxygen_has_a_homeostatic_drive(self):
        oxy = _body_sensors()["oxygen"]
        assert "drive" in oxy, "oxygen needs an air-hunger drive (Exp 60)"
        assert oxy["drive"]["drift_mode"] == "homeostatic"
        assert oxy["drive"]["set_point"] == 20
        assert oxy["drive"]["drift_rate"] == 0.0  # world-owned: the bridge writes air truth

    def test_drive_oxygen_is_in_the_fear_allowlist(self):
        assert "drive:oxygen" in NACConfig().cluster_fear_failure_modes
        assert "drive:health" in NACConfig().cluster_fear_failure_modes  # not dropped

    def test_drive_oxygen_books_fear_on_the_underwater_cluster(self):
        nac = NAc(NACConfig())
        nac.record_cluster_fear(AGENT, WATER, "drive:oxygen", 1.0)
        assert nac.cluster_fear(AGENT, WATER) < 0.0
        assert nac.cluster_fear(AGENT, SHORE) == 0.0  # specific to where the pain co-occurred


class TestSurfaceAffordance:
    def test_surface_declared_on_the_body(self):
        data = yaml.safe_load(_BODY.read_text())
        assert "surface" in yaml.dump(data), "surface affordance missing (Exp 60)"

    def test_threat_need_matches_surface(self):
        # substring match in recommend_action: "surface" (a threat affordance)
        # must be reachable from the threat need, or learned drowning-fear scores
        # nothing (Exp 58's dead-read-path failure).
        assert "surface" in _DRIVE_TOOL_AFFINITIES["threat"]
        tool = "minecraft_player_surface"
        assert any(kw in tool for kw in _DRIVE_TOOL_AFFINITIES["threat"])
