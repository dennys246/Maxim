"""Exp 60 (drowning-avoidance) substrate additions — structural guards.

Locks the three additions the four-lens review required BEFORE the water
apparatus / harness are built (docs/experiments/exp60_drowning_avoidance_prereg.md):

1. `is_in_water` — a binary world sensor (the stable pre-damage underwater cue).
2. an air-hunger `drive` on `oxygen` → failure mode ``drive:oxygen`` in the
   Wire-4 fear allowlist (so drowning books a drowning-SPECIFIC fear before
   tissue damage, not generic injury fear).
3. an `escape_water` affordance matched by ``_DRIVE_TOOL_AFFINITIES["threat"]``
   via the existing "escape" keyword (so learned drowning-fear actually scores
   the swim-up action — the Exp-58 dead-read-path bug must not recur here;
   named `escape_water` not `surface` to avoid a cross-body false-match, S2).

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


def _body_affordances() -> dict:
    data = yaml.safe_load(_BODY.read_text())

    def find(o):
        if isinstance(o, dict):
            if "affordances" in o:
                return o["affordances"]
            for v in o.values():
                r = find(v)
                if r:
                    return r
        return None

    return find(data) or {}


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


class TestEscapeWaterAffordance:
    def test_escape_water_declared_as_an_affordance(self):
        affs = _body_affordances()
        assert "escape_water" in affs, f"escape_water affordance missing (Exp 60); have {sorted(affs)}"
        assert "surface" not in affs, "renamed to escape_water to avoid the cross-body 'surface' false-match (S2)"

    def test_threat_need_reaches_escape_water_via_escape_keyword(self):
        # recommend_action substring-matches the tool name against the threat
        # affinity keywords. escape_water must match (via the existing "escape"),
        # or learned drowning-fear scores nothing (Exp 58's dead-read-path).
        tool = "minecraft_player_escape_water"
        matched = [kw for kw in _DRIVE_TOOL_AFFINITIES["threat"] if kw in tool]
        assert matched == ["escape"], f"expected only 'escape' to match, got {matched}"

    def test_no_generic_surface_keyword_added_to_threat(self):
        # guard S2: a generic "surface" keyword would false-match other bodies'
        # affordances (e.g. alien_xenomorph climb_surface).
        assert "surface" not in _DRIVE_TOOL_AFFINITIES["threat"]


class TestSaturationRestsAtTheBridgeClamp:
    """Exp 60 gate (ii) first live run: `saturation` sat at an extreme in BOTH situations and
    its constant mass lifted cos(shore, submerged) from 0.787 to 0.8502. The game never rests at
    the old midpoint (5): a fed bot reads the bridge clamp (10), a drained bot 0. The range must
    put the FED clamp at the A4-neutral midpoint (docs/wiring/cosine-separation-is-directional.md
    corollary 6)."""

    BRIDGE_CLAMP = 10  # scripts/minecraft_bridge/index.js: Math.min(10, bot.foodSaturation)

    def test_fed_state_is_the_midpoint_and_the_initial(self):
        s = _body_sensors()["saturation"]
        lo, hi = s["range"]
        assert (lo + hi) / 2 == self.BRIDGE_CLAMP == s["initial"]

    def test_bridge_clamp_is_what_the_test_assumes(self):
        js = (Path(__file__).resolve().parents[2] / "scripts" / "minecraft_bridge" / "index.js").read_text()
        assert f"saturation: Math.min({self.BRIDGE_CLAMP}, bot.foodSaturation" in js
