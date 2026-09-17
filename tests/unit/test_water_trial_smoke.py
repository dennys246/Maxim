"""The shared water-classroom seed context (`scripts/survival_world/water_trial.py::WaterTrial`)
proven able to tick and act OFFLINE, against the scripted water bridge (Exp 61 architecture lens
S10; the wiring lesson `docs/wiring/harness-loop-must-be-proven-live.md`).

What this proves, in the order the harnesses use it: the bridge roster + cadence check passes on
the player roster; the full loop ticks on the shore (the #732/#733 arc); shore and submerged encode
to distinct live world clusters; the escape actuation check surfaces the bot through the BRIDGE
with ZERO executor calls (the executor-lens invariant of Exp 60); a placement with no fear is
censored with no executor call (the structural floor); after two saturating fear writes on the
water cluster a placement EXECUTES `escape_water` through the real loop and classifies surfaced;
and the staging close (`exp61_run.close_and_stage`) persists that fear — the hub-session trap the
review found (the loop's own session pair had closed the hub; the staged nac carried fear 0)
goes RED here on the pre-fold code and green after.

Not proven here: propose-only TRAINING (needs the body's oxygen drive to publish pain through the
sync pump — a live-apparatus property) and the live bridge's timing. Those are the one-pair dry
run's job (build step 4).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world.exp61_run import close_and_stage  # noqa: E402
from survival_world.scripted_water import ScriptedWaterBridge, ScriptedWaterControl  # noqa: E402
from survival_world.water_trial import Refusal, WaterTrial  # noqa: E402

SHORE = {"x": 10.0, "y": 64.0, "z": 10.0}
SUBMERGED = {"x": 10.0, "y": 60.0, "z": 20.0}
GEOM = {"shore": [10.0, 64.0, 10.0], "submerged": [10.0, 60.0, 20.0], "deaths_objective": "exp60_deaths"}
# Exp 60's constants, shortened so the smoke runs in seconds; the SHAPE of every check is the same.
FAST = {
    "loop_hz": 4.0,
    "loop_warm_s": 0.5,
    "loop_liveness_s": 2.0,
    "loop_liveness_min_ticks": 3,
    "bridge_state_interval_max_s": 0.15,
    "placements_per_probe": 1,
    "shore_roam_s": 1.0,
    "death_cap": 2,
    "K_usable_episodes": 1,
    "usable_oxygen_max": 12.0,
    "usable_pain_intensity_min": 1.0,
    "specificity_ratio": 0.2,
    "fingerprint": {},
}


def _build(tmp_path: Path, srv: ScriptedWaterBridge, agent_id: str):
    from maxim.simulation.minecraft_harness import MinecraftSyncPump, build_minecraft_aut
    from survival_world.common import make_fresh_encoder

    home = tmp_path / agent_id
    aut = build_minecraft_aut(
        agent_id=agent_id, bridge_port=srv.port, persistence_dir=str(home), entity_ref="bodies/minecraft_player"
    )
    aut.bio.memory_hub.on_session_start()
    encoder = make_fresh_encoder(aut)
    pump = MinecraftSyncPump(aut, interval_s=0.25)
    pump.start()
    return aut, encoder, pump, home


@pytest.mark.timeout(180)
def test_water_trial_ticks_acts_and_the_staging_close_persists_fear(tmp_path: Path) -> None:
    srv = ScriptedWaterBridge(shore=SHORE, submerged=SUBMERGED)
    rcon = ScriptedWaterControl(srv)
    agent_id = "smoke_water"
    aut, encoder, pump, home = _build(tmp_path, srv, agent_id)
    trial = WaterTrial(
        aut=aut,
        rcon=rcon,
        username="maxim",
        geom=GEOM,
        frozen=FAST,
        probe_cap_s=3.0,
        train_cap_s=8.0,
        persistence_dir=home,
        agent_id=agent_id,
        encoder=encoder,
        settle_guard={"is_raining": 0.0, "nearest_player_dist": 64.0},
    )
    trial.attach_instruments()
    try:
        # roster + freshness on the player roster
        cadence = trial.check_bridge()
        assert cadence is not None and cadence <= 0.15
        # the full loop ticks on the shore (the wake-source + autonomy arc)
        assert trial.check_liveness() >= FAST["loop_liveness_min_ticks"]
        trial.check_gamerules()
        # shore and submerged are distinct LIVE clusters
        shore_c, water_c = trial.check_clusters_distinct()
        assert shore_c and water_c and shore_c != water_c
        # actuation through the BRIDGE only — never the executor
        trial.resolve_tools()
        res = trial.check_escape_actuation()
        assert res["t_surface"] is not None and res["t_surface"] < 2.5
        assert trial.calls == [], "the actuation check must not reach the executor (it would book a positive link)"
        assert "escape_water" in srv.actions
        trial.check_no_positive_escape_link()
        trial.deaths0 = trial.deaths()
        # no fear → the structural floor: censored, no executor call
        pl0 = trial.placement("nofear")
        assert not pl0["surfaced"] and pl0["calls"] == [], pl0
        # two saturating writes on the water cluster → the loop EXECUTES the escape and surfaces
        aut.bio.nac.record_cluster_fear(agent_id, water_c, "drive:oxygen", 1.0)
        aut.bio.nac.record_cluster_fear(agent_id, water_c, "drive:oxygen", 1.0)
        assert aut.bio.nac.anticipatory_threat_need(agent_id, {"world": water_c}) > 0.5
        pl1 = trial.placement("fear")
        assert pl1["escape_water_calls"] >= 1, pl1["calls"]
        assert pl1["surfaced"] and pl1["latency_s"] is not None and pl1["latency_s"] < 3.0, pl1
        # the loop's own session pair closed the hub; the trial re-opened it after every loop run, so
        # the staging close persists the fear booked AFTER those loops (the S1 trap, guarded)
        stage = tmp_path / "stage"
        trial.detach_instruments()
        close_and_stage(aut, pump, stage)
        staged = json.loads((stage / "aut_nac.json").read_text())
        keys = [k for k in staged.get("cluster_fear", {}) if k.split("\x1f")[1] == water_c]
        assert keys and staged["cluster_fear"][keys[0]] == -1.0, staged.get("cluster_fear")
        ec = json.loads((stage / "aut_ec.json").read_text())
        assert water_c in ec["substrate_nodes"] and ec["substrate_nodes"][water_c]["modality"] == "world"
    finally:
        try:
            aut.client.close()
        except Exception:
            pass
        srv.close()


def test_staging_close_refuses_when_the_hub_session_is_not_active() -> None:
    """The negative arm: a hub whose close returns {} persisted nothing — never copy a stale file."""

    class _Hub:
        def on_session_start(self):
            return {"already_active": 0}

        def on_session_end(self):
            return {}

    class _Bio:
        memory_hub = _Hub()

        def on_session_end(self):
            return {}

    class _Client:
        closed = False

        def close(self):
            self.closed = True

    class _Aut:
        bio = _Bio()
        client = _Client()
        persistence_dir = "/nonexistent"

    class _Pump:
        def stop(self):
            return None

    aut = _Aut()
    with pytest.raises(Refusal, match="nothing persisted"):
        close_and_stage(aut, _Pump(), None)
    assert aut.client.closed  # the client is closed even on refusal (NIT N3)


@pytest.mark.timeout(60)
def test_settle_guard_refuses_rain_and_an_absent_key(tmp_path: Path) -> None:
    srv = ScriptedWaterBridge(shore=SHORE, submerged=SUBMERGED)
    rcon = ScriptedWaterControl(srv)
    aut, encoder, pump, home = _build(tmp_path, srv, "smoke_guard")
    try:
        time.sleep(0.5)
        bad = WaterTrial(
            aut=aut,
            rcon=rcon,
            username="maxim",
            geom=GEOM,
            frozen=FAST,
            probe_cap_s=3.0,
            train_cap_s=8.0,
            persistence_dir=home,
            agent_id="smoke_guard",
            encoder=encoder,
            settle_guard={"is_raining": 1.0},  # the fake never rains → the guard must refuse
        )
        with pytest.raises(Refusal, match="settle guard is_raining"):
            bad.rescue("guard")
        absent = WaterTrial(
            aut=aut,
            rcon=rcon,
            username="maxim",
            geom=GEOM,
            frozen=FAST,
            probe_cap_s=3.0,
            train_cap_s=8.0,
            persistence_dir=home,
            agent_id="smoke_guard",
            encoder=encoder,
            settle_guard={"no_such_sensor": 0.0},
        )
        with pytest.raises(Refusal, match="absent"):
            absent.rescue("guard")
    finally:
        try:
            pump.stop()
            aut.client.close()
        except Exception:
            pass
        srv.close()


@pytest.mark.timeout(120)
def test_without_the_reopen_the_staging_close_refuses_instead_of_staging_stale_fear(tmp_path: Path) -> None:
    """The hub-session trap in its original shape: the liveness loop's own session pair closes the
    hub; fear booked afterwards would never reach the staged nac. With the trial's re-open disabled
    and the hub refusing to re-open, the close must REFUSE (nothing persisted) — never copy the
    liveness-close snapshot as if it were the trained donor."""
    srv = ScriptedWaterBridge(shore=SHORE, submerged=SUBMERGED)
    rcon = ScriptedWaterControl(srv)
    agent_id = "smoke_trap"
    aut, encoder, pump, home = _build(tmp_path, srv, agent_id)
    trial = WaterTrial(
        aut=aut,
        rcon=rcon,
        username="maxim",
        geom=GEOM,
        frozen=FAST,
        probe_cap_s=3.0,
        train_cap_s=8.0,
        persistence_dir=home,
        agent_id=agent_id,
        encoder=encoder,
    )
    trial.reopen_hub_session = lambda: None  # type: ignore[method-assign]  — the pre-fold behaviour
    trial.attach_instruments()
    try:
        assert trial.check_liveness() >= FAST["loop_liveness_min_ticks"]  # the loop closes the hub session here
        aut.bio.nac.record_cluster_fear(agent_id, "some-water-node", "drive:oxygen", 1.0)
        trial.detach_instruments()
        aut.bio.memory_hub.on_session_start = lambda: {}  # type: ignore[method-assign]  — a hub that will not re-open
        with pytest.raises(Refusal, match="nothing persisted"):
            close_and_stage(aut, pump, tmp_path / "stage_trap")
        assert not (tmp_path / "stage_trap" / "aut_nac.json").exists()
    finally:
        try:
            aut.client.close()
        except Exception:
            pass
        srv.close()
