"""R3's unit of measurement — `WaterTrial.lethal_event` — proven OFFLINE on the scripted water bridge
before it touches the rig, with the three endings the design names and the pilot's must-nots fixed:

- SURFACE: an attached fresh agent books fear on its own cluster from the scripted oxygen pain and
  surfaces by its own executed escape; the shore teleport is issued BEFORE the loop is stopped (the
  pilot joined first); no linger; pain-seconds integrate from the sample series; the decision
  provenance of the executed escape is captured and drive-decisive.
- DEATH: a DETACHED agent on a bridge with a scripted drowning (no innate route offline — the scripted
  body has no health→threat lag to exploit, it just dies) reads the death from the `deaths` objective
  per sample, corroborated by the respawn discontinuity, and ends without a teleport.
- CAP: a detached agent on a bridge with no damage never surfaces and never dies → the cap, a Refusal.

Plus the preflights: the `deaths` objective exists / resets / reads back; a parse failure RAISES (never
0); the true food state parses from `data get entity`; the surface-cell check; a gamerule SET reads
back; an unknown gamerule name is an InstrumentError, never a recorded absence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world.common import InstrumentError  # noqa: E402
from survival_world.scripted_water import (  # noqa: E402
    LockstepTime,
    ScriptedWaterBridge,
    ScriptedWaterControl,
    StepClock,
)
from survival_world.setup_world import water_anchor_record, water_classroom_geometry  # noqa: E402
from survival_world.water_trial import R3_GAMERULES, Refusal, WaterTrial, _detach_fear_subscriber  # noqa: E402

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


def _xyz(t: list) -> dict[str, float]:
    return {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])}


def _record() -> dict:
    rec = water_anchor_record(water_classroom_geometry(10, 20, depth=5, shore_y=64))
    rec["measured"] = {"t_damage_onset_min_s": 16.067}
    return rec


def _rig(
    tmp_path: Path,
    agent_id: str,
    *,
    damage_onset_s: float | None,
    detach: bool,
    damage_per_s: float = 2.0,
    respawn_lag_s: float | None = None,
    escape_delay_s: float = 0.3,
    lockstep_patch=None,
):
    """``lockstep_patch`` (a pytest ``monkeypatch``) runs the scripted world on a step clock the
    harness advances (#951), so a window the test reads (the respawn lag) is world time, not wall."""
    from maxim.simulation.minecraft_harness import MinecraftSyncPump, build_minecraft_aut
    from survival_world.common import make_fresh_encoder

    rec = _record()
    world_clock = StepClock() if lockstep_patch is not None else None
    srv = ScriptedWaterBridge(
        shore=_xyz(rec["shore"]),
        submerged=_xyz(rec["submerged"]),
        damage_onset_s=damage_onset_s,
        damage_per_s=damage_per_s,
        respawn_lag_s=respawn_lag_s,
        escape_delay_s=escape_delay_s,
        clock=world_clock,
    )
    rcon = ScriptedWaterControl(srv, gamerules={r: v for r, v in R3_GAMERULES})
    home = tmp_path / agent_id
    aut = build_minecraft_aut(
        agent_id=agent_id, bridge_port=srv.port, persistence_dir=str(home), entity_ref="bodies/minecraft_player"
    )
    aut.bio.memory_hub.on_session_start()
    if detach:
        assert _detach_fear_subscriber(aut) >= 1
    encoder = make_fresh_encoder(aut)
    pump = MinecraftSyncPump(aut, interval_s=0.25)
    pump.start()
    trial = WaterTrial(
        aut=aut,
        rcon=rcon,
        username="maxim",
        geom=rec,
        frozen=FAST,
        probe_cap_s=3.0,
        train_cap_s=12.0,
        persistence_dir=home,
        agent_id=agent_id,
        encoder=encoder,
        settle_guard={"is_raining": 0.0, "nearest_player_dist": 64.0},
    )
    trial.attach_instruments()
    trial.resolve_tools()
    if lockstep_patch is not None:
        # one 4 Hz tick per sleep at least: `lethal_event` paces with sleep(period - elapsed), which
        # sleeps 0 on an overrun tick — without a quantum a slow runner would freeze the world
        lockstep = LockstepTime(world_clock, min_step_s=1.0 / FAST["loop_hz"])
        lockstep_patch.setattr(sys.modules[WaterTrial.__module__], "time", lockstep)
    return srv, rcon, aut, trial


def _close(srv, aut, trial):
    trial.detach_instruments()
    try:
        aut.client.close()
    except Exception:
        pass
    srv.close()


@pytest.mark.timeout(240)
def test_lethal_event_surface_by_own_escape_teleports_before_the_loop_stops(tmp_path: Path) -> None:
    srv, rcon, aut, trial = _rig(tmp_path, "lethal_surface", damage_onset_s=None, detach=False)
    try:
        trial.rescue("ready")
        ev = trial.lethal_event("surface", cap_s=25.0)
    finally:
        _close(srv, aut, trial)
    assert ev["end"] == "surface" and ev["survived"] is True and ev["t_surface"] is not None, ev["end"]
    assert ev["escaped_before_damage"] is True and ev["health_lost"] == 0.0
    assert any(str(c["tool"]).endswith("escape_water") and not c["post_event"] for c in ev["calls"]), ev["calls"]
    assert all("t_return" in c and "detail" in c for c in ev["calls"])
    # the in-situ learner booked fear on its own cluster from the scripted oxygen pain
    assert any(p["failure_mode"] == "drive:oxygen" for p in ev["pain_publishes"]) and trial.fear_dump()
    # pain-seconds integrate the SAMPLE series (not the publishes): oxygen paid, health not
    assert ev["pain_seconds"]["oxygen"] is not None and ev["pain_seconds"]["oxygen"] > 0.0
    assert ev["pain_seconds"]["health"] == 0.0
    # decision provenance captured: the executed escape was drive-decisive
    from survival_world.exp61_run import decision_decisive

    assert ev["executed_escape_event"] is not None and decision_decisive(ev["executed_escape_event"])[0]
    # the exit teleport was issued BEFORE the loop stopped: the shore `tp` precedes the post-window reads
    tps = [i for i, c in enumerate(rcon.commands) if c.startswith("tp ")]
    deaths_reads = [i for i, c in enumerate(rcon.commands) if c.startswith("scoreboard players get")]
    assert tps and deaths_reads and tps[-1] < deaths_reads[-1]
    # one wall clock: ticks carry `t` on the window's clock; proposals precede the executed call
    assert ev["ticks"] and all("t" in t for t in ev["ticks"]) and ev["tick_period_median_s"]
    assert ev["food_at_teleport"]["foodSaturationLevel"] == 20.0  # the TRUE reservoir (the sensed value is 10)
    assert ev["max_state_age_s"] is not None
    assert ev["deaths_delta_after"] == 0 and ev["guard_breach"] is None


@pytest.mark.timeout(240)
def test_lethal_event_death_reads_the_objective_and_the_respawn(tmp_path: Path, monkeypatch) -> None:
    # 40 hp/s → dead 0.5 s (world time) after onset. The scoreboard leads the respawn by 0.6 s of WORLD
    # time and the harness steps the world 0.25 s at a time (#951: lockstep), so at least two samples land
    # inside the lag whatever the runner's speed — the race is certain here. The escape takes 2 s of world
    # time to land (> the 0.5 s damage window + the 0.6 s respawn lag), so no reflex escape, however its
    # ticks fall, can surface the agent before the death or land in the death→respawn gap.
    srv, rcon, aut, trial = _rig(
        tmp_path,
        "lethal_death",
        damage_onset_s=1.5,
        detach=True,
        damage_per_s=40.0,
        respawn_lag_s=0.6,
        escape_delay_s=2.0,
        lockstep_patch=monkeypatch,
    )
    try:
        trial.rescue("ready")
        ev = trial.lethal_event("death", cap_s=25.0)
    finally:
        _close(srv, aut, trial)
    assert ev["end"] == "death" and ev["survived"] is False and ev["t_death"] is not None, ev["end"]
    assert ev["t_first_damage"] is not None and ev["min_health"] is not None and ev["min_health"] < 20.0
    assert ev["health_lost"] > 0 and ev["pain_seconds"]["health"] > 0.0
    # detached: NOTHING acts before damage exists. A call before the damage onset (1.5 s after the window
    # opens; a world step never outruns the wall clock, so a wall `t` < 1.5 is before it) would be
    # anticipation without fear — a detach that failed on some other path (fear_dump covers only the NAc
    # half). A reflex call INSIDE the damage window is allowed: whether a 4 Hz tick lands in a 0.5 s window
    # is scheduling (#951: asserting "no call before the death" flaked three times). A post-event flee
    # during the corroboration samples is the reflex firing on a respawned body — flagged, not counted.
    assert all(c["post_event"] or c["t"] >= 1.5 for c in ev["calls"]), ev["calls"]
    assert trial.fear_dump() == {}
    # the respawn discontinuity within the death sample: health 20, out of the water, at the shore
    last = ev["samples"][-1]
    assert last["deaths_delta"] == 1 and not last["in_water"] and last["health"] == 20.0 and last["saturation"] == 5.0
    assert srv.deaths == 1 and ev["deaths_delta_after"] == 1
    # the RACE was exercised: the first `deaths` sample was still a pre-death snapshot (in water), and
    # the corroboration kept sampling to the respawn discontinuity
    first_death = next(x for x in ev["samples"] if x["deaths_delta"] > 0)
    assert first_death["in_water"] and ev["t_death"] == first_death["t"] and last["t"] > first_death["t"]


@pytest.mark.timeout(240)
def test_lethal_event_innate_health_reflex_surfaces_a_detached_agent_after_damage(tmp_path: Path) -> None:
    """The pilot's lethal_A, offline: no fear subscriber, the scripted drowning at the game's 2 hp/s from
    a short onset; the innate `health → threat` need fires once health leaves its band and the agent
    surfaces WITH damage — never before damage, with health-pain seconds paid."""
    srv, rcon, aut, trial = _rig(tmp_path, "lethal_innate", damage_onset_s=1.5, detach=True, damage_per_s=2.0)
    try:
        trial.rescue("ready")
        ev = trial.lethal_event("innate", cap_s=25.0)
    finally:
        _close(srv, aut, trial)
    assert ev["end"] == "surface" and ev["survived"] is True, ev["end"]
    assert ev["escaped_before_damage"] is False and ev["health_lost"] > 0 and ev["min_health"] < 14.0
    assert ev["pain_seconds"]["health"] > 0.0 and trial.fear_dump() == {}
    assert any(str(c["tool"]).endswith("escape_water") and not c["post_event"] for c in ev["calls"]), ev["calls"]
    from survival_world.exp61_run import decision_decisive

    assert ev["executed_escape_event"] is not None and decision_decisive(ev["executed_escape_event"])[0]


@pytest.mark.timeout(120)
def test_lethal_event_cap_is_a_refusal_with_the_shore_teleport(tmp_path: Path) -> None:
    srv, rcon, aut, trial = _rig(tmp_path, "lethal_cap", damage_onset_s=None, detach=True)
    try:
        trial.rescue("ready")
        with pytest.raises(Refusal, match="cap") as ei:
            trial.lethal_event("cap", cap_s=4.0)
    finally:
        _close(srv, aut, trial)
    ev = ei.value.partial["event"]
    assert ev["end"] == "cap" and ev["survived"] is False and ev["calls"] == []
    # the shore teleport was issued at the cap (cleanup, not rescue semantics)
    assert any(c.startswith("tp ") for c in rcon.commands)


def test_deaths_parse_failure_raises_and_preflight_resets(tmp_path: Path) -> None:
    rec = _record()
    srv = ScriptedWaterBridge(shore=_xyz(rec["shore"]), submerged=_xyz(rec["submerged"]))
    rcon = ScriptedWaterControl(srv, gamerules={r: v for r, v in R3_GAMERULES})

    class _T(WaterTrial):
        def __init__(self):  # noqa: D401 — minimal: only the RCON-facing methods are exercised
            self.rcon, self.username, self.geom = rcon, "maxim", rec

    t = _T()
    srv.deaths = 3
    assert t.deaths() == 3
    t.preflight_deaths_objective()
    assert t.deaths() == 0
    assert t.read_food_state()["foodSaturationLevel"] == 20.0  # true reservoir, not the sensed clamp
    t.check_surface_cell_air()
    srv.surface_air = False
    with pytest.raises(Refusal, match="not air"):
        t.check_surface_cell_air()
    assert "false" in t.set_gamerule("naturalRegeneration", "false").lower()
    t.check_gamerules((("drowningDamage", "true"), ("doInsomnia", "false")))
    with pytest.raises(InstrumentError, match="not a rule"):
        t.check_gamerules((("doDrowningDamage", "true"),))
    rcon.command = lambda cmd: "garbage"  # type: ignore[method-assign]
    with pytest.raises(InstrumentError, match="unreadable"):
        t.deaths()
    srv.close()


def test_attaching_instruments_twice_refuses(monkeypatch):
    """Exp 62 runs two pools per agent — a second attach would wrap the first trial's spy.

    The prereg's apparatus paragraph names this: "ONE instrument attach — the executor spy must
    not double-wrap". Double-wrapping would count every call twice and fire the pain subscriber
    twice per publish, which reads as a louder agent rather than as a defect.
    """
    import types

    from survival_world.common import InstrumentError
    from survival_world.water_trial import WaterTrial

    trial = object.__new__(WaterTrial)
    trial.calls = []
    trial._orig_execute = lambda action: None  # already attached
    trial.aut = types.SimpleNamespace(
        bio=types.SimpleNamespace(pain_bus=types.SimpleNamespace(subscribe=lambda _cb: None)),
        executor=types.SimpleNamespace(execute=lambda action: None),
    )
    with pytest.raises(InstrumentError, match="already attached"):
        trial.attach_instruments()


def test_a_second_trial_on_the_same_agent_refuses_to_double_wrap():
    """Exp 62 runs one trial per pool on one agent — the case a per-trial flag cannot see.

    The first trial's guard lives on `self`; a second trial has its own. The sentinel rides on the
    installed wrapper instead, so the agent itself carries the "already instrumented" fact.
    """
    import types

    from survival_world.common import InstrumentError
    from survival_world.water_trial import WaterTrial

    def _wrapped(action):
        return None

    _wrapped._water_trial_spy = True  # what the FIRST trial installed
    second = object.__new__(WaterTrial)
    second.calls = []
    second.aut = types.SimpleNamespace(
        bio=types.SimpleNamespace(pain_bus=types.SimpleNamespace(subscribe=lambda _cb: None)),
        executor=types.SimpleNamespace(execute=_wrapped),
    )
    with pytest.raises(InstrumentError, match="double-wraps"):
        second.attach_instruments()
