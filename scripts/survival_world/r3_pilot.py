"""R3 PILOT — the measured diagnostic the four-lens review asked for BEFORE any R3 harness is built
(``docs/experiments/r3_survival_benchmark_prereg.md`` §Build order step 1). Dev tooling, not in the
wheel. Everything here is recorded as a diagnostic; nothing is data.

What it measures, live on the Exp 60 water classroom, one fresh agent per row:

1. ``apparatus``   — the Exp 60/61 preflights on a throwaway agent, PLUS the world facts v1 never
                     verified: ``naturalRegeneration``, ``doDrowningDamage``, ``difficulty`` (read,
                     recorded); the ``flee`` anchor preflight through the BRIDGE on the shore (latency +
                     detail — the default anchor is world spawn, a no-path from a sealed room); the
                     ascent timing at two placement heights (seconds per block, bridge-side escape);
                     exhaustion AT REST for 60 s with the loop live (food / saturation stamped).
2. ``lethal_A``    — a fresh agent with the Wire-4 fear subscriber DETACHED, teleported to the pool
                     floor with the loop live and NOT rescued: does the innate ``health → threat``
                     need execute ``escape_water`` once damage lands (predicted ≈ 21 s), or does it
                     drown (predicted death ≈ 25–33 s)? Tick-by-tick.
3. ``lethal_B``    — the same with the subscriber ATTACHED (production default): does the agent book
                     fear on its own cluster and escape BEFORE damage (predicted ≈ 9–10 s)?
4. ``drown_on`` / ``drown_off`` — a DELIBERATE drowning with the loop live, game-native: the pool's
                     air cells are capped with stone for the window (restored after), so the agent
                     tries and cannot surface; ``naturalRegeneration`` true, then false, then restored.
                     Records the damage/death timeline, the ``deaths`` objective per sample, the
                     respawn discontinuity, and whether the loop keeps ticking through death.

Rows go to ``--out`` (default ``/tmp/r3_pilot.jsonl``). The window never rescues; the shore teleport
happens AFTER the window has ended (surfaced / dead / capped), which is the R3 gauntlet's shape.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _provenance import in_process_code_provenance  # noqa: E402
from exp56 import common as C  # noqa: E402  — REPO_ROOT + RconControl live here (as exp61_run)
from survival_world.common import InstrumentError, settle_until, sync_snapshot  # noqa: E402
from survival_world.exp60_run import ANCHOR_FILE, APPARATUS_RECORD, FROZEN  # noqa: E402
from survival_world.exp60_water_check import STALE_MAX_CONSECUTIVE, STALE_STATE_S  # noqa: E402
from survival_world.exp61_run import _build, _load_json, _rss_mb, close_and_stage  # noqa: E402
from survival_world.water_trial import (  # noqa: E402
    Refusal,
    WaterTrial,
    _detach_fear_subscriber,
    _f,
    _telemetry_ticks,
    min_pain_edge_s,
)

LETHAL_CAP_S = 45.0  # an agent alive underwater past this is an instrument fault (death ≈ 25–33 s)
DROWN_CAP_S = 60.0
REST_S = 60.0
POST_DEATH_S = 6.0  # keep sampling after a death to see the respawn + the loop still ticking
EXTRA_RULES = ("naturalRegeneration", "doDrowningDamage", "doInsomnia", "doWeatherCycle")


def _now_row(kind: str, **fields: Any) -> dict[str, Any]:
    return {"ts": time.time(), "kind": kind, "rss_mb": _rss_mb(), **fields}


def _write(out: Path, row: dict[str, Any]) -> None:
    with out.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


class Pilot:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        apparatus = _load_json(C.REPO_ROOT / APPARATUS_RECORD)
        self.geom = _load_json(ANCHOR_FILE)
        self.pain_edge_min = min_pain_edge_s(apparatus)
        self.t_damage_onset = float(self.geom["measured"]["t_damage_onset_min_s"])
        self.rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
        self.workdir = Path(args.workdir).expanduser()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.out = Path(args.out)
        self.provenance = in_process_code_provenance(
            C.REPO_ROOT, __import__("maxim").__file__, out_path=self.out, allow_dirty=args.allow_dirty
        )
        sub = self.geom["submerged"]
        # the record's exact submerged point, as WaterTrial teleports (the proven seam)
        self.floor = {"x": float(sub[0]), "y": float(sub[1]), "z": float(sub[2])}
        # the pool's first AIR layer is `surface_y` (the record's `shore_y`; water occupies
        # shore_y-depth .. shore_y-1); the cap is the 3x3 over the pool centre at that layer
        surf_y = int(self.geom["surface_y"])
        self.cap_slab = (sub[0] - 1, surf_y, sub[2] - 1, sub[0] + 1, surf_y, sub[2] + 1)

    # ── assembly ──

    def trial(self, agent_id: str, *, detach_fear: bool) -> tuple[WaterTrial, Any, Any]:
        home = self.workdir / agent_id
        aut, encoder, pump = _build(self.args, agent_id=agent_id, home=home)
        detached = _detach_fear_subscriber(aut) if detach_fear else 0
        trial = WaterTrial(
            aut=aut,
            rcon=self.rcon,
            username=self.args.username,
            geom=self.geom,
            frozen=FROZEN,
            probe_cap_s=self.pain_edge_min - FROZEN["probe_cap_margin_s"],
            train_cap_s=self.t_damage_onset - FROZEN["train_cap_margin_s"],
            persistence_dir=home,
            agent_id=agent_id,
            encoder=encoder,
            settle_guard={"is_raining": 0.0, "nearest_player_dist": 64.0},
        )
        trial.detached_subscribers = detached
        trial.attach_instruments()
        trial.resolve_tools()
        return trial, aut, pump

    def read_rules(self) -> dict[str, str]:
        from survival_world.water_trial import GAMERULES

        rules = {rule: self.rcon.command(f"gamerule {rule}").strip() for rule, _ in GAMERULES}
        for rule in EXTRA_RULES:
            rules[rule] = self.rcon.command(f"gamerule {rule}").strip()
        rules["difficulty"] = self.rcon.command("difficulty").strip()
        return rules

    # ── the no-rescue live window ──

    def live_window(
        self,
        trial: WaterTrial,
        *,
        label: str,
        enter: Any,
        until: Any,
        cap_s: float,
        linger_after_stop_s: float = 0.0,
    ) -> dict[str, Any]:
        """Like ``WaterTrial.loop_window`` but it NEVER rescues inside the window: the loop runs, the
        agent is `enter()`-ed, samples (with food/saturation/y/deaths) run at 4 Hz until `until(s)` or
        the cap; the shore teleport happens only AFTER the window (cleanup, not rescue)."""
        from maxim.simulation.minecraft_harness import run_minecraft_aut
        from maxim.simulation.substrate_telemetry import SubstrateTelemetry

        stop = threading.Event()
        calls0 = len(trial.calls)
        telem_path = trial.persistence_dir / f"telemetry_{label}_{int(time.time() * 1000)}.jsonl"
        telem = SubstrateTelemetry(log_path=telem_path, agent_id=trial.agent_id)
        loop = threading.Thread(
            target=run_minecraft_aut,
            args=(trial.aut,),
            kwargs={
                "max_steps": 100_000,
                "target_hz": FROZEN["loop_hz"],
                "stop_event": stop,
                "substrate_telemetry": telem,
            },
            daemon=True,
        )
        loop.start()
        time.sleep(FROZEN["loop_warm_s"])
        deaths0 = trial.deaths()
        samples: list[dict[str, Any]] = []
        stale = 0
        t0 = enter()
        stopped_at = None
        linger_until = None
        try:
            while True:
                t = time.monotonic() - t0
                if linger_until is not None and time.monotonic() >= linger_until:
                    break
                if linger_until is None and t >= cap_s:
                    break
                s = self.sample_full(trial, t0, deaths0)
                if s is None:
                    stale += 1
                    if stale >= STALE_MAX_CONSECUTIVE:
                        raise InstrumentError(f"{label}: bridge stopped delivering fresh state inside the window")
                else:
                    stale = 0
                    samples.append(s)
                    if linger_until is None and until(s):
                        stopped_at = s["t"]
                        if linger_after_stop_s > 0:
                            linger_until = time.monotonic() + linger_after_stop_s
                        else:
                            break
                time.sleep(0.25)
        finally:
            t_end = time.monotonic()
            stop.set()
            loop.join(timeout=20.0)
            stuck = loop.is_alive()
            try:
                trial.stop_motion()
                self.rcon.teleport(trial.username, trial.shore)  # cleanup AFTER the window, never a rescue
            except Exception as exc:
                print(f"WARNING: post-window teleport raised: {exc!r}")
            trial.reopen_hub_session()
        if stuck:
            raise Refusal(f"{label}: loop thread did not stop")
        ticks = _telemetry_ticks(telem_path, t0)
        window_calls = [{**c, "t": round(c["t"] - t0, 3)} for c in trial.calls[calls0:]]
        return {
            "label": label,
            "samples": samples,
            "calls": window_calls,
            "ticks": ticks,
            "n_ticks": len(ticks),
            "stopped_at": stopped_at,
            "t_end": round(t_end - t0, 3),
            "pain": [
                {"t": round(p["t"] - t0, 3), "failure_mode": p["failure_mode"], "intensity": p["intensity"]}
                for p in trial.signals
                if t0 <= p["t"] <= t_end
            ],
            "deaths_delta": trial.deaths() - deaths0,
        }

    def sample_full(self, trial: WaterTrial, t0: float, deaths0: int) -> dict[str, Any] | None:
        if trial.aut.client.state_age_s() > STALE_STATE_S:
            return None
        vm = sync_snapshot(trial.aut)
        if vm is None or "is_in_water" not in vm:
            return None
        return {
            "t": round(time.monotonic() - t0, 3),
            "in_water": _f(vm, "is_in_water", 0) >= 0.5,
            "health": _f(vm, "health", 20.0),
            "oxygen": _f(vm, "oxygen", 20.0),
            "food": _f(vm, "food", -1.0),
            "saturation": _f(vm, "saturation", -1.0),
            "y": _f(vm, "y_altitude", -1.0),
            "deaths_delta": trial.deaths() - deaths0,  # per sample over local RCON (the R3 detector shape)
        }

    # ── helpers ──

    def submerge_to(self, trial: WaterTrial, pos: dict[str, float], label: str) -> float:
        trial.stop_motion()
        t_tp = time.monotonic()
        self.rcon.teleport(trial.username, pos)
        if settle_until(trial.aut, lambda vm: _f(vm, "is_in_water", 0) >= 0.5, timeout_s=3.0) is None:
            self.rcon.teleport(trial.username, trial.shore)
            raise Refusal(f"{label}: is_in_water did not reflect the teleport to {pos}")
        return t_tp

    def bridge_escape_timing(self, trial: WaterTrial, pos: dict[str, float], label: str) -> dict[str, Any]:
        """Bridge-side escape from `pos` (never the executor); returns t_surface + bridge detail."""
        t0 = self.submerge_to(trial, pos, label)
        outcome: dict[str, Any] = {}

        def _go() -> None:
            try:
                outcome.update(trial.aut.client.call_action("escape_water", {}))
            except Exception as exc:
                outcome.update({"ok": False, "detail": repr(exc)})

        th = threading.Thread(target=_go, daemon=True)
        th.start()
        surfaced = None
        while time.monotonic() - t0 < 12.0:
            s = trial.sample(t0)
            if s is not None and not s["in_water"]:
                surfaced = s["t"]
                break
            time.sleep(0.25)
        th.join(timeout=10.0)
        self.rcon.teleport(trial.username, trial.shore)
        trial.rescue(label)
        return {"pos": pos, "t_surface": surfaced, "bridge": {k: outcome.get(k) for k in ("ok", "detail")}}

    def flee_preflight(self, trial: WaterTrial) -> dict[str, Any]:
        """`flee` through the BRIDGE on the shore: the anchor must be reachable fast (v1's default was
        world spawn, ≈ 69 blocks away through stone = a multi-second no-path)."""
        trial.rescue("flee-preflight")
        t0 = time.monotonic()
        try:
            out = trial.aut.client.call_action("flee", {})
        except Exception as exc:
            out = {"ok": False, "detail": repr(exc)}
        return {"latency_s": round(time.monotonic() - t0, 3), "bridge": {k: out.get(k) for k in ("ok", "detail")}}

    def cap_pool(self, on: bool) -> None:
        x0, y, z0, x1, _, z1 = self.cap_slab
        block = "minecraft:stone" if on else "minecraft:air"
        self.rcon.command(f"fill {x0} {y} {z0} {x1} {y} {z1} {block}")

    def set_regen(self, on: bool) -> str:
        return self.rcon.command(f"gamerule naturalRegeneration {'true' if on else 'false'}").strip()

    # ── rows ──

    def row_apparatus(self) -> None:
        trial, aut, pump = self.trial("r3pilot_apparatus", detach_fear=False)
        row: dict[str, Any] = {"rules": self.read_rules()}
        try:
            row["bridge_state_interval_s"] = trial.check_bridge()
            row["loop_liveness_ticks"] = trial.check_liveness()
            trial.check_gamerules()
            row["clusters_distinct"] = list(trial.check_clusters_distinct())
            trial.rescue("apparatus")
            row["flee_preflight"] = self.flee_preflight(trial)
            floor = dict(self.floor)
            two_up = {**floor, "y": floor["y"] + 2.0}
            row["ascent_from_floor"] = self.bridge_escape_timing(trial, floor, "ascent-floor")
            row["ascent_from_floor_plus_2"] = self.bridge_escape_timing(trial, two_up, "ascent-plus2")
            a, b = row["ascent_from_floor"]["t_surface"], row["ascent_from_floor_plus_2"]["t_surface"]
            row["ascent_s_per_block"] = round((a - b) / 2.0, 3) if a is not None and b is not None else None
            row["column_depth"] = self.geom.get("depth")
            # exhaustion at rest, loop live, on the shore
            rest = self.live_window(
                trial,
                label="rest",
                enter=lambda: (trial.rescue("rest"), time.monotonic())[1],
                until=lambda s: False,
                cap_s=REST_S,
            )
            row["rest"] = {
                "n_samples": len(rest["samples"]),
                "n_ticks": rest["n_ticks"],
                "food_first": rest["samples"][0]["food"] if rest["samples"] else None,
                "food_last": rest["samples"][-1]["food"] if rest["samples"] else None,
                "saturation_first": rest["samples"][0]["saturation"] if rest["samples"] else None,
                "saturation_last": rest["samples"][-1]["saturation"] if rest["samples"] else None,
                "calls": rest["calls"],
            }
            row["refusal"] = None
        except (Refusal, InstrumentError) as exc:
            row["refusal"] = str(exc)
            row.update(getattr(exc, "partial", None) or {})
        finally:
            trial.detach_instruments()
            try:
                close_and_stage(aut, pump, None)
            except Exception as exc:
                print(f"WARNING: apparatus close: {exc!r}")
        _write(self.out, _now_row("apparatus", provenance=self.provenance, **row))
        print(f"apparatus: {json.dumps({k: v for k, v in row.items() if k not in ('rest',)}, default=str)[:600]}")

    def row_lethal(self, kind: str, *, detach_fear: bool) -> None:
        trial, aut, pump = self.trial(f"r3pilot_{kind}", detach_fear=detach_fear)
        row: dict[str, Any] = {"subscriber_detached": detach_fear, "detached_count": trial.detached_subscribers}
        try:
            trial.rescue("lethal-ready")
            row["fear_before"] = trial.fear_dump()
            row["links_before"] = trial.positive_escape_links()
            win = self.live_window(
                trial,
                label=kind,
                enter=lambda: self.submerge_to(trial, self.floor, kind),
                until=lambda s: (not s["in_water"]) or s["deaths_delta"] > 0,
                cap_s=LETHAL_CAP_S,
                linger_after_stop_s=2.0,
            )
            surf = next((s["t"] for s in win["samples"] if not s["in_water"] and s["deaths_delta"] == 0), None)
            death = next((s["t"] for s in win["samples"] if s["deaths_delta"] > 0), None)
            row.update(
                {
                    "t_surface": surf,
                    "t_death": death,
                    "escaped_before_damage": surf is not None and surf < self.t_damage_onset,
                    "min_health": min(s["health"] for s in win["samples"]) if win["samples"] else None,
                    "min_oxygen": min(s["oxygen"] for s in win["samples"]) if win["samples"] else None,
                    "first_pain": win["pain"][:3],
                    "n_pain": len(win["pain"]),
                    "calls": win["calls"],
                    # tick clock = the loop's FIRST tick (≈ loop_warm_s before the teleport)
                    "first_proposals": [
                        (t["t_from_first_tick"], t["proposal"], (t.get("drives") or {}).get("threat"))
                        for t in win["ticks"]
                        if t.get("proposal")
                    ][:6],
                    "n_ticks": win["n_ticks"],
                    "capped": surf is None and death is None,
                    "fear_after": trial.fear_dump(),
                    "links_after": trial.positive_escape_links(),
                    "samples": win["samples"],
                    "refusal": None,
                }
            )
        except (Refusal, InstrumentError) as exc:
            row["refusal"] = str(exc)
            row.update(getattr(exc, "partial", None) or {})
        finally:
            trial.detach_instruments()
            try:
                close_and_stage(aut, pump, None)
            except Exception as exc:
                print(f"WARNING: {kind} close: {exc!r}")
        _write(self.out, _now_row(kind, provenance=self.provenance, **row))
        print(
            f"{kind}: surface={row.get('t_surface')} death={row.get('t_death')} before_damage={row.get('escaped_before_damage')} "
            f"min_health={row.get('min_health')} calls={[(c['tool'], c['success']) for c in row.get('calls', [])]} "
            f"fear_after={row.get('fear_after')} refusal={row.get('refusal')}"
        )

    def row_drown(self, kind: str, *, regen_on: bool) -> None:
        trial, aut, pump = self.trial(f"r3pilot_{kind}", detach_fear=False)
        row: dict[str, Any] = {"regen_on": regen_on}
        try:
            row["regen_set"] = self.set_regen(regen_on)
            trial.rescue("drown-ready")
            self.cap_pool(True)
            win = self.live_window(
                trial,
                label=kind,
                enter=lambda: self.submerge_to(trial, self.floor, kind),
                until=lambda s: s["deaths_delta"] > 0,
                cap_s=DROWN_CAP_S,
                linger_after_stop_s=POST_DEATH_S,
            )
            death = next((s["t"] for s in win["samples"] if s["deaths_delta"] > 0), None)
            after = [s for s in win["samples"] if death is not None and s["t"] > death]
            damage_first = next((s["t"] for s in win["samples"] if s["health"] < 20.0), None)
            row.update(
                {
                    "t_first_damage": damage_first,
                    "t_death": death,
                    "deaths_delta": win["deaths_delta"],
                    "health_series": [(s["t"], s["health"]) for s in win["samples"]],
                    "respawn_first_sample": after[0] if after else None,
                    # tick clock = the loop's first tick (≈ loop_warm_s before the teleport); approximate
                    "ticks_after_death": sum(
                        1
                        for t in win["ticks"]
                        if death is not None and t["t_from_first_tick"] > death + FROZEN["loop_warm_s"]
                    ),
                    "n_ticks": win["n_ticks"],
                    "calls": win["calls"],
                    "capped_window": death is None,
                    "refusal": None,
                }
            )
        except (Refusal, InstrumentError) as exc:
            row["refusal"] = str(exc)
        finally:
            try:
                self.cap_pool(False)
            finally:
                row["regen_restored"] = self.set_regen(True)
            trial.detach_instruments()
            try:
                close_and_stage(aut, pump, None)
            except Exception as exc:
                print(f"WARNING: {kind} close: {exc!r}")
        _write(self.out, _now_row(kind, provenance=self.provenance, **row))
        print(
            f"{kind}: first_damage={row.get('t_first_damage')} death={row.get('t_death')} ticks_after_death={row.get('ticks_after_death')} "
            f"respawn={row.get('respawn_first_sample')} refusal={row.get('refusal')}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rcon-host", default="127.0.0.1")
    p.add_argument("--rcon-port", type=int, default=25575)
    p.add_argument("--rcon-password", required=True)
    p.add_argument("--username", default="maxim")
    p.add_argument("--bridge-host", default="127.0.0.1")
    p.add_argument("--bridge-port", type=int, default=25567)
    p.add_argument("--workdir", default="~/r3_pilot")
    p.add_argument("--out", default="/tmp/r3_pilot.jsonl")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument(
        "--only", default="", help="comma list of rows to run: apparatus,lethal_A,lethal_B,drown_on,drown_off"
    )
    args = p.parse_args()
    pilot = Pilot(args)
    only = {s for s in args.only.split(",") if s}
    steps = [
        ("apparatus", pilot.row_apparatus),
        ("lethal_A", lambda: pilot.row_lethal("lethal_A", detach_fear=True)),
        ("lethal_B", lambda: pilot.row_lethal("lethal_B", detach_fear=False)),
        ("drown_on", lambda: pilot.row_drown("drown_on", regen_on=True)),
        ("drown_off", lambda: pilot.row_drown("drown_off", regen_on=False)),
    ]
    print(f"r3 pilot -> {pilot.out}; pain edge {pilot.pain_edge_min:.2f}s damage onset {pilot.t_damage_onset:.2f}s")
    for name, fn in steps:
        if only and name not in only:
            continue
        print(f"\n=== {name} ===")
        fn()
    print(f"\nr3 pilot done -> {pilot.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
