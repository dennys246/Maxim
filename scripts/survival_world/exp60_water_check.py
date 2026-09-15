#!/usr/bin/env python3
"""Exp 60 water classroom — the LIVE apparatus instrument check (gated evidence).

Chunk (i) of the Exp 60 build (docs/experiments/exp60_drowning_avoidance_prereg.md): the
water classroom built by ``setup_world.py water_classroom`` must be MEASURED to do what the
design assumes before the separability probe (chunk ii) or the trial harness (chunk iii)
touch it. Every environment-lens "live unknown" (rationale/exp60-drowning/environment.md
E2/E4/E5, wiring SF-2/SF-3) is a check here, read through the SENSED body values
(``common.sync_snapshot`` — range-clamp aware, docs/wiring/sensor-range-clamps.md), never a
timer or the action's self-report:

``W1 shore``      dry rest baseline: is_in_water 0, on_ground 1, oxygen full, health full,
                  hostile_count 0, nearest_hostile_dist at the 64 horizon (= sensor neutral).
``W2 dive``       teleport to the pool FLOOR: is_in_water flips to 1 fast; the idle bot SINKS
                  (no auto-float — flag and depth hold for 6 s); oxygen depletes monotonically
                  through the pain edge (< 14 bubbles) to 0; exactly ONE drowning-damage tick
                  is allowed to land so the damage-onset edge of the DV window is a measured
                  number (expected ~15 s), then rescue. Gate: onset in [12, 20] s, health >= 16.
``W3 recover``    rescue to the dry shore restores sensed oxygen >= 19 within 10 s (the
                  homeostatic latch clears only on OBSERVED recovery — wiring SF-3).
``W4 escape``     the REAL registered ``escape_water`` tool, executed through
                  ``aut.executor.execute`` (the production consumer, as Exp 58's flee preflight),
                  puts the head in air within 6 s of the call (bridge truth ``is_in_water`` 0;
                  the bridge caps the action at 8 s; the window is 15 s).
``W5 sink-back``  informational: how long after the action releases jump the head is back
                  underwater — tells the harness how promptly rescue must follow a surface.

Three cycles; PASS requires every gated check on every cycle (min/max gating, as the Phase-0
light check: a one-cycle flake is the regression this pins). Instrument evidence only — no
credit path runs, no learned bias forms. Every completed run lands a record through the gated
evidence path (an absent record must mean "never ran"). Exit codes: 0 pass, 2 usage,
3 provenance/dirty-tree refusal, 4 measured FAIL or instrument error (``instrument_error``
separates them).

LIVE-ONLY: run ON the box hosting the bridge (server up, bridge connected, ``prepare`` and
``water_classroom`` already run):

    python scripts/survival_world/exp60_water_check.py --rcon-password '<pw>' --username maxim
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _provenance import (  # noqa: E402
    DirtyTreeError,
    ProvenanceError,
    evidence_out_paths_or_exit,
    in_process_code_provenance,
)
from exp56 import common as C  # noqa: E402
from survival_world.common import InstrumentError, settle_until, sync_snapshot  # noqa: E402

ANCHOR_FILE = Path.home() / ".maxim" / "exp60_water_classroom.json"

# ── Gates (fixed by intent before the first live run; the prereg freezes them) ──
CYCLES = 3
SAMPLE_S = 0.25  # the bridge/pump cadence (4 Hz)
IN_WATER_WITHIN_S = 3.0  # teleport -> is_in_water 1 (settle-until-reflected)
SINK_HOLD_S = 6.0  # idle bot must stay submerged this long (no auto-float)
SINK_Y_TOLERANCE = 1.0  # feet stay within this of the pool floor while holding
OXYGEN_FULL = 20.0  # bridge oxygenLevel [0,20]; body range [0,40] puts rest at neutral
OXYGEN_PAIN_EDGE = 13.0  # drive set_point 20 - comfort_band 6: pain fires below 14 bubbles
DAMAGE_ONSET_WINDOW_S = (12.0, 20.0)  # ~15 s of air in 1.20.4; the DV window's far edge
DIVE_CAP_S = 22.0  # rescue regardless (death ~25 s from full health)
RESCUE_HEALTH_MIN = 16.0  # at most two 2-hp drowning ticks may land before the rescue
RECOVER_OXYGEN_MIN = 19.0
RECOVER_WITHIN_S = 10.0
SURFACE_WITHIN_S = 6.0  # escape_water: head in air within this (bridge action cap is 8 s)
SINKBACK_CAP_S = 10.0
HOSTILE_HORIZON = 64.0  # bridge cap for nearest_hostile_dist == the sensor's neutral midpoint
FROZEN_GAMERULES = {"doMobSpawning": "false", "doDaylightCycle": "false", "doWeatherCycle": "false"}


# ─────────────────────────── pure evaluators (unit-tested) ───────────────────────────


def evaluate_dive(samples: list[dict[str, Any]], *, floor_y: float) -> dict[str, Any]:
    """Classify one dive from ``(t, oxygen, health, y, in_water)`` samples (t from teleport).

    Pure: no I/O. Returns the measured edges plus ``pass`` and the failing ``reasons``.
    """
    reasons: list[str] = []
    t_in = next((s["t"] for s in samples if s["in_water"]), None)
    if t_in is None or t_in > IN_WATER_WITHIN_S:
        reasons.append(f"is_in_water never reached 1 within {IN_WATER_WITHIN_S}s (t_in={t_in})")
    hold = [s for s in samples if t_in is not None and t_in <= s["t"] <= t_in + SINK_HOLD_S]
    sink_hold = bool(hold) and all(s["in_water"] and s["y"] <= floor_y + SINK_Y_TOLERANCE for s in hold)
    if not sink_hold:
        reasons.append(f"bot did not stay submerged at the floor for {SINK_HOLD_S}s (auto-float / drift)")
    wet = [s for s in samples if s["in_water"]]
    oxy = [s["oxygen"] for s in wet]
    monotone = all(b <= a for a, b in zip(oxy, oxy[1:]))
    if not monotone:
        reasons.append("oxygen rose while submerged (air pocket / head cell not water)")
    t_pain = next((s["t"] for s in wet if s["oxygen"] <= OXYGEN_PAIN_EDGE), None)
    t_zero = next((s["t"] for s in wet if s["oxygen"] <= 0), None)
    full_health = samples[0]["health"] if samples else None
    t_damage = next((s["t"] for s in samples if full_health is not None and s["health"] < full_health), None)
    lo, hi = DAMAGE_ONSET_WINDOW_S
    if t_damage is None or not (lo <= t_damage <= hi):
        reasons.append(f"first drowning damage not in [{lo}, {hi}]s (t_damage={t_damage})")
    if t_pain is None or (t_damage is not None and t_pain >= t_damage):
        reasons.append(f"oxygen pain edge (<= {OXYGEN_PAIN_EDGE}) not reached before damage (t_pain={t_pain})")
    health_end = samples[-1]["health"] if samples else None
    if health_end is None or health_end < RESCUE_HEALTH_MIN:
        reasons.append(f"health at rescue {health_end} < {RESCUE_HEALTH_MIN} (rescue too slow)")
    return {
        "t_in_water": t_in,
        "sink_hold": sink_hold,
        "oxygen_monotone": monotone,
        "t_pain_edge": t_pain,
        "t_oxygen_zero": t_zero,
        "t_damage_onset": t_damage,
        "health_at_rescue": health_end,
        "n_samples": len(samples),
        "pass": not reasons,
        "reasons": reasons,
    }


def evaluate_surface(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify one escape from ``(t, in_water, oxygen)`` samples (t from the tool call).

    ``t_surface`` = first head-in-air read; ``t_sinkback`` = first re-submerge after it
    (informational — the harness's rescue budget).
    """
    reasons: list[str] = []
    t_surface = next((s["t"] for s in samples if not s["in_water"]), None)
    if t_surface is None or t_surface > SURFACE_WITHIN_S:
        reasons.append(f"head not in air within {SURFACE_WITHIN_S}s of escape_water (t_surface={t_surface})")
    t_sinkback = None
    if t_surface is not None:
        t_sinkback = next((s["t"] for s in samples if s["t"] > t_surface and s["in_water"]), None)
    return {
        "t_surface": t_surface,
        "t_sinkback": t_sinkback,
        "n_samples": len(samples),
        "pass": not reasons,
        "reasons": reasons,
    }


# ─────────────────────────────────── live ───────────────────────────────────


def _f(vm: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(vm.get(key, default))
    except (TypeError, ValueError):
        return default


def _sample(aut: Any, t0: float) -> dict[str, Any] | None:
    vm = sync_snapshot(aut)
    if vm is None:
        return None
    return {
        "t": round(time.monotonic() - t0, 3),
        "oxygen": _f(vm, "oxygen", OXYGEN_FULL),
        "health": _f(vm, "health", 20.0),
        "y": _f(vm, "y_altitude", 0.0),
        "in_water": _f(vm, "is_in_water", 0.0) >= 0.5,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/experiments/data/exp60_water_apparatus.json")
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25567)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", default=os.environ.get("SURVIVAL_RCON_PASSWORD", ""))
    ap.add_argument("--username", default="maxim", help="the bridge bot's username")
    ap.add_argument("--cycles", type=int, default=CYCLES)
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    if not args.rcon_password:
        ap.error("--rcon-password (or SURVIVAL_RCON_PASSWORD) is required")

    out_arg = Path(args.out)
    out_abs = out_arg if out_arg.is_absolute() else (C.REPO_ROOT / out_arg)
    out_path = evidence_out_paths_or_exit(
        C.REPO_ROOT,
        [str(out_abs)],
        write_experiment_results=args.write_experiment_results,
        allow_dirty=args.allow_dirty,
    )[0]

    import maxim  # the code under test — in-process, so the in-process provenance door

    try:
        provenance = in_process_code_provenance(
            C.REPO_ROOT, maxim.__file__, out_path=out_path, allow_dirty=args.allow_dirty
        )
    except (DirtyTreeError, ProvenanceError) as exc:
        print(f"[FAIL] provenance: {exc}")
        return 3

    from maxim.simulation.minecraft_harness import build_minecraft_aut

    report: dict[str, Any] = {
        "ts": time.time(),
        "gates": {
            "cycles": args.cycles,
            "in_water_within_s": IN_WATER_WITHIN_S,
            "sink_hold_s": SINK_HOLD_S,
            "oxygen_pain_edge": OXYGEN_PAIN_EDGE,
            "damage_onset_window_s": list(DAMAGE_ONSET_WINDOW_S),
            "rescue_health_min": RESCUE_HEALTH_MIN,
            "recover_within_s": RECOVER_WITHIN_S,
            "surface_within_s": SURFACE_WITHIN_S,
            "hostile_horizon": HOSTILE_HORIZON,
        },
        "provenance": provenance,
        "cycles": [],
        "instrument_error": None,
    }

    def _finish(code: int) -> int:
        cycles = report["cycles"]
        report["all_pass"] = bool(cycles) and len(cycles) == args.cycles and all(c["pass"] for c in cycles)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k] for k in ("cycles", "instrument_error", "all_pass")}, indent=2))
        print(f"exp60 water apparatus: {'PASS' if report['all_pass'] else 'FAIL'} -> {out_path}")
        return code

    try:
        geom = json.loads(ANCHOR_FILE.read_text())
    except (OSError, ValueError) as exc:
        report["instrument_error"] = f"no water classroom record at {ANCHOR_FILE} ({exc}) — build first"
        print(f"INSTRUMENT ERROR: {report['instrument_error']}")
        return _finish(4)
    shore = {"x": float(geom["shore"][0]), "y": float(geom["shore"][1]), "z": float(geom["shore"][2])}
    sub = {"x": float(geom["submerged"][0]), "y": float(geom["submerged"][1]), "z": float(geom["submerged"][2])}
    report["apparatus"] = {"anchor_file": str(ANCHOR_FILE), **{k: geom[k] for k in ("shore", "submerged", "depth")}}

    persistence_dir = tempfile.mkdtemp(prefix="exp60_water_check_")
    aut = build_minecraft_aut(
        agent_id="exp60_water_check",
        bridge_port=args.bridge_port,
        bridge_host=args.bridge_host,
        persistence_dir=persistence_dir,
        entity_ref="bodies/minecraft_player",
    )
    rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)

    def _heal() -> None:
        rcon.command(f"effect give {args.username} minecraft:instant_health 1 10 true")
        rcon.command(f"effect give {args.username} minecraft:saturation 1 10 true")

    def _rescue() -> None:
        rcon.teleport(args.username, shore)

    try:
        # Startup gate: the bridge must deliver the Exp 60 sensors before anything is measured.
        if settle_until(aut, lambda vm: "is_in_water" in vm and "oxygen" in vm, timeout_s=8.0) is None:
            raise InstrumentError(
                "bridge never delivered a snapshot carrying is_in_water + oxygen — bridge dead/busy "
                "or body regression; check the bridge terminal for 'bridge busy: one client at a time'."
            )
        # Apparatus-owned world conditions are VERIFIED, not toggled (the Phase-0 instrument
        # check restores doMobSpawning to true on exit — a stale world must refuse here).
        for rule, want in FROZEN_GAMERULES.items():
            resp = rcon.command(f"gamerule {rule}").strip().lower()
            if want not in resp:
                raise InstrumentError(
                    f"gamerule {rule} is not {want} ({resp!r}) — re-run `setup_world.py prepare` "
                    "and `setup_world.py water_classroom` before the check."
                )
        tool = next((t for t in aut.executor.registry.list() if t.endswith("_escape_water")), None)
        if tool is None:
            raise InstrumentError("no *_escape_water tool registered — body/executor mis-wired (Exp 60 substrate)")

        for cycle in range(args.cycles):
            rec: dict[str, Any] = {"cycle": cycle}
            # ── W1 shore baseline ──
            _rescue()
            _heal()
            vm = settle_until(
                aut,
                lambda vm: (
                    _f(vm, "is_in_water", 1) < 0.5
                    and _f(vm, "on_ground", 0) >= 0.5
                    and _f(vm, "oxygen", 0) >= RECOVER_OXYGEN_MIN
                    and _f(vm, "health", 0) >= 20.0
                ),
                timeout_s=RECOVER_WITHIN_S,
            )
            if vm is None:
                raise InstrumentError("shore baseline never settled (dry, grounded, full air, full health)")
            hostile_count = _f(vm, "hostile_count", 99)
            nearest = _f(vm, "nearest_hostile_dist", 0)
            w1 = {
                "is_in_water": _f(vm, "is_in_water", -1),
                "on_ground": _f(vm, "on_ground", -1),
                "oxygen": _f(vm, "oxygen", -1),
                "health": _f(vm, "health", -1),
                "y_altitude": _f(vm, "y_altitude", -1),
                "hostile_count": hostile_count,
                "nearest_hostile_dist": nearest,
                "pass": hostile_count == 0 and nearest >= HOSTILE_HORIZON,
            }
            rec["w1_shore"] = w1

            # ── W2 dive: sample until the FIRST damage tick (or the cap), then rescue ──
            rcon.teleport(args.username, sub)
            t0 = time.monotonic()
            samples: list[dict[str, Any]] = []
            while time.monotonic() - t0 < DIVE_CAP_S:
                s = _sample(aut, t0)
                if s is not None:
                    samples.append(s)
                    if s["health"] < 20.0:
                        break
                time.sleep(SAMPLE_S)
            _rescue()
            t_rescue = time.monotonic()
            if not samples:
                raise InstrumentError("no dive samples (bridge stopped delivering state mid-dive)")
            w2 = evaluate_dive(samples, floor_y=sub["y"])
            w2["series"] = [(s["t"], s["oxygen"], s["health"]) for s in samples]
            rec["w2_dive"] = w2

            # ── W3 recovery on the dry shore ──
            vm = settle_until(
                aut,
                lambda vm: _f(vm, "is_in_water", 1) < 0.5 and _f(vm, "oxygen", 0) >= RECOVER_OXYGEN_MIN,
                timeout_s=RECOVER_WITHIN_S,
            )
            t_recover = None if vm is None else round(time.monotonic() - t_rescue, 3)
            rec["w3_recover"] = {"t_recover": t_recover, "pass": t_recover is not None}
            _heal()

            # ── W4 escape_water through the REAL executor (bridge truth decides) ──
            rcon.teleport(args.username, sub)
            if settle_until(aut, lambda vm: _f(vm, "is_in_water", 0) >= 0.5, timeout_s=IN_WATER_WITHIN_S) is None:
                raise InstrumentError("W4: is_in_water did not reflect the submerged teleport")
            outcome: dict[str, Any] = {}

            def _run_tool(tool_name: str = tool, sink: dict[str, Any] = outcome) -> None:
                try:
                    out = aut.executor.execute({"tool_name": tool_name, "params": {}})
                    sink["success"] = getattr(out, "success", None)
                    sink["error"] = getattr(out, "error", None)
                    sink["result"] = str(getattr(out, "result", ""))[:160]
                except Exception as exc:  # recorded as the outcome, not swallowed
                    sink["success"] = False
                    sink["error"] = repr(exc)

            t0 = time.monotonic()
            th = threading.Thread(target=_run_tool, daemon=True)
            th.start()
            esc: list[dict[str, Any]] = []
            surfaced_at: float | None = None
            while True:
                s = _sample(aut, t0)
                if s is not None:
                    esc.append(s)
                    if surfaced_at is None and not s["in_water"]:
                        surfaced_at = s["t"]
                    elif surfaced_at is not None and s["in_water"]:
                        break  # sink-back observed
                now = time.monotonic() - t0
                if not th.is_alive() and (surfaced_at is None or now - surfaced_at > SINKBACK_CAP_S):
                    break
                if now > SURFACE_WITHIN_S + SINKBACK_CAP_S + 8.0:  # never hang on a stuck action
                    break
                time.sleep(SAMPLE_S)
            th.join(timeout=2.0)
            w4 = evaluate_surface(esc)
            w4["tool"] = tool
            w4["executor"] = outcome
            rec["w4_escape"] = w4
            rec["w5_sinkback_s"] = w4["t_sinkback"]
            _rescue()
            _heal()

            rec["pass"] = bool(w1["pass"] and w2["pass"] and rec["w3_recover"]["pass"] and w4["pass"])
            report["cycles"].append(rec)
            print(
                f"cycle {cycle}: shore={'ok' if w1['pass'] else 'FAIL'} "
                f"dive={'ok' if w2['pass'] else 'FAIL'} (in_water {w2['t_in_water']}s, pain {w2['t_pain_edge']}s, "
                f"damage {w2['t_damage_onset']}s) recover={t_recover}s "
                f"escape={'ok' if w4['pass'] else 'FAIL'} (surface {w4['t_surface']}s, sinkback {w4['t_sinkback']}s)"
            )
            for r in w2["reasons"] + w4["reasons"]:
                print(f"   - {r}")
    except InstrumentError as exc:
        report["instrument_error"] = str(exc)
        print(f"INSTRUMENT ERROR: {exc}")
        return _finish(4)
    except Exception as exc:  # record the failure as evidence, then surface it fully
        report["instrument_error"] = f"unexpected: {exc!r}"
        traceback.print_exc()
        return _finish(4)
    finally:
        # Guarded individually: a broken RCON must not mask the original error or leave
        # the bot submerged. The classroom stays built and doMobSpawning stays FALSE —
        # both are apparatus-owned conditions, not this check's to restore.
        try:
            _rescue()
            _heal()
        except Exception as exc:
            print(f"WARNING: could not rescue/heal the bot on exit: {exc!r}")
        rcon.close()
        try:
            aut.bio.on_session_end()
        except Exception as exc:
            print(f"WARNING: bio teardown raised: {exc!r}")
        try:
            aut.client.close()
        except (OSError, ConnectionError) as exc:
            print(f"WARNING: bridge client close raised: {exc!r}")
        shutil.rmtree(persistence_dir, ignore_errors=True)  # throwaway fresh substrate

    code = _finish(0 if all(c["pass"] for c in report["cycles"]) else 4)
    if code != 0:
        print(
            "  A gated check failed on at least one cycle — read the cycle's `reasons`. The pool\n"
            "  is not run-authorized until every cycle passes; chunk (ii)'s separability probe\n"
            "  must not run on an apparatus whose dynamics are unverified."
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
