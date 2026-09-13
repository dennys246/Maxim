#!/usr/bin/env python3
"""1.3 survival world Phase 0 — Step-1 instrument checks (light + world-channel separability).

Two checks gate the Phase-1 prereg (roadmap_1_3.md Phase 0: "verify sensor separability
through the real encoder before any claim"):

``check1_light``  the perceived-brightness sensor (docs/wiring/world-light-sensing.md) reads
    both sides live, EVERY cycle: >= 13 on the daylight surface, <= 1 inside an RCON-built
    roofed box (min/max gated, not medians — an intermittent flicker of the very regression
    this pins, the 0 -> 15 block-light fix, must not hide under a median).

``check2_world_separability``  the exp56 check-1 pattern on the SURVIVAL body: rest->dark
    onsets through the production encode (``_encode_current_clusters`` with the production
    encoder wiring — ec + atl + nac, see ``common.make_fresh_encoder``); separation (onset
    changes the WORLD cluster) and stability (repeat re-completes it) both >= 0.70, with a
    MINIMUM-N gate (>= 3/4 of attempted transitions and repeats must actually encode) so a
    mostly-dead encoder can neither pass vacuously nor fail as a fake measurement. This is
    the L11 question made concrete: the survival body declares 17 ``modality: world``
    sensors, the A4 nonlinear gain is ALREADY the shipped default for the world modality
    (``SensorEncoderConfig.gain_modalities``), and the 1.1.4 re-measure still scored A4
    separation 0.0566 << 0.70 on a SINGLE-sensor swing at this N (docs/limits/
    l11_sensor_dilution.md) — a dark-box transition moves several sensors at once (light,
    y_altitude, sky exposure), so whether the REAL transition separates is a genuine
    measurement and a FAIL is a live possibility. A FAIL here BLOCKS the Phase-1 prereg.

Instrument evidence, not a behavioural claim: no credit path runs, no learned bias forms.
Every completed run — PASS, measured FAIL, or instrument error — lands a record through the
gated-evidence path so the Phase-1 prereg can cite it (an absent record must mean "never
ran", not "failed"). Exit codes: 0 pass, 2 usage, 3 provenance/dirty-tree refusal,
4 measured FAIL or instrument error (the record's ``instrument_error`` field separates them).

LIVE-ONLY (no --mock): run ON the box hosting the bridge (it binds 127.0.0.1), server up +
bridge connected + prepare already run:

    python scripts/survival_world/instrument_check.py --rcon-password '<pw>' --username maxim
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from _provenance import (  # noqa: E402
    DirtyTreeError,
    ProvenanceError,
    evidence_out_paths_or_exit,
    in_process_code_provenance,
)
from exp56 import common as C  # noqa: E402
from survival_world.common import (  # noqa: E402
    InstrumentError,
    bot_pos,
    build_dark_box,
    make_fresh_encoder,
    remove_dark_box,
    settle_until,
)

SEPARABILITY_BAR = 0.70  # exp56 Phase-0 check-1 bar, reused deliberately (same instrument class)
DAY_LIGHT_MIN = 13.0  # noon surface reads 15; margin for edge-of-chunk shade
DARK_LIGHT_MAX = 1.0  # roofed stone box: block 0, sky 0
CYCLES = 20  # rest, dark, dark per cycle (exp56 check-1 shape)
LIGHT_KEY = "light_level"  # the body's declared world light sensor (minecraft_player.yaml)
BOX_ALTITUDE = 80  # box floor this far above the rest anchor's y (clear of local terrain)
SETTLE_TIMEOUT_S = 10.0
Y_TOLERANCE = 3.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/experiments/data/survival_phase0.json")
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

    # Anchor a relative --out to the REPO root, not the cwd — a cwd-relative path silently
    # escapes the D27 evidence redirect and the dirty-tree gate (governance is path-based).
    out_arg = Path(args.out)
    out_abs = out_arg if out_arg.is_absolute() else (C.REPO_ROOT / out_arg)
    out_path = evidence_out_paths_or_exit(
        C.REPO_ROOT,
        [str(out_abs)],
        write_experiment_results=args.write_experiment_results,
        allow_dirty=args.allow_dirty,
    )[0]

    import maxim  # the code under test — imported in-process, so use the in-process door

    try:
        provenance = in_process_code_provenance(
            C.REPO_ROOT, maxim.__file__, out_path=out_path, allow_dirty=args.allow_dirty
        )
    except (DirtyTreeError, ProvenanceError) as exc:
        print(f"[FAIL] provenance: {exc}")
        return 3

    from maxim.runtime.agent_loop import _encode_current_clusters
    from maxim.simulation.minecraft_harness import build_minecraft_aut

    report: dict = {
        "ts": time.time(),
        "bars": {
            "separability": SEPARABILITY_BAR,
            "day_light_min": DAY_LIGHT_MIN,
            "dark_light_max": DARK_LIGHT_MAX,
        },
        "provenance": provenance,
        "world_sensor_count_note": "17 modality:world sensors (L11 dilution context)",
        "instrument_error": None,
    }

    def _finish(code: int) -> int:
        report["all_pass"] = bool(
            report.get("check1_light", {}).get("pass") and report.get("check2_world_separability", {}).get("pass")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        printable = {k: v for k, v in report.items() if k.startswith("check") or k == "instrument_error"}
        print(json.dumps(printable, indent=2))
        print(f"survival phase0: {'PASS' if report['all_pass'] else 'FAIL'} -> {out_path}")
        return code

    persistence_dir = tempfile.mkdtemp(prefix="survival_phase0_")
    aut = build_minecraft_aut(
        agent_id="survival_phase0",
        bridge_port=args.bridge_port,
        bridge_host=args.bridge_host,
        persistence_dir=persistence_dir,
        entity_ref="bodies/minecraft_player",
    )
    encoder = make_fresh_encoder(aut)
    rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)

    box: tuple[int, int, int] | None = None
    rest: dict[str, float] | None = None
    mob_spawning_disabled = False
    try:
        # Startup gate: the bridge must actually deliver state before anything is measured —
        # connect() confirms nothing by default, and a one-client-bridge rejection would
        # otherwise read as body-initial sensor values for the whole run.
        first = settle_until(aut, lambda vm: LIGHT_KEY in vm, timeout_s=8.0)
        if first is None:
            raise InstrumentError(
                "bridge never delivered a snapshot carrying the light sensor "
                f"({LIGHT_KEY!r}) — bridge dead/busy, or body regression; "
                "check the bridge terminal for 'bridge busy: one client at a time'."
            )

        # Frozen conditions the world was prepared with; re-assert (idempotent, cheap).
        rcon.command("time set day")
        rcon.command("gamerule doDaylightCycle false")
        # No NEW spawns while the instrument runs (the dark box is a spawnable space when the
        # bot is > 24 blocks away). Restored in the finally; existing cave mobs remain —
        # honest world-channel noise, not controlled away.
        rcon.command("gamerule doMobSpawning false")
        mob_spawning_disabled = True

        rx, ry, rz = bot_pos(rcon, args.username)
        rest = {"x": rx, "y": ry, "z": rz}
        # Roofed box well above the LOCAL surface (derived from the bot's y, not a hardcoded
        # altitude — 1.20 terrain can reach y~200, and carving terrain then "restoring" with
        # air would silently mutate the shared world). Interior floor top = y0 + 1.
        by0 = min(int(ry) + BOX_ALTITUDE, 250)
        bx, bz = int(rx) + 48, int(rz)
        build_dark_box(rcon, bx, by0, bz)
        box = (bx, by0, bz)
        dark_tp = {"x": float(bx), "y": float(by0 + 1), "z": float(bz)}
        report["apparatus"] = {"rest": rest, "box": {"x": bx, "y0": by0, "z": bz}}

        light_rest: list[float] = []
        light_dark: list[float] = []
        ids: list[tuple[bool, str | None]] = []  # (in_dark, world_cluster_id)
        settle_timeouts = 0
        for _cycle in range(args.cycles):
            for in_dark in (False, True, True):  # rest, onset, repeat — exp56 check-1 shape
                target_y = dark_tp["y"] if in_dark else rest["y"]
                rcon.teleport(args.username, dark_tp if in_dark else rest)
                # Settle on POSITION truth (y separates the conditions by ~80 blocks and is
                # independent of the light metric under test) — the settle-until-reflected
                # pattern; a blind sleep can sample the previous condition (exp56's measured
                # stale-snapshot incident).
                vm = settle_until(
                    aut,
                    lambda vm, ty=target_y: abs(float(vm.get("y_altitude", 1e9)) - ty) <= Y_TOLERANCE,
                    timeout_s=SETTLE_TIMEOUT_S,
                )
                if vm is None:
                    settle_timeouts += 1
                    continue  # no sample: never mislabel a condition
                try:
                    lv = float(vm[LIGHT_KEY])
                except (KeyError, TypeError, ValueError):
                    lv = None
                if lv is not None:
                    (light_dark if in_dark else light_rest).append(lv)
                clusters = _encode_current_clusters(encoder, "survival_phase0", aut.executor)
                ids.append((in_dark, clusters.get("world")))
        if settle_timeouts > args.cycles:  # more than a third of samples never settled
            raise InstrumentError(
                f"{settle_timeouts}/{args.cycles * 3} teleports never reflected in bridge state "
                f"within {SETTLE_TIMEOUT_S}s — server lag or bridge stall; nothing was mislabeled "
                "(unsettled samples are dropped), but the run is too sparse to gate on."
            )
        if not light_rest or not light_dark:
            raise InstrumentError(
                f"no light samples collected (rest={len(light_rest)}, dark={len(light_dark)}) "
                f"despite the startup gate seeing {LIGHT_KEY!r} — bridge died mid-run?"
            )

        report["check1_light"] = {
            "day_min": min(light_rest),
            "day_median": sorted(light_rest)[len(light_rest) // 2],
            "dark_max": max(light_dark),
            "dark_median": sorted(light_dark)[len(light_dark) // 2],
            "n_rest": len(light_rest),
            "n_dark": len(light_dark),
            "settle_timeouts": settle_timeouts,
            # Min/max gated: every cycle must read both sides (docstring rationale).
            "pass": min(light_rest) >= DAY_LIGHT_MIN and max(light_dark) <= DARK_LIGHT_MAX,
        }

        transitions = separated = repeats = stable = 0
        for i in range(1, len(ids)):
            prev_dark, prev_id = ids[i - 1]
            in_dark, cid = ids[i]
            if prev_id is None or cid is None:
                continue
            if not prev_dark and in_dark:
                transitions += 1
                separated += int(cid != prev_id)
            elif prev_dark and in_dark:
                repeats += 1
                stable += int(cid == prev_id)
        if transitions == 0 or repeats == 0:
            raise InstrumentError(
                "zero usable transitions/repeats — the encoder returned no world cluster at "
                "all (a dead instrument, NOT a measured separability failure; do not read "
                "this as the L11 outcome)."
            )
        min_n = max(1, (3 * args.cycles) // 4)
        separation = separated / transitions
        stability = stable / repeats
        report["check2_world_separability"] = {
            "transitions": transitions,
            "separation": round(separation, 4),
            "repeats": repeats,
            "stability": round(stability, 4),
            "none_encodes": sum(1 for _, cid in ids if cid is None),
            "min_n": min_n,
            # Ratio bars AND a minimum-N gate: 3/3 on a mostly-dead encoder must not pass.
            "pass": (
                separation >= SEPARABILITY_BAR
                and stability >= SEPARABILITY_BAR
                and transitions >= min_n
                and repeats >= min_n
            ),
        }
    except InstrumentError as exc:
        report["instrument_error"] = str(exc)
        print(f"INSTRUMENT ERROR: {exc}")
        return _finish(4)
    except Exception as exc:  # record the failure as evidence, then surface it fully
        report["instrument_error"] = f"unexpected: {exc!r}"
        traceback.print_exc()
        return _finish(4)
    finally:
        # Cleanup commands individually guarded: if RCON is the thing that broke, an
        # unguarded cleanup raise would MASK the original exception and abandon the rest
        # of the restore (bot sealed in the box, spawning left off, box left in the world).
        if mob_spawning_disabled:
            try:
                rcon.command("gamerule doMobSpawning true")
            except Exception as exc:
                print(f"WARNING: could not restore doMobSpawning: {exc!r}")
        if rest is not None:
            try:
                rcon.teleport(args.username, rest)
            except Exception as exc:
                print(f"WARNING: could not teleport bot back to rest: {exc!r}")
        if box is not None:
            try:
                remove_dark_box(rcon, *box)
            except Exception as exc:
                print(f"WARNING: could not remove the dark box at {box}: {exc!r}")
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

    code = _finish(0 if (report["check1_light"]["pass"] and report["check2_world_separability"]["pass"]) else 4)
    if code != 0:
        if not report["check1_light"]["pass"]:
            print(
                "  check1: light did not read both sides on every cycle — inspect the bridge's\n"
                "  perceivedLight (docs/wiring/world-light-sensing.md has the manual check)."
            )
        if not report["check2_world_separability"]["pass"]:
            print(
                "  check2: the world channel did not separate rest vs dark through the real\n"
                "  encoder — the L11 17-sensor dilution outcome, now on a MULTI-sensor swing.\n"
                "  NOTE the A4 nonlinear gain is ALREADY the shipped default for the world\n"
                "  modality, and the 1.1.4 re-measure scored it 0.0566 << 0.70 on a single-\n"
                "  sensor swing at this N (docs/limits/l11_sensor_dilution.md) — so this is a\n"
                "  real ceiling, not a missing toggle. Open paths: gain-exponent/threshold\n"
                "  work, or channel grouping; each is a selection-dynamics change needing its\n"
                "  own re-baseline. The Phase-1 prereg is BLOCKED until this passes."
            )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
