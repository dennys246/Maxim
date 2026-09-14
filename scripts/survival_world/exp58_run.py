#!/usr/bin/env python3
"""Exp 58 LIVE trial harness — Claim B: learned dark-fear (FEAR vs ABLATED arms).

Runs the frozen prereg's Claim-B design (docs/experiments/exp58_survival_wants_prereg.md,
WITH its dated 2026-09-14 harness addendum) against the live 1.20.4 classroom. Per
seed × arm:

  1. PREFLIGHTS (all before any measurement; refusal = exit 3/4, no data):
     - offline gates record exists with all_pass true (prereg instrument-gate stop rule);
     - frozen-apparatus fingerprint asserted + stamped (explore weight, encoder pattern
       threshold, θ/α/cap/allowlist, subscriber intensity floor, health drive spec) and
       the usable-episode bound checked against the body's comfort band;
     - classroom geometry verified live (light 15 at the safe anchor, 0 in the dark room);
     - FLEE ACTUATION CHECK (verify-actuation-before-theorizing): with the door OPEN, one
       real executor flee from inside the dark room must cross the door plane — the
       review's DNR-1 measured that the previous design could not (closed door is
       impassable to the pathfinder; bot.spawnPoint is the WORLD spawn — the bridge now
       takes --flee_x/--flee_z and the harness opens the door for probes).
  2. PRE-PROBE: kill-sweep + full-heal, DOOR OPEN (probes are mob-free so containment is
     moot), P shepherded placements into the dark room with the FULL loop live; DV =
     LATENCY-TO-EXIT (RCON ground truth, censored at the window cap). Executed-action
     counts per window recorded (the lit-activity gate's DV).
  3. TRAINING: DOOR CLOSED (containment), K confined damage episodes via PROPOSE-ONLY
     ticks (the conditioning chamber — no execution: a failed flee would book
     arm-asymmetric negative credit on the read path itself; record_outcome verified
     unreachable on this path). Usable episode = pain PUBLISHED with the dark
     world-cluster active AND health ≤ 13; the NOTED WORLD CLUSTER IS CAPTURED PER
     EPISODE — the live G2: training-majority cluster must equal the probe-identified
     dark cluster or the seed is REFUSED (an unreadable fear write must never ship as a
     behavioural null).
  4. POST-PROBE: identical to (2). Flee negative-link count recorded (probe
     self-poisoning audit).
  5. Per-seed record appended (run_id-stamped) through the gated evidence path. Under-K,
     failed live-G2, or death-cap seeds are stamped REFUSED and their behavioural DVs
     are not written.

Run ON the bridge box, server + bridge up (bridge started with --flee_x/--flee_z from
the classroom build output), classroom built:

    python scripts/survival_world/exp58_run.py --arm fear --rcon-password '<pw>' \\
        --username maxim --write-experiment-results
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
import uuid
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
from survival_world.common import InstrumentError, bot_pos, read_vital, settle_until  # noqa: E402

FROZEN = {
    "K_usable_episodes": 10,
    "placements_per_probe": 6,
    "placement_window_s": 45.0,
    "episode_timeout_s": 120.0,
    "usable_health_max": 13.0,  # must sit BELOW set_point - comfort_band (asserted live)
    "seeds": (11, 12, 13, 14, 15),
    "loop_hz": 4.0,
    "death_cap": 2,
    # Frozen-apparatus fingerprint (asserted against live config at start):
    "fingerprint": {
        "cluster_fear_alpha": 0.5,
        "max_cluster_fear": 1.0,
        "cluster_fear_threshold": 0.5,
        "cluster_fear_failure_modes": ["drive:health"],
        "encoder_pattern_threshold": 0.85,
        "substrate_explore_bonus_weight": 0.0,
    },
}
GATES_RECORD = "docs/experiments/data/exp58_offline_gates.json"


class Refusal(RuntimeError):
    """A prereg stop rule fired — the seed/run must not produce a verdict row."""


def _detach_fear_subscriber(aut) -> int:
    bus = aut.bio.pain_bus
    targets = [cb for cb in list(bus._pain_signal_subs) if "cluster_fear" in getattr(cb, "__qualname__", "")]
    for cb in targets:
        bus.unsubscribe(cb)
    return len(targets)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("fear", "ablated"), required=True)
    ap.add_argument("--seeds", type=int, default=len(FROZEN["seeds"]))
    ap.add_argument("--out", default="docs/experiments/data/exp58_claim_b.jsonl")
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25567)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", required=True)
    ap.add_argument("--username", default="maxim")
    ap.add_argument("--door-x", type=float, default=None)
    ap.add_argument("--dark-x", type=float, default=None)
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    out_arg = Path(args.out)
    out_abs = out_arg if out_arg.is_absolute() else (C.REPO_ROOT / out_arg)
    out_path = evidence_out_paths_or_exit(
        C.REPO_ROOT,
        [str(out_abs)],
        write_experiment_results=args.write_experiment_results,
        allow_dirty=args.allow_dirty,
    )[0]

    import maxim

    try:
        provenance = in_process_code_provenance(
            C.REPO_ROOT, maxim.__file__, out_path=out_path, allow_dirty=args.allow_dirty
        )
    except (DirtyTreeError, ProvenanceError) as exc:
        print(f"[FAIL] provenance: {exc}")
        return 3

    # ── Preflight: gates record (prereg instrument-gate stop rule) ──
    gates_path = C.REPO_ROOT / GATES_RECORD
    try:
        gates = json.loads(gates_path.read_text())
    except OSError:
        print(f"[FAIL] offline gates record missing: {gates_path} — gates must pass before live trials")
        return 3
    if not gates.get("all_pass"):
        print(f"[FAIL] offline gates record does not carry all_pass=true: {gates_path}")
        return 3

    from maxim.runtime.agent_loop import _encode_current_clusters, propose_via_substrate
    from maxim.similarity.encoder import SensorEncoderConfig
    from maxim.simulation.minecraft_harness import MinecraftSyncPump, build_minecraft_aut, run_minecraft_aut
    from survival_world.common import make_fresh_encoder

    run_id = uuid.uuid4().hex[:12]
    rcon = C.RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    # Geometry from the classroom's RECORDED anchor, never re-derived from the
    # bot's live position (a half-block teleport drift put door_x off by one in
    # the first dry-run — geometry must be the built truth, not wherever the
    # bot happens to stand). --door-x/--dark-x still override for ad-hoc use.
    anchor_file = Path.home() / ".maxim" / "exp58_classroom.json"
    try:
        geom = json.loads(anchor_file.read_text())
    except OSError:
        print(f"[FAIL] classroom geometry not found: {anchor_file} — run `setup_world.py classroom` first")
        return 3
    ax, ay, az = (float(v) for v in geom["anchor"])
    door_x = args.door_x if args.door_x is not None else float(geom["door_x"])
    dark_x = args.dark_x if args.dark_x is not None else float(geom["dark_x"])
    safe = {"x": ax, "y": ay, "z": az}
    dark = {"x": dark_x, "y": ay, "z": az}
    print(f"run {run_id}: arm={args.arm} anchor=({ax:.0f},{ay:.0f},{az:.0f}) door_x={door_x:.0f} dark_x={dark_x:.0f}")

    def _set_door(open_: bool) -> None:
        state = "true" if open_ else "false"
        dx, dz = int(door_x), int(az)
        rcon.command(f"setblock {dx} {int(ay)} {dz} minecraft:oak_door[facing=east,half=lower,open={state}]")
        rcon.command(f"setblock {dx} {int(ay) + 1} {dz} minecraft:oak_door[facing=east,half=upper,open={state}]")

    def _sweep(radius: int = 64) -> None:
        # `execute at <bot>` so distance measures from the CLASSROOM, not the
        # console/world-spawn origin (executor finding 5: a classroom >64 from
        # world spawn made the old sweep a silent no-op).
        rcon.command(f"execute at {args.username} run kill @e[type=minecraft:zombie,distance=..{radius}]")

    def _heal() -> None:
        rcon.command(f"effect give {args.username} minecraft:instant_health 1 10 true")
        rcon.command(f"effect give {args.username} minecraft:saturation 1 10 true")

    def _deaths() -> int:
        resp = rcon.command(f"scoreboard players get {args.username} exp58_deaths")
        try:
            return int(resp.split(" has ")[1].split()[0])
        except (IndexError, ValueError):
            return 0

    def _in_dark() -> bool:
        x, _, _ = bot_pos(rcon, args.username)
        return x > door_x

    records: list[dict] = []
    exit_code = 0
    try:
        for seed in FROZEN["seeds"][: args.seeds]:
            print(f"\n=== arm={args.arm} seed={seed} ===")
            persistence_dir = tempfile.mkdtemp(prefix=f"exp58_{args.arm}_{seed}_")
            agent_id = f"exp58_{args.arm}_s{seed}"
            from maxim.simulation.minecraft import MinecraftClient

            # Per-seed reconnect races the one-client bridge slot (freed by an
            # ASYNC close event) — confirm+retry per the exp56 pattern that
            # fixed the 88%-in campaign crash (executor finding 9).
            client = MinecraftClient(args.bridge_host, args.bridge_port)
            client.connect(confirm_timeout_s=4.0, retries=8, backoff_s=0.5)
            aut = build_minecraft_aut(
                agent_id=agent_id,
                bridge_port=args.bridge_port,
                bridge_host=args.bridge_host,
                persistence_dir=persistence_dir,
                entity_ref="bodies/minecraft_player",
                client=client,
            )
            encoder = make_fresh_encoder(aut)
            pump = MinecraftSyncPump(aut, interval_s=0.25)
            pump.start()
            # run_agentic_loop writes CWD-relative state files — point them at
            # the seed's throwaway dir, not the repo (executor NIT).
            prev_cwd = os.getcwd()
            os.chdir(persistence_dir)
            record: dict = {
                "ts": time.time(),
                "run_id": run_id,
                "arm": args.arm,
                "seed": seed,
                "frozen": FROZEN,
                "provenance": provenance,
                "gates_record_ts": gates.get("ts"),
                "refusal": None,
            }
            try:
                # ── Fingerprint assertions (frozen-apparatus stop rule) ──
                cfg = aut.bio.nac.config
                fp = FROZEN["fingerprint"]
                live_fp = {
                    "cluster_fear_alpha": cfg.cluster_fear_alpha,
                    "max_cluster_fear": cfg.max_cluster_fear,
                    "cluster_fear_threshold": cfg.cluster_fear_threshold,
                    "cluster_fear_failure_modes": sorted(cfg.cluster_fear_failure_modes),
                    "encoder_pattern_threshold": float(SensorEncoderConfig().pattern_threshold),
                    "substrate_explore_bonus_weight": float(getattr(cfg, "substrate_explore_bonus_weight", 0.0)),
                }
                record["fingerprint_live"] = live_fp
                if live_fp != fp:
                    raise Refusal(f"config fingerprint drift: {live_fp} != {fp}")
                # Damage-arithmetic vs the body's declared band (band-edge trap):
                drive = aut.executor.embodiment.root.drive_specs.get("health")
                if drive is None or FROZEN["usable_health_max"] >= drive.set_point - drive.comfort_band:
                    raise Refusal("usable_health_max does not sit below the health comfort band")

                if args.arm == "ablated":
                    detached = _detach_fear_subscriber(aut)
                    record["detached_subscribers"] = detached
                    if detached != 1:
                        raise Refusal(f"expected exactly 1 fear subscriber, detached {detached}")

                if settle_until(aut, lambda vm: vm.get("light_level") is not None, timeout_s=10.0) is None:
                    raise Refusal("bridge never delivered state")

                # ── Geometry + actuation preflights ──
                _set_door(False)  # darkness is a DOOR-CLOSED property (open leaks platform light)
                _sweep()
                rcon.teleport(args.username, safe)
                if settle_until(aut, lambda vm: vm.get("light_level") == 15, timeout_s=10.0) is None:
                    raise Refusal("safe anchor does not read light 15 — classroom geometry / relight?")
                rcon.teleport(args.username, dark)
                if settle_until(aut, lambda vm: vm.get("light_level") == 0, timeout_s=10.0) is None:
                    raise Refusal("dark room does not read light 0 (door closed) — classroom geometry / relight?")
                # Flee-actuation check with the door CLOSED — flee itself must
                # open it (canOpenDoors) AND cross the plane; if it can't, the
                # seed refuses before any measurement.
                flee_tool = next(t for t in aut.executor.registry.list() if t.endswith("_flee"))
                out = aut.executor.execute({"tool_name": flee_tool, "params": {}})
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline and _in_dark():
                    time.sleep(0.5)
                if _in_dark():
                    raise Refusal(
                        f"flee actuation check FAILED (success={getattr(out, 'success', None)}, "
                        f"error={getattr(out, 'error', None)!r}) — bridge --flee_x/--flee_z set? door open?"
                    )
                print("  preflight: flee actuation OK (crossed the door plane)")

                def _stop_motion() -> None:
                    # Clear any residual pathfinder goal (bridge `stop`): a goto
                    # promise survives Python timeouts AND /tp; a stale goal
                    # self-moves the bot through later phases (finding 1).
                    try:
                        aut.client.call_action("stop", {})
                    except Exception as exc:
                        print(f"WARNING: stop action raised: {exc!r}")

                _stop_motion()
                # Pre-training dark-cluster identity (the live-G2 triple: this,
                # the training majority, and the post-training probe cluster).
                _set_door(False)  # flee left it open; the dark read needs it closed (no leak)
                rcon.teleport(args.username, dark)
                if settle_until(aut, lambda vm: vm.get("light_level") == 0, timeout_s=10.0) is None:
                    raise Refusal("pre-training dark settle failed")
                pre_dark_cluster = _encode_current_clusters(encoder, agent_id, aut.executor).get("world")
                rcon.teleport(args.username, safe)
                deaths0 = _deaths()

                def _probe(label: str) -> dict:
                    # Door stays CLOSED: the flee affordance opens it to exit
                    # (bridge Movements canOpenDoors=true), so the dark chamber
                    # stays sealed/dark until the bot chooses to leave — no
                    # block-light leak from the lit safe chamber during the window.
                    _set_door(False)
                    _sweep()
                    _heal()
                    if settle_until(aut, lambda vm: (vm.get("food") or 0) >= 16, timeout_s=10.0) is None:
                        raise Refusal(f"{label}-probe: satiation (food >= 16) never settled (S7)")
                    stop = threading.Event()
                    actions0 = len(getattr(aut.executor, "_tools_succeeded", []) or [])
                    loop = threading.Thread(
                        target=run_minecraft_aut,
                        args=(aut,),
                        kwargs={"max_steps": 100_000, "target_hz": FROZEN["loop_hz"], "stop_event": stop},
                        daemon=True,
                    )
                    loop.start()
                    latencies = []
                    censored = 0
                    clean_placements = 0
                    try:
                        for i in range(FROZEN["placements_per_probe"]):
                            _heal()
                            _stop_motion()  # no residual goal carries into the placement
                            rcon.teleport(args.username, dark)
                            t0 = time.monotonic()
                            lat = FROZEN["placement_window_s"]
                            was_censored = True
                            while time.monotonic() - t0 < FROZEN["placement_window_s"]:
                                # Per-poll sweep: the spawner fires whenever the bot is in
                                # range (5-15 s delay) — one sweep at probe start is NOT
                                # mob-free (executor finding 2). Radius 16 covers the room.
                                _sweep(radius=16)
                                if not _in_dark():
                                    lat = time.monotonic() - t0
                                    was_censored = False
                                    break
                                time.sleep(0.5)
                            hp_end = read_vital(aut, "health")
                            placement_clean = hp_end is not None and hp_end >= 18.0
                            if placement_clean:
                                clean_placements += 1
                            latencies.append(round(lat, 2))
                            censored += int(was_censored)
                            print(
                                f"  {label} placement {i}: latency {lat:.1f}s"
                                f"{' (censored)' if was_censored else ''}"
                                f"{'' if placement_clean else ' [DIRTY: hp<18]'}"
                            )
                            rcon.teleport(args.username, safe)
                            time.sleep(2.0)
                    finally:
                        stop.set()
                        loop.join(timeout=15.0)
                        if loop.is_alive():
                            raise Refusal(f"{label}-probe loop thread did not stop (finding 4)")
                        _stop_motion()
                    actions = len(getattr(aut.executor, "_tools_succeeded", []) or []) - actions0
                    return {
                        "latencies_s": latencies,
                        "censored": censored,
                        "clean_placements": clean_placements,
                        "actions_executed": actions,
                        "cap_s": FROZEN["placement_window_s"],
                    }

                pre = _probe("pre")

                # ── Training (door closed; propose-only conditioning) ──
                _set_door(False)
                _stop_motion()  # a residual flee goal would walk the bot out (finding 1)
                usable = 0
                attempts = 0
                episode_clusters: list[str] = []
                pubs_start = aut.bio.pain_bus.get_stats().get("total_published", 0)
                deadline = time.monotonic() + FROZEN["K_usable_episodes"] * FROZEN["episode_timeout_s"] * 1.5
                while usable < FROZEN["K_usable_episodes"] and time.monotonic() < deadline:
                    attempts += 1
                    _heal()
                    rcon.teleport(args.username, dark)
                    ep_end = time.monotonic() + FROZEN["episode_timeout_s"]
                    while time.monotonic() < ep_end:
                        pubs_before = aut.bio.pain_bus.get_stats().get("total_published", 0)
                        propose_via_substrate(
                            nac=aut.bio.nac, agent_id=agent_id, executor=aut.executor, sensor_encoder=encoder
                        )
                        pubs_after = aut.bio.pain_bus.get_stats().get("total_published", 0)
                        hp = read_vital(aut, "health")
                        noted = aut.bio.nac.active_clusters(agent_id).get("world")
                        if (
                            pubs_after > pubs_before
                            and noted
                            and _in_dark()
                            and hp is not None
                            and hp <= FROZEN["usable_health_max"]
                        ):
                            episode_clusters.append(noted)
                            usable += 1
                            print(f"  training: usable episode {usable}/{FROZEN['K_usable_episodes']}")
                            break
                        time.sleep(1.0 / FROZEN["loop_hz"])
                    rcon.teleport(args.username, safe)
                    _heal()
                    # Latch clearing requires an evaluation OBSERVING recovery — settle the
                    # SENSED health first (bridge 500ms + pump 250ms lag; finding 12).
                    if settle_until(aut, lambda vm: (vm.get("health") or 0) >= 18.0, timeout_s=10.0) is None:
                        raise Refusal("healed health never settled — latch cannot clear")
                    for _ in range(4):  # healthy ticks: latch observes recovery
                        propose_via_substrate(
                            nac=aut.bio.nac, agent_id=agent_id, executor=aut.executor, sensor_encoder=encoder
                        )
                        time.sleep(0.25)
                    if _deaths() - deaths0 > FROZEN["death_cap"]:
                        raise Refusal(f"death cap exceeded ({_deaths() - deaths0})")
                record["training"] = {
                    "usable_episodes": usable,
                    "attempts": attempts,
                    "pain_published": aut.bio.pain_bus.get_stats().get("total_published", 0) - pubs_start,
                    "episode_clusters": episode_clusters,
                    "deaths": _deaths() - deaths0,
                }
                if usable < FROZEN["K_usable_episodes"]:
                    raise Refusal(f"only {usable}/{FROZEN['K_usable_episodes']} usable episodes")

                # ── Cluster identification + LIVE G2 gate ──
                _set_door(False)  # dark cluster is identified at light 0 (door closed, no leak)
                _sweep()
                rcon.teleport(args.username, dark)
                if settle_until(aut, lambda vm: vm.get("light_level") == 0, timeout_s=10.0) is None:
                    raise Refusal("post-training dark settle failed")
                dark_cluster = _encode_current_clusters(encoder, agent_id, aut.executor).get("world")
                rcon.teleport(args.username, safe)
                if settle_until(aut, lambda vm: vm.get("light_level") == 15, timeout_s=10.0) is None:
                    raise Refusal("post-training lit settle failed")
                lit_cluster = _encode_current_clusters(encoder, agent_id, aut.executor).get("world")
                majority = max(set(episode_clusters), key=episode_clusters.count) if episode_clusters else None
                record["live_g2"] = {
                    "pre_training_dark_cluster": pre_dark_cluster,
                    "training_majority_cluster": majority,
                    "probe_dark_cluster": dark_cluster,
                    "pass": bool(majority) and majority == dark_cluster,
                }
                # Diagnosability: the agent's full fear map (wrong-cluster writes visible).
                record["cluster_fear_dump"] = {
                    f"{cid}|{fm}": v
                    for (aid, cid, fm), v in getattr(aut.bio.nac, "_cluster_fear", {}).items()
                    if aid == agent_id
                }
                if not record["live_g2"]["pass"]:
                    raise Refusal(
                        "LIVE G2 FAILED: fear trained on a different world cluster than the probe "
                        "activates — an unreadable write must not ship as a behavioural null"
                    )
                record["dark_fear"] = round(aut.bio.nac.cluster_fear(agent_id, dark_cluster), 4)
                record["lit_fear"] = round(aut.bio.nac.cluster_fear(agent_id, lit_cluster), 4) if lit_cluster else 0.0

                post = _probe("post")
                sig = f"tool:{flee_tool}"
                record["flee_negative_links"] = len(aut.bio.nac.get_negative_outcomes(sig))
                record["pre"] = pre
                record["post"] = post
                print(
                    f"seed {seed}: pre={pre['latencies_s']} post={post['latencies_s']} "
                    f"fear={record['dark_fear']} flee_neg_links={record['flee_negative_links']}"
                )
            except (Refusal, InstrumentError) as exc:
                record["refusal"] = str(exc)
                print(f"REFUSED seed {seed}: {exc}")
                exit_code = 4
            finally:
                try:
                    pump.stop()
                except Exception as exc:
                    print(f"WARNING: pump stop raised: {exc!r}")
                try:
                    aut.bio.on_session_end()
                except Exception as exc:
                    print(f"WARNING: bio teardown raised: {exc!r}")
                try:
                    aut.client.close()
                except (OSError, ConnectionError) as exc:
                    print(f"WARNING: client close raised: {exc!r}")
                os.chdir(prev_cwd)
                shutil.rmtree(persistence_dir, ignore_errors=True)
                _sweep()
            records.append(record)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
    finally:
        try:
            _set_door(False)
        except Exception as exc:
            print(f"WARNING: could not close the door: {exc!r}")
        rcon.close()

    ok = sum(1 for r in records if r.get("refusal") is None)
    print(f"\narm={args.arm} run={run_id}: {ok}/{len(records)} seeds clean -> {out_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
