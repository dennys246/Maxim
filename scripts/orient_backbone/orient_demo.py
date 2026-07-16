#!/usr/bin/env python
"""Orient demo — run the LEARNED policy live: the robot turns toward your voice.

The show-it-off script (and the embryo of the 1.1 `--embodiment` hardware
runtime): loads a trained orient NAc (your own from live_3_learn.py, a merged
one from orient_merge_arm.py, or one extracted from a queen-mind bundle),
prints what it knows (the frozen-policy probe), then tracks sound until
Ctrl+C — greedy substrate-primary selection, no exploration, no apparatus,
no LLM. Walk around the room and talk; the robot follows.

    ~/Envs/maxim-env/bin/python scripts/orient_backbone/orient_demo.py \\
        --host <robot-ip> [--learn]

--learn keeps the potential_diff credit flowing (the demo keeps getting
better and saves on exit); without it the policy is frozen — pure playback.
Offline logic check: --dry-run.

This is deliberately NOT the experiment harness: no probes-during-run, no
epsilon, no trial caps. For the pre-registered measurement protocol, use
live_3_learn.py (Exp 45).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from live_3_learn import AGENT_ID, ARGMAX, az_bin, load_orient_actions, probe_policy
from live_common import DryRig, JsonlLog, LiveRig, gated_azimuth, preflight, resolve_host

from maxim.decisions.nac import NAc, NACConfig

DEFAULT_NAC_PATHS = (
    "~/.maxim/orient_live/nac_merged.json",  # the merge-arm showcase artifact
    "~/.maxim/orient_live/nac_reachy.json",  # the primary training NAc
)


def resolve_nac_path(explicit: str | None) -> str | None:
    if explicit:
        p = os.path.expanduser(explicit)
        return p if os.path.exists(p) else None
    for cand in DEFAULT_NAC_PATHS:
        p = os.path.expanduser(cand)
        if os.path.exists(p):
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="Reachy daemon IP (or $MAXIM_REACHY_HOST)")
    ap.add_argument("--nac-path", default=None, help=f"trained NAc json (default: first of {DEFAULT_NAC_PATHS})")
    ap.add_argument("--learn", action="store_true", help="keep learning while demoing (saves on exit)")
    ap.add_argument("--flip-sign", action="store_true", help="Step-2-calibrated sign flag")
    ap.add_argument(
        "--step-scale", type=float, default=1.0, help="rescale the YAML orient magnitudes (1.0 = as declared)"
    )
    ap.add_argument("--max-yaw", type=float, default=1.4, help="clamp cumulative body yaw (rad)")
    ap.add_argument("--duration", type=float, default=0.6)
    ap.add_argument("--settle", type=float, default=0.4)
    ap.add_argument("--log", default="/tmp/orient_demo.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="offline logic check (no robot)")
    args = ap.parse_args()

    sign_mult = -1.0 if args.flip_sign else 1.0
    poll_s = 0.0 if args.dry_run else 0.15

    def pace(seconds: float) -> None:
        if not args.dry_run:
            time.sleep(seconds)

    action_deltas, band = load_orient_actions()
    names = list(action_deltas)

    nac_path = resolve_nac_path(args.nac_path)
    if nac_path is None:
        print("[FAIL] no trained NAc found. Train one first (live_3_learn.py --perturb --fresh),")
        print("       merge one (orient_merge_arm.py --save-merged), or extract a queen-mind")
        print("       bundle (maxim substrate import) and pass it via --nac-path.")
        return 2
    nac = NAc(NACConfig(persistence_path=nac_path))
    ok, err = nac.load_safe(nac_path)
    print(f"[nac] loaded {nac_path} (ok={ok}{', ' + err if err else ''})")

    # Show what the substrate knows BEFORE any motion — the demo's honest opener.
    p = probe_policy(nac, names, action_deltas)
    print(
        f"[policy] probe correctness={p['correctness']:.2f} "
        f"magnitude={p['magnitude_appropriateness']:.2f}  " + str({b: v["argmax"] for b, v in p["bins"].items()})
    )
    if p["correctness"] < 1.0:
        print("[warn] policy is not fully converged — bins probing None will HOLD instead of turning.")
        print("       (train more via live_3_learn.py, or run this demo with --learn)")

    log = JsonlLog(args.log)
    if args.dry_run:
        rig = DryRig(theta_src=-0.7)
        print(f"[dry] source starts at world bearing {rig.theta_src:+.2f} rad (it wanders)")
    else:
        host, source = resolve_host(args.host)
        if host is None:
            print("[FAIL] no robot address: --host <ip> or export MAXIM_REACHY_HOST=<ip>")
            log.close()
            return 2
        print(f"[host] using {host} (source: {source})")
        preflight(host)
        rig = LiveRig(host)
        rig.recenter()

    log.write("demo_start", nac_path=nac_path, learn=args.learn, flip_sign=args.flip_sign)
    print("\n[demo] tracking — talk from anywhere in the FRONT hemisphere; Ctrl+C to stop.\n")

    ticks = 0
    max_dry_ticks = 200  # dry-run: bounded (live runs until Ctrl+C)
    try:
        while True:
            ticks += 1
            if args.dry_run and ticks > max_dry_ticks:
                break
            az = gated_azimuth(rig.reader, k=3, timeout_s=2.0, poll_s=poll_s)
            if az is None:
                pace(0.3)  # quiet room — just wait for the gate
                continue
            state = az_bin(az, band)
            if state == "center":
                pace(0.3)
                continue

            rec = nac.recommend_action(
                agent_id=AGENT_ID,
                available_tools=names,
                current_drives=None,
                current_cluster_id=state,
                min_confidence=ARGMAX,
            )
            if rec is None:
                # Nothing learned for this bin: a demo HOLDS rather than guessing.
                log.write("hold_unlearned", state=state, az=round(az, 3))
                pace(0.5)
                continue
            action = rec["tool_name"]
            delta = action_deltas[action] * args.step_scale * sign_mult
            target_yaw = max(-args.max_yaw, min(args.max_yaw, rig.body_yaw + delta))
            if target_yaw == rig.body_yaw:
                print(f"      at yaw limit ({rig.body_yaw:+.2f}) — recentering")
                log.write("yaw_limit", body_yaw=round(rig.body_yaw, 3))
                rig.recenter()
                continue
            rig.goto_body_yaw(target_yaw, duration=args.duration)
            pace(args.duration + args.settle)

            record = {"state": state, "action": action, "az": round(az, 3), "body_yaw": round(rig.body_yaw, 3)}
            if args.learn:
                az_after = gated_azimuth(rig.reader, k=3, timeout_s=2.0, poll_s=poll_s)
                if az_after is not None:
                    relief = abs(az) - abs(az_after)
                    nac.update_cluster_reward(AGENT_ID, state, f"tool:{action}", relief)
                    record["relief"] = round(relief, 4)
            print(
                f"      az {az:+.2f} ({state}) -> {action}" + (f"  relief={record.get('relief')}" if args.learn else "")
            )
            log.write("track", **record)
    except KeyboardInterrupt:
        print("\n[demo] stopping...")
    finally:
        if not args.dry_run:
            rig.recenter()
        if args.learn:
            nac.save(nac_path)
            print(f"[nac] saved continued learning -> {nac_path}")
        log.write("demo_end", ticks=ticks)
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
