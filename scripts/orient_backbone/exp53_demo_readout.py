#!/usr/bin/env python
"""Exp 53 DEMO — a nursery-taught infant glances toward whoever speaks (for the camera).

NOT EVIDENCE. Same loaded files, same production path as the Exp 53 harness
(``exp53_cross_context_readout.py``: LoadedAgent → _encode_current_clusters →
propose_via_substrate → ReachyOrientMotorBackend; nothing credits the NAc), but no
apparatus: the base is not rotated to place the source — a person simply speaks from a
side. One decision per speech onset, then a refractory pause, because the learned map is
three azimuth bins and a free-running loop would oscillate for a speaker on the right
(measured: right → one correct step to +0.2–0.3, which sits in the centre bin whose bias
is ``turn_left``). The taught agent glances toward the speaker; a control agent loaded the
same way sits still.

    export PYTHONPATH="$PWD/src"
    python scripts/orient_backbone/exp53_demo_readout.py --host 10.6.0.63 --agent taught_seed43
    python scripts/orient_backbone/exp53_demo_readout.py --host 10.6.0.63 --agent satiated_seed43   # the control shot
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import exp53_cross_context_readout as h  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default=None)
    ap.add_argument(
        "--agent", default="taught_seed43", help="label from the manifest (taught_seed43 / satiated_seed43 …)"
    )
    ap.add_argument("--manifest", default=str(_HERE.parent.parent / "docs/experiments/data/53_agents_manifest.json"))
    ap.add_argument("--delta", type=float, default=0.30, help="body-yaw step per glance (rad); 0.30 = Exp 53b")
    ap.add_argument("--min-az", type=float, default=0.10, help="|az| below this counts as already facing the speaker")
    ap.add_argument("--refractory", type=float, default=4.0, help="seconds to hold after a glance")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument(
        "--recenter-after", type=float, default=20.0, help="seconds of silence before drifting back to centre"
    )
    ap.add_argument(
        "--chain",
        type=int,
        default=1,
        help="max steps per speech onset; extra steps only while the substrate repeats the SAME direction "
        "and the speaker is still off-centre (stops on reversal — demo-side rule, see docstring)",
    )
    ap.add_argument("--log", default="/tmp/exp53_demo.jsonl")
    args = ap.parse_args(argv)

    os.environ["MAXIM_SUBSTRATE_TOOL_WHITELIST"] = "turn_left,turn_right"
    os.environ.pop("MAXIM_PLACE_CODE_EXTEROCEPTION", None)
    h.DELTAS.update({"turn_left": +args.delta, "turn_right": -args.delta})
    manifest = h._load_manifest(args.manifest)
    spec = next(a for a in manifest["agents"] if a["label"] == args.agent)
    host, _ = h.resolve_host(args.host)
    if host is None:
        print("[FAIL] --host <ip> or MAXIM_REACHY_HOST")
        return 2

    from maxim.simulation import sim_logger

    sink = h._ProvenanceSink()
    sim_logger.register_sim_sink(sink)
    agent = h.LoadedAgent(spec, 0.0)  # frozen policy: the learned bias decides, nothing else
    rig = h.LiveReadoutRig(host)
    log = h.JsonlLog(args.log)
    print(f"[demo] {args.agent}: {spec['bias_entries']} learned biases; δ={args.delta} rad; NOT EVIDENCE")
    print("[demo] speak from a side; the robot takes one glance per speech onset, then holds.")
    t0 = time.time()
    last_glance = -1e9
    last_speech = time.time()
    try:
        rig.recenter()
        while time.time() - t0 < args.duration:
            az = h.gated_azimuth(rig.reader, k=3, timeout_s=1.5, poll_s=0.15)
            now = time.time()
            if az is None:
                if now - last_speech > args.recenter_after and abs(rig._body_yaw_deg()) > 3:
                    print("  (quiet) drifting back to centre")
                    rig.recenter()
                    last_speech = now
                continue
            last_speech = now
            if now - last_glance < args.refractory:
                continue
            if abs(az) < args.min_az:
                print(f"  az {az:+.2f} — already facing the speaker")
                last_glance = now
                continue
            d = h.decide(agent, rig, sink)
            aff = d["affordance"]
            if aff is None:
                print(f"  az {az:+.2f} → no learned preference — sits still (control behaviour)")
                log.write("demo_decision", agent=args.agent, az=az, affordance=None, evidence=False)
                last_glance = now
                continue
            steps = 0
            first_aff = aff
            az_now = az
            while aff is not None and steps < args.chain:
                res = rig.execute(aff)
                steps += 1
                time.sleep(0.8)
                az2 = h.gated_azimuth(rig.reader, k=3, timeout_s=1.5, poll_s=0.15)
                print(
                    f"  az {az_now:+.2f} → {aff} (margin {d['learned_margin']}) → az {az2 if az2 is None else round(az2, 2)}"
                )
                log.write(
                    "demo_glance",
                    agent=args.agent,
                    step=steps,
                    az_before=az_now,
                    affordance=aff,
                    az_after=az2,
                    learned_margin=d["learned_margin"],
                    success=bool(getattr(res, "success", False)),
                    evidence=False,
                )
                if az2 is None or abs(az2) < args.min_az or steps >= args.chain:
                    break
                az_now = az2
                d = h.decide(agent, rig, sink)
                aff = d["affordance"]
                if aff != first_aff:
                    print(f"  (substrate says {aff} — reversal; holding)")
                    break
            last_glance = now
    except KeyboardInterrupt:
        print("[demo] stopped")
    finally:
        assert agent.files_unchanged(), "persisted files changed — the demo must never write"
        rig.close()
        sim_logger.unregister_sim_sink(sink)
    return 0


if __name__ == "__main__":
    sys.exit(main())
