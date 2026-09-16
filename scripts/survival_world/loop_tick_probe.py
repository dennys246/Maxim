#!/usr/bin/env python3
"""Exp 60 diagnostic — WHERE does the live agent loop spend its time between substrate ticks?

The second one-seed run (bridge at 100 ms, cadence preflight PASSED) still read ``ticks=1`` per
4.3 s window: the loop's substrate branch fired once and not again. Offline (fake bridge) the same
loop ticks every 0.5–1 s. So on the live box each loop ITERATION is slow for a reason the fake
bridge does not reproduce — an installed optional dependency (embedding model), disk persistence,
the real percept content, something else. This probe MEASURES it instead of guessing:

* runs the FULL loop (``run_minecraft_aut``, substrate-primary, the harness's exact kwargs) for
  ``--seconds`` on the shore with a ``SubstrateTelemetry`` writer → prints every substrate tick's
  time from loop start (the tick rate the harness's windows depend on);
* runs ``cProfile`` ON THE LOOP THREAD (``threading.setprofile``) → prints the top functions by
  cumulative time, so the hog is named by measurement.

Diagnostic only: no gated record, fresh throwaway persistence, no world changes (the bot is not
teleported). Run on the bridge box with the bridge up (at whatever cadence you want to test):

    export PYTHONPATH="$PWD/src"
    python scripts/survival_world/loop_tick_probe.py --seconds 8 --bridge-port 25567
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25567)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument(
        "--fake", action="store_true", help="use the in-process FakeBridgeServer (100 ms) instead of a live bridge"
    )
    args = ap.parse_args(argv)

    from maxim.simulation.minecraft_harness import build_minecraft_aut, run_minecraft_aut
    from maxim.simulation.substrate_telemetry import SubstrateTelemetry

    port = args.bridge_port
    srv = None
    if args.fake:
        from maxim.simulation.minecraft_harness import FakeBridgeServer

        srv = FakeBridgeServer(state_interval_s=0.1, events=False)  # the live condition: no text events
        port = srv.port
    pdir = tempfile.mkdtemp(prefix="loop_tick_probe_")
    aut = build_minecraft_aut(
        agent_id="loop_tick_probe",
        bridge_port=port,
        bridge_host=args.bridge_host,
        persistence_dir=pdir,
        entity_ref="bodies/minecraft_player",
    )
    time.sleep(0.5)
    ages = []
    for _ in range(30):
        ages.append(aut.client.state_age_s())
        time.sleep(0.05)
    print(
        f"bridge snapshot age (s) over 1.5 s: min {min(ages):.3f} max {max(ages):.3f} (a fresh bridge keeps this under its interval)"
    )

    telem_path = Path(pdir) / "telemetry.jsonl"
    telem = SubstrateTelemetry(log_path=telem_path, agent_id="loop_tick_probe")
    prof = cProfile.Profile()
    stop = threading.Event()

    # The substrate branch runs ONLY while no proposal is pending. Log every install/clear
    # of the controller's pending_proposal with the proposal's SOURCE, so a proposal that is
    # installed by another path and never executed (the live symptom: one substrate tick,
    # then silence) is named — strategy, tool, approval flag, plan text.
    from maxim.runtime.loop_controller import LoopController

    t_start = time.monotonic()
    installs: list[dict] = []
    _orig_prop = LoopController.pending_proposal

    def _set(self, proposal):  # type: ignore[no-untyped-def]
        if proposal is None:
            installs.append({"t": round(time.monotonic() - t_start, 3), "event": "clear"})
        else:
            act = getattr(proposal, "action", None) or {}
            installs.append(
                {
                    "t": round(time.monotonic() - t_start, 3),
                    "event": "install",
                    "strategy": getattr(proposal, "strategy_used", None),
                    "tool": act.get("tool_name") if isinstance(act, dict) else None,
                    "requires_approval": bool(getattr(proposal, "requires_approval", False)),
                    "has_plan_text": bool(getattr(proposal, "plan_text", None)),
                    "confidence": getattr(proposal, "confidence", None),
                    "reasoning": (getattr(proposal, "reasoning", "") or "")[:80],
                }
            )
        _orig_prop.fset(self, proposal)

    LoopController.pending_proposal = property(_orig_prop.fget, _set)

    def _target() -> None:
        prof.enable()
        try:
            run_minecraft_aut(aut, max_steps=1_000_000, target_hz=4.0, stop_event=stop, substrate_telemetry=telem)
        finally:
            prof.disable()

    th = threading.Thread(target=_target, daemon=True)
    try:
        th.start()
        time.sleep(args.seconds)
    finally:
        stop.set()
        th.join(timeout=30.0)
        LoopController.pending_proposal = _orig_prop  # class-wide patch: restore no matter what
    joined = not th.is_alive()

    rows = []
    for ln in telem_path.read_text().splitlines():
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass
    if rows:
        first = rows[0]["ts"]
        ticks = [round(r["ts"] - first, 2) for r in rows]
        print(f"\nsubstrate ticks in {args.seconds:.0f} s: {len(rows)}  (s from first tick): {ticks[:20]}")
        gaps = [round(b - a, 2) for a, b in zip(ticks, ticks[1:])]
        print(f"tick gaps: {gaps[:20]}")
    else:
        print(f"\nsubstrate ticks in {args.seconds:.0f} s: 0 — the loop never reached its substrate branch")
    print(f"loop thread stopped cleanly: {joined}")
    print(f"\npending_proposal timeline ({len(installs)} events; s from loop start):")
    for ev in installs[:40]:
        print("  ", json.dumps(ev))
    if not installs:
        print("   (no proposal was ever installed or cleared)")

    out = io.StringIO()
    st = pstats.Stats(prof, stream=out)
    st.sort_stats("cumulative").print_stats(args.top)
    print("\n=== loop thread profile (top by cumulative time) ===")
    print(out.getvalue())
    try:
        aut.client.close()
    except Exception as exc:
        print(f"WARNING: client close raised: {exc!r}")
    if srv is not None:
        srv.close()
    shutil.rmtree(pdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
