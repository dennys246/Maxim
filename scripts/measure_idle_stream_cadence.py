#!/usr/bin/env python3
"""Measure the console /ws record cadence while the agent is IDLE.

Open bug 1 in docs/bugs/console_seam_findings.md: with a talk loop alive and
nothing happening, `hippocampus` and `scn` records arrive at roughly the loop's
2 Hz. The emitters are per-OPERATION (`sim_memory` on store/recall, `sim_scn` on
SCN registration), not per-tick — so something performs a memory operation every
iteration. This measures WHICH kinds arrive and how fast, instead of reasoning
about it (two symptoms in that same batch turned out to be instrument artifacts).

Attaches to a RUNNING `maxim serve`, sits idle, and reports per-kind counts and
rate. Sends no SubscribeFrame — it records the unfiltered firehose, which is
what a client sees by default.

    # terminal 1
    maxim serve
    # terminal 2 — start a talk loop so there IS an idle loop to observe,
    # then measure. Without a turn first, no loop exists and the answer is
    # trivially zero (which is itself worth confirming — see --no-warmup).
    python scripts/measure_idle_stream_cadence.py --seconds 60

Interpretation:
  * ~2/s hippocampus+scn  → confirms the report; the loop is doing memory work
    per iteration even with no stimulus.
  * only heartbeats       → the cadence was tied to an ACTIVE turn, not idle,
    and the original observation needs re-scoping.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
import urllib.request


def warmup(base: str, text: str) -> None:
    """Send one talk turn so a live loop exists to observe."""
    req = urllib.request.Request(
        f"{base}/api/run",
        data=json.dumps({"mode": "talk", "input": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"warmup turn (this costs one real LLM call) → {text!r}")
    with urllib.request.urlopen(req, timeout=600) as r:
        print("  ", json.loads(r.read()).get("detail", ""))


def measure(ws_url: str, seconds: float) -> None:
    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("pip install websockets") from None

    kinds: collections.Counter[str] = collections.Counter()
    first_seen: dict[str, float] = {}
    seqs: list[int] = []
    t0 = time.time()
    print(f"listening {seconds:.0f}s (idle — do not interact) …")
    with connect(ws_url, max_size=None) as ws:
        while time.time() - t0 < seconds:
            remaining = seconds - (time.time() - t0)
            try:
                raw = ws.recv(timeout=max(0.5, remaining))
            except TimeoutError:
                continue
            evt = json.loads(raw)
            k = evt.get("kind", "?")
            kinds[k] += 1
            first_seen.setdefault(k, time.time() - t0)
            if isinstance(evt.get("seq"), int):
                seqs.append(evt["seq"])

    elapsed = time.time() - t0
    total = sum(kinds.values())
    print(f"\n{total} events in {elapsed:.1f}s = {total / elapsed:.2f}/s\n")
    print(f"{'kind':16} {'count':>6} {'per_sec':>8}  first_at")
    for k, n in kinds.most_common():
        print(f"{k:16} {n:>6} {n / elapsed:>8.2f}  {first_seen[k]:.1f}s")

    non_meta = {k: n for k, n in kinds.items() if k not in ("heartbeat", "identity", "run", "dropped")}
    print()
    if not non_meta:
        print("VERDICT: only meta-kinds while idle — the reported ~2/s was NOT an idle")
        print("         cadence. Re-scope the observation to active turns.")
    else:
        rate = sum(non_meta.values()) / elapsed
        print(f"VERDICT: {rate:.2f}/s of real records while idle: {dict(non_meta)}")
        print("         Confirms the report. Next: find the per-iteration caller —")
        print("         `hippocampus` points at store/recall, `scn` at SCN registration.")
    gaps = sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)
    print(f"seq gaps (dropped events): {gaps}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--no-warmup", action="store_true", help="Skip the talk turn (measures a loop-less server)")
    ap.add_argument("--warmup-text", default="Hi — just checking in, no need to do anything.")
    args = ap.parse_args()

    if not args.no_warmup:
        warmup(args.base, args.warmup_text)
        print("settling 5s before measuring …")
        time.sleep(5)
    measure(args.base.replace("http", "ws", 1) + "/ws", args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
