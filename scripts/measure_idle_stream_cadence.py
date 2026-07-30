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

    # The REPORT was specifically ~2/s of hippocampus+scn. Judge against THAT,
    # not against "any non-meta kind" — the first version of this verdict
    # counted `pipeline` at 0.1/s as a confirmation, which is a 20x miss and
    # exactly the kind of instrument error this whole exercise keeps hitting.
    reported = {k: n for k, n in kinds.items() if k in ("hippocampus", "scn")}
    other = {k: n for k, n in kinds.items() if k not in ("heartbeat", "identity", "run", "dropped", *reported)}
    reported_rate = sum(reported.values()) / elapsed
    print()
    if reported_rate >= 1.0:
        print(f"VERDICT: CONFIRMED — {reported_rate:.2f}/s hippocampus+scn while idle {dict(reported)}.")
        print("         Next: find the per-iteration caller (store/recall vs SCN registration).")
    elif reported:
        print(f"VERDICT: PARTIAL — hippocampus+scn present but only {reported_rate:.2f}/s, far below")
        print(f"         the reported ~2/s {dict(reported)}. Likely tied to activity, not idle.")
    else:
        print("VERDICT: NOT REPRODUCED — zero hippocampus/scn records while idle.")
        print("         The reported ~2/s was an ACTIVE-turn cadence, not an idle one.")
        print("         Re-scope: measure during a turn, not after it.")
    if other:
        print(
            f"         (other non-meta traffic, unrelated to the report: {dict(other)} "
            f"= {sum(other.values()) / elapsed:.2f}/s)"
        )
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
