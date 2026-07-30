#!/usr/bin/env python3
"""Attribute the `respond` fixation: learned saturation, or prior/framing?

Three hypotheses are on record and NONE is attributed:

  A. PROMPT FRAMING — docs/bugs/sim_embodiment_followups.md Issue 1
     (2026-04-19): the orchestrator addresses the AUT as if it were a human,
     triggering respond/request_interaction.
  B. CREDIT-ON-EXECUTION SATURATION — `respond` always succeeds, so
     record_outcome books +1 every turn and its causal confidence snowballs
     (docs/plans/deferred/credit_on_progress_not_execution.md).
  C. THE LLM'S OWN PRIOR — a model asked "what next?" with a respond tool
     available reaches for it.

THE DISCRIMINATOR IS THE TRAJECTORY, not the rate:

  * B is LEARNED, so respond-preference must GROW across turns and NAc
    confidence for respond must climb.
  * A and C are PRIORS, so respond-preference is high from TURN ONE and flat.

So: run N turns that each have an obvious non-respond action available, and
record per turn (a) whether respond was used, (b) which other tools ran, and
(c) NAc's causal confidence for respond. A flat-high curve falsifies B as the
DOMINANT cause; a rising curve supports it.

This does NOT use the substrate-primary explore bonus: `respond` is not in that
repertoire, so it cannot discriminate here (an earlier framing of this
experiment got that wrong).

    maxim serve                     # terminal 1
    python scripts/measure_respond_fixation.py --turns 6   # terminal 2

Costs one real LLM call per turn. On a 14B local model expect ~90s/turn.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

#: Prompts that each have an OBVIOUS non-respond action. If the agent responds
#: to these instead of acting, that is the fixation under study. Deliberately
#: mundane and repeatable — no novelty confound.
PROBES = [
    "List the files in the current directory.",
    "What files are in this folder? Use a tool to check.",
    "Read the first few lines of README.md.",
    "Search the web for today's date.",
    "List the directory contents again.",
    "Check what's in README.md one more time.",
    "Look up the current weather.",
    "Show me the files here.",
]


def post(base: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        f"{base}/api/run",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def nac_respond_confidence(agent_id: str = "console_agent") -> float | None:
    """Read NAc's learned confidence for `respond`, if the store is readable."""
    try:
        import json as _json
        from pathlib import Path

        from maxim.utils.paths import agent_data

        p = Path(agent_data(agent_id)) / "nac.json"
        if not p.is_file():
            # Before the bio_stack NAc-persistence fix this file NEVER existed,
            # so this read silently returned None every turn and the confidence
            # column was meaningless. It is also only written at session END,
            # so expect None until the handle stops.
            return None
        data = _json.loads(p.read_text())
        best = 0.0
        for link in (data.get("causal_links") or {}).values() if isinstance(data.get("causal_links"), dict) else []:
            sig = str(link.get("event_signature", ""))
            if "respond" in sig:
                best = max(best, float(link.get("confidence", 0.0) or 0.0))
        return best
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--out", default="respond_fixation.jsonl")
    args = ap.parse_args()

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("pip install websockets") from None

    ws_url = args.base.replace("http", "ws", 1) + "/ws"
    rows = []
    print(f"{args.turns} turns — each has an obvious non-respond action available.\n")
    # ping_interval=None is LOAD-BEARING: each turn blocks in a synchronous POST
    # for up to ~2 minutes without servicing the socket, so the default keepalive
    # ping times out and the server closes with 1011. The first run of this
    # harness died exactly that way mid-measurement — and, worse, the turns that
    # DID complete recorded no tools, so a broken instrument produced
    # plausible-looking "the agent never acted" rows. Events are collected on a
    # background thread for the same reason.
    with connect(ws_url, max_size=None, ping_interval=None) as ws, open(args.out, "w") as fh:
        import queue
        import threading

        inbox: queue.Queue = queue.Queue()
        stop = threading.Event()

        def _pump():
            while not stop.is_set():
                try:
                    inbox.put(json.loads(ws.recv(timeout=1.0)))
                except TimeoutError:
                    continue
                except Exception:
                    return

        threading.Thread(target=_pump, daemon=True).start()
        for i in range(args.turns):
            probe = PROBES[i % len(PROBES)]
            conf_before = nac_respond_confidence()
            t0 = time.time()
            detail = post(args.base, {"mode": "talk", "input": probe}).get("detail", "")
            elapsed = time.time() - t0

            # Drain whatever this turn produced (from the pump, not the socket).
            tools: list[str] = []
            replied = False
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    evt = inbox.get(timeout=0.5)
                except Exception:
                    continue
                k = evt.get("kind")
                if k == "motor":
                    msg = evt.get("message", "")
                    for name in ("respond", "speak", "list_directory", "read_file", "glob", "internet_search", "bash"):
                        if name in msg:
                            tools.append(name)
                            break
                elif k == "response":
                    replied = bool(evt.get("data", {}).get("text"))

            conf_after = nac_respond_confidence()
            used_respond = "respond" in tools or "speak" in tools
            acted = [t for t in tools if t not in ("respond", "speak")]
            row = {
                "turn": i + 1,
                "probe": probe,
                "used_respond": used_respond,
                "other_tools": acted,
                "replied": replied,
                "nac_respond_conf_before": conf_before,
                "nac_respond_conf_after": conf_after,
                "seconds": round(elapsed, 1),
                "detail": detail,
            }
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"turn {i + 1}: respond={'Y' if used_respond else 'n'} "
                f"other={acted or '-'} replied={'Y' if replied else 'n'} "
                f"nac_conf={conf_after} ({elapsed:.0f}s)"
            )
        stop.set()

    # ── verdict ──
    print("\n" + "=" * 62)
    first_half = rows[: max(1, len(rows) // 2)]
    second_half = rows[len(rows) // 2 :]
    r1 = sum(r["used_respond"] for r in first_half) / len(first_half)
    r2 = sum(r["used_respond"] for r in second_half) / len(second_half)
    print(f"respond-rate  first half: {r1:.0%}   second half: {r2:.0%}")
    confs = [r["nac_respond_conf_after"] for r in rows if r["nac_respond_conf_after"] is not None]
    if confs:
        print(f"NAc respond confidence: {confs[0]:.3f} → {confs[-1]:.3f}")
    else:
        print("NAc respond confidence: unreadable (no nac.json yet — turns may not have persisted)")

    print()
    if r1 >= 0.8 and r2 >= 0.8:
        print("VERDICT → flat-HIGH from turn one. Favours A (prompt framing) / C (LLM")
        print("          prior). Saturation (B) is NOT the dominant cause; it cannot")
        print("          explain fixation that exists before any learning.")
    elif r2 > r1 + 0.2:
        print("VERDICT → RISING. Supports B (credit-on-execution saturation): the")
        print("          preference is being learned within the session.")
    elif r1 < 0.3 and r2 < 0.3:
        print("VERDICT → the agent mostly ACTED. No fixation reproduced on these")
        print("          probes — re-scope (the April report was a sim orchestrator")
        print("          context, not console talk).")
    else:
        print("VERDICT → mixed / underpowered. Raise --turns before concluding.")
    print(f"\nrows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
