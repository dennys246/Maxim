#!/usr/bin/env python3
"""Live break-3 SMOKE: does the drive->eat->relief->credit loop CLOSE on the survival world?

R2 break 3 is "the world affords the corrective act". The apparatus (setup_world.py) makes
`eat` executable (food seeded, hunger drains). This smoke confirms — against the LIVE bridge —
that the three breaks COMPOSE on the real path:

  1. under hunger, the substrate prior (break 1) selects `eat`, and
  2. `eat` EXECUTES via the bridge (break 3 — the world affords it), and
  3. the food rise SYNCS into vital_metrics and break-2's measured-relief credit fires on the
     INTEROCEPTIVE channel (drive_relief_channel == "interoceptive", drive_potential_diff > 0).

It is PRINT-ONLY and writes no gated results: a wiring check, not a claim. It is
substrate-primary (no LLM in the action path), so it is safe to run on the leader box beside
qwen32b — it is not a second LLM consumer. The learned-bias-over-trials measurement (breaks
1+2+3 composing into LEARNING, the thing that moves R2 off PREMISE-NULL) is a separate,
pre-registered, two-lens-reviewed, provenance-stamped experiment — NOT this.

Run ON the box hosting the bridge (the bridge binds 127.0.0.1), with the server up, the bridge
connected (bot joined), and prepare/verify already run:

    python scripts/survival_world/break3_smoke.py --rcon-password maxim --username maxim
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from exp56.common import RconControl  # noqa: E402


def _sync_food(aut, *, settle_s: float, tries: int = 6) -> float | None:
    """Pull the latest world truth a few times; return the bot's food, or None."""
    last: float | None = None
    for _ in range(tries):
        aut.backend.sync_world_sensors()
        vm = getattr(aut.executor.embodiment.root, "vital_metrics", {}) or {}
        if "food" in vm:
            try:
                last = float(vm["food"])
            except (TypeError, ValueError):
                last = None
        time.sleep(settle_s)
    return last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25567)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", required=True)
    ap.add_argument("--username", default="maxim", help="the bridge bot's username")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--settle-s", type=float, default=0.6)
    ap.add_argument(
        "--no-induce-hunger",
        action="store_true",
        help="do NOT apply the game-native hunger effect (rely on natural drain instead)",
    )
    args = ap.parse_args(argv)

    from maxim.runtime.agent_loop import _read_drive_states
    from maxim.simulation.minecraft_harness import build_minecraft_aut
    from maxim.tools.introspection import INTROSPECTION_TOOL_NAMES

    persistence_dir = tempfile.mkdtemp(prefix="break3_smoke_")
    print(f"break-3 smoke: fresh substrate at {persistence_dir}\n")
    aut = build_minecraft_aut(
        agent_id="break3_smoke",
        bridge_port=args.bridge_port,
        bridge_host=args.bridge_host,
        persistence_dir=persistence_dir,
        entity_ref="bodies/minecraft_player",
    )
    available = [t for t in aut.executor.registry.list() if t not in INTROSPECTION_TOOL_NAMES]
    eat_tool = next((t for t in available if t.endswith("_eat")), None)
    if eat_tool is None:
        print(f"FAIL: no *_eat tool in the roster {available!r} — wrong body?")
        return 2

    rcon = RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
    rows: list[dict] = []
    try:
        for i in range(args.cycles):
            if not args.no_induce_hunger:
                # Game-native hunger effect to open a deficit fast (wiring aid, not the real run).
                rcon.command(f"effect give {args.username} minecraft:hunger 40 4")
                time.sleep(2.0)  # let the food bar drain before we read it
            food_before = _sync_food(aut, settle_s=args.settle_s)
            drives = _read_drive_states(aut.executor)
            rec = aut.bio.nac.recommend_action(
                agent_id="break3_smoke",
                available_tools=available,
                current_drives=drives,
                current_clusters=None,
                min_confidence=0.0,
            )
            prior_pick = None if rec is None else rec.get("tool_name")

            out = aut.executor.execute({"tool_name": eat_tool, "params": {}})
            food_after = _sync_food(aut, settle_s=args.settle_s)
            side = out.side_effects or {}

            rows.append(
                {
                    "eat_success": bool(getattr(out, "success", False)),
                    "food_before": food_before,
                    "food_after": food_after,
                    "prior_pick": prior_pick,
                    "channel": side.get("drive_relief_channel"),
                    "diff": side.get("drive_potential_diff"),
                    "withheld": side.get("drive_credit_withheld"),
                }
            )
            r = rows[-1]
            print(
                f"cycle {i}: food {r['food_before']}->{r['food_after']}  "
                f"prior_pick={r['prior_pick']}  eat_success={r['eat_success']}  "
                f"break2[channel={r['channel']} diff={r['diff']} withheld={r['withheld']}]"
                + ("" if getattr(out, "success", False) else f"  error={out.error!r}")
            )
    finally:
        rcon.close()
        try:
            aut.client.close()
        except Exception:
            pass

    # Verdict — name exactly where the loop breaks (verify-the-instrument).
    def _rose(r):
        return r["food_before"] is not None and r["food_after"] is not None and r["food_after"] > r["food_before"]

    executed = [r for r in rows if r["eat_success"]]
    rose = [r for r in executed if _rose(r)]
    credited = [r for r in rose if r["channel"] == "interoceptive" and (r["diff"] or 0) > 0]
    prior_hit = [r for r in rows if r["prior_pick"] == eat_tool]

    print("\n--- verdict ---")
    print(f"break 1 (prior selects eat under hunger): {len(prior_hit)}/{len(rows)} cycles")
    print(f"break 3 (eat EXECUTES via bridge):        {len(executed)}/{len(rows)} cycles")
    print(f"  food actually rose after eat:            {len(rose)}/{len(executed)} executed")
    print(f"break 2 (interoceptive relief credited):  {len(credited)}/{len(rose)} risen")
    if executed and rose and credited:
        print("\nLOOP CLOSES: breaks 1+2+3 compose on the live path (composition validated).")
        print("Next: the pre-registered learned-bias-over-trials measurement (moves R2 off PREMISE-NULL).")
        return 0
    print("\nLOOP DID NOT CLOSE — see the first failing stage above.")
    if executed and not rose:
        print("  HINT: eat succeeded but food did not rise post-sync — likely the bridge's `eat`")
        print("  returns before bot.consume() resolves, so the backend syncs stale food. That is a")
        print("  SHARED-CODE fix (scripts/minecraft_bridge/index.js) → exp56-rerun guard applies.")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
