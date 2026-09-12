#!/usr/bin/env python3
"""Live break-3 SMOKE: does the drive->eat->relief->credit loop CLOSE on the survival world?

R2 break 3 is "the world affords the corrective act". The apparatus (setup_world.py) makes
`eat` executable (food seeded, hunger drains). This smoke confirms — against the LIVE bridge —
that the three breaks COMPOSE on the real path:

  1. under a REAL hunger deficit, the substrate prior (break 1) selects `eat`, and
  2. `eat` EXECUTES via the bridge (break 3 — the world affords it), and
  3. the food rise SYNCS into vital_metrics and break-2's measured-relief credit fires on the
     INTEROCEPTIVE channel (drive_relief_channel == "interoceptive", drive_potential_diff > 0).

Minecraft hunger drains SLOWLY (a saturation buffer burns off first), so this first drains food
below the deprivation threshold with a strong game-native hunger effect BEFORE testing eat —
otherwise food sits satiated (>16), no deficit exists, and both the prior and the credit path
correctly do nothing (the first smoke's false negative).

PRINT-ONLY, no gated results — a wiring check, not a claim. Substrate-primary (no LLM in the
action path), so safe on the leader box beside qwen32b. The learned-bias-over-trials
measurement is a separate pre-registered, reviewed, provenance-stamped run.

Run ON the box hosting the bridge (it binds 127.0.0.1), server up + bridge connected + prepare
already run:

    python scripts/survival_world/break3_smoke.py --rcon-password maxim --username maxim
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from exp56.common import RconControl  # noqa: E402


def _food(aut) -> float | None:
    """Sync world truth and return the bot's current food, or None."""
    aut.backend.sync_world_sensors()
    vm = getattr(aut.executor.embodiment.root, "vital_metrics", {}) or {}
    if "food" not in vm:
        return None
    try:
        return float(vm["food"])
    except (TypeError, ValueError):
        return None


def _drain_until_hungry(aut, rcon, username: str, *, target: float, timeout_s: float) -> float | None:
    """Apply a strong game-native hunger effect and poll until food <= target."""
    print(f"draining food below {target} (game-native hunger effect) …")
    t0 = time.time()
    f = _food(aut)
    while time.time() - t0 < timeout_s:
        # Re-apply each poll (idempotent) — a single application can wear off before we reach target.
        rcon.command(f"effect give {username} minecraft:hunger 1000 20")
        time.sleep(1.5)
        f = _food(aut)
        print(f"  food={f}")
        if f is not None and f <= target:
            break
    # Clear the effect so it does not fight the eat relief we are about to measure.
    rcon.command(f"effect clear {username} minecraft:hunger")
    time.sleep(0.5)
    return _food(aut)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=25567)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=25575)
    ap.add_argument("--rcon-password", required=True)
    ap.add_argument("--username", default="maxim", help="the bridge bot's username")
    ap.add_argument("--eats", type=int, default=4, help="max eat cycles from the deficit")
    ap.add_argument("--target-food", type=float, default=3.0, help="drain food to at/below this first")
    # A heavily-saturated bot (from a prior run's eating) drains slowly — the food bar only
    # falls once the saturation buffer is gone — so give the drain real headroom.
    ap.add_argument("--drain-timeout-s", type=float, default=120.0)
    ap.add_argument("--satisfaction", type=float, default=16.0, help="stop eating at/above this")
    args = ap.parse_args(argv)

    from maxim.runtime.agent_loop import _read_drive_states
    from maxim.runtime.tool_dispatch import read_learning_side_effects
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

    # Wiring diagnostic (the thing my unit test hand-set — confirm the REAL AUT populates it):
    live = getattr(aut.executor.embodiment, "live_world_set_sensors", None)
    print(f"live_world_set_sensors (must contain 'food'): {sorted(live) if live else live}\n")

    rcon = None
    rows: list[dict] = []
    try:
        rcon = RconControl(args.rcon_host, args.rcon_port, args.rcon_password)
        deficit_food = _drain_until_hungry(
            aut, rcon, args.username, target=args.target_food, timeout_s=args.drain_timeout_s
        )
        print(f"\nreached food={deficit_food} — now testing the loop under a real deficit\n")

        # break 1, live: with a genuine deficit, does the prior pick eat?
        drives = _read_drive_states(aut.executor)
        rec = aut.bio.nac.recommend_action(
            agent_id="break3_smoke",
            available_tools=available,
            current_drives=drives,
            current_clusters=None,
            min_confidence=0.0,
        )
        prior_pick = None if rec is None else rec.get("tool_name")
        print(f"break-1 prior under deficit: drives={drives} -> prior_pick={prior_pick}\n")

        for i in range(args.eats):
            food_before = _food(aut)
            if food_before is not None and food_before >= args.satisfaction:
                # Satiated — no deficit left to relieve; eating here only tests the food
                # cap, not the loop. Stop rather than log noisy near-cap eats.
                print(f"eat {i}: skipped (food {food_before} >= satisfaction {args.satisfaction})")
                break
            out = aut.executor.execute({"tool_name": eat_tool, "params": {}})
            food_after = _food(aut)
            # The canonical parser the real turn uses (agent_loop / exp56 execute_and_record),
            # not a raw side_effects dict read. NOTE: this reads the relief SIGNAL break-2
            # produces; it does NOT call record_outcome, so no cluster reward is BOOKED here.
            side = read_learning_side_effects(out)
            rows.append(
                {
                    "eat_success": bool(getattr(out, "success", False)),
                    "food_before": food_before,
                    "food_after": food_after,
                    "channel": side.drive_relief_channel,
                    "diff": side.drive_potential_diff,
                    "withheld": side.drive_credit_withheld,
                }
            )
            r = rows[-1]
            print(
                f"eat {i}: food {r['food_before']}->{r['food_after']}  success={r['eat_success']}  "
                f"break2[channel={r['channel']} diff={r['diff']} withheld={r['withheld']}]"
                + ("" if r["eat_success"] else f"  error={out.error!r}")
            )
    finally:
        if rcon is not None:
            rcon.close()
        try:
            aut.client.close()
        except Exception:
            pass
        shutil.rmtree(persistence_dir, ignore_errors=True)  # throwaway fresh substrate

    def _rose(r):
        return r["food_before"] is not None and r["food_after"] is not None and r["food_after"] > r["food_before"]

    executed = [r for r in rows if r["eat_success"]]
    rose = [r for r in executed if _rose(r)]
    credited = [r for r in rose if r["channel"] == "interoceptive" and (r["diff"] or 0) > 0]

    break1 = prior_pick == eat_tool
    print("\n--- verdict ---")
    print(f"break 1 (prior picks eat under a real deficit): {break1} (picked {prior_pick}; drives={drives})")
    print(f"break 3 (eat EXECUTES via bridge):              {len(executed)}/{len(rows)} eats")
    print(f"  food actually rose after eat:                  {len(rose)}/{len(executed)} executed")
    print(f"break 2 (interoceptive relief SIGNAL):          {len(credited)}/{len(rose)} risen")
    # COMPOSITION requires ALL THREE: the prior fired (break 1), eat executed and food rose
    # (break 3), and every rise produced an interoceptive relief signal (break 2). The signal is
    # the credit path's INPUT; it is not enough alone — without break 1 the deficit was too mild
    # and the prior never chose eat.
    if break1 and executed and rose and credited and len(credited) == len(rose):
        print("\nCOMPOSITION VALIDATED: prior->eat->measured interoceptive relief SIGNAL fires live")
        print("(break 1 selects eat, break 3 executes it, break 2 measures + routes the relief).")
        print("SCOPE: this validates the credit path's INPUT (side effects) — NOT the cluster")
        print("BOOKING (record_outcome) or a learned bias, which are the pre-registered")
        print("measurement's job (the run that moves R2 off PREMISE-NULL).")
        return 0
    print("\nLOOP DID NOT CLOSE — see the failing stage(s) above.")
    if not break1:
        print(f"  break 1: the prior did not pick eat — deficit likely too mild (food={deficit_food}).")
        print("  Drain deeper (lower --target-food / raise --drain-timeout-s) so hunger is real.")
    if executed and not rose:
        print("  HINT: eat succeeded but food did not rise post-sync — likely the bridge's `eat`")
        print("  returns before bot.consume() resolves, so the backend syncs stale food. That is a")
        print("  SHARED-CODE fix (scripts/minecraft_bridge/index.js) -> exp56-rerun guard applies.")
    elif rose and not credited:
        print("  HINT: food rose but break-2 did not credit — inspect drive_relief_channel/diff above")
        print("  against live_world_set_sensors (must contain 'food') and the food drive thresholds.")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
