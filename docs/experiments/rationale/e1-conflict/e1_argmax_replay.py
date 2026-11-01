"""E1 argmax replay (roadmap_1_4.md §Phase 2, "before any build" (i)): where does the teleported conflict cross?

The agent is submerged at the food, hungry, with a learned water fear. E1 v2 says the first executed choice is a pure
function of protocol constants: `hunger × 0.7` on `eat` against `fear × 0.7` on `escape_water`/`flee`, both behind
the 0.5 activation floor. This replay checks that claim on the SHIPPED code, before any harness exists. Nothing here
is a mirror:
- the body is the real `bodies/minecraft_player` (ComponentRegistry), read by the shipped
  `substrate_proposal._read_drive_states` (so the hunger need is the real `corrective_need_intensity` of the `food`
  DriveSpec);
- the fear is written by the real `NAc.record_cluster_fear` (failure mode `drive:oxygen`, the Exp 60 allowlist) and read
  by the real `NAc.anticipatory_threat_need`, merged into `threat` by max (the three lines of
  `substrate_proposal.propose_via_substrate`'s Wire-4 read, copied because that function also encodes and ticks);
- causal links come from the real `tool_dispatch.record_outcome` (the shipped write a successful or failed call books);
- the choice is the real `NAc.recommend_action` (explore weight 0.0 as every survival harness pins it; drive gate both
  ways, since `sim.drive_gate_enabled` is config, not pinned).

Grid: food 0..20 (hunger need = (16 − food)/10 on the shipped spec) × fear {0, 0.49, 0.5, 0.75 (D, ingested), 1.0 (C,
self-learned cap)} × three rosters × four causal-history states × drive gate off/on. Inputs are all committed: the body
YAML, the B1 campaign's stamped 26-tool set, and Phase 0 item 4's proposed verb names.

Run from the repo root:  PYTHONPATH="$PWD/src" python docs/experiments/rationale/e1-conflict/e1_argmax_replay.py
It rewrites `e1_argmax_replay.json` beside itself (the committed result) and prints the summary.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.disable(logging.WARNING)
sys.path.insert(0, "src")

from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.embodiment.body import Embodiment  # noqa: E402
from maxim.embodiment.component_registry import ComponentRegistry  # noqa: E402
from maxim.runtime.executor import Executor  # noqa: E402
from maxim.runtime.tool_policy import ToolPolicy  # noqa: E402
from maxim.runtime.substrate_proposal import _read_drive_states  # noqa: E402
from maxim.runtime.tool_dispatch import record_outcome  # noqa: E402
from maxim.tools.registry import ToolRegistry  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "e1_argmax_replay.json"
B1_ROWS = HERE.parents[1] / "data" / "b1_exp60_floor_48ce40118c20.jsonl"
AGENT, WORLD = "e1_aut", "w_water"
P = "minecraft_player_"
# Phase 0 item 4's param-free verbs (roadmap_1_4.md §Phase 0 (4)); proposed names, not yet shipped.
ITEM4_VERBS = ("swim_forward", "swim_back", "swim_left", "swim_right", "swim_up", "sink", "turn_left", "turn_right")
FEARS = (0.0, 0.49, 0.5, 0.75, 1.0)
FOODS = tuple(range(0, 21))


def _b1_roster() -> list[str]:
    """The 26-tool set B1 stamped per seed (campaign 48ce40118c20), the shipped body's live roster."""
    for line in B1_ROWS.read_text().splitlines():
        sets = (json.loads(line).get("policy") or {}).get("tool_sets")
        if sets:
            return list(sets[0])
    raise SystemExit("no tool_sets in the B1 rows: the committed input moved")


def rosters() -> dict[str, list[str]]:
    shipped = _b1_roster()
    with_verbs = shipped + [P + v for v in ITEM4_VERBS]
    # A variant body renames every affordance `{body}_{affordance}` (roadmap §Bodies); the prefix is a guess, and the
    # replay shows whether the prefix can move a tie (it cannot among body tools, which share it).
    variant = [t.replace(P, "minecraft_e1_") for t in with_verbs]
    return {"shipped_26": shipped, "shipped_plus_item4": with_verbs, "variant_body_plus_item4": variant}


@functools.lru_cache(maxsize=None)
def _drives_at(food: int) -> tuple[tuple[str, float], ...]:
    """The shipped drive read at this food level (cached: the body is rebuilt once per level, not per cell)."""
    return tuple(sorted(_read_drive_states(_executor_at(food)).items()))


def _executor_at(food: float) -> Executor:
    root = ComponentRegistry().instantiate("bodies/minecraft_player")
    root.vital_metrics.update(food=float(food), health=20.0, oxygen=14.0, saturation=0.0, is_in_water=1.0)
    ex = Executor(tool_registry=ToolRegistry(), policy=ToolPolicy.unrestricted(reason="E1 argmax replay"))
    ex.embodiment = Embodiment(root=root)
    return ex


def _book(nac: NAc, tool: str, *, success: bool) -> None:
    record_outcome(
        bridge_booked=None, refused=None, agent_id=AGENT, tool_name=tool, success=success, result_summary="replay",
        error=None if success else "replay failure", reasoning="", recent_outcomes=[], max_recent=10,
        llm_worker=None, context_pool=_Ctx(), nac=nac, embodiment_failed=False,
    )  # fmt: skip


class _Ctx:
    def add_outcome(self, **_: Any) -> None:
        pass


HISTORIES = {
    "fresh": (),
    # One surfacing in training or a prior placement: the escape link forms on the first success (roadmap §Phase 0).
    "escape_succeeded_once": (("escape_water", True),),
    # One eat with bread in hand at food < 20 (E1's own protocol books this on its first eat).
    "eat_succeeded_once": (("eat", True),),
    # `flee` fails fast underwater (B1 apparatus); the 0.77 s tie-break R3 measured.
    "flee_failed_once": (("flee", False),),
}


def _nac(fear: float, history: tuple[tuple[str, bool], ...], tools: list[str], gate: bool) -> NAc:
    nac = NAc(config=NACConfig(substrate_explore_bonus_weight=0.0, drive_gate_enabled=gate))
    remaining = fear
    while remaining > 1e-12:  # valence -= alpha(0.5) * intensity, so intensity 2*remaining reaches -fear exactly
        step = min(remaining, nac.config.cluster_fear_alpha * 1.0)
        nac.record_cluster_fear(AGENT, WORLD, "drive:oxygen", step / nac.config.cluster_fear_alpha)
        remaining -= step
    for short, ok in history:
        (full,) = [t for t in tools if t.endswith("_" + short) and not t.startswith("read_")]
        _book(nac, full, success=ok)
    return nac


def cell(food: int, fear: float, tools: list[str], history: str, gate: bool) -> dict[str, Any]:
    nac = _nac(fear, HISTORIES[history], tools, gate)
    drives = dict(_drives_at(food))
    clusters = {"world": WORLD, "interoception": "i_any"}
    fear_need = float(nac.anticipatory_threat_need(AGENT, clusters))
    if fear_need > 0.0:  # substrate_proposal.propose_via_substrate's Wire-4 merge, verbatim in effect
        drives["threat"] = max(float(drives.get("threat", 0.0) or 0.0), fear_need)
    rec = nac.recommend_action(agent_id=AGENT, available_tools=tools, current_drives=drives, current_clusters=clusters)
    return {
        "food": food,
        "hunger_need": round(drives.get("hunger", 0.0), 4),
        "fear": fear,
        "threat_need": round(drives.get("threat", 0.0), 4),
        "winner": None
        if rec is None
        else rec["tool_name"].split("_", 2)[-1]
        if rec["tool_name"].startswith("minecraft_")
        else rec["tool_name"],
        "winner_full": None if rec is None else rec["tool_name"],
        "confidence": None if rec is None else round(float(rec["confidence"]), 4),
    }


def crossing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Lowest food at which the winner is NOT `eat` (scanning food upward = need downward)."""
    eat_foods = [r["food"] for r in rows if r["winner"] and r["winner"].endswith("eat")]
    return {"eat_wins_at_food": eat_foods, "eat_wins_iff_food_le": max(eat_foods) if eat_foods else None}


def main() -> int:
    out: dict[str, Any] = {
        "_format_version": "1.0",
        "kind": "e1_argmax_replay",
        "inputs": {},
        "summary": [],
        "cells": [],
    }
    rs = rosters()
    out["inputs"] = {
        "rosters": rs,
        "fears": FEARS,
        "foods": FOODS,
        "histories": {k: list(v) for k, v in HISTORIES.items()},
    }
    for rname, tools in rs.items():
        for hist in HISTORIES:
            for gate in (False, True):
                for fear in FEARS:
                    rows = [cell(f, fear, tools, hist, gate) for f in FOODS]
                    out["cells"].extend({"roster": rname, "history": hist, "drive_gate": gate, **r} for r in rows)
                    winners = sorted({str(r["winner"]) for r in rows})
                    out["summary"].append(
                        {
                            "roster": rname,
                            "history": hist,
                            "drive_gate": gate,
                            "fear": fear,
                            "winners": winners,
                            **crossing(rows),
                        }
                    )
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'roster':26s} {'history':22s} gate fear  eat iff food<=  winners")
    for s in out["summary"]:
        print(
            f"{s['roster']:26s} {s['history']:22s} {int(s['drive_gate'])}    {s['fear']:<4}  {str(s['eat_wins_iff_food_le']):>13s}  {','.join(s['winners'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
