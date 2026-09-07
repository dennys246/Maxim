"""R2 — premise check: do the minecraft_player world-owned drives move behaviour?

The Minecraft survival ladder (docs/plans/minecraft_benchmark.md Part II) rung R2
asks whether the world-owned `health` (HomeostaticDriveSpec) and `food`
(EntropicDriveSpec) drives measurably change action selection toward corrective
affordances (`eat` when hungry, `attack_nearest` when threatened) — "the game
drains it, not the model." R3/R4 do not run if they do not.

This probe measures the INTRINSIC path AS SHIPPED, no drive-path fix (fixing it
to make survival rewarding would be engineering the outcome, D1's spirit). On a
substrate-primary agent, first-contact action selection has exactly one active
signal on a fresh substrate — the NAc cold-start drive prior (learned bias,
causal links and reward bias are all zero before any experience; the explore
bonus is default-0). So a fresh-NAc probe of `recommend_action` over the real
tool roster ISOLATES the drive prior exactly. It is a pure function of
(available_tools, current_drives) — a live server would add noise, not signal,
on this channel — so the offline probe is the exact instrument, not an
approximation.

It reports, for STARVING+HURT vs SATIATED+HEALTHY, which action the drive prior
selects and its reasoning. A moving-behaviour PASS would select a corrective
affordance under deficit and something else (or nothing) when satisfied. Run:

    export PYTHONPATH="$PWD/src"   # if running from a worktree
    python scripts/r2_drive_premise_probe.py --write-experiment-results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _provenance import executed_code_provenance, evidence_out_paths  # noqa: E402

from maxim.embodiment.spec import resolve_entity_spec, _parse_entity  # noqa: E402
from maxim.embodiment.component_registry import ComponentRegistry  # noqa: E402
from maxim.embodiment.tool_bridge import generate_tools_for_entity  # noqa: E402
from maxim.tools.registry import ToolRegistry  # noqa: E402
from maxim.tools.introspection import INTROSPECTION_TOOL_NAMES  # noqa: E402
from maxim.runtime.agent_loop import _read_drive_states  # noqa: E402
from maxim.decisions.nac import NAc, NACConfig  # noqa: E402

BODY_REF = "bodies/minecraft_player"
CORRECTIVE = {"minecraft_player_eat", "minecraft_player_attack_nearest"}

# The states probed: raw sensor values as the bridge writes them (food 20 = full,
# health 20 = full; both bodies deficient in STARVING+HURT).
STATES = [
    ("STARVING+HURT", {"food": 2.0, "health": 5.0}),
    ("SATIATED+HEALTHY", {"food": 20.0, "health": 20.0}),
]


def _build() -> tuple[object, list[str]]:
    creg = ComponentRegistry()
    root = _parse_entity(resolve_entity_spec(BODY_REF, creg))
    registry = ToolRegistry()
    generate_tools_for_entity(root, registry)
    # Mirror runtime/agent_loop.propose_via_substrate exactly.
    available = [t for t in registry.list() if t not in INTROSPECTION_TOOL_NAMES]
    return root, available


def _probe_state(root, available, values: dict[str, float]) -> dict:
    root.vital_metrics.update(values)
    executor = SimpleNamespace(embodiment=SimpleNamespace(root=root))
    drives = _read_drive_states(executor)
    nac = NAc(config=NACConfig())  # fresh — no learned bias / causal / reward
    rec = nac.recommend_action(
        agent_id="r2_probe",
        available_tools=available,
        current_drives=drives,
        current_clusters=None,
        min_confidence=0.0,  # do not gate — surface whatever the prior scores
    )
    selected = None if rec is None else rec.get("tool_name")
    return {
        "drives_read": drives,
        "selected": selected,
        "reasoning": None if rec is None else rec.get("reasoning"),
        "selected_is_corrective": selected in CORRECTIVE,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/experiments/data/r2_drive_premise.json")
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    out_path = evidence_out_paths(
        REPO_ROOT,
        [args.out],
        write_experiment_results=args.write_experiment_results,
        allow_dirty=args.allow_dirty,
    )[0]
    prov = executed_code_provenance(REPO_ROOT, "maxim", out_path=out_path, allow_dirty=args.allow_dirty)

    root, available = _build()
    per_state = {}
    for label, values in STATES:
        per_state[label] = _probe_state(root, available, dict(values))

    starve = per_state["STARVING+HURT"]
    full = per_state["SATIATED+HEALTHY"]
    # Behaviour "moves" iff a deficit selects a corrective affordance AND the
    # satisfied state does not select the same thing (state-dependence).
    moves_behaviour = starve["selected_is_corrective"] and (starve["selected"] != full["selected"])
    record = {
        "ts": time.time(),
        "rung": "R2",
        "body_ref": BODY_REF,
        "provenance": prov,
        "available_tools": sorted(available),
        "corrective_affordances": sorted(CORRECTIVE),
        "states": per_state,
        "moves_behaviour": moves_behaviour,
        "verdict": "PREMISE-HELD" if moves_behaviour else "PREMISE-NULL",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")

    print(json.dumps({k: record[k] for k in ("rung", "verdict", "moves_behaviour")}, indent=2))
    for label, r in per_state.items():
        print(f"  {label}: drives={r['drives_read']} → {r['selected']} ({r['reasoning']})")
    print(f"  record → {out_path}")
    # A null is a valid result and exits 0 (it ships as a null, Exp 53 shape);
    # only an apparatus error is nonzero.
    return 0


if __name__ == "__main__":
    sys.exit(main())
