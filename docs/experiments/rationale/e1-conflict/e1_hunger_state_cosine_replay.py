"""E1 hunger-state cosine replay (roadmap_1_4.md §Phase 2, "before any build" (ii)): does the water fear's key survive hunger?

Wire 4 writes the fear onto the WORLD cluster only (`pain_bus.create_pain_cluster_fear_subscriber`), and on the shipped
`minecraft_player` body `food` and `saturation` are WORLD-modality sensors. So the axis E1 titrates (hunger) sits inside
the fear's own key. This replay asks, on the committed live vectors, which hunger states share the trained water
cluster, which share the SHORE's cluster (so a hungry agent could not even form a water-specific fear in situ), and
what two variant-body geometry changes do to both.

Inputs: the LIVE gate-(ii) per-sensor normalized values of the Exp 60 pool (`exp60_geometry_2026-09-15b.json`: shore =
`v_safe`, submerged = `v_dark`, both recorded FED: food 20/40 and saturation 10/20, the bridge clamp); the shipped
`_stable_basis`; the shipped gain law w = (2|v − 0.5|)^3. The script REFUSES to print unless it reproduces the record's
live cos(shore, submerged) to 0.002 (the guard `exp62_cross_pool_replay.py` uses). Food/saturation are normalized on the
shipped ranges ([0, 40] and [0, 20]). Minecraft drains saturation to 0 before food falls, so a game-natively hungry
agent (food < 16, the hunger need's onset) carries saturation 0; the `sat 10` rows are a protocol-held state
(`effect give … saturation`), not a natural one.

Approximation, stated: the EC joins a node at cos ≥ 0.85 to that node's pattern; pairwise cosine against the TRAINING
vector stands in for the node pattern (a running centroid would sit between training states and only blur the edges).

Run from the repo root:  PYTHONPATH="$PWD/src" python docs/experiments/rationale/e1-conflict/e1_hunger_state_cosine_replay.py
It rewrites `e1_hunger_state_cosine_replay.json` beside itself (the committed result) and prints the tables.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")
from maxim.similarity.encoder import _stable_basis  # noqa: E402

DIM, P, TH = 384, 3.0, 0.85
HERE = Path(__file__).resolve().parent
REC = HERE.parents[1] / "data" / "exp60_geometry_2026-09-15b.json"
OUT = HERE / "e1_hunger_state_cosine_replay.json"
rec = json.loads(REC.read_text())
SHORE = {s["sensor"]: s["v_safe"] for s in rec["per_sensor"]}
SUB = {s["sensor"]: s["v_dark"] for s in rec["per_sensor"]}
LIVE = float(rec["cosine"]["a4_gained"])
FOODS = (20, 16, 14, 12, 10, 9, 8, 7, 6, 4, 2, 0)  # hunger need = (16 − food)/10, clipped to [0, 1]

# The geometries a prereg could freeze. `shipped` is today's body; the two variants are body-YAML changes a variant
# body could declare (roadmap §Phase 2: "a variant body may re-declare saturation's range so 'hungry' is silent"),
# each a geometry change that needs its own replay — this one.
GEOMETRIES: dict[str, dict[str, Any]] = {
    "shipped": {"note": "food [0,40] rest 20; saturation [0,20] rest 10; both world modality"},
    "sat_out_of_world": {"drop": ("saturation",), "note": "saturation moved off the world modality (or not declared)"},
    "hunger_out_of_world": {
        "drop": ("saturation", "food"),
        "note": "food and saturation both off the world modality: hunger silent to the world encoder (the drive still reads)",
    },
}


def embed(vmap: dict[str, float], drop: tuple[str, ...] = ()) -> list[float]:
    vec = [0.0] * DIM
    for name, v in vmap.items():
        if name in drop:
            continue
        w = (abs(v - 0.5) * 2.0) ** P
        if w == 0.0:
            continue
        lo = _stable_basis(name, DIM, salt="low")
        hi = _stable_basis(name, DIM, salt="high")
        for i in range(DIM):
            vec[i] += w * ((1.0 - v) * lo[i] + v * hi[i])
    return vec


def cos(a: list[float], b: list[float]) -> float:
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na and nb else 0.0


def state(base: dict[str, float], food: float, sat: float) -> dict[str, float]:
    return {**base, "food": food / 40.0, "saturation": sat / 20.0}


def natural_sat(food: float) -> float:
    """Game-native: saturation is 0 whenever the agent is hungry at all (food < 16); fed agents sit at the clamp 10."""
    return 10.0 if food >= 16 else 0.0


def main() -> int:
    base = cos(embed(SHORE), embed(SUB))
    print(f"live gate (ii): replayed cos(shore, submerged) = {base:.4f}  (record {LIVE})")
    if abs(base - LIVE) > 0.002:
        print("REFUSED: the replay does not reproduce the live record")
        return 2
    out: dict[str, Any] = {
        "_format_version": "1.0",
        "kind": "e1_hunger_state_cosine_replay",
        "inputs": {
            "record": REC.name,
            "live_cos": LIVE,
            "replayed_cos": round(base, 4),
            "threshold": TH,
            "foods": FOODS,
        },
        "geometries": {},
    }
    for gname, g in GEOMETRIES.items():
        drop = tuple(g.get("drop", ()))
        sub_fed = embed(state(SUB, 20, 10), drop)
        rows = []
        for f in FOODS:
            sat = natural_sat(f)
            sub_f = embed(state(SUB, f, sat), drop)
            shore_f = embed(state(SHORE, f, sat), drop)
            rows.append(
                {
                    "food": f,
                    "saturation": sat,
                    "hunger_need": round(max(0.0, min(1.0, (16 - f) / 10)), 2),
                    # A fed-trained fear (Exp 60/61's protocol trains fed) probed at this hunger level:
                    "cos_sub_fed_vs_sub_here": round(cos(sub_fed, sub_f), 4),
                    # Can THIS hunger state form a water-specific fear in situ? (shore and water must separate)
                    "cos_shore_here_vs_sub_here": round(cos(shore_f, sub_f), 4),
                    # The protocol-held alternative: saturation pinned at the fed clamp while food falls
                    "cos_sub_fed_vs_sub_here_sat_held": round(cos(sub_fed, embed(state(SUB, f, 10), drop)), 4),
                }
            )
        # Train-hungry titration: a fear trained at food T (natural saturation) probed at every other level.
        matrix = {
            str(t): {
                str(f): round(
                    cos(embed(state(SUB, t, natural_sat(t)), drop), embed(state(SUB, f, natural_sat(f)), drop)), 4
                )
                for f in FOODS
            }
            for t in (12, 8, 4)
        }
        out["geometries"][gname] = {"note": g["note"], "drop": list(drop), "rows": rows, "train_hungry_matrix": matrix}
        print(f"\n=== {gname}: {g['note']}  ('*' = below {TH})")
        print(" food sat need | fed-trained→here  shore↔sub here | sat-held fed→here")
        for r in rows:
            a, b, c = (
                r["cos_sub_fed_vs_sub_here"],
                r["cos_shore_here_vs_sub_here"],
                r["cos_sub_fed_vs_sub_here_sat_held"],
            )
            print(
                f"  {r['food']:3d} {r['saturation']:3.0f} {r['hunger_need']:4.2f} |  {a:.4f}{'*' if a < TH else ' '}"
                f"          {b:.4f}{'*' if b < TH else ' '}   |  {c:.4f}{'*' if c < TH else ' '}"
            )
        print(" train-hungry (natural saturation) → probe food:  " + " ".join(f"{f:>6d}" for f in FOODS))
        for t, row in matrix.items():
            print(
                f"   trained at food {t:>2s}:                     "
                + " ".join(f"{v:.3f}{'*' if v < TH else ' '}" for v in row.values())
            )
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
