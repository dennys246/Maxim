"""E1 v3 toy model: the want against the fear, learned at the food. Can a reasonable floor lose, and can Maxim?

Not data and not a gate: the offline check the owner's 2026-10-16 decision asks for ("redesign E1 so a reasonable
no-bio floor CAN lose", baseline_floors.md §Why; B1's `rationale/b1-exp60-floor/learner_sim.py` is the precedent).
It is the evidence behind the floor decision taken 2026-10-18 (E1 prereg §Owner decisions, OE1).

The episode (live-like timings from Exp 60/R3/B1's records; every number is a toy assumption the pilot re-measures):
the agent starts submerged AT the food, oxygen 20, draining 8/6 per s; the saturating air-hunger pain lands at
t = 6 s (oxygen 12) and is terminal (rescue, as in Exp 60's training). It holds BREAD bread. `eat` blocks EAT_S and
adds 5 food if it holds bread and food < 20, else fails fast. `escape_water` blocks 2.6 s and ends the episode
surfaced. Every other tool returns in 0.5 s (`flee` fails underwater; reads succeed and do nothing). The world key is
ONE cluster at every hunger level: the variant body drops `saturation` from the world modality (owner decision OE2;
`docs/experiments/rationale/e1-conflict/e1_hunger_state_cosine_replay.py`, geometry `sat_out_of_world`).

Arms:
- MAXIM: the REAL NAc. `recommend_action` (explore 0.0), `record_cluster_fear` on the pain (Wire 4's write),
  `anticipatory_threat_need` (its read), and every call booked through the real `tool_dispatch.record_outcome`.
- TWIN: the mechanism twin (baseline_floors.md §Why): MAXIM with Wire 4's fear store replaced by a TD(0) state value on
  the same cluster (V ← V + 0.5(−1 − V) per pain; the need is −V when it clears θ = 0.5). Everything else is MAXIM's.
- Q_EXPLORE: tabular Q-learning, B1's frozen constants (α 0.5, γ 1, ε 0.2 train / 0 probe, Q₀ −2), over the full
  26-tool repertoire; reward = B1's oxygen cost −(20 − oxygen)/20 per 0.5 s, +1 per bread eaten (the relief term E1
  adds), −2 at the terminal pain (B1's PAIN_TERMINAL + SATURATING_PAIN). Key = the world cluster × a hunger band (the
  stand-in for B1's cluster TUPLE, whose interoception entry carries the hunger need; the real split is a pilot
  measurement).
- Q_RAW: the same learner keyed on raw sensors (is_in_water × hunger band × B1's oxygen bins).
- Q_PRIOR: Q_RAW limited to the prior's tools (`eat`, `flee`, `escape_water`): how much the prior carries (reported).
- Q_WORLD: the same learner keyed on the world cluster alone (no need in its state).
- RANDOM: uniform over the 26 tools.

Protocol: K training episodes alternating food 12 and 4, then ONE probe episode at each held-out food 14, 8, 2 (the
learner keeps learning, as B1's probes do). DVs, gated separately (never a weighted payoff, whose weights would pick
the winner): AVOID = P(pain) on the probe; WANT = bread eaten on the probe. Seeds: 200 per stochastic arm; MAXIM and
TWIN are deterministic.

Run from the repo root:  PYTHONPATH="$PWD/src" python docs/experiments/rationale/e1-conflict/learner_sim.py
It rewrites `learner_sim_results.json` beside itself and prints the tables. About 2 minutes.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

logging.disable(logging.WARNING)
sys.path.insert(0, "src")
from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.runtime.tool_dispatch import record_outcome  # noqa: E402

OUT = Path(__file__).with_name("learner_sim_results.json")
P = "minecraft_player_"
SHORT = ["attack_nearest", "eat", "escape_water", "flee", "mine_block", "move_to", "place_block", "turn"]
_READS = (
    "distance_from_spawn food health hostile_count is_in_water is_raining light_level look_pitch nearest_hostile_dist "
    "nearest_player_dist on_ground oxygen saturation speed time_of_day xp_level y_altitude"
).split()
TOOLS = [P + s for s in SHORT] + [f"read_{P}{s}" for s in _READS] + ["sense_minecraft_player"]  # B1's stamped 26
EAT, ESC, FLEE = P + "eat", P + "escape_water", P + "flee"
ESC_S, STEP, PAIN_T, TH = 2.6, 0.5, 6.0, 0.5
TRAIN_FOODS, PROBE_FOODS = (12, 4), (14, 8, 2)
AID = "e1_aut"


class _Ctx:
    def add_outcome(self, **_: Any) -> None:
        pass


def hunger_need(food: float) -> float:
    """The shipped `food` DriveSpec's corrective need (satisfaction 16, deprivation 6)."""
    return max(0.0, min(1.0, (16 - food) / 10))


def oxygen_at(t: float) -> float:
    return 20 - 8 / 6 * t


class Maxim:
    def __init__(self, *, td: bool = False) -> None:
        self.nac = NAc(NACConfig(substrate_explore_bonus_weight=0.0))
        self.td, self.v = td, 0.0

    def select(self, food: float, oxy: float) -> str | None:
        drives: dict[str, float] = {"food": food}  # the raw drive (skipped by the raw-sensor guard), as shipped
        if hunger_need(food) > 0:
            drives["hunger"] = hunger_need(food)
        if self.td:
            fear = -self.v if -self.v >= TH else 0.0
        else:
            fear = self.nac.anticipatory_threat_need(AID, {"world": "w_water"})
        if fear > 0:
            drives["threat"] = fear
        r = self.nac.recommend_action(
            agent_id=AID, available_tools=TOOLS, current_drives=drives, current_clusters={"world": "w_water"}
        )
        return None if r is None else r["tool_name"]

    def outcome(self, tool: str, ok: bool, reward: float, food: float, oxy: float, done: bool) -> None:
        record_outcome(
            bridge_booked=None, refused=None, agent_id=AID, tool_name=tool, success=ok, result_summary="toy",
            error=None if ok else "toy failure", reasoning="", recent_outcomes=[], max_recent=10,
            llm_worker=None, context_pool=_Ctx(), nac=self.nac, embodiment_failed=False,
        )  # fmt: skip

    def pain(self) -> None:
        if self.td:
            self.v += 0.5 * (-1.0 - self.v)
        else:
            self.nac.record_cluster_fear(AID, "w_water", "drive:oxygen", 1.0)

    probe = False


class TabularQ:
    def __init__(self, key: str, seed: int, *, prior: bool = False) -> None:
        self.q: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: -2.0))
        self.key, self.rng = key, random.Random(seed)
        self.acts = [EAT, ESC, FLEE] if prior else TOOLS
        self.pending: tuple | None = None
        self.probe = False

    def _k(self, food: float, oxy: float) -> tuple:
        hb = 0 if food >= 16 else 1 if food >= 11 else 2 if food >= 6 else 3
        ob = 3 if oxy >= 15 else 2 if oxy >= 10 else 1 if oxy >= 5 else 0  # B1's RAW_OXYGEN_BINS
        return {"world": ("w",), "world_hunger": ("w", hb), "raw": ("w", hb, ob)}[self.key]

    def select(self, food: float, oxy: float) -> str:
        s = self._k(food, oxy)
        if not self.probe and self.rng.random() < 0.2:
            a = self.rng.choice(self.acts)
        else:
            best = max(self.q[s][x] for x in self.acts)
            a = self.rng.choice([x for x in self.acts if self.q[s][x] == best])
        self.pending = (s, a)
        return a

    def outcome(self, tool: str, ok: bool, reward: float, food: float, oxy: float, done: bool) -> None:
        assert self.pending is not None
        s, a = self.pending
        target = reward if done else reward + max(self.q[self._k(food, oxy)][x] for x in self.acts)
        self.q[s][a] += 0.5 * (target - self.q[s][a])

    def pain(self) -> None:
        pass


class Random_:
    def __init__(self, seed: int) -> None:
        self.rng, self.probe = random.Random(seed), False

    def select(self, food: float, oxy: float) -> str:
        return self.rng.choice(TOOLS)

    def outcome(self, *_: Any) -> None:
        pass

    def pain(self) -> None:
        pass


def episode(agent: Any, food0: float, *, bread: int, eat_s: float) -> tuple[int, int]:
    """One episode; returns (bread eaten, pain 0/1)."""
    t, food, held, eaten = 0.0, float(food0), bread, 0
    while True:
        if t >= PAIN_T:
            agent.pain()
            return eaten, 1
        a = agent.select(food, oxygen_at(t))
        if a is None:  # Maxim idles when nothing clears min_confidence
            t += STEP
            continue
        dur = {EAT: eat_s, ESC: ESC_S}.get(a, STEP)
        t1 = min(t + dur, PAIN_T)
        cost = -sum((20 - oxygen_at(t + i * STEP)) / 20 for i in range(max(1, int(round((t1 - t) / STEP)))))
        if t + dur > PAIN_T:  # the pain lands inside the blocking call
            agent.outcome(a, False, cost - 2.0, food, oxygen_at(t1), True)
            agent.pain()
            return eaten, 1
        if a == ESC:
            agent.outcome(a, True, cost, food, oxygen_at(t1), True)
            return eaten, 0
        ok = a == EAT and held > 0 and food < 20
        if ok:
            food, held, eaten = min(20.0, food + 5), held - 1, eaten + 1
        succeeded = ok or a.startswith(("read_", "sense_"))
        agent.outcome(a, succeeded, cost + (1.0 if ok else 0.0), food, oxygen_at(t1), False)
        t = t1


ARMS: dict[str, tuple[Callable[[int], Any], bool]] = {
    "MAXIM": (lambda s: Maxim(), False),
    "TWIN": (lambda s: Maxim(td=True), False),
    "Q_EXPLORE": (lambda s: TabularQ("world_hunger", s), True),
    "Q_RAW": (lambda s: TabularQ("raw", s), True),
    "Q_PRIOR": (lambda s: TabularQ("raw", s, prior=True), True),
    "Q_WORLD": (lambda s: TabularQ("world", s), True),
    "RANDOM": (lambda s: Random_(s), True),
}


def run_arm(name: str, *, k: int, seeds: int, bread: int, eat_s: float) -> dict[str, Any]:
    make, stochastic = ARMS[name]
    n = seeds if stochastic else 1
    curve = [[0.0, 0.0] for _ in range(k)]
    probe = {f: [0.0, 0.0] for f in PROBE_FOODS}
    for s in range(n):
        agent = make(s)
        for e in range(k):
            ea, pa = episode(agent, TRAIN_FOODS[e % len(TRAIN_FOODS)], bread=bread, eat_s=eat_s)
            curve[e][0] += ea / n
            curve[e][1] += pa / n
        agent.probe = True
        for f in PROBE_FOODS:
            ea, pa = episode(agent, f, bread=bread, eat_s=eat_s)
            probe[f][0] += ea / n
            probe[f][1] += pa / n
    return {
        "n": n,
        "curve": [{"want": round(c[0], 3), "p_pain": round(c[1], 3)} for c in curve],
        "probe": {str(f): {"want": round(v[0], 3), "p_pain": round(v[1], 3)} for f, v in probe.items()},
    }


def main() -> int:
    out: dict[str, Any] = {"_format_version": "1.0", "kind": "e1_conflict_toy", "main": {}, "sensitivity": []}
    k, seeds = 12, 200
    print(f"main: K={k}, train foods {TRAIN_FOODS}, probe foods {PROBE_FOODS}, bread 2, eat 1.6 s; cells WANT/P(pain)")
    for name in ARMS:
        r = run_arm(name, k=k, seeds=seeds, bread=2, eat_s=1.6)
        out["main"][name] = r
        cells = "  ".join(f"food {f}: {v['want']:.2f}/{v['p_pain']:.2f}" for f, v in r["probe"].items())
        print(f"  {name:10s} probe {cells}")
    m, q = out["main"]["MAXIM"]["probe"], out["main"]["Q_EXPLORE"]["probe"]
    floor_can_lose = any(q[f]["p_pain"] > m[f]["p_pain"] for f in q)
    maxim_can_lose = any(q[f]["want"] > m[f]["want"] for f in q)
    out["floor_can_lose_on_AVOID"], out["maxim_can_lose_on_WANT"] = floor_can_lose, maxim_can_lose
    print(f"floor CAN lose on AVOID: {floor_can_lose};  Maxim CAN lose on WANT: {maxim_can_lose}")
    print("sensitivity (probe at food 8): bread × eat duration")
    for bread in (1, 2, 3, 6):
        for eat_s in (0.8, 1.6):
            row = {"bread": bread, "eat_s": eat_s}
            for name in ("MAXIM", "TWIN", "Q_EXPLORE", "Q_RAW", "Q_PRIOR"):
                row[name] = run_arm(name, k=k, seeds=100, bread=bread, eat_s=eat_s)["probe"]["8"]
            out["sensitivity"].append(row)
            print(
                f"  bread {bread} eat {eat_s}: "
                + "  ".join(
                    f"{n} {row[n]['want']:.2f}/{row[n]['p_pain']:.2f}"
                    for n in ("MAXIM", "TWIN", "Q_EXPLORE", "Q_RAW", "Q_PRIOR")
                )
            )
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
