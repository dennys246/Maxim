"""B1 learner toy model: what the frozen tabular learner (scripts/survival_world/floor_policies.py) does on a water
episode with live-like timings, before any rig time (owner decision OD5, 2026-10-11).

Not data and not a gate: an offline model that motivated the v4 learner constants (γ = 1, Q₀ = −2). It drives the
REAL ``TabularQPolicy`` (only γ and Q₀ vary) through Exp 60's schedule — 6 greedy pre-probe placements (cap 4.3 s,
truncated), K training episodes (ε = 0.2; the saturating pain at ≈ 6 s is terminal −1 plus its own intensity −1),
then the greedy first post placement, which succeeds only if ``escape_water`` is picked early enough to surface
before the 4.3 s cap. Assumed timings (from Exp 60's records and the bridge): selects every 0.5 s in water; every
non-escape tool fails or returns in < 0.5 s with ``params={}``; ``escape_water`` blocks ≈ 2.6 s until the head
clears; oxygen 20 → 12 over 6 s. One water key (the live EC may split it, which only slows learning — see the row's
key-coverage diagnostic). Run:

    PYTHONPATH="$PWD/src" python docs/experiments/rationale/b1-exp60-floor/learner_sim.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from survival_world import floor_policies as FP  # noqa: E402

N_TOOLS, ESC, ESC_DUR, DRAIN, PAIN_T, PROBE_CAP, SELECT_S = 26, "t_escape_water", 2.6, 8 / 6, 6.0, 4.3, 0.5
TOOLS = [ESC, "t_flee"] + [f"t_other{i}" for i in range(N_TOOLS - 2)]


def first_post_success(seed: int, arm: str, *, k: int, gamma: float, q_init: float) -> bool:
    clock = {"t": 0.0}
    pain: list[float] = []
    FP.Q_INIT = q_init  # the module constant `_row` reads (restored by the caller)
    pol = FP.TabularQPolicy(
        seed=seed,
        key_mode="raw" if arm == "raw" else "clusters",
        prior_only=arm == "prior",
        was_executed=lambda a, s: True,
        pain_since=lambda: sum(pain.pop() for _ in range(len(pain))),
        clock=lambda: clock["t"],
        gamma=gamma,
    )

    def episode(train: bool) -> bool:
        pol.clear_pending()
        t, base = 0.0, clock["t"]

        def end(at: float, reason: str) -> None:
            clock["t"] = base + at
            if reason == "pain":
                pain.append(FP.SATURATING_PAIN)
            pol.end_episode(reason)

        while True:
            if train and t >= PAIN_T:
                return end(PAIN_T, "pain") or False
            if not train and t >= PROBE_CAP:
                return end(PROBE_CAP, "cap") or False
            clock["t"] = base + t
            obs = {"is_in_water": 1.0, "oxygen": 20 - DRAIN * t}
            pick = pol.select(agent_id="x", available_tools=TOOLS, clusters={"world": "water"}, observation=obs)
            if pick["tool_name"] == ESC:
                done = t + ESC_DUR
                if train and done >= PAIN_T:
                    return end(PAIN_T, "pain") or False
                if not train and done > PROBE_CAP:
                    return end(PROBE_CAP, "cap") or False
                end(done, "surfaced")
                return True
            t += SELECT_S

    pol.set_probe_mode(True)
    for _ in range(6):
        episode(False)
    pol.set_probe_mode(False)
    for _ in range(k):
        episode(True)
    pol.set_probe_mode(True)
    return episode(False)


def table(n: int = 400) -> list[tuple[str, dict[str, float]]]:
    saved = FP.Q_INIT
    rows = []
    try:
        for label, kw in [
            ("v3 frozen: γ=0.9, Q₀=0, K=10", {"k": 10, "gamma": 0.9, "q_init": 0.0}),
            ("v3, K=30", {"k": 30, "gamma": 0.9, "q_init": 0.0}),
            ("γ=1, Q₀=0, K=10", {"k": 10, "gamma": 1.0, "q_init": 0.0}),
            ("γ=0.9, Q₀=−2, K=10", {"k": 10, "gamma": 0.9, "q_init": -2.0}),
            ("v4: γ=1, Q₀=−2, K=10", {"k": 10, "gamma": 1.0, "q_init": -2.0}),
        ]:
            rows.append(
                (label, {a: sum(first_post_success(s, a, **kw) for s in range(n)) / n for a in ("explore", "prior", "raw")})
            )
    finally:
        FP.Q_INIT = saved
    rng = random.Random(0)
    draws = int((PROBE_CAP - ESC_DUR) / SELECT_S) + 1
    rows.append(("RANDOM (uniform draws)", {"any": sum(any(rng.choice(TOOLS) == ESC for _ in range(draws)) for _ in range(20000)) / 20000}))
    return rows


if __name__ == "__main__":
    for label, rates in table():
        print(f"{label:32s} " + "  ".join(f"{a}={v:.2f}" for a, v in rates.items()))
