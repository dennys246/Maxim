#!/usr/bin/env python
"""Reproduce the tool-competition failure — through the REAL record_outcome.

The embodied cradle_mother sim measured at chance because the infant chose
``sense_presence`` (a tool that "always succeeds") over turning: substrate-primary
credits tool EXECUTION success as if it were goal/drive progress, so a
mechanically-successful distractor snowballs causal + cluster credit and drowns
the specific operant orient signal. See
docs/plans/deferred/credit_on_progress_not_execution.md.

This reproduces that in a CLEAN scripted setting (no LLM, deterministic) using the
REAL ``runtime/tool_dispatch.record_outcome`` + ``NAc.recommend_action`` — proving
the failure is in the production credit path, not an artifact of the sim's
narrator. An infant learns to orient (mother feeds it via
``credit_operant_reward`` when it turns toward the sound); we run it WITH and
WITHOUT an always-succeed ``sense`` distractor in the tool set.

Read: ``no_distractor`` → the infant orients (chooses a turn, correctly).
``with_distractor`` → ``sense`` snowballs and the infant stops orienting (orient
rate collapses) — the embodied failure, reproduced. This is the BEFORE; the
credit-on-progress fix should make ``with_distractor`` match ``no_distractor``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)
# Reproduce the DEFAULT sim credit behaviour — no operant-only mode, no whitelist.
os.environ.pop("MAXIM_OPERANT_ONLY_CREDIT", None)
os.environ.pop("MAXIM_SUBSTRATE_TOOL_WHITELIST", None)

from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.embodiment.body import Embodiment  # noqa: E402
from maxim.embodiment.spec import _parse_entity  # noqa: E402
from maxim.embodiment.tool_bridge import ModulatorAffordanceTool  # noqa: E402
from maxim.runtime.tool_dispatch import record_outcome  # noqa: E402
from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402
from maxim.similarity.encoder import SensorEncoder  # noqa: E402

TURNS = ["turn_left", "turn_right"]
SENSE = "sense"  # always-succeeds distractor (no drive effect) — the sense_presence stand-in
AGENT = "infant"
EPSILON = 0.2
ORIENTED = 0.15
GAIN = 1.0


class _NullPool:
    def add_outcome(self, **kwargs):
        pass


def _body():
    d = {
        "name": "infant",
        "entity_type": "body",
        "sensors": {"azimuth": {"unit": "normalized", "range": [-1, 1], "initial": 0.0}},
        "modulators": {
            "orient": {
                "abstract": True,
                "affordances": {
                    "turn_left": {"params": {}, "description": "l", "self_effect": {"azimuth": 0.3}},
                    "turn_right": {"params": {}, "description": "r", "self_effect": {"azimuth": -0.3}},
                },
            }
        },
    }
    return Embodiment(root=_parse_entity(d))


def sound_azimuth(rng):
    return (-1.0 if rng.random() < 0.5 else 1.0) * float(rng.uniform(0.3, 0.9))


def run(*, with_distractor, seed, ticks=1500):
    rng = np.random.default_rng(seed)
    nac = NAc(NACConfig())
    enc = SensorEncoder(
        ec=EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception", "audio"}))), atl=None
    )
    emb = _body()
    body = emb.root
    mod = body.modulators["orient"]
    tools = {a: ModulatorAffordanceTool(body, mod, a, mod.affordances[a], a, embodiment=emb) for a in TURNS}
    pool = _NullPool()
    recent: list = []
    available = TURNS + ([SENSE] if with_distractor else [])

    for _ in range(ticks):
        az = sound_azimuth(rng)
        body.vital_metrics["azimuth"] = az
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az}, ranges={"azimuth": (-1.0, 1.0)})
        if rng.random() < EPSILON:
            action = available[int(rng.integers(len(available)))]
        else:
            rec = nac.recommend_action(
                agent_id=AGENT,
                available_tools=available,
                current_drives=None,
                current_cluster_id=state,
                min_confidence=0.0,
            )
            action = rec["tool_name"] if rec else available[int(rng.integers(len(available)))]

        if action == SENSE:
            # Always succeeds, no drive effect — the tool-execution-success pollution.
            record_outcome(
                agent_id=AGENT,
                tool_name=SENSE,
                success=True,
                result_summary="ok",
                error=None,
                reasoning="",
                recent_outcomes=recent,
                max_recent=20,
                llm_worker=None,
                context_pool=pool,
                nac=nac,
                cluster_id=state,
                embodiment_failed=False,
                drive_potential_diff=None,
            )
            continue

        # a turn: execute, then the mother feeds if it oriented (operant credit)
        tools[action].execute()
        az_after = float(body.vital_metrics.get("azimuth", az))
        progress = abs(az) - abs(az_after)
        record_outcome(
            agent_id=AGENT,
            tool_name=action,
            success=True,
            result_summary="ok",
            error=None,
            reasoning="",
            recent_outcomes=recent,
            max_recent=20,
            llm_worker=None,
            context_pool=pool,
            nac=nac,
            cluster_id=state,
            embodiment_failed=False,
            drive_potential_diff=progress if abs(progress) > 1e-9 else None,
        )
        if abs(az_after) <= ORIENTED:
            nac.credit_operant_reward(AGENT, GAIN)

    # ---- eval: greedy — what does the infant DO, and does it orient correctly? --
    er = np.random.default_rng(seed + 10_000)
    n_orient = n_correct = total = 0
    for _ in range(400):
        az = sound_azimuth(er)
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az}, ranges={"azimuth": (-1.0, 1.0)})
        rec = nac.recommend_action(
            agent_id=AGENT, available_tools=available, current_drives=None, current_cluster_id=state, min_confidence=0.0
        )
        action = rec["tool_name"] if rec else available[int(er.integers(len(available)))]
        total += 1
        if action in TURNS:
            n_orient += 1
            want = "turn_left" if az < 0 else "turn_right"
            n_correct += action == want
    return {
        "orient_rate": n_orient / total,  # fraction of the time it chose to TURN (not sense)
        "directedness": n_correct / n_orient if n_orient else 0.0,  # of turns, fraction correct
    }


def main():
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    print("tool-competition reproduction — orient tool vs an always-succeed `sense` distractor")
    print("(real record_outcome + recommend_action, no LLM)\n")
    print(f"{'arm':16} {'orient_rate':>12} {'directedness':>13}")
    print("-" * 44)
    res = {}
    for arm, wd in (("no_distractor", False), ("with_distractor", True)):
        rr = [run(with_distractor=wd, seed=s) for s in seeds]
        orr = float(np.mean([r["orient_rate"] for r in rr]))
        dr = float(np.mean([r["directedness"] for r in rr]))
        res[arm] = (orr, dr)
        print(f"{arm:16} {orr:12.2f} {dr:13.2f}")

    print("\nverdict:")
    nd_or, nd_dir = res["no_distractor"]
    wd_or, wd_dir = res["with_distractor"]
    reproduced = nd_or >= 0.8 and nd_dir >= 0.8 and (nd_or - wd_or) >= 0.3
    print(f"  no_distractor: orients {nd_or:.2f} @ {nd_dir:.2f} directed;  with_distractor: orients {wd_or:.2f}")
    print(f"  FAILURE REPRODUCED (distractor suppresses orienting): {'YES' if reproduced else 'NO'}")
    if reproduced:
        print("\n  → the sense distractor snowballs tool-success credit and the infant stops orienting,")
        print("    exactly as the embodied sim. Next: the credit-on-progress fix should close the gap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
