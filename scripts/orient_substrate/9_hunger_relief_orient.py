#!/usr/bin/env python
"""Exp 52 Phase A — does HUNGER RELIEF teach orienting? (scripted; seconds; no LLM)

Pre-registration: docs/experiments/protocols/exp52_nurture_preregistration.md.
Extends probe 4 (``4_operant_learning_curve.py``) with the one change Exp 52 tests:
the mother's operant credit is no longer a constant ``GAIN`` — it is the SIGN of
the drive relief her feed produced in the infant (``drive_comfort_progress`` over
the drives the feed touched; zero relief → NO credit). The body's hunger drive is
LIVE here (probe 4 froze it at drift 0.0): hunger drifts up every tick, a
contingent feed relieves it, and the credit is whatever that relief was worth.

Arms (all with the same external ε-greedy explorer; hunger drift on unless noted):
  taught   : contingent feed (infant's turn moved TOWARD the sound) → relief-credit.
  satiated : SAME contingency, hunger held at 0 (no drift, initial 0) → the feed
             relieves nothing → NO credit is minted. The discriminating arm: if it
             learns, the credit is not coming from relief.
  yoked    : fed on the SAME tick schedule as the matched taught seed, regardless
             of the infant's own turn → relief-credit lands on whatever action was
             pending (superstition control). Runs on an INDEPENDENT RNG stream
             (seed + YOKED_SEED_OFFSET): with the taught seed's stream it would
             replay the taught trajectory action-for-action and "learn" by
             construction (found in the 2026-08-25 harness dry run).
  no_feed  : never fed (named as in Phase B / Exp 48).

``--credit constant`` reproduces probe 4's by-fiat credit (constant GAIN on every
contingent feed) as the A/B against Exp 46 — the harness change must not move
that curve.

Gates (frozen in the pre-registration, amendment 1; printed and written to --json):
  LEARNED          taught settled (mean of last 4 bins) ≥ 0.80 AND rose ≥ 0.15
                   from the PRE-LEARNING BASELINE = the first FIRST_N ticks pooled
                   across seeds (probe 4's convention — the mechanism learns in
                   ~10 ticks, so "bin 1" is already learned and a bin-based rise
                   is structurally unattainable: S7). Baseline ≥ 0.80 →
                   LEARNED-AT-CEILING (settled ≥ 0.80), reported as such.
  HUNGER-NECESSARY taught settled ≥ satiated settled + 0.20 AND satiated ≤ 0.60.
  MOTHER-NECESSARY taught settled ≥ max(yoked, none) settled + 0.20.
  MECHANISM SANITY (apparatus, not science — any failure voids the run):
                   satiated credits == 0; taught credits == taught feeds;
                   every taught credit has reward +1.

Usage::

    PYTHONPATH=src python scripts/orient_substrate/9_hunger_relief_orient.py --seeds 8 \\
        --json docs/experiments/data/52_phaseA_scripted.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.pop("MAXIM_NAC_REWARD_BIAS_DISABLED", None)

from maxim.decisions.nac import NAc, NACConfig  # noqa: E402
from maxim.embodiment.body import Embodiment  # noqa: E402
from maxim.embodiment.sem import drive_comfort_progress  # noqa: E402
from maxim.embodiment.spec import _parse_entity  # noqa: E402
from maxim.embodiment.tool_bridge import ModulatorAffordanceTool  # noqa: E402
from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402
from maxim.similarity.encoder import SensorEncoder  # noqa: E402

ACTIONS = ["turn_left", "turn_right"]
AGENT = "infant"
MIN_CONF = 0.0
GAIN = 1.0  # probe 4's constant credit (the --credit constant A/B)
HUNGER_DRIFT = 0.05  # per tick, clamped to 1.0 (hungry within a few ticks)
FEED_AMOUNT = 0.5
ARMS = ("taught", "satiated", "yoked", "no_feed")

# Frozen gates (exp52_nurture_preregistration.md §Phase A)
LEARNED_MIN = 0.80
RISE_MIN = 0.15
HUNGER_MARGIN = 0.20
SATIATED_MAX = 0.60
MOTHER_MARGIN = 0.20
SETTLE_BINS = 4
FIRST_N = 5  # pre-learning baseline window (chance by construction), as probe 4
YOKED_SEED_OFFSET = 100_003  # independent RNG stream for the yoked infant


def _body(*, hungry: bool):
    """Infant: azimuth SENSOR with NO drive + a live entropic hunger drive.

    ``hungry=False`` is the satiated arm: hunger starts at 0 and never drifts, so a
    feed relieves nothing.
    """
    d = {
        "name": "infant",
        "entity_type": "body",
        "sensors": {
            "azimuth": {"unit": "normalized", "range": [-1, 1], "initial": 0.0},  # NO drive
            "hunger": {
                "unit": "ratio",
                "range": [0, 1],
                "initial": 0.5 if hungry else 0.0,
                "drive": {
                    "drift_mode": "entropic",
                    "drift_direction": "up",
                    "drift_rate": 0.0,  # ticked manually below (scripted, not wall-clock)
                    "deprivation_threshold": 0.7,
                    "deprivation_pain": 0.3,
                    "satisfaction_threshold": 0.3,
                },
            },
        },
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


def make_encoder():
    ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"interoception", "audio"})))
    return SensorEncoder(ec=ec, atl=None)


def sound_azimuth(rng):
    side = -1.0 if rng.random() < 0.5 else 1.0
    return side * float(rng.uniform(0.3, 0.9))


def feed(body, *, credit: str) -> tuple[float, float | None, float]:
    """The mother feeds: hunger −FEED_AMOUNT (clamped). Returns (relief, reward, hunger_before).

    ``reward`` is what the operant credit will carry: ``credit="relief"`` → the
    SIGN of the relief, or None when |relief| ≤ 1e-9 (nothing to reinforce);
    ``credit="constant"`` → GAIN regardless of state (probe 4's by-fiat credit).
    """
    vm = body.vital_metrics
    spec = body.drive_specs["hunger"]
    before = float(vm["hunger"])
    after = max(0.0, min(1.0, before - FEED_AMOUNT))
    vm["hunger"] = after
    relief = drive_comfort_progress(spec, before, after)
    if credit == "constant":
        return relief, GAIN, before
    if abs(relief) <= 1e-9:
        return relief, None, before
    return relief, (1.0 if relief > 0 else -1.0), before


def run(arm: str, *, seed: int, ticks: int, bin_size: int, epsilon: float, credit: str, feed_schedule=None):
    """One infant's session. Returns (per-bin directedness, raw directed flags, telemetry)."""
    rng = np.random.default_rng(seed + YOKED_SEED_OFFSET if arm == "yoked" else seed)
    nac = NAc(NACConfig())
    enc = make_encoder()
    emb = _body(hungry=(arm != "satiated"))
    body = emb.root
    mod = body.modulators["orient"]
    tools = {a: ModulatorAffordanceTool(body, mod, a, mod.affordances[a], a, embodiment=emb) for a in ACTIONS}
    hungry = arm != "satiated"

    bins: list[list[int]] = []
    cur: list[int] = []
    raw: list[int] = []
    fed_ticks: list[int] = []
    credits: list[float] = []
    reliefs: list[float] = []
    hungers_at_feed: list[float] = []
    for t in range(ticks):
        # hunger drifts up every tick (entropic, direction up) — scripted tick, not wall-clock
        if hungry:
            body.vital_metrics["hunger"] = min(1.0, float(body.vital_metrics["hunger"]) + HUNGER_DRIFT)

        az = sound_azimuth(rng)
        body.vital_metrics["azimuth"] = az
        state = enc.encode_sensors(agent_id=AGENT, sensors={"azimuth": az}, ranges={"azimuth": (-1.0, 1.0)})

        if rng.random() < epsilon:
            action = ACTIONS[int(rng.integers(len(ACTIONS)))]
        else:
            rec = nac.recommend_action(
                agent_id=AGENT,
                available_tools=ACTIONS,
                current_drives=None,
                current_cluster_id=state,
                min_confidence=MIN_CONF,
            )
            action = rec["tool_name"] if rec else ACTIONS[int(rng.integers(len(ACTIONS)))]

        nac.set_pending_operant_action(AGENT, state, f"tool:{action}")
        tools[action].execute()  # azimuth += ±0.3
        az_after = float(body.vital_metrics.get("azimuth", az))
        progress = abs(az) - abs(az_after)
        directed = progress > 1e-9
        cur.append(1 if directed else 0)
        raw.append(1 if directed else 0)

        # Mother: WHEN does she feed?
        if arm in ("taught", "satiated"):
            should_feed = directed
        elif arm == "yoked":
            should_feed = feed_schedule is not None and t in feed_schedule
        else:
            should_feed = False

        if should_feed:
            relief, reward, hunger_before = feed(body, credit=credit)
            fed_ticks.append(t)
            reliefs.append(relief)
            hungers_at_feed.append(hunger_before)
            if reward is not None:
                nac.credit_operant_reward(AGENT, reward)
                credits.append(reward)

        if len(cur) >= bin_size:
            bins.append(cur)
            cur = []
    if cur:
        bins.append(cur)
    telemetry = {
        "fed": len(fed_ticks),
        "credits": len(credits),
        "credit_rewards": sorted(set(credits)),
        "relief_min": min(reliefs) if reliefs else None,
        "relief_max": max(reliefs) if reliefs else None,
        # How hungry the infant was when fed — the honesty payload behind "fed
        # while hungry": the sign-only credit ignores magnitude by design, so
        # HUNGER-NECESSARY discriminates nonzero vs zero relief, not hungry vs sated.
        "hunger_at_feed_min": min(hungers_at_feed) if hungers_at_feed else None,
        "hunger_at_feed_median": float(np.median(hungers_at_feed)) if hungers_at_feed else None,
        "fed_ticks": fed_ticks,
    }
    return [sum(b) / len(b) for b in bins], raw, telemetry


def _provenance_block(out_path: str, allow_dirty: bool) -> dict:
    """Which code ran (scripts/_provenance.py::in_process_code_provenance, by path).

    Refuses (exit 3) when the imported ``maxim`` is not this repo's ``src``, and when
    ``--json`` points under ``docs/experiments/data/`` from a dirty ``src``/``scripts``
    tree unless ``--allow-dirty`` — which stamps ``allow_dirty: true`` here (the report's
    provenance block) so the write-up cannot omit it (roadmap 1.1.x item 16.7).
    """
    import maxim

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts"))
    import _provenance

    try:
        return _provenance.in_process_code_provenance(
            repo, getattr(maxim, "__file__", None), out_path=out_path or None, allow_dirty=allow_dirty
        )
    except _provenance.ProvenanceError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(3) from exc


def main() -> int:
    p = argparse.ArgumentParser(description="Exp 52 Phase A — hunger-relief-taught orienting (scripted)")
    p.add_argument("--ticks", type=int, default=600)
    p.add_argument("--bin", type=int, default=50)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--credit", default="relief", choices=["relief", "constant"])
    p.add_argument("--json", default="", help="write the full report (curves, telemetry, gates, provenance)")
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="write a GATED record (docs/experiments/data/) from a dirty src/scripts tree; stamps allow_dirty: true "
        "into every record (default: refuse, exit 3 — docs/lessons/experiment-prereg-precedes-data.md)",
    )
    args = p.parse_args()

    prov = _provenance_block(args.json, args.allow_dirty)
    seeds = list(range(args.seeds))
    print(
        f"Exp 52 Phase A — credit={args.credit} ticks={args.ticks} bin={args.bin} "
        f"ε={args.epsilon} seeds={args.seeds} hunger_drift={HUNGER_DRIFT}/tick feed={FEED_AMOUNT}\n"
    )
    curves: dict[str, np.ndarray] = {}
    baseline: dict[str, float] = {}
    tel: dict[str, list[dict]] = {}
    schedules: dict[int, list[int]] = {}
    for arm in ARMS:
        results = []
        for s in seeds:
            sched = None
            if arm == "yoked":
                sched = schedules.get(s)
                # The schedule comes from the taught seed; without it the arm would
                # silently degrade to `none` and MOTHER-NECESSARY would pass on a
                # fake control (review fold).
                assert sched is not None, f"yoked seed {s}: taught must run first (ARMS order)"
                sched = set(sched)
            r = run(
                arm,
                seed=s,
                ticks=args.ticks,
                bin_size=args.bin,
                epsilon=args.epsilon,
                credit=args.credit,
                feed_schedule=sched,
            )
            if arm == "taught":
                schedules[s] = r[2]["fed_ticks"]  # yoked seeds inherit the taught seed's feed schedule
            results.append(r)
        per_seed = [r[0] for r in results]
        n_bins = min(len(c) for c in per_seed)
        mat = np.array([c[:n_bins] for c in per_seed])
        curves[arm] = mat.mean(axis=0)
        # pre-learning baseline: the first FIRST_N ticks of every seed, pooled
        baseline[arm] = float(np.mean([x for r in results for x in r[1][:FIRST_N]]))
        tel[arm] = [r[2] for r in results]

    n_bins = len(curves["taught"])
    header = "bin:       " + " ".join(f"{i + 1:>5}" for i in range(n_bins))
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        print(f"{arm:10} " + " ".join(f"{v:5.2f}" for v in curves[arm]))

    settled = {arm: float(np.mean(curves[arm][-SETTLE_BINS:])) for arm in ARMS}
    print("\nverdict:")
    print(f"  first-{FIRST_N}-ticks baseline: taught={baseline['taught']:.2f} (chance ≈ 0.50)")
    print("  settled:   " + "  ".join(f"{a}={settled[a]:.2f}" for a in ARMS))

    at_ceiling = baseline["taught"] >= LEARNED_MIN
    if at_ceiling:
        learned = settled["taught"] >= LEARNED_MIN
    else:
        learned = settled["taught"] >= LEARNED_MIN and (settled["taught"] - baseline["taught"]) >= RISE_MIN
    hunger_nec = (settled["taught"] - settled["satiated"]) >= HUNGER_MARGIN and settled["satiated"] <= SATIATED_MAX
    mother_nec = (settled["taught"] - max(settled["yoked"], settled["no_feed"])) >= MOTHER_MARGIN
    # Mechanism sanity (apparatus): what the credit path DID, independent of behaviour.
    sat_credits = sum(x["credits"] for x in tel["satiated"])
    taught_fed = sum(x["fed"] for x in tel["taught"])
    taught_credits = sum(x["credits"] for x in tel["taught"])
    taught_rewards = sorted({r for x in tel["taught"] for r in x["credit_rewards"]})
    # S5: the yoked arm must receive exactly the taught seed's feeds.
    yoked_rate_matched = all(y["fed"] == t["fed"] for y, t in zip(tel["yoked"], tel["taught"]))
    if args.credit == "relief":
        sanity = (
            taught_fed > 0
            and sat_credits == 0
            and taught_credits == taught_fed
            and taught_rewards == [1.0]
            and yoked_rate_matched
        )
    else:
        sanity = taught_fed > 0 and taught_credits == taught_fed and taught_rewards == [GAIN] and yoked_rate_matched
    if at_ceiling:
        print(
            f"  LEARNED-AT-CEILING (baseline {baseline['taught']:.2f} ≥ {LEARNED_MIN}; settled ≥ {LEARNED_MIN}): "
            f"{'PASS' if learned else 'FAIL'}"
        )
    else:
        print(
            f"  LEARNED (taught settled ≥ {LEARNED_MIN} and rose ≥ {RISE_MIN} from the first-{FIRST_N}-ticks baseline): "
            f"{'PASS' if learned else 'FAIL'}"
        )
    print(
        f"  HUNGER-NECESSARY (taught ≥ satiated + {HUNGER_MARGIN}, satiated ≤ {SATIATED_MAX}): "
        f"{'PASS' if hunger_nec else 'FAIL'}"
    )
    print(f"  MOTHER-NECESSARY (taught ≥ max(yoked, no_feed) + {MOTHER_MARGIN}): {'PASS' if mother_nec else 'FAIL'}")
    _h = [x["hunger_at_feed_median"] for x in tel["taught"] if x["hunger_at_feed_median"] is not None]
    if _h:
        print(
            f"  hunger at feed (taught, median of per-seed medians): {float(np.median(_h)):.3f} — sign-only credit; see pre-reg §Arms"
        )
    print(
        f"  MECHANISM SANITY (satiated credits={sat_credits}; taught fed={taught_fed} credits={taught_credits} "
        f"rewards={taught_rewards}; yoked feeds == taught feeds: {yoked_rate_matched}): "
        f"{'OK' if sanity else 'VOID — apparatus, not a result'}"
    )
    if not sanity:
        print("\n**VOID — the credit path did not behave as declared; fix the apparatus before reading the curves.**")
    elif learned and hunger_nec and mother_nec:
        print(
            "\n**PASS — the infant learned to orient from hunger-relieved feeding alone; a feed without need taught nothing.**"
        )
    elif learned:
        print("\n**PARTIAL — learned, but a control gate failed (see above).**")
    else:
        print("\n**FAIL — LEARNED did not clear; per the pre-registration Phase B does not run.**")

    if args.json:
        report = {
            "experiment": "exp52_phaseA_scripted",
            "credit": args.credit,
            "params": {
                "ticks": args.ticks,
                "bin": args.bin,
                "seeds": args.seeds,
                "epsilon": args.epsilon,
                "hunger_drift": HUNGER_DRIFT,
                "feed_amount": FEED_AMOUNT,
            },
            "curves": {a: [float(x) for x in curves[a]] for a in ARMS},
            "baseline_first_ticks": baseline,
            "baseline_n_ticks": FIRST_N,
            "at_ceiling": at_ceiling,
            "settled": settled,
            "telemetry": {a: [{k: v for k, v in x.items() if k != "fed_ticks"} for x in tel[a]] for a in ARMS},
            "gates": {
                "LEARNED": learned,
                "HUNGER_NECESSARY": hunger_nec,
                "MOTHER_NECESSARY": mother_nec,
                "MECHANISM_SANITY": sanity,
            },
            "provenance": prov,
        }
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nreport written: {args.json}")
    return 0 if sanity else 4


if __name__ == "__main__":
    sys.exit(main())
