#!/usr/bin/env python3
"""Encoding bake-off — which sensor-encoding configuration survives a big body?

L11 (`docs/limits/README.md`, tracking doc `docs/limits/l11_sensor_dilution.md`)
records that the shipped encoding dilutes as ``cos ~ 1 - 0.57/N`` and that
DISCRIMINATION — telling *which* sensor moved — is the ceiling behind it. Three
candidate mitigations were measured synthetically on 2026-09-01. This harness
runs them head to head, against the REAL ``EntorhinalCortex`` rather than a model
of it, so the comparison measures the shipped mechanism.

**The arms are not three mutually exclusive options.** The scaled threshold and
per-type grouping are COMPLEMENTARY (grouping alone does not clear the fixed
0.85 bar; the threshold alone leaves two sensors in one channel confusable), and
the nonlinear gain is an alternative ENCODING that composes with either. So the
honest arm set is six, including the current encoding as a control — without the
control, "best" is unanchored.

**The metric is frozen here, before any run**, per the house rule that gates are
chosen before data:

  separation     fraction of meaningful single-sensor excursions that allocate a
                 NEW cluster                                        (want ~1.0)
  stability      fraction of small all-sensor jitters that COMPLETE onto the
                 same cluster                                       (want ~1.0)
  discrimination fraction of distinct-sensor excursion PAIRS landing in
                 DIFFERENT clusters                                 (want ~1.0)
  economy        clusters allocated per 100 presented states     (a COST, want low)

  PRIMARY = min(separation, stability, discrimination)  — the weakest link.

The min is deliberate: a configuration can trivially max any one of the first
three by sacrificing another (a threshold of 1.0 separates everything and is
useless; a threshold of 0 completes everything and is useless). Economy is
reported alongside as a cost, never folded into the primary — more clusters is a
latency problem against an unbounded O(N_nodes x d) EC scan (D51), not a quality
problem, and mixing them would hide the trade.

**Scope, stated so the output is not over-read.** The bodies are SYNTHETIC at
counts a Minecraft-scale body would reach, because no shipped body exceeds ~12
sensors — the regime of interest does not exist yet. Real drives correlate
(hunger and fatigue drift together) and these do not; that is L11 open question
2 and this harness does not answer it. Treat the result as "which candidate is
worth building", not "which candidate is validated".

Usage
-----
    python scripts/encoding_bakeoff.py --sensors 50 --trials 200
    python scripts/encoding_bakeoff.py --sensors 12,30,50,100 --json out.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402
from maxim.similarity.encoder import _stable_basis  # noqa: E402

DIM = 384

# The frozen arm set. `gain` and `group` are encoding/structure choices;
# `k` is None for a fixed threshold or a float for `1 - k/N`.
ARMS: tuple[dict[str, Any], ...] = (
    {"name": "A0 current (control)", "k": None, "groups": 1, "gain": False},
    {"name": "A1 scaled threshold", "k": 0.30, "groups": 1, "gain": False},
    {"name": "A2 grouping only", "k": None, "groups": 5, "gain": False},
    {"name": "A3 threshold + grouping", "k": 0.30, "groups": 5, "gain": False},
    {"name": "A4 nonlinear gain", "k": None, "groups": 1, "gain": True},
    {"name": "A5 gain + threshold", "k": 0.30, "groups": 1, "gain": True},
)

FIXED_THRESHOLD = 0.85  # SensorEncoderConfig.pattern_threshold, the shipped value
GAIN_EXPONENT = 3.0


def _embed(sensors: dict[str, float], *, gain: bool) -> list[float]:
    """Mirror `similarity/encoder.py::_sensor_embed`, optionally gain-weighted.

    Kept local rather than imported so the gain variant is a one-line diff from
    the shipped formula and the difference is auditable here.
    """
    out = [0.0] * DIM
    for name, value in sorted(sensors.items()):
        lo = _stable_basis(f"{name}:low", DIM)
        hi = _stable_basis(f"{name}:high", DIM)
        w = ((abs(value - 0.5) * 2.0) ** GAIN_EXPONENT) if gain else 1.0
        for i in range(DIM):
            out[i] += w * ((1.0 - value) * lo[i] + value * hi[i])
    return out


def _partition(names: list[str], groups: int) -> list[list[str]]:
    """Split sensors into `groups` channels — the per-type-modality stand-in."""
    if groups <= 1:
        return [names]
    return [names[i::groups] for i in range(groups)]


def _cluster_ids(
    ec_by_channel: dict[int, EntorhinalCortex],
    state: dict[str, float],
    channels: list[list[str]],
    *,
    gain: bool,
    threshold: float,
) -> tuple[str, ...]:
    """Present a state and return the cluster id per channel — the real EC path."""
    ids: list[str] = []
    for idx, members in enumerate(channels):
        sub = {n: state[n] for n in members}
        vec = _embed(sub, gain=gain)
        ec = ec_by_channel[idx]
        res = ec.pattern_complete_or_separate(vec, modality=f"ch{idx}", threshold=threshold)
        if res.is_new:
            # EC's separation path deliberately allocates an id WITHOUT
            # registering it — its own comment: "the caller (LinguisticEncoder)
            # registers via register_substrate_node after ATL activation
            # succeeds. This keeps EC stateless for the separation path."
            # A harness that skips this half measures an EC that never
            # remembers anything: every state looks new, stability reads 0.00
            # and the node store stays empty. (It did, on the first run here.)
            ec.register_substrate_node(res.node_id, vec, f"ch{idx}")
        ids.append(str(res.node_id))
    return tuple(ids)


def run_arm(arm: dict[str, Any], n_sensors: int, trials: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    names = [f"s{i}" for i in range(n_sensors)]
    channels = _partition(names, int(arm["groups"]))
    per_channel_n = max(1, n_sensors // max(1, int(arm["groups"])))
    k = arm["k"]
    threshold = FIXED_THRESHOLD if k is None else (1.0 - float(k) / per_channel_n)

    frozen = frozenset(f"ch{j}" for j in range(len(channels)))
    ecs: dict[int, EntorhinalCortex] = {
        i: EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozen)) for i in range(len(channels))
    }

    sep = stab = disc = 0
    presented = 0
    all_ids: set[tuple[str, ...]] = set()

    for _ in range(trials):
        rest = {n: rng.uniform(0.30, 0.70) for n in names}
        base = _cluster_ids(ecs, rest, channels, gain=arm["gain"], threshold=threshold)
        all_ids.add(base)
        presented += 1

        # separation: a decisive single-sensor excursion must make a NEW cluster
        a_name = rng.choice(names)
        exc_a = dict(rest)
        exc_a[a_name] = rng.choice([0.0, 1.0])
        ids_a = _cluster_ids(ecs, exc_a, channels, gain=arm["gain"], threshold=threshold)
        all_ids.add(ids_a)
        presented += 1
        if ids_a != base:
            sep += 1

        # stability: a 2% all-sensor jitter must COMPLETE onto the same cluster
        noisy = {n: min(1.0, max(0.0, v + rng.gauss(0, 0.02))) for n, v in rest.items()}
        ids_n = _cluster_ids(ecs, noisy, channels, gain=arm["gain"], threshold=threshold)
        all_ids.add(ids_n)
        presented += 1
        if ids_n == base:
            stab += 1

        # discrimination: a DIFFERENT sensor excursion must land elsewhere
        others = [n for n in names if n != a_name]
        if others:
            b_name = rng.choice(others)
            exc_b = dict(rest)
            exc_b[b_name] = rng.choice([0.0, 1.0])
            ids_b = _cluster_ids(ecs, exc_b, channels, gain=arm["gain"], threshold=threshold)
            all_ids.add(ids_b)
            presented += 1
            if ids_b != ids_a:
                disc += 1

    separation = sep / trials
    stability = stab / trials
    discrimination = disc / trials
    total_nodes = sum(len(getattr(ec, "_substrate_nodes", {}) or {}) for ec in ecs.values())
    return {
        "arm": arm["name"],
        "n_sensors": n_sensors,
        "channels": len(channels),
        "per_channel_n": per_channel_n,
        "threshold": round(threshold, 5),
        "separation": round(separation, 4),
        "stability": round(stability, 4),
        "discrimination": round(discrimination, 4),
        "primary_min": round(min(separation, stability, discrimination), 4),
        "clusters_per_100_states": round(100.0 * total_nodes / max(1, presented), 2),
        "distinct_cluster_tuples": len(all_ids),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensors", default="12,30,50,100", help="comma-separated sensor counts")
    p.add_argument("--trials", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", default="", help="write full results here (not a gated path)")
    args = p.parse_args(argv)

    counts = [int(x) for x in args.sensors.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for n in counts:
        print(f"\n=== {n} sensors, {args.trials} trials ===")
        print(f"{'arm':<26}{'thr':>8}{'sep':>7}{'stab':>7}{'disc':>7}{'PRIMARY':>9}{'clu/100':>9}")
        print("-" * 73)
        for arm in ARMS:
            r = run_arm(arm, n, args.trials, args.seed)
            rows.append(r)
            print(
                f"{r['arm']:<26}{r['threshold']:>8.4f}{r['separation']:>7.2f}"
                f"{r['stability']:>7.2f}{r['discrimination']:>7.2f}"
                f"{r['primary_min']:>9.2f}{r['clusters_per_100_states']:>9.1f}"
            )
        best = max((r for r in rows if r["n_sensors"] == n), key=lambda r: r["primary_min"])
        print(f"  -> best by PRIMARY at N={n}: {best['arm']} ({best['primary_min']:.2f})")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"metric": "min(separation, stability, discrimination)", "rows": rows}, indent=2))
        print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
