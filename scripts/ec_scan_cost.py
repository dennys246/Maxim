#!/usr/bin/env python3
"""EC substrate-node scan cost at A4's allocation rate — the 1.1.4 prerequisite measurement.

The A4 nonlinear gain (the 1.1.4 encoding change, selected by the 2026-09-01
bake-off — `docs/limits/l11_sensor_dilution.md` §Bake-off) allocates ~120x the
control's clusters, and `similarity/ec.py::pattern_complete_or_separate` is an
exact O(N_nodes x d) Python loop with no cap, no pruning and no index — the
degenerate `LSHIndex` (bugs ledger D51) is NOT on this path, so fixing it would
not help. Production shares ONE EC across all modality channels and the scan
iterates the WHOLE store, filtering modality in Python, so per-encode cost
scales with the TOTAL node count across channels. This harness measures that
cost against the REAL `EntorhinalCortex` before A4 ships, per the plan
(`docs/plans/world_seam_1_1_4.md` PR 0 / decision D4).

**Metric, horizon and decision rule are frozen here, before any run**, per the
house rule that gates are chosen before data.

Measurements
------------
(a) ORGANIC growth: one shared EC; per simulated tick, one A4-gained encode on
    a `world` channel (N sensors, default 50) and one on an `interoception`
    channel (6 sensors) via the real complete-or-separate + register protocol.
    States follow the bake-off's presentation cycle (rest / single-sensor
    excursion / 2% jitter / second excursion), which reproduces A4's measured
    allocation regime. Per-call wall time recorded; median/p95 over a sliding
    window reported at store-size checkpoints. The audio channel is omitted
    (place-code + gain composition is plan decision D3, unresolved); its store
    contribution is small at control allocation rates and its omission makes
    this an UNDER-estimate of store size, noted in the output.
(b) CONTROLLED scan timing: fresh EC pre-loaded to fixed store sizes
    {100, 1k, 2k, 5k, 10k, 20k} (60% world / 30% interoception / 10% audio),
    50 timed world-channel probes each; linear fit gives the per-node
    coefficient used for the horizon projection.

Horizon (frozen): a 4-hour session at 2 ticks/s = 28,800 ticks; projected
store = organic allocation rate (measured in (a), per channel, per presented
state) x 28,800 per channel. Ignoring the production min_delta gate and using
the bake-off's excursion-heavy state cycle both push the store UP — this is
deliberately an upper-bound horizon; the controlled curve makes the projection
exact for whatever store a real session reaches.

Decision rule (frozen; plan decision D4)
----------------------------------------
Let P95(S) be the controlled-curve p95 per-encode scan cost at store size S,
and S_hz the projected horizon store.

1. P95(S_hz) <= 5 ms          -> A4 ships bare; the number is recorded.
2. P95(S_hz) >  5 ms          -> a store bound is a prerequisite shipping in
   the SAME PR as A4. The cap N_cap = the largest store size with
   P95 <= 5 ms, read off the controlled curve.
3. N_cap < 1,000 nodes        -> a cap would discard within-session
   representation (1,000 =~ the allocation of a single ~30-min A4 session);
   an index replacing the exact scan becomes the prerequisite and 1.1.4
   re-sequences.

The 5 ms bar: substrate-primary ticks at ~2 Hz; 3 channels x 5 ms = 15 ms/tick
=~ 3% of the tick budget spent scanning, the most we will spend on a lookup.

Scope, stated so the output is not over-read: synthetic uncorrelated sensors
(L11 open question 2 unchanged); cost measurement ONLY — separation/stability/
discrimination were the bake-off's job, not this harness's.

(c) EXPLORATORY remedy sizing — outside the frozen verdict, labeled as such in
    the output: a numpy-vectorized EXACT scan (same cosine, same store, one
    matrix-vector product) timed at the same store sizes, to size the cheapest
    semantics-preserving remedy against the ANN-index alternative if the
    verdict is not ship-bare. It informs the REMEDY choice; it cannot change
    the verdict, whose rule is frozen above.

Usage
-----
    python scripts/ec_scan_cost.py --json docs/experiments/data/ec_scan_cost_2026-09-03.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402
from maxim.similarity.encoder import _stable_basis  # noqa: E402

DIM = 384
GAIN_EXPONENT = 3.0  # the bake-off's A4 value, unswept (L11 open question 1a)
FIXED_THRESHOLD = 0.85  # SensorEncoderConfig.pattern_threshold, the shipped value

TICK_HZ = 2.0
SESSION_HOURS = 4.0
HORIZON_TICKS = int(TICK_HZ * SESSION_HOURS * 3600)  # 28,800

P95_BAR_MS = 5.0
CAPACITY_FLOOR_NODES = 1000

CONTROLLED_SIZES = (100, 1_000, 2_000, 5_000, 10_000, 20_000)
CONTROLLED_PROBES = 50
CONTROLLED_MIX = (("world", 0.60), ("interoception", 0.30), ("audio", 0.10))


def _embed_a4(sensors: dict[str, float]) -> list[float]:
    """The bake-off's A4 arm, verbatim (encoding_bakeoff.py::_embed, gain=True)."""
    out = [0.0] * DIM
    for name, value in sorted(sensors.items()):
        lo = _stable_basis(f"{name}:low", DIM)
        hi = _stable_basis(f"{name}:high", DIM)
        w = (abs(value - 0.5) * 2.0) ** GAIN_EXPONENT
        for i in range(DIM):
            out[i] += w * ((1.0 - value) * lo[i] + value * hi[i])
    return out


def _timed_complete(ec: EntorhinalCortex, vec: list[float], modality: str) -> tuple[float, Any]:
    t0 = time.perf_counter()
    res = ec.pattern_complete_or_separate(
        vec,
        modality=modality,
        threshold=FIXED_THRESHOLD,
        # None = explicit geometry opt-out: this harness measures SCAN COST,
        # not cross-space isolation, and every node it stores is same-space.
        geometry=None,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if res.is_new:
        # The caller registers — EC's separation path is deliberately stateless
        # (same protocol note as encoding_bakeoff.py::_cluster_ids).
        ec.register_substrate_node(res.node_id, vec, modality)
    return elapsed_ms, res


def _bakeoff_states(rng: random.Random, names: list[str]) -> list[dict[str, float]]:
    """One bake-off presentation cycle: rest, excursion A, jitter, excursion B."""
    rest = {n: rng.uniform(0.30, 0.70) for n in names}
    a_name = rng.choice(names)
    exc_a = dict(rest)
    exc_a[a_name] = rng.choice([0.0, 1.0])
    noisy = {n: min(1.0, max(0.0, v + rng.gauss(0, 0.02))) for n, v in rest.items()}
    states = [rest, exc_a, noisy]
    others = [n for n in names if n != a_name]
    if others:
        exc_b = dict(rest)
        exc_b[rng.choice(others)] = rng.choice([0.0, 1.0])
        states.append(exc_b)
    return states


def run_organic(n_world: int, world_states: int, seed: int, checkpoint_every: int) -> dict[str, Any]:
    rng = random.Random(seed)
    world_names = [f"w{i}" for i in range(n_world)]
    intero_names = [f"d{i}" for i in range(6)]
    ec = EntorhinalCortex(ECConfig())  # default frozen set {"interoception","audio"}; world updates centroids

    latencies: list[float] = []
    checkpoints: list[dict[str, Any]] = []
    presented = {"world": 0, "interoception": 0}

    while presented["world"] < world_states:
        for w_state, i_state in zip(_bakeoff_states(rng, world_names), _bakeoff_states(rng, intero_names)):
            ms_w, _ = _timed_complete(ec, _embed_a4(w_state), "world")
            latencies.append(ms_w)
            presented["world"] += 1
            ms_i, _ = _timed_complete(ec, _embed_a4(i_state), "interoception")
            presented["interoception"] += 1
            del ms_i  # interoception cost is the same scan; world latencies carry the metric

            if presented["world"] % checkpoint_every == 0:
                window = latencies[-checkpoint_every:]
                checkpoints.append(
                    {
                        "presented_world_states": presented["world"],
                        "total_nodes": len(getattr(ec, "_substrate_nodes", {}) or {}),
                        "world_p50_ms": round(statistics.median(window), 3),
                        "world_p95_ms": round(_p95(window), 3),
                    }
                )
                print(
                    f"  organic: {presented['world']:>6} world states, "
                    f"{checkpoints[-1]['total_nodes']:>6} nodes, "
                    f"p50 {checkpoints[-1]['world_p50_ms']:>8.3f} ms, "
                    f"p95 {checkpoints[-1]['world_p95_ms']:>8.3f} ms"
                )

    total_nodes = len(getattr(ec, "_substrate_nodes", {}) or {})
    return {
        "n_world_sensors": n_world,
        "world_states_presented": presented["world"],
        "interoception_states_presented": presented["interoception"],
        "total_nodes": total_nodes,
        "nodes_per_100_presented_states": round(
            100.0 * total_nodes / (presented["world"] + presented["interoception"]), 2
        ),
        "checkpoints": checkpoints,
    }


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[idx]


def run_controlled(n_world: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 1)
    world_names = [f"w{i}" for i in range(n_world)]
    rows: list[dict[str, Any]] = []
    ec = EntorhinalCortex(ECConfig())
    registered = 0
    for size in CONTROLLED_SIZES:
        while registered < size:
            # Random unit-scale vectors are fine here: cost depends on store
            # size and dim, not on content; the modality mix mirrors a shared
            # production EC so the Python-side modality filter is exercised.
            r = registered / max(1, size)
            modality = next(m for m, cut in _mix_cuts() if r < cut)
            vec = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
            ec.register_substrate_node(str(uuid.uuid4()), vec, modality)
            registered += 1
        probe_ms: list[float] = []
        for _ in range(CONTROLLED_PROBES):
            state = {n: rng.uniform(0.0, 1.0) for n in world_names}
            vec = _embed_a4(state)
            t0 = time.perf_counter()
            ec.pattern_complete_or_separate(vec, modality="world", threshold=FIXED_THRESHOLD, geometry=None)
            probe_ms.append((time.perf_counter() - t0) * 1000.0)
            # deliberately NOT registering: store size must stay fixed per row
        rows.append(
            {
                "store_nodes": size,
                "p50_ms": round(statistics.median(probe_ms), 3),
                "p95_ms": round(_p95(probe_ms), 3),
            }
        )
        print(f"  controlled: {size:>6} nodes, p50 {rows[-1]['p50_ms']:>8.3f} ms, p95 {rows[-1]['p95_ms']:>8.3f} ms")
    return rows


def _mix_cuts() -> list[tuple[str, float]]:
    cuts: list[tuple[str, float]] = []
    acc = 0.0
    for modality, frac in CONTROLLED_MIX:
        acc += frac
        cuts.append((modality, acc))
    cuts[-1] = (cuts[-1][0], 1.01)
    return cuts


def run_exploratory_vectorized(n_world: int, seed: int) -> list[dict[str, Any]]:
    """Phase (c): numpy-vectorized exact scan — remedy sizing, NOT the verdict."""
    import numpy as np

    rng = random.Random(seed + 2)
    world_names = [f"w{i}" for i in range(n_world)]
    rows: list[dict[str, Any]] = []
    for size in CONTROLLED_SIZES:
        world_count = int(size * CONTROLLED_MIX[0][1])
        store = np.array([[rng.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in range(world_count)])
        norms = np.linalg.norm(store, axis=1)
        probe_ms: list[float] = []
        for _ in range(CONTROLLED_PROBES):
            state = {n: rng.uniform(0.0, 1.0) for n in world_names}
            vec = np.array(_embed_a4(state))
            t0 = time.perf_counter()
            sims = store @ vec / (norms * (np.linalg.norm(vec) or 1.0))
            best = int(np.argmax(sims)) if sims.size else -1
            del best
            probe_ms.append((time.perf_counter() - t0) * 1000.0)
        rows.append(
            {
                "store_nodes_total": size,
                "world_nodes_scanned": world_count,
                "p50_ms": round(statistics.median(probe_ms), 3),
                "p95_ms": round(_p95(probe_ms), 3),
            }
        )
        print(
            f"  vectorized (exploratory): {size:>6} nodes, "
            f"p50 {rows[-1]['p50_ms']:>7.3f} ms, p95 {rows[-1]['p95_ms']:>7.3f} ms"
        )
    return rows


def decide(organic: dict[str, Any], controlled: list[dict[str, Any]]) -> dict[str, Any]:
    # HARNESS NOTE (2026-09-03): the first full run's decide() computed n_cap as
    # max(largest under-bar measured row, a value extrapolated from a two-point
    # fit of the LARGEST rows). That fit's intercept was -40.5 ms, so the
    # extrapolation manufactured n_cap = 3,094 while the measured rows put the
    # 5 ms crossing between 100 nodes (2.5 ms) and 1,000 nodes (23 ms) — an
    # implementation contradicting the frozen rule's own words ("read off the
    # controlled curve"), flipping the verdict to cap-prerequisite. The rule is
    # unchanged; this implementation now reads the curve: piecewise-linear
    # interpolation between adjacent MEASURED rows, no extrapolation.
    xs = [row["store_nodes"] for row in controlled]
    ys = [row["p95_ms"] for row in controlled]

    # Least-squares fit over all controlled rows — used for the horizon
    # projection only, never for n_cap.
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    coeff = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
    intercept = mean_y - coeff * mean_x

    per_state_alloc = organic["total_nodes"] / (
        organic["world_states_presented"] + organic["interoception_states_presented"]
    )
    horizon_store = int(per_state_alloc * HORIZON_TICKS * 2)  # 2 encodes per tick (world + intero)
    p95_at_horizon = coeff * horizon_store + intercept

    n_cap = 0
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 <= P95_BAR_MS:
            n_cap = x0
            if y1 > y0:
                crossing = x0 + (P95_BAR_MS - y0) / (y1 - y0) * (x1 - x0)
                n_cap = int(min(crossing, x1))
    if ys and ys[-1] <= P95_BAR_MS:
        n_cap = xs[-1]

    if p95_at_horizon <= P95_BAR_MS:
        verdict = "ship-bare"
    elif n_cap >= CAPACITY_FLOOR_NODES:
        verdict = "cap-prerequisite"
    else:
        verdict = "index-prerequisite"
    return {
        "per_node_coeff_ms": round(coeff, 6),
        "intercept_ms": round(intercept, 4),
        "organic_alloc_nodes_per_state": round(per_state_alloc, 4),
        "horizon_ticks": HORIZON_TICKS,
        "projected_horizon_store_nodes": horizon_store,
        "projected_p95_at_horizon_ms": round(p95_at_horizon, 2),
        "p95_bar_ms": P95_BAR_MS,
        "n_cap_nodes_at_bar": n_cap,
        "capacity_floor_nodes": CAPACITY_FLOOR_NODES,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world-sensors", type=int, default=50)
    p.add_argument("--world-states", type=int, default=5000, help="organic-phase world states presented")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--json", default="", help="write full results here")
    p.add_argument("--allow-dirty", action="store_true", help="stamp allow_dirty into the record")
    args = p.parse_args(argv)

    provenance: dict[str, object] | None = None
    if args.json:
        # Gated-record preflight (item 16.7): refuse to write evidence from a
        # dirty src/scripts tree, and assert the imported `maxim` is THIS
        # repo's src (L01) — exit 3 on either, before any measurement runs.
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        import maxim
        from _provenance import DirtyTreeError, ProvenanceError, in_process_code_provenance

        try:
            provenance = in_process_code_provenance(
                _REPO_ROOT, maxim.__file__, out_path=args.json, allow_dirty=args.allow_dirty
            )
        except (ProvenanceError, DirtyTreeError) as exc:
            print(f"[FAIL] gated-record preflight: {exc}", file=sys.stderr)
            return 3

    print(f"=== organic growth: {args.world_sensors} world sensors, {args.world_states} states ===")
    organic = run_organic(args.world_sensors, args.world_states, args.seed, args.checkpoint_every)
    print(f"\n=== controlled scan timing: sizes {CONTROLLED_SIZES} ===")
    controlled = run_controlled(args.world_sensors, args.seed)
    decision = decide(organic, controlled)
    print("\n=== exploratory: vectorized exact scan (remedy sizing, not the verdict) ===")
    exploratory = run_exploratory_vectorized(args.world_sensors, args.seed)

    print("\n=== decision (rule frozen in module docstring) ===")
    for key, value in decision.items():
        print(f"  {key}: {value}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "harness": "scripts/ec_scan_cost.py",
                    "provenance": provenance,
                    "metric": "per-encode wall ms of pattern_complete_or_separate (p50/p95)",
                    "decision_rule": (
                        f"P95(horizon)<= {P95_BAR_MS}ms -> ship-bare; else cap at largest store "
                        f"with P95<={P95_BAR_MS}ms, shipped in A4's PR; cap < "
                        f"{CAPACITY_FLOOR_NODES} nodes -> index-prerequisite"
                    ),
                    "gain_exponent": GAIN_EXPONENT,
                    "threshold": FIXED_THRESHOLD,
                    "dim": DIM,
                    "seed": args.seed,
                    "organic": organic,
                    "controlled": controlled,
                    "decision": decision,
                    "exploratory_vectorized_exact_scan": {
                        "note": "remedy sizing only — outside the frozen verdict",
                        "rows": exploratory,
                    },
                },
                indent=2,
            )
        )
        print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
