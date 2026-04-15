"""P3a Stage 2 — retrieval metric extractor + sweep runner.

Provides:

- ``precision_at_k`` / ``recall_at_k`` / ``f1_at_k`` per-probe metrics.
- ``ProbeResult`` / ``SweepResult`` dataclasses.
- ``run_probes(retriever, probes)`` — run the retriever over all probes.
- ``aggregate_seeds(results)`` — mean/std/per-seed stats across seeds.
- ``compare_to_baseline(mechanism, baseline)`` — the "beats
  baseline_mean + 2×baseline_std" check the plan's pass gate requires.

All callers pass a ``retriever`` as a simple callable
``Callable[[str, int], list[tuple[str, float]]]`` (cue → ranked list).
The Hippocampus ``retrieve_on_cue`` fits this shape directly; the
TF-IDF baseline exposes the same callable.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# A retriever returns a ranked list of (node_id, score) pairs for a
# given cue. Limit specifies how many to return.
Retriever = Callable[[str, int], list[tuple[str, float]]]


# ─────────────────────────────────────────────────────────────────────────
# Per-probe metrics
# ─────────────────────────────────────────────────────────────────────────


def precision_at_k(retrieved: list[str], ground_truth: set[str], k: int) -> float:
    """Fraction of the top-k retrieved nodes that are in ground_truth.

    ``precision@k = |retrieved_top_k ∩ ground_truth| / k``

    Returns 0.0 if k == 0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for node in top_k if node in ground_truth)
    return hits / k


def recall_at_k(retrieved: list[str], ground_truth: set[str], k: int) -> float:
    """Fraction of ground_truth recovered in the top-k retrieved nodes.

    ``recall@k = |retrieved_top_k ∩ ground_truth| / |ground_truth|``

    Returns 0.0 if ground_truth is empty.
    """
    if not ground_truth:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for node in top_k if node in ground_truth)
    return hits / len(ground_truth)


def f1_at_k(retrieved: list[str], ground_truth: set[str], k: int) -> float:
    """F1 of precision@k and recall@k."""
    p = precision_at_k(retrieved, ground_truth, k)
    r = recall_at_k(retrieved, ground_truth, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ─────────────────────────────────────────────────────────────────────────
# Probe / sweep dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeResult:
    """Result of a single retrieval probe."""

    cue: str
    targets: frozenset[str]
    retrieved: tuple[tuple[str, float], ...]
    precision: float
    recall: float
    f1: float


@dataclass
class SweepResult:
    """Aggregated metrics over all probes for a single (seed, retriever) run."""

    seed: int
    n_probes: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    per_probe: list[ProbeResult] = field(default_factory=list)


@dataclass
class AggregateResult:
    """Mean/std across seeds."""

    n_seeds: int
    mean_precision: float
    std_precision: float
    mean_recall: float
    std_recall: float
    mean_f1: float
    std_f1: float
    per_seed: list[SweepResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Sweep runner
# ─────────────────────────────────────────────────────────────────────────


def run_probes(
    retriever: Retriever,
    probes: list[dict[str, Any]],
    *,
    k: int | None = None,
    seed: int = 0,
) -> SweepResult:
    """Run ``retriever`` on every probe and aggregate per-probe metrics.

    ``k`` defaults to the number of targets per probe (so precision@k
    and recall@k both max at 1.0 for a perfect retriever). Pass an
    explicit ``k`` to test recall@k for larger k.
    """
    results: list[ProbeResult] = []
    for probe in probes:
        cue = probe["cue"]
        targets = frozenset(probe["targets"])
        probe_k = k if k is not None else len(targets)
        retrieved = retriever(cue, probe_k)
        retrieved_ids = [node for node, _ in retrieved]
        p = precision_at_k(retrieved_ids, targets, probe_k)
        r = recall_at_k(retrieved_ids, targets, probe_k)
        f = f1_at_k(retrieved_ids, targets, probe_k)
        results.append(
            ProbeResult(
                cue=cue,
                targets=targets,
                retrieved=tuple(retrieved),
                precision=p,
                recall=r,
                f1=f,
            )
        )

    if not results:
        return SweepResult(
            seed=seed,
            n_probes=0,
            mean_precision=0.0,
            mean_recall=0.0,
            mean_f1=0.0,
            per_probe=[],
        )

    return SweepResult(
        seed=seed,
        n_probes=len(results),
        mean_precision=statistics.fmean(r.precision for r in results),
        mean_recall=statistics.fmean(r.recall for r in results),
        mean_f1=statistics.fmean(r.f1 for r in results),
        per_probe=results,
    )


def aggregate_seeds(results: list[SweepResult]) -> AggregateResult:
    """Aggregate per-seed metrics with mean + std.

    Uses ``statistics.stdev`` (sample stdev, ddof=1) when there are
    ≥2 seeds; for 1 seed, std = 0.0 rather than raising.
    """
    if not results:
        raise ValueError("aggregate_seeds: empty results list")

    def _mean_std(values: list[float]) -> tuple[float, float]:
        m = statistics.fmean(values)
        s = statistics.stdev(values) if len(values) >= 2 else 0.0
        return m, s

    precs = [r.mean_precision for r in results]
    recs = [r.mean_recall for r in results]
    f1s = [r.mean_f1 for r in results]

    mp, sp = _mean_std(precs)
    mr, sr = _mean_std(recs)
    mf, sf = _mean_std(f1s)

    return AggregateResult(
        n_seeds=len(results),
        mean_precision=mp,
        std_precision=sp,
        mean_recall=mr,
        std_recall=sr,
        mean_f1=mf,
        std_f1=sf,
        per_seed=results,
    )


# ─────────────────────────────────────────────────────────────────────────
# Baseline comparison — the pass gate from the plan
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BaselineComparison:
    """Outcome of comparing mechanism vs baseline by baseline_mean+2×std."""

    mechanism_mean_f1: float
    baseline_mean_f1: float
    baseline_std_f1: float
    margin: float  # mechanism_mean_f1 - (baseline_mean_f1 + 2 × baseline_std_f1)
    beats_baseline: bool


def compare_to_baseline(
    mechanism: AggregateResult,
    baseline: AggregateResult,
) -> BaselineComparison:
    """Does ``mechanism`` beat ``baseline`` by ``baseline_mean + 2 × std``?

    This is the plan's Stage 2 pass gate language; the mechanism must
    clear the 2σ envelope of the baseline to count as a real win
    (matches the P2 methodology).
    """
    gate = baseline.mean_f1 + 2 * baseline.std_f1
    margin = mechanism.mean_f1 - gate
    return BaselineComparison(
        mechanism_mean_f1=mechanism.mean_f1,
        baseline_mean_f1=baseline.mean_f1,
        baseline_std_f1=baseline.std_f1,
        margin=margin,
        beats_baseline=margin > 0,
    )


# ─────────────────────────────────────────────────────────────────────────
# Helpers — math / safety
# ─────────────────────────────────────────────────────────────────────────


def safe_log(x: float, base: float = math.e) -> float:
    """log with floor-at-epsilon so 0.0 doesn't crash the baseline."""
    if x <= 0:
        return 0.0
    return math.log(x) / math.log(base)
