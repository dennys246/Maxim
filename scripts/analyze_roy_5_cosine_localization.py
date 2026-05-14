#!/usr/bin/env python
"""Roy-5a cosine-localization disambiguator.

Reads persisted EC state (``aut_ec.json``, written by
``EntorhinalCortex.save()`` since PR #248) from a Roy-5a priming session
plus each arm's session, computes three pairwise cosine matrices between
priming and arm-A centroids, and decodes the max cosine over
food-bearing priming centroids into one of three pre-registered
sub-hypotheses (H1c / H1b / H1a) per
``docs/plans/roy_5_encoder_alignment_disambiguator.md`` Stage 1.

The three matrices are computed per modality pair:

- ``M_tt`` — priming text-modality centroids vs arm text-modality
  centroids.
- ``M_dt`` — priming interoception centroids vs arm text centroids
  (cross-modality; Roy-4 confirmed both modalities fire in both phases
  so this matrix should be non-empty).
- ``M_dd`` — priming interoception vs arm interoception.

The food-bearing priming centroid set is extracted from the priming
session's ``aut_nac.json`` ``cluster_reward_bias`` map using the same
unit-separator (``\\x1f``) compound-key parsing the Roy-4 analyzer uses:
``"{agent_id}\\x1f{cluster_id}\\x1f{tool:sense_food_source}"``.

Pre-registered decoding (in this file AND in the printed summary):

================================  ============================================  ============================================================
``max(M_tt food-bearing)``        Verdict                                       Stage 2 next step
================================  ============================================  ============================================================
``>= 0.40``                       H1c — threshold/centroid tuning               2a — ``MAXIM_EC_PATTERN_THRESHOLD_TEXT`` / ``MAXIM_EC_FROZEN_TEXT`` sweep
``0.20 <= max < 0.40``            H1b — encoder model fit                       2b — kill ``LinguisticEncoder._get_encoder`` singleton, A/B alternative ST models
``< 0.20``                        H1a — encoder subspace incompatibility        2c → Stage 3 cradle-arc redesign + Hebbian retest
================================  ============================================  ============================================================

Exit code:

- ``0`` — clean H1c / H1b / H1a verdict on arm A.
- ``2`` — indeterminate (missing required input, no food-bearing priming
  centroids found, no text-modality nodes in priming OR arm A).
- ``1`` — analysis ran but arm A produced no usable signal (empty
  matrices on all three pairs). Should never happen if Roy-5a ran to
  completion against the standard fixture.

Usage::

    python scripts/analyze_roy_5_cosine_localization.py \\
        --priming-dir ~/.maxim/sim_reports/<priming_sid> \\
        --arm-a-dir   ~/.maxim/sim_reports/<arm_a_sid> \\
        --arm-b-dir   ~/.maxim/sim_reports/<arm_b_sid> \\
        --arm-c-dir   ~/.maxim/sim_reports/<arm_c_sid> \\
        --output-json /tmp/roy_5a_analysis.json

The verdict is gated on arm A (the substrate-primed arm). Arms B and C
exist for cross-checking — their EC dumps tend to be tiny (blank
substrate) but the analyzer still computes their matrices for the
output JSON bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# ─────────────────────────────────────────────────────────────────────────
# JSON loading (fail-soft)
# ─────────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON dict, or None on missing / corrupt file."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# ─────────────────────────────────────────────────────────────────────────
# Math helpers (intentionally dependency-free — keeps the script runnable
# in any Maxim checkout without dragging in numpy for a one-shot)
# ─────────────────────────────────────────────────────────────────────────


def _cosine(u: list[float], v: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns 0.0 on length mismatch or zero-norm to match the
    ``_cosine_similarity`` helper in ``similarity/ec.py`` and
    ``analysis/substrate_diff.py::_cosine``.
    """
    if len(u) != len(v) or not u:
        return 0.0
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


# ─────────────────────────────────────────────────────────────────────────
# EC state ingestion
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Centroid:
    """One substrate node's persisted state."""

    node_id: str
    embedding: tuple[float, ...]
    modality: str


def load_ec_centroids(ec_dump_path: Path) -> list[Centroid]:
    """Read ``aut_ec.json`` and return its substrate-node centroids.

    The on-disk schema written by ``EntorhinalCortex.save()`` is::

        {
            "_format_version": "1.0",
            "version": "1.0",
            "config": {...},
            "lsh": {...},
            "inverted": {...},
            "signatures": {...},
            "substrate_nodes": {
                "<uuid>": {
                    "embedding": [<float>, ...],
                    "modality": "<text|interoception|...>",
                    "count": <int>
                },
                ...
            }
        }

    Missing file / corrupt file returns an empty list — the analyzer
    surfaces this via the per-arm summary's "0 nodes loaded" line.
    """
    data = _load_json(ec_dump_path)
    if data is None:
        return []
    nodes = data.get("substrate_nodes")
    if not isinstance(nodes, dict):
        return []
    out: list[Centroid] = []
    for nid, ndata in nodes.items():
        if not isinstance(ndata, dict):
            continue
        emb = ndata.get("embedding")
        mod = ndata.get("modality")
        if not isinstance(emb, list) or not isinstance(mod, str):
            continue
        try:
            vec = tuple(float(x) for x in emb)
        except (TypeError, ValueError):
            continue
        out.append(Centroid(node_id=str(nid), embedding=vec, modality=mod))
    return out


# ─────────────────────────────────────────────────────────────────────────
# Food-bearing priming centroid extraction (mirrors Roy-4 analyzer)
# ─────────────────────────────────────────────────────────────────────────


def load_priming_food_clusters(nac_path: Path) -> set[str]:
    """Return EC cluster IDs keyed to sense_food_source bias in priming.

    The flat compound shape current ``NAc.to_dict`` writes is::

        ``"{agent_id}\\x1f{cluster_id}\\x1f{tool:sense_food_source}": <bias>``

    The middle field is the EC cluster UUID. The 3-field UTS-separator
    pattern is identical to ``analyze_roy_4_coactivation.py``; tests
    pin this so the two analyzers cannot drift on extraction semantics.

    A legacy nested shape (``_cluster_reward_bias``) is also accepted as
    a fallback for synthetic test fixtures and any future NAc dump
    variant.
    """
    if not nac_path.exists():
        return set()
    with nac_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    food_clusters: set[str] = set()

    flat = data.get("cluster_reward_bias")
    if isinstance(flat, dict) and flat:
        for compound_key in flat.keys():
            parts = str(compound_key).split("\x1f")
            if len(parts) == 3 and parts[2].endswith("sense_food_source"):
                food_clusters.add(parts[1])
        if food_clusters:
            return food_clusters

    nested = data.get("_cluster_reward_bias", {})
    if isinstance(nested, dict):
        for agent_or_cluster, inner in nested.items():
            if isinstance(inner, dict):
                for cluster_id, tool_map in inner.items():
                    if isinstance(tool_map, dict) and "sense_food_source" in tool_map:
                        food_clusters.add(str(cluster_id))
                if "sense_food_source" in inner and isinstance(inner.get("sense_food_source"), (int, float)):
                    food_clusters.add(str(agent_or_cluster))
    return food_clusters


# ─────────────────────────────────────────────────────────────────────────
# Matrix computation
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CosineMatrix:
    """Pairwise cosine matrix between two centroid sets, plus headlines.

    ``rows``: priming-side node_ids (matching ``rows_modality``).
    ``cols``: arm-side node_ids (matching ``cols_modality``).
    ``values``: ``values[i][j] = cosine(rows[i].embedding, cols[j].embedding)``.
    ``max_over_rows``: per-row max — index aligns with ``rows``.
    """

    rows: tuple[str, ...]
    cols: tuple[str, ...]
    rows_modality: str
    cols_modality: str
    values: tuple[tuple[float, ...], ...]
    max_over_rows: tuple[float, ...]

    @property
    def is_empty(self) -> bool:
        return not self.rows or not self.cols


def _filter_by_modality(centroids: Iterable[Centroid], modality: str) -> list[Centroid]:
    return [c for c in centroids if c.modality == modality]


def _compute_matrix(
    priming_centroids: list[Centroid],
    arm_centroids: list[Centroid],
    *,
    rows_modality: str,
    cols_modality: str,
) -> CosineMatrix:
    """Pairwise cosine between (priming ∩ rows_modality) × (arm ∩ cols_modality).

    Returns an empty CosineMatrix (is_empty=True) when either side has
    zero centroids of the requested modality. Important regression
    guard: M_dt (priming interoception × arm text) is the cross-modality
    matrix; Roy-4's finding was that both modalities fire in both
    phases, so this matrix should be non-empty on real data even
    though row/col modalities differ.
    """
    rows = _filter_by_modality(priming_centroids, rows_modality)
    cols = _filter_by_modality(arm_centroids, cols_modality)
    if not rows or not cols:
        return CosineMatrix(
            rows=tuple(c.node_id for c in rows),
            cols=tuple(c.node_id for c in cols),
            rows_modality=rows_modality,
            cols_modality=cols_modality,
            values=(),
            max_over_rows=(),
        )
    values: list[tuple[float, ...]] = []
    max_per_row: list[float] = []
    for r in rows:
        row_vals = tuple(_cosine(list(r.embedding), list(c.embedding)) for c in cols)
        values.append(row_vals)
        max_per_row.append(max(row_vals) if row_vals else 0.0)
    return CosineMatrix(
        rows=tuple(c.node_id for c in rows),
        cols=tuple(c.node_id for c in cols),
        rows_modality=rows_modality,
        cols_modality=cols_modality,
        values=tuple(values),
        max_over_rows=tuple(max_per_row),
    )


def _max_over_food_clusters(matrix: CosineMatrix, food_clusters: set[str]) -> float:
    """Max cosine across only the rows whose node_id is in food_clusters.

    Returns ``-inf`` if no food-bearing rows are in the matrix (e.g.,
    M_dt where all food clusters are text-modality but rows are
    interoception). Callers translate ``-inf`` to "not applicable" in
    the verdict.
    """
    if matrix.is_empty:
        return float("-inf")
    food_max = float("-inf")
    for nid, row_max in zip(matrix.rows, matrix.max_over_rows):
        if nid in food_clusters and row_max > food_max:
            food_max = row_max
    return food_max


# ─────────────────────────────────────────────────────────────────────────
# Verdict decoding (PRE-REGISTERED — do not relax thresholds without a
# new plan + experiment doc; the H1a/b/c thresholds are the load-bearing
# contract Stage 2's implementation branches consume)
# ─────────────────────────────────────────────────────────────────────────


# H1c lower bound is the current ECConfig.pattern_complete_threshold
# (P1-tuned, paraphrase-mpnet@0.40). H1b lower bound is a conservative
# floor below which "the encoder doesn't even see these as related" is
# the parsimonious explanation. These constants are the public contract
# of this analyzer — tests pin the exact boundary semantics.
H1C_LOWER_BOUND: float = 0.40
H1B_LOWER_BOUND: float = 0.20


@dataclass(frozen=True)
class Verdict:
    """Pre-registered Roy-5a decoding outcome."""

    sub_hypothesis: str  # "H1c" | "H1b" | "H1a" | "INDETERMINATE"
    max_food_cosine_tt: float
    next_stage: str
    description: str


def decode_verdict(max_food_tt: float) -> Verdict:
    """Pre-registered H1c/H1b/H1a decoding for arm A's M_tt food cosine.

    Boundary semantics (verified by the unit tests):
    - ``max >= 0.40`` → H1c
    - ``0.20 <= max < 0.40`` → H1b
    - ``max < 0.20`` → H1a (includes ``-inf`` "no food-bearing rows in
      M_tt", which is itself diagnostic of the H1a "structurally
      incompatible subspaces" case — food clusters either don't exist
      in text modality at all, or they're so dissimilar to arm A's text
      that no row contributes a non-trivial max)

    The "no food-bearing priming centroids in priming's NAc dump"
    case is handled upstream (the caller short-circuits to
    ``INDETERMINATE`` before reaching this function).
    """
    if max_food_tt >= H1C_LOWER_BOUND:
        return Verdict(
            sub_hypothesis="H1c",
            max_food_cosine_tt=max_food_tt,
            next_stage="Stage 2a — threshold/centroid tuning sweep",
            description=(
                "Text embeddings are within EC's pattern_complete_threshold "
                "of 0.40 — the encoder DOES see priming food clusters and "
                "arm-A test text as similar. Pattern completion misses "
                "because of frozen-prototype or running-mean drift "
                "interactions, not encoder fit. Ship Stage 2a: "
                "MAXIM_EC_PATTERN_THRESHOLD_TEXT + MAXIM_EC_FROZEN_TEXT "
                "env-var sweep against the Roy-2c fixture."
            ),
        )
    if max_food_tt >= H1B_LOWER_BOUND:
        return Verdict(
            sub_hypothesis="H1b",
            max_food_cosine_tt=max_food_tt,
            next_stage="Stage 2b — encoder A/B sweep",
            description=(
                "Text embeddings are moderately close (0.20-0.40) but below "
                "EC's completion threshold. The encoder MIGHT close the gap "
                "with a different sentence-transformer model. Ship Stage 2b: "
                "kill the LinguisticEncoder._get_encoder process-wide "
                "singleton, A/B 2-3 alternative models against the Roy-2c "
                "fixture."
            ),
        )
    return Verdict(
        sub_hypothesis="H1a",
        max_food_cosine_tt=max_food_tt,
        next_stage="Stage 2c → Stage 3 cradle-arc redesign + Hebbian retest",
        description=(
            "Text embeddings are structurally incomparable (max cosine "
            "< 0.20) — the encoder doesn't see priming food clusters and "
            "arm-A test text as related at all. Threshold tuning cannot "
            "rescue this; encoder A/B at the same modality has limited "
            "upside. Redirect to Stage 3: redesigned cradle priming arc "
            "with deliberate (sensor, drive, narrator-utterance) co-firing "
            "+ Hebbian retest. PASS resurrects cross_modal_substrate_"
            "binding.md; FAIL promotes encoder replacement to 1.2+."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────
# Per-arm analysis bundle
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArmResult:
    """Cosine matrices + headline cosines for one arm."""

    arm: str
    priming_text_count: int
    priming_interoception_count: int
    arm_text_count: int
    arm_interoception_count: int
    food_cluster_count: int
    food_in_text_modality_count: int
    M_tt: CosineMatrix
    M_dt: CosineMatrix
    M_dd: CosineMatrix
    max_food_cosine_tt: float
    max_food_cosine_dt: float
    max_food_cosine_dd: float


def analyze_arm(
    *,
    arm_label: str,
    priming_centroids: list[Centroid],
    arm_centroids: list[Centroid],
    food_clusters: set[str],
) -> ArmResult:
    """Compute all three matrices + the food-bearing maxima for one arm."""
    pt = _filter_by_modality(priming_centroids, "text")
    pi = _filter_by_modality(priming_centroids, "interoception")
    at = _filter_by_modality(arm_centroids, "text")
    ai = _filter_by_modality(arm_centroids, "interoception")

    m_tt = _compute_matrix(priming_centroids, arm_centroids, rows_modality="text", cols_modality="text")
    m_dt = _compute_matrix(
        priming_centroids,
        arm_centroids,
        rows_modality="interoception",
        cols_modality="text",
    )
    m_dd = _compute_matrix(
        priming_centroids,
        arm_centroids,
        rows_modality="interoception",
        cols_modality="interoception",
    )

    food_in_text = len({c.node_id for c in pt} & food_clusters)

    return ArmResult(
        arm=arm_label,
        priming_text_count=len(pt),
        priming_interoception_count=len(pi),
        arm_text_count=len(at),
        arm_interoception_count=len(ai),
        food_cluster_count=len(food_clusters),
        food_in_text_modality_count=food_in_text,
        M_tt=m_tt,
        M_dt=m_dt,
        M_dd=m_dd,
        max_food_cosine_tt=_max_over_food_clusters(m_tt, food_clusters),
        max_food_cosine_dt=_max_over_food_clusters(m_dt, food_clusters),
        max_food_cosine_dd=_max_over_food_clusters(m_dd, food_clusters),
    )


# ─────────────────────────────────────────────────────────────────────────
# CLI plumbing
# ─────────────────────────────────────────────────────────────────────────


def _fmt_cosine(v: float) -> str:
    return "n/a" if v == float("-inf") else f"{v:.4f}"


def _print_arm_summary(result: ArmResult) -> None:
    print(f"\n=== Arm {result.arm} ===")
    print(f"  Priming centroids:  text={result.priming_text_count}  interoception={result.priming_interoception_count}")
    print(f"  Arm centroids:      text={result.arm_text_count}  interoception={result.arm_interoception_count}")
    print(
        f"  Food-bearing priming clusters: {result.food_cluster_count}  (in text modality: {result.food_in_text_modality_count})"
    )
    print("  Max cosine over food-bearing priming centroids:")
    print(f"    M_tt (priming text × arm text):                {_fmt_cosine(result.max_food_cosine_tt)}")
    print(f"    M_dt (priming interoception × arm text):       {_fmt_cosine(result.max_food_cosine_dt)}")
    print(f"    M_dd (priming interoception × arm interoception): {_fmt_cosine(result.max_food_cosine_dd)}")


def _matrix_to_dict(m: CosineMatrix) -> dict[str, Any]:
    return {
        "rows": list(m.rows),
        "cols": list(m.cols),
        "rows_modality": m.rows_modality,
        "cols_modality": m.cols_modality,
        "values": [[round(v, 6) for v in row] for row in m.values],
        "max_over_rows": [round(v, 6) for v in m.max_over_rows],
        "is_empty": m.is_empty,
    }


def _arm_to_dict(r: ArmResult) -> dict[str, Any]:
    return {
        "arm": r.arm,
        "priming_text_count": r.priming_text_count,
        "priming_interoception_count": r.priming_interoception_count,
        "arm_text_count": r.arm_text_count,
        "arm_interoception_count": r.arm_interoception_count,
        "food_cluster_count": r.food_cluster_count,
        "food_in_text_modality_count": r.food_in_text_modality_count,
        "max_food_cosine_tt": (None if r.max_food_cosine_tt == float("-inf") else round(r.max_food_cosine_tt, 6)),
        "max_food_cosine_dt": (None if r.max_food_cosine_dt == float("-inf") else round(r.max_food_cosine_dt, 6)),
        "max_food_cosine_dd": (None if r.max_food_cosine_dd == float("-inf") else round(r.max_food_cosine_dd, 6)),
        "M_tt": _matrix_to_dict(r.M_tt),
        "M_dt": _matrix_to_dict(r.M_dt),
        "M_dd": _matrix_to_dict(r.M_dd),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Roy-5a post-hoc cosine-localization analyzer. Decodes max "
            "cosine over food-bearing priming centroids vs arm A into one "
            "of three pre-registered sub-hypotheses (H1c / H1b / H1a) per "
            "docs/plans/roy_5_encoder_alignment_disambiguator.md Stage 1."
        )
    )
    parser.add_argument(
        "--priming-dir",
        type=Path,
        required=True,
        help="Path to the priming session directory (contains aut_ec.json + aut_nac.json).",
    )
    parser.add_argument(
        "--arm-a-dir",
        type=Path,
        required=True,
        help="Path to arm A's session directory (substrate-primed, neutral prompt).",
    )
    parser.add_argument(
        "--arm-b-dir",
        type=Path,
        default=None,
        help="Optional: arm B session directory (blank substrate, persona prompt).",
    )
    parser.add_argument(
        "--arm-c-dir",
        type=Path,
        default=None,
        help="Optional: arm C session directory (blank substrate, neutral prompt).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="If provided, write the full analysis bundle as JSON to this path.",
    )
    args = parser.parse_args(argv)

    # ── Load priming centroids + food clusters ────────────────────────
    priming_ec_path = args.priming_dir / "aut_ec.json"
    priming_nac_path = args.priming_dir / "aut_nac.json"
    priming_centroids = load_ec_centroids(priming_ec_path)
    food_clusters = load_priming_food_clusters(priming_nac_path)

    print(f"Priming dir:         {args.priming_dir}")
    print(f"  EC centroids:      {len(priming_centroids)} from {priming_ec_path}")
    print(f"  Food-bearing clusters: {len(food_clusters)} from {priming_nac_path}")

    if not priming_centroids:
        print(
            "\nERROR: priming session has no EC centroids. Confirm PR #248 was "
            "merged before the Roy-5a run (aut_ec.json should land in "
            "every session_dir).",
            file=sys.stderr,
        )
        return 2
    if not food_clusters:
        print(
            "\nERROR: priming session's aut_nac.json contains no "
            "sense_food_source bias entries. Roy-5a verdict cannot be "
            "decoded — confirm priming arc fired sense_food_source.",
            file=sys.stderr,
        )
        return 2

    # ── Per-arm analysis ──────────────────────────────────────────────
    arm_inputs = [
        ("a", args.arm_a_dir),
        ("b", args.arm_b_dir),
        ("c", args.arm_c_dir),
    ]
    arm_results: list[ArmResult] = []
    for label, dir_path in arm_inputs:
        if dir_path is None:
            continue
        arm_ec_path = dir_path / "aut_ec.json"
        arm_centroids = load_ec_centroids(arm_ec_path)
        print(f"Arm {label} dir:           {dir_path}")
        print(f"  EC centroids:      {len(arm_centroids)} from {arm_ec_path}")
        result = analyze_arm(
            arm_label=label,
            priming_centroids=priming_centroids,
            arm_centroids=arm_centroids,
            food_clusters=food_clusters,
        )
        arm_results.append(result)

    # ── Verdict (gated on arm A) ──────────────────────────────────────
    arm_a = next((r for r in arm_results if r.arm == "a"), None)
    if arm_a is None:
        print("ERROR: no arm A result; cannot decode verdict.", file=sys.stderr)
        return 2

    verdict = decode_verdict(arm_a.max_food_cosine_tt)

    print("\n" + "=" * 70)
    print(f"ROY-5a VERDICT: {verdict.sub_hypothesis}")
    print("=" * 70)
    print(f"  max(M_tt food-bearing) = {_fmt_cosine(verdict.max_food_cosine_tt)}")
    print(
        f"  Decoding: max < {H1B_LOWER_BOUND} → H1a; {H1B_LOWER_BOUND} ≤ max < {H1C_LOWER_BOUND} → H1b; max ≥ {H1C_LOWER_BOUND} → H1c"
    )
    print(f"  Next stage: {verdict.next_stage}")
    print()
    print(f"  {verdict.description}")
    print("=" * 70)

    for r in arm_results:
        _print_arm_summary(r)

    # ── Optional JSON output ──────────────────────────────────────────
    if args.output_json is not None:
        bundle = {
            "verdict": {
                "sub_hypothesis": verdict.sub_hypothesis,
                "max_food_cosine_tt": round(verdict.max_food_cosine_tt, 6)
                if verdict.max_food_cosine_tt != float("-inf")
                else None,
                "next_stage": verdict.next_stage,
                "description": verdict.description,
            },
            "thresholds": {
                "H1C_LOWER_BOUND": H1C_LOWER_BOUND,
                "H1B_LOWER_BOUND": H1B_LOWER_BOUND,
            },
            "priming": {
                "dir": str(args.priming_dir),
                "centroid_count": len(priming_centroids),
                "food_cluster_count": len(food_clusters),
                "food_clusters": sorted(food_clusters),
            },
            "arms": {r.arm: _arm_to_dict(r) for r in arm_results},
        }
        args.output_json.write_text(json.dumps(bundle, indent=2, sort_keys=True))
        print(f"\nWrote analysis bundle to {args.output_json}")

    # ── Exit code ─────────────────────────────────────────────────────
    if arm_a.M_tt.is_empty and arm_a.M_dt.is_empty and arm_a.M_dd.is_empty:
        print(
            "WARNING: arm A produced no centroids in any modality — verdict is based on empty matrices.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
