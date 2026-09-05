#!/usr/bin/env python3
"""Roy paraphrase-collapse diagnostic.

Walks a hand-curated list of paraphrase pairs and distractors through the
production text encoder path (LinguisticEncoder + EC.pattern_complete_or_separate)
and answers two questions:

    1. (Single-cell mode, default) Does intra-modal text clustering work on
       Roy CLI percept text at the current default config (decomposition off,
       frozen_centroid_modalities = {"interoception"}, threshold 0.40)?

    2. (Matrix mode, --matrix) Which (decomposition × frozen-text-centroid ×
       threshold) cell satisfies the sequential pair ≥ 70% / distractor < 30%
       gate with the smallest delta from current defaults?

The matrix mode implements Phase 1 of docs/plans/archive/ec_centroid_drift_fix.md.

Pre-registered single-cell outcomes (see docs/experiments/24_roy_paraphrase_diagnostic.md):

  CROSS_MODAL_ONLY     pair collapse ≥ 70% AND distractor collapse < 30%
                       → text clustering works; H1a is the only gap.

  EC_THRESHOLD_TUNING  pair collapse < 70% AND ≥1 pair has pre-EC cosine
                       above the EC threshold but disjoint isolated nodes
                       → centroid drift or threshold-tuning issue.

  SUBSTRATE_BROKEN     isolated pair collapse < 70% OR isolated distractor
                       collapse ≥ 30%
                       → substrate concept formation broken on Roy regime.

  CENTROID_DRIFT_COLLAPSE  isolated mode passes the gate but sequential
                           mode fails it (added 2026-05-23; the data shape
                           the original three did not anticipate).

Usage (single-cell, current defaults):

    MAXIM_SUBSTRATE_PATH=1 python scripts/diagnose_roy_paraphrase_collapse.py \\
        --input data/roy_paraphrase_pairs.json \\
        --output /tmp/roy_paraphrase_collapse.json

Usage (matrix sweep, Phase 1):

    MAXIM_SUBSTRATE_PATH=1 python scripts/diagnose_roy_paraphrase_collapse.py \\
        --matrix \\
        --input data/roy_paraphrase_pairs.json \\
        --output-dir /tmp/roy_matrix

Substrate-only; no LLM calls; runs in < 5 min wall (matrix in ~1-2 min).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Make `python scripts/diagnose_roy_paraphrase_collapse.py` work in a repo
# checkout without requiring `pip install -e .`. Mirrors the pattern in
# scripts/behavioral_convergence_*.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ─────────────────────────────────────────────────────────────────────────
# Cell config
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CellConfig:
    """One matrix cell — a specific EC / encoder configuration to test."""

    decomposition: bool  # wire a ConceptDecomposer into LinguisticEncoder
    frozen_text_centroid: bool  # add "text" to ECConfig.frozen_centroid_modalities
    threshold: float  # ECConfig.pattern_complete_threshold

    def label(self) -> str:
        d = "d1" if self.decomposition else "d0"
        f = "f1" if self.frozen_text_centroid else "f0"
        t = f"t{int(self.threshold * 100):02d}"
        return f"{d}_{f}_{t}"

    def delta_from_current(self) -> tuple[int, int, float]:
        """Sort key for "smallest config delta from current defaults".

        Current defaults: decomposition=False, frozen_text_centroid=False,
        threshold=0.40. Smaller tuple = closer to current defaults.
        """
        return (
            int(self.decomposition),
            int(self.frozen_text_centroid),
            abs(self.threshold - 0.40),
        )


# Current defaults — the baseline cell that single-cell mode runs by default.
CURRENT_DEFAULTS = CellConfig(decomposition=False, frozen_text_centroid=False, threshold=0.40)

# Phase 1 matrix axes per docs/plans/archive/ec_centroid_drift_fix.md.
MATRIX_CELLS: list[CellConfig] = [
    CellConfig(decomposition=d, frozen_text_centroid=f, threshold=t)
    for d in (False, True)
    for f in (False, True)
    for t in (0.40, 0.50, 0.60, 0.70)
]


# ─────────────────────────────────────────────────────────────────────────
# Encoder + EC construction per cell
# ─────────────────────────────────────────────────────────────────────────


def _build_stack(cell: CellConfig):
    """Return (encoder, ec, ec_config) for a cell.

    Encoder shares the lazy-loaded sentence-transformers model across cells
    via the LinguisticEncoder module-level singleton; only the EC and the
    encoder wrapper are freshly constructed per cell.
    """
    from maxim.similarity.ec import ECConfig, EntorhinalCortex
    from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder

    fcm = frozenset({"interoception", "text"}) if cell.frozen_text_centroid else frozenset({"interoception"})
    ec_config = ECConfig(
        pattern_complete_threshold=cell.threshold,
        frozen_centroid_modalities=fcm,
    )
    ec = EntorhinalCortex(config=ec_config)

    class _StubATL:
        def __init__(self) -> None:
            self.activations: list[tuple[str, str, str]] = []

        def activate_substrate_node(self, *, node_id, text, substrate_modality, embedding_text):
            self.activations.append((node_id, text, substrate_modality))

    decomposer = None
    if cell.decomposition:
        from maxim.similarity.decomposer import ConceptDecomposer

        decomposer = ConceptDecomposer()

    encoder = LinguisticEncoder(
        ec=ec,
        atl=_StubATL(),
        config=EncoderConfig(),
        decomposer=decomposer,
    )
    return encoder, ec, ec_config


def _encode_text(encoder, ec, text: str, cell: CellConfig) -> tuple[set[str], list[float]]:
    """Encode a string through the cell's configured path.

    Returns (set_of_active_node_ids, whole-string-embedding).

    With decomposition off, the set has one element. With decomposition on,
    the set has one element per noun chunk extracted by spaCy. The
    whole-string embedding is returned in both modes for pre-EC cosine
    reporting.
    """
    whole_emb = encoder.embed(text)
    if cell.decomposition:
        # encode_decomposed routes through ConceptDecomposer + per-chunk EC
        # pattern_complete_or_separate. Returns list of node IDs; we collect
        # into a set for collapse-via-intersection semantics.
        node_ids = encoder.encode_decomposed(text, "text", agent_id="diagnostic-agent")
        return set(node_ids), whole_emb
    result = ec.pattern_complete_or_separate(embedding=whole_emb, modality="text")
    if result.is_new:
        ec.register_substrate_node(result.node_id, whole_emb, "text")
    return {result.node_id}, whole_emb


def _is_collapsed(nodes_a: set[str], nodes_b: set[str]) -> bool:
    """Two strings collapse iff their active-node-ID sets intersect.

    Generalizes the single-node "same node_id" check to the decomposed
    case (multiple node IDs per string). A pair "collapses" if they share
    AT LEAST ONE concept-level node.
    """
    return bool(nodes_a & nodes_b)


# ─────────────────────────────────────────────────────────────────────────
# Cell runner
# ─────────────────────────────────────────────────────────────────────────


def run_cell(pairs: list[dict], distractors: list[dict], cell: CellConfig) -> dict[str, Any]:
    """Run sequential + isolated diagnostic for one cell and return its result.

    Returns a dict mirroring the single-cell output format, plus the cell
    config under `_meta.cell_config`.
    """
    from maxim.similarity.ec import _cosine_similarity

    encoder, ec, ec_config = _build_stack(cell)
    encoder._ensure_model()
    if encoder.using_fallback:
        return {"_error": "encoder using bag-of-words fallback", "_meta": {"cell_config": asdict(cell)}}

    # Walk unique strings in first-seen order — mirrors how a Roy run
    # encounters sequential CLI percepts. Order matters because pattern
    # completion is centroid-dependent.
    seen: dict[str, None] = {}
    order: list[tuple[str, str]] = []
    for p in pairs:
        for half in ("a", "b"):
            t = p[half]
            if t not in seen:
                seen[t] = None
                order.append(("pair", t))
    for d in distractors:
        for half in ("a", "b"):
            t = d[half]
            if t not in seen:
                seen[t] = None
                order.append(("distractor", t))

    string_nodes: dict[str, set[str]] = {}
    string_emb: dict[str, list[float]] = {}
    walk_log: list[dict] = []
    for kind, text in order:
        nodes, emb = _encode_text(encoder, ec, text, cell)
        string_nodes[text] = nodes
        string_emb[text] = emb
        walk_log.append(
            {
                "kind": kind,
                "text": text,
                "node_ids": sorted(nodes),
                "node_count": len(nodes),
            }
        )

    def _isolated_collapse(text_a: str, text_b: str) -> tuple[bool, int]:
        """Fresh EC + encoder for just this pair; return (collapsed, total_isolated_nodes)."""
        iso_encoder, iso_ec, _ = _build_stack(cell)
        # Reuse whole-string embeddings only — but with decomposition, the
        # chunks need re-encoding via encode_decomposed, so re-run the full
        # encode path. With decomposition off, this re-embeds — cheap.
        nids_a, _ = _encode_text(iso_encoder, iso_ec, text_a, cell)
        nids_b, _ = _encode_text(iso_encoder, iso_ec, text_b, cell)
        return _is_collapsed(nids_a, nids_b), iso_ec.substrate_node_count

    def _score(entries: list[dict]) -> tuple[list[dict], int, int]:
        results: list[dict] = []
        n_seq = 0
        n_iso = 0
        for e in entries:
            a, b = e["a"], e["b"]
            cos = _cosine_similarity(string_emb[a], string_emb[b])
            collapsed_seq = _is_collapsed(string_nodes[a], string_nodes[b])
            collapsed_iso, iso_nodes = _isolated_collapse(a, b)
            n_seq += int(collapsed_seq)
            n_iso += int(collapsed_iso)
            results.append(
                {
                    "id": e["id"],
                    "class": e["class"],
                    "a": a,
                    "b": b,
                    "nodes_a_seq": sorted(string_nodes[a]),
                    "nodes_b_seq": sorted(string_nodes[b]),
                    "cosine_preEC": round(cos, 4),
                    "collapsed_sequential": collapsed_seq,
                    "collapsed_isolated": collapsed_iso,
                    "isolated_total_nodes": iso_nodes,
                }
            )
        return results, n_seq, n_iso

    pair_results, pair_seq, pair_iso = _score(pairs)
    dist_results, dist_seq, dist_iso = _score(distractors)
    n_pairs = len(pair_results)
    n_dist = len(dist_results)

    pair_rate_seq = pair_seq / n_pairs if n_pairs else 0.0
    pair_rate_iso = pair_iso / n_pairs if n_pairs else 0.0
    dist_rate_seq = dist_seq / n_dist if n_dist else 0.0
    dist_rate_iso = dist_iso / n_dist if n_dist else 0.0

    return {
        "_meta": {
            "cell_config": asdict(cell),
            "cell_label": cell.label(),
            "encoder_model": encoder.config.model_name,
            "ec_pattern_complete_threshold": ec_config.pattern_complete_threshold,
            "frozen_centroid_modalities": sorted(ec_config.frozen_centroid_modalities),
            "decomposition": "on" if cell.decomposition else "off",
            "n_pairs": n_pairs,
            "n_distractors": n_dist,
        },
        "summary": {
            "pair_collapse_rate_sequential": round(pair_rate_seq, 4),
            "pair_collapse_rate_isolated": round(pair_rate_iso, 4),
            "distractor_collapse_rate_sequential": round(dist_rate_seq, 4),
            "distractor_collapse_rate_isolated": round(dist_rate_iso, 4),
            "pair_collapsed_count_sequential": pair_seq,
            "pair_collapsed_count_isolated": pair_iso,
            "distractor_collapsed_count_sequential": dist_seq,
            "distractor_collapsed_count_isolated": dist_iso,
            "ec_substrate_node_count_after_sequential_walk": ec.substrate_node_count,
        },
        "pairs": pair_results,
        "distractors": dist_results,
        "walk_log": walk_log,
    }


# ─────────────────────────────────────────────────────────────────────────
# Verdict logic (single-cell mode)
# ─────────────────────────────────────────────────────────────────────────


def classify_verdict(cell_result: dict, pair_min: float, dist_max: float) -> tuple[str, str]:
    """Map a cell's summary to one of the four pre-registered verdicts."""
    s = cell_result["summary"]
    pair_seq = s["pair_collapse_rate_sequential"]
    pair_iso = s["pair_collapse_rate_isolated"]
    dist_seq = s["distractor_collapse_rate_sequential"]
    dist_iso = s["distractor_collapse_rate_isolated"]
    threshold = cell_result["_meta"]["ec_pattern_complete_threshold"]
    high_cos_disjoint = sum(
        1 for r in cell_result["pairs"] if not r["collapsed_isolated"] and r["cosine_preEC"] >= threshold
    )

    centroid_drift = (pair_iso < pair_min and pair_seq >= pair_min) or (dist_iso < dist_max and dist_seq >= dist_max)

    if centroid_drift:
        return (
            "CENTROID_DRIFT_COLLAPSE",
            f"Isolated per-pair clustering looks reasonable (pairs {pair_iso:.0%} collapse / "
            f"distractors {dist_iso:.0%} collapse), but sequential encoding produces runaway "
            f"centroid drift (pairs {pair_seq:.0%} / distractors {dist_seq:.0%}) — most strings "
            f"end up in one mega-node. Substrate concept formation is structurally broken under "
            f"sequential streaming input, which IS the Roy regime. Running-mean centroid update "
            f"is the load-bearing failure.",
        )
    if pair_iso >= pair_min and dist_iso < dist_max:
        return (
            "CROSS_MODAL_ONLY",
            f"Isolated pair collapse {pair_iso:.0%} ≥ {pair_min:.0%} AND distractor collapse "
            f"{dist_iso:.0%} < {dist_max:.0%}. Sequential: pairs {pair_seq:.0%} / distractors "
            f"{dist_seq:.0%}. Intra-modal text clustering WORKS on Roy fixtures.",
        )
    if pair_iso < pair_min and high_cos_disjoint > 0:
        return (
            "EC_THRESHOLD_TUNING",
            f"Isolated pair collapse {pair_iso:.0%} < {pair_min:.0%}, but {high_cos_disjoint} "
            f"pair(s) had pre-EC cosine ≥ EC threshold ({threshold}) yet landed in disjoint "
            f"isolated nodes. Suggests threshold tuning (H1c shape).",
        )
    why = []
    if pair_iso < pair_min:
        why.append(f"isolated pair collapse {pair_iso:.0%} < {pair_min:.0%}")
    if dist_iso >= dist_max:
        why.append(f"isolated distractor collapse {dist_iso:.0%} ≥ {dist_max:.0%}")
    return (
        "SUBSTRATE_BROKEN",
        f"Failing gate ({'; '.join(why)}). Sequential: pairs {pair_seq:.0%} / distractors "
        f"{dist_seq:.0%}. Intra-modal text clustering is BROKEN on Roy fixtures even without "
        f"cross-pair contamination.",
    )


# ─────────────────────────────────────────────────────────────────────────
# Matrix runner + ranking
# ─────────────────────────────────────────────────────────────────────────


def run_matrix(
    pairs: list[dict],
    distractors: list[dict],
    pair_min: float,
    dist_max: float,
) -> dict[str, Any]:
    """Run all 2x2x4 cells and rank the passing ones by smallest delta.

    Pass criterion (per docs/plans/archive/ec_centroid_drift_fix.md Phase 1):
      sequential pair collapse >= pair_min AND
      sequential distractor collapse < dist_max.

    Among passing cells, sort by CellConfig.delta_from_current() (smallest
    first). The winner is the lex-smallest delta.
    """
    cells: list[dict[str, Any]] = []
    for cell in MATRIX_CELLS:
        print(f"  cell {cell.label()}: running...", file=sys.stderr)
        result = run_cell(pairs, distractors, cell)
        verdict, blurb = classify_verdict(result, pair_min, dist_max)
        result["verdict"] = verdict
        result["verdict_blurb"] = blurb
        cells.append(result)

    passing_indices = [
        i
        for i, c in enumerate(cells)
        if c["summary"]["pair_collapse_rate_sequential"] >= pair_min
        and c["summary"]["distractor_collapse_rate_sequential"] < dist_max
    ]
    # Sort passing cells by delta-from-defaults (smallest first).
    passing_indices.sort(key=lambda i: CellConfig(**cells[i]["_meta"]["cell_config"]).delta_from_current())

    winner_idx = passing_indices[0] if passing_indices else None
    return {
        "cells": cells,
        "passing_cell_labels": [cells[i]["_meta"]["cell_label"] for i in passing_indices],
        "winner_cell_label": cells[winner_idx]["_meta"]["cell_label"] if winner_idx is not None else None,
        "winner_cell_config": cells[winner_idx]["_meta"]["cell_config"] if winner_idx is not None else None,
        "winner_summary": cells[winner_idx]["summary"] if winner_idx is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────
# Pretty printers
# ─────────────────────────────────────────────────────────────────────────


def _print_single_cell(result: dict, pair_min: float, dist_max: float) -> None:
    meta = result["_meta"]
    s = result["summary"]
    print()
    print("=== ROY PARAPHRASE COLLAPSE DIAGNOSTIC ===")
    print(f"cell:                  {meta['cell_label']}")
    print(f"encoder:               {meta['encoder_model']}")
    print(f"EC threshold:          {meta['ec_pattern_complete_threshold']}")
    print(f"frozen centroids:      {','.join(meta['frozen_centroid_modalities'])}")
    print(f"decomposition:         {meta['decomposition']}")
    print(f"MAXIM_SUBSTRATE_PATH:  {os.environ.get('MAXIM_SUBSTRATE_PATH', '<unset>')}")
    print()
    print(f"PAIRS ({meta['n_pairs']} total — want ≥{pair_min:.0%} collapse)")
    print(f"  {'id':<32} {'cos':>6}  {'seq':<10} {'iso':<10}  class")
    for r in result["pairs"]:
        flag_seq = "COLLAPSED" if r["collapsed_sequential"] else "separated"
        flag_iso = "COLLAPSED" if r["collapsed_isolated"] else "separated"
        print(f"  {r['id']:<32} {r['cosine_preEC']:>6.3f}  {flag_seq:<10} {flag_iso:<10}  {r['class']}")
    print(
        f"  → sequential pair collapse: {s['pair_collapsed_count_sequential']}/{meta['n_pairs']} = "
        f"{s['pair_collapse_rate_sequential']:.0%}"
    )
    print(
        f"  → isolated   pair collapse: {s['pair_collapsed_count_isolated']}/{meta['n_pairs']} = "
        f"{s['pair_collapse_rate_isolated']:.0%}"
    )
    print()
    print(f"DISTRACTORS ({meta['n_distractors']} total — want <{dist_max:.0%} collapse)")
    print(f"  {'id':<32} {'cos':>6}  {'seq':<10} {'iso':<10}  class")
    for r in result["distractors"]:
        flag_seq = "FALSE-COLL" if r["collapsed_sequential"] else "separated"
        flag_iso = "FALSE-COLL" if r["collapsed_isolated"] else "separated"
        print(f"  {r['id']:<32} {r['cosine_preEC']:>6.3f}  {flag_seq:<10} {flag_iso:<10}  {r['class']}")
    print(
        f"  → sequential distractor collapse: {s['distractor_collapsed_count_sequential']}/"
        f"{meta['n_distractors']} = {s['distractor_collapse_rate_sequential']:.0%}"
    )
    print(
        f"  → isolated   distractor collapse: {s['distractor_collapsed_count_isolated']}/"
        f"{meta['n_distractors']} = {s['distractor_collapse_rate_isolated']:.0%}"
    )
    print()
    print(f"EC substrate nodes after sequential walk: {s['ec_substrate_node_count_after_sequential_walk']}")
    print()
    print(f"VERDICT: {result['verdict']}")
    for line in result["verdict_blurb"].split(". "):
        line = line.strip()
        if line:
            print(f"  {line}{'.' if not line.endswith('.') else ''}")
    print()


def _print_matrix(matrix: dict, pair_min: float, dist_max: float) -> None:
    print()
    print("=== ROY PARAPHRASE COLLAPSE MATRIX SWEEP ===")
    print(f"gate: sequential pair ≥ {pair_min:.0%} AND sequential distractor < {dist_max:.0%}")
    print()
    print(
        f"  {'cell':<14} {'pair_seq':>8} {'pair_iso':>8} {'dist_seq':>8} {'dist_iso':>8} "
        f"{'#nodes':>7}  {'gate':<6}  verdict"
    )
    for c in matrix["cells"]:
        s = c["summary"]
        passes = s["pair_collapse_rate_sequential"] >= pair_min and s["distractor_collapse_rate_sequential"] < dist_max
        gate = "PASS" if passes else "fail"
        print(
            f"  {c['_meta']['cell_label']:<14} "
            f"{s['pair_collapse_rate_sequential']:>8.0%} "
            f"{s['pair_collapse_rate_isolated']:>8.0%} "
            f"{s['distractor_collapse_rate_sequential']:>8.0%} "
            f"{s['distractor_collapse_rate_isolated']:>8.0%} "
            f"{s['ec_substrate_node_count_after_sequential_walk']:>7} "
            f"  {gate:<6}  {c['verdict']}"
        )
    print()
    if matrix["winner_cell_label"]:
        print(f"PASSING CELLS ({len(matrix['passing_cell_labels'])}): {', '.join(matrix['passing_cell_labels'])}")
        print(f"WINNER (smallest delta from defaults): {matrix['winner_cell_label']}")
        print(f"  config: {matrix['winner_cell_config']}")
        print(f"  summary: {matrix['winner_summary']}")
    else:
        print("NO CELLS PASS THE GATE.")
        print("  Phase 1 loops back — extend matrix axes or commit to a more aggressive fix")
        print("  (member-count cap, sparse coding) per docs/plans/archive/ec_centroid_drift_fix.md.")
    print()


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="data/roy_paraphrase_pairs.json")
    parser.add_argument(
        "--output",
        default=None,
        help="Single-cell mode: write JSON to this path. Ignored in --matrix mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Matrix mode: write per-cell JSON + matrix.json to this directory.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run the 2x2x4 (decomp × frozen-text × threshold) matrix sweep per "
        "Phase 1 of docs/plans/archive/ec_centroid_drift_fix.md.",
    )
    args = parser.parse_args()

    # D26/D27: a degraded (hash-fallback) encoder must error on the APPARATUS —
    # this diagnostic's cosine numbers are meaningless on hash embeddings.
    from maxim.similarity.encoder import require_semantic_encoder

    require_semantic_encoder(context="diagnose_roy_paraphrase_collapse (D27)")

    if os.environ.get("MAXIM_SUBSTRATE_PATH", "").strip().lower() not in ("1", "true", "yes", "on"):
        print(
            "WARNING: MAXIM_SUBSTRATE_PATH is not set. This script bypasses "
            "MemoryHub so it will still run, but a production Roy run without "
            "the env var silently disables text-modality routing. Re-run with "
            "MAXIM_SUBSTRATE_PATH=1 to keep parity with the Roy regime.",
            file=sys.stderr,
        )

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2
    with input_path.open() as f:
        data = json.load(f)

    pairs = data["pairs"]
    distractors = data["distractors"]
    decision = data["_meta"]["decision_thresholds"]
    pair_min = float(decision["pair_collapse_min"])
    dist_max = float(decision["distractor_collapse_max"])

    if args.matrix:
        print(f"Running matrix sweep: {len(MATRIX_CELLS)} cells...", file=sys.stderr)
        matrix = run_matrix(pairs, distractors, pair_min, dist_max)
        _print_matrix(matrix, pair_min, dist_max)
        if args.output_dir:
            out_dir = Path(args.output_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            for c in matrix["cells"]:
                (out_dir / f"cell_{c['_meta']['cell_label']}.json").write_text(json.dumps(c, indent=2))
            (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2))
            print(f"Wrote {len(matrix['cells'])} per-cell JSON + matrix.json to {out_dir}", file=sys.stderr)
        return 0

    # Single-cell mode (default — runs current defaults).
    result = run_cell(pairs, distractors, CURRENT_DEFAULTS)
    if "_error" in result:
        print(f"ERROR: {result['_error']}", file=sys.stderr)
        return 3
    verdict, blurb = classify_verdict(result, pair_min, dist_max)
    result["verdict"] = verdict
    result["verdict_blurb"] = blurb
    _print_single_cell(result, pair_min, dist_max)
    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Wrote JSON: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
