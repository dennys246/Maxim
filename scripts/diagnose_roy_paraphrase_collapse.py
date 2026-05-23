#!/usr/bin/env python3
"""Roy paraphrase-collapse diagnostic.

Walks a hand-curated list of paraphrase pairs and distractors through the
production text encoder path (LinguisticEncoder + EC.pattern_complete_or_separate
at modality='text', default threshold 0.40) and answers the open question:

    Does intra-modal text clustering work on Roy CLI percept text?

Pre-registered outcomes (see docs/experiments/24_roy_paraphrase_diagnostic.md):

  CROSS_MODAL_ONLY     pair collapse ≥ 70% AND distractor collapse < 30%
                       → text clustering works; H1a is the only gap; JEPA /
                         cross-modal binding direction stands.

  EC_THRESHOLD_TUNING  pair collapse < 70% AND ≥1 pair has pre-EC cosine
                       above the EC threshold but disjoint node IDs
                       → centroid drift or threshold-tuning issue; sweep
                         MAXIM_EC_PATTERN_THRESHOLD_TEXT next.

  SUBSTRATE_BROKEN     pair collapse < 70% OR distractor collapse ≥ 30%
                       → substrate concept formation broken on Roy regime;
                         encoder / substrate write path needs rework BEFORE
                         further persona work AND BEFORE 1.0 V1 validation.

Usage:

    MAXIM_SUBSTRATE_PATH=1 python scripts/diagnose_roy_paraphrase_collapse.py \\
        --input data/roy_paraphrase_pairs.json \\
        --output /tmp/roy_paraphrase_collapse.json

Substrate-only; no LLM calls; runs in < 5 min wall.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make `python scripts/diagnose_roy_paraphrase_collapse.py` work in a repo
# checkout without requiring `pip install -e .`. Mirrors the pattern in
# scripts/behavioral_convergence_*.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/roy_paraphrase_pairs.json")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Stdout always prints a human summary.",
    )
    args = parser.parse_args()

    # Match the Roy runner env regime. The script itself does NOT route through
    # MemoryHub (so it works even with the env var unset), but production Roy
    # runs without MAXIM_SUBSTRATE_PATH=1 silently disable text-modality routing
    # per feedback_substrate_path_env_var_for_roy. Warn loudly if the regime is
    # different so the operator notices.
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

    from maxim.similarity.ec import ECConfig, EntorhinalCortex, _cosine_similarity
    from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder

    # Minimal ATL stub. LinguisticEncoder calls atl.activate_substrate_node
    # unconditionally; we capture the calls for diagnostic visibility but
    # don't need real ATL state — node_id assignment is owned by EC.
    class _StubATL:
        def __init__(self) -> None:
            self.activations: list[tuple[str, str, str]] = []

        def activate_substrate_node(self, *, node_id, text, substrate_modality, embedding_text):
            self.activations.append((node_id, text, substrate_modality))

    atl = _StubATL()
    ec = EntorhinalCortex(config=ECConfig())  # default pattern_complete_threshold=0.40
    encoder = LinguisticEncoder(ec=ec, atl=atl, config=EncoderConfig())

    # Force model load and refuse to run if sentence-transformers is not
    # available — the bag-of-words fallback would produce structurally
    # meaningless cosine similarities and the diagnostic would be a no-op.
    encoder._ensure_model()
    if encoder.using_fallback:
        print(
            "ERROR: LinguisticEncoder is using the bag-of-words hash fallback. "
            "Install sentence-transformers (pip install pymaxim[semantic]) so "
            "the diagnostic uses the same encoder as production.",
            file=sys.stderr,
        )
        return 3

    # Walk unique strings in first-seen order — mirrors how a Roy run
    # encounters sequential CLI percepts. Order matters because pattern
    # completion is centroid-dependent.
    seen: dict[str, None] = {}
    order: list[tuple[str, str]] = []  # (origin_kind, text)
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

    string_node: dict[str, str] = {}
    string_emb: dict[str, list[float]] = {}
    walk_log: list[dict] = []

    for kind, text in order:
        emb = encoder.embed(text)
        result = ec.pattern_complete_or_separate(embedding=emb, modality="text")
        if result.is_new:
            ec.register_substrate_node(result.node_id, emb, "text")
        string_node[text] = result.node_id
        string_emb[text] = emb
        walk_log.append(
            {
                "kind": kind,
                "text": text,
                "node_id": result.node_id,
                "ec_similarity_to_match": round(result.similarity, 4),
                "is_new_node": result.is_new,
            }
        )

    def _isolated_collapse(text_a: str, text_b: str) -> tuple[bool, float, float]:
        """Fresh EC, encode A then B, return (collapsed, sim_b_to_a, cosine_preEC).

        Removes cross-pair centroid contamination. Mirrors the test of "if I
        had only seen these two strings, would the substrate cluster them?"
        """
        iso_ec = EntorhinalCortex(config=ECConfig())
        emb_a = string_emb[text_a]
        emb_b = string_emb[text_b]
        ra = iso_ec.pattern_complete_or_separate(embedding=emb_a, modality="text")
        if ra.is_new:
            iso_ec.register_substrate_node(ra.node_id, emb_a, "text")
        rb = iso_ec.pattern_complete_or_separate(embedding=emb_b, modality="text")
        if rb.is_new:
            iso_ec.register_substrate_node(rb.node_id, emb_b, "text")
        collapsed = ra.node_id == rb.node_id
        cos = _cosine_similarity(emb_a, emb_b)
        return collapsed, rb.similarity, cos

    # Score pairs: collapse = same node_id. Track BOTH sequential and isolated
    # results — sequential mirrors a Roy run, isolated removes centroid drift
    # contamination so the pair-collapse signal is unconfounded.
    pair_results = []
    pair_collapsed_seq = 0
    pair_collapsed_iso = 0
    for p in pairs:
        a, b = p["a"], p["b"]
        node_a, node_b = string_node[a], string_node[b]
        cos = _cosine_similarity(string_emb[a], string_emb[b])
        collapsed_seq = node_a == node_b
        collapsed_iso, sim_iso, _ = _isolated_collapse(a, b)
        pair_collapsed_seq += int(collapsed_seq)
        pair_collapsed_iso += int(collapsed_iso)
        pair_results.append(
            {
                "id": p["id"],
                "class": p["class"],
                "a": a,
                "b": b,
                "node_a_seq": node_a,
                "node_b_seq": node_b,
                "cosine_preEC": round(cos, 4),
                "collapsed_sequential": collapsed_seq,
                "collapsed_isolated": collapsed_iso,
                "isolated_ec_similarity": round(sim_iso, 4),
            }
        )

    # Score distractors: collapse = BAD outcome.
    dist_results = []
    dist_collapsed_seq = 0
    dist_collapsed_iso = 0
    for d in distractors:
        a, b = d["a"], d["b"]
        node_a, node_b = string_node[a], string_node[b]
        cos = _cosine_similarity(string_emb[a], string_emb[b])
        collapsed_seq = node_a == node_b
        collapsed_iso, sim_iso, _ = _isolated_collapse(a, b)
        dist_collapsed_seq += int(collapsed_seq)
        dist_collapsed_iso += int(collapsed_iso)
        dist_results.append(
            {
                "id": d["id"],
                "class": d["class"],
                "a": a,
                "b": b,
                "node_a_seq": node_a,
                "node_b_seq": node_b,
                "cosine_preEC": round(cos, 4),
                "collapsed_sequential": collapsed_seq,
                "collapsed_isolated": collapsed_iso,
                "isolated_ec_similarity": round(sim_iso, 4),
            }
        )

    n_pairs = len(pair_results)
    n_dist = len(dist_results)
    pair_rate_seq = pair_collapsed_seq / n_pairs if n_pairs else 0.0
    dist_rate_seq = dist_collapsed_seq / n_dist if n_dist else 0.0
    pair_rate_iso = pair_collapsed_iso / n_pairs if n_pairs else 0.0
    dist_rate_iso = dist_collapsed_iso / n_dist if n_dist else 0.0

    # Third-outcome diagnostic: pairs whose pre-EC embedding cosine is at or
    # above the EC threshold but whose ISOLATED node IDs are disjoint anyway.
    # Run on isolated mode because sequential is contaminated by centroid drift.
    ec_threshold = ec.config.pattern_complete_threshold
    high_cos_disjoint = [r for r in pair_results if not r["collapsed_isolated"] and r["cosine_preEC"] >= ec_threshold]

    # Verdict logic — pre-registered outcomes, applied to ISOLATED mode because
    # sequential mode entangles per-pair clustering with cross-pair centroid
    # drift. The sequential numbers are reported separately as the production-
    # realistic dynamic; the isolated numbers are the clean per-pair signal.
    # If the two disagree sharply, the dynamic itself is the finding.
    pair_rate = pair_rate_iso
    dist_rate = dist_rate_iso
    centroid_drift_collapse = (
        pair_rate_iso < pair_min  # isolated says encoder can't cluster
        and pair_rate_seq >= pair_min  # but sequential mega-collapses everything
    ) or (
        # OR isolated separates distractors fine, but sequential collapses them
        dist_rate_iso < dist_max and dist_rate_seq >= dist_max
    )

    if centroid_drift_collapse:
        verdict = "CENTROID_DRIFT_COLLAPSE"
        verdict_blurb = (
            f"Isolated per-pair clustering looks reasonable (pairs {pair_rate_iso:.0%} "
            f"collapse / distractors {dist_rate_iso:.0%} collapse), but sequential "
            f"encoding produces runaway centroid drift (pairs {pair_rate_seq:.0%} / "
            f"distractors {dist_rate_seq:.0%}) — most strings end up in one mega-node. "
            f"The substrate concept formation is structurally broken under sequential "
            f"streaming input, which IS the Roy regime. Running-mean centroid update "
            f"is the load-bearing failure. Fix BEFORE further persona work AND BEFORE "
            f"1.0 V1 cross-session validation."
        )
    elif pair_rate >= pair_min and dist_rate < dist_max:
        verdict = "CROSS_MODAL_ONLY"
        verdict_blurb = (
            f"Isolated pair collapse {pair_rate_iso:.0%} ≥ {pair_min:.0%} AND "
            f"distractor collapse {dist_rate_iso:.0%} < {dist_max:.0%}. "
            f"Sequential: pairs {pair_rate_seq:.0%} / distractors {dist_rate_seq:.0%}. "
            f"Intra-modal text clustering WORKS on Roy fixtures. The persona-convergence "
            f"gap is purely the cross-modal binding gap (H1a). User's worry was N=1; "
            f"JEPA / cross-modal binding direction stands."
        )
    elif pair_rate < pair_min and high_cos_disjoint:
        verdict = "EC_THRESHOLD_TUNING"
        verdict_blurb = (
            f"Isolated pair collapse {pair_rate_iso:.0%} < {pair_min:.0%}, but "
            f"{len(high_cos_disjoint)} pair(s) had pre-EC cosine ≥ EC threshold "
            f"({ec_threshold}) yet landed in disjoint isolated nodes. Suggests "
            f"threshold tuning (H1c shape). Next step: sweep "
            f"MAXIM_EC_PATTERN_THRESHOLD_TEXT before deciding the substrate path is broken."
        )
    else:
        verdict = "SUBSTRATE_BROKEN"
        why = []
        if pair_rate < pair_min:
            why.append(f"isolated pair collapse {pair_rate_iso:.0%} < {pair_min:.0%}")
        if dist_rate >= dist_max:
            why.append(f"isolated distractor collapse {dist_rate_iso:.0%} ≥ {dist_max:.0%}")
        verdict_blurb = (
            f"Failing gate ({'; '.join(why)}). Sequential: pairs {pair_rate_seq:.0%} / "
            f"distractors {dist_rate_seq:.0%}. Intra-modal text clustering is BROKEN "
            f"on Roy fixtures even without cross-pair contamination. Substrate concept "
            f"formation fails on the regime the harness uses, not just cross-modal "
            f"alignment. Encoder / substrate write path needs rework BEFORE further "
            f"persona work AND BEFORE 1.0 V1 cross-session validation."
        )

    out = {
        "_meta": {
            "encoder_model": encoder.config.model_name,
            "ec_pattern_complete_threshold": ec_threshold,
            "decomposer": "off (whole-string encoding)",
            "MAXIM_SUBSTRATE_PATH": os.environ.get("MAXIM_SUBSTRATE_PATH", ""),
            "n_pairs": n_pairs,
            "n_distractors": n_dist,
            "input_file": str(input_path),
        },
        "summary": {
            "pair_collapse_rate_sequential": round(pair_rate_seq, 4),
            "pair_collapse_rate_isolated": round(pair_rate_iso, 4),
            "distractor_collapse_rate_sequential": round(dist_rate_seq, 4),
            "distractor_collapse_rate_isolated": round(dist_rate_iso, 4),
            "pair_collapsed_count_sequential": pair_collapsed_seq,
            "pair_collapsed_count_isolated": pair_collapsed_iso,
            "distractor_collapsed_count_sequential": dist_collapsed_seq,
            "distractor_collapsed_count_isolated": dist_collapsed_iso,
            "high_cosine_disjoint_pair_count": len(high_cos_disjoint),
            "ec_substrate_node_count_after_sequential_walk": ec.substrate_node_count,
        },
        "verdict": verdict,
        "verdict_blurb": verdict_blurb,
        "pairs": pair_results,
        "distractors": dist_results,
        "walk_log": walk_log,
    }

    # Human stdout summary.
    print()
    print("=== ROY PARAPHRASE COLLAPSE DIAGNOSTIC ===")
    print(f"encoder:               {encoder.config.model_name}")
    print(f"EC threshold:          {ec_threshold}")
    print(f"MAXIM_SUBSTRATE_PATH:  {os.environ.get('MAXIM_SUBSTRATE_PATH', '<unset>')}")
    print(f"input:                 {input_path}")
    print()
    print(f"PAIRS ({n_pairs} total — want ≥{pair_min:.0%} collapse)")
    print(f"  {'id':<32} {'cos':>6}  {'seq':<10} {'iso':<10}  class")
    for r in pair_results:
        flag_seq = "COLLAPSED" if r["collapsed_sequential"] else "separated"
        flag_iso = "COLLAPSED" if r["collapsed_isolated"] else "separated"
        print(f"  {r['id']:<32} {r['cosine_preEC']:>6.3f}  {flag_seq:<10} {flag_iso:<10}  {r['class']}")
    print(f"  → sequential pair collapse: {pair_collapsed_seq}/{n_pairs} = {pair_rate_seq:.0%}")
    print(f"  → isolated   pair collapse: {pair_collapsed_iso}/{n_pairs} = {pair_rate_iso:.0%}")
    print()
    print(f"DISTRACTORS ({n_dist} total — want <{dist_max:.0%} collapse)")
    print(f"  {'id':<32} {'cos':>6}  {'seq':<10} {'iso':<10}  class")
    for r in dist_results:
        flag_seq = "FALSE-COLL" if r["collapsed_sequential"] else "separated"
        flag_iso = "FALSE-COLL" if r["collapsed_isolated"] else "separated"
        print(f"  {r['id']:<32} {r['cosine_preEC']:>6.3f}  {flag_seq:<10} {flag_iso:<10}  {r['class']}")
    print(f"  → sequential distractor collapse: {dist_collapsed_seq}/{n_dist} = {dist_rate_seq:.0%}")
    print(f"  → isolated   distractor collapse: {dist_collapsed_iso}/{n_dist} = {dist_rate_iso:.0%}")
    print()
    print(f"EC substrate nodes after sequential walk: {ec.substrate_node_count}")
    print(f"Pairs with pre-EC cosine ≥ threshold but disjoint ISOLATED nodes: {len(high_cos_disjoint)}")
    print()
    print(f"VERDICT: {verdict}")
    for line in verdict_blurb.split(". "):
        line = line.strip()
        if line:
            print(f"  {line}{'.' if not line.endswith('.') else ''}")
    print()

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"Wrote JSON: {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
