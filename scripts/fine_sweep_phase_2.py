#!/usr/bin/env python3
"""Fine-grained threshold sweep on both P1 and Roy paraphrase fixtures.

Phase 2 of docs/plans/archive/ec_centroid_drift_fix.md — sample the
ECConfig.pattern_complete_threshold at 0.01 granularity around the
Phase 1 matrix's 0.05-grid winner, on BOTH fixtures, to find the
single threshold that strictly dominates the 0.40 baseline on each.

Loads the sentence-transformers model once, runs all (threshold × fixture)
combinations, prints a combined table, persists a JSON bundle. The
combined gate is:

  strict P1 ≥ 90%  AND  Roy sequential pair ≥ 70%  AND  Roy sequential
  distractor < 30%

Usage from the repo root:

    PYTHONPATH=src MAXIM_SUBSTRATE_PATH=1 python scripts/fine_sweep_phase_2.py

Cost: ~5 min wall on M3 Mac after the sentence-transformers model is
warm. Zero $ (substrate-only).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

# Roy fixture lives at data/roy_paraphrase_pairs.json once PR #259 merges.
# Until then, the fixture must be copied locally — point this at the
# right location.
ROY_FIXTURE = REPO / "data" / "roy_paraphrase_pairs.json"
if not ROY_FIXTURE.exists():
    fallback = Path("/tmp/roy_pairs.json")
    if fallback.exists():
        ROY_FIXTURE = fallback


def _evidence_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EC drift Phase 2 fine sweep (D27 evidence policy)")
    ap.add_argument(
        "--write-experiment-results",
        action="store_true",
        help="update the COMMITTED results record (refuses a dirty tree)",
    )
    ap.add_argument("--allow-dirty", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _evidence_args()
    # D26/D27: a degraded (hash-fallback) encoder must error on the APPARATUS,
    # never publish over the science.
    from maxim.similarity.encoder import require_semantic_encoder

    require_semantic_encoder(context="fine_sweep_phase_2 (D27)")

    # Fail-fast (review fold): resolve the evidence path — including the
    # dirty-tree refusal on the opt-in — BEFORE the sweep runs, not after.
    sys.path.insert(0, str(REPO / "scripts"))
    from _provenance import evidence_out_paths_or_exit

    [out_path] = evidence_out_paths_or_exit(
        REPO,
        [REPO / "docs" / "experiments" / "results" / "ec_drift_phase_2_fine_sweep.json"],
        write_experiment_results=args.write_experiment_results,
        allow_dirty=args.allow_dirty,
    )
    if not ROY_FIXTURE.exists():
        print(
            f"ERROR: Roy fixture not found. Looked at "
            f"{REPO / 'data' / 'roy_paraphrase_pairs.json'} and /tmp/roy_pairs.json. "
            f"Once PR #259 merges this lives in data/.",
            file=sys.stderr,
        )
        return 2

    # Sample at 0.01 around the 0.05-grid winner. Includes 0.40 (baseline)
    # and 0.50 (failed 0.05-grid winner) as anchors so the table shows the
    # full transition. Phase 2's 0.05-grid sampling missed the 0.44 sweet
    # spot — this script demonstrates the "0.01 fine-tune when the moment
    # calls for it" process rule named in docs/experiments/26_*.md.
    thresholds = [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]

    from maxim.memory.atl import ATL
    from maxim.similarity.ec import ECConfig, EntorhinalCortex
    from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder

    from tests.substrate.p1_metrics import compute_p1_metrics, load_clusters_from_fixture

    p1_fixture_path = REPO / "scenarios" / "substrate" / "paraphrase_clusters.yaml"
    p1_clusters = load_clusters_from_fixture(str(p1_fixture_path))

    # Force model load once via a throwaway encoder.
    warm = LinguisticEncoder(
        ec=EntorhinalCortex(),
        atl=ATL(),
        config=EncoderConfig(model_name="paraphrase-mpnet-base-v2"),
    )
    warm.embed("warmup")

    # P1 sweep — 10 shuffled seeds per threshold.
    print(f"\n=== P1 fine sweep ({len(thresholds)} thresholds × 10 seeds) ===", flush=True)
    p1_results: dict[float, dict] = {}
    for thresh in thresholds:
        collapse = []
        cross = []
        growth = []
        seeds_pass = 0
        for seed in range(10):
            ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=thresh))
            atl = ATL()
            encoder = LinguisticEncoder(ec=ec, atl=atl, config=EncoderConfig(model_name="paraphrase-mpnet-base-v2"))
            metrics = compute_p1_metrics(
                ec=ec,
                atl=atl,
                clusters=p1_clusters,
                encoder=encoder,
                shuffle=True,
                seed=seed,
            )
            collapse.append(metrics.collapse_rate)
            cross.append(metrics.cross_cluster_rate)
            growth.append(metrics.node_growth_final_20pct)
            if metrics.passes_p1():
                seeds_pass += 1
        mc = statistics.mean(collapse)
        sc = statistics.stdev(collapse)
        mx = statistics.mean(cross)
        mg = statistics.mean(growth)
        p1_results[thresh] = {
            "collapse_mean": round(mc, 4),
            "collapse_std": round(sc, 4),
            "cross_mean": round(mx, 4),
            "growth_mean": round(mg, 4),
            "seeds_pass": seeds_pass,
        }
        print(
            f"  t{int(thresh * 100):02d}  collapse={mc:.1%} ±{sc:.1%}  cross={mx:.1%}  growth={mg:.1%}  seeds={seeds_pass}/10",
            flush=True,
        )

    # Roy sweep — inline single-cell logic (no decomposition, no frozen-text).
    with ROY_FIXTURE.open() as f:
        roy_data = json.load(f)

    def _run_roy_cell(thresh: float) -> dict:
        """Sequential walk over the Roy fixture at a given threshold."""
        ec = EntorhinalCortex(ECConfig(pattern_complete_threshold=thresh))
        encoder = LinguisticEncoder(ec=ec, atl=ATL(), config=EncoderConfig(model_name="paraphrase-mpnet-base-v2"))
        # First-seen order across pairs then distractors — matches the
        # diagnostic script's protocol exactly.
        seen: dict[str, None] = {}
        order: list[str] = []
        for p in roy_data["pairs"]:
            for half in ("a", "b"):
                if p[half] not in seen:
                    seen[p[half]] = None
                    order.append(p[half])
        for d in roy_data["distractors"]:
            for half in ("a", "b"):
                if d[half] not in seen:
                    seen[d[half]] = None
                    order.append(d[half])
        node_of: dict[str, str] = {}
        for text in order:
            emb = encoder.embed(text)
            result = ec.pattern_complete_or_separate(embedding=emb, modality="text")
            if result.is_new:
                ec.register_substrate_node(result.node_id, emb, "text")
            node_of[text] = result.node_id
        pair_collapsed = sum(1 for p in roy_data["pairs"] if node_of[p["a"]] == node_of[p["b"]])
        dist_collapsed = sum(1 for d in roy_data["distractors"] if node_of[d["a"]] == node_of[d["b"]])
        return {
            "pair_seq": round(pair_collapsed / len(roy_data["pairs"]), 4),
            "dist_seq": round(dist_collapsed / len(roy_data["distractors"]), 4),
            "ec_nodes": ec.substrate_node_count,
        }

    print(f"\n=== Roy fine sweep ({len(thresholds)} thresholds) ===", flush=True)
    roy_results: dict[float, dict] = {}
    for thresh in thresholds:
        r = _run_roy_cell(thresh)
        roy_results[thresh] = r
        print(
            f"  t{int(thresh * 100):02d}  pair_seq={r['pair_seq']:.0%}  dist_seq={r['dist_seq']:.0%}  ec_nodes={r['ec_nodes']}",
            flush=True,
        )

    # Combined ranking — find cells where strict P1 (≥ 90%) AND Roy gate pass.
    print("\n=== Combined sweep — both fixtures ===", flush=True)
    print(
        f"  {'cell':<8} {'P1 collapse':>11} {'P1 cross':>9} {'P1 seeds':>9}  "
        f"{'Roy pair':>9} {'Roy dist':>9} {'Roy nodes':>10}  {'gate':<8}"
    )
    combined = []
    for thresh in thresholds:
        p1 = p1_results[thresh]
        r = roy_results[thresh]
        p1_strict = p1["collapse_mean"] >= 0.90
        roy_pass = r["pair_seq"] >= 0.70 and r["dist_seq"] < 0.30
        passes = p1_strict and roy_pass
        gate = "PASS" if passes else ("fail-Roy" if not roy_pass else "fail-P1")
        print(
            f"  t{int(thresh * 100):02d}     {p1['collapse_mean']:>10.1%} {p1['cross_mean']:>8.1%} "
            f"{p1['seeds_pass']:>8}/10  {r['pair_seq']:>8.0%} {r['dist_seq']:>8.0%} {r['ec_nodes']:>9}   {gate:<8}"
        )
        if passes:
            combined.append(
                {
                    "threshold": thresh,
                    "p1_collapse": p1["collapse_mean"],
                    "p1_cross": p1["cross_mean"],
                    "p1_seeds": p1["seeds_pass"],
                    "roy_pair_seq": r["pair_seq"],
                    "roy_dist_seq": r["dist_seq"],
                    "roy_ec_nodes": r["ec_nodes"],
                }
            )

    print(f"\nPASSING CELLS (strict P1 ≥ 90% AND Roy gate): {len(combined)}")
    if combined:
        # Rank by P1 collapse rate descending, then by seeds_pass descending,
        # then by cross-cluster ascending.
        combined.sort(key=lambda x: (-x["p1_collapse"], -x["p1_seeds"], x["p1_cross"]))
        winner = combined[0]
        print(f"BEST CELL: t{int(winner['threshold'] * 100):02d}")
        print(
            f"  P1: collapse={winner['p1_collapse']:.1%}, cross={winner['p1_cross']:.1%}, seeds={winner['p1_seeds']}/10"
        )
        print(
            f"  Roy: pair={winner['roy_pair_seq']:.0%}, dist={winner['roy_dist_seq']:.0%}, "
            f"nodes={winner['roy_ec_nodes']}"
        )

    # Persist.
    out = {
        "thresholds": thresholds,
        "p1": {f"{t:.2f}": v for t, v in p1_results.items()},
        "roy": {f"{t:.2f}": v for t, v in roy_results.items()},
        "passing_strict_p1_and_roy": combined,
        "winner": combined[0] if combined else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
