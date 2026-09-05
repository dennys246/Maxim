"""Exp D8 — read-side mutation measurement (1.2 gate 3): accept or separate.

Pre-registration: docs/experiments/protocols/d8_read_mutation_preregistration.md
(THE authority on any divergence; this docstring mirrors it).

D8 (bugs ledger): `bio_enrichment`'s recall path calls
`EC.pattern_complete_or_separate(embedding, "text", geometry=...)` per enrichment
query; text is not a frozen-centroid modality, so every completing recall moves
the matched centroid ~1/(n+1) and increments its member count. This harness
measures the "querying degrades text-cluster resolution" claim on the REAL
encode/recall path and computes the accept-or-separate verdict with
:func:`decide_verdict` — the frozen decision rule, no operator judgment.

Arms: BASELINE (shipped behavior), FROZEN (text centroids frozen — drift must be
exactly 0.0 or the instrument is broken, exit 4), AMPLIFIED (R x 5 — mean drift
must strictly exceed BASELINE's or the meter cannot see, exit 4).

Metrics: M1 probe identity churn (48 encode texts, completed on save/load CLONES
so probing never mutates the store under test); M2 per-node centroid cosine
before/after (mean + min); M3 count provenance (reported only — structural fact
for gate 4, never folded into the verdict).

Decision rule (frozen): separate-required iff churn > 2/48 OR min cosine < 0.98;
else accept.

Data: docs/experiments/data/d8_read_mutation_<date>.json — written only with
--write-experiment-results, behind the gated-record provenance preflight
(clean tree, this repo's maxim).

In-process harness — imports maxim, spawns nothing.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Frozen apparatus constants (pre-registered; changing any after first data
# requires an amendment header in the protocol doc).
# ---------------------------------------------------------------------------

REPEATS = 6
AMPLIFY_FACTOR = 5
CHURN_MAX_PROBES = 2  # of 48
DRIFT_MIN_COS = 0.98
COMPLETE_MIN_FRAC = 0.90
ENCODER_MODEL = "paraphrase-mpnet-base-v2"

# 16 concepts x (3 encode + 2 held-out query) percept-like variants. Authored
# pre-registration; near-paraphrase within concept, distinct across concepts.
CORPUS: dict[str, dict[str, list[str]]] = {
    "kitchen_kettle": {
        "encode": [
            "The kettle on the stove is whistling loudly",
            "A kettle whistles loudly on the stove",
            "The stovetop kettle has started whistling",
        ],
        "query": [
            "The kettle is whistling on the stove again",
            "That kettle on the stove keeps whistling",
        ],
    },
    "door_creak": {
        "encode": [
            "The wooden door creaks when it swings open",
            "That wooden door creaks as it opens",
            "The door made of wood creaks while opening",
        ],
        "query": [
            "The wooden door is creaking open",
            "A creak comes from the wooden door opening",
        ],
    },
    "dog_bark": {
        "encode": [
            "A large dog is barking at the mail carrier",
            "The big dog barks at the mail carrier",
            "A large dog barks loudly at the mail carrier",
        ],
        "query": [
            "The big dog is barking at the mail carrier again",
            "That large dog keeps barking at the mail carrier",
        ],
    },
    "library_ladder": {
        "encode": [
            "The tall ladder slides along the library bookshelves",
            "A tall ladder slides down the bookshelves of the library",
            "The library's tall ladder glides along its bookshelves",
        ],
        "query": [
            "The tall ladder keeps sliding along the library shelves",
            "That ladder slides across the library bookshelves",
        ],
    },
    "sword_rust": {
        "encode": [
            "The old sword's blade is covered in rust",
            "Rust covers the blade of the old sword",
            "The blade of the old sword has rusted over",
        ],
        "query": [
            "The old sword blade is rusty all over",
            "Rust has spread across the old sword's blade",
        ],
    },
    "campfire_smoke": {
        "encode": [
            "Smoke rises from the dying campfire embers",
            "The dying campfire's embers send up smoke",
            "Smoke drifts up from the campfire's dying embers",
        ],
        "query": [
            "The campfire embers are still smoking",
            "Smoke keeps rising off the dying campfire",
        ],
    },
    "robot_arm_lift": {
        "encode": [
            "The robot arm lifts the red block from the table",
            "A robot arm picks the red block up off the table",
            "The robotic arm lifts the red block off the table",
        ],
        "query": [
            "The robot arm is lifting the red block again",
            "The robotic arm picks up the red block from the table",
        ],
    },
    "battery_low": {
        "encode": [
            "The battery indicator shows the charge is nearly empty",
            "The battery gauge shows charge almost empty",
            "The charge indicator shows the battery is nearly drained",
        ],
        "query": [
            "The battery indicator says the charge is almost gone",
            "The gauge shows the battery charge nearly empty",
        ],
    },
    "chess_checkmate": {
        "encode": [
            "The black queen delivers checkmate on the chessboard",
            "Checkmate comes from the black queen on the chessboard",
            "On the chessboard the black queen delivers checkmate",
        ],
        "query": [
            "The black queen just delivered checkmate in the chess game",
            "Checkmate by the black queen ends the chess match",
        ],
    },
    "spreadsheet_error": {
        "encode": [
            "The spreadsheet formula returns a divide by zero error",
            "A divide by zero error comes from the spreadsheet formula",
            "The formula in the spreadsheet throws a divide by zero error",
        ],
        "query": [
            "The spreadsheet keeps showing a divide by zero error",
            "That formula returns a divide by zero error again",
        ],
    },
    "engine_stall": {
        "encode": [
            "The car engine stalls at the traffic light",
            "At the traffic light the car's engine stalls",
            "The car's engine keeps stalling at the light",
        ],
        "query": [
            "The engine stalled again at the traffic light",
            "The car stalls its engine at the light",
        ],
    },
    "bird_nest": {
        "encode": [
            "A small bird builds a nest in the oak tree",
            "The small bird is building its nest in the oak",
            "In the oak tree a small bird builds a nest",
        ],
        "query": [
            "The little bird keeps building its nest in the oak tree",
            "A small bird is making a nest in the oak",
        ],
    },
    "sailboat_tack": {
        "encode": [
            "The white sailboat tacks across the crowded harbor",
            "Across the crowded harbor the white sailboat tacks",
            "The white sailboat is tacking through the crowded harbor",
        ],
        "query": [
            "The white sailboat keeps tacking across the harbor",
            "That white sailboat tacks its way through the harbor",
        ],
    },
    "wind_leaves": {
        "encode": [
            "The wind rustles the dry autumn leaves",
            "Dry autumn leaves rustle in the wind",
            "The wind is rustling through the dry autumn leaves",
        ],
        "query": [
            "Autumn leaves keep rustling in the wind",
            "The dry leaves rustle as the wind blows",
        ],
    },
    "printer_jam": {
        "encode": [
            "The office printer jams on the second page",
            "The printer in the office jams at page two",
            "The office printer has jammed on page two again",
        ],
        "query": [
            "The printer jammed again on the second page",
            "The office printer keeps jamming at page two",
        ],
    },
    "cat_windowsill": {
        "encode": [
            "The gray cat sleeps on the sunny windowsill",
            "A gray cat is sleeping on the sunlit windowsill",
            "The gray cat naps on the sunny windowsill",
        ],
        "query": [
            "The gray cat is asleep on the windowsill in the sun",
            "That gray cat keeps sleeping on the sunny windowsill",
        ],
    },
}


def _refuse(msg: str) -> None:
    print(f"[REFUSED — apparatus] {msg}", file=sys.stderr)
    raise SystemExit(4)


def _preflight(json_path: str | None, allow_dirty: bool):
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import maxim
    from _provenance import DirtyTreeError, ProvenanceError, in_process_code_provenance

    try:
        return in_process_code_provenance(_REPO_ROOT, maxim.__file__, out_path=json_path, allow_dirty=allow_dirty)
    except (ProvenanceError, DirtyTreeError) as exc:
        print(f"[FAIL] gated-record preflight: {exc}", file=sys.stderr)
        raise SystemExit(3)


# ---------------------------------------------------------------------------
# Apparatus
# ---------------------------------------------------------------------------


def _build_encoder():
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.similarity.encoder import LinguisticEncoder, require_semantic_encoder

    # The hash fallback is a refusal, not a fallback: its vectors are not the
    # production text geometry, so drift measured on them answers nothing.
    try:
        require_semantic_encoder(ENCODER_MODEL, context="exp-d8 read-mutation measurement")
    except Exception as exc:
        _refuse(f"real semantic encoder unavailable: {exc}")
    scratch_ec = EntorhinalCortex()  # embed() records provenance into its EC
    enc = LinguisticEncoder(ec=scratch_ec, atl=None)
    return enc


def _fresh_ec(freeze_text: bool):
    from dataclasses import replace

    from maxim.similarity.ec import ECConfig, EntorhinalCortex

    config = ECConfig()
    if freeze_text:
        config = replace(
            config,
            frozen_centroid_modalities=frozenset(config.frozen_centroid_modalities) | {"text"},
        )
    return EntorhinalCortex(config=config)


def _encode_store(enc, ec) -> dict[str, str]:
    """Encode the corpus via the production protocol; return concept -> node id.

    Refuses (exit 4) unless every concept forms exactly one node with >= 3
    members and no two concepts share a node — a corpus that shatters or
    collapses is an apparatus failure, not a finding.
    """
    concept_nodes: dict[str, set[str]] = {}
    for concept, texts in CORPUS.items():
        ids: set[str] = set()
        for text in texts["encode"]:
            emb = enc.embed(text)
            geom = enc.geometry_for(emb, "text")
            result = ec.pattern_complete_or_separate(embedding=emb, modality="text", geometry=geom)
            if result.is_new:
                ec.register_substrate_node(result.node_id, emb, "text", geometry=geom)
            ids.add(result.node_id)
        concept_nodes[concept] = ids

    for concept, ids in concept_nodes.items():
        if len(ids) != 1:
            _refuse(f"concept {concept!r} shattered into {len(ids)} nodes (corpus must cluster)")
    flat = [next(iter(ids)) for ids in concept_nodes.values()]
    if len(set(flat)) != len(flat):
        _refuse("two concepts collapsed into one node (corpus must separate)")
    mapping = {concept: next(iter(ids)) for concept, ids in concept_nodes.items()}
    for concept, nid in mapping.items():
        count = ec.substrate_node_metadata(nid)["member_count"]
        if count < 3:
            _refuse(f"concept {concept!r} node has member_count {count} < 3")
    return mapping


def _snapshot(ec) -> dict[str, dict[str, Any]]:
    """{node_id: {embedding, member_count}} via the public metadata accessor."""
    out: dict[str, dict[str, Any]] = {}
    for nid in list(ec._substrate_nodes.keys()):  # noqa: SLF001 — id listing only; values via public accessor
        meta = ec.substrate_node_metadata(nid)
        out[nid] = {"embedding": list(meta["embedding"]), "member_count": meta["member_count"]}
    return out


def _clone_ec(ec):
    """Save/load round-trip so probing never mutates the store under test."""
    from maxim.similarity.ec import EntorhinalCortex

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "ec.json")
        ec.save(path)
        clone = EntorhinalCortex(config=ec.config)
        clone.load(path)
    return clone


def _probe(ec, enc) -> dict[str, str]:
    """Complete every encode text against a CLONE; return text -> node id."""
    clone = _clone_ec(ec)
    out: dict[str, str] = {}
    for concept, texts in CORPUS.items():
        for text in texts["encode"]:
            emb = enc.embed(text)
            geom = enc.geometry_for(emb, "text")
            result = clone.pattern_complete_or_separate(embedding=emb, modality="text", geometry=geom)
            out[text] = result.node_id if not result.is_new else f"SEPARATED:{concept}"
    return out


def _workload(ec, enc, repeats: int) -> dict[str, int]:
    """Replay bio_enrichment's recall shape: query variants, no registration."""
    completions = 0
    separations = 0
    for texts in CORPUS.values():
        for text in texts["query"]:
            for _ in range(repeats):
                emb = enc.embed(text)
                geom = enc.geometry_for(emb, "text")
                result = ec.pattern_complete_or_separate(embedding=emb, modality="text", geometry=geom)
                if result.is_new:
                    separations += 1  # bio_enrichment does not register on is_new
                else:
                    completions += 1
    return {"completions": completions, "separations": separations}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _drift(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, float]:
    cosines = [_cosine(before[nid]["embedding"], after[nid]["embedding"]) for nid in before if nid in after]
    return {"mean_cos": sum(cosines) / len(cosines), "min_cos": min(cosines)}


# ---------------------------------------------------------------------------
# The frozen decision rule — the protocol's own verdict function.
# ---------------------------------------------------------------------------


def decide_verdict(metrics: dict[str, Any]) -> str:
    """separate-required iff churn > 2/48 OR min per-node cosine < 0.98; else accept."""
    if metrics["churned_probes"] > CHURN_MAX_PROBES:
        return "separate-required"
    if metrics["drift"]["min_cos"] < DRIFT_MIN_COS:
        return "separate-required"
    return "accept"


# ---------------------------------------------------------------------------


def run(args) -> int:
    out_path = args.json if args.write_experiment_results else None
    provenance = _preflight(out_path, args.allow_dirty)

    enc = _build_encoder()

    # --- BASELINE arm -----------------------------------------------------
    ec = _fresh_ec(freeze_text=False)
    _encode_store(enc, ec)
    pre_snapshot = _snapshot(ec)
    pre_probe = _probe(ec, enc)
    traffic = _workload(ec, enc, REPEATS)
    post_snapshot = _snapshot(ec)
    post_probe = _probe(ec, enc)

    total_queries = traffic["completions"] + traffic["separations"]
    if total_queries and traffic["completions"] / total_queries < COMPLETE_MIN_FRAC:
        _refuse(
            f"only {traffic['completions']}/{total_queries} workload queries completed "
            "(< 90% — this measures node creation, not reconsolidation)"
        )

    churned = [t for t in pre_probe if pre_probe[t] != post_probe[t]]
    drift = _drift(pre_snapshot, post_snapshot)
    pre_counts = sum(v["member_count"] for v in pre_snapshot.values())
    post_counts = sum(v["member_count"] for v in post_snapshot.values())

    # --- FROZEN instrument arm -------------------------------------------
    ec_frozen = _fresh_ec(freeze_text=True)
    _encode_store(enc, ec_frozen)
    frozen_pre = _snapshot(ec_frozen)
    _workload(ec_frozen, enc, REPEATS)
    frozen_post = _snapshot(ec_frozen)
    # "Exactly zero drift" = BIT-IDENTICAL embeddings (a cosine of a vector
    # with itself rounds to 0.999... in floats, so a cosine==1.0 test would
    # refuse a healthy instrument).
    frozen_moved = [
        nid for nid in frozen_pre if frozen_post.get(nid, {}).get("embedding") != frozen_pre[nid]["embedding"]
    ]
    frozen_drift = _drift(frozen_pre, frozen_post)
    if frozen_moved:
        _refuse(f"FROZEN arm moved {len(frozen_moved)} centroid(s) — the instrument is broken")

    # --- AMPLIFIED instrument arm ----------------------------------------
    ec_amp = _fresh_ec(freeze_text=False)
    _encode_store(enc, ec_amp)
    amp_pre = _snapshot(ec_amp)
    _workload(ec_amp, enc, REPEATS * AMPLIFY_FACTOR)
    amp_drift = _drift(amp_pre, _snapshot(ec_amp))
    if not (amp_drift["mean_cos"] < drift["mean_cos"]):
        _refuse(
            f"AMPLIFIED arm mean drift ({1 - amp_drift['mean_cos']:.6f}) is not greater than "
            f"BASELINE ({1 - drift['mean_cos']:.6f}) — the meter cannot see the effect"
        )

    metrics: dict[str, Any] = {
        "churned_probes": len(churned),
        "churned_texts": churned,
        "n_probes": len(pre_probe),
        "drift": drift,
        "workload": traffic,
        "count_provenance": {
            "encode_counts": pre_counts,
            "post_workload_counts": post_counts,
            "recall_increments": post_counts - pre_counts,
            "note": "structural fact for gate 4 (ec_merge weights by member_count); never folded into the verdict",
        },
        "instrument_arms": {"frozen_drift": frozen_drift, "amplified_drift": amp_drift},
    }
    verdict = decide_verdict(metrics)

    print(f"D8 read-mutation measurement — verdict: {verdict}")
    print(f"  churn: {len(churned)}/{len(pre_probe)} probes (rule: > {CHURN_MAX_PROBES} -> separate)")
    print(
        f"  drift: mean_cos {drift['mean_cos']:.6f}, min_cos {drift['min_cos']:.6f} (rule: min < {DRIFT_MIN_COS} -> separate)"
    )
    print(
        f"  counts: {pre_counts} encode -> {post_counts} post-workload (+{post_counts - pre_counts} from recall; reported, not folded)"
    )
    print(f"  workload: {traffic['completions']} completions / {traffic['separations']} separations")

    if out_path:
        record = {
            "_format_version": "1.0",
            "experiment": "d8_read_mutation",
            "protocol": "docs/experiments/protocols/d8_read_mutation_preregistration.md",
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
            "constants": {
                "repeats": REPEATS,
                "amplify_factor": AMPLIFY_FACTOR,
                "churn_max_probes": CHURN_MAX_PROBES,
                "drift_min_cos": DRIFT_MIN_COS,
                "encoder_model": ENCODER_MODEL,
            },
            "metrics": metrics,
            "verdict": verdict,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        from maxim.utils.atomic_io import atomic_write_json

        atomic_write_json(out_path, record)
        print(f"  wrote {out_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_json = str(
        _REPO_ROOT / "docs" / "experiments" / "data" / f"d8_read_mutation_{datetime.now(timezone.utc):%Y%m%d}.json"
    )
    ap.add_argument(
        "--json", default=default_json, help="gated record path (only written with --write-experiment-results)"
    )
    ap.add_argument(
        "--write-experiment-results",
        action="store_true",
        help="write the gated record (clean tree enforced by the provenance preflight)",
    )
    ap.add_argument("--allow-dirty", action="store_true", help="explicitly allow a dirty tree for the gated write")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
