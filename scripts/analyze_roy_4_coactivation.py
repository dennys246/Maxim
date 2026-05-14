#!/usr/bin/env python
"""Roy-4 post-hoc co-activation analysis.

Reads ``sim_ec_activation`` events from a Roy-4 run's JSONL log,
computes the pairwise co-activation matrix per ``agent_id`` over the
priming phase, applies a proposed Hebbian binding rule, and reports
whether any test-phase active nodes have would-have-bound edges to
priming ``sense_food_source`` clusters.

This script is the validation gate for Stage 2-6 of
``docs/plans/cross_modal_substrate_binding.md`` (1.1 plan). Per
``release_0_9_1.md`` Stage 0d the answer is binary:

- **PASS** — at least one test-phase active node has a would-have-bound
  edge to a priming sense_food_source cluster. The binding mechanism
  would have closed the Roy-2c gap. Greenlight 1.1 Stages 2-6.
- **FAIL** — no would-have-bound edges between priming and test
  clusters under the proposed rule. Encoder alignment is too severe;
  cancel the binding plan and redirect to a 1.2+ encoder-replacement
  research direction.

The Hebbian rule is the design under test:

- Tick bucket: events sharing the same integer ``tick`` field
  ("same-tick or adjacent-tick co-activation" per the plan's tick-
  discrete temporal semantics section).
- Min co-firing count threshold: pair must co-fire in at least N ticks
  during priming (default 5).
- Salience weighting: each co-firing event contributes
  ``min(activation_a, activation_b)``; the would-have-bound edge weight
  is the sum across all co-firings.
- Min bound-strength threshold: only edges whose summed weight clears a
  floor are kept (default 0.5 — equivalent to a 0.1-activation pair
  co-firing 5 times, OR a 0.5-activation pair co-firing once with
  N=1; the rule documented in the 1.1 plan uses N=5).

Usage::

    python scripts/analyze_roy_4_coactivation.py \\
        --priming-jsonl /tmp/roy_4_ec_trace_priming.jsonl \\
        --arm-a-jsonl /tmp/roy_4_ec_trace_arm_a.jsonl \\
        --arm-b-jsonl /tmp/roy_4_ec_trace_arm_b.jsonl \\
        --arm-c-jsonl /tmp/roy_4_ec_trace_arm_c.jsonl \\
        --priming-tool-clusters /path/to/sense_food_source_cluster_ids.json \\
        --output-json /tmp/roy_4_analysis.json

If the cluster-IDs JSON is omitted, the analyzer infers priming
``sense_food_source`` clusters by reading the priming session's
``aut_nac.json`` file for any cluster whose ``_cluster_reward_bias``
map includes a ``sense_food_source`` entry. Pass ``--priming-nac`` to
point at the NAc dump.

When pointed at a single JSONL file via ``--jsonl``, the script reads
all events and partitions them by ``agent_id`` (priming session ids
have a distinct prefix per Roy runner conventions); the partition
heuristic isn't bulletproof so the recommended path is the explicit
per-arm flag set above.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ─────────────────────────────────────────────────────────────────────────────
# Event ingestion
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EcEvent:
    """One sim_ec_activation record."""

    tick: int
    elapsed_s: float
    agent_id: str | None
    active_node_id: str
    activation_strength: float
    modality_tag: str
    modality: str
    is_new: bool


def _record_is_ec_activation(record: dict[str, Any]) -> bool:
    """Detect a sim_ec_activation record in mixed-event JSONL streams.

    The Roy runner emits a single MAXIM_LOG_FILE JSONL stream containing
    HTTP, peer-backend, role, and bio-system events. We pluck only the
    subset emitted by ``similarity/ec.py::_emit_ec_activation``. Both
    persistence shapes (native sim_log and MAXIM_LOG_FILE bridge) carry
    ``subsystem == "EC_TRACE"`` as a top-level field; the bridge format
    additionally tags ``e == "sim_ec_trace"``.
    """
    if record.get("subsystem") == "EC_TRACE":
        return True
    if record.get("e") == "sim_ec_trace":
        return True
    # Defensive fallback for any future variant nesting under "data".
    data = record.get("data") or {}
    if isinstance(data, dict) and data.get("subsystem") == "EC_TRACE":
        return True
    return False


def _record_to_event(record: dict[str, Any]) -> EcEvent | None:
    """Convert a raw JSONL record to an EcEvent, returning None on miss.

    Two on-disk shapes exist depending on which sink the JSONL came from:

    1. **Native sim_log file** (path passed to ``enable_sim_logging``):
       ``{"t": elapsed_s, "subsystem": "EC_TRACE", "message": ...,
       "data": {"tick": ..., "active_node_id": ..., ...},
       "agent_id": ..., "agent": ...}``
       — all EC fields live inside ``data``.

    2. **MAXIM_LOG_FILE bridge file** (StructuredFormatter):
       ``{"t": wall_clock, "e": "sim_ec_trace", "subsystem": "EC_TRACE",
       "elapsed_s": ..., "tick": ..., "active_node_id": ...,
       "agent_id": ...}``
       — EC fields are flat at the top level; ``elapsed_s`` is explicit.

    Detection: if a top-level ``tick`` key exists, treat as bridge
    format; otherwise read from ``data`` (native sim_log shape).
    """
    if "tick" in record:
        # Bridge format — flat top-level fields
        source = record
        elapsed_s = float(record.get("elapsed_s", record.get("t", 0.0)))
        agent_id = record.get("agent_id")
    else:
        # Native sim_log JSONL — EC fields nested under "data"
        source = record.get("data") or {}
        elapsed_s = float(record.get("t", 0.0))
        agent_id = record.get("agent_id")
    try:
        return EcEvent(
            tick=int(source["tick"]),
            elapsed_s=elapsed_s,
            agent_id=agent_id if agent_id is not None else None,
            active_node_id=str(source["active_node_id"]),
            activation_strength=float(source["activation_strength"]),
            modality_tag=str(source["modality_tag"]),
            modality=str(source["modality"]),
            is_new=bool(source["is_new"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_ec_events(path: Path) -> list[EcEvent]:
    """Stream a JSONL file, return only the EC trace records."""
    events: list[EcEvent] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _record_is_ec_activation(record):
                continue
            ev = _record_to_event(record)
            if ev is not None:
                events.append(ev)
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Co-activation analysis
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CoActivationStats:
    """Stats for one (node_a, node_b) pair (canonically ordered)."""

    pair: tuple[str, str]
    cofire_tick_count: int = 0
    weight_sum: float = 0.0
    modality_pair: tuple[str, str] = ("", "")


@dataclass
class BindingEdge:
    """A would-have-bound edge under the proposed Hebbian rule."""

    node_a: str
    node_b: str
    weight: float
    cofire_tick_count: int
    modality_pair: tuple[str, str]


def _group_by_tick(events: Iterable[EcEvent]) -> dict[int, list[EcEvent]]:
    """Group events by their integer ``tick`` field."""
    by_tick: dict[int, list[EcEvent]] = defaultdict(list)
    for ev in events:
        by_tick[ev.tick].append(ev)
    return by_tick


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Order so ``(a, b)`` and ``(b, a)`` collapse to the same key."""
    return (a, b) if a <= b else (b, a)


def compute_coactivation(
    events: list[EcEvent],
) -> dict[tuple[str, str], CoActivationStats]:
    """Build a pairwise co-activation matrix from one agent's events.

    Same-tick co-activation: nodes a and b co-fire in tick t iff both
    have an EcEvent with ``tick == t``. The contribution to the pair's
    weight is ``min(activation_a, activation_b)``; the pair's
    ``cofire_tick_count`` increments by one per co-firing tick (NOT per
    event count — multiple events of the same node in one tick collapse
    to one node-activation for that tick, taking the max activation).
    """
    # Within a tick, multiple events for the same node collapse to one
    # node-activation using max(activation_strength).
    by_tick = _group_by_tick(events)
    stats: dict[tuple[str, str], CoActivationStats] = {}
    for tick, tick_events in by_tick.items():
        # node → (max_activation, modality_tag)
        node_state: dict[str, tuple[float, str]] = {}
        for ev in tick_events:
            existing = node_state.get(ev.active_node_id)
            if existing is None or ev.activation_strength > existing[0]:
                node_state[ev.active_node_id] = (ev.activation_strength, ev.modality_tag)
        nodes = list(node_state.items())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_a, (act_a, mod_a) = nodes[i]
                node_b, (act_b, mod_b) = nodes[j]
                pair = _canonical_pair(node_a, node_b)
                # Restore the modality pair to match the canonical pair ordering.
                if pair == (node_a, node_b):
                    mod_pair = (mod_a, mod_b)
                else:
                    mod_pair = (mod_b, mod_a)
                stat = stats.setdefault(
                    pair,
                    CoActivationStats(pair=pair, modality_pair=mod_pair),
                )
                stat.cofire_tick_count += 1
                stat.weight_sum += min(act_a, act_b)
    return stats


def apply_hebbian_rule(
    coact: dict[tuple[str, str], CoActivationStats],
    *,
    min_cofire_count: int = 5,
    min_weight: float = 0.5,
) -> list[BindingEdge]:
    """Filter co-activation pairs to the would-have-bound edge list.

    A pair survives iff it co-fires in at least ``min_cofire_count``
    ticks AND its salience-weighted weight sum clears ``min_weight``.
    Returns the edge list sorted by weight descending.
    """
    edges: list[BindingEdge] = []
    for pair, stat in coact.items():
        if stat.cofire_tick_count < min_cofire_count:
            continue
        if stat.weight_sum < min_weight:
            continue
        edges.append(
            BindingEdge(
                node_a=pair[0],
                node_b=pair[1],
                weight=stat.weight_sum,
                cofire_tick_count=stat.cofire_tick_count,
                modality_pair=stat.modality_pair,
            )
        )
    edges.sort(key=lambda e: e.weight, reverse=True)
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Priming sense_food_source cluster inference
# ─────────────────────────────────────────────────────────────────────────────


def load_priming_food_clusters(nac_path: Path) -> set[str]:
    """Read priming aut_nac.json, return EC cluster IDs keyed to sense_food_source.

    The on-disk NAc dump keys ``cluster_reward_bias`` with a unit-
    separator (``\\x1f``) compound::

        ``"{agent_id}\\x1f{cluster_id}\\x1f{tool:sense_food_source}": <bias>``

    The middle field is the EC cluster UUID; we extract it for every
    entry whose third field ends in ``sense_food_source``. The nested
    per-agent shape is preserved as a fallback for synthetic test
    fixtures and any future NAc dump variant.
    """
    if not nac_path.exists():
        return set()
    with nac_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    food_clusters: set[str] = set()

    # Canonical flat-compound shape (current NAc.to_dict format).
    flat = data.get("cluster_reward_bias")
    if isinstance(flat, dict) and flat:
        for compound_key in flat.keys():
            parts = str(compound_key).split("\x1f")
            if len(parts) == 3 and parts[2].endswith("sense_food_source"):
                food_clusters.add(parts[1])
        if food_clusters:
            return food_clusters

    # Legacy / synthetic shapes.
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


# ─────────────────────────────────────────────────────────────────────────────
# Pass/Fail determination
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AnalysisResult:
    """The Roy-4 outcome bundle."""

    pass_criteria_met: bool
    test_arm: str
    priming_food_clusters: list[str] = field(default_factory=list)
    test_phase_active_nodes: list[str] = field(default_factory=list)
    bound_priming_test_pairs: list[BindingEdge] = field(default_factory=list)
    priming_total_events: int = 0
    priming_unique_nodes: int = 0
    priming_total_pairs: int = 0
    priming_bound_edge_count: int = 0
    test_total_events: int = 0
    rule_min_cofire_count: int = 5
    rule_min_weight: float = 0.5


def _binding_edge_to_dict(edge: BindingEdge) -> dict[str, Any]:
    return {
        "node_a": edge.node_a,
        "node_b": edge.node_b,
        "weight": round(edge.weight, 4),
        "cofire_tick_count": edge.cofire_tick_count,
        "modality_pair": list(edge.modality_pair),
    }


def analyze_arm(
    priming_events: list[EcEvent],
    test_events: list[EcEvent],
    *,
    test_arm: str,
    food_clusters: set[str],
    min_cofire_count: int,
    min_weight: float,
) -> AnalysisResult:
    """Compute the pass/fail outcome for one test arm.

    Pass criteria: at least one test-phase active node has a would-have-
    bound edge under the proposed rule that connects to a priming
    sense_food_source cluster.
    """
    coact = compute_coactivation(priming_events)
    bound_edges = apply_hebbian_rule(coact, min_cofire_count=min_cofire_count, min_weight=min_weight)
    test_node_ids = {ev.active_node_id for ev in test_events}
    matched: list[BindingEdge] = []
    for edge in bound_edges:
        # The edge connects two priming-phase nodes. We need to check
        # whether (one endpoint) is in food_clusters AND (the other
        # endpoint OR the same endpoint) appears in test_node_ids.
        a_food = edge.node_a in food_clusters
        b_food = edge.node_b in food_clusters
        a_test = edge.node_a in test_node_ids
        b_test = edge.node_b in test_node_ids
        # Two satisfying shapes:
        #  - food_cluster ↔ test_phase node (direct bind)
        #  - food_cluster ↔ food_cluster where one is also test_phase (rare)
        if (a_food and b_test) or (b_food and a_test):
            matched.append(edge)
        elif a_food and b_food and (a_test or b_test):
            matched.append(edge)
    return AnalysisResult(
        pass_criteria_met=len(matched) > 0,
        test_arm=test_arm,
        priming_food_clusters=sorted(food_clusters),
        test_phase_active_nodes=sorted(test_node_ids),
        bound_priming_test_pairs=matched,
        priming_total_events=len(priming_events),
        priming_unique_nodes=len({e.active_node_id for e in priming_events}),
        priming_total_pairs=len(coact),
        priming_bound_edge_count=len(bound_edges),
        test_total_events=len(test_events),
        rule_min_cofire_count=min_cofire_count,
        rule_min_weight=min_weight,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _print_arm_summary(result: AnalysisResult) -> None:
    print(f"\n=== Arm {result.test_arm} ===")
    print(f"  Priming events:           {result.priming_total_events}")
    print(f"  Priming unique nodes:     {result.priming_unique_nodes}")
    print(f"  Priming co-firing pairs:  {result.priming_total_pairs}")
    print(
        f"  Priming would-have-bound edges (≥{result.rule_min_cofire_count} ticks,"
        f" weight ≥{result.rule_min_weight}): {result.priming_bound_edge_count}"
    )
    print(f"  Priming food clusters:    {len(result.priming_food_clusters)}")
    print(f"  Test events:              {result.test_total_events}")
    print(f"  Test active nodes:        {len(result.test_phase_active_nodes)}")
    print(f"  Matching priming↔test edges: {len(result.bound_priming_test_pairs)}")
    outcome = "PASS" if result.pass_criteria_met else "FAIL"
    print(f"  Outcome: {outcome}")
    if result.bound_priming_test_pairs:
        print("  Top 5 matched edges:")
        for edge in result.bound_priming_test_pairs[:5]:
            print(
                f"    {edge.node_a[:8]}↔{edge.node_b[:8]}"
                f"  w={edge.weight:.3f}  cofire={edge.cofire_tick_count}"
                f"  mod={edge.modality_pair}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Roy-4 post-hoc co-activation analyzer. Determines whether "
            "priming-cluster ↔ test-cluster pairs would have been linked "
            "under a proposed Hebbian rule. PASS → greenlight 1.1 "
            "Stages 2-6 of cross_modal_substrate_binding.md; FAIL → "
            "cancel the binding plan and redirect to 1.2+ encoder "
            "replacement."
        )
    )
    parser.add_argument(
        "--priming-jsonl",
        type=Path,
        required=True,
        help="JSONL log from the priming session (MAXIM_LOG_FILE or sim session log).",
    )
    parser.add_argument(
        "--arm-a-jsonl",
        type=Path,
        required=True,
        help="JSONL log from arm A (substrate-primed, neutral).",
    )
    parser.add_argument(
        "--arm-b-jsonl",
        type=Path,
        default=None,
        help="Optional: JSONL log from arm B (blank, hungry-infant persona).",
    )
    parser.add_argument(
        "--arm-c-jsonl",
        type=Path,
        default=None,
        help="Optional: JSONL log from arm C (blank, neutral).",
    )
    parser.add_argument(
        "--priming-nac",
        type=Path,
        default=None,
        help=("Path to priming session's aut_nac.json — used to identify sense_food_source clusters."),
    )
    parser.add_argument(
        "--priming-food-clusters",
        type=Path,
        default=None,
        help=(
            "Optional: explicit JSON list of priming cluster IDs known "
            "to carry sense_food_source bias. Overrides --priming-nac."
        ),
    )
    parser.add_argument(
        "--min-cofire-count",
        type=int,
        default=5,
        help="Hebbian rule: minimum co-firing tick count for a pair to bind (default 5).",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.5,
        help="Hebbian rule: minimum summed salience weight for a pair to bind (default 0.5).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="If provided, write the full analysis result as JSON to this path.",
    )
    args = parser.parse_args(argv)

    # ── Load food clusters ─────────────────────────────────────────────
    food_clusters: set[str] = set()
    if args.priming_food_clusters is not None:
        with args.priming_food_clusters.open("r", encoding="utf-8") as f:
            food_clusters = set(json.load(f))
        print(f"Loaded {len(food_clusters)} food clusters from {args.priming_food_clusters}")
    elif args.priming_nac is not None:
        food_clusters = load_priming_food_clusters(args.priming_nac)
        print(f"Inferred {len(food_clusters)} food clusters from {args.priming_nac}")
    else:
        print(
            "WARNING: no --priming-nac or --priming-food-clusters supplied; "
            "treating all priming clusters as candidates. The PASS criterion "
            "becomes 'any priming cluster ↔ test cluster bound edge'."
        )

    # ── Load priming events ───────────────────────────────────────────
    priming_events = load_ec_events(args.priming_jsonl)
    print(f"Priming: {len(priming_events)} EC events from {args.priming_jsonl}")
    if not priming_events:
        print(
            "ERROR: priming JSONL contains no sim_ec_activation records. "
            "Confirm MAXIM_EC_TRACE_ACTIVATIONS=1 was set in the runner "
            "environment and that the JSONL path matches MAXIM_LOG_FILE."
        )
        return 2

    # If we have no food cluster filter, fall back to "any priming node
    # that appears more than once" as a coarse signal of substrate-acquired
    # cluster — better than empty.
    if not food_clusters:
        node_counts: dict[str, int] = defaultdict(int)
        for ev in priming_events:
            node_counts[ev.active_node_id] += 1
        food_clusters = {n for n, c in node_counts.items() if c >= 2}
        print(f"  Fallback: treating {len(food_clusters)} priming nodes (those firing ≥2 times) as cluster candidates.")

    # ── Per-arm analysis ──────────────────────────────────────────────
    arm_paths = [
        ("a", args.arm_a_jsonl),
        ("b", args.arm_b_jsonl),
        ("c", args.arm_c_jsonl),
    ]
    arm_results: list[AnalysisResult] = []
    for arm_label, path in arm_paths:
        if path is None:
            continue
        test_events = load_ec_events(path)
        print(f"Arm {arm_label}: {len(test_events)} EC events from {path}")
        result = analyze_arm(
            priming_events=priming_events,
            test_events=test_events,
            test_arm=arm_label,
            food_clusters=food_clusters,
            min_cofire_count=args.min_cofire_count,
            min_weight=args.min_weight,
        )
        arm_results.append(result)

    # ── Headline outcome (gated on arm A, the substrate-primed arm) ──
    arm_a = next((r for r in arm_results if r.test_arm == "a"), None)
    if arm_a is None:
        print("ERROR: no arm A result; cannot determine PASS/FAIL.", file=sys.stderr)
        return 2

    print("\n" + "=" * 60)
    if arm_a.pass_criteria_met:
        print("ROY-4 OUTCOME: PASS")
        print("Cross-modal binding plan is JUSTIFIED. Stages 2-6 of cross_modal_substrate_binding.md greenlit for 1.1.")
    else:
        print("ROY-4 OUTCOME: FAIL")
        print(
            "Encoder alignment is too severe for Hebbian binding alone. "
            "Cancel cross_modal_substrate_binding.md Stages 2-6; redirect "
            "to a 1.2+ encoder-replacement research direction."
        )
    print("=" * 60)

    for result in arm_results:
        _print_arm_summary(result)

    # ── Optional JSON output ──────────────────────────────────────────
    if args.output_json is not None:
        bundle = {
            "rule": {
                "min_cofire_count": args.min_cofire_count,
                "min_weight": args.min_weight,
            },
            "priming_food_clusters": sorted(food_clusters),
            "arms": {
                r.test_arm: {
                    "pass_criteria_met": r.pass_criteria_met,
                    "priming_total_events": r.priming_total_events,
                    "priming_unique_nodes": r.priming_unique_nodes,
                    "priming_total_pairs": r.priming_total_pairs,
                    "priming_bound_edge_count": r.priming_bound_edge_count,
                    "test_total_events": r.test_total_events,
                    "test_active_node_count": len(r.test_phase_active_nodes),
                    "matched_edges": [_binding_edge_to_dict(e) for e in r.bound_priming_test_pairs],
                }
                for r in arm_results
            },
            "outcome": "PASS" if arm_a.pass_criteria_met else "FAIL",
        }
        args.output_json.write_text(json.dumps(bundle, indent=2, sort_keys=True))
        print(f"\nWrote analysis bundle to {args.output_json}")

    return 0 if arm_a.pass_criteria_met else 1


if __name__ == "__main__":
    sys.exit(main())
