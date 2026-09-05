"""Bayesian-aggregation tests for the Hivemind ``nac_merge`` + ``ec_merge``.

v1_refinement.md §B5 PR B. Validates the design rules locked
2026-05-30:

1. Zero prior for unobserved entries.
2. Valence-distinct CausalLinks stay separate.
3. EC centroid uses member_count-weighted mean.
4. Cosine threshold default 0.44 matches ``ECConfig``.

Plus the contracts the module docstring promises:

- Pure functions (inputs unchanged after the call).
- Commutativity for merged values (contributor ORDER may differ).
- Idempotence on identical inputs.
- Backward-compat for pre-fold ``count`` key in EC node dicts.
"""

from __future__ import annotations

import pytest

import copy
from typing import Any

from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.hivemind.merge import (
    CONSENSUS_SOURCE,
    NAC_KEY_SEP,
    ec_merge,
    ec_merge_aligned,
    nac_merge_many,
    rekey_nac_state,
    substrate_merge,
    nac_merge,
)
from maxim.similarity.ec import EntorhinalCortex


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


def _link(
    *,
    event_sig: str,
    outcome_sig: str,
    valence: Valence,
    observation_count: int = 1,
    predicted_value: float = 0.6,
    source: str = "local",
    contributors: tuple[str, ...] = (),
    domain: str | None = None,
) -> dict[str, Any]:
    return CausalLink(
        id=f"link-{event_sig}-{outcome_sig}",
        event_type="tool",
        event_signature=event_sig,
        event_context={},
        outcome_type="tool_result",
        outcome_signature=outcome_sig,
        outcome_valence=valence,
        temporal_delta=TemporalDelta(),
        predicted_value=predicted_value,
        observation_count=observation_count,
        source=source,
        contributors=contributors,
        domain=domain,
    ).to_dict()


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": "1.0",
        "links": {},
        "outcome_index": {},
        "priors": {},
        "total_observations": 0,
        "reward_bias": {},
        "goal_reward_bias": {},
        "cluster_reward_bias": {},
        "percept_valences": {},
        "event_outcome_welford": {},
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────
# nac_merge — link merging
# ─────────────────────────────────────────────────────────────────────────


def test_link_unique_to_left_preserved() -> None:
    """A link present only in left passes through unchanged."""
    left = _state(links={"e1": [_link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE)]})
    right = _state()
    merged = nac_merge(left, right, left_source="A", right_source="B")
    assert "e1" in merged["links"]
    assert len(merged["links"]["e1"]) == 1
    # Source unchanged (still local — solo contributor).
    assert merged["links"]["e1"][0]["source"] == "local"


def test_link_pair_aggregates_counts_and_contributors() -> None:
    """A shared link aggregates counts, populates contributors, sets consensus."""
    left = _state(
        links={
            "e1": [
                _link(
                    event_sig="e1",
                    outcome_sig="o1",
                    valence=Valence.POSITIVE,
                    observation_count=4,
                    predicted_value=0.5,
                    source="A",
                ),
            ]
        }
    )
    right = _state(
        links={
            "e1": [
                _link(
                    event_sig="e1",
                    outcome_sig="o1",
                    valence=Valence.POSITIVE,
                    observation_count=6,
                    predicted_value=1.0,
                    source="B",
                ),
            ]
        }
    )
    merged = nac_merge(left, right, left_source="A", right_source="B")
    [link] = merged["links"]["e1"]
    # Counts sum
    assert link["observation_count"] == 10
    # Weighted-mean predicted_value: (4*0.5 + 6*1.0) / 10 = 0.8
    assert link["predicted_value"] == 0.8
    # Source resolves to consensus, contributors carry both
    assert link["source"] == CONSENSUS_SOURCE
    assert tuple(link["contributors"]) == ("A", "B")


def test_valence_distinct_links_stay_separate() -> None:
    """A POSITIVE link from A and a NEGATIVE link from B for the same event
    are kept as TWO distinct links — they have different outcome_signatures
    because the outcome_signature embeds valence.
    """
    left = _state(
        links={
            "e1": [
                _link(event_sig="e1", outcome_sig="o1:positive", valence=Valence.POSITIVE, source="A"),
            ]
        }
    )
    right = _state(
        links={
            "e1": [
                _link(event_sig="e1", outcome_sig="o1:negative", valence=Valence.NEGATIVE, source="B"),
            ]
        }
    )
    merged = nac_merge(left, right, left_source="A", right_source="B")
    assert len(merged["links"]["e1"]) == 2
    valences = {link["outcome_valence"] for link in merged["links"]["e1"]}
    assert valences == {"positive", "negative"}


# ─────────────────────────────────────────────────────────────────────────
# nac_merge — scalar fields
# ─────────────────────────────────────────────────────────────────────────


def test_reward_bias_unique_to_one_side_preserved() -> None:
    """Zero-prior rule: unique key passes through unchanged, not pulled toward 0."""
    left = _state(reward_bias={"agent1:nodeA": 0.15})
    right = _state(reward_bias={"agent1:nodeB": 0.10})
    merged = nac_merge(left, right, left_source="A", right_source="B")
    assert merged["reward_bias"]["agent1:nodeA"] == 0.15
    assert merged["reward_bias"]["agent1:nodeB"] == 0.10


def test_reward_bias_shared_key_averages_and_clamps() -> None:
    """Shared key: unweighted mean, clamped to [0, max_reward_bias]."""
    left = _state(reward_bias={"agent1:nodeA": 0.18})
    right = _state(reward_bias={"agent1:nodeA": 0.14})
    merged = nac_merge(left, right, left_source="A", right_source="B", max_reward_bias=0.20)
    # Mean is 0.16 — under the 0.20 cap, no clamping needed.
    assert merged["reward_bias"]["agent1:nodeA"] == 0.16


def test_reward_bias_clamps_at_max() -> None:
    """A mean exceeding max_reward_bias clamps to the cap."""
    left = _state(reward_bias={"k": 0.30})
    right = _state(reward_bias={"k": 0.30})
    merged = nac_merge(left, right, left_source="A", right_source="B", max_reward_bias=0.20)
    assert merged["reward_bias"]["k"] == 0.20


def test_percept_valence_signed_clamp() -> None:
    """percept_valences use [-1.0, 1.0] clamp, not the reward_bias cap."""
    left = _state(percept_valences={"k": -1.2})
    right = _state(percept_valences={"k": -0.5})
    merged = nac_merge(left, right, left_source="A", right_source="B")
    # Mean is -0.85, within bounds.
    assert merged["percept_valences"]["k"] == -0.85


# ─────────────────────────────────────────────────────────────────────────
# Welford parallel merge
# ─────────────────────────────────────────────────────────────────────────


def test_welford_parallel_merge_chan_formula() -> None:
    """Verify Chan's parallel-Welford formula by comparison to a hand calc.

    Stream [1.0, 2.0, 3.0]: n=3, mean=2.0, m2=2.0 (sum of (x-mean)^2).
    Stream [4.0, 5.0]: n=2, mean=4.5, m2=0.5.
    Merged: n=5, mean=3.0, m2=10.0.
    """
    left = _state(event_outcome_welford={"k": {"mean": 2.0, "m2": 2.0, "n": 3.0}})
    right = _state(event_outcome_welford={"k": {"mean": 4.5, "m2": 0.5, "n": 2.0}})
    merged = nac_merge(left, right, left_source="A", right_source="B")
    w = merged["event_outcome_welford"]["k"]
    assert w["n"] == 5.0
    assert w["mean"] == 3.0
    assert w["m2"] == 10.0


# ─────────────────────────────────────────────────────────────────────────
# nac_merge — invariants
# ─────────────────────────────────────────────────────────────────────────


def test_purity_inputs_not_mutated() -> None:
    """The function must not mutate its input dicts."""
    left = _state(links={"e1": [_link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, source="A")]})
    right = _state(links={"e1": [_link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, source="B")]})
    left_copy = copy.deepcopy(left)
    right_copy = copy.deepcopy(right)
    nac_merge(left, right, left_source="A", right_source="B")
    assert left == left_copy
    assert right == right_copy


def test_commutativity_for_merged_values() -> None:
    """``nac_merge(a, b) == nac_merge(b, a)`` modulo contributor order."""
    a = _state(
        reward_bias={"k": 0.10},
        links={"e1": [_link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, source="A")]},
    )
    b = _state(
        reward_bias={"k": 0.16},
        links={"e1": [_link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, source="B")]},
    )
    ab = nac_merge(a, b, left_source="A", right_source="B")
    ba = nac_merge(b, a, left_source="B", right_source="A")
    # Scalar fields commute exactly.
    assert ab["reward_bias"] == ba["reward_bias"]
    # Link merged values commute; contributor SET is identical, ORDER may differ.
    [ab_link] = ab["links"]["e1"]
    [ba_link] = ba["links"]["e1"]
    assert ab_link["observation_count"] == ba_link["observation_count"]
    assert ab_link["predicted_value"] == ba_link["predicted_value"]
    assert set(ab_link["contributors"]) == set(ba_link["contributors"])


def test_idempotence_on_identical_inputs() -> None:
    """``nac_merge(a, a, left_source=X, right_source=X)`` is a stable transform.

    After dedup the contributors collapse to ``(X,)``; counts sum;
    means stay identical (mean(v, v) = v).
    """
    a = _state(
        reward_bias={"k": 0.10},
        links={
            "e1": [_link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, observation_count=3, source="X")]
        },
    )
    merged = nac_merge(a, copy.deepcopy(a), left_source="X", right_source="X")
    # Scalar values unchanged.
    assert merged["reward_bias"]["k"] == 0.10
    # Link counts SUM (treating each side as separate observations).
    [link] = merged["links"]["e1"]
    assert link["observation_count"] == 6
    # Contributors dedup to a single entry.
    assert tuple(link["contributors"]) == ("X",)
    # Source stays singular (one contributor → not consensus).
    assert link["source"] == "X"


def test_round_trip_through_nac_load_state() -> None:
    """A merged state dict loads cleanly into a live NAc.

    Construct two NAcs, dump each, merge the dumps, load the merged
    state into a fresh NAc, and verify a representative key persists.
    """
    nac_a = NAc(config=NACConfig())
    nac_a._reward_bias[("agent1", "node-1")] = 0.12
    nac_b = NAc(config=NACConfig())
    nac_b._reward_bias[("agent1", "node-1")] = 0.16

    merged = nac_merge(nac_a.dump(), nac_b.dump(), left_source="A", right_source="B")
    nac_out = NAc(config=NACConfig())
    nac_out.load_state(merged)
    assert nac_out._reward_bias[("agent1", "node-1")] == 0.14  # mean of 0.12 + 0.16


# ─────────────────────────────────────────────────────────────────────────
# ec_merge
# ─────────────────────────────────────────────────────────────────────────


def _ec_node(
    *,
    embedding: list[float],
    modality: str = "text",
    member_count: int = 1,
    source: str = "local",
    domain: str | None = None,
    contributors: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "embedding": embedding,
        "modality": modality,
        "member_count": member_count,
        "source": source,
        "domain": domain,
        "contributors": list(contributors),
    }


def test_ec_merge_unique_nodes_preserved() -> None:
    """Distinct modalities can never match — both nodes pass through."""
    left = {"n1": _ec_node(embedding=[1.0, 0.0], modality="text", source="A")}
    right = {"n2": _ec_node(embedding=[1.0, 0.0], modality="vision", source="B")}
    merged = ec_merge(left, right, left_source="A", right_source="B")
    assert set(merged.keys()) == {"n1", "n2"}


def test_ec_merge_cosine_match_weighted_centroid() -> None:
    """Two nodes with identical-direction embeddings merge; centroid is
    member_count-weighted mean.

    Left node: embedding [1,0,0], member_count=10
    Right node: embedding [3,0,0] (same direction, magnitude irrelevant for cosine),
                member_count=5
    Expected centroid: (10*[1,0,0] + 5*[3,0,0]) / 15 = [25/15, 0, 0] ≈ [1.667, 0, 0]
    """
    left = {"n1": _ec_node(embedding=[1.0, 0.0, 0.0], member_count=10, source="A")}
    right = {"n2": _ec_node(embedding=[3.0, 0.0, 0.0], member_count=5, source="B")}
    merged = ec_merge(left, right, left_source="A", right_source="B", cosine_threshold=0.5)
    # The merged node is keyed under the left ID since right matched into it.
    assert "n1" in merged
    assert "n2" not in merged
    merged_emb = merged["n1"]["embedding"]
    assert abs(merged_emb[0] - 25.0 / 15.0) < 1e-9
    assert merged_emb[1:] == [0.0, 0.0]
    assert merged["n1"]["member_count"] == 15
    assert merged["n1"]["source"] == CONSENSUS_SOURCE
    assert tuple(merged["n1"]["contributors"]) == ("A", "B")


def test_ec_merge_below_threshold_inserts_new() -> None:
    """When the best cosine is below threshold, the right node lands as
    a new entry (no merge).
    """
    left = {"n1": _ec_node(embedding=[1.0, 0.0, 0.0], source="A")}
    right = {"n2": _ec_node(embedding=[0.0, 1.0, 0.0], source="B")}  # orthogonal
    merged = ec_merge(left, right, left_source="A", right_source="B", cosine_threshold=0.5)
    assert set(merged.keys()) == {"n1", "n2"}
    assert merged["n1"]["source"] == "A"
    assert merged["n2"]["source"] == "B"


def test_ec_merge_id_collision_suffixed() -> None:
    """Two contributors with the same node_id that DON'T pattern-match
    must coexist — the right side's node is suffixed with its source.
    """
    left = {"shared": _ec_node(embedding=[1.0, 0.0, 0.0], source="A")}
    right = {"shared": _ec_node(embedding=[0.0, 1.0, 0.0], source="B")}  # orthogonal
    merged = ec_merge(left, right, left_source="A", right_source="B", cosine_threshold=0.5)
    assert "shared" in merged
    assert "shared#B" in merged
    assert merged["shared"]["source"] == "A"
    assert merged["shared#B"]["source"] == "B"


def test_ec_merge_accepts_pre_fold_count_key() -> None:
    """Backward-compat: input nodes using the pre-fold ``count`` key
    (instead of ``member_count``) still merge correctly.
    """
    # Both sides use the old "count" key.
    left = {"n1": {"embedding": [1.0, 0.0, 0.0], "modality": "text", "count": 4, "source": "A"}}
    right = {"n2": {"embedding": [1.0, 0.0, 0.0], "modality": "text", "count": 6, "source": "B"}}
    merged = ec_merge(left, right, left_source="A", right_source="B", cosine_threshold=0.5)
    # They match (identical direction) — merged into n1.
    assert "n1" in merged
    # Output uses the canonical member_count key.
    assert merged["n1"]["member_count"] == 10


def test_ec_merge_round_trip_through_ec_load() -> None:
    """A merged substrate_nodes dict round-trips through the EC save/load path.

    Build two ECs, register a node in each, save both, slice out the
    substrate_nodes from each payload, merge, and reload into a fresh EC.
    """
    ec_a = EntorhinalCortex()
    ec_a.register_substrate_node("node-a", [1.0, 0.0, 0.0], "text", source="A", domain="combat")
    ec_b = EntorhinalCortex()
    ec_b.register_substrate_node("node-b", [0.99, 0.0, 0.0], "text", source="B", domain="combat")

    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        path_a = Path(td) / "ec_a.json"
        ec_a.save(str(path_a))
        path_b = Path(td) / "ec_b.json"
        ec_b.save(str(path_b))

        left_nodes = json.loads(path_a.read_text())["substrate_nodes"]
        right_nodes = json.loads(path_b.read_text())["substrate_nodes"]

        merged_nodes = ec_merge(left_nodes, right_nodes, left_source="A", right_source="B")
        # Reload via a synthetic EC payload.
        merged_payload = {
            "_format_version": "1.0",
            "version": "1.0",
            "config": {
                "num_lsh_tables": 4,
                "bits_per_table": 8,
                "default_k": 10,
                "min_similarity": 0.3,
                "enable_semantic": False,
            },
            "lsh": {},
            "inverted": {},
            "signatures": {},
            "substrate_nodes": merged_nodes,
        }
        merged_path = Path(td) / "merged.json"
        merged_path.write_text(json.dumps(merged_payload))

        ec_out = EntorhinalCortex()
        ec_out.load(str(merged_path))

    # The merge collapsed the two text nodes (cosine ≈ 1.0) into one.
    assert ec_out.substrate_node_count == 1
    # The surviving node carries CONSENSUS_SOURCE and both contributors.
    [(nid, _)] = list(ec_out._substrate_nodes.items())
    meta = ec_out.substrate_node_metadata(nid)
    assert meta is not None
    assert meta["source"] == CONSENSUS_SOURCE
    assert meta["domain"] == "combat"
    # CRITICAL regression guard (pre-fold bug): the on-disk ``count`` key
    # and the metadata-accessor ``member_count`` alias must both survive
    # the merge → save → load round-trip. Pre-fold, ``ec_merge`` emitted
    # only ``member_count``, so ``EC.load()`` reset count to 1.
    assert meta["member_count"] == 2


# ─────────────────────────────────────────────────────────────────────────
# Fold regression guards — CRITICAL clamp ranges
# ─────────────────────────────────────────────────────────────────────────


def test_goal_reward_bias_signed_clamp_preserves_negative() -> None:
    """goal_reward_bias is SIGNED — negative values are no-go signals and
    must survive the merge.

    Pre-fold the unsigned clamp ``[0, max_reward_bias]`` clipped every
    negative goal bias to 0, silently destroying ThoughtGate
    "skip deliberation" learning.
    """
    left = _state(goal_reward_bias={"goal1": -0.15})
    right = _state(goal_reward_bias={"goal1": -0.05})
    merged = nac_merge(left, right, left_source="A", right_source="B")
    # Mean is -0.10, within signed clamp [-0.20, +0.20].
    assert merged["goal_reward_bias"]["goal1"] == -0.10


def test_cluster_reward_bias_uses_max_cluster_param() -> None:
    """cluster_reward_bias uses NACConfig's separate max_cluster_reward_bias
    (default 1.0, 5× wider than max_reward_bias=0.20).

    Pre-fold the 0.20 cap silently compressed Wire-A's primary
    action-selection signal by 5×.
    """
    left = _state(cluster_reward_bias={"k": 0.80})
    right = _state(cluster_reward_bias={"k": 0.60})
    merged = nac_merge(left, right, left_source="A", right_source="B")
    # Default max_cluster_reward_bias=1.0; mean 0.70 is well within.
    assert merged["cluster_reward_bias"]["k"] == 0.70


def test_cluster_reward_bias_clamps_at_max_cluster_param() -> None:
    """At the cluster cap, the merge clamps correctly."""
    left = _state(cluster_reward_bias={"k": 1.4})
    right = _state(cluster_reward_bias={"k": 1.4})
    merged = nac_merge(left, right, left_source="A", right_source="B", max_cluster_reward_bias=1.0)
    assert merged["cluster_reward_bias"]["k"] == 1.0


# ─────────────────────────────────────────────────────────────────────────
# Fold regression guard — IMPORTANT temporal_delta union
# ─────────────────────────────────────────────────────────────────────────


def test_temporal_delta_unions_observed_deltas() -> None:
    """``observed_deltas`` from BOTH sides survive the link merge.

    Pre-fold, only left's deltas survived — right's observations
    vanished silently, breaking B2 oscillator imminence prediction.
    """
    left_link = _link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, source="A")
    left_link["temporal_delta"] = {"observed_deltas": [1.0, 2.0]}
    right_link = _link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, source="B")
    right_link["temporal_delta"] = {"observed_deltas": [3.0, 4.0]}
    left = _state(links={"e1": [left_link]})
    right = _state(links={"e1": [right_link]})
    merged = nac_merge(left, right, left_source="A", right_source="B")
    [merged_link] = merged["links"]["e1"]
    assert merged_link["temporal_delta"]["observed_deltas"] == [1.0, 2.0, 3.0, 4.0]


def test_temporal_delta_truncated_to_100() -> None:
    """Merged observed_deltas truncate to the last 100 per the existing
    TemporalDelta.add_observation invariant.
    """
    left_link = _link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, source="A")
    left_link["temporal_delta"] = {"observed_deltas": list(range(60))}  # 60 entries
    right_link = _link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, source="B")
    right_link["temporal_delta"] = {"observed_deltas": list(range(60, 120))}  # 60 entries
    left = _state(links={"e1": [left_link]})
    right = _state(links={"e1": [right_link]})
    merged = nac_merge(left, right, left_source="A", right_source="B")
    [merged_link] = merged["links"]["e1"]
    deltas = merged_link["temporal_delta"]["observed_deltas"]
    assert len(deltas) == 100
    # Trailing 100 of the concatenated 120-entry list = [20..119].
    assert deltas[0] == 20
    assert deltas[-1] == 119


# ─────────────────────────────────────────────────────────────────────────
# Fold regression guard — IMPORTANT reserved-source namespace
# ─────────────────────────────────────────────────────────────────────────


def test_left_source_rejects_reserved_prefix() -> None:
    """Contributor IDs in the reserved ``_*`` namespace are rejected at
    entry so they can't shadow sentinel markers like CONSENSUS_SOURCE.
    """
    import pytest

    with pytest.raises(ValueError, match="reserved prefix"):
        nac_merge(_state(), _state(), left_source="_consensus", right_source="B")


def test_right_source_rejects_reserved_prefix() -> None:
    import pytest

    with pytest.raises(ValueError, match="reserved prefix"):
        nac_merge(_state(), _state(), left_source="A", right_source="_evil")


def test_ec_merge_rejects_reserved_prefix() -> None:
    import pytest

    with pytest.raises(ValueError, match="reserved prefix"):
        ec_merge({}, {}, left_source="_consensus", right_source="B")


def test_consensus_source_is_underscore_prefixed() -> None:
    """The sentinel value lives in the reserved namespace by design."""
    assert CONSENSUS_SOURCE.startswith("_")


# ─────────────────────────────────────────────────────────────────────────
# Fold regression guard — IMPORTANT poison-resistance hook reservations
# ─────────────────────────────────────────────────────────────────────────


def test_trusted_sources_filters_untrusted_links() -> None:
    """``trusted_sources`` is the 1.2 poison-resistance reservation — when
    non-None, right-side links whose contributor set is not a subset of
    the trusted set are dropped before merge.
    """
    left = _state(
        links={"e1": [_link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, source="A")]},
    )
    right = _state(
        links={
            "e1": [
                _link(event_sig="e1", outcome_sig="o2", valence=Valence.POSITIVE, source="trusted"),
                _link(event_sig="e1", outcome_sig="o3", valence=Valence.POSITIVE, source="untrusted"),
            ]
        }
    )
    merged = nac_merge(
        left,
        right,
        left_source="A",
        right_source="trusted",
        trusted_sources=frozenset({"A", "trusted"}),
    )
    outcome_sigs = {link["outcome_signature"] for link in merged["links"]["e1"]}
    # left's o1 + right's o2 admitted; right's o3 (source="untrusted") dropped.
    assert outcome_sigs == {"o1", "o2"}


def test_validate_link_filters_callback() -> None:
    """``validate_link`` callback rejects links whose return is falsy."""
    left = _state()
    right = _state(
        links={
            "e1": [
                _link(event_sig="e1", outcome_sig="o1", valence=Valence.POSITIVE, source="B"),
                _link(event_sig="e1", outcome_sig="o2", valence=Valence.POSITIVE, source="B"),
            ]
        }
    )
    merged = nac_merge(
        left,
        right,
        left_source="A",
        right_source="B",
        validate_link=lambda ld: ld["outcome_signature"] != "o2",
    )
    outcome_sigs = {link["outcome_signature"] for link in merged["links"].get("e1", [])}
    assert outcome_sigs == {"o1"}


# ─────────────────────────────────────────────────────────────────────────
# Fold regression guard — NICE determinism
# ─────────────────────────────────────────────────────────────────────────


def test_nac_merge_output_iteration_order_is_sorted() -> None:
    """Sorted key iteration produces bit-identical output across runs.

    Matters for PR D bundle hashing (signature + content addressing).
    """
    left = _state(
        links={
            "z_event": [_link(event_sig="z_event", outcome_sig="o1", valence=Valence.POSITIVE)],
            "a_event": [_link(event_sig="a_event", outcome_sig="o1", valence=Valence.POSITIVE)],
        }
    )
    right = _state(
        links={
            "m_event": [_link(event_sig="m_event", outcome_sig="o1", valence=Valence.POSITIVE)],
        }
    )
    merged = nac_merge(left, right, left_source="A", right_source="B")
    assert list(merged["links"].keys()) == ["a_event", "m_event", "z_event"]


# ─────────────────────────────────────────────────────────────────────────
# Cross-encoder-space safety (2026-08-06 design-review findings)
# ─────────────────────────────────────────────────────────────────────────


def test_cosine_returns_zero_on_dimension_mismatch() -> None:
    """Different-length vectors come from different encoder spaces.

    Pre-fix ``zip`` truncated to the shorter vector, so the pair scored a
    PARTIAL cosine over the overlapping prefix.
    """
    from maxim.hivemind.merge import _cosine

    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0, 0.0, 0.0]
    assert _cosine(a, b) == 0.0
    # Same-length still behaves normally.
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_ec_merge_does_not_merge_across_encoder_dimensions() -> None:
    """A 384-dim and a 768-dim node of the SAME modality must stay separate.

    The pre-fix partial cosine over the shared prefix was 1.0 here — far
    above the 0.44 threshold — so the two silently merged and one
    contributor's centroid absorbed a vector from a different space.
    ``ec_merge`` gates on ``modality`` only and EC node payloads carry no
    encoder identity, so nothing else would have caught it.
    """
    left = {"n_small": {"embedding": [1.0, 0.0, 0.0], "modality": "text", "count": 1}}
    right = {"n_big": {"embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "modality": "text", "count": 1}}

    merged = ec_merge(left, right, left_source="A", right_source="B")

    assert set(merged) == {"n_small", "n_big"}, "different-dim nodes must not merge"
    assert merged["n_small"]["embedding"] == [1.0, 0.0, 0.0]
    assert merged["n_big"]["embedding"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_hivemind_frozen_modalities_match_ec_default() -> None:
    """The duplicated literal must track ``ECConfig``'s (pinned, not typed).

    ``merge.py`` deliberately avoids importing internal modules, so the
    frozen-modality set is duplicated. It silently diverged — ``"audio"``
    was in ``ECConfig`` but not here, so a default-argument ``ec_merge``
    running-mean-updated audio centroids across contributors, exactly
    what the local EC forbids for that modality.
    """
    from maxim.hivemind.merge import DEFAULT_FROZEN_CENTROID_MODALITIES
    from maxim.similarity.ec import ECConfig

    assert DEFAULT_FROZEN_CENTROID_MODALITIES == ECConfig().frozen_centroid_modalities


def test_ec_merge_freezes_audio_centroid_by_default() -> None:
    """Audio centroids must not drift across contributors by default."""
    left = {"n1": {"embedding": [1.0, 0.0], "modality": "audio", "count": 1}}
    right = {"n2": {"embedding": [0.9, 0.436], "modality": "audio", "count": 1}}

    merged = ec_merge(left, right, left_source="A", right_source="B")

    assert merged["n1"]["embedding"] == [1.0, 0.0], "audio centroid must stay frozen"
    assert merged["n1"]["count"] == 2, "counts still aggregate"


# ─────────────────────────────────────────────────────────────────────────
# D43 — alignment-preserving merge
# ─────────────────────────────────────────────────────────────────────────


class TestD43AlignmentPreservingMerge:
    """`ec_merge_aligned` returns the map `ec_merge` used to discard."""

    def _nodes(self, spec):
        return {nid: {"embedding": emb, "modality": mod, "count": 1} for nid, (emb, mod) in spec.items()}

    def test_a_folded_node_maps_to_its_survivor(self):
        left = self._nodes({"L1": ([1.0, 0.0, 0.0], "text")})
        right = self._nodes({"R1": ([1.0, 0.0, 0.0], "text")})
        res = ec_merge_aligned(left, right, left_source="A", right_source="B")
        assert res.id_map["R1"] == "L1"
        assert set(res.nodes) == {"L1"}

    def test_an_inserted_node_maps_to_itself(self):
        """Every right-side id gets an entry, so a re-key never silently drops."""
        left = self._nodes({"L1": ([1.0, 0.0, 0.0], "text")})
        right = self._nodes({"R1": ([0.0, 1.0, 0.0], "text")})
        res = ec_merge_aligned(left, right, left_source="A", right_source="B")
        assert res.id_map["R1"] == "R1"

    def test_every_right_id_is_in_the_map(self):
        left = self._nodes({"L1": ([1.0, 0.0, 0.0], "text")})
        right = self._nodes(
            {"R1": ([1.0, 0.0, 0.0], "text"), "R2": ([0.0, 1.0, 0.0], "text"), "R3": ([0.0, 0.0, 1.0], "text")}
        )
        res = ec_merge_aligned(left, right, left_source="A", right_source="B")
        assert set(res.id_map) == {"R1", "R2", "R3"}

    def test_sensor_modalities_align_at_their_own_threshold(self):
        """THE half no plan document named.

        `ec_merge`'s 0.44 default is tuned for paraphrase-mpnet TEXT.
        Interoception clusters — the ones that key `cluster_reward_bias` — are
        formed at 0.85. Two sensor states that are merely *similar* (cos ~0.6)
        must NOT fold, or returning the id map produces a confidently wrong
        alignment, which is strictly worse than today's honestly missing one.
        """
        import math

        theta = math.acos(0.6)
        left = self._nodes({"L1": ([1.0, 0.0], "interoception")})
        right = self._nodes({"R1": ([math.cos(theta), math.sin(theta)], "interoception")})

        res = ec_merge_aligned(left, right, left_source="A", right_source="B")
        assert res.id_map["R1"] == "R1", "cos 0.6 must NOT fold at the sensor threshold 0.85"

        # ...while the same pair folds under the text default, which is exactly
        # the collapse this guards against.
        loose = ec_merge_aligned(
            left, right, left_source="A", right_source="B", modality_thresholds={"interoception": 0.44}
        )
        assert loose.id_map["R1"] == "L1"

    def test_ec_merge_wrapper_keeps_the_old_contract(self):
        left = self._nodes({"L1": ([1.0, 0.0, 0.0], "text")})
        right = self._nodes({"R1": ([1.0, 0.0, 0.0], "text")})
        assert (
            ec_merge(left, right, left_source="A", right_source="B")
            == ec_merge_aligned(left, right, left_source="A", right_source="B").nodes
        )


class TestD43Rekey:
    """`rekey_nac_state` — the fold that makes a foreign want readable."""

    def _state(self, aid="A", cid="C1"):
        sep = NAC_KEY_SEP
        return {
            "cluster_reward_bias": {sep.join((aid, cid, "tool:x")): 0.8},
            "cluster_reward_source": {sep.join((aid, cid, "tool:x")): "operant"},
        }

    def test_cluster_id_is_rewritten_through_the_map(self):
        out = rekey_nac_state(self._state(), {"C1": "SURVIVOR"})
        assert list(out["cluster_reward_bias"]) == [NAC_KEY_SEP.join(("A", "SURVIVOR", "tool:x"))]

    def test_agent_id_normalises_at_the_boundary(self):
        """Closes the second axis without touching a single persisted file."""
        out = rekey_nac_state(self._state(aid="DONOR"), {"C1": "S"}, to_agent_id="RECEIVER")
        assert list(out["cluster_reward_bias"])[0].split(NAC_KEY_SEP)[0] == "RECEIVER"

    def test_an_unmapped_cluster_is_DROPPED_not_passed_through(self):
        """A donor bias whose cluster did not survive names a node the receiver
        cannot reach. Keeping it is what made the union grow while contributing
        exactly 0.0 — dropping is the honest fold."""
        out = rekey_nac_state(self._state(), {})
        assert out["cluster_reward_bias"] == {}

    def test_credit_source_is_rekeyed_alongside_the_bias(self):
        out = rekey_nac_state(self._state(), {"C1": "S"})
        assert list(out["cluster_reward_source"]) == list(out["cluster_reward_bias"])

    def test_input_is_not_mutated(self):
        st = self._state()
        rekey_nac_state(st, {"C1": "S"})
        assert list(st["cluster_reward_bias"])[0].split(NAC_KEY_SEP)[1] == "C1"


class TestD43RestoredState:
    """The merge no longer DELETES the receiver's own state."""

    def test_cluster_reward_source_survives_a_merge(self):
        sep = NAC_KEY_SEP
        k = sep.join(("A", "C", "tool:x"))
        left = {"cluster_reward_source": {k: "operant"}}
        merged = nac_merge(left, {}, left_source="A", right_source="B")
        assert merged["cluster_reward_source"][k] == "operant", "the merge wiped the receiver's own provenance"

    def test_disagreeing_sources_promote_to_mixed_one_way(self):
        sep = NAC_KEY_SEP
        k = sep.join(("A", "C", "tool:x"))
        merged = nac_merge(
            {"cluster_reward_source": {k: "operant"}},
            {"cluster_reward_source": {k: "drive_relief"}},
            left_source="A",
            right_source="B",
        )
        assert merged["cluster_reward_source"][k] == "mixed"

    def test_saved_at_survives_and_keeps_the_younger(self):
        merged = nac_merge(
            {"saved_at": "2026-01-01T00:00:00Z"},
            {"saved_at": "2026-06-01T00:00:00Z"},
            left_source="A",
            right_source="B",
        )
        assert merged["saved_at"] == "2026-06-01T00:00:00Z"


class TestD43NWayFold:
    """N→1 semantics were not undecided — they were decided wrong."""

    def _s(self, v):
        return {"cluster_reward_bias": {NAC_KEY_SEP.join(("A", "C", "tool:x")): v}}

    def test_equal_weight_across_contributors(self):
        """Pairwise left-fold gives the LAST contributor 1/2 the pooled bias at
        N=4 (weights 1/8, 1/8, 1/4, 1/2). Equal weight is the whole point."""
        states = [self._s(0.0), self._s(0.0), self._s(0.0), self._s(1.0)]
        merged = nac_merge_many(states, sources=["a", "b", "c", "d"])
        got = merged["cluster_reward_bias"][NAC_KEY_SEP.join(("A", "C", "tool:x"))]
        assert abs(got - 0.25) < 1e-9, f"expected the equal-weight mean 0.25, got {got}"

    def test_pairwise_fold_would_have_given_the_last_contributor_half(self):
        """Pins the defect this replaces, so the difference is legible."""
        states = [self._s(0.0), self._s(0.0), self._s(0.0), self._s(1.0)]
        merged = states[0]
        for i, st in enumerate(states[1:], start=1):
            merged = nac_merge(merged, st, left_source="a", right_source=f"c{i}")
        got = merged["cluster_reward_bias"][NAC_KEY_SEP.join(("A", "C", "tool:x"))]
        assert abs(got - 0.5) < 1e-9, f"pairwise should over-weight the last contributor, got {got}"

    def test_absence_is_no_evidence_not_a_zero_vote(self):
        """Zero-prior rule preserved: divide by the contributors that HELD the
        key, not by N."""
        merged = nac_merge_many([self._s(1.0), {}], sources=["a", "b"])
        got = merged["cluster_reward_bias"][NAC_KEY_SEP.join(("A", "C", "tool:x"))]
        assert abs(got - 1.0) < 1e-9

    def test_single_state_is_returned_unchanged(self):
        assert nac_merge_many([self._s(0.7)], sources=["a"])["cluster_reward_bias"]

    def test_length_mismatch_is_loud(self):
        with pytest.raises(ValueError, match="length mismatch"):
            nac_merge_many([self._s(0.1), self._s(0.2)], sources=["only-one"])


class TestSubstrateMerge:
    """D43 follow-up: the COMPOSITION, which is what consumers call.

    D43 shipped `ec_merge_aligned`, `rekey_nac_state` and `nac_merge_many` and
    left their composition to call sites — of which there were zero. Every
    shipped consumer still called bare `nac_merge`, so through every real path
    a merged want still read out as 0.0. These pin the composed contract and,
    more importantly, its ORDER, which is the part that is easy to get wrong
    and silent when wrong.
    """

    @staticmethod
    def _nodes(prefix: str, vecs: dict[str, list[float]], modality: str = "interoception") -> dict:
        return {
            f"{prefix}-{k}": {"embedding": v, "modality": modality, "count": 1, "source": prefix, "domain": None}
            for k, v in vecs.items()
        }

    def test_donor_bias_survives_onto_a_receiver_cluster_id(self):
        """The whole point: a re-keyed bias names a node the receiver has."""
        left_nodes = self._nodes("L", {"a": [1.0, 0.0], "b": [0.0, 1.0]})
        right_nodes = self._nodes("R", {"a": [1.0, 0.0], "b": [0.0, 1.0]})
        donor_key = NAC_KEY_SEP.join(("agentR", "R-a", "tool:x"))

        result = substrate_merge(
            receiver_nac={},
            receiver_ec=left_nodes,
            donor_nac={"cluster_reward_bias": {donor_key: 0.9}},
            donor_ec=right_nodes,
            receiver_source="recv",
            donor_source="donor",
            receiver_agent_id="agentL",
        )

        assert result.biases_dropped == 0
        assert result.biases_rekeyed == 1
        merged_keys = list(result.nac["cluster_reward_bias"])
        assert len(merged_keys) == 1
        agent, cluster, tool = merged_keys[0].split(NAC_KEY_SEP)
        assert agent == "agentL", "agent_id must normalise at the ingestion boundary"
        assert cluster in result.ec_nodes, "the re-keyed cluster must be a node in the merged EC"
        assert tool == "tool:x"

    def test_an_unalignable_donor_bias_is_dropped_and_counted(self):
        """Dropping is the honest fold — and it must be VISIBLE.

        D43's defining property was reporting success while contributing
        nothing. A donor cluster that does not survive is dropped rather than
        passed through, and `biases_dropped` says so.
        """
        result = substrate_merge(
            receiver_nac={},
            receiver_ec=self._nodes("L", {"a": [1.0, 0.0]}),
            donor_nac={"cluster_reward_bias": {NAC_KEY_SEP.join(("agentR", "ghost", "tool:x")): 0.9}},
            donor_ec=self._nodes("R", {"a": [1.0, 0.0]}),
            receiver_source="recv",
            donor_source="donor",
        )
        assert result.biases_dropped == 1
        assert result.biases_rekeyed == 0
        assert not result.nac.get("cluster_reward_bias")

    def test_the_receivers_own_biases_are_not_rekeyed(self):
        """Re-keying the receiver would be the same bug in mirror image."""
        recv_key = NAC_KEY_SEP.join(("agentL", "L-a", "tool:x"))
        result = substrate_merge(
            receiver_nac={"cluster_reward_bias": {recv_key: 0.5}},
            receiver_ec=self._nodes("L", {"a": [1.0, 0.0]}),
            donor_nac={},
            donor_ec=self._nodes("R", {"a": [1.0, 0.0]}),
            receiver_source="recv",
            donor_source="donor",
            receiver_agent_id="agentL",
        )
        assert recv_key in result.nac["cluster_reward_bias"]

    def test_unknown_keyword_is_loud(self):
        """kwargs fan out to two different merges — a typo must not vanish."""
        with pytest.raises(TypeError, match="unexpected keyword"):
            substrate_merge(
                receiver_nac={},
                receiver_ec={},
                donor_nac={},
                donor_ec={},
                receiver_source="a",
                donor_source="b",
                cosine_threshhold=0.5,
            )

    def test_interoception_uses_the_sensor_threshold_not_the_text_default(self):
        """The half no plan document named.

        `ec_merge`'s 0.44 default is tuned for paraphrase-mpnet TEXT;
        interoception clusters — the ones that key `cluster_reward_bias` —
        form at 0.85. Two clearly-distinct sensor states must not fold.
        """
        import math

        theta = math.acos(0.6)
        left = self._nodes("L", {"a": [1.0, 0.0]})
        right = self._nodes("R", {"a": [math.cos(theta), math.sin(theta)]})
        result = substrate_merge(
            receiver_nac={},
            receiver_ec=left,
            donor_nac={},
            donor_ec=right,
            receiver_source="recv",
            donor_source="donor",
        )
        assert len(result.ec_nodes) == 2, "cos 0.60 must NOT fold at the interoception threshold 0.85"

        text_left = self._nodes("L", {"a": [1.0, 0.0]}, modality="text")
        text_right = self._nodes("R", {"a": [math.cos(theta), math.sin(theta)]}, modality="text")
        text_result = substrate_merge(
            receiver_nac={},
            receiver_ec=text_left,
            donor_nac={},
            donor_ec=text_right,
            receiver_source="recv",
            donor_source="donor",
        )
        assert len(text_result.ec_nodes) == 1, "cos 0.60 SHOULD fold at the text threshold 0.44"


class TestTightenOnlyClampSeam:
    """The 1.2 poison-resistance clamp at its decided seam (substrate_merge).

    The adapter-level behavioral tests live in test_hivemind_ingest.py;
    these pin the SEAM itself — the clamp fires inside substrate_merge for
    every consumer, and the sign-scope guarantee holds by execution against
    the real taught archive (the gate-6 gauntlet's seed-43 state), not by
    inspection: the pre-clamp fold path is re-composed by hand and the
    post-merge state must be identical.
    """

    ARCHIVE = "docs/experiments/data/53_agents/taught_seed43"

    def _load_archive_pair(self) -> tuple[dict[str, Any], dict[str, Any]]:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / self.ARCHIVE
        nac_state = json.loads((root / "aut_nac.json").read_text())
        ec_payload = json.loads((root / "aut_ec.json").read_text())
        nac_state.pop("_format_version", None)
        return nac_state, ec_payload["substrate_nodes"]

    def test_no_op_clamp_returns_same_object(self) -> None:
        from maxim.hivemind.merge import tighten_negative_biases

        merged = {"cluster_reward_bias": {"k": 0.5}}
        receiver = {"cluster_reward_bias": {"k": 0.4}}
        out, count = tighten_negative_biases(merged, receiver)
        assert out is merged  # no-op case: trivially byte-identical
        assert count == 0

    def test_clamp_restores_receiver_negative(self) -> None:
        from maxim.hivemind.merge import tighten_negative_biases

        merged = {"cluster_reward_bias": {"k": 0.0, "other": 0.3}}
        receiver = {"cluster_reward_bias": {"k": -0.9}}
        out, count = tighten_negative_biases(merged, receiver)
        assert out["cluster_reward_bias"]["k"] == -0.9
        assert out["cluster_reward_bias"]["other"] == 0.3
        assert count == 1
        assert merged["cluster_reward_bias"]["k"] == 0.0  # input not mutated

    def test_taught_seed43_merge_is_byte_untouched_by_the_clamp(self) -> None:
        """The non-negotiable sign-scope guard, BY EXECUTION on real data.

        The taught seed-43 archive carries only positive valence (the
        taught want). Merging it through substrate_merge (clamp active)
        must produce a NAc state identical to the pre-clamp composition
        (ec_merge_aligned + rekey_nac_state + nac_merge, hand-composed
        exactly as substrate_merge did before the clamp landed). A clamp
        leaking into positive folds would silently change benchmark arm 2.
        """
        from maxim.hivemind.bundle import scrub_nac_state_for_bundle

        donor_nac_raw, donor_ec = self._load_archive_pair()
        donor_nac = scrub_nac_state_for_bundle(donor_nac_raw)
        # Two receiver arms: a FRESH receiver and the taught seed-42 peer
        # (the gauntlet's merge pair).
        import json
        from pathlib import Path

        root42 = Path(__file__).resolve().parents[2] / "docs/experiments/data/53_agents/taught_seed42"
        recv_nac = json.loads((root42 / "aut_nac.json").read_text())
        recv_nac.pop("_format_version", None)
        recv_nac = scrub_nac_state_for_bundle(recv_nac)
        recv_ec = json.loads((root42 / "aut_ec.json").read_text())["substrate_nodes"]

        for receiver_nac, receiver_ec in (({}, {}), (recv_nac, recv_ec)):
            result = substrate_merge(
                receiver_nac=receiver_nac,
                receiver_ec=receiver_ec,
                donor_nac=donor_nac,
                donor_ec=donor_ec,
                receiver_source="local",
                donor_source="nursery-43",
                receiver_agent_id="recv",
            )
            # The pre-clamp path, composed by hand (what substrate_merge
            # did before the clamp landed).
            baseline_aligned = ec_merge_aligned(
                receiver_ec,
                donor_ec,
                left_source="local",
                right_source="nursery-43",
            )
            baseline_rekeyed = rekey_nac_state(donor_nac, baseline_aligned.id_map, to_agent_id="recv")
            baseline_nac = nac_merge(
                receiver_nac,
                baseline_rekeyed,
                left_source="local",
                right_source="nursery-43",
            )
            assert result.biases_tightened == 0
            assert result.nac == baseline_nac
            # And the taught want actually landed (not a vacuous equality
            # between two empty folds).
            assert result.nac["cluster_reward_bias"], "taught biases must survive the merge"


class TestInherentKeysTransport:
    """inherent_bias_keys must ride nac_merge and rekey_nac_state (a
    rebuilt-dict merge that dropped the marker would be the D43
    delete-state shape one field over)."""

    def test_nac_merge_unions_inherent_keys(self) -> None:
        key_l = NAC_KEY_SEP.join(("a", "c1", "t"))
        key_r = NAC_KEY_SEP.join(("a", "c2", "t"))
        left = {"inherent_bias_keys": [key_l], "cluster_reward_bias": {key_l: -0.5}}
        right = {"inherent_bias_keys": [key_r], "cluster_reward_bias": {key_r: -0.5}}
        merged = nac_merge(left, right, left_source="l", right_source="r")
        assert merged["inherent_bias_keys"] == sorted([key_l, key_r])

    def test_nac_merge_emits_empty_list_when_neither_side_has_markers(self) -> None:
        merged = nac_merge({}, {}, left_source="l", right_source="r")
        assert merged["inherent_bias_keys"] == []

    def test_rekey_maps_inherent_keys_and_drops_unmapped(self) -> None:
        key_mapped = NAC_KEY_SEP.join(("aut", "c1", "t"))
        key_dropped = NAC_KEY_SEP.join(("aut", "gone", "t"))
        state = {
            "cluster_reward_bias": {key_mapped: -0.5, key_dropped: -0.5},
            "inherent_bias_keys": [key_mapped, key_dropped],
        }
        out = rekey_nac_state(state, {"c1": "local9"}, to_agent_id="recv")
        assert out["inherent_bias_keys"] == [NAC_KEY_SEP.join(("recv", "local9", "t"))]
