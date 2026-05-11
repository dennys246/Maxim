"""Unit tests for the substrate divergence analyzer.

Roy harness session R2 of 5 — proves the analyzer:
1. detects a non-zero L2 divergence when one ``reward_bias`` key
   differs between two synthetic NAc snapshots,
2. returns zero divergence when the snapshots are identical,
3. returns ``available=False`` (without raising) when one side is
   missing the snapshot file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxim.analysis.substrate_diff import (
    AtlDiff,
    EcDiff,
    HippocampusDiff,
    NacDiff,
    atl_diff,
    ec_diff,
    hippocampus_diff,
    nac_diff,
    substrate_diff,
    substrate_diff_to_json,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _nac_payload(reward_bias: dict[str, float]) -> dict:
    return {
        "_format_version": "1.0",
        "version": "1.0",
        "links": {},
        "outcome_index": {},
        "priors": {},
        "total_observations": 0,
        "reward_bias": reward_bias,
        "goal_reward_bias": {},
    }


def test_nac_diff_detects_differing_percept_valence(tmp_path: Path) -> None:
    """Two NAc snapshots that differ on a single key produce a non-zero L2."""
    dir_a = tmp_path / "session_a"
    dir_b = tmp_path / "session_b"

    payload_a = _nac_payload({"agentA:node_x": 0.50, "agentA:node_y": 0.30})
    # Add a reserved 1.1 percept_valences field on both sides — the
    # analyzer should compute its L2 when both sides expose it.
    payload_a["percept_valences"] = {
        "agentA:guard:betrayal": 0.80,
        "agentA:trader:cooperate": 0.10,
    }
    payload_b = _nac_payload({"agentA:node_x": 0.10, "agentA:node_y": 0.30})
    payload_b["percept_valences"] = {
        "agentA:guard:betrayal": 0.20,
        "agentA:trader:cooperate": 0.10,
    }

    _write_json(dir_a / "aut_nac.json", payload_a)
    _write_json(dir_b / "aut_nac.json", payload_b)

    diff = nac_diff(dir_a, dir_b)
    assert isinstance(diff, NacDiff)
    assert diff.available is True
    # Single key differs by 0.40 → L2 == 0.40
    assert diff.reward_bias_l2 == pytest.approx(0.40, abs=1e-9)
    # Top delta points at the differing key with the right sign.
    top_key, top_delta = diff.reward_bias_top_deltas[0]
    assert top_key == "agentA:node_x"
    assert top_delta == pytest.approx(0.40, abs=1e-9)
    # _percept_valences support: surfaces when both sides provide it.
    assert diff.percept_valence_available is True
    assert diff.percept_valence_l2 == pytest.approx(0.60, abs=1e-9)


def test_nac_diff_identical_snapshots_yield_zero(tmp_path: Path) -> None:
    """Identical NAc snapshots produce zero divergence and empty top-delta lists."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    payload = _nac_payload({"agentA:node_x": 0.50, "agentA:node_y": 0.30})
    payload["goal_reward_bias"] = {"survive": 0.20}

    _write_json(dir_a / "aut_nac.json", payload)
    _write_json(dir_b / "aut_nac.json", payload)

    diff = nac_diff(dir_a, dir_b)
    assert diff.available is True
    assert diff.reward_bias_l2 == 0.0
    assert diff.reward_bias_top_deltas == ()
    assert diff.goal_reward_bias_l2 == 0.0
    assert diff.causal_link_count_delta == 0
    # _percept_valences absent on both sides → marked unavailable, never
    # raises.
    assert diff.percept_valence_available is False


def test_nac_diff_missing_file_returns_unavailable(tmp_path: Path) -> None:
    """Missing ``aut_nac.json`` on one side returns available=False, no raise."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_json(dir_a / "aut_nac.json", _nac_payload({"agentA:node_x": 0.5}))
    # No file under dir_b.

    diff = nac_diff(dir_a, dir_b)
    assert isinstance(diff, NacDiff)
    assert diff.available is False
    assert "missing" in diff.reason.lower()
    # __str__ must work on unavailable diffs without raising.
    assert "available=false" in str(diff)


def test_hippocampus_diff_valence_distribution(tmp_path: Path) -> None:
    """KS statistic should be > 0 when valence distributions clearly differ."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    episodes_a = [
        {
            "id": f"ep_{i}",
            "start_tick": i,
            "end_tick": i,
            "channel": "c",
            "sender_ids": [],
            "thread_id": None,
            "activated_nodes": [],
            "reward_events": [],
            "scn_tag": None,
            "valence": 0.8,
        }
        for i in range(20)
    ]
    episodes_b = [
        {
            "id": f"ep_{i}",
            "start_tick": i,
            "end_tick": i,
            "channel": "c",
            "sender_ids": [],
            "thread_id": None,
            "activated_nodes": [],
            "reward_events": [],
            "scn_tag": None,
            "valence": -0.8,
        }
        for i in range(20)
    ]
    memories = [{"id": "m1", "timestamp": 0.0, "perception": {"salience": 0.5}}]

    payload_a = {"_format_version": "1.0", "memories": memories, "episodes": episodes_a}
    payload_b = {"_format_version": "1.0", "memories": memories, "episodes": episodes_b}
    _write_json(dir_a / "aut_hippocampus.json", payload_a)
    _write_json(dir_b / "aut_hippocampus.json", payload_b)

    diff = hippocampus_diff(dir_a, dir_b)
    assert isinstance(diff, HippocampusDiff)
    assert diff.available is True
    assert diff.episode_count_a == 20
    assert diff.episode_count_b == 20
    # Disjoint distributions → KS == 1.0
    assert diff.valence_ks_statistic == pytest.approx(1.0, abs=1e-9)
    assert diff.valence_mean_a == pytest.approx(0.8, abs=1e-9)
    assert diff.valence_mean_b == pytest.approx(-0.8, abs=1e-9)


def test_ec_diff_modality_and_pairs(tmp_path: Path) -> None:
    """EC diff reports node count, modality histogram, and top cosine pairs."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    nodes_a = {
        "n_text_1": {"embedding": [1.0, 0.0, 0.0], "modality": "text", "count": 1},
        "n_text_2": {"embedding": [0.0, 1.0, 0.0], "modality": "text", "count": 1},
    }
    nodes_b = {
        "n_text_1b": {"embedding": [0.99, 0.01, 0.0], "modality": "text", "count": 1},
        "n_vision_1": {"embedding": [0.0, 0.0, 1.0], "modality": "vision", "count": 1},
    }
    payload_a = {"_format_version": "1.0", "version": "1.0", "substrate_nodes": nodes_a}
    payload_b = {"_format_version": "1.0", "version": "1.0", "substrate_nodes": nodes_b}
    _write_json(dir_a / "ec.json", payload_a)
    _write_json(dir_b / "ec.json", payload_b)

    diff = ec_diff(dir_a, dir_b, top_n=3)
    assert isinstance(diff, EcDiff)
    assert diff.available is True
    assert diff.node_count_a == 2
    assert diff.node_count_b == 2
    assert diff.modality_histogram_a == {"text": 2}
    assert diff.modality_histogram_b == {"text": 1, "vision": 1}
    # Top pair should be the near-collinear text vectors.
    assert diff.closest_node_pairs[0][0] == "n_text_1"
    assert diff.closest_node_pairs[0][1] == "n_text_1b"
    assert diff.closest_node_pairs[0][2] == pytest.approx(0.99995, abs=1e-3)


def test_atl_diff_name_overlap(tmp_path: Path) -> None:
    """ATL diff reports concept count delta + Jaccard over concept names."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    concepts_a = {
        "c1": {"id": "c1", "timestamp": 0.0, "name": "guard", "category": "person"},
        "c2": {"id": "c2", "timestamp": 0.0, "name": "trader", "category": "person"},
    }
    concepts_b = {
        "c1": {"id": "c1", "timestamp": 0.0, "name": "guard", "category": "person"},
        "c3": {"id": "c3", "timestamp": 0.0, "name": "scout", "category": "person"},
    }
    _write_json(dir_a / "atl.json", {"_format_version": "1.0", "concepts": concepts_a})
    _write_json(dir_b / "atl.json", {"_format_version": "1.0", "concepts": concepts_b})

    diff = atl_diff(dir_a, dir_b)
    assert isinstance(diff, AtlDiff)
    assert diff.available is True
    assert diff.shared_names == ("guard",)
    assert diff.only_in_a == ("trader",)
    assert diff.only_in_b == ("scout",)
    assert diff.jaccard == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_substrate_diff_aggregates_and_serializes(tmp_path: Path) -> None:
    """``substrate_diff`` aggregates all four axes and ``__str__`` is non-empty."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    _write_json(dir_a / "aut_nac.json", _nac_payload({"agentA:n": 0.5}))
    _write_json(dir_b / "aut_nac.json", _nac_payload({"agentA:n": 0.0}))

    diff = substrate_diff(dir_a, dir_b)
    text = str(diff)
    assert "Substrate divergence" in text
    assert "NAc diff" in text
    # EC + ATL files weren't written → fail-soft markers.
    assert "EC diff: available=false" in text
    assert "ATL diff: available=false" in text
    assert "Hippocampus diff: available=false" in text

    payload = substrate_diff_to_json(diff)
    assert payload["nac"]["available"] is True
    assert payload["nac"]["reward_bias_l2"] == pytest.approx(0.5, abs=1e-9)
    assert payload["ec"]["available"] is False
    assert payload["atl"]["available"] is False
    assert payload["hippocampus"]["available"] is False


def test_corrupt_file_returns_unavailable(tmp_path: Path) -> None:
    """A corrupt JSON file yields available=False, never raises."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "aut_nac.json").write_text("{not valid json", encoding="utf-8")
    _write_json(dir_b / "aut_nac.json", _nac_payload({}))

    diff = nac_diff(dir_a, dir_b)
    assert diff.available is False
    assert "corrupt" in diff.reason.lower() or "unreadable" in diff.reason.lower()
