"""Loaders prune and COUNT dangling cross-store refs (#1438 PR 1a; docs/plans/bounded_stop.md §5).

A session end saves the hippocampus first; a stop or a crash before a dependant's save leaves that dependant's file
holding refs to memories the saved hippocampus no longer has. Today nothing prunes them: ATL's ``ref_count``
shields its concepts from forgetting, and NAc's ``memory_ids`` flow into semantic promotion. (SCN and EC are not
pruned: their ids are not the hippocampus's.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _agent(home: Path) -> Any:
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.tools.registry import ToolRegistry

    config = AgentConfig(
        agent_id="prune_agent",
        role="pc",
        persistence_dir=str(home / "prune_agent"),
        with_bio_stack=True,
        with_executor=False,
        with_pain_bridge=False,
        with_fear_gate=False,
    )
    return AgentFactory(base_data_dir=home).create_full_agent(config, tool_registry=ToolRegistry(), auto_load=True)


def _drop_from_hippocampus_file(home: Path, memory_id: str) -> None:
    """What a cut-short end leaves: the hippocampus saved WITHOUT the memory, its dependants saved WITH it."""
    path = home / "prune_agent" / "hippocampus.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["memories"] = [m for m in data["memories"] if m["id"] != memory_id]
    path.write_text(json.dumps(data), encoding="utf-8")


def _concepts_referencing(hub: Any, memory_id: str) -> list[str]:
    return [c.id for c in hub.atl if memory_id in (getattr(c, "memory_refs", {}) or {}).get("hippocampus", ())]


def test_a_memory_missing_from_the_saved_hippocampus_is_pruned_from_atl_and_counted(tmp_path: Path) -> None:
    first = _agent(tmp_path)
    victim = first.hippocampus.store_observation("the red kettle sits on the left burner of the stove")
    first.hippocampus.flush(timeout=5.0)
    first.memory_hub._concept_extractor.flush(timeout=5.0)
    # The extractor's own linking depends on the text pipeline; link one concept by hand so the ATL row is exercised.
    concept_id, _ = first.memory_hub.atl.find_or_create(name="kettle", category="object")
    first.memory_hub.atl.get(concept_id).add_ref("hippocampus", victim)
    assert _concepts_referencing(first.memory_hub, victim)  # precondition
    first.shutdown()
    _drop_from_hippocampus_file(tmp_path, victim)

    second = _agent(tmp_path)

    hub = second.memory_hub
    assert victim not in {m.id for m in hub.hippocampus}
    assert _concepts_referencing(hub, victim) == []
    assert hub._ownership.current.start_stats["pruned_refs"]["atl_hippocampus_refs"] >= 1


def test_a_fear_on_a_node_ec_still_holds_is_kept_and_one_on_a_vanished_node_is_pruned(tmp_path: Path) -> None:
    """An early version passed EC's own (valid) ids as the ids to DROP and wiped every live fear."""
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    hub.ec.register_substrate_node("node_alive", modality="world", embedding=[1.0, 0.0, 0.0])
    hub.nac.record_cluster_fear("prune_agent", "node_alive", "drive:oxygen", 1.0)
    hub.nac.record_cluster_fear("prune_agent", "node_gone", "drive:oxygen", 1.0)
    assert hub.nac.cluster_fear("prune_agent", "node_gone") < 0  # precondition: both fears booked

    assert hub.nac.prune_ec_keys(hub.ec.substrate_node_ids()) == 1

    assert hub.nac.cluster_fear("prune_agent", "node_alive") < 0
    assert hub.nac.cluster_fear("prune_agent", "node_gone") == 0.0


def test_nothing_is_pruned_against_a_store_that_did_not_load_its_file(tmp_path: Path) -> None:
    """A target that came up empty (no file, an unreadable or a refused one) makes every ref look dangling."""
    hub = _agent(tmp_path).memory_hub  # a fresh home: no store read a file
    hub.nac.record_cluster_fear("prune_agent", "node_x", "drive:oxygen", 1.0)
    concept_id, _ = hub.atl.find_or_create(name="kettle", category="object")
    hub.atl.get(concept_id).add_ref("hippocampus", "a_memory_not_in_this_hippocampus")
    assert not hub.hippocampus.store_file_loaded() and not hub.ec.store_file_loaded()  # preconditions
    hub._refs_pruned = False  # run the once-per-hub pass again, as at a first open

    assert hub._prune_dangling_refs() == {}
    assert hub.nac.cluster_fear("prune_agent", "node_x") < 0
    assert _concepts_referencing(hub, "a_memory_not_in_this_hippocampus")
