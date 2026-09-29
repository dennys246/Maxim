"""#972 -- a write-but-don't-read agent restores NOTHING, including at session start.

The sim orchestrator's agent is built with ``AgentConfig(load_persisted=False)`` ("the orch must NOT
restore it"), and ``build_bio_stack`` honoured that for Hippocampus, NAc/EC and SCN. But the hub's
``on_session_start`` restored the ATL (and the AngularGyrus, and the cross-layer graph) from the same
home regardless, so last run's concepts came back every time. The factory's own skeleton hub did the
same for ``create_agent(auto_load=False)``. The hub now learns the builder's choice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _seed_atl_and_ag(home: Path) -> None:
    from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig
    from maxim.memory.atl import ATL, ATLConfig

    home.mkdir(parents=True, exist_ok=True)
    atl = ATL(ATLConfig(persistence_path=str(home / "atl.json")))
    atl.find_or_create("last_runs_wolf", "object")
    atl.save()
    ag = AngularGyrus(AngularGyrusConfig(persistence_path=str(home / "angular_gyrus.json"), seed_built_ins=False))
    ag._seed_built_ins()
    ag.save()
    # A marker only a restore can produce: the file's stats.
    data = json.loads((home / "angular_gyrus.json").read_text())
    data["stats"]["total_stores"] = 987654
    (home / "angular_gyrus.json").write_text(json.dumps(data))


def _names(atl) -> set[str]:
    return {getattr(c, "name", None) for c in atl.recall(limit=100)}


def test_the_write_but_dont_read_agent_does_not_restore_its_atl_or_ag(tmp_path: Path) -> None:
    from maxim.runtime.bio_stack import build_bio_stack

    _seed_atl_and_ag(tmp_path)
    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent", load_persisted=False)
    try:
        stack.memory_hub.on_session_start()
        assert "last_runs_wolf" not in _names(stack.atl)
        assert stack.angular_gyrus._total_stores != 987654  # nothing from the file
        assert stack.memory_hub.load_persisted is False
    finally:
        stack.memory_hub.on_session_end()
        stack.memory_hub.shutdown()


def test_a_normal_agent_still_restores_them(tmp_path: Path) -> None:
    from maxim.runtime.bio_stack import build_bio_stack

    _seed_atl_and_ag(tmp_path)
    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent")
    try:
        stack.memory_hub.on_session_start()
        assert "last_runs_wolf" in _names(stack.atl)
        assert stack.angular_gyrus._total_stores == 987654
        assert stack.memory_hub.load_persisted is True
    finally:
        stack.memory_hub.on_session_end()
        stack.memory_hub.shutdown()


def test_the_orchestrator_shaped_full_agent_does_not_restore_its_atl(tmp_path: Path) -> None:
    """The issue's guard: AgentConfig(load_persisted=False), session started by the factory."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    home = tmp_path / "orchestrator"
    _seed_atl_and_ag(home)
    instance = AgentFactory(base_data_dir=tmp_path).create_full_agent(
        AgentConfig(agent_id="orchestrator", persistence_dir=str(home), load_persisted=False, with_bio_stack=True)
    )
    try:
        assert "last_runs_wolf" not in _names(instance.memory_hub.atl)
    finally:
        instance.shutdown()
    # Write-but-don't-read: its declared overwrite replaced last run's file with this run's (empty) ATL.
    assert "last_runs_wolf" not in (home / "atl.json").read_text()


def test_a_fresh_factory_agent_does_not_restore_at_session_start(tmp_path: Path) -> None:
    """create_agent(auto_load=False) skipped the ATL at construction, then its hub read atl.json anyway."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    home = tmp_path / "npc"
    _seed_atl_and_ag(home)
    before = (home / "atl.json").read_bytes()
    agent = AgentFactory(base_data_dir=tmp_path).create_agent(AgentConfig(agent_id="npc"), auto_load=False)
    assert "last_runs_wolf" not in _names(agent.atl)
    agent.atl.find_or_create("this_runs_bear", "object")
    agent.shutdown()
    # Not write-but-don't-read: nothing was declared, so its save over the existing file is refused
    # (#939/#971) and the file is untouched -- the create_npc_agent consequence, followed up in #985.
    assert (home / "atl.json").read_bytes() == before


def test_build_memory_hub_requires_the_decision() -> None:
    """A default is how this happened: forgetting the decision is a TypeError, not a restore."""
    import inspect

    from maxim.integration.memory_hub import build_memory_hub

    param = inspect.signature(build_memory_hub).parameters["load_persisted"]
    assert param.default is inspect.Parameter.empty and param.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("load_persisted", [False, True])
def test_the_hub_learns_the_bio_stack_choice(tmp_path: Path, load_persisted: bool) -> None:
    from maxim.runtime.bio_stack import build_bio_stack

    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent", load_persisted=load_persisted)
    try:
        assert stack.memory_hub.load_persisted is load_persisted
    finally:
        stack.memory_hub.shutdown()


def test_the_cross_layer_graph_is_not_restored_either(tmp_path: Path) -> None:
    """Nothing persists the graph in production today; the gate holds for it all the same."""
    from maxim.memory.cross_layer import CrossLayerEdgeType, CrossLayerGraph
    from maxim.runtime.bio_stack import build_bio_stack

    path = tmp_path / "cross_layer_graph.json"
    writer = CrossLayerGraph(persistence_path=str(path))
    writer.add_edge("hippocampus", "m1", "atl", "c1", next(iter(CrossLayerEdgeType)))
    writer.save()
    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent", load_persisted=False)
    try:
        stack.memory_hub._cross_layer._persistence_path = str(path)
        stack.memory_hub.on_session_start()
        assert stack.memory_hub._cross_layer.stats()["total_edges"] == 0
    finally:
        stack.memory_hub.shutdown()
