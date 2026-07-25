"""Regression guards for HANDLE seam (a) — persistent-agent campaign injection.

docs/plans/console_handle_campaign_injection.md. The failure mode this seam
kills is SILENT mis-routing of learning (an Adventure run learning into a
throwaway "sim_aut" the Talk agent never reads), so the guards pin, against a
REAL bio-stack (no LLM, no full sim — house style per
tests/integration/test_orchestrator_sandbox.py):

1. adoption binds the campaign's aut_* surfaces to the persistent agent's
   OWN subsystems, and a campaign episode persists in the agent's home;
2. imagined=True fiction facts decay while real learning persists on the
   persistent agent's NAc;
3. the persistent_agent-is-None path is unchanged (defaults pinned; existing
   orchestrator tests cover the body);
4. stop routes FULL consolidation (#427's explicit flavor).
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from maxim.simulation.orchestrator import (
    _AdoptedAgent,
    _CampaignToolLease,
    _adopt_persistent_agent,
    _setup_sim_sandbox,
    start_simulation_mode,
)
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry


def _stub_tool(tool_name: str) -> Tool:
    class _T(Tool):
        name = tool_name
        description = "stub"
        input_schema: dict[str, Any] = {}

        def execute(self, **kwargs: Any) -> Any:
            return ToolOutput(success=True, output="ok")

    return _T()


def _build_persistent_agent(tmp_path, agent_id: str = "console_agent"):
    """A REAL persistent agent: full bio-stack + executor, home in tmp_path."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    config = AgentConfig(
        agent_id=agent_id,
        role="pc",
        persistence_dir=str(tmp_path / agent_id),
        with_bio_stack=True,
        with_executor=True,
        with_pain_bridge=False,
        with_fear_gate=False,
    )
    factory = AgentFactory(base_data_dir=tmp_path)
    registry = ToolRegistry()
    registry.register(_stub_tool("persistent_tool"))
    return factory.create_full_agent(config, tool_registry=registry, auto_load=True)


# ─────────────────────────────────────────────────────────────────────────
# Guard 1 — adoption binds the persistent agent's own subsystems
# ─────────────────────────────────────────────────────────────────────────


class TestAdoption:
    def test_adoption_binds_persistent_systems(self, tmp_path):
        inst = _build_persistent_agent(tmp_path)
        adopted = _adopt_persistent_agent(inst)
        assert isinstance(adopted, _AdoptedAgent)
        assert adopted.instance is inst
        assert adopted.agent_id == "console_agent"
        assert adopted.registry is inst.tool_registry
        assert adopted.pain_bus is inst.pain_bus
        # The lease snapshot covers the agent's pre-campaign tools.
        assert "persistent_tool" in adopted.lease.baseline

    def test_campaign_episode_lands_in_persistent_home(self, tmp_path):
        """The seam's whole point: an episode stored through the ADOPTED
        binding is recallable from the persistent agent's OWN hippocampus,
        persisted under the agent's home — not a session AUT file."""
        inst = _build_persistent_agent(tmp_path)
        adopted = _adopt_persistent_agent(inst)

        # What the orchestrator binds as aut_hippocampus IS the persistent one.
        aut_hippocampus = adopted.instance.hippocampus
        assert aut_hippocampus is inst.hippocampus

        aut_hippocampus.store_observation("The dragon spared the village of Emberfall")
        persist_path = aut_hippocampus.config.persistence_path
        assert persist_path is not None
        assert str(tmp_path / "console_agent") in str(persist_path), (
            "persistence must live in the agent's home, not a sim tmpdir"
        )
        aut_hippocampus.save(persist_path)

        # A fresh Hippocampus loading the persistent HOME sees the episode.
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        fresh = Hippocampus(HippocampusConfig(persistence_path=str(persist_path)))
        fresh.load(str(persist_path))
        results = fresh.recall(query="dragon village", limit=5)
        assert results, "campaign episode must be recallable from the persistent agent's home"

    def test_adoption_rejects_missing_subsystems(self, tmp_path):
        from maxim.runtime.agent_factory import AgentConfig, AgentInstance

        bare = AgentInstance(
            agent_id="console_agent",
            role="pc",
            config=AgentConfig(agent_id="console_agent"),
        )
        with pytest.raises(ValueError, match="missing"):
            _adopt_persistent_agent(bare)

    def test_adoption_rejects_sim_aut_identity(self, tmp_path):
        inst = _build_persistent_agent(tmp_path, agent_id="sim_aut2")
        inst.agent_id = "sim_aut"
        with pytest.raises(ValueError, match="sim_aut"):
            _adopt_persistent_agent(inst)

    def test_adoption_rejects_empty_agent_id(self, tmp_path):
        inst = _build_persistent_agent(tmp_path)
        inst.agent_id = ""
        with pytest.raises(ValueError, match="agent_id"):
            _adopt_persistent_agent(inst)

    def test_adoption_rejects_hub_agent_id_mismatch(self, tmp_path):
        """Producer/consumer key alignment: the loop keys attribution off
        memory_hub.agent_id — a divergent instance.agent_id would silently
        split learning across two keys."""
        inst = _build_persistent_agent(tmp_path)
        inst.agent_id = "someone_else"
        with pytest.raises(ValueError, match="disagrees"):
            _adopt_persistent_agent(inst)

    def test_entity_ref_incompatible_with_injection(self, tmp_path):
        """entity_ref grafts a sim-owned embodiment; a persistent agent owns
        its own (declared at handle construction). Fails loud + early."""
        inst = _build_persistent_agent(tmp_path)
        with pytest.raises(ValueError, match="entity_ref"):
            start_simulation_mode(
                goal="incompat check",
                persistent_agent=inst,
                entity_ref="weapons/rusty_sword",
            )


# ─────────────────────────────────────────────────────────────────────────
# Guard 2 — imagined fiction decays; real learning persists
# ─────────────────────────────────────────────────────────────────────────


class TestImaginedFictionProvenance:
    def test_imagined_link_decays_real_link_persists(self, tmp_path):
        from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence

        inst = _build_persistent_agent(tmp_path)
        nac = inst.nac
        assert nac is not None

        def _link(link_id: str, signature: str) -> CausalLink:
            return CausalLink(
                id=link_id,
                event_type="tool",
                event_signature=signature,
                event_context={},
                outcome_type="tool_result",
                outcome_signature="result",
                outcome_valence=Valence.POSITIVE,
                temporal_delta=TemporalDelta(observed_deltas=(1.0,)),
                confidence=0.8,
            )

        nac._links["pet_the_dog"] = [_link("real", "pet_the_dog")]
        nac._links["ghost_merchant_barter"] = [_link("fiction", "ghost_merchant_barter")]

        # Session-end sequence the orchestrator runs (branch 6) — against
        # the PERSISTENT agent's NAc.
        tagged = nac.tag_imagined_links(frozenset({"npcs/ghost_merchant"}))
        assert tagged == 1
        decayed = nac.decay_imagined_links(0.5)
        assert decayed == 1

        real = nac._links["pet_the_dog"][0]
        fiction = nac._links["ghost_merchant_barter"][0]
        assert real.confidence == pytest.approx(0.8), "real learning must NOT decay"
        assert real.imagined is False
        assert fiction.imagined is True
        assert fiction.confidence == pytest.approx(0.4), "fiction must decay by the 0.5 factor"


# ─────────────────────────────────────────────────────────────────────────
# Branch 4 — the campaign tool lease restores the persistent registry
# ─────────────────────────────────────────────────────────────────────────


class TestCampaignToolLease:
    def test_restore_returns_registry_to_precampaign_state(self):
        registry = ToolRegistry()
        registry.register(_stub_tool("persistent_tool"))
        registry.register(_stub_tool("bash"))

        lease = _CampaignToolLease.snapshot(registry)

        # Campaign mutations: add sim tools, drop a "noisy" persistent tool
        # (the orchestrator's _irrelevant_tools loop shape).
        registry.register(_stub_tool("choose"))
        registry.register(_stub_tool("memory_recall"))
        lease.record_removal("bash", registry)
        registry.deregister("bash")

        assert set(registry.list_all()) == {"persistent_tool", "choose", "memory_recall"}

        dropped, restored = lease.restore(registry)
        assert set(dropped) == {"choose", "memory_recall"}
        assert restored == ["bash"]
        assert set(registry.list_all()) == lease.baseline == {"persistent_tool", "bash"}

    def test_record_removal_ignores_campaign_added_tools(self):
        registry = ToolRegistry()
        lease = _CampaignToolLease.snapshot(registry)
        registry.register(_stub_tool("choose"))
        lease.record_removal("choose", registry)  # not in baseline → not captured
        registry.deregister("choose")
        dropped, restored = lease.restore(registry)
        assert dropped == [] and restored == []
        assert registry.list_all() == []


# ─────────────────────────────────────────────────────────────────────────
# Guard 3 — persistent_agent is None path unchanged
# ─────────────────────────────────────────────────────────────────────────


class TestNonePathUnchanged:
    def test_persistent_agent_defaults_to_none(self):
        sig = inspect.signature(start_simulation_mode)
        assert sig.parameters["persistent_agent"].default is None

    def test_sandbox_helper_builds_own_bus_by_default(self):
        sandbox, _, pain_bus = _setup_sim_sandbox(backend="tmpdir", populate=False)
        try:
            assert pain_bus is not None
        finally:
            if sandbox is not None:
                sandbox.cleanup()

    def test_sandbox_helper_adopts_injected_bus(self):
        from maxim.proprioception.pain_bus import build_pain_bus

        injected = build_pain_bus(hippocampus=None, nac=None)
        sandbox, _, pain_bus = _setup_sim_sandbox(backend="tmpdir", populate=False, pain_bus=injected)
        try:
            assert pain_bus is injected, "an injected persistent bus must be routed, not rebuilt"
        finally:
            if sandbox is not None:
                sandbox.cleanup()


# ─────────────────────────────────────────────────────────────────────────
# Guard 4 — stop routes FULL consolidation (#427)
# ─────────────────────────────────────────────────────────────────────────


class _SpyHub:
    def __init__(self) -> None:
        self.full_calls = 0
        self.lightweight_calls = 0

    def on_session_end(self) -> dict:
        self.full_calls += 1
        return {}

    def on_session_end_lightweight(self) -> dict:
        self.lightweight_calls += 1
        return {}


class TestStopConsolidation:
    def _instance_with_spy_hub(self):
        from maxim.runtime.agent_factory import AgentConfig, AgentInstance

        hub = _SpyHub()
        inst = AgentInstance(
            agent_id="console_agent",
            role="pc",
            config=AgentConfig(agent_id="console_agent"),
            memory_hub=hub,
        )
        return inst, hub

    def test_shutdown_default_is_full(self):
        inst, hub = self._instance_with_spy_hub()
        inst.shutdown()
        assert hub.full_calls == 1 and hub.lightweight_calls == 0

    def test_shutdown_explicit_lightweight(self):
        inst, hub = self._instance_with_spy_hub()
        inst.shutdown(consolidation="lightweight")
        assert hub.full_calls == 0 and hub.lightweight_calls == 1

    def test_shutdown_rejects_unknown_flavor(self):
        inst, _ = self._instance_with_spy_hub()
        with pytest.raises(ValueError, match="consolidation"):
            inst.shutdown(consolidation="fast")

    def test_aut_loop_gets_full_consolidation_when_injected(self):
        """Wiring pin (branch 5): the AUT's run_agentic_loop call carries the
        explicit "full" override for an injected persistent agent and None
        (today's derive-from-sim-flag behavior) otherwise."""
        src = inspect.getsource(start_simulation_mode)
        assert 'consolidation="full" if persistent_agent is not None else None' in src


# ─────────────────────────────────────────────────────────────────────────
# MaximHandle — the headless HANDLE flavor
# ─────────────────────────────────────────────────────────────────────────


class TestMaximHandle:
    def test_handle_builds_persistent_agent_in_home(self, tmp_path):
        from maxim.console.handle import MaximHandle

        handle = MaximHandle(agent_id="console_agent", home=tmp_path / "home")
        try:
            inst = handle.instance
            assert inst.agent_id == "console_agent"
            assert inst.executor is not None
            assert inst.memory_hub is not None
            assert inst.hippocampus is not None
            # And it is adoptable — the injection precondition.
            adopted = _adopt_persistent_agent(inst)
            assert adopted.instance is inst
        finally:
            handle.stop()

    def test_handle_rejects_sim_aut_identity(self, tmp_path):
        from maxim.console.handle import MaximHandle

        with pytest.raises(ValueError, match="sim_aut"):
            MaximHandle(agent_id="sim_aut", home=tmp_path)

    def test_stop_routes_full_consolidation_and_is_idempotent(self, tmp_path):
        from maxim.console.handle import MaximHandle

        handle = MaximHandle(agent_id="console_agent", home=tmp_path / "home")
        hub = _SpyHub()
        handle.instance.memory_hub = hub
        handle.stop()
        assert hub.full_calls == 1, "stop() must run FULL consolidation (#427)"
        handle.stop()
        assert hub.full_calls == 1, "second stop must be a no-op"

    def test_play_campaign_missing_path_fails_loud(self, tmp_path):
        from maxim.console.handle import MaximHandle

        handle = MaximHandle(agent_id="console_agent", home=tmp_path / "home")
        try:
            with pytest.raises(FileNotFoundError):
                handle.play_campaign(tmp_path / "nope.yaml")
        finally:
            handle.stop()

    def test_play_campaign_refused_after_stop(self, tmp_path):
        from maxim.console.handle import MaximHandle

        handle = MaximHandle(agent_id="console_agent", home=tmp_path / "home")
        handle.stop()
        with pytest.raises(RuntimeError, match="stopped"):
            handle.play_campaign(tmp_path / "any.yaml")
