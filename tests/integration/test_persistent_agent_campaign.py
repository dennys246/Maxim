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

    def test_adoption_requires_full_bio_surface(self, tmp_path):
        """pain_bus/hippocampus/nac are REQUIRED — a missing pain_bus would
        make the sandbox silently build a learner-less bus and orphan every
        pain signal (review fold: Exec #6 / bio-fidelity F5)."""
        for attr in ("pain_bus", "hippocampus", "nac"):
            inst = _build_persistent_agent(tmp_path / attr)
            setattr(inst, attr, None)
            with pytest.raises(ValueError, match="missing"):
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
    """Pins the NAc tag/decay MECHANISM on a persistent agent's NAc.

    HONESTY NOTE (three-lens review, cross-confirmed): the orchestrator's
    branch-6 block is structurally INERT on the injected DM path today (the
    imagination trigger requires entity_ref, which injection forbids) —
    campaign-DECLARED fiction persists as real learning by design intent;
    provenance for campaign-declared entities is a tracked follow-up in
    docs/plans/console_handle_campaign_injection.md. This test guards the
    mechanism that follow-up will wire, not the current wiring.
    """

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
        persistent = _stub_tool("persistent_tool")
        bash = _stub_tool("bash")
        registry.register(persistent)
        registry.register(bash)

        lease = _CampaignToolLease.snapshot(registry)

        # Campaign mutations: add sim tools, drop a "noisy" persistent tool
        # (the orchestrator's _irrelevant_tools loop shape — no bookkeeping
        # needed; the snapshot holds the baseline objects).
        registry.register(_stub_tool("choose"))
        registry.register(_stub_tool("memory_recall"))
        registry.deregister("bash")

        assert set(registry.list_all()) == {"persistent_tool", "choose", "memory_recall"}

        dropped, restored = lease.restore(registry)
        assert set(dropped) == {"choose", "memory_recall"}
        assert restored == ["bash"]
        assert set(registry.list_all()) == set(lease.baseline) == {"persistent_tool", "bash"}
        assert registry.get("bash") is bash, "the ORIGINAL persistent tool object must come back"

    def test_restore_rebinds_replaced_baseline_tools(self):
        """ToolRegistry.register silently overwrites same-name entries; a
        name-only diff cannot see a campaign registration that REPLACED a
        persistent tool (pinning dead sim state). The object-identity
        snapshot must rebind the baseline object. (Review fold: Exec #7.)"""
        registry = ToolRegistry()
        original = _stub_tool("memory_recall")
        registry.register(original)

        lease = _CampaignToolLease.snapshot(registry)
        registry.register(_stub_tool("memory_recall"))  # campaign shadows it
        assert registry.get("memory_recall") is not original

        dropped, restored = lease.restore(registry)
        assert dropped == [] and restored == ["memory_recall"]
        assert registry.get("memory_recall") is original

    def test_restore_is_idempotent(self):
        """The handle's exception-path safety net calls restore() after the
        orchestrator's normal-path restore — the second call must no-op."""
        registry = ToolRegistry()
        registry.register(_stub_tool("persistent_tool"))
        lease = _CampaignToolLease.snapshot(registry)
        registry.register(_stub_tool("choose"))

        lease.restore(registry)
        dropped, restored = lease.restore(registry)
        assert dropped == [] and restored == []
        assert set(registry.list_all()) == {"persistent_tool"}


# ─────────────────────────────────────────────────────────────────────────
# Guard 3 — persistent_agent is None path unchanged
# ─────────────────────────────────────────────────────────────────────────


class TestNonePathUnchanged:
    def test_persistent_agent_defaults_to_none(self):
        sig = inspect.signature(start_simulation_mode)
        assert sig.parameters["persistent_agent"].default is None

    def test_injection_branch_gates_are_wired(self):
        """Source pins for branches 1-3 (matching the branch-5 pin below):
        reverting any of these gates would silently mis-route learning while
        every helper-level test still passes (review fold: Exec #9)."""
        src = inspect.getsource(start_simulation_mode)
        # Branch 1: factory call skipped on adoption
        assert "_aut_instance = _adopted.instance" in src
        # Branch 2: resume-session file-load gated
        assert "if persistent_agent is None and resume_session" in src
        # Branch 3: session-dir AUT snapshot skipped
        assert "if persistent_agent is not None:" in src and "no session AUT snapshot" in src
        # Bash env not armed for adopted agents
        assert "if _adopted is None:\n        os.environ.setdefault" in src
        # /new recursion keeps the injection
        assert "persistent_agent=persistent_agent" in src

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
# Repeated sessions on one persistent hub (review fold: Arch #1 / Exec #4)
# ─────────────────────────────────────────────────────────────────────────


class TestRepeatedSessions:
    def test_sleep_with_autosave_does_not_deadlock(self, tmp_path):
        """Pre-fix, hippocampus.sleep() self-deadlocked whenever
        auto_save_after_sleep=True (the default) met a live persistence_path:
        _sleep saved UNDER the write lock, and save→dump takes a read lock on
        the same non-reentrant RWLock. CI never saw it because conftest sets
        auto_save_after_sleep=False; every persistent agent's full
        consolidation hits this combination. Run in a thread with a bounded
        join so a regression FAILS rather than hanging the suite."""
        import threading

        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        path = tmp_path / "hippocampus.json"
        hippo = Hippocampus(HippocampusConfig(persistence_path=str(path), auto_save_after_sleep=True))
        hippo.store_observation("a memory to consolidate and save")

        done = threading.Event()
        results: dict = {}

        def _run() -> None:
            results.update(hippo.sleep())
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        assert done.wait(timeout=30.0), "hippocampus.sleep() deadlocked (auto-save under write lock)"
        assert path.exists(), "auto-save after sleep must still write the persistence file"

    def test_concept_extractor_revives_on_next_session(self, tmp_path):
        """Full consolidation shuts the ConceptExtractor worker down; a
        persistent agent's NEXT session (campaign #2, Talk mode) must revive
        it — otherwise episodes capture while ATL concept extraction is
        silently dead."""
        inst = _build_persistent_agent(tmp_path)
        hub = inst.memory_hub
        ce = hub._concept_extractor
        assert ce is not None, "persistent hub must have a concept extractor wired"
        assert ce._worker.is_alive()

        hub.on_session_start()
        hub.on_session_end()
        assert not ce._worker.is_alive(), "full consolidation stops the worker"

        hub.on_session_start()
        assert ce._worker.is_alive(), "second session must revive concept extraction"
        hub.on_session_end()

    def test_dn_pain_bridge_unsubscribe_leaves_persistent_learners(self, tmp_path):
        """A sim-scoped DefaultNetwork subscribes its PainCircuitBridge to
        the persistent bus; teardown must unsubscribe it or every campaign
        accumulates a dead subscriber (latent duplicate NAc pain learner).
        Pins the mechanism + the orchestrator teardown wiring."""
        from maxim.default_network import DefaultNetworkConfig
        from maxim.runtime.bootstrap import build_default_network

        inst = _build_persistent_agent(tmp_path)
        bus = inst.pain_bus
        before = len(bus._pain_signal_subs)

        dn = build_default_network(
            nac=inst.nac,
            maxim=None,
            pain_bus=bus,
            config=DefaultNetworkConfig(enabled=True, publish_actions=False, fear_gate_enabled=False),
        )
        if dn is None or dn.pain_bridge is None:
            pytest.skip("DefaultNetwork unavailable in this environment")
        assert len(bus._pain_signal_subs) > before, "DN bridge subscribes to the persistent bus"

        bus.unsubscribe(dn.pain_bridge._on_pain)
        assert len(bus._pain_signal_subs) == before, "unsubscribe must fully un-pin the sim DN"

        # Wiring pin: the orchestrator's adopted teardown performs this.
        src = inspect.getsource(start_simulation_mode)
        assert "aut_pain_bus.unsubscribe(_dn_bridge._on_pain)" in src


# ─────────────────────────────────────────────────────────────────────────
# Post-merge round: stop-vs-loop session-end race (Exec #4 / Arch #1)
# ─────────────────────────────────────────────────────────────────────────


class TestSessionEndRace:
    def test_concurrent_session_end_consolidates_exactly_once(self, tmp_path):
        """The unlocked check-then-act on _session_active let a shutdown-hook
        stop() and the campaign loop's own session-end BOTH run full
        consolidation concurrently. The atomic test-and-clear admits exactly
        one; the loser gets the honest no-op {}."""
        import threading

        inst = _build_persistent_agent(tmp_path)
        hub = inst.memory_hub
        hub.on_session_start()

        barrier = threading.Barrier(2)
        results: list[dict] = []

        def _end() -> None:
            barrier.wait()
            results.append(hub.on_session_end())

        threads = [threading.Thread(target=_end) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
        assert len(results) == 2
        non_empty = [r for r in results if r]
        assert len(non_empty) == 1, f"exactly one caller must consolidate, got {len(non_empty)}"

    def test_handle_stop_waits_for_campaign_then_proceeds_loudly(self, tmp_path, caplog):
        """stop() waits (bounded) on the campaign lock; on expiry it proceeds
        with a WARNING rather than hanging forever on a wedged campaign."""
        import logging

        from maxim.console.handle import MaximHandle

        handle = MaximHandle(agent_id="console_agent", home=tmp_path / "home")
        hub = _SpyHub()
        handle.instance.memory_hub = hub

        handle._campaign_lock.acquire()  # simulate a live (wedged) campaign
        try:
            with caplog.at_level(logging.WARNING, logger="maxim.console.handle"):
                handle.stop(campaign_wait_s=0.2)
        finally:
            handle._campaign_lock.release()
        assert hub.full_calls == 1, "stop must still consolidate after the bounded wait"
        assert any("campaign still running" in r.message for r in caplog.records)


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
