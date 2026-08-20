"""Tests for maxim.runtime.agent_factory + agent_pool — multi-agent infrastructure."""

from __future__ import annotations

import threading

import pytest

from maxim.runtime.agent_factory import AgentConfig, AgentFactory, AgentInstance
from maxim.runtime.agent_pool import AgentPool, TurnResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def factory(tmp_path):
    """Create an AgentFactory with temp data directory."""
    return AgentFactory(base_data_dir=tmp_path / "agents")


@pytest.fixture
def guard_config():
    return AgentConfig(
        agent_id="npc_guard",
        role="npc",
        personality="A stern town guard.",
        remembers=True,
        learns=True,
    )


@pytest.fixture
def merchant_config():
    return AgentConfig(
        agent_id="npc_merchant",
        role="npc",
        personality="A shrewd merchant.",
        remembers=True,
        learns=False,
    )


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig(agent_id="test")
        assert cfg.role == "npc"
        assert cfg.remembers is True
        assert cfg.learns is True
        assert cfg.tool_whitelist is None
        assert cfg.personality is None

    def test_custom_config(self):
        cfg = AgentConfig(
            agent_id="pc",
            role="pc",
            model_profile="large",
            tool_whitelist={"speak", "fight"},
            personality="A brave warrior.",
        )
        assert cfg.role == "pc"
        assert cfg.model_profile == "large"
        assert "speak" in cfg.tool_whitelist


# ---------------------------------------------------------------------------
# AgentFactory
# ---------------------------------------------------------------------------


class TestAgentFactory:
    def test_create_agent_returns_instance(self, factory, guard_config):
        instance = factory.create_agent(guard_config)
        assert isinstance(instance, AgentInstance)
        assert instance.agent_id == "npc_guard"
        assert instance.role == "npc"

    def test_creates_hippocampus_when_remembers(self, factory, guard_config):
        instance = factory.create_agent(guard_config)
        assert instance.hippocampus is not None

    def test_skips_hippocampus_when_not_remembers(self, factory):
        cfg = AgentConfig(agent_id="forgetful", remembers=False)
        instance = factory.create_agent(cfg)
        assert instance.hippocampus is None

    def test_creates_nac_when_learns(self, factory, guard_config):
        instance = factory.create_agent(guard_config)
        assert instance.nac is not None

    def test_skips_nac_when_not_learns(self, factory, merchant_config):
        instance = factory.create_agent(merchant_config)
        assert instance.nac is None

    def test_creates_memory_hub(self, factory, guard_config):
        instance = factory.create_agent(guard_config)
        assert instance.memory_hub is not None

    def test_creates_tool_registry(self, factory, guard_config):
        instance = factory.create_agent(guard_config)
        assert instance.tool_registry is not None

    def test_personality_preserved(self, factory, guard_config):
        instance = factory.create_agent(guard_config)
        assert instance.personality == "A stern town guard."

    def test_agents_have_separate_memory(self, factory, guard_config, merchant_config):
        guard = factory.create_agent(guard_config)
        merchant = factory.create_agent(merchant_config)

        # Separate hippocampus instances
        assert guard.hippocampus is not merchant.hippocampus

        # Separate persistence paths
        guard_path = guard.hippocampus.config.persistence_path
        merchant_path = merchant.hippocampus.config.persistence_path
        assert guard_path != merchant_path

    def test_per_agent_persistence_directory(self, factory, tmp_path):
        cfg = AgentConfig(agent_id="test_agent")
        factory.create_agent(cfg)
        agent_dir = tmp_path / "agents" / "test_agent"
        assert agent_dir.is_dir()

    def test_create_npc_agent_convenience(self, factory):
        instance = factory.create_npc_agent(
            npc_name="aldric",
            personality="A battle-hardened captain.",
        )
        assert instance.agent_id == "npc_aldric"
        assert instance.role == "npc"
        assert instance.personality == "A battle-hardened captain."
        assert instance.hippocampus is not None
        assert instance.nac is not None


class TestAgentInstance:
    def test_export_memories(self, factory, guard_config):
        instance = factory.create_agent(guard_config)
        export = instance.export_memories()
        assert export["agent_id"] == "npc_guard"
        assert export["role"] == "npc"
        assert "episodic_memories" in export

    def test_shutdown_safe(self, factory, guard_config):
        instance = factory.create_agent(guard_config)
        # Should not raise
        instance.shutdown()


# ---------------------------------------------------------------------------
# F2: create_full_agent
# ---------------------------------------------------------------------------


class TestCreateFullAgent:
    """Tests for create_full_agent — the F2 canonical agent construction."""

    def test_full_agent_with_bio_stack(self, factory):
        """Bio-stack is constructed when with_bio_stack=True."""
        cfg = AgentConfig(
            agent_id="full_bio",
            with_bio_stack=True,
        )
        instance = factory.create_full_agent(cfg)
        assert instance.bio_stack is not None
        assert instance.pain_bus is not None
        assert instance.hippocampus is not None
        assert instance.nac is not None
        assert instance.memory_hub is not None

    def test_full_agent_without_bio_stack(self, factory):
        """Bio-stack is NOT constructed when with_bio_stack=False."""
        cfg = AgentConfig(agent_id="no_bio")
        instance = factory.create_full_agent(cfg)
        assert instance.bio_stack is None
        assert instance.pain_bus is None

    def test_full_agent_with_executor(self, factory):
        """Executor is constructed when with_executor=True + tool_registry provided."""
        from maxim.tools.registry import ToolRegistry

        cfg = AgentConfig(
            agent_id="full_exec",
            with_bio_stack=True,
            with_executor=True,
        )
        registry = ToolRegistry()
        instance = factory.create_full_agent(cfg, tool_registry=registry)
        assert instance.executor is not None
        assert instance.tool_registry is registry

    def test_full_agent_executor_requires_registry(self, factory):
        """create_full_agent raises ValueError without tool_registry."""
        cfg = AgentConfig(
            agent_id="no_reg",
            with_executor=True,
        )
        with pytest.raises(ValueError, match="tool_registry"):
            factory.create_full_agent(cfg)

    def test_full_agent_shutdowns_partial_bio_stack_on_executor_failure(self, factory, monkeypatch):
        """A late factory failure cannot orphan an unreturned bio stack."""
        from maxim.tools.registry import ToolRegistry

        cfg = AgentConfig(
            agent_id="partial_exec",
            with_bio_stack=True,
            with_executor=True,
        )
        shutdown_instances = []
        original_shutdown = AgentInstance.shutdown

        def record_shutdown(instance, **kwargs):
            shutdown_instances.append(instance)
            return original_shutdown(instance, **kwargs)

        def fail_executor(*args, **kwargs):
            raise RuntimeError("executor failed")

        monkeypatch.setattr(AgentInstance, "shutdown", record_shutdown)
        monkeypatch.setattr("maxim.runtime.bootstrap.build_executor", fail_executor)

        with pytest.raises(RuntimeError, match="executor failed"):
            factory.create_full_agent(cfg, tool_registry=ToolRegistry())

        assert len(shutdown_instances) == 1
        assert shutdown_instances[0].bio_stack is not None

    def test_full_agent_with_fear_gate(self, factory):
        """FearGatedExecutor wraps the executor when with_fear_gate=True."""
        from maxim.tools.registry import ToolRegistry

        cfg = AgentConfig(
            agent_id="full_fear",
            with_bio_stack=True,
            with_executor=True,
            with_fear_gate=True,
        )
        registry = ToolRegistry()
        instance = factory.create_full_agent(cfg, tool_registry=registry)
        assert instance.executor is not None
        # FearGatedExecutor wraps the inner executor
        from maxim.runtime.fear_gate import FearGatedExecutor

        assert isinstance(instance.executor, FearGatedExecutor)

    def test_full_agent_memory_upgraded_from_bio_stack(self, factory):
        """Memory subsystems are upgraded to bio-stack versions."""
        cfg = AgentConfig(
            agent_id="upgraded",
            with_bio_stack=True,
        )
        instance = factory.create_full_agent(cfg)
        # hippocampus/nac/memory_hub should come from bio-stack
        assert instance.hippocampus is instance.bio_stack.hippocampus
        assert instance.nac is instance.bio_stack.nac
        assert instance.memory_hub is instance.bio_stack.memory_hub

    def test_full_agent_stops_superseded_skeleton_worker(self, factory, monkeypatch):
        """Bio-stack upgrade leaves exactly its returned MemoryHub worker alive."""
        skeleton_hubs = []
        original_create = factory.create_agent

        def capture_skeleton(*args, **kwargs):
            instance = original_create(*args, **kwargs)
            skeleton_hubs.append(instance.memory_hub)
            return instance

        monkeypatch.setattr(factory, "create_agent", capture_skeleton)
        cfg = AgentConfig(agent_id="worker_ownership", with_bio_stack=True)
        instance = factory.create_full_agent(cfg)

        skeleton = skeleton_hubs[0]
        assert skeleton is not instance.memory_hub
        assert skeleton._concept_extractor._worker.is_alive() is False
        assert instance.memory_hub._concept_extractor._worker.is_alive() is True

        instance.shutdown()
        assert instance.memory_hub._concept_extractor._worker.is_alive() is False

    def test_full_agent_interrupt_during_skeleton_cleanup_unwinds_both_hubs(self, factory, monkeypatch):
        """Ownership transfers before superseded-hub cleanup can be interrupted."""
        import maxim.runtime.bio_stack as bio_stack_module

        skeleton_hubs = []
        returned_bio = []
        original_create = factory.create_agent
        original_build_bio = bio_stack_module.build_bio_stack

        def capture_skeleton(*args, **kwargs):
            instance = original_create(*args, **kwargs)
            skeleton = instance.memory_hub
            original_shutdown = skeleton.shutdown
            shutdown_calls = 0

            def interrupt_once():
                nonlocal shutdown_calls
                shutdown_calls += 1
                if shutdown_calls == 1:
                    raise KeyboardInterrupt
                original_shutdown()

            skeleton.shutdown = interrupt_once
            skeleton_hubs.append(skeleton)
            return instance

        def capture_bio(**kwargs):
            bio = original_build_bio(**kwargs)
            returned_bio.append(bio)
            return bio

        monkeypatch.setattr(factory, "create_agent", capture_skeleton)
        monkeypatch.setattr(bio_stack_module, "build_bio_stack", capture_bio)
        cfg = AgentConfig(agent_id="interrupt_ownership", with_bio_stack=True)

        with pytest.raises(KeyboardInterrupt):
            factory.create_full_agent(cfg)

        assert skeleton_hubs[0]._concept_extractor._worker.is_alive() is False
        assert returned_bio[0].memory_hub._concept_extractor._worker.is_alive() is False

    def test_full_agent_bio_failure_does_not_persist_empty_skeleton(self, factory, monkeypatch):
        """Resource rollback cannot overwrite restored agent memory."""
        from unittest.mock import MagicMock

        skeletons = []
        original_create = factory.create_agent

        def capture_skeleton(*args, **kwargs):
            instance = original_create(*args, **kwargs)
            instance.hippocampus.save = MagicMock()
            instance.nac.save = MagicMock()
            skeletons.append(instance)
            return instance

        def fail_bio(**kwargs):
            raise RuntimeError("bio construction failed")

        monkeypatch.setattr(factory, "create_agent", capture_skeleton)
        monkeypatch.setattr("maxim.runtime.bio_stack.build_bio_stack", fail_bio)

        with pytest.raises(RuntimeError, match="bio construction failed"):
            factory.create_full_agent(AgentConfig(agent_id="rollback", with_bio_stack=True))

        skeleton = skeletons[0]
        skeleton.hippocampus.save.assert_not_called()
        skeleton.nac.save.assert_not_called()
        assert skeleton.memory_hub._concept_extractor._worker.is_alive() is False

    def test_full_agent_pre_built_pain_bus(self, factory):
        """Pre-built PainBus is passed through to build_bio_stack."""
        from maxim.proprioception.pain_bus import PainBus

        pre_bus = PainBus(_allow_raw=True)
        cfg = AgentConfig(
            agent_id="pre_bus",
            with_bio_stack=True,
        )
        instance = factory.create_full_agent(cfg, pain_bus=pre_bus)
        assert instance.pain_bus is pre_bus

    def test_full_agent_separate_persistence(self, factory, tmp_path):
        """Two full agents get separate persistence directories."""
        cfg_a = AgentConfig(agent_id="agent_a", with_bio_stack=True)
        cfg_b = AgentConfig(agent_id="agent_b", with_bio_stack=True)
        a = factory.create_full_agent(cfg_a)
        b = factory.create_full_agent(cfg_b)
        # Separate hippocampus instances
        assert a.hippocampus is not b.hippocampus
        assert a.nac is not b.nac

    def test_full_agent_backward_compat(self, factory):
        """create_full_agent with no F2 flags behaves like create_agent."""
        cfg = AgentConfig(agent_id="compat")
        instance = factory.create_full_agent(cfg)
        assert instance.bio_stack is None
        assert instance.executor is None
        assert instance.hippocampus is not None  # from create_agent


# ---------------------------------------------------------------------------
# AgentPool
# ---------------------------------------------------------------------------


class TestAgentPool:
    def test_add_and_get(self, factory, guard_config):
        pool = AgentPool()
        guard = factory.create_agent(guard_config)
        pool.add(guard)

        assert pool.size == 1
        assert "npc_guard" in pool.agent_ids
        assert pool.get_agent("npc_guard") is guard

    def test_remove(self, factory, guard_config):
        pool = AgentPool()
        guard = factory.create_agent(guard_config)
        pool.add(guard)
        pool.remove("npc_guard")

        assert pool.size == 0
        with pytest.raises(KeyError):
            pool.get_agent("npc_guard")

    def test_remove_clears_per_agent_stash(self, factory, guard_config):
        """remove() must drop module-level bio_integration stash entries
        so a future agent that reuses the id doesn't inherit stale state.
        """
        from maxim.runtime import bio_integration

        pool = AgentPool()
        guard = factory.create_agent(guard_config)
        pool.add(guard)

        # Seed all three stash dicts for this agent.
        bio_integration._episode_ticks["npc_guard"] = 42
        bio_integration._latest_pain_intensity["npc_guard"] = 0.7
        bio_integration._latest_substrate_nodes["npc_guard"] = ("node-a",)

        pool.remove("npc_guard")

        assert "npc_guard" not in bio_integration._episode_ticks
        assert "npc_guard" not in bio_integration._latest_pain_intensity
        assert "npc_guard" not in bio_integration._latest_substrate_nodes

    def test_run_turn(self, factory, guard_config):
        pool = AgentPool()
        guard = factory.create_agent(guard_config)
        pool.add(guard)

        result = pool.run_turn("npc_guard", "A stranger approaches the gate.")
        assert isinstance(result, TurnResult)
        assert result.agent_id == "npc_guard"
        assert result.error is None
        assert result.duration_ms > 0

    def test_run_round_sequential(self, factory, guard_config, merchant_config):
        pool = AgentPool()
        pool.add(factory.create_agent(guard_config))
        pool.add(factory.create_agent(merchant_config))

        results = pool.run_round(
            {
                "npc_guard": "A hooded figure approaches.",
                "npc_merchant": "The market opens at dawn.",
            },
            concurrent=False,
        )

        assert "npc_guard" in results
        assert "npc_merchant" in results
        assert results["npc_guard"].error is None
        assert results["npc_merchant"].error is None

    def test_run_round_concurrent(self, factory, guard_config, merchant_config):
        pool = AgentPool()
        pool.add(factory.create_agent(guard_config))
        pool.add(factory.create_agent(merchant_config))

        results = pool.run_round(
            {
                "npc_guard": "A hooded figure approaches.",
                "npc_merchant": "The market opens at dawn.",
            },
            concurrent=True,
        )

        assert len(results) == 2
        for result in results.values():
            assert result.error is None

    def test_broadcast_percept(self, factory, guard_config, merchant_config):
        pool = AgentPool()
        pool.add(factory.create_agent(guard_config))
        pool.add(factory.create_agent(merchant_config))

        results = pool.broadcast_percept("The town bell rings loudly.")
        assert len(results) == 2

    def test_export_all_memories(self, factory, guard_config, merchant_config):
        pool = AgentPool()
        pool.add(factory.create_agent(guard_config))
        pool.add(factory.create_agent(merchant_config))

        exports = pool.export_all_memories()
        assert "npc_guard" in exports
        assert "npc_merchant" in exports

    def test_shutdown(self, factory, guard_config):
        pool = AgentPool()
        pool.add(factory.create_agent(guard_config))
        pool.shutdown()
        assert pool.size == 0

    def test_missing_agent_raises(self):
        pool = AgentPool()
        with pytest.raises(KeyError, match="not found"):
            pool.get_agent("nonexistent")


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_pool_access(self, factory):
        """Multiple threads adding/running agents concurrently."""
        pool = AgentPool()
        errors: list[str] = []

        def worker(idx: int) -> None:
            try:
                cfg = AgentConfig(agent_id=f"agent_{idx}", personality=f"Agent {idx}")
                instance = factory.create_agent(cfg)
                pool.add(instance)
                result = pool.run_turn(f"agent_{idx}", "Hello")
                if result.error:
                    errors.append(f"agent_{idx}: {result.error}")
            except Exception as e:
                errors.append(f"agent_{idx}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert pool.size == 8

        pool.shutdown()

    def test_tool_registry_concurrent_access(self):
        """ToolRegistry handles concurrent register/deregister."""
        from maxim.tools.registry import ToolRegistry
        from maxim.tools.base import Tool

        class DummyTool(Tool):
            def __init__(self, name: str):
                self.name = name
                self.description = f"Dummy tool {name}"
                self.input_schema = {}

            def execute(self, **kwargs):
                return {"result": self.name}

        registry = ToolRegistry()
        errors: list[str] = []

        def writer(idx: int) -> None:
            try:
                for i in range(50):
                    registry.register(DummyTool(f"tool_{idx}_{i}"))
                    registry.list()
                    if i % 3 == 0:
                        registry.deregister(f"tool_{idx}_{i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # Should have registered tools (some deregistered)
        assert len(registry.list()) > 0
