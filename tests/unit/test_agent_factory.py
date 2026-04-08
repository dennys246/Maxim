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
