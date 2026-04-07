"""Unit tests for MemoryAgent.standalone() factory method (Phase 5)."""

from __future__ import annotations


class TestMemoryAgentStandalone:
    """Test MemoryAgent.standalone() factory."""

    def test_standalone_creates_agent(self):
        """MemoryAgent.standalone() creates working agent without external wiring."""
        from maxim.agents.memory_agent import MemoryAgent

        agent = MemoryAgent.standalone()
        assert agent is not None

    def test_standalone_has_bus(self):
        """Standalone agent creates its own AgentBus."""
        from maxim.agents.memory_agent import MemoryAgent

        agent = MemoryAgent.standalone()
        assert hasattr(agent, "_bus") or hasattr(agent, "bus")
