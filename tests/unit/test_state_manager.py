"""Unit tests for state manager."""

from __future__ import annotations



class TestStateManagerBasic:
    """Test basic state manager functionality."""

    def test_initial_state(self):
        """Initial state matches config."""
        from maxim.modes.state_manager import StateManager, StateManagerConfig

        config = StateManagerConfig(
            initial_mode="active",
            initial_strategy="explore",
            initial_processing_state="awake",
        )
        manager = StateManager(config)

        assert manager.operational_mode == "active"
        assert manager.strategy == "explore"
        assert manager.processing_state == "awake"

    def test_default_state(self):
        """Default state is passive/observe/awake."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        assert manager.operational_mode == "passive"
        assert manager.strategy == "observe"
        assert manager.processing_state == "awake"


class TestStateManagerProcessingState:
    """Test processing state changes."""

    def test_set_processing_state(self):
        """Can set processing state."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        assert manager.set_processing_state("sleep") is True
        assert manager.processing_state == "sleep"
        assert manager.is_sleeping is True

    def test_set_processing_state_no_change(self):
        """Returns False when state unchanged."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        assert manager.set_processing_state("awake") is False

    def test_set_invalid_processing_state(self):
        """Invalid state returns False."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        assert manager.set_processing_state("invalid") is False


class TestStateManagerStrategy:
    """Test strategy changes."""

    def test_set_strategy(self):
        """Can set strategy."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        assert manager.set_strategy("explore") is True
        assert manager.strategy == "explore"

    def test_set_strategy_auto_wake(self):
        """Strategy change wakes from sleep."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()
        manager.set_processing_state("sleep")

        manager.set_strategy("explore")

        assert manager.processing_state == "awake"

    def test_set_strategy_no_auto_wake(self):
        """Can disable auto-wake on strategy change."""
        from maxim.modes.state_manager import StateManager, StateManagerConfig

        config = StateManagerConfig(auto_wake_on_strategy_change=False)
        manager = StateManager(config)
        manager.set_processing_state("sleep")

        manager.set_strategy("explore")

        assert manager.processing_state == "sleep"


class TestStateManagerOperationalMode:
    """Test operational mode changes."""

    def test_set_operational_mode(self):
        """Can set operational mode."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        assert manager.set_operational_mode("active") is True
        assert manager.operational_mode == "active"
        assert manager.is_active is True

    def test_set_singularity_mode(self):
        """Can set singularity mode."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        assert manager.set_operational_mode("singularity") is True
        assert manager.operational_mode == "singularity"
        assert manager.is_active is True

    def test_set_invalid_mode(self):
        """Invalid mode returns False."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        assert manager.set_operational_mode("invalid") is False


class TestStateManagerCallbacks:
    """Test callback functionality."""

    def test_callback_called_on_change(self):
        """Callbacks are called on state changes."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()
        changes = []

        def callback(state_type, old, new):
            changes.append((state_type, old, new))

        manager.add_callback(callback)
        manager.set_strategy("explore")

        assert len(changes) == 1
        assert changes[0] == ("strategy", "observe", "explore")

    def test_remove_callback(self):
        """Removed callbacks are not called."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()
        changes = []

        def callback(state_type, old, new):
            changes.append((state_type, old, new))

        manager.add_callback(callback)
        manager.remove_callback(callback)
        manager.set_strategy("explore")

        assert len(changes) == 0


class TestStateManagerAgent:
    """Test agent notification."""

    def test_notifies_agent(self):
        """Agent methods are called on state changes."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()
        notifications = []

        class MockAgent:
            def set_processing_state(self, state):
                notifications.append(("processing_state", state))

            def set_strategy(self, strategy):
                notifications.append(("strategy", strategy))

            def set_operational_mode(self, mode):
                notifications.append(("operational_mode", mode))

        manager.set_agent(MockAgent())
        manager.set_processing_state("sleep")
        manager.set_strategy("explore")
        manager.set_operational_mode("active")

        assert ("processing_state", "sleep") in notifications
        assert ("strategy", "explore") in notifications
        assert ("operational_mode", "active") in notifications


class TestStateManagerConvenienceMethods:
    """Test convenience methods."""

    def test_request_shutdown(self):
        """request_shutdown sets flag."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        assert manager.shutdown_requested is False
        manager.request_shutdown()
        assert manager.shutdown_requested is True

    def test_strategy_convenience_methods(self):
        """Strategy convenience methods work."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        manager.request_strategy_explore()
        assert manager.strategy == "explore"

        manager.request_strategy_research()
        assert manager.strategy == "research"

        manager.request_strategy_assist()
        assert manager.strategy == "assist"

    def test_mode_convenience_methods(self):
        """Mode convenience methods work."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        manager.request_mode_active()
        assert manager.operational_mode == "active"

        manager.request_mode_passive()
        assert manager.operational_mode == "passive"


class TestStateManagerSerialization:
    """Test state serialization."""

    def test_to_dict(self):
        """State can be serialized to dict."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()
        manager.set_operational_mode("active")
        manager.set_strategy("explore")
        manager.request_shutdown()

        data = manager.to_dict()

        assert data["operational_mode"] == "active"
        assert data["strategy"] == "explore"
        assert data["shutdown_requested"] is True

    def test_from_dict(self):
        """State can be loaded from dict."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()
        manager.from_dict({
            "operational_mode": "singularity",
            "strategy": "research",
            "processing_state": "sleep",
        })

        assert manager.operational_mode == "singularity"
        assert manager.strategy == "research"
        assert manager.processing_state == "sleep"
