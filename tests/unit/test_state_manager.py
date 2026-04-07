"""Unit tests for state manager."""

from __future__ import annotations


class TestStateManagerBasic:
    """Test basic state manager functionality."""

    def test_initial_state(self):
        """Initial state matches config."""
        from maxim.modes.state_manager import StateManager, StateManagerConfig

        config = StateManagerConfig(
            initial_mode="active",
            initial_processing_state="awake",
        )
        manager = StateManager(config)

        assert manager.operational_mode == "active"
        assert manager.processing_state == "awake"

    def test_default_state(self):
        """Default state is passive/awake."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()

        assert manager.operational_mode == "passive"
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
        manager.set_operational_mode("active")

        assert len(changes) == 1
        assert changes[0] == ("operational_mode", "passive", "active")

    def test_remove_callback(self):
        """Removed callbacks are not called."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()
        changes = []

        def callback(state_type, old, new):
            changes.append((state_type, old, new))

        manager.add_callback(callback)
        manager.remove_callback(callback)
        manager.set_operational_mode("active")

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

            def set_operational_mode(self, mode):
                notifications.append(("operational_mode", mode))

        manager.set_agent(MockAgent())
        manager.set_processing_state("sleep")
        manager.set_operational_mode("active")

        assert ("processing_state", "sleep") in notifications
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
        manager.request_shutdown()

        data = manager.to_dict()

        assert data["operational_mode"] == "active"
        assert data["shutdown_requested"] is True

    def test_from_dict(self):
        """State can be loaded from dict."""
        from maxim.modes.state_manager import StateManager

        manager = StateManager()
        manager.from_dict(
            {
                "operational_mode": "singularity",
                "processing_state": "sleep",
            }
        )

        assert manager.operational_mode == "singularity"
        assert manager.processing_state == "sleep"
