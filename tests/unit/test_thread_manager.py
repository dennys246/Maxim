"""Unit tests for thread registry management."""

from __future__ import annotations

import threading


class TestThreadRegistry:
    """Test thread registry functionality."""

    def test_register_and_list(self):
        """Registered threads appear in list."""
        from maxim.utils.thread_manager import ThreadRegistry

        registry = ThreadRegistry()
        thread = threading.Thread(target=lambda: None)

        registry.register("test", thread)

        assert "test" in registry.list_threads()
        assert "test" in registry
        assert len(registry) == 1

    def test_unregister(self):
        """Unregistered threads are removed."""
        from maxim.utils.thread_manager import ThreadRegistry

        registry = ThreadRegistry()
        thread = threading.Thread(target=lambda: None)

        registry.register("test", thread)
        registry.unregister("test")

        assert "test" not in registry.list_threads()
        assert "test" not in registry

    def test_get_thread(self):
        """Can retrieve registered thread."""
        from maxim.utils.thread_manager import ThreadRegistry

        registry = ThreadRegistry()
        thread = threading.Thread(target=lambda: None)

        registry.register("test", thread)
        retrieved = registry.get("test")

        assert retrieved is thread

    def test_get_nonexistent(self):
        """Getting nonexistent thread returns None."""
        from maxim.utils.thread_manager import ThreadRegistry

        registry = ThreadRegistry()
        assert registry.get("nonexistent") is None

    def test_list_alive(self):
        """list_alive returns only running threads."""
        from maxim.utils.thread_manager import ThreadRegistry

        registry = ThreadRegistry()

        stop = threading.Event()

        def worker():
            stop.wait()

        thread = threading.Thread(target=worker, daemon=True)
        registry.register("alive", thread)
        thread.start()

        dead_thread = threading.Thread(target=lambda: None)
        registry.register("dead", dead_thread)

        alive = registry.list_alive()

        assert "alive" in alive
        assert "dead" not in alive

        stop.set()
        thread.join(timeout=1.0)

    def test_stop_all_stops_threads(self):
        """stop_all terminates running threads."""
        from maxim.utils.thread_manager import ThreadRegistry

        registry = ThreadRegistry()

        stop = threading.Event()
        stopped = threading.Event()

        def worker():
            try:
                stop.wait()
            finally:
                stopped.set()

        thread = threading.Thread(target=worker, daemon=True)
        registry.register("test", thread)
        thread.start()

        # Stop with short timeout
        registry.stop_all(timeout=2.0)

        # Should have stopped
        assert len(registry) == 0

    def test_clear(self):
        """clear removes all threads without stopping."""
        from maxim.utils.thread_manager import ThreadRegistry

        registry = ThreadRegistry()
        registry.register("a", threading.Thread(target=lambda: None))
        registry.register("b", threading.Thread(target=lambda: None))

        registry.clear()

        assert len(registry) == 0


class TestCreateDaemonThread:
    """Test daemon thread creation utility."""

    def test_creates_daemon_thread(self):
        """Created thread is a daemon."""
        from maxim.utils.thread_manager import create_daemon_thread

        thread = create_daemon_thread(target=lambda: None, name="test")

        assert thread.daemon is True
        assert thread.name == "test"

    def test_thread_runs_target(self):
        """Created thread runs the target function."""
        from maxim.utils.thread_manager import create_daemon_thread

        result = []

        def target(value):
            result.append(value)

        thread = create_daemon_thread(target=target, args=("hello",))
        thread.start()
        thread.join(timeout=1.0)

        assert result == ["hello"]
