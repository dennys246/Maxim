"""Smoke tests for the 6 original public API verbs + list_models.

These tests mock LLM and hardware dependencies so they run offline.
They verify that each verb can be called without crashing and returns
the expected type.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# configure
# ─────────────────────────────────────────────────────────────────────────────


def test_configure_sets_verbosity():
    """configure() completes without error and sets agentic verbosity."""
    from maxim.api import configure

    # Should not raise
    configure(verbosity=0)
    configure(verbosity=2)
    configure(verbosity=3)


def test_configure_sets_debug_traces():
    """configure(debug='hippo,nac') sets the corresponding env vars."""
    from maxim.api import configure

    configure(debug="hippo,nac")
    assert os.environ.get("MAXIM_HIPPO_TRACE") == "1"
    assert os.environ.get("MAXIM_NAC_TRACE") == "1"

    # Clean up
    os.environ.pop("MAXIM_HIPPO_TRACE", None)
    os.environ.pop("MAXIM_NAC_TRACE", None)


# ─────────────────────────────────────────────────────────────────────────────
# diagnose
# ─────────────────────────────────────────────────────────────────────────────


def test_diagnose_returns_report():
    """diagnose() returns a DiagnosticReport with platform info and results."""
    from maxim.api import DiagnosticReport, diagnose

    report = diagnose()
    assert isinstance(report, DiagnosticReport)
    assert report.platform is not None
    assert isinstance(report.sections, list)


# ─────────────────────────────────────────────────────────────────────────────
# observe / introspect
# ─────────────────────────────────────────────────────────────────────────────


def test_observe_returns_dict():
    """observe() returns a dict with subsystem keys."""
    from maxim.api import observe

    result = observe()
    assert isinstance(result, dict)
    # Should have at least some known subsystem keys
    assert "memory" in result or "hippocampus" in result or len(result) > 0


def test_introspect_is_observe_alias():
    """introspect() returns the same result as observe()."""
    import maxim

    # Both should work and return dicts
    r1 = maxim.observe()
    r2 = maxim.introspect()
    assert type(r1) is type(r2)


# ─────────────────────────────────────────────────────────────────────────────
# list_models
# ─────────────────────────────────────────────────────────────────────────────


def test_list_models_returns_grouped_dict():
    """list_models() returns dict with 'local' and 'cloud' keys."""
    from maxim.api import ModelInfo, list_models

    result = list_models()
    assert "local" in result
    assert "cloud" in result
    assert len(result["local"]) > 0
    assert len(result["cloud"]) > 0

    # Every entry should be a ModelInfo
    for m in result["local"]:
        assert isinstance(m, ModelInfo)
        assert m.cloud is False
    for m in result["cloud"]:
        assert isinstance(m, ModelInfo)
        assert m.cloud is True
        assert m.api_key_env  # cloud models must have an env var


def test_list_models_importable_from_maxim():
    """list_models is accessible as maxim.list_models()."""
    import maxim

    result = maxim.list_models()
    assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# _validate_model
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_model_accepts_known_local_profile():
    """Valid local model names pass validation without error."""
    from maxim.api import _validate_model

    # Local models don't need SDK or API key — should not raise
    _validate_model("mistral-7b")
    _validate_model("smollm-1.7b")


def test_validate_model_rejects_unknown_profile():
    """Unknown model name raises ConfigurationError."""
    from maxim.api import _validate_model
    from maxim.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="Unknown model"):
        _validate_model("nonexistent-model-xyz")


def test_validate_model_checks_cloud_requirements():
    """Cloud model without SDK or API key raises ConfigurationError."""
    from maxim.api import _validate_model
    from maxim.exceptions import ConfigurationError

    # Temporarily remove the key
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        # Should raise for either missing SDK or missing API key
        with pytest.raises(ConfigurationError):
            _validate_model("claude-sonnet")
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


# ─────────────────────────────────────────────────────────────────────────────
# connect (error case)
# ─────────────────────────────────────────────────────────────────────────────


def test_connect_missing_robot_raises():
    """connect() with a nonexistent robot type raises a clear error."""
    from maxim.api import connect

    with pytest.raises(Exception):
        connect("nonexistent_robot_xyz_999", timeout=1.0)


# ───────────────────────────────────────────────────────────────────────────
# run contract (D15/D16)
# ────────────────────────────────────────────────────────────────────────


class _RunTestWorker:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _RunTestAgent:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.goal = SimpleNamespace(robot=None)
        self.__class__.instances.append(self)

    def wire_robot(self, robot):
        self.goal.robot = robot


@pytest.fixture
def mocked_run_runtime(monkeypatch):
    """Replace the heavy run stack while preserving facade data flow."""
    import maxim.api as api

    _RunTestWorker.instances.clear()
    _RunTestAgent.instances.clear()
    router = SimpleNamespace(n_ctx=8192, get_token_counter=lambda: None)
    state = SimpleNamespace(data={})
    registry = MagicMock(name="tool_registry")
    executor = MagicMock(name="executor")
    loop_calls = []
    registry_calls = []

    monkeypatch.setattr(api, "_validate_model", lambda model: None)
    monkeypatch.setattr(api, "_resolve_model", lambda model: model)
    monkeypatch.setattr(api, "configure", lambda **kwargs: None)
    monkeypatch.setattr(api, "_inject_pending_tools", lambda tool_registry: None)
    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", lambda: (router, None))
    monkeypatch.setattr("maxim.agents.llm_worker.LLMWorker", _RunTestWorker)
    monkeypatch.setattr("maxim.agents.maxim_agent.MaximAgent", _RunTestAgent)
    monkeypatch.setattr("maxim.runtime.bootstrap.build_environment", lambda **kwargs: MagicMock())
    monkeypatch.setattr("maxim.runtime.bootstrap.build_state", lambda: state)
    monkeypatch.setattr("maxim.runtime.bootstrap.build_memory", lambda: MagicMock())
    monkeypatch.setattr("maxim.runtime.bootstrap.build_decision_engine", lambda: MagicMock())
    monkeypatch.setattr("maxim.runtime.bootstrap.build_executor", lambda *args, **kwargs: executor)
    monkeypatch.setattr("maxim.runtime.bootstrap.build_evaluators", lambda: [])

    def build_tool_registry(**kwargs):
        registry_calls.append(kwargs)
        return registry

    def run_agentic_loop(**kwargs):
        loop_calls.append(kwargs)

    monkeypatch.setattr("maxim.runtime.bootstrap.build_tool_registry", build_tool_registry)
    monkeypatch.setattr("maxim.runtime.agent_loop.run_agentic_loop", run_agentic_loop)

    return SimpleNamespace(
        api=api,
        state=state,
        registry=registry,
        executor=executor,
        loop_calls=loop_calls,
        registry_calls=registry_calls,
    )


def test_run_delivers_goal_through_runtime_input_mailbox(tmp_path, mocked_run_runtime):
    """A stable goal argument must become observable loop input."""
    mocked_run_runtime.api.run(
        model="test-profile",
        goal="inspect the workspace",
        home_dir=str(tmp_path),
        learning=False,
    )

    assert mocked_run_runtime.state.data["pending_cli_input"] == "inspect the workspace"
    assert len(mocked_run_runtime.loop_calls) == 1


def test_run_rejects_robot_in_headless_mode(tmp_path, mocked_run_runtime):
    """Contradictory hardware options fail loudly instead of ignoring robot."""
    from maxim.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="headless=False"):
        mocked_run_runtime.api.run(
            model="test-profile",
            robot="simulated",
            home_dir=str(tmp_path),
            learning=False,
        )


def test_run_rejects_blank_goal(tmp_path, mocked_run_runtime):
    """A present-but-empty goal cannot silently degrade to interactive mode."""
    from maxim.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="non-empty"):
        mocked_run_runtime.api.run(
            model="test-profile",
            goal="   ",
            home_dir=str(tmp_path),
            learning=False,
        )

    assert _RunTestWorker.instances == []


def test_run_rejects_concurrent_invocation(tmp_path, mocked_run_runtime):
    """Process-global routing state cannot be mutated by overlapping runs."""
    from maxim.exceptions import ConfigurationError

    assert mocked_run_runtime.api._RUN_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(ConfigurationError, match="Only one maxim.run"):
            mocked_run_runtime.api.run(
                model="test-profile",
                goal="second run",
                home_dir=str(tmp_path),
                learning=False,
            )
    finally:
        mocked_run_runtime.api._RUN_LOCK.release()


def test_run_wires_and_disconnects_robot(tmp_path, mocked_run_runtime, monkeypatch):
    """The acquired controller reaches live tools and its lease is released."""
    robot = SimpleNamespace(robot_id="api-robot", disconnect=MagicMock())
    lease = mocked_run_runtime.api._RunRobotLease(
        controller=robot,
        registration_key="simulated",
        owns_connection=True,
        woke_for_run=True,
    )
    release = MagicMock()
    monkeypatch.setattr(mocked_run_runtime.api, "_connect_robot_for_run", lambda robot_type: lease)
    monkeypatch.setattr(mocked_run_runtime.api, "_release_run_robot", release)

    mocked_run_runtime.api.run(
        model="test-profile",
        robot="simulated",
        headless=False,
        home_dir=str(tmp_path),
        learning=False,
    )

    assert mocked_run_runtime.registry_calls == [{"maxim": robot}]
    assert _RunTestAgent.instances[0].goal.robot is robot
    release.assert_called_once_with(lease)


def test_run_restores_environment_and_stops_worker_on_setup_failure(
    tmp_path,
    mocked_run_runtime,
    monkeypatch,
):
    """Cleanup covers failures after worker start but before loop entry."""
    monkeypatch.setenv("MAXIM_LLM_ENABLED", "original-enabled")
    monkeypatch.setenv("MAXIM_LLM_PROFILE", "original-profile")

    class BrokenAgent:
        def __init__(self, **kwargs):
            raise RuntimeError("agent setup failed")

    monkeypatch.setattr("maxim.agents.maxim_agent.MaximAgent", BrokenAgent)

    with pytest.raises(RuntimeError, match="agent setup failed"):
        mocked_run_runtime.api.run(
            model="test-profile",
            home_dir=str(tmp_path),
            learning=False,
        )

    assert _RunTestWorker.instances[0].started is True
    assert _RunTestWorker.instances[0].stopped is True
    assert os.environ["MAXIM_LLM_ENABLED"] == "original-enabled"
    assert os.environ["MAXIM_LLM_PROFILE"] == "original-profile"


def test_run_disconnects_robot_on_late_setup_failure(tmp_path, mocked_run_runtime, monkeypatch):
    """A controller connected during setup is closed if registry build fails."""
    robot = SimpleNamespace(robot_id="api-robot", disconnect=MagicMock())
    lease = mocked_run_runtime.api._RunRobotLease(
        controller=robot,
        registration_key="simulated",
        owns_connection=True,
        woke_for_run=True,
    )
    release = MagicMock()
    monkeypatch.setattr(mocked_run_runtime.api, "_connect_robot_for_run", lambda robot_type: lease)
    monkeypatch.setattr(mocked_run_runtime.api, "_release_run_robot", release)
    monkeypatch.setattr(
        "maxim.runtime.bootstrap.build_tool_registry",
        MagicMock(side_effect=RuntimeError("registry setup failed")),
    )

    with pytest.raises(RuntimeError, match="registry setup failed"):
        mocked_run_runtime.api.run(
            model="test-profile",
            robot="simulated",
            headless=False,
            home_dir=str(tmp_path),
            learning=False,
        )

    assert _RunTestWorker.instances[0].stopped is True
    release.assert_called_once_with(lease)


def test_run_leaves_preexisting_robot_connected(tmp_path, mocked_run_runtime, monkeypatch):
    """run() must not tear down a controller that the caller already owned."""
    robot = SimpleNamespace(robot_id="api-robot", disconnect=MagicMock())
    lease = mocked_run_runtime.api._RunRobotLease(
        controller=robot,
        registration_key="simulated",
        owns_connection=False,
        woke_for_run=False,
    )
    monkeypatch.setattr(mocked_run_runtime.api, "_connect_robot_for_run", lambda robot_type: lease)

    mocked_run_runtime.api.run(
        model="test-profile",
        robot="simulated",
        headless=False,
        home_dir=str(tmp_path),
        learning=False,
    )

    assert _RunTestAgent.instances[0].goal.robot is robot
    robot.disconnect.assert_not_called()


def test_run_robot_lease_wakes_sleeps_and_unregisters(monkeypatch):
    """A fresh run-owned controller is operational and fully unwound."""
    import maxim.api as api
    from maxim.hardware.registry import RobotRegistry

    RobotRegistry.reset_instance()
    try:
        lease = api._connect_robot_for_run("simulated")
        registry = RobotRegistry()

        assert lease.owns_connection is True
        assert lease.woke_for_run is True
        assert lease.controller.state.is_awake is True
        assert registry.get_robot(lease.registration_key) is lease.controller

        api._release_run_robot(lease)

        assert lease.controller.state.is_awake is False
        assert lease.controller.is_connected() is False
        assert registry.get_robot(lease.registration_key) is None
    finally:
        RobotRegistry.reset_instance()


def test_run_robot_lease_preserves_preexisting_awake_controller():
    """Caller-owned awake state and connection survive run cleanup."""
    import maxim.api as api
    from maxim.hardware.registry import RobotRegistry
    from maxim.hardware.simulation import SimulatedController

    RobotRegistry.reset_instance()
    registry = RobotRegistry()
    registry.register_controller_type("simulated", SimulatedController)
    controller = registry.connect_robot(robot_id="simulated", robot_type="simulated")
    assert controller is not None and controller.wake_up()
    try:
        lease = api._connect_robot_for_run("simulated")
        assert lease.owns_connection is False
        assert lease.woke_for_run is False

        api._release_run_robot(lease)

        assert controller.is_connected() is True
        assert controller.state.is_awake is True
    finally:
        registry.disconnect_all()
        RobotRegistry.reset_instance()


def test_run_robot_lease_uses_registry_key_when_controller_id_differs():
    """Plugin-defined controller IDs cannot leave stale registry entries."""
    import maxim.api as api
    from maxim.hardware.registry import RobotRegistry
    from maxim.hardware.simulation import SimulatedController

    class PluginController(SimulatedController):
        def __init__(self):
            super().__init__(robot_id="plugin-internal-id")

    RobotRegistry.reset_instance()
    registry = RobotRegistry()
    registry.register_controller_type("plugin", PluginController)
    try:
        lease = api._connect_robot_for_run("plugin")
        assert lease.registration_key == "plugin"
        assert lease.controller.robot_id == "plugin-internal-id"

        api._release_run_robot(lease)

        assert registry.get_robot("plugin") is None
    finally:
        registry.disconnect_all()
        RobotRegistry.reset_instance()


def test_run_robot_lease_does_not_mutate_existing_primary():
    """Explicit run routing does not claim process-global primary ownership."""
    import maxim.api as api
    from maxim.hardware.registry import RobotRegistry
    from maxim.hardware.simulation import SimulatedController

    RobotRegistry.reset_instance()
    registry = RobotRegistry()
    registry.register_controller_type("simulated", SimulatedController)
    registry.register_controller_type("selected", SimulatedController)
    primary = registry.connect_robot(robot_id="primary", robot_type="simulated", set_primary=True)
    try:
        lease = api._connect_robot_for_run("selected")
        assert registry.primary is primary

        api._release_run_robot(lease)

        assert registry.primary is primary
    finally:
        registry.disconnect_all()
        RobotRegistry.reset_instance()


def test_run_robot_acquisition_unwinds_keyboard_interrupt():
    """Interrupting wake-up restores state and removes a fresh registration."""
    import maxim.api as api
    from maxim.hardware.registry import RobotRegistry
    from maxim.hardware.simulation import SimulatedController

    class InterruptedController(SimulatedController):
        instances = []

        def __init__(self, robot_id):
            super().__init__(robot_id)
            self.slept = False
            self.__class__.instances.append(self)

        def wake_up(self):
            raise KeyboardInterrupt

        def goto_sleep(self):
            self.slept = True
            return super().goto_sleep()

    RobotRegistry.reset_instance()
    registry = RobotRegistry()
    registry.register_controller_type("interrupted", InterruptedController)
    try:
        with pytest.raises(KeyboardInterrupt):
            api._connect_robot_for_run("interrupted")

        controller = InterruptedController.instances[0]
        assert controller.slept is True
        assert controller.is_connected() is False
        assert registry.get_robot("interrupted") is None
    finally:
        registry.disconnect_all()
        RobotRegistry.reset_instance()


def test_run_robot_release_retains_connection_when_sleep_fails():
    """Unknown motor state must remain recoverable through the registry."""
    import maxim.api as api
    from maxim.exceptions import HardwareError
    from maxim.hardware.registry import RobotRegistry
    from maxim.hardware.simulation import SimulatedController

    class SleeplessController(SimulatedController):
        def goto_sleep(self):
            return False

    RobotRegistry.reset_instance()
    registry = RobotRegistry()
    registry.register_controller_type("sleepless", SleeplessController)
    try:
        lease = api._connect_robot_for_run("sleepless")

        with pytest.raises(HardwareError, match="connection was retained"):
            api._release_run_robot(lease)

        assert registry.get_robot("sleepless") is lease.controller
        assert lease.controller.is_connected() is True
    finally:
        registry.disconnect_all()
        RobotRegistry.reset_instance()


def test_run_robot_release_retains_connection_when_disconnect_fails():
    """Transport-close failure is surfaced without losing the recovery handle."""
    import maxim.api as api
    from maxim.exceptions import HardwareError
    from maxim.hardware.registry import RobotRegistry
    from maxim.hardware.simulation import SimulatedController

    class FailedDisconnectController(SimulatedController):
        def disconnect(self):
            raise RuntimeError("transport still live")

    RobotRegistry.reset_instance()
    registry = RobotRegistry()
    registry.register_controller_type("failed_disconnect", FailedDisconnectController)
    try:
        lease = api._connect_robot_for_run("failed_disconnect")

        with pytest.raises(HardwareError, match="connection was retained"):
            api._release_run_robot(lease)

        assert registry.get_robot("failed_disconnect") is lease.controller
    finally:
        RobotRegistry.reset_instance()


def test_run_robot_wake_failure_reports_disconnect_retention():
    """A failed wake never hides a transport that could not be released."""
    import maxim.api as api
    from maxim.exceptions import MaximConnectionError
    from maxim.hardware.registry import RobotRegistry
    from maxim.hardware.simulation import SimulatedController

    class FailedWakeAndDisconnectController(SimulatedController):
        def wake_up(self):
            return False

        def disconnect(self):
            raise RuntimeError("transport still live")

    RobotRegistry.reset_instance()
    registry = RobotRegistry()
    registry.register_controller_type("failed_wake_disconnect", FailedWakeAndDisconnectController)
    try:
        with pytest.raises(MaximConnectionError, match="connection was retained"):
            api._connect_robot_for_run("failed_wake_disconnect")

        assert registry.get_robot("failed_wake_disconnect") is not None
    finally:
        RobotRegistry.reset_instance()


def test_run_shutdowns_bio_instance(tmp_path, mocked_run_runtime, monkeypatch):
    """The stable facade persists and shuts down a learning-enabled instance."""
    instance = SimpleNamespace(
        executor=mocked_run_runtime.executor,
        hippocampus=None,
        memory_hub=None,
        pain_bus=None,
        shutdown=MagicMock(),
    )

    class FakeFactory:
        def __init__(self, **kwargs):
            pass

        def create_full_agent(self, config, *, tool_registry):
            return instance

    monkeypatch.setattr("maxim.runtime.agent_factory.AgentFactory", FakeFactory)

    mocked_run_runtime.api.run(
        model="test-profile",
        goal="learn safely",
        home_dir=str(tmp_path),
        learning=True,
    )

    instance.shutdown.assert_called_once_with()


def test_run_cleanup_continues_after_worker_stop_failure(tmp_path, mocked_run_runtime, monkeypatch):
    """One cleanup failure cannot skip robot release or env restoration."""
    robot = SimpleNamespace(robot_id="api-robot")
    lease = mocked_run_runtime.api._RunRobotLease(
        controller=robot,
        registration_key="simulated",
        owns_connection=True,
        woke_for_run=True,
    )
    release = MagicMock()
    monkeypatch.setattr(mocked_run_runtime.api, "_connect_robot_for_run", lambda robot_type: lease)
    monkeypatch.setattr(mocked_run_runtime.api, "_release_run_robot", release)
    monkeypatch.setenv("MAXIM_LLM_ENABLED", "before-enabled")
    monkeypatch.setenv("MAXIM_LLM_PROFILE", "before-profile")

    def fail_stop(self):
        self.stopped = True
        raise RuntimeError("worker stop failed")

    monkeypatch.setattr(_RunTestWorker, "stop", fail_stop)

    with pytest.raises(RuntimeError, match="worker stop failed"):
        mocked_run_runtime.api.run(
            model="test-profile",
            goal="cleanup",
            robot="simulated",
            headless=False,
            home_dir=str(tmp_path),
            learning=False,
        )

    release.assert_called_once_with(lease)
    assert os.environ["MAXIM_LLM_ENABLED"] == "before-enabled"
    assert os.environ["MAXIM_LLM_PROFILE"] == "before-profile"


# ─────────────────────────────────────────────────────────────────────────────
# Thread safety
# ─────────────────────────────────────────────────────────────────────────────


def test_event_subscription_thread_safe():
    """on() and unsubscribe() work under concurrent access."""
    import threading

    from maxim.api import on

    handles = []
    errors = []

    def subscribe_many():
        try:
            for i in range(50):
                h = on("test_event", lambda e: None)
                handles.append(h)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=subscribe_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(handles) == 200

    # Unsubscribe all
    for h in handles:
        h.unsubscribe()
