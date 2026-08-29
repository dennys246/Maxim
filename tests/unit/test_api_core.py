"""Smoke tests for the 6 original public API verbs + list_models.

These tests mock LLM and hardware dependencies so they run offline.
They verify that each verb can be called without crashing and returns
the expected type.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import logging

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
    monkeypatch.setattr(api, "_inject_registered_tools", lambda tool_registry: None)
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


# ─────────────────────────────────────────────────────────────────────────────
# N2 (score card 2026-08-27, Runtime "Upgrade to C+"): the API's own shutdown
# must produce loadable state. Black-box, through the PUBLIC verbs only.
#
# Verified to fail on the pre-fix runtime: `AgentInstance.shutdown()` calls
# `memory_hub.on_session_end()`, an atomic test-and-CLEAR that returns {} when no
# session was started — and neither create.agent() nor load.agent() opened one. The
# cycle wrote ONLY hippocampus.json + nac.json (saved directly by shutdown) and
# dropped ec.json / scn.json / atl.json, so every later load.agent() logged
# "Half-present NAc/EC pair" — the orphaned-bias state D2/D17 were fixed to DETECT,
# produced by the API itself. Root cause fixed at the lifecycle (the instance opens
# the session it later closes), not by making shutdown save more things.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def api_home(tmp_path, monkeypatch):
    """An isolated MAXIM_DATA_HOME with the path caches reset around it."""
    from maxim.utils.paths import _reset_caches

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("MAXIM_LLM_ENABLED", "0")
    _reset_caches()
    yield tmp_path
    _reset_caches()


def _mutate_substrate(hub) -> str:
    """Write one SCN signature, one EC signature and one ATL concept. Returns the concept id."""
    from maxim.similarity.ec import SituationSignature
    from maxim.time.temporal_signature import TemporalSignature

    hub.scn.register("probe_memory_1", TemporalSignature.now(), significance=0.9)
    hub.ec.register(
        "probe_memory_1",
        signature=SituationSignature(
            semantic_hash=(1, 2, 3),
            structural_hash=42,
            temporal_hash=(9, 3, 1, 8),
            context_hash=7,
            tool_name="probe_tool",
            outcome_type="success",
            mode="test",
            goal_keywords=("probe",),
        ),
    )
    concept_id, _ = hub.atl.find_or_create("probe_concept", category="test", definition="a probe")
    return concept_id


def test_create_mutate_shutdown_load_round_trips_ec_scn_atl(api_home, caplog):
    """create.agent → mutate SCN/EC/ATL → shutdown() → load.agent() (N2)."""
    from maxim import create, load

    caplog.set_level(logging.WARNING)
    agent = create.agent("api_round_trip")
    concept_id = _mutate_substrate(agent.memory_hub)
    assert len(agent.memory_hub.scn) == 1
    assert len(agent.memory_hub.ec) == 1
    agent.shutdown()

    agent_dir = api_home / "agents" / "api_round_trip"
    for name in ("ec.json", "scn.json", "atl.json", "nac.json", "hippocampus.json"):
        assert (agent_dir / name).exists(), f"{name} was not written by the API's own shutdown()"

    caplog.clear()
    reloaded = load.agent("api_round_trip")
    try:
        hub = reloaded.memory_hub
        assert hub.scn.get_signature("probe_memory_1") is not None, "SCN signature did not round-trip"
        assert hub.ec.get_signature("probe_memory_1") is not None, "EC signature did not round-trip"
        assert hub.atl.get(concept_id) is not None, "ATL concept did not round-trip"
        assert len(hub.scn) == 1 and len(hub.ec) == 1
        half_present = [r.getMessage() for r in caplog.records if "Half-present" in r.getMessage()]
        assert not half_present, f"load.agent() warned on the API's own output: {half_present}"
    finally:
        reloaded.shutdown()


def test_load_mutate_shutdown_load_round_trips(api_home):
    """The same contract on the LOAD path — a second session must persist too."""
    from maxim import create, load

    create.agent("api_second_session").shutdown()
    first = load.agent("api_second_session")
    concept_id = _mutate_substrate(first.memory_hub)
    first.shutdown()

    second = load.agent("api_second_session")
    try:
        assert second.memory_hub.scn.get_signature("probe_memory_1") is not None
        assert second.memory_hub.ec.get_signature("probe_memory_1") is not None
        assert second.memory_hub.atl.get(concept_id) is not None
    finally:
        second.shutdown()


def test_session_start_is_idempotent(api_home):
    """The runtime also opens a session (start_bio_session) on an adopted instance;
    a second open must be a no-op, not a re-restore that clears ATL."""
    from maxim import create

    agent = create.agent("api_idempotent")
    try:
        concept_id, _ = agent.memory_hub.atl.find_or_create("kept", category="test", definition="in memory only")
        assert agent.memory_hub.on_session_start() == {"already_active": 1}
        assert agent.memory_hub.atl.get(concept_id) is not None, "a second session start discarded in-memory ATL state"
    finally:
        agent.shutdown()


def test_shutdown_is_idempotent_and_does_not_reopen(api_home, caplog):
    """A second shutdown() must not raise, resurrect the session, or re-save."""
    from maxim import create

    agent = create.agent("api_double_shutdown")
    _mutate_substrate(agent.memory_hub)
    agent.shutdown()
    assert agent.memory_hub._session_active is False
    ec = api_home / "agents" / "api_double_shutdown" / "ec.json"
    assert ec.exists()
    stamp = (ec.stat().st_mtime_ns, ec.read_bytes())

    caplog.set_level(logging.WARNING)
    agent.shutdown()
    assert agent.memory_hub._session_active is False
    assert (ec.stat().st_mtime_ns, ec.read_bytes()) == stamp, "the second shutdown re-wrote persisted state"
    # Quiet: this hub DID have a session, the first shutdown closed it. The D41 guard
    # fires only for a hub where none was ever opened (its own test below).
    assert not [r for r in caplog.records if "no session ever started" in r.getMessage()]


def test_full_agent_construction_opens_the_session_on_the_hub_it_keeps(api_home):
    """create_full_agent REPLACES instance.memory_hub with the bio-stack's hub, so a
    session opened during create_agent lands on a throwaway skeleton and the real hub
    stays closed — D41 one layer down, on the path every runtime caller uses
    (api.run, cli, orchestrator AUT/NPC, console handle). Both lenses of the review
    round found it; reproduced black-box before the fix:
        with_bio_stack=True → shutdown wrote ONLY hippocampus.json + nac.json.
    """
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.time.temporal_signature import TemporalSignature

    factory = AgentFactory()
    agent = factory.create_full_agent(AgentConfig(agent_id="full_round_trip", with_bio_stack=True))
    assert agent.memory_hub._session_active is True, "the session was opened on a hub that was then discarded"
    agent.memory_hub.scn.register("probe_memory_1", TemporalSignature.now(), significance=0.9)
    concept_id, _ = agent.memory_hub.atl.find_or_create("probe_concept", category="test", definition="a probe")
    agent.shutdown()

    agent_dir = api_home / "agents" / "full_round_trip"
    for name in ("ec.json", "scn.json", "atl.json"):
        assert (agent_dir / name).exists(), f"{name} was dropped by the bio-stack construction path"

    reloaded = factory.create_full_agent(AgentConfig(agent_id="full_round_trip", with_bio_stack=True), auto_load=True)
    try:
        # D42: build_bio_stack used to construct a PATHLESS SCN, so on_session_end had
        # nothing to save to and temporal state vanished between runtime sessions.
        assert reloaded.memory_hub.scn.persistence_path is not None
        assert reloaded.memory_hub.scn.get_signature("probe_memory_1") is not None
        assert reloaded.memory_hub.atl.get(concept_id) is not None
    finally:
        reloaded.shutdown()


def test_the_loop_closing_its_own_session_does_not_make_shutdown_cry_wolf(api_home, caplog):
    """The agent loop opens and closes a session AROUND a run, nested inside the
    instance's lifetime. That is normal, so the D41 detector must stay quiet — a guard
    that fires on every clean runtime shutdown is a guard nobody reads (second review
    round found it firing on api.run(), the CLI, the orchestrator and MaximHandle.stop()).
    It must still fire for a hub where NO session was ever opened, which is the defect.
    """
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.runtime.bio_integration import end_bio_session, start_bio_session

    agent = AgentFactory().create_full_agent(AgentConfig(agent_id="loop_nesting", with_bio_stack=True))
    enabled = start_bio_session(memory_hub=agent.memory_hub, hippocampus=agent.hippocampus)
    end_bio_session(memory_hub=agent.memory_hub, hippocampus=agent.hippocampus, memory_hub_enabled=enabled)
    assert agent.memory_hub._session_active is False, "the loop should have closed the session"

    caplog.set_level(logging.WARNING)
    agent.shutdown()
    cried = [r.getMessage() for r in caplog.records if "no session ever started" in r.getMessage()]
    assert not cried, f"the D41 guard fired on a normal loop-then-shutdown sequence: {cried}"


def test_a_hub_that_never_opened_a_session_still_warns_on_both_closers(caplog):
    """The defect itself: closing a session nobody opened persists nothing."""
    import threading

    from maxim.integration.memory_hub import MemoryHub

    for closer in ("on_session_end", "on_session_end_lightweight"):
        hub = MemoryHub.__new__(MemoryHub)
        hub._session_active = False
        hub._session_ever_started = False
        hub._session_flag_lock = threading.Lock()
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert getattr(MemoryHub, closer)(hub) == {}
        assert any("no session ever started" in r.getMessage() for r in caplog.records), closer


def test_write_but_dont_read_agents_do_not_restore_scn(api_home):
    """`load_persisted=False` (the orchestrator NPC) must not read a previous
    session's temporal state — the D42 fix must not change that contract."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.time.temporal_signature import TemporalSignature

    factory = AgentFactory()
    first = factory.create_full_agent(AgentConfig(agent_id="npc_style", with_bio_stack=True))
    first.memory_hub.scn.register("probe_memory_1", TemporalSignature.now(), significance=0.9)
    first.shutdown()

    assert (api_home / "agents" / "npc_style" / "scn.json").exists(), (
        "nothing was persisted, so the assertion below would pass vacuously"
    )
    npc = factory.create_full_agent(
        AgentConfig(agent_id="npc_style", with_bio_stack=True, load_persisted=False), auto_load=False
    )
    try:
        assert npc.memory_hub.scn.get_signature("probe_memory_1") is None
    finally:
        npc.shutdown()


def test_a_corrupt_scn_file_is_preserved_not_overwritten(api_home, caplog):
    """D42 bound scn.json to the runtime path, which means session end can now REWRITE
    it — so an unreadable file must be moved aside first, never silently destroyed."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    factory = AgentFactory()
    first = factory.create_full_agent(AgentConfig(agent_id="corrupt_scn", with_bio_stack=True))
    first.shutdown()
    scn_json = api_home / "agents" / "corrupt_scn" / "scn.json"
    assert scn_json.exists()
    scn_json.write_text("{not json at all")

    caplog.set_level(logging.WARNING)
    second = factory.create_full_agent(AgentConfig(agent_id="corrupt_scn", with_bio_stack=True))
    try:
        assert second.memory_hub.scn.persistence_path is None, (
            "a corrupt scn.json stayed bound, so session end would overwrite it"
        )
        assert any("unreadable" in r.getMessage() for r in caplog.records)
    finally:
        second.shutdown()
    assert scn_json.read_text() == "{not json at all", "the unreadable SCN file was destroyed"
