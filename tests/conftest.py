"""Shared pytest fixtures for Maxim tests.

This module provides reusable fixtures for testing core components.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import Mock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Multi-agent isolation fixture (P4 rule, CLAUDE.md L43). Importing
# the module is enough to register the `multi_agent_modes` fixture for
# all tests via the `pytest_plugins` mechanism below.
pytest_plugins = ["tests.multi_agent_fixtures"]


@pytest.fixture(autouse=True)
def _isolate_maxim_llm_profile_env():
    """Save and restore ``MAXIM_LLM_PROFILE`` across every test.

    Several paths in ``build_primary_router`` (the persisted-model
    restore at lane_backends.py:~741, the ``_maybe_pin_pre_upgrade_profile``
    migration, and ``_apply_local_llm_override``) read, set, or clear
    this env var as intentional side effects on the running process.
    Without isolation, a test that calls ``build_primary_router`` on a
    developer machine with a real ``~/.maxim/util/active_llm_model.txt``
    leaks the persisted profile into every subsequent test's env,
    breaking assertions that expect ``MAXIM_LLM_PROFILE`` to be absent
    or to match a specific value.

    This fixture runs for every test (not just tests/unit/) so CI
    machines with pre-existing Maxim state or test runs with --last
    behavior stay clean.
    """
    saved = os.environ.pop("MAXIM_LLM_PROFILE", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_LLM_PROFILE", None)
        if saved is not None:
            os.environ["MAXIM_LLM_PROFILE"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_role_env():
    """Scrub ``MAXIM_ROLE`` across every test (Plan 2 R2a).

    ``runtime/role.py::detect_and_apply_role`` exports the detected role
    to env so downstream code (``runtime/llm_server.py::_model_state_file``)
    can read it. Tests that exercise role detection or persisted-model
    paths would leak the value into every later test that constructs the
    runtime. Follow the auto-download scrub pattern: always unset on entry,
    restore any pre-existing user value on exit.
    """
    saved = os.environ.pop("MAXIM_ROLE", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_ROLE", None)
        if saved is not None:
            os.environ["MAXIM_ROLE"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_auto_download_env():
    """Scrub ``MAXIM_AUTO_DOWNLOAD_MODELS`` across every test.

    P5's ``ensure_available`` reads this env var to decide whether to
    skip the download prompt. ``cli_utils.normalize_args`` sets it when
    ``--auto-download`` is passed. Without isolation, the
    ``test_normalize_args_sets_env_when_flag_set`` test would leak the
    env var into every later test that calls ``build_primary_router``,
    causing a real GGUF download against the developer's home directory.

    Always start each test with the var unset; restore it on the way out
    so any user-set value survives the test session.
    """
    saved = os.environ.pop("MAXIM_AUTO_DOWNLOAD_MODELS", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_AUTO_DOWNLOAD_MODELS", None)
        if saved is not None:
            os.environ["MAXIM_AUTO_DOWNLOAD_MODELS"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_log_display_env():
    """Scrub log/display env vars introduced by feat/log-display-improvements.

    ``MAXIM_REPORT_JSON`` is set by ``cli.py`` when ``--report-json`` is
    passed and read by the orchestrator at sim-end to emit the report.
    ``MAXIM_LOG_FILE_MAX_BYTES`` and ``MAXIM_LOG_FILE_BACKUP_COUNT``
    are read by ``configure_logging`` when attaching the JSONL handler
    for ``MAXIM_LOG_FILE``.  Per the lessons-learned rule that opt-in
    env vars in hot startup paths need autouse scrubs (CLAUDE.md),
    pair every new ``MAXIM_*`` with isolation here so a CLI test that
    sets one of them does not leak into every later test that calls
    ``build_primary_router`` or ``configure_logging``.

    Always start each test with the vars unset; restore on the way out
    so any user-set values survive the test session.
    """
    keys = ("MAXIM_REPORT_JSON", "MAXIM_LOG_FILE_MAX_BYTES", "MAXIM_LOG_FILE_BACKUP_COUNT")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        yield
    finally:
        for k in keys:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]


@pytest.fixture(autouse=True)
def _isolate_maxim_nac_min_confidence():
    """Scrub ``MAXIM_NAC_MIN_CONFIDENCE`` across every test.

    Added in 0.9.1 (release_0_9_1.md Stage 0a). Overrides the
    ``min_confidence`` threshold in ``propose_via_substrate`` for
    Roy-2c (H1 vs H2 disambiguator). Per CLAUDE.md "opt-in env vars
    in hot startup paths need autouse scrubs", pair the env-var
    reader at agent_loop._resolve_min_confidence with this scrub so
    a CLI/Roy test that sets the var does not leak into every later
    test that constructs the agent loop.
    """
    saved = os.environ.pop("MAXIM_NAC_MIN_CONFIDENCE", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_NAC_MIN_CONFIDENCE", None)
        if saved is not None:
            os.environ["MAXIM_NAC_MIN_CONFIDENCE"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_ec_trace_env():
    """Scrub ``MAXIM_EC_TRACE_ACTIVATIONS`` across every test.

    Added in 0.9.1 (release_0_9_1.md Stage 0d). Gates per-tick
    ``sim_ec_activation`` JSONL emission from
    ``EntorhinalCortex.pattern_complete_or_separate``. The Roy-4
    iteration sets this in the runner environment to capture the
    co-activation matrix for the proposed cross-modal Hebbian binding
    rule validation. Per CLAUDE.md "opt-in env vars in hot startup
    paths need autouse scrubs", pair the env-var reader at
    ``similarity/ec.py::_ec_trace_enabled`` with this scrub so a test
    that sets the var does not leak emission events into every later
    test that constructs an EC.
    """
    saved = os.environ.pop("MAXIM_EC_TRACE_ACTIVATIONS", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_EC_TRACE_ACTIVATIONS", None)
        if saved is not None:
            os.environ["MAXIM_EC_TRACE_ACTIVATIONS"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_disable_cluster_bias_annotation():
    """Scrub ``MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION`` across every test.

    Added in 0.9.1 (release_0_9_1.md Stage 2). Gates Wire-A's
    cluster-bias annotation read at the agent-loop LLM-submission site.
    Default OFF (annotation ON) per the release plan; the Roy-3
    ablation iteration sets this to ``1`` to compare annotation-on
    vs annotation-off arms. Per CLAUDE.md "opt-in env vars in hot
    startup paths need autouse scrubs", pair the env-var reader at
    ``runtime/agent_loop.py`` with this scrub so a Roy-3 ablation test
    does not leak annotation-disabled state into every later test
    that constructs the agent loop.
    """
    saved = os.environ.pop("MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION", None)
        if saved is not None:
            os.environ["MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_disable_variance_annotation():
    """Scrub ``MAXIM_DISABLE_VARIANCE_ANNOTATION`` across every test.

    Added in 0.9.1 (release_0_9_1.md Stage 4 — Wire 1). Gates the
    risk-sensitive variance annotation read at the agent-loop
    LLM-submission site (mirrors Wire-A's ablation gate above).
    Default OFF (annotation ON) per the release plan; Roy-3 may
    set this to ``1`` to compare variance-annotation-on vs -off
    arms. Per CLAUDE.md "opt-in env vars in hot startup paths
    need autouse scrubs", pair the env-var reader at
    ``runtime/agent_loop.py`` with this scrub so a Roy-3 ablation
    test does not leak annotation-disabled state into every later
    test that constructs the agent loop.
    """
    saved = os.environ.pop("MAXIM_DISABLE_VARIANCE_ANNOTATION", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_DISABLE_VARIANCE_ANNOTATION", None)
        if saved is not None:
            os.environ["MAXIM_DISABLE_VARIANCE_ANNOTATION"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_disable_imagination_substrate_signal():
    """Scrub ``MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL`` across every test.

    Added in W2 MVP (imagination_substrate_signals.md Hookup 1). Gates
    the substrate-aware manifest read at the sim orchestrator scene-load
    site (parallel to Wire-A's ablation gate above). Default OFF
    (substrate-aware manifest ON); Roy iterations set this to ``1`` to
    measure W2's contribution by comparing on-vs-off arms. Per
    CLAUDE.md "opt-in env vars in hot startup paths need autouse
    scrubs", pair the env-var reader at ``simulation/orchestrator.py``
    with this scrub so a Roy ablation test does not leak the disabled
    state into every later test that constructs the AUT orchestrator.
    """
    saved = os.environ.pop("MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL", None)
        if saved is not None:
            os.environ["MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_deep_embodiment():
    """Reset ``resolution._resolved_depth`` + scrub ``MAXIM_DEEP_EMBODIMENT``.

    The module-level cache in ``embodiment/resolution.py`` persists across
    tests in the same process. Without isolation, any test that calls
    ``get_embodiment_depth()`` caches the value and leaks to later tests.
    """
    saved = os.environ.pop("MAXIM_DEEP_EMBODIMENT", None)
    try:
        from maxim.embodiment.resolution import reset_depth

        reset_depth()
    except ImportError:
        pass
    try:
        yield
    finally:
        try:
            from maxim.embodiment.resolution import reset_depth

            reset_depth()
        except ImportError:
            pass
        os.environ.pop("MAXIM_DEEP_EMBODIMENT", None)
        if saved is not None:
            os.environ["MAXIM_DEEP_EMBODIMENT"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_substrate_path_env():
    """Scrub ``MAXIM_SUBSTRATE_PATH`` across every test.

    P1's ``MemoryHub._wire_substrate_encoder`` reads this env var to
    decide whether to activate the substrate encoding path. Without
    isolation, tests that set this var leak the encoder into every
    later test that constructs a MemoryHub.
    """
    saved = os.environ.pop("MAXIM_SUBSTRATE_PATH", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_SUBSTRATE_PATH", None)
        if saved is not None:
            os.environ["MAXIM_SUBSTRATE_PATH"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_concept_decomposition_env():
    """Scrub ``MAXIM_CONCEPT_DECOMPOSITION`` across every test.

    ``MemoryHub._wire_substrate_encoder`` reads this env var to decide
    whether to construct a ConceptDecomposer. Without isolation, tests
    that set this var leak spaCy model loading into every later test
    that constructs a MemoryHub with MAXIM_SUBSTRATE_PATH=1.
    """
    saved = os.environ.pop("MAXIM_CONCEPT_DECOMPOSITION", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_CONCEPT_DECOMPOSITION", None)
        if saved is not None:
            os.environ["MAXIM_CONCEPT_DECOMPOSITION"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_llm_call_timeout_env():
    """Scrub ``MAXIM_LLM_CALL_TIMEOUT_S`` across every test (Plan 3.5 R2).

    ``LLMWorker.__init__`` reads this env var to override the agent-level
    LLM call timeout. Tests that set it to simulate fast timeouts or slow
    timeouts would leak the value into every later test that constructs
    an LLMWorker, breaking assertions about default behavior and causing
    flaky timing tests. Follow the Plan 2 R2a pattern: always unset on
    entry, restore any pre-existing user value on exit.
    """
    saved = os.environ.pop("MAXIM_LLM_CALL_TIMEOUT_S", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_LLM_CALL_TIMEOUT_S", None)
        if saved is not None:
            os.environ["MAXIM_LLM_CALL_TIMEOUT_S"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_auto_undrain_probe_interval_env():
    """Scrub ``MAXIM_AUTO_UNDRAIN_PROBE_INTERVAL_S`` across every test (Plan 4 C4.6).

    ``AutoUndrainProber`` reads this env var at construction. A test that
    sets a fast interval for unit testing would leak into later tests that
    construct the runtime, spawning real background threads with unexpected
    intervals. Same pattern as ``_isolate_maxim_auto_download_env``.
    """
    saved = os.environ.pop("MAXIM_AUTO_UNDRAIN_PROBE_INTERVAL_S", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_AUTO_UNDRAIN_PROBE_INTERVAL_S", None)
        if saved is not None:
            os.environ["MAXIM_AUTO_UNDRAIN_PROBE_INTERVAL_S"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_cancellation_contextvar():
    """Scrub the cancellation ``ContextVar`` between tests (Plan 3.5 R4).

    ``maxim.utils.cancellation._cancel_event_var`` is module-level state
    that persists across tests in the same pytest process. If a test (or
    a sim helper invoked from a test) calls ``set_cancel_event`` and
    leaks the binding without ``reset_cancel_event``, every later test
    that exercises ``LLMWorker._call_llm_with_timeout`` inherits the
    stale Event — which can cause confusing failures where the orphan
    thread sees a wrong Event reference and either fires cancellation
    spuriously or fails to fire it at all.

    Pattern: force the binding to None on test entry, restore the prior
    binding on exit. Same shape as the env-var scrubs above but applied
    to a ContextVar instead of os.environ.
    """
    from maxim.utils.cancellation import (
        reset_cancel_event,
        set_cancel_event,
    )

    scrub_token = set_cancel_event(None)
    try:
        yield
    finally:
        reset_cancel_event(scrub_token)


@pytest.fixture(autouse=True)
def _isolate_maxim_request_context_contextvar():
    """Scrub the ``RequestContext`` ``ContextVar`` between tests (Plan 4 A.2).

    Plan 4 A.2 added a boundary ``set_context(normalized_ctx)`` call in
    ``LLMWorker._call_llm_with_timeout`` so that
    ``maxim.utils.http._current_context`` populates X-Maxim-* outbound
    headers and acts as a fallback for
    ``_normalize_request_context(None)``. If a test leaks a binding
    without calling ``reset_context``, every later test that asserts on
    ``_normalize_request_context(None) is empty`` or that reads
    ``current_context()`` inherits the stale ``RequestContext``.

    This is the exact same bug class the pre-merge review flagged as
    latent in ``tests/unit/test_backend_error_taxonomy.py::
    test_normalize_request_context_handles_none``. The fixture forces
    the binding to None on test entry and restores the prior binding on
    exit — matching the cancellation-contextvar pattern above.
    """
    from maxim.utils.http import reset_context, set_context

    scrub_token = set_context(None)
    try:
        yield
    finally:
        reset_context(scrub_token)


@pytest.fixture(autouse=True)
def _isolate_discovery_state():
    """Reset SEM Tool Discovery module-level state between tests.

    ``tools/discovery.py`` has module-level mutable dicts (``_tool_last_used``
    and ``_goal_selected``) that persist across tests in the same process.
    A test that calls ``mark_tool_used("slash", 5)`` would leak the entry
    into every later test that calls ``evict_stale_discoveries``.

    Pattern: reset on entry so every test starts with a clean slate.
    """
    from maxim.tools.discovery import reset_discovery_state

    reset_discovery_state()
    yield
    reset_discovery_state()


@pytest.fixture(autouse=True)
def _isolate_maxim_nac_temporal_credit_weight_env():
    """Scrub ``MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT`` across every test.

    NAc.__init__ reads this env var to override the temporal credit weight
    for SCN-coupled eligibility. A test that sets a custom weight would
    leak into every later test that constructs a NAc instance.
    """
    saved = os.environ.pop("MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT", None)
        if saved is not None:
            os.environ["MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT"] = saved


@pytest.fixture(autouse=True)
def _isolate_maxim_nac_cluster_reward_bias_decay_tau_env():
    """Scrub ``MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU`` across every test.

    NAc.__init__ reads this env var to override the Wire-A cluster-keyed
    reward-bias decay timescale. A test that sets a custom tau would leak
    into every later test that constructs a NAc instance and silently
    change cluster bias decay rates project-wide.
    """
    saved = os.environ.pop("MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU", None)
    try:
        yield
    finally:
        os.environ.pop("MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU", None)
        if saved is not None:
            os.environ["MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU"] = saved


# ─────────────────────────────────────────────────────────────────────────────
# Memory Types Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_perception():
    """Standard test perception."""
    from maxim.memory.types import Perception

    return Perception(
        observations={"x": 320, "y": 240},
        detected_objects=["mug", "table"],
        detected_people=["person_1"],
        salience=0.6,
        novelty=0.5,
        cli_input=None,
        transcript=None,
    )


@pytest.fixture
def high_salience_perception():
    """High-salience perception for immediate promotion tests."""
    from maxim.memory.types import Perception

    return Perception(
        observations={"x": 320, "y": 240},
        detected_objects=["fire", "emergency"],
        detected_people=[],
        salience=0.98,
        novelty=0.95,
        cli_input="urgent situation",
        transcript=None,
    )


@pytest.fixture
def sample_context():
    """Standard test context."""
    from maxim.memory.types import Context

    return Context(
        active_goal="find mug",
        active_mode="exploration",
    )


@pytest.fixture
def sample_decision():
    """Standard test decision."""
    from maxim.memory.types import Decision

    return Decision(
        intent={"goal": "find mug", "action": "look_around"},
        reasoning="Looking for mug in workspace",
        confidence=0.8,
    )


@pytest.fixture
def sample_action():
    """Standard test action."""
    from maxim.memory.types import Action

    return Action(
        tool_name="look_around",
        tool_params={"direction": "left"},
    )


@pytest.fixture
def sample_outcome():
    """Standard successful outcome."""
    from maxim.memory.types import Outcome

    return Outcome(
        success=True,
        result={"found": True, "object": "mug"},
        error=None,
        evaluations=[],
    )


@pytest.fixture
def failed_outcome():
    """Standard failed outcome."""
    from maxim.memory.types import Outcome

    return Outcome(
        success=False,
        result=None,
        error="Target not found",
        evaluations=[],
    )


@pytest.fixture
def complete_memory_args(sample_perception, sample_context, sample_decision, sample_action, sample_outcome):
    """All arguments needed for Hippocampus.capture()."""
    return {
        "perception": sample_perception,
        "context": sample_context,
        "decision": sample_decision,
        "action": sample_action,
        "outcome": sample_outcome,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core System Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def hippocampus():
    """Fresh Hippocampus instance with no persistence."""
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    config = HippocampusConfig(
        persistence_path=None,  # No disk persistence
        enable_sleep_consolidation=True,
        auto_save_after_sleep=False,
    )
    return Hippocampus(config)


@pytest.fixture
def scn():
    """Fresh SCN instance."""
    from maxim.time.scn import SCN

    return SCN()


@pytest.fixture
def nac():
    """Fresh NAc instance."""
    from maxim.decisions.nac import NAc, NACConfig

    config = NACConfig(
        max_links=1000,
        min_confidence_threshold=0.3,
        base_learning_rate=0.2,
    )
    return NAc(config)


@pytest.fixture
def focus_learner(tmp_path):
    """Fresh FocusLearner with temp persistence."""
    from maxim.proprioception.focus_learner import FocusLearner, FocusLearnerConfig

    config = FocusLearnerConfig(
        initial_gain=0.7,
        learning_rate=0.2,
        persist_path=str(tmp_path / "focus_learner.json"),
        auto_save_interval=0,  # Disable auto-save for tests
        min_result_delay=0.0,  # No delay for testing
    )
    return FocusLearner(config=config)


@pytest.fixture
def escalation_bridge(hippocampus, scn, nac, tmp_path):
    """Fresh EscalationLearningBridge with temp persistence."""
    from maxim.bridges.escalation_bridge import EscalationLearningBridge

    return EscalationLearningBridge(
        hippocampus=hippocampus,
        scn=scn,
        nac=nac,
        persist_path=str(tmp_path / "escalation.json"),
        auto_save_interval=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def temporal_signature_morning():
    """Morning temporal signature (9am Monday)."""
    from maxim.time.temporal_signature import TemporalSignature

    return TemporalSignature(
        timestamp=1704103200.0,  # Example timestamp
        circadian_phase=9 / 24,  # 9am
        weekly_phase=0 / 7,  # Monday
        monthly_phase=0.25,  # Week 1
        annual_phase=0.0,  # January
    )


@pytest.fixture
def temporal_signature_evening():
    """Evening temporal signature (8pm Friday)."""
    from maxim.time.temporal_signature import TemporalSignature

    return TemporalSignature(
        timestamp=1704499200.0,  # Example timestamp
        circadian_phase=20 / 24,  # 8pm
        weekly_phase=4 / 7,  # Friday
        monthly_phase=0.75,  # Week 3
        annual_phase=0.5,  # June
    )


# ─────────────────────────────────────────────────────────────────────────────
# Causal Link Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def valence_positive():
    """Positive valence enum."""
    from maxim.decisions.causal_link import Valence

    return Valence.POSITIVE


@pytest.fixture
def valence_negative():
    """Negative valence enum."""
    from maxim.decisions.causal_link import Valence

    return Valence.NEGATIVE


@pytest.fixture
def valence_neutral():
    """Neutral valence enum."""
    from maxim.decisions.causal_link import Valence

    return Valence.NEUTRAL


# ─────────────────────────────────────────────────────────────────────────────
# Mock Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_planner():
    """Mock planner that returns configurable plans."""
    planner = Mock()
    planner.propose_plans = Mock(return_value=[])
    return planner


@pytest.fixture
def mock_policy():
    """Mock policy that accepts all and scores uniformly."""
    policy = Mock()
    policy.allow = Mock(return_value=True)
    policy.score = Mock(return_value=1.0)
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def create_memory_batch(hippocampus, count: int, **overrides) -> list[str]:
    """Create multiple memories with default values.

    Args:
        hippocampus: Hippocampus instance
        count: Number of memories to create
        **overrides: Override any memory component

    Returns:
        List of memory IDs
    """
    from maxim.memory.types import Action, Context, Decision, Outcome, Perception

    ids = []
    for i in range(count):
        perception = overrides.get(
            "perception",
            Perception(
                observations={"i": i},
                detected_objects=[f"obj_{i}"],
                detected_people=[],
                salience=0.5,
                novelty=0.5,
            ),
        )
        context = overrides.get(
            "context",
            Context(active_goal=f"goal_{i}", active_mode="exploration"),
        )
        decision = overrides.get(
            "decision",
            Decision(intent={"goal": f"goal_{i}"}, reasoning="test", confidence=0.8),
        )
        action = overrides.get(
            "action",
            Action(tool_name=f"tool_{i}", tool_params={}),
        )
        outcome = overrides.get(
            "outcome",
            Outcome(success=True, result={"i": i}),
        )

        memory_id = hippocampus.capture(
            perception=perception,
            context=context,
            decision=decision,
            action=action,
            outcome=outcome,
        )
        ids.append(memory_id)

    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Planning Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_bus():
    """Fresh AgentBus instance for plan tests."""
    from maxim.agents.bus import AgentBus

    return AgentBus()


@pytest.fixture
def long_horizon_config():
    """Default LongHorizonConfig for tests."""
    from maxim.planning.plan_document import LongHorizonConfig

    return LongHorizonConfig()


def make_phase(
    plan_id: str = "plan-1",
    index: int = 0,
    description: str = "Test phase",
    status: str = "PENDING",
    sub_goals: list | None = None,
    expected_inputs: dict | None = None,
    expected_outputs: dict | None = None,
    phase_id: str | None = None,
) -> Any:
    """Factory for creating test Phase instances."""
    from maxim.planning.plan_document import Phase, PhaseStatus

    return Phase(
        id=phase_id or f"phase-{plan_id}-{index}",
        description=description,
        status=PhaseStatus[status],
        plan_id=plan_id,
        index=index,
        sub_goals=sub_goals or [],
        expected_inputs=expected_inputs or {},
        expected_outputs=expected_outputs or {},
    )


def make_plan_document(
    num_phases: int = 3,
    objective: str = "Test objective",
    status: str = "ACTIVE",
    plan_id: str = "plan-1",
    with_sub_goals: bool = False,
) -> Any:
    """Factory for creating test PlanDocument instances.

    Creates phases with proper linking. If with_sub_goals is True,
    each phase gets 2 sub-goals with the second depending on the first.
    """
    from maxim.agents.bus import FailureStrategy, SubGoal, SubGoalStatus
    from maxim.planning.plan_document import Phase, PhaseStatus, PlanDocument, PlanStatus

    phases = []
    for i in range(num_phases):
        phase_status = PhaseStatus.ACTIVE if i == 0 else PhaseStatus.PENDING
        sgs = []
        if with_sub_goals:
            sg1 = SubGoal(
                id=f"sg-{plan_id}-{i}-0",
                description=f"Sub-goal {i}.0",
                tool_name=f"tool_{i}_0",
                tool_params={"phase": i, "step": 0},
                status=SubGoalStatus.PENDING,
                on_failure=FailureStrategy.REPLAN,
            )
            sg2 = SubGoal(
                id=f"sg-{plan_id}-{i}-1",
                description=f"Sub-goal {i}.1",
                tool_name=f"tool_{i}_1",
                tool_params={"phase": i, "step": 1},
                status=SubGoalStatus.PENDING,
                depends_on=[sg1.id],
                on_failure=FailureStrategy.RETRY,
            )
            sgs = [sg1, sg2]

        phase = Phase(
            id=f"phase-{plan_id}-{i}",
            description=f"Phase {i}: step {i}",
            status=phase_status,
            plan_id=plan_id,
            index=i,
            sub_goals=sgs,
            expected_outputs={f"output_{i}": f"result from phase {i}"},
        )
        if i == 0:
            phase.started_at = time.time()
        phases.append(phase)

    now = time.time()
    return PlanDocument(
        id=plan_id,
        objective=objective,
        created_at=now,
        updated_at=now,
        status=PlanStatus[status],
        phases=phases,
        current_phase_index=0,
    )


def make_plan_manager(
    tmp_path: Path,
    bus: Any = None,
    config: Any = None,
    services: Any = None,
) -> Any:
    """Factory for creating test PlanManager instances."""
    from maxim.agents.bus import AgentBus
    from maxim.planning.plan_document import LongHorizonConfig
    from maxim.planning.plan_manager import PlanManager, PlanServices

    plans_dir = str(tmp_path / "plans")
    return PlanManager(
        plans_dir=plans_dir,
        bus=bus or AgentBus(),
        config=config or LongHorizonConfig(),
        services=services or PlanServices(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM Fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm(monkeypatch):
    """Mock LLM router that returns canned responses.

    Patches build_primary_router() to return a fake router so tests
    don't need a real LLM backend.

    Usage::

        def test_something(mock_llm):
            router, manager = mock_llm
            result = router.generate("hello")
            assert result == '{"response": "mock"}'
    """

    class FakeRouter:
        def generate(self, prompt, **kw):
            return '{"response": "mock"}'

        def generate_json(self, prompt, **kw):
            return {"response": "mock"}

        def stop(self):
            pass

    class FakeLaneManager:
        def stop(self):
            pass

    router = FakeRouter()
    manager = FakeLaneManager()

    monkeypatch.setattr(
        "maxim.runtime.lane_backends.build_primary_router",
        lambda **kw: (router, manager),
    )
    return router, manager


# ─── HTTP response factory (Plan 1 R1 loose ends) ────────────────────────
#
# Tests can be migrated to this helper incrementally — Plan 2 R2b will use
# it for BackendError test fixtures. Existing hand-rolled Response stubs
# in tests/unit/test_http_client.py and elsewhere stay as-is until then.


def make_http_response(
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
    body: Any = None,
    elapsed_ms: float = 1.0,
    endpoint: str | None = None,
    request_id: str = "test-req-id",
):
    """Factory for ``maxim.utils.http.Response`` test stubs.

    Use this instead of hand-rolling ``Response(...)`` in every test. Saves
    duplication and keeps the stub shape consistent as the Response
    dataclass evolves.

    ``body`` accepts:
      - ``None`` → empty bytes
      - ``bytes`` → passed through verbatim
      - ``str`` → encoded as UTF-8
      - any other JSON-serializable object → ``json.dumps`` then UTF-8 encoded

    Usage as a fixture::

        def test_something(http_response_factory):
            resp = http_response_factory(status=429, body={"error": "rate limited"})

    Direct import also works::

        from tests.conftest import make_http_response
        resp = make_http_response(status=200)
    """
    import json as _json

    from maxim.utils import http as _http

    if body is None:
        content = b""
    elif isinstance(body, bytes):
        content = body
    elif isinstance(body, str):
        content = body.encode("utf-8")
    else:
        content = _json.dumps(body).encode("utf-8")

    return _http.Response(
        status=status,
        headers=dict(headers or {}),
        content=content,
        elapsed_ms=elapsed_ms,
        endpoint=endpoint or _http._EXTERNAL_ENDPOINT,
        request_id=request_id,
    )


@pytest.fixture
def http_response_factory():
    """Pytest fixture wrapper around :func:`make_http_response`."""
    return make_http_response
