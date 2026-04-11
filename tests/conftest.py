"""Shared pytest fixtures for Maxim tests.

This module provides reusable fixtures for testing core components.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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
