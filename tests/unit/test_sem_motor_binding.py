"""SEM motor binding guards (sem_motor_binding.md Phase 1).

The orient modulator's spec-declared affordances become REAL body turns on
live via a backend attached through the existing ``attach_backends`` seam.
Pins, per the three-lens design review:

- SIM BYTE-IDENTITY: no factory → SpecModulator stub semantics, modeled
  self_effect + drive_potential_diff exactly as before;
- the factory binds ONLY the ``orient`` modulator, deltas read from the
  YAML's own ``self_effect["head_yaw"]``;
- the backend dispatches a body-yaw goto (head rides along via the
  controller's compose — no explicit head command), verifies actuation by
  readback, and a rejected/short motion is a REAL failure;
- build_executor unions the backend's ``world_owned_sensors`` into
  ``live_world_set_sensors`` and binds the embodiment;
- the credit-mill guard: a live-owned drive sensor in the declared effect
  yields NO modeled write, NO drive_potential_diff, and the
  ``drive_credit_withheld`` marker — and the consumer suppresses the flat
  +1 tool-success cluster floor on that marker (a real turn in a silent
  room must not mint direction-blind credit).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from maxim.embodiment.component_registry import ComponentRegistry
from maxim.hardware.reachy.motor_backend import (
    ReachyOrientMotorBackend,
    make_reachy_orient_factory,
)
from maxim.decisions.nac import NAc
from maxim.proprioception.pain_bus import build_pain_bus


def _bus():
    return build_pain_bus(hippocampus=None, nac=None)


def _nac():
    return NAc()


def _reachy():
    return ComponentRegistry().instantiate("bodies/reachy_mini")


def _orient_modulator(entity):
    for ent in entity.walk():
        mod = ent.modulators.get("orient")
        if mod is not None:
            return ent, mod
    raise AssertionError("reachy_mini has no orient modulator")


class _FakeRobot:
    def __init__(self, *, accept=True, body_yaw=0.0, track=True):
        self.targets = []
        self._accept = accept
        self._body_yaw = body_yaw
        self._track = track

    def is_connected(self):
        return True

    def get_current_pose(self):
        return {"yaw": self._body_yaw, "body_yaw": self._body_yaw}

    def goto_target(self, target):
        self.targets.append(target)
        if self._accept and self._track and target.body_yaw is not None:
            self._body_yaw = target.body_yaw
        return self._accept


class TestFactoryBinding:
    def test_binds_only_orient_with_yaml_deltas(self):
        entity = _reachy()
        factory = make_reachy_orient_factory(_FakeRobot())
        bound = {}
        for ent in entity.walk():
            for mname, mod in ent.modulators.items():
                backend = factory(ent, mname, mod)
                if backend is not None:
                    bound[mname] = backend
        assert set(bound) == {"orient"}
        deltas = bound["orient"]._deltas
        assert deltas["turn_left"] == pytest.approx(0.3)
        assert deltas["turn_right_big"] == pytest.approx(-0.9)

    def test_build_executor_attaches_and_claims_world_owned(self):
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        registry = ToolRegistry()
        executor = build_executor(
            registry,
            pain_bus=_bus(),
            nac=_nac(),
            entity_ref="bodies/reachy_mini",
            component_registry=ComponentRegistry(),
            modulator_factory=make_reachy_orient_factory(_FakeRobot()),
        )
        emb = executor.embodiment
        assert emb is not None
        assert "head_yaw" in emb.live_world_set_sensors
        assert "body_yaw" in emb.live_world_set_sensors
        _, mod = _orient_modulator(emb.root)
        backend = mod._backend
        assert isinstance(backend, ReachyOrientMotorBackend)
        assert backend._embodiment is emb  # bind_embodiment ran

    def test_no_factory_is_byte_identical_stub(self):
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        registry = ToolRegistry()
        executor = build_executor(
            registry,
            pain_bus=_bus(),
            nac=_nac(),
            entity_ref="bodies/reachy_mini",
            component_registry=ComponentRegistry(),
        )
        emb = executor.embodiment
        _, mod = _orient_modulator(emb.root)
        assert mod._backend is None
        assert "head_yaw" not in emb.live_world_set_sensors
        # Stub execute: modeled semantics, success True, credit flows.
        tool = executor.registry.get("reachy_mini_turn_left")
        out = tool.execute()
        assert out.success is True
        assert (out.side_effects or {}).get("drive_potential_diff") is not None
        assert "drive_credit_withheld" not in (out.side_effects or {})


class TestMotorBackendDispatch:
    def _backend(self, robot, maxim=None):
        entity = _reachy()
        factory = make_reachy_orient_factory(robot, maxim=maxim)
        ent, mod = _orient_modulator(entity)
        return factory(ent, "orient", mod), entity

    def test_turn_left_dispatches_body_only_goto(self):
        robot = _FakeRobot(body_yaw=0.0)
        backend, _ = self._backend(robot)
        result = backend.execute("turn_left", {})
        assert result.success is True
        assert len(robot.targets) == 1
        t = robot.targets[0]
        # Body-yaw command; head NOT explicitly commanded — the controller
        # composes the head to ride along (head-frame invariant machinery).
        assert t.body_yaw == pytest.approx(0.3)
        assert t.head_yaw is None
        assert result.metadata["reached"] is True

    def test_rejected_motion_is_real_failure(self):
        backend, _ = self._backend(_FakeRobot(accept=False))
        result = backend.execute("turn_left", {})
        assert result.success is False

    def test_confirmed_shortfall_is_failure(self):
        # Dispatch accepted but the body never moved (track=False).
        backend, _ = self._backend(_FakeRobot(track=False))
        result = backend.execute("turn_left_big", {})
        assert result.success is False
        assert result.metadata["reached"] is False

    def test_body_limit_clamps(self):
        robot = _FakeRobot(body_yaw=math.radians(159.0))
        backend, _ = self._backend(robot)
        result = backend.execute("turn_left_big", {})
        assert result.metadata["clamped_to_body_limit"] is True
        assert robot.targets[0].body_yaw == pytest.approx(math.radians(160.0))

    def test_unknown_affordance_fails(self):
        backend, _ = self._backend(_FakeRobot())
        assert backend.execute("fly", {}).success is False

    def test_dn_inhibited_and_synced(self):
        maxim = MagicMock()
        robot = _FakeRobot()
        backend, _ = self._backend(robot, maxim=maxim)
        backend.execute("turn_left", {})
        maxim._default_network.inhibit.assert_called_once()
        maxim._clear_motor_queue.assert_called_once()
        maxim.sync_head_position.assert_called_once()


class TestCreditMillGuard:
    """A motor-bound live turn must book NO modeled azimuth credit and must
    NOT fall through to the flat +1 cluster floor."""

    def _live_turn_output(self):
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        executor = build_executor(
            ToolRegistry(),
            pain_bus=_bus(),
            nac=_nac(),
            entity_ref="bodies/reachy_mini",
            component_registry=ComponentRegistry(),
            modulator_factory=make_reachy_orient_factory(_FakeRobot()),
        )
        # Simulate the DoAFeed's live claim on azimuth (the runtime does
        # this at feed construction).
        executor.embodiment.live_world_set_sensors.add("azimuth")
        tool = executor.registry.get("reachy_mini_turn_left")
        return tool.execute(), executor

    def test_no_modeled_azimuth_credit_and_withheld_marker(self):
        out, executor = self._live_turn_output()
        assert out.success is True  # real dispatch succeeded
        side = out.side_effects or {}
        assert "drive_potential_diff" not in side, (
            "modeled azimuth credit leaked on a live-owned sensor — the phantom credit mill is back"
        )
        assert side.get("drive_credit_withheld") is True

    def test_consumer_suppresses_floor_on_withheld(self):
        from maxim.runtime.tool_dispatch import record_outcome

        nac = MagicMock()
        record_outcome(
            agent_id="a",
            tool_name="reachy_mini_turn_left",
            success=True,
            result_summary="ok",
            error=None,
            reasoning="",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=MagicMock(),
            nac=nac,
            cluster_id="cluster-xyz",
            drive_credit_withheld=True,
        )
        assert not nac.update_cluster_reward.called, (
            "the flat +1 tool-success floor fired for a drive-touched-but-"
            "unmeasured live turn — silence-heavy sessions would mint "
            "direction-blind cluster credit (probe-3 floor drowning)"
        )

    def test_floor_still_flows_without_marker(self):
        from maxim.runtime.tool_dispatch import record_outcome

        nac = MagicMock()
        record_outcome(
            agent_id="a",
            tool_name="reachy_mini_turn_left",
            success=True,
            result_summary="ok",
            error=None,
            reasoning="",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=MagicMock(),
            nac=nac,
            cluster_id="cluster-xyz",
        )
        assert nac.update_cluster_reward.called


class TestReviewFoldGuards:
    """Guards for the two-lens review folds (sem_motor_binding.md)."""

    def test_pose_unreadable_refuses_to_guess(self):
        """E2: a missing body_yaw in the pre-pose must FAIL, not assume 0 -
        assuming would command a swing of up to the full body angle."""

        class _BlindRobot(_FakeRobot):
            def get_current_pose(self):
                return {"yaw": 0.1}  # joint read failed - no body_yaw

        entity = _reachy()
        factory = make_reachy_orient_factory(_BlindRobot())
        ent, mod = _orient_modulator(entity)
        backend = factory(ent, "orient", mod)
        result = backend.execute("turn_left", {})
        assert result.success is False
        assert "unreadable" in (result.error or "")

    def test_measured_body_yaw_world_set(self):
        """F5: the readback writes BOTH head_yaw and body_yaw into the
        entity sensors - the declared body_yaw sensor must not stay frozen
        at 0 while the body really rotates."""
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        executor = build_executor(
            ToolRegistry(),
            pain_bus=_bus(),
            nac=_nac(),
            entity_ref="bodies/reachy_mini",
            component_registry=ComponentRegistry(),
            modulator_factory=make_reachy_orient_factory(_FakeRobot()),
        )
        tool = executor.registry.get("reachy_mini_turn_left")
        out = tool.execute()
        assert out.success is True
        root = executor.embodiment.root
        assert root.vital_metrics["body_yaw"] == pytest.approx(0.3, abs=0.01)

    def test_always_active_filter_returns_the_orient_family_only(self):
        """The unions consume the body's ALWAYS-ACTIVE affordances (the
        reflexive turn_* vocabulary), not every goal-gated affordance."""
        from maxim.embodiment.tool_bridge import always_active_sem_tools
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        executor = build_executor(
            ToolRegistry(),
            pain_bus=_bus(),
            nac=_nac(),
            entity_ref="bodies/reachy_mini",
            component_registry=ComponentRegistry(),
        )
        names = {t.name for t in always_active_sem_tools(executor.registry)}
        # The reflexive vocabulary: the four turns + listen (all declared
        # always_active in the YAML). Goal-gated affordances (look_at,
        # antenna moves, ...) must NOT be in the set.
        assert {
            "reachy_mini_turn_left",
            "reachy_mini_turn_right",
            "reachy_mini_turn_left_big",
            "reachy_mini_turn_right_big",
        } <= names
        assert names <= {
            "reachy_mini_turn_left",
            "reachy_mini_turn_right",
            "reachy_mini_turn_left_big",
            "reachy_mini_turn_right_big",
            "reachy_mini_listen",
        }

    def test_learned_index_registration_makes_tools_renderable(self):
        """E1: the passive-mode filtered prompt renderer partitions the
        LearnedToolIndex's OWN universe - SEM tools must be registered into
        it post-build_executor or they render nowhere on live's default
        mode."""
        from maxim.embodiment.tool_bridge import always_active_sem_tools
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.learned_index import LearnedToolIndex
        from maxim.tools.registry import ToolRegistry

        executor = build_executor(
            ToolRegistry(),
            pain_bus=_bus(),
            nac=_nac(),
            entity_ref="bodies/reachy_mini",
            component_registry=ComponentRegistry(),
        )
        index = LearnedToolIndex()
        for t in always_active_sem_tools(executor.registry):
            index.register_tool(t)
        assert "reachy_mini_turn_left_big" in index._tool_keywords
