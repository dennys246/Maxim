"""Unit tests for build_memory_hub — the canonical MemoryHub construction door.

Tests the structural-enforcement invariants from memory_hub_unification.md:
- Builder always calls .connect() → three always-created bridges alive
- Required keyword-only core args (positional rejected)
- Explicit None opt-out for optional systems
- Bridge deps forwarded to .connect()

Raw MemoryHub() construction requires ``_allow_raw=True`` post-C6 flip
(PR #301) — tests that need a bare hub pass the keyword explicitly. The
structural enforcement lives at BOTH the type (TypeError when omitted)
AND the production door (the builder).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def core_systems():
    """Create the four required core bio-systems."""
    from maxim.decisions.nac import NAc
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.time.scn import SCN

    return {
        "hippocampus": Hippocampus(config=HippocampusConfig(auto_save_after_sleep=False)),
        "scn": SCN(),
        "nac": NAc(),
        "ec": EntorhinalCortex(),
    }


class TestBuildMemoryHub:
    """Tests for the build_memory_hub builder function."""

    # ── Signature enforcement ────────────────────────────────────────────

    def test_positional_args_rejected(self, core_systems):
        """All args are keyword-only — positional construction is a TypeError."""
        from maxim.integration.memory_hub import build_memory_hub

        with pytest.raises(TypeError, match="positional"):
            build_memory_hub(
                core_systems["hippocampus"],
                core_systems["scn"],
                core_systems["nac"],
                core_systems["ec"],
            )

    def test_missing_hippocampus_raises_type_error(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub

        with pytest.raises(TypeError):
            build_memory_hub(
                agent_id="default_agent",
                scn=core_systems["scn"],
                nac=core_systems["nac"],
                ec=core_systems["ec"],
            )

    def test_missing_scn_raises_type_error(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub

        with pytest.raises(TypeError):
            build_memory_hub(
                agent_id="default_agent",
                hippocampus=core_systems["hippocampus"],
                nac=core_systems["nac"],
                ec=core_systems["ec"],
            )

    def test_missing_nac_raises_type_error(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub

        with pytest.raises(TypeError):
            build_memory_hub(
                agent_id="default_agent",
                hippocampus=core_systems["hippocampus"],
                scn=core_systems["scn"],
                ec=core_systems["ec"],
            )

    def test_missing_ec_raises_type_error(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub

        with pytest.raises(TypeError):
            build_memory_hub(
                agent_id="default_agent",
                hippocampus=core_systems["hippocampus"],
                scn=core_systems["scn"],
                nac=core_systems["nac"],
            )

    # ── Core invariant: three always-created bridges are alive ───────────

    def test_returns_memory_hub_instance(self, core_systems):
        from maxim.integration.memory_hub import MemoryHub, build_memory_hub

        hub = build_memory_hub(agent_id="default_agent", **core_systems)
        assert isinstance(hub, MemoryHub)

    def test_plan_bridge_always_created(self, core_systems):
        """PlanHistoryBridge is alive on every hub from the builder.

        Pre-Wave-2, cli.py and agent_factory.py sites had this as None
        because .connect() was never called (Gap A + Gap B).
        """
        from maxim.integration.memory_hub import build_memory_hub

        hub = build_memory_hub(agent_id="default_agent", **core_systems)
        assert hub._plan_bridge is not None

    def test_escalation_bridge_always_created(self, core_systems):
        """EscalationLearningBridge is alive on every hub from the builder."""
        from maxim.integration.memory_hub import build_memory_hub

        hub = build_memory_hub(agent_id="default_agent", **core_systems)
        assert hub._escalation_bridge is not None

    def test_fear_bridge_always_created(self, core_systems):
        """FearCircuitBridge is alive on every hub from the builder."""
        from maxim.integration.memory_hub import build_memory_hub

        hub = build_memory_hub(agent_id="default_agent", **core_systems)
        assert hub._fear_bridge is not None

    # ── Optional bio-systems forwarded ───────────────────────────────────

    def test_atl_forwarded(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub
        from maxim.memory.atl import ATL, ATLConfig

        atl = ATL(config=ATLConfig())
        hub = build_memory_hub(agent_id="default_agent", **core_systems, atl=atl)
        assert hub.atl is atl

    def test_angular_gyrus_forwarded(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub
        from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig

        ag = AngularGyrus(config=AngularGyrusConfig())
        hub = build_memory_hub(agent_id="default_agent", **core_systems, angular_gyrus=ag)
        assert hub.angular_gyrus is ag

    def test_cerebellum_forwarded(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub

        sentinel = object()
        hub = build_memory_hub(agent_id="default_agent", **core_systems, cerebellum=sentinel)
        assert hub.cerebellum is sentinel

    def test_embodiment_forwarded(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub

        sentinel = object()
        hub = build_memory_hub(agent_id="default_agent", **core_systems, embodiment=sentinel)
        assert hub.embodiment is sentinel

    def test_worker_pool_forwarded(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub

        sentinel = object()
        hub = build_memory_hub(agent_id="default_agent", **core_systems, worker_pool=sentinel)
        assert hub.worker_pool is sentinel

    # ── Bridge deps forwarded to .connect() ──────────────────────────────

    def test_spatial_bridge_created_when_spatial_passed(self, core_systems):
        from unittest.mock import MagicMock

        from maxim.integration.memory_hub import build_memory_hub

        spatial = MagicMock()
        hub = build_memory_hub(agent_id="default_agent", **core_systems, spatial=spatial)
        assert hub._spatial_bridge is not None

    def test_salience_bridge_created_when_salience_passed(self, core_systems):
        from unittest.mock import MagicMock

        from maxim.integration.memory_hub import build_memory_hub

        salience = MagicMock()
        hub = build_memory_hub(agent_id="default_agent", **core_systems, salience=salience)
        assert hub._salience_bridge is not None

    def test_no_spatial_bridge_when_spatial_omitted(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub

        hub = build_memory_hub(agent_id="default_agent", **core_systems)
        assert hub._spatial_bridge is None

    def test_no_salience_bridge_when_salience_omitted(self, core_systems):
        from maxim.integration.memory_hub import build_memory_hub

        hub = build_memory_hub(agent_id="default_agent", **core_systems)
        assert hub._salience_bridge is None

    def test_fear_agent_stored(self, core_systems):
        """fear_agent is stored by .connect() but not used by any bridge today."""
        from unittest.mock import MagicMock

        from maxim.integration.memory_hub import build_memory_hub

        fear_agent = MagicMock()
        hub = build_memory_hub(agent_id="default_agent", **core_systems, fear_agent=fear_agent)
        assert hub._fear_agent is fear_agent

    # ── Regression guard: bare MemoryHub() requires explicit _allow_raw=True ─

    def test_raw_constructor_requires_explicit_allow_raw(self, core_systems):
        """Post-C6 hard-error flip (v1_refinement §C6), raw MemoryHub()
        construction without ``_allow_raw=True`` raises ``TypeError``.
        Tests that need a bare hub must opt in explicitly. Structural
        enforcement lives at the production door (build_memory_hub).
        """
        import pytest

        from maxim.integration.memory_hub import MemoryHub

        with pytest.raises(TypeError, match=r"build_memory_hub"):
            MemoryHub(**core_systems)

        # With explicit opt-in: bare construction works (no .connect()
        # called — bridges remain None).
        hub = MemoryHub(**core_systems, _allow_raw=True)
        assert hub._plan_bridge is None
        assert hub._escalation_bridge is None
        assert hub._fear_bridge is None

    # ── Gap A/B regression: builder fixes the silent-no-op shape ─────────

    def test_gap_a_regression_bridges_alive_with_minimal_args(self, core_systems):
        """Reproduces the Gap A/B bug shape: construct with only core systems.

        Pre-Wave-2, this was exactly what cli.py and agent_factory.py did.
        The builder must produce a hub with all three always-created bridges
        alive even when no optional args are passed.
        """
        from maxim.integration.memory_hub import build_memory_hub

        hub = build_memory_hub(agent_id="default_agent", **core_systems)

        # All three always-created bridges must be alive
        assert hub._plan_bridge is not None, "PlanHistoryBridge dead — Gap A regression"
        assert hub._escalation_bridge is not None, "EscalationLearningBridge dead — Gap A regression"
        assert hub._fear_bridge is not None, "FearCircuitBridge dead — Gap A regression"

        # Bridge methods should return non-default values (proving they're wired)
        # get_plan_templates returns [] from a real bridge (vs [] from no-bridge default)
        # The distinction is that the bridge.is_healthy is True
        assert hub.planning is not None
        assert hub.planning.is_healthy

        assert hub.escalation is not None
        assert hub.escalation.is_healthy

        assert hub.fear is not None
        assert hub.fear.is_healthy

    # ── Double .connect() guard (review fold I1) ─────────────────────────

    def test_double_connect_preserves_bridge_identity(self, core_systems):
        """Calling .connect() twice does NOT recreate the three always-created
        bridges — the guard prevents silent double-construction.

        This is the Reachy agentic_runtime.py pattern: build_memory_hub
        calls .connect() once, then the late spatial/salience wiring calls
        .connect() again.
        """
        from maxim.integration.memory_hub import build_memory_hub

        hub = build_memory_hub(agent_id="default_agent", **core_systems)

        # Capture bridge identity from builder
        plan_id = id(hub._plan_bridge)
        escalation_id = id(hub._escalation_bridge)
        fear_id = id(hub._fear_bridge)

        # Second .connect() — simulates the Reachy late-wiring pattern
        hub.connect()

        # Same objects, not recreated
        assert id(hub._plan_bridge) == plan_id
        assert id(hub._escalation_bridge) == escalation_id
        assert id(hub._fear_bridge) == fear_id


class TestOnSessionEndLightweight:
    """Tests for MemoryHub.on_session_end_lightweight (R0.2)."""

    def test_nac_decay_fires(self, core_systems):
        from maxim.integration.memory_hub import MemoryHub

        hub = MemoryHub(**core_systems, _allow_raw=True)
        hub._session_active = True
        hub._session_start_time = 0.0
        results = hub.on_session_end_lightweight()
        assert results.get("nac_decayed") is True

    def test_session_active_resets(self, core_systems):
        from maxim.integration.memory_hub import MemoryHub

        hub = MemoryHub(**core_systems, _allow_raw=True)
        hub._session_active = True
        hub._session_start_time = 0.0
        hub.on_session_end_lightweight()
        assert hub._session_active is False

    def test_idempotent_double_call(self, core_systems):
        from maxim.integration.memory_hub import MemoryHub

        hub = MemoryHub(**core_systems, _allow_raw=True)
        hub._session_active = True
        hub._session_start_time = 0.0
        hub.on_session_end_lightweight()
        # Second call returns empty (not active)
        assert hub.on_session_end_lightweight() == {}

    def test_lightweight_flag_in_results(self, core_systems):
        from maxim.integration.memory_hub import MemoryHub

        hub = MemoryHub(**core_systems, _allow_raw=True)
        hub._session_active = True
        hub._session_start_time = 0.0
        results = hub.on_session_end_lightweight()
        assert results.get("lightweight") is True

    def test_returns_session_duration(self, core_systems):
        import time

        from maxim.integration.memory_hub import MemoryHub

        hub = MemoryHub(**core_systems, _allow_raw=True)
        hub._session_active = True
        hub._session_start_time = time.time()
        results = hub.on_session_end_lightweight()
        assert "session_duration_seconds" in results
        assert results["session_duration_seconds"] >= 0.0
