"""Tests for ``runtime/bootstrap.py::build_executor``.

The canonical agent-construction site after the executor bootstrap
unification plan landed. This file replaces ``test_embodiment_bootstrap.py``
which exercised the previous helper-discipline shape.

Three identical bug instances (sem_execution_hook Stages 1, 2, 2c)
showed that "remember to call the helper after build_executor" is not a
strong enough invariant. ``build_executor`` now requires an explicit
``pain_bus=`` decision — forgetting becomes a ``TypeError`` instead of a
silent no-op.

See ``docs/plans/archive/executor_bootstrap_unification.md``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Required-keyword contract — the structural enforcement
# ---------------------------------------------------------------------------


class TestBuildExecutorRequiredKeyword:
    """``pain_bus`` is a required keyword-only arg with no default. The
    structural intent: forgetting the bridge decision is impossible.
    ``pain_bus=None`` is a legal explicit opt-out; missing the kwarg
    entirely is a TypeError."""

    def test_missing_pain_bus_kwarg_raises_type_error(self):
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        # Tight match: must be the missing-keyword-only TypeError, not
        # some other TypeError that incidentally contains "pain_bus".
        with pytest.raises(TypeError, match=r"missing.*keyword-only argument.*pain_bus"):
            build_executor(ToolRegistry())  # type: ignore[call-arg]

    def test_pain_bus_none_is_explicit_opt_out_no_bridge(self):
        """Sandbox executors / headless tests pass ``pain_bus=None``
        explicitly. The executor exists but has no bridge; tool
        outcomes are not learned."""
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        executor = build_executor(ToolRegistry(), pain_bus=None, permissions=None)

        assert executor is not None
        assert executor._tool_pain_bridge is None

    def test_positional_pain_bus_rejected_keyword_only(self):
        """``pain_bus`` is keyword-only — passing it positionally
        should fail at the signature level."""
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        with pytest.raises(TypeError):
            build_executor(ToolRegistry(), PainBus(_allow_raw=True))  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Bridge wiring — fires when the caller opts in
# ---------------------------------------------------------------------------


class TestBuildExecutorBridgeWiring:
    """When ``pain_bus`` (or ``pain_detector``) is provided, the bridge
    is constructed and attached to the inner Executor. This is the
    invariant that closed the CLI gap in sem_execution_hook Stage 2 —
    now structurally enforced."""

    def test_bridge_wired_with_nac_only_no_subscription_source(self):
        """C2 cross-confirmed regression guard: the bridge's PRIMARY
        value is direct attribution via record_tool_embodiment_failure
        (Stage 1). Subscription via pain_bus/pain_detector is the
        SECONDARY out-of-band path. A caller with NAc but no
        subscription source (e.g., the sim orchestrator's AUT, where
        NAc is subscribed to the bus directly via
        create_pain_nac_subscriber) MUST get a bridge for direct
        attribution. Pre-fold the bridge was gated on
        pain_bus|pain_detector — direct-attribution-only callers had
        to pass a no-op PainDetector to trick the constructor."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()

        executor = build_executor(
            ToolRegistry(),
            pain_bus=None,
            permissions=None,
            pain_detector=None,
            nac=nac,
        )

        assert isinstance(executor._tool_pain_bridge, ToolPainBridge), (
            "build_executor with nac=NAc, pain_bus=None, pain_detector=None "
            "must construct a bridge for direct attribution. The bridge is "
            "the primary path for record_tool_embodiment_failure (Stage 1)."
        )

    def test_no_bridge_when_nac_is_none(self):
        """Inverse of the above: explicit opt-out via nac=None
        produces no bridge regardless of any subscription source."""
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        executor = build_executor(
            ToolRegistry(),
            pain_bus=None,
            permissions=None,
            nac=None,
        )

        assert executor._tool_pain_bridge is None

    def test_bridge_wired_when_pain_bus_provided(self):
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)

        executor = build_executor(
            ToolRegistry(),
            pain_bus=pain_bus,
            permissions=None,
            nac=nac,
        )

        assert isinstance(executor._tool_pain_bridge, ToolPainBridge)

    def test_bridge_wired_when_pain_detector_provided(self):
        """Legacy Reachy path uses ``pain_detector`` instead of a
        PainBus. Same bridge construction, different subscription."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_detector = MagicMock()

        executor = build_executor(
            ToolRegistry(),
            pain_bus=None,
            permissions=None,
            pain_detector=pain_detector,
            nac=nac,
        )

        assert isinstance(executor._tool_pain_bridge, ToolPainBridge)

    def test_no_bridge_without_nac_raises(self):
        """A bridge with ``nac=None`` has no meaningful behavior. Fail
        fast so the caller passes NAc or opts out explicitly."""
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        with pytest.raises(ValueError, match="nac"):
            build_executor(
                ToolRegistry(),
                pain_bus=PainBus(_allow_raw=True),
                permissions=None,
                nac=None,
            )


# ---------------------------------------------------------------------------
# Fail-fast preconditions — checked BEFORE any construction
# ---------------------------------------------------------------------------


class TestBuildExecutorFailFastPreconditions:
    """Precondition checks must run BEFORE any object construction so
    a ValueError leaves no half-built state. Cross-confirmed Stage 2
    review finding: pre-fold the helper built the bridge BEFORE
    checking, leaving a dead-subscription bridge stuck on the
    executor when callers recovered from the exception."""

    def test_pain_bus_and_pain_detector_together_raises(self):
        """Passing both is ambiguous — ToolPainBridge uses pain_bus if
        set and silently drops pain_detector. Fail fast."""
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)
        pain_detector = MagicMock()

        with pytest.raises(ValueError, match="pain_bus AND pain_detector"):
            build_executor(
                ToolRegistry(),
                pain_bus=pain_bus,
                permissions=None,
                pain_detector=pain_detector,
                nac=nac,
            )

    def test_entity_ref_without_pain_bus_raises(self):
        """Embodiment._publish_pain emits through the bus — without a
        bus, embodiment failures fire into the void."""
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()

        with pytest.raises(ValueError, match="pain_bus"):
            build_executor(
                ToolRegistry(),
                pain_bus=None,
                permissions=None,
                nac=nac,
                entity_ref="weapons/rusty_sword",
                component_registry=ComponentRegistry(),
            )

    def test_entity_ref_without_component_registry_raises(self):
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)

        with pytest.raises(ValueError, match="component_registry"):
            build_executor(
                ToolRegistry(),
                pain_bus=pain_bus,
                permissions=None,
                nac=nac,
                entity_ref="weapons/rusty_sword",
                component_registry=None,
            )


# ---------------------------------------------------------------------------
# Embodiment loading — activates when entity_ref is provided
# ---------------------------------------------------------------------------


class TestBuildExecutorEmbodiment:
    """When ``entity_ref`` is provided, the function loads the
    component, wraps it in ``Embodiment``, and registers affordance
    tools. Uses bundled ``weapons/rusty_sword`` — no mocks in the SEM
    chain."""

    def test_rusty_sword_ref_registers_affordance_tools(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)
        registry = ToolRegistry()

        executor = build_executor(
            registry,
            pain_bus=pain_bus,
            permissions=None,
            nac=nac,
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
        )

        # The embodiment is stashed on the executor for caller access.
        embodiment = executor.embodiment
        assert isinstance(embodiment, Embodiment)

        tool_names = set(executor.registry.list())
        for expected in (
            "rusty_sword_slash",
            "rusty_sword_parry",
            "rusty_sword_throw",
            "rusty_sword_sharpen",
            "rusty_sword_repair",
        ):
            assert expected in tool_names, f"expected {expected!r} in executor.registry, got {sorted(tool_names)}"

    def test_embodiment_holds_pain_bus_reference(self):
        """The Embodiment must hold the pain_bus reference so
        evaluate_failures can publish PainSignals — the cascade Stage 1
        fixed depends on this link."""
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)

        executor = build_executor(
            ToolRegistry(),
            pain_bus=pain_bus,
            permissions=None,
            nac=nac,
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
        )

        embodiment = executor.embodiment
        assert embodiment._pain_bus is pain_bus

    def test_missing_entity_ref_raises_with_actionable_hint(self):
        """A typo in entity_ref must fail with a hint listing real
        alternatives — never silently no-op."""
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.exceptions import ComponentNotFoundError
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = MagicMock()
        pain_bus = PainBus(_allow_raw=True)

        with pytest.raises(ComponentNotFoundError) as exc_info:
            build_executor(
                ToolRegistry(),
                pain_bus=pain_bus,
                permissions=None,
                nac=nac,
                entity_ref="weapons/nonexistent_sword",
                component_registry=ComponentRegistry(),
            )
        msg = str(exc_info.value)
        assert "weapons/nonexistent_sword" in msg
        assert "weapons/rusty_sword" in msg


# ---------------------------------------------------------------------------
# End-to-end cascade through the new constructor
# ---------------------------------------------------------------------------


class TestBuildExecutorEndToEndCascade:
    """The integration story: after build_executor, invoking a
    generated affordance tool through executor.execute drives the full
    Stage 1 pain cascade. NAc records a NEGATIVE link on failure. Any
    failure here means the cascade is broken at one of: bridge
    construction, executor wiring, embodiment loading, side_effects
    routing, or NAc record_tool_embodiment_failure."""

    def test_rusty_sword_slash_to_shatter_records_negative(self):
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        nac = NAc()
        pain_bus = PainBus(_allow_raw=True)

        executor = build_executor(
            ToolRegistry(),
            pain_bus=pain_bus,
            permissions=None,
            nac=nac,
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
        )

        # Drive durability to zero so shatter fires on the next
        # evaluate_failures() call. Deterministic at this durability —
        # verified 2026-04-14, regression-guarded here.
        embodiment = executor.embodiment
        sword = embodiment.root
        sword.vital_metrics["durability"] = 0.0

        result = executor.execute(
            {
                "tool_name": "rusty_sword_slash",
                "params": {"target": "dummy", "force": 0.9},
            }
        )

        assert result.success is True
        assert result.side_effects is not None
        assert result.side_effects.get("embodiment_failures"), (
            "rusty_sword at durability=0.0 did not produce "
            "embodiment_failures; check embodiment.evaluate_failures() "
            "wiring in ModulatorAffordanceTool.execute"
        )

        prediction = nac.predict(
            event_type="tool",
            event_signature="tool:rusty_sword_slash",
        )
        assert prediction is not None, (
            "NAc has no prediction for tool:rusty_sword_slash after a "
            "deterministic shatter. The cascade is broken — either the "
            "bridge was not attached by build_executor, "
            "record_tool_embodiment_failure was not called, or NAc did "
            "not form a causal link."
        )
        assert prediction.predicted_valence == Valence.NEGATIVE


# ---------------------------------------------------------------------------
# permissions= — the gate the factory now arms (1.1.3)
# ---------------------------------------------------------------------------


class TestBuildExecutorPermissions:
    """``build_executor(permissions=)`` existed since C4 but no console
    caller passed it, so ``Executor._permissions`` was ``None`` for every
    ``MaximHandle`` agent. These pin that the parameter really arms the
    gate; ``test_agent_factory.py`` pins that the factory passes it."""

    def _registry(self):
        from maxim.tools.base import Tool, ToolOutput
        from maxim.tools.registry import ToolRegistry

        class _Stub(Tool):
            name = "stub_tool"
            description = "stub"
            input_schema: dict = {}

            def execute(self, **kwargs):
                return ToolOutput(success=True, output="ran")

        registry = ToolRegistry()
        registry.register(_Stub())
        return registry

    def test_permissions_none_leaves_gate_off(self):
        from maxim.runtime.bootstrap import build_executor

        executor = build_executor(self._registry(), pain_bus=None, permissions=None)
        assert executor._permissions is None
        assert executor.execute({"tool_name": "stub_tool", "params": {}}).success is True

    def test_permissions_arm_the_executor_gate(self):
        from maxim.agents.permissions import AgentPermissions
        from maxim.runtime.bootstrap import build_executor

        perms = AgentPermissions(tool_allow=frozenset({"respond"}))
        executor = build_executor(self._registry(), pain_bus=None, permissions=perms)
        assert executor._permissions is perms
        result = executor.execute({"tool_name": "stub_tool", "params": {}})
        assert result.success is False
        assert "allow-list" in (result.error or "")


class TestD79GenerationCollaborators:
    """D79 fix (b): the executor's generation-relevant collaborators are
    declared constructor fields, and BOTH generation sites (bootstrap
    initial + acquisition regeneration) run through the one seam,
    ``Executor.generate_entity_tools``. Pre-fix, ``build_executor`` took
    ``entity_map``/``cerebellum`` and threaded them to initial generation
    only — the Executor stashed neither, so Mechanism-B acquisition was a
    silent no-op through the canonical builder (the third
    takes-but-does-not-stash miss at this seam)."""

    def _build(self, **kw):
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        return build_executor(
            ToolRegistry(),
            pain_bus=PainBus(_allow_raw=True),
            permissions=None,
            nac=MagicMock(),
            entity_ref="weapons/rusty_sword",
            component_registry=ComponentRegistry(),
            **kw,
        )

    def test_build_executor_stashes_generation_collaborators(self):
        from maxim.embodiment.entity_map import EntityMap

        entity_map = EntityMap()
        cerebellum = MagicMock(name="cerebellum")
        executor = self._build(entity_map=entity_map, cerebellum=cerebellum)
        assert executor._entity_map is entity_map, "build_executor took entity_map but did not stash it (D79)"
        assert executor._cerebellum is cerebellum, "build_executor took cerebellum but did not stash it (D79)"

    def test_one_generation_seam_threads_every_collaborator(self, monkeypatch):
        """Both call sites go through generate_entity_tools, which passes
        ALL declared collaborators — a spy pins the kwargs so dropping one
        (the D77 shape) fails here, not silently in a sim."""
        import maxim.embodiment.tool_bridge as tool_bridge
        from maxim.runtime.executor import Executor
        from maxim.tools.registry import ToolRegistry

        calls = []

        def _spy(entity, registry, **kwargs):
            calls.append(kwargs)
            return {}

        monkeypatch.setattr(tool_bridge, "generate_tools_for_entity", _spy)
        embodiment = MagicMock(name="embodiment")
        cerebellum = MagicMock(name="cerebellum")
        entity_map = MagicMock(name="entity_map")
        executor = Executor(ToolRegistry(), embodiment=embodiment, cerebellum=cerebellum, entity_map=entity_map)
        executor.generate_entity_tools(MagicMock(name="entity"))
        assert calls, "generate_entity_tools must route through generate_tools_for_entity"
        kwargs = calls[0]
        assert kwargs["embodiment"] is embodiment
        assert kwargs["cerebellum"] is cerebellum
        assert kwargs["entity_map"] is entity_map

    def test_mechanism_b_acquisition_works_through_the_canonical_builder(self):
        """The behavioral pin: an acquirable entity registered in the
        entity_map handed to build_executor gets its tools registered on
        acquisition. Pre-fix this was a silent no-op (the executor's
        entity_map was permanently None), which made this exact assertion
        fail."""
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.embodiment.entity_map import EntityMap

        entity_map = EntityMap()
        bread = ComponentRegistry().instantiate("items/minecraft_bread")
        entity_map.register(bread)
        executor = self._build(entity_map=entity_map)

        executor._handle_entity_acquisition({"entity_acquired": "minecraft_bread"})
        assert executor.registry.get("minecraft_bread_eat_bread") is not None, (
            "acquisition through the canonical builder must register the item's tools (D79)"
        )

    def test_bootstrap_initial_generation_routes_through_the_seam(self, monkeypatch):
        """Review-round finding 2: without this, reverting bootstrap to a
        hand-threaded generate_tools_for_entity call (the pre-fix shape)
        leaves every other guard green."""
        from maxim.runtime.executor import Executor

        calls = []
        real = Executor.generate_entity_tools

        def _spy(self, entity):
            calls.append(entity)
            return real(self, entity)

        monkeypatch.setattr(Executor, "generate_entity_tools", _spy)
        executor = self._build()
        assert calls, "build_executor's initial generation must route through Executor.generate_entity_tools"
        assert "rusty_sword_slash" in set(executor.registry.list())


class TestInteractiveAttributionGate:
    """The executor's interactive gate must not fail open (#864, review round).

    This guards the PRIMARY contamination path. `build_pain_bus`'s own docstring records that
    tool-invoked pain reaches NAc through `ToolPainBridge` *regardless* of the bus subscriptions,
    so the three gates in `proprioception/pain_bus.py` cover only out-of-band pain and this one
    covers the rest. It used to be a bare `except Exception: pass` that left `_suppress_nac` False
    on any failure — the same fail-toward-contamination shape, on the bigger door, uninstrumented.
    """

    def _executor(self):
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        executor = build_executor(ToolRegistry(), pain_bus=None, permissions=None, pain_detector=None, nac=MagicMock())
        executor._tool_pain_bridge = MagicMock()
        return executor

    def test_a_broken_interactive_read_does_not_attribute_to_nac(self, monkeypatch):
        """Version-independent: patches the reader, not the fix's own shape, so it runs against
        the defective code too. With the old bare `except Exception: pass` this passed straight
        through and `record_tool_start` fired on a human-directed call."""
        import maxim.simulation.sim_logger as sim_logger

        def _boom():
            raise RuntimeError("cannot read interactive mode")

        monkeypatch.setattr(sim_logger, "get_interactive_mode", _boom)
        executor = self._executor()
        with pytest.raises(RuntimeError):
            executor.execute({"tool_name": "nope", "params": {}})
        assert not executor._tool_pain_bridge.mock_calls, (
            f"a broken gate attributed a tool call to NAc: {executor._tool_pain_bridge.mock_calls}"
        )

    def test_a_human_driven_tool_call_is_not_attributed(self, monkeypatch):
        import maxim.simulation.sim_logger as sim_logger

        monkeypatch.setattr(sim_logger, "get_interactive_mode", lambda: sim_logger.InteractiveMode.ON)
        executor = self._executor()
        executor.execute({"tool_name": "nope", "params": {}})
        # `record_tool_start` specifically: the gate suppresses ATTRIBUTION, not every interaction
        # with the bridge (an ungated `pop_invocation_rpe` follows on the same call).
        assert not executor._tool_pain_bridge.record_tool_start.called

    def test_an_agent_driven_tool_call_IS_attributed(self, monkeypatch):
        """The anti-vacuity arm: without it, a gate stuck permanently on would pass both above."""
        import maxim.simulation.sim_logger as sim_logger

        monkeypatch.setattr(sim_logger, "get_interactive_mode", lambda: sim_logger.InteractiveMode.OFF)
        executor = self._executor()
        executor.execute({"tool_name": "nope", "params": {}})
        assert executor._tool_pain_bridge.record_tool_start.called, "the gate suppressed agent-driven attribution"

    def test_the_gate_is_not_swallowed(self):
        import ast
        import inspect
        import textwrap

        from maxim.runtime.executor import Executor

        tree = ast.parse(textwrap.dedent(inspect.getsource(Executor.execute)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            branches = [*node.body, *node.orelse, *node.finalbody, *(s for h in node.handlers for s in h.body)]
            dumped = ast.dump(ast.Module(body=branches, type_ignores=[]))
            assert "get_interactive_mode" not in dumped, "the interactive gate is inside a try block again"
