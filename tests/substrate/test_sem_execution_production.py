"""SEM execution end-to-end production test (Stage 3 of sem_execution_hook).

This is the production-path counterpart to ``test_sem_pain_cascade.py``.
The PoC file uses a hand-rolled ``PoCAgent`` that records actions
directly into NAc, bypassing the ``runtime/executor.py`` boundary. That
was correct for proving the cascade *mechanism* in P2 Stage 2, but it
left the production path (LLM → executor.execute → ModulatorAffordanceTool
→ ToolPainBridge.record_tool_embodiment_failure → NAc) uncovered.

This file covers it. **No PoC harness — every test drives the cascade
through the real ``build_executor`` constructor, the real
``Executor.execute({"tool_name": ...})`` dispatch, the real
``ModulatorAffordanceTool.execute`` forward path, and the real Stage 1
direct-attribution wiring on ``record_tool_embodiment_failure``.**

If a regression breaks any of those layers — bridge construction,
executor routing, ``ToolOutput.side_effects`` plumbing, NAc direct
attribution — at least one assertion in this file should fail loudly
and name the broken layer.

Read alongside:
- ``docs/plans/archive/sem_execution_hook.md`` Stage 3 + Stage 4 sections
- ``docs/plans/archive/executor_bootstrap_unification.md`` for the build_executor
  signature contract this test relies on
- ``tests/substrate/test_sem_pain_cascade.py`` for the PoC mechanism
  tests (still useful as smaller-surface regression guards; this file
  is the production-path superset)
"""

from __future__ import annotations

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.proprioception.pain_bus import PainBus
from maxim.runtime.bootstrap import build_executor
from maxim.tools.registry import ToolRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Shared world setup (production-path equivalent of _build_world)
# ─────────────────────────────────────────────────────────────────────────────


def _build_production_world(*, sword_durability: float = 1.0):
    """Construct a real production agent stack via build_executor.

    Returns a dict mirroring _build_world() in test_sem_pain_cascade
    but with the production constructor in place of the PoC harness.
    """
    pain_bus = PainBus(_allow_raw=True)
    nac = NAc(NACConfig(temporal_window_seconds=60.0))
    component_registry = ComponentRegistry()

    executor = build_executor(
        ToolRegistry(),
        pain_bus=pain_bus,
        permissions=None,
        nac=nac,
        entity_ref="weapons/rusty_sword",
        component_registry=component_registry,
    )

    embodiment = executor.embodiment
    assert embodiment is not None, (
        "build_executor with entity_ref must populate executor.embodiment "
        "(declared field, NOT an attribute stash). If this fails, the "
        "executor_bootstrap_unification I1 fold has regressed."
    )
    sword = embodiment.root
    sword.vital_metrics["durability"] = sword_durability

    return {
        "executor": executor,
        "embodiment": embodiment,
        "sword": sword,
        "nac": nac,
        "pain_bus": pain_bus,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 core: end-to-end cascade through executor.execute
# ─────────────────────────────────────────────────────────────────────────────


class TestSEMProductionCascade:
    """End-to-end SEM execution tests against the real rusty_sword YAML
    via the production build_executor → executor.execute path.

    Every layer that can break the cascade is asserted explicitly so
    a regression names its own broken layer in the failure message:

    1. ``build_executor`` constructs the bridge       (assert in helper)
    2. ``executor.embodiment`` is set                 (assert in helper)
    3. ``executor.execute`` returns success           (test 1)
    4. ``ToolOutput.side_effects`` carries failures   (test 1)
    5. ``record_tool_embodiment_failure`` is invoked  (test 1, NAc state)
    6. NAc direct attribution forms the link         (test 2)
    7. ``nac.predict`` returns NEGATIVE              (test 2)
    8. Repeated cascades strengthen confidence       (test 3)
    9. Per-entity isolation holds                    (test 4)
    10. Policy preference flips after learning      (test 5)
    """

    def test_rusty_sword_slash_at_zero_durability_produces_side_effects(self):
        """Layer 3-4 guard: the production tool returns a successful
        ToolOutput with embodiment_failures populated. If this fails,
        the regression is in ``ModulatorAffordanceTool.execute``,
        ``embodiment.evaluate_failures()``, or the side_effects
        plumbing in ``runtime/executor.py``."""
        world = _build_production_world(sword_durability=0.0)
        executor = world["executor"]

        result = executor.execute(
            {
                "tool_name": "rusty_sword_slash",
                "params": {"target": "training_dummy", "force": 0.9},
            }
        )

        assert result.success is True, (
            "executor.execute returned success=False for a healthy tool "
            f"invocation: {result}. Tool itself must succeed; failure "
            "lives in side_effects per Stage 1 contract."
        )
        assert result.side_effects is not None, (
            "ModulatorAffordanceTool.execute did not populate "
            "ToolOutput.side_effects. Stage 1 invariant broken — see "
            "src/maxim/embodiment/tool_bridge.py::ModulatorAffordanceTool.execute"
        )
        failures = result.side_effects.get("embodiment_failures")
        assert failures, (
            f"side_effects['embodiment_failures'] is empty after slashing "
            f"a sword at durability=0.0: side_effects={result.side_effects}. "
            "Either evaluate_failures() did not fire, or shatter's "
            "trigger threshold has changed in weapons/rusty_sword.yaml."
        )

    def test_nac_records_negative_after_one_cascade(self):
        """Layer 5-7 guard: a single cascade produces a NEGATIVE
        prediction. If this fails, the regression is in
        ``ToolPainBridge.record_tool_embodiment_failure``, NAc's direct-
        attribution path, or the executor's side_effects routing."""
        world = _build_production_world(sword_durability=0.0)
        executor = world["executor"]
        nac = world["nac"]

        # Sanity: no prediction before the cascade runs
        assert nac.predict(event_type="tool", event_signature="tool:rusty_sword_slash") is None

        executor.execute(
            {
                "tool_name": "rusty_sword_slash",
                "params": {"target": "training_dummy", "force": 0.9},
            }
        )

        prediction = nac.predict(
            event_type="tool",
            event_signature="tool:rusty_sword_slash",
        )
        assert prediction is not None, (
            "NAc has no prediction for tool:rusty_sword_slash after a "
            "deterministic shatter. The cascade is broken at one of: "
            "(a) ToolPainBridge attached by build_executor, (b) executor "
            "routing on side_effects['embodiment_failures'], (c) "
            "record_tool_embodiment_failure called with non-empty "
            "failures list, (d) NAc.record_outcome forms the link. "
            "Check src/maxim/runtime/executor.py and "
            "src/maxim/bridges/tool_pain_bridge.py."
        )
        assert prediction.predicted_valence == Valence.NEGATIVE, (
            f"Expected NEGATIVE prediction, got {prediction.predicted_valence}. "
            "Either the bridge is reporting POSITIVE on a failure "
            "(record_tool_embodiment_failure valence bug) or the test "
            "is asking the wrong NAc instance."
        )

    def test_repeated_cascades_strengthen_negative_confidence(self):
        """Layer 8 guard: multiple pain events grow link confidence.

        Confidence grows as ``sqrt(observation_count)`` per
        ``CausalLink.record_observation``. Across 3 cascades we expect
        strict monotonic growth. If confidences flat-line, either the
        link is being recreated (not updated) on each cascade or the
        bridge is no-op'ing on follow-up failures."""
        world = _build_production_world(sword_durability=0.0)
        executor = world["executor"]
        nac = world["nac"]
        sword = world["sword"]

        confidences: list[float] = []
        for iteration in range(3):
            # Reset durability + clear the shatter's active state so it
            # can fire again. Mirrors the PoC test pattern.
            sword.vital_metrics["durability"] = 0.0
            for failure_mode in sword.failure_modes:
                failure_mode.active = False

            executor.execute(
                {
                    "tool_name": "rusty_sword_slash",
                    "params": {"target": "training_dummy", "force": 0.9},
                }
            )
            prediction = nac.predict(
                event_type="tool",
                event_signature="tool:rusty_sword_slash",
            )
            assert prediction is not None, (
                f"iteration {iteration}: prediction missing — cascade stopped firing on follow-up failures"
            )
            confidences.append(prediction.confidence)

        assert confidences[-1] > confidences[0], (
            f"Confidence did not grow across 3 cascades: {confidences}. "
            "Either CausalLink.record_observation is no-op'ing on "
            "duplicates or the link is being recreated each time."
        )

    def test_per_entity_isolation_through_production_path(self):
        """Layer 9 guard: learning on sword A does not leak into
        the prediction surface of sword B. Mirrors the PoC's
        per-entity isolation test through the production constructor."""
        # World A — slashes a doomed sword, learns NEGATIVE
        world_a = _build_production_world(sword_durability=0.0)
        world_a["executor"].execute(
            {
                "tool_name": "rusty_sword_slash",
                "params": {"target": "training_dummy", "force": 0.9},
            }
        )
        prediction_a = world_a["nac"].predict(
            event_type="tool",
            event_signature="tool:rusty_sword_slash",
        )
        assert prediction_a is not None
        assert prediction_a.predicted_valence == Valence.NEGATIVE

        # World B — independent NAc, never slashed anything
        world_b = _build_production_world(sword_durability=1.0)
        prediction_b = world_b["nac"].predict(
            event_type="tool",
            event_signature="tool:rusty_sword_slash",
        )
        assert prediction_b is None, (
            "World B's NAc has a prediction it never earned — "
            "cross-world contamination. Each build_executor call should "
            "produce an isolated bridge attached to that call's NAc."
        )

    # NOTE: The original Stage 3 file shipped a
    # `test_policy_story_drop_weapon_after_learning_slash_is_painful`
    # test, but the pre-merge review (I6 cross-confirmed) caught it
    # asserting on a trivially-true premise: `rusty_sword_repair` is
    # already a registered affordance, so `nac.predict(repair) is
    # None` is true for ANY untried tool name and proves nothing
    # about cross-tool contamination. The load-bearing NEGATIVE
    # signal is already covered by `test_nac_records_negative_after_one_cascade`
    # and the confidence-growth test. A real cross-tool policy story
    # would need to record a POSITIVE outcome via nac.record_outcome
    # AND a NEGATIVE one and assert the relative ordering; that's a
    # different test (and more work) and not load-bearing for the
    # Stage 3 cascade integrity claim. Trivially-green tests pollute
    # bisects, so it was deleted entirely rather than left as a
    # weakened guard.

    def test_executor_embodiment_field_is_declared_not_stash(self):
        """Layer 2 explicit guard: ``executor.embodiment`` is a real
        declared field on the Executor class, not an attribute stash.
        This is the executor_bootstrap_unification I1 cross-confirmed
        review finding — verify the fix has not regressed."""
        import inspect

        from maxim.runtime.executor import Executor

        # The field must be declared in __init__ so it survives even
        # when no entity_ref is passed.
        executor = build_executor(ToolRegistry(), pain_bus=None, permissions=None)
        assert hasattr(executor, "embodiment"), (
            "Executor missing the declared `embodiment` field. The "
            "executor_bootstrap_unification I1 fold made this a real "
            "constructor parameter, not an attribute stash. If this "
            "assertion fails, someone has regressed Executor.__init__."
        )
        assert executor.embodiment is None, (
            f"executor.embodiment should default to None when entity_ref was not passed; got {executor.embodiment!r}"
        )
        # Sanity: it's a real PARAMETER on Executor.__init__, not just
        # a local variable inside the function body. The pre-merge
        # review (I4 cross-confirmed) caught that `co_varnames`
        # includes locals, so a future refactor to **kwargs +
        # `kwargs.get("embodiment")` would have kept the old assertion
        # green even though the parameter was gone. Use
        # inspect.signature to check the actual parameter surface.
        params = inspect.signature(Executor.__init__).parameters
        assert "embodiment" in params, (
            "Executor.__init__ no longer has `embodiment` as a declared "
            "parameter. The declared-field invariant is broken — the "
            "constructor must accept embodiment as a keyword argument, "
            "not stash it post-construction via setattr."
        )
