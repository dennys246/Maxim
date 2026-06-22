"""Substrate-primary scene-harm wiring — regression guards (B4 + B5).

Covers docs/plans/substrate_primary_cradle_readiness.md:
  * B4: phase-activated scene affordances receive the AUT embodiment in
    substrate-primary so their self_effect writes to the agent's body —
    and DEFAULT to embodiment=None so LLM-AUT (Exp 37/38) stays byte-identical
    (scene harm there arrives via the narrator-driven Layer-2 proximity path).
  * B5: record_outcome treats a tool result carrying embodiment_failures as a
    NEGATIVE outcome, not POSITIVE-on-mechanical-success.
"""

from __future__ import annotations

from maxim.simulation.generative_runner import _activate_phase_entities
from maxim.tools.registry import ToolRegistry

_HEARTH_REF = "items/cradle_false_hearth"


# ── B4: embodiment threading + LLM-AUT default safety ────────────────────


def test_phase_activation_default_embodiment_none_preserves_llm_aut():
    """Default (no embodiment kwarg) → scene affordance tools keep
    ``_embodiment is None``. This is the LLM-AUT path; passing embodiment
    there would double-count with the Layer-2 proximity harm and change
    Exp 37/38. Pins byte-identical LLM-AUT behavior."""
    reg = ToolRegistry()
    n = _activate_phase_entities((_HEARTH_REF,), reg, set())
    assert n > 0
    warm = reg.get("hearth_warm_self")
    assert warm._embodiment is None


def test_phase_activation_threads_embodiment_for_substrate_primary():
    """When an embodiment is passed (substrate-primary), scene affordance
    tools receive it, so their self_effect writes to the agent's body."""
    reg = ToolRegistry()
    sentinel_embodiment = object()
    sentinel_entity_map = _RecordingEntityMap()
    _activate_phase_entities(
        (_HEARTH_REF,),
        reg,
        set(),
        embodiment=sentinel_embodiment,
        entity_map=sentinel_entity_map,
    )
    warm = reg.get("hearth_warm_self")
    assert warm._embodiment is sentinel_embodiment
    # entity_map is also threaded (used by sense/resolution)
    assert sentinel_entity_map.registered, "entity_map.register was not called"


class _RecordingEntityMap:
    """Minimal EntityMap stand-in that records register() calls."""

    def __init__(self):
        self.registered = False

    def register(self, entity):  # noqa: D401 - stub
        self.registered = True


# ── B5: embodiment_failed → negative learning outcome ────────────────────


class _CtxStub:
    def add_outcome(self, **kwargs):  # noqa: D401 - stub
        pass


def _record(nac, *, success, embodiment_failed):
    from maxim.runtime.tool_dispatch import record_outcome

    record_outcome(
        agent_id="a",
        tool_name="hearth_warm_self",
        success=success,
        result_summary="warmed",
        error=None,
        reasoning="",
        recent_outcomes=[],
        max_recent=10,
        llm_worker=None,
        context_pool=_CtxStub(),
        nac=nac,
        embodiment_failed=embodiment_failed,
    )


def test_record_outcome_embodiment_failed_books_negative_not_positive():
    """A mechanically-successful action that harmed the body must book a
    NEGATIVE causal link, not a positive one (B5)."""
    from maxim.decisions.nac import NAc, NACConfig

    nac = NAc(NACConfig())
    _record(nac, success=True, embodiment_failed=True)
    sig = "tool:hearth_warm_self"
    assert nac.get_negative_outcomes(sig), "expected a negative link for harmful success"
    assert not nac.get_positive_outcomes(sig), "must NOT book a positive link when harmed"


def test_record_outcome_clean_success_still_positive():
    """Regression: a clean success (no embodiment failure) is still POSITIVE."""
    from maxim.decisions.nac import NAc, NACConfig

    nac = NAc(NACConfig())
    _record(nac, success=True, embodiment_failed=False)
    sig = "tool:hearth_warm_self"
    assert nac.get_positive_outcomes(sig)
    assert not nac.get_negative_outcomes(sig)


def test_repeated_embodiment_failure_never_books_positive():
    """Two embodiment-failed outcomes (e.g. two harmful trials) still produce
    only negative links — no spurious positive ever sneaks in (B5)."""
    from maxim.decisions.nac import NAc, NACConfig

    nac = NAc(NACConfig())
    _record(nac, success=True, embodiment_failed=True)
    _record(nac, success=True, embodiment_failed=True)
    sig = "tool:hearth_warm_self"
    assert nac.get_negative_outcomes(sig)
    assert not nac.get_positive_outcomes(sig)


def test_negative_links_use_max_not_sum_in_scoring():
    """B5's no-double-penalty correctness is load-bearing on recommend_action
    subtracting the MAX over negative links (nac.py), NOT the sum — that's why
    the executor-bridge negative + record_outcome negative (same event sig)
    don't compound. Pin it: 3 negative links + a cold-affinity boost (0.7).
    MAX → score ≈ 0.7 − 0.5·0.6 = 0.4 (selectable); a SUM would drive it well
    below the gate (IDLE)."""
    from maxim.decisions.causal_link import Valence
    from maxim.decisions.nac import NAc, NACConfig

    nac = NAc(NACConfig())
    for outcome in ("failure:a", "failure:b", "failure:c"):
        nac.observe(
            event_type="tool",
            event_signature="tool:hearth_warm_self",
            outcome_type="tool_result",
            outcome_signature=outcome,
            outcome_valence=Valence.NEGATIVE,
            delta_seconds=0.0,
        )
    assert len(nac.get_negative_outcomes("tool:hearth_warm_self")) == 3
    rec = nac.recommend_action(
        agent_id="a",
        available_tools=["hearth_warm_self"],
        current_drives={"cold": 1.0},
        min_confidence=0.3,
    )
    assert rec is not None and rec["tool_name"] == "hearth_warm_self"
