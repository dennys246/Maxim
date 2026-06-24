"""Substrate-primary scene-harm wiring — regression guards (B4 + B5 + B6).

Covers docs/plans/substrate_primary_cradle_readiness.md:
  * B4: phase-activated scene affordances receive the AUT embodiment in
    substrate-primary so their self_effect writes to the agent's body —
    and DEFAULT to embodiment=None so LLM-AUT (Exp 37/38) stays byte-identical
    (scene harm there arrives via the narrator-driven Layer-2 proximity path).
  * B5: record_outcome treats a tool result carrying embodiment_failures as a
    NEGATIVE outcome, not POSITIVE-on-mechanical-success.
  * B6: propose_via_substrate filters read-only cognitive introspection tools
    out of the recommend_action candidate set so always-succeeding meta-tools
    can't snowball and starve the embodied affordances (the meta-tool fixation
    that VOID'd the Exp 42 triage). LLM-AUT is unaffected (it never calls this).
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


# ── B6: introspection-tool filtering in the substrate-primary candidate set ──


class _RecordingNac:
    """Stub NAc that captures the candidate set propose_via_substrate passes."""

    def __init__(self):
        self.seen_tools: list[str] | None = None

    def recommend_action(self, *, agent_id, available_tools, **kwargs):  # noqa: D401 - stub
        self.seen_tools = list(available_tools)
        return None


class _StubRegistry:
    def __init__(self, names):
        self._names = list(names)

    def list(self):  # noqa: D401 - stub
        return list(self._names)


class _StubExecutor:
    def __init__(self, names):
        self.registry = _StubRegistry(names)
        self.embodiment = None  # → _read_drive_states returns {}


def test_introspection_names_cover_the_registered_tools():
    """The filter set is derived from the tool classes, so it stays in sync."""
    from maxim.tools.introspection import (
        INTROSPECTION_TOOL_NAMES,
        SystemStatsTool,
        TemporalPatternsTool,
    )

    assert SystemStatsTool.name in INTROSPECTION_TOOL_NAMES
    assert TemporalPatternsTool.name in INTROSPECTION_TOOL_NAMES
    # the meta-tools that dominated the Exp 42 triage are all covered
    for n in (
        "temporal_patterns",
        "system_stats",
        "energy_status",
        "causal_links",
        "concept_query",
        "memory_recall",
        "similarity_search",
    ):
        assert n in INTROSPECTION_TOOL_NAMES


def test_propose_via_substrate_excludes_introspection_tools():
    """Embodied affordances reach recommend_action; introspection tools do not."""
    from maxim.runtime.agent_loop import propose_via_substrate
    from maxim.tools.introspection import INTROSPECTION_TOOL_NAMES

    names = [
        "temporal_patterns",
        "system_stats",
        "memory_recall",
        "warmth_beta_safe_warm_self",
        "warmth_alpha_harm_touch",
        "sense_presence",  # sensing is NOT introspection — kept
    ]
    nac = _RecordingNac()
    propose_via_substrate(nac=nac, agent_id="a", executor=_StubExecutor(names))
    assert nac.seen_tools is not None, "recommend_action was not reached"
    assert not (set(nac.seen_tools) & INTROSPECTION_TOOL_NAMES)
    assert "warmth_beta_safe_warm_self" in nac.seen_tools
    assert "warmth_alpha_harm_touch" in nac.seen_tools
    assert "sense_presence" in nac.seen_tools  # sensing preserved


def test_propose_via_substrate_idle_when_only_introspection():
    """If every candidate is an introspection tool, the substrate has no
    embodied action to propose → None (IDLE), not a meta-tool fidget."""
    from maxim.runtime.agent_loop import propose_via_substrate

    nac = _RecordingNac()
    out = propose_via_substrate(nac=nac, agent_id="a", executor=_StubExecutor(["system_stats", "temporal_patterns"]))
    assert out is None
    assert nac.seen_tools is None  # filtered to empty before recommend_action


# ── B7: drive-gating (motivated attention) in recommend_action ────────────


def _nac_with_snowballed(tool: str, *, n: int = 6, gate: bool = True, **cfg):
    """An NAc where ``tool`` has a strong positive causal link (snowballed).

    ``gate`` defaults True because the B7 tests exercise the gate; the SHIPPED
    NACConfig default is OFF (opt-in per experiment), so callers must enable it
    explicitly — which is what the experiment harness/bio_stack does."""
    from maxim.decisions.causal_link import Valence
    from maxim.decisions.nac import NAc, NACConfig

    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.0, drive_gate_enabled=gate, **cfg))
    for _ in range(n):
        nac.observe(
            event_type="tool",
            event_signature=f"tool:{tool}",
            outcome_type="tool_result",
            outcome_signature="success",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=0.0,
        )
    return nac


_TOOLS = ["sense_presence", "fire_pit_warm_self"]  # irrelevant (snowballed) vs cold-relevant


def test_drive_gate_forbids_irrelevant_tool_under_intense_drive():
    """An intense cold drive narrows selection to the cold-relevant tool even
    though the irrelevant tool has the higher learned (snowballed) score —
    the fix for the sense_presence fixation that VOID'd the triage."""
    nac = _nac_with_snowballed("sense_presence", gate=True)  # opt-in (experiment) gating
    rec = nac.recommend_action(agent_id="a", available_tools=_TOOLS, current_drives={"cold": 0.9}, min_confidence=0.3)
    assert rec is not None and rec["tool_name"] == "fire_pit_warm_self"


def test_drive_gate_disabled_is_the_shipped_default():
    """Gate OFF is the SHIPPED default → legacy argmax: the snowballed
    irrelevant tool wins. Proves the gate (not some other change) is what flips
    selection, and that enabling it is opt-in (global semantics unchanged)."""
    nac = _nac_with_snowballed("sense_presence", gate=False)
    rec = nac.recommend_action(agent_id="a", available_tools=_TOOLS, current_drives={"cold": 0.9}, min_confidence=0.3)
    assert rec is not None and rec["tool_name"] == "sense_presence"


def test_drive_gate_noop_when_drive_below_threshold():
    """Below the gate threshold the drive isn't intense enough to narrow
    attention → legacy argmax (the agent may sense when not cold)."""
    nac = _nac_with_snowballed("sense_presence", gate=True)
    rec = nac.recommend_action(agent_id="a", available_tools=_TOOLS, current_drives={"cold": 0.3}, min_confidence=0.3)
    assert rec is not None and rec["tool_name"] == "sense_presence"


def test_drive_gate_does_not_override_explore_first():
    """Explore-first still guarantees a forced trial of every tool before
    gating engages — so the harmful affordance the discrimination needs gets
    surfaced. With exploration on and the relevant tool already tried but the
    irrelevant one untried, explore-first picks the untried one despite the
    intense drive."""
    from maxim.decisions.nac import NAc, NACConfig

    nac = NAc(NACConfig(substrate_explore_bonus_weight=1.5, drive_gate_enabled=True))
    # mark fire (relevant) as already tried; sense_presence still untried
    nac._ever_selected.add(("a", "fire_pit_warm_self"))
    rec = nac.recommend_action(agent_id="a", available_tools=_TOOLS, current_drives={"cold": 0.9}, min_confidence=0.3)
    assert rec is not None and rec["tool_name"] == "sense_presence"  # explore-first wins


# ── B8: delta-attribution (bystander affordances not blamed for lingering harm) ──


def test_drive_failure_sensor_parsing():
    from maxim.embodiment.tool_bridge import _drive_failure_sensor

    assert _drive_failure_sensor("drive:arms.thermal:discomfort") == "arms.thermal"
    assert _drive_failure_sensor("drive:cold:deprived") == "cold"
    assert _drive_failure_sensor("laceration") is None  # standard failure_mode → unfiltered
    assert _drive_failure_sensor("drive:x") is None


def test_intrinsic_harm_only_blames_band_exceeding_delta():
    """The harmful warm (+0.6 > band 0.5) is intrinsically harmful; the safe
    warm (+0.05) is not — so a safe affordance executing while the harmful
    source's breach lingers is never blamed (the Exp 42 pollution fix)."""
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.embodiment.tool_bridge import _intrinsically_harmful_sensors

    body = ComponentRegistry().instantiate("bodies/infant_humanoid_chilled")
    # arms.thermal is a homeostatic sub-sensor with comfort_band 0.5
    assert "arms.thermal" in _intrinsically_harmful_sensors(body, {"cold": -0.3, "arms.thermal": 0.6})
    assert "arms.thermal" not in _intrinsically_harmful_sensors(body, {"cold": -0.3, "arms.thermal": 0.05})
    # relief-direction cold delta never counts as harmful
    assert "cold" not in _intrinsically_harmful_sensors(body, {"cold": -0.3})


def test_execute_delta_attribution_causing_vs_bystander_on_chilled_body():
    """End-to-end through ModulatorAffordanceTool.execute on the qualified
    `arms.thermal` path: the harmful warm (+0.6) reports its OWN breach; a safe
    warm (+0.05) executing while that breach LINGERS is NOT blamed. This is the
    load-bearing claim B8 exists to protect, on the real Exp 42 fixtures."""
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.embodiment.tool_bridge import generate_tools_for_entity

    reg = ComponentRegistry()
    body = Embodiment(root=reg.instantiate("bodies/infant_humanoid_chilled"))
    harm = reg.instantiate("items/warmth_alpha_harm")
    safe = reg.instantiate("items/warmth_alpha_safe")
    treg = ToolRegistry()
    htools = {t.name: t for t in generate_tools_for_entity(harm, treg, embodiment=body)}
    stools = {t.name: t for t in generate_tools_for_entity(safe, treg, embodiment=body)}

    def thermal_failures(out):
        return [f for f in (out.side_effects or {}).get("embodiment_failures", []) if "arms.thermal" in f["name"]]

    # Causing tool: +0.6 breaches arms.thermal → reports its own drive failure.
    out_harm = htools["warmth_alpha_harm_warm_self"].execute()
    assert thermal_failures(out_harm), "harmful warm must report its own arms.thermal breach (causing tool)"

    # arms.thermal now lingers breached (default slow drift). The bystander safe
    # warm (+0.05, not intrinsically harmful) must NOT inherit the blame.
    out_safe = stools["warmth_alpha_safe_warm_self"].execute()
    assert not thermal_failures(out_safe), "safe warm must NOT be blamed for the lingering breach (delta-attribution)"


def test_introspection_tool_classes_all_in_filter_set():
    """Every read-only cognitive Tool defined in tools/introspection.py is a
    member of INTROSPECTION_TOOL_NAMES — so adding a new introspection tool that
    forgets to join the set is caught here (the set is hand-curated by class)."""
    import inspect

    from maxim.tools import introspection as intro
    from maxim.tools.base import Tool

    missing = []
    for _name, obj in inspect.getmembers(intro, inspect.isclass):
        if issubclass(obj, Tool) and obj is not Tool and obj.__module__ == intro.__name__:
            tool_name = getattr(obj, "name", None)
            if tool_name and tool_name not in intro.INTROSPECTION_TOOL_NAMES:
                missing.append(tool_name)
    assert not missing, f"introspection Tool(s) missing from INTROSPECTION_TOOL_NAMES: {missing}"
