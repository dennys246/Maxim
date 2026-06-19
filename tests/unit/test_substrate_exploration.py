"""Substrate exploration policy — regression guards.

Covers the novelty-bonus-before-gate selection added to
``NAc.recommend_action`` per docs/plans/substrate_exploration_policy.md
(1.1). The policy breaks the deterministic-argmax fixation that the
2026-06-17 spike quantified: a safe tool's causal-link confidence snowballs
while never-tried interaction affordances stay flat and are never selected.

Invariants pinned here (see the plan's "DO NOT BREAK" section):
  * OFF (``substrate_explore_bonus_weight == 0.0``) == byte-identical legacy.
  * The bonus lifts never-tried *available* tools into candidacy (zero-base
    tools must be able to enter ``scores``) — but never proposes a tool
    outside ``available_tools``, and the min_confidence gate still applies.
  * Exploration writes only ``_visit_count``; ``reward_bias`` is untouched.
  * Visit counts decay deterministically (no RNG); the lever is reproducible.
"""

from __future__ import annotations

from maxim.decisions.nac import NAc, NACConfig

TOOLS = ["safe_food", "hearth_warm_self", "hearth_touch"]


def _rec(nac: NAc, tools=TOOLS, *, drives=None, gate=0.3):
    return nac.recommend_action(
        agent_id="a",
        available_tools=tools,
        current_drives=drives,
        min_confidence=gate,
    )


# ── OFF == legacy ────────────────────────────────────────────────────────


def test_exploration_off_returns_none_when_no_scored_tools():
    """OFF + no learned/drive signal → every tool scores 0 → IDLE (None).

    This is the legacy "Random selection is explicitly NOT a fallback"
    contract: with nothing learned and no drive, the substrate has no
    opinion and proposes nothing.
    """
    nac = NAc(NACConfig())  # default weight 0.0
    assert _rec(nac) is None


def test_exploration_off_equals_legacy_selection():
    """OFF must be byte-identical to the pre-policy argmax.

    With a drive that scores ``safe_food`` above the gate and the two
    hearth tools at 0, OFF selects ``safe_food`` and never surfaces the
    zero-base hearth tools.
    """
    nac = NAc(NACConfig())
    drives = {"food": 1.0}  # drive-affinity name-match lifts safe_food only
    picks = {(_rec(nac, drives=drives) or {}).get("tool_name") for _ in range(10)}
    assert picks == {"safe_food"}


# ── ON lifts unvisited tools ─────────────────────────────────────────────


def test_exploration_lifts_unvisited_tool():
    """ON: the novelty bonus alone lifts zero-base tools over the gate.

    The whole point — the spike showed hearth_warm_self/touch have zero
    base score, so a "redistribute among already-scored tools" design
    would never surface them.
    """
    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.4))  # 0.4 > gate 0.3
    explored = {(_rec(nac) or {}).get("tool_name") for _ in range(15)}
    explored.discard(None)
    assert explored == set(TOOLS), f"exploration did not surface all tools: {explored}"


def test_exploration_only_proposes_available_tools():
    """Exploration never invents a tool outside available_tools."""
    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.5))
    for _ in range(20):
        r = _rec(nac)
        if r is not None:
            assert r["tool_name"] in TOOLS


def test_exploration_respects_min_confidence_gate():
    """A bonus weight below the gate, with no other signal, still IDLEs.

    Novelty alone cannot manufacture an action when it can't clear the
    confidence gate — exploration is "try viable candidates," not "act
    randomly when clueless."
    """
    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.1))  # 0.1 < gate 0.3
    assert _rec(nac, gate=0.3) is None


def test_visit_count_increments_on_selection():
    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.4))
    _rec(nac)
    total = sum(nac._visit_count.values())
    assert total == 1.0
    assert all(k[0] == "a" for k in nac._visit_count)


def test_exploration_spreads_across_tools_not_fixated():
    """ON breaks fixation: across N selections, >1 distinct tool is tried.

    Contrast the OFF/legacy behaviour where one tool would win the argmax
    every tick once it accrues any lead.
    """
    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.4))
    picks = [(_rec(nac) or {}).get("tool_name") for _ in range(15)]
    distinct = {p for p in picks if p}
    assert len(distinct) > 1


def test_explore_first_untried_beats_high_score_tried_tool():
    """The 2026-06-18 bug: an untried tool must be selected over a tried tool
    that has accrued a high base score (drive affinity / causal snowball).

    A score-term-only design fails this — per-tick visit decay resurrects the
    tried tool's novelty so it re-ties/out-scores untried tools. The hard
    explore-first gate fixes it: while any untried tool exists, it wins.
    """
    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.5))
    drives = {"food": 1.0}  # gives a food-named tool a large drive-affinity base
    # Round 1: only sense_food available → it gets tried + a high base score.
    for _ in range(3):
        nac.recommend_action(
            agent_id="a",
            available_tools=["sense_food"],
            current_drives=drives,
            min_confidence=0.3,
        )
    assert ("a", "sense_food") in nac._ever_selected
    # Round 2: introduce an untried zero-base tool. Despite sense_food's higher
    # raw score, explore-first must surface the untried tool.
    r = nac.recommend_action(
        agent_id="a",
        available_tools=["sense_food", "hearth_warm_self"],
        current_drives=drives,
        min_confidence=0.3,
    )
    assert r is not None and r["tool_name"] == "hearth_warm_self"


def test_exploration_first_visits_every_tool_before_repeat():
    """Explore-FIRST guarantee: every available tool is selected exactly once
    before any tool is selected a second time.

    This is the fix for the 2026-06-17 spike's leaky-novelty finding — bare
    decaying novelty re-explored high-alphabetical tools and starved the
    low-alphabetical interaction affordances Exp 41 must surface. The sticky
    ``_ever_selected`` set gives never-tried tools the full dominating bonus.
    """
    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.5))  # > gate 0.3
    first_n = [(_rec(nac) or {}).get("tool_name") for _ in range(len(TOOLS))]
    assert sorted(first_n) == sorted(TOOLS), f"not all tools tried first: {first_n}"
    assert len(set(first_n)) == len(TOOLS), "a tool repeated before all were tried"


# ── isolation: exploration must not corrupt learned state ────────────────


def test_exploration_does_not_mutate_reward_bias():
    """The novelty term is selection-local; it never writes reward_bias."""
    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.5))
    for _ in range(10):
        _rec(nac)
    assert nac._reward_bias == {}
    assert nac._cluster_reward_bias == {}


# ── decay ────────────────────────────────────────────────────────────────


def test_exploration_visit_decay():
    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.4, substrate_explore_decay_tau=50.0))
    _rec(nac)
    before = sum(nac._visit_count.values())
    assert before > 0
    nac.decay_exploration_visits()
    after = sum(nac._visit_count.values())
    assert after < before


def test_exploration_visit_decay_prunes_to_empty():
    """Repeated decay drives a single visit below the 0.05 prune floor."""
    nac = NAc(NACConfig(substrate_explore_bonus_weight=0.4, substrate_explore_decay_tau=2.0))
    _rec(nac)
    for _ in range(50):
        if not nac._visit_count:
            break
        nac.decay_exploration_visits()
    assert nac._visit_count == {}


def test_decay_exploration_visits_noop_when_off():
    """Off path never populates the map → decay is a cheap no-op."""
    nac = NAc(NACConfig())  # weight 0.0
    _rec(nac, drives={"food": 1.0})
    assert nac._visit_count == {}
    assert nac.decay_exploration_visits() == 0


# ── determinism (no RNG) ─────────────────────────────────────────────────


def test_exploration_is_deterministic():
    """Two fresh NAcs with the same weight produce identical pick sequences."""
    n1 = NAc(NACConfig(substrate_explore_bonus_weight=0.4))
    n2 = NAc(NACConfig(substrate_explore_bonus_weight=0.4))
    picks1 = [(_rec(n1) or {}).get("tool_name") for _ in range(10)]
    picks2 = [(_rec(n2) or {}).get("tool_name") for _ in range(10)]
    assert picks1 == picks2


# ── config plumbing (sim.substrate_explore_bonus_weight) ─────────────────


def test_config_field_registered_in_field_to_env():
    from maxim.runtime.config_loader import _FIELD_TO_ENV

    assert _FIELD_TO_ENV["sim.substrate_explore_bonus_weight"] == "MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT"


def test_config_default_is_off():
    from maxim.runtime.config_loader import SimConfigSection

    assert SimConfigSection().substrate_explore_bonus_weight == 0.0


def test_config_coercion_clamps_negative_to_zero():
    from maxim.runtime.config_loader import _coerce_for_field

    assert _coerce_for_field("-1.0", "sim.substrate_explore_bonus_weight") == 0.0
    assert _coerce_for_field("0.4", "sim.substrate_explore_bonus_weight") == 0.4
