#!/usr/bin/env python3
"""Valence annotation PoC — demonstrates SEM pain → Hebbian edge valence.

Scenario
--------
1. An agent perceives several concepts ("rusty sword", "heavy", "sharp",
   "sunny day", "warm breeze") across two episodes.
2. During the rusty-sword episode, a pain reaction fires (simulating a
   CerebellumModulator failure on a "swing" affordance).
3. Episode 2 (sunny-day) has no reactions (neutral valence).
4. After both episodes close, we measure:
   (a) edges from the sword episode carry negative valence
   (b) edges from the day episode have NO valence metadata
   (c) spreading_activation from "rusty_sword" cue propagates negative valence
   (d) a clean hippocampus (no pain history) shows zero valence

This script uses no LLM calls and is fully deterministic.

Usage
-----
    python scripts/valence_annotation_poc.py

Expected output: all assertions pass, summary printed.
"""

from __future__ import annotations

import sys
import time

# Ensure src/ is importable when run from repo root
sys.path.insert(0, "src")

from maxim.agents.bus import EdgeType
from maxim.decisions.causal_link import Valence
from maxim.memory.episode import CaptureEvent
from maxim.memory.hippocampus import (
    EpisodeConfig,
    HebbianConfig,
    Hippocampus,
    HippocampusConfig,
)
from maxim.reactions.types import Reaction, ReactionContext


def make_hippocampus() -> Hippocampus:
    return Hippocampus(
        HippocampusConfig(
            episode=EpisodeConfig(
                boundary_tick_gap=50,
                hebbian=HebbianConfig(init=0.3, delta=0.1, max_weight=1.0),
            )
        )
    )


def make_pain_reaction(source: str, intensity: float = 0.8) -> Reaction:
    return Reaction(
        kind="pain",
        intensity=intensity,
        valence=Valence.NEGATIVE,
        timestamp=time.time(),
        context=ReactionContext(),
        source=source,
    )


def run_poc() -> dict:
    """Run the PoC and return measurements."""
    results = {}

    # -- Agent with pain experience --
    h = make_hippocampus()

    # Episode 1: rusty sword encounter (tick 0-10)
    # Concepts: rusty_sword, heavy, sharp
    sword_nodes = ("text_rusty_sword", "text_heavy", "text_sharp")
    h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=sword_nodes))

    # Pain fires during this episode: swing affordance fails
    h.capture_reaction(
        make_pain_reaction(
            source="cerebellum:rusty_sword.motor.swing",
            intensity=0.8,
        )
    )
    ep1 = h.finalize_pending_episode()
    assert ep1 is not None
    results["episode_1_valence"] = ep1.valence

    # Episode 2: sunny day (tick 100-110), no reactions
    day_nodes = ("text_sunny_day", "text_warm_breeze")
    h.observe_episode_event(CaptureEvent(tick=100, channel="text", activated_nodes=day_nodes))
    ep2 = h.finalize_pending_episode()
    assert ep2 is not None
    results["episode_2_valence"] = ep2.valence

    # -- Measurement (a): sword edges have negative valence --
    sword_edge = h._binding_graph.find_edge("text_rusty_sword", "text_heavy", EdgeType.ASSOCIATES)
    assert sword_edge is not None, "Sword→heavy edge missing"
    results["sword_heavy_edge_valence"] = sword_edge.metadata.get("valence", 0.0)
    assert sword_edge.metadata.get("valence", 0.0) < 0.0, (
        f"Expected negative valence on sword edge, got {sword_edge.metadata}"
    )

    sword_sharp = h._binding_graph.find_edge("text_rusty_sword", "text_sharp", EdgeType.ASSOCIATES)
    results["sword_sharp_edge_valence"] = sword_sharp.metadata.get("valence", 0.0)

    # -- Measurement (b): day edges have NO valence metadata --
    day_edge = h._binding_graph.find_edge("text_sunny_day", "text_warm_breeze", EdgeType.ASSOCIATES)
    assert day_edge is not None, "Day→breeze edge missing"
    results["day_edge_has_valence"] = "valence" in day_edge.metadata
    assert "valence" not in day_edge.metadata, f"Day edge should have no valence, got {day_edge.metadata}"

    # -- Measurement (c): spreading activation propagates negative valence --
    activated = h.retrieve_on_cue("text_rusty_sword", include_valence=True)
    results["sword_cue_results"] = [
        {"node": n, "activation": round(a, 4), "valence": round(v, 4)} for n, a, v in activated
    ]
    # All co-activated nodes from the sword episode should carry negative valence
    for node, activation, valence in activated:
        if node in ("text_heavy", "text_sharp"):
            assert valence < 0.0, f"Expected negative valence for {node}, got {valence}"
    results["sword_cue_all_negative"] = all(v < 0.0 for n, a, v in activated if n in ("text_heavy", "text_sharp"))

    # Sunny day cue should have zero valence (no pain in that episode)
    day_results = h.retrieve_on_cue("text_sunny_day", include_valence=True)
    results["day_cue_results"] = [
        {"node": n, "activation": round(a, 4), "valence": round(v, 4)} for n, a, v in day_results
    ]
    for node, activation, valence in day_results:
        assert valence == 0.0, f"Expected zero valence for day cue node {node}, got {valence}"
    results["day_cue_all_zero_valence"] = True

    # -- Measurement (d): clean agent shows zero valence --
    h_clean = make_hippocampus()
    h_clean.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=sword_nodes))
    # No pain reaction
    h_clean.finalize_pending_episode()
    clean_results = h_clean.retrieve_on_cue("text_rusty_sword", include_valence=True)
    results["clean_agent_results"] = [
        {"node": n, "activation": round(a, 4), "valence": round(v, 4)} for n, a, v in clean_results
    ]
    for node, activation, valence in clean_results:
        assert valence == 0.0, f"Clean agent: expected zero valence for {node}, got {valence}"
    results["clean_agent_all_zero_valence"] = True

    # -- Persistence round-trip --
    state = h.dump()
    h_restored = make_hippocampus()
    h_restored.load_state(state)

    restored_edge = h_restored._binding_graph.find_edge("text_rusty_sword", "text_heavy", EdgeType.ASSOCIATES)
    results["persistence_valence_preserved"] = (
        restored_edge is not None and restored_edge.metadata.get("valence", 0.0) == results["sword_heavy_edge_valence"]
    )

    return results


def main() -> None:
    print("=" * 60)
    print("Valence Annotation PoC — SEM Pain → Hebbian Edge Valence")
    print("=" * 60)
    print()

    results = run_poc()

    print("Episode 1 (sword + pain):")
    print(f"  Net valence: {results['episode_1_valence']:.3f}")
    print(f"  sword→heavy edge valence: {results['sword_heavy_edge_valence']:.3f}")
    print(f"  sword→sharp edge valence: {results['sword_sharp_edge_valence']:.3f}")
    print()

    print("Episode 2 (sunny day, no reactions):")
    print(f"  Net valence: {results['episode_2_valence']:.3f}")
    print(f"  Day edge has valence metadata: {results['day_edge_has_valence']}")
    print()

    print("Spreading activation from 'rusty_sword' cue:")
    for r in results["sword_cue_results"]:
        print(f"  {r['node']}: activation={r['activation']}, valence={r['valence']}")
    print(f"  All sword-episode nodes negative: {results['sword_cue_all_negative']}")
    print()

    print("Spreading activation from 'sunny_day' cue:")
    for r in results["day_cue_results"]:
        print(f"  {r['node']}: activation={r['activation']}, valence={r['valence']}")
    print(f"  All day-episode nodes zero valence: {results['day_cue_all_zero_valence']}")
    print()

    print("Clean agent (same sword, no pain):")
    for r in results["clean_agent_results"]:
        print(f"  {r['node']}: activation={r['activation']}, valence={r['valence']}")
    print(f"  All zero valence: {results['clean_agent_all_zero_valence']}")
    print()

    print(f"Persistence round-trip: valence preserved = {results['persistence_valence_preserved']}")
    print()

    print("=" * 60)
    print("ALL ASSERTIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
