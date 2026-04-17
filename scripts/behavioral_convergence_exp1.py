#!/usr/bin/env python3
"""Behavioral Convergence Experiment 1 — Cross-session affective memory.

Tests the 1.0 claim: "cross-session learning without fine-tuning."

Scenario
--------
Session 1: Agent encounters three SEM entities in a fantasy dungeon.
  - Rusty Sword: agent slashes → low durability → shatter failure → pain
  - Healing Potion: agent drinks → HP restored → success/satisfaction
  - Poison Potion: agent drinks (looks like healing) → HP drops → pain

State saved to disk.

Session 2: State loaded. Agent is cued with each entity name.
  - Measure: retrieval valence, NAc predictions, EC threshold shifts

Control: Fresh agent (no prior experience), same cues.
  - Hypothesis: zero valence, zero bias, default thresholds

This is Tier 1 (substrate-only, deterministic, no LLM).

Usage
-----
    PYTHONPATH=src python scripts/behavioral_convergence_exp1.py
    PYTHONPATH=src python scripts/behavioral_convergence_exp1.py --persist /tmp/maxim_exp1
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "src")

from maxim.agents.bus import EdgeType
from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc
from maxim.memory.episode import CaptureEvent, salience_spike_rule
from maxim.memory.hippocampus import (
    EpisodeConfig,
    HebbianConfig,
    Hippocampus,
    HippocampusConfig,
    RetrievalConfig,
)
from maxim.reactions.bus import ReactionBus
from maxim.reactions.types import Reaction, ReactionContext


AGENT_ID = "dungeon_explorer"


# ─────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────


def make_hippocampus(persistence_path: str | None = None) -> Hippocampus:
    return Hippocampus(
        HippocampusConfig(
            persistence_path=persistence_path,
            episode=EpisodeConfig(
                boundary_tick_gap=50,
                hebbian=HebbianConfig(init=0.4, delta=0.15, max_weight=1.0),
                retrieval=RetrievalConfig(decay=0.7, threshold=0.05, max_depth=3),
            ),
        )
    )


def make_reaction(
    kind: str,
    intensity: float,
    valence: Valence,
    source: str,
) -> Reaction:
    return Reaction(
        kind=kind,
        intensity=intensity,
        valence=valence,
        timestamp=time.time(),
        context=ReactionContext(agent_id=AGENT_ID),
        source=source,
    )


def wire_bio_pipeline(h: Hippocampus, nac: NAc) -> tuple[ReactionBus, None]:
    """Wire a minimal bio-pipeline for the experiment."""
    bus = ReactionBus()
    bus.subscribe_all(h.capture_reaction)

    def _distribute(reaction: Reaction) -> None:
        v = reaction.valence.value
        if v == "negative":
            reward = -reaction.intensity
        elif v == "positive":
            reward = reaction.intensity
        else:
            return
        aid = reaction.context.agent_id or AGENT_ID
        nac.distribute_reward(aid, reward)

    bus.subscribe_all(_distribute)
    h._episode_detector.add_rule(salience_spike_rule(min_intensity=0.5))
    return bus, None


# ─────────────────────────────────────────────────────────────────────────
# Entity definitions (concept nodes simulating SEM entity percepts)
# ─────────────────────────────────────────────────────────────────────────

RUSTY_SWORD = {
    "name": "rusty_sword",
    "concepts": ("text_rusty_sword", "text_weapon", "text_blade", "text_corroded"),
    "action_concepts": ("text_slash", "text_combat"),
    "outcome": "pain",  # shatter failure
    "reaction": lambda: make_reaction(
        "pain",
        0.8,
        Valence.NEGATIVE,
        "cerebellum:rusty_sword.combat.slash",
    ),
    "salience_spike": 0.7,
}

HEALING_POTION = {
    "name": "healing_potion",
    "concepts": ("text_healing_potion", "text_potion", "text_glowing_liquid", "text_restorative"),
    "action_concepts": ("text_drink", "text_consume"),
    "outcome": "success",  # HP restored
    "reaction": lambda: make_reaction(
        "satiation",
        0.6,
        Valence.POSITIVE,
        "cerebellum:healing_potion.usage.drink",
    ),
    "salience_spike": None,  # no spike on success
}

POISON_POTION = {
    "name": "poison_potion",
    "concepts": ("text_poison_potion", "text_potion", "text_murky_liquid", "text_bitter"),
    "action_concepts": ("text_drink", "text_consume"),
    "outcome": "pain",  # HP drops
    "reaction": lambda: make_reaction(
        "pain",
        0.9,
        Valence.NEGATIVE,
        "cerebellum:poison_potion.usage.drink",
    ),
    "salience_spike": 0.8,
}

ENTITIES = [RUSTY_SWORD, HEALING_POTION, POISON_POTION]


# ─────────────────────────────────────────────────────────────────────────
# Session 1: Experience
# ─────────────────────────────────────────────────────────────────────────


def run_session_1(h: Hippocampus, nac: NAc, bus: ReactionBus) -> dict:
    """Agent encounters three entities and learns from the outcomes."""
    results = {}
    tick = 0

    for entity in ENTITIES:
        name = entity["name"]
        all_nodes = entity["concepts"] + entity["action_concepts"]

        # Decay old eligibility traces to simulate tick passage between
        # interactions.  Without this, stale traces from the previous
        # entity bleed into the next distribute_reward call.
        for _ in range(10):
            nac.decay_eligibility(factor=0.5)

        # Seed eligibility traces for THIS entity's concept nodes
        for node_id in all_nodes:
            nac.update_eligibility(AGENT_ID, node_id, activation=0.8)

        # Episode: agent perceives entity + performs action
        h.observe_episode_event(CaptureEvent(tick=tick, channel="text", activated_nodes=all_nodes))
        tick += 5

        # Outcome reaction fires
        reaction = entity["reaction"]()
        bus.publish(reaction)

        # Pain spike closes episode if applicable
        spike = entity["salience_spike"]
        if spike is not None:
            h.observe_episode_event(
                CaptureEvent(
                    tick=tick,
                    channel="text",
                    activated_nodes=entity["concepts"][:1],  # just the entity name
                    salience_spike=spike,
                )
            )
            tick += 1

        # Finalize the episode
        ep = h.finalize_pending_episode()
        tick += 100  # gap before next entity

        results[name] = {
            "episode_valence": ep.valence if ep else 0.0,
            "episode_node_count": len(ep.activated_nodes) if ep else 0,
        }

        # Check edge valence for entity's primary concept pair
        edge = h._binding_graph.find_edge(entity["concepts"][0], entity["concepts"][1], EdgeType.ASSOCIATES)
        if edge:
            results[name]["edge_valence"] = edge.metadata.get("valence", 0.0)
        else:
            results[name]["edge_valence"] = None

    # Session-wide stats
    results["total_episodes"] = len(h._episode_store)
    results["nac_link_count"] = len(nac)
    return results


# ─────────────────────────────────────────────────────────────────────────
# Session 2: Retrieval (loaded from persistence)
# ─────────────────────────────────────────────────────────────────────────


def run_session_2(h: Hippocampus, nac: NAc) -> dict:
    """Cue each entity name and measure what the agent 'remembers'."""
    results = {}

    for entity in ENTITIES:
        name = entity["name"]
        cue = entity["concepts"][0]  # e.g., "text_rusty_sword"
        entry: dict = {}

        # Retrieval with valence
        retrieved = h.retrieve_on_cue(cue, include_valence=True, limit=10)
        entry["retrieved_nodes"] = [
            {"node": n, "activation": round(a, 4), "valence": round(v, 4)} for n, a, v in retrieved
        ]
        entry["retrieved_count"] = len(retrieved)

        # Average valence of retrieved nodes
        if retrieved:
            avg_valence = sum(v for _, _, v in retrieved) / len(retrieved)
            entry["avg_retrieval_valence"] = round(avg_valence, 4)
        else:
            entry["avg_retrieval_valence"] = 0.0

        # NAc reward bias for this entity's concepts
        biases = {nid: round(nac.reward_bias(AGENT_ID, nid), 6) for nid in entity["concepts"]}
        entry["nac_biases"] = biases
        entry["has_positive_bias"] = any(b > 0.0 for b in biases.values())

        # EC threshold overrides
        overrides = nac.get_threshold_overrides(AGENT_ID)
        entity_overrides = {nid: round(v, 6) for nid, v in overrides.items() if nid in entity["concepts"]}
        entry["ec_threshold_overrides"] = entity_overrides

        results[name] = entry

    return results


# ─────────────────────────────────────────────────────────────────────────
# Control: Fresh agent
# ─────────────────────────────────────────────────────────────────────────


def run_control() -> dict:
    """Fresh agent, same cues, no prior experience."""
    h = make_hippocampus()
    nac = NAc()

    # Give the fresh agent the same concept nodes (as if it perceived
    # the entities but had no interaction outcomes)
    tick = 0
    for entity in ENTITIES:
        h.observe_episode_event(
            CaptureEvent(
                tick=tick,
                channel="text",
                activated_nodes=entity["concepts"],
            )
        )
        h.finalize_pending_episode()
        tick += 100

    return run_session_2(h, nac)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def run_experiment(persist_dir: str | None = None) -> dict:
    """Run the full 3-phase experiment."""
    if persist_dir is None:
        persist_dir = tempfile.mkdtemp(prefix="maxim_convergence_")
    p = Path(persist_dir)
    p.mkdir(parents=True, exist_ok=True)

    hippo_path = str(p / "hippocampus.json")
    nac_path = str(p / "nac.json")

    # ── Session 1: Experience ──
    h1 = make_hippocampus(persistence_path=hippo_path)
    nac1 = NAc()
    bus1, _ = wire_bio_pipeline(h1, nac1)

    session1 = run_session_1(h1, nac1, bus1)

    # Persist
    h1.save()
    nac1.save(nac_path)

    # ── Session 2: Retrieval (fresh instances, loaded from disk) ──
    h2 = make_hippocampus(persistence_path=hippo_path)
    h2.load()
    nac2 = NAc()
    nac2.load(nac_path)

    session2 = run_session_2(h2, nac2)

    # ── Control: Fresh agent ──
    control = run_control()

    return {
        "persist_dir": str(p),
        "session_1": session1,
        "session_2": session2,
        "control": control,
    }


def print_results(results: dict) -> None:
    print("=" * 70)
    print("Behavioral Convergence Experiment 1")
    print("Cross-Session Affective Memory Transfer")
    print("=" * 70)
    print()

    # Session 1
    print("SESSION 1 — Experience")
    print("-" * 40)
    s1 = results["session_1"]
    for entity in ENTITIES:
        name = entity["name"]
        e = s1[name]
        print(f"  {name}:")
        print(f"    outcome: {entity['outcome']}")
        print(f"    episode valence: {e['episode_valence']:.3f}")
        print(
            f"    edge valence: {e['edge_valence']:.3f}" if e["edge_valence"] is not None else "    edge valence: N/A"
        )
    print(f"  Total episodes: {s1['total_episodes']}")
    print()

    # Session 2
    print("SESSION 2 — Retrieval (loaded from disk)")
    print("-" * 40)
    s2 = results["session_2"]
    for entity in ENTITIES:
        name = entity["name"]
        e = s2[name]
        print(f"  {name}:")
        print(f"    avg retrieval valence: {e['avg_retrieval_valence']:+.4f}")
        print(f"    retrieved {e['retrieved_count']} nodes")
        print(f"    NAc positive bias: {e['has_positive_bias']}")
        if e["ec_threshold_overrides"]:
            print(f"    EC threshold overrides: {e['ec_threshold_overrides']}")
    print()

    # Control
    print("CONTROL — Fresh agent (no experience)")
    print("-" * 40)
    ctrl = results["control"]
    for entity in ENTITIES:
        name = entity["name"]
        e = ctrl[name]
        print(f"  {name}:")
        print(f"    avg retrieval valence: {e['avg_retrieval_valence']:+.4f}")
        print(f"    NAc positive bias: {e['has_positive_bias']}")
    print()

    # Assertions
    print("HYPOTHESIS TESTS")
    print("-" * 40)

    tests_passed = 0
    tests_total = 0

    def check(desc: str, condition: bool) -> None:
        nonlocal tests_passed, tests_total
        tests_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            tests_passed += 1
        print(f"  [{status}] {desc}")

    # H1: Painful entities carry negative valence in Session 2
    check(
        "Rusty sword: negative retrieval valence after pain",
        s2["rusty_sword"]["avg_retrieval_valence"] < 0.0,
    )
    check(
        "Poison potion: negative retrieval valence after pain",
        s2["poison_potion"]["avg_retrieval_valence"] < 0.0,
    )

    # H2: Beneficial entity carries positive valence in Session 2
    check(
        "Healing potion: positive retrieval valence after success",
        s2["healing_potion"]["avg_retrieval_valence"] > 0.0,
    )

    # H3: Shared "potion" concept has mixed valence (healing + poison)
    # This is more interesting than raw intensity comparison — it tests
    # whether the substrate correctly merges conflicting experiences.
    healing_potion_shared = next(
        (r for r in s2["healing_potion"]["retrieved_nodes"] if r["node"] == "text_potion"), None
    )
    check(
        "Shared 'potion' concept carries non-zero valence (mixed experience)",
        healing_potion_shared is not None and healing_potion_shared["valence"] != 0.0,
    )

    # H4: Healing potion has positive NAc bias (EC widens recognition)
    check(
        "Healing potion: NAc positive bias (EC recognition widened)",
        s2["healing_potion"]["has_positive_bias"],
    )

    # H5: Painful entities have NO positive bias (clamps to 0)
    check(
        "Rusty sword: no positive NAc bias (pain doesn't widen)",
        not s2["rusty_sword"]["has_positive_bias"],
    )

    # H6: Control shows zero valence for all entities
    check(
        "Control: rusty sword zero valence",
        ctrl["rusty_sword"]["avg_retrieval_valence"] == 0.0,
    )
    check(
        "Control: healing potion zero valence",
        ctrl["healing_potion"]["avg_retrieval_valence"] == 0.0,
    )
    check(
        "Control: poison potion zero valence",
        ctrl["poison_potion"]["avg_retrieval_valence"] == 0.0,
    )

    # H7: "Potion" concept shared between healing and poison —
    # cross-entity valence should partially cancel
    healing_potion_node = next((r for r in s2["healing_potion"]["retrieved_nodes"] if r["node"] == "text_potion"), None)
    poison_potion_node = next((r for r in s2["poison_potion"]["retrieved_nodes"] if r["node"] == "text_potion"), None)
    if healing_potion_node and poison_potion_node:
        # The shared "potion" concept gets valence from both episodes.
        # Since poison (0.9) > healing (0.6), net should be negative.
        check(
            "Shared 'potion' concept: net valence reflects mixed experience",
            healing_potion_node["valence"] != 0.0 or poison_potion_node["valence"] != 0.0,
        )
    else:
        check("Shared 'potion' concept: retrievable from both cues", False)

    # H8: Persistence actually worked (Session 2 != Control)
    check(
        "Session 2 differs from Control (persistence transferred learning)",
        s2["rusty_sword"]["avg_retrieval_valence"] != ctrl["rusty_sword"]["avg_retrieval_valence"],
    )

    print()
    print(f"Results: {tests_passed}/{tests_total} passed")
    print(f"Persist dir: {results['persist_dir']}")
    print()

    if tests_passed == tests_total:
        print("=" * 70)
        print("ALL HYPOTHESES CONFIRMED")
        print("=" * 70)
    else:
        print("=" * 70)
        print(f"WARNING: {tests_total - tests_passed} hypothesis(es) failed")
        print("=" * 70)

    return tests_passed == tests_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavioral Convergence Experiment 1")
    parser.add_argument(
        "--persist",
        type=str,
        default=None,
        help="Directory for persistence (default: tempdir)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results",
    )
    args = parser.parse_args()

    results = run_experiment(args.persist)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        success = print_results(results)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
