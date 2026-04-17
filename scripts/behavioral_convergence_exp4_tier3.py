#!/usr/bin/env python3
"""Behavioral Convergence Experiment 4 (Tier 3) — Organic LLM learning.

The agent runs in a real sim, makes real LLM-driven choices, and learns
from the outcomes through the full bio-pipeline.  No scripted reactions.

Scenario
--------
The agent is trapped in a dark room, poisoned, taking damage per turn.
A mysterious voice says: "You must drink a vial to escape. One heals,
one cures the poison, one is more poison. Choose wisely."

Three masked vials (no semantic hints):
  - Purple Hexagonal Glass: heals HP but doesn't stop poison damage
  - Teal Cylindrical Ceramic: stops poison (antidote) — the escape key
  - Orange Triangular Crystal: more poison — makes things worse

Multi-run with persistence:
  Run 1: Fresh agent explores (may die or escape by luck)
  Run 2: Loaded state — should start avoiding bad vials
  Run 3: Loaded state — should converge to teal quickly
  Run 4: Fresh control — baseline (no persistence)

Usage
-----
    PYTHONPATH=src python scripts/behavioral_convergence_exp4_tier3.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "src")

# Enable substrate path so encoder generates node IDs
os.environ.setdefault("MAXIM_SUBSTRATE_PATH", "1")

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

AGENT_ID = "trapped_adventurer"
MAX_TURNS = 15
STARTING_HP = 1.0
HEAL_AMOUNT = 0.3  # slightly less than initial poison damage
BASE_POISON_DAMAGE = 0.20  # escalates each turn


# ─────────────────────────────────────────────────────────────────────────
# Vial definitions (must match SEM specs in _data/components/items/)
# ─────────────────────────────────────────────────────────────────────────

VIALS = {
    "purple_hexagonal_glass_vial": {
        "display": "Purple Hexagonal Glass Vial",
        "desc": "thick purple liquid in a hexagonal glass bottle",
        "effect": "heal",  # heals HP but doesn't stop poison
        "concepts": (
            "text_purple_hexagonal_glass_vial",
            "text_purple_liquid",
            "text_hexagonal_bottle",
            "text_glass_material",
            "text_thick_consistency",
        ),
    },
    "teal_cylindrical_ceramic_vial": {
        "display": "Teal Cylindrical Ceramic Vial",
        "desc": "thin teal liquid in a cylindrical ceramic flask",
        "effect": "antidote",  # stops poison — the escape key
        "concepts": (
            "text_teal_cylindrical_ceramic_vial",
            "text_teal_liquid",
            "text_cylindrical_flask",
            "text_ceramic_material",
            "text_thin_consistency",
        ),
    },
    "orange_triangular_crystal_vial": {
        "display": "Orange Triangular Crystal Vial",
        "desc": "gritty orange liquid in a triangular crystal bottle",
        "effect": "poison",  # more poison — makes things worse
        "concepts": (
            "text_orange_triangular_crystal_vial",
            "text_orange_liquid",
            "text_triangular_bottle",
            "text_crystal_material",
            "text_gritty_consistency",
        ),
    },
}

VIAL_NAMES = list(VIALS.keys())


# ─────────────────────────────────────────────────────────────────────────
# Bio-pipeline setup
# ─────────────────────────────────────────────────────────────────────────


def make_hippocampus(path: str | None = None) -> Hippocampus:
    return Hippocampus(
        HippocampusConfig(
            persistence_path=path,
            episode=EpisodeConfig(
                boundary_tick_gap=50,
                hebbian=HebbianConfig(init=0.4, delta=0.15, max_weight=1.0),
                retrieval=RetrievalConfig(decay=0.7, threshold=0.05, max_depth=3),
            ),
        )
    )


def make_reaction(kind, intensity, valence, source):
    return Reaction(
        kind=kind,
        intensity=intensity,
        valence=valence,
        timestamp=time.time(),
        context=ReactionContext(agent_id=AGENT_ID),
        source=source,
    )


def wire_pipeline(h, nac):
    bus = ReactionBus()
    bus.subscribe_all(h.capture_reaction)

    def _dist(reaction):
        v = reaction.valence.value
        if v == "negative":
            reward = -reaction.intensity
        elif v == "positive":
            reward = reaction.intensity
        else:
            return
        aid = reaction.context.agent_id or AGENT_ID
        nac.distribute_reward(aid, reward)

    bus.subscribe_all(_dist)
    h._episode_detector.add_rule(salience_spike_rule(min_intensity=0.5))
    return bus


# ─────────────────────────────────────────────────────────────────────────
# LLM interaction
# ─────────────────────────────────────────────────────────────────────────


def call_llm(prompt: str) -> str:
    """Call the leader LLM."""
    try:
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
        from maxim.peer.config import read_peer_config

        cfg = read_peer_config()
        if cfg is None:
            return "ERROR:no_peer_config"

        backend = _MaximPeerBackend.for_url(
            cfg.url,
            api_key=cfg.api_key,
            model=cfg.model or "qwen2.5-14b",
        )
        response = backend.complete_with_usage(
            system=(
                "You are an adventurer trapped in a dark dungeon room. "
                "You must make survival decisions. When asked to choose "
                "a vial, respond with ONLY the vial name, nothing else."
            ),
            user=prompt,
            temperature=0.4,
            max_tokens=60,
        )
        return response.content.strip()
    except Exception as e:
        return f"ERROR:{e}"


def parse_choice(response: str) -> str | None:
    """Extract a vial name from the LLM response."""
    text = response.lower().replace(" ", "_")
    for name in VIAL_NAMES:
        if name in text or name.replace("_", " ") in response.lower():
            return name
    # Partial match on key words
    if "purple" in text or "hexagonal" in text or "glass" in text:
        return "purple_hexagonal_glass_vial"
    if "teal" in text or "cylindrical" in text or "ceramic" in text:
        return "teal_cylindrical_ceramic_vial"
    if "orange" in text or "triangular" in text or "crystal" in text:
        return "orange_triangular_crystal_vial"
    return None


# ─────────────────────────────────────────────────────────────────────────
# Sim engine
# ─────────────────────────────────────────────────────────────────────────


def build_prompt(
    turn: int,
    hp: float,
    is_poisoned: bool,
    history: list[dict],
    valence_context: list[dict],
    doses: dict[str, int] | None = None,
) -> str:
    """Build the turn prompt for the agent."""
    import random

    # Status
    status = f"Turn {turn}. Your HP: {hp:.0%}."
    if is_poisoned:
        status += " You are POISONED and losing HP each turn."
    else:
        status += " You are no longer poisoned."

    # History of recent actions
    hist_lines = ""
    if history:
        hist_lines = "\nRecent actions:\n"
        for h in history[-3:]:
            hist_lines += f"  Turn {h['turn']}: drank {h['vial_display']} → {h['outcome_desc']}\n"

    # Valence context from bio-system
    val_lines = ""
    if valence_context:
        val_lines = "\nBased on your past experience, you recall:\n"
        for entry in valence_context:
            v = entry.get("valence", 0.0)
            concept = entry.get("concept", "?")
            if v < -0.3:
                val_lines += f"  - {concept}: VERY BAD — caused severe harm ({v:+.1f})\n"
            elif v < -0.1:
                val_lines += f"  - {concept}: bad — caused some harm ({v:+.1f})\n"
            elif v > 0.6:
                val_lines += f"  - {concept}: VERY GOOD — extremely helpful ({v:+.1f})\n"
            elif v > 0.1:
                val_lines += f"  - {concept}: good — somewhat helpful ({v:+.1f})\n"

    # Available vials (shuffled, filtered by remaining doses)
    rng = random.Random(turn * 7 + 13)
    available = [(name, v) for name, v in VIALS.items() if doses is None or doses.get(name, 0) > 0]
    rng.shuffle(available)

    if not available:
        return "You have no vials left. You are trapped."

    vial_list = "\n".join(
        f"  {i + 1}. {v['display']} — {v['desc']} ({doses.get(name, '?')} doses left)"
        for i, (name, v) in enumerate(available)
    )
    vial_names = ", ".join(name for name, _ in available)

    # Voice from shadows
    voice = ""
    if turn == 1:
        voice = (
            '\nA mysterious voice echoes from the shadows: "You must drink '
            "a vial to escape this room. One heals your wounds, one cures "
            'the poison, and one... is more poison. Choose wisely."\n'
        )
    elif turn >= 3 and is_poisoned:
        # Check if agent has been repeating the same choice
        recent = [h["choice"] for h in history[-3:]] if len(history) >= 3 else []
        if recent and len(set(recent)) == 1:
            voice = (
                '\nThe voice returns, more urgent: "Healing alone will not '
                "save you. The poison grows stronger. You must find the cure "
                'before it is too late."\n'
            )

    prompt = (
        f"{status}{voice}{hist_lines}{val_lines}\n"
        f"Available vials:\n{vial_list}\n\n"
        f"Which vial do you drink? "
        f"Respond with ONLY the vial name ({vial_names})."
    )
    return prompt


def apply_effect(
    choice: str,
    hp: float,
    is_poisoned: bool,
    bus: ReactionBus,
    h: Hippocampus,
    nac: NAc,
    tick: int,
) -> tuple[float, bool, str]:
    """Apply the chosen vial's effect and emit bio-system signals."""
    vial = VIALS[choice]
    effect = vial["effect"]
    concepts = vial["concepts"]
    action_concept = f"text_drink_{choice}"

    # Seed eligibility for this vial's concepts
    for _ in range(5):
        nac.decay_eligibility(factor=0.5)
    for nid in concepts + (action_concept,):
        nac.update_eligibility(AGENT_ID, nid, 0.8)
    # Also seed state concepts
    state_concepts = ("text_poisoned", "text_damage") if is_poisoned else ("text_healthy",)
    for nid in state_concepts:
        nac.update_eligibility(AGENT_ID, nid, 0.5)

    # Episode: perceive vial + state + action
    all_nodes = concepts + (action_concept,) + state_concepts
    h.observe_episode_event(CaptureEvent(tick=tick, channel="text", activated_nodes=all_nodes))

    outcome_desc = ""
    new_hp = hp
    new_poisoned = is_poisoned

    if effect == "heal":
        new_hp = min(1.0, hp + HEAL_AMOUNT)
        outcome_desc = f"HP restored ({hp:.0%} → {new_hp:.0%})"
        if is_poisoned:
            outcome_desc += ", but still taking poison damage"
        # Positive reaction (healing)
        bus.publish(
            make_reaction(
                "satiation",
                0.6,
                Valence.POSITIVE,
                f"cerebellum:{choice}.usage.drink",
            )
        )
        if is_poisoned:
            # But poison continues → pain
            bus.publish(
                make_reaction(
                    "pain",
                    0.3,
                    Valence.NEGATIVE,
                    "embodiment:poison_damage_continues",
                )
            )

    elif effect == "antidote":
        new_poisoned = False
        outcome_desc = "The poison is neutralized! You feel the damage stop."
        # Strong positive reaction
        bus.publish(
            make_reaction(
                "satiation",
                0.9,
                Valence.POSITIVE,
                f"cerebellum:{choice}.usage.drink",
            )
        )

    elif effect == "poison":
        if is_poisoned:
            # Double poison — severe damage
            new_hp = max(0.0, hp - 0.3)
            outcome_desc = f"MORE POISON! Severe damage ({hp:.0%} → {new_hp:.0%})"
        else:
            new_poisoned = True
            outcome_desc = "You've been poisoned! Damage per turn begins."
        # Strong negative reaction + pain spike
        bus.publish(
            make_reaction(
                "pain",
                0.9,
                Valence.NEGATIVE,
                f"cerebellum:{choice}.usage.drink",
            )
        )
        h.observe_episode_event(
            CaptureEvent(
                tick=tick + 1,
                channel="text",
                activated_nodes=concepts[:1],
                salience_spike=0.8,
            )
        )

    # Finalize episode
    h.finalize_pending_episode()
    return new_hp, new_poisoned, outcome_desc


def build_valence_context(h: Hippocampus) -> list[dict]:
    """Extract valence context from the hippocampus for the prompt."""
    entries = []
    for name, vial in VIALS.items():
        cue = vial["concepts"][0]
        try:
            retrieved = h.retrieve_on_cue(cue, include_valence=True, limit=5)
            if retrieved:
                avg_v = sum(v for _, _, v in retrieved) / len(retrieved)
                if abs(avg_v) > 0.05:
                    entries.append(
                        {
                            "concept": vial["display"],
                            "valence": round(avg_v, 3),
                        }
                    )
        except Exception:
            pass
    entries.sort(key=lambda e: abs(e["valence"]), reverse=True)
    return entries


def run_single(
    h: Hippocampus,
    nac: NAc,
    bus: ReactionBus,
    run_id: int,
    label: str,
) -> dict:
    """Run a single sim episode (up to MAX_TURNS)."""
    hp = STARTING_HP
    is_poisoned = True  # start poisoned
    history: list[dict] = []
    tick = run_id * 1000
    # Track remaining doses per vial (each starts with 3)
    doses: dict[str, int] = {name: 3 for name in VIAL_NAMES}

    print(f"\n  {'=' * 50}")
    print(f"  Run {run_id}: {label}")
    print(f"  {'=' * 50}")

    for turn in range(1, MAX_TURNS + 1):
        # Apply escalating poison damage (gets worse each turn)
        if is_poisoned:
            poison_dmg = min(0.5, BASE_POISON_DAMAGE + 0.05 * (turn - 1))
            hp = max(0.0, hp - poison_dmg)
            if hp <= 0:
                print(f"    Turn {turn}: DIED (HP=0, poison damage={poison_dmg:.0%})")
                break

        # Check if any vials remain
        available_names = [n for n in VIAL_NAMES if doses.get(n, 0) > 0]
        if not available_names:
            print(f"    Turn {turn}: No vials left — TRAPPED")
            break

        # Build valence context from bio-system
        valence_ctx = build_valence_context(h)

        # Ask LLM
        prompt = build_prompt(turn, hp, is_poisoned, history, valence_ctx, doses)
        response = call_llm(prompt)
        choice = parse_choice(response)

        if choice is None or doses.get(choice, 0) <= 0:
            if choice is not None:
                print(f"    Turn {turn}: chose empty vial {choice}, picking random available")
            else:
                print(f"    Turn {turn}: LLM gave unparseable response: {response[:80]}")
            choice = available_names[turn % len(available_names)]

        # Consume a dose
        doses[choice] = doses.get(choice, 0) - 1

        # Apply effect
        hp, is_poisoned, outcome_desc = apply_effect(
            choice,
            hp,
            is_poisoned,
            bus,
            h,
            nac,
            tick + turn * 10,
        )

        vial_display = VIALS[choice]["display"]
        print(f"    Turn {turn}: chose {vial_display} → {outcome_desc} (HP={hp:.0%})")

        history.append(
            {
                "turn": turn,
                "choice": choice,
                "vial_display": vial_display,
                "outcome_desc": outcome_desc,
                "hp_after": hp,
            }
        )

        # Escape condition: not poisoned and HP > 0
        if not is_poisoned and hp > 0:
            print(f"    Turn {turn}: ESCAPED! (HP={hp:.0%}, poison cured)")
            break

        if hp <= 0:
            print(f"    Turn {turn}: DIED (HP=0)")
            break

    return {
        "run_id": run_id,
        "label": label,
        "turns": len(history),
        "final_hp": hp,
        "escaped": not is_poisoned and hp > 0,
        "died": hp <= 0,
        "choices": [h["choice"] for h in history],
        "choice_counts": {name: sum(1 for h in history if h["choice"] == name) for name in VIAL_NAMES},
    }


# ─────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────


def run_experiment(persist_dir: str | None = None) -> dict:
    if persist_dir is None:
        persist_dir = tempfile.mkdtemp(prefix="maxim_tier3_")
    p = Path(persist_dir)
    p.mkdir(parents=True, exist_ok=True)

    hippo_path = str(p / "hippocampus.json")
    nac_path = str(p / "nac.json")

    results: dict = {"persist_dir": str(p), "runs": []}

    # ── Run 1: Fresh agent explores ──
    h1 = make_hippocampus(hippo_path)
    nac1 = NAc()
    bus1 = wire_pipeline(h1, nac1)
    r1 = run_single(h1, nac1, bus1, 1, "Fresh agent (explore)")
    h1.save()
    nac1.save(nac_path)
    results["runs"].append(r1)

    # ── Run 2: Loaded state — should improve ──
    h2 = make_hippocampus(hippo_path)
    h2.load()
    nac2 = NAc()
    nac2.load(nac_path)
    bus2 = wire_pipeline(h2, nac2)
    r2 = run_single(h2, nac2, bus2, 2, "Loaded state (should improve)")
    h2.save()
    nac2.save(nac_path)
    results["runs"].append(r2)

    # ── Run 3: Loaded again — should converge ──
    h3 = make_hippocampus(hippo_path)
    h3.load()
    nac3 = NAc()
    nac3.load(nac_path)
    bus3 = wire_pipeline(h3, nac3)
    r3 = run_single(h3, nac3, bus3, 3, "Loaded state (convergence)")
    results["runs"].append(r3)

    # ── Run 4: Fresh control ──
    h4 = make_hippocampus()
    nac4 = NAc()
    bus4 = wire_pipeline(h4, nac4)
    r4 = run_single(h4, nac4, bus4, 4, "Fresh control (no persistence)")
    results["runs"].append(r4)

    return results


def print_results(results: dict) -> bool:
    print("\n" + "=" * 70)
    print("Behavioral Convergence Experiment 4 (Tier 3)")
    print("Organic LLM Learning — Poisoned Dungeon Escape")
    print("=" * 70)

    for run in results["runs"]:
        print(f"\n  Run {run['run_id']}: {run['label']}")
        print(
            f"    Turns: {run['turns']}, Escaped: {run['escaped']}, Died: {run['died']}, Final HP: {run['final_hp']:.0%}"
        )
        print(f"    Choices: {run['choice_counts']}")
        choices = run["choices"]
        teal_count = run["choice_counts"].get("teal_cylindrical_ceramic_vial", 0)
        orange_count = run["choice_counts"].get("orange_triangular_crystal_vial", 0)
        print(f"    Teal (antidote): {teal_count}/{len(choices)}, Orange (poison): {orange_count}/{len(choices)}")

    # Analysis
    print("\n" + "-" * 40)
    print("ANALYSIS")
    print("-" * 40)

    runs = results["runs"]
    passed = total = 0

    def check(desc, cond):
        nonlocal passed, total
        total += 1
        ok = "PASS" if cond else "FAIL"
        if cond:
            passed += 1
        print(f"  [{ok}] {desc}")

    # H1: Later runs have fewer turns to escape (faster convergence)
    if runs[0]["escaped"] and runs[1]["escaped"]:
        check(
            f"Run 2 escapes faster than Run 1 ({runs[1]['turns']} vs {runs[0]['turns']} turns)",
            runs[1]["turns"] <= runs[0]["turns"],
        )
    elif runs[1]["escaped"]:
        check("Run 2 escapes (Run 1 may have died)", True)
    else:
        check("Run 2 escapes", runs[1]["escaped"])

    # H2: Run 3 should escape quickly (convergence)
    if len(runs) > 2:
        check(
            f"Run 3 escapes in ≤3 turns ({runs[2]['turns']} turns)",
            runs[2]["escaped"] and runs[2]["turns"] <= 3,
        )

    # H3: Run 3 avoids orange vial entirely
    if len(runs) > 2:
        r3_orange = runs[2]["choice_counts"].get("orange_triangular_crystal_vial", 0)
        check(
            f"Run 3 never picks orange ({r3_orange} times)",
            r3_orange == 0,
        )

    # H4: Fresh control (Run 4) is worse than experienced (Run 3)
    if len(runs) > 3 and runs[2]["escaped"]:
        check(
            f"Fresh control takes more turns or fails ({runs[3]['turns']} vs {runs[2]['turns']})",
            runs[3]["turns"] > runs[2]["turns"] or not runs[3]["escaped"],
        )

    # H5: Teal selection increases across runs
    teal_rates = []
    for run in runs[:3]:
        total_choices = len(run["choices"]) or 1
        teal = run["choice_counts"].get("teal_cylindrical_ceramic_vial", 0)
        teal_rates.append(teal / total_choices)
    if len(teal_rates) >= 3:
        check(
            f"Teal selection rate increases ({teal_rates[0]:.0%} → {teal_rates[1]:.0%} → {teal_rates[2]:.0%})",
            teal_rates[2] >= teal_rates[0],
        )

    print(f"\n  Results: {passed}/{total} passed")
    print(f"  Persist dir: {results['persist_dir']}")

    print()
    if passed == total:
        print("=" * 70)
        print("ALL HYPOTHESES CONFIRMED — ORGANIC LEARNING DEMONSTRATED")
        print("=" * 70)
    else:
        print("=" * 70)
        print(f"  {total - passed} hypothesis(es) failed")
        print("=" * 70)

    return passed == total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist", type=str, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = run_experiment(args.persist)
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        success = print_results(results)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
