#!/usr/bin/env python
"""Habituation — can a novel sound triumph over a wall of familiar noise?

Novelty is over the JOINT ``(sound-content × direction)``, not direction alone: a
familiar car engine heard from the road habituates, but the SAME engine from your
bedroom is a rare (content, direction) pair → surprising. This is a violated
expectation in CONTEXT, the brain's real model of "what comes from where".

The salience gate: an agent facing a SCENE of concurrent sounds attends to the
most salient one. Salience of a sound =
    novelty(joint)                # decays with repetition — habituation
  + reward_weight × value(content)   # a learned-important content (mother's voice)
                                     # stays salient DESPITE familiarity
Novelty uses the ``tools/novelty.py`` decay (0.85 per repetition).

The stress test (the point): habituate a growing WALL of background noises, then
drop in ONE subtle novel sound. Does the agent still catch it?
  - HABITUATING agent: the noise is suppressed (novelty→0), so the novel sound
    (novelty≈1) stands alone → caught, even against 40 noises. Habituation is
    what MAKES novelty-detection-in-noise possible.
  - FLAT agent (no habituation): every sound is equally "novel" → the target is 1
    of N+1 → it drowns as the noise grows. This is the control that shows
    habituation has a FUNCTION, not just a decay curve.

Reward protection: the mother's voice (a rewarded content) stays attended even
when as familiar as the noise — the cocktail-party effect (tune out the hum, snap
to your name).

Federation (individual vs COLLECTIVE habituation): merge the crèche's familiarity
counts (a hivemind pools its expectation model — what sound comes from where is
normal). A crèche-raised agent then suppresses noises the COLLECTIVE found common
even if it never personally heard them, so the novel sound triumphs for it too.

NOTE (honest scope): joint novelty is tracked DIRECTLY over ``(content, direction)``
here — the substrate's LinguisticEncoder is text-based and does not cleanly cluster
a numeric (content×direction) code, so EC-native joint clustering (which would add
generalization: similar sounds share habituation) is a follow-up. The salience/
novelty model + the novel-in-noise question are faithful.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

DECAY = 0.85  # tools/novelty.py: novelty decays 0.85 per repetition
REWARD_W = 1.0
DIRECTIONS = (0, 1, 2, 3)  # a few directions; the joint is (content, direction)


def novelty(counts, joint, habituate=True):
    return (DECAY ** counts[joint]) if habituate else 1.0


def salience(content, direction, counts, value, *, habituate=True):
    return novelty(counts, (content, direction), habituate) + REWARD_W * value.get(content, 0.0)


def attend(scene, counts, value, rng, *, habituate=True):
    """Return the content the agent attends to (max salience) from a scene of
    (content, direction) sounds. Ties broken RANDOMLY (a flat agent has no basis
    to prefer, so it lands at true chance rather than a deterministic first-pick)."""
    jitter = rng.random(len(scene)) * 1e-9
    scored = [(salience(c, d, counts, value, habituate=habituate) + jitter[i], c) for i, (c, d) in enumerate(scene)]
    return max(scored, key=lambda x: x[0])[1]


def habituate_background(counts, noises, reps, rng):
    """Expose the agent to the background noise scene ``reps`` times (each noise
    from its home direction), accruing familiarity counts."""
    for _ in range(reps):
        for c, d in noises:
            counts[(c, d)] += 1


def stress_trial(*, n_noise, reps, habituate, seed, with_mother=False, mother_reps=0):
    """One agent: habituate n_noise background noises, then face a scene of those
    noises + one NOVEL sound. Returns (caught_novel, caught_mother_if_present)."""
    rng = np.random.default_rng(seed)
    counts = defaultdict(int)
    value = {"mother": 3.0} if with_mother else {}
    # each noise has a home direction; the novel is a new content from some direction
    noises = [(f"noise{i}", int(DIRECTIONS[i % len(DIRECTIONS)])) for i in range(n_noise)]
    habituate_background(counts, noises, reps, rng)
    if with_mother:
        # the mother's voice is ALSO heard often (as familiar as the noise)
        for _ in range(mother_reps):
            counts[("mother", 0)] += 1

    # the test scene: all the (now-familiar) noises + one subtle novel sound
    novel = ("novel", int(rng.integers(len(DIRECTIONS))))
    scene = list(noises) + [novel]
    if with_mother:
        scene.append(("mother", 0))
    winner = attend(scene, counts, value, rng, habituate=habituate)
    return (winner == "novel", winner == "mother")


def federation_trial(*, n_noise, n_agents, reps, per_agent_frac, seed):
    """Individual vs COLLECTIVE habituation. Each infant hears only a RANDOM
    SUBSET of the background noises (partial exposure), so alone its un-heard
    noises are NOT habituated and compete with the novel sound. The crèche pools
    every infant's familiarity counts, so collectively every noise is habituated —
    a crèche-raised agent suppresses noises it never personally heard, and the
    novel sound triumphs for it. Returns (solo_catches_novel, creche_catches_novel)."""
    rng = np.random.default_rng(seed)
    noises = [(f"noise{i}", int(DIRECTIONS[i % len(DIRECTIONS)])) for i in range(n_noise)]
    novel = ("novel", int(rng.integers(len(DIRECTIONS))))
    k = max(1, int(round(n_noise * per_agent_frac)))
    agent_counts = []
    for _ in range(n_agents):
        c = defaultdict(int)
        for idx in rng.choice(n_noise, size=k, replace=False):
            c[noises[idx]] += reps
        agent_counts.append(c)
    solo = agent_counts[0]
    pooled = defaultdict(int)
    for c in agent_counts:
        for key, v in c.items():
            pooled[key] += v
    scene = list(noises) + [novel]
    solo_catch = attend(scene, solo, {}, rng, habituate=True) == "novel"
    creche_catch = attend(scene, pooled, {}, rng, habituate=True) == "novel"
    return solo_catch, creche_catch


def main():
    p = argparse.ArgumentParser(description="Habituation: novel sound in noise")
    p.add_argument(
        "--reps", type=int, default=30, help="how many times the background noise is heard (habituation depth)"
    )
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--noise-levels", default="1,5,10,20,40")
    args = p.parse_args()

    levels = [int(x) for x in args.noise_levels.split(",")]
    print(f"habituation — novel sound in a wall of noise (reps={args.reps}, {args.seeds} seeds/level)\n")
    print(f"{'noises':>7} | {'habituating catch-novel':>24} | {'flat (no habit.) catch-novel':>28}")
    print("-" * 68)
    for n in levels:
        hab = np.mean([stress_trial(n_noise=n, reps=args.reps, habituate=True, seed=s)[0] for s in range(args.seeds)])
        flat = np.mean([stress_trial(n_noise=n, reps=args.reps, habituate=False, seed=s)[0] for s in range(args.seeds)])
        print(f"{n:>7} | {hab:>24.2f} | {flat:>28.2f}   (chance {1 / (n + 1):.2f})")

    # reward protection: the mother's voice, as familiar as the noise, stays attended
    print("\nreward protection (cocktail party) — mother's voice among 20 habituated noises:")
    caught_m = np.mean(
        [
            stress_trial(n_noise=20, reps=args.reps, habituate=True, seed=s, with_mother=True, mother_reps=args.reps)[1]
            for s in range(args.seeds)
        ]
    )
    print(f"  attends to the mother's (equally-familiar) voice: {caught_m:.2f}  (habituation alone would suppress it)")

    # federation: individual vs collective habituation (each infant hears half the noises)
    print("\nfederation — individual vs COLLECTIVE habituation (20 noises, each infant hears half):")
    solo = np.mean(
        [
            federation_trial(n_noise=20, n_agents=12, reps=args.reps, per_agent_frac=0.5, seed=s)[0]
            for s in range(args.seeds)
        ]
    )
    creche = np.mean(
        [
            federation_trial(n_noise=20, n_agents=12, reps=args.reps, per_agent_frac=0.5, seed=s)[1]
            for s in range(args.seeds)
        ]
    )
    print(f"  solo (heard half the noises) catches novel  : {solo:.2f}   (un-heard noises compete)")
    print(f"  crèche (pooled counts) catches novel        : {creche:.2f}   (collective habituation)")

    print("\nverdict:")
    hab40 = np.mean([stress_trial(n_noise=40, reps=args.reps, habituate=True, seed=s)[0] for s in range(args.seeds)])
    flat40 = np.mean([stress_trial(n_noise=40, reps=args.reps, habituate=False, seed=s)[0] for s in range(args.seeds)])
    print(
        f"  HABITUATION ENABLES NOVELTY-IN-NOISE (habituating catches novel in 40-noise wall, flat drowns): "
        f"{'PASS' if hab40 >= 0.9 and flat40 <= 0.2 else 'FAIL'}  (hab={hab40:.2f} flat={flat40:.2f})"
    )
    print(f"  REWARD PROTECTS SALIENCE (mother's voice survives habituation): {'PASS' if caught_m >= 0.9 else 'FAIL'}")
    print(
        f"  COLLECTIVE > INDIVIDUAL HABITUATION (crèche catches novel more than solo): {'PASS' if creche >= solo + 0.2 else 'FAIL'}  (solo={solo:.2f} crèche={creche:.2f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
