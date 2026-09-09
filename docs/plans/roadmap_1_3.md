# Roadmap 1.3 — "Oasis-2": the survival world

**Scoped 2026-09-09** (owner decision: survival world ← 1.3; perception fabric + microduck +
Exp 55 → 1.4 — see the rescope note in [roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md)). This is
the phased plan and dependency map; the world/classroom designs live in
[survival_world_1_3.md](survival_world_1_3.md) and [intrinsic_motivation_1_3.md](intrinsic_motivation_1_3.md),
and the Minecraft benchmark in [minecraft_benchmark.md](minecraft_benchmark.md) Part II.

## The thesis

1.2 "Oasis" showed a *taught* want transfers between independent agents (Exp 56 EARNED) and
scales per-participant (Exp 57 PARTIAL). **1.3 shifts the reward from a teacher to the game
itself, and shares a *survival* want:** an agent learns, the hard way, that something in the
world keeps it alive; that want transfers to a second agent, which survives its first night
never having been bitten. The reframe from Exp 52/56's teacher-minted credit to game-native
relief is the whole 1.3 move.

## The reframe that makes 1.3 tractable — mechanisms first

1.3 is not "build all the classrooms." It is **build (up to) three mechanisms; the classrooms
fall out of whichever ones exist.** Every rung is marked by the mechanism it needs, so none
reads as ready-to-build before its mechanism does. The two load-bearing nulls from 1.2 are the
build list, not obstacles: **R2** (drives don't yet move behaviour — the survival loop's three
breaks) and **R1** (no cross-context generalization — the substrate is exact-key).

## Phases (by dependency)

### Phase 0 — Platform + instrument *(cheap, gating; do first)*
- Stand up the chosen **modern MC version (1.20.1 / 1.20.4)**, port the bridge/sensors, and
  **verify sensor separability through the real encoder before any claim** (the `light_level`-
  dead lesson: a classroom the sensors can't separate can't be learned).
- Resolve the re-baseline: moving off 1.16.5 re-triggers the Exp 56/57 guards — re-run the
  shared-want fabric on 1.20.1 or keep it pinned; do not run two versions long-term.
- Platform facts (see [survival_world_1_3.md](survival_world_1_3.md) §"World vs. substrate" and
  §"Minecraft version"): Java Edition only (mineflayer); classrooms built by **RCON command
  scripts**, nothing hand-built; the operator hosts (Paper server + bridge + harness).

### Phase 1 — The survival loop (R2's three breaks) *(THE gate; nothing dynamic works without it)*
- Build: (1) drive→corrective-action affinity; (2) measured-relief credit for interoceptive
  world drives; (3) executable corrective acts in-world.
- **Validate with the dining hall** (spawn low-health → learn to `eat` → relief; the R2-break-1
  flagship) and the cheap **dark=danger avoidance** probe (rides the already-wired negative-
  credit path — the most tractable first contingency).

### Phase 2 — Shared survival wants *(the pivotal may-fail claim; the 1.3 headline)*
Agent A learns "dark = danger" / eat-when-hungry the hard way → exports its substrate → agent B
ingests it → **B survives its first night better, never bitten.** Needs only Phase 1 + the 1.2
fabric — **not** R1 or R4 — so it is the honest minimal 1.3 that can actually ship. A recorded
failure ships as a failure.

### Phase 3 — R3 survival benchmark *(instrument + frozen baseline, Goldilocks-calibrated)*
Measures the survival advantage the learned drives buy. **Single-step wolf defense** rides here
(wild wolf attacks the agent's attacker → health relief → operant credit). Instrument first,
verdict second.

### Phase 4 — R1 generalization channel *(frontier; likely 1.3-late / 1.4)*
Cross-context. Unlocks the dining-hall→wolf *transfer*, spatial "resource is over there," and
cross-layout. A genuine architectural build — the substrate is exact-key today (R1
CACHE-CONFIRMED).

### Phase 5 — R4 delayed / multi-step credit *(frontier; likely 1.3-late / 1.4)*
Unlocks **crafting** (the delayed-credit showcase, not a recipe cache), **wolf upkeep**,
**farming**, **shelter**. The canonical "does tick-anchored credit reach a delayed
construction?" question.

### Phase 6 — Intrinsic motivation *(own line, parallel — sibling of R3/R4)*
`success × novelty` vs learning-progress (`Δsuccess`); the **mining classroom** is its testbed;
guardrail — it must NOT silently power R3 (declared ablation arm or its own line, never an
undeclared default). Full design: [intrinsic_motivation_1_3.md](intrinsic_motivation_1_3.md).

## Critical path vs. frontier

- **Shippable 1.3 = Phase 0 → 1 → 2 (+3).** A shared *survival* want, measured. That is the
  defensible minimum and the headline.
- **R4-dependent classrooms (crafting, farming, wolf upkeep) realistically slip to 1.3-late or
  1.4** — build them only after R4 exists, or they become recipe caches (R1's trap).
- **Breeding is 1.4+** (population dynamics, a different axis).

## Discipline that carries (every rung)

- **D1 — game-native pressure only.** Reward arrives as a game-native relief through the operant-
  credit path, never hand-coded; no synthetic sensor (no `is_in_cave` flag — the agent forms the
  concept from real sensors, or it hasn't learned anything).
- **Verify the instrument first.** Confirm the encoder separates a classroom's situation before
  running the experiment (the R1 threshold issue and the dead `light_level` are the standing
  reminders).
- **A classroom passes one test:** its reward is a game-native relief through the credit path.
  If hand-coded, it is engineered and teaches nothing real.

## Not in 1.3 (moved / parallel)

- **Perception fabric + reflex tier + microduck + Exp 55 → 1.4** (the hardware thread; sequencing
  in [roadmap_1_3_path.md](roadmap_1_3_path.md), now the 1.4 plan). Independent of the Minecraft
  survival world.
