# Dungeon Master Extensions Plan

> **Status:** Not started. Depends on [Dungeon Master Persona MVP](dungeon_master_persona.md) shipping and accumulating real usage.
>
> **Summary:** Optional follow-on capabilities for DM. Each extension is independently scoppable and should only ship if MVP usage reveals actual demand. Listed in rough priority order, but the *real* ordering comes from MVP pain points.

## Ordering Philosophy

MVP tells us which extensions matter. Don't commit to a sequence upfront. After MVP lands, collect evidence:

- How many campaigns get hand-authored? If one, architect is urgent. If many, hand-authoring is fine.
- Do users complain about encounter repetition? If yes, library is urgent.
- Does the AUT steamroll campaigns? If yes, adaptation is urgent.
- Do encounters corrupt downstream AUT state? If yes, sub-sim isolation matters.
- Are users asking for dice they can't predict? True-random RNG.

---

## Extension A — Reusable Encounter Library (~300 LOC)

**Motivation:** hand-authored campaigns share encounter patterns (ambushes, moral dilemmas, treasure rooms). A curated library lets campaign authors compose rather than retype.

**Design:**
- `scenarios/encounter_library/<category>/<descriptive_name>_v<N>.yaml` layout
- Manifest header per encounter: `tags`, `difficulty_range`, `narrative_role`, `required_flags`
- Campaigns reference library entries by ref string: `library/ambush/forest_bandits_v3`
- MVP-style inline encounters still supported; library is additive

**New files:**
- `src/maxim/simulation/encounter_schema.py` (~80) — standalone `Encounter` + `EncounterManifest`
- `src/maxim/simulation/encounter_loader.py` (~100) — ref resolution, manifest indexing
- `scenarios/encounter_library/**` — 6–10 seed encounters across categories

**Modified:**
- `src/maxim/simulation/campaign_schema.py` — support `encounter_refs` alongside inline definitions
- `src/maxim/simulation/dm_runtime.py` — resolve refs at campaign load

**Ship gate:** second campaign authored against library is demonstrably faster than authoring inline.

---

## Extension B — Interactive Adventure Architect Persona (~350 LOC)

**Motivation:** hand-authoring campaign YAML is the primary MVP friction. An architect persona interviews the user and generates campaigns.

**Depends on:** [Interactive Simulation Prompts](interactive_sim_prompts.md) (the `ask_user` tool)

**Depends on:** Extension A (library) — architect composes from library; without it the architect is generating from scratch every time, which was the whole problem.

**Design:**
- `adventure_architect` persona with ~8-question interview budget
- Interview phases: theme → scope → constraints → draft → refine
- Architect calls `browse_encounters`, `emit_campaign`
- Validator hard-fails; architect iterates to repair
- Output written to `scenarios/generated/{session}_{slug}.yaml`
- User-reviewed: architect proposes, user approves via `ask_user` before finalizing

**New files:**
- Entry in `src/maxim/simulation/personas.py` for `adventure_architect`
- `src/maxim/simulation/tools_dm.py` — add `emit_campaign`, `browse_encounters`, `propose_draft`

**Ship gate:** architect produces a campaign in <5 minutes of user interview that runs end-to-end without manual edits.

---

## Extension C — Adaptive Difficulty & Narrative Pacing (~200 LOC)

**Motivation:** static branches feel canned. Reading AUT internals lets DM adapt mid-campaign.

**Depends on:** `InspectAUTTool` (from Realtime Refinement, already shipped)

**Design:** before committing to implementation, **define the adaptation metrics first**. Candidate signals:
- AUT turns-per-encounter (baseline vs. current)
- AUT pain trend (from `inspect_aut(pain_history)`)
- AUT memory recall on prior NPCs (from `inspect_aut(memory_recall)`)
- AUT causal link count for repeated patterns (from `inspect_aut(causal_links)`)

Each metric needs an operational definition + threshold tuning before any adaptation rule is written. Expect 2–3 iterations of tuning per metric.

**Design phase (before implementation):**
- Run 5–10 MVP campaigns, log all candidate signals
- Identify which signals actually correlate with "too easy" / "too hard" outcomes
- Only then write adaptation rules

**Ship gate:** running the same campaign twice with different AUT behavior produces measurably different branch paths.

---

## Extension D — Encounter Isolation via Sub-Sims (~?? LOC, uncertain)

**Motivation:** MVP runs one long sim. If one encounter corrupts AUT state (wedged memory, stuck plan), the rest of the campaign is compromised. Sub-sim isolation lets each encounter run in its own scoped context.

**The hard question:** what does a persistent-memory sub-sim mean architecturally? Today's sub-sims spawn fresh AUTs. For DM isolation to work with narrative continuity, we'd need either:

1. **Nested goal scopes** — same AUT, orchestrator pushes a new goal onto a stack. Not really a "sub-sim." Clean but requires reworking `spawn_sub_simulation` semantics.
2. **Serialized AUT state** — fork AUT memory, run encounter, merge back. Complex, risk of state divergence.
3. **Recap-only** — fresh AUT per encounter, DM composes a "previously on…" prelude from NPC registry. Simple but loses real memory continuity.

**Do not start this extension** until we understand from MVP usage whether isolation is actually a problem. If MVP campaigns run fine without isolation, this extension is not needed.

**If it becomes needed:** likely pick option (3) recap-only first as the cheapest path, consider (1) nested goal scopes if recap proves insufficient.

---

## Extension E — True-Random RNG Option (~15 LOC)

**Motivation:** sometimes you want genuinely non-reproducible dice.

**Design:**
```yaml
campaign:
  seed: 42                # seeded (default)
  # or
  randomness: true_random  # non-reproducible, flagged in report
```

`dm_runtime` exposes a factory returning `random.Random(seed)` or `random.SystemRandom()`. Report notes which mode was used (replayability impact).

**Ship anytime.** Trivial.

---

## Extension F — Encounter Merging / Mashup (~180 LOC, speculative)

**Motivation:** dynamic composition — "an ambush during a moral dilemma" = merge two library encounters.

**Defer indefinitely.** Merge semantics (NPC reconciliation, choice union, outcome blending) are genuinely hard and the use case is speculative. Only revisit if users actually request this after Extensions A+B ship.

---

## Extension G — Chained Generation + Execution Pipeline (~50 LOC)

**Depends on:** Extension B (architect persona)

**Design:** `dm_full_pipeline` persona that chains architect → DM runner in one CLI invocation, writing the campaign to a temp path and immediately executing.

**Ship gate:** architect + MVP both stable.

---

## Suggested Ordering (conditional on MVP learnings)

**Most likely path:**
1. MVP ships
2. **Extension E** (true random) — 15 LOC, may as well
3. **Extension A** (library) — if repetition becomes painful
4. **Extension B** (architect) — once A exists to compose from
5. **Extension C** (adaptation) — after collecting MVP metric data
6. **Extension G** (pipeline) — after B ships
7. **Extension D** (isolation) — only if MVP reveals state corruption
8. **Extension F** (merging) — only on explicit user demand

**Total scope if all extensions ship:** ~1,095 LOC on top of the ~560 LOC MVP. But most extensions should *never* ship unless MVP usage demands them.

## Ties to Other Plans

| Plan | Relationship |
|------|-------------|
| [DM MVP](dungeon_master_persona.md) | **Prerequisite** — extensions layer onto MVP |
| [Interactive Simulation Prompts](interactive_sim_prompts.md) | Needed for Extension B |
| [Simulation Entity Naming](sim_entity_naming.md) | Soft win once NPC dialogue volume grows |
| **Realtime Refinement** (core done) | Extension C consumes `InspectAUTTool` |
| **Multi-LLM Scaling** (not started) | Architect + classification use cheap-lane model; synergistic, not required |
