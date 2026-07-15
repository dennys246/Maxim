# Controlled LLM-primary embodied-choice harness (Exp 44 vehicle)

**Status:** PLANNED (2026-07-15). Supersedes the "use the fixture
orchestrator" idea — the fixture path is the wrong vehicle (below). This is
the real set of changes to make the generative `cradle_pref` route a
*controlled* multi-turn safe-vs-harm embodied experiment for an LLM-primary
agent-under-test.

## Why NOT the fixture orchestrator (settled)

`FixtureDrivenOrchestrator` (S1) is a **scripted percept-replay** harness:
narrator-free, LLM-optional, feeds a FIXED YAML percept sequence and snapshots
substrate state. It runs a live AUT loop, but:
- **No reactive world** — percept N+1 is fixed YAML regardless of the AUT's
  turn-N action (`fixture_orchestrator.py:206`).
- **No scene-affordance registration** — `warmth_*` are never callable; the
  `generate_tools_for_entity(world_entities)` machinery lives only in the
  generative branch. Fixture YAML has no scene-entity field.
- Making it host the choice = re-implementing the generative scene machinery
  inside a stimulus-replay schema, while the pieces it would reuse (AUT loop,
  `actions.jsonl`, Exp 42 analyzer) are already reusable from the generative
  path. Net-negative.

## The vehicle that already exists

The generative `cradle_pref_a`/`cradle_pref_b` arcs (`arcs.py:496-526`) each
declare exactly the dilemma (`world_entities: [warmth_alpha_harm,
warmth_beta_safe]`, swapped for arm B), with **empty phase instructions**.
`scripts/benchmark_exp42_preference.py` already has `--aut-mode llm-primary`,
`--aut-model`, arm-env recording, and the mode-agnostic `actions.jsonl`
metric. `cradle_pref_a` resolves as an arc → `generative=True`. Everything
routes correctly. The controlled-experiment work is a small set of gaps, not
a new harness.

## The gaps, in priority order

### G1 — LLM-primary scene harm is non-deterministic (BLOCKING for validity)

**The crux.** `orchestrator.py:1640`:
`_gen_embodiment = _aut_instance.embodiment if aut_mode == "substrate-primary" else None`.
In LLM-primary, scene entities register with `embodiment=None`, so
`warmth_alpha_harm`'s `self_effect` (arms.thermal +0.6 harm) does NOT
deterministically write to the body — harm is expected via the
"narrator-driven Layer-2 proximity path" (comment `orchestrator.py:1633-1636`).
So the safe-vs-harm *signal* the whole experiment measures depends on the
orchestrator LLM improvising it. An AUT can warm at the harmful source and
feel nothing → cannot learn safe > harm → the metric degenerates. This is
the FLOOR under the imagination-noise problem, and likely why earlier runs
showed `warmth_*_observe` but no real safe-vs-harm choice.

**Constraint:** the `embodiment=None` in LLM-primary is DELIBERATE — the
comment warns threading it "would double-count and change Exp 37/38" (harm
would arrive both via `self_effect` AND the narrator-proximity path). So the
fix must be **gated to Exp 44** and validated not to regress 37/38.

**Design decision (pre-register before implementing):**
- Option A — **deterministic self_effect harm, narrator-proximity OFF** for
  the controlled run: thread `embodiment=_aut_instance.embodiment` into
  scene-entity activation for LLM-primary too, AND suppress the
  narrator-driven Layer-2 proximity writes so harm arrives via exactly one
  channel (`self_effect`). This is the *controlled* choice — deterministic,
  attributable, no double-count. Localized to `orchestrator.py:1640-1641` +
  gating the Layer-2 proximity path + a validation that Exp 37/38 (which
  rely on the narrator path) are untouched because they run a different
  arc/flag.
- Option B — accept narrator-mediated harm (status quo). Rejected: it is the
  exact improvisation the controlled experiment must remove.

Prefer A behind an Exp-44 gate (e.g. an arc/flag that says "this is a
controlled deterministic-embodiment run"). The delta-attribution scope note
(B8, `tool_bridge.py:430-437`) and the pending `transition_based_drive_pain.md`
work both touch how this harm attributes to NAc — coordinate.

### G2 — LLM-primary lacks the salience/drive-gating filter (VALIDITY)

Substrate-primary uses NAc drive-gating (`MAXIM_SIM_DRIVE_GATE_ENABLED=1`) so
a cold body makes warming the only rewarding act, and reward-irrelevant tools
are never selected. LLM-primary hands the AUT every tool with no equivalent
salience filter → a cautious 32B fritters on `respond`/`sense`. Even with a
clean scene (imagination gated) and deterministic harm (G1), the AUT may not
prioritize warming. Decide: (a) an introspection-tool filter for LLM-primary
(mirror the substrate-primary `INTROSPECTION_TOOL_NAMES` exclusion in
`propose_via_substrate`), and/or (b) rely on the body_state prompt (arm C —
the whole point of Exp 44) to supply the urgency. Note the circularity: G2 is
partly what Exp 44 is *testing* (does body_state make the agent prioritize
warming), so over-filtering could mask the effect. Pre-register the filter
decision so it doesn't confound the arms.

### G3 — narrator prose (LOWER — probably tolerable)

Empty phase instructions remove narrator *direction* but not *prose volume*:
`NARRATOR_SYSTEM_GENERATION` (`narrator.py:77-94`) hard-codes "vivid,
immersive, 3-5 sentences." There is no terse-narrator knob. The Exp 44
pre-reg already manages the narrator-state confound by pinning the narrator
snapshot across arms, so this may not need a fix. IF per-turn atmospheric
prose (dungeon framing) is judged to derail behavior beyond what
snapshot-pinning controls, add a one-branch terse mode to `narrator.py`
(skip generation / emit a fixed line) — far cheaper than any harness change.

## Already shipped (this session)

- **Imagination fully gated** — `MAXIM_DISABLE_IMAGINATION` universal switch
  (constructs `ImaginationTrigger(enabled=False)`; both `process_percept` and
  `process_manifest` respect it). Kills the world-builder manifest + per-turn
  entity design. Controlled arcs now present only their declared entities.
- **n_ctx single-source** standard (`maxim config set llm.n_ctx`) — fixes the
  budgeter-vs-server 500s.

## Sequencing

1. **Confirm the gated seed** (imagination off, running now): does the AUT
   engage warmth affordances at all, and — critically — does calling
   `warmth_alpha_harm` produce ANY harm signal? If harm is absent, that's G1
   confirmed live (the `embodiment=None` floor), regardless of the Tool Usage
   headline.
2. **G1 first** — it's the validity floor; no arm result means anything until
   harm is deterministic. Pre-register Option A, gate to Exp 44, validate
   37/38 untouched, one seed to confirm safe-vs-harm learning appears.
3. **G2** — pre-register the salience-filter decision (careful re: confounding
   the arms).
4. **G3** — only if prose is shown to derail beyond snapshot-pinning.
5. Then run the arms (A/B/C) per `acting_coach_body_state_ablation.md`.

## Honest note on scope / divergence

Exp 44 has surfaced a five-layer stack of setup issues (model → n_ctx → scene
distractors → per-turn imagination → non-deterministic harm), each found only
after fixing the last. This is the CLAUDE.md "divergence → step back" pattern.
The through-line: **the whole Exp 42 apparatus (harness, scene, metric, harm
model) was built for substrate-primary; LLM-primary is not a drop-in.** This
plan is the deliberate accounting of what LLM-primary actually needs. Worth
deciding explicitly, before more seeds, whether the LLM-primary body_state
question is worth this build-out now, or whether it waits behind a purpose-
built LLM-primary embodied harness.

## Non-goals

- Not building out the fixture orchestrator (settled: wrong vehicle).
- Not changing Exp 37/38's narrator-mediated-harm path (G1 is Exp-44-gated).
- Not the SEM environmental-proximity redesign
  (`sem_environmental_proximity_sensing.md`) — orthogonal; that grounds the
  cold *drive*, this fixes scene *harm* determinism + salience.
