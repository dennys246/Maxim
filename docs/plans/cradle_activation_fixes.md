# Cradle / Exp 37 — Empirical Reframing

**Status:** revised 2026-06-04 after [PE_DIAG] instrumented smoke
**Blocks:** Exp 37 cross-session graduation
**Supersedes:** the 2026-06-04 morning draft of this file (which speculated about Bug 3 as the killer; empirically refuted below)

## Empirical evidence that anchored the reframe

A `[PE_DIAG]`-instrumented smoke run from a non-leader machine on
2026-06-04 evening (after architecturally fixing the routing path,
documented in [[project-exp37-blocked-cradle-activation]]) produced this
clean trace from the cradle scenario, fire_pit Arm A, qwen2.5-14b on the
leader:

```
[PE_DIAG] TURN-LOOP-START arc_name='cradle' num_phases=7
         phases_have_world_entities=[True, False, True, False, True, False, False]
[PE_DIAG] PHASE-OBJECT phase_name='exploration'
         world_entities=('items/cradle_fire_pit', 'items/cradle_food', 'items/cradle_cool_air')
[PE_DIAG] INSTANTIATE-OK ref='items/cradle_fire_pit'
         entity=Entity('fire_pit', type='hazard', sensors=['heat_output', 'fuel'], modulators=['flame'])
[PE_DIAG] TOOLS-REGISTERED ref='items/cradle_fire_pit' count=5
[PE_DIAG] RETURN tools_generated=14
```

Subsequent phase transitions activated `items/cradle_sharp_rock` +
`items/cradle_blanket` cleanly when the narrator advanced to phase 2.

## Findings, in priority order

### Finding A — Bug 3 (phase entity activation) is REFUTED

The earlier draft of this plan listed three speculative mechanisms for why
`_activate_phase_entities` might silently fail. The [PE_DIAG] trace
falsifies ALL three:

| Mechanism | Status |
|---|---|
| Narrator never advances phase_idx | DEAD — went 0→1→2→3 cleanly across 12 turns |
| Arc.phases deserialization drops world_entities | DEAD — `('items/cradle_fire_pit', ...)` came through intact |
| Silent exception in registry.instantiate or imagination fallback | DEAD — `INSTANTIATE-OK` fired with a fully-formed Entity |

The cradle's curated entities **are** loading correctly. The fire_pit
entity is registered with `sensors=['heat_output', 'fuel']`,
`modulators=['flame']`, and 5 tools (including the thermal-contact
affordances Exp 37 measures). The earlier evidence (orchestrator
"Manifest pre-trigger" extracting `stone_pit` / `teddy_bear` /
`wooden_door`) was the orchestrator's PARALLEL imagined entity
population — additive to the cradle entities, not replacing them. The
prior actions.jsonl showed `fire_pit_observe` / `fire_pit_touch` in
the available-tools list; I missed that detail and over-indexed on the
scene-description phrases.

### Finding B — Bug 2 is the ACTUAL primary blocker

Confirmed: ~60% of all cradle turns burn on
`Tool not registered: 'respond'. Did you mean: infant_humanoid_respond?`
errors plus parallel `think` failures (`Missing required input: thought`)
plus the cascading `_llm_unavailable` fallbacks they trigger.

Smoke (2026-06-04 evening, clean routing, fresh laptop run):
```
✗ respond: 8
✗ _llm_unavailable: 5
✓ infant_humanoid_use: 2
✗ think: 2
✓ sense_food_source: 1
✓ fire_pit_observe: 1
✓ blanket_touch: 1
✓ read_food_source_freshness: 1
```

6 successful tool calls out of 12 turns. The remaining 10 wasted on
tool-name mismatches.

The mismatch chain:
- `prompt_builder.py` has 20+ hardcoded `'respond'` references telling LLM to use it
- `runtime/generative_runner.py:365-372` deregisters `respond` and `say` for embodied arcs (forces use of body-prefixed `infant_humanoid_respond`)
- Net: prompt instruction contradicts registry state, agent loops on the wrong name

The "respond" deregister was originally added as a 14B-specific
workaround. It now needs a real fix at the prompt-builder layer — the
deregister should NOT exist if prompt_builder produces tool-aware text.

### Finding C — NEW: LLM priors override "infant" persona

Even with 6 working tool calls, the agent's behavioral choice was:

- `blanket_touch ✓` — touched the SAFE object
- `fire_pit_observe ✓` — only OBSERVED the dangerous one, did not touch
- `affordance_preference_safe_count: 1, failed_count: 0, safe_fraction: 1.0` — perfect risk avoidance

This is **textbook cautious adult behavior**. The 14B Qwen brings
strong priors about fire being dangerous and overrides the "infant
who doesn't know fire is hot" persona that the cradle scenario
assumes. Stronger models (Sonnet, the pre-reg's primary) would be
*even more* risk-averse.

The cradle scenario design assumption was that an "infant" agent
would curiously touch the fire, get burned, form aversive memory.
This is the LOAD-BEARING mechanism Exp 37 measures: substrate
transfer of pain-from-fire-touch from Arm A to Arm B. If Arm A's
agent never touches the fire (because LLM priors prevent it), no
aversive substrate forms, no transfer can be measured.

This is NOT a bug. It's an experimental-design problem revealed by
the bug-fix process exposing the actual agent behavior.

### Finding D — Bug 1 (turns_completed AttributeError) is real but cosmetic

`campaign_runner.py:55: result.turns_completed` (should be `result.total_turns`).
Catches as `Generative campaign error` but the trial result is already
written to JSONL. Doesn't block anything; trivial fix.

## Revised fix sequencing

**P0 (mechanical, ship now):** Bug 1 — `turns_completed` rename + regression test that exercises the display path. One-line src fix, ~30 min total.

**P1 (real engineering work):** Bug 2 — reconcile `prompt_builder.py` with the embodied-arc tool registry. Three options:
- (A) `prompt_builder.py` introspects the registry to find the actually-registered speak-equivalent tool; emits `infant_humanoid_respond` for embodied, `respond` for conversational
- (B) Drop the deregister in `generative_runner.py:365-372` and instead use mode-detection (substrate-primary vs conversational) in `prompt_builder.py` to control which tool-discipline instructions get emitted
- (C) `prompt_builder.py` enumerates available tools dynamically and just lists them, dropping all the hardcoded `'respond'` / `'think'` / `'memory_recall'` instructional text

Option (B) probably cleanest — substrate-primary mode is already a flag
in the runtime, just needs to gate the conversational-tool instructions.

Estimated 3-5 hours including tests + cradle integration smoke.

**P2 (experimental design refinement, USE existing bio-machinery):**
Chosen path 2026-06-04: leverage the homeostatic drive system the
cradle already has, instead of fighting the LLM's adult priors. The
infant_humanoid body already declares a `core_temperature` drive
(set_point 0.0, comfort_band 0.4, starts at -0.15) and the phase-0
world includes a `cool_air` entity. The bio-mechanism for natural
fire-seeking behavior is already wired — the existing Exp 37 design
just wasn't tuned to activate it.

Mechanism (no code structure changes, just calibration):
- Cool air drops `core_temperature` further into discomfort zone
- Drive-pain accumulates when out of comfort band → motivates seeking warmth
- Agent moves toward fire_pit for thermal relief (overrides the
  "fire is dangerous" prior because internal drive is louder)
- Proximity warming feels good → reinforces approach
- Touching the flames directly still triggers thermal_contact
  failure mode → forms aversive memory specifically for `fire_pit_touch`
- Substrate transfer to Arm B: agent inherits *both* the
  approach-to-warm association (positive) and the don't-touch
  association (negative). Discrimination between safe-proximity and
  damaging-contact is the substrate signal Exp 37 measures.

Implementation tweaks needed in `_data/components/`:
- `bodies/infant_humanoid.yaml`: possibly lower initial core_temperature
  or tighten comfort_band so drive-pain triggers sooner
- `items/cradle_cool_air.yaml`: verify the affordance actually drops
  body core_temperature (not just simulates wind on skin)
- `items/cradle_fire_pit.yaml`: distinguish `warm_self` (proximity
  affordance, positive thermal sensor read, no pain) from `touch_fire`
  / `pick_up` (contact affordance, triggers thermal_contact failure
  mode). May already be the case — needs audit.

Experimental metric becomes richer:
- Old (still primary): `failure_class_action_count = count(touch(fire_pit) | pick_up(fire_pit))` — counts the AVERSIVE contact
- New (corroborating, NOT replacement): `fire_approach_action_count = count(any affordance that produces warming on fire_pit)` — counts the APPROACH that the substrate should encourage in Arm B vs more hesitancy on Arm A

Hypothesis on Arm B vs Arm A:
- Approach count should be UNCHANGED or HIGHER on Arm B
  (substrate transferred "fire = warm" positive memory)
- Touch count should be LOWER on Arm B (substrate transferred
  "touch = pain" aversive memory)

This is a richer measurement than the pre-reg's binary, and it
exercises both the positive and negative substrate edges — a stronger
test of the substrate-transfer claim.

Pre-reg amendment: keep `failure_class_action_count` as the primary
metric (variance-survival rule unchanged), add the
`fire_approach_action_count` as a corroborating signal. Document the
homeostatic-drive activation in the protocol so the scenario
calibration is reproducible.

Scope: small. Most of the YAML and bio-machinery exists. The work is
calibration tweaks + analyzer additions + one pre-reg amendment.

**P3 (still real but lowest priority):** the orchestrator's parallel
imagined entities (`stone_pit`, `teddy_bear`, `wooden_door`) running
alongside curated entities. Not blocking Exp 37 (cradle entities load
correctly), but a separate question of whether the imagined-entity
overlay confuses the agent or just adds noise.

## Architecture lesson captured

The 4-hour 530 debugging odyssey was downstream of running the harness
on the same machine as the leader. Sub-sims spawn local llama-cpps,
collide with leader's llama-cpp on port 8100, cascade through
proxy-on-8099 → cloudflared-as-leader-child. Empirically resolved by
running harness on a DIFFERENT machine: the laptop has its own peer.yml,
sub-sims auto-detect peer correctly, send HTTPS through the tunnel, no
collision. See [[project-exp37-blocked-cradle-activation]] for the full
cascade analysis. Add to CLAUDE.md "Lessons learned" once we have the
landing patch.

## Carryforward state

- `diag/pe-instrumentation` branch pushed (not for merge). Revert before
  next session OR cherry-pick the harness `--language-model` drop fix
  separately and toss the rest.
- `cradle_activation_fixes.md` (this file) replaces the morning draft.
- `project_exp37_blocked_cradle_activation.md` memory needs revising
  to match findings A/B/C/D above.
- Pre-reg PR #304 may need amendment depending on P2 choice; defer
  until after P1 ships and we have a real measurement.

## Suggested PR sequencing

1. **PR A — revert PE_DIAG instrumentation** (~5 min, removes
   debug-level warnings; or just delete `diag/pe-instrumentation`
   branch and start clean)
2. **PR B — keep the harness `--language-model` drop** (uncommitted
   on main checkout; small, real fix, ship independently)
3. **PR C — Bug 1 `turns_completed` rename + regression test** (P0)
4. **PR D — Bug 2 prompt_builder + generative_runner reconciliation**
   (P1, largest)
5. **Decision: P2 scenario-design path** — discuss with the user
   before any code, may need pre-reg amendment
