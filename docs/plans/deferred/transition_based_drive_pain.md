> **PHASES 0-1 SHIPPED (2026-07-28, `feat/transition-drive-pain`, PR #435)** as Stage 0a of
> [live_audio_orient_wiring.md](../live_audio_orient_wiring.md).
>
> **SHIPPED SHAPE DIFFERS FROM THIS PLAN — read this before Phase 3.** The plan proposed
> latching BOTH attribution channels and then retiring B8 (Phase 3). A pre-merge two-lens
> round **disproved that design**: both lenses independently reproduced, on the real Exp 42
> fixtures, that latching the direct channel starves B8 (which is state-INDEPENDENT and
> therefore the only thing that still separates causer from bystander at sensor
> **saturation**), so every *repeat* harmful affordance emitted no `embodiment_failures`
> → `learn_success` flipped True → the collateral-harm gate went silent → the harmful
> hearth booked **positive** cluster reward. What shipped instead:
> - **Channel 1 (returned `FailureEvent`s): unchanged, state-based.** B8 filters it and
>   **stays load-bearing — Phase 3's "remove B8" premise is retracted.**
> - **Channel 2 (`_publish_drive_pain` → PainBus): severity-latched** — fires on band
>   entry and on material re-injury (bio-faithful sensitization, not silence), with
>   hysteresis on recovery so a noisy world-set sensor at the band edge cannot chatter.
> - **Latch owner: `Entity.drive_breach_severity`**, not the `Embodiment` wrapper — fixes
>   ephemeral per-invocation wrappers, reparenting, same-name siblings, and key leaks.
>
> Regression guards: `tests/unit/test_transition_drive_pain.py` (19) +
> `test_substrate_primary_scene_harm.py::test_execute_delta_attribution_causing_vs_bystander_on_chilled_body`
> (repeat-causer arm — verified to FAIL on the latched-direct-channel design). Phase 2
> unit/substrate blast radius green (161 tests). Phase 4 (CLAUDE.md invariant) shipped.
>
> **VALIDATION STATUS (2026-07-28):**
> - **Exp 42 triage re-run: DONE — GRADUATE #6 HOLDS (2026-07-29).** 40 sub-sims, 0 failed,
>   0 floored; H1 0.996/1.000, C1 +0.996, C2 PASS in BOTH configurations, with the gating
>   toggle verified fired (`env_drive_gate_enabled` 1 vs 0) and `executed_git_hash` pinned
>   to the fold in every record. Post-fold is **identical** to the pre-fold main baseline —
>   the predicted result, since Exp 42 rides the direct channel the fold deliberately left
>   state-based. Caveat recorded: `safe_pref` is saturated (SD 0.000 across all four
>   conditions), so this supports "did not break discrimination" but is weak evidence for
>   "changed nothing". Full writeup:
>   [docs/experiments/42b_drive_pain_fold_revalidation.md](../../experiments/42b_drive_pain_fold_revalidation.md).
>   (A 2026-07-28 attempt was retracted as invalid — wrong checkout; its numbers serve as
>   the main-side baseline.)
> - **SCN oscillator cold-start floor: CLOSED — concern VOID (2026-07-29).** Measured on a
>   real substrate-primary cradle run: `aut_scn.json` carries 10 event signatures and
>   **zero** `drive:*` ones. Cause is not density but wiring — `Body._emit_drive_temporal_event`
>   early-returns on `self._distributor is None`, and `build_executor` gives its
>   `distributor=` to `ToolPainBridge` (bootstrap.py:439) but not to `Embodiment`
>   (bootstrap.py:456). **No drive TemporalEvent has ever reached the oscillator in any
>   production run**, so this fold cannot have changed drive-phase density. The one-line
>   wire is deliberately NOT applied here (it would activate a dead learning path — its own
>   change, its own validation); recorded as dormant infrastructure in CLAUDE.md.

> **PHASES 0-1 SHIPPED (2026-07-28, `feat/transition-drive-pain`)** as Stage 0a of
> [live_audio_orient_wiring.md](../live_audio_orient_wiring.md) (the Track-2 live-azimuth
> wiring is the trigger that fired). The breach latch lives on
> `Embodiment.__init__::_drive_breach`; both channels now fire on band entry only.
> Regression guard: `tests/unit/test_transition_drive_pain.py` (5 transition assertions
> verified failing on the pre-fix emitter). Phase 2 unit/substrate blast radius is green
> (142 tests); **OUTSTANDING: the Exp 42 triage re-run** (both arms, confirming GRADUATE
> holds without leaning on B8) — which then gates **Phase 3 (B8 disposition)**; B8 is kept
> as belt-and-suspenders until that run lands. Phase 4 (CLAUDE.md invariant) shipped in
> the same commit.

> **DEFERRED (2026-07-15 plans audit):** Not shipped — `Body.evaluate_failures` is still state-based (re-fires per tick, no breach latch); B8 delta-attribution remains the only patch covering channel 1. Correct, well-scoped root-cause work (~25–40 LOC + blast-radius validation) but off the Exp 44 critical path, and touching shared embodiment code mid-experiment is the wrong moment. **Revive when:** a second drive-pain-attribution consumer appears, channel-2 (PainBus) mis-attribution actually bites an experiment (e.g. a safe source accruing spurious negative like pre-B8 Exp 42), or before any change to `evaluate_failures` cadence.
>
> **⚠️ REVIVE TRIGGER FIRED (2026-07-17):** Track 1 of `embodiment_runtime_wiring.md` added a **per-live-loop-iteration** `evaluate_failures()` call (`agent_loop.py::tick_embodiment_drift`) so the llm-primary body drifts instead of freezing — i.e. it *changed the `evaluate_failures` cadence*, the exact trigger named above. The three-lens implementation review cross-confirmed this (bio-fidelity SF-1, architecture NH-3, executor #2). It was shipped as a should-fix, not a blocker, because the flood is dampened to **valence noise, not false causal links** (the drift tick discards its direct FailureEvents so only the PainBus channel fires; the PainBus refractory caps it to ~2 Hz; the `_context_similarity` denominator mismatch keeps it from linking to actions), AND it is **latent for the shipped reachy body** (its only drive, azimuth, is world-set with `drift_rate 0` at `initial 0` → centered, no breach, until DoA is fed in Track 2). **This plan is now on the near path**, not the far one: it should land before (a) Track 2 feeds live azimuth into the body, or (b) a body with self-drifting drives is declared for a live/opp-in run, or (c) Exp 44 embodied llm-primary numbers are re-baselined against the new cadence.


**Status:** Draft. Written 2026-06-23 from the Exp 42 PR #380 two-lens review (the Architecture-lens A-CRIT1 + the cross-confirmed root-cause finding). Exp 42 shipped a delta-attribution *filter* (B8) at the affordance layer as a scoped unblocker; this plan addresses the underlying cause so the fix lands once, at the source, for both attribution channels — instead of being re-patched per consumer.

## Front-gate scope pressure

This is **not** a new mechanism — it modifies the existing `Body.evaluate_failures` so its drive-pain emission is transition-based rather than state-based. No new bus, bridge, builder, or config knob. It rides entirely on the existing `FailureEvent` / `PainBus` surfaces. The bar is therefore "is this the right behavior for the existing emitter," not "should this exist."

## Why this plan exists

`embodiment/body.py::Body.evaluate_failures` re-evaluates every drive spec on **every call** and, for any sensor currently out of band, **re-fires** a `FailureEvent` *and* calls `_publish_drive_pain` ([body.py:242-277](../../src/maxim/embodiment/body.py)). It has no memory of the prior breach state, so a sensor that stays out of band emits a fresh pain event **every tick** it remains breached.

`evaluate_failures` is called from two places per tick in the embodied loop — the per-tick poll (`propose_via_substrate` / `EmbodimentPerceptSource.next_percept`) and per-affordance (`ModulatorAffordanceTool.execute`). Both see the same lingering breach, so the pain fans out to **two attribution channels**, neither of which can tell *which action caused this tick's breach*:

1. **Direct channel** — the returned `FailureEvent`s become `ToolOutput.side_effects["embodiment_failures"]`, which `runtime/executor.py` routes to `ToolPainBridge.record_tool_embodiment_failure` → NAc negative on the **executing tool**.
2. **PainBus channel** — `_publish_drive_pain` → `PainBus` → `create_pain_nac_subscriber`, which attributes via `_context_similarity` to a pending action event.

Once a harmful affordance pushes a sensor out of band, the breach lingers (slow homeostatic recovery), so **every subsequent action** executing during the breach is blamed for harm it didn't cause. In Exp 42 this collapsed the safe-vs-harm discrimination (the safe warmth source accrued `neg ≈ 0.96`, identical to the harmful one).

**B8 (Exp 42) patched channel 1 only**, with a delta heuristic ("attribute a drive failure to an affordance only if its own delta would breach a healthy sensor"). That unblocked the experiment, but:

- It leaves **channel 2 (PainBus) unfiltered** — it happens not to re-pollute in Exp 42 (safe net stayed `+0.99`), but the correctness rests on undocumented `_context_similarity` scores, exactly the fragile coupling the P2 `_context_similarity` lessons warn about.
- It is a **delta-only heuristic**: an affordance whose delta is below the band but that tips an already-near-breach sensor over is a genuine partial cause yet is spared (false-negative attribution → lost learning signal).
- It is a **per-consumer patch** of a single-source defect. The next consumer of drive-pain inherits the same mis-attribution.

The root cause is that drive-pain is **state-based** ("is breached now") rather than **transition-based** ("just entered breach"). Fixing it at the emitter fixes both channels and removes the need for the per-affordance heuristic.

## The fix

Make `Body.evaluate_failures` emit a drive `FailureEvent` + `_publish_drive_pain` **on band entry only** (within-band → out-of-band transition), latched per `(entity_path, drive_name)` and cleared on the reverse transition.

- A breach is fired on the **tick/affordance-execute that causes the crossing** → attributable to the causing action. The harmful affordance's `execute` calls `evaluate_failures` immediately after applying its `self_effect`, so the crossing (and the pain) fire inside *its* execute → direct-channel attribution lands on the harmful tool. A bystander affordance executing later sees no transition (already latched out) → emits nothing → is not blamed. **Both channels** become clean at the source.
- Re-injury is still captured: if the sensor recovers within band and is pushed out again, that is a new transition and fires again.

### Onset pain vs sustained motivation — the key distinction

A cold agent *is* in ongoing discomfort, so "fire once on entry" must not lose the motivational signal. It doesn't, because the two signals live in different places:

- **Motivation** (the agent should keep acting on the need) is carried by the **drive value itself** — `_read_drive_states` reads `Entity.vital_metrics` / modulator sub-sensors directly, and the acting-coach / drive-affinity heuristic read the drive value, **not** the `FailureEvent`. Onset-only emission does not touch this.
- **Learning attribution** (which action caused harm → NAc causal link) is what the `FailureEvent` / PainBus pain feeds. This is precisely the signal that should be onset/cause-based, not re-fired every tick.

So transition-based emission keeps motivation intact while making attribution correct. (If a future consumer genuinely needs a throttled *sustained*-discomfort signal, add it as a distinct, clearly-named event at a deliberate cadence — do not revert to per-tick re-fire of the onset event.)

## Phases

**Phase 0 — characterize (pin current behavior).** Add a test reproducing the per-tick re-fire and the bystander pollution on *both* channels (the gap A-CRIT1 named): a body with a homeostatic sub-sensor, one harmful + one gentle affordance sharing it, assert that pre-fix the gentle affordance's NAc attribution inherits the lingering breach via the PainBus subscriber path.

**Phase 1 — transition latch.** Add a per-`(entity_path, drive_name)` breach-state set on `Body`; in `evaluate_failures`, fire the drive `FailureEvent` + `_publish_drive_pain` only on within→out transition, clear on out→within. Applies uniformly to homeostatic discomfort and entropic deprivation. The latch must be consistent across the poll and execute call sites (single state on the `Body`).

**Phase 2 — validate blast radius.** This is shared embodiment code that Exp 37/38 and the SEM pain cascade depend on. Run `tests/substrate/test_sem_pain_cascade.py`, `tests/unit/test_self_effect.py`, the embodiment suites, and re-run the **Exp 42 triage** (both arms) — confirm GRADUATE holds *without* leaning on B8's channel-1 heuristic, and confirm Exp 37/38 behavior is unchanged (a smoke run if the harness is available). Audit every consumer of drive `FailureEvent` / drive PainBus signals for an assumption of per-tick re-fire (e.g. escalating-pain logic).

**Phase 3 — disposition B8.** Once transition-based attribution is validated, B8's delta-only filter in `ModulatorAffordanceTool.execute` is **redundant for the direct channel** (the causing-tick already fires only on the causer). Decide: remove the delta heuristic (simplify) while **keeping** the unfiltered-fallback + error-logging structure from the review fold, or keep it as documented belt-and-suspenders. Either way, drop the "two-channel scope" caveat from the docstrings once both channels are clean.

**Phase 4 — record the invariant.** Add an `[engineering]` CLAUDE.md invariant ("drive-pain is transition-based; emitted on band entry, not per-tick") with the Phase-0/1 tests as the `Regression guard:`. Update `docs/user/tool_side_effects.md` (`embodiment_failures` no longer needs the delta-attribution caveat if B8 is removed).

## Risks

- **Exp 37/38 depend on drive-pain firing.** If onset-only changes the pain *cadence* they see, behavior could shift. Mitigate via Phase 2 validation (cascade tests + a harness smoke) before merge.
- **A consumer relies on per-tick re-fire** (escalation / repeated reinforcement). Phase 2 audit; if found, give it the distinct throttled sustained signal rather than reverting.
- **Latch state + persistence.** The breach-state set is session-runtime only (not persisted); confirm it resets correctly on body re-instantiation and doesn't leak across agents (per-`Body` instance, keyed by entity path).
- **Timing vs the drift-before-check.** `evaluate_failures` applies wall-clock drift before the band check ([body.py:145-154](../../src/maxim/embodiment/body.py)); the latch must be evaluated against the post-drift value so a sensor that drifts back within band correctly clears, enabling a later genuine re-breach.

## What this does NOT do

- Does not change the drive *value* dynamics (`tick_vital_drift`), the affinity heuristic, or `_read_drive_states` — motivation is untouched.
- Does not change standard (non-drive) `failure_mode` evaluation — those already fire on their own trigger semantics and are out of scope.
- Does not add a config knob — transition-based is the correct default, not an option.

## Sizing

Small-to-moderate: ~1 state field + transition logic in `evaluate_failures` (~25-40 LOC), Phase-0/1 tests (~6-10), Phase-2 validation (re-run existing suites + Exp 42 triage), optional B8 removal (net **negative** LOC). The cost is in **validation discipline** (shared path → Exp 37/38 + cascade), not in code volume.
