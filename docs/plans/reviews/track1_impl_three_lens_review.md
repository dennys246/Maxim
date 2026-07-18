# Three-lens implementation review — Track 1 embodiment-runtime-wiring

**Date:** 2026-07-17. **Lenses:** Architecture, Executor/correctness, Bio-fidelity (three parallel
reviewers). **Reviewed diff:** the three Track-1 implementation commits on
`feat/audio-percept-step3-runtime` (per-iteration tick, declaration-driven body wiring, Layer 3a+3b) —
458 lines across `agent_loop.py`, `hardware/config.py`, `agentic_runtime.py`, `acting_coach.py` + 3 test
files. This reviews the CODE (the earlier round reviewed the plans).

**Unanimous verdict: merge-able, no blockers.** All three confirmed the change is genuinely opt-in
(byte-identical when no `config.body` is declared and the flag is off), the `tick_vital_drift` CI grep is
not tripped, the `build_executor` triad is guarded before the call (bad declaration → bodiless, never a
crash), `.embodiment` resolves through every executor wrapper, and the coach parse matches the real
`format_body_state_for_prompt` output. Findings below were folded in commit following this record.

## Folded

**F1 — `get(robot_id) or get_primary()` could wire a foreign robot's body** *(cross-confirmed: Architecture
#1 + Executor #1)*. On a concrete-id miss in a multi-robot file, the `or get_primary()` fallback silently
adopted the first/primary robot's `config.body`. Naive fix (concrete-miss → bodiless) would break the
DEFAULT single-robot case (runtime name `reachy_mini` ≠ yaml key `primary`, so the id always misses and
the fallback is load-bearing). **Fold:** fall back to primary only when unambiguous — a single robot, or
one explicitly marked `primary`; a genuinely ambiguous multi-robot-no-primary miss → bodiless + warn.
Guard: `TestResolveBodyWiring` (9 cases incl. name/key mismatch preserved, multi-robot-no-primary miss →
bodiless, explicit-primary miss → primary, exact match wins).

**F2 — per-iteration `evaluate_failures()` trips the deferred transition-based-drive-pain trigger + changes
Exp 44 cadence** *(cross-confirmed: Bio SF-1 + Architecture NH-3 + Executor #2 — highest consensus)*. The
continuous tick makes drive-pain state-based (re-fires per standing breach) instead of onset-based, which
is verbatim the revival trigger in `deferred/transition_based_drive_pain.md` ("before any change to
evaluate_failures cadence"). Dampened to **valence noise, not false causal links** (drift tick discards its
direct FailureEvents → only PainBus fires; PainBus refractory caps ~2 Hz; `_context_similarity` mismatch
prevents linking), and **latent for the shipped reachy body** (azimuth is world-set `drift_rate 0` at
`initial 0` → no breach until DoA is fed in Track 2). **Fold (documentation, not suppression — the cadence
change is the intended fix):** a CADENCE CAVEAT in the `tick_embodiment_drift` docstring; the deferred plan
marked **⚠️ REVIVE TRIGGER FIRED** and moved to the near path (land before Track 2 feeds azimuth, before a
self-drifting body is declared, or before Exp 44 numbers are re-baselined); CLAUDE.md invariant updated.
**Exp 44 embodied llm-primary numbers need re-validation against the new cadence.**

**F3 — tests covered leaf helpers, not the production wiring** *(cross-confirmed: Architecture SF-2 +
Executor #3)*. **Fold:** `TestResolveBodyWiring` exercises `_resolve_body_wiring` itself (gating, warning,
fallback, bodiless, success-passes-ref); `test_roundtrip_from_real_embodiment` drives a real `Embodiment`
through `format_body_state_for_prompt` into `_compose_drive_modulation` so a reword of the body.py
descriptor literals can't silently break the coach parse.

**F4 — `descriptor == "rising"` exact-match fragility** *(Executor #4)*. **Fold:** `"rising" in descriptor`,
consistent with the other two branches (which already tolerate the trailing `intensity`/`discomfort`
numbers). Guard: `test_rising_with_trailing_suffix_still_reported`.

**Bio SF-2/SF-3 — azimuth pain mis-scale + homeostatic-shape borrow** *(Bio, single-lens, latent)*. `pain_scale
1.0` makes a fully off-center sound louder in the NAc pain channel than genuine noxious modes (thermal 0.4,
camera 0.6) — an inversion; and modeling an exteroceptive bearing as a homeostatic drive is a mild category
error. Latent (azimuth world-set at 0 until DoA is fed). **Fold (document + defer to Track 2):** a BIO NOTE +
`TODO(Track 2)` in `reachy_mini.yaml` — revisit `pain_scale` (toward 0.2-0.3) and re-measure the orient loop
when wiring the azimuth feed; downstream consumers must not assume interoceptive self-return semantics.

## Endorsed (not findings)

- The coach action-neutrality rewrite is **biologically correct** (restores the afferent/efferent split — the
  body reports state, the cortex/LLM selects the action), not a loss *(Bio)*.
- Loop placement (after pause/stop/shutdown early-exits, before the idle gate) is correct: freeze-states skip
  the tick, a sitting robot still drifts *(Architecture, verified line-by-line)*.
- No CI-grep violation, no frozen-dataclass change, no new env-var tunable *(all three)*.

## Tracked (not folded this round)

- **Layer 3a flag coupling** *(Architecture NH-4)*: the live-robot `body_state` prompt is gated behind the
  Exp 44 ablation flag `MAXIM_ENABLE_BODY_STATE_PROMPT`. Consistent with the AgentFactory seam, but when
  `body_state` graduates from experiment to feature, BOTH seams (`_maybe_wire_body_state` and the Reachy
  runtime wiring) must promote/rename the gate together or the live path silently stays inert. Tracked in the
  Track 1 plan.
- **transition_based_drive_pain.md** implementation itself — its trigger fired (F2); it is now near-path but
  is its own scoped change, not part of this branch.
