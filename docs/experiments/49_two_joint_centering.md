# Exp 49 — Two-joint centering: does the runtime USE the body-turn capability appropriately?

**Status:** PRE-REGISTERED (draft pending owner-visible commit). Phase 3 Arm 1 of
[sem_motor_binding.md](../plans/sem_motor_binding.md); owner-designed scenario
(2026-08-04): "a simulation that requires both a combination of body movement
and head movement to center on a sound for reward."

## Hypothesis

**H1 (capability composition):** with sound sources at bearings BEYOND the neck's
reach (±40–160°), an agent with motor-bound body-turn affordances centers the
source; a head-only agent (no SEM motor binding — focus_on_sound alone) plateaus
at the neck limit and cannot.

**H2 (usage appropriateness, LLM-primary):** given the honest fell-short/clamped
feedback, the LLM composes focus_on_sound + the correct-DIRECTION turn tool.
Pre-registered metric: direction-correct first body turn per trial (the
2026-08-04 live session showed wrong-direction turns under sparse speech —
this arm measures it under dense speech with the full feedback loop).

**H3 (measured credit, secondary):** with dense synthetic speech, the Phase-2
measured relief credit books non-zero cluster rewards whose SIGN matches the
turn's true direction-progress in ≥95% of credited turns (the honesty gates
should make wrong-sign credit rare-to-impossible in a fixed-source world).

## Harness (honest physics — the non-negotiables)

Runs the FULL live stack — agentic runtime, DoA feed, focus_on_sound, SEM motor
binding, Phase-2 measurement — against `SimulatedController` (which honors the
world-frame pose contract) plus a synthetic `DoAReader`:

- Source at FIXED world bearing S per trial (never teleports mid-trial).
- Reader computes the head-relative offset from the controller's CURRENT world
  head yaw (head world = body + head-relative, the frame contract), applies the
  LINEAR-ARRAY FOLD honestly (a rear source reads as its front mirror), inverts
  `doa_to_azimuth` so the PRODUCTION mapping code runs, and adds Gaussian noise
  (σ from the live DoA characterization, ~0.03 az).
- Speech density: dense (every read speech-gated true) for the main arms —
  the live-conversation sparse case is what Exp 49 deliberately removes as a
  confound; a sparse arm may follow.
- Neck envelope clamped at the measured ~±22° (not the optimistic ±45).

## Arms

| Arm | Motor binding | Proposer | Trials |
|---|---|---|---|
| A head-only | OFF (factory None) | LLM (Qwen32B via tunnel) | 10 bearings × L/R counterbalanced |
| B full | ON | LLM | same bearings |
| C substrate | ON | substrate-primary (trained policy import if key-compatible; else fresh) | same |

Bearings: ±{40, 60, 90, 120, 160}° (all beyond neck reach). Trial ends on
CENTERED (|az| < 0.11 sustained 2 readings) or 12 actions or 180 s.

## Metrics (pre-registered)

- centering rate per arm (H1 gate: B > A with zero overlap on far bins)
- time-to-center / actions-to-center
- direction-correct first body turn (H2: ≥80% in B)
- credited-turn sign-accuracy (H3: ≥95%)
- body-turn usage rate on far bins (0% expected in A — it has no tool)

## Confounds pre-answered

- Provenance: harness asserts repo interpreter (scripts/_provenance.py pattern)
  and stamps executed_git_hash per trial record.
- No imagination distractors (MAXIM_DISABLE_IMAGINATION=1).
- The LLM sees identical prompts modulo the tool list (arm A simply lacks the
  turn tools — the same mode filter path, not a prompt hack).
- Fixed seed per trial for noise; bearings counterbalanced L/R to cancel any
  residual sign asymmetry.

## What would falsify what

- B ≈ A on far bins → the binding/prompt surface isn't reaching the LLM (wiring
  bug, not cognition) — check the described-tools render first.
- B centers but H2 < 50% → the LLM flails and gets there by feedback random
  walk; the note/description language needs work (or Phase 3's reflex layer is
  the answer, not deliberation).
- H3 < 95% → a measurement honesty gate is leaking; STOP and audit before any
  policy claims (the credit path is upstream of everything).
