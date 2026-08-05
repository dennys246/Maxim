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

## Amendments (2026-08-04, pre-A/B — two-lens review + the H3 STOP clause)

Recorded BEFORE any LLM-arm data exists. Arm C exploratory data was taken
pre-amendment and triggered amendment 2; it will be re-run under the amended
definitions.

1. **Arm A is a head-only BODY, not an unbound binding.** The two-lens review
   caught that `motor_binding: false` alone leaves the SEM turn tools
   registered with stub-SUCCESS semantics — a placebo arm (the LLM believes it
   turned), which also exempts arm A from the action cap (stub turns emit no
   motion events). Arm A now declares `bodies/reachy_mini_headonly` — the
   bundled body with the `orient` modulator removed (generated at harness
   runtime from the bundled YAML; `entity.name` unchanged so every other tool
   name is identical). The turn tools are genuinely absent, exactly as the
   confounds section claims. `motor_binding: false` is retained as
   belt-and-suspenders.
2. **H3 truth is the FOLDED sensor truth, not unfolded theta.** The first
   arm C run fired the pre-registered STOP clause (overall sign-accuracy 0.42).
   The audit found the honesty gates CLEAN and the metric definition wrong:
   split by fold region, sign-accuracy was 0.89 sub-fold (|θ|≤90°) vs 0.08
   beyond — behind the linear array's fold a CORRECT turn toward the source
   honestly *increases* the folded reading (θ 160°→143° reads as 20°→37°), so
   sensor-faithful credit and unfolded truth have opposite signs there by
   physics. H3 is now scored against the change in noiseless folded
   ``|az_true|`` (what any honest measurement of this sensor can know); the
   unfolded divergence is reported separately as ``credited_fold_divergent`` —
   the sensor's physical blind spot, itself a finding: a credit-following
   learner is genuinely pushed toward the 180° false equilibrium on rear
   sources (present on real hardware too; the far-bin arm C failures are this).
3. **Centered = unfolded truth.** The criterion is |θ| < 9.9° (|az| 0.11 in
   the unfolded frame) sustained 2 truth reads — folded az reads ~0 when
   facing exactly AWAY, so gating on the sensor would score a 180°-wrong head
   as centered. (This was always the harness intent; the original wording
   said "|az|" without naming the frame.)
4. **Arm C operating point (pre-registered here):** substrate import is the
   live runtime's `nac.json` + `ec.json` PAIR (the pair rule — biases key on
   EC node ids); `MAXIM_NAC_MIN_CONFIDENCE=0` (cluster bias is capped ~0.2,
   below the 0.3 default gate — the known clamp-vs-threshold asymmetry; a
   fresh-substrate arm under the default gate is inert by design and was
   recorded as such); `MAXIM_SUBSTRATE_TOOL_WHITELIST=turn_left,turn_right`
   (the probe-3 floor lesson). NOTE the whitelist means C's action space
   excludes `focus_on_sound` while B's includes it — B-vs-C comparisons
   conflate proposer with repertoire and are secondary.
5. **Determinism caveat:** the per-trial seed fixes the noise/speech DRAW
   SEQUENCE, not the trajectory — read counts are wall-clock-paced, so trials
   are statistically controlled, not bit-reproducible.
