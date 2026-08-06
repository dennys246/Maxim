# Exp 49 — Two-joint centering: does the runtime USE the body-turn capability appropriately?

**Status:** COMPLETE (2026-08-04; H3 attribution corrected 2026-08-06). H1 supported, H2 1.00, H3 1.00 (B) / 0.969 (C) — both pass.
Results below the Amendments section. Phase 3 Arm 1 of
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

## Results (2026-08-04, post-amendment runs; harness @ a2f1abef)

10 trials/arm, bearings ±{40,60,90,120,160}°, dense speech, σ=0.03, neck
envelope ±22°, caps 12 actions / 180 s. Arm A/B proposer: Qwen32B via the
big-mac-mini tunnel (~40–90 s per action). Arm C: substrate-primary with the
live runtime's nac.json+ec.json pair imported (one trained bias:
`turn_right` @ +0.101 on one audio cluster), `MAXIM_NAC_MIN_CONFIDENCE=0`.
Per-trial JSONL + provenance in the session scratchpad
(`exp49_arm{A,B,C}_full` / `exp49_armC_v2`); every record carries
`executed_git_hash` + import hashes.

| Metric | A head-only | B full (LLM) | C substrate |
|---|---|---|---|
| centering rate | **0/10** | **4/10** | 5/10 |
| far bins (±120/160) centered | 0/4 | 2/4 (both ±120) | 0/4 |
| body-turn usage | 0.0 | 1.0 | 1.0 |
| first body turn correct (H2) | n/a (no tool) | **10/10 = 1.00** | 5/10 |
| mean time-to-center | — | 86.8 s | 4.65 s |
| credited turns (H3 sign acc.) | 0 | **23/23 = 1.00** | **125/129 = 0.969** (corrected 2026-08-06) |
| credited fold-divergent | 0 | 7 | 77 |
| plateaued at neck limit | 6/10 | 0 | 0 |

- **H1 SUPPORTED.** Head-only cannot do the task: 0/10, zero body turns,
  60% of trials parked at the neck envelope (the others under-aimed at the
  folded image and idled). Motor-bound B centers 4/10 overall and 2/4 far
  bins with zero overlap against A. B's four timeouts are CLOCK-bound, not
  capability-bound: every timeout trial had direction-correct body turns in
  progress at 3–5 actions when the 180 s cap hit (the LLM's per-action
  latency is the binding constraint; the pre-registered cap charitably
  assumed faster deliberation).
- **H2 PASSED at 1.00** (gate ≥0.80): the LLM's first body turn was
  direction-correct in 10/10 trials under dense speech with the honest
  fell-short/clamped feedback — the 2026-08-04 live wrong-direction turns do
  not reproduce once speech is dense and the feedback loop is closed.
- **H3 PASSED on BOTH arms** (gate ≥0.95): arm B 23/23 = 1.00; arm C
  **125/129 = 0.969** after the 2026-08-06 extractor-attribution correction
  (below). The 4 remaining arm C mismatches are pure noise floor: true
  sensor progress ≈ 0.0036 az (0.3°) under σ = 0.03 read noise — an honest
  measurement legitimately sign-flips there. The measurement honesty gates
  survive with zero confirmed leaks.
- **CORRECTION (2026-08-06): the "one-turn-stale capture frame" residual
  hypothesis in the first version of this section is REFUTED.** A published
  0.915 arm C H3 attributed 6 full-magnitude mismatches to a stale mirror
  frame under instantaneous sim turns. Event-level reconstruction showed the
  measured before/after pairs were **frame-clean all along** — every
  credit's `before` chains exactly from the previous credit's `after`. The
  liar was the harness's credit-to-motion matcher: the credit event is
  emitted at the END of its turn's execute, and under back-to-back
  substrate actions the NEXT turn's motion event lands within the 10 ms
  JSONL timestamp rounding — a `<=` matcher attributed those credits to the
  wrong turn and scored honest negative credit (a turn genuinely away from
  center) as sign-flipped. Fixed to strictly-before matching (a credit can
  never tie with its OWN motion — the ≥0.3 s measurement settle guarantees
  separation; rounding is monotonic so the next motion never rounds below
  its predecessor's credit). Pinned by
  `test_credit_tied_with_next_motion_attributes_to_own_turn`. Consequence:
  **hardware H3 claims are no longer instrument-blocked** — the previously
  planned "stale-frame fix" in the backend gates is unnecessary; no
  production code changed.
- **Secondary finding — the fold splits the two intelligences.** The
  deliberative LLM crosses the linear array's front/back fold (centers
  ±120° rear sources) because its policy is not credit-following; the
  substrate is genuinely TRAPPED by the fold's 180° false equilibrium on
  far bins (77 fold-divergent credits — honest measurement punishing
  correct turns behind the fold), while being ~20× faster than the LLM
  where its trained policy applies (4.65 s vs 86.8 s to center). This is
  the cleanest argument yet for the layered design (fast substrate reflex
  inside the fold, deliberation for the ambiguous rear half) and for the
  bin/production key unification + Phase-3 reflex work.
- **Also recorded:** a FRESH substrate (no import) is inert on this task by
  design — recommend_action is positive-gated with no azimuth→turn affinity
  entry, and the one live-trained bias (+0.101) sits under the 0.3 default
  confidence gate (the clamp-vs-threshold asymmetry). Within-trial learning
  is visible in arm C traces: overshoot books negative credit, the policy
  switches to the opposite turn and servos onto center.
