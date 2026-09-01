# The ternary-collapse sweep (2026-08-31)

Systematic audit for good/neutral/bad representations collapsed into a
boolean, run before fixing D53 so the fix could be wired everywhere rather
than patched at one call site. Four parallel readers over partitioned
surfaces, then synthesis; every load-bearing claim re-verified by hand
against the source, which mattered — the readers contradicted each other on
the single most consequential point.

## What the sweep was looking for

`Valence` is a live four-value enum and `decisions/causal_link.py::_VALENCE_TO_REWARD`
maps it canonically:

    POSITIVE 1.0   NEGATIVE 0.0   NEUTRAL 0.5   UNKNOWN 0.5

`NEUTRAL = 0.5` is the Rescorla-Wagner prior midpoint, consumed by
`CausalLink.update_prediction_rw` and by Welford online variance. It is the
correct target for "this happened, but it teaches nothing directional."

Two shapes, stated so they could not be answered by grep:

- **(a) collapsed ternary** — a live 3+-valued representation squashed into
  an `if/else`.
- **(b) sign-only sink** — a parameter that only accepts a sign or a bool, so
  neutrality is *unrepresentable* and defaults to the positive floor.
  **Shape (b) is the dangerous one: it looks like a design choice, not a bug.**

## The site table

| file | symbol | shape | a NEUTRAL outcome booked | EARNED? | status |
|---|---|---|---|---|---|
| `runtime/tool_dispatch.py` | `record_outcome` (valence → `nac.observe`) | a | `POSITIVE` → causal link → a flat **+0.50 rising to +0.64** into `recommend_action`, vs a 0.3 gate | **Exp 42, Exp 52 Phase B** | FIXED |
| `runtime/tool_dispatch.py` | `record_outcome` (`credit_goal`) | a | `+1.0`; α 0.15 / cap 0.20 → two ticks saturate | Exp 42 | FIXED |
| `runtime/tool_dispatch.py` | `record_outcome` (`cluster_reward` else-arm) | a | `+1.0`; ~7 ticks saturate the cap | **Exp 42** | FIXED |
| `runtime/tool_dispatch.py` | `record_outcome` (measured zero-progress drive) | a | fell through to the `+1` floor | Exp 42 | FIXED |
| `bridges/tool_pain_bridge.py` | `record_tool_complete(success: bool)` | **b** | `POSITIVE` on every completion; caller hardcoded `True` | **Exp 52** | FIXED |
| `runtime/executor.py` | `Executor.execute` | a | `record_tool_complete(success=True)`, no third arm | Exp 52 | FIXED |
| `hardware/reachy/motor_backend.py` | `ReachyOrientMotorBackend.execute` | **b** | `success = reached is not False` — unverified → `True` | **in Exp 53/53b/54's path** | FIXED |
| `tools/reachy.py` | `FocusOnSoundTool.execute` | **b** | `success=True`, no `side_effects` at all | D53 seed | FIXED |
| `decisions/causal_link.py` | `_record_observation_unlocked` | b | the 3-way valence EMA is **structurally dead** — link ids hash `outcome_signature`, which embeds success/failure, so a link is only re-observed with its own valence | indirect | OPEN (D56) |
| `agents/exec_agent.py` | `_evaluate_staging` | a | `1.0 if success else 0.0` starves `_eval_valence_extremity` (`abs(v−0.5)*2`) → constant 1.0 | Exp 52 (memory promotion) | OPEN (D56) |
| `integration/bio_enrichment.py` | `_query_nac` | a | `else: "neutral"` — NEUTRAL and UNKNOWN merge irreversibly | prompt surface | OPEN (D56) |
| `embodiment/tool_bridge.py` | `ModulatorAffordanceTool.execute` | a | measured-zero → withheld; modeled-zero → `+1`, same function | Phase-2 | FIXED (both now withhold) |
| `bridges/pain_bridge.py` | `record_action_complete` | — | **correct** ternary; caller `_maybe_turn_around` hardcodes `success=True` | no | OPEN (D56) |
| `prompts/acting_coach.py` | `_compose_nac_annotations` | a | neutral rows silently dropped from "Learned Experience" | no | OPEN (D56) |
| `bridges/salience_bridge.py` | `enrich_salience`, `record_interaction` | a + b | `Outcome.success is None` → booked as FAILURE via `else False` | no | OPEN (D56) |
| `integration/memory_hub.py` | `_wire_sensitization::modulation_lookup` | a | documented neutral behaviour is **unreachable**; an all-neutral class reads extremity 1.0 — maximum sensitization, the inverse of intent | no | OPEN (D56) |
| `bridges/planning_bridge.py` | `get_tool_success_rate`, `record_plan_outcome` | a + b | neutral links sit in the denominator only → all-neutral history scores 0.0, *worse* than no history (0.5) | no — never instantiated | OPEN (D56) |
| `embodiment/cerebellum.py` | `form_engram` | a | `success = outcome_valence != "NEGATIVE"` | no — no callers | OPEN (D56) |

## The three findings that changed the fix

**1. D53's row said "no EARNED row is bounded." That was wrong.**
`recommend_action` has a `causal` component — `score += best_pos` where
`best_pos = max(link.confidence)`, **unweighted** (the negative term is
weighted 0.5). A link's confidence is **0.50** on creation and **0.64+** once
re-observed (`min(0.99, 0.5 + 0.1 * (count ** 0.5))`), against a
`min_confidence` gate of 0.3 — so it clears the gate from the very first
observation. The collapsed valence *did* reach action selection — not through `reward_bias`
(`_record_outcome_impl` writes neither bias dict) but through the causal
term. `MAXIM_OPERANT_ONLY_CREDIT` **cannot** suppress it: the drive branch is
evaluated first and `operant_only` gates only the `cluster_reward` else-arm,
while `learn_success`, the POSITIVE link and `credit_goal(+1.0)` all fire
above it. **Exp 52's "the mother is the sole teacher" held for the cluster
surface only.** Exp 42 runs `--aut-mode substrate-primary` with
`drive_relief_only=False` and no `OPERANT_ONLY`, so all four `tool_dispatch`
sites were live in its credit path — and its graduation row carries no
re-run trigger naming the credit rule.

**2. The same collapse was hard-coded one layer BELOW `record_outcome`**, in
`ReachyOrientMotorBackend`, which Exp 53/53b/54 do attach. Their claims
survive only because those harnesses measure `achieved_delta_rad`
independently. That is luck-shaped, not design-shaped.

**3. D53's preferred fix was not viable as written.**
`proprioception/pain.py::PainDetector._check_movement_failure` is live but
reachable only from the camera path, is a one-sided *pain* detector (bad /
less-bad, never neutral), and writes the `"movement"` namespace rather than
`"tool"`. Reviving it is a rewrite. **The comparator already existed, twice,
in the right namespace and already tri-valued** — `FocusOnSoundTool` and
`ReachyOrientMotorBackend` both compute `reached: bool | None` and both threw
it away at the `ToolOutput` boundary, because `Executor.execute` reads
`side_effects` and never `metadata`.

## The pattern worth remembering

**The sinks were innocent.** `update_cluster_reward(reward: float)`,
`credit_goal(reward: float)` and `NAc.observe(outcome_valence: Valence)` all
already accepted neutrality. Every collapse was at a caller or in a
`success: bool` parameter.

The tier is idiomatic in this codebase — `cluster_bias_annotation.bias_to_band`
ships a five-band renderer with a written rationale for withholding the gloss
from neutral; `focus_learner._report_to_nac` is a correct three-way;
`bio_stack._distribute_reward_from_reaction` traces `reward_skip_neutral`;
and `cradle_mother.reactive_mother_tick` refuses to round zero up, which is
the only reason Exp 52's satiated control arm works at all. So the systemic
shape is not "nobody thought about neutral" — it is **correct neutral-aware
consumers starved by boolean producers.**

Two decisions fell out of the arithmetic rather than taste. Both sinks
accumulate `current + alpha * reward`, so `reward = 0.0` is an *exact no-op*
— which means booking 0.0 and skipping the write are behaviourally identical
for the bias, and skipping additionally avoids materialising phantom entries
and promoting a triple's credit-source to `"mixed"`. Hence: skip, don't write
zero.

## Collateral, unrelated to the tier

- `PainDetector._emit_pain` hand-built its `Reaction` and bypassed
  `reactions/compat.py::pain_signal_to_reaction`, so `agent_id` was dropped
  and **every** PainDetector-origin pain distributed exactly zero reward
  (D54). Root cause was deeper than the bypass: `PainDetector` had no
  `agent_id` at all.
- `FocusLearner.save` hand-rolled `open()+json.dump`, violating both the
  `atomic_write_json` and `_format_version` invariants (D55).

## Process notes

- **A regression guard was holding the defect in place.**
  `test_zero_progress_falls_back_to_tool_success` pinned the `+1` floor. It
  was inverted, not deleted — and the inverted version immediately caught a
  real regression introduced by the fix (a *failed* tool whose drive measured
  zero briefly booked nothing instead of `-1`).
- **The parallel readers disagreed on the load-bearing point** — one said the
  collapsed valence reached action selection, the other said it could not.
  Resolving it required reading `recommend_action` directly. A single reader
  would have shipped whichever answer it happened to hold.
- Roughly a third of the tier surface outside the hot partitions is dead code
  (`PlanningBridge` never instantiated, `Cerebellum.form_engram` and
  `MemoryHub.record_interaction` with no callers, `ValenceSignal` unconnected
  at both ends) carrying latent copies of the same bug. Relevant to
  dormancy-over-deletion: they will re-enter live paths silently if wired.
