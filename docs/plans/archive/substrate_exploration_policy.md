# Substrate Exploration Policy — break substrate-primary fixation so the substrate can adapt

> **ARCHIVED (2026-07-15 plans audit):** ✅ SHIPPED + VALIDATED. Exploration policy (`sim.substrate_explore_bonus_weight` + explore-first hard gate) landed with Exp 41 plumbing (PR #379). Exp 41 fired 2026-06-19 → VOID (design, not mechanism); successor Exp 42 GRADUATED using this exact policy (PR #380). The `Authorization gate open` header is stale.


**Target version:** 1.1
**Status:** Draft — **spike-validated 2026-06-17** (mechanism + lever confirmed empirically; see Empirical validation below). Authorization gate open.
**Owns:** `decisions/nac.py::NAc.recommend_action` (selection), a new per-`(agent, tool)` visit counter on `NAc`, `runtime/agent_loop.py` §8.5 (per-tick decay), `runtime/config_loader.py` (`sim.substrate_explore_*` fields).
**Companion plans:** [grounded_language_acquisition.md](../grounded_language_acquisition.md) (substrate-primary AUT mode — the upstream this plugs into), [behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md) (#6).
**Validated by:** [../experiments/41_substrate_primary_exploration.md](../../experiments/41_substrate_primary_exploration.md).

---

## Why this plan exists

[grounded_language_acquisition.md](../grounded_language_acquisition.md) Phase -1 shipped substrate-primary action generation: `NAc.recommend_action()` produces an action from causal-link confidence + reward bias + a drive-affinity heuristic, and `propose_via_substrate()` wraps it into an `LLMProposal` with no LLM in the loop. Phase 0 shipped the cradle harness and the `SubstrateTelemetry` JSONL writer.

Then Exp 39 (and the 1.0 graduation triage for row #6) hit the wall: **the substrate-primary proposer fixates.** It forms EC clusters and NAc causal links and emits drive-conditioned proposals, but it **selects the same high-scoring tool every tick and never tries an alternative**, so it cannot discover that a different action is better. The 2026-05-11 re-run in `grounded_language_acquisition.md` already saw the symptom: "6 cluster updates all on the same tool (`sense_food_source`) ... the cold-start path collapses to one drive-affinity match and loops on it."

The mechanical cause is exact and small. `NAc.recommend_action` ends in a single deterministic argmax ([decisions/nac.py:1741](../../src/maxim/decisions/nac.py)):

```python
best_tool = max(scores, key=lambda t: (scores[t], t))   # ties break by name
...
if best_score < min_confidence:    # nac.py:1755
    return None                    # IDLE — "Random selection is explicitly NOT a fallback" (docstring nac.py:1577)
```

`scores` is a sum of *static learned components*: causal-link confidence, `reward_bias` (clamped `[0, max_reward_bias=0.20]`, never negative — [nac.py:1673,1831](../../src/maxim/decisions/nac.py)), `cluster_reward_bias`, and the drive-affinity cold-start heuristic ([nac.py:1690-1709](../../src/maxim/decisions/nac.py)). There is **no stochasticity, no novelty bonus, no visit-count term, no softmax anywhere.** Whatever wins `max()` first keeps winning. Nothing in the system rewards an under-tried action, so the substrate cannot explore its way out of a prior-favored-but-wrong habit. That is the missing piece named for 1.1+ across the graduation notes.

This plan adds the minimal mechanism that lets the substrate try alternatives, so that Exp 41 can test whether the unmasked substrate learns to override its own built-in drive-affinity prior from embodied feedback alone.

## Front-gate scope pressure (CLAUDE.md "Working principles for new mechanisms")

> *"Does this need to be its own mechanism, or can it ride on existing infrastructure?"*

Candidate hosts surveyed:

| Candidate | Why insufficient / sufficient |
|---|---|
| `salience/novelty.py::ThreadSafeNoveltyTracker` | Vision/gaze-scoped (COCO track_ids); feeds DefaultNetwork, **not** `recommend_action`. No tool-level visit signal exists. Cannot ride on it. |
| `DefaultNetwork` arousal gate / `suggest_exploration_direction` | Spatial/imagination exploration, not action-selection over tools. Wrong layer. |
| `NAc.reward_bias` / `cluster_reward_bias` | These *add to the winner*; they cannot lift an under-tried loser. Reusing them re-creates fixation. |
| **`NAc.recommend_action` selection step itself** | **This is the right host.** The deterministic `max()` at [nac.py:1741](../../src/maxim/decisions/nac.py) is the single chokepoint through which every substrate-primary action passes. |

**Verdict — rides on existing infrastructure; no new bus / bridge / bio-system.** This is an *additive selection policy* on `NAc.recommend_action` plus one small per-`(agent, tool)` visit-count dict (mirroring the existing `_reward_bias` dict pattern) and its per-tick decay in §8.5. The specific reason a new mechanism is *not* warranted: the selection chokepoint and the per-agent-keyed state pattern both already exist; exploration is a modification of the score-then-argmax, not a new subsystem.

### Working principles applied
- **Principle 1 — `[engineering]` only.** Every invariant below enters as `[engineering]`. It graduates to `[behavioral]` only if Exp 41 earns it. Naming a knob "exploration" does not make it adaptive — Exp 41 decides that.
- **Principle 2 — dormancy over deletion.** If Exp 41 returns FAIL/FAIL (fixation is deeper than selection), the policy is marked `Dormant since <date>: Exp 41 showed selection-stochasticity insufficient` and left wired with `epsilon=0` as the default (≡ legacy argmax), not deleted.
- **Principle 3 — front-gate.** Applied above.

## Current mechanism background (what already ships)

- Selection: `NAc.recommend_action` ([nac.py:1565-1782](../../src/maxim/decisions/nac.py)), deterministic argmax at :1741, IDLE-when-no-opinion contract at :1755/:1577.
- Wrapper: `propose_via_substrate` ([runtime/agent_loop.py:755-853](../../src/maxim/runtime/agent_loop.py)); substrate-primary branch at [agent_loop.py:2803](../../src/maxim/runtime/agent_loop.py); `min_confidence` resolution `_resolve_min_confidence` ([agent_loop.py:730-752](../../src/maxim/runtime/agent_loop.py), default `0.3`).
- Per-tick decay home: §8.5 BIO-SYSTEM PER-TICK MAINTENANCE ([agent_loop.py:3756-3776](../../src/maxim/runtime/agent_loop.py)) — already calls `decay_eligibility`, `decay_reward_biases`, etc.
- Config pattern (per `feedback_prefer_config_over_new_env_vars.md`, exemplar `sim.aut_turn_timeout_s`): field on a `*ConfigSection` → `_FIELD_TO_ENV` entry → `_coerce_for_field` clamp → `resolve_setting()` at build ([config_loader.py:281-296,364,566-573](../../src/maxim/runtime/config_loader.py)).
- Seeding: `utils/seeding.py` (S4) — exploration RNG must be seeded from here for reproducible experiments.

## Phases

### Phase 0 — instrument fixation (the baseline we must beat) — small
Add a fixation metric so improvement is measurable, before changing selection. Read from the existing `SubstrateTelemetry` proposal stream:
- `proposal_entropy` and `unique_tools_per_window` over the session (`substrate_telemetry.jsonl::proposal.tool`).
- Pin the current behavior: a test asserting the deterministic argmax produces `unique_tools == 1` on the degenerate fixture. This is the "before" number Exp 41 compares against.
**Owns:** `simulation/substrate_telemetry.py` (derived-metric helper), `tests/...`.

### Phase 1 — exploration-aware selection (the core, ~60-90 LOC) — moderate
Replace the bare argmax with an exploration-aware selection at [nac.py:1741](../../src/maxim/decisions/nac.py). **Recommended primary design: novelty-bonus-before-gate** (see Open Q1):
- Maintain `self._visit_count: dict[(agent_id, tool), int]` (init near [nac.py:429](../../src/maxim/decisions/nac.py), same shape as `_reward_bias`).
- Before the argmax, add a novelty bonus to each *already-scored* tool: `scores[t] += bonus_weight / (1 + visit_count[t])` (UCB-flavored; under-tried tools get lifted, well-tried ones decay toward their learned score).
- Then the existing argmax + `min_confidence` gate run **unchanged**. Increment `visit_count` for the selected tool.
- A separate `epsilon` knob (ε-greedy) is the simpler fallback design retained behind the same config field; `epsilon=0` and `bonus_weight=0` ≡ legacy deterministic argmax (the regression anchor).
**Config:** `sim.substrate_explore_epsilon` (default `0.0`) and `sim.substrate_explore_bonus_weight` (default `0.0`) via the 4-step config pattern; resolved at `NAc` construction (AgentFactory build) and threaded onto `NACConfig`. Default-off means **zero behavior change** for every non-experimental caller.
**RNG:** seeded via `utils/seeding.py`; the seed flows from the harness so Exp 41 arms are reproducible.

### Phase 2 — visit-count decay — small
Decay `_visit_count` per tick in §8.5 ([agent_loop.py:3768](../../src/maxim/runtime/agent_loop.py)), alongside the existing decay calls, so the exploration bonus fades like the other biases and the agent re-explores after the world changes (phase transitions). Decay shape mirrors `decay_reward_biases` (tau-based).

### Phase 3 — behavioral validation
Run [../experiments/41_substrate_primary_exploration.md](../../experiments/41_substrate_primary_exploration.md). Outcome decides whether any invariant graduates to `[behavioral]` (GRADUATE), stays `[engineering]` (PARTIAL), or the mechanism goes Dormant (FAIL/FAIL).

## Sizing

| Phase | Scope | LOC | Risk |
|---|---|---|---|
| 0 | Fixation metric + baseline pin | ~40 | Low |
| 1 | Exploration-aware selection + visit-count state + config | ~80 | Medium (touches the selection chokepoint) |
| 2 | Per-tick visit-count decay | ~25 | Low |
| 3 | Exp 41 (separate doc) | — | — |
| **Total** | | **~145 src + tests** | |

**Risk shape:** the only medium-risk surface is the `recommend_action` selection edit — it is the single path every substrate-primary action flows through, and it must preserve the IDLE-when-no-opinion contract and default to byte-identical legacy behavior when off. Both are pinned by regression tests below.

## DO NOT BREAK (load-bearing invariants — all enter `[engineering]`)

- **[engineering] Exploration-off ≡ legacy deterministic argmax.** With `epsilon=0` and `bonus_weight=0`, `recommend_action` returns exactly what [nac.py:1741](../../src/maxim/decisions/nac.py) returns today. This is the regression anchor and the Exp 41 control arm. *Regression guard:* `tests/unit/test_nac.py::TestExplorationPolicy::test_off_equals_legacy_argmax`.
- **[engineering] Exploration never proposes a tool outside `available_tools`; IDLE-when-no-available-tools is preserved.** The novelty bonus lifts never-tried tools that ARE in the active registry (the spike showed the target affordances have zero base score, so a "redistribute among already-scored tools only" design would never surface them — the bonus must be able to add a zero-base *available* tool to `scores`). But it operates strictly over `available_tools`; it never invents a tool, and `propose_via_substrate` still returns `None` when the active registry is empty. The "Random selection is explicitly NOT a fallback" contract ([nac.py:1577](../../src/maxim/decisions/nac.py)) holds — the bonus is deterministic, not a random pick. *Regression guard:* `test_exploration_only_proposes_available_tools` + `test_exploration_off_returns_none_when_no_scored_tools`.
- **[engineering] OFF (`substrate_explore_bonus_weight == 0.0`) is byte-identical legacy argmax.** The default and the Exp 41 control arm; the bonus branch is fully short-circuited at weight 0. *Regression guard:* `test_exploration_off_equals_legacy_selection`.
- **[engineering] `reward_bias` clamp `[0, max_reward_bias]` is untouched.** Exploration adds a *separate* novelty term to the local `score` and writes only to `_visit_count`; it never writes into `reward_bias`/`_cluster_reward_bias` (which would corrupt the learning signal and re-create the key-embedded-statistic class of bug). *Regression guard:* `test_exploration_does_not_mutate_reward_bias`.
- **[engineering] New tunables route through `config.json`, not a primary `MAXIM_*` env var** (per `feedback_prefer_config_over_new_env_vars.md`). `sim.substrate_explore_bonus_weight` is the field; the `MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT` env override is the auto-derived harness knob. *Regression guard:* `tests/unit/test_config_loader.py` substrate-explore coercion + `_FIELD_TO_ENV` registration.
- **[engineering] The shipped lever is the deterministic novelty bonus (no RNG).** Runs are reproducible by construction — no seeding needed. If an ε-greedy comparator is added later (Open Q1), it MUST seed from `utils/seeding.py`; the novelty-bonus path has no such dependency. *Regression guard:* `test_exploration_visit_decay` (deterministic decay) + `test_exploration_is_deterministic`.

## What this plan does NOT do
- Does not touch the **LLM-primary** action path (the default `--aut-mode llm-primary`). Exploration is substrate-primary-only; under LLM-primary the LLM already provides behavioral diversity.
- Does not add curiosity to vision/gaze/spatial (`ThreadSafeNoveltyTracker` stays as-is).
- Does not change reward learning, causal-link formation, or the drive-affinity heuristic. It only changes *which scored action is selected*, not *how actions are scored or learned*.
- Does not attempt cross-session exploration persistence — visit counts are session-scoped (like eligibility traces), not persisted.

## Open questions

1. **Novelty-bonus-before-gate vs ε-greedy vs softmax.** *Author recommendation:* ship the UCB-style novelty-bonus as primary (it lifts under-tried-but-plausible actions across the `min_confidence` gate, which pure ε-greedy among above-threshold candidates cannot do when only the fixated tool clears threshold), keep ε-greedy behind the same config field as a simpler comparator, and defer softmax (adds a temperature knob and obscures the IDLE contract). Exp 41's exploratory ε-sweep informs the default.
2. **Should the novelty bonus be allowed to push a sub-threshold action *over* `min_confidence`?** *Author recommendation:* yes — that is precisely the fixation-breaking behavior (the alternative action is sub-threshold *because* it's never been tried). But cap the bonus so it cannot lift a truly-zero-score tool (one the substrate has no representation for) over the floor, preserving IDLE. Implement as: bonus applies only to tools with a non-zero base score component.
3. **Visit-count decay tau.** *Author recommendation:* start at the `reward_bias_decay_tau` neighborhood (50 ticks) so exploration re-opens on roughly the same timescale the learned biases fade; expose it as a config field but do not over-tune before Exp 41.
4. **Default-on or default-off after Exp 41 GRADUATE?** *Author recommendation:* even on GRADUATE, keep substrate-primary exploration default-off in 1.1 and flip it on only with the substrate-primary AUT mode's own maturation in [grounded_language_acquisition.md](../grounded_language_acquisition.md) Phase 1+. The two should graduate together.

## Authorization gate
Proceed to Phase 0+1 only on explicit authorization. Phase 3 (Exp 41) requires the experiment doc's setup deliverables (deceptive arc + analyzer) to land first.

## Empirical validation (2026-06-17 spike)

A pre-build spike (substrate-primary on `cradle_prelinguistic_deceptive`, exploration OFF = current code, smollm-1.7b, $0 local) confirmed the mechanism and the lever, and ruled out a false lead:

- **Activation is NOT the blocker (ruled out).** After phase-0 entity activation the substrate proposer's candidate set grows **20 → 35 tools**, including the full hearth set (`sense_hearth`, `hearth_observe`, `hearth_warm_self`, `hearth_touch`). Same registry object on both ends (`generate_tools_for_entity` registers via plain `register()` → always-active, into the same `aut_registry` the proposer reads via `executor.registry.list()`). An earlier "candidate set = 20, no hearth" reading was a first-tick timing artifact (the proposer logged before the generative loop activated phase 0).
- **The blocker is selection fixation, and it's quantified.** Cold start: every tool ties at 0.352; `sense_food_source` wins the name-sort tiebreak, executes successfully, and its causal-link confidence snowballs: **1.004 → 1.297 → 1.405 → 1.48 → 1.557 → 1.657** across ticks while every other tool stays flat at ~0.35–0.42. The deterministic argmax then locks on it for the whole run (170/170 actions in the 10-turn run).
- **`hearth_warm_self` / `hearth_touch` never enter the top-6.** Only the passive `read_hearth_heat_output` appears (~0.35). The interaction affordances have no causal link, no reward bias, and get no drive-affinity boost → never tried → never accumulate signal → never picked. Self-reinforcing.
- **The plan's chosen lever (novelty-bonus-before-gate) is validated.** `warm_self`/`touch` have **0 visits**, so a `bonus_weight / (1 + visit_count)` term is exactly what lifts a never-tried, no-signal tool over the `min_confidence` gate and forces a trial — at which point pain → negative link → the override question becomes measurable. ε-greedy alone among above-threshold candidates would not surface them (only `sense_food_source` clears the gate once it snowballs).

**Two caveats the spike surfaced (carried into Open Questions / the Exp 41 scenario design):**
- **Drive design:** `warm_self` gets no drive-affinity boost while `sense_food_source` rides the hunger drive; the infant's `core_temperature` deficit (−0.15) recovered to 0 over the run. Even with exploration, a *sustained* thermal deficit (and a constrained food escape-valve) is needed so warmth-seeking stays salient after the first trial. This is a `cradle_prelinguistic_deceptive` arc/drive-tuning task, tracked in [41_substrate_primary_exploration.md](../../experiments/41_substrate_primary_exploration.md) §2.
- **F1 (spurious positive link):** once `warm_self` executes it returns `success=True` (the embodiment failure rides in `side_effects`), booking a positive link that competes 1:1 with the 0.5-weighted negative — the within-session learning signal (Exp 41 H2) will be shallow until this is addressed.

## Iteration log
- **2026-06-17** — Pre-build spike. Mechanism confirmed: fixation, not activation. Lever confirmed: novelty-bonus-before-gate. No-regret pieces landed alongside: `cradle_prelinguistic_deceptive` arc + `substrate_telemetry` `causal_links` field. Proceeding to Phase 0+1.
- **2026-06-18** — Phase 0/1/2 SHIPPED (NACConfig `substrate_explore_bonus_weight`/`substrate_explore_decay_tau`; `_visit_count` + novelty-bonus in `recommend_action`; `decay_exploration_visits` in agent_loop §8.5; `config.json::sim.substrate_explore_bonus_weight` + `_FIELD_TO_ENV` + bio_stack wiring; 16 unit tests; conftest scrub). **Open Q1 RESOLVED → explore-first**: a build-and-verify spike showed bare decaying-novelty is *leaky* — per-tick decay lets an already-tried high-alphabetical tool (`temporal_patterns`, which also snowballs via causal-link success) recover novelty and re-win the name tiebreak, so exploration never descends to the low-alphabetical tools. Fix shipped: a sticky `_ever_selected` set gives never-tried tools the FULL bonus weight, guaranteeing one trial of every *available* tool before exploitation (pinned by `test_exploration_first_visits_every_tool_before_repeat`). End-to-end this broke the absolute single-tool fixation (1 → 14 distinct tools explored).
- **2026-06-18 — "affordance-filtering" hypothesis FALSIFIED; real cause was a self-inflicted exploration bug, now fixed.** A first pass suspected interaction affordances (`hearth_warm_self`/`touch`) were being deactivated by orchestrator goal-relevance curation. Direct candidate-set instrumentation disproved it: the substrate proposer's candidate set reaches **35 tools and stays there with every interaction affordance present** (`hearth_warm_self`/`touch`/`observe`, `food_source_eat`) — they were always available. The real bug was in THIS policy: the soft novelty-bonus + per-tick visit-count **decay resurrected a tried tool's novelty back toward the full weight** (`weight/(1+visits)` with `visits→0`), so already-tried high-alphabetical tools kept re-tying at the full bonus and the name-tiebreak starved the low-alphabetical `hearth_*` tools — `hearth_warm_self` sat in `scores` at the full 1.5 for 122 ticks and was never selected. The sticky `_ever_selected` set only switched the *formula*, not the *magnitude*. **Fix: explore-first is now a HARD GATE** (`nac.py::recommend_action`) — while any scored tool is untried this session, selection is restricted to untried tools, independent of decay/causal score. **End-to-end result (substrate-primary, `cradle_prelinguistic_deceptive`, weight 1.5): all 35 tools proposed AND executed exactly once before exploitation — including `hearth_warm_self`/`touch`/`observe`/`food_source_eat`.** The exploration policy now fully achieves its purpose: it surfaces the harmful affordance Exp 41 must test. Regression guard: `test_explore_first_untried_beats_high_score_tried_tool`. Remaining for a valid Exp 41 run: `cradle_prelinguistic_deceptive` drive tuning (sustained thermal deficit so warm-seeking stays salient after the one exploratory trial) + F1 (the `success=True` spurious-positive-link signal quality).
