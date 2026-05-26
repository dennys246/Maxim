# Decay + consolidation calibration framework

**Target version:** 1.1+ (post-tau-split, post-SCN-tying).
**Status:** Draft. Plan written 2026-05-26, prompted by user's observation during the post-Roy-3c-bisect session: "the ultimate goal of decay is to enable forgetting more or less of unimportant or not used information — theoretically can't we create within a simulation have a particular percept occur and adjust the decay so it just begins to form a short term memory, then in another phase we'd do the same but for long-term memories and consolidations."
**Owns:** [`src/maxim/decisions/nac.py`](../../src/maxim/decisions/nac.py) (NAc decay parameters), [`src/maxim/memory/`](../../src/maxim/memory/) (tier-transition mechanisms), `scripts/calibrate_decay.py` (new), `docs/experiments/3X_decay_calibration_baseline.md` (new), `docs/experiments/3X_decay_calibration_results.md` (new).
**Companion plans:** [cluster_reward_bias_decay_tau_split.md](cluster_reward_bias_decay_tau_split.md) (Phase A of the prerequisite chain), [scn_decay_anchoring.md](scn_decay_anchoring.md) *(to be written as Phase C kickoff output)* (Phase B prerequisite), [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) (the bio-fidelity thesis this plan operationalizes), [persona_convergence_crucible.md](persona_convergence_crucible.md) (whose iterations would consume calibrated values), [memory_consolidation_practice.md](memory_consolidation_practice.md) (companion living doc — calibration framework feeds back into the consolidation mechanism this practice doc refines).

## Why this plan exists

Maxim's substrate has five distinct decay functions in NAc — `reward_bias`, `goal_reward_bias`, `cluster_reward_bias`, `percept_valence`, and `eligibility` — plus the memory tier transitions (FORMING → SHORT_TERM → LONG_TERM) which themselves have pressure-decay dynamics. All of them are currently parameterized by **hand-picked constants** chosen from neuroscience priors + use-case math.

This was correct for 0.7-0.9.x when the substrate was being bootstrapped: hand-picking gives plausible values to ship, and the validation iterations (Roy series) surface when the values are wrong. But it scales badly:

- Each new consumer of a decay parameter inherits an inherited-by-accident default (the canonical example: Wire-A inheriting `reward_bias_decay_tau=50.0` for `_cluster_reward_bias` even though the 50.0 was sized for EC threshold modulation, not multi-turn substrate-voice annotation — see [cluster_reward_bias_decay_tau_split.md](cluster_reward_bias_decay_tau_split.md)).
- The five decay functions don't act independently — they couple through the memory tier transitions and through downstream learning consumers (SCN temporal credit depends on eligibility, NAc reward attribution depends on cluster reward bias, etc.). Tuning one in isolation can break the implicit coordination.
- Hand-picked values are not portable across deployments. Even after the SCN-tying follow-up makes them hardware-independent in the *time unit* sense, the *correct numeric value* still varies by hardware (because the substrate's tick rate, scenario diversity, and consumer-readout requirements differ across deployments).

**Calibration-by-simulation is the bio-coherent answer.** Biological systems don't store decay constants in a config file — synaptic decay rates are *shaped by experience*, with Hebbian-style mechanisms continually adjusting the timescales of plasticity to match the agent's task structure. The path forward is to let the substrate's own behavioral targets drive the decay calibration: define the tier transitions you want, run scenarios that exercise them, search the tau space until the observed behavior matches.

## Framing rule

**This plan does NOT replace Phase A/B/C of the [tau-split](cluster_reward_bias_decay_tau_split.md) and [SCN-anchoring](scn_decay_anchoring.md) work.** It depends on them as prerequisites:

- **Without the tau split**, each tau isn't independently addressable. The calibration framework can't tune `cluster_reward_bias_decay_tau` separately from `reward_bias_decay_tau` if they're the same field.
- **Without SCN-anchoring**, calibrated tau values are in agent-loop-tick units. Effective decay rate is hardware-dependent (~10x across deployments per [feedback_decay_is_tick_anchored_not_wall_clock](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_decay_is_tick_anchored_not_wall_clock.md)). Calibration produces non-portable constants.
- **The tau-split Phase B (Roy-3a-retry)** is the *first manual calibration data point* — one sample of "tau=300 produces what behavior on Roy-3a." This plan generalizes that single sample into a systematic loop.

Sequencing: tau-split ships → SCN-anchoring ships → THIS framework ships. Each preserves the prior's calibration math. Calibrated tau values from this framework become the new defaults that replace the hand-picked 50.0 / 200.0 / 300.0.

## The five decay functions and their natural calibration scenarios

| Decay function | Current default | Bio role | Canonical calibration scenario |
|---|---:|---|---|
| `reward_bias` | 50 | EC threshold modulation — fast adaptation so recognition threshold tracks recent reward signal | Single rewarded event → tune so threshold modulation decays to negligible (<10% of peak) within K ticks where K matches the expected "this reward is no longer current" timescale |
| `goal_reward_bias` | 50 (inherits) | Goal-conditioned reward attribution per Stage 4 Wire 1 | Goal stack push → reward → goal stack pop. Tune so the goal-specific bias survives the goal's lifetime but extinguishes once the goal is dead |
| `cluster_reward_bias` | 300 (post-tau-split) | Wire-A substrate-voice annotation across multi-turn test phase | Priming cluster receives N rewards, then is silent for M ticks. Tune so the cluster bias is still above the 0.5 "strongly rewarding" band at the consumer-readout point (Wire-A's mid-test-arm render) |
| `percept_valence` | 200 | Pavlovian aversion — burned-by-dragon stays aversive across sessions | Aversive percept + outcome → identical percept later. Tune so the aversion bias survives sleep/replay consolidation but extinguishes if the percept is repeatedly observed without aversion outcome |
| `eligibility` | (per-tick decay rate) | Fast-decay trace for credit assignment within an event window | Event start → reward at +K ticks. Tune so eligibility trace covers the event-to-reward window cleanly without bleeding across episode boundaries |

Each scenario has a measurable behavioral readout. The framework iterates tau until the readout matches the target.

**Plus the memory-tier consolidation parameters** (out of scope for Phase 1 of this plan; see "Consolidation undercooked" below):
- `_PROMOTION_PRESSURE_THRESHOLD = 3.0` (SHORT_TERM → LONG_TERM gate)
- `_PRESSURE_DECAY_RATE` (how fast accumulated promotion pressure decays)
- Sleep-cycle consolidation timing
- Access-context-diversity scoring weight

These are NOT decay parameters in the same sense, but they couple to decay through the tier-transition dynamics. Calibrating decay without also calibrating consolidation produces tau values that match the decay-side behavior but may produce wrong tier-transition timing. **This is the user-flagged "consolidation still needs developing" concern** and is explicitly Phase 4+ work.

## What this does NOT do (in Phase 1)

- **Does NOT make decay learnable** (no per-cluster, per-modality, or per-agent dynamic decay rates). That's the "deeper version" — see below. Phase 1 is one-shot calibration → static config defaults, not continuous learning.
- **Does NOT touch the consolidation mechanism itself** in Phase 1. The framework tunes *decay rates* against tier-transition targets; tuning *the tier-transition mechanics themselves* is Phase 4+ once the decay side stabilizes.
- **Does NOT replace the Roy iteration loop.** Roy iterations test integrated persona behavior; this framework tests individual decay-function calibration. They're complementary: Roy validates that calibrated values produce sensible persona behavior; calibration provides the values Roy validates.
- **Does NOT auto-deploy calibrated values to production.** Phase 1 ships calibrated values as the *new manual defaults* via a config PR. Continuous auto-calibration (the framework runs on every release, the defaults update via CI) is Phase 5+ work.

## Open questions for review (lots — this is an aspirational plan)

### Calibration target definition

**Q1.** For each decay function, what's the precise target behavioral readout? Three candidate target shapes:
- **Time-to-decay-below-threshold**: tune tau so `bias(tick=K) < T` for chosen K, T
- **Time-to-tier-transition**: tune tau so the underlying memory enters a target tier at tick K
- **Consumer-readout match**: tune tau so a downstream consumer (Wire-A's annotation, NAc.propose_via_substrate's tool selection, etc.) produces the expected output at the target time

Different targets produce different optimal tau values. Phase 1 of this plan defines one target per decay function based on the dominant consumer.

**Q2.** How are competing objectives reconciled? `cluster_reward_bias_decay_tau` needs to satisfy "slow enough for Wire-A to render at test time" AND "fast enough that learned aversions extinguish when appropriate" AND "bio-plausible relative to associative-memory timescales." Pareto-front sweep + manual tradeoff? Or weighted single-objective with weights from review?

**Q3.** How is multi-decay coupling handled? If you tune `cluster_reward_bias_decay_tau` in isolation, the resulting value might break `eligibility_decay` coupling (SCN temporal credit). Sequential per-function calibration with full-Roy validation between each, OR joint multi-dimensional search?

### Scenario library

**Q4.** What's the minimal canonical scenario set per decay function? Single-percept, single-reward, multi-reward, paired-aversion-then-extinction, etc. The set defines the calibration framework's discriminative power.

**Q5.** Are scenarios deterministic or stochastic? Deterministic gives reproducible calibration but may overfit to one scenario shape. Stochastic (LLM-driven narrator like the cradle arcs) gives natural variability but adds noise — n=3 minimum per [feedback_n3_minimum_for_partial_vs_saturated](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_n3_minimum_for_partial_vs_saturated.md).

### Mechanics

**Q6.** Search strategy: binary search on tau, grid search, gradient-free optimization (Nelder-Mead, etc.), Bayesian optimization? Cost per scenario run is ~minutes; total search budget across five decay functions is hours-to-days.

**Q7.** Per-tau convergence criterion: when do we declare a tau value "matches the target"? Tolerance bounds + n=3 stability? Exact match?

**Q8.** What hardware baseline are calibrated values valid against? If SCN-anchoring is correctly implemented, the answer is "any hardware after the SCN constant is set." If not, calibrated values are leader-specific (RTX 5080 per `user_hardware_setup`).

### Consolidation expansion (Phase 4+)

**Q9.** The consolidation parameters (`_PROMOTION_PRESSURE_THRESHOLD`, pressure decay rate, access-context-diversity scoring) are not decay parameters in the same sense, but they couple to decay through tier-transition dynamics. Does the framework expand to calibrate them too, or do they stay separately tuned?

**Q10.** Sleep-cycle consolidation timing is currently hand-picked. Does the framework calibrate sleep cycle length against "what fraction of FORMING memories should consolidate per sleep cycle" targets?

**Q11.** If a calibration scenario produces a target that the *current consolidation mechanism cannot meet at any tau value*, the framework's verdict is "consolidation mechanism needs revision before calibration is meaningful." That's a useful diagnostic but it shifts work to consolidation-mechanism development. How do we sequence that?

### The deeper version

**Q12.** Should decay become per-cluster (or per-modality, or per-agent) learnable rather than globally parameterized? Hebbian-style mechanisms calibrate weights through experience; decay rates could be similarly calibrated continuously rather than via one-shot framework runs. This is the biologically-honest direction but a substantially bigger change. Phase 1 stays with globally-parameterized decay + framework-calibrated defaults; the learnable version is Phase 6+ or a separate plan.

## Sizing (rough, pending Phase 1 detailed design)

| Phase | Item | LOC (rough) | Wall time | Risk |
|---|---|---|---|---|
| 0 | Detailed design pass: pick targets per decay function, define scenario library, choose search strategy | 0 src + ~300 LOC plan refinement | ~1 week | low |
| 1 | Per-decay-function canonical scenario implementation (one scenario per function, scripts/calibrate_decay.py) | ~400 src + ~200 tests | ~2 weeks | medium |
| 2 | Binary-search calibration loop, per-function, with convergence criteria | ~200 src + ~100 tests | ~1 week | medium |
| 3 | Sequential calibration of all five decay functions; full-Roy validation between each; surface coupling failures | ~50 src (config update) + runner days | ~2-3 weeks | high (coupling failures are real) |
| 4 | Consolidation parameter expansion (`_PROMOTION_PRESSURE_THRESHOLD`, pressure decay rate); requires consolidation-mechanism stability first | ~150 src + tests | depends on consolidation maturity | high |
| 5 | CI-integrated calibration: framework re-runs on each release, surfaces drift, suggests config updates | ~300 src + CI integration | ~1-2 weeks | medium |
| 6 | Per-cluster / per-modality / per-agent learnable decay (separate plan candidate) | substantial | months | research |
| **Total (Phases 0-5)** | | **~1400 LOC + ~7-10 weeks** | | |

Phase 6 is open-ended research; explicitly NOT included in the Phases 0-5 estimate.

## What this plan REPLACES vs PRESERVES

**Replaces:**
- Hand-picked decay tau defaults across the five NAc decay functions
- The implicit assumption that decay tau values can be set once at config-design time and never revisited

**Preserves:**
- Wire-A's `bee42ca` bio-fidelity decay correction (the *call* is still wired in; only the *value* changes via calibration)
- All five decay functions' bio-roles (calibration tunes timescale, not function)
- The tier-transition semantics (FORMING/SHORT_TERM/LONG_TERM remain; calibration tunes the gates, not the tiers)
- Manual override via env var (the calibrated defaults remain overridable for diagnostic / A-B testing)

## Why this is a 1.1+ target, not 1.0

The 1.0 release exit criteria per [v1_refinement.md](v1_refinement.md) focus on validation + bio-system stabilization + sensorimotor grounding + cleanup + docs. The decay calibration framework is **bio-system optimization**, not stabilization — it improves on already-shipped parameters, which is post-1.0 work.

For 1.0, the path is:
1. Ship the tau-split (closes the inherited-by-accident Wire-A bug)
2. Ship SCN-anchoring (makes existing tau values portable)
3. Run Roy iterations on the cleaned-up parameter set to surface remaining manual-tuning gaps

The calibration framework absorbs the surfaced gaps into a systematic process post-1.0.

If a 1.0 Roy iteration produces a specific decay-tuning surprise that the calibration framework would have caught, accelerate Phase 0 + Phase 1 for that specific decay function as a bridge before 1.1.

## Cross-references to companion docs

- **[memory_consolidation_practice.md](memory_consolidation_practice.md)** — living practice doc that refines the SHORT_TERM → LONG_TERM consolidation mechanism. Phase 4 of this plan depends on consolidation mechanism maturity tracked there.
- **[behavioral_convergence_practice.md](behavioral_convergence_practice.md)** — living doc that tracks "does the agent actually get better across sessions." Calibrated decay values feed into this; the practice doc surfaces when calibration produced values that don't translate to behavioral improvement.
- **[persona_convergence_crucible.md](persona_convergence_crucible.md)** — Roy iterations consume calibrated values; their results validate the calibration framework's choices.

## Authorization gate

Phase 0 (detailed design pass) is a 1-week plan refinement, no source changes — can start anytime post-Phase-C of the tau-split chain. Phase 1+ (implementation) needs explicit user authorization per the standing "open as PR, surface before merge" rule.

## Naming convention note

The user named this plan `decay_consolidation_calibration_plan` in the kickoff conversation. The "consolidation" word is load-bearing — it signals that the framework eventually expands beyond decay-only into the SHORT_TERM ↔ LONG_TERM mechanism. Phase 4 is where consolidation enters scope; Phases 1-3 are decay-only. Don't drop "consolidation" from the name even though Phase 1-3 don't touch it directly.
