# 0.9.1 — Substrate-Annotates-LLM-Context Release

**Target version:** 0.9.1 (bump from 0.9.0)
**Status:** Active. Plan written 2026-05-13.
**Owns:** [decisions/nac.py](../../src/maxim/decisions/nac.py), [runtime/agent_loop.py](../../src/maxim/runtime/agent_loop.py), [runtime/gating.py](../../src/maxim/runtime/gating.py), [embodiment/](../../src/maxim/embodiment/), [proprioception/pain_bus.py](../../src/maxim/proprioception/pain_bus.py), [simulation/sim_logger.py](../../src/maxim/simulation/sim_logger.py), [prompts/](../../src/maxim/prompts/)
**Companion plans:** [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) (Wires 1+2+3 scoped here), [persona_convergence_crucible.md](persona_convergence_crucible.md) (Roy iterations that motivate the release)

## Why 0.9.1 (not 1.0)

The 1.0 release plan ([v1_refinement.md](v1_refinement.md)) explicitly **deferred** the bio_emergent_persona_foundations wires to 1.1+ on the rationale that [docs/experiments/12_v1_phased_attribution.md](../experiments/12_v1_phased_attribution.md) Phase A reproduced cross-session recall without scaffolds. That rationale held until the Roy harness produced a different falsification target: cross-session recall happens, but **substrate-acquired bias does not translate into action selection**.

Five Roy iterations (Roy-0 through Roy-2pc) shipped a symmetric structural-vs-behavioral gap:

- **Roy-0**: substrate-primary throughout, rehearsal fixture — cluster monoculture.
- **Roy-1a**: llm-primary at test, original holdout — wire structurally preserved, behaviorally inert (LLM proposer doesn't consume cluster bias).
- **Roy-1b**: substrate-primary at test, original holdout — wire consumed but held-out percepts don't fire priming clusters.
- **Roy-2**: llm-primary + multi-arc priming + original holdout — multi-arc priming did NOT widen cluster vocabulary; clean A-vs-C tool-family divergence via salience-mediated LLM-prompt path only.
- **Roy-2pc** (PR #243): substrate-primary + multi-arc priming + **engineered-overlap fixture** — byte-identical action distribution across all three arms. Engineering semantic overlap is insufficient to break the gap.

`cluster_reward_bias_l2` reproduces within 1% across all five iterations (substrate wire is rock-solid). The behavioral pathway from cluster bias to action selection has at least one (possibly two) blocking gates under substrate-primary AND it isn't read at all under llm-primary. The architectural fix that routes around both gates is **substrate-annotates-LLM-context** — surface bio-state at the LLM prompt where the proposer can read it across percept regimes the substrate didn't directly drill.

0.9.1 ships that pattern as a focused release. 1.0 still ships when [v1_refinement.md](v1_refinement.md) §D1-D3 docs work + C4/C5/C6 deprecation cycle complete; the 1.0 disposition rationale updates from "Phase A reproduced recall without scaffolds" to "Phase A reproduced recall, Roy iterations established the need for annotation wires, 0.9.1 ships them, 1.0 stabilizes the surface."

## Naming reconciliation

The Roy iteration docs use "Wire 1" loosely to mean the **design pattern**: substrate annotates LLM context. The [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) plan uses "Wire 1" specifically for **risk-sensitive action annotation** (one application of the pattern, on `CausalLink` variance). The pattern as a whole is Wires 1+2+3 in that doc.

**None of the existing three wires directly surface `cluster_reward_bias` to the prompt** — which is what Roy-2pc's specific finding points at. 0.9.1 therefore ships **a fourth annotation wire** (Wire-A: cluster-bias annotation) alongside the existing three.

| Wire | Source plan | Annotation surfaces |
|---|---|---|
| **A** | NEW (this plan) | `NAc._cluster_reward_bias` for currently-active EC clusters → per-tool `[recently rewarding]` / `[unfamiliar]` hints |
| **1** | foundations | `CausalLink.variance_estimate` → per-tool `[high variance]` / `[reliable]` hints |
| **2** | foundations | `NAc._percept_valences[entity_class, failure_mode]` → percept salience modulation (Pavlovian) |
| **3** | foundations | `Embodiment.get_disabled_affordances()` → tool list filter |

## Scope

| Stage | Item | LOC | Persistence | Frozen impact |
|---|---|---|---|---|
| 0a | Roy-2c probe (`min_confidence=0.0`, H1 vs H2 disambiguator) | ~10 | none | none |
| 0b | Stage 0 telemetry (`agent_id` on action JSONL, NAc snapshot at session-end, entity_class on MOTOR/PERCEPT) | ~150 | `_format_version` bump on action JSONL | none |
| 0c | `recommend_action` instrumentation (EC activation, proposal confidence, bias consulted per turn) | ~80 | none | none |
| 0d | Roy-4 EC-activation instrumentation (1.1 cross-modal binding validation prereq) | ~80 | none | none |
| 1 | **Wire 3** — Embodiment-state → tool filter | ~80 | none | none |
| 2 | **Wire-A** — Cluster-bias annotation (NEW) | ~150 | none | none |
| 3 | **Wire 2** — Pavlovian percept aversion | ~250 | new dict on NAc, `_format_version` bump | GatingContext `learned_aversions` add |
| 4 | **Wire 1** — Risk-sensitive action annotation | ~200 | none | OutcomePrediction `uncertainty_interval` add |
| 5 | Roy-3 validation iteration on 0.9.1 substrate | ~30 (spec only) | none | none |
| **Total 0.9.1** | | **~1030** | 2 format-version bumps | 2 frozen-dataclass field appends |

Ordering rationale:
1. **Stage 0a-d first** because telemetry blocks observation. Roy-2c (one env var) lands before any wire work; Stage 0b-c lands as the structural prerequisite for measuring whether subsequent wires produced behavioral signal; Stage 0d lands the EC-activation instrumentation needed for Roy-4 (the 1.1 cross-modal binding validation prereq).
2. **Wire 3 second** (embodiment filter) — smallest, no persistence, demonstrates the framing without risk.
3. **Wire-A third** — directly addresses Roy-2pc finding, no persistence, ships before Wire 2's persistence change.
4. **Wire 2 fourth** — only persistence change; lands with full schema discipline. Pre-merge two-lens review must check for the latent-bridge×subscriber trap referenced in [CLAUDE.md](../../CLAUDE.md) (Wave 1 biosystem_unification lesson).
5. **Wire 1 last** — depends on Wire 2 having generated variance data to weigh.
6. **Roy-3** — validation iteration with all four wires active. Same priming as Roy-2pc, both AUT modes, both fixtures (original holdout + engineered overlap). Answers "did the annotation pattern produce the cross-arm behavioral divergence the cluster-bias path couldn't"?
7. **Roy-4** (parallel) — runs once Stage 0d ships. Validates the 1.1 cross-modal binding plan's design before its implementation lands. Can run any time after 0d ships; doesn't block the wire stages.

Per [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md), each Stage 1-4 gets a pre-merge two-lens review (Executor + Architecture lenses). Stage 0a-c can ship under a single review since the changes are observability-only.

## Stage 0a — Roy-2c probe (`min_confidence=0.0`)

**Why first:** Roy-2pc's H1 vs H2 hypotheses cannot be distinguished without this probe. Cheap (one env-var change + new iteration spec); informs whether Wire-A's annotation needs to surface raw `cluster_reward_bias` values or whether the consumer is the only block.

**Implementation:**
- Add `MAXIM_NAC_MIN_CONFIDENCE` env-var override at [agent_loop.py::propose_via_substrate](../../src/maxim/runtime/agent_loop.py). Default 0.3 (current behavior); env var, if set and parseable as float, overrides.
- Document in [CLAUDE.md](../../CLAUDE.md) env-var table.
- Add `conftest.py` autouse scrub for the new env var (per the [opt-in env vars in hot startup paths](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_opt_in_env_in_hot_paths.md) feedback rule).

**Roy-2c iteration:**
- `scenarios/roy/roy_2c_iteration.yaml`: name=roy-2c, same priming + arms + fixture as Roy-2pc, `MAXIM_NAC_MIN_CONFIDENCE=0.0` set via runner (or doc'd in protocol).
- Expected diagnostic outcomes (pre-registered):
  - **A > B > C on `sense_food_source` counts** → H2 confirmed (gate was the block); Wire-A still ships but documentation notes that lowering the gate is a viable interim too.
  - **A ≈ B ≈ C reproduces** → H1 confirmed (LinguisticEncoder→EC alignment is the block); Wire-A is the only architectural fix.
  - **A < C** → unexpected; investigate before Wire-A design.

## Stage 0b — Telemetry instrumentation

Lifted verbatim from [bio_emergent_persona_foundations.md § Stage 0](bio_emergent_persona_foundations.md). Independent of all wires; ships ahead of them.

- Thread `agent_id` + `session_id` into every action record via the existing `RequestContext` ContextVar at [utils/http.py](../../src/maxim/utils/http.py).
- Add `entity_class` field to MOTOR/PERCEPT events (sim_log subsystems).
- Save NAc snapshots at session boundary (not just final).
- `_format_version` bump per CC1 contract.

## Stage 0c — `recommend_action` instrumentation

**Why:** Roy-2pc's headline diagnosis: "single-experiment H1-vs-H2 disambiguation is structurally impossible without per-turn observability". Future Roy iterations need this to interpret results.

**Implementation:**
- New per-turn JSONL event `sim_recommend_action` (sim_log subsystem) emitted from [decisions/nac.py::recommend_action](../../src/maxim/decisions/nac.py). Fields: `agent_id`, `tick`, `current_cluster_id`, `cluster_reward_bias_consulted` (the value read from `_cluster_reward_bias` for the active cluster), `best_tool`, `best_score`, `min_confidence`, `passed_gate` (bool).
- No persistence change (event stream only).
- The event MUST emit even when `recommend_action` returns `None` (sub-threshold path). Without this, Roy iterations can't tell whether the gate fired vs the consumer didn't run at all.

## Stage 0d — Roy-4 EC-activation instrumentation (validation prereq for 1.1 cross-modal binding)

**Why:** Roy-2c confirmed H1 (LinguisticEncoder → EC alignment is the block). Wire-A ships as the interim signal-surfacing fix, but the structural fix is [`cross_modal_substrate_binding.md`](cross_modal_substrate_binding.md) (1.1 plan): EC nodes that co-fire across modalities acquire Hebbian binding edges. Before that plan invests in implementation, **Roy-4 validates that priming-cluster ↔ test-cluster pairs WOULD have been linked under a proposed binding rule.** If Roy-4 fails (the pairs genuinely never co-fire), the deeper fix is replacing LinguisticEncoder with an aligned multimodal encoder — a different research direction. Roy-4 is the cheap gate that prevents misallocation.

**Implementation:**
- New env-var `MAXIM_EC_TRACE_ACTIVATIONS=1` gating a per-tick `sim_ec_activation` JSONL event from [similarity/ec.py::pattern_complete_or_separate](../../src/maxim/similarity/ec.py). Fields: `agent_id`, `tick`, `active_node_id`, `activation_strength`, `modality_tag` (sensor / linguistic / drive — derived from the encoder source).
- `conftest.py` autouse scrub for the env var per [feedback_opt_in_env_in_hot_paths.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_opt_in_env_in_hot_paths.md).
- `scenarios/roy/roy_4_iteration.yaml`: same priming + fixture + arms as Roy-2c. Single-variable change: `MAXIM_EC_TRACE_ACTIVATIONS=1` in runner environment.
- **Analysis (post-run):** compute pairwise co-activation matrix across the priming session JSONL. For each pair `(node_a, node_b)` where both fired in the same tick at least N times during priming, mark them as "would-have-bound." Then check whether any test-phase active nodes are in the would-have-bound neighborhood of a priming `sense_food_source` cluster.

**Pre-registered diagnostic:**
- **At least one test-phase active node has a would-have-bound edge to a priming cluster** → cross-modal binding plan justified; Stage 2+ of [cross_modal_substrate_binding.md](cross_modal_substrate_binding.md) is greenlit for 1.1.
- **No would-have-bound edges exist between priming and test clusters** → encoder alignment is so severe that even Hebbian binding wouldn't form the link. Cancel the 1.1 binding plan; redirect to a 1.2+ encoder replacement research direction.

**Sizing:** ~50 LOC instrumentation + ~30 LOC analysis script + outcome doc. Single-session experiment.

**Owns:** [docs/experiments/21_roy_4.md](../experiments/21_roy_4.md), [docs/experiments/protocols/21_roy_4_reproduction.md](../experiments/protocols/21_roy_4_reproduction.md), [scenarios/roy/roy_4_iteration.yaml](../../scenarios/roy/roy_4_iteration.yaml).

## Stage 1 — Wire 3: embodiment-state → tool filter

Lifted from [bio_emergent_persona_foundations.md § Wire 3](bio_emergent_persona_foundations.md). No design changes from the foundations doc; recapped here for self-containment:

- New `Embodiment.get_disabled_affordances() -> set[str]` returning affordances routed through components below integrity threshold.
- New `Embodiment.get_degraded_affordances() -> dict[str, float]` returning `affordance_name → integrity` for partially-damaged paths.
- Hook in [agent_loop.py](../../src/maxim/runtime/agent_loop.py) before tool description assembly: filter disabled tools; append `[DAMAGED: integrity 0.X]` to degraded ones.
- Default thresholds: `integrity < 0.3` disables; `integrity < 0.6` annotates.

**Test surface:**
- Unit: thresholds behave as documented.
- Integration: damage a component in a sim; verify the affordance disappears from the next prompt's tool list.

**Frozen contract impact:** None.

## Stage 2 — Wire-A: cluster-bias annotation (NEW)

**The Roy-2pc-specific INTERIM fix.** Surfaces `NAc._cluster_reward_bias` to the LLM prompt for currently-active EC clusters, so the LLM proposer can read substrate-acquired bias across percept regimes the substrate didn't directly drill.

**Coexists with the structural fix in 1.1.** Wire-A is *static signal-surfacing* — it renders associations the substrate already encoded (via tool-name keys that survive the encoder gap). The *structural* fix is cross-modal binding ([`cross_modal_substrate_binding.md`](cross_modal_substrate_binding.md)) which lets the substrate learn new cross-modal associations via Hebbian co-activation. Wire-A surfaces tool-level bias *immediately* for the llm-primary path; the binding plan teaches the substrate to form proper cross-modal edges over experience. Both ship; neither supersedes the other. Wire-A remains the llm-primary surface even after binding lands because llm-primary's proposer doesn't call `recommend_action` and therefore doesn't consume bound-edge neighborhoods.

### What's there today
- `NAc._cluster_reward_bias: dict[str, dict[str, float]]` (keyed `agent_id → cluster_key → bias`). Existing data structure; no schema change needed.
- `NAc.recommend_action` already consults this per `current_cluster_id` argument.
- **Nothing surfaces this map at LLM prompt time.** The llm-primary AUT proposer never sees it.

### Implementation

**REVISED 2026-05-13 per Roy-2c finding (H1 confirmed): the original active-cluster-restricted design has been replaced with an agent-wide aggregation.** Roy-2c demonstrated that the priming-acquired EC clusters and the engineered test-fixture EC clusters are structurally disjoint under LinguisticEncoder embedding. Restricting Wire-A's bias rendering to "clusters that match the current percept's active set" reproduces exactly the bug that motivated the wire's existence — the active-cluster intersection with priming clusters is empty in the failure mode this wire is designed to fix.

- New method `NAc.get_agent_tool_biases(*, agent_id: str, top_n: int = 5) -> list[tuple[str, float]]`:
  - Iterate `_cluster_reward_bias[agent_id]` across ALL clusters (no active-cluster filter).
  - Aggregate per-tool by max(|bias|) across all clusters for that agent.
  - Return top-N `(tool_name, bias)` pairs sorted by absolute bias descending.
  - Rationale: the priming substrate's tool-level signal ("this agent has experienced strong reward on `sense_food_source`") survives the encoder-alignment gap; the cluster-level signal ("on these specific EC clusters") does not. Wire-A surfaces the surviving granularity.
- New `PromptBuilder` section: `cluster_bias_annotations`. Renders the top-N biases as a structured block in the prompt:
  ```
  Substrate associations from prior experience:
    sense_food_source  [strongly rewarding from prior experience]
    infant_humanoid_pick_up  [neutral / mixed]
  ```
- Prompt section ranks at IMPORTANT priority (per [feedback_prompt_section_priority.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_prompt_section_priority.md)).
- Bias value mapping: `bias >= 0.5` → "strongly rewarding"; `0.1 ≤ bias < 0.5` → "mildly rewarding"; `-0.1 ≤ bias < 0.1` → "neutral / mixed"; `-0.5 < bias < -0.1` → "mildly aversive"; `bias ≤ -0.5` → "strongly aversive".
- Hidden under env var `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION=1` for the Roy-3 ablation (default ON in 0.9.1).

### Frozen contract impact
None. `_cluster_reward_bias` already persists; we're adding a read path. No new dataclass fields, no persistence schema changes.

### Behavioral signal
The Roy-2pc-specific question is whether arm A's llm-primary proposer reads "sense_food_source [strongly rewarding from prior experience]" and uses it on the engineered fixture's food-themed percepts. **Pre-registered Roy-3 outcome:**
- Arm A `sense_food_source` count > arms B/C → cluster wire IS expressible via annotation; pattern works.
- Arm A ≈ arms B/C → annotation didn't reach the proposer's decision pathway; either prompt-budget eviction or the LLM didn't bias on the hint. Investigate prompt rendering.

### Test surface
- Unit: `get_agent_tool_biases` aggregates across all clusters per agent, not per active cluster set.
- Unit: bias-string mapping covers all ranges.
- Unit: empty `_cluster_reward_bias[agent_id]` returns empty list (cold-start agent).
- Integration: a sim with primed substrate produces a prompt containing the annotation block.
- Integration: the env-var disable surfaces an annotation-free prompt (Roy-3 ablation prerequisite).
- **Regression guard for the Roy-2c finding:** a sim where the test percept's active EC clusters are DISJOINT from the priming clusters MUST still produce the annotation block. The annotation is agent-wide, not active-cluster-restricted.

## Stage 3 — Wire 2: Pavlovian percept aversion

Lifted from [bio_emergent_persona_foundations.md § Wire 2](bio_emergent_persona_foundations.md). No design changes; the field reservations already shipped (PR #216, 2026-04-30 under V1 Phase A). 0.9.1 implements the consumer.

- Persist `NAc._percept_valences: dict[tuple[str, str], float]` keyed by `(entity_class, failure_mode)`.
- `_format_version` bump on NAc dump; backward-compat reader returns empty dict.
- `record_percept_valence(entity_class, failure_mode, valence, *, agent_id)` + `get_percept_valence(...)` with keyword-only `agent_id` per CLAUDE.md per-agent stash rule.
- Decay: extend [agent_loop.py](../../src/maxim/runtime/agent_loop.py) section 8.5 to call `decay_percept_valences` per-tick.
- New `PainBus` subscriber `create_percept_valence_subscriber(nac)` auto-wired via `build_pain_bus`. **Pre-merge review must check** the latent-bridge×subscriber trap (per [pain_bus_unification.md](pain_bus_unification.md)) — Wire-A's annotation read site and Wire 2's PainBus subscriber must not double-attribute the same outcome.
- Read site: extend `GatingContext` with `learned_aversions: dict | None = None` (frozen-safe additive field; already reserved). `TextSalienceScorer._compute_salience` queries it on percept arrival.

**Frozen contract impact:**
- `GatingContext.learned_aversions` reserved field is now wired. Audit gate: docstring update declares the addition.
- NAc `_format_version` bump.

**Test surface:**
- Unit: round-trip persistence; missing field → empty dict.
- Integration: PainBus subscriber auto-wires through `build_pain_bus`.
- Multi-agent: two agents sharing one NAc instance attribute valence to distinct `agent_id` keys (the CC4 rule).

## Stage 4 — Wire 1: risk-sensitive action annotation

Lifted from [bio_emergent_persona_foundations.md § Wire 1](bio_emergent_persona_foundations.md). No design changes; `OutcomePrediction.uncertainty_interval` already reserved (PR #216). 0.9.1 implements the producer + consumer.

- `CausalLink.variance_estimate: float = 0.0` (mutable, no frozen impact). Update via Welford's online variance in `record_outcome`.
- `OutcomePrediction.uncertainty_interval: tuple[float, float] = (0.0, 0.0)` populated from variance + observation count.
- `NAc.get_action_risk_profile(event_sig, *, agent_id) -> dict[str, float]` returning `{action_signature → risk_score}`.
- Tool description assembly in [agent_loop.py](../../src/maxim/runtime/agent_loop.py): append `[high variance]` / `[reliable]` annotations.

**Honest scope caveat (preserved from foundations doc):** behavioral effect goes through the LLM. Hybrid bio-system + LLM, not pure substrate-driven. A cleaner post-1.0 design adds a real risk-weighted action ranker that pre-filters tools before the LLM sees them. The hybrid version ships in 0.9.1 to keep scope tight.

**Frozen contract impact:**
- `OutcomePrediction.uncertainty_interval` reserved field now populated. Audit gate: docstring update.
- `CausalLink` is mutable; new field invisible to frozen-contract surface.

## Stage 5 — Roy-3 validation iteration

After Wires A+1+2+3 ship, Roy-3 runs against the same multi-arc priming + held-out fixtures as Roy-2/Roy-2pc to measure whether the annotation pattern produced the cross-arm behavioral divergence the cluster-bias path couldn't.

**Spec:** `scenarios/roy/roy_3_iteration.yaml`. Two sub-iterations (single spec, distinct yamls):
- Roy-3a: llm-primary at test, **original** held-out fixture (Roy-1a replay with annotations).
- Roy-3b: llm-primary at test, **engineered overlap** fixture (Roy-2pc replay with annotations under llm-primary so the cluster-bias annotation has best read-chance).

**Pre-registered diagnostic outcomes:**
- **A > C on `sense_food_source` counts in Roy-3b** → Wire-A annotation reached LLM proposer's decision pathway; pattern works on engineered overlap. Wire-A clearly justified.
- **A > C on tool-family divergence in Roy-3a (richer than Roy-2's 17/3/2 vs 21/5/1/1)** → Pavlovian aversion (Wire 2) + risk annotation (Wire 1) compound the existing salience-mediated signal.
- **A ≈ B ≈ C across both fixtures** → annotation pattern alone is insufficient. Re-scope: pre-filter ranker (post-1.0 cleaner design) becomes load-bearing; investigate prompt-budget eviction.

Roy-3 is also the first Roy iteration with full Stage 0c `recommend_action` instrumentation, so disambiguation between annotation-reaches-LLM vs LLM-reads-but-ignores becomes structurally possible.

### Stage 5 outcome (2026-05-24/25)

Roy-3 shipped ([23_roy_3.md](../experiments/23_roy_3.md), [PR #258](https://github.com/dennys246/Maxim/pull/258)). Pre-registered "A ≈ B ≈ C across both fixtures" reproduced — annotation was wired end-to-end but the LLM saw `[neutral / mixed]` at test time (max(|bias|) = 0.036 in Roy-3a, 0.098 in Roy-3b, both below the 0.1 "mildly rewarding" floor). The two Roy-3 follow-up items:

1. **Bisect the priming-side regression** (Wire 1 vs Wire 2 vs interaction): **CLOSED** by Roy-3c-bisect ([29_roy_3c_bisect.md](../experiments/29_roy_3c_bisect.md), [PR #266](https://github.com/dennys246/Maxim/pull/266)). Verdict: the wires did NOT cause the regression. Two axes, two outside causes: (a) key count 6→2 is non-code environmental drift in the encoder layer (env-var refuted by A1, narrator drift refuted by A3); (b) bias magnitude saturated→partial→decayed-to-neutral is Wire-A's intentional bio-fidelity decay correction (bee42ca), confirmed behaviorally by A2.
2. **"Decide whether Wire-A's render needs a raw priming snapshot floor"**: **SUPERSEDED** by [cluster_reward_bias_decay_tau_split.md](cluster_reward_bias_decay_tau_split.md). The bisect's magnitude-axis finding surfaced that Wire-A's tau was inherited by accident from `reward_bias_decay_tau=50.0` (sized for EC threshold modulation, NOT multi-turn substrate-voice annotation). Splitting the tau and tuning it to ~300 (the same pattern Wire 2's `percept_valence_decay_tau=200.0` already used) is the bio-fidelity-respecting fix — preserves the decay correction the bee42ca fold shipped, and lets Wire-A's annotation be expressive at test time without a separate "raw snapshot" mechanism. Phase 3 is a Roy-3a-retry validating the tune.

**0.9.1 release status:** ships unchanged per Roy-3's original recommendation. The tau-split work is independent of 0.9.1's shipped surface and is a 0.9.2 / 1.0 follow-up.

## Cross-cutting: persistence schema

Two format-version bumps:

1. **Action JSONL** (Stage 0b): adds `agent_id` + `entity_class` fields. `_format_version: "1.1"` on the wrapper.
2. **NAc dump** (Stage 3 Wire 2): adds `_percept_valences` dict. `_format_version: "1.1"` already declared on the schema; backward-compat reader handles missing field as empty dict.

Reader policy (mirrors existing patterns):
- Missing field → empty default; one `_format_version` drift warning per file_type per process per CC1.

## Cross-cutting: frozen contract impact

Per [CLAUDE.md](../../CLAUDE.md) CC3 audit rules:
- `GatingContext.learned_aversions: dict | None = None` reserved field WIRED in Stage 3.
- `OutcomePrediction.uncertainty_interval: tuple[float, float] = (0.0, 0.0)` reserved field WIRED in Stage 4.
- No new frozen dataclasses introduced.
- No existing frozen dataclasses modified beyond reading reserved fields.

Audit gate: each Stage's docstring updates declare the reserved field is now active. No backward-incompatible changes.

## Cross-cutting: env-var inventory

New env vars introduced in 0.9.1:

| Env var | Stage | Default | Purpose |
|---|---|---|---|
| `MAXIM_NAC_MIN_CONFIDENCE` | 0a | unset → 0.3 | Override `recommend_action` threshold for Roy-2c / Wire-A ablation |
| `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION` | 2 | unset → enabled | Ablation gate for Wire-A |

Both get `conftest.py` autouse env-scrub fixtures per [feedback_opt_in_env_in_hot_paths.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_opt_in_env_in_hot_paths.md). Added to CLAUDE.md env-var table.

## Sizing summary

| Stage | Item | LOC |
|---|---|---|
| 0a | Roy-2c env-var + iteration spec | ~30 |
| 0b | Stage 0 telemetry | ~150 |
| 0c | `recommend_action` instrumentation | ~80 |
| 1 | Wire 3 (embodiment filter) | ~80 |
| 2 | Wire-A (cluster-bias annotation) | ~150 |
| 3 | Wire 2 (Pavlovian) | ~250 |
| 4 | Wire 1 (risk annotation) | ~200 |
| 5 | Roy-3 spec + outcome doc | ~30 (spec) |
| **Total** | | **~970** |

Estimated calendar: 4-6 days for Stages 0-5, including pre-merge two-lens reviews per stage.

## Definition of done

- All stages shipped to main behind pre-merge two-lens reviews.
- Roy-2c artifact in `~/.maxim/roy/roy-2c/` with H1 vs H2 disambiguation logged in [docs/experiments/20_roy_2c.md](../experiments/20_roy_2c.md) (or whichever id sequence applies).
- Roy-3 outcome doc with cross-arm behavioral divergence measured under annotation pattern.
- `_percept_valences` round-trips through dump/load (Wire 2).
- Sim with `MAXIM_LOG_FILE` produces JSONL records with `agent_id` on every action AND per-turn `sim_recommend_action` events.
- Damaged-component test: an agent's tool list visibly drops affordances on integrity drop.
- Pavlovian test: an agent's percept salience score on `(entity_class, failure_mode)` shifts measurably after a single pain event with that signature, persists across session restart.
- Cluster-bias annotation test: a sim with primed substrate produces a prompt block surfacing the top-N per-tool biases.
- Version bumped to 0.9.1 in `pyproject.toml` + `src/maxim/__init__.py`.

## What 0.9.1 does NOT do

- **No pre-filter ranker.** Wire 1's hybrid LLM-annotation design ships; a real risk-weighted action ranker that pre-filters tools before the LLM sees them is post-1.0 cleanup if Wire 1 turns out to be insufficient.
- **No cross-session annotation persistence beyond what Wires 2 already brings.** Wire-A reads `_cluster_reward_bias` which already persists; Wire 2 adds `_percept_valences` which persists. Wire 1 and Wire 3 are session-scoped reads against persisted producers — no new persistence surfaces.
- **No streak detection (Wire 4) or oscillator coupling (Wire 5).** These remain 1.1+ per the foundations doc deferral rule.
- **No min_confidence default change.** Roy-2c probes via env var; if H2 is confirmed, the disposition is *to ship Wire-A regardless* — lowering the gate is an interim only if Wire-A turns out insufficient.

## Risk register

| Risk | Mitigation |
|---|---|
| Wire-A's prompt section gets evicted by the budgeter under high-percept-load runs | Section sits at IMPORTANT priority; pre-merge review must verify it survives a sim with WMS at capacity. Roy-3a is the natural surface test. |
| Wire 2's PainBus subscriber double-attributes with `ToolPainBridge` | Pre-merge review enforces the latent-bridge×subscriber trap check ([pain_bus_unification.md](pain_bus_unification.md) "Latent risk surfaced during pre-merge review"). Regression test: `tests/unit/test_pain_bus.py::TestBuildPainBus::test_subscriber_does_not_link_pending_tool_event`. |
| Roy-3 reproduces Roy-2pc identically (annotation pattern insufficient) | 0.9.1 still ships — the wires close architectural gaps regardless of persona behavior per the foundations doc framing rule. Roy-3's negative result escalates the post-1.0 pre-filter-ranker design to 1.1 instead of "experimental limbo through 1.1+". |
| Prompt budget exhausted under all four annotation sections at once | Per-section priority ranking handles this; pre-merge review must verify Wire 3 (most important — physical damage) survives budget pressure over Wire 1 (least important — variance). |

## 1.0 implications

0.9.1 changes the 1.0 plan's bio_emergent_persona_foundations disposition from "deferred to 1.1+" to "shipped in 0.9.1". Update [v1_refinement.md](v1_refinement.md) execution-order and 1.0 exit-criteria sections post-merge.

Outside of that one line item, **1.0 scope is unchanged**. D1-D3 docs work + C4/C5/C6 deprecation cycle still gate 1.0 release; 0.9.1 ships the wires inside the existing 0.9 deprecation window without disrupting the 1.0 timeline.

## References

- [persona_convergence_crucible.md](persona_convergence_crucible.md) — Roy harness; Roy-0 through Roy-2pc iteration log.
- [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) — Wires 1+2+3 design (lifted here for Stages 1, 3, 4); field reservations already shipped.
- [docs/experiments/19_roy_2pc.md](../experiments/19_roy_2pc.md) — positive-control negative result; the empirical floor for this release.
- [v1_refinement.md](v1_refinement.md) — 1.0 plan whose bio_emergent_persona disposition this release supersedes.
- [pain_bus_unification.md](pain_bus_unification.md) — latent-bridge×subscriber trap reference for Wire 2 review.
- [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md) — pre-merge review template.
