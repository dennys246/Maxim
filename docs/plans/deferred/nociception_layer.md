# Nociception layer — pain says what it is, consumers say what they take

> **DEFERRED 2026-09-24 (owner decision), revised the same day after a four-lens review** (reports in
> [../reviews/nociception_layer/](../reviews/nociception_layer/): architecture, bio-fidelity, wiring,
> risk/ledger; all four: *adopt the deferral, with changes* — folded below, §Review record).
> Step 1 ships inside memory-strength Phase 2S-c (`PainSignal.kind`, `proprioception/pain.py::classify_pain`;
> **pending that PR's merge**); the rest is recorded here, not built.
> **Revive when**, whichever comes first:
> (a) [adaptive_nociception.md](adaptive_nociception.md) revives — its producer-side gain is step 5;
> (b) [reflex_layering.md](reflex_layering.md) route 3 revives — its step 3a hands felt-pain modulation here;
> (c) a pain consumer **added after 2026-09-24** re-derives meaning from `pain_type` + `source` instead
> of reading `kind`, or needs a kind `PainKind` does not carry;
> (d) **any R4 credit-routing PR opens** (roadmap 1.4 Phase 5): R4 routes trace credit to the selection
> surface, which turns F1 from near-inert into live action learning — **step 3 (the F1 fix) must land
> before or with it**;
> (e) memory-strength **reads a pain-derived tag in a validation run** (the opt-in `strength` strategy
> in a survival or sim campaign) while F3 is unfixed;
> (f) F1, F1b or F2 is measured to move behaviour.

## Why this exists

The owner asked, while reviewing Phase 2S-c: *"Is it possible we need to better abstract pain into a
sort of factory and engine with paths from potential sources coming in on standard ways — why do we
have a ToolPainBridge, should it just be PainBridge?"*

The audit's answer: the name is not the problem; **meaning is**. A pain signal does not say what it
is, so each consumer decides. 2S-c's first draft is the incident: its bridge consumed the PainBus
without the classification `pain_bus._pain_encoding` applies, and both review lenses measured air
hunger recorded as pain at 0.7, fear at 0.5 and a failing tool at 0.3+. Owner decision 4 of the memory
plan says pain is nociception only; nothing in the type enforced it.

**The rule genuinely differs by consumer** — memory must exclude air hunger; fear must include it as
drowning's unconditioned stimulus; reward pays drive pain today and the survival loop relies on it.
So the goal is not one rule for everyone but **one vocabulary**, with each consumer's rule written in
it, in one place, and pinned.

## The pain system as it stands (verified on `main`, 2026-09-24; corrected by the wiring lens)

**Producers.** `PainSignal` is constructed in 6 modules (`embodiment/body.py`, `simulation/tools.py`,
`simulation/sandbox.py`, `proprioception/pain_bus.py`, `proprioception/perceived_pain.py`,
`proprioception/pain.py::PainDetector`); a pain `Reaction` (`kind="pain"`) directly in 7
(`reactions/compat.py`, `runtime/pain_interceptor.py`, `runtime/sim_adapter.py`,
`simulation/sandbox.py`, `simulation/conversational_source.py`, `proprioception/perceived_pain.py`,
`embodiment/backends/cerebellum_modulator.py`).

**Two buses, by design.** `PainBus` carries rich free-form context; `ReactionBus` is the typed
isolation surface. PainBus forwards each signal as a `Reaction`; ReactionBus pain is bridged back to
PainBus subscribers as a **rebuilt** `PainSignal` (`_reaction_to_pain_signal`) — which **drops
`context["source"]`** and parses the `PainType` from the `Reaction.source` suffix, defaulting to
`EXTERNAL_SIGNAL` when it cannot (F3). History: [archive/pain_bus_unification.md](../archive/pain_bus_unification.md),
[archive/reaction_bus_unification.md](../archive/reaction_bus_unification.md); the open deferred shell
[pain_bus_bridge_subscriber_unification.md](pain_bus_bridge_subscriber_unification.md) is the prior
record of the subscriber overlap (F2).

**Consumers, and the rule each applies today:**

| Consumer | Surface | Rule today |
|---|---|---|
| `create_pain_memory_subscriber` (Hippocampus capture) | PainBus | captures **every** signal ≥ 0.4; only its encoding's `pain` field is nociceptive (`_pain_encoding`: drive → `extra["drive_pain"]`, anticipated → `extra["anticipated_pain"]`) |
| `hippocampus.capture_reaction` → `Episode.finalize` net valence (`bio_stack` step 4b) | ReactionBus (`subscribe_all`) | every reaction, **anticipated pain included** (F1b) |
| `create_percept_valence_subscriber` (Wire-2) | PainBus | every signal above threshold, keyed `(agent, entity_class, failure_mode)`; human-driving gate |
| `create_pain_nac_subscriber` (NAc causal events) | PainBus | every signal above threshold; human-driving gate |
| `create_pain_cluster_fear_subscriber` → `NAc.record_cluster_fear` (Wire-4 fear) | PainBus | subscriber unfiltered; **the authority is `NAc.config.cluster_fear_failure_modes` = `{drive:health, drive:oxygen}`**, also checked by hivemind ingest/bundle (a wire boundary) |
| `bio_stack._distribute_reward_from_reaction` (eligibility-trace reward) | ReactionBus (`subscribe_all`) | every negative reaction with a real `agent_id`: pays **anticipated** pain (F1) **and drive pain, incl. `drive:oxygen`, which the survival loop relies on** |
| `ToolPainBridge` (action-outcome attribution; per-invocation pain for encoding since 2S-c) | PainBus (or `PainDetector` callbacks) | attribution: tool-failure types; embodiment: out-of-band only; encoding (2S-c): NOCICEPTIVE via `kind`. No threshold, no human-driving gate; also writes SCN and the credit distributor |
| `PainCircuitBridge` (`bridges/pain_bridge.py`, Default Network) | PainBus (or detector) | **no type filter** — any pain while a movement is pending |
| `_sim_log_reaction` | ReactionBus | telemetry |
| harness instruments (e.g. `water_trial._record_pain`, the Exp 62 instrument) | PainBus | experiment-specific |

Plus `api.py`'s `on("pain_signal")` → `PainSignalEvent` public surface (CC2, field-additive), for
which no publisher was found — possibly dead; verify before step 2 touches it.

## Verified findings (2026-09-24)

- **F1 — anticipated pain is paid out as real negative reward.** `perceived_pain.py` publishes its
  prediction as `Reaction(kind="pain", valence=NEGATIVE, context=ReactionContext(agent_id=...))`;
  `_distribute_reward_from_reaction` pays `-intensity` for any negative reaction whose `agent_id` is
  neither `None` nor `WORLD_AGENT_ID`. **Live on the `maxim --sim` orchestrator AUT path** (every
  orchestrator sim wires `PerceivedPainAssessor(agent_id="sim_aut", pain_bus=aut_pain_bus)` onto the
  bus `build_bio_stack` subscribed); not live in the Minecraft harness or the Reachy runtime. The 0.5 s
  refractory only rate-limits it. **Near-inert today**: the reward lands on a clamped surface that
  selection does not read — which trigger (d) exists to watch. Because the intensity comes from NAc's
  own confidence, the prediction confirms itself and can never extinguish. **Impact unmeasured**; the
  cheapest bound is one logged sim counting anticipated-pain reactions against
  `pain_chain.distribute_returned`, plus a unit replay. A discrete, verified defect: it belongs in a
  GitHub issue now (the register's "where a thing goes" rule), fixed by step 3.
- **F1b — the same leak into memory.** `hippocampus.capture_reaction` folds anticipated pain into
  `Episode.finalize`'s net valence; via the lossy bridge it also reaches `create_pain_nac_subscriber`.
- **F2 — three NAc writers on overlapping pain, not two bridges.** `ToolPainBridge` and
  `PainCircuitBridge` share a bus (`--sim --embodiment`, Reachy) but their effective inputs are
  disjoint; no double attribution between them was found. The real overlap is
  `create_pain_nac_subscriber` and `ToolPainBridge._on_embodiment_pain`, both calling
  `record_outcome_full` on the same out-of-band embodiment pain; NAc consumes linked pending events, so
  the result depends on subscription order — mostly harmless, unmeasured. Counting reward distribution,
  four paths put pain into NAc.
- **F3 — the ReactionBus→PainBus rebuild misclassifies.** Dropping `context["source"]` and defaulting
  unparseable suffixes (cerebellum, sim_adapter, conversational_source) to `EXTERNAL_SIGNAL` makes them
  NOCICEPTIVE under step 1 — in the memory capture's `pain` field (as before step 1) and in 2S-c's felt
  pain. Consistent with the pre-2S-c classification, so not a 2S-c regression; harmless while the
  strength tag is opt-in and unvalidated (trigger (e)). Fixed by step 2.

## Design principles

1. **A pain says what it is, once, on the type — and stores it.** `PainKind` (NOCICEPTIVE, DRIVE,
   ANTICIPATORY, FRUSTRATION, EXHAUSTION) via `classify_pain(pain_type, source)`; an unclassified
   `PainType` raises. Step 1 ships it as a derived property; step 2 **stores** it at construction (on
   `Reaction` too), so it survives the Reaction→PainSignal round trip and fails at construction rather
   than in a consumer.
2. **A consumer's rule is written in that vocabulary, in one place, and pinned** — `(kind, failure_mode)`
   pairs where a kind alone is too coarse (Wire-4's allowlist is exactly that: NOCICEPTIVE `drive:health`
   and DRIVE `drive:oxygen`, not "all NOCICEPTIVE + all DRIVE").
3. **Parallel pathways, not one adapted value** (bio-fidelity S4). Peripheral / spinal gain (habituation
   and sensitization to the stimulus) lives at the producer (adaptive_nociception); **expectation and
   context modulation is a separate stage consumers declare**, as they declare kinds — the
   sensory-discriminative, affective and amygdala (fear) routes do not share one number.
4. **Anticipation is not an outcome.** A prediction is never paid out as reward at its level; what
   reinforces avoidance is fear **reduction** (two-factor avoidance). The replacement learning signal
   for step 3 is the **signed change in anticipated pain across an action** (a rise punishes, relief
   reinforces) — or, if not built, a recorded known gap.
5. **No new bus.** PainBus and ReactionBus stay.

## Front-gate (roadmap 1.4 Phase 5) — corrected

*Does this need its own mechanism, or can it ride existing infrastructure?* It rides both buses,
`PainDetector` and the builders. Genuinely new, and why: (i) **a kind on the typed surface** — a
ReactionBus consumer can already read the `PainType` from the `source` suffix, but **not the drive line**
(`pain_signal_to_reaction` drops `context["source"]`), so drive vs nociceptive is unrecoverable there;
(ii) **a declared rule per consumer** — an optional `kinds=` filter would not have prevented 2S-c's
omission, so it earns its place only as a **required, keyword-only** subscription parameter; otherwise a
check inside each handler (which also covers the `add_pain_callback` fallback both bridges still use)
is enough; (iii) the producer-side gain stage (adaptive_nociception).

## Steps (dependency order, revised after review: the F1 fix moved ahead of the consumer migration)

1. **`PainSignal.kind` + `classify_pain` + `failure_pain_kind`.** In Phase 2S-c (pending merge): the
   memory capture's encoding and the 2S-c bridge read it; `_pain_encoding` output-identical (420-case
   comparison, round-two review). The 2S-c PR states its ToolPainBridge change is record-only and leaves
   attribution unchanged (SEM cascade row discharged).
2. **Store kind; put it on `Reaction`.** Move `PainKind` to a leaf module (`proprioception/pain.py`
   already imports `reactions.types` — putting it there cycles); an additive defaulted field on
   `Reaction` (the SHAPE-FROZEN marker allows it) **with the isolation review the marker requires**;
   `pain_signal_to_reaction` and `_reaction_to_pain_signal` carry it, fixing F3. Verify the `api.py`
   `PainSignalEvent` surface (dead or live) before touching it.
3. **Fix F1 / F1b** *(v1's step 4)*. Reward distribution and `capture_reaction` never take ANTICIPATORY at its
   level; name the replacement signal (principle 4) or record the gap. **Keep DRIVE paying reward** unless
   the owner decides otherwise — excluding it re-runs the survival rows (§Gates). Land before or with any
   R4 routing PR (trigger (d)).
4. **Consumers declare their rule** *(v1's step 3)*. Migrate each consumer one at a time, preserving its rule
   exactly, guarded by a **golden table generated from pre-migration code** (below). Wire-4's allowlist
   stays the authority — `cluster_fear_failure_modes` is a hivemind wire boundary; express it as
   `(kind, failure_mode)` pairs, never as a kind set.
5. **Producer-side adaptation.** adaptive_nociception.md's per-source gain, plus the separate
   expectation-modulation stage (principle 3). Frozen during experiment campaigns; fingerprinted.
6. *(narrowed)* **Say the bridge boundaries in docstrings.** `ToolPainBridge` = action-outcome
   attribution, `PainCircuitBridge` = motion harm, `create_pain_nac_subscriber` = out-of-band pain; name
   F2's real overlap and settle it with pain_bus_bridge_subscriber_unification.md. **No rename, and no
   "one pain path into NAc"** — the subscribers write separate NAc maps by design.

**Guard spec for step 4, the consumer migration (risk lens).** Per consumer, commit before the migration a golden table from the
pre-migration code over the product of: every `PainType`; sources `drive:{oxygen,health,hunger,saturation}`,
`""`, a tool source, `cerebellum:*`, `perceived_pain:anticipated`; origin (PainBus-direct vs
ReactionBus-bridged); `agent_id` (set, `WORLD_AGENT_ID`, `None`) — recording the accept flag and the
exact delivered value. The migration reproduces it exactly, and each guard is proven by deleting its
filter.

## Taxonomy questions the revive must settle (bio-fidelity lens)

- **`SAFETY_VIOLATION`** is classed NOCICEPTIVE (inherited from `NOCICEPTIVE_PAIN_TYPES`), but its own
  comment calls it a "FearAgent-detected threat" — anticipatory. No producer today; decide before one
  appears.
- **`drive:health`** is an engineering proxy for injury, not nociception: hypoxic health loss while
  drowning is near-painless, and the aversive part is air hunger (DRIVE). Label it a proxy and carry the
  damage cause where the game gives it — **without** touching Wire-4's US set.
- **Air hunger excluded from `pain`** is sound only if drive pressure demonstrably raises the memory tag
  during a breach (the relevance gate can leave it out); pin that with a check.
- **FRUSTRATION**'s learning signal is the prediction error (frustrative nonreward), not a fixed
  intensity; **COGNITIVE_OVERLOAD** reads as effort cost; **EXHAUSTION** overlaps the body's fatigue
  drive; **MOVEMENT_FAILURE** is a motor prediction error, not tissue damage. Fine as step-1 labels;
  revisit with step 5.

## What it must not do

- Change what fear learns from: Wire-4's `(drive:health, drive:oxygen)` set is what Exp 60/61/62 EARNED
  and a hivemind wire boundary.
- Stop paying DRIVE pain as reward without re-running the survival rows.
- Merge the buses or add one; make pain adaptive inside a consumer; rename without a deprecation path.

## What gates it

- **Four-lens design review** before any build beyond step 1 (this plan's own review was the design
  pass for the plan, not for a build).
- **Ledger, by step** (risk lens):

| Step | Rows |
|---|---|
| 1 (2S-c) | none if output-identical; SEM pain → NAc cascade discharge stated |
| 2 kind on Reaction | the generic PainBus/ReactionBus trigger: SEM cascade, row 9; isolation review |
| 3 F1 / F1b | SEM cascade; Exp 42, 45, 52, 56, 57 and the Exp 48 row; Exp 60/61/62 **and R3 §Outcome if DRIVE's reward changes** |
| 4 consumers declare | Exp 60/61/62 (Wire-4 allowlist); SEM cascade; row 9 (percept valence); Exp 10 (memory consumer, discharged by the golden table) |
| 5 adaptation | everything above that reads intensity; row 9; Exp 60/61/62 (oxygen US magnitude vs θ) |
| 6 boundaries | SEM cascade; Exp 42 |

- **Behaviour tiers** ([behavior_tiers.md](behavior_tiers.md)): classification is tier 1; adaptation is
  tier 2 with a learned gain (M6).
- **1.4 guardrail:** nothing here may silently power E1–E3. E1's argmax reads only hunger × fear, so
  deferral is safe for E1; trigger (d) covers E3.

## Review record (2026-09-24)

Four parallel lenses read v1 (reports in [../reviews/nociception_layer/](../reviews/nociception_layer/)).
Folded: corrected producer counts (6 / 7) and the consumer table (three missed consumers, four misstated
rules); F1 scoped to the orchestrator path, marked near-inert-until-R4, with a measurement; F1b and F3
added; F2 re-pointed from the two bridges to the subscriber/bridge overlap; step order (v1 numbering) 1→2→4→3; v1 step 3's
cluster-fear migration **rejected as written** (a kind set is far wider than the allowlist); step 6's
rename and "one path" **dropped**; kind to be stored, not derived, from step 2; the parallel-pathways
principle and the fear-reduction replacement signal for F1; triggers (d) R4 and (e) strength-reads added,
(c) date-anchored; the golden-table guard spec; the taxonomy questions. Not adopted: none of the lenses'
findings was rejected.

## Links

- [adaptive_nociception.md](adaptive_nociception.md) — producer-side gain (step 5).
- [reflex_layering.md](reflex_layering.md) — its step 3a hands felt-pain modulation here.
- [pain_bus_bridge_subscriber_unification.md](pain_bus_bridge_subscriber_unification.md) — F2's prior record.
- [behavior_tiers.md](behavior_tiers.md) M6 (pain intensity).
- [../memory_strength_and_forgetting.md](../memory_strength_and_forgetting.md) decision 4 and Phase 2S-c.
- [../roadmap_1_4.md](../roadmap_1_4.md) Phase 5 (R4 credit routing; the mechanisms rule).
- `docs/agents/embodiment.md` §2 — the three pain/credit channels.
