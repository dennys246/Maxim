# Bio-memory & substrate — working brief

> Part of the CLAUDE.md satellite layer. Read this whole file before editing `src/maxim/memory/`, `decisions/`, `similarity/`, `integration/memory_hub.py`, `hivemind/`, `time/`, `agents/bus.py` (tiers/valence), or the substrate side of `imagination/`. The slim CLAUDE.md core + this brief are intended to be sufficient context for work in this area. Full incident narratives: docs/lessons/.

## 1. Mental model

The substrate is one directed data-flow chain; every invariant below pins one link of it:

```
percept text / sensor reading
  → LinguisticEncoder (similarity/encoder.py; decomposition strategy per surface:
      SpaCyNounChunkStrategy for percepts, AffordanceDecompositionStrategy for affordance names)
  → EC pattern_complete_or_separate (similarity/ec.py: match existing node vs allocate new;
      centroid update UNLESS modality is frozen)
  → ATL semantic graph + Hippocampus episodic tiers (FORMING → SHORT_TERM → LONG_TERM)
  → NAc (decisions/nac.py: reward_bias / cluster_reward_bias keyed on EC node ids,
      causal links keyed on event/outcome signatures, Welford variance on the parent aggregation)
  → SCN temporal credit (decisions/temporal_credit.py::TemporalCreditDistributor —
      the single intake fanning out to NAc eligibility, SCN bins, oscillator phases)
  → back into behavior: prompt annotations (Wire-A cluster-bias, Wire-1 variance band)
      and substrate-primary action selection (agent_loop::propose_via_substrate / NAc.recommend_action).
```

Attribution back into the chain has a strict preference order: **direct lookup key first** (`(tool_name, invocation_id)` → `NAc.record_outcome`), context similarity only as the out-of-band fallback (`record_outcome_full`, directional `_context_similarity`).

**EC threshold table — the numbers people confuse (a real incident: sensor threshold assumed 0.44):**

| Surface | Config field | Default | Notes |
|---|---|---|---|
| Text / linguistic | `ECConfig.pattern_complete_threshold` (similarity/ec.py) | **0.44** | Raised from 0.40 to stop centroid drift (L03); Roy-5 H1C boundary tracks this default |
| Sensor / drive encoding | `SensorEncoderConfig.pattern_threshold` (similarity/encoder.py) | **0.85** | NOT 0.44 — separate config class, separate surface |
| Frozen-centroid modalities | `ECConfig.frozen_centroid_modalities` | `{"interoception", "audio"}` | No running-mean centroid update; must equal hivemind `DEFAULT_FROZEN_CENTROID_MODALITIES` (test-pinned equality) |
| NAc recognition widening | `NAc.get_threshold_overrides(base_threshold=)` | tracks live EC threshold | Callers MUST pass `self.ec.config.pattern_complete_threshold`; the `None` fallback (0.44) is legacy-test-only |

Construction of this whole stack is owned by the builder family (`build_bio_stack` → `build_memory_hub` / `build_pain_bus` / etc.) — also see docs/agents/runtime-tools.md for the builders and the agent-loop call sites; docs/agents/embodiment.md for the pain/credit channels that feed NAc; docs/agents/persistence-config.md for `_format_version`, atomic writes, and stable-hash rules.

## 2. Key files

| Area | Key files |
|---|---|
| Memory | `memory/hippocampus.py`, `memory/concept_extractor.py`, `memory/store.py` (protocols), `memory/percept_trace_buffer.py` (τ-decay ring buffer) |
| Causal learning | `decisions/nac.py` (reward bias, eligibility traces, distribute_reward, **goal_reward_bias**), `decisions/causal_link.py` (CausalLink, percept_refs) |
| Temporal credit | `decisions/temporal_credit.py` (**TemporalCreditDistributor** — NAc+SCN composition, fast-decay + phase-similarity credit, **anticipatory_pre_activate**), `time/temporal_event.py` (TemporalEvent envelope), `decisions/valence_signal.py` (ValenceSignal transport for WMS salience modulation) |
| Substrate encoding | `similarity/encoder.py` (LinguisticEncoder, SensorEncoder, `_normalize_value`), `similarity/ec.py` (pattern_complete_or_separate, centroid update), `similarity/decomposer.py` (decomposition strategies) |
| Valence | `memory/episode.py` (Episode.valence, apply_hebbian_on_close, salience_spike_rule), `agents/bus.py` (propagate_valence), `memory/hippocampus.py` (capture_reaction, include_valence) |
| Cross-layer wiring | `integration/memory_hub.py` (single coordinator; `build_memory_hub`) |
| Hivemind | `hivemind/merge.py` (nac_merge/ec_merge), `hivemind/bundle.py` (ZIP bundles + scrub), `hivemind/identity.py` (quarantine heuristic), `hivemind/cli.py` (`maxim substrate` verbs) |

## 3. Invariants & lessons

- **[engineering] Memory tier progression is one-way**: FORMING → SHORT_TERM → LONG_TERM. Don't skip or reverse. WORKING is not a tier — it's an Exec-owned `WorkingMemorySet` (active reference layer). See `agents/working_memory.py`. Regression guard: [src/maxim/agents/bus.py](src/maxim/agents/bus.py) — `TierTransitionError` raised on skip/reverse attempts; tier-progression code structurally enforces direction.

- **[engineering] Hippocampus, NAc, and ATL maintain SEPARATE EpisodicMemory instances** — this is intentional coexistence, not tech debt. Don't merge. Regression guard: [src/maxim/integration/memory_hub.py::build_memory_hub](src/maxim/integration/memory_hub.py) — each bio-system takes its own EpisodicMemory instance via constructor params.

- **[behavioral] Running-mean centroid drift in cosine-similarity pattern completion.** Detection rule: when validating a new EC modality or substrate-encoding path, ALWAYS measure both isolated (fresh EC per item) and sequential (one EC, all items) — sharp disagreement = drift. Fix rule: frozen-prototype semantics (`ECConfig.frozen_centroid_modalities`) or raise `pattern_complete_threshold` (text: 0.40 → 0.44). NAc coupling: production callers MUST pass `self.ec.config.pattern_complete_threshold` to `NAc.get_threshold_overrides(base_threshold=)`. Sweeps default to 0.05 granularity; 0.01 only at a regression boundary. Full history: [docs/lessons/ec-centroid-drift.md](docs/lessons/ec-centroid-drift.md). Regression guards: [tests/unit/test_ec_centroid_drift_fix.py](tests/unit/test_ec_centroid_drift_fix.py) (4 tests pinning default + parameterization + fallback + clamp floor), [tests/unit/test_roy_5_cosine_localization.py::test_h1c_lower_bound_tracks_ec_default](tests/unit/test_roy_5_cosine_localization.py) (Roy-5 H1C boundary tracks EC default). Roy experiment: [docs/experiments/27_ec_drift_phase_4_behavioral.md](docs/experiments/27_ec_drift_phase_4_behavioral.md) + [docs/experiments/22_roy_5a.md](docs/experiments/22_roy_5a.md) (H1C boundary).

- **[engineering] Key-embedded values produce structurally-degenerate statistics.** Before adding a statistic field to an entity, list the entity's key fields and confirm the statistic varies over them; if the measured dimension is part of the key, the accumulator belongs on the parent aggregation, not the keyed entity (canonical: reward variance moved from `CausalLink` — keyed on valence-embedding `outcome_signature` — up to `NAc._event_outcome_welford`). Full history: [docs/lessons/key-embedded-degenerate-statistics.md](docs/lessons/key-embedded-degenerate-statistics.md). Regression guard: [src/maxim/decisions/nac.py](src/maxim/decisions/nac.py) — variance accumulator lives on `NAc._event_outcome_welford` (parent aggregation), not on keyed `CausalLink`; class docstring documents the move.

- **[behavioral] Context-similarity attribution: directional denominator, and direct-key-first.** (1) `NAc._context_similarity` denominator is `len(ctx1)`, not the key union — the function is directional (ctx1 = pending-event/stored-link side; ctx2 = outcome/query side; extra outcome-side keys do NOT dilute the score). If you add a new caller that needs symmetric similarity, build a separate function — do NOT touch `_context_similarity`'s denominator. (2) Context similarity is the *fallback* for out-of-band attribution; it is NEVER the right mechanism when you have a direct lookup key. If a new code path wants `record_outcome_full` + context similarity, ask first whether a direct key (e.g. `(tool_name, invocation_id)`) is available and prefer `record_outcome`. `ToolPainBridge._on_embodiment_pain` guards on `bool(self._pending_tools)` to prevent double-recording while a tool is in flight. Full history: [docs/lessons/context-similarity-directional-denominator.md](docs/lessons/context-similarity-directional-denominator.md) + [docs/lessons/direct-key-over-context-similarity.md](docs/lessons/direct-key-over-context-similarity.md). Regression guards: `tests/unit/test_nac.py::TestContextSimilarity` (7 tests) + `tests/unit/test_pain_bus.py::TestCreatePainNacSubscriber::test_pain_attributes_to_pending_action_via_context_similarity` + `tests/substrate/test_sem_pain_cascade.py` end-to-end. Roy experiment: [docs/experiments/p2_sem_pain_cascade.md](docs/experiments/p2_sem_pain_cascade.md) (end-to-end cascade validation on rusty_sword fixture; validates tool-invoked SEM affordance learning end-to-end via direct-attribution path).

- **[engineering] NAc and EC persist as a PAIR in `build_bio_stack` (`nac.json` + `ec.json`); decay-on-load lives in `NAc.load()` and NEVER in `load_state()`** (biases key on EC node ids — restoring either alone leaves them silently dangling; `load_state` stays byte-faithful for hivemind `nac_merge`). `apply_decay=False` is REQUIRED where wall-clock gaps are not agent-experienced time: the `--resume-sim` restore and read-only observers (`maxim.load.nac`, `maxim.observe`). `NAc.save()` — not `dump()` — stamps `saved_at`. The orchestrator NPC passes `AgentConfig(load_persisted=False)` (write-but-don't-read). Full history: [docs/lessons/nac-ec-persist-pair.md](docs/lessons/nac-ec-persist-pair.md). Regression guard: [tests/unit/test_nac_persistence_decay.py](tests/unit/test_nac_persistence_decay.py) (decay schedules, opt-out, bool-`saved_at`, corrupt-value coercion, recovery reset) + [tests/integration/test_cross_session_persistence.py](tests/integration/test_cross_session_persistence.py) (two-session two-process content round-trip; verified to fail on both no-persistence and a simulated save-only truncating implementation).

- **[engineering] `TemporalCreditDistributor.record_event` is the canonical intake for EVERY temporally-anchored event — do NOT build an "SCN bus"; a new consumer belongs INSIDE `record_event`.** When adding a producer: use the required fields (`event_id`, `event_type`, `event_signature`, `agent_id`, `temporal_sig` — note `temporal_sig`/`context`, NOT `temporal_signature`/`metadata`), pass the distributor explicitly (required keyword-only, `None` = explicit opt-out), and do NOT wrap the emit in a bare `except Exception` — a swallowed `TypeError` is exactly how the drive path stayed dead unnoticed (only 1 of 6 declared event_type categories has a producer; see [docs/plans/deferred/scn_event_producer_gap.md](docs/plans/deferred/scn_event_producer_gap.md)). Full history: [docs/lessons/scn-record-event-intake.md](docs/lessons/scn-record-event-intake.md). Regression guard: none yet — this documents an ABSENCE by design (missing-is-the-signal); `scripts/check_oscillator_coldstart.py` reports `drive=0` with a pointer to the plan, and the "every declared category has a producer or is marked reserved" test is the tracked follow-up that would make it enforceable.

- **[engineering] `Hippocampus.recall()` always calls `memory.touch()` on each result** and adds RECALL entries to WorkingMemorySet when `working_memory=` is provided (reconsolidation pull into active context). Content is NOT mutated on access. Regression guard: [src/maxim/memory/hippocampus_retrieval.py::RetrievalMixin.recall](src/maxim/memory/hippocampus_retrieval.py) (mixed into `Hippocampus`) + recall-side tests in [tests/integration/test_memory_hub.py](tests/integration/test_memory_hub.py).

- **[engineering] SHORT_TERM → LONG_TERM promotion is pressure-based** (Stage 7): each context-diverse recall accrues `promotion_pressure` via `_compute_access_score`; wall-clock elapsed time decays pressure (`_PRESSURE_DECAY_RATE`); threshold crossing (`_PROMOTION_PRESSURE_THRESHOLD = 3.0`) triggers promotion. Context diversity uses query-string hashing — identical queries don't accumulate. FORMING → SHORT_TERM remains outcome-triggered. The old `should_promote()` methods on `MemoryItem`/`WorkingMemoryEntry` in `agents/bus.py` have been removed. Regression guard: [src/maxim/memory/hippocampus_consolidation.py](src/maxim/memory/hippocampus_consolidation.py) (`_compute_access_score`, `_PROMOTION_PRESSURE_THRESHOLD`) + tier-progression tests in [tests/integration/test_memory_hub.py](tests/integration/test_memory_hub.py).

- **[engineering] `MemoryRecord` now carries `promotion_pressure`, `last_scored_at`, `access_contexts`** — all three fields deserialize with backward-compatible defaults (0.0, 0.0, empty deque) from old persisted data. Regression guard: [src/maxim/memory/types.py::MemoryRecord](src/maxim/memory/types.py) + [tests/integration/test_persistence_compat.py](tests/integration/test_persistence_compat.py).

- **[engineering] `Episode.valence` defaults to 0.0 on old data.** Backward compatible. Old episode dicts without the valence field deserialize cleanly. Regression guard: [src/maxim/memory/episode.py::Episode](src/maxim/memory/episode.py) (dataclass default) + [tests/integration/test_persistence_compat.py](tests/integration/test_persistence_compat.py).

- **[engineering] `spreading_activation(propagate_valence=False)` returns `dict[str, float]` unchanged.** The `propagate_valence=True` path returns `dict[str, tuple[float, float]]`. Existing callers are unaffected. Regression guard: [src/maxim/agents/bus.py::spreading_activation](src/maxim/agents/bus.py) — the `@overload` signatures document both return shapes.

- **[engineering] NAc `_reward_bias` clamps to [0, max_reward_bias].** Negative rewards (pain) produce 0.0 bias. Bias only widens EC recognition, never narrows. Pain avoidance is handled by valence annotation on edges, not by reward bias. Regression guard: [src/maxim/decisions/nac.py](src/maxim/decisions/nac.py) — clamp implemented inside `_reward_bias` update path.

- **[engineering] `BioStack.save_cerebellum()` must be called at session end.** Without it, learned forward models are lost. Regression guard: [src/maxim/runtime/bio_stack.py::BioStack.save_cerebellum](src/maxim/runtime/bio_stack.py) + session-end callers in [src/maxim/runtime/bio_stack.py::BioStack](src/maxim/runtime/bio_stack.py) (`on_session_end` path) and [src/maxim/runtime/agent_factory.py](src/maxim/runtime/agent_factory.py).

- **[engineering] NAc per-tick decay is wired into agent_loop.py section 8.5.** Both `decay_eligibility()` and `decay_reward_biases()` run every tick, conditional on `_loop_nac is not None`. Prior to 2026-04-24, these methods were defined but never called from production code — eligibility traces and reward biases persisted indefinitely. The fix is load-bearing for affordance concept transfer (SCN temporal coupling assumes traces decay). Regression guard: [src/maxim/runtime/agent_loop.py](src/maxim/runtime/agent_loop.py) section 8.5 — decay calls co-located with loop tick.

- **[behavioral] SCN temporal coupling for eligibility traces (first SCN-substrate PoC).** When fast-decay eligibility traces expire, `distribute_reward` falls back to temporal-phase similarity via `NAc._temporal_anchors` at `NACConfig.temporal_credit_weight` (default 0.3x). Session-scoped — NOT persisted; cross-session transfer uses persisted `reward_bias`; anchors prune when both trace expired AND older than `temporal_window_seconds`. Full history: [docs/lessons/scn-temporal-coupling-eligibility.md](docs/lessons/scn-temporal-coupling-eligibility.md). Roy experiment: [docs/experiments/temporal_credit_validation.md](docs/experiments/temporal_credit_validation.md) (named-experiment citation pending stricter Roy validation per the borderline note).

- **[engineering] The SCN oscillator is enabled by default in `build_bio_stack`** (`scn.enable_oscillator()` after construction); anticipatory pre-activation runs via `TemporalCreditDistributor.anticipatory_pre_activate(agent_id)` once per tick before `distribute()`; cold-start guard: <3 observations per event type → 0.0 imminence; `_event_phases` is written only under the distributor's RLock and persists via `scn.dump()`. Full history: [docs/lessons/scn-oscillator-default-on.md](docs/lessons/scn-oscillator-default-on.md). Regression guard: [src/maxim/runtime/bio_stack.py::build_bio_stack](src/maxim/runtime/bio_stack.py) (oscillator enable at construction) + [src/maxim/decisions/temporal_credit.py](src/maxim/decisions/temporal_credit.py) (TemporalCreditDistributor composition).

- **[engineering] Affordance names are encoded through a SEPARATE `LinguisticEncoder` from the percept encoder** (shared EC/ATL/NAc backing; affordance side uses `AffordanceDecompositionStrategy`, percept side `SpaCyNounChunkStrategy`). Use the `AFFORDANCE_STRATEGY` singleton for annotation lookups and the shared `_make_aff_encoder()` factory for new affordance-encoder constructions. Full history: [docs/lessons/affordance-encoder-separate.md](docs/lessons/affordance-encoder-separate.md). Regression guard: [src/maxim/similarity/decomposer.py](src/maxim/similarity/decomposer.py) (`AFFORDANCE_STRATEGY` singleton) + [src/maxim/imagination/trigger.py](src/maxim/imagination/trigger.py) (`_make_aff_encoder` factory).

- **[engineering] Signed (`[-1,1]`) sensors MUST be encoded WITH their range through `SensorEncoder`, or they FOLD** (the range-blind map aliases center 0.0 with hard-left −1.0 — left/right azimuth collapse into one EC cluster). The no-range path stays byte-identical to pre-P1; callers needing sign preserved thread `ranges={name:(lo,hi)}`. `_read_drive_ranges` and `_read_drive_states` MUST emit the same drive set (a value with no range silently re-folds), and a malformed YAML range is skipped per-sensor, never raised (a raise silently disables ALL substrate encoding). Ranges must be in the same UNITS as the values. Full history: [docs/lessons/sensor-encoding-range-aware.md](docs/lessons/sensor-encoding-range-aware.md). Regression guard: [tests/unit/test_normalize_value_range_aware.py](tests/unit/test_normalize_value_range_aware.py) (byte-identical legacy incl. the fold, monotonic range-aware, `[0,1]` identity, left/right separation, the two-walk drift guard on real reachy+infant bodies, malformed-range skip) + [scripts/orient_substrate/2_full_path_probe.py](scripts/orient_substrate/2_full_path_probe.py).

- **[engineering] `src/maxim/hivemind/` is the substrate-sharing layer.** `nac_merge`/`ec_merge` are pure functions (inputs never mutated) requiring keyword-only `left_source=`/`right_source=` routed through `_validate_source` (rejects non-strings, empties, and the reserved `_*` namespace — any new source-taking entry point must route through it). `ec_merge` respects `frozen_centroid_modalities` — do NOT bypass or mutate the default (it must equal `ECConfig.frozen_centroid_modalities`; the equality is test-pinned after a silent divergence shipped drift on `"audio"`). Bundles NEVER include hippocampus episodes, and the `nac.json` slice is content-scrubbed at composition UNCONDITIONALLY (never at capture); every ZIP entry routes through `_safe_join` (ZIP-slip). New merge entry points MUST reserve the `trusted_sources`/`validate_link`/`validate_node` parameter shape for 1.2 poison resistance; new bundle SLICES require the schema bump + migration (additive manifest keys follow the `signer_identity` `.get` precedent). Full history: [docs/lessons/hivemind-substrate-sharing.md](docs/lessons/hivemind-substrate-sharing.md). Regression guard: [tests/unit/test_hivemind_merge.py](tests/unit/test_hivemind_merge.py) (Welford parallel-merge correctness, valence-distinct-stays-separate, sorted-key determinism, frozen-modality preservation, reserved-namespace rejection) + [tests/unit/test_hivemind_identity.py](tests/unit/test_hivemind_identity.py) (short-proper-noun coverage, threshold semantics) + [tests/unit/test_hivemind_bundle.py](tests/unit/test_hivemind_bundle.py) (ZIP-slip rejection, migration-seam, float-precision survival, identity filter, end-to-end round-trip through real NAc + EC instances) + [tests/unit/test_artifact_stamping.py](tests/unit/test_artifact_stamping.py) (16 tests incl. the CLI read-from-payload seam + old-file compat both directions).

## 4. Live gotchas / known gaps

- **Isolated-vs-sequential drift detection** (from the centroid-drift lesson): any new EC modality or encoding path must be measured both ways — fresh EC per item AND one EC streaming all items. Sharp disagreement is the diagnostic signature of progressive centroid drift; do not ship on isolated-only numbers.
- **Decay is tick-anchored, not wall-clock — with ONE exception:** in-session decay (`decay_eligibility`, `decay_reward_biases`, cluster-bias tau) runs per loop tick; the only wall-clock decay is `NAc.load()`'s decay-on-load, and even that must be skipped (`apply_decay=False`) at `--resume-sim` and read-only observers.
- **SCN oscillator runs on one input channel of six:** only `tool` TemporalEvents have a producer; the drive emitter is unwired AND malformed (dies in a silent `except Exception` swallow) — do not wire the distributor without also fixing the constructor call, and vice versa. Inventory + revival criteria: [docs/plans/deferred/scn_event_producer_gap.md](docs/plans/deferred/scn_event_producer_gap.md). The dead emitter itself is embodiment-side — also see docs/agents/embodiment.md.
- **Behavioral graduation ledger:** [docs/plans/behavioral_graduation_candidates.md](docs/plans/behavioral_graduation_candidates.md) tracks which substrate claims are Earned vs pending; major changes here (encoder swaps, threshold moves, EC/NAc refactors) must check its **Re-run on:** triggers.
- **Pre-fix persisted hashes are permanently dead:** files written before the stable-hash conversion can never match recomputed hashes; loaders WARN on missing `hash_scheme` marker (see stable-hash invariant in docs/agents/persistence-config.md — the converted sites live in `similarity/signature.py`, `memory/context_index.py`, `similarity/lsh.py`, `similarity/semantic.py`, `decisions/nac.py`).
- **`MAXIM_PLACE_CODE_EXTEROCEPTION` default-ON blockers:** flipping it changes EC cluster identity for the audio channel (re-validation trigger for Exp 48 / Exp 49 H3); known follow-ups are min_confidence recalibration, the hivemind merge dim-guard (old/new-geometry audio nodes merge invisibly because audio is frozen), and an `ec` invalidate command.
- **LLM-imagined entities skip substrate encoding** (known gap; see memory file `feedback_imagined_entity_affordance_encoding_gap.md`) and imagined-entity links decay 50% at session end via `NAc.decay_imagined_links(0.5)`.

Cross-refs: docs/agents/embodiment.md for the pain/credit channel map (B8 delta-attribution, channel split, motor credit — NAc is the consumer); docs/agents/runtime-tools.md for the builder family (`build_bio_stack`/`build_memory_hub`/`build_pain_bus`), per-agent stash rules, and agent-loop tick sites; docs/agents/persistence-config.md for the autosave-under-RWLock deadlock rule (subject: Hippocampus), stable-hash, `_format_version`, and the hivemind bundle/ZIP wire format; docs/agents/simulation-experiments.md for the ablation-arm index using the env vars below.

## 4b. Naming conventions — cross-system consistency (rehomed from AGENTS.md, 2026-08-19)

When multiple systems share a functional role they **must** use the same method and
property names — the `MemoryLayer` protocol maps cleanly onto concrete
implementations only because the names match. If two modules perform the same
operation, name it identically; don't invent synonyms.

| Operation | Canonical name | DO NOT use |
|---|---|---|
| Full sleep-cycle management (compress + remove + preserve) | `sleep()` | `cleanup()`, `gc()` |
| Promote important memories to long-term | `consolidate()` | `promote()`, `commit()` |
| Store a new record | `capture()` / `store()` | `add()`, `insert()`, `create()` |
| Retrieve by ID | `get()` | `fetch()`, `find_by_id()` |
| Query by filters | `recall()` | `search()`, `query()`, `find()` |
| Persist to disk | `save()` / `load()` | `dump()`, `serialize()`, `write()` |
| Internal association graph | `graph` (property) | `_graph`, `get_graph()` |
| Layer identifier | `layer_name` (property) | `name`, `type`, `kind` |
| Record access tracking | `touch()` | `update_access()`, `mark_read()` |

**Extending:** add the canonical name here before implementing a new shared
operation; rename inconsistent legacy names during the next change to that system.

## 5. Env vars owned

- `MAXIM_SUBSTRATE_PATH` = enable LinguisticEncoder → EC → ATL dual-write = `runtime/agent_loop.py` / encoder wiring
- `MAXIM_CONCEPT_DECOMPOSITION` = enable noun-phrase decomposition before EC (needs spaCy) = `similarity/decomposer.py`
- `MAXIM_NAC_MIN_CONFIDENCE` = override `propose_via_substrate` min_confidence (default 0.3; 0.0 bypasses cold-start gate) = `runtime/agent_loop.py`
- `MAXIM_NAC_REWARD_BIAS_DISABLED` = Exp 37 ablation arm: no-op the three reward-bias surfaces (read once at NAc construction) = `decisions/nac.py` (x-ref simulation-experiments)
- `MAXIM_EC_TRACE_ACTIVATIONS` = per-tick `sim_ec_activation` JSONL events = `similarity/ec.py`
- `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION` = Wire-A cluster-bias prompt annotation off (ablation) = producer in `runtime/agent_loop.py`
- `MAXIM_DISABLE_VARIANCE_ANNOTATION` = Wire-1 variance-band annotation off (ablation) = producer in `runtime/agent_loop.py`
- `MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU` = Wire-A cluster-bias decay tau, clamped [50,1000], invalid → 300 + WARNING = `decisions/nac.py`
- `MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT` = temporal-phase fallback credit weight (default 0.3x) = `decisions/nac.py` / `NACConfig`
- `MAXIM_PLACE_CODE_EXTEROCEPTION` = audio channel emits Gaussian population code over azimuth instead of raw scalar (default OFF; see gotcha above) = `runtime/agent_loop.py::_read_exteroceptive_states` (x-ref embodiment)
- `MAXIM_HIPPO_TRACE` = per-operation Hippocampus trace events = `memory/hippo_tracer.py`
- `MAXIM_ATL_TRACE` = per-operation ATL trace events = `memory/atl_tracer.py`
- `MAXIM_NAC_TRACE` = per-operation NAc trace events = `decisions/nac_tracer.py`

Long rationales live in the pre-diet snapshot: [docs/lessons/claude-md-2026-08-13-pre-diet.md](docs/lessons/claude-md-2026-08-13-pre-diet.md).

## 6. Lesson archive

- [docs/lessons/ec-centroid-drift.md](docs/lessons/ec-centroid-drift.md)
- [docs/lessons/key-embedded-degenerate-statistics.md](docs/lessons/key-embedded-degenerate-statistics.md)
- [docs/lessons/context-similarity-directional-denominator.md](docs/lessons/context-similarity-directional-denominator.md)
- [docs/lessons/direct-key-over-context-similarity.md](docs/lessons/direct-key-over-context-similarity.md)
- [docs/lessons/nac-ec-persist-pair.md](docs/lessons/nac-ec-persist-pair.md)
- [docs/lessons/scn-record-event-intake.md](docs/lessons/scn-record-event-intake.md)
- [docs/lessons/scn-temporal-coupling-eligibility.md](docs/lessons/scn-temporal-coupling-eligibility.md)
- [docs/lessons/scn-oscillator-default-on.md](docs/lessons/scn-oscillator-default-on.md)
- [docs/lessons/affordance-encoder-separate.md](docs/lessons/affordance-encoder-separate.md)
- [docs/lessons/sensor-encoding-range-aware.md](docs/lessons/sensor-encoding-range-aware.md)
- [docs/lessons/hivemind-substrate-sharing.md](docs/lessons/hivemind-substrate-sharing.md)