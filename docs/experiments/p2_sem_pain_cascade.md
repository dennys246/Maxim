# P2 — SEM Pain Cascade (Stage 2 integration)

**Date:** 2026-04-13
**Phase:** P2 Stage 2 (end-to-end Percept → Reaction → Learning loop)
**Status:** recorded (mechanism tests green; sentence-transformers sweep deferred to Stage 3)
**Code version:** `feat/substrate-p2-finish` branch, worktree @ `.worktrees/p2/`
**Decision:** Stage 2 integration demonstrated end-to-end on real bundled `weapons/rusty_sword` component. Agent prefers `drop_weapon` over `slash` after one pain-learning cycle. Proceed to Stage 3 (sentence-transformers validation sweep + lab notebook archive).

## Hypothesis

A full SEM pain cascade — affordance use → sensor failure threshold → `PainBus.publish` → NAc causal learning → `nac.predict` → agent policy pick — can run end-to-end on the real bundled component registry with no mocks in the middle. The pre-Stage-2 codebase had every layer present individually, but a three-layer dewiring prevented any path through in production:

1. `body.py:_publish_pain` built a thin `Reaction` whose `ReactionContext.bindings` only carried `entity_path` — `source`, `failure_mode`, `sensor_readings` were stripped before any subscriber saw them.
2. `create_pain_nac_subscriber` recorded tautological `pain → pain` causal links via `nac.observe`, which could never satisfy a `nac.predict("action", ...)` query, and silently swallowed every exception.
3. `NAc._context_similarity` used `len(keys_union)` as the denominator, silently diluting legitimate matches whenever the outcome side carried more keys than the pending event — this broke every `record_outcome_full` call without `attributed_event_signature`, including the pre-existing `ToolPainBridge._on_embodiment_pain` path.

The CI tests for each layer passed because they constructed their inputs directly and bypassed the upstream lossy path. Production was silently broken.

## Methodology

### Mechanism tests — fast suite (synthetic / real-component integration)

Two test surfaces, both in `tests/substrate/`:

- **`test_components_smoke.py`** — standing YAML drift guard. Iterates every bundled `_data/components/**/*.yaml` through `ComponentRegistry.instantiate()`, reads all scalar sensors, runs `evaluate_failures()` without exception, and drives every weapon component's primary scalar-sensor failure past its trigger threshold to verify it fires. For `weapons/rusty_sword` specifically, attaches a `PainBus` subscriber and verifies the fired PainSignal carries the rich-context dialect (`source=embodiment`, `failure_mode=shatter`, `entity` contains `rusty_sword`).

- **`test_sem_pain_cascade.py`** — integration PoC. Instantiates `weapons/rusty_sword` via `ComponentRegistry`, wraps in `Embodiment`, wires `PainBus → create_pain_nac_subscriber → NAc`. A `PoCAgent` harness records actions (`record_event("action", "slash:rusty_sword", context={source, entity})`) before driving durability below the shatter threshold. The agent's policy picks between `slash` and `drop_weapon` by querying `nac.predict` and scoring by least-negative-valence.

### Fixtures

- **`src/maxim/_data/components/weapons/rusty_sword.yaml`** — bundled weapon component with `durability` sensor (initial 0.3), `combat.slash` affordance, and `shatter` failure mode triggered at `durability < 0.1` with `pain: 0.6`. No drift needed — this YAML was already aligned to the Stage 2 architecture.

### Validation sweep (slow suite, Stage 3 pending)

`TestP2ValidationSweep` in `test_p2_reward_modulation.py` requires `sentence-transformers`. Not run in Stage 2; reserved for Stage 3 once the env is ready.

### Hardware

MacBook Pro M3, Python 3.12, CPU-only. Mechanism + smoke tests run in ~4s.

## Results

### Mechanism tests

| Suite | Tests | Result | Notes |
|---|---|---|---|
| `test_nac.py::TestContextSimilarity` | 7 | PASS | Regression guards for the directional `_context_similarity`, including an inverse "outcome-extra-keys-do-not-dilute" test |
| `test_pain_bus.py::TestPainBusDirectDispatch` | 5 | PASS | Full-context delivery, unsubscribe, exception safety |
| `test_pain_bus.py::TestPainBusFallbackDispatch` | 2 | PASS | Sandbox-style direct-reaction publishes still reach direct subs; no double-delivery |
| `test_pain_bus.py::TestPainBusRefractory` | 5 | PASS | Same entity gated; **different entity NOT gated** (C1 regression guard); different failure mode NOT gated; window clears; get_stats counts direct subs |
| `test_pain_bus.py::TestCreatePainNacSubscriber` | 5 | PASS | Threshold short-circuit, attribution via context-similarity, predict returns NEGATIVE, no-match no-op, exceptions logged |
| `test_components_smoke.py::TestAllComponentsLoad` | 4 | PASS | 54 bundled components load; 4 legacy body-spec YAMLs filtered (tracked follow-up) |
| `test_components_smoke.py::TestDurabilityFailureCascade` | 2 | PASS | `rusty_sword` shatter fires rich-context PainSignal; every weapon's scalar-sensor failure mode triggers |
| `test_sem_pain_cascade.py::TestSEMPainCascadePoC` | 6 | PASS | End-to-end loop, no cross-entity leakage, repeated-pain confidence strictly monotonic |
| `test_embodiment_failures.py` | 13 | PASS | Includes 3 pre-existing `TestToolPainBridgeEmbodiment` tests that were dead-code in production pre-Stage-2 and are now exercised for real |

**Total Stage 2 surface: 49 tests, all green.** Broader fast suite: 4211 passing (2 environmental flakes unrelated to Stage 2).

### Headline PoC result

`test_agent_prefers_drop_weapon_after_learning_slash_is_painful` runs the full cascade:

```
1. Agent records slash:rusty_sword as a pending NAc event
   (context: {source: embodiment, entity: <rusty_sword_path>})
2. sword.vital_metrics["durability"] = 0.05 (below shatter threshold 0.1)
3. Embodiment.evaluate_failures() detects shatter
4. body.py:_publish_pain builds PainSignal(
       pain_type=EXTERNAL_SIGNAL, intensity=0.6,
       context={source: "embodiment", entity: <path>,
                entity_type: "weapon", failure_mode: "shatter",
                composes: [], sensor_readings: {durability: 0.05,
                sharpness: 0.5, weight: 1.2},
                entity_path: <path>}
   )
5. PainBus.publish dispatches to create_pain_nac_subscriber
   with full context (no lossy round-trip)
6. Subscriber calls nac.record_outcome_full(
       outcome_type="pain",
       outcome_signature="pain:embodiment:<path>:shatter",
       outcome_valence=NEGATIVE,
       context=<full 7-key rich context>
   )
7. NAc._record_outcome_impl walks _pending_events, computes
   _context_similarity(event_ctx_2keys, outcome_ctx_7keys)
   = matches/len(event_ctx) = 2/2 = 1.0, matches >= threshold 0.5
   → creates CausalLink slash:<path> → pain with Valence.NEGATIVE
8. agent.choose(["slash", "drop_weapon"]):
   - nac.predict("slash", ...)       → NEGATIVE (confidence ~0.55)
   - nac.predict("drop_weapon", ...) → None (no history, neutral)
   - score(slash) = -0.55, score(drop_weapon) = 0
   - choose -> "drop_weapon"
```

Repeated-pain test: over 3 iterations, `observation_count` reaches exactly 3 and confidence grows strictly: `0.55 → 0.64 → 0.67` (per the `sqrt(observation_count)` update rule in `CausalLink.record_observation`).

Cross-entity isolation: two separate `rusty_sword` instances on separate NAcs — agent A's pain learning on sword A produces no prediction for agent B on sword B.

### Root-cause fixes uncovered during the pre-merge review round

Two parallel reviewers (Executor + Architecture lenses) reviewed the Stage 2 branch pre-commit. Consolidated findings:

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | CRITICAL | `NAc._context_similarity` union-of-keys denominator dilutes rich-context outcomes. The initial Stage 2 draft worked around this with a slim 2-key attribution context in `create_pain_nac_subscriber` — a band-aid because `ToolPainBridge._on_embodiment_pain` passes 7 keys with the same bug unfixed. | Root-cause fix: denominator → `len(ctx1)` (event-side). Slim-context workaround removed. |
| 2 | CRITICAL | PainBus refractory keyed on `reaction_bus`'s `(kind, source)` collapses two distinct entities firing embodiment pain in the same 0.5s window — `pain_signal_to_reaction` synthesizes `source` from `pain_type` alone. | PainBus now has its own `(entity, failure_mode)` refractory gate, finer-grained than reaction_bus's. Regression guard: `test_different_entity_NOT_refractory_gated`. |
| 3 | CRITICAL | `test_repeated_pain_strengthens_negative_confidence` used a loose `>=` assertion that would pass even if iterations 2–3 silently no-op'd. | Tightened to strict `>`, added `observation_count == 3` assertion, added per-iteration event-count sanity check. |
| 4 | IMPORTANT | `ContextVar`-based signal stash in the original PainBus design had re-entrancy hazards (nested publishes) and invisible coupling with `@resilient` retry / async contexts. | Deleted the ContextVar entirely. PainBus now dispatches direct subscribers from `publish()` inline, then forwards to `reaction_bus` with a `_suppress_bridge` flag preventing double-delivery. |
| 5 | IMPORTANT | `test_unsubscribe_stops_delivery` didn't actually test unsubscribe — both publishes were refractory-blocked. | Fixed to use distinct `entity` kwargs so refractory doesn't hide the behavior. |
| 6 | IMPORTANT | `test_pain_below_threshold_is_ignored` used `len(nac) == 0` which wouldn't distinguish a short-circuit from a post-check refactor. | Mocked NAc, asserts `record_outcome_full.call_count == 0`. |
| 7 | IMPORTANT | `get_stats` subscriber count didn't include direct PainBus subscribers. | `get_stats` now sums `len(_pain_signal_subs) + reaction_bus.get_stats["subscriber_count"]` and exposes `direct_pain_subscribers` separately. |
| 8 | IMPORTANT | `PoCAgent` harness pretends to resolve "RC3" (production SEM action recording) but actually punts. | Added `TODO(substrate-p2-followup)` comment in the harness docstring naming the specific call sites that need wiring (`runtime/executor.py` or `embodiment/motor.py`). |
| 9 | MINOR | `_is_legacy_body_spec` reads private registry attributes. | Kept as-is with a follow-up comment in `component_registry.py::_read_component_header`. |

## How to replicate

### Fast-path (what Stage 2 ran)

```bash
cd /path/to/Maxim
python -m pytest \
  tests/substrate/test_sem_pain_cascade.py \
  tests/substrate/test_components_smoke.py \
  tests/unit/test_pain_bus.py \
  tests/unit/test_nac.py::TestContextSimilarity \
  tests/unit/test_embodiment_failures.py \
  -xvs
```

Expected: 49 tests pass, ~4s runtime on an M3 MacBook. No additional dependencies beyond the core install (no `sentence-transformers` needed).

### Full P2 substrate suite (Stage 1 reward modulation + Stage 2 pain cascade)

```bash
python -m pytest tests/substrate/ tests/unit/test_pain_bus.py \
  tests/unit/test_nac.py::TestContextSimilarity \
  tests/unit/test_substrate_recognition.py \
  -q
```

Expected: 66+ passing (Stage 1 mechanism tests at 11 + Stage 2 at 49 + legacy substrate recognition at 36 + 12 slow-suite skipped without `sentence-transformers`).

### Stage 3 — sentence-transformers validation sweep (not run in Stage 2)

When `sentence-transformers` is installed:

```bash
pip install 'pymaxim[semantic]'
python -m pytest tests/substrate/test_p2_reward_modulation.py::TestP2ValidationSweep -xvs
```

This runs the slow-suite `test_sweep_10_seeds` which produces `docs/experiments/results/p2_reward_modulation_sweep.json` with mean/std over 10 shuffled seeds at `paraphrase-mpnet-base-v2` @ threshold 0.55 and reward 2.0. Pass criteria: mean target reduction ≥30%, mean distractor interference ≤5%. Record the summary at `docs/experiments/p2_reward_modulation_sweep.md` in the P1 template style.

### Single-seed smoke against real embeddings (between mechanism and full sweep)

```bash
python -m pytest tests/substrate/test_p2_reward_modulation.py::TestP2ValidationSweep::test_single_seed -xvs
```

### Observing the cascade live (optional)

To eyeball the pain cascade wiring in a sim run with the leader model:

```bash
MAXIM_LANE_TRACE=1 MAXIM_PROVENANCE_VERBOSITY=2 \
  maxim --sim "test sword durability" \
  --language-model mistral-7b --sandbox tmpdir
```

Watch for `pain_published` log events and `_context_similarity` debug lines. Cap the run to ~60s with Ctrl+C.

## Files touched in Stage 2

Production:
- `src/maxim/decisions/nac.py` — `_context_similarity` denominator root-cause fix
- `src/maxim/proprioception/pain_bus.py` — full `PainBus` rewrite + `create_pain_nac_subscriber` rewrite
- `src/maxim/embodiment/body.py::_publish_pain` — rich-context `PainSignal` via `PainBus.publish`
- `src/maxim/embodiment/component_registry.py::_read_component_header` — legacy YAML gap follow-up comment

Tests (new):
- `tests/unit/test_pain_bus.py` (19 tests)
- `tests/substrate/test_components_smoke.py` (6 tests)
- `tests/substrate/test_sem_pain_cascade.py` (6 tests)

Tests (modified):
- `tests/unit/test_nac.py::TestContextSimilarity` (7 new regression guards)
- `tests/unit/test_embodiment_failures.py::test_embodiment_evaluates_composed` (dialect update)

Docs:
- `docs/plans/substrate_recognition.md` — Stage 2 section
- `CLAUDE.md` — 2 new load-bearing invariants (NAc similarity convention, PainBus layering)
- `docs/experiments/p2_sem_pain_cascade.md` (this file)

## Load-bearing invariants for future sessions

1. **`NAc._context_similarity` denominator is `len(ctx1)`**, not the key union. All in-file callers pass `(event_or_stored_link, outcome_or_query)` in that order. Adding extra keys on the outcome/query side must NOT dilute the score. If a new caller needs symmetric semantics, build a separate function.
2. **PainBus is the rich-context layer; ReactionBus is the typed isolation surface.** Bio-internal publishers (body, sandbox, tool failure) call `PainBus.publish(PainSignal(...))`. Direct `PainBus.subscribe` callers receive the full `signal.context`. `reaction_bus` subscribers receive the strict typed `Reaction`. Do not route rich cause-description through `ReactionContext.bindings`.
3. **PainBus refractory is per-`(entity, failure_mode)`**, not per-`(kind, source)`. Two distinct entities firing embodiment pain in the same tick must both deliver — the regression guard is `test_different_entity_NOT_refractory_gated`.
4. **`create_pain_nac_subscriber` passes the full `signal.context`** through to `record_outcome_full`, not a slim attribution dict. If a future session is tempted to re-add a slim workaround, the fix belongs in `NAc._context_similarity`, not the subscriber.
5. **No `ContextVar` signal stash in PainBus.** The original Stage 2 draft used one; it was removed because of re-entrancy hazards under `@resilient` retry and async contexts. Direct subscribers are dispatched inline from `publish()`.
