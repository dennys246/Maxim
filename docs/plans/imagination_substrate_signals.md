# Imagination substrate-signal hookup

**Status:** **Hookup 1 SHIPPED** (PR pending, 2026-05-27). Hookups 2+3 remain 1.1+. Surfaced by [30_wire_a_tau_validation.md](../experiments/30_wire_a_tau_validation.md) Finding 3. Reframed 2026-05-27 — see "1.0 scope reframe" below.
**Trigger:** Roy-3a-retry NULL outcome — Wire-A annotated `sense_food_source [strongly rewarding from prior experience]` but no food entity was in scene. Imagination *should* be able to dream up a food-source entity to make the substrate-favored tool invokable, but the trigger is percept-text-bound and substrate signals (NAc, Wire-A, ATL) don't reach it.
**Target:** **1.0 (MVP-scoped: Hookup 1 only)** + 1.1+ (Hookups 2+3). See "1.0 scope reframe" below.

## W2 MVP shipment recap (2026-05-27, PR pending)

**Scope landed:** Hookup 1 ONLY (substrate-aware manifest generation). `Narrator.generate_scene_manifest()` gains optional `nac_top_biases` parameter; the AUT orchestrator passes `aut_nac.get_agent_tool_biases(agent_id=<canonical AUT id>, top_n=5)` at scene-load time.

**Substrate-voice consistency** — pre-merge two-lens review surfaced as bio-fidelity Critical finding C1 that the original implementation diverged from Wire-A's rendering shape (leaked raw floats, dropped "from prior experience" framing, emitted neutral entries unconditionally). Folded before merge by routing W2 through `compose_cluster_bias_annotation_section` directly so manifest LLM and AUT proposer LLM see the same substrate voice. The manifest-specific directive line is appended below the shared section.

**Ablation gate** — added `MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL` env-var (parallel to Wire-A's `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION`) so Roy iterations can ablate W2 without re-implementing the gate. Shares the `annotation_disabled_via_env` truthy-parser. Conftest autouse scrub added in the same commit (per CLAUDE.md "opt-in env vars need autouse scrubs").

**Agent_id resolution** — pulls from `aut_memory_hub.agent_id` (canonical AUT source, fallback `"sim_aut"`) instead of a fresh literal, so a future agent_id refactor can't silently null the lookup. Cross-confirmed M3 finding from both review lenses.

**Self-reinforcing loop (Open Question #5):** intentionally NOT mitigated in MVP. The loop closes across sessions, not within (manifest is one-shot at scene-load, biases drift during the session, next scene reads drifted biases). Empirical-grounding gate ("≥N% of past sessions") becomes a Hookup-2 prerequisite if the Roy iteration shows pathological reinforcement. Documented in the `_compose_substrate_context` docstring.

**Two substrate render surfaces by design:** W2 (manifest LLM) and Wire-A (AUT proposer LLM) both surface `_cluster_reward_bias`. Disjoint action spaces (entity selection vs tool selection) so not strict double-counting; shared renderer keeps the substrate voice unified across surfaces. Documented in the `_compose_substrate_context` docstring.

**Deferred from MVP:** drives-routed-through-manifest (homeostatic/entropic) — bio-fidelity reviewer m3 flagged as a natural Hookup-2 candidate but explicitly out of scope.

**Tests:** 9 tests in `tests/unit/test_imagination.py::TestGenerateSceneManifestSubstrateAware` pinning byte-identical no-op path, shared-renderer reuse, no-raw-float-leakage, "from prior experience" framing, all-neutral filter, ordering, and band routing.

**Next iteration:** Roy-3a-retry-W1+W2 to measure end-to-end gap closure. Per Principle 4 (cycle convergence vs divergence): if the integration iteration's primary criterion fails AND post-hoc findings spawn new follow-up plans, bird's-eye to encoder replacement rather than ship Hookups 2+3 immediately.

## 1.0 scope reframe (2026-05-27)

Originally targeted as 1.1+. Reframed during the post-Phase-C strategic discussion: **this plan is 1.0 critical path because the substrate→action conversion question is the 1.0 thesis bottleneck.** Roy-3a-retry showed imagination is substrate-blind even when Wire-A annotation is strong — the simulation can't dream the missing entity into scene, so Wire-A's signal has nowhere to land. Complement to [sense_tool_registry.md](sense_tool_registry.md): registry makes invisible tools visible, this plan makes scenes dynamically populated to match substrate preferences.

**1.0 MVP scope — Hookup 1 only (the smallest of the three candidates):**

- **Substrate-aware manifest** — pass `nac_top_biases` (output of `NAc.get_agent_tool_biases`) to `Narrator.generate_scene_manifest()` at [narrator.py:472](../../src/maxim/simulation/narrator.py). The manifest LLM call gains substrate context in its prompt so the LLM-generated manifest can include entities that activate substrate-favored tools.
- **Reuses existing pipeline** — the manifest pre-trigger runs once at scene load (already in production for cradle); this hookup adds context to its existing LLM call. No new periodic-callback machinery, no new trigger surface.
- **~20-30 LOC additive** + ~50 LOC tests. Smallest of the three Hookups by far.

**Deferred to 1.1+ (Hookups 2+3, post-Roy-iteration verdict):**
- **Hookup 2** — per-tick subscriber for missing high-bias tools (Medium-sized; checks Wire-A annotation vs active roster at LLM-submission time, requests imagination for missing tools).
- **Hookup 3** — arousal-gate relaxation for first-reaction-to-novel-percept ticks (Small).

The MVP is the cheapest path to test the strict-reading hypothesis "closing the substrate-blindness gap lets Wire-A's annotation translate to behavior." Hookups 2+3 wait on the Roy iteration's verdict (per Principle 4's two-divergence-in-a-row watch-point — if the manifest hookup alone doesn't close the gap, the next iteration is likely the trigger to bird's-eye to encoder replacement rather than ship more mechanism-layer patches).

## Front-gate scope pressure (retroactive)

Added 2026-05-27 per CLAUDE.md Principle 3.

**Question:** does this need to be its own mechanism, or can it ride on existing infrastructure?

**Existing infrastructure surveyed:**

| Candidate | Why insufficient (or sufficient) |
|---|---|
| `ImaginationTrigger.process_percept()` ([imagination/trigger.py:579](../../src/maxim/imagination/trigger.py)) | **Already the right entry point** — needs an additive `tool_bias_context` parameter, not a new trigger surface. Candidate (2) below rides here |
| `Narrator.generate_scene_manifest()` ([narrator.py:472](../../src/maxim/simulation/narrator.py)) | **Already the right entry point** for one-shot scene-load substrate injection. Candidate (1) below rides here (~20 LOC additive) |
| `StructuredContext.cluster_bias_annotations` (Wire-A render at [agent_loop.py:2864](../../src/maxim/runtime/agent_loop.py)) | **Already computes the substrate signal** — the hookup is *consume it from where it's already calculated*, not produce a new signal |
| `NAc.get_agent_tool_biases()` | Already provides the data; no new accumulator needed |
| `ComponentIndex` two-layer discovery (alias + embedding) | Solves *which* entity to design once imagination fires — wrong layer for *whether* to trigger. Stays in the design pipeline, not the trigger decision |
| Bio-pipeline pre-deliberation enrichment ([integration/bio_enrichment.py](../../src/maxim/integration/bio_enrichment.py)) | Tempting but wrong abstraction layer — enrichment shapes the LLM prompt, not the imagination trigger. Adding imagination calls inside enrichment would couple two systems that should stay layered |
| DN arousal gate ([trigger.py:706](../../src/maxim/imagination/trigger.py)) | Candidate (3) below proposes *relaxing*, not replacing — additive |

**Verdict:** could-ride-on-existing. All three highest-priority hookups are **additive parameters on existing methods**, not new mechanisms. No new bus, bridge, registry, or builder. **Caveat:** Candidate (2) introduces a new *caller* (per-tick subscriber at LLM-submission time) — itself additive wiring, not a new producer-side infrastructure, but it does change the call-frequency profile of `process_percept` and merits load-test review.

**Specific reason:** the substrate signals already exist (Wire-A output, NAc tool biases) and the imagination entry points already exist (`process_percept`, `process_manifest`). The gap is purely *additive wiring* — connecting computed signal A to consumer B. Front-gate scope pressure is satisfied without proposing new mechanism surface.

## Why this plan exists

In Roy-3a's test arm, the substrate has strong signal that `sense_food_source` is rewarding (Wire-A annotation `[strongly rewarding from prior experience]` at every LLM submission; cluster_reward_bias 0.997 → 0.753 across the arm; 660 ATL "Concept reinforced" events for `sense_food_source (action)` during priming). The simulation could supply a food entity by triggering imagination — that's the cradle precedent. But the percepts in the Roy-1 holdout fixture are pure body-sensation ("heat blooms across your fingertips", "something soft drapes against your cheek"), and imagination's regex+stop-word entity-phrase extractor finds zero entity indicators. So imagination never tries.

The runtime log makes the gap concrete:
- `Imagination: no entity phrases from 'heat blooms across your fingertips.'` ([trigger.py:604](../../src/maxim/imagination/trigger.py)) — extractor runs, returns empty.
- 74 `Imagination skipped: no percept_text (obs keys: [])` events — most ticks have no percept at all.
- 2 successful imagination resolutions (during scene-manifest pre-trigger) — designed "practice dummy for sword" and "practice a long staff" from the LLM-generated manifest. **Neither came from substrate signal; both came from the test scenario's narrator-LLM output.**

Imagination is alive but **substrate-blind**.

## The architectural smell

The imagination trigger has three entry points:

1. `process_percept(observation)` ([trigger.py:579](../../src/maxim/imagination/trigger.py)) — reads `transcript`/`raw_transcript_text`/`cli_input` from the observation dict. Substrate-blind.
2. `process_manifest(manifest_text)` ([trigger.py:837](../../src/maxim/imagination/trigger.py)) — reads an LLM-generated scene manifest once at scene-load time. Substrate-blind.
3. `SenseToolsTool._imagination_trigger.process_percept()` ([discovery.py:188](../../src/maxim/tools/discovery.py)) — passes the sense_tools query string back through `process_percept`. Substrate-blind (the query came from the LLM, not from the substrate directly).

None of these subscribe to:
- `NAc.get_agent_tool_biases()` — Wire-A's input, the agent-wide tool reward signal.
- `NAc._cluster_reward_bias` — directly.
- ATL concept activation surface.
- Hippocampus enrichment (recalls past episodes — could surface "you used `sense_food_source` 135 times in past sessions").
- The `cluster_bias_annotations` already in `StructuredContext` ([agent_loop.py:2864](../../src/maxim/runtime/agent_loop.py)) — Wire-A's render output is right there in scope but never consulted by imagination.

The result is a feedback gap. The substrate accumulates strong preferences; Wire-A surfaces them to the LLM; the LLM cannot act on them when the tool isn't in scene; imagination doesn't know to dream up the missing entity. The cradle sim covers this by hardcoding `world_entities` in the arc metadata; Roy-3a uses a generic goal string so the manifest pre-trigger generates unrelated entities.

## The 9-gate chain

Per the architecture investigation, the trigger has nine gates between "request" and "entity in scene", every gate silently skip-tolerant:

1. **Entity-phrase extraction** ([trigger.py:275](../../src/maxim/imagination/trigger.py)) — regex + hardcoded `_ENTITY_INDICATORS` (lines 132-252) + hardcoded `_STOP_WORDS` (lines 85-94). Pure-sensation language extracts `[]`.
2. **Mention threshold** ([trigger.py:683](../../src/maxim/imagination/trigger.py)) — needs ≥2 mentions of the phrase before designing. Default `imagination_threshold=2`.
3. **ImaginationCache** ([trigger.py:629](../../src/maxim/imagination/trigger.py)) — short-circuits on cache hit. A *failed* prior design also caches and prevents retry within the session.
4. **ComponentIndex two-layer lookup** ([trigger.py:650](../../src/maxim/imagination/trigger.py)) — alias hash + embedding cosine. Miss is silent (no log of near-threshold misses).
5. **DN arousal gate** ([trigger.py:706](../../src/maxim/imagination/trigger.py)) — blocks during LLM-primary high-load. Logs at DEBUG only.
6. **Energy budget gate** ([trigger.py:717](../../src/maxim/imagination/trigger.py)) — shared LLM energy lane. After ~3 AUT turns the budget exhausts and imagination starves silently.
7. **Per-phrase design guard** ([trigger.py:735](../../src/maxim/imagination/trigger.py)) — race-safety against concurrent design of the same head noun.
8. **EntityDesigner LLM call** ([designer.py:54](../../src/maxim/imagination/designer.py)) — 5 distinct return-None failure modes (schema validation, sensor sanity, dedup, etc.), none logged at WARNING.
9. **Scene registration** ([trigger.py:767](../../src/maxim/imagination/trigger.py)) — exception-swallowed silent skip if EntityMap is None.

In Roy-3a's test arm, gates 1-2 fail first (extraction returns empty, then threshold), so the rest never fire. But even if substrate-signal hookup brought a phrase to the trigger, gates 5-6 would likely block during the same high-arousal/high-energy-load test arm.

## Load-bearing invariants (DO NOT BREAK)

Surfaced by the architecture review in [30_wire_a_tau_validation.md](../experiments/30_wire_a_tau_validation.md):

1. **Per-phrase design guard** ([trigger.py:735](../../src/maxim/imagination/trigger.py)) is a concurrency safeguard against orchestrator + AUT both calling imagination simultaneously. Don't remove or weaken without an explicit replacement.

2. **Shared energy budget** ([trigger.py:717](../../src/maxim/imagination/trigger.py)) is intentional parity enforcement between AUT deliberation workload and imagination workload. Splitting the budget per-lane needs a deliberate design decision, not a drive-by tune.

3. **Cradle's substrate-blind design is by intent.** Cradle works because its arc metadata explicitly lists `world_entities`; the imagination trigger receives explicit phase entities, not substrate signals. The current substrate-blindness is not an accident — it's a layering choice to keep imagination decoupled from the substrate's accumulator-state feedback loops. Any substrate-signal hookup must preserve the *option* to run substrate-blind (cradle's pattern stays valid).

## Sketch of the contract surface

(Not a code proposal — just the contract pieces a hookup would need to nail down.)

**Three highest-priority candidate hookups** (from the architecture investigation, ranked by cost):

1. **Manifest generation is percept-blind — SMALL (~20 LOC).** Pass `nac_top_biases` (output of `get_agent_tool_biases`) to `generate_scene_manifest()` ([narrator.py:472](../../src/maxim/simulation/narrator.py)). The manifest LLM call gains substrate context in its prompt, so the manifest can include entities that activate substrate-favored tools. This is the **cheapest** path because it reuses the manifest pre-trigger pipeline (a one-shot at scene load) instead of adding a per-tick subscriber.

2. **Imagination trigger lacks substrate signal hookup — MEDIUM.** Add an optional `tool_bias_context` parameter to `process_percept()`. At LLM-submission time in the agent loop (after Wire-A annotation is computed at [agent_loop.py:2864](../../src/maxim/runtime/agent_loop.py)), check whether any top-biased tools are absent from the active tool roster. If so, request imagination with a synthetic phrase like `"The agent recalls: {tool_name} could be useful here"`. This is a per-tick subscriber, more powerful than (1) but with more design surface (when to fire? rate limits? avoid LLM cost blow-up?).

3. **Arousal gate too aggressive during percept arrival — SMALL.** [trigger.py:706](../../src/maxim/imagination/trigger.py) should allow imagination during *first-reaction-to-novel-percept* ticks regardless of arousal. Roy-3a's LLM-primary mode keeps arousal elevated continuously; the gate blocks imagination exactly when the substrate would want it firing.

These are independent and combinable. (1) is the cheapest proof-of-value; (3) is small and addresses a known mis-tuning; (2) is the most powerful but largest design surface.

**What stays substrate-blind:**

- Cradle's `_activate_phase_entities()` path ([generative_runner.py:93](../../src/maxim/simulation/generative_runner.py)) — keep as the substrate-blind reference flow. The substrate-aware hookups are *additive* alternatives that surface when arc metadata is sparse.
- The basic `process_percept` percept-text extraction — still fires when percept text contains entity indicators. The substrate hookup is a *fallback when extraction returns empty AND substrate has signal for missing tools.*

## Phasing

Not detailed at this DRAFT stage. The natural shape is:

- **Phase 0** — design pass + this plan refinement. Decide which of the three candidate hookups land first. Cost estimates + interaction with [sense_tool_registry.md](sense_tool_registry.md).
- **Phase 1** — manifest substrate-context (cheapest). Land + write a Roy-3a-variant experiment that re-runs the same fixture with a substrate-aware manifest. Pass criterion: Arm A produces ≥1 `sense_food_source` call (the original Phase 3 PRIMARY criterion).
- **Phase 2** — per-tick subscriber (Hookup 2). Larger design surface; depends on rate-limit + cost-of-imagination calibration.
- **Phase 3** — arousal-gate fix (Hookup 3). Small standalone fix; could land in parallel.
- **Phase 4** — Roy-3a re-run with all three landed. Measures the dynamic gap closure end-to-end.

## What this NOT solves

- Sense-tool LLM-visibility heterogeneity. That's [sense_tool_registry.md](sense_tool_registry.md). Complementary: that plan lets the LLM *see* what's not in scene; this plan lets the substrate dream the missing entity into scene. Either could close the Roy-3a gap independently; both together is the robust fix.
- The tick-anchored decay bio-fidelity gap. That's the planned `scn_decay_anchoring.md` (Phase C of the tau-split kickoff, not yet drafted).
- General imagination quality (entity design realism, sensor calibration). Out of scope.

## Authorization gate

Drafted as `feat/wire-a-tau-split-phase-3-validation` branch fold. Phase 0 design pass starts on explicit user authorization; not currently a 1.0 gate. If the user prioritizes this over sense-tool-registry, Hookup 1 (manifest substrate-context) is the smallest unit of validation and the natural first lift.

## Open questions

1. Is the substrate-aware manifest a one-shot at scene-load (cheaper), or should it re-evaluate when Wire-A's biases shift mid-session (more powerful, more LLM cost)?
2. When the synthetic phrase is generated (Hookup 2), should it surface to the LLM as imagination-internal-state (transparent: "I'm imagining a food source because substrate wants it") or stay hidden (LLM just sees the new entity appear)? Transparent is more honest but invites self-fulfilling-prophecy artifacts.
3. Does substrate-driven imagination need a kill switch for adversarial / red-team scenarios where the substrate has acquired pathological preferences? Likely yes — a per-session opt-out env var or config flag.
4. The architecture review flagged that the shared energy budget is load-bearing. If substrate-driven imagination becomes a busy code path, it'll starve AUT deliberation. What's the budgeting story?
5. **Self-reinforcing preference loop (bio-fidelity concern).** Phase 3 bio-fidelity review flagged that hooking NAc top-biases into the LLM-generated scene manifest closes the substrate-signal gap but risks a feedback loop: substrate prefers tool X → manifest generates entity for X → agent uses X → substrate reinforces X → next manifest is more X-biased. In biology, top-down priming is constrained by perceptual groundedness (V1's top-down signal *modulates* what's perceivable, doesn't *generate* new percepts). Maxim's manifest runs *before* the scene with no ground-truth correction. Candidate constraint: substrate-biased entities appear in the manifest only if they were present in ≥N% of past sessions (an empirical-grounding check the substrate has to earn). Audit this when Hookup 1 lands.
