# Substrate Plan — Bio-Stack Convergence + Prompt Layer

**Status:** Active, gating 1.0
**Supersedes:** `percept_substrate_plan.md` (v1), `salience_abstraction_plan.md`, **`embodiment_voice_plan.md`** (merged in — see Track B)
**Structure:** Proof-obligation phases (P1–P6) on the substrate track, composition phases (B1–B5) on the prompt-layer track, gated by a Foundations phase (F0) and a falsification Pilot (P0).

## Goal

Prove that a bio-stack of concentrated percept flow, per-modality pattern separation/completion, hippocampal episode binding, and reward-modulated Hebbian consolidation can learn across sessions **without LLM fine-tuning** — and prove it against baselines a skeptic would actually propose, not against strawmen. Demonstrate this through extended convergence simulations that each test one behavioral claim against (a) a degenerate negative control and (b) a **plausible simple baseline** that a critic would reach for first. The bar is "significantly better than the plausible baseline," not "significantly better than random."

The prompt-layer track (Track B, formerly `embodiment_voice_plan.md`) is merged in because it touches the same text-to-prompt surface the substrate migration touches — running them as separate plans was a merge-conflict generator and the "P1 and B1 should land together" sequencing note in the old plan already admitted as much.

## How to use this plan

This is not an implementation checklist ordered by LOC. Each phase is a **proof obligation**: a specific, observable behavior that must be demonstrated by an extended convergence simulation before the next phase's work is gated.

**The execution contract:**

1. Build the minimum implementation needed to run a phase's convergence sim. Not the full system — just enough to make the sim meaningful.
2. Author the test fixtures (labeled ground truth) that the sim requires.
3. Run the sim at extended scale using **both harness tiers** (see below), seeded for reproducibility, with multiple seeds for statistical significance.
4. Compare results against both the pass criteria **and two controls**: a degenerate negative control *and* a plausible simple baseline (see "Baselines and statistical hygiene").
5. **If it passes both:** proceed. The minimum implementation stays.
6. **If it fails:** explore the phase's swap points. Re-run the sim. Iterate until it passes or you've exhausted the swap points.
7. **If all swap points fail:** that's a real finding. Revisit the architectural commitments before doing more work.

### Harness tiers

Every phase runs against two harness tiers:

- **Unit sim** — fixture-driven orchestrator (simulator_upgrades_plan S1) running a substrate-specific YAML fixture through the `ConversationalSource.inject_cli()` percept pipeline, with the mock LLM (S2) providing canned AUT responses. ≤100 turns per seed, fully seeded via `--seed` (S4), runs in <60s per seed on local hardware. Used for swap-point iteration. **Does not touch a live LLM.**
- **System sim** — full Maxim agent running a realistic scenario, 500–1000 turns, multiply seeded (≥10 seeds for most phases, ≥20 for P4). Runs the live LLM where narrative coherence matters; uses the mock LLM for deterministic mechanism tests. Catches integration failures the unit sim misses.

A phase passes only when **both** tiers pass. The unit sim proves the mechanism; the system sim proves the mechanism survives integration.

**Crucially:** neither tier is "build a bespoke harness per phase." The unit sim is a YAML fixture + metric extractor plugin running through the existing fixture-driven orchestrator. The system sim is the existing Maxim agent with a substrate-specific scenario file. Per-phase harness LOC is ~100 LOC for the metric extractor plugin, not the ~200 LOC-per-phase bespoke framework the prior plan drafts budgeted.

## Baselines and statistical hygiene

Every phase that claims a behavior runs against a **degenerate negative control** (random assignment, zeroed modulation, coin-flip boundaries) — this catches gross wiring bugs. A phase that fails this has a plumbing problem, not an architectural one.

On top of that, some phases have a **plausible baseline** — a simple alternative a skeptic would reach for first. Plausible baselines get used in one of two ways:

- **As a gate** where the baseline attacks the same claim the architecture makes. The architecture must beat the baseline by `baseline_mean + 2×baseline_std` across ≥10 seeds. **Applies to P3a (TF-IDF vs Hebbian episode binding), P4 (OpenCLIP shared space vs hippocampus-mediated binding — the 1.0-gating head-to-head for commitment #3), and P6 (LRU vs graded decay).**
- **As a regression check** where the baseline exists on adjacent territory but doesn't attack the same claim. The architecture should not be *catastrophically* below the baseline, but beating it is not the point — the architecture's value is proven in later phases. **Applies to P1 (FAISS+cosine), P2 (unrewarded baseline of the architecture itself), and P3b (metadata-grep).**

The distinction: on a gate phase, losing to the baseline invalidates the architectural commitment. On a regression-check phase, a large regression indicates a mechanism bug but losing-by-a-little is fine.

**Effect sizes, not p-values.** Before a gate phase runs, measure the baseline across ≥10 seeds, publish mean + std, and pin the architecture's pass threshold at `baseline_mean + 2×baseline_std` *before* the architecture is evaluated. Pre-registration prevents post-hoc threshold massaging. No p-values, no Bonferroni — for small-sample engineering validation, frequentist hypothesis testing is theater. Raw effect sizes are the right tool.

**Absolute numbers alone are not evidence.** A phase that hits 90% in isolation tells us nothing without at least the degenerate control for comparison. Absolute thresholds are a minimum viability gate, not the whole test.

## Core architectural commitments

These are the invariants the phases test. They should not be re-litigated without a failed phase forcing the conversation.

### 1. Concentrated flow, not fan-out
All modalities normalize to the same `Percept` structure and flow through one pipeline. Bio-systems subscribe to the unified stream and never receive modality-specific inputs. Adding a new modality = adding one encoder.

### 2. Per-modality computation, shared storage
EC (computation — pattern separation/completion) operates per-modality via tag filtering. One `EntorhinalCortex` class, no subclasses. ATL is the node store: every node is tagged by modality, within-modality edges live in ATL filtered by tag. No separate "modality graph" data structure.

### 3. Hippocampus is the only cross-modal binder
ATL never has a cross-modality edge. Cross-modal recall works through hippocampus episode reconstruction, not shared embedding space.

### 4. NAc modulates recognition, does not gate it
EC does recognition. NAc modulates EC's threshold per node via reward bias maintained on eligibility traces. Recognition stays automatic; reward selectively sharpens it.

### 5. SCN tags episodes, does not do credit assignment
SCN provides circadian phase tags for context retrieval. Credit assignment uses eligibility traces at seconds-to-minutes timescales.

### 6. Communication channels are context, not modality
SMS, email, Slack, direct speech, narrative text from a DM — all of these are **TEXT modality** with different channel metadata. They share the same ATL text nodes and the same linguistic encoder. What differs is channel metadata in `Percept.context`, per-channel episode boundary rules, a priori salience, and retrieval filters. See the "Communication channels" section below.

## Current state — verified against code

Ground-truthed against the repo (not the plan's prior aspirations):

- **[`Percept` dataclass](../../src/maxim/agents/bus.py)** already has `embedding`, `sensory` (rich SensoryTag with submodality/spatial/entity), `salience`, and `metadata: dict`. **What's missing is schema enforcement:** no TEXT/VISION enum, no conventional `context` keys (`channel`, `sender`, `thread_id`, `timestamp`), no `scn_tag`. The previous plan said "none exist" — it was wrong. This is a migration, not a green-field field addition.
- **[`EC`](../../src/maxim/similarity/ec.py)** is a pure LSH + inverted-index wrapper. No `pattern_complete_or_separate`, no threshold modulation, no per-modality dispatch. Accurate.
- **[`ATL`](../../src/maxim/memory/atl.py)** is a concept graph with no modality fields on nodes and no edge-type enforcement by modality. Accurate.
- **[`Hippocampus`](../../src/maxim/memory/hippocampus.py)** stores `EpisodicMemory` objects but has **no Episode dataclass, no co-occurrence tracking, no Hebbian link state**. Persistence via `atomic_write_json` already works ([hippocampus_persistence.py](../../src/maxim/memory/hippocampus_persistence.py)).
- **[`NAc`](../../src/maxim/decisions/nac.py)** has Rescorla-Wagner. **It does not have eligibility traces** — the prior plan was wrong about this. No per-event trace decay, no τ parameter, no per-node reward bias keyed by ATL node ID. P2 builds all of this.
- **Text input bypasses the percept layer entirely.** `transcript_chunk` flows straight into [prompt_builder.py](../../src/maxim/agents/prompt_builder.py) and [context_pool.py](../../src/maxim/runtime/context_pool.py). **No `LinguisticEncoder` exists.** This is the biggest single gap and the highest-risk change in the plan.
- **No `VisionEncoder` / CLIP integration.** `test_vision_engine.py` is detection + IoU tracking, not embedding.
- **No substrate fixtures directory** in `tests/fixtures/`.
- **`atomic_write_json`** is already wired through ATL, hippocampus, semantics, cross_layer — P3.5 is mostly NAc save/load plus a cross-layer round-trip harness.

## F0 — Foundations wave (split into [foundations_plan.md](foundations_plan.md))

The foundation fixes that used to live in this document as F0.1–F0.8 have been split into their own plan to keep substrate_plan focused on proof-obligation phases. See [foundations_plan.md](foundations_plan.md) for the eight items (NAc wiring + save/load signature, PerceptTraceBuffer, NarrativeModulator ghost removal, Percept context schema, agent_id threading + SCN race fix, Percept factory consolidation, tier transition assertions, Sensor→Percept contract), their dependency order, and per-item exit criteria.

> **⚠ F0 naming collision with peer_leader_flexibility_plan.** `peer_leader_flexibility_plan` also uses "F0" for its own foundation wave (filelock primitive, storage reporter, leader pinning — already landed on main as commits `411d2c0` and `b705504`). **That's a different F0 from this plan's F0.1–F0.8.** When you see a commit message like `feat(runtime): F0 foundation wave`, check which plan it belongs to before assuming it's the substrate foundation. The substrate F0 is "bug fixes and structural refinements unblocking substrate P1+"; the peer_leader F0 is "filelock + storage + leader pinning infrastructure for the routing path." No file overlap between the two waves. The naming is entrenched on both sides and a rename would be churn, so disambiguation lives in reader vigilance.

**Why they block P1+:** each item fixes a real bug or plugs a load-bearing gap the substrate phases would otherwise silently build on top of — bugs we'd only find mid-P2 or mid-P3 if we let the phases run first. The wave is ~1,130 LOC across eight small PRs and takes ~2 weeks.

**Gate:** P0 pilot does not open until all eight items have landed and the previously-failing `test_record_plan_outcome` is green. See [foundations_plan.md](foundations_plan.md) for the exit criteria of the wave as a whole.

## Persistence as cross-phase contract

**Rule:** every phase's unit sim ends with a subprocess round-trip smoke test. Serialize whatever state the phase added, spawn a subprocess via `subprocess.Popen`, reload, re-run a held-out 10% probe from the same fixtures, verify the probe's output matches pre-shutdown within tolerance.

Each phase picks its own probe when it gets there — the rule is uniform, the details are phase-specific. P3.5 is still the full round-trip certification at scale; P5 stress-tests it; every other phase uses the smoke test to catch "my new component has unserializable state" on day one instead of three phases later.

**Cost:** ~50 LOC per phase, shared utility from simulator_upgrades_plan S3.

**Why this matters:** the 1.0 claim is cross-**session** learning. If persistence is not a continuous obligation from P1 forward, the claim is built on faith until P3.5.

## P0 — Fixture-Difficulty Pilot (formerly plausible-baseline falsification)

Before committing to the substrate architecture, run the cheapest possible sanity check: **are the P1 fixtures hard enough to tell us anything?** The prior draft of this phase framed it as a falsification test with a pass bar. That framing was wrong — FAISS+cosine doesn't attack the same claim the architecture makes, so it can't *falsify* the architecture (see "Baselines and statistical hygiene" for the full argument). What it *can* do is measure whether the fixtures are easy enough for a trivial baseline to solve. If they are, the fixtures are too easy and need to be harder before P1 can mean anything.

**Hypothesis:** P1's paraphrase-cluster fixtures are difficult enough that a trivial sentence-transformer + cosine-similarity + fixed-threshold baseline does *not* trivially solve them. If the baseline hits ≥85% paraphrase collapse, the fixtures are too easy — not because the architecture is wrong, but because the test isn't testing anything.

**Steps:**
1. Author P1's fixtures first (`tests/fixtures/substrate/paraphrase_clusters.yaml`, ≥50 clusters, 2–3 days using the sim-as-fixture-debugger workflow — see "Fixture authoring workflow" below).
2. Implement the trivial baseline against `BenchmarkRunner`'s existing `baseline_path` hook. `sentence-transformers` embeds each sentence (run both `all-MiniLM-L6-v2` and `all-mpnet-base-v2`), stored in FAISS, cluster membership by cosine threshold. ~100 LOC of baseline module + ~20 LOC of BenchmarkRunner wiring.
3. Run it on the fixtures via the fixture-driven orchestrator from [simulator_upgrades_plan.md](simulator_upgrades_plan.md) S1. Publish mean + std over 10 seeds.
4. **Decision gate:**
   - **Baseline ≥85%:** the fixtures are too easy. Author harder clusters (more paraphrase variation, more near-miss distractors) and re-run. The pilot is not a plan-killer; it's a fixture-quality gate.
   - **Baseline 60–85%:** fixtures are well-calibrated. Proceed with P1, register the baseline score as P1's **regression check** — not a pass bar, just a floor.
   - **Baseline <60%:** fixtures may be too hard. Check that they are solvable by a careful human reader; if yes, proceed.

**Why this is cheap:** the fixtures are P1's fixtures anyway. The baseline is ~100 LOC. With the sim upgrades in place, running it takes ~30 seconds per seed on local hardware. Total cost is ~2–3 days including fixture authoring and ~0 cloud cost.

**Exit:** Published baseline score (both sentence-transformer models), decision recorded in the plan, P1's sanity-floor threshold written into this document, fixtures calibrated to be neither trivially-easy nor unsolvable.

**Dependencies:** foundations_plan, simulator_upgrades_plan. The sim upgrades are what make this pilot cheap; without them, P0 is back to "bespoke harness from scratch."

## Fixture authoring workflow — sim as fixture debugger

Fixture quality is load-bearing for the proof-obligation framing. A rushed fixture is worse than a delayed phase — rushed fixtures give false-positive passes. The workflow leverages the simulator as a fixture debugger, not just a test runner:

1. **Rough draft** (1 day). Author a first-pass fixture with ~50% confidence.
2. **Sim replay + inspect** (1 day). Run the rough fixture through the fixture-driven orchestrator (S1). Inspect `aut_hippocampus.json`, `aut_nac.json`, and ATL state snapshots. Did the fixture do what you intended? The sim tells you what the substrate actually sees, not what you *think* it sees.
3. **Refine** (0.5 day). Where the substrate did something unexpected, the fixture entry was usually ambiguous. Rewrite, re-run, iterate.
4. **Freeze before implementation starts.** Once the fixture behaves the way you intended, freeze it. Don't edit it mid-phase. If you find a real problem in the fixture during implementation, re-run P0-style baseline calibration on the revised version before continuing — not a silent tweak.

**Total per-phase fixture cost: ~2.5 days** (down from the original ~3–4 day estimate), with higher quality because the sim is giving feedback instead of you authoring in the dark. This workflow is only possible once the sim upgrades land.

**Critical discipline:** the frozen 60% must not be edited after a phase's implementation starts. If you find the holdout is wrong mid-phase, that's a sign to re-run P0-style fixture-difficulty calibration, not to silently fix the holdout. False positives come from editing the holdout to match what the implementation happens to produce.

## Communication channels as episode scaffolds

Each communication channel has natural structure that maps to episode boundaries and retrieval metadata. The plan treats them uniformly as TEXT percepts with channel-specific context.

| Channel | Episode boundary | Sender identity | Latency | Default salience signal |
|---|---|---|---|---|
| **SMS** | Contact + temporal gap (>30min closes) | Phone number / contact | Seconds–minutes | High if contact is known; low for unknown senders |
| **Email** | `In-Reply-To` thread chain | From address | Minutes–days | Subject keywords + sender reputation |
| **Slack** | Thread if in-thread; otherwise channel + burst window (e.g., 10min of activity) | User ID | Seconds | `@mention` → high; DM → high; channel broadcast → medium; reaction-only → low |
| **Direct speech** (sim/DM) | Scene change signal from orchestrator | NPC ID or `self` | Real-time | Scene-driven |
| **Narrative** (DM prose) | Scene change signal | `narrator` | Real-time | Stake-driven (DM can tag) |
| **Self-authored** (agent's own output) | Inherits from the channel it's sent to | `self` | Inherits | Same as the channel it targets |

**Incoming vs outgoing:** the agent's own speech is a TEXT percept with `sender=self`. Outgoing messages flow through the linguistic encoder before being sent. This is load-bearing: the agent can retrieve its own past statements via the same mechanism it retrieves others' speech. Without this, the agent cannot track its own commitments against incoming messages.

**Long-latency credit assignment — honest framing:** Email and async Slack create gaps longer than the eligibility-trace decay τ. When a reply arrives three days later, the trace is gone. The architecture's claim is that the reply pattern-completes to the concept's ATL node, retrieves the original episode via hippocampus, and the retrieved context becomes the current context. **This is not a free validation** — it depends on P3 retrieval already working. P3b's channel-integration sim tests this pathway explicitly rather than assuming it.

**Salience:** channel salience feeds WhereCoord (the salience layer) as an a priori weight on `Percept.salience`. NAc's reward-based threshold modulation is orthogonal — NAc learns from outcomes, channel salience is prior-based. Both inputs affect EC's threshold.

---

## Track A — Substrate phases

### P1 — Stable within-modality recognition under controlled paraphrase

**Hypothesis:** The EC + modality-tagged ATL pipeline collapses paraphrases of the same labeled referent to a single stable ATL node. The mechanism behaves as designed (threshold modulation + pattern completion do real work), the behavior survives a persistence round-trip, and the result is not catastrophically worse than a trivial baseline.

**Scoping change from the prior draft.** P1 no longer uses "beats the plausible baseline by 2 std" as a pass criterion. The plausible baseline (FAISS + cosine) attacks sentence similarity, which is not what the architecture is designed for — the architecture's value is in enabling *downstream* behavior (cross-modal binding, episode retrieval, reward-gated recognition), proven in later phases. Holding P1 to a FAISS bar is evaluating a cognitive architecture as a sentence embedder and is unfair to commitment #2.

**Instead, P1 has mechanistic criteria plus a sanity floor.** The mechanism must do what we designed it to do; the sanity floor exists only to catch gross mechanism bugs. The real "does the architecture earn its complexity" test is **P4**, where the baseline (CLIP shared space) attacks the exact claim the architecture makes.

**Convergence simulation:** ≥50 paraphrase clusters, 10–30 sentences each, no reward, no episodes, no hippocampus binding.

**Pass criteria (all gates must fire):**
- **Paraphrase collapse:** ≥90% of within-cluster presentations activate the same node.
- **Cluster distinctness:** ≤5% of distinct clusters collapse into shared nodes.
- **Node stability:** node count plateaus; <10% growth over the final 20% of the run.
- **Modality isolation:** in a mixed-modality probe (adding ~10 non-text percepts to the stream), no text cluster collapses into a non-text node. Tests that commitment #2 is actually enforced.
- **Persistence round-trip:** per the cross-phase contract — serialize ATL, subprocess reload, re-run a held-out 10% of clusters, ≥95% of pre-shutdown node activations preserved.
- **Sanity floor:** architecture's paraphrase collapse rate is within 5 pp of the FAISS + cosine baseline from the P0 pilot. Being lower than the baseline is acceptable if the mechanism is sound; being *much* lower flags a mechanism bug and is a fail.
- **Beats degenerate negative control** (random node assignment) by >30 pp.
- Report mean + std across ≥10 seeds. Effect sizes in raw units. No p-values.

**Why no p-values here:** this is small-sample engineering validation, not inferential statistics. "Beats baseline by 2 std across 10 seeds" is the right tool for this scale. Frequentist hypothesis testing on small samples is theater.

**Plausible baseline (sanity floor only, not pass bar):** sentence-transformer embedding + FAISS cosine top-1 + fixed threshold. ~100 LOC. Scored in P0 pilot and kept in-tree for ongoing sanity-floor checks.

**Negative control:** random node assignment (every percept → fresh random node).

**Swap points (if the architecture fails to beat the plausible baseline), in order of increasing cost:**
1. Similarity metric — cosine → euclidean → learned metric over frozen embedding (cheap).
2. Pattern completion threshold — static per-modality → adaptive-per-node (cheap).
3. Encoding granularity — whole sentence → noun-phrase span → head word + dependency context. Pipeline change, budget a day per swap.
4. Embedding model — sentence-transformers → syntactically-aware encoder → dedicated entity encoder.
5. Add shallow coreference resolution pre-encoder. ~a week; real scope expansion.

**Test fixtures:** `tests/fixtures/substrate/paraphrase_clusters.yaml` — ≥50 hand-authored clusters with labeled referents and paraphrase families. **These are authored during the P0 pilot** and re-used here.

**Minimum implementation:**
- F0.4 `Percept` schema + F0.6 factories (prereqs)
- `LinguisticEncoder` producing `Percept(modality="text")` — lands via B1 (Track B), see interleave note
- `EntorhinalCortex.pattern_complete_or_separate(percept, modality)` returning activated or new node
- `ATL` with modality-tagged nodes, tag-filtered queries, edge enforcement
- **Text-to-prompt migration** — see dedicated section below
- **`SimulationReport.phase_metrics` extension** (~30 LOC): add a `phase_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)` to [`SimulationReport`](../../src/maxim/simulation/report.py) so metric extractor plugins can write per-phase sections (`phase_metrics["p1"] = {...}`) without editing report.py for every future phase. Lands as part of P1 since P1 is the first phase that writes metrics; subsequent phases populate their own keys.
- **P1 metric extractor plugin** (~100 LOC): computes paraphrase collapse rate, cluster distinctness, node count stability from ATL state snapshots, writes to `phase_metrics["p1"]`
- Fixture YAML for the unit sim + scenario file for the system sim (authored via the P0 sim-as-fixture-debugger workflow)
- FAISS sanity-floor baseline wired through `BenchmarkRunner.baseline_path`

**Dependencies:** foundations_plan, simulator_upgrades_plan, P0, B1+B2.

### P2 — Reward-modulated recognition sharpens rewarded nodes

**Hypothesis:** After a reward event credited to node X, near-miss percepts that previously pattern-separated now pattern-complete to X. Recognition radius expands for behaviorally relevant stimuli and decays when reinforcement stops. Per-agent isolation holds: rewarding a node for agent A does not affect agent B's recognition of the same concept.

**Scoping change from the prior draft.** P2 also moves to mechanistic criteria — same rationale as P1. The "static top-K-neighbor merge" baseline is a credible alternative *implementation* but not a credible alternative *claim*, so it no longer gates. The real proof that reward-modulated recognition earns its complexity is **the combination of P4 (cross-modal binding) and behavioral scenarios captured in the convergence practice log** — not a sentence-cluster collapse rate.

**Convergence simulation:**
Seed near-duplicate paraphrases of a target concept plus distractors. Baseline pass without reward; count distinct nodes. Reset. Apply a reward event to the target's first presentation. Run the rewarded condition. Count nodes.

**Pass criteria (all gates must fire):**
- **Rewarded-node collapse:** ≥30% fewer distinct nodes in the rewarded cluster vs unrewarded baseline.
- **Non-interference:** distractor node count matches the unrewarded baseline within ±5%.
- **Decay:** recognition radius returns toward baseline over a defined timescale after reinforcement stops, driven by `PerceptTraceBuffer` decay.
- **Per-agent isolation (F0.5 verification):** run the same sim with two agents. Rewarding agent A's target produces no change in agent B's node count for the same concept.
- **Persistence round-trip:** serialize ATL + NAc per-node bias + `PerceptTraceBuffer` snapshot, subprocess reload, re-run rewarded probes, verify bias still modulates EC threshold.
- **Beats degenerate negative control** (α = 0, NAc reward bias forced off) — should show no recognition-radius change at all.
- **Sanity floor:** architecture's rewarded-node collapse delta is not negative vs an unrewarded run of the architecture itself. (Reward should sharpen, not blur.)
- Report mean + std across ≥10 seeds. Effect sizes in raw units.

**Static top-K-neighbor merge (reference implementation only, not a gate):** kept in-tree as a reference point in the convergence practice log. If a future behavioral scenario shows the architecture doing something the top-K merge cannot, that's evidence for the complexity. The sentence-cluster metric is not where that evidence lives.

**Negative control:** NAc reward bias forced to 0 (α = 0).

**Swap points:**
- NAc reward bias decay rate (τ)
- Threshold modulation strength (α)
- Eligibility trace timescale
- Per-node vs per-cluster threshold modulation
- Reward magnitude scaling

**Test fixtures:** reuse P1's paraphrase clusters, add reward-event annotations to a subset.

**Minimum implementation:**
- P1 plus F0.2 `PerceptTraceBuffer` (shared trace buffer, not NAc-owned) plus
- NAc per-node reward bias keyed by `(agent_id, node_id)` (F0.5 prereq — multi-agent correctness)
- NAc reads from `PerceptTraceBuffer` when crediting reward events
- EC threshold formula: `threshold = base - α × nac.reward_bias(agent_id, nearest)`
- **Reaction abstraction Phase 5** (folded from [reaction_abstraction_plan](reaction_abstraction_plan.md)): NAc causal link table gains structured `percept_refs: tuple[TraceSnapshot, ...]` column so queries can run by percept involvement, not just hash match. Per-node reward bias keys off `(agent_id, node_id)`.
- **P2 metric extractor plugin** (~100 LOC): rewarded-node collapse delta, distractor non-interference, decay timescale, per-agent isolation verification
- Fixture YAML reusing P1 paraphrase clusters with reward annotations on a subset

**P2 proof-of-concept: SEM-to-SEM pain cascade simulation**

A hard PoC verifying the Percept→Reaction→Learning loop works end-to-end through SEM entity interaction. Uses the sword component (`_data/components/weapons/`) and the agent's body as two interacting SEM entities.

*Scenario:* Agent wields a sword with a `durability` sensor and a failure mode at `durability < 0.1, pain: 0.8`. Over 3 encounters:

1. **Encounter 1 — Use while healthy.** Sword durability starts at 1.0. Agent uses `slash` affordance 3 times. Durability decreases per use. No failure mode fires. NAc observes neutral outcomes. Hippocampus records successful actions.

2. **Encounter 2 — Use into failure.** Durability crosses the 0.1 threshold. `Embodiment.evaluate_failures()` triggers → `Reaction(kind="pain", intensity=0.8, source="embodiment:external_signal")` flows through ReactionBus. NAc learns `(slash_sword, low_durability_context) → NEGATIVE`. CascadeResolver propagates sensor changes to the body entity (pain reading increases on the body's proprioception sensor).

3. **Encounter 3 — Avoidance test.** Sword is re-presented at low durability. Agent is offered `slash` vs `drop_weapon`. NAc prediction for `slash` in low-durability context should return NEGATIVE valence with confidence > 0.5. The bio-stack should prefer `drop_weapon` (or at minimum, suppress `slash` confidence relative to Encounter 1).

*Pass criteria:*
- **Pain reaction fires** at the correct threshold (durability < 0.1). Verify via `ReactionBus.history(kind="pain")`.
- **NAc learns context-conditional avoidance:** `nac.predict(event_signature="slash_sword", context={"durability": "low"})` returns a prediction with `valence=NEGATIVE` and `confidence > 0.5` after Encounter 2.
- **Hippocampus records the pain episode** with salience > 0.8 (from `create_pain_memory_subscriber` intensity boost).
- **Contrast with healthy state:** same `nac.predict` with `context={"durability": "high"}` returns NEUTRAL or POSITIVE, showing the avoidance is state-conditional, not a blanket weapon aversion.
- **Persistence round-trip:** save NAc + Hippocampus state after Encounter 2, reload into fresh subprocess, Encounter 3 avoidance behavior survives.

*Implementation:* ~150 LOC fixture YAML + ~80 LOC test harness using S1 fixture orchestrator. The sword component already exists; the test wires it through AgentFactory → CerebellumModulator (with reaction_bus) → ReactionBus → NAc bridge → prediction query. No new SEM types needed — the PoC exercises the existing infrastructure under adversarial conditions.

*Why this belongs in P2:* it tests reward-modulated recognition (P2's claim) through a concrete embodied scenario rather than synthetic paraphrase clusters. If the agent can learn "don't slash with a degraded sword" purely from bio-stack state (no LLM fine-tuning, no hand-coded rule), that's a concrete demonstration of the substrate's learning capability. The PoC also stress-tests the reaction_abstraction Phase 2 infrastructure (ReactionBus, CerebellumModulator emission, pain bridges) against a real scenario.

**Dependencies:** P1, F0.2 (PerceptTraceBuffer), reaction_abstraction Phases 1–4 (all landed), simulator_upgrades_plan.

### P3a — Episode binding produces retrieval on partial cue (synthetic only)

**Split from old P3.** Tests the *mechanism* — Hebbian link formation and partial-cue retrieval — on synthetic episodes with hand-authored ground truth. No channel rules. If this fails, the mechanism is wrong. If this passes and P3b fails, only the channel rules are wrong.

**Hypothesis:** Nodes co-occurring in the same hippocampus episode form durable links; presenting a partial cue retrieves the others, **by a margin greater than a TF-IDF bag-of-concepts baseline**.

**Convergence simulation:** 100 hand-authored synthetic episodes with explicit ground-truth co-occurrence structure. Probe with partial cues, measure retrieval precision/recall.

**Pass criteria (all gates must fire):**
- **Minimum viability:** precision >0.70, recall >0.70 against ground truth; node count per episode stays in 5–50 range.
- **Beats the TF-IDF gate baseline:** F1 exceeds `baseline_mean + 2×baseline_std` of the TF-IDF baseline. Hebbian episode binding and TF-IDF co-occurrence are attacking the same task — this is a fair head-to-head.
- **Beats degenerate negative control** (coin-flip episode boundaries) by a large margin.
- **Persistence round-trip:** per the cross-phase contract.
- Report mean + std across ≥10 seeds.

**Plausible baseline:** bag-of-concepts TF-IDF over episode windows, retrieval by cosine match between cue and episode vector. ~150 LOC.

**Swap points:**
- Hebbian link strength function
- Link decay rate
- Consolidation trigger
- Retrieval mechanism — direct episode lookup → spreading activation → top-K weighted

**Test fixtures:** `tests/fixtures/substrate/synthetic_episodes.yaml` — 100 synthetic episodes with labeled co-occurrence. 1–2 days authoring.

**Minimum implementation:**
- P2 plus
- `Episode` dataclass: `id`, `start_tick`, `end_tick`, `channel`, `sender_ids: set[str]`, `thread_id: str | None`, `activated_nodes: set[NodeId]`, `reward_events: list[RewardEvent]`, `scn_tag: CircadianContext`
- Hippocampus episode store with episode-to-node edges
- Generic (not channel-specific) episode boundary via tick gap + explicit scene signal
- Hebbian within-ATL edge updates on episode close
- Retrieval path

**Dependencies:** P2.

### P3b — Channel integration: episode boundary rules + filtered retrieval

**Split from old P3.** Tests that per-channel boundary rules produce useful episodes on realistic data. If this fails and P3a passed, the architectural claim is safe but the channel configuration is wrong — tunable, not fundamental.

**Hypothesis:** Per-channel episode boundary rules (SMS contact+gap, narrative scene change) produce episodes whose channel-filtered retrieval ("what did X say about Y") **beats a metadata-only filter baseline**.

**Convergence simulation:** 100 realistic episodes drawn from SMS + narrative direct speech. Ground truth is author-labeled: which messages belong to which conversation, which concepts are co-mentioned. Probe: given a concept cue, retrieve episodes containing it, filtered by `sender` or `channel`.

**Pass criteria (all gates must fire):**
- **Minimum viability:** precision >0.70, recall >0.70 on channel-filtered retrieval.
- **Specificity under overlap:** when two episodes share a concept, cueing retrieves both.
- **Regression check vs metadata-grep baseline:** architecture's F1 is not catastrophically below the grep baseline. The baseline doesn't attack the same claim, but a large regression would indicate a mechanism bug.
- **Persistence round-trip:** per the cross-phase contract.
- **Beats degenerate negative control** (random episode boundaries) by a large margin.
- Report mean + std across ≥10 seeds.

**Plausible baseline (sanity floor only):** on a cue, return every message containing a substring match for the cue, filtered by sender/channel metadata. This is "what grep gives you."

**Swap points:**
- Per-channel boundary rules, tuned independently (SMS gap window, narrative scene signal)
- Within-channel burst detection (Slack)
- Thread chain traversal (email)

**Test fixtures:** `tests/fixtures/substrate/channel_episodes.yaml` — 100 realistic SMS + narrative episodes with channel metadata, sender labels, co-mention ground truth. 1–2 days authoring.

**Minimum implementation:**
- P3a plus
- Per-channel episode boundary rules (SMS + narrative only for 0.3)
- SMS and narrative channel adapters (narrative exists in sim runtime; SMS is a ~100 LOC fixture-reader stub)
- Retrieval filter by `sender` / `channel`

**Dependencies:** P3a.

### P3.5 — Basic cross-session persistence

**Hypothesis:** ATL nodes, hippocampus episodes, NAc reward biases, and channel episode structure survive serialization/deserialization. A reloaded system recognizes the same nodes, retrieves the same episodes, and respects the same reward biases as it did pre-shutdown.

**Why here and not P5:** the 1.0 claim is cross-**session** learning. Without persistence landing before P4, P4's "session 2" is actually "same process, new test" — weaker than the 1.0 claim. P3.5 is minimum-viable persistence: save, load, verify identity. Decay and GC land in P6.

**Most of the work already exists.** `atomic_write_json` is wired through ATL, hippocampus, semantics, and cross_layer. NAc has partial save/load but needs the new per-node reward bias fields and the eligibility trace state added. The main new work is a cross-layer round-trip harness, not the serialization itself.

**Convergence simulation:**
Run P3b's channel-integration sim through to episode close. Serialize the full bio-stack state with `atomic_io`. **Spawn a new Python process via `subprocess.Popen`**, not a new object in the same interpreter. Load state from disk. Re-run only the retrieval probes. Verify results match pre-shutdown.

**Pass criteria (no plausible baseline — this is a round-trip test):**
- Node identity: every pre-shutdown node has a matching post-reload node with the same `node_id`.
- Edge weights match pre-shutdown within floating-point tolerance.
- Episode retrieval works identically (precision/recall within 2%).
- NAc per-node reward biases round-trip (requires foundations_plan F0.9 NAc signature alignment).
- `PerceptTraceBuffer` state round-trips (even if decayed to zero on reload, the decay is deterministic).

**Known exclusions from the round-trip contract (not failures):**
- **ATL callbacks are not persisted.** `_on_concept_captured` and `_on_concept_deleted` at [atl.py:83-84](../../src/maxim/memory/atl.py#L83-L84) are live `Callable` objects that cannot pickle; this is by-design per the iceberg sweep audit. Callers re-register callbacks post-load. The P3.5 harness explicitly treats callbacks as a known exclusion — a missing callback post-reload is not a test failure.

**Swap points:**
- Persistence format: JSON → msgpack → structured store
- Load order: ATL → hippocampus → NAc, or a different sequence
- What to persist: all nodes / top-K by connectivity / only consolidated

**Minimum implementation:**
- P3b plus foundations_plan F0.1 (NAc save/load signature already aligned as part of F0.1)
- NAc save/load covering per-node reward bias (signature already aligned)
- Cross-layer round-trip harness using simulator_upgrades_plan S3 subprocess harness
- Checksum verification
- **Schema-versioned snapshot protocol** — see sub-section below

### P3.5.1 — Schema-versioned snapshot protocol (sub-section of P3.5)

**Why this belongs in P3.5 and not a post-1.0 refactor:** P3.5 is already building cross-layer round-trip serialization from scratch. Every bio-system already rolls its own `save()`/`load()`, which F0.1 partially aligns for NAc but doesn't unify across ATL, Hippocampus, NAc, SCN, and the new `PerceptTraceBuffer`. Building *just enough* of a unified snapshot protocol as part of P3.5 is strictly cheaper than shipping five divergent persistence implementations and retrofitting schema versioning later when a field change silently invalidates saved sessions.

**This is load-bearing for the 1.0 research claim.** The claim is *cross-session learning*. Saved sessions *are* the data of that claim. If a field change in 0.5 silently breaks every session saved in 0.4, the empirical base erodes without anyone noticing — which is the kind of bug that kills a research project not because it crashes, but because the evidence trail quietly rots.

**Minimum protocol** (defined as part of the incremental contracts layer — see "Contracts layer" section):

```python
class BioSystemSnapshot(Protocol):
    schema_version: int

    def dump(self) -> dict[str, Any]: ...

    @classmethod
    def load(cls, state: dict[str, Any]) -> Self: ...

def migrate(old_state: dict, from_v: int, to_v: int) -> dict: ...
```

Every bio-system (ATL, Hippocampus, NAc, SCN, `PerceptTraceBuffer`) satisfies the Protocol. A new `SessionSnapshot` composes all bio-system snapshots into one atomic round-trippable unit:

```python
@dataclass
class SessionSnapshot:
    schema_version: int
    atl: dict[str, Any]
    hippocampus: dict[str, Any]
    nac: dict[str, Any]
    scn: dict[str, Any]
    trace_buffer: dict[str, Any]
    agent_id: str
    saved_at: float  # wall-clock for audit, not for logic
```

Loading a `SessionSnapshot` checks `schema_version` against the current code's expected version. If older, call `migrate()` with explicit from→to version pair. If newer, fail loud — don't guess.

**Minimum migration machinery** (not a full migration framework):
- A single `migrate()` function per bio-system that dispatches on `from_v → to_v` pairs
- Initial version is `1` for every bio-system; no migrations needed at P3.5 time
- The *first* field change in 0.4+ adds the first migration (trivial, ~20 LOC per bio-system)
- Migration functions are pure: `dict → dict`, no side effects, no imports from the rest of the substrate
- A test harness that loads a sample 0.3 snapshot and verifies it works after a hypothetical 0.4 change

**What this is not:**
- Not a plugin discovery system (that's [deferred/bio_system_plugin_plan.md](../deferred/bio_system_plugin_plan.md))
- Not a generic migration framework like Alembic (too heavy, YAGNI)
- Not backwards-compatible serialization for arbitrary versions (we only support migration forward, one version at a time)
- Not a session diff/merge tool (post-1.0 if ever)

**Scope:** ~300 LOC net. This adds to P3.5's budget but *replaces* the ad-hoc persistence alignment that was already implied by P3.5's "cross-layer round-trip harness" bullet. The net cost of doing this properly vs. ad-hoc is ~100–150 LOC extra, and the schema versioning is free insurance against the class of bug that would otherwise kill the cross-session claim.

**Exit:**
- `BioSystemSnapshot` Protocol defined in `src/maxim/contracts/biosystem.py` (alongside the incremental contracts layer work)
- All five bio-systems implement the Protocol
- `SessionSnapshot` round-trips through the subprocess harness
- Unit test loads a hand-crafted v1 snapshot and verifies every field survives
- Unit test loads a hypothetical v2 snapshot with a migration stub and verifies the migration runs
- Unit test loads a v3 snapshot (unknown future version) and fails loud

**Dependencies:** P3b, foundations_plan F0.1, simulator_upgrades_plan S3, and the `BioSystemSnapshot` Protocol (which this sub-section defines — the contracts-layer work accretes here rather than waiting).

### P4 — Cross-modal binding via hippocampus

**Hypothesis:** Nodes of different modalities co-occurring in the same episode can cue each other across modality boundaries through episode reconstruction, **by a margin greater than a shared-embedding-space baseline**. This is the architecture's central claim.

**Vision encoder is real, not toy.** A scalar-state toy encoder with a small finite value set produces a trivially-passable test. Minimal vision (single-object CLIP embedding, ~100 LOC) exercises the claim honestly.

**The plausible baseline matters more here than anywhere else.** P4's claim is that hippocampus-episode binding is *superior to* a shared-embedding-space approach. If a CLIP-text-and-CLIP-vision shared space (projected into one space, cosine retrieval) solves the mug test at the same rate, the entire "no shared space, bind through episodes" commitment (#3) is rhetorical. P4 must beat this baseline to justify commitment #3.

**Convergence simulation — the mug test, persisted:**
Session 1: run a scripted sim where a text `"mug"` percept and a vision percept of a mug co-occur in multiple episodes from the narrative channel. Apply a reward event in one of them. Save state (uses P3.5). **Spawn a new Python process via `subprocess.Popen`**, load state from disk. Session 2: present text `"mug"` alone. Measure whether the vision mug node is retrieved. Run the symmetric test (vision → text).

**Pass criteria (all gates must fire, and P4 is where the gates bite hardest):**
- **Minimum viability:** forward retrieval >80% of seeded trials; reverse retrieval >80%; false-binding rate <10%; rewarded retrieval margin >15% over non-rewarded.
- **Beats the OpenCLIP shared-embedding-space gate baseline:** architecture's forward + reverse retrieval F1 exceeds `baseline_mean + 2×baseline_std` of the CLIP baseline. **This is the whole justification for commitment #3.** Losing here is a plan-ending finding, and the fallback in "If the whole thing fails" is what we go do next.
- **Beats degenerate modality-agnostic control** (single pool, no modality tags) by a large margin.
- **Persistence round-trip:** the mug test *is* a persistence test — session 1 writes, subprocess loads, session 2 probes.
- Report mean + std across ≥20 seeds. P4 doubles the seed count of earlier phases because it is the 1.0-gating phase and small effect sizes here are not acceptable.

**Plausible baseline:** text and vision both embedded via OpenCLIP into its shared space. Retrieval by cosine top-K across the joint pool. No hippocampus, no episodes, no modality tagging. ~150 LOC. **This is the critic's first move, and the phase exists because commitment #3 is worthless if this baseline wins.**

**Negative control:** modality-agnostic lookup ignoring modality tags.

**Swap points:**
- Episode reconstruction mechanism (direct lookup → spreading activation → weighted top-K)
- Number of co-occurrences for durable cross-modal link (1 → 3 → 5)
- Cross-modal link strength function
- Retrieval threshold

**Test fixtures:** ~30 scripted episodes with known text-vision pairings, small object set (mug, cup, bowl, plate), labeled ground truth.

**Minimum implementation:**
- P3.5 plus
- Minimal `VisionEncoder` using CLIP (single-object image → embedding → `Percept(modality="vision")`)
- Cross-modal retrieval path: text cue → ATL text node → hippocampus episode lookup → reconstruct episode → retrieve vision nodes
- Symmetric vision-cue path
- Mug test harness with **real subprocess boundary**
- CLIP shared-space baseline implementation for ongoing re-runs

**Dependencies:** P3.5. **This phase is the 0.3 → 0.4 gate.**

### P5 — Robust cross-session persistence under stress

**Hypothesis:** The bio-stack state survives serialization across varied content distributions, high node counts, dense episode graphs, and concurrent channel activity.

**Convergence simulation:**
Long-running mixed-channel sim (SMS + email + Slack + narrative + vision) to 10,000+ nodes and 1,000+ episodes. Serialize every 100 episodes. Reload at each checkpoint. Verify retrieval quality does not degrade across reloads.

**Pass criteria:**
- State size stays bounded and explicable (linear or sub-linear in node count).
- Retrieval precision/recall stable across ≥10 save/reload cycles.
- Load time <5s for 10k nodes.
- Checksum verification on every node and edge.

**Plausible baseline:** same workload run without any pruning or consolidation — just append-only. If retrieval F1 is the same, the consolidation logic isn't doing work.

**Negative control:** skip persistence entirely (in-process continuous run).

**Swap points:**
- Persistence format (if P3.5 picked JSON, P5 may force msgpack for size)
- Incremental vs full writes
- Compression
- Index structures at load time

**Dependencies:** P4.

### P6 — Extinction without reinforcement

**Hypothesis:** Associations not reinforced decay predictably. Reinforced associations persist. The system forgets appropriately.

**Temporal model (new — this was undefined before):** simulated time advances via **explicit SCN phase ticks**, not wall clock. One "simulated session" = one SCN cycle, driven by a fixture-specified tick schedule. Wall-clock time is not used anywhere in the decay path. This keeps tests deterministic and fast.

**Convergence simulation:**
Long-running sim with two node groups: Group A receives periodic reinforcement, Group B does not. Advance SCN phase per schedule to simulate multi-session duration (leveraging P5 persistence). Measure Hebbian link strength, retrieval probability, node presence.

**Pass criteria (all gates must fire):**
- Group A retrieval stays high (>80% of initial).
- Group B retrieval drops below 20% within N simulated sessions.
- No catastrophic forgetting: Group A not collaterally damaged by Group B decay.
- Node count bounded: orphaned Group B nodes pruned after decay.
- **Beats the LRU gate baseline:** the architecture's retention curve for Group A exceeds `baseline_mean + 2×baseline_std` of the LRU baseline under the same capacity constraint. If LRU matches, graded decay adds nothing and the mechanism is unjustified.
- **Persistence round-trip with decay state:** verify decay schedule round-trips deterministically — a subprocess reload resumes decay from the same point.
- Report mean + std across ≥10 seeds.

**Plausible baseline:** LRU cache with fixed capacity. Evict least-recently-accessed nodes when full. ~80 LOC.

**Negative control:** no decay (Hebbian strength frozen).

**Swap points:**
- Decay function shape (linear / exponential / power)
- Per-tier decay rates (FORMING / SHORT_TERM / LONG_TERM)
- Pruning threshold
- Reinforcement semantics (full reset vs partial)

**Dependencies:** P5.

### P8 — Minimum-viable sleep replay and consolidation

**Why this exists.** Forgetting (P6) is half of what biological consolidation does. The other half is *active strengthening* of rewarded associations during offline (sleep) phases, without new input. Hippocampus-to-cortex replay is the mechanism by which episodic memory becomes semantic memory in brains. A substrate that never consolidates will show stale performance over long-session use — the agent remembers what happened but never *learns from what it remembers when offline*. P8 is the minimum mechanism that proves offline learning can happen at all; the research program on top of that mechanism lives in [memory_consolidation_practice.md](../deferred/memory_consolidation_practice.md).

**Scoping discipline:** P8 is **deliberately not ambitious**. One replay strategy, one scheduling rule, one measurable improvement. Alternative strategies, episode-selection tuning, wake-vs-sleep trade-offs, hippocampus-to-ATL promotion rules, and interference management all live in the consolidation practice doc, not here. This phase exists to ship a mechanism; the living doc exists to refine it.

**Hypothesis:** During an explicit sleep phase (SCN-driven tick), replaying the top-N most-rewarded episodes from the current session strengthens within-episode Hebbian links, and retrieval F1 on those episodes measurably improves after sleep — **without new input between the pre-sleep and post-sleep measurements**.

**Convergence simulation:**
Run a P3b-style channel-integration sim to populate ATL + hippocampus + NAc. Measure retrieval F1 on a held-out probe set (pre-sleep). Advance SCN to a sleep phase. Run the replay loop: for the top-N rewarded episodes, replay their activation sequences through EC + ATL with Hebbian link updates enabled; do not emit any new percepts. Measure retrieval F1 on the same probe set (post-sleep). Compare.

**Pass criteria (all gates must fire):**
- **Improvement without input:** post-sleep retrieval F1 on replayed-episode probes exceeds pre-sleep F1 by a pre-registered margin (baseline of the phase is the pre-sleep score itself — this is a within-subject comparison).
- **No collateral damage:** F1 on non-replayed probes stays within ±5% of pre-sleep. Replay should strengthen, not overwrite.
- **Determinism:** the replay loop is deterministic for a given seed + state. Running it twice on the same pre-sleep state produces identical post-sleep state.
- **Persistence round-trip:** serialize post-sleep state, subprocess reload, verify replay-strengthened links survive.
- **Beats degenerate negative control** (replay without Hebbian updates — should show no F1 change).
- Report mean + std across ≥10 seeds.

**No plausible baseline.** P8's claim is "the mechanism works at all." There isn't an "alternative replay" to compete with at this minimum-viable scope — that's what the living doc is for.

**Swap points (if P8's minimum mechanism doesn't work):**
- Replay source: top-N by reward → top-N by Hebbian centrality → top-N by recency
- Replay count N
- Hebbian update strength during replay (same as online, or scaled down)
- Sleep-phase tick budget (how many "cycles" of replay per sleep phase)

**If the minimum works, *everything else* about replay goes into the practice doc.** Alternative strategies, production tuning, interference analysis, wake-replay, hippocampus-to-ATL promotion rules — all living-doc territory.

**Test fixtures:** reuse P3b fixtures. P8's probe set is held out from P3b's training set; no new fixtures required for the minimum phase.

**Minimum implementation:**
- P6 plus
- Explicit SCN sleep-phase detection (tick-driven)
- Replay loop: episode selection → activation sequence → EC + ATL with Hebbian updates, no percept emission
- Pre/post measurement harness
- Determinism test

**Dependencies:** P6 (decay must exist for replay to make sense — otherwise you're just strengthening everything).

---

## Track B — Prompt layer (merged from embodiment_voice_plan)

Track B was previously a sibling plan. It's merged here because B1's PromptAssembler and P1's text-to-prompt migration touch the same files and the same hot path. Running them as separate tracks against the same surface was a merge-conflict generator. Track B's phases land interleaved with Track A per the roadmap.

### B1 — PromptAssembler (single composition point) — 0.3

One class that takes structured inputs and produces the final system message. Replaces the four scattered prompt locations (hardcoded strings in narrator, inline persona strings, template `.txt` files, YAML one-liners) with a composable pipeline.

```python
PromptAssembler.compose(
    identity: Persona,           # who the character IS
    sensors: SensorState,        # what they perceive right now (from SEM)
    affordances: list[Action],   # what they can do (from SEM + tool registry)
    scene: SceneContext,         # what's happening around them (from DM)
    memory: MemorySummary,       # relevant recalled episodes/concepts (from MemoryHub)
    coach: ActingCoach | None,   # meta-guidance layer (B3)
) -> SystemMessage
```

**Files touched:** new `prompts/assembler.py`, refactor [agents/prompt_builder.py](../../src/maxim/agents/prompt_builder.py) to delegate, deprecate ad-hoc injection in [prompts/prompt_profiles.py](../../src/maxim/prompts/prompt_profiles.py).

**Exit:** All NPC and planning-agent system messages flow through `PromptAssembler.compose`. `grep -r "system_message = f\""` returns nothing outside the assembler. `MemorySummary` consumes P1's ATL output.

**Interleave with Track A:** B1 lands **together** with P1's text-to-prompt migration. The migration section below is the combined work order.

**Scope:** ~500 LOC refactor.

**Dependencies:** F0.3 (NarrativeModulator ghost removed), F0.4 (Percept schema).

### B2 — (folded into F0.3)

The `NarrativeModulator` ghost removal landed in F0 rather than Track B because it's a prereq for everything else. The NPC persona-through-prompt rewire lands with B1.

### B3 — Acting Coach layer — 0.4

Meta-prompt scaffold around identity. Gives the LLM explicit instructions on role values, speech register, failure modes, continuity contract. Optional per-character — simple guards don't need one, campaign-critical NPCs do.

**Files touched:** new `prompts/acting_coach.py`, campaign YAML schema extension.

**Exit:** Blind A/B test: acting-coach NPC behavior is measurably more consistent across a multi-turn encounter than the same NPC without one (deterministic seed + fixed scene).

**Dependencies:** B1.

### B4 — Replanning with failure diagnosis — 0.4 (gates 1.0)

Rewrite [replanning.txt](../../src/maxim/_data/prompts/planning/replanning.txt) with real structure: failure point, observed evidence, prior replan attempts (not stateless), root cause hypothesis, alternative approaches, selected approach + rationale, revised plan. Persist replan attempts in-session.

**Exit:** Induced failure scenario — first plan fails deterministically; second plan differs structurally; third attempt does not repeat either earlier approach. This feeds Track A's P3b channel-retrieval path (past failure episodes become retrievable cues).

**Dependencies:** B1, P3a (episode retrieval of prior attempts).

### B5 — Embodiment/narrative separation — 0.4

Formalize SEM → embodiment inputs, DM → narrative inputs, PromptAssembler → composition. Add a lint-style contract test that narrative modules don't reach into SEM internals and vice versa.

**Dependencies:** B1.

---

## Text-to-prompt migration (dedicated risk section)

**Why this has its own section:** this is the single highest-risk change in the plan. Text content is carried on `Percept.transcript_chunk: str | None` at [bus.py:126](../../src/maxim/agents/bus.py#L126), but it never flows through any encoder — consumers pull it directly and format it into prompt lines. The field is *on* the Percept; the *processing* bypasses the substrate layers. The P1 claim requires text to flow through `LinguisticEncoder → Percept(with embedding + ATL node ref) → EC pattern completion → ATL → MemorySummary → PromptAssembler`. Botching this breaks the live agent runtime — every sim, every interactive session.

**Concrete starting points (from code audit — preserved so a cold-start session can find the current state without re-doing the investigation):**

**Current producers of `Percept.transcript_chunk`** (where text enters the system today):
- [runtime/agent_loop.py:736](../../src/maxim/runtime/agent_loop.py#L736) — main loop sets `transcript_chunk=transcript` on percepts during the perception phase
- [agents/perception_agent.py:223](../../src/maxim/agents/perception_agent.py#L223) — `PerceptionAgent` constructs percepts with transcript content; also has explicit `transcript_chunk=None` paths at lines 312, 360 for non-text percepts

**Current consumers** (the four places that pull the string directly and bypass substrate processing):
- [agents/prompt_builder.py:380-381](../../src/maxim/agents/prompt_builder.py#L380-L381) — `if percept.transcript_chunk: lines.append(f'Heard: "{percept.transcript_chunk[:200]}"')`. This is the main prompt-layer consumer — the one B1's PromptAssembler will replace.
- [agents/context_pool.py:255-256](../../src/maxim/agents/context_pool.py#L255-L256) — `if percept.transcript_chunk: parts.append(f'Heard: "{percept.transcript_chunk[:100]}"')`. Same pattern, different truncation. Second prompt-layer consumer.
- [runtime/sim_adapter.py:56-63](../../src/maxim/runtime/sim_adapter.py#L56-L63) — sim adapter reads `transcript_chunk` to build observation dicts for the sim runtime. This is *not* a prompt-layer consumer; it's the sim's observation pipeline. **Leave this alone during migration** — the sim adapter can continue reading the string field as long as producers still set it.
- [runtime/skill_matcher.py:162-163](../../src/maxim/runtime/skill_matcher.py#L162-L163) — skill matcher reads `transcript_chunk` for tool-selection heuristics. Same "not prompt-layer" status — leave it alone during migration.

**Migration approach: keep `transcript_chunk` as-is, add the substrate path alongside it.** The migration is *additive* to the Percept schema — introduce a new field for the ATL node reference (tentatively `Percept.text_node_id: NodeId | None = None`), run the text through `LinguisticEncoder` to populate both the embedding and the ATL ref, and let `PromptBuilder` / `context_pool` progressively shift from reading `transcript_chunk` to reading from `MemorySummary`. The two consumers that are *not* prompt-layer (`sim_adapter`, `skill_matcher`) keep reading `transcript_chunk` and don't need to change.

**Why this is lower-risk than the prior draft implied:** the earlier plan draft said "text currently flows straight from `transcript_chunk` into the prompt builder, bypassing the percept layer entirely." That was half-right — the field *is* on the Percept, but the *processing* bypasses the substrate. Once you see the call graph concretely (2 prompt-layer consumers + 2 non-prompt-layer consumers), the migration is: leave the 2 non-prompt consumers alone, dual-write for the 2 prompt consumers, flip the flag. **It's four call sites, not a cross-cutting refactor.** The dual-write / shadow-read / cutover ceremony still applies, but the blast radius is contained.

**Open question for the implementer:** should the `LinguisticEncoder` run synchronously during percept construction in [perception_agent.py](../../src/maxim/agents/perception_agent.py) (blocking the percept producer on an embedding call), or asynchronously (emit the percept immediately, populate `text_node_id` via a callback)? Synchronous is simpler but adds latency to the perception path. Asynchronous is more complex but preserves the existing perception timing. Default to synchronous for the migration — optimize later if the latency shows up in sim metrics.


**The migration must land with a feature flag and dual-write:**

1. **Phase 1 — dual path (behind flag).** Introduce `LinguisticEncoder`, route text through it *in addition to* the existing direct-to-prompt path. Both write; only the legacy path reads. Run the full test suite + a short sim. Verify parity.
2. **Phase 2 — shadow read.** PromptAssembler reads from `MemorySummary` (substrate path) alongside the legacy path. Compare outputs on a fixed seed; log divergences. Hold until divergence is understood, not tolerated.
3. **Phase 3 — cutover.** Flip the flag. Substrate path is authoritative. Legacy path still writes for rollback.
4. **Phase 4 — legacy removal.** Once a full release cycle passes with no rollbacks, delete the legacy path.

**Rollback:** flipping the flag must fully revert behavior. An integration test pins this: same seed + same input → same prompt under the legacy flag.

**Parity test:** a fixed-seed sim produces a prompt byte-for-byte identical between legacy and dual-write phases. This is the hard gate before touching Phase 2.

**Scope:** ~600 LOC combined (encoder + dual-write plumbing + shadow-read diff + flag wiring). This is *on top of* B1's 500-LOC assembler refactor. Budget accordingly.

## Version path / roadmap

The `0.3` gate includes three prerequisite waves (Cleanup, foundations_plan, simulator_upgrades_plan) plus the P0 pilot and B1+P1 combined migration before substrate phases start. This sequencing is what makes the substrate phases cheap by the time they run.

| Version | Phases that must pass | What it proves |
|---|---|---|
| **0.2.2** | Cleanup Wave (ships first — removes friction from the hot path B1+P1 will rewrite) | CLI rot cleared |
| **0.3-pre** | [foundations_plan](foundations_plan.md), [simulator_upgrades_plan](simulator_upgrades_plan.md), **P0** fixture-difficulty pilot, **B1+P1** combined migration | Foundations solid; substrate phases cheap to run; fixtures calibrated; text flows through percepts end-to-end |
| **0.3-minimum** (fallback if scope runs long) | Everything in 0.3-pre plus **P1, P2, P3.5** | Mechanism + reward modulation + persistence certification proven. Enough for a defensible version bump even if P3a/P3b/P4 slip to 0.3.1. |
| **0.3-target** | 0.3-minimum plus **P3a, P3b, P4** (real cross-modal, OpenCLIP head-to-head — the 1.0-gating head-to-head) | Architecture's mechanism works, survives persistence, beats head-to-head baselines, cross-modal binding proven across real process boundary |
| **0.4** | P4 re-passed with production vision + email/Slack channels, **B3 Acting Coach**, **B4 Replanning** (gates 1.0), **B5 embodiment/narrative separation** | Architecture generalizes; NPCs coherent; replanning recovers from failure |
| **0.5** | P5 (stress persistence), P6 (extinction vs LRU), **P8** (minimum-viable sleep replay — the mechanism, not the research program) | System persists under load, forgets appropriately, actively strengthens rewarded associations offline. Consolidation lives as a biological-class mechanism, not just a metaphor. |
| **1.0** | Stress-test sim combining all phases with full channel diversity; B4 passing; [behavioral_convergence_practice.md](../deferred/behavioral_convergence_practice.md) and [memory_consolidation_practice.md](../deferred/memory_consolidation_practice.md) seeded with their first rounds of experiments | Cross-session learning without fine-tuning, at realistic scale, with coherent voice, with ongoing research programs for behavior change and consolidation refinement |

**0.3-minimum vs 0.3-target:** the plan explicitly allows a partial 0.3 to ship as a version bump if P3a/P3b/P4 slip. This prevents the "everything-or-nothing" trap where any slippage looks like failure. **Slipping P3a/P3b/P4 to 0.3.1 is normal re-planning, not plan failure.** The reason P3.5 is in 0.3-minimum and P3a/P3b are not: P3.5 is a correctness contract (serialization round-trips identically), which can be tested on P1/P2 state alone; P3a/P3b add episode binding, which is architecturally later.

**Demo rule:** if a wave ends without a runnable demo, the wave isn't done. No moving on until you can show it. This is the forcing function against "I'll come back to it later." Each wave's demo is obvious from its exit criteria — if you can't point at a command that runs and a file you can read, the wave isn't done.

### Why each wave exists

**Why Cleanup Wave ships first:** [cleanup_wave.md](cleanup_wave.md) touches the CLI + prompt-bootstrap hot path that B1+P1 will rewrite. Clearing it first removes a full class of merge conflicts. Cleanup Wave is small (~800 LOC, mostly deletions), near-done, and not on the critical path — which is exactly why it should ship before the critical-path work begins.

**Why foundations_plan is its own wave:** the foundation fixes are load-bearing for every substrate phase and were wrongly assumed to exist in the prior plan. Fixing them as a clean block with its own CI gate is cheaper than discovering them mid-P2. See [foundations_plan.md](foundations_plan.md).

**Why simulator_upgrades_plan is its own wave:** the sim upgrades drop per-phase harness cost from ~200 LOC to ~100 LOC, unlock deterministic testing without live LLMs, and enable the persistence subprocess harness that every subsequent phase depends on. ~1 week of work that saves ~1.5 weeks across the substrate phases, plus unlocks local-only substrate validation (no cloud cost). See [simulator_upgrades_plan.md](simulator_upgrades_plan.md).

**Why only P0 as a pilot (not P0 + P4-mini):** earlier drafts had two pilots. P4-mini was cut because its "minimum cross-modal binding" wasn't actually testing commitment #3 — it was testing "does OpenCLIP win at vision retrieval on a toy fixture," which the literature already answers. The real test requires real substrate state (hippocampus episodes, reward-gated binding, persistence) and happens at P4 proper. P0 remains as a single fixture-difficulty pilot; the OpenCLIP baseline is authored during P0 and carried forward into P4 so the number is pinned before P4 starts, but there is no separate "feasibility" gate.

**Why B1 merges into P1:** they touch the same files. See the dedicated migration section.

**Channel rollout across versions:**
- **0.3:** SMS + narrative
- **0.4:** Email + Slack
- **0.5:** Any remaining channels + multi-channel stress tests

## Scope honesty — updated

**Note:** sim harness LOC dropped significantly after the [simulator_upgrades_plan](simulator_upgrades_plan.md) audit found that `ConversationalSource.inject_cli()`, `ScenarioSource`, `BenchmarkRunner.baseline_path`, and the session-report builder already provide ~80% of the harness infrastructure. Per-phase harness work is now ~100 LOC for a metric extractor plugin, not ~200 LOC for a bespoke framework. The net scope savings (~1,100 LOC of harness) mostly offset the cost of the sim upgrades themselves.

| Wave | Item | Scope | Notes |
|---|---|---|---|
| **Prereq** | [foundations_plan](foundations_plan.md) (F0.1–F0.7) | ~1,010 LOC | Blocks substrate phases. See separate plan. |
| **Prereq** | [simulator_upgrades_plan](simulator_upgrades_plan.md) (S1–S4) | ~800 LOC | Fixture orchestrator, mock LLM, persistence subprocess harness, deterministic seeding. Blocks P0. |
| **Prereq** | **P0** fixture-difficulty pilot | ~100 LOC (FAISS baseline wired into `BenchmarkRunner`) + ~150 LOC (OpenCLIP baseline carried forward into P4) + ~100 LOC metric extractor | Uses sim upgrades. Also pins the OpenCLIP baseline number before P4 starts. |
| **0.3** | **B1+P1 combined migration** | ~1,100 LOC (500 assembler + 600 text-to-prompt dual-write) | Critical path. See dedicated migration section. |
| 0.3 | P1 substrate additions | ~300 LOC (EC pattern completion + ATL modality tagging) + ~100 LOC metric extractor | |
| 0.3 | P2 | ~300 LOC + ~100 LOC metric extractor | Uses F0.2 trace buffer + F0.5 agent_id threading. |
| 0.3 | P3a | ~400 LOC (Episode + Hebbian + consolidation) + ~100 LOC metric extractor | |
| 0.3 | P3b | ~250 LOC (channel rules + SMS stub) + ~100 LOC metric extractor | |
| 0.3 | P3.5 | ~200 LOC (NAc save/load + cross-layer round-trip) + ~100 LOC metric extractor | Uses S3 subprocess harness, not bespoke. |
| 0.3 | P4 | ~500 LOC (real vision encoder + cross-modal path + mug harness) + ~100 LOC metric extractor | OpenCLIP baseline reused from P0. |
| **0.4** | B3 Acting Coach | ~300 LOC | |
| 0.4 | B4 Replanning | ~400 LOC | Gates 1.0. |
| 0.4 | B5 separation | ~150 LOC | |
| **0.5** | P5 | ~400 LOC + ~100 LOC metric extractor | |
| 0.5 | P6 | ~300 LOC + ~100 LOC metric extractor | |
| 0.5 | **P8 minimum-viable consolidation** | ~350 LOC + ~100 LOC metric extractor | Replay mechanism only; research program lives in consolidation practice doc. |
| All phases | Persistence round-trip plugin per phase | ~50 LOC × 9 phases = ~450 LOC | Uses S3 harness; per-phase wiring only. |
| — | Fixture authoring time | ~2.5 days × 9 phases | Sim-as-fixture-debugger workflow. |

**Totals:**
- **Prereqs (foundations + sim upgrades + P0):** ~1,980 LOC (foundations dropped from ~1,410 to ~1,130 after F0 simplification; P4-mini cut)
- **0.3:** ~3,250 LOC (includes B1+P1 migration)
- **0.4:** ~850 LOC
- **0.5:** ~1,350 LOC
- **Grand total to 1.0:** ~**7,430 LOC system + ~1,100 LOC metric extractor plugins + ~3 weeks of fixture authoring time**

The LOC shrinkage comes from the simplification pass: F0 cut from 10 items to 8, F0.8 trimmed from ~320 to ~100 LOC, P4-mini cut as a separate pilot. The remaining LOC is higher leverage — the sim upgrades pay dividends on every phase, forever, and unlock the living docs' experiments too.

### Sim time budget (not cloud cost)

Running substrate phases on local RTX 5080, with the sim upgrades in place:

- **Per-seed unit sim:** ~30–60 seconds (fixture-driven orchestrator + mock LLM, no live model calls)
- **Per-phase unit validation** (10 seeds × ~1 minute = ~10 minutes)
- **Full substrate unit validation** (9 phases × 10 minutes = ~90 minutes)
- **System sim tier** (uses live LLM for narrative coherence where needed, ~5–10 minutes per seed × ≥5 seeds × selected phases ≈ 2–4 hours)
- **Overnight full run of all phases × 10 unit seeds + selected system sims:** ~6 hours

**Cloud LLM spend for substrate validation:** $0. The whole substrate validation runs on local hardware. Cloud LLM is only used for optional system-sim system-level scenarios where narrative quality matters (B3/B4 evaluation in 0.4) — those are budgeted separately and optional.

**One-time sim time for the larger milestones:**
- P0 pilot: ~30 minutes on local hardware
- B1+P1 migration parity tests: ~1 hour
- P4 full mug test: ~2 hours (real vision + persistence round-trip)
- Full 1.0 stress-test sim: ~8 hours (the biggest single run)

These are all overnight-scale, not week-scale. The plan's pace is not sim-time bound.

## Non-goals

- **No custom-trained encoders or LLMs.** This is the strongest non-goal in the plan. Training a nanoGPT / small transformer on Maxim's own sim corpus — whether as an encoder, a decoder, or a sim backend — pre-bakes knowledge into the system and defeats the cross-session learning claim. A critic would correctly say "you fine-tuned a model on your data." Encoders and LLMs used by the plan must be off-the-shelf: `sentence-transformers`, OpenCLIP, cloud LLMs, local llama.cpp backends. Any learning the plan claims must happen in ATL + NAc + Hippocampus at inference time, never in a trained model's weights. This is load-bearing for the 1.0 claim, not a scope decision, and it is non-negotiable.
- **No thalamus / wiring registry.** `MemoryHub` coordinates, `AgentBus` does pub/sub, `default_network/gate.py` already claims the thalamus metaphor.
- **No shared embedding space across modalities.** Each modality keeps its native space. Cross-modal binding is relational through hippocampus episodes. **P4 must beat an OpenCLIP shared-space head-to-head baseline or commitment #3 is revisited.**
- **No modality-specific EC subclasses.** One `EntorhinalCortex`, N modality-keyed tag filters.
- **No per-channel modality.** SMS/email/Slack are TEXT with context, not separate modalities.
- **No NAc as recognition gate.** NAc modulates EC's threshold per node.
- **No NAc-owned eligibility traces.** Traces live in the shared `PerceptTraceBuffer` (F0.2). NAc reads them.
- **No SCN in credit assignment.** SCN tags episodes and drives sleep phases (P8). Credit assignment uses `PerceptTraceBuffer`.
- **No wall-clock dependency in decay or replay.** SCN phase ticks drive simulated time (P6, P8).
- **No store protocol wiring for Mother Maxim.** Deferred.
- **No POG integration.** Deferred until convergence is proven.
- **No projection layers between modality spaces.** That's fine-tuning in disguise.
- **No phase skipping.** Phases build on each other.
- **No pass on absolute numbers alone.** Absolute numbers meet minimum viability gates, never prove behavior by themselves. Degenerate controls alone are not evidence; gate baselines apply where the baseline attacks the same claim (P3a, P4, P6); regression checks apply elsewhere (P1, P2, P3b).
- **No p-values or Bonferroni corrections** in the pass criteria. Effect sizes across ≥10 seeds (≥20 for P4) are the right tool for small-sample engineering validation.
- **No new LLM router features.** Use the existing [models/language/router.py](../../src/maxim/models/language/router.py).
- **No vision or audio prompting in Track B.** Text-only scope for 1.0. Multi-modal prompt composition is 1.1+.
- **No consolidation research program inside substrate_plan.** P8 ships the minimum mechanism; [memory_consolidation_practice.md](../deferred/memory_consolidation_practice.md) hosts the research program on top of it.
- **No behavioral-convergence gates inside substrate_plan.** Behavior change is observed, not gated — it lives in [behavioral_convergence_practice.md](../deferred/behavioral_convergence_practice.md).

## If the whole thing fails

Specific fallbacks per commitment:

- **P1 fails mechanistic criteria** (paraphrase collapse < 90% or catastrophically below FAISS sanity floor) **→ mechanism bug.** Iterate swap points in order. If all fail, each modality gets its own storage class with its own recognition logic. Lose commitment #2's "one EC" invariant. ~2 weeks of refactor.
- **P2 fails mechanistic criteria** (rewarded-node collapse ≤ 0% delta, or per-agent isolation breaks) **→ mechanism bug or F0.5 regression.** Check F0.5 first; then iterate swap points. If all fail: reward gates recognition instead of modulating it. NAc becomes a gate. Biologically unfaithful but implementable. ~1 week.
- **P3a fails vs TF-IDF head-to-head → commitment #3 wrong.** Fallback: reintroduce shared ATL centroids for cross-modal concepts, with explicit projection into a shared symbolic space. ~1–2 weeks.
- **P3b fails** but P3a passes → channel boundary rules are wrong, not the architecture. Iterate per-channel boundary tuning. This is the safe failure.
- **P4 fails vs OpenCLIP head-to-head → commitment #3 very wrong.** This is the plan's biggest risk. Fallback: hippocampus-as-binder replaced by direct cross-modal edges in ATL gated by reward, or explicit CLIP-style shared space. Near-full rewrite. ~4+ weeks, substantial replanning.
- **P6 fails vs LRU → graded decay unjustified.** Fallback: use LRU. Simpler, less biological, works. Drop the "graded decay per tier" commitment and document it as a simplification.
- **P8 fails to improve F1 after replay → consolidation mechanism broken.** Iterate replay swap points (source, count, update strength, budget). If all fail: promote the mechanism question into [memory_consolidation_practice.md](../deferred/memory_consolidation_practice.md) and defer consolidation until the practice doc produces a working design. Do not block 1.0 on it — P8 becomes post-1.0 if the minimum mechanism doesn't work.

If multiple phases fail in ways that indicate the architecture is fundamentally wrong, **stop building and write a new plan before committing more effort**.

## Living-doc discipline

Living docs become graveyards without ongoing use. The goal is for [behavioral_convergence_practice.md](../deferred/behavioral_convergence_practice.md) and [memory_consolidation_practice.md](../deferred/memory_consolidation_practice.md) to accumulate at least one experiment entry per version bump, so the empirical base grows alongside the code. This is a soft discipline, not a CI gate — for a single-developer project, a hard gate against yourself is LARPing. Treat it as a signal: if a version ships and neither practice doc has a new entry, ask why. Usually the answer is "I skipped evaluation to hit a deadline" and that's a trade-off worth making explicit, not pretending it didn't happen.

The 1.0 release should have at least one entry in each practice doc. Enforce that one on yourself at tag time.

## Contracts layer (incremental, opportunistic)

Separate from the proof-obligation phase work, this plan establishes formal Protocols for the key extension points as they come up. The goal is **extensibility and platform-readiness**, not a standalone refactor. Protocols land naturally during the work that already touches each boundary:

- **`LLMBackend` Protocol** — defined as the first step of [simulator_upgrades_plan](simulator_upgrades_plan.md) S2, since S2 needs a target to implement against. Reverse-engineered from the four existing backends. ~50 LOC.
- **`BioSystem` Protocol** — defined during substrate phase work (P1–P8) as each phase touches a bio-system. Captures the common contract: `save`, `load`, `snapshot`, `on_percept`, `on_tick`. Lives in `src/maxim/contracts/biosystem.py`. Not wired to a plugin discovery system — that's post-1.0 (see [deferred/bio_system_plugin_plan.md](../deferred/bio_system_plugin_plan.md)).
- **`BioSystemSnapshot` Protocol** — defined as part of P3.5's schema-versioned snapshot sub-section (see P3.5.1 above). Schema versioning is load-bearing for the 1.0 cross-session learning claim; the Protocol accretes in P3.5 rather than waiting.
- **`Sensor` / `PerceptProducer` Protocols** — defined during foundations_plan F0.6 (Percept factory consolidation) and F0.8 (Sensor→Percept contract). These are natural landing spots because those items already touch the surface.
- **`Reporter` Protocol** — optional, defined if/when SimulationReport gets a second implementation. Don't build it speculatively.
- **`EventBus` Protocol + typed events** — *defined* here as part of the contracts layer, but the *implementation* is deferred to post-1.0. The current fragmented bus situation (five transports, direct callbacks, partial `MemoryHub` mediation) doesn't block the substrate claim — it's a platform-readiness concern. See [deferred/unified_event_bus_plan.md](unified_event_bus_plan.md). Definition is cheap (~100 LOC of Protocol + event dataclasses); implementation is the 3–5 week refactor that lives in the deferred plan.

**Why this matters for platform ambition:** if Maxim eventually has external contributors adding bio-systems, sensors, or LLM backends, **the Protocols have to exist before anyone writes against a duck-typed interface**. Once external code depends on a duck type, the duck type is de-facto frozen and refactoring it becomes a breaking change. Defining Protocols now — even if they're not enforced with ABC inheritance yet — gives `mypy` something to check and gives future contributors a clear target.

**Non-goal: no standalone contracts refactor PR.** Protocols accrete as the work that needs them happens. Writing a "contracts package" in isolation before the substrate phases would be speculative abstraction.

## Track B dependency trace

Track B phases (B1, B3, B4, B5) were merged into this plan from the old `embodiment_voice_plan.md`. B1 interleaves with P1 via the combined migration. The rest were written as "lands in 0.4" but needed an explicit trace of what they depend on in 0.3 to avoid latent ordering bugs.

| Track B phase | Version | Depends on (0.3 deliverables) | Depends on (0.4 or earlier prereqs) | Blocker risk |
|---|---|---|---|---|
| **B1** PromptAssembler | 0.3 (with P1) | F0.3 (ghost removed), F0.4 (Percept schema) | — | None — combined migration handles it |
| **B3** Acting Coach | 0.4 | B1 must exist | F0.4 Percept schema | None — B3 is additive on top of B1 |
| **B4** Replanning with memory | 0.4 | B1, **P3a** (past-failure episode retrieval), **B3** (continuity contract needs Acting Coach's state awareness) | `replanning.txt` rewrite, `loop_controller.py` history threading | **Medium risk:** B4 depends on P3a being green. If 0.3-minimum ships without P3a, B4 must wait for 0.3.1. |
| **B5** SEM/DM separation | 0.4 | B1 (single composition point) | Contract test infra | None — B5 is a lint-style contract check |

**The one real ordering risk:** **B4 depends on P3a.** If 0.3 ships as 0.3-minimum (without P3a), B4 cannot land in 0.4 because B4 needs past-failure episode retrieval — which is P3a's output. In that case, B4 slips to 0.4.1 or moves behind P3a in 0.3.1. This is not a plan-killer; it's a flag to watch during execution.

**Other Track B items (B3, B5) are additive on top of B1 and can land in 0.4 regardless of P3a's status.** Only B4 is coupled to substrate phase ordering.

## Open questions

- **OpenCLIP model choice for P4:** `ViT-B/32` is the default starting point. Larger models are more capable but slower. Pilot with two sizes and pick the one that makes the baseline hardest to beat.
- **Fixture authoring:** confirmed 60% upfront + 40% during-implementation with the upfront 60% held out frozen. This matches the "best foundation possible" goal and avoids fixture leakage.
- **Sentence-transformer model for P0 and P1:** run P0 with `all-MiniLM-L6-v2` AND `all-mpnet-base-v2`. Use the stronger model as the sanity floor for P1 — the tougher bar.
- **P8 sleep-phase cadence:** how often should sleep phases fire during a sim? Once per "session" (each time the agent stops for a clear end-of-interaction), or on an SCN-phase schedule? This is a design question the practice doc will answer; for the minimum-viable P8, one sleep phase per sim run suffices.
- **`build_primary_router` env-mutation side effect (adjacent bug, tracked here for visibility).** Discovered during the Cleanup Wave session: `_read_persisted_model` reads from `os.environ` and `build_primary_router` mutates `os.environ` as a startup side effect. Test isolation is patched with a band-aid in `conftest.py` that restores env state between tests, but the production code still has the latent bug — concurrent tests, multi-agent sessions, or any caller that reads `MAXIM_LLM_PROFILE` expecting the user's setting can see whatever startup last wrote. **Proper fix:** refactor `_read_persisted_model` to accept the profile name as a parameter threaded from the caller, not from env. Env vars become a CLI-entry-point concern, read once at startup. Scope estimate: ~50–100 LOC depending on call-graph depth. **Owned by whoever is working on [peer_leader_flexibility_plan.md](peer_leader_flexibility_plan.md)** — this note is here for cross-plan visibility only; the fix lands in its own small refactor commit after the peer-leader plan's current waves, and the conftest band-aid should be removable once it lands. This is not a substrate blocker.
