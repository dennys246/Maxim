# Substrate Plan — Bio-Stack Convergence

**Status:** Active, gating 1.0
**Supersedes:** `percept_substrate_plan.md` (v1), `salience_abstraction_plan.md`
**Structure:** Proof-obligation phases (P1–P6). Each phase is a falsifiable behavioral claim, not an implementation chunk.

## Goal

Prove that a bio-stack of concentrated percept flow, per-modality pattern separation/completion, hippocampal episode binding, and reward-modulated Hebbian consolidation can learn across sessions **without LLM fine-tuning**. Demonstrate this through extended convergence simulations that each test one behavioral claim against a negative control, with explicit swap-points for when a claim fails.

## How to use this plan

This is not an implementation checklist ordered by LOC. Each phase is a **proof obligation**: a specific, observable behavior that must be demonstrated by an extended convergence simulation before the next phase's work is gated.

**The execution contract:**

1. Build the minimum implementation needed to run a phase's convergence sim. Not the full system — just enough to make the sim meaningful.
2. Author the test fixtures (labeled ground truth) that the sim requires.
3. Run the sim at extended scale using **both harness tiers** (see below), seeded for reproducibility, with multiple seeds for statistical significance.
4. Compare results against both the pass criteria **and a negative control** — a deliberately broken version of the system on the same input. If the architecture's numbers aren't significantly better than the negative control, the phase has not passed regardless of absolute numbers.
5. **If it passes:** proceed. The minimum implementation stays.
6. **If it fails:** explore the phase's swap points. Re-run the sim. Iterate until it passes or you've exhausted the swap points.
7. **If all swap points fail:** that's a real finding. Revisit the architectural commitments before doing more work.

### Harness tiers

Every phase runs against two harness tiers:

- **Unit sim** — direct API loop, stripped to the minimum components under test, ≤100 turns, fully seeded, runs in <60s. Used for swap-point iteration. Does not touch the full Maxim agent.
- **System sim** — full Maxim agent, 500–1000 turns, multiply seeded (≥5 seeds), runs as part of the version's validation gate. Catches integration failures the unit sim misses.

A phase passes only when **both** tiers pass. The unit sim proves the mechanism; the system sim proves the mechanism survives integration.

### Negative controls

Every pass criterion is accompanied by a **negative control** — an intentionally degenerate implementation run on identical input. Examples:

- **P1 negative control:** random node assignment (every percept gets a fresh random node). Should fail referential stability.
- **P2 negative control:** no threshold modulation (α = 0). Should show no recognition radius expansion on rewarded nodes.
- **P3 negative control:** random co-occurrence (episode boundaries assigned by coin flip). Should fail retrieval F1.
- **P4 negative control:** modality-agnostic lookup (ignore modality tags). Should either fail entirely or succeed trivially, either of which invalidates the result.

If the architecture's metrics are not statistically distinguishable from its negative control at p<0.05 across ≥5 seeds, the phase has not passed.

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

SMS, email, Slack, direct speech, narrative text from a DM — all of these are **TEXT modality** with different channel metadata. They share the same ATL text nodes (a "mug" is a "mug" whether it came from SMS or email) and the same linguistic encoder. What differs is:

- **Channel metadata** carried in `Percept.context`: `channel`, `sender`, `recipients`, `thread_id`, `subject`, `latency_class`
- **Episode boundary rules** per channel (see P3 swap points)
- **A priori salience** per channel (an SMS from a known contact is more salient than a Slack channel broadcast; this feeds WhereCoord, not NAc reward)
- **Retrieval filters** — "what did my wife say about mugs" filters episodes by `sender=wife, channel=sms`

This means channels never split the ATL node store. The agent that sees "mug" in SMS and "mug" in email recognizes them as the same concept; the *episodes* they belong to are different, and episode retrieval can be channel-filtered. See the "Communication channels" section below for details.

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

**Incoming vs outgoing:** the agent's own speech is a TEXT percept with `sender=self`. Outgoing messages flow through the linguistic encoder before being sent. This is load-bearing for one thing: the agent can retrieve its own past statements via the same mechanism it retrieves others' speech. Without this, the agent cannot track its own commitments against incoming messages.

**Long-latency credit assignment:** Email and async Slack create gaps longer than eligibility trace decay (τ ≈ 30s). When an email reply arrives three days later referencing a concept from the original thread, the trace is gone — so credit assignment cannot flow directly. Instead, the reply's content pattern-completes to the concept's ATL node, retrieves the original episode via hippocampus, and the *retrieved* context becomes the current context. This is episode-reconstruction doing the work eligibility traces cannot. **This is a free validation of the architecture:** the mechanism for long-latency comms is the same mechanism as the mug test.

**Salience:** channel salience feeds WhereCoord (the salience layer) as an a priori weight on the `Percept.salience` field. NAc's reward-based threshold modulation is orthogonal — NAc learns from outcomes, channel salience is prior-based. Both inputs affect EC's threshold.

## Current state

- [`Percept` dataclass](../../src/maxim/agents/bus.py) exists at bus.py:118-174. No `embedding`, no `modality`, no `scn_tag`, no channel context.
- [`EC`](../../src/maxim/similarity/ec.py) is an LSH index. No pattern separation/completion semantics.
- [`ATL`](../../src/maxim/memory/atl.py) is a typed concept graph. Modality tagging and edge enforcement are new.
- [`Hippocampus`](../../src/maxim/memory/hippocampus.py) holds episodes but doesn't serve as the cross-modal binding layer yet.
- [`NAc`](../../src/maxim/decisions/nac.py) has Rescorla-Wagner + eligibility traces. Per-node reward bias keyed by ATL node ID is new.
- [`SCN`](../../src/maxim/time/scn.py) has phase calculation. Episode tagging hook is new.
- Text input currently bypasses the percept layer entirely and flows straight into prompts. **Biggest single gap.**
- No communication channel integrations exist. SMS/email/Slack adapters are new (though several of them can be stubbed during the plan and built later — see phase notes).

## Proof-obligation phases

Each phase has: **Hypothesis**, **Convergence simulation**, **Pass criteria**, **Negative control**, **Swap points**, **Test fixtures**, **Minimum implementation**, **Dependencies**.

---

### P1 — Stable within-modality recognition under controlled paraphrase

**Hypothesis:** Text percepts that refer to the *same explicitly-labeled referent* activate a single stable ATL node under paraphrase variation. This phase does **not** test disambiguation of genuinely ambiguous references — that's P3's job with episode context. P1 tests only the claim: *given coreference is known*, can the encoder + EC + ATL represent it as one node.

**This scoping is deliberate.** Sentence embeddings cannot solve referential disambiguation on their own — that problem requires discourse context, which hippocampus provides in P3. P1's job is to prove that when coreference is labeled, the pipeline correctly collapses paraphrases to one node. If even this fails, no swap point will save us.

**Convergence simulation:**
Run a scripted text stream from a labeled paraphrase fixture. Each fixture cluster contains 10–30 sentences that all refer to the same labeled concept under varied surface forms ("the mug," "that mug of mine," "my favorite mug," "pick up the mug"). ≥50 clusters per run. No reward, no episodes, no hippocampus binding.

**Pass criteria:**
- **Paraphrase collapse:** ≥90% of within-cluster presentations activate the same node.
- **Cluster distinctness:** ≤5% of distinct clusters collapse into shared nodes.
- **Node stability:** node count plateaus; <10% growth over the final 20% of the run.
- **Statistical significance vs negative control** at p<0.05 across 5 seeds.

**Negative control:** random node assignment (every percept creates a fresh node, up to a cap). Should score near-zero on paraphrase collapse.

**Swap points (if the sim fails), in order of increasing cost:**
1. **Similarity metric** — cosine → euclidean → learned metric over frozen embedding (cheap).
2. **Pattern completion threshold** — static per-modality → adaptive-per-node (cheap).
3. **Encoding granularity** — whole sentence → noun-phrase span → head word + dependency context. **This is a pipeline change, not a config flag.** Budget a full day per swap.
4. **Embedding model** — sentence-transformers → a syntactically-aware encoder → a dedicated entity encoder. Pipeline change.
5. **Add shallow coreference resolution pre-encoder** — resolve pronouns/definite descriptions before encoding. ~a week of work; real scope expansion.

**Test fixtures:** `tests/fixtures/substrate/paraphrase_clusters.yaml` — ≥50 hand-authored clusters with labeled referents and paraphrase families. Author budget: 1–2 days of focused work. These fixtures are load-bearing; don't cut corners.

**Minimum implementation:**
- `Percept` dataclass with `embedding`, `modality`, `context` (dict with at least `channel`, `sender`, `thread_id`, `timestamp` fields), `salience`
- `LinguisticEncoder` producing `Percept(modality=TEXT)`
- `EntorhinalCortex.pattern_complete_or_separate(percept, modality)` returning activated or new node
- `ATL` with modality-tagged nodes, tag-filtered queries, edge enforcement
- Migration of text-to-prompt call sites through the encoder
- Unit sim harness + system sim harness
- Paraphrase cluster fixtures

**Dependencies:** None. P1 is the bottom of the stack.

---

### P2 — Reward-modulated recognition sharpens rewarded nodes

**Hypothesis:** After a reward event credited to node X, near-miss percepts that previously pattern-separated now pattern-complete to X. Recognition radius expands for behaviorally relevant stimuli.

**Convergence simulation:**
Seed a cluster of near-duplicate paraphrases of a target concept, plus distractors. Run a baseline pass without reward; count distinct nodes. Reset. Apply a reward event to the first presentation of the target. Run the same cluster with reward applied. Count nodes in the rewarded condition.

**Pass criteria:**
- Rewarded-node collapse: ≥30% fewer distinct nodes in the rewarded cluster vs baseline.
- Non-interference: distractor node count matches baseline within noise (±5%).
- Decay: recognition radius returns toward baseline over a defined timescale after reinforcement stops.
- Statistical significance vs negative control at p<0.05 across 5 seeds.

**Negative control:** NAc reward bias forced to 0 (α = 0). Should show no recognition radius change.

**Swap points:**
- NAc reward bias decay rate (τ)
- Threshold modulation strength (α)
- Eligibility trace timescale
- Per-node vs per-cluster threshold modulation
- Reward magnitude scaling

**Test fixtures:** reuse P1's paraphrase clusters, add reward-event annotations to a subset.

**Minimum implementation:**
- P1 plus
- NAc per-node reward bias keyed by ATL node ID (new data structure)
- Eligibility trace integration: encoder events → NAc updates
- EC threshold formula: `threshold = base - α * nac.reward_bias(nearest)`
- Unit + system sim harnesses

**Dependencies:** P1.

---

### P3 — Episode binding produces retrieval on partial cue

**Hypothesis:** Nodes co-occurring in the same hippocampus episode form durable links; presenting a partial cue retrieves the others. Episode boundary rules generalize across communication channels via the channel-specific config.

**This is where channel integration first lands.** P3's convergence sim uses realistic channel structures, not abstract synthetic episodes.

**Convergence simulation:**
Generate two fixture sets and run both:

1. **Synthetic baseline:** 100 hand-authored episodes with explicit ground-truth co-occurrence structure. Probe with partial cues, measure retrieval precision/recall.
2. **Channel-integration run:** 100 realistic episodes drawn from at least two channels (start with SMS + narrative direct speech, since they're the cheapest to integrate). Ground truth is author-labeled: which messages belong to which conversation, which concepts are co-mentioned. Probe: given a concept cue, retrieve episodes containing it, optionally filtered by `sender` or `channel`.

Both runs must pass independently.

**Pass criteria:**
- **Synthetic sim:** precision >0.70, recall >0.70 against ground truth.
- **Channel sim:** precision >0.70, recall >0.70 on channel-filtered retrieval ("what did X say about Y").
- **Boundary stability:** node count per episode stays in 5–50 range; no runaway episode growth.
- **Specificity under overlap:** when two episodes share a concept, cueing the concept retrieves both (disambiguation is good, not a failure).
- Statistical significance vs negative control at p<0.05 across 5 seeds.

**Negative control:** random episode boundaries (assign percepts to episodes by coin flip). Should fail retrieval F1 dramatically.

**Swap points:**
- **Episode boundary definition** — start with per-channel rules (SMS: contact+30min gap; narrative: scene change signal). If retrieval fails, swap toward semantic-gap detection or SCN phase boundaries. Per-channel rules can be tuned independently.
- **Hebbian link strength function** — how much does one co-occurrence contribute?
- **Link decay rate**
- **Consolidation trigger** — when does an episode "close" and propagate links to ATL?
- **Retrieval mechanism** — direct episode lookup → spreading activation → top-K weighted

**Test fixtures:**
- `tests/fixtures/substrate/synthetic_episodes.yaml` — 100 synthetic episodes with labeled co-occurrence.
- `tests/fixtures/substrate/channel_episodes.yaml` — 100 realistic SMS + narrative episodes with channel metadata, sender labels, and co-mention ground truth.
- Author budget: 2–3 days.

**Minimum implementation:**
- P2 plus
- `Episode` dataclass: `id`, `start_tick`, `end_tick`, `channel`, `sender_ids: set[str]`, `thread_id: str | None`, `activated_nodes: set[NodeId]`, `reward_events: list[RewardEvent]`, `scn_tag: CircadianContext`
- Hippocampus episode store with episode-to-node edges
- Per-channel episode boundary rules (start with SMS and narrative; email/Slack deferred unless adapter exists)
- Hebbian within-ATL edge updates on episode close
- Retrieval path with optional channel/sender filter
- SMS and narrative channel adapters (narrative already exists in the sim runtime; SMS is a new ~100 LOC stub that can read from a fixture file)

**Dependencies:** P2.

---

### P3.5 — Basic cross-session persistence

**Hypothesis:** ATL nodes, hippocampus episodes, NAc reward biases, and channel episode structure survive serialization/deserialization. A reloaded system recognizes the same nodes, retrieves the same episodes, and respects the same reward biases as it did pre-shutdown.

**Why here and not P5:** The 1.0 claim is cross-**session** learning. Without persistence landing before P4, P4's "session 2" is actually "same process, new test" — a weaker claim than the one 1.0 requires. P3.5 is minimum-viable persistence: save, load, verify identity. Decay and GC land later in P6.

**Convergence simulation:**
Run P3's channel-integration sim through to episode close. Serialize the full bio-stack state with `atomic_io`. Spawn a new process. Load state. Re-run only the retrieval probes. Verify results match pre-shutdown.

**Pass criteria:**
- Node identity survives: every pre-shutdown node has a matching post-reload node with the same `node_id`.
- Edge weights survive: within-ATL edge weights match pre-shutdown within floating-point tolerance.
- Episode retrieval works identically: the same probes produce the same results (precision/recall within 2% of pre-shutdown).
- Reward biases survive: NAc per-node reward bias values round-trip.
- **No negative control needed** — this phase tests serialization round-trip, not learning.

**Swap points:**
- Persistence format: JSON → msgpack → a more structured store
- What to persist: all nodes / top-K by connectivity / only consolidated
- Load order: ATL → hippocampus → NAc, or a different sequence

**Minimum implementation:**
- P3 plus
- Save/load methods on ATL, hippocampus, NAc using `atomic_io.atomic_write_json`
- Round-trip test harness

**Dependencies:** P3.

---

### P4 — Cross-modal binding via hippocampus

**Hypothesis:** Nodes of different modalities co-occurring in the same episode can cue each other across modality boundaries through episode reconstruction. This is the architecture's central claim.

**This phase uses a minimal real vision encoder, not a toy intero encoder.** A scalar-state toy encoder with a small finite set of values produces an impoverished embedding space that makes pattern completion trivial — the phase would "pass" without the binding mechanism being tested. Minimal vision (single-object CLIP embedding, ~100 LOC) actually exercises the claim.

**Convergence simulation — the mug test, persisted:**
Session 1: run a scripted sim where a text `"mug"` percept and a vision percept of a mug co-occur in multiple episodes from the narrative channel. Apply a reward event in one of them. Save state (uses P3.5). **Spawn a new process**, load state. Session 2: present the text `"mug"` alone. Measure whether the vision mug node is retrieved.

Run the symmetric test too: session 1 vision cue, session 2 text recall. Cross-modal retrieval should work both directions.

**Pass criteria:**
- Forward retrieval: text cue retrieves vision node on >80% of seeded trials.
- Reverse retrieval: vision cue retrieves text node on >80% of seeded trials.
- Modality isolation: cueing text does not retrieve unrelated vision nodes (false binding rate <10%).
- Reward gating: rewarded co-occurrences retrieved more reliably than non-rewarded (>15% margin).
- Statistical significance vs negative control at p<0.05 across 5 seeds.

**Negative control:** modality-agnostic lookup — ignore modality tags entirely and treat vision and text as one big pool. If this passes, the result is meaningless because the test was trivial. If this fails, the architecture's claim is distinct from the baseline.

**Swap points:**
- Episode reconstruction mechanism (direct lookup → spreading activation → weighted top-K)
- Number of co-occurrences for durable cross-modal link (1 → 3 → 5)
- Cross-modal link strength function
- Retrieval threshold

**Test fixtures:** ~30 scripted episodes with known text-vision pairings, authored with a small set of object images (mug, cup, bowl, plate) and labeled ground truth.

**Minimum implementation:**
- P3.5 plus
- Minimal `VisionEncoder` using CLIP or equivalent (single-object image → embedding → `Percept(modality=VISION)`)
- Cross-modal retrieval path: text cue → ATL text node → hippocampus episode lookup → reconstruct episode → retrieve vision nodes
- Symmetric path for vision cue
- Mug test harness with persisted session boundary

**Dependencies:** P3.5. **This phase is the 0.3→0.4 gate.**

---

### P5 — Robust cross-session persistence under stress

**Hypothesis:** The bio-stack state survives serialization across varied content distributions, high node counts, dense episode graphs, and concurrent channel activity. Not just "it round-trips" (P3.5) but "it round-trips under realistic load."

**Convergence simulation:**
Run a long-running mixed-channel sim (SMS + email + Slack + narrative + vision) to 10,000+ nodes and 1,000+ episodes. Serialize periodically (every 100 episodes). Reload at each checkpoint. Verify retrieval quality does not degrade across reloads.

**Pass criteria:**
- State size stays bounded and explicable (linear or sub-linear in node count).
- Retrieval precision/recall stable across ≥10 save/reload cycles.
- Load time stays under a defined threshold (e.g., <5s for 10k nodes).
- No silent corruption: every node and edge round-trips verified by checksum.

**Negative control:** skip persistence entirely (compare against in-process continuous run). Persistence should not degrade retrieval relative to in-process.

**Swap points:**
- Persistence format (if P3.5 picked JSON, P5 might force msgpack for size)
- Incremental vs full writes
- Compression
- Index structures at load time

**Dependencies:** P4.

---

### P6 — Extinction without reinforcement

**Hypothesis:** Associations not reinforced decay predictably. Reinforced associations persist. The system forgets appropriately.

**Convergence simulation:**
Long-running sim with two node groups: Group A receives periodic reinforcement, Group B does not. Measure Hebbian link strength, retrieval probability, and node presence over simulated multi-session duration (leveraging P5 persistence).

**Pass criteria:**
- Group A retrieval stays high (>80% of initial).
- Group B retrieval drops below 20% within N simulated sessions.
- No catastrophic forgetting: Group A not collaterally damaged by Group B decay.
- Node count bounded: orphaned Group B nodes pruned after decay.
- Statistical significance vs negative control at p<0.05 across 5 seeds.

**Negative control:** no decay (Hebbian strength frozen). Should show no forgetting and unbounded node growth.

**Swap points:**
- Decay function shape (linear / exponential / power)
- Per-tier decay rates (FORMING / WORKING / SHORT_TERM / LONG_TERM)
- Pruning threshold
- Reinforcement semantics (single re-encounter resets decay fully, or partially)

**Dependencies:** P5.

---

## Version path

| Version | Phases that must pass | What it proves |
|---|---|---|
| **0.3** | P1, P2, P3, **P3.5**, P4 (with minimal vision) | Architecture's central claim holds end-to-end, including across a real process boundary |
| **0.4** | P4 re-passed with production-quality vision + expanded channel coverage (add email, Slack) | Architecture generalizes beyond minimal scope |
| **0.5** | P5, P6 | System persists under load and forgets appropriately |
| **1.0** | Stress-test sim combining all phases with full channel diversity | Cross-session learning without LLM fine-tuning, at realistic scale |

**Why P4 is in 0.3 with real vision:** the minimal vision encoder is ~100 LOC of CLIP wiring. The toy intero alternative would produce a trivially-passable test that tells us nothing. The extra cost of real vision in 0.3 is small; the validity gain is large.

**Why P3.5 is in 0.3:** without persistence before P4, P4's "session 2" is a misnomer and the 1.0 claim is weaker than advertised.

**Channel rollout across versions:**
- **0.3:** SMS + narrative (cheapest integrations; narrative already exists in sim runtime)
- **0.4:** Add email + Slack (real adapters, not stubs)
- **0.5:** Any remaining channels + multi-channel stress tests

## Scope honesty

| Phase | Implementation | Sim harness | Fixtures | Total |
|---|---|---|---|---|
| P1 | ~800 LOC (includes text-to-prompt migration — the big hidden cost) | ~200 LOC | 1–2 days authoring | ~1,000 LOC + fixtures |
| P2 | ~300 LOC | ~150 LOC | reuse P1 fixtures | ~450 LOC |
| P3 | ~600 LOC (episode model + boundary rules + SMS adapter stub) | ~250 LOC | 2–3 days authoring | ~850 LOC + fixtures |
| P3.5 | ~300 LOC | ~150 LOC | reuse P3 fixtures | ~450 LOC |
| P4 | ~500 LOC (vision encoder + cross-modal path + mug test harness) | ~200 LOC | 1 day authoring | ~700 LOC + fixtures |
| P5 | ~400 LOC | ~300 LOC | generated | ~700 LOC |
| P6 | ~300 LOC | ~250 LOC | generated | ~550 LOC |
| **Total to 1.0** | **~3,200 LOC system** | **~1,500 LOC harness** | **~1 week fixtures** | **~4,700 LOC + fixture authoring time** |

This is roughly 50% bigger than the previous estimate. The previous estimate omitted: sim harness infrastructure as its own budget, negative control implementations, ground-truth fixture authoring, and the text-to-prompt migration scope. Honest scoping beats optimistic scoping.

## Non-goals

- **No thalamus / wiring registry.** `MemoryHub` coordinates, `AgentBus` does pub/sub, `default_network/gate.py` already claims the thalamus metaphor.
- **No shared embedding space across modalities.** Each modality keeps its native space. Cross-modal binding is relational through hippocampus episodes.
- **No modality-specific EC subclasses.** One `EntorhinalCortex`, N modality-keyed tag filters.
- **No per-channel modality.** SMS/email/Slack are TEXT with context, not separate modalities.
- **No NAc as recognition gate.** NAc modulates EC's threshold per node.
- **No SCN in credit assignment.** SCN tags episodes. Credit assignment uses eligibility traces.
- **No store protocol wiring for Mother Maxim.** Deferred.
- **No POG integration.** Deferred until convergence is proven.
- **No projection layers between modality spaces.** That's fine-tuning in disguise.
- **No phase skipping.** Phases build on each other.
- **No pass without negative control.** Absolute numbers are not evidence.

## Cross-pollination with embodiment_voice_plan

- [`embodiment_voice_plan.md`](embodiment_voice_plan.md) B1 (PromptAssembler) consumes P1's text node activations as the "memory" layer of the prompt. As P2 lands, the prompt's recalled content becomes reward-gated.
- B4 (replanning with failure diagnosis) overlaps with P3's co-episode retrieval: past failure episodes become retrievable cues that feed replanning.
- The linguistic encoder from P1 is the path for user/DM/NPC speech into the substrate — and, once channels are integrated in P3, the path for SMS/email/Slack messages too.
- **Sequencing:** P1 and B1 should land together. B1 cannot meaningfully consume memory state until P1 has a stable ATL. P1 cannot meaningfully validate its migration until B1 routes text through the encoder.

## If the whole thing fails

Specific fallbacks per commitment, sketched so that a failed phase triggers scoped exploration:

- **P1 fails across all swap points → commitment #2 wrong.** Fallback: each modality gets its own storage class (`TextStore`, `VisionStore`, ...) with its own recognition logic. Lose the "one EC" invariant. Concrete sketch: split `ATL` into per-modality stores, add a thin coordinator for episode binding. Rough cost: 2 weeks of refactor.
- **P3 fails across all swap points → commitment #3 wrong.** Fallback: reintroduce shared ATL centroids for concepts that co-occur across modalities, with explicit projection from per-modality embeddings into a shared symbolic space. The "no shared space" commitment relaxes. Concrete sketch: add a `ConceptCentroid` layer above modality-tagged nodes. Rough cost: 1–2 weeks.
- **P4 fails across all swap points → commitment #3 very wrong.** Fallback: hippocampus-as-binder architecture replaced by direct cross-modal edges in ATL, gated by reward. This is a near-full rewrite of the substrate plan. Rough cost: 4+ weeks, substantial replanning.
- **P2 fails across all swap points → commitment #4 wrong.** Fallback: reward gates recognition instead of modulating it. NAc becomes a gate. Biologically unfaithful but implementable. Rough cost: 1 week.

If multiple phases fail in ways that indicate the architecture is fundamentally wrong, stop building and write a new plan before committing more effort.
