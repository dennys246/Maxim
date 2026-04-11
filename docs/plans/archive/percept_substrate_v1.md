# Unified Percept Substrate + Reward-Modulated Cross-Region Binding

**Status:** Draft — proposed 2026-04-10
**Target versions:** 0.3 (substrate) → 0.4 (reward-modulated learning) → 0.5 (sleep-phase consolidation)
**Owner:** Denny

---

## Why this plan exists

A pre-publication code review of v1.0.0 surfaced two related findings:

1. **The bio-inspired naming is half-earned.** NAc and Cerebellum implement genuine functional analogs of their brain namesakes (Rescorla-Wagner reward learning; forward-model prediction error). ATL, Angular Gyrus, SCN, and Default Network use neuroscience vocabulary but implement conventional agent components (concept store, math engine, time-bin cache, behavior arbiter) without the cross-region mechanisms that make those brain regions distinctive.
2. **NAc learns causal links that the agent loop does not consult at decision time.** The reward-learning component is bookkeeping, not behavior. The bio-stack runs in parallel to the LLM-driven decision path rather than shaping it.

The root cause of both findings is the same: **percepts are not stored in a unified representation.** Text tokens, vision frames, audio transcripts, proprioceptive reads, and interoceptive signals each live in their own format, so Hippocampus, EC, and ATL cannot operate on a shared substrate. They end up as parallel stores with thin bridges (`fear_bridge`, `salience_bridge`, `planning_bridge`) instead of an interlocking memory system.

This plan proposes the substrate refactor that closes both gaps and makes the bio-stack load-bearing.

---

## Conceptual model

In the brain, percepts arrive at sensory cortices, are routed through the entorhinal cortex (EC) into the hippocampus, indexed against episodic memory, and bound at convergence zones (ATL) into multimodal concepts. A neuromodulatory signal — dopaminergic projections from the VTA gated by NAc — modulates which co-active associations consolidate. Memory recall is spreading activation through these reward-weighted links, not lookup over a flat store.

Maxim should approximate this with three layers:

1. **A unified `Percept` type** every modality produces, carrying a shared-space embedding alongside its raw content.
2. **A pre-hippocampal stage** (EC) that routes incoming percepts: pattern-separates novel embeddings, pattern-completes near-duplicates, and decides whether to extend the current episode or open a new one.
3. **Reward-modulated Hebbian binding** where NAc RPE signals strengthen or weaken the cross-percept links that were co-active in a recent eligibility window. ATL maintains concept centroids in the shared embedding space rather than a typed concept graph.

Recall, in this design, is spreading activation through the reward-weighted link graph — what gets retrieved depends on what the system has historically found rewarding to retrieve, not just on cosine similarity to the current query.

---

## What this unlocks

- **NAc actually gates behavior.** Strengthened links *are* the retrieval pathway, so consulting them on the next perception is unavoidable. This fixes the "NAc bookkeeping never consulted" finding directly.
- **EC has a real job.** Pattern separation / completion at the embedding layer is exactly what EC does in the brain. Currently EC is a placeholder.
- **ATL becomes a real convergence zone.** Concept centroids in shared space can bind cross-modally — the defining property of ATL — instead of being a typed graph that doesn't actually integrate modalities.
- **`energy/` and `harm/` earn their keep.** Energy gates expensive sleep-phase consolidation; harm gates sleep-phase web search. Two currently-skeletal subsystems become load-bearing.
- **The system can demonstrably improve on a task across sessions without fine-tuning the LLM.** This is the smallest unit of non-trivial machine learning that isn't just prompt engineering, and it is the threshold at which "cognitive architecture" is earned rather than claimed.

---

## Phase 0.3 — Unified Percept substrate (no learning changes)

The goal of 0.3 is to land the substrate without changing any learning behavior. Ship a refactor that everything compiles against, then iterate.

### 0.3.1 — Define the `Percept` type

Create `src/maxim/perception/percept.py`:

```python
from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np

Modality = Literal["text", "vision", "audio", "proprio", "interoception"]

@dataclass
class Percept:
    modality: Modality
    content: Any                          # raw payload (tokens, frame, waveform, sensor read)
    embedding: np.ndarray                 # unified-space embedding — load-bearing
    timestamp: float                      # epoch seconds; SCN reads this
    source: str                           # which sensor / agent / channel produced it
    salience: float = 0.0                 # default_network writes this on arrival
    context_tag: str | None = None        # episodic binding handle (set by EC)
    metadata: dict[str, Any] = field(default_factory=dict)
```

The `embedding` field is the load-bearing piece. Without a shared space, "cross-modal binding" is two graphs with a bridge.

### 0.3.2 — Pick the embedding strategy

Decision required up front. Options:

- **A small learned projection head per modality into a shared space.** Cheap, no extra heavy deps, accuracy depends on training data. We can bootstrap with random projections + Hebbian updates from co-occurrence.
- **A pretrained multimodal encoder (CLIP, ImageBind, or similar).** Higher fidelity, adds a torch dependency, but the `semantic` extra already pulls torch + sentence-transformers so this is incremental.
- **Hybrid:** sentence-transformers for text, a separate vision encoder, and a learned projection from each into a shared space. Most flexible, most code.

**Recommendation:** start with sentence-transformers for text (already an optional extra) and a 384-d random projection for non-text modalities, replaceable later. Document the dimension choice in `perception/embedding_config.py`. The embedding layer should be swappable; do not hard-code the encoder anywhere outside `perception/`.

### 0.3.3 — Migrate perception sources to produce `Percept`s

Touch points:
- Text: wherever LLM input/output currently flows into Hippocampus capture (search `hippocampus.capture(`).
- Vision: `data/camera/` and `models/vision/` consumers.
- Audio: `models/audio/` consumers.
- Proprio/interoception: any sensor reads that currently drop into `default_network`.

Each producer becomes a thin adapter that emits `Percept` objects on a single bus. The bus is the new entry point — Hippocampus, EC, ATL, default_network, and salience all subscribe.

### 0.3.4 — Refactor Hippocampus to consume `Percept`s

`memory/hippocampus.py` currently captures heterogeneous payloads. Migrate `capture()` to take `Percept` (or a sequence). Episodic records should store the percept references (or their IDs), not their raw content — raw content lives in the percept store, not duplicated.

### 0.3.5 — Give EC a real implementation

Currently `_ec` is a placeholder referenced from NAc. In 0.3, give it a job:

`src/maxim/memory/ec.py`:
- Maintains a recent-percept embedding index (FAISS or numpy depending on extras).
- On each incoming percept: computes nearest neighbor in the recent window.
- If similarity > threshold → pattern completion: extends the current episode (`context_tag` carries forward).
- If similarity < threshold → pattern separation: opens a new episode (`context_tag` rotates).
- Hands the (now-tagged) percept to Hippocampus and ATL.

This is the gateway. Everything that wants to participate in episodic memory must come through EC.

### 0.3.6 — ATL concept centroids in shared space

`memory/atl.py` currently maintains a typed concept graph. In 0.3, add a parallel representation: each concept is a centroid in the embedding space with a member set. On a new percept, ATL finds the nearest centroid; if close enough, the percept joins; otherwise a new concept seed is created with the percept as its first member. The typed graph stays for back-compatibility but is no longer the primary representation — promote it to "labels on centroids" in 0.4.

### 0.3.7 — Convergence test framework (even with trivial initial tests)

Create `tests/convergence/` with the harness in 0.3 even though there's nothing to converge on yet. We will need it in 0.4 and writing the harness under time pressure is bad. Initial tests should at minimum:

- Run a fixed sequence of percepts through the substrate and assert episode boundaries land where expected.
- Round-trip a percept through the bus and back via Hippocampus recall.
- Verify the embedding dimension is consistent across modalities.

Mark all convergence tests with the existing `learning` pytest marker.

### 0.3.8 — Don't break existing tests

The 3,500 unit tests are characterization tests for current behavior. Keep them green throughout the refactor by routing legacy capture paths through an adapter that wraps non-`Percept` inputs into `Percept`s on the way in. Delete the adapter at the end of 0.4 once everything produces `Percept`s natively.

**0.3 exit criterion:** every perception source produces `Percept`s, EC routes them, Hippocampus and ATL consume them, all existing tests still pass, and the convergence harness exists with at least three trivial tests.

---

## Phase 0.4 — Reward-modulated Hebbian binding

The goal of 0.4 is to wire NAc RPE into link updates over the percept graph and prove the system stabilizes and improves.

### 0.4.1 — Eligibility traces

Each percept that flows through EC gets pushed onto a decaying eligibility trace (recent-window queue with exponential weights). The trace is the credit-assignment vehicle: when a reward arrives, only percepts in the trace are eligible for link strengthening.

`src/maxim/memory/eligibility.py`:
- `push(percept_id, timestamp)` — append with current weight 1.0.
- `decay(now)` — apply exponential decay; drop entries below epsilon.
- `eligible() -> list[(percept_id, weight)]` — current eligible set with weights.

Trace length and decay constant are config (`NACConfig.eligibility_window_s`, `NACConfig.eligibility_decay`).

### 0.4.2 — Reward-modulated link updates

When NAc emits an RPE (`update_prediction_rw` already does this), broadcast it to the link graph. Each pair `(p_i, p_j)` in the eligibility set gets a Hebbian update:

```
Δw_ij = η · rpe · trace_weight(p_i) · trace_weight(p_j)
w_ij ← clip(w_ij + Δw_ij, 0, w_max)
```

Implementation note: store links as a sparse adjacency keyed on percept *centroid* IDs (from ATL), not raw percept IDs — otherwise the link graph grows linearly with percept count and memory blows up. Centroid IDs are bounded by concept count.

### 0.4.3 — Pruning and homeostatic plasticity

Reward-modulated Hebbian systems spiral without inhibition. Implement:

- **Per-link saturation:** `w_max` ceiling, already in the update rule above.
- **Periodic decay:** every N seconds, all link weights decay by a small factor. Forgetting is a feature.
- **Bottom-quantile pruning:** when link count exceeds budget, drop the bottom 5% by weight.
- **Per-node saturation:** cap the sum of outgoing weights from any centroid; renormalize when exceeded.

Tunables in `NACConfig`. Document defaults.

### 0.4.4 — Recall = spreading activation

Modify `Hippocampus.recall()` and `ATL.recall_associated()` to walk the reward-weighted link graph from the query's centroid, accumulating activation along weighted edges with decay per hop. What reaches an activation threshold becomes the recall set, ranked by accumulated activation (not raw cosine similarity).

The legacy similarity-based recall stays as a fallback for queries that don't land on a known centroid.

### 0.4.5 — Convergence tests (the load-bearing ones)

This is the test suite that decides whether 0.4 ships. Required:

- **Stability:** run 10,000 random percepts through the system; assert link count stays bounded, no link weight diverges, no centroid count explodes. Run with multiple seeds.
- **Reward learning:** define a synthetic task where percept type A predicts reward and percept type B does not. After N exposures, assert the A→reward link weight is significantly higher than B→reward.
- **Cross-session improvement:** run a fixed task twice in the same session; assert recall on the second run produces the rewarded percepts faster (fewer hops, higher activation) than on the first.
- **Pruning correctness:** generate enough percepts to trigger pruning; assert the pruned links are the lowest-weight links and no high-weight links are lost.
- **Homeostatic decay:** verify that without continued reinforcement, link weights decay to baseline over a known timescale.

If any of these tests don't pass, 0.4 doesn't ship. This is the test suite that converts "bio-inspired" from claim to demonstration.

### 0.4.6 — Wire NAc into the agent loop's decision path

Once recall is reward-weighted, the agent loop's perception → recall → context-assembly pipeline already consults NAc-shaped knowledge implicitly. But also: before each LLM proposal, query the link graph for the top-K most strongly linked centroids to the current context, surface them as "remembered relevant" hints in the prompt. This makes the NAc influence explicit and inspectable.

**0.4 exit criterion:** all five convergence tests pass, the agent loop consults the link graph on every perception, and a documented benchmark shows measurable cross-session improvement on at least one task.

---

## Phase 0.5 — Sleep-phase consolidation with active search

The goal of 0.5 is to let the system consolidate during quiescence, optionally including active search.

### 0.5.1 — Replay loop

During sleep (already a state in `modes/`), the existing Hippocampus consolidation loop already exists. Extend it to:

- Sample episodes from the eligibility-weighted recent buffer (recent + high-RPE first).
- Re-emit each episode's percepts through the Hebbian update path with reduced learning rate.
- Update centroid positions and link weights as if the episode were happening live.
- Prune low-weight links aggressively during sleep (more aggressive than waking).

### 0.5.2 — Energy-gated search

`energy/` becomes the gate. Sleep-phase consolidation is allowed when:

- The system is in sleep mode.
- Energy budget is above a configured threshold.
- A "things to investigate" queue is non-empty (high-uncertainty episodes, contradictions, unresolved questions flagged during waking).

When all three are true, the consolidation loop may issue search calls (LLM, web, prior conversations) to find supporting or contradicting information for queue items. Results re-enter the percept bus during sleep and consolidate normally.

### 0.5.3 — Harm gating for active search

`harm/` gates the search half. Sleep-phase web search requires:

- Explicit per-session user consent (default off).
- A domain allow-list.
- A rate limit.
- A query log the user can audit.

Without consent, sleep-phase consolidation runs but does not issue external requests — it only replays what's already in memory.

### 0.5.4 — SCN scheduling

`SCN` decides *when* sleep happens and *which* time bins to preferentially replay. Recent episodes consolidate first; distant episodes are revisited rarely but not never. This gives SCN a real circadian job instead of being a static time-bin cache.

**0.5 exit criterion:** sleep-phase consolidation runs without unbounded resource use, energy and harm gates are honored, and a benchmark shows that running a sleep cycle between two task exposures produces a larger improvement than two consecutive waking exposures.

---

## Risks and how we address them

| Risk | Mitigation |
|------|-----------|
| **Credit assignment is crude.** Eligibility traces are an approximation; long-horizon rewards may strengthen the wrong links. | Start with short windows (seconds to tens of seconds). Add longer-horizon credit assignment only if a benchmark demands it. Don't try to be clever in 0.4. |
| **Runaway feedback loops.** Reinforced links fire more, get reinforced more, dominate retrieval. | Per-link saturation, per-node saturation, periodic decay, bottom-quantile pruning. All four required, not optional. The convergence tests must catch divergence before ship. |
| **Embedding-space choice is load-bearing.** A bad shared space makes "cross-modal binding" cosmetic. | Make the embedding layer swappable. Bootstrap with sentence-transformers + projection; revisit with a real multimodal encoder once the substrate is proven. |
| **Sleep-phase search is expensive and has safety implications.** | Default off. Energy + harm gates required. Per-session consent. Domain allow-list. Audit log. 0.5 ships with the gates *first* and the search behind them. |
| **Testing this is hard — unit tests don't cover convergence behavior.** | The convergence test framework is part of 0.3's exit criterion, not 0.4's. Build the harness before you need it. |
| **The refactor breaks existing tests.** | Adapter pattern: legacy capture paths wrap raw inputs into `Percept`s during the transition. Delete the adapter at end of 0.4. |
| **Scope creep into other subsystems.** | Each phase has an explicit exit criterion. Do not start the next phase until the current one's criterion is met. |

---

## Sequencing summary

| Version | Theme | Exit criterion |
|---------|-------|----------------|
| **0.2** (publish now) | Research preview, namespace claim, packaging hygiene. No architecture changes. | Two cosmetic fixes (gitignore, dist cleanup). All existing tests pass. Wheel builds. |
| **0.3** | Unified `Percept` substrate. EC routes. ATL centroids in shared space. No learning changes. | Every perception source produces `Percept`s. EC routes them. Existing tests green. Convergence harness exists. |
| **0.4** | Reward-modulated Hebbian binding. NAc actually gates behavior. | Five convergence tests pass. Agent loop consults link graph. Cross-session improvement benchmark shows gain. |
| **0.5** | Sleep-phase consolidation, energy-gated, harm-gated. | Sleep cycle produces measurable improvement over consecutive waking exposures. Energy + harm gates honored. |
| **1.0** | Stability promise. | The system can demonstrably improve on a task across sessions without fine-tuning the LLM, with a test that proves it. |

The 1.0 threshold is the *only* one that should bear that label. Until then, the package is 0.x and the API may move.

---

## Honest framing

This plan is the version of "bio-inspired cognitive architecture" that earns the phrase. It is not AGI. It is not consciousness. It is the smallest credible instance of a learned memory substrate that operates *independently of the LLM's weights* and shapes the LLM's behavior through what it surfaces at recall time. If it works, the bio-inspired naming becomes load-bearing. If it doesn't converge in convergence tests, we learn something specific about why and fix it before claiming otherwise.

Either outcome is more valuable than shipping 1.0 with a bio-stack that runs in parallel to the decision path.
