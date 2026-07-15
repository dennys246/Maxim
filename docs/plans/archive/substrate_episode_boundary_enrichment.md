# Substrate — Episode boundary enrichment

**Status:** PARTIAL (2026-04-18). Stage 1 (tool execution boundary) SHIPPED in 0.4. Stage 3 (pain/salience spike) SHIPPED via sem_learning_loop.md. Stage 2 (semantic shift) remains — ship before P6.
**Scope:** ~200–400 LOC (3 new boundary rules + CaptureEvent extensions + tests).
**Target version:** post-0.3. Ships AFTER P4 Stage 3 proves the base substrate claim.
**Parent:** None (standalone). Extends `memory/episode.py` boundary rule surface.
**Blocks:** Nothing directly. Improves episode quality for all downstream substrate phases.

## Motivation

Episode boundary detection currently has three rules ([episode.py:199-234](../../src/maxim/memory/episode.py#L199)):

1. `tick_gap_rule(max_gap)` — silence gap
2. `channel_change_rule()` — channel switch
3. `scn_tag_change_rule()` — scene-tag change

These cover the basics but miss three biologically meaningful boundary signals that the runtime already produces. The result is episodes that are too coarse — a single episode can span a tool execution (binding pre-tool context to post-tool results as one undifferentiated blob), a topic pivot mid-conversation (binding unrelated concepts), or a pain spike (blurring the moment of surprise into surrounding routine). Coarser episodes mean noisier Hebbian edges, which dilutes the signal-to-noise ratio that spreading activation has to work with.

The `EpisodeBoundaryDetector` is already designed for pluggable rule extension ([episode.py:319-347](../../src/maxim/memory/episode.py#L319)) — rules are `Callable[[PendingEpisodeState, CaptureEvent], bool]`, evaluated via `any()`, commutative. Adding new rules is zero-touch to existing code. The work is in (a) enriching `CaptureEvent` with the signals the new rules need, and (b) wiring the signal producers (executor, encoder, PainBus) to populate those fields.

## New boundary rules

### Rule 1 — Tool execution boundary

**Signal:** the agent executed a tool between the previous event and the current one.
**Why it matters:** tool execution fundamentally changes context. "I'm about to search for X" and "search returned Y results" are different episodic contexts — binding them in the same episode is correct (the search intent relates to the results), but binding the pre-search *conversation* to the post-search *results* through the same episode creates Hebbian edges between unrelated concepts. Breaking the episode at tool execution creates three clean episodes: `[pre-search conversation]`, `[search intent + search results]`, `[post-search conversation]`.
**Biological analogy:** action execution as an episodic boundary — performing a deliberate action ("I opened the drawer") creates a memory segmentation point.

```python
def tool_execution_rule() -> BoundaryRule:
    """Close the pending episode when the incoming event was preceded
    by a tool execution."""
    def _rule(pending: PendingEpisodeState, event: CaptureEvent) -> bool:
        return event.after_tool_execution is True
    return _rule
```

**CaptureEvent extension:** `after_tool_execution: bool = False`. Set by the agent loop / executor integration point after a tool completes, before the next percept is captured.

**Design choice — close BEFORE the tool result, not after:** the tool result event starts a new episode (possibly with the tool's output nodes as activated_nodes). The pre-tool conversation is in the closing episode. If the tool result should bind to the search intent, the executor emits the intent as the last event of the old episode OR as the first event of the new one — TBD during implementation based on which produces better Hebbian structure.

### Rule 2 — Semantic shift detection

**Signal:** the incoming text percept's embedding has low cosine similarity to the previous text percept's embedding.
**Why it matters:** topic changes within a conversation are invisible to the existing rules — same channel, same scene, no silence gap — but create episodic boundaries in human memory. "We were talking about the project deadline, then someone mentioned their vacation" should be two episodes. Without this rule, the deadline concepts and vacation concepts get Hebbian-bound, creating noise edges that dilute retrieval quality.
**Biological analogy:** this is the most directly hippocampal rule. fMRI studies show hippocampal episode boundary signals correlate with contextual shift detection, independent of spatial scene change.

```python
def semantic_shift_rule(threshold: float = 0.40) -> BoundaryRule:
    """Close the pending episode when the incoming event's embedding
    diverges from the episode's centroid by more than (1 - threshold)
    cosine distance."""
    def _rule(pending: PendingEpisodeState, event: CaptureEvent) -> bool:
        if event.embedding is None or pending.centroid_embedding is None:
            return False
        similarity = cosine_similarity(event.embedding, pending.centroid_embedding)
        return similarity < threshold
    return _rule
```

**CaptureEvent extension:** `embedding: ndarray | None = None`. Populated by the substrate encoding path (LinguisticEncoder output) when available. `None` for events that don't carry text embeddings (vision-only, system events).

**PendingEpisodeState extension:** `centroid_embedding: ndarray | None = None`. Running centroid of all text embeddings seen so far in this episode (incremental update on each text event, same math as EC's centroid update).

**Threshold calibration:** 0.40 is a starting point. Needs a sweep against real conversation data — too low (0.20) and only hard topic jumps fire; too high (0.60) and natural conversational drift creates tiny episodes. Stage 1 includes a calibration pass using the P2/P5 fixture conversations.

**O(1) concern:** cosine similarity on 768-dim vectors is ~1μs. Well within the per-rule O(1) budget noted in the EpisodeBoundaryDetector docstring.

### Rule 3 — Pain/salience spike boundary

**Signal:** a PainSignal was published between the previous event and the current one, above a salience threshold.
**Why it matters:** surprising or emotionally charged events create sharper episodic boundaries in biological memory. You remember the moment before and after a shock as distinct episodes, not one blurred sequence. The substrate should model this — a pain spike during tool execution or environmental interaction should create a clean episodic boundary so the "what went wrong" context is in its own episode, not diluted by surrounding routine percepts.
**Biological analogy:** amygdala-mediated episodic segmentation. Emotionally salient events trigger hippocampal pattern separation, creating distinct memory traces for pre-event and post-event contexts.

```python
def salience_spike_rule(min_intensity: float = 0.5) -> BoundaryRule:
    """Close the pending episode when the incoming event follows a
    pain/salience spike above the given intensity threshold."""
    def _rule(pending: PendingEpisodeState, event: CaptureEvent) -> bool:
        return event.salience_spike is not None and event.salience_spike >= min_intensity
    return _rule
```

**CaptureEvent extension:** `salience_spike: float | None = None`. Set by the PainBus integration point when a PainSignal with `intensity >= min_intensity` was published since the last capture event. The agent loop or MemoryHub bridge translates PainBus signals into this field on the next CaptureEvent.

**Intensity threshold:** 0.5 is a starting point — this is relative to PainSignal.intensity which ranges [0, 1]. Only high-salience events (failed tool executions, embodiment pain, unexpected environmental changes) should break episodes; low-level background signals (slight discomfort, minor uncertainty) should not.

## What is NOT a new rule (and why)

- **Agent action boundary** (break when agent speaks/acts): too aggressive as a default. In a multi-turn conversation the agent speaks every other turn — this would create 2-event micro-episodes that can't accumulate meaningful Hebbian structure. Could be a channel-specific rule via P3b's `channel_specific_rule` wrapper for specific use cases (e.g., tool-heavy channels where each action is independent), but not a default.

- **Temporal duration cap** ("no episode longer than N seconds"): crude. The tick_gap_rule already handles silence, and the new semantic-shift rule handles topic drift within active conversation. A hard cap would chop useful long episodes (extended tool use, deep focused conversation) at arbitrary boundaries. If this turns out to be needed after P5 stress testing reveals pathologically long episodes, add it then with data.

- **Modality change boundary** (break when text → vision or vice versa): explicitly rejected in P4 Stage 1 design ([episode.py:126-133](../../src/maxim/memory/episode.py#L126)). The entire cross-modal binding mechanism depends on one episode containing BOTH text and vision nodes co-activating. A modality-change rule would prevent the mechanism from firing.

## CaptureEvent changes (summary)

Three new optional fields, all defaulting to the no-op value:

```python
@dataclass(frozen=True)
class CaptureEvent:
    # ... existing fields ...
    after_tool_execution: bool = False          # Rule 1
    embedding: np.ndarray | None = None         # Rule 2
    salience_spike: float | None = None         # Rule 3
```

Plus one new mutable field on `PendingEpisodeState`:

```python
centroid_embedding: np.ndarray | None = None    # Rule 2 running centroid
```

All existing callers that construct `CaptureEvent(tick=..., channel=...)` continue to work unchanged — the new fields are optional with safe defaults. No boundary rule fires on `None` / `False` values.

## Stages

### Stage 1 — Tool execution boundary

Smallest blast radius, clearest signal, most mechanical (no threshold calibration needed).

1. Add `after_tool_execution: bool = False` to `CaptureEvent`
2. Implement `tool_execution_rule()` in `episode.py`
3. Wire in the executor → agent loop path: after `Executor.execute()` returns, set the flag on the next `CaptureEvent`
4. Test: episode with [text, text, TOOL, text, text] produces 2 episodes, not 1
5. Regression: P4 mug test still passes (no tool execution in the fixture)
6. Optional: sweep a P5-style fixture comparing episode count / Hebbian edge density with and without the rule

### Stage 2 — Semantic shift detection

Requires embedding on CaptureEvent + running centroid on PendingEpisodeState.

1. Add `embedding: ndarray | None = None` to `CaptureEvent`
2. Add `centroid_embedding: ndarray | None = None` to `PendingEpisodeState` with incremental update in `_apply_event_to_pending`
3. Implement `semantic_shift_rule(threshold)` in `episode.py`
4. **Calibration sweep:** run against real conversational data (P2 fixture, P5 fixture when available) to find the threshold where episode count is reasonable (not too many micro-episodes, not too few mega-episodes). Report threshold vs episode-count curve.
5. Test: conversation with mid-stream topic pivot produces 2 episodes; same-topic conversation stays as 1
6. Regression: P4 mug test unaffected (single-concept inputs have no "previous embedding" to compare against)

### Stage 3 — Pain/salience spike boundary

Requires PainBus → CaptureEvent bridge.

1. Add `salience_spike: float | None = None` to `CaptureEvent`
2. Implement `salience_spike_rule(min_intensity)` in `episode.py`
3. Wire in: agent loop subscribes to PainBus (or MemoryHub surfaces it), records latest PainSignal intensity since last capture, populates the field on the next CaptureEvent
4. Test: pain signal with intensity 0.7 between two text events produces 2 episodes; pain signal with intensity 0.2 does not
5. Regression: existing SEM pain cascade test unaffected (pain cascade operates at the NAc learning layer, not the episode boundary layer — the boundary enrichment makes the episodes AROUND the pain sharper, but doesn't change the pain attribution path)
6. Integration test: tool failure → PainSignal → episode boundary → next episode captures recovery behavior → NAc links the recovery to the failure. This is the "the agent remembers what went wrong and what it did about it as two distinct episodes" story.

### Stage 4 — Measurement and tuning (optional, post-P5)

1. Instrument episode statistics: mean/median/p99 episode duration (ticks), node count per episode, Hebbian edge count per episode
2. Compare old rules vs enriched rules on P5 stress test fixture
3. Tune thresholds based on real data (semantic shift threshold, salience intensity floor)
4. Document recommended defaults per use case (conversation, tool-heavy, embodied, simulation)

## When to execute

**Not before P4 Stage 3 ships.** Same reasoning as concept decomposition — the base substrate claim must be proven first. Episode boundary enrichment improves signal quality, but if the underlying Hebbian mechanism doesn't beat OpenCLIP, sharper episodes won't save it.

**Ideal moment:** after P4 (cross-modal proven), alongside or after concept decomposition, before P5 stress test. P5 exercises the substrate under realistic multi-turn, multi-tool agent workloads — exactly the scenario where the current three rules are insufficient. Having richer boundaries before P5 means the stress test measures the enriched system, not the minimal one.

**Stage ordering within this plan:** Stage 1 (tool boundary) can ship independently and has zero calibration risk. Stage 2 (semantic shift) needs a calibration sweep and should ship before any fixture that measures Hebbian quality. Stage 3 (pain/salience) depends on how far the PainBus → MemoryHub integration has progressed (Wave 2 of biosystem_unification may be a natural moment).

**Trigger condition:** P4 Stage 3 PASSES. If P4 fails, defer indefinitely.

## Cross-references — where enriched boundaries would help

- **[substrate_concept_decomposition.md](substrate_concept_decomposition.md):** concept decomposition creates more nodes per episode. Without boundary enrichment, a long conversation episode could accumulate dozens of noun-phrase nodes with O(n^2) Hebbian edges. Tool and semantic-shift boundaries keep episode size bounded, making the interaction between the two plans healthy rather than explosive.
- **[substrate_binding_persistence.md](substrate_binding_persistence.md) P5 (stress test):** P5's multi-turn fixtures will exercise realistic episode lengths. Having enriched boundaries before P5 means the stress test results reflect production-like episode structure.
- **[substrate_binding_persistence.md](substrate_binding_persistence.md) P8 (sleep replay):** sleep replay pumps O(episodes) events through the detector in batch ([episode.py:327-336](../../src/maxim/memory/episode.py#L327) perf note). More episodes from richer boundaries means more replay iterations — the O(1)-per-rule invariant becomes more load-bearing. Rule 2 (semantic shift with embedding cosine) is the heaviest per-event computation; verify it stays under budget during P8 implementation.
- **[archive/biosystem_unification.md](biosystem_unification.md) Wave 2 (MemoryHub):** Stage 3 (pain/salience spike) needs the PainBus → episode-capture bridge. If Wave 2 ships the MemoryHub unification first, Stage 3 can wire through MemoryHub's existing capture callback surface. If this plan ships first, it builds a minimal bridge that Wave 2 later absorbs.
- **[behavioral_convergence_practice.md](../deferred/behavioral_convergence_practice.md):** the "does the agent get better across sessions" question is sensitive to episode quality. Sharper episodes mean cleaner Hebbian structure, which means retrieval quality improves faster — or at least, regressions from noisy episodes are eliminated as a confound.

## Risks

1. **Over-segmentation.** If all three new rules fire aggressively, episodes could become too short (2–3 events) to accumulate meaningful Hebbian structure. Mitigation: minimum episode duration floor (configurable, default ~3 events). If an episode has fewer than N activated nodes, the close is deferred. This is a Stage 4 tuning concern, not a Stage 1 blocker.
2. **Semantic shift threshold sensitivity.** Too low and it never fires; too high and every sentence is a new episode. Mitigation: Stage 2 includes a calibration sweep and the threshold is a constructor parameter, not a constant.
3. **CaptureEvent field creep.** Three new optional fields is fine; ten would be a smell. If a future plan needs more fields, consider refactoring CaptureEvent into a core struct + an optional metadata dict. Not needed yet.
4. **Embedding on CaptureEvent couples the boundary detector to the encoder.** If the encoder changes (concept decomposition ships multiple embeddings per input), the centroid update logic needs adjustment. Mitigation: centroid update uses the same incremental math as EC; concept decomposition plan is aware of this dependency.
