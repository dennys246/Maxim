# Causal Memory Integration Plan

> **Status:** Not started. Depends on cognitive pain system (implemented) and RPE flow (implemented).

Wire NAc's causal learning into EC's similarity space and hippocampus's associative graph. Currently, causal links are siloed in NAc's flat dict — EC can't find similar causal patterns, the hippocampal graph has no causal edges (EdgeType.CAUSES exists but is never created), and spreading activation can't traverse causal chains.

---

## Current State (Verified Against Code)

### What Works

| System | What It Does | Storage |
|--------|-------------|---------|
| **NAc** | Rescorla-Wagner causal learning (event → outcome) | `_links: dict[str, list[CausalLink]]` — flat, by event_signature |
| **EC** | Multi-modal similarity (LSH, structural, temporal, semantic) | `_signatures: dict[str, SituationSignature]` — by memory_id |
| **Hippocampus** | Episodic memory with associative graph | `_memories` + `_graph` with ASSOCIATES edges |
| **ToolPainBridge** | Routes tool errors → NAc + SCN, exposes RPE | `_pending_tools`, `_last_rpe` |

### What's Missing

1. **EC doesn't index causal links.** NAc can find links by exact event_signature, but can't answer "what causal patterns are similar to this one?" EC has the similarity engine but doesn't know about causal links.

2. **CAUSES edges never created.** `EdgeType.CAUSES` exists in the enum (bus.py:77). `spreading_activation()` already traverses it (hippocampus.py). But NO code path creates CAUSES edges. The graph only has ASSOCIATES edges (from `_form_associations()` during capture).

3. **NAc context similarity is partial.** `predict()` already scores links by `confidence * (0.5 + 0.5 * context_similarity)` (nac.py:471), but `_context_similarity()` is key-by-key exact matching. EC's LSH + semantic embeddings would provide much richer similarity. EC results can be appended to the `event_links` list before the existing scoring logic — predict() will rank them naturally.

4. **CausalLink.memory_ids is a weak backlink.** CausalLinks store memory IDs that contributed to the link, but the hippocampal graph has no edge FROM those memories TO the causal outcome. You can go CausalLink → memory_ids → episodes, but you can't go episode → spreading_activation → related causal outcomes.

---

## Biological Mapping

| Brain Structure | Maxim Component | Role in Causal Memory |
|----------------|----------------|----------------------|
| **Entorhinal Cortex** | `EC` | Gateway — indexes causal patterns in similarity space, enables "find similar causes" |
| **Hippocampus** | `Hippocampus._graph` | Episodic binding — CAUSES edges link event episodes to outcome episodes |
| **Nucleus Accumbens** | `NAc` | Reward learning — computes RPE, updates predicted_value via Rescorla-Wagner |
| **SCN** | `SCN` | Temporal context — when did this causal pattern occur? (already wired) |
| **Amygdala** | `FearAgent + PainDetector` | Aversive gating — should we avoid this action? (already wired) |

The flow mirrors dopaminergic signaling: NAc detects surprise (RPE), hippocampus forms the episodic memory, EC indexes it for future retrieval, and spreading activation through CAUSES edges enables causal reasoning ("last time I did X in context Y, Z happened").

---

## Implementation

### Design: NAc as the Trigger Point

The NAc computes RPE on every `record_outcome()` call. This is the natural trigger for all downstream memory operations — the RPE magnitude determines what gets wired:

```
NAc.record_outcome() → CausalLink updated → RPE computed
    │
    ├─ RPE > 0.3 (surprising) ──► Create CAUSES edge in hippocampus graph
    │                            ► Register causal pattern in EC
    │                            ► Boost hippocampus capture salience
    │
    └─ RPE ≤ 0.3 (routine) ────► Update predicted_value only (existing behavior)
                                 ► No graph edge, no EC registration
```

NAc already returns the updated CausalLinks (with `last_rpe` set) from `record_outcome()`. The calling bridge (ToolPainBridge) has access to hippocampus and EC through its constructor params. All wiring flows from NAc's surprise signal.

### Phase 1: Create CAUSES Edges in Hippocampus Graph

When NAc records a surprising outcome (RPE > 0.3), create CAUSES edges between event episodes and the outcome episode in the hippocampal graph.

**CORRECTION:** `_form_associations()` in hippocampus.py does NOT call EC — it uses hippocampus's own `_recall_similar_unlocked()`. CAUSES edge creation is a new code path, not extending `_form_associations()`.

**Where:** In `ToolPainBridge`, after `nac.record_outcome()` returns updated links. ToolPainBridge gets a new optional `hippocampus` parameter:

```python
class ToolPainBridge:
    def __init__(self, nac, pain_detector, scn=None, hippocampus=None):
        # ... existing ...
        self._hippocampus = hippocampus

def _create_causal_edges(self, links: list[CausalLink], outcome_memory_id: str | None):
    """Create CAUSES edges from event episodes to outcome episode.
    Only for surprising outcomes (RPE > 0.3)."""
    if not outcome_memory_id or not self._hippocampus:
        return
    for link in links:
        if (link.last_rpe or 0.0) < 0.3:
            continue  # Skip routine outcomes
        for event_mem_id in link.memory_ids[-5:]:
            if event_mem_id != outcome_memory_id:
                try:
                    self._hippocampus.graph.add_edge(
                        source=event_mem_id,
                        target=outcome_memory_id,
                        edge_type=EdgeType.CAUSES,
                        weight=link.confidence,
                    )
                except Exception:
                    pass  # Memory may have been evicted
```

Call `_create_causal_edges()` from both `record_tool_complete()` (success path) and `_on_pain()` (failure path), after NAc outcome recording.

**Effect:** `spreading_activation()` already traverses CAUSES edges (verified: bus.py:1138 checks `EdgeType.ASSOCIATES` and `EdgeType.CAUSES`). Now it finds them. Recalling an event episode activates related outcome episodes through causal chains.

### Phase 2: Register Causal Links in EC Similarity Space

Index established CausalLinks in EC so "find similar causal patterns" works. Only register links with `observation_count >= 3` (not one-off events).

**Where:** In NAc, after creating/updating a CausalLink in `record_outcome_full()`. NAc gets a new optional `ec` parameter:

```python
class NAc:
    def __init__(self, config=None, ec=None):
        # ... existing ...
        self._ec = ec

# In record_outcome_full(), after link is created/updated:
if self._ec is not None and link.observation_count >= 3:
    sig = SituationSignature(
        structural_hash=hash(f"{link.event_signature}:{link.outcome_signature}"),
        temporal_hash=(0, 0, 0, 0),  # SCN bins if available
        tool_name=link.event_signature.split(":")[-1] if ":" in link.event_signature else "",
        outcome_type=link.outcome_valence.value,
        mode="",
        goal_keywords=tuple(link.event_context.get("goal", "").split()[:3]),
        context_hash=hash(frozenset(sorted(link.event_context.items()))),
        semantic_hash=(),
    )
    self._ec.register(f"causal:{link.id}", sig)
```

**Deregistration:** When a CausalLink's confidence drops below 0.1 or it's evicted, remove from EC:
```python
if self._ec is not None:
    self._ec.remove_signature(f"causal:{link.id}")
```

**Effect:** EC can now answer "find causal patterns similar to this tool execution."

### Phase 3: EC-Enhanced Causal Prediction

Augment NAc's `predict()` with EC similarity results. The existing predict() already scores links by `confidence * (0.5 + 0.5 * context_similarity)` (nac.py:471). EC results can be appended to `event_links` before this scoring — predict() ranks them naturally.

**Where:** In `NAc.predict()`, before the existing scoring logic:

```python
def predict(self, event_type, event_signature, context=None):
    event_links = self._links.get(event_signature, [])
    
    # NEW: augment with EC-similar causal patterns (gated by config)
    if self._ec is not None and self._config.use_ec_similarity:
        sig = self._build_causal_signature(event_signature, context)
        similar = self._ec.find_similar(sig, k=10, min_similarity=0.5)
        for causal_id, score in similar:
            if causal_id.startswith("causal:"):
                link_id = causal_id[7:]
                link = self._get_link_by_id(link_id)
                if link and link not in event_links:
                    event_links.append(link)
    
    # Existing scoring logic handles ranking (line 458+)
    ...
```

**Config flag:** Add `use_ec_similarity: bool = False` to `NACConfig`. Defaults to OFF until Phases 1-2 are validated. This makes Phase 3 independently deployable and reversible.

**Caution:** EC results may surface causal links with different event_signatures but similar contexts. The existing context_similarity scoring (line 471) handles ranking — high context_match links score higher. Start with `use_ec_similarity=False` and enable after observing Phase 1-2 behavior.

### Phase 4: RPE-Driven Capture Salience

Close the loop: boost hippocampus capture salience by RPE magnitude so surprising outcomes form stronger memories.

**Where:** In `agent_loop.py`, before `capture_from_loop_async` is called (lines 1293-1305). The observation dict already supports `salience` (hippocampus.py:469 reads `observation.get("salience", 0.5)`):

```python
# After tool execution, before capture:
rpe = executor.get_last_rpe()
if rpe > 0.0:
    current_salience = observation.get("salience", 0.5)
    observation["salience"] = min(1.0, current_salience + rpe * 0.5)  # RPE boosts up to +0.5
```

**Effect:** Surprising tool outcomes (high RPE) produce high-salience memories that hit the 0.7 threshold for consolidation candidates or 0.95 for immediate long-term promotion. Routine outcomes keep their original salience.

---

## Implementation Sequencing

| Phase | What | Effort | Impact | Dependencies |
|-------|------|--------|--------|-------------|
| **1** | CAUSES edges in hippocampus graph | Small | High | ToolPainBridge + hippocampus param |
| **2** | Register causal links in EC | Medium | High | NAc + EC param, SituationSignature building |
| **3** | EC-enhanced causal prediction | Medium | High | Phase 2, careful tuning needed |
| **4** | RPE-driven capture salience | Small | Medium | get_last_rpe() (already implemented) |

Phase 1 and Phase 4 are independent and can be done in parallel. Phase 2 before Phase 3 (indexing before querying). Phase 3 requires the most care — it changes prediction behavior.

---

## Wiring Changes

### ToolPainBridge

```python
class ToolPainBridge:
    def __init__(self, nac, pain_detector, scn=None, hippocampus=None):
        # ... existing ...
        self._hippocampus = hippocampus  # NEW: for CAUSES edges
```

### NAc

```python
class NAc:
    def __init__(self, config=None, ec=None):
        # ... existing ...
        self._ec = ec  # NEW: for causal link indexing (Phase 2) + similarity queries (Phase 3)

class NACConfig:
    # ... existing fields ...
    use_ec_similarity: bool = False  # NEW: Phase 3 flag, default OFF
```

### MemoryHub wiring (agentic_runtime.py)

```python
# Currently:
nac = NAc(config=nac_config)
tool_pain_bridge = ToolPainBridge(nac=nac, pain_detector=detector, scn=scn)

# After:
nac = NAc(config=nac_config, ec=ec)  # NAc can index causal links in EC
tool_pain_bridge = ToolPainBridge(
    nac=nac, pain_detector=detector, scn=scn,
    hippocampus=hippocampus,  # ToolPainBridge can create CAUSES edges
)
```

---

## What NOT to Change

- **NAc's internal `_links` storage** — it's the primary causal link store. EC provides similarity indexing ON TOP of it, not replacing it.
- **EC's SituationSignature structure** — it already has the fields needed (tool_name, outcome_type, temporal_hash, context_hash). Just populate them from CausalLink data.
- **Hippocampus capture logic** — the interestingness criteria (salience > 0.7 → consolidation candidate) are correct. We just boost salience with RPE, not change the thresholds.
- **Spreading activation parameters** — decay=0.5, max_depth=3, threshold=0.05 are tuned for ASSOCIATES edges. CAUSES edges should use the same parameters initially.

---

## Risks

1. **CAUSES edge proliferation.** Mitigated: only create CAUSES edges when `last_rpe > 0.3` (Phase 1 code guards this). Routine outcomes (low RPE) don't create edges.

2. **EC index bloat.** Mitigated: only register links with `observation_count >= 3` (Phase 2 code guards this). One-off events don't clutter EC.

3. **Prediction behavior change (Phase 3).** Mitigated: `NACConfig.use_ec_similarity` defaults to `False`. Must be explicitly enabled after Phases 1-2 are validated. Existing scoring logic (context_similarity weighting) handles ranking naturally.

4. **Circular activation.** CAUSES + ASSOCIATES edges could create loops (A associates B, B causes C, C associates A). Mitigated: spreading_activation already bounds with max_depth=3 and decay=0.5.

5. **Stale EC entries.** If a CausalLink is evicted or confidence drops, EC must deregister it. Mitigated: Phase 2 includes deregistration logic on confidence < 0.1.

---

## Tests Needed

- CAUSES edge created when RPE > threshold (Phase 1)
- CAUSES edge NOT created for routine outcomes (Phase 1)
- Spreading activation traverses CAUSES edges (Phase 1)
- CausalLink registered in EC with correct signature (Phase 2)
- EC.find_similar returns causal links (Phase 2)
- NAc.predict finds similar causal patterns via EC (Phase 3)
- Capture salience boosted by RPE magnitude (Phase 4)
- RPE=0 produces no salience boost (Phase 4)
