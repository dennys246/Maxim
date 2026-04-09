# Salience Abstraction Plan

**Goal:** Decouple the salience/attention system from vision-specific assumptions (pixels, bounding boxes, gaze saccades) and rebuild it around modality-agnostic primitives with pluggable coordinate systems — while deepening integration with the bio-subsystems (ATL, EC, Hippocampus, NAc, SCN).

**Motivation:** The current salience system was built for a camera-equipped robot. It works, but it leaks pixel coordinates into the generic salience API, forcing the narrative transcriber to fake bounding boxes and making the system unusable for text-only, audio, or SEM-based percepts. Meanwhile, salience and the bio-stack are barely connected — ConceptExtractor ignores salience, NAc doesn't feed back to attention, and ATL concepts don't influence what the agent notices.

**Principle:** "Where" is always part of the abstraction — never optional — but the coordinate system is pluggable. Vision uses pixels, text uses narrative positions, embodiment uses entity-relative coordinates, and temporal uses SCN bins. The salience system doesn't know which one it's using.

---

## Current Architecture (what exists)

```
┌──────────────────────────────────────────────────────────┐
│  Default Network                                         │
│                                                          │
│  ┌─────────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ SalienceNetwork │  │ SalienceMap │  │GazeController│  │
│  │ (WHAT)          │  │ (WHERE)     │  │ (HOW)       │  │
│  │ - track_id      │  │ - pixel grid│  │ - saccade   │  │
│  │ - bbox          │  │ - COCO IDs  │  │ - fixate    │  │
│  │ - novelty decay │  │ - person    │  │ - explore   │  │
│  │ - interest      │  │   detection │  │             │  │
│  └────────┬────────┘  └──────┬──────┘  └──────┬──────┘  │
│           │                  │                 │         │
│  ┌────────┴────────┐  ┌─────┴───────┐  ┌─────┴───────┐  │
│  │MovementDetector │  │AttentionNet │  │ GazeHistory │  │
│  │ - frame_size    │  │ - pixel grid│  │ - pixel pos │  │
│  │ - peripheral    │  │ - visit map │  │ - IOR decay │  │
│  └─────────────────┘  └─────────────┘  └─────────────┘  │
└──────────────────────────────────────────────────────────┘
         ↓ (detections)            ↑ (gaze commands)
    ┌────────────┐           ┌──────────┐
    │ YOLO / RTM │           │ Reachy   │
    │ (vision)   │           │ (motors) │
    └────────────┘           └──────────┘

Bio-system connections: ALMOST NONE
  - Hippocampus stores salience/novelty floats on Perception (passive)
  - SalienceMemoryBridge enriches queries post-hoc (not encoding)
  - ConceptExtractor ignores salience entirely
  - NAc doesn't feed back to attention
  - ATL concepts don't influence what agent notices
  - SCN temporal bins exist but aren't used in salience
```

### Vision-Coupled Problems

| Component | Vision Coupling | Impact |
|-----------|----------------|--------|
| `SalienceNetwork` | `bbox`, `position_u/v` in pixels, `get_objects_at_position(radius_pixels)` | Forces NarrativeTranscriber to fake pixel bboxes |
| `SalienceMap` | `frame_width/height`, `PERSON_CLASS_ID=0`, pixel grid | Completely unusable outside vision |
| `GazeController` | `saccade_threshold_pixels`, fixation timing | Only makes sense with a camera head |
| `AttentionNetwork` | `image_width/height`, pixel-to-grid conversion | Only makes sense with camera FOV |
| `MovementDetector` | `frame_size`, `peripheral_threshold` in pixels | Only makes sense with visual field |
| `NarrativeTranscriber` | `_POSITION_MAP` fakes pixel coords from "left"/"center"/"right" | Symptom of the abstraction gap |
| `TrackedObject` | `position_u`, `position_v`, `bbox_w`, `bbox_h` | Pixel semantics baked into data model |

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Salience System (modality-agnostic)                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │ SalienceNetwork  (refactored core)               │           │
│  │                                                  │           │
│  │  SalienceItem                                    │           │
│  │  ├── id: str                                     │           │
│  │  ├── label: str                                  │           │
│  │  ├── source: PerceptSource (vision|text|sem|…)   │           │
│  │  ├── confidence: float                           │           │
│  │  ├── where: WhereCoord                           │           │
│  │  └── concept_id: str | None  (ATL link)          │           │
│  │                                                  │           │
│  │  WhereCoord (protocol)                           │           │
│  │  ├── distance_to(other) → float                  │           │
│  │  ├── region() → str                              │           │
│  │  └── as_dict() → dict                            │           │
│  │                                                  │           │
│  │  Methods:                                        │           │
│  │  ├── update(items: list[SalienceItem])            │           │
│  │  ├── get_salience(id) → float                    │           │
│  │  ├── get_top(n) → list[SalienceItem]             │           │
│  │  ├── query(filters) → list[SalienceItem]         │           │
│  │  ├── get_nearby(where, radius) → list            │           │
│  │  └── to_context_str() → str (modality-aware)     │           │
│  └──────────────────────┬───────────────────────────┘           │
│                         │                                       │
│  ┌──────────────────────┴───────────────────────────┐           │
│  │ AttentionField  (abstract spatial tracking)      │           │
│  │                                                  │           │
│  │  Methods:                                        │           │
│  │  ├── record_focus(where: WhereCoord, success)    │           │
│  │  ├── get_inhibition(where) → float (IOR)         │           │
│  │  ├── suggest_next() → WhereCoord                 │           │
│  │  └── get_exploration_score(where) → float        │           │
│  │                                                  │           │
│  │  Implementations:                                │           │
│  │  ├── PixelField(w, h, grid)  ← current vision   │           │
│  │  ├── NarrativeField(regions)  ← text sim         │           │
│  │  ├── EntityField(entity_registry) ← SEM          │           │
│  │  └── NullField() ← headless/benchmark            │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │ ChangeDetector  (abstract motion/change)         │           │
│  │                                                  │           │
│  │  Methods:                                        │           │
│  │  ├── update(items: list[SalienceItem]) → scores  │           │
│  │  └── get_change_score(id) → float                │           │
│  │                                                  │           │
│  │  Implementations:                                │           │
│  │  ├── MotionDetector(frame_size) ← vision         │           │
│  │  ├── NarrativeChangeDetector() ← text (new       │           │
│  │  │     entities, state changes, surprises)        │           │
│  │  └── SensorChangeDetector() ← SEM sensor deltas  │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │ FocusController  (abstract attention dynamics)   │           │
│  │                                                  │           │
│  │  FocusState: ATTENDING | SHIFTING | SCANNING     │           │
│  │                                                  │           │
│  │  Implementations:                                │           │
│  │  ├── GazeFocusController() ← vision (saccades)   │           │
│  │  ├── NarrativeFocusController() ← text (which    │           │
│  │  │     entity to ask about / examine next)        │           │
│  │  └── NullFocusController() ← headless            │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### WhereCoord Implementations

| Implementation | Coordinate Type | distance_to() | region() | Used By |
|---------------|----------------|---------------|----------|---------|
| `PixelWhere(u, v, w, h)` | Pixel bbox | Euclidean distance | "top-left" / "center" / etc. | Vision pipeline |
| `NarrativeWhere(position, scene)` | Named position + scene ref | Ordinal distance (same/adjacent/far) | The position name itself | Sim bridge, DM runtime |
| `EntityWhere(entity_name, sensor_ref)` | SEM entity + optional sensor | Graph distance (hops in entity tree) | Entity name | SEM/embodiment |
| `TemporalWhere(scn_phase, hour_bin)` | SCN temporal coordinate | Circular distance on 24h clock | "morning" / "night" / etc. | SCN-aware salience |
| `SemanticWhere(concept_id, atl_ref)` | ATL concept space | Concept similarity (ATL graph distance) | Concept category | ATL integration |

Key insight: `WhereCoord` is a protocol (structural typing), not a base class. Any object with `distance_to()`, `region()`, and `as_dict()` qualifies. This means vision, text, SEM, temporal, and semantic coordinates all participate in the same salience computations without inheritance.

---

## Bio-Subsystem Integration

This is the real payoff — salience becomes a hub that connects to every bio-system, not just a standalone module.

### ATL → Salience: Concept Recognition Boost

**Current gap:** When the agent sees "a guard," salience treats it as a novel detection. But ATL may already have a rich concept for "guard" with relationships to "danger," "authority," "weapon." That knowledge should boost salience.

**Integration:**
1. On every `SalienceNetwork.update()`, look up each item's label in ATL via `find_or_create()`
2. If ATL returns an existing concept (not newly created), set `item.concept_id` and apply a **familiarity boost** inversely proportional to novelty — "I know what this is, and it matters"
3. Concepts with high relationship density (many connections in ATL graph) get a **semantic richness boost** — richly connected concepts are more attention-worthy
4. Concepts linked to the current goal via ATL relationships get a **goal relevance boost**

**New field on SalienceItem:** `concept_id: str | None` — links the detection to its ATL concept, enabling downstream systems to reason about what's being attended to.

### Salience → Hippocampus: Encoding Gate

**Current gap:** `Perception.salience` is stored passively. ConceptExtractor extracts from all percepts equally, regardless of salience.

**Integration:**
1. ConceptExtractor receives salience scores with each captured memory
2. **High-salience concepts** (> 0.7) get extracted with boosted initial confidence
3. **Low-salience detections** (< 0.3) are skipped entirely — no concept extraction for background noise
4. Concepts extracted from high-salience moments get a `ConceptProvenance.SALIENT_CAPTURE` tag, marking them as attention-driven discoveries

**Touch:** `concept_extractor.py` — add salience-gated extraction in `on_memory_captured()`.

### NAc → Salience: Reward-Driven Attention

**Current gap:** NAc learns causal links (action → outcome) but doesn't feed back to what the agent attends to. If "talking to the guard" repeatedly leads to good outcomes, the agent should *notice* guards more.

**Integration:**
1. NAc emits reward signals when causal predictions are confirmed (RPE)
2. On positive RPE, identify the entities present during the rewarded action
3. Boost those entities' labels in SalienceNetwork's interest set dynamically
4. On negative RPE (surprise failures), boost novelty for the associated entities — "pay more attention to this, my predictions were wrong"

**New callback:** `on_reward(action, outcome, entities_present)` registered in MemoryHub wiring.

### SCN → Salience: Temporal Attention Modulation

**Current gap:** SCN tracks circadian bins and temporal patterns, but salience weights are static.

**Integration:**
1. SCN provides a `temporal_salience_modifier()` based on current circadian phase
2. During typical "active" hours, attention thresholds are lower (notice more)
3. During typical "quiet" hours, only high-salience items break through
4. Entities that consistently appear at certain times of day get a **temporal familiarity boost** — "I always see this at this time"

**Touch:** `SalienceNetwork.get_salience()` queries SCN modifier.

### EC → Salience: Similarity-Based Priming

**Current gap:** EC stores similarity signatures but doesn't prime attention. If the current situation is similar to a past high-reward situation, the agent should preemptively attend to the same entities.

**Integration:**
1. When entering a new scene/encounter, EC queries for similar past situations
2. Entities that appeared in similar high-reward past situations get a **priming boost**
3. This is spreading activation from EC → SalienceNetwork, not just post-hoc enrichment
4. The existing SalienceMemoryBridge becomes a thin wrapper over this mechanism

**Touch:** Refactor `SalienceMemoryBridge.enrich_salience()` to use EC priming proactively.

### SEM → Salience: Sensor-Driven Attention

**Current gap:** SEM entities have sensors (hp, trust, durability) that change over time, but the salience system doesn't notice these changes.

**Integration:**
1. `SensorChangeDetector` monitors SEM sensor deltas between updates
2. Large sensor changes (hp dropping, trust shifting) generate salience spikes
3. Entities approaching failure modes get boosted salience — "this thing is about to break"
4. `EntityWhere` coordinates enable `get_nearby()` to find related entities in the SEM tree

**New:** `SensorChangeDetector` as a `ChangeDetector` implementation.

### Full Bio-System Wiring Diagram

```
                    ┌─────────────┐
         ┌─────────│     ATL     │──────────┐
         │ concept │  (concepts) │ concept  │
         │ lookup  └──────┬──────┘ richness │
         ▼                │ concept          │
   ┌─────────────┐        │ extraction       │
   │  Salience   │◄───────┘                  │
   │  Network    │                           │
   │             │◄──── NAc reward ──── ┌────┴────┐
   │  - what     │      feedback        │   NAc   │
   │  - where    │                      │(causal) │
   │  - novelty  │◄──── SCN temporal ── └─────────┘
   │  - change   │      modifier   ┌─────────┐
   │             │◄────────────────│   SCN   │
   │             │                 │ (time)  │
   │             │◄──── EC priming └─────────┘
   │             │      (similar   ┌─────────┐
   │             │       scenes)   │   EC    │
   └──────┬──────┘                 │(similar)│
          │                        └─────────┘
          │ salience-gated
          │ encoding
          ▼
   ┌─────────────┐
   │ Hippocampus │
   │ (episodes)  │
   └─────────────┘
```

---

## Phase Plan

### Phase S-0: Core Abstractions (~300 LOC)

**Goal:** Define the modality-agnostic protocols without breaking existing vision code.

**Files:**
- **New:** `src/maxim/salience/protocols.py` — `SalienceItem`, `WhereCoord` protocol, `PerceptSource` enum
- **New:** `src/maxim/salience/where.py` — `PixelWhere`, `NarrativeWhere`, `EntityWhere`, `TemporalWhere`, `SemanticWhere`
- **New:** `src/maxim/attention/protocols.py` — `AttentionField`, `ChangeDetector`, `FocusController` protocols

**Design rules:**
- `WhereCoord` is a `Protocol` (structural typing), not ABC
- `SalienceItem` is a frozen dataclass, not a dict — no more loose detection dicts
- `PerceptSource` enum: `VISION`, `NARRATIVE`, `SEM`, `AUDIO`, `TEMPORAL`
- All distance calculations are via `WhereCoord.distance_to()`, never raw pixel math
- `as_dict()` on every WhereCoord for serialization/persistence

**Tests:** Protocol conformance tests for each WhereCoord implementation.

### Phase S-1: Refactor SalienceNetwork (~400 LOC, net ~-100)

**Goal:** SalienceNetwork speaks `SalienceItem` and `WhereCoord` instead of detection dicts and pixel coords.

**Changes:**
- `update_from_detections()` → `update(items: list[SalienceItem])` (keep old method as thin adapter)
- `TrackedObject` fields: `position_u/v`, `bbox_w/h` → `where: WhereCoord`
- `get_objects_at_position(pixel_pos, radius_pixels)` → `get_nearby(where: WhereCoord, radius: float)`
- `to_context_str()` becomes modality-aware via `WhereCoord.region()` — prints "in the server room" not "at (320, 240)"
- `SalienceConfig.interest_labels` stays generic (works for any modality)

**Adapter layer:** `VisionSalienceAdapter` converts YOLO detections → `SalienceItem` with `PixelWhere`. Keeps all existing vision code working without changes to the YOLO pipeline.

**NarrativeTranscriber cleanup:** Stops faking pixel bboxes. Produces `SalienceItem` with `NarrativeWhere` directly. `_POSITION_MAP` deleted.

**Tests:** Existing salience tests adapted + new tests for each WhereCoord path.

### Phase S-2: Abstract Attention Layer (~350 LOC, net ~-150)

**Goal:** Extract `AttentionField`, `ChangeDetector`, `FocusController` as abstract interfaces with vision as one implementation.

**Changes:**
- `AttentionNetwork` → `PixelField(AttentionField)` — keeps all pixel grid logic, just behind the protocol
- `SalienceMap` → `PixelAttentionMap` — internal to vision, not part of generic API
- `GazeController` → `GazeFocusController(FocusController)` — saccade/fixation behind `ATTENDING`/`SHIFTING`/`SCANNING`
- `MovementDetector` → `MotionDetector(ChangeDetector)` — pixel motion behind `get_change_score()`
- `GazeHistory` → internal to `PixelField` (inhibition-of-return is spatial, stays with the pixel impl)

**New implementations:**
- `NarrativeField(AttentionField)` — tracks which narrative entities/locations have been "attended to" (examined, spoken to, visited)
- `NarrativeChangeDetector(ChangeDetector)` — detects new entities appearing, state changes in dialogue, surprise reveals
- `NarrativeFocusController(FocusController)` — manages which entity to examine next in text mode
- `EntityField(AttentionField)` — SEM entity tree as attention space, `distance_to()` = graph hops
- `SensorChangeDetector(ChangeDetector)` — monitors SEM sensor deltas for salience spikes
- `NullField` / `NullChangeDetector` / `NullFocusController` — for headless/benchmark mode

**Default network update:** Factory selects implementations based on active percept sources (vision detected → pixel impls, sim mode → narrative impls, SEM active → entity impls).

### Phase S-3: ATL Integration (~200 LOC)

**Goal:** ATL concepts inform salience, and salience gates concept extraction.

**Changes:**

**ATL → Salience (concept recognition boost):**
- `SalienceNetwork.update()` calls `atl.find_or_create()` for each item's label
- If concept exists: set `item.concept_id`, apply familiarity boost
- Concepts with high degree (many ATL relationships) get semantic richness boost
- Concepts related to current goal via ATL graph get goal relevance boost
- New config: `concept_boost: float = 0.2`, `richness_boost: float = 0.1`, `goal_concept_boost: float = 0.15`

**Salience → ConceptExtractor (encoding gate):**
- `on_memory_captured()` receives salience from the captured memory's `Perception`
- Items with salience > 0.7: extract with boosted confidence (1.2x)
- Items with salience < 0.3: skip extraction entirely
- New provenance: `ConceptProvenance.SALIENT_CAPTURE`

**Touch:** `concept_extractor.py`, `memory_hub.py` (wiring), `atl.py` (no API changes, just consumption).

### Phase S-4: NAc + EC + SCN Integration (~250 LOC)

**Goal:** Reward signals, similarity priming, and temporal modulation feed into salience.

**NAc → Salience (reward-driven attention):**
- Register callback in MemoryHub: `nac.on_observation()` → check RPE
- Positive RPE: add associated entity labels to `SalienceConfig.interest_labels` dynamically
- Negative RPE: boost novelty for associated entities (force re-attention)
- Interest labels decay over time (prevent permanent attention capture)
- New: `DynamicInterestTracker` manages NAc-driven interest evolution

**EC → Salience (similarity priming):**
- On scene/encounter entry, EC queries similar past situations
- Entities from high-reward similar situations get a priming boost
- Refactor `SalienceMemoryBridge` to use EC proactively (at scene entry), not just reactively
- Bridge becomes thinner — delegates to EC for similarity, ATL for concepts

**SCN → Salience (temporal modulation):**
- `SalienceNetwork.get_salience()` queries SCN for temporal modifier
- Active circadian phase → lower thresholds (notice more)
- Rest phase → higher thresholds (only high-salience breaks through)
- New config: `scn_modulation_range: tuple[float, float] = (0.8, 1.2)` — ±20% salience adjustment

### Phase S-5: SEM Integration + Campaign Validation (~200 LOC)

**Goal:** SEM sensor changes drive salience, cyberpunk campaign validates the full stack.

**SEM → Salience:**
- `SensorChangeDetector` monitors entity sensor deltas between DM runtime turns
- Large delta (> 20% of range) → salience spike for that entity
- Entity approaching failure mode → sustained salience boost
- `EntityWhere` enables `get_nearby()` over the SEM entity tree

**Campaign validation:**
- Run `neon_gauntlet_v1.yaml` with the refactored salience system
- Verify: drone in alley gets novelty decay on repeat encounter
- Verify: fixer's trust sensor changes trigger salience spike before betrayal
- Verify: new arm's proprioception sensor drives attention during recalibration
- Verify: ATL concept for "guard" boosts salience for corpo_guard encounters
- Add expectation: `salience.concept_recognition_events: 3` (ATL recognized entities)

---

## Migration Strategy

**Phase S-0 and S-1 are backwards-compatible.** The new protocols coexist with the old API. `update_from_detections()` stays as a thin adapter calling `update()` internally. Vision code doesn't change until S-2.

**Phase S-2 is the breaking change** for the default network — it replaces direct SalienceMap/GazeController construction with factory-selected implementations. But this is internal wiring, not public API.

**Phase S-3/S-4 are purely additive** — new callbacks and wiring in MemoryHub, no existing API changes.

**Phase S-5 validates** the full stack end-to-end against the cyberpunk campaign.

## File Impact Summary

| Phase | New Files | Modified Files | LOC (net) |
|-------|-----------|---------------|-----------|
| S-0 | `salience/protocols.py`, `salience/where.py`, `attention/protocols.py` | — | +300 |
| S-1 | `salience/adapters.py` | `salience/salience_network.py`, `simulation/narrative_transcriber.py` | -100 |
| S-2 | `attention/pixel_field.py`, `attention/narrative_field.py`, `attention/entity_field.py`, `attention/sensor_change.py`, `attention/null_field.py` | `attention/salience_map.py`, `attention/gaze_controller.py`, `default_network/network.py` | -150 |
| S-3 | — | `memory/concept_extractor.py`, `integration/memory_hub.py` | +200 |
| S-4 | `salience/interest_tracker.py` | `bridges/salience_bridge.py`, `integration/memory_hub.py` | +250 |
| S-5 | `attention/sensor_change.py` | `simulation/dm_runtime.py` | +200 |
| **Total** | **~10 new** | **~10 modified** | **~+700 net** |

## Invariants

- **WhereCoord is a protocol, not a base class.** Don't add ABC inheritance.
- **SalienceItem is frozen.** Mutations go through SalienceNetwork methods, not by modifying items.
- **Vision code stays in vision-specific implementations.** No pixel math in the generic layer.
- **Bio-system integration is callback-based.** Salience doesn't import ATL/NAc/SCN directly — MemoryHub wires the callbacks.
- **`update_from_detections()` adapter stays until all vision callers migrate.** Don't delete it in S-1.
- **No new dependencies.** WhereCoord implementations use only stdlib + numpy (optional).

## Testing Strategy

- **S-0:** Protocol conformance tests (every WhereCoord impl satisfies the protocol)
- **S-1:** Port existing SalienceNetwork tests to use SalienceItem. Verify NarrativeTranscriber no longer produces fake pixel bboxes. Verify `to_context_str()` prints modality-appropriate text.
- **S-2:** Verify default network initializes correct implementations per mode. Verify vision pipeline still works end-to-end with PixelField.
- **S-3:** Test concept recognition boost in isolation. Test salience-gated extraction (high/low salience memories). Integration test via MemoryHub.
- **S-4:** Test NAc interest evolution (positive RPE → boosted interest). Test EC priming (similar scene → entity priming). Test SCN modulation (temporal salience adjustment).
- **S-5:** Full campaign run with expectations. Sensor change detection unit tests. `novelty_decay_observed` expectation on cyberpunk gauntlet.
