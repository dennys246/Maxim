# Gating Abstraction Plan

**Status:** Ready (2026-04-20)
**Scope:** 0.7 — Simulation Scalability (prerequisite for Deliberative Thinking)
**Depends on:** Default Network (existing), Reactions/ReactionBus (existing)
**Blocks:** Deliberative Thinking L0 (BioEnrichmentPipeline needs TextSalienceScorer)

---

## Problem

The system needs novelty/salience gating in multiple places but currently has three fragmented implementations:

1. **ThalamicGate** (`default_network/gate.py`) — priority cascade for visual percepts, tightly coupled to COCO classes, track IDs, and Reachy hardware
2. **ImaginationTrigger** (`imagination/trigger.py`) — ad-hoc mention-counting + temporal arousal check via `dn.imagination_allowed()`
3. **PainBus** (`proprioception/pain_bus.py`) — refractory window per `(entity, failure_mode)`

The ThalamicGate's core architecture is clean (modular callbacks, adaptive thresholds, generic config) but its implementation is 60-70% vision-bound through percept field assumptions and COCO semantics. Text percepts pass through it by accident (the `has_speech=True` path), not by design.

For v1.0, the system needs:
- Text gating (bio-enrichment pipeline, deliberative thinking)
- Audio gating (future: speech salience, environmental sounds)
- Cross-modal salience boosting (hearing → vision attention)
- Energy-aware gating (SCN fatigue → aggressive filtering)
- Learning to gate (NAc feedback on what was worth attending to)

None of these fit cleanly into the current ThalamicGate without it becoming a god-class.

## Architecture

**Shared scoring protocol + shared adaptive threshold controller. NOT a unified gate class.**

Each gating site has different *decision structure* (cascade vs refractory vs budget). What's actually shared across all of them:

1. How to score an input (novelty × salience × goal relevance → number)
2. How to adapt the threshold over time based on outcomes
3. How to report the decision (uniform logging/provenance)

```
runtime/gating.py (new module)
├── SalienceScorer (Protocol)
│     score(input, context) → GateScore
├── GateScore (frozen dataclass)
│     novelty: float, salience: float, goal_relevance: float, combined: float
├── GateDecision (frozen dataclass)
│     passed: bool, score: GateScore, reason: str, threshold_used: float
├── AdaptiveThresholdController (extracted from gate.py)
│     record_outcome(), adapt(), threshold (property)
├── AdaptiveThresholdConfig (extracted from gate.py)
│     target_rate, adaptation_rate, min_threshold, max_threshold, ...
└── GatingContext (frozen dataclass)
      active_goal: str | None, energy: float, recent_inputs: int

Consumers (each composes scorer + controller):
├── ThalamicGate (DN)        → VisionSalienceScorer + cascade + motor routing
├── BioEnrichmentPipeline    → TextSalienceScorer (EC pattern sep + goal keywords)
├── ImaginationTrigger       → TextSalienceScorer + energy budget + temporal minimum
└── Future: AudioScorer, ProprioceptionScorer, CrossModalBooster
```

### Why Not a Unified Gate Class

The decision structures are fundamentally different:

| Site | Structure | Why different |
|------|-----------|---------------|
| ThalamicGate | Priority cascade (7 checks, first match wins) | Vision has many bypass conditions (attention locks, speech, safety) |
| BioEnrichment | Simple threshold (novel enough? → enrich) | No cascade needed, just yes/no |
| Imagination | Budget gate (novelty + energy + temporal spacing) | Rate-limited by design — too expensive to fire freely |
| PainBus | Refractory window (recent fire → suppress) | Time-based suppression, not salience |

Forcing these into one `UnifiedGate.evaluate()` produces either a method full of modality switches or sites overriding most of the logic. Both are worse than composable building blocks.

### SalienceScorer Protocol

```python
from typing import Protocol, Any

class SalienceScorer(Protocol):
    """Score an input's novelty and salience for gating decisions.

    Each modality implements this independently:
    - Vision: track-ID novelty + COCO class salience + spatial relevance
    - Text: EC pattern separation + goal-keyword overlap + NAc familiarity
    - Audio: volume spike + speech detection + directional novelty
    - Pain: intensity as salience, refractory recency as inverse novelty
    """
    def score(self, input: Any, context: GatingContext) -> GateScore: ...
```

### AdaptiveThresholdController (Extracted)

Already exists in `default_network/gate.py` lines 87-340. The existing implementation is modality-agnostic — it tracks:
- Escalation rate (adjusts toward target, default 5%)
- Outcome quality (action_taken vs ignored vs goal_created)
- Load signal (LLM queue depth → generalize to "processing load")
- Urgency bias (fear/risk factor → generalize to "energy urgency")

**Zero behavioral change on extraction.** ThalamicGate imports from the new location instead of owning the class.

### Cross-Modal Salience Boosting (v1.0)

NOT a gate concern — it's a *scorer* concern. When audio detects a sound in a direction, it publishes a `SalienceBoost(direction, magnitude)` on the ReactionBus. The vision scorer subscribes and temporarily lowers its threshold in that spatial region. Gates stay modality-local; scorers coordinate through the existing bus.

### Energy-Aware Gating (v1.0)

Each `AdaptiveThresholdController` already accepts a load signal. Generalize to an `energy: float` parameter from SCN (via `GatingContext.energy`). Tired agent = higher thresholds = more aggressive filtering. One field per gating site, no new abstraction.

### Learning to Gate (v1.0)

The controller's `record_outcome()` method is the hook. When NAc records positive valence for an escalated input, call `record_outcome("action_taken")`. When an escalation leads to no learning, call `record_outcome("ignored")`. The adaptive loop already uses this to tune thresholds. Wire it, don't redesign it.

## Stages

### G0 — Extract `runtime/gating.py` (~80 LOC)

**New file:** `runtime/gating.py`

Extract from `default_network/gate.py`:
- `AdaptiveThresholdController` class (lines 87-340)
- `AdaptiveThresholdConfig` dataclass
- New: `SalienceScorer` protocol
- New: `GateScore`, `GateDecision`, `GatingContext` dataclasses

**Modified:** `default_network/gate.py` — import `AdaptiveThresholdController` from `runtime/gating.py` instead of defining it locally. Zero behavioral change.

**Validation:** All existing DN tests pass unchanged. The gate works identically — it just imports from a different module.

### G1 — TextSalienceScorer (~60 LOC)

**New:** `TextSalienceScorer` class in `runtime/gating.py` (or `integration/text_scorer.py` if it needs bio-system imports)

```python
class TextSalienceScorer:
    """Score text inputs for novelty and salience.

    Novelty: EC pattern_separate_or_complete distance (0=exact match, 1=completely new)
    Salience: goal-keyword overlap + NAc reward history for recognized concepts
    """
    def __init__(self, *, ec: EC | None = None, nac: NAc | None = None): ...

    def score(self, text: str, context: GatingContext) -> GateScore:
        novelty = self._compute_novelty(text)      # EC pattern separation
        salience = self._compute_salience(text, context)  # goal keywords + NAc
        combined = novelty * salience
        return GateScore(novelty=novelty, salience=salience,
                         goal_relevance=self._goal_overlap(text, context),
                         combined=combined)

    def _compute_novelty(self, text: str) -> float:
        """EC pattern separation: how different is this from recent inputs?"""
        # If EC available: pattern_separate_or_complete returns separation distance
        # Fallback: simple keyword overlap with recent history

    def _compute_salience(self, text: str, context: GatingContext) -> float:
        """Goal keyword overlap + NAc reward signal for recognized concepts."""
```

**Used by:** BioEnrichmentPipeline (L0 of deliberative thinking plan) as the novelty gate.

### G2 — Wire ImaginationTrigger (~30 LOC)

**Modified:** `imagination/trigger.py`

Replace the ad-hoc gating:
- Remove mention-counting threshold (currently hardcoded)
- Replace `self._dn.imagination_allowed()` with `TextSalienceScorer.score() + threshold check`
- Keep temporal minimum (minimum interval between fires) as a parameter on the trigger, not on DN
- Trigger gets its own `AdaptiveThresholdController` instance for learning what's worth imagining

**Result:** ImaginationTrigger no longer depends on the full DefaultNetwork for its gating decision. It uses the same scoring protocol as BioEnrichmentPipeline. DN dependency becomes optional (only needed for motor control in Reachy mode).

### G3 — ThalamicGate cleanup (~100 LOC)

**Modified:** `default_network/gate.py`

- Extract COCO_CLASSES and vision-specific scoring into `VisionSalienceScorer` implementing `SalienceScorer`
- Gate's `evaluate()` delegates scoring to the vision scorer
- Cascade logic stays (it's DN-specific decision structure, not generic)
- Add `Percept.modality` routing: if modality is text, delegate to TextSalienceScorer instead of VisionSalienceScorer
- Remove the `has_speech` special case — fold it into TextSalienceScorer (speech always scores max salience)

**Result:** ThalamicGate becomes a thin orchestrator that delegates scoring to modality-specific scorers and applies DN-specific cascade logic on top.

## Sequencing with Deliberative Thinking Plan

```
G0 (extract gating.py)
  ↓
G1 (TextSalienceScorer)     ←  prerequisite for Deliberative Thinking L0
  ↓
L0 (BioEnrichmentPipeline)  ←  uses TextSalienceScorer as novelty gate
  ↓
L1 (ThinkTool enrichment)
  ↓
G2 (ImaginationTrigger)     ←  can ship in parallel with L1
  ↓
L2 (Active deliberation)
  ↓
G3 (ThalamicGate cleanup)   ←  can ship in parallel with L2
```

G0 + G1 are the minimum prerequisite. G2 and G3 improve architecture but don't block the thinking plan.

## Invariants

- **`AdaptiveThresholdController` is stateful and per-consumer.** Each gating site owns its own instance with domain-appropriate parameters. Controllers are NOT shared across sites.
- **`SalienceScorer` is stateless (or internally manages its own state).** Scorers can cache recent inputs for novelty computation but don't depend on external mutable state except their constructor dependencies (EC, NAc).
- **Gate decisions are logged uniformly via `GateDecision`.** Every gating site produces the same structured decision record for provenance/debugging.
- **ThalamicGate retains its cascade logic.** The refactoring extracts the *scoring* and *threshold adaptation*, not the *decision structure*. The priority cascade (attention lock > goal > speech > interest > novelty*salience > anomaly > safety) is DN-specific and stays.
- **Vision-specific code stays in `default_network/`.** `VisionSalienceScorer`, COCO classes, track-ID logic remain in the DN module. Only the shared abstractions move to `runtime/gating.py`.
- **`runtime/gating.py` has no bio-system imports at the module level.** The protocol is defined with `TYPE_CHECKING` only. Concrete scorers that need EC/NAc live in `integration/` or their respective domain modules.

## LOC Estimate

| Stage | LOC | Notes |
|-------|-----|-------|
| G0 | ~80 | Extract + protocol + dataclasses |
| G1 | ~60 | TextSalienceScorer |
| G2 | ~30 | ImaginationTrigger refactor |
| G3 | ~100 | ThalamicGate cleanup + VisionSalienceScorer |
| **Total** | **~270** | |

## Open Questions

- **EC availability in sim mode:** Does EC exist when running `--sim` without `MAXIM_SUBSTRATE_PATH=1`? If not, TextSalienceScorer needs a fallback (keyword-based novelty without EC). Check the sim bio-stack construction path.
- **AdaptiveThresholdController persistence:** Currently the controller's learned thresholds persist within DN's session state. After extraction, each consumer needs its own persistence path (or accepts ephemeral thresholds per session).
- **GatingContext.energy source:** SCN provides circadian energy. But does SCN exist in all agent configurations? If not, default energy=1.0 (fully energized, no gating tightening).
- **ImaginationTrigger DN decoupling:** G2 removes the DN dependency for gating. But ImaginationTrigger still uses DN for the "arousal gate" (`imagination_allowed`). Should the temporal-minimum logic move into the trigger's own controller, or should DN retain a `is_available_for_imagination()` method that only checks inhibition state?
