# Percept Reflex System — Automatic Body Responses

**Status:** DESIGN
**Depends on:** Component-level damage (shipped), Proprioceptive Discovery (planned)

## Motivation

Auto-damage in `SendMessageTool` is a special case bolted onto the wrong abstraction. It detects attack keywords in narration text and applies damage — but this is just ONE instance of a general pattern: "percept contains sensory signal → body responds automatically."

A real body does many things automatically:
- **Attack detected** → component damage (currently: hack in SendMessageTool)
- **Bright flash** → squint (visibility drops)
- **Loud noise** → startle (awareness spikes, stamina dips)
- **Cold/heat exposure** → shiver/sweat (stamina drain)
- **Falling/impact** → brace (damage distribution, leg stress)
- **Smoke/poison** → cough (stamina drain, awareness drop)
- **Social threat** → heightened awareness (awareness spikes)

These are all the same pattern and deserve a unified abstraction.

## The Three Response Layers

The body's automatic response system has three layers, each operating at a different timescale:

```
Layer 1: REFLEXES (this plan)
  When: DURING percept processing (same tick, before deliberation)
  What: Innate pattern → automatic body response
  Source: Entity spec (archetype-defined, body-specific)
  Example: Attack text → component damage, fire text → burn pain
  Analog: Spinal cord reflex arc

Layer 2: PRE-EMPTION (existing — perceived_pain.py)
  When: BEFORE action execution (anticipatory)
  What: Learned prediction → anticipatory pain/aversion
  Source: NAc causal links (experience-based)
  Example: "last time I touched fire it hurt" → anticipatory pain
  Analog: ACC/vmPFC anticipatory aversion

Layer 3: LEARNING (existing — pain_bus + NAc)
  When: AFTER action execution (outcome)
  What: Outcome evaluation → causal link formation
  Source: Tool results, failure modes, pain signals
  Example: "fire_breath → damage" NAc link, reward_bias update
  Analog: Dopamine prediction error
```

Reflexes are INNATE (defined in spec, not learned). Pre-emption is LEARNED (NAc predictions). Learning closes the loop (outcomes update predictions). Together they form the body's full automatic response stack.

## Design

### Core Abstraction

```python
@dataclass(frozen=True)
class ReflexSpec:
    """A single percept reflex defined in the entity spec.

    Detects a pattern in percept text and triggers an automatic
    body response.  Reflexes fire during bio-enrichment processing,
    BEFORE the LLM deliberates — the body responds before the mind
    decides.
    """
    name: str                              # e.g., "attack_flinch"
    triggers: tuple[str, ...]              # keyword patterns to detect
    response_type: str                     # "damage", "sensor_adjust", "pain", "reaction"
    target_component: str                  # modulator name (e.g., "torso", "head", "legs")
    intensity_scale: dict[str, float]      # intensity keyword → multiplier
    base_intensity: float                  # default intensity when no keyword match
    cooldown_s: float = 2.0               # min seconds between firings (prevent spam)
```

### Entity Spec Format

Reflexes are declared per-archetype (shared across all entities of that type) or per-entity (overrides):

```yaml
# In _data/reflexes/humanoid.yaml (archetype-level, included by entity spec)
reflexes:
  attack_flinch:
    detect_keywords: [attack, strikes, hits, slashes, bites, claws, stabs, smashes]
    response: {tool: damage_component, params: {component: torso}}
    base_intensity: 0.15
    intensity_scale:
      devastating: 0.30
      massive: 0.25
      critical: 0.25
      powerful: 0.20
      light: 0.05
      glancing: 0.05
      minor: 0.05
    cooldown_s: 1.0
    suppressible: true   # pre-emption can reduce intensity

  fire_burn:
    detect_keywords: [fire, flame, burn, scorch, engulf, inferno]
    response: {tool: damage_component, params: {component: torso, source: fire}}
    base_intensity: 0.15
    cooldown_s: 1.0
    suppressible: true

  impact_brace:
    detect_keywords: [fall, crash, slam, impact, thrown, collide]
    response: {tool: damage_component, params: {component: legs}}
    base_intensity: 0.10
    cooldown_s: 2.0
    suppressible: true

  environment_cold:
    detect_keywords: [cold, freezing, frost, blizzard, ice]
    response: {tool: set_entity_sensor, params: {sensor: stamina, value: -0.05}}
    base_intensity: 0.05
    cooldown_s: 10.0
    suppressible: false   # can't learn to not feel cold

  startle:
    detect_keywords: [explosion, roar, scream, thunderous, deafening]
    response: {tool: set_entity_sensor, params: {sensor: awareness, value: -0.1}}
    base_intensity: 0.1
    cooldown_s: 5.0
    suppressible: true   # soldiers can learn to suppress startle
```

A dragon's reflexes would be DIFFERENT — no fire_burn (immune to fire), but has wing_damage_flinch (protective wing fold on impact).

### Processing Pipeline

Reflexes fire inside `BioEnrichmentPipeline.enrich()` — the same pipeline that already processes every percept. This is the natural home because:

1. It runs BEFORE the LLM deliberates (same tick as percept)
2. It already has access to the embodiment (for sensor/component modification)
3. It already computes salience/threat (the scorer's output can gate reflexes)
4. It already returns structured results that flow into the prompt

```
Percept text → BioEnrichmentPipeline.enrich()
  ├── Novelty gate (existing)
  ├── Hippocampus query (existing)
  ├── NAc predictions (existing)
  ├── ATL spreading activation (existing)
  ├── ComponentIndex affordances (existing)
  ├── *** Reflex evaluation (NEW) ***
  │     ├── For each registered reflex:
  │     │     ├── Check trigger keywords against text
  │     │     ├── Check cooldown (skip if too recent)
  │     │     ├── Compute intensity (base + keyword scale)
  │     │     ├── Apply response via DamageComponentTool / sensor adjustment
  │     │     └── Emit Reaction on ReactionBus for downstream learning
  │     └── Return fired reflex names in EnrichmentResult.reflexes_fired
  └── Return EnrichmentResult (existing fields + reflexes_fired)
```

### Connection to Pre-emption

When a reflex fires, it produces a `Reaction` on the `ReactionBus`. This feeds into:
1. **NAc** — records "attack_percept → damage" causal link (reflexes ARE learning signals)
2. **Hippocampus** — episodic capture of the reflex event
3. **Pre-emption** — over time, NAc learns to predict the reflex before it fires. "This situation usually triggers attack_flinch, so I should anticipate pain." The pre-emption layer becomes the LEARNED version of the innate reflex.

This creates a beautiful development arc:
- **Session 1:** Reflexes fire reactively (innate)
- **Session 2+:** Pre-emption starts anticipating (learned from reflex outcomes)
- **Mature agent:** Pre-emption dominates, reflexes only fire on genuinely surprising stimuli

### Migration: Remove auto-damage from SendMessageTool

The `_detect_attack` function and its auto-damage call in `SendMessageTool.execute()` become the `attack_flinch` reflex. The keyword list (`_ATTACK_KEYWORDS`) moves to the entity spec. SendMessageTool becomes clean — it just sends messages.

```python
# Before (hack in SendMessageTool):
if self._embodiment is not None:
    is_attack, amount, source = _detect_attack(text)
    if is_attack: apply_damage(...)

# After (reflex in BioEnrichmentPipeline):
# SendMessageTool: just sends. No damage logic.
# BioEnrichmentPipeline.enrich(): evaluates reflexes automatically.
# attack_flinch reflex: detects attack → DamageComponentTool → pain → learning
```

## Review Findings (3-lens parallel review, 2026-04-25)

### Extensibility + Plugin Lens

1. **[HIGH] Response dispatch: Tool calls, not enum.** The `response_type` enum is a closed dispatch system. Third-party developers can't add custom response types. **Resolution:** Define responses as tool invocation specs in YAML: `response: {tool: damage_component, params: {component: torso}}`. This reuses existing tool infrastructure, gets `@resilient` coverage, and produces `ToolOutput.side_effects` the executor already processes. A `ReflexHandler` protocol with registered handlers covers non-tool responses.

2. **[HIGH] Telemetry: `reflexes_fired` in EnrichmentResult.** Add the field + `sim_enrichment("reflex", ...)` logging. Without it, reflex firings are invisible.

3. **[MEDIUM] Reflex chaining via ReactionBus.** After reflexes fire, check if emitted Reactions match other reflexes. Cap cascade depth at 2-3. Use existing ReactionBus refractory as spam guard.

4. **[MEDIUM] Suppression: `suppressible` flag.** Check NAc for learned suppression before firing. One dict lookup per reflex, negligible cost.

5. **[LOW] Cross-entity: tag Reactions with `source_agent_id`.** Defer to multi-agent milestone.

### Bio-Fidelity + Learning Dynamics Lens

6. **[HIGH] Habituation: exposure-count decay.** Multiply intensity by `1 / (1 + k * exposure_count)` per reflex per context. Reset on context change (new attacker, new environment). The cooldown_s stays for rate-limiting; habituation is the separate adaptation curve.

7. **[HIGH] Double-counting: pain pipeline ONLY.** Reflexes cause damage → pain → NAc. Do NOT also emit a separate Reaction to NAc. Choose one path. The pain pipeline is more mature and better tested.

8. **[HIGH] Pre-emption suppression: bracing reduces damage.** When a reflex fires, check if pre-emption already fired for the same percept this tick. If so, reduce reflex intensity by the pre-emption intensity (clamped to zero). "I was braced, the blow hurts less." This creates the correct gradient: learning to anticipate REDUCES net pain. Without this, the system has a pathological gradient where learning makes the agent suffer MORE.

9. **[MEDIUM] Sensitization: damaged parts are hypersensitive.** Scale intensity by `1 + factor * (1 - component.integrity)`. Creates realistic cascade where damage begets more pain from further damage to the same region.

10. **[MEDIUM] Affordance death spiral: confidence threshold on body-part extraction.** If ambiguous, default to torso (no survival-critical affordances gated at low integrity). Only target specific limbs when text is unambiguous.

### Standardization + API Consistency Lens

11. **[HIGH] `triggers` → `detect_keywords`.** The word "trigger" is already taken by `FailureMode.trigger: {field, op, value}`. One letter apart, different semantics. Rename to avoid spec-author confusion.

12. **[MEDIUM] Canonical `build_reflex_registry(*, entity_spec=, pain_bus=, embodiment=)` builder.** Follows the `build_pain_bus` / `build_executor` pattern. Required keyword-only args prevent silent wiring failures.

13. **[MEDIUM] Registry lives on Embodiment** (one per agent). Passed to bio-enrichment as parameter. Multi-agent isolation preserved.

14. **[LOW] Archetype-level reflex files.** Reflexes in separate `_data/reflexes/humanoid.yaml` files, not inline in entity spec. Reduces spec bloat (base_humanoid.yaml stays under 150 lines).

15. **[MEDIUM] Three plans need coordination doc.** Component damage, proprioceptive discovery, and percept reflexes all modify EnrichmentResult + entity YAML + enrichment→executor flow. Need a shared "Embodiment V2 Coordination" one-pager with implementation order and interface contract.

## Revised Key Decisions (post-review)

1. **Responses are tool invocation specs, not enum dispatch.** YAML declares `response: {tool: damage_component, params: {...}}`. Custom handlers via `ReflexHandler` protocol for non-tool responses.

2. **Pain pipeline is the ONLY NAc learning path.** Reflexes cause damage → pain → NAc. No separate Reaction→NAc path. Prevents double-counting.

3. **Pre-emption suppresses reflex intensity.** `effective_intensity = reflex_intensity * (1 - preemption_intensity)`. Learning to anticipate reduces damage. Correct biological gradient.

4. **Habituation + sensitization.** Intensity = `base * habituation_decay * sensitization_boost` where `habituation_decay = 1 / (1 + k * exposure_count)` and `sensitization_boost = 1 + s * (1 - component.integrity)`.

5. **Body-part extraction has confidence threshold.** Ambiguous targets → torso default. Prevents affordance death spirals from misparsed text.

6. **`detect_keywords` not `triggers`.** Avoids naming collision with FailureMode triggers.

7. **Archetype-level reflex files.** Separate YAML, `!include` from entity spec.

8. **Canonical `build_reflex_registry()` builder** with required keyword-only args.

## Files (revised)

| File | Change | LOC |
|------|--------|-----|
| `embodiment/reflex.py` | NEW: ReflexSpec, ReflexHandler protocol, ReflexRegistry, evaluate_reflexes(), build_reflex_registry() | ~180 |
| `embodiment/spec.py` | Parse `reflexes` with `detect_keywords` from entity YAML | ~25 |
| `integration/bio_enrichment.py` | Call reflex evaluation, add `reflexes_fired` to EnrichmentResult, telemetry | ~30 |
| `_data/reflexes/humanoid.yaml` | NEW: base humanoid reflexes (attack_flinch, fire_burn, impact_brace, startle, environment_cold) | ~50 |
| `_data/reflexes/quadruped.yaml` | NEW: dragon reflexes (fire-immune, wing-fold, tail-guard) | ~30 |
| `simulation/tools.py` | Remove auto-damage from SendMessageTool | -30 |
| `runtime/bootstrap.py` | Wire build_reflex_registry into build_bio_stack | ~10 |
| **Total** | | **~295 net** |
