# Cradle of Artificial Civilization — non-linguistic sensorimotor learning

**Status:** Shell plan (2026-04-24), research-grade
**Scope:** ~500-800 LOC across embodiment, proprioception, simulation
**Depends on:** [orchestrator_sem_damage.md](orchestrator_sem_damage.md) (reactive chain must work first), [temporal_credit_integration.md](temporal_credit_integration.md) (shipped — learning infrastructure)
**Gates:** None; research exploration
**Branch:** TBD

---

## Vision

A newborn agent that learns from sensation, not language.  No prompt engineering, no tool descriptions, no linguistic scaffolding — just a body with sensors, a world with stimuli, and the bio-pipeline learning what hurts, what helps, and what to pay attention to.

The cradle is an environment where we strip away the LLM's world knowledge and force learning through direct sensorimotor experience.  The agent doesn't "know" fire is dangerous because GPT said so — it knows because touching fire triggered pain in its thermal sensors, the pain signal propagated through PainBus to NAc, and NAc formed a causal link between the thermal spike and negative valence.

**Bio-plausible framing:** Human infants learn sensorimotor contingencies before language.  They learn that touching hot things hurts (somatosensory → pain → avoidance), that hunger is resolved by feeding (interoceptive → motor → reward), and that certain textures predict certain outcomes (tactile → associative → prediction).  This happens through distributed body sensors, not through linguistic instruction.

The existing Maxim bio-pipeline (PainBus → NAc → hippocampus → TemporalCreditDistributor → ValenceSignal) is already the right learning substrate.  What's missing is the **sensory input layer** — distributed body sensors that produce non-linguistic signals the pipeline can learn from.

## Design

### Layer 1: Somatosensory Registry

Distributed sensor patches across body regions, each registering a specific sensation modality.

```yaml
# Extension to base_humanoid.yaml
entity:
  name: base_humanoid
  somatosensory:
    left_hand:
      thermal: { range: [-1, 1], initial: 0.0 }   # -1=cold, 0=neutral, +1=hot
      pressure: { range: [0, 1], initial: 0.0 }    # 0=none, 1=crushing
      sharp: { range: [0, 1], initial: 0.0 }       # 0=none, 1=piercing
      texture: { range: [0, 1], initial: 0.5 }     # 0=smooth, 1=rough
    right_hand:
      thermal: { range: [-1, 1], initial: 0.0 }
      pressure: { range: [0, 1], initial: 0.0 }
      sharp: { range: [0, 1], initial: 0.0 }
      texture: { range: [0, 1], initial: 0.5 }
    torso:
      thermal: { range: [-1, 1], initial: 0.0 }
      pressure: { range: [0, 1], initial: 0.0 }
      impact: { range: [0, 1], initial: 0.0 }      # 0=none, 1=blunt trauma
    face:
      thermal: { range: [-1, 1], initial: 0.0 }
      light: { range: [0, 1], initial: 0.5 }       # 0=dark, 1=blinding
      chemical: { range: [0, 1], initial: 0.0 }    # 0=none, 1=irritant
```

**Implementation:** `SomatosensoryPatch` dataclass in `embodiment/somatosensory.py`.  Each patch is a named body region with a dict of modality → current value.  The registry is a `dict[str, SomatosensoryPatch]` on `Entity`.

**Failure modes from somatosensory:**
- `thermal > 0.7` on any patch → `burn` failure mode → PainSignal(intensity=thermal, source=patch_name)
- `thermal < -0.7` → `frostbite` failure mode
- `pressure > 0.8` → `crush` failure mode
- `sharp > 0.5` → `laceration` failure mode
- Two modalities co-occurring amplify: `thermal > 0.5 AND sharp > 0.3` → `searing_wound` (higher pain)

These feed directly into the existing `body.tick()` → `FailureMode.evaluate()` → `PainBus.publish()` pipeline.

### Layer 2: Interoceptive Drive System

Internal state sensors that produce motivational signals — hunger, thirst, fatigue, temperature regulation.

```yaml
interoception:
  hunger:
    range: [0, 1]
    initial: 0.0
    drift: 0.002          # increases per tick (~0.1/minute at 2Hz)
    satisfaction_action: eat
    deprivation_threshold: 0.7   # pain signal fires above this
    satisfaction_threshold: 0.3  # reward signal fires when dropping below
  thirst:
    range: [0, 1]
    initial: 0.0
    drift: 0.003
    satisfaction_action: drink
    deprivation_threshold: 0.6
    satisfaction_threshold: 0.3
  core_temperature:
    range: [-1, 1]
    initial: 0.0           # 0 = comfortable
    drift: 0.0             # environment-driven, not time-driven
    pain_band: [-0.6, 0.6] # pain outside this range
```

**Key insight:** Interoceptive drives don't require language.  Hunger rises over time.  When it crosses the deprivation threshold, a PainSignal fires (low intensity, persistent).  When the agent eats and hunger drops below the satisfaction threshold, a positive Reaction fires.  NAc learns: `eat → hunger_drops → positive`.  No prompt needed — the causal link forms from the sensory experience alone.

### Layer 3: Cradle Environment

A minimal simulation environment designed to teach non-linguistic contingencies:

```
Cradle scenario: "infant_learning"
  - No goal prompt (the agent discovers its own drives)
  - Environment objects:
    - fire_pit: thermal=0.9 when touched, visible at distance
    - water_source: satisfies thirst when interacted with
    - food_source: satisfies hunger when interacted with
    - sharp_rock: sharp=0.8 when grasped
    - soft_blanket: texture=0.1 (smooth), thermal=0.2 (warm)
  - The agent has only motor primitives: reach, grasp, move, release
  - No tool descriptions beyond the motor name
  - LLM prompt is minimal: "you are a body. explore."
```

**The experiment:**

1. **Phase 1: Pain avoidance (~10 min)** — Agent touches fire_pit.  Thermal sensor spikes → burn failure mode → PainSignal → NAc negative link on "fire_pit" + "reach" co-occurrence.  Does the agent avoid reaching toward fire_pit on subsequent encounters?

2. **Phase 2: Drive satisfaction (~15 min)** — Hunger rises.  Agent explores.  Eventually interacts with food_source.  Hunger drops → positive Reaction → NAc positive link on "food_source" + "eat".  Does the agent seek food_source when hungry?

3. **Phase 3: Texture discrimination (~10 min)** — Sharp_rock and soft_blanket look similar at distance.  Grasping sharp_rock → laceration pain.  Grasping soft_blanket → no pain + comfort (thermal warmth).  Does the agent learn to discriminate by texture memory?

4. **Phase 4: Cross-session transfer (~15 min)** — Resume session.  Does the agent avoid fire_pit WITHOUT being burned again?  Does it seek food_source when hungry WITHOUT re-discovering it?  This is the same cross-session test as temporal credit Set 1, but with non-linguistic signals.

### Layer 4: Memory Integration

Somatosensory events need to reach hippocampus for episodic memory, not just NAc for causal learning.

**Percept creation from somatosensory:**
- When a somatosensory patch changes significantly (delta > 0.2), create a Percept:
  - `content = f"{patch_name}:{modality}:{value:.2f}"`
  - `sensory = SensoryTag(modality=SensoryModality.TOUCH)`  
  - No natural language — just the sensor reading
- This Percept flows through the standard pipeline: PerceptTraceBuffer → hippocampus → episode

**Substrate encoding:**
- Somatosensory percepts get encoded via `LinguisticEncoder.encode()` using the content string
- EC pattern-completes "left_hand:thermal:0.85" with previous thermal experiences
- ATL forms substrate concepts for sensory patterns: "hot_touch", "sharp_grasp"
- NAc eligibility traces link these concepts to outcomes via temporal credit

**Bio-enrichment:**
- When the agent deliberates, bio-enrichment queries hippocampus: "what happened last time I felt this?"
- Recalled somatosensory episodes provide non-linguistic context: the agent "remembers" the burn
- This memory modulates behavior without the LLM being told "fire is dangerous"

### Layer 5: Somatosensory Homunculus (future)

Inspired by the cortical homunculus — a map of body representation where more sensitive regions have more sensor patches.  Hands and face get higher-resolution patches than torso.  Damage to a region reduces sensitivity (numbness), creating a degraded-perception learning problem similar to the Darkened Cavern campaign.

This is explicitly deferred — Layer 1-4 must prove the concept first.

## Relationship to existing systems

| System | Role in cradle |
|--------|---------------|
| PainBus | Receives somatosensory pain signals, routes to NAc + hippocampus |
| NAc | Forms causal links from somatosensory events → outcomes |
| TemporalCreditDistributor | Credits somatosensory-linked substrate nodes after temporal delay |
| ValenceSignal | Modulates WMS thought salience from somatosensory valence |
| Hippocampus | Stores somatosensory episodes for cross-session recall |
| ThoughtGate | Fires on novel somatosensory input (first burn → high novelty) |
| `_goal_reward_bias` | Learns drive-satisfaction goals ("eat when hungry" → positive bias) |

**Key: the cradle uses NO new learning infrastructure.**  Everything runs on the pipeline shipped in temporal_credit_integration.  The novel contribution is the sensory input layer and the experimental protocol.

## Staging

### Stage 1: Somatosensory Registry (~100 LOC)

`embodiment/somatosensory.py`: `SomatosensoryPatch`, `SomatosensoryRegistry`.  YAML schema extension for `base_humanoid`.  Wire into `body.tick()` for failure mode evaluation.

### Stage 2: Interoceptive Drives (~80 LOC)

Drive system with drift, deprivation pain, satisfaction reward.  Wire hunger/thirst into existing `base_humanoid` sensors (they already exist — just need drift + thresholds).

### Stage 3: Cradle Environment (~150 LOC)

Minimal sim scenario with environmental objects that produce somatosensory stimuli on interaction.  The orchestrator narrates object proximity; SEM damage wiring (from `orchestrator_sem_damage.md`) handles actual sensor changes.

### Stage 4: Percept + Memory Integration (~100 LOC)

Somatosensory → Percept creation → PerceptTraceBuffer → hippocampus pipeline.  Substrate encoding for sensory patterns.

### Stage 5: Experiments + Validation (~50 LOC)

Four-phase experiment protocol.  Automated via script like `temporal_credit_validation.sh`.

### Stage 6: Somatosensory Homunculus (deferred)

Variable-resolution body map.  Damage-induced numbness.  Deferred until Stages 1-5 prove the concept.

## Validation criteria

| Phase | Hypothesis | Pass condition |
|-------|-----------|----------------|
| 1 | Pain avoidance | Agent avoids fire_pit after first burn (measured by action selection frequency) |
| 2 | Drive satisfaction | Agent seeks food_source when hunger > 0.7 (measured by time-to-eat decreasing across episodes) |
| 3 | Texture discrimination | Agent prefers soft_blanket over sharp_rock after first laceration (measured by grasp target selection) |
| 4 | Cross-session transfer | Session 2 agent avoids fire_pit without re-experiencing burn + seeks food when hungry without re-discovering food_source |

## The research claim

"A bio-inspired cognitive architecture demonstrates non-linguistic sensorimotor learning: an agent with no prior knowledge learns pain avoidance, drive satisfaction, and texture discrimination through direct bodily experience, retaining these associations across sessions without fine-tuning."

This is the cradle of artificial civilization — the moment the agent stops being a language model with a body bolted on, and starts being a body that happens to have language.
