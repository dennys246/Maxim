# Asset Foundry Guide

The Asset Foundry is an autonomous pipeline that generates, validates, tests, and scores SEM (Sensor-Entity-Modulator) components. It expands the component library without manual YAML authoring while stress-testing the bio-stack against novel entity designs.

## Quick Start

```bash
# Generate 10 fantasy weapons (template-based, no LLM needed)
maxim --foundry "medieval weapons" --foundry-genre fantasy --foundry-category weapons

# Generate mixed components for a theme
maxim --foundry "underwater civilization" --foundry-count 20 --foundry-genre fantasy

# Dry run — generate + validate only, skip gauntlet testing
maxim --foundry "horror creatures" --foundry-count 5 --foundry-genre horror --foundry-dry-run

# With LLM for creative generation (requires configured LLM)
maxim --foundry "cyberpunk weapons" --foundry-genre cyberpunk --foundry-category weapons --llm mistral-7b
```

## How It Works

The foundry runs four phases in sequence:

### Phase F-0: Generation

Takes a theme prompt, genre, and optional category. Generates candidate SEM component specs using the EntityDesigner:
- With an LLM: creative generation guided by the SEM schema
- Without an LLM: template-based fallback with sensible defaults per entity type

Each candidate gets a unique name, genre tags, and is saved as YAML to `~/.maxim/foundry/{run_id}/candidates/`.

Sub-themes are rotated automatically to avoid repetition (e.g., "close-range melee", "long-range with overheating", "stealth with limited charges").

### Phase F-1: Validation

Rejects malformed or nonsensical specs before wasting simulation cost:

| Check | What It Catches |
|-------|----------------|
| Schema | Missing required fields (name, sensors, modulators) |
| Sensor sanity | Reversed ranges, initial outside range, zero-width ranges |
| Affordance sanity | Empty modulators, missing descriptions |
| Failure mode sanity | Triggers referencing nonexistent sensors, pain outside [0,1] |
| Semantic sanity | Very large range spans, unreachable failure thresholds |

### Phase F-2: SEM Protocol Tests + Gauntlet

**8 structural tests** (fast, no LLM):
1. Instantiation via `_parse_entity()`
2. Sensor initialization within range
3. Affordance enumeration with descriptions
4. Tool generation without collisions
5. Failure mode trigger validation
6. Entity tree composition (`walk()`)
7. Vital metrics population
8. Embodiment wrapping

**3-encounter gauntlet** (exercises the bio-stack):
1. **Discovery** — observe sensors via sense tools
2. **Interaction** — invoke affordance tools, check NAc learning
3. **Stress** — push sensors toward failure thresholds, verify pain cascade

Each gauntlet gets a fresh bio-stack (Hippocampus, NAc, PainBus) for isolation.

### Phase F-3: Scoring + Curation

Ranks candidates on 4 dimensions:

| Dimension | Weight | Measures |
|-----------|--------|----------|
| Hippocampal engagement | 30% | Did the agent remember this entity? |
| Causal learning | 30% | Did NAc learn cause-effect from this entity? |
| Pain/failure activation | 20% | Did failure modes fire? Did pain cascade work? |
| Affordance usage | 20% | How many affordances were actually used? |

**Buckets:**
- **Promote** (score > 0.7): High-quality, ready for human review
- **Review** (0.4-0.7): Interesting but flawed
- **Reject** (< 0.4): Low engagement or broken

## Output Structure

Each run produces:
```
~/.maxim/foundry/{run_id}/
  config.yaml          # Theme, genre, count, model
  candidates/          # Raw generated YAML specs
  rejected/            # Failed validation (with error details)
  results/             # Per-candidate gauntlet results + scores
  promoted/            # Top scorers (ready for human review)
  scores.json          # Machine-readable scoring summary
  report.md            # Human-readable report
```

## CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--foundry` | str | — | Theme prompt (e.g., "cyberpunk weapons") |
| `--foundry-count` / `--count` | int | 10 | Components to generate |
| `--foundry-genre` | str | fantasy | Genre tag |
| `--foundry-category` | str | auto | Category (weapons, creatures, npcs, items, environments, vehicles, bodies) |
| `--foundry-dry-run` | flag | off | Generate + validate only |

## Entity Context in Prompts

When running with `--embodiment`, the agent's prompt automatically includes an **entity context section** summarizing the entity's capabilities:

```
=== Entity: plasma_lance (weapon) ===
plasma_thrust: High damage thrust, drains charge quickly (params: target: str, force: float)
sweep_strike: Wide area attack, moderate charge drain (params: target: str)
heat_vent: Release built-up heat, brief vulnerability window
Failure risk: heat >= 450 triggers overheated; charge < 0.05 triggers charge_depleted.
```

This is auto-composed from the SEM entity spec and injected at IMPORTANT priority — after the identity section but before tool guidance. It complements `body_state` (which shows current sensor readings like "charge: 0.3") by giving the agent a static capability reference.

The entity context section is wired in both the CLI non-sim path and the simulation orchestrator AUT path.

## Synonym Generation + ComponentIndex

LLM-generated entities now include a `synonyms` field — a list of 5-10 alternative names a user might use to refer to the entity. This feeds the **ComponentIndex** semantic discovery layer.

**How it works in the foundry pipeline:**
1. `EntityDesigner.design()` generates synonyms as part of the spec
2. Promoted components carry synonyms into `~/.maxim/components/`
3. On next startup, `ComponentIndex` indexes those synonyms into Layer 1 (alias table)
4. Natural language queries resolve instantly: `"old iron door"` → `environments/rusty_gate`

**Dedup via ComponentIndex:** Before promoting a candidate, the foundry calls `index.dedup_check(spec, threshold=0.80)`. If a near-duplicate exists (cosine similarity ≥ 0.80 against an existing component), the candidate is skipped with a log message: "Skipped {name} — similar to existing {match.ref} (cosine {score:.2f})".

See the [Embodiment Guide — ComponentIndex](../embodiment_guide.md#componentindex--semantic-discovery) for the full API and architecture.

## Using Promoted Components

Promoted components are saved to `~/.maxim/foundry/{run_id}/promoted/`. To use one:

```bash
# Run a sim with a promoted component
maxim --sim "test the plasma cutter" --embodiment weapons/plasma_cutter

# Copy to user components directory for permanent availability
cp ~/.maxim/foundry/20260419_180000/promoted/plasma_cutter.yaml ~/.maxim/components/weapons/
```

Components in `~/.maxim/components/` are automatically discovered by the ComponentRegistry.

## Cost

| Operation | Local LLM | Cloud LLM |
|-----------|-----------|-----------|
| Generate (per entity) | ~500 tokens, $0.00 | ~$0.005 |
| Gauntlet (3 encounters) | No LLM needed | $0.00 |
| Validate + SEM tests | No LLM | $0.00 |
| **Typical run (10 entities)** | **3-5 min, $0.00** | **~$0.05** |

## Extending the Scoring Rubric

The scoring system is extensible via `ScoringConfig`:

```python
from maxim.simulation.foundry import FoundryRunner, ScoringConfig

config = ScoringConfig(
    dimensions={
        "hippocampal_engagement": 0.25,
        "causal_learning": 0.25,
        "pain_failure": 0.25,
        "affordance_usage": 0.25,
    },
    promote_threshold=0.6,   # Lower bar for promotion
    reject_threshold=0.3,
)

runner = FoundryRunner(
    theme="test weapons",
    genre="fantasy",
    scoring_config=config,
)
result = runner.run(count=5)
```

## Invariants

- Foundry **never auto-commits** to the component library. Human review required.
- Generated components pass the **same validation** as hand-written ones.
- Each gauntlet gets a **fresh bio-stack**. Zero state leakage.
- **Infra failures are not candidate failures.** Setup crashes are logged separately.

## See Also

- [Embodiment Guide](../embodiment_guide.md) — SEM entity system overview
- [YAML Reference](../embodiment_yaml_reference.md) — Component YAML format
- [Asset Foundry Plan](../plans/deferred/asset_foundry_plan.md) — Design document
