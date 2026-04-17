# Valence Annotation PoC — SEM Pain to Hebbian Edge Valence

**Date:** 2026-04-17
**Status:** PASS
**Plan:** [substrate_valence_annotation.md](../plans/archive/substrate_valence_annotation.md) Stage 3

## Scenario

Two episodes, one agent, no LLM calls (fully deterministic):

1. **Episode 1 (rusty sword + pain):** Agent perceives concepts `text_rusty_sword`, `text_heavy`, `text_sharp`. A pain reaction fires (intensity 0.8, `Valence.NEGATIVE`) simulating a failed "swing" affordance from `CerebellumModulator`.

2. **Episode 2 (sunny day):** Agent perceives `text_sunny_day`, `text_warm_breeze`. No reactions fire.

## Measurements

### (a) Edge valence annotation

| Edge | Valence |
|------|---------|
| `text_rusty_sword` → `text_heavy` | **-0.800** |
| `text_rusty_sword` → `text_sharp` | **-0.800** |
| `text_sunny_day` → `text_warm_breeze` | **no valence metadata** |

Pain in Episode 1 annotates all co-activated edges with negative valence. Episode 2 (neutral) has no valence metadata on its edges.

### (b) Spreading activation valence propagation

From `text_rusty_sword` cue (`include_valence=True`):

| Node | Activation | Valence |
|------|-----------|---------|
| `text_heavy` | 0.21 | **-0.800** |
| `text_sharp` | 0.21 | **-0.800** |

From `text_sunny_day` cue:

| Node | Activation | Valence |
|------|-----------|---------|
| `text_warm_breeze` | 0.21 | **0.000** |

Negative valence propagates through the binding graph only for concepts associated with the painful episode.

### (c) Clean agent control

A second hippocampus with the same sword concepts but **no pain reaction**:

| Node | Activation | Valence |
|------|-----------|---------|
| `text_heavy` | 0.21 | **0.000** |
| `text_sharp` | 0.21 | **0.000** |

Confirms valence comes from the reaction, not from concept co-activation alone.

### (d) Persistence round-trip

`h.dump()` → `h2.load_state(state)` → edge valence preserved exactly (-0.800 = -0.800).

## Reproduction

```bash
PYTHONPATH=src python scripts/valence_annotation_poc.py
```

Or via tests:

```bash
python -m pytest tests/substrate/test_valence_annotation.py -v
```

26 tests covering:
- Reaction capture into pending episode (3 tests)
- Episode valence computation (7 tests)
- Edge valence annotation with decay (5 tests)
- Persistence round-trip (4 tests)
- Spreading activation valence propagation (5 tests)
- retrieve_on_cue include_valence (2 tests)

## Key design decisions

1. **Reactions stored transiently on `PendingEpisodeState`**, net valence computed at `finalize()` and stored as a simple float on the frozen `Episode` dataclass. No coupling between `episode.py` and `reactions/types.py`.

2. **Valence decay factor = 0.95** applied to existing edge valence before adding the new episode's contribution. Prevents saturation when an agent repeatedly interacts with the same entity.

3. **Zero-valence episodes skip annotation entirely** — no unnecessary metadata written. Old episodes without the `valence` field default to 0.0 on load.

4. **`propagate_valence=False` is the default** for `spreading_activation()` — no behavioral change for existing callers. When `True`, return type changes from `dict[str, float]` to `dict[str, tuple[float, float]]`.

## Connection to concept decomposition

This PoC uses sentence-level node IDs (`text_rusty_sword`). With concept decomposition enabled (`MAXIM_CONCEPT_DECOMPOSITION=1`), the same scenario would produce finer-grained nodes (`chunk_rusty_sword`, `chunk_heavy`, `chunk_sharp`) — each getting its own valence annotation. The valence signal would be more precise: "rusty sword" carries negative valence, but the action concept "swing" would not (it's not a co-activated node in the substrate).

## What's next

- **Stage 4 (positive valence):** `CerebellumModulator._emit_success_reaction()` for successful affordances → symmetric positive edge annotation.
- **Agent-level consumption:** Prompt assembler could include valence as context ("you have negative associations with X"), or DefaultNetwork could trigger avoidance reactions.
- **NAc `distribute_reward` wiring:** The reaction-capture infrastructure from this plan can feed the reward-bias path, connecting SEM reactions to EC's threshold adjustment.
