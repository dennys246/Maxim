# Display Simplification Plan

**Goal:** Replace 4 overlapping output systems and 5+ display flags with a single `--display` flag that controls what the user sees. Default should be clean narrative interaction, not a wall of system internals.

**Status:** Planning  
**Priority:** First post-publication UX improvement (v1.0.1)  
**Estimated effort:** ~400 LOC across ~15 files

---

## The Problem Today

### Four output systems dumping to the same terminal

| System | Source | Controls | Default |
|--------|--------|----------|---------|
| sim_logger | `sim_log()` calls with subsystem labels | `--show` channels | **ALL** — every subsystem visible |
| structured_logging | Agentic event buffer | `--agentic-verbosity` 0-3 | Level 1 (goals, tools) |
| Direct print() | ~40 hardcoded `print()` calls in orchestrator, campaign_runner, etc. | None — always prints | Always on |
| Bio-system tracers | hippo/nac/atl detailed traces | `--debug` subsystems | Off (good) |

### Five flags that overlap and confuse

```
--verbosity 0|1|2          # Python logging level
--agentic-verbosity 0|1|2|3  # Agentic event buffer level
--show bio|exec|sim|...    # sim_logger channel filter
--debug hippo|nac|all      # Bio-system detailed traces
--no-agentic-console       # Suppress agentic event output
```

Users have to understand 4 systems and 5 flags to control what they see. Most just want "show me the story" or "show me everything."

### Result: noisy defaults

A user running `maxim --sim "test memory"` sees:
```
  SIMULATION MODE — ADVERSARIAL persona
  0.12s [PERCEPT     ] [ScenarioSource] You enter the tavern...
  0.14s [HIPPOCAMPUS ] Captured memory id=abc123 (sal=0.8, nov=0.3)
  0.15s [NAc         ] Link: see_merchant → approach (RPE=0.45, conf=0.91)
  0.16s [SCN         ] Temporal bin: evening (sig=0.32)
  0.17s [ATL         ] Concept: tavern (cat=location, conf=0.88)
  0.18s [EXEC        ] Tool: speak(message="Hello, merchant")
  0.19s [PIPELINE    ] LLM call: 847 tokens, 1.2s
  0.20s [FEAR        ] Approved: speak — no safety concern
  Saving orchestrator memory...
  Building simulation report...
```

When they probably wanted:
```
  Scene: You enter the tavern. A merchant watches you from behind the counter.
  Agent: "Hello, merchant. What do you have for sale?"
  > What do you do? [browse / haggle / leave]
```

---

## Proposed Design: One Flag, Three Tiers

### `--display` flag

```
--display interaction   # (DEFAULT) Clean narrative — percepts, actions, prompts
--display bio           # + condensed bio-system annotations
--display debug         # + full system traces, timing, internal state
```

That's it. One flag, three values, clear names.

### What each tier shows

#### `--display interaction` (default)

The user sees only what matters for following the story:

| What | Example |
|------|---------|
| Scene delivery | `Scene: You enter the tavern. A merchant watches you from behind the counter.` |
| Agent actions | `Agent: speak("Hello, merchant")` |
| Agent responses | `Agent: "I'd like to see your wares."` |
| Choice prompts | `> What do you do? [browse / haggle / leave]` |
| Entity state (if present) | `[guard: trust=0.4, suspicion=0.6]` |
| Turn markers | `--- Turn 3 ---` |
| Final summary | `Finished: 12 turns, 8 actions, completed` |

What's hidden:
- All bio-system events (hippo, nac, scn, atl, fear, pain)
- Pipeline internals (LLM calls, token counts, timing)
- Orchestrator status messages ("Saving memory...", "Building report...")
- System startup/shutdown messages

#### `--display bio`

Everything in `interaction`, plus condensed one-line bio annotations:

| What | Example |
|------|---------|
| Memory capture | `  [memory] Captured: "merchant offered healing potion"` |
| Memory recall | `  [memory] Recalled: 3 memories about "merchant"` |
| Causal learning | `  [causal] Learned: threaten → hostility (conf 0.82)` |
| Pain signal | `  [pain] Weapon shattered — intensity 0.6` |
| Fear gate | `  [safety] Blocked: rm -rf / (fear threshold exceeded)` |
| Concept formed | `  [concept] New: "tavern" (category: location)` |
| Temporal pattern | `  [temporal] Evening encounters tend to be hostile` |

Format: indented, muted color, brief. Not the raw subsystem trace — a human-readable summary.

#### `--display debug`

Everything in `bio`, plus full system output:

| What | Example |
|------|---------|
| Full sim_logger traces | `0.14s [HIPPOCAMPUS] Captured memory id=abc123 (sal=0.8, nov=0.3)` |
| Pipeline timing | `0.19s [PIPELINE] LLM call: 847 tokens, 1.2s, model=mistral-7b` |
| Agentic events | `[AGENTIC] loop_iteration | cycle=47, dt=0.03s` |
| Bio-system tracers | `[CAPTURE] #1 abc123def sal=0.82 nov=0.45 ✓ memory_recall` |
| Orchestrator status | `Saving orchestrator memory...` |
| All current output | Everything that shows today |

This is what `--verbosity 2 --show all --debug all` produces today — just behind a single flag.

---

## Flag Consolidation

### Before (5 flags)

```
--verbosity 0|1|2
--agentic-verbosity 0|1|2|3
--show bio|exec|sim|memory|safety|all
--debug hippo|nac|atl|scn|all
--no-agentic-console
```

### After (1 primary flag + 1 escape hatch)

```
--display interaction|bio|debug     # Controls everything (default: interaction)
--trace hippo|nac|atl|scn|all       # Detailed bio-system traces to stderr (unchanged)
```

The `--trace` flag survives because it's genuinely different — it produces detailed per-operation traces to stderr for debugging specific subsystems. Everything else collapses into `--display`.

### Backward compatibility

The old flags continue to work but are hidden from `--help`:

| Old flag | New equivalent | Behavior |
|----------|---------------|----------|
| `--verbosity 0` | `--display interaction` | Mapped internally |
| `--verbosity 2` | `--display debug` | Mapped internally |
| `--show bio` | `--display bio` | Mapped internally |
| `--show all` | `--display debug` | Mapped internally |
| `--debug hippo` | `--trace hippo` | Renamed but same behavior |
| `--agentic-verbosity N` | Removed | Controlled by display tier |
| `--no-agentic-console` | Removed | `interaction` tier suppresses it |

If a user passes both `--display` and an old flag, `--display` wins with a deprecation warning.

---

## Implementation Plan

### Phase D-0: Display tier infrastructure (~80 LOC)

Add `DisplayTier` enum and global state to sim_logger.py:

```python
class DisplayTier(enum.IntEnum):
    INTERACTION = 0  # Percepts, actions, prompts only
    BIO = 1          # + condensed bio annotations
    DEBUG = 2        # + full system traces

_display_tier: DisplayTier = DisplayTier.INTERACTION
```

Add `display_log()` function that respects the tier:

```python
def display_log(tier: DisplayTier, message: str, **kwargs):
    """Log a message that only appears at the specified tier or above."""
    if _display_tier >= tier:
        _emit(message, **kwargs)
```

**Files:** sim_logger.py  
**Tests:** Test tier filtering, test default is INTERACTION

### Phase D-1: Interaction tier — clean narrative output (~120 LOC)

New display functions for the clean narrative layer:

```python
def display_scene(text: str): ...        # Scene delivery
def display_action(tool: str, params: dict): ...  # Agent action
def display_response(text: str): ...     # Agent response text
def display_prompt(request: PromptRequest): ...   # Choice prompt
def display_entity_state(entities: dict): ...     # Entity sensors
def display_turn(n: int): ...            # Turn marker
def display_summary(result: SimulationResult): ...  # Final summary
```

Wire these into:
- `dm_runtime.py` — scene delivery, choice prompts, entity state
- `orchestrator.py` — turn markers, final summary
- Agent loop response path — action + response display

**Files:** sim_logger.py, dm_runtime.py, orchestrator.py, campaign_runner.py  
**Tests:** Test each display function produces expected output

### Phase D-2: Bio tier — condensed annotations (~100 LOC)

Add condensed bio-system formatters:

```python
def display_memory_capture(content: str): ...
def display_memory_recall(query: str, count: int): ...
def display_causal_learn(event: str, outcome: str, confidence: float): ...
def display_pain(source: str, intensity: float): ...
def display_fear_gate(tool: str, approved: bool): ...
def display_concept(name: str, category: str): ...
```

Wire these as callbacks from the existing bio-system code — each subsystem already has logging points, we just need to add a `display_*` call alongside the existing `sim_log()` call.

**Files:** sim_logger.py, hippocampus.py (or capture path), nac.py (observe path), pain_bus.py, fear agent  
**Tests:** Test bio annotations appear at BIO tier, hidden at INTERACTION

### Phase D-3: Silence direct print() calls (~80 LOC, tedious but simple)

Audit all ~40 `print()` calls in simulation/runtime code. For each one:
- **Is it narrative?** (scene, response, prompt) → Replace with `display_*()` call
- **Is it status?** ("Saving memory...", "Building report...") → Replace with `display_log(DEBUG, ...)`
- **Is it a report?** (final summary) → Replace with `display_summary()`

This is the most tedious phase but the most impactful — it's what makes the default quiet.

**Files:** orchestrator.py, campaign_runner.py, agent_loop.py, loop_controller.py, report.py, interactive.py, research_orchestrator.py  
**Tests:** Test that INTERACTION tier produces no system status messages

### Phase D-4: CLI flag wiring (~40 LOC)

Add `--display` to cli_parser.py, wire into the tier system:

```python
core.add_argument(
    "--display",
    type=str,
    default="interaction",
    choices=["interaction", "bio", "debug"],
    help="Output detail level: interaction (clean narrative, DEFAULT), "
         "bio (+ memory/learning annotations), debug (+ full system traces).",
)
```

Map old flags to new tier. Rename `--debug` to `--trace`. Hide deprecated flags from help.

**Files:** cli_parser.py, cli.py, api.py (configure)  
**Tests:** Test flag parsing, test backward compat mapping

### Phase D-5: Python API alignment (~30 LOC)

Update `maxim.configure()` to accept the new tier:

```python
maxim.configure(display="interaction")  # or "bio" or "debug"
```

Update `maxim.imagine()` and `maxim.campaign()` defaults to use INTERACTION tier.

**Files:** api.py  
**Tests:** Test configure sets tier correctly

---

## Phase Summary

| Phase | What | LOC | Key outcome |
|-------|------|-----|-------------|
| D-0 | DisplayTier enum + display_log() | ~80 | Infrastructure |
| D-1 | Interaction display functions | ~120 | Clean narrative output |
| D-2 | Bio-tier condensed annotations | ~100 | Human-readable bio events |
| D-3 | Silence ~40 print() calls | ~80 | Default is quiet |
| D-4 | CLI flag wiring + backward compat | ~40 | One flag to rule them all |
| D-5 | Python API alignment | ~30 | configure(display=...) |
| **Total** | | **~450** | |

---

## User-Facing Result

### Before

```bash
maxim --sim "test memory"
# 50+ lines of system output per turn
```

### After

```bash
maxim --sim "test memory"
# Clean narrative: scenes, actions, prompts

maxim --sim "test memory" --display bio
# + memory captures, causal learning, pain signals

maxim --sim "test memory" --display debug
# Full system output (same as today)

maxim --sim "test memory" --trace hippo
# + detailed hippocampus operation trace to stderr
```

---

## Invariants

- **INTERACTION tier must never produce system-internal output.** If a user sees `[PIPELINE]` or "Saving orchestrator memory..." in interaction mode, that's a bug.
- **BIO tier annotations must be human-readable.** Not `id=abc123 sal=0.82 nov=0.45` — instead `Captured: "merchant offered healing potion"`.
- **DEBUG tier must produce everything current output produces.** No regression for power users.
- **Old flags must not break.** Deprecation warning, then mapped to new tier.
- **`--trace` stays separate.** It's stderr, detailed, per-subsystem — different from display tier.
- **Python API and CLI must be consistent.** `maxim.configure(display="bio")` = `--display bio`.

---

## Testing Strategy

- **D-0:** Unit test DisplayTier filtering (message at BIO tier hidden when INTERACTION active)
- **D-1:** Integration test: run a mock campaign at INTERACTION tier, assert no subsystem labels in output
- **D-2:** Integration test: run at BIO tier, assert bio annotations present, no raw traces
- **D-3:** Grep test: assert zero `print()` calls in simulation/ and runtime/ (all replaced)
- **D-4:** CLI test: `--display bio` sets correct tier; `--verbosity 2` maps to DEBUG; `--display` + `--show` warns
- **D-5:** API test: `configure(display="interaction")` then `imagine()` produces clean output
