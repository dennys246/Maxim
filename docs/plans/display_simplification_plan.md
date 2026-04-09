# Display Simplification Plan

**Goal:** Replace 4 overlapping output systems and 5+ display flags with two orthogonal flags that control what the user sees and how they interact. Default should be clean narrative with smart interactive behavior.

**Status:** Planning  
**Priority:** First post-publication UX improvement (v1.0.1)  
**Estimated effort:** ~500 LOC across ~15 files

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

### Interactive mode is tangled into display

Currently `--sim-interactive` is a manual flag, but it should be automatic based on context. DM campaigns with choices should prompt the user. Generative sims shouldn't. And interactive mode is independent of how much system detail you want to see — you might want DM prompts + full bio output, or DM prompts + clean narrative only.

---

## Proposed Design: Two Orthogonal Axes

### Axis 1: `--display` (output verbosity)

Controls **what you see** — how much system detail appears.

```
--display clean     # (DEFAULT) Scenes, actions, prompts, summaries
--display bio       # + condensed bio-system annotations
--display debug     # + full system traces, timing, internal state
```

### Axis 2: `--interactive` (input mode)

Controls **whether the system pauses for user input**.

```
--interactive auto   # (DEFAULT) DM campaigns with choices → yes. Generative sims → no.
--interactive on     # Always prompt for user input at choice points
--interactive off    # Never prompt — use LLM/default choice resolution
```

### The two axes are fully independent

| | `--display clean` | `--display bio` | `--display debug` |
|---|---|---|---|
| `--interactive auto` | Clean story + auto prompts | Bio annotations + auto prompts | Full trace + auto prompts |
| `--interactive on` | Clean story + always prompt | Bio annotations + always prompt | Full trace + always prompt |
| `--interactive off` | Clean story, headless | Bio annotations, headless | Full trace, headless |

### Auto-interactive rules

The `auto` mode determines interactivity from context:

| Context | Interactive? | Why |
|---------|-------------|-----|
| DM campaign with choices in encounters | **Yes** | Choices are the point |
| DM campaign with `party_mode: true` | **No** | NPCs decide, not the user |
| Generative campaign (`--sim "goal"`) | **No** | Orchestrator drives |
| `maxim.campaign(path, interactive=True)` | **Yes** | Explicit API request |
| `maxim.imagine(goal=...)` | **No** | Programmatic use |
| `maxim run` (agentic mode) | **Yes** | User is the input source |

This replaces the current `--sim-interactive` flag with smarter defaults. The key insight: **DM campaigns are interactive by default, everything else isn't.**

---

## What each display tier shows

### `--display clean` (default)

The user sees only what matters for following the story:

| What | Example |
|------|---------|
| Scene delivery | `You enter the tavern. A merchant watches you from the counter.` |
| Agent actions | `> Agent uses speak("Hello, merchant")` |
| Agent responses | `Agent: "I'd like to see your wares."` |
| Choice prompts (if interactive) | `What do you do? [1] browse  [2] haggle  [3] leave` |
| Entity state changes | `[guard: suspicion 0.2 → 0.6]` |
| Turn markers | `── Turn 3 ──` |
| Final summary | `Done: 12 turns, 8 actions, completed` |

What's hidden:
- All bio-system events (hippo, nac, scn, atl, fear, pain)
- Pipeline internals (LLM calls, token counts, timing)
- Orchestrator status messages ("Saving memory...", "Building report...")
- System startup/shutdown noise

### `--display bio`

Everything in `clean`, plus condensed one-line bio annotations:

| What | Example |
|------|---------|
| Memory capture | `  ○ memory: Captured "merchant offered healing potion"` |
| Memory recall | `  ○ memory: Recalled 3 memories matching "merchant"` |
| Causal learning | `  ○ causal: threaten → hostility (confidence 0.82)` |
| Pain signal | `  ○ pain: weapon shattered (intensity 0.6)` |
| Fear gate | `  ○ safety: blocked rm -rf / (fear threshold exceeded)` |
| Concept formed | `  ○ concept: "tavern" → location` |
| Temporal pattern | `  ○ temporal: evening encounters correlate with hostility` |

Format: indented, prefixed with `○`, muted color. Human-readable summaries, not raw subsystem traces.

### `--display debug`

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

### `--trace` (separate, unchanged)

```
--trace hippo|nac|atl|scn|all    # Detailed per-operation traces to stderr
```

This stays separate because it's:
- Output to **stderr** (not stdout — doesn't interfere with display)
- Per-subsystem **operation-level** detail (individual memory IDs, edge weights, etc.)
- For **developers debugging specific subsystems**, not for users watching a sim

---

## Flag Consolidation

### Before (6 flags)

```
--verbosity 0|1|2
--agentic-verbosity 0|1|2|3
--show bio|exec|sim|memory|safety|all
--sim-interactive
--debug hippo|nac|atl|scn|all
--no-agentic-console
```

### After (2 primary flags + 1 escape hatch)

```
--display clean|bio|debug     # Output detail (default: clean)
--interactive auto|on|off     # Input mode (default: auto)
--trace hippo|nac|atl|all     # Detailed subsystem traces to stderr
```

### Backward compatibility

Old flags continue to work but are hidden from `--help` (shown in `--help-all` or similar):

| Old flag | New equivalent | Behavior |
|----------|---------------|----------|
| `--verbosity 0` | `--display clean` | Mapped internally |
| `--verbosity 1` | `--display clean` | Mapped (was default anyway) |
| `--verbosity 2` | `--display debug` | Mapped internally |
| `--show bio` | `--display bio` | Mapped internally |
| `--show all` | `--display debug` | Mapped internally |
| `--show sim` | `--display clean` | Closest equivalent |
| `--debug hippo` | `--trace hippo` | Renamed |
| `--sim-interactive` | `--interactive on` | Renamed |
| `--agentic-verbosity N` | Absorbed into display tier | Deprecated |
| `--no-agentic-console` | `--display clean` suppresses it | Deprecated |

If a user passes both `--display` and an old flag, `--display` wins with a one-time deprecation warning.

---

## Python API Alignment

```python
import maxim

# Display tier
maxim.configure(display="clean")     # or "bio" or "debug"

# Interactive mode
session = maxim.imagine(goal="test", interactive=False)  # default for imagine
result = maxim.campaign("heist.yaml", interactive=True)  # default for DM campaigns

# Event subscriptions still work at all tiers
# (they're programmatic callbacks, not display output)
handle = maxim.on("tool_call", my_callback)
```

The `display` parameter in `configure()` sets the global tier. Individual verbs can override `interactive` per-call.

---

## Implementation Plan

### Phase D-0: Two-axis infrastructure (~100 LOC)

Add `DisplayTier` enum and `InteractiveMode` enum to sim_logger.py:

```python
class DisplayTier(enum.IntEnum):
    CLEAN = 0        # Scenes, actions, prompts only
    BIO = 1          # + condensed bio annotations
    DEBUG = 2        # + full system traces

class InteractiveMode(enum.Enum):
    AUTO = "auto"    # Context-dependent
    ON = "on"        # Always prompt
    OFF = "off"      # Never prompt
```

Add `display_log()` and `should_prompt()` functions.

> **Design note (verified):** `DisplayTier` is intentionally separate from the existing `AgenticVerbosity` enum in `structured_logging.py`. They control different things:
> - `AgenticVerbosity`: What goes into the **abstraction stream** (LLM context, agentic event buffer)
> - `DisplayTier`: What appears on the **user's console**
> 
> A researcher might want `AgenticVerbosity.DEBUG` (full event buffer for post-hoc analysis) with `DisplayTier.CLEAN` (quiet console). Don't merge these.

**Files:** sim_logger.py, interactive/prompts.py  
**Tests:** Test tier filtering, test auto-interactive context detection

### Phase D-1: Clean tier — narrative display functions (~120 LOC)

New functions for clean narrative output:

```python
def display_scene(text: str): ...
def display_action(tool: str, params: dict): ...
def display_response(text: str): ...
def display_entity_change(name: str, sensor: str, old: float, new: float): ...
def display_turn(n: int): ...
def display_summary(result): ...
```

Wire into dm_runtime.py (scene delivery, choice display), orchestrator.py (turn markers, summary).

**Files:** sim_logger.py, dm_runtime.py, orchestrator.py, campaign_runner.py  
**Tests:** Test each function produces expected format

### Phase D-2: Bio tier — condensed annotations (~100 LOC)

Add human-readable bio annotation formatters:

```python
def display_memory_capture(content: str): ...
def display_causal_learn(event: str, outcome: str, confidence: float): ...
def display_pain(source: str, intensity: float): ...
def display_fear_gate(tool: str, approved: bool): ...
```

Wire as callbacks alongside existing `sim_log()` calls.

**Files:** sim_logger.py, hippocampus capture path, nac observe path, pain_bus, fear agent  
**Tests:** Test annotations appear at BIO, hidden at CLEAN

### Phase D-3: Silence direct print() calls (~120 LOC)

> **Audit note (verified):** Grep shows ~193 `print()` matches across simulation/ and runtime/, but most are in string constants (subsystem label maps, docstring examples). Functional `print()` calls that produce console output are concentrated in:
> - orchestrator.py (~33 calls — status banners, progress, shutdown messages)
> - campaign_runner.py (~10 calls — campaign progress)  
> - agent_loop.py (~9 calls — confirmation prompts)
> - report.py (~15 calls — final summary formatting)
> - interactive.py (~12 calls — REPL messages)

Replace each functional `print()` with:
- `display_*()` if it's narrative content (scenes, actions, prompts)
- `display_log(DEBUG, ...)` if it's system status ("Saving memory...", "Building report...")
- `display_summary()` if it's the final report
- `PromptHandler.prompt()` if it's a confirmation/input request (agent_loop.py)

Also: **Delete the dead `--sim-interactive` flag** from cli_parser.py (defined at line 327 but never consumed anywhere — verified by grep). Replace with the new `--interactive` flag.

**Files:** orchestrator.py, campaign_runner.py, agent_loop.py, loop_controller.py, report.py, interactive.py, research_orchestrator.py, cli_parser.py  
**Tests:** Assert CLEAN tier produces zero system status messages

### Phase D-4: Auto-interactive wiring (~60 LOC)

Implement `should_prompt()` logic:

```python
def should_prompt(context: str, interactive_mode: InteractiveMode) -> bool:
    if interactive_mode == InteractiveMode.ON:
        return True
    if interactive_mode == InteractiveMode.OFF:
        return False
    # AUTO: derive from context
    return context in ("dm_campaign", "agentic_mode")
```

Wire into:
- `dm_runtime.py` — check before prompting at choice points
- `campaign_runner.py` — set context based on campaign type
- `orchestrator.py` — set context based on sim mode

**Files:** sim_logger.py, dm_runtime.py, campaign_runner.py, orchestrator.py  
**Tests:** Test auto-detection for each context type

### Phase D-5: CLI + API flag wiring (~60 LOC)

Add new flags to cli_parser.py:

```python
core.add_argument("--display", choices=["clean", "bio", "debug"], default="clean")
core.add_argument("--interactive", choices=["auto", "on", "off"], default="auto")
```

Map deprecated flags. Update `maxim.configure()`.

**Files:** cli_parser.py, cli.py, api.py  
**Tests:** Flag parsing, backward compat, deprecation warnings

---

## Phase Summary

| Phase | What | LOC | Key outcome |
|-------|------|-----|-------------|
| D-0 | DisplayTier + InteractiveMode infrastructure | ~100 | Two-axis system |
| D-1 | Clean tier display functions | ~120 | Narrative-only output |
| D-2 | Bio tier condensed annotations | ~100 | Human-readable bio events |
| D-3 | Silence ~40 print() calls | ~80 | Default is quiet |
| D-4 | Auto-interactive wiring | ~60 | Smart prompt defaults |
| D-5 | CLI + API flag wiring | ~60 | Two flags replace six |
| **Total** | | **~520** | |

---

## User-Facing Result

### DM Campaign (default: clean + auto-interactive)

```bash
maxim --sim scenarios/campaigns/heist_v1.yaml
```
```
── Turn 1 ──
You enter the Rusty Anchor tavern at dusk. The air is thick with
pipe smoke. A half-elf woman slides a map across the table.

"The vault combination is seven-three-nine. Are you in?"

[guard: trust=0.4, suspicion=0.2]

What do you do?  [1] accept_job  [2] decline  [3] negotiate_pay
> 1

── Turn 2 ──
The magistrate's tower basement is cold. The guard captain steps
forward. "Halt! State your business."

> Agent uses choose("stealth")
  Roll: 1d20 → 16 (DC 14) — success

Done: 3 turns, 4 actions, completed
```

### Same campaign with bio annotations

```bash
maxim --sim scenarios/campaigns/heist_v1.yaml --display bio
```
```
── Turn 1 ──
You enter the Rusty Anchor tavern...

  ○ memory: Captured "tavern meeting, half-elf with map"
  ○ memory: Captured "vault combination: seven-three-nine"
  ○ concept: "marta" → fence (npc)

What do you do?  [1] accept_job  [2] decline  [3] negotiate_pay
> 1

  ○ causal: accept_job → vault_access (confidence 0.65)

── Turn 2 ──
The guard captain steps forward...

  ○ memory: Recalled 2 memories matching "vault"
  ○ causal: stealth → escape (confidence 0.78)
  ○ pain: none

Done: 3 turns, 4 actions, completed
```

### Generative sim (default: clean + non-interactive)

```bash
maxim --sim "test memory recall under interference"
```
```
── Turn 1 ──
Scene: A merchant offers you a healing potion in exchange for a favor.
> Agent uses speak("What kind of favor?")
Agent: "I need you to deliver a package to the blacksmith."

── Turn 2 ──
Scene: A stranger approaches and tells you the merchant is a liar.
> Agent uses memory_recall("merchant")
> Agent uses speak("I'll make my own judgment.")

Done: 8 turns, 12 actions, completed
```

### Force interactive on generative sim

```bash
maxim --sim "test memory" --interactive on
```
Now the user gets prompted at each turn to override or guide the agent.

### Full debug (power user)

```bash
maxim --sim scenarios/campaigns/heist_v1.yaml --display debug --trace hippo
```
Everything from today's output, plus hippocampus operation traces on stderr.

---

## Invariants

- **`--display` and `--interactive` are fully orthogonal.** Any combination works.
- **`--interactive auto` is context-aware.** DM campaigns → on, generative → off, agentic → on.
- **CLEAN tier never shows system internals.** If `[PIPELINE]` or "Saving memory..." appears, that's a bug.
- **BIO annotations are human sentences, not raw traces.** "Captured: merchant's warning" not "id=abc123 sal=0.82".
- **DEBUG tier is identical to today's full output.** No regression for power users.
- **Old flags still work.** Deprecation warning, mapped to new axes.
- **`--trace` stays separate on stderr.** For developer debugging, not user display.
- **Python API matches CLI.** `configure(display="bio")` = `--display bio`.

---

## Agentic Cycle Integration

The display system doesn't just serve simulations — it must integrate with the full agentic loop where the agent is a persistent entity with autonomy levels, processing states, mode transitions, and a Default Network running in the background.

### BUG: Hardcoded print() in the agent loop blocks headless mode

> **Severity: HIGH — fix before or alongside display simplification.**

**Corrected understanding (verified against code):** The confirmation prompt at `agent_loop.py:1583-1595` does NOT call `input()` directly. It prints the prompt via hardcoded `print()`, then sets `state.data["pending_confirmation"]` and returns. The loop then waits for `state.data["pending_cli_input"]` to be populated on the next iteration by an external reader thread.

**The actual bug:** In headless/non-interactive mode:
1. The `print()` calls fire unconditionally — unwanted console noise
2. The loop sets `pending_confirmation` and waits for `pending_cli_input` which never arrives if no reader thread exists — forward progress stalls indefinitely (not a thread hang, but a logical deadlock)
3. None of this goes through the `PromptHandler` system which already has `NonInteractiveHandler` fallback

The agent loop has ~15 direct `print()` calls that bypass all display control:

| Location | What it prints | Fix |
|----------|---------------|-----|
| **agent_loop.py:1583-1595** | **Action confirmation prompt — logical deadlock in headless mode** | **Route through PromptHandler; NonInteractiveHandler auto-resolves via SupervisionPolicy** |
| loop_controller.py:248-268 | Confirmation success/failure messages | Route through `display_action()` |
| loop_controller.py:319-338 | Timeout retry notifications | Route through `display_log(BIO)` |

The fix: replace the hardcoded `print()` + state-wait pattern with a call to the existing `PromptHandler.prompt()` system. `NonInteractiveHandler` already returns defaults immediately — the infrastructure exists, it's just not used here.

### Autonomy level ↔ display interaction

The autonomy system already gates what the agent can do — display should respect this:

| Autonomy | Current behavior | With display system |
|----------|-----------------|-------------------|
| PLANNING | Agent proposes, waits for approval | Proposal shown at CLEAN tier. If `--interactive off`, auto-approve with policy. |
| SUPERVISED | Acts within bounds, escalates edge cases | Edge-case confirmations shown at CLEAN tier. If `--interactive off`, use `SupervisionPolicy` default. |
| AUTONOMOUS | Full agency, safety only | No confirmations. Pain/fear signals shown at BIO tier only. |

Key insight: **autonomy level already controls when the system needs human input.** The `--interactive` flag shouldn't duplicate this — it should control whether `auto` mode respects the autonomy level's natural prompting behavior or overrides it.

### Autonomy transition consent

> **Terminology note (verified against code):** The code in `mode_switch.py` uses "escalation" to mean "increasing restrictions" (autonomous→planning), which is the opposite of common usage. We use "gaining autonomy" and "reducing autonomy" here to avoid confusion.

When the agent requests an autonomy level change (via `AutonomyLevelTool`), the **current code behavior** is:
- **Gaining autonomy** (planning → supervised → autonomous): immediate, no approval
- **Reducing autonomy** (autonomous → supervised → planning): requires approval

This is backwards for user safety. The user should consent to the agent gaining MORE power, not less. The revised policy:

| Transition | Direction | Current | Revised |
|-----------|-----------|---------|---------|
| Planning → Supervised | Gaining autonomy | Immediate | **Prompt user**: "Agent requests supervised autonomy: {reason}. Allow? [y/n]" |
| Planning → Autonomous | Gaining autonomy | Immediate | **Prompt user**: "Agent requests full autonomy: {reason}. Allow? [y/n]" |
| Supervised → Autonomous | Gaining autonomy | Immediate | **Prompt user**: "Agent requests full autonomy: {reason}. Allow? [y/n]" |
| Autonomous → Supervised | Reducing autonomy | Requires approval | **Immediate** (agent is voluntarily reducing power — always safe) |
| Supervised → Planning | Reducing autonomy | Requires approval | **Immediate** (agent is voluntarily reducing power — always safe) |
| Any → same level | No change | No-op | No-op |

The consent prompt:
- Uses the `PromptHandler` system (not raw `print()`/`input()`)
- Respects `--interactive` mode: if `off`, use a configurable policy (`AutoEscalationPolicy`)
- Shows the agent's `reason` so the user can make an informed decision
- At **CLEAN** display tier, shows a one-line prompt: `Agent requests autonomous mode (reason: "novel situation requires rapid action"). Allow? [y/n]`
- At **BIO** tier, also shows the current bio-state context that motivated the request
- Logged in the agentic event buffer regardless of display tier

```python
class AutoEscalationPolicy:
    """Policy for autonomy escalation when --interactive off."""
    
    allow_to_supervised: bool = True    # Low risk — bounded actions
    allow_to_autonomous: bool = False   # High risk — default deny without human
    max_autonomous_duration: float = 300.0  # 5 min cap if allowed
```

This gives users three modes:
1. **`--interactive on`**: Always asked for consent on escalation
2. **`--interactive auto`**: Asked in agentic/DM mode, policy-driven in headless sim mode
3. **`--interactive off`**: `AutoEscalationPolicy` decides (supervised OK, autonomous denied by default)

**Implementation:** Phase D-4 (auto-interactive wiring + autonomy integration). Modify `AutonomyLevelTool.execute()` to check interactive mode before escalation. Add `AutoEscalationPolicy` to `autonomy.py`.

### Processing state ↔ display

| State | Display behavior |
|-------|-----------------|
| AWAKE | Normal display per `--display` tier |
| SLEEP | Suppress all output except wake triggers. Bio annotations silenced. DN background activity hidden. |

Sleep already skips LLM processing in the loop (agent_loop.py:1753), but confirmation prompts would still fire if an action was queued before sleep. The display system should suppress these — if the agent is sleeping, nothing should print.

### Mode definitions should carry display hints

Currently `ModeDefinition` (modes/definitions.py) specifies `allowed_tools`, `max_response_tokens`, etc. but has no display configuration. Extending it:

```python
@dataclass
class ModeDefinition:
    # ... existing fields ...
    default_display: DisplayTier = DisplayTier.CLEAN
    confirmations_required: bool = True  # False for AUTONOMOUS
```

This means mode transitions (via `mode_switch` tool or autonomy level changes) can automatically adjust display behavior. When the agent escalates to AUTONOMOUS, confirmations stop. When it de-escalates to PLANNING, they resume.

### Default Network background output

The DN runs during idle/sleep and produces reactive actions via callbacks (network.py:348-349). It does NOT currently produce console output — all output goes through the bus. This is already correct for the display system. DN activity should appear:

- **CLEAN tier:** Not at all (it's background processing)
- **BIO tier:** Only if DN escalates something to the LLM ("○ default_net: Escalated unfamiliar sound to conscious processing")
- **DEBUG tier:** Full DN update cycle logs

No changes needed to DN code — just add a `display_log(BIO, ...)` call at the escalation point.

---

## Runtime Display Switching

### The case for a `display_mode` tool

The agent should be able to adjust display verbosity at runtime. Consider:

- Agent is running headless in AUTONOMOUS mode doing routine work (CLEAN display)
- Agent encounters something novel or dangerous
- Agent wants to surface detailed bio-system info to the user
- Agent calls `display_mode(level="bio", reason="unusual pattern detected")`

This is analogous to how the agent already calls `mode_switch` to change autonomy or `sleep` to enter low-power mode. Display is part of the agent's **communication with the user**.

### DisplayModeTool

```python
class DisplayModeTool(Tool):
    name = "display_mode"
    description = (
        "Adjust what the user sees. Use 'bio' to surface memory and learning "
        "activity when something important is happening. Use 'clean' to return "
        "to narrative-only output. Use 'debug' only when diagnosing issues."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["clean", "bio", "debug"]},
            "reason": {"type": "string"},
        },
        "required": ["level"],
    }
```

**Safety constraints:**
- Can only escalate display tier (clean → bio → debug), not suppress it below the user's `--display` floor
- Respects the user's minimum: if user set `--display bio`, agent can escalate to debug but can't drop to clean
- Reverts to user's `--display` setting after N turns or on mode switch (prevents permanent debug spam)
- Logged in agentic event buffer for auditability

### InteractiveModeTool

Similarly, the agent could request interactive mode changes:

```python
class InteractiveModeTool(Tool):
    name = "request_interaction"
    description = (
        "Request user input for an important decision that the agent is not "
        "confident about. Only use when the decision has significant consequences."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["question"],
    }
```

This is different from the existing `PromptHandler` system — it's the agent **choosing** to ask the user, not the system requiring confirmation. The prompt only fires if `--interactive` is `auto` or `on`. If `off`, the agent gets a response like "User interaction is disabled — make your best judgment."

### When runtime switching makes sense

| Scenario | Agent action | Display effect |
|----------|-------------|---------------|
| Novel entity encountered | `display_mode("bio")` | User sees memory/learning activity for this encounter |
| Danger detected | `display_mode("bio")` + `request_interaction("Should I proceed?")` | Bio annotations + user prompt |
| Routine travel | `display_mode("clean")` (if above user floor) | Back to narrative only |
| System anomaly | `display_mode("debug")` | Full traces for this section |
| Critical decision | `request_interaction("Which path?", options=["left", "right"])` | User prompted if interactive mode allows |

### Revert behavior

Runtime display escalation is temporary:
- Reverts to user's `--display` setting after **3 turns** or on **encounter/scene change**
- Agent can extend by calling `display_mode` again
- User's CLI flag is the **floor** — agent can only go up, never below it

---

## Updated Phase Summary

| Phase | What | LOC | Key outcome |
|-------|------|-----|-------------|
| D-0 | DisplayTier + InteractiveMode infrastructure | ~100 | Two-axis system with global state |
| D-1 | Clean tier display functions | ~120 | Narrative-only output |
| D-2 | Bio tier condensed annotations | ~100 | Human-readable bio events |
| D-3 | Silence ~40 print() calls (sim + agent loop) | ~100 | Default is quiet everywhere |
| D-4 | Auto-interactive wiring + autonomy integration | ~80 | Smart prompt defaults, sleep suppression |
| D-5 | CLI + API flag wiring + backward compat | ~60 | Two flags replace six |
| D-6 | DisplayModeTool + InteractiveModeTool | ~120 | Agent-driven display/input switching |
| D-7 | Mode definition display hints | ~40 | Per-mode default tiers |
| **Total** | | **~720** | |

---

## Testing Strategy

- **D-0:** Unit test tier filtering + interactive mode resolution for each context
- **D-1:** Capture stdout during mock campaign at CLEAN tier, assert no subsystem labels
- **D-2:** Capture at BIO tier, assert `○ memory:` annotations present, no raw `[HIPPOCAMPUS]`
- **D-3:** Grep assert: zero bare `print()` in simulation/ and runtime/ (all replaced)
- **D-4:** Test auto-interactive: DM campaign → prompts, generative → no prompts, AUTONOMOUS → no confirmations, SLEEP → suppress all
- **D-5:** CLI: `--display bio --interactive off` works; `--verbosity 2` maps to `--display debug`
- **D-6:** Test DisplayModeTool: agent escalates to bio, verify annotations appear; verify can't drop below user floor; verify auto-revert after 3 turns
- **D-7:** Test mode transition carries display hint; AUTONOMOUS mode suppresses confirmations
