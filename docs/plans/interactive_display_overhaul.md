# Interactive Display Overhaul — agent focus, thinking panel, dynamic resize

**Status:** Implementation-ready plan (2026-04-23)
**Scope:** ~450-550 LOC net
**Depends on:** [pfc_deliberation_cycle.md](pfc_deliberation_cycle.md) (SHIPPED in PR #178 — deliberation events flowing)
**Target version:** 0.8 patch

---

## Problem

The current interactive display is a single interleaved log panel shared by all agents. As the system matures — multi-agent sims, PFC deliberation cycles, richer bio-system output — the display has three gaps:

1. **No agent isolation.** AUT, orchestrator, and NPC agents all write to one log. During multi-agent sims, the interleaved output is noisy. Users can't focus on one agent's behavior.

2. **No thinking visibility.** The PFC deliberation cycle produces reasoning text (the LLM's chain of thought) that's richer than a log line. Currently it's compressed into one-line `[THOUGHT]` and `[DELIBERATION]` log entries at the BIO display tier. Users can't see *what* the agent is thinking, only *that* it's thinking.

3. **Fixed layout.** The bio log panel is one size. Users who want deep bio-system traces get cramped; users who only care about narrative waste space on dimmed grey lines they never read. No way to adjust without changing `--display` tier (which filters *content*, not *space*).

## Design

### Panel layout: split the log into two rows

The flexible log panel splits into two sub-panels with adjustable ratio:

```
+------------------------------------------+
| Title (fixed 3-4 lines)                  |
+------------------------------------------+
| Status bar (fixed 3 lines)               |
|  + focused agent indicator + layout tag  |
+------------------------------------------+
| Warnings (optional, fixed)               |
+------------------------------------------+
| Agent Log (ratio, scrollable)            |
|   - bio subsystems dimmed                |
|   - filtered by focused agent            |
+------------------------------------------+
| Thinking (ratio, scrollable)             |
|   - LLM reasoning text                   |
|   - cycle number + elapsed               |
|   - enrichment summary                   |
+------------------------------------------+
| Input (fixed 4+ lines)                   |
+------------------------------------------+
```

The thinking panel shows:
- The LLM's `reasoning` text from the current deliberation cycle (the full text, not a truncated log line)
- Cycle indicator: `Cycle 2/3 . 12.4s`
- Enrichment summary: which bio-systems contributed (hippocampus, NAc, EC) and how many sections
- When no deliberation is active: panel shows last completed deliberation summary or collapses to minimum height (1-2 lines: "No active deliberation")

The split replaces the current single flexible `Layout(log_panel)` in `_build_layout()` ([display.py:446](src/maxim/interactive/display.py#L446)) with:
```python
Layout(name="body"),  # replaces Layout(log_panel)
...
layout["body"].split_column(
    Layout(log_panel, ratio=log_ratio),
    Layout(thinking_panel, ratio=thinking_ratio),
)
```

Default ratio: 3:1 (log gets 75%, thinking gets 25%). Adjustable via shift+up/down.

**Extension interaction.** When extensions are registered (DM panels, etc.), the current code splits into two *columns* (log left ratio 2, extensions right ratio 1) at [display.py:432](src/maxim/interactive/display.py#L432). With the overhaul, the body area splits into two columns first, then the left column splits into two rows (log + thinking). The right column remains extensions. Extensions never lose their space.

### Agent focus switching: shift+left/right

**Structural change to `_log_lines`.** The current `_log_lines: deque[str]` stores Rich markup strings with agent prefixes baked in at [display.py:219](src/maxim/interactive/display.py#L219). There's no way to filter by agent without parsing markup. Change to:

```python
@dataclass(frozen=True, slots=True)
class _LogEntry:
    agent: str | None  # nickname or None (system)
    markup: str        # full Rich markup string (as today)

_log_lines: deque[_LogEntry]
```

This preserves existing rendering (iterate and join `.markup`) but enables O(1) agent filtering.

**Focus cycle:** ALL -> AUT -> ORCH -> NPC1 -> NPC2 -> ... -> ALL

**Implementation:** `MaximDisplay` gains:
- `_focused_agent: str | None` -- `None` means ALL (no filter)
- `_agent_roster: list[str]` -- populated via new `register_agent(nickname)` method, called from `sim_logger.register_agent_nickname()`
- `focus_next()` / `focus_prev()` -- cycle through roster
- `_filter_lines(lines: list[_LogEntry]) -> list[_LogEntry]` -- filter predicate used by `_build_layout`

**Status bar indicator:** When focused, status bar gains a right-aligned tag `[Agent: AUT]` or `[Agent: ALL]`. Rendered inside the existing status panel at [display.py:344](src/maxim/interactive/display.py#L344) by appending to `status_text`.

**Scroll state per agent:** Each agent gets its own `_scroll_offset` so switching agents preserves scroll position. Store as `_scroll_offsets: dict[str | None, int]` keyed by agent name (None = ALL). The current `_scroll_offset` field at [display.py:127](src/maxim/interactive/display.py#L127) becomes a property that reads from this dict.

### Thinking panel scrolling: shift+up/down

Shift+up/down scrolls the thinking panel, mirroring plain up/down for the log panel. Same step size (3 lines per press). Shift+up scrolls older (toward the start of reasoning text), shift+down scrolls newer. The thinking panel gets its own `_thinking_scroll_offset: int` field (0 = bottom). Reset to 0 whenever `set_thinking()` is called with new content (new cycle starts — auto-scroll to latest).

### Dynamic section resize: ctrl+shift+up/down

Ctrl+Shift+up increases the thinking panel ratio (and decreases log). Ctrl+Shift+down does the opposite. Presets rather than continuous:

| Preset | Log ratio | Thinking ratio | Use case |
|--------|-----------|----------------|----------|
| 0 | 5 | 1 | Minimal thinking, max log (narrative focus) |
| 1 | 3 | 1 | Default |
| 2 | 2 | 1 | Balanced |
| 3 | 1 | 1 | Equal split |
| 4 | 1 | 2 | Thinking focus |
| 5 | 1 | 3 | Deep deliberation debugging |

Presets are a list of `(log_ratio, thinking_ratio)` tuples. Ctrl+Shift+up increments the index (more thinking), Ctrl+Shift+down decrements (more log). Clamps at boundaries (no wrap). Current preset shown in status bar: `[Layout: 3:1]`.

### Keyboard: modified-arrow escape sequences

**Key binding summary:**

| Keys | Sequence / Char | Action |
|------|-----------------|--------|
| Up/Down | `\x1b[A`/`\x1b[B` | Scroll log panel (existing) |
| Left/Right | `\x1b[D`/`\x1b[C` | Page up / jump to bottom in log (existing) |
| Shift+Up/Down | `\x1b[1;2A`/`\x1b[1;2B` | Scroll thinking panel |
| Shift+Left/Right | `\x1b[1;2D`/`\x1b[1;2C` | Agent focus prev/next |
| Option+`=` | `≠` (`\u2260`) | Resize: more thinking |
| Option+`-` | `–` (`\u2013`) | Resize: less thinking |

Mental model: **plain = log, shift = thinking/focus, Option+/-  = layout.**

**macOS terminal note:** macOS Terminal.app and iTerm2 strip modifier bits from arrow escape sequences (Ctrl+Shift+Up arrives as plain Up). Resize uses the Unicode characters that Option+key produces instead — these bypass escape sequences entirely. The `_KEYMAP` still contains `[1;3A`/`[1;3B` (Option+arrow escape codes) as fallback for terminals that do pass modifier sequences through (e.g. Linux xterm).

**Critical design issue found during investigation.** The current escape handling at [orchestrator.py:1633](src/maxim/simulation/orchestrator.py#L1633) reads exactly 2 bytes after ESC with `os.read(fd, 2)`. Plain arrow keys send 3 total bytes (`\x1b` + `[A`), so reading 2 works. But modified arrows send 6 total bytes (`\x1b` + `[1;2A` or `\x1b` + `[1;5A`), meaning 5 bytes follow ESC. The current code would read `[1` (2 bytes), match nothing, and leave `;2A` in the stdin buffer — those bytes would then be interpreted as printable characters and appear in the user's input line.

**Fix: variable-length escape accumulator.** After detecting ESC and the 50ms select guard, read bytes one at a time until a terminating character (uppercase letter A-Z or `~`) is seen or a timeout/max-length (8 bytes) is reached. This handles:
- Plain arrows: `[A` (2 bytes, terminates on `A`)
- Shift+arrows: `[1;2A` (4 bytes, terminates on `A`)
- Ctrl+Shift+arrows: `[1;6A` (4 bytes, terminates on `A`)
- Other modified keys: `[5~` (PgUp), etc.

```python
# After ESC detected and select guard passed:
seq_bytes = bytearray()
for _ in range(8):  # max escape sequence length
    _ready, _, _ = select.select([stdin], [], [], 0.01)  # 10ms per byte
    if not _ready:
        break
    b = _os_mod.read(fd, 1)
    if not b:
        break
    seq_bytes.extend(b)
    ch = b[0]
    # Escape sequences terminate on A-Z or ~
    if (0x41 <= ch <= 0x5A) or (0x61 <= ch <= 0x7A) or ch == 0x7E:
        break

seq = seq_bytes.decode("ascii", errors="replace")
```

Then dispatch via keymap dict:

```python
_KEYMAP: dict[str, str] = {
    # Plain arrows — log scroll (existing behavior)
    "[A": "scroll_up",
    "[B": "scroll_down",
    "[C": "scroll_bottom",
    "[D": "scroll_page_up",
    # Shift+arrows — thinking scroll + agent focus
    "[1;2A": "scroll_thinking_up",
    "[1;2B": "scroll_thinking_down",
    "[1;2C": "focus_next",
    "[1;2D": "focus_prev",
    # Ctrl+Shift+arrows — layout resize
    "[1;6A": "resize_thinking_more",
    "[1;6B": "resize_thinking_less",
}
```

Actions map to display methods:
```python
action = _KEYMAP.get(seq)
if action and display is not None:
    method = getattr(display, action, None)
    if method:
        method()
```

**Terminal compatibility.** Shift+arrow (`\x1b[1;2X`) and Ctrl+arrow (`\x1b[1;5X`) escape sequences are standard xterm/VT220 and work in iTerm2, Terminal.app (macOS), GNOME Terminal, and Windows Terminal. Some older terminals may not send them — degrade gracefully (unknown sequence = ignored). The variable-length accumulator also prevents buffer pollution from unrecognized sequences.

### Thinking panel data flow

The reasoning text exists inside `_run_deliberation_cycles` at [agent_loop.py:404](src/maxim/runtime/agent_loop.py#L404) (cycle 1 reasoning) and [agent_loop.py:497](src/maxim/runtime/agent_loop.py#L497) (subsequent cycle reasoning). Currently it's logged as a truncated summary via `sim_log("DELIBERATION", ...)`.

New data path:
1. `_run_deliberation_cycles` and the cycle 1 enrichment block call a new `sim_logger.sim_deliberation_update()` function with: cycle number, max cycles, reasoning text (full), enrichment section names, elapsed time.
2. `sim_deliberation_update()` calls `display.set_thinking(...)` on the active display, passing the full reasoning text + metadata.
3. `MaximDisplay.set_thinking()` stores the data in `_thinking_state: _ThinkingState | None` and calls `_refresh()`.
4. `_build_layout()` renders `_thinking_state` into the thinking panel.
5. On deliberation end (ready_to_act or max cycles), call `sim_deliberation_end()` which sets the thinking panel to a summary view ("Deliberated 2 cycles in 8.3s").

```python
@dataclass
class _ThinkingState:
    reasoning: str         # Full LLM reasoning text
    cycle: int             # Current cycle number
    max_cycles: int        # Hard cap
    enrichment_tags: list[str]  # ["hippocampus", "nac", "ec"]
    started_at: float      # time.monotonic() for elapsed display
    agent: str | None      # Agent nickname
    completed: bool = False  # True = show summary instead of live text
```

### Non-sim mode

The thinking panel only appears during sim/interactive mode where `MaximDisplay` is active. CLI non-sim mode uses plain logging -- PFC deliberation events go to the standard logger (and MAXIM_LOG_FILE JSONL) as they do today.

---

## Integration points (exact locations)

| What | Where | Line(s) | Action |
|------|-------|---------|--------|
| Layout construction (no extensions) | [display.py](src/maxim/interactive/display.py) | 438-449 | Split `Layout(log_panel)` into body with log+thinking rows |
| Layout construction (with extensions) | [display.py](src/maxim/interactive/display.py) | 418-436 | Left column of two-column split gets log+thinking rows |
| Log line storage | [display.py](src/maxim/interactive/display.py) | 120, 219 | Change `deque[str]` to `deque[_LogEntry]`, update append |
| Visible line computation | [display.py](src/maxim/interactive/display.py) | 363-382 | Account for thinking panel height in `visible_lines` calc |
| Scroll offset field | [display.py](src/maxim/interactive/display.py) | 127 | Replace with `_scroll_offsets` dict |
| Scroll method | [display.py](src/maxim/interactive/display.py) | 293-301 | Read/write from `_scroll_offsets[_focused_agent]` |
| `page_height` property | [display.py](src/maxim/interactive/display.py) | 252-255 | Account for thinking panel in height calc |
| Status bar text | [display.py](src/maxim/interactive/display.py) | 344 | Append focus indicator + layout preset |
| Escape sequence read | [orchestrator.py](src/maxim/simulation/orchestrator.py) | 1633-1648 | Replace 2-byte read with variable-length accumulator + keymap |
| Arrow key dispatch | [orchestrator.py](src/maxim/simulation/orchestrator.py) | 1637-1648 | Replace if/elif chain with `_KEYMAP` dict lookup |
| Deliberation cycle 1 enrichment log | [agent_loop.py](src/maxim/runtime/agent_loop.py) | 918-922 | Add `sim_deliberation_update()` call with reasoning text |
| Deliberation cycles 2+ log | [agent_loop.py](src/maxim/runtime/agent_loop.py) | 438-447, 491, 508, 523 | Add `sim_deliberation_update()` / `sim_deliberation_end()` calls |
| Agent nickname registration | [sim_logger.py](src/maxim/simulation/sim_logger.py) | 89-92 | Also call `display.register_agent(nickname)` |
| Display tier for THOUGHT/DELIBERATION | [sim_logger.py](src/maxim/simulation/sim_logger.py) | 355-356 | Unchanged -- log lines still BIO tier; thinking panel is separate channel |

---

## Staging

### Stage 1: Thinking panel (ships alone)

Add the thinking panel split, `_ThinkingState`, `set_thinking()`, `_thinking_scroll_offset`, `scroll_thinking_up()`/`scroll_thinking_down()`, `sim_deliberation_update()` / `sim_deliberation_end()`, and the data path from `_run_deliberation_cycles` through sim_logger to the display. Also refactor `_log_lines` to `deque[_LogEntry]` (needed for Stage 2 but cleaner to do now since we're touching `_build_layout` anyway). Also refactor the escape sequence reader to the variable-length accumulator and add `_KEYMAP` with shift+up/down for thinking scroll (plain arrows migrate to the keymap too).

**Does NOT include:** agent focus or resize presets.

**Files touched:**
| File | Change | LOC est |
|------|--------|---------|
| `interactive/display.py` | `_LogEntry` dataclass, `_ThinkingState` dataclass, `set_thinking()` method, `_thinking_scroll_offset`, `scroll_thinking_up()`/`scroll_thinking_down()`, `_build_layout` body split, visible line calc adjustment | +140, -15 |
| `simulation/orchestrator.py` | Variable-length escape accumulator, `_KEYMAP` dict (plain arrows + shift+up/down), replace if/elif chain | +35, -15 |
| `simulation/sim_logger.py` | `sim_deliberation_update()`, `sim_deliberation_end()` functions | +40 |
| `runtime/agent_loop.py` | Call `sim_deliberation_update/end` at cycle boundaries | +15 |
| **Stage 1 net** | | **~+200** |

**Validation:** Run a sim with `--display bio` and confirm the thinking panel shows reasoning text during deliberation and "No active deliberation" otherwise. Verify shift+up/down scrolls the thinking panel. Verify plain arrows still scroll the log (regression). Verify the log panel height adapts correctly (doesn't overflow terminal).

### Stage 2: Agent focus switching

Add `_focused_agent`, `_agent_roster`, `register_agent()`, `focus_next()`/`focus_prev()`, `_filter_lines()`, per-agent `_scroll_offsets`. Refactor the escape sequence reader to variable-length accumulator and add shift+left/right handling via `_KEYMAP`.

**Depends on Stage 1** (needs `_LogEntry.agent` field for filtering + `_KEYMAP` infrastructure).

**Files touched:**
| File | Change | LOC est |
|------|--------|---------|
| `interactive/display.py` | `_focused_agent`, `_agent_roster`, `register_agent()`, `focus_next()`/`focus_prev()`, `_filter_lines()`, `_scroll_offsets` dict, status bar focus indicator | +80, -10 |
| `simulation/orchestrator.py` | Add shift+left/right entries to `_KEYMAP` | +2 |
| `simulation/sim_logger.py` | `register_agent_nickname` also calls `display.register_agent()` | +5 |
| **Stage 2 net** | | **~+77** |

**Validation:** Run a multi-agent sim. Verify shift+right cycles ALL -> AUT -> ORCH -> NPC1 -> ALL. Verify log filters correctly. Verify scroll position is preserved per agent. Test on iTerm2 and Terminal.app.

### Stage 3: Dynamic resize

Add resize presets, shift+up/down handling, layout preset indicator in status bar.

**Depends on Stage 1** (needs `_KEYMAP` infrastructure from Stage 1).

**Files touched:**
| File | Change | LOC est |
|------|--------|---------|
| `interactive/display.py` | `_RESIZE_PRESETS`, `_resize_index`, `resize_thinking_more()`/`resize_thinking_less()`, preset in `_build_layout` ratios, preset in status bar | +40 |
| `simulation/orchestrator.py` | Add ctrl+shift+up/down entries to `_KEYMAP` | +2 |
| **Stage 3 net** | | **~+42** |

**Validation:** Verify Ctrl+Shift+up/down cycles through presets. Verify preset indicator updates in status bar. Verify panels resize visually.

---

## Known risks (validated against code)

### 1. Escape sequence buffer pollution (CRITICAL, addressed in design)

The shell plan's `_KEYMAP` was correct in concept but the underlying read mechanism was wrong. The current `os.read(fd, 2)` at [orchestrator.py:1633](src/maxim/simulation/orchestrator.py#L1633) cannot handle 6-byte shift+arrow sequences. Unread bytes pollute the input buffer. The variable-length accumulator in the design section solves this.

**Validation needed:** Manual test of shift+arrow on macOS Terminal.app and iTerm2. Verify no stray characters appear in input buffer. Also test that plain arrows still work (regression).

### 2. Thinking panel during one-shot turns

Most turns don't trigger multi-cycle deliberation. The thinking panel needs graceful empty state -- not a blank box but a minimal view. Design: when `_thinking_state is None` or `_thinking_state.completed`, show a dim 1-line summary ("No active deliberation" or "Deliberated 2 cycles in 8.3s"). The panel still occupies its ratio share -- collapsing it dynamically would cause jarring layout shifts at 4 FPS refresh.

### 3. Terminal height pressure

Current `visible_lines` calculation at [display.py:370](src/maxim/interactive/display.py#L370) already subtracts fixed panel heights from terminal height. Adding the thinking panel (ratio-based, not fixed) means the log panel shrinks. On a 24-line terminal at 3:1 ratio with all fixed panels, the log gets ~7 lines and thinking gets ~2 lines. This is tight but functional. At 1:3 ratio (deep thinking), the log gets ~2 lines -- usable for focused debugging but not general use. The preset system lets users choose.

**Mitigation:** Document that terminals under 30 lines tall should use preset 0 (5:1) or disable the thinking panel entirely (future `--no-thinking-panel` flag if needed).

### 4. Agent roster ordering

NPCs are created dynamically during sim. The roster needs to update live as agents register via `sim_logger.register_agent_nickname()`. Focus cycling must be stable -- don't reorder the roster when new agents join, just append. `register_agent()` appends to `_agent_roster` only if the nickname isn't already present.

### 5. Log filtering changes visible line count

When switching from ALL to AUT, the visible log shrinks (filtered lines removed). The per-agent scroll offset for AUT might point past the end of its filtered log. `_build_layout` must clamp `_scroll_offsets[_focused_agent]` against the filtered line count at render time.

### 6. Extension panel interaction

The current extension layout ([display.py:432-434](src/maxim/interactive/display.py#L432)) splits the body into two columns. The thinking panel split must nest inside the left column, not replace the two-column split. The code path is: body -> split_row(left_col ratio 2, ext_col ratio 1), then left_col -> split_column(log_panel ratio X, thinking_panel ratio Y). Verified that Rich `Layout` supports this nesting.

### 7. Thread safety of thinking state

`set_thinking()` is called from the agent loop thread (via sim_logger). `_build_layout()` reads `_thinking_state` from the main thread's refresh. Both paths already acquire `_lock` (RLock), so this is safe. No new threading concern.

### 8. Refresh rate

Adding a second scrollable panel increases `_build_layout()` work per frame. At 4 FPS ([display.py:149](src/maxim/interactive/display.py#L149)), each frame has 250ms budget. The thinking panel is a simple `Panel(Text(...))` render -- negligible cost. Agent filtering adds one list comprehension per frame. No refresh rate concern.

---

## Test strategy

### Unit tests (no terminal needed)

1. **`_LogEntry` filtering.** Create a `MaximDisplay`, call `log()` with various agent values, verify `_filter_lines()` returns correct subsets.

2. **Focus cycling.** Register 3 agents, call `focus_next()` 4 times, verify the cycle: None -> "AUT" -> "ORCH" -> "NPC1" -> None.

3. **Per-agent scroll offsets.** Set focus to "AUT", scroll up, switch to "ORCH", verify offset is 0. Switch back to "AUT", verify offset is preserved.

4. **Resize presets.** Call `resize_thinking_more()` repeatedly, verify index clamps at max. Call `resize_thinking_less()`, verify it decrements. Verify `_build_layout()` uses correct ratios (mock console height).

5. **Thinking state lifecycle.** Call `set_thinking(...)`, verify `_thinking_state` is set. Call `set_thinking(completed=True)`, verify completed flag. Call `set_thinking(None)`, verify cleared.

6. **Thinking scroll.** Call `set_thinking(...)` with long reasoning text, call `scroll_thinking_up()`, verify `_thinking_scroll_offset` increments. Call `set_thinking(...)` with new content, verify offset resets to 0.

7. **Escape sequence accumulator.** Unit-test the accumulator function in isolation (extracted as a helper): feed byte sequences for plain arrow (`[A`), shift+arrow (`[1;2A`), ctrl+arrow (`[1;5A`), unknown sequence, lone ESC. Verify correct sequence string output and that no bytes are left unconsumed.

### Integration tests (terminal-dependent, manual)

8. **Modified arrows on macOS.** Run `maxim --sim "test" --interactive --sim-max-turns 3 --display bio`. Press shift+up/down (thinking scroll), shift+right/left (agent focus), Ctrl+Shift+up/down (resize). Verify each action works, no stray characters in input.

9. **Multi-agent deliberation visibility.** Run a multi-agent sim. Verify thinking panel shows AUT's reasoning during deliberation, then shows "No active deliberation" when idle.

10. **Log files for automated verification.** Use `MAXIM_LOG_FILE=/tmp/maxim.jsonl` to capture structured events. Verify `sim_deliberation_update` records contain full reasoning text, cycle number, and enrichment tags. This can be automated without a terminal.

---

## What NOT to do

- **Don't auto-collapse the thinking panel.** Dynamic height changes cause layout jitter at 4 FPS. Use the ratio system instead -- users who don't want the panel use preset 0 (5:1).
- **Don't filter the thinking panel by display tier.** The thinking panel is a separate channel from the log. It always shows reasoning when a deliberation is active, regardless of CLEAN/BIO/DEBUG tier setting. Log-line THOUGHT/DELIBERATION entries still respect the tier.
- **Don't store reasoning text in `_log_lines`.** The thinking panel has its own state (`_ThinkingState`). Log lines are one-line summaries; thinking state is the full multi-paragraph reasoning. Conflating them breaks both use cases.
- **Don't wrap at preset boundaries.** Ctrl+Shift+up at max preset (1:3) should no-op, not wrap to 5:1. Users expect "more thinking" to stop at "max thinking", not jump to "no thinking". Same for Ctrl+Shift+down.

## Relationship to other plans

- **Depends on** [pfc_deliberation_cycle.md](pfc_deliberation_cycle.md) -- SHIPPED (PR #178). Deliberation reasoning text is flowing.
- **Composes with** [goal_depth_integration.md](goal_depth_integration.md) -- when GOAL WMS entries exist, the thinking panel could show active goal context alongside reasoning.
- **Replaces** the `DeliberationExtension` approach originally proposed in the PFC plan -- that used a column side-panel which is too narrow for reasoning text.
