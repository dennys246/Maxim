# Interactive Display Overhaul — agent focus, thinking panel, dynamic resize

**Status:** Shell plan (2026-04-22)
**Scope:** ~400-500 LOC
**Depends on:** [pfc_deliberation_cycle.md](pfc_deliberation_cycle.md) (deliberation events must be flowing first)
**Target version:** 0.8 patch (ships after PFC cycle)

---

## Problem

The current interactive display is a single interleaved log panel shared by all agents. As the system matures — multi-agent sims, PFC deliberation cycles, richer bio-system output — the display has three gaps:

1. **No agent isolation.** AUT, orchestrator, and NPC agents all write to one log. During multi-agent sims, the interleaved output is noisy. Users can't focus on one agent's behavior.

2. **No thinking visibility.** The PFC deliberation cycle produces reasoning text (the LLM's chain of thought) that's richer than a log line. Currently it's either invisible or compressed into a one-line `[THOUGHT]` log entry. Users can't see *what* the agent is thinking, only *that* it's thinking.

3. **Fixed layout.** The bio log panel is one size. Users who want deep bio-system traces get cramped; users who only care about narrative waste space on dimmed grey lines they never read. No way to adjust without changing `--display` tier (which filters *content*, not *space*).

## Design

### Panel layout: split the log into two rows

The flexible log panel splits into two sub-panels with adjustable ratio:

```
┌─────────────────────────────────┐
│ Title (fixed 3-4 lines)        │
├─────────────────────────────────┤
│ Status bar (fixed 3 lines)     │
│  + focused agent indicator     │
├─────────────────────────────────┤
│ Warnings (optional, fixed)     │
├─────────────────────────────────┤
│ Agent Log (ratio, scrollable)  │
│   — bio subsystems dimmed      │
│   — filtered by focused agent  │
├─────────────────────────────────┤
│ Thinking (ratio, scrollable)   │
│   — LLM reasoning text         │
│   — cycle number + elapsed     │
│   — enrichment summary         │
├─────────────────────────────────┤
│ Input (fixed 4+ lines)         │
└─────────────────────────────────┘
```

The thinking panel shows:
- The LLM's `reasoning` text from the current deliberation cycle (not a log line — the actual text)
- Cycle indicator: `Cycle 2/3 · 12.4s`
- Enrichment summary: which bio-systems contributed (hippocampus, NAc, EC) and how many sections
- When no deliberation is active: panel shows last completed deliberation summary or collapses to minimum height (1-2 lines: "No active deliberation")

The split replaces the current single flexible `Layout(log_panel)` with:
```python
Layout(log_panel, ratio=log_ratio),
Layout(thinking_panel, ratio=thinking_ratio),
```

Default ratio: 3:1 (log gets 75%, thinking gets 25%). Adjustable via shift+up/down.

### Agent focus switching: shift+left/right

Every log line is already tagged with an agent nickname via `sim_logger.register_agent_nickname()`. Agent focus filters both the log panel and the thinking panel to show only one agent's output.

**Focus cycle:** ALL → AUT → ORCH → NPC1 → NPC2 → ... → ALL

**Implementation:** `MaximDisplay` gains:
- `_focused_agent: str | None` — `None` means ALL (no filter)
- `_agent_roster: list[str]` — populated from `register_agent_nickname` calls
- `focus_next()` / `focus_prev()` — cycle through roster
- `_should_show(agent: str | None) -> bool` — filter predicate used by `log()` and thinking panel

**Status bar indicator:** When focused, status bar shows `[Agent: AUT]` or `[Agent: ALL]` in the border title or as a right-aligned tag. Uses the existing status panel — no new panel needed.

**Scroll state per agent:** Each agent gets its own `_scroll_offset` so switching agents preserves scroll position. Store as `_scroll_offsets: dict[str | None, int]` keyed by agent name (None = ALL).

### Dynamic section resize: shift+up/down

Shift+up increases the thinking panel ratio (and decreases log). Shift+down does the opposite. Presets rather than continuous:

| Preset | Log ratio | Thinking ratio | Use case |
|--------|-----------|----------------|----------|
| 0 | 5 | 1 | Minimal thinking, max log (narrative focus) |
| 1 | 3 | 1 | Default |
| 2 | 2 | 1 | Balanced |
| 3 | 1 | 1 | Equal split |
| 4 | 1 | 2 | Thinking focus |
| 5 | 1 | 3 | Deep deliberation debugging |

Presets are a list of `(log_ratio, thinking_ratio)` tuples. Shift+up increments the index (more thinking), shift+down decrements (more log). Wraps at boundaries. Current preset shown in status bar: `[Layout: 3:1]`.

### Keyboard: shift+arrow escape sequences

Current arrow handling in orchestrator's stdin loop detects `\x1b[A/B/C/D`. Shift+arrow produces `\x1b[1;2A/B/C/D`. Refactor the hardcoded switch into a keymap dict:

```python
_KEYMAP = {
    # Existing
    "\x1b[A": "scroll_up",
    "\x1b[B": "scroll_down",
    "\x1b[C": "scroll_bottom",
    "\x1b[D": "scroll_page_up",
    # New
    "\x1b[1;2A": "resize_thinking_more",    # shift+up
    "\x1b[1;2B": "resize_thinking_less",    # shift+down
    "\x1b[1;2C": "focus_next_agent",        # shift+right
    "\x1b[1;2D": "focus_prev_agent",        # shift+left
}
```

The keymap is declarative — actions map to display methods. The stdin loop becomes:
```python
action = _KEYMAP.get(sequence)
if action:
    getattr(display, action)()
```

**Terminal compatibility note:** Shift+arrow escape sequences (`\x1b[1;2X`) are standard xterm/VT220 and work in iTerm2, Terminal.app, GNOME Terminal, and Windows Terminal. Some older terminals may not send them — degrade gracefully (unknown sequence = ignored).

### Non-sim mode

The thinking panel only appears during sim/interactive mode where `MaximDisplay` is active. CLI non-sim mode uses plain logging — PFC deliberation events go to the standard logger (and MAXIM_LOG_FILE JSONL) as they do today.

## What changes

| File | Change | LOC |
|------|--------|-----|
| `interactive/display.py` | Split log panel into log + thinking rows. Add `_focused_agent`, `_agent_roster`, `focus_next/prev`, `_should_show` filter, per-agent scroll offsets, resize presets. Thinking panel rendering (cycle indicator, reasoning text, enrichment summary). | +200 |
| `simulation/orchestrator.py` | Refactor stdin arrow handling into `_KEYMAP` dict. Add shift+arrow sequence detection. Route new actions to display methods. | +40, -20 |
| `simulation/sim_logger.py` | Route deliberation reasoning text to `display.set_thinking()` (new method) in addition to log lines. Pass full reasoning text, not just summary. | +20 |
| `runtime/agent_loop.py` | Pass reasoning text from deliberation cycle to sim_logger for thinking panel (minor — the text already exists, just needs routing). | +10 |
| **Net** | | **~+270** |

## Known friction points

1. **Thinking panel during one-shot turns.** Most turns don't trigger deliberation. The panel needs graceful empty state — not a blank box, but a minimal collapsed view ("idle" or last deliberation summary). Consider: auto-collapse to 1 line when idle and expand when deliberation starts, overriding the user's ratio preset temporarily.

2. **Agent roster ordering.** NPCs are created dynamically during sim. The roster needs to update live as agents register. Focus cycling should be stable — don't reorder the roster when new agents join, just append.

3. **Scroll interaction with agent filter.** When switching from ALL to AUT, the visible log shrinks (filtered). The scroll offset for AUT might point past the end of its filtered log. Clamp on focus switch.

4. **Rich Live refresh rate.** Adding a second scrollable panel doubles the rendering work per refresh. Current refresh is ~4Hz. Monitor for frame drops on older terminals. The thinking panel updates infrequently (once per deliberation cycle, ~10-15s), so it shouldn't add meaningful load.

5. **Escape sequence buffering.** Raw `os.read(fd, 1)` reads one byte at a time. Shift+arrow is 6 bytes (`\x1b[1;2A`). The current code already handles multi-byte sequences (3-byte arrow detection). Extending to 6 bytes needs the same buffering logic but longer — verify the read loop accumulates correctly without blocking on partial sequences.

## Validation

1. Run a multi-agent sim and verify shift+right/left cycles through agents correctly:
   ```bash
   PYTHONPATH=src python -m maxim --sim "two agents explore a dungeon" --interactive --sim-max-turns 6 --display bio
   ```

2. Verify shift+up/down resizes panels and the ratio indicator updates in the status bar.

3. Verify the thinking panel shows reasoning text during deliberation and collapses gracefully when idle.

4. Verify scroll position is preserved per-agent when switching focus.

5. Test on both iTerm2 and Terminal.app to confirm shift+arrow escape sequences work.

## Relationship to other plans

- **Depends on** [pfc_deliberation_cycle.md](pfc_deliberation_cycle.md) — deliberation reasoning text must be flowing before the thinking panel has content to show.
- **Composes with** [goal_depth_integration.md](goal_depth_integration.md) — when GOAL WMS entries exist, the thinking panel could show active goal context alongside reasoning.
- **Replaces** the `DeliberationExtension` approach originally proposed in the PFC plan — that used a column side-panel which is too narrow for reasoning text.
