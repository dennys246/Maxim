# Bug: Raw print() calls corrupt MaximDisplay panel

**Status:** Fixed (0.3.2, 2026-04-18)
**Severity:** Medium — display activates but visual output is corrupted
**Affects:** `maxim --sim "..." --interactive` (generative + DM campaigns)
**Does NOT affect:** `--sim interactive` (REPL mode), headless, non-interactive, tests
**Resolution:** Replaced `input()` with raw terminal reader (`termios`/`tty` + `os.read`). Keystrokes render inside the Live panel via `display.set_prompt()`. Spinner, stall warnings, responses, and Python logging all route through the display. See `project_032_interactive_mode.md` memory for full details.

## Symptoms

When `MaximDisplay` is active (`--interactive` flag), the rich `Live` panel renders but is repeatedly corrupted by raw `print()` calls that bypass the `sim_logger` routing:

1. **Orchestrator spinner** (`[K ⠋ Orchestrator planning next probe...`) — printed directly to stdout with ANSI escape codes, overlaps the panel borders
2. **Agent responses** (`[Maxim]: ...` + `maxim>` prompt) — the `respond` tool outputs directly via `print()`, not through `sim_log()` or `display_scene()`
3. **Turn summaries** (`Turn 1 complete: 6 action(s) [...]`) — printed directly

The panel redraws correctly between corruptions (the `Live.update()` cycle works), but each raw `print()` injects a line that conflicts with the rich layout, causing the visual "flickering panel" effect.

## Root cause

`sim_logger._emit()` correctly routes through `MaximDisplay.log()` when the display is active. But several output paths in the runtime bypass `sim_log()` / `_emit()` entirely:

### Stray print() paths to trace

| Source | What it prints | File | Probable location |
|---|---|---|---|
| Orchestrator spinner | `⠋ Orchestrator planning next probe... (Ns)` | `simulation/orchestrator.py` | Spinner uses direct `print()` with `\r` + `[K` ANSI clear |
| Agent respond tool | `[Maxim]: ...` + `maxim>` prompt | `tools/response.py` or `runtime/loop_controller.py` | Response output path |
| Turn summary | `Turn N complete: ...` | `simulation/orchestrator.py` | Post-turn summary |
| Shutdown messages | `Shutting down agent loops...` etc. | `simulation/orchestrator.py` or `cli.py` | Cleanup path |
| Stall warning | `⚠ Stall detected (#1, ...)` | `simulation/orchestrator.py` | Stall detector |

## Fix approach

For each stray `print()` path, the fix is one of:

1. **Route through `display.set_status()`** — for status/progress info (spinner, turn count). The display status bar is designed for exactly this.
2. **Route through `_emit()` / `display_scene()`** — for content the user should see (agent responses, turn summaries). These already route through the display when active.
3. **Suppress when display is active** — for ANSI control sequences (`\r`, `[K`) that are only meaningful in raw terminal mode. Check `get_active_display() is not None` before emitting.

### Investigation steps

1. `grep -rn "print(" src/maxim/simulation/orchestrator.py` — find all direct prints in orchestrator
2. `grep -rn "print(" src/maxim/tools/response.py` — find response output path
3. `grep -rn "print(" src/maxim/runtime/loop_controller.py` — find loop output paths
4. `grep -rn "\\\\r\|\\[K" src/maxim/` — find ANSI cursor-control sequences (spinner patterns)
5. For each hit: is it user-facing content or control output? Route accordingly.

### Design constraint

The fix must be **backward compatible** — when `get_active_display()` is `None` (no `--interactive`, no rich, no TTY), all paths must produce identical output to today. The display routing is an **addition**, not a replacement.

### Estimated scope

~50-100 LOC across 2-3 files. The per-path fix is 2-5 lines (check for active display, route through it or suppress). The investigation is the bulk of the work — tracing which `print()` calls are user-facing vs control sequences.

## Related

- [interactive_experience_031.md](../plans/interactive_experience_031.md) Stage 5 — wired the core display routing
- [prompt_b3_b5_track.md](../plans/prompt_b3_b5_track.md) — DisplayExtension implementations build on working display
- Plan risk #5 identified this: "stray `print()` calls from tool outputs or third-party code could still interfere"

## Target

0.3.2 patch — fix after 0.3.1 ships with the core interactive infrastructure.
