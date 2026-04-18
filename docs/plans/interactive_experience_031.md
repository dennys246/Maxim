# Interactive experience fixes — 0.3.1

**Status:** Draft (reviewed 2026-04-18, 3-lens + simplification + risk reviews folded, MaximDisplay wiring reinstated, introspection tools folded in as Stage 8)
**Scope:** ~700-800 LOC across 8 stages (fixes + wiring + cleanup + introspection tools).
**Target version:** 0.3.1 (patch release).
**Gates:** Nothing. Quality-of-life fixes for interactive/simulation user experience.
**Depends on:** Nothing — all changes are to existing shipped infrastructure.
**Blocks:** prompt_b3_b5_track.md (Acting Coach) benefits from a working interactive layer first.

## Goal

Fix the interactive infrastructure so the agent can reliably distinguish real user input from auto-defaults, the narrator degrades gracefully, and dead abstractions stop accumulating. These are plumbing fixes — the Acting Coach (B3) is the fancy faucet that goes on top.

## Motivation

The 0.3 release proved cross-session learning works. But the interactive experience — the part where a human actually sits at the terminal — has bugs that silently break the contract between agent and user, and ~400 LOC of speculative abstractions that have zero production callers.

**Bugs:**

1. **`RequestInteractionTool` lies to the agent.** When interactive mode is OFF, the tool returns `success=True` with `"Interaction disabled — make your best judgment."` The LLM sees this output string and has no way to distinguish "user said yes" from "I auto-defaulted." Three distinct failure modes produce identical output: interactive disabled, handler missing, handler failed. ([tools/display.py:136-140](../../src/maxim/tools/display.py#L136))

2. **Narrator fallback text breaks immersion.** When LLM generation fails, the narrator returns `"The journey continues. [Phase: {phase}]"` — visible brackets and phase labels bleed through to the user. ([narrator.py:235](../../src/maxim/simulation/narrator.py#L235), [narrator.py:291](../../src/maxim/simulation/narrator.py#L291))

3. **No handler selection logging.** The user doesn't know if rich or plain prompts are active. Handler creation failures are logged at DEBUG. ([cli.py:1114-1119](../../src/maxim/cli.py#L1114))

4. **`create_handler()` silently falls back on unknown modes.** Passing `mode="foo"` silently gives you PlainPromptHandler with no warning. ([prompts.py:395](../../src/maxim/interactive/prompts.py#L395))

5. **Story context uses char count as token proxy.** 800 chars ≈ 200 tokens is a rough approximation that breaks with unicode, code blocks, or special characters. ([narrator.py:329-334](../../src/maxim/simulation/narrator.py#L329))

**Dead complexity** (identified by simplification review):

6. **`MaximDisplay` was never wired to `sim_logger`.** The foundational buildout plan ([foundational_buildout_plan.md:998-1002](../../docs/plans/archive/foundational_buildout_plan.md#L998)) explicitly designed the integration: "When `MaximDisplay` is active, `sim_log()` routes to `display.log()` instead." The display class is fully built — `start()`, `stop()`, `log()`, `set_status()`, `set_prompt()`, rich `Live` panel layout, extension protocol. The `create_display()` factory exists. What's missing is the routing in `sim_log()` to check for an active display and use it. With 51+ `sim_log` call sites across the runtime, wiring this one integration point gives the entire system a rich panel UI for free when `--interactive` is on.

7. **Minor dead code.** `PromptType.SHORT_TEXT`, `LONG_TEXT`, `NUMERIC`, `RATING` — zero call sites construct these. `poll_freeform()` — zero callers. `freeze_context()` — test-only. `revert_after_turns` parameter on `agent_escalate_display()` — never implemented by any caller. These are small and should be cleaned up, but are not the priority.

## Pre-merge review findings (folded 2026-04-18)

Three parallel reviews (Executor, Architecture, UX) plus a Simplification review produced these cross-confirmed findings:

### Critical: `ToolOutput.metadata` never reaches the LLM (all 3 lenses)

The original Stage 1 design added `was_prompted: bool` to `ToolOutput.metadata`. But:
- `ToolResult` in [agents/bus.py:828-850](../../src/maxim/agents/bus.py#L828) has **no metadata field** — it carries `tool_call_id, tool_name, success, result, error, params`.
- The LLM only sees `ToolResult.result` (a string).
- Adding metadata to `ToolOutput` is invisible to the agent's reasoning.

**Resolution:** Encode the signal into `ToolOutput.output` — the string the LLM actually reads. Three distinct strings for three distinct states (see Stage 1 below). This is simpler, requires no plumbing changes, and the LLM gets the information it needs.

### Medium: Token-counting API is wrong (Architecture lens)

The plan said `hasattr(self._llm, 'count_tokens')`. The actual API is `LLMRouter.get_token_counter()` at [router.py:220-227](../../src/maxim/models/language/router.py#L220), which returns a `TokenCounter` object. Fixed in Stage 4.

### Medium: Phase names are free-form strings (Architecture lens)

`NarrativePhase.name` at [arcs.py:32-44](../../src/maxim/simulation/arcs.py#L32) is a plain `str`, not an enum. Builtin arcs use names like `"seed"`, `"conflict"`, `"reflection"`. YAML-loaded arcs can use anything. The fallback lookup table must have a robust default for unknown phase names. Fixed in Stage 2.

### Major: Stage 5 gold-plates dead code (Simplification lens)

The original Stage 5 added NUMERIC validation and LONG_TEXT multiline support for `PromptType` variants that have **zero production callers**. Nobody constructs `PromptRequest(prompt_type=PromptType.NUMERIC)` anywhere in `src/`. Adding validation loops and multiline input for unused types is the wrong direction — these types should be removed, not enhanced. Stage 5 is replaced with a dead-code cleanup stage.

### Low: `ToolOutput.metadata` is `dict[str, Any]`, never `None` (Executor lens)

The field at [tools/base.py:43](../../src/maxim/tools/base.py#L43) is `field(default_factory=dict)`, not `dict | None`. The original plan's risk section was wrong about the type. Moot now since we're using output strings instead of metadata.

## Stages

### Stage 1 — `RequestInteractionTool` honest reporting

**The critical fix.** The LLM must know whether interaction actually occurred, through the output string it reads.

**What's built:**

Change the `ToolOutput.output` string to carry the signal directly — three distinct strings for three states:

1. **Interactive disabled** (line 137-140): `"Interaction disabled — make your best judgment. You are operating autonomously; no user was consulted."` (was: `"Interaction disabled — make your best judgment."`)
2. **No handler / handler failed** (line 161-172): `"Question displayed to user but response could not be collected. Proceed with your best judgment."` (was: `"Question displayed to user. Response will come on next turn."` — a lie, since no response mechanism exists on this path)
3. **Real response** (line 154-157): `"User responded: {response.value}"` (unchanged)

Also populate `ToolOutput.metadata` for programmatic consumers (the agent loop's `on_step` callback, environment integrations, future `ToolResult` metadata wiring):
- `metadata={"was_prompted": False, "reason": "interactive_mode_off"}` for state 1
- `metadata={"was_prompted": False, "reason": "no_handler"}` for state 2
- `metadata={"was_prompted": True}` for state 3

**Why both output string AND metadata:** The output string is the LLM-facing contract (what the agent reads now). The metadata is the programmatic contract (what the agent loop / future `ToolResult` extension can consume). The output string is the load-bearing fix; the metadata is forward-compatible prep that costs 1 line per branch.

**Why not `success=False`:** the agent shouldn't treat "interaction disabled" as a tool failure that triggers retries or error handling. It's a valid state — the agent just needs to know it's in autonomous mode.

**Pass gate:**
- `test_request_interaction_disabled_output_says_autonomous`: interactive OFF + non-critical → output contains "autonomously"
- `test_request_interaction_critical_overrides_off`: interactive OFF + `critical=True` → `should_prompt` returns True, real prompt fires
- `test_request_interaction_prompted_output_says_responded`: interactive ON + handler → output starts with "User responded:"
- `test_request_interaction_no_handler_output_says_not_collected`: handler is None → output contains "could not be collected"

**Scope:** ~20 LOC in `tools/display.py`, ~40 LOC tests.

### Stage 2 — Narrator fallback immersion

**What's built:**
- Replace bracket-tagged fallback text in `Narrator.generate()` (line 235) and `Narrator.generate_single()` (line 291) with immersive alternatives.
- Small lookup dict mapping known phase name substrings to fallback text. Phase names are free-form strings (not an enum), so the lookup matches on substrings (`"conflict"` matches `"escalation_conflict"`) with a robust default for unrecognized phases:

```python
_FALLBACK_SCENES = {
    "intro": "A new scene begins to unfold before you...",
    "establish": "The world around you takes shape, details sharpening...",
    "conflict": "Tension hangs thick in the air...",
    "escalat": "The stakes rise around you...",
    "reflect": "A quiet moment settles, inviting contemplation...",
    "recall": "Something stirs in the back of your mind...",
}
_FALLBACK_DEFAULT = "The story continues to unfold around you..."
```

- Log the generation failure at WARNING (currently DEBUG at lines 230 and 286) so the operator knows narration fell back — the user sees immersive text, the operator sees the diagnostic.

**Pass gate:**
- `test_narrator_fallback_no_brackets`: fallback text contains no `[` or `]` characters
- `test_narrator_fallback_phase_aware`: known phases produce distinct fallback text; unknown phases get `_FALLBACK_DEFAULT`
- `test_narrator_generate_failure_logs_warning`: LLM exception → WARNING log emitted

**Scope:** ~40 LOC in `narrator.py`, ~30 LOC tests.

### Stage 3 — Handler selection logging + unknown-mode warning

**What's built:**
- `create_handler()` logs at INFO which handler was selected: `"PromptHandler: RichPromptHandler (mode=auto, tty=True)"`.
- Unknown mode string raises `ValueError` instead of silently falling back. The function's docstring already documents the valid modes — silent fallback hides misconfiguration.
- CLI handler creation in `cli.py` logs at INFO on success, WARNING on failure (upgrade from DEBUG).

**Safety check (Executor lens):** All production call sites pass hardcoded `"auto"` — [cli.py:1116](../../src/maxim/cli.py#L1116) and [orchestrator.py:341](../../src/maxim/simulation/orchestrator.py#L341). No user-controlled strings reach `create_handler()`. The `ValueError` is safe.

**Pass gate:**
- `test_create_handler_unknown_mode_raises`: `create_handler("foo")` raises `ValueError`
- `test_create_handler_logs_selection`: INFO log emitted with handler class name and mode
- Existing `test_create_handler_*` tests updated for `ValueError` on unknown modes

**Scope:** ~20 LOC in `prompts.py`, ~10 LOC in `cli.py`, ~20 LOC tests.

### Stage 4 — Story context token-aware truncation

**What's built:**
- Replace char-count proxy in `Narrator._update_story_context()` with a word-count heuristic: `len(text.split())` as a token proxy. Word count tracks actual token count within ~20% for English text (vs char count which diverges by 2-4x on code/unicode).
- Cap at ~150 words (≈ 200 tokens) instead of 800 chars.
- If the LLM router exposes `get_token_counter()` ([router.py:220-227](../../src/maxim/models/language/router.py#L220)), use it for accurate counting. Check via `hasattr(self._llm, 'get_token_counter')`. Fall back to word count if unavailable or if the call fails.

**Design choice — why not require a tokenizer:** the narrator's token budget is approximate (it's a rolling context window, not a hard limit). A 20% error from word count is acceptable here. A hard tokenizer dependency would couple narrator.py to a specific model family.

**Pass gate:**
- `test_story_context_truncation_word_count`: 300-word story context truncates to ~150 words
- `test_story_context_unicode_handled`: unicode-heavy text doesn't blow past the budget the way char-count would
- Regression: existing narrator tests pass unchanged

**Scope:** ~25 LOC in `narrator.py`, ~25 LOC tests.

### Stage 5 — Wire `MaximDisplay` into `sim_logger`

**The missing integration.** `MaximDisplay` is fully built ([interactive/display.py](../../src/maxim/interactive/display.py)) — rich `Live` panels with agent log, status bar, input area, extension protocol. The foundational buildout plan ([foundational_buildout_plan.md:998-1002](../../docs/plans/archive/foundational_buildout_plan.md#L998)) designed the routing: "When `MaximDisplay` is active, `sim_log()` routes to `display.log()` instead." This routing was never wired. With 51+ `sim_log` call sites across the runtime, wiring this one integration point gives the entire system a rich panel UI for free.

**What's built:**

1. **Thread-safe `MaximDisplay`:** Add `threading.RLock` to `MaximDisplay`. All mutations — `log()`, `set_status()`, `set_prompt()`, `_refresh()` — acquire the lock. The orchestrator runs 3+ concurrent threads (`sim.aut`, `sim.stdin`, `sim.stall` + main) that all emit `sim_log()` events. Without the lock, concurrent `display.log()` calls race on the deque append + `_live.update()` — one thread's update overwrites another mid-render. RLock (not Lock) because `log()` calls `_refresh()` internally.

2. **Module-level display reference in `sim_logger.py`:** Add `_active_display: MaximDisplay | None = None` with thread-safe `set_active_display(display)` / `get_active_display()` accessor functions (guarded by a module-level `threading.Lock`). When set, `sim_log()` routes to `display.log()` instead of `print()`. When not set, falls back to current ANSI-print behavior.

3. **Route `_emit()` through the display when active:** The routing must happen in `_emit()` ([sim_logger.py:171-177](../../src/maxim/simulation/sim_logger.py#L171)), not just in `sim_log()`. All the display-tier functions (`display_scene()`, `display_action()`, `display_response()`, `display_turn()`) call `_emit()` — if `_emit()` still calls `print()` when a display is active, the raw ANSI output corrupts the `Live` panel. `_emit()` checks `get_active_display()` and routes to `display.log()` when active, `print()` when not.

4. **Route `display_turn()` through `display.set_status(turn=str(n))`** when active — the turn number updates the status bar instead of printing a separator line.

5. **Wire `create_display()` in `cli.py`** alongside the existing `create_handler()` call (~line 1116). When `--interactive` is on (or AUTO resolves to on), create the display, call `start()`, and call `set_active_display()`. Call `stop()` at session end.

6. **atexit + SIGINT cleanup for `Live.stop()`:** Register `_cleanup_display()` via `atexit` in `sim_logger.py` (alongside the existing `_cleanup_log_file` handler). This calls `get_active_display().stop()` if the display is still active at process exit. Without this, Ctrl+C or unhandled exceptions leave the terminal in raw mode (cursor hidden, input corrupted, user must manually `reset`). Also add explicit `display.stop()` in `cli.py`'s shutdown path and SIGINT handler.

7. **Graceful degradation:** If `rich` is not installed or stdout is not a TTY, `create_display("auto")` returns `None`, `set_active_display(None)` is a no-op, and all 51+ call sites fall through to the existing ANSI-print path. Zero behavioral change for headless/piped/non-rich environments.

**What this does NOT do (yet):**
- No `DisplayExtension` implementations (DM campaign panels, character sheets). That's B3/DM territory. The extension protocol is ready but this stage just wires the core display.
- No `PromptHandler` ↔ `MaximDisplay` integration (showing prompts in the input panel). The display has `set_prompt()` / `clear_prompt()` but wiring it to `RequestInteractionTool` is a follow-up — the tool currently prints via `display_scene()` which will automatically route through the display once wired.

**Design choices:**
- **Module reference, not mutable global import:** Per the [mutable globals lesson](../../CLAUDE.md) (`feedback_module_extraction.md`), `sim_logger` stores the display as `_active_display` and callers use `sim_logger.get_active_display()`. The display is set via function call, not imported by name.
- **RLock in display, Lock for module accessor:** The display needs RLock because `log()` → `_refresh()` is re-entrant. The module-level `_active_display` accessor uses a plain Lock since get/set are non-re-entrant.
- **JSONL writes are unaffected by display routing:** The `sim_log()` JSONL path ([sim_logger.py:465-469](../../src/maxim/simulation/sim_logger.py#L465)) always persists every event regardless of display tier or active display. The display routing only replaces terminal output. This is intentional — JSONL is "recorder for everything," display is "what the user sees."

**Pass gate:**
- `test_sim_log_routes_to_display_when_active`: set active display → `sim_log("HIPPO", "test")` → `display._log_lines` contains the entry, stdout has no ANSI output
- `test_sim_log_falls_back_when_no_display`: no display set → `sim_log()` prints to stdout as before
- `test_display_scene_routes_through_display`: active display → `display_scene("text")` → appears in display log
- `test_display_turn_updates_status`: active display → `display_turn(5)` → `display._status["turn"] == "5"`
- `test_cli_creates_display_when_interactive`: mock `create_display` → verify called when interactive mode is on
- `test_display_concurrent_log_no_corruption`: 8 threads × 50 `sim_log()` calls → all entries present in display, no exceptions
- `test_display_atexit_stops_live`: register atexit → call handler → `display._live is None`
- Visual: run `maxim --sim "test memory" --interactive` → see rich panel layout instead of raw ANSI lines

**Scope:** ~80 LOC in `sim_logger.py` + `display.py`, ~20 LOC in `cli.py`, ~80 LOC tests.

### Stage 6 — Light prompt cleanup

**What's removed** (genuinely unused, no planned callers):

1. **Unused `PromptType` variants:** Remove `SHORT_TEXT`, `LONG_TEXT`, `NUMERIC`, `RATING` from the enum. Keep `SINGLE_CHOICE`, `MULTI_CHOICE`, `CONFIRM`, `FREEFORM` — the four types with production callers. If B3 or another plan needs these types, re-add WITH a caller in the same commit.
2. **`freeze_context()` helper** ([prompts.py:44-46](../../src/maxim/interactive/prompts.py#L44)): zero production callers, wraps `tuple(kwargs.items())`. **Must also remove from `interactive/__init__.py`** — it's exported in `__all__` at [interactive/__init__.py:26,34](../../src/maxim/interactive/__init__.py#L26). Remove the import and the `__all__` entry.
3. **`poll_freeform()` ABC method** ([prompts.py:93-99](../../src/maxim/interactive/prompts.py#L93)): declared, never overridden, never called.
4. **`PromptRequest` fields never set in production:** `min_selections`, `max_selections`, `value_range`, `context`. The single production call site at [tools/display.py:147](../../src/maxim/tools/display.py#L147) only sets `prompt_type`, `question`, `options`.
5. **`revert_after_turns` parameter** on `agent_escalate_display()` ([sim_logger.py:94](../../src/maxim/simulation/sim_logger.py#L94)): declared, no caller implements the revert.

**What stays:**
- **`MaximDisplay` + `DisplayExtension`** — now wired in Stage 5. Earning its keep.
- **`CallbackPromptHandler`** — mechanism for the Python API (`pymaxim`) and future game-engine integration (see [game_npc_integration.md](game_npc_integration.md)). 38 LOC, no churn.
- **`PromptRequest.default` and `timeout_sec`** — production-relevant (read by `select.select()` in `PlainPromptHandler`).

**Pass gate:**
- All existing tests pass (update test references to removed types)
- `ruff check` clean
- Grep: `PromptType.NUMERIC`, `PromptType.RATING`, `PromptType.LONG_TEXT`, `PromptType.SHORT_TEXT` → zero matches in `src/`
- Grep: `freeze_context` → zero matches in `src/` (removed from `__init__.py` and `prompts.py`)

**Scope:** net ~-60 LOC (removal > addition). ~20 LOC test updates.

### Stage 7 — Integration smoke test

**What's built:**
- One end-to-end test that wires `RequestInteractionTool` → handler → validates the full round trip: tool receives question + options, handler fires, tool returns user's actual response with output starting with `"User responded:"`.
- One test for the disabled path: interactive OFF → output contains `"autonomously"`, no handler invoked.
- One test for the no-handler path: handler is None, interactive ON → output contains `"could not be collected"`.

**Pass gate:**
- `test_interaction_tool_end_to_end_with_handler`: handler returns "option B" → tool output contains `"User responded: option B"`
- `test_interaction_tool_end_to_end_disabled`: interactive OFF → output contains `"autonomously"`
- `test_interaction_tool_end_to_end_no_handler`: handler is None → output contains `"could not be collected"`

**Scope:** ~50 LOC tests.

### Stage 8 — Agent introspection tools

**4 high-priority tools from [tool_refinement_plan.md](tool_refinement_plan.md)** — ~100 LOC each, no prerequisites, all reading from existing data sources. These give the agent self-awareness about its own learning, memory, performance, and pain state.

**What's built:**

1. **`nac_stats`** — total observations, causal link count, top-rewarded tools (by mean RPE), RPE distribution (mean/std/min/max over recent window). Source: `decisions/nac.py` internal state. The agent can reason about what it's learned: "I've observed 47 tool outcomes, `memory_recall` has the highest reward signal, `filesystem_write` has negative valence."

2. **`memory_pressure`** — per-tier counts (FORMING/WORKING/SHORT_TERM/LONG_TERM), promotion rate (promotions per minute over last 10 min), total episodes, oldest/newest episode timestamps. Source: `memory/hippocampus.py` internal state. The agent can assess its own memory health: "I have 200 episodes, 15 in WORKING tier, promotion rate is 2/min."

3. **`loop_stats`** — current Hz, average cycle time (last 100 cycles), total steps since boot, time since last action, time since last percept. Source: `runtime/loop_controller.py` counters. The agent can diagnose its own performance: "I'm running at 4Hz, average cycle is 250ms, last action was 30s ago."

4. **`pain_triggers_active`** — currently active pain triggers with intensity, source entity, failure mode, and time since onset. Source: `proprioception/pain_bus.py` recent signals. The agent can reason about its own discomfort: "I have an active pain trigger from `rusty_sword` entity, `grip_failure` mode, intensity 0.7, onset 5s ago."

**Design principles** (from tool_refinement_plan.md):
- Read-only (no mutation)
- Size-capped (~4KB per response)
- Context-gated (only registered when relevant data sources exist)
- Secrets stay opaque (no API keys, no raw memory content)

**All four tools go in a single new file:** `src/maxim/tools/agent_introspection.py`. Registered in the tool registry alongside existing introspection tools. Each tool follows the existing `Tool` ABC pattern in [tools/base.py](../../src/maxim/tools/base.py).

**Pass gate:**
- `test_nac_stats_returns_structured_data`: agent with NAc → tool returns dict with `total_observations`, `causal_link_count`, `top_rewarded_tools`
- `test_nac_stats_without_nac`: agent without NAc → tool returns `success=False, error="NAc not available"`
- `test_memory_pressure_returns_tier_counts`: agent with hippocampus → tool returns per-tier counts
- `test_loop_stats_returns_timing`: agent in loop → tool returns Hz, cycle time, steps
- `test_pain_triggers_active_reflects_recent_pain`: fire PainSignal → tool returns it in active triggers
- `test_pain_triggers_empty_when_no_pain`: no pain signals → tool returns empty list

**Scope:** ~400 LOC in `tools/agent_introspection.py`, ~150 LOC tests.

## What this plan does NOT include

- **Acting Coach (B3).** Prompt engineering on top of working plumbing — ship after this. See [prompt_b3_b5_track.md](prompt_b3_b5_track.md). Stage 5's `MaximDisplay` wiring gives B3 a working panel system to build `DisplayExtension` implementations on. **Chain: interactive_experience_031 (0.3.1) → B3 acting coach (0.5) → B4 replanning (0.5, 1.0-GATING).**
- **`DisplayExtension` implementations.** DM campaign panels (character sheet, inventory, encounter info) are B3/DM territory. Stage 5 wires the core display; extensions come later.
- **Prompt type re-addition.** If B3 or another plan needs `NUMERIC`, `RATING`, `LONG_TEXT`, or `SHORT_TEXT`, add them back WITH a production caller in the same commit. Don't add types without callers.
- **`ToolResult` metadata wiring.** Stage 1 adds metadata to `ToolOutput` for forward compatibility. Wiring it through `ToolResult` to the agent bus is a deeper refactor (touches [agents/bus.py](../../src/maxim/agents/bus.py)) that belongs in `agent_factory_canonicalization.md`, not a 0.3.1 patch.
- **Display tier for PromptAssembler.** `compose_observation_section()` includes substrate memory regardless of display tier. Phase 3 concern.
- **Token-exact counting.** Word-count heuristic is sufficient for narrator's rolling context.
- **Model capability detection for `use_two_call`.** Parameter exists and works — callers just always pass the default. Follow-up item.
- **InteractiveMode simplification.** The simplification review noted that the 3-state enum (AUTO/ON/OFF) could be a bool + critical override. True, but changing it touches all callers of `set_interactive_mode()` and `should_prompt()` — more churn than a 0.3.1 patch warrants.
- **`should_prompt` layer migration.** `should_prompt()` lives in `simulation/sim_logger.py` but is consumed by `tools/display.py` — mild layering violation (tools → simulation). Moving it to `interactive/mode.py` or similar is cleaner but not blocking.
- **`PromptHandler` ↔ `MaximDisplay` input panel integration.** ✅ DONE — `RequestInteractionTool` now calls `set_prompt()` before prompting and `clear_prompt()` in a `finally` block after. Questions appear in the dedicated input panel at the bottom of the display, not just in the log.

## Stage ordering

Stages 1-4 are independent and can ship in any order (or in parallel). Stage 5 (`MaximDisplay` wiring) is independent of 1-4 but should ship before B3. Stage 6 (cleanup) is independent. Stage 7 depends on Stage 1 (tests the output strings). Stage 8 (introspection tools) is fully independent — no dependencies on any other stage.

**Recommended order:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8.

Stages 1-4 and 8 can all run in parallel (they touch different files). Stage 5 is the largest and riskiest (thread safety, atexit). Stage 7 validates Stage 1. Stage 8 is independent and can ship in the same PR or separately.

## Risks

1. **Stage 1 output string change.** The old fallback string `"Question displayed to user. Response will come on next turn."` is asserted in [test_request_interaction_tool.py:104](../../tests/unit/test_request_interaction_tool.py#L104) (`assert "next turn" in output`). This test will break — update it to match the new string (`"could not be collected"`). The disabled-path string change (`"Interaction disabled"` → adds `"autonomously"`) has no existing test assertions on the added text, so it's safe.
2. **Stage 3 ValueError on unknown mode.** Any caller passing a typo'd mode string will now crash instead of silently working. All production call sites pass hardcoded `"auto"` — verified by Executor review.
3. **Stage 5 thread safety.** The orchestrator runs 3+ concurrent threads. Without the RLock added in this stage, concurrent `display.log()` calls race on deque append + `_live.update()`. The lock adds ~5 LOC to `MaximDisplay` but is non-optional — omitting it creates a sporadic race condition under multi-turn simulation load.
4. **Stage 5 terminal corruption on exit.** If `rich.live.Live` is active when the process exits (Ctrl+C, exception, `os._exit()`), the terminal stays in raw mode — cursor hidden, input corrupted. The atexit handler + SIGINT cleanup added in this stage prevent this. Without them, every `--interactive` session that ends abnormally leaves a corrupted terminal requiring `reset`.
5. **Stage 5 `print()` corruption.** When `Live` is active, raw `print()` calls outside `sim_log()` can corrupt the display. Mitigation: the routing happens in `_emit()` which is the single chokepoint for all `display_*()` functions — so all 51+ sim_log call sites are covered. Stray `print()` from tool outputs or third-party code could still interfere. `Live(screen=False)` (already in `MaximDisplay.__init__`) helps but doesn't fully prevent this.
6. **Stage 6 `freeze_context` export.** `freeze_context` is exported from [interactive/__init__.py:26,34](../../src/maxim/interactive/__init__.py#L26). Removing it from `prompts.py` without removing the `__init__.py` import/export would cause `ImportError` at package load time. Must update both files in the same commit.
7. **Stage 6 removal of PromptType variants.** Any downstream code importing `PromptType.NUMERIC` etc. will get `AttributeError`. Mitigation: zero production callers in `src/`, not part of public `pymaxim` API surface.

## Files touched

| File | Stages |
|---|---|
| `src/maxim/tools/display.py` | 1 |
| `src/maxim/simulation/narrator.py` | 2, 4 |
| `src/maxim/interactive/prompts.py` | 3, 6 |
| `src/maxim/interactive/display.py` | 5 |
| `src/maxim/simulation/sim_logger.py` | 5, 6 |
| `src/maxim/cli.py` | 3, 5 |
| `tests/unit/test_interactive.py` | 1, 3, 5, 6, 7 |
| `tests/unit/test_generative_campaign.py` | 2, 4 |
| `tests/unit/test_request_interaction_tool.py` | 1, 7 |
| `src/maxim/tools/agent_introspection.py` (new) | 8 |
| `tests/unit/test_agent_introspection.py` (new) | 8 |

## Review history

- **2026-04-18 (draft):** Initial 6-stage plan.
- **2026-04-18 (review round 1):** 3-lens parallel review (Executor, Architecture, UX). Critical finding: `ToolOutput.metadata` never reaches the LLM — Stage 1 rewritten to use output strings. Token-counting API corrected (Stage 4). Phase name free-form string handling added (Stage 2).
- **2026-04-18 (simplification review):** ~400 LOC dead interactive code identified. Original Stage 5 (prompt type enhancement) replaced with cleanup.
- **2026-04-18 (user review):** `MaximDisplay` is NOT dead code — it's half-built infrastructure where the hard part (display class) is done and the easy part (sim_logger routing) was never wired. The foundational buildout plan explicitly designed this integration. Stage 5 rewritten from "remove MaximDisplay" to "wire MaximDisplay into sim_logger." Prompt type cleanup moved to Stage 6 (smaller scope). Stage 7 added for integration test. Plan grows from 6 to 7 stages but the core display wiring is the high-value addition — it gives `--interactive` users the rich panel UI that was always intended.
- **2026-04-18 (risk + execution/architecture review):** Two critical risks folded into Stage 5: (1) thread safety — `MaximDisplay` needs `RLock` because 3+ orchestrator threads call `sim_log()` concurrently; (2) atexit handler for `Live.stop()` — without it Ctrl+C leaves terminal corrupted. Additional findings: routing must happen in `_emit()` not just `sim_log()` (all `display_*` functions call `_emit`); Stage 1 test at `test_request_interaction_tool.py:104` asserts on old string; `freeze_context` is exported from `interactive/__init__.py` and must be removed there too. Game NPC integration scoped as separate plan — interactive layer is orthogonal to agent control surface.
