# Mode System Refactor Plan

> **Status:** Not started. Independent of bio-system hardening / DM MVP — can be interleaved or run after.
>
> **Last updated:** 2026-04-07
>
> **Summary:** Collapse the modes system from three orthogonal dimensions (operational mode + processing state + strategy) down to a single **autonomy level** (passive/active/singularity). Remove all behavioral steering machinery (strategies, LiveModeIntent, CuriosityManager, exploration system). Make sleep/wake a **tool the agent calls** instead of a separate state machine dimension.
>
> **Note:** Bio-system hardening and DM MVP are actively in progress. This plan is designed to not conflict with either, but verify imports/line numbers before implementing — files may have shifted.

## Why

The modes system was designed when the agent pipeline was simpler. Today, behavioral flexibility is handled better by the systems that have grown up around it:

- **Goal agent** already drives task switching and strategic reasoning
- **NAc** learns what approaches work through causal observation
- **Default Network** handles reactive behavioral shifts (orienting, social, startle)
- **LLM context** naturally adapts tool selection to the situation

What remains valuable is the **safety/permissions tier** — controlling what the agent *can* do (tool access, filesystem, initiative caps). The behavioral steering on top of that (strategies, exploration budgets, curiosity decay, agent-defined intent) is redundant with the agent's natural reasoning.

Sleep/wake as a separate ProcessingState dimension is over-engineered. The agent should decide to sleep the same way it decides anything else — by calling a tool. Wake happens naturally when user input arrives.

## What stays vs what goes

### Keep (autonomy layer)
- `OperationalMode` enum: passive / active / singularity
- `ProcessingState` enum: awake / sleep (kept as enum for type safety + serialization; toggled by sleep tool + auto-wake)
- Per-mode tool permissions (`allowed_tools`, `forbidden_tools`, capability flags)
- Per-mode filesystem gating (`get_mode_filesystem_config()`)
- Per-mode Default Network config (`DefaultNetworkModeConfig`)
- Per-mode `max_initiative` cap
- Per-mode `context_prompt` (the one behavioral injection that earns its keep)
- `StateManager` (simplified to track autonomy level + sleep flag)
- `ModeDefinition` dataclass (trimmed — see dead fields below)
- Sleep config (`SLEEP_CONFIG`, `is_wake_keyword()`)
- `AutonomyLevelTool` in `mode_switch.py` (approval-gated escalation)

### Remove

**Modules (delete entirely):**
- `modes/strategies.py` — `StrategyLibrary`, `BUILTIN_STRATEGIES`, outcome recording (~417 LOC)
- `modes/live_intent.py` — `LiveModeIntent`, `LiveModeIntentStore`, `get_live_mode_with_intent()` (~270 LOC)
- `modes/exploration.py` — `ExplorationPolicy`, `ExplorationConstraints`, `ExplorationBudget`, `CuriosityManager`, `ExplorationSession`, `AdversarialFocusValidator`, `FocusDecomposer`, `parse_explore_command()` (~657 LOC)
- `tools/define_live_intent.py` — 4 LiveModeIntent tools (~317 LOC)

**From `definitions.py`:**
- `Strategy` dataclass (lines 69-101)
- `STRATEGIES` dict (lines 106-198) + `get_strategy()`, `get_strategy_for_keyword()`
- `LEGACY_MODE_MAPPING` + `_create_legacy_mode()` (lines 657-706)
- `get_exploration_mode_with_policy()` (lines 767-790) — never called outside definitions.py
- `get_mode_for_context()` (lines 793-821) — **never called anywhere**
- `get_state_for_context()` (lines 824-861) — **never called anywhere**
- `parse_state_command()` (lines 864-927) — **only defined in definitions.py, never called**
- `outcome_memory_key` field on `ModeDefinition` — **set everywhere, read nowhere**
- `preferred_strategies`, `avoid_strategies` fields on `ModeDefinition`
- Strategy-related parts of `MaximState` (`get_strategy()`, `get_effective_initiative()` strategy branch)

**From other files:**
- `StrategyInfo` dataclass (from `llm_types.py`)
- Strategy injection in prompt builder
- Exploration mode branch in agent loop + CLI
- Strategy convenience methods in `selfy.py` and `state_manager.py`

### Transform
- Sleep/wake: `ProcessingState` stays as enum internally, but the transition is triggered by the agent calling a `sleep` tool (not by the state machine). Auto-wake on user input.
- `ModeSwitchTool`: remove "sleep" from its `VALID_MODES` — sleep routes through `SleepTool` only, preventing two paths to the same state.

### Migrate: `simulation/personas.py`
- All 8 simulation personas are currently `Strategy` objects imported from `maxim.modes.definitions`
- **Replace with a `Persona` dataclass** defined in `personas.py` itself — personas are orchestrator personality configs, not behavioral strategies
- Fields: `name`, `description`, `focus`, `context_prompt`, `max_initiative` (subset of Strategy that personas actually use)
- `keywords`, `preferred_tools`, `avoid_tools`, `response_style` are unused by personas — drop them

---

## Phase 1: Migrate personas off Strategy (~40 LOC)

Do this first since it's the only consumer where `Strategy` removal would break simulation functionality.

### 1.1 `simulation/personas.py`
- Define a `Persona` dataclass at module level: `name`, `description`, `focus`, `context_prompt`, `max_initiative`
- Change `SIMULATION_PERSONAS: dict[str, Strategy]` → `dict[str, Persona]`
- Replace `Strategy(...)` constructors with `Persona(...)`
- Remove `from maxim.modes.definitions import Strategy`
- Verify no other simulation code accesses Strategy-specific fields (`keywords`, `preferred_tools`, etc.) on personas

---

## Phase 2: Strip consumers of strategy/exploration imports (~350 LOC removed)

Update all files that import from the doomed modules so they stop depending on them. No behavior change yet — just decouple.

### 2.1 `runtime/agent_loop.py`
- Remove exploration mode branch (~lines 2063-2113) that builds `ExplorationPolicy`, uses `get_strategy_library()`, builds `StrategyInfo` list
- Remove `StrategyInfo` import
- Simplify mode_info construction to just `get_mode()` for the operational mode

### 2.2 `agents/prompt_builder.py`
- Remove `STRATEGIES` import from definitions (imported twice — lines 122 and 878)
- Remove strategy context injection blocks (lines that look up `request.current_strategy` and inject strategy name/focus/context_prompt)
- Mode `context_prompt` injection stays

### 2.3 `agents/llm_types.py`
- Remove `StrategyInfo` dataclass (lines 82-87)
- Remove from `LLMRequest`: `strategies`, `current_strategy` fields
- Keep `ModeInfo` unchanged

### 2.4 `agents/llm_worker.py`
- Remove `strategies` and `current_strategy` parameters from request construction (lines ~561, 608)

### 2.5 `agents/__init__.py`
- Remove `StrategyInfo` from imports and `__all__`

### 2.6 `conscience/selfy.py`
- Remove `request_strategy_*()` methods (6 methods, lines 543-559)
- Remove `current_strategy` property
- Keep `request_sleep()`, `request_wake()`, `request_mode_*()`, `operational_mode`, `processing_state`

### 2.7 `runtime/bootstrap.py`
- Remove LiveModeIntent tools registration block (~lines 189-219)

### 2.8 `agents/perception_agent.py`
- Remove `_parse_explore_command()` and its import of `parse_explore_command` (line 125)
- Remove exploration command detection

### 2.9 `cli.py`
- Remove all 6 exploration CLI args: `--explore`/`-e`, `--exploration-focus`, `--exploration-allow-scripts`, `--exploration-allow-training`, `--list-sessions`, `--resume-session`
- Remove entire exploration mode handler branch (~lines 1482-1591, ~120 LOC)

### 2.10 `tools/mode_switch.py`
- Simplify `VALID_MODES` from 9 entries to `{"passive", "active", "singularity"}` — explicitly remove "sleep" (will be handled by `SleepTool`)
- `AutonomyLevelTool` stays unchanged

---

## Phase 3: Simplify core modes modules (~500 LOC removed)

### 3.1 `modes/definitions.py`
- Remove `Strategy` dataclass (lines 69-101), `STRATEGIES` dict (lines 106-198), `get_strategy()`, `get_strategy_for_keyword()`
- Remove from `ModeDefinition`: `preferred_strategies`, `avoid_strategies`, `outcome_memory_key`
- Remove `get_exploration_mode_with_policy()` (lines 767-790)
- Remove `get_mode_for_context()` (lines 793-821) — orphaned, zero callers
- Remove `get_state_for_context()` (lines 824-861) — orphaned, zero callers
- Remove `parse_state_command()` (lines 864-927) — orphaned, zero callers
- Remove `LEGACY_MODE_MAPPING`, `_create_legacy_mode()`, `MODES` dict (lines 657-706)
- Remove `MaximState.get_strategy()`, simplify `get_effective_initiative()` to just return mode's `max_initiative`
- Remove TYPE_CHECKING imports for `ExplorationPolicy`, `LiveModeIntentStore`
- Simplify `get_mode()`: remove `intent_store` parameter, look up from `OPERATIONAL_MODES` directly
- Add simple legacy name map for backward compat:
  ```python
  _LEGACY_NAME_MAP = {
      "observe": "passive", "reflection": "passive", "sleep": "passive", "train": "passive",
      "live": "active", "active-assistance": "active", "active_assistance": "active",
      "exploration": "active", "research": "active",
  }
  ```
- `list_modes()` returns `["passive", "active", "singularity"]`

### 3.2 `modes/state_manager.py`
- Remove `set_strategy()`, strategy property, strategy from `AgentProtocol`
- Remove from `StateManagerConfig`: `initial_strategy`, `auto_wake_on_strategy_change`
- Remove `strategy` from `to_dict()` / `from_dict()` (keep graceful ignore of unknown keys for old persisted state via `.get()`)
- Remove `request_strategy_*()` convenience methods (lines 293-309)
- Simplify callback `state_type` to only "processing_state" and "operational_mode"

---

## Phase 4: Delete dead modules (~1,660 LOC removed)

Delete these files entirely:
- `src/maxim/modes/strategies.py` (~417 LOC)
- `src/maxim/modes/live_intent.py` (~270 LOC)
- `src/maxim/modes/exploration.py` (~657 LOC)
- `src/maxim/tools/define_live_intent.py` (~317 LOC)

Update `modes/__init__.py`:
- Remove all imports from `strategies`, `exploration`, `live_intent`
- Remove from `__all__`: `Strategy`, `StrategyLibrary`, `get_strategy_library`, `ExplorationPolicy`, `ExplorationConstraints`, `ExplorationBudget`, `ExplorationSession`, `ExplorationSubGoal`, `CuriosityManager`, `AdversarialFocusValidator`, `FocusDecomposer`, `parse_explore_command`, `get_exploration_mode_with_policy`

---

## Phase 5: Sleep tool (~60 LOC added)

### 5.1 Create `src/maxim/tools/sleep.py`

```python
class SleepTool(BaseTool):
    """Agent calls this to enter low-power mode when idle.

    The agent decides to sleep when there's nothing to do.
    Wake is automatic — any user input triggers wake via is_wake_keyword()
    and the agent loop's existing sleep-check logic.
    """
    name = "sleep"
    # Takes optional 'reason' parameter (string, for provenance/logging)
    # Calls state_manager.set_processing_state(ProcessingState.SLEEP)
    # Returns confirmation with reason logged
```

No separate "wake" tool — wake is implicit on user input. The agent loop already checks `is_sleeping` and calls `set_processing_state("awake")` when input arrives.

### 5.2 Register in `runtime/bootstrap.py`

### 5.3 Update `data/util/phrase_responses.json`
- Keep "maxim" → wake (auto-wake keyword)
- Keep "maxim sleep" → now routes to the sleep tool instead of `request_sleep`
- Remove strategy-switching phrase responses ("maxim observe", etc.)

---

## Phase 6: Update tests (~200 LOC changed)

### 6.1 `tests/unit/test_state_manager.py`
- Remove `TestStateManagerStrategy` class
- Remove strategy references from serialization tests
- Remove strategy from MockAgent protocol
- Keep all processing_state and operational_mode tests

### 6.2 Other test files
- Remove `StrategyInfo` references from `test_llm_types.py`, `test_llm_worker_pool.py`
- Check for exploration/strategy refs in any other test files
- Add test for `SleepTool` (verify state transition, verify auto-wake path)
- Add test for `Persona` dataclass in personas test if one exists

### 6.3 Grep sweep
Search for any remaining references to deleted symbols across the entire codebase:
```bash
rg "StrategyInfo|StrategyLibrary|LiveModeIntent|LiveModeIntentStore|ExplorationPolicy|ExplorationBudget|ExplorationConstraints|CuriosityManager|ExplorationSession|FocusDecomposer|AdversarialFocusValidator|parse_explore_command|get_strategy_library|get_exploration_mode_with_policy|get_mode_for_context|get_state_for_context|parse_state_command|outcome_memory_key|define_live_intent" src/ tests/
```

---

## Phase 7: Cleanup

### 7.1 Documentation
- Simplify `docs/user/modes-guide.md` to document autonomy levels only
- Update `docs/user/tools.md`: remove LiveModeIntent tools, add sleep tool
- Update CLAUDE.md: simplify mode system description, remove strategy references

### 7.2 Persisted data (non-breaking)
- `data/agents/*/modes/live_intent.json` — orphaned after removal. Safe to leave (ignored) or delete.
- `data/agents/*/exploration_sessions/*.json` — orphaned. Same treatment.
- Note: `from_dict()` on StateManager should silently ignore `strategy` key so old persisted state files don't crash on load.

### 7.3 Verification
```bash
ruff check src/ tests/
ruff format src/ tests/
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py
```

---

## Estimate

~1,900 LOC removed, ~100 LOC added. Net reduction **~1,800 LOC**. Touches ~20 files, deletes 4.

(Previous estimate was ~1,240 LOC net — the audit found more dead code: orphaned functions in definitions.py, the full exploration CLI branch, personas migration.)

---

## Risk notes

- **`personas.py` is the only breaking consumer of `Strategy`** — Phase 1 handles this first so simulation keeps working throughout the refactor.
- **`from_dict()` backward compat** — old persisted state may contain `strategy` keys. The existing `.get()` pattern handles this gracefully (returns default, ignores unknown keys). No migration needed.
- **Two paths to sleep** — current `ModeSwitchTool` accepts "sleep" as a valid mode AND there will be a new `SleepTool`. Phase 2.10 explicitly removes "sleep" from `ModeSwitchTool.VALID_MODES` to prevent ambiguity.
- **DN mode config is orthogonal** — `DefaultNetworkModeConfig` on each `OperationalMode` survives untouched. DN behavior is per-autonomy-level, not per-strategy.
- **Active hardening/DM work** — bio-system hardening and DM MVP are in progress concurrently. This plan doesn't touch orchestrator wiring, bio-system init, SEM entities, or campaign infrastructure. Verify line numbers before implementing — files may have shifted.

---

## Integration with active work

### Relationship to Bio-System Wiring Hardening
**Independent.** The hardening plan fixes sim-mode initialization of bio-systems (orchestrator wiring, NAc observe, SCN/EC init, PainBus→NAc) and now also includes bio-system consolidation (Phase 7: absorb dormant systems like energy, skills, provenance) and dead runtime cleanup (Phase 8). None of that touches the modes system. The one thing to watch: if hardening adds mode-aware logic (e.g., "only wire X in active mode"), use the simplified autonomy level names, not strategy names.

**Division of dead code cleanup:** This plan removes modes-specific dead code (~1,800 LOC: strategies, exploration, LiveModeIntent). The hardening plan's Phase 8 removes general dead code (~434 LOC: resilient.py, session.py, debug_status_server.py, monitor_registry.py, dead planner methods). No overlap.

### Relationship to DM MVP
**Independent but beneficial to do first.** The DM plan adds a `dungeon_master` persona and campaign runtime. The DM persona would use the new `Persona` dataclass (Phase 1 of this plan creates it). If DM ships first on the old `Strategy`-based persona system, this plan's Phase 1 migrates it — no conflict either way.

### Recommended sequencing
1. **Bio-System Wiring Hardening** (in progress) — no dependency
2. **Mode Refactor** — clean, independent, reduces surface area by ~1,800 LOC
3. **DM MVP** — builds on clean foundation, uses `Persona` dataclass

Alternatively, interleave mode refactor with hardening if you want variety. They don't conflict.
