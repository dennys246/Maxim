# Safe Freezing Plan

Systematic audit and freezing of Maxim's dataclass configurations for thread safety and immutability. Originally item #7 from the claw-code upgrade (now implemented and documented in ARCHITECTURE.md/DECISIONS.md). Split into its own plan because it requires per-class verification across 30+ config dataclasses.

---

## Motivation

Multiple threads (LLMWorker, agent loop, video capture, worker pool) reference shared config objects. Mutable configs risk accidental cross-thread mutation causing non-deterministic behavior. Freezing configs prevents this at construction time.

## Current State

- **30+ `*Config` dataclasses** across `src/maxim/`
- **4 are currently frozen** (from claw-code upgrade): `ToolOutput`, `LLMProposal`, `SkillResult`, `LongHorizonConfig`
- **At least 2 are actively mutated post-construction:**
  - `LLMAgentConfig` — `llm_agent.py:350-354` directly assigns `.profile`, `.quantization`, `.model_path` in `switch_model()`
  - `RTSPBridgeConfig` — `skills/rtsp_streaming.py` directly assigns `.rtsp_url`, `.fps` in `execute()`
- **1 confirmed safe to freeze immediately:** `LongHorizonConfig` (plan_document.py:57-80) — all scalar fields (bool, int, float, str), no mutations found

## Approach

### Phase 1: Freeze confirmed-safe configs

**DONE** — `LongHorizonConfig`, `ToolOutput`, `LLMProposal`, `SkillResult` all frozen. `ToolOutput` and `SkillResult` have `metadata: dict` fields (frozen but unhashable). `LongHorizonConfig` is fully hashable (all scalars).

### Phase 2: Audit remaining configs

For each config dataclass:

1. **Grep for mutations:** `config.field_name =` patterns in all files that import/use the config
2. **Check field types:** Mutable defaults (`list`, `dict`, `set`) prevent hashing but NOT freezing — frozen + unhashable is valid (prevents mutation, just can't use as dict key)
3. **Check `from_dict()` patterns:** If deserialization constructs then mutates, must refactor to build the full object in one shot
4. **Check `field(init=False)`:** `dataclasses.replace()` won't work for these fields

### Phase 3: Refactor mutation sites

For configs that are mutated post-construction:

```python
# Before (direct mutation — breaks with frozen=True):
self._config.profile = new_profile

# After (replace pattern — works with frozen=True):
self._config = dataclasses.replace(self._config, profile=new_profile)
```

**Risk:** `dataclasses.replace()` creates a NEW object. All references to the old config object become stale. This is NOT a simple find-and-replace — callers that hold a reference to the old config must be identified:
- If config is passed to constructor: the component keeps the old reference
- If config is accessed via property: safe (always reads current)
- If config is stored in a closure: stale reference

For `LLMAgentConfig.switch_model()`: the config is stored as `self._agent_config` on the agent — `.replace()` would update the agent's reference but any thread that cached the old config would see stale values. Must verify no thread caches it.

### Phase 4: Freeze and test

1. Add `frozen=True` to each audited config
2. Run full test suite
3. Grep for any remaining `config.field =` patterns (should be zero)

## Config Inventory (to audit)

Non-exhaustive list of config dataclasses to check:

| Config | Module | Priority | Notes |
|--------|--------|----------|-------|
| `LongHorizonConfig` | planning/plan_document.py | **DONE** | Safe — all scalars, no mutations |
| `ToolOutput` | tools/base.py | **DONE** | Frozen, unhashable (dict field) |
| `LLMProposal` | agents/llm_types.py | **DONE** | Frozen, unhashable (list fields) |
| `SkillResult` | skills/base.py | **DONE** | Frozen, unhashable (dict field) |
| `HippocampusConfig` | memory/hippocampus.py | High | Shared across threads |
| `NACConfig` | decisions/nac.py | High | Shared across threads |
| `ConsolidationConfig` | memory/consolidation.py | Medium | Used by sleep system |
| `DefaultNetworkConfig` | default_network/ | Medium | Controls reactive behaviors |
| `AttentionConfig` | attention/ | Medium | Gaze control |
| `ExplorationConfig` | modes/ | Medium | Exploration policy |
| `LLMAgentConfig` | agents/ | Low | **MUTATED** in switch_model() — requires refactor |
| `RTSPBridgeConfig` | skills/ | Low | **MUTATED** in execute() — requires refactor |
| `PainConfig` | proprioception/pain.py | Medium | Used by PainDetector |
| `EnergyConfig` | energy/ | Low | Tracking only |
| `ContextPoolConfig` | runtime/ | Low | Context management |

## Dependencies

- None — this is independent of the claw-code upgrade plan
- Can be done incrementally (one config at a time)
- Each frozen config is independently testable
