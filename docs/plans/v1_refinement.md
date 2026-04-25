# v1 Refinement — Removing Silent Backward Compatibility

**Status:** PLANNING (post-1.0, pre-v1.0 release)

## Motivation

Several backward-compatibility shims exist that silently accept under-specified inputs instead of failing loudly. Per the CLAUDE.md lesson "push silent-no-op invariants into types, not helpers," these are silent-failure risks that accumulate over time. Before v1.0 release, each should either become a loud error or be structurally enforced.

## Items

### 1. Modulators without sensors

**Current:** `SpecModulator` with no `sensors` silently returns `compute_integrity() = 1.0`. A modulator that should have body-part sensors but doesn't (because the YAML author forgot) appears "always healthy" — damage to it is a no-op, and the agent never learns to protect that body part.

**v1 behavior:** Require every modulator to declare at least one sensor. Modulators that are purely capability axes (no physical substrate) should declare `abstract: true` to explicitly opt out.

```yaml
# v1: explicit opt-out
modulators:
  communication:
    abstract: true  # no physical substrate — no sensors needed
    affordances:
      speak: {params: {message: str}}
  wing:
    sensors:  # REQUIRED — forgetting is a parse-time error
      membrane_integrity: {range: [0,1], initial: 1.0}
    affordances:
      take_flight: ...
```

**Migration:** CI validation script flags all seed components missing modulator sensors. Each gets either `sensors:` or `abstract: true`. User components in `~/.maxim/components/` get a deprecation warning at parse time for one minor version, then a hard error.

### 2. DamageEntityTool shim

**Current:** `DamageEntityTool` exists as a deprecated shim delegating to `DamageComponentTool`. Old orchestrator prompts still work.

**v1 behavior:** Remove `DamageEntityTool` entirely. Update all orchestrator prompts to use `damage_component`. The CI grep allowlist in `test.yml` blocks new callers.

### 3. Entity health as direct sensor

**Current:** `health` can be either a direct entity-level sensor (old format) or `health: derived` (new format, computed from component integrities). Both work silently.

**v1 behavior:** If an entity has modulators with sensors AND a direct `health` sensor (not `derived`), emit a parse-time warning: "Entity has component sensors but health is not derived — damage to components won't affect entity health." One minor version of warnings, then require `health: derived` when any modulator has sensors.

### 4. Probe compat shims

**Current:** `probe_llm_server`, `llm_server_responding_at`, `_probe_once` exist as deprecated shims. CI grep allowlist limits callers to 4 sites.

**v1 behavior:** Remove shims. Migrate remaining 4 callers to `_MaximPeerBackend.for_url(...).health_check()`.

### 5. `SendMessageTool._detect_attack` dead code

**Current:** `_detect_attack()` and `_ATTACK_KEYWORDS` are dead code after auto-damage removal. Still importable, still in memory, could confuse future contributors.

**v1 behavior:** Delete `_detect_attack`, `_ATTACK_KEYWORDS`, and all related comments. If keyword-based attack detection is ever needed again, it belongs in the orchestrator prompt (LLM decides what's an attack), not in Python substring matching.

### 6. Raw `PainBus()` / `ReactionBus()` / `MemoryHub()` construction

**Current:** Tests use raw constructors (~30+16+14 sites). Production code uses builders. The structural enforcement lives at the production door, not the type.

**v1 behavior:** Add `_allow_raw=False` parameter to raw constructors. Production paths use builders. Test paths pass `_allow_raw=True`. Any call without `_allow_raw=True` raises `TypeError` with a migration hint pointing to the builder.

## Timing

These are all breaking changes. They ship in the v1.0 release (or a 0.9 "deprecation warnings" release followed by a 1.0 "hard errors" release). The two-phase approach (warn then error) is preferred for items 1, 3, and 6 where external users may have custom specs/code. Items 2, 4, and 5 are internal and can be hard-removed in one step.
