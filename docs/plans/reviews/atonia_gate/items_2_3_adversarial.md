# Atonia gate, items 2 and 3: adversarial design review

Reviewer: adversarial design pass (CLAUDE.md "Faster without lowering the bar", item 1), read-only, 2026-10-24.
Target: `docs/plans/atonia_gate_approach_notes.md` §0–§3 (DRAFT v1), checked against the worktree at `d9dffe6e`.
Paths are under `src/maxim/` unless they start with `scripts/` or `docs/`. Classes: DO-NOT-BUILD / SHOULD-FIX / NIT;
each finding is also tagged **fail-open** (lets something through) or **fail-closed** (refuses or breaks).

No finding is DO-NOT-BUILD. The shape is right: one effect set per class, frozen when the class is created, checked
against the object that actually runs. But two spec bugs would fail open as written (1, 2), and several plumbing gaps
let an effect past the ceiling the design claims (4, 6, 7, 11, 13). All of them should be folded before code.

## Findings

### 1. SHOULD-FIX (fail-open once someone "repairs" it): `effects(tool) <= ceiling` does not work on `enum.Flag`
The note gives the admit rule as `effect_of(tool) <= self.ceiling` and the type as `enum.Flag` (V2, §2 "One admit
path", §3 "Dispatch checks"). On CPython 3.12, `enum.Flag` defines no `<=`. Running `E.A <= E.A|E.B` raises
`TypeError: '<=' not supported`. The obvious way to make it run is to switch to `enum.IntFlag`, and then `<=` compares
the integers. With READ=1, SELF_SCOPE=2, WRITE_LOCAL=4, NETWORK=8, `WRITE_LOCAL <= READ|NETWORK` is **True**. That is
exactly Profile A's case: a write tool admitted under the network-read ceiling (checked in a scratch interpreter).
**Fold:** add one helper, `effect_within(e, ceiling) -> bool`, defined as `(e & ~ceiling) == Effect(0)`, and use it at
every check. Add a test that `Effect` is not an `int` subclass, and a test that WRITE_LOCAL is refused under
READ|NETWORK. Forbid `<=` and `in` on `Effect` in the note.

### 2. SHOULD-FIX (fail-open): `Effect(0)` is admitted under every ceiling
Under subset semantics an empty declaration passes every ceiling, and it also passes the "declares its own effect"
census. A tool that writes `effect = Effect(0)` ("no effect") gets in everywhere a READ tool would not.
**Fold:** `Tool.__init_subclass__` rejects `Effect(0)`, and so does any value that is not an `Effect`. The lowest
legal declaration is `READ`.

### 3. SHOULD-FIX (fail-closed, breaks third-party code; the miss path is unspecified)
- The rule "a mismatch between record and attribute fails admission loudly" is ambiguous. Suppose the check compares
  the record with `getattr(cls, "effect")`. A subclass of a READ class that declares nothing then has record `ALL`
  and an inherited attribute of `READ`. That is a mismatch, so it is refused even by a registry with no ceiling. This
  contradicts "A registry with no ceiling admits everything, so `register_tool` keeps working". **Fold:** compare the
  record against `cls.__dict__.get("effect", <absent>)`, never against `getattr`.
- `effect_of` on a type that has no record is unspecified. Registries accept any object today:
  `tools/registry.py::ToolRegistry.register` takes no `isinstance` check, `api.py::register_tool` takes `tool: Any`,
  and tests register `MagicMock`s. An object whose `__class__` property claims to be a READ Tool also passes
  `isinstance`. **Fold:** state that `effect_of` reads `type(obj)` (never `obj.__class__`), and that a miss or a
  non-`Tool` type returns `Effect.ALL` without raising `KeyError` and without being caught and swallowed.
- Checked and not a problem: `__init_subclass__` fires under `ABCMeta` and for `type(name, (Tool,), ns)`. `copy`
  keeps the class. Reassigning `__class__` changes which methods run, so it changes the effect consistently. A module
  `reload` gives a miss, which is `ALL` once the miss rule above is adopted. Nested `FunctionTool` classes cannot be
  pickled anyway.

### 4. SHOULD-FIX (fail-open): "what it can do" under-declares tools whose effect comes from an injected collaborator
A per-class effect has to be the union over every collaborator the constructor, or later wiring, can supply. Several
classifications in the note, or the obvious ones, miss this:
- `embodiment/tool_bridge.py::ModulatorAffordanceTool` is proposed as `ACTUATE | READ`. Its `execute` calls
  `self._modulator.execute(...)` (L576), and the modulator may be a backend that talks over a socket:
  `embodiment/backends/minecraft.py` (the Minecraft bridge) or the Reachy motor backend. That is NETWORK. Speech
  affordances set `speech_params` (`_affordance_speech_params`), which is COMMS. A successful call can also return
  `entity_acquired`, which makes the executor register more tools (`runtime/executor.py::_handle_entity_acquisition`),
  so it is composite.
- `tools/response.py::RespondTool` writes files (`utils/response_output.py` L225 `write_text`). It can also speak
  through an injected `tts_engine` and `speaker_fn`, which is ACTUATE and possibly NETWORK. So it is
  COMMS | WRITE_LOCAL | ACTUATE, not "COMMS".
- `tools/discovery.py::SensePresenceTool` is a "sense" tool, and it is also `auto_fire`. It triggers the imagination
  pipeline (`self._imagination_trigger.process_percept`, L197), which makes LLM calls and spawns entities.
- Collaborators are attached **after admission**: `simulation/orchestrator.py` (around L1640–1666) assigns
  `_imagination_trigger`, `_nac`, `_atl` and `_component_index` onto tools already in the registry. Freezing the
  effect at class creation does not cover what is wired in later.

**Fold:** write the rule as "the union over every collaborator type the class accepts, including ones set after
construction". Add one table row per collaborator-backed class. State that a tool which calls the LLM router is
NETWORK (because a lane may be remote).

### 5. SHOULD-FIX (fail-open by definition gap): the six flags are named but never defined
- SELF_SCOPE vs WRITE_LOCAL is undefined. Hippocampus and NAc state persists under `~/.maxim`, so is a memory write
  SELF_SCOPE or WRITE_LOCAL? Item 7 needs the answer.
- NETWORK has to be defined as **egress**. A GET can carry data out in its URL. Calling it a "network read" (V2,
  Profile A) undersells it in the AD9 effect table.
- Tools that widen the agent's own authority have no class: `tools/mode_switch.py::ModeSwitchTool`,
  `AutonomyLevelTool`, `tools/internet_search.py::InternetAccessTool`, and
  `tools/filesystem.py::RequestDirectoryChangeTool` (which widens `allowed_dirs`). If they are filed under SELF_SCOPE,
  a SELF_SCOPE ceiling admits authority changes. **Fold:** add a `GOVERN` flag, or classify these tools as `ALL`.
- The note contradicts itself. §4.2's `ProposeActionTool` is declared READ, yet "its `execute` appends one bounded
  record to item 5's log". By the note's own rule that is WRITE_LOCAL, and the observer's ceiling is READ. **Fold:**
  either the rule explicitly exempts the session's own audit log (and says why), or the declaration changes.

### 6. SHOULD-FIX (fail-open): auto-fire tools run outside every ceiling the executor holds
`runtime/loop_perception.py` (auto_sense, L188–203) runs `tool.execute()` on every tool from `get_auto_fire_tools()`
on every tick, without going through `Executor`. `auto_fire` is read with `getattr`, so an instance can set it. The
registry ceiling defaults to `ALL`, and the policy ceiling is never consulted on this path. V6's census lists the site
but bounds nothing, and the sub-sim child (finding 13) runs this loop. **Fold:** `_admit` refuses `auto_fire` on
anything outside `READ`, and the check reads the class's own declaration, not the instance's.

### 7. SHOULD-FIX (fail-open): the registry ceiling and the policy ceiling are unrelated, and the registry's is mutable
§2 puts a ceiling on the registry, and §3 puts a second one on `ToolPolicy`. Nothing ties the two together. The direct
dispatch sites (V6, finding 6) see only the registry's ceiling, which defaults to `ALL`. `self.ceiling` is also a
plain attribute (§2 "checks `effect_of(tool) <= self.ceiling`"), and #1431's "switchable" Talk ceiling invites a
setter. Lowering a ceiling does not evict a tool already admitted, and `activate_scene` re-enables tools without
re-checking them. **Fold:**
- `Executor.__init__` refuses a registry whose ceiling is wider than the policy's, or `build_executor` sets it from the
  policy.
- The registry ceiling is set at construction only and exposed as a read-only property.
- Talk toggles narrow at dispatch, through a narrower-only filter, and never raise a ceiling.

### 8. SHOULD-FIX (fail-open for future code): "one admit path" is not structural
- `simulation/tools.py::SimToolRegistry` (L156) is **not** a `ToolRegistry` subclass. It has its own `_tools` dict and
  no lock, so "`SimToolRegistry.register` calls `_admit`" means a second copy of the check.
- `loop_perception.py` explicitly supports "out-of-tree ToolRegistry subclasses", and a subclass that overrides
  `register` skips `_admit`.
- `Executor` accepts any duck-typed registry.
- The proposed grep for `_tools[` assignments misses `.update`, `.setdefault` and rebinding (`self._tools = {...}`).

**Fold:** one module-level `admit(tool, ceiling)` that both registries call. Name the executor's dispatch check as the
authority and the registry check as early refusal. Replace the grep with a behavioural sweep: every write method of
every registry class, given an `ALL` tool under a READ ceiling, must raise.

### 9. SHOULD-FIX (fail-closed, leaves state behind): entity generation registers partway, then raises
`embodiment/tool_bridge.py::generate_tools_for_entity` registers as it goes: sense tools at L1067 and L1079, then
affordances at L1101. Under a READ ceiling the sense tools land and the first `ModulatorAffordanceTool` raises
`ToolEffectRefused`, so the entity ends up half-registered. `simulation/dm_runtime.py::_register_entity_tools` only
records `_registered_tools[name]` on success, so the tools that did land are never deregistered when the entity
leaves. **Fold:** pre-check every generated tool before registering any of them, or refuse per tool and return the
refused list. Record what was actually registered either way.

### 10. SHOULD-FIX (fail-open where authority is keyed by name): replacing a same-name tool stays legal under `ALL`
`api.py::_inject_registered_tools` runs after the built-ins, and its own docstring says "last-wins". So
`register_tool(X)` with `X.name == "think"` replaces `ThinkTool` in every run. Several authorities are keyed by name:
`ToolPolicy.only(names)`, the mode tool lists, and `ALWAYS_ALLOWED_TOOLS`. The `kind:` selectors can be spoofed
through `register(kind=)` (`object.__setattr__`). With a policy whose ceiling is `ALL`, the allow-list then admits the
impostor. **Fold:**
- Refuse every same-name replacement unless the caller passes `replace=True`; `_CampaignToolLease.restore` uses it.
- `ToolPolicy.only` takes a required, explicit `ceiling` with no default.
- The docs say a name allow-list is bounded only by its ceiling.

### 11. SHOULD-FIX (fail-open): the `ToolPolicy` constructors leak, and a deny-only policy has no form
- `ToolPolicy(permissions=..., ceiling=Effect.ALL, reason=...)` and `dataclasses.replace(policy, ceiling=Effect.ALL)`
  produce a policy equivalent to `unrestricted` while bypassing both the `unrestricted(` census and `narrowed`.
  **Fold:** require a private token in `__post_init__`, and have the census also flag `ToolPolicy(` and
  `replace(` on a policy.
- `agents/permissions.py::tool_permissions_from_settings` (L40–58) returns a **deny-only** `AgentPermissions`
  (`allow=None`, `deny={...}`) from `tools.deny` / `MAXIM_TOOLS_DENY`. The note says it "becomes
  `ToolPolicy.only(...)`", but `only` needs names. The only other constructor is `unrestricted`, which carries no
  permissions, so the console's working deny list would be dropped. That is a regression of a gate that exists today.
  **Fold:** add `ToolPolicy.from_permissions(perms, *, ceiling, reason)`, or give `unrestricted` a `deny=` argument.
  Add a test showing that a console deny of `bash` still refuses `bash`.

### 12. SHOULD-FIX: V3's soft landing makes the O1 red gate vacuous and adds an unlisted allow-all
- `build_executor(policy=None)` raises, while `AgentConfig` with `None` only warns. So `create_full_agent` must
  translate `None` into `ToolPolicy.unrestricted(...)`. Under that translation the O1 strict xfail ("`maxim.run`'s
  executor has a non-`None` policy") flips green on a rename, with behaviour unchanged. The translation site is also
  an `unrestricted` caller missing from the pinned list.
- `api.py::run` builds `AgentConfig` without permissions (L725–733). Its `learning=False` branch (L753) has no
  `AgentConfig` at all, so "`api.run` takes the policy from `AgentConfig`" is wrong for that branch.
- The note names the field `AgentConfig(permissions=None)` in V3 and `AgentConfig.policy` in §3. If both fields exist,
  which wins is undefined.
- "Raises at 1.4.0" is held only by attention. It needs a version-keyed strict test.
- Checked and not reopened: the observer, because §4.2 builds its executor without `AgentConfig`.

**Fold:**
- The O1 gate asserts the policy's identity (reason and ceiling), not that it is non-`None`.
- Pin the translation site and its reason in the census.
- Write one sentence on what `learning=False` passes.
- One field name.
- A strict xfail keyed on `__version__ >= 1.4`.

### 13. SHOULD-FIX (fail-open): the sub-sim's "parent policy" is the wrong object, and the child loop escapes it
`SpawnSubSimulationTool` is a narrator tool (`orchestrator.py` L1366, `SimToolRegistry()`). The policy given to it at
registration is independent of the executor that later dispatches it, so the two can drift. The narrator is likely to
be `unrestricted`, which makes `narrowed()` vacuous. The child registry comes from `build_tool_registry(...,
operational_mode="active")` (`simulation/tools.py` L1498) with no ceiling, and the child runs the full
`run_agentic_loop`, including auto_sense (finding 6). **Fold:**
- Build the spawn tool and its executor in one builder, from one policy.
- The child's registry ceiling is the child policy's ceiling.
- Name which parent counts: the narrator's policy, or the AUT's.

### 14. SHOULD-FIX (fail-closed, behaviour regression): a frozen alias snapshot breaks DM encounters
`simulation/dm_runtime.py` L317 calls `self._executor.register_aliases(...)` **after** construction for each
encounter's choices, and calls `remove_aliases` later. A snapshot taken at construction loses them, so choosing a
choice by name stops resolving to `choose`. Aliases are also not load-bearing for security: the ceiling check runs on
the resolved object, and the allow-list runs on the canonical name. **Fold:** give each executor its own alias table,
copied at construction and changed by `register_aliases` on that executor only. Drop the security claim from the
alias row.

### 15. SHOULD-FIX (fail-open for future code): the AST census has blind spots
- Direct-dispatch receivers have arbitrary names: `_af_tool.execute()` (`loop_perception.py` L203), `eph_tool.execute`
  (`simulation/tools.py` L864), `tool.execute` (`integration/bio_enrichment.py` L508/L512), and
  `tools[action].execute()` (a subscript receiver) at nine sites in `scripts/orient_substrate/*.py`, which the census
  walks. A matcher can also be evaded with `fn = tool.run; fn(...)`, `functools.partial(tool.execute, ...)`,
  `pool.submit(tool.run, ...)` or `getattr(tool, "execute")`.
- The `policy=` check gets `**kwargs` splats wrong in both directions. The runtime `TypeError` is the real gate there.
- `unrestricted(` is evaded by a method alias, a non-literal `reason`, or a direct `ToolPolicy(...)` (finding 11).

**Fold:** keep the AST census as a list, and back it at runtime. `Executor.execute` sets a contextvar.
`Tool.__init_subclass__` wraps each class's own `execute`, logging a structured event (raising under the test flag)
when it is called outside a dispatch context, unless the call site holds a named V6 exemption.

### 16. NIT (fail-closed): the prompt roster must apply the ceiling too
`Executor.permits` / `_runnable_tools` → `_permission_denial` is name-based. If the ceiling is checked only in
`execute`, the model is offered tools that can only ever be refused, which D82 forbids. **Fold:** add the ceiling to
`_permission_denial`, resolving the object once.

### 17. NIT (fail-closed, misleading error): `SimToolRegistry.get` should raise `KeyError`, not return `None`
`Executor._run_started` catches only `KeyError` around `registry.get`. Given `None`, it reaches `_outcome_start(None,
...)` and `None.run`, an `AttributeError` reported as "execution failed". **Fold:** raise `KeyError`. Separately,
`_FallbackRedirectTool._registered_tools` is class-level state shared by every `SimToolRegistry`.

### 18. NIT: `ConcurrentExecutor` needs no policy of its own
`runtime/concurrent_executor.py::ConcurrentExecutor.__init__(executor: Any, ...)` wraps an `Executor`. Giving it "the
required argument" would create a second policy. (The claim that it has no construction site is correct; there is
none in `src/`, `scripts/` or `tests/`.) **Fold:** type its argument as `Executor` and check it with `isinstance`.

### 19. NIT: `narrowed()` and the semantics of the permissions it carries
- Literal set intersection of allow-lists that mix `kind:` selectors and names drops entries, which fails closed.
- `clearance` and `sem_access_rules` have no narrowing rule, and `AgentPermissions.can_access_sem` has **zero callers**
  in `src/`. The SEM half of the policy is decorative, so do not claim the policy enforces it.
- A ceiling built by complement (`~Effect.WRITE_LOCAL`) silently absorbs every flag added later. Forbid it.

### 20. NIT: factory classes and the declaration census
- `tools/reachy_stubs.py::create_stub_tool` assigns class attributes after creation, and `api.py::tool` builds a
  nested `FunctionTool`. The effect must go into the class-body namespace (a closure variable); assigning it after
  creation trips the mismatch rule.
- A subclass walk sees the module-level `StubTool`s, but a `FunctionTool` exists only after decoration. The census
  must call the factories.
- Skip abstract classes (`inspect.isabstract`).
- The own-`__dict__` rule breaks no intermediate today: every in-tree tool subclasses `Tool` directly.
- I count 83 concrete subclasses plus the per-call factory classes, not 85.

### 21. NIT (fail-closed): a stale positional call can bind silently
After `permissions` is removed, `Executor(reg, pd, bridge, X, policy=p)` binds `X` to `embodiment` without any error.
**Fold:** make every parameter after `tool_registry` keyword-only.

## Factual claims checked
- Correct: `ToolRegistry.register` silently replaces a same-name tool. `forbidden_categories` reads `action["category"]`
  (`agents/autonomy.py` L307). `ALWAYS_ALLOWED_TOOLS` lists `move` as "no side effects". `_mode_denial` returns `None`
  with no mode source and no grant. `find` and `get` are separate lookups (`resolve_tool_name` / `_permission_denial`
  vs `_run_started`). Six `build_executor` sites pass a literal `None`, and `create_full_agent` passes
  `config.permissions` (default `None`), which makes seven. The six direct `ToolRegistry(` / `SimToolRegistry(` sites
  in `src/` and the two in `scripts/` match the note. `ConcurrentExecutor` has no construction site.
- Wrong or incomplete: "one admit path" (finding 8); the `api.run` plumbing (finding 12); the console mapping
  (finding 11); `ProposeActionTool` as READ (finding 5); the alias snapshot as a security fix (finding 14); the V6
  census list, which omits the nine `scripts/orient_substrate` sites (finding 15).

**Verdict:** SOUND WITH CONDITIONS: no DO-NOT-BUILD, but fold findings 1, 2, 4, 6, 7, 11 and 13 (fail-open) and 12 (vacuous red gate) before code, and give the folded note another delta-scoped round.
