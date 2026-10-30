# Wire-integrity review: #1315 item 2 (`Tool.effect` + registry ceiling)

Worktree `.worktrees/v-effect`, commit e37f2ea1, diff `origin/main...HEAD`. Read-only review. Method: docs/CODE_REVIEW.md
"The wire-integrity lens". Evidence used: AST scans, greps over src/ scripts/ tests/ docs/, child-python repros
(`review_wire/repro.py`, `probe_names.py`, `scan_undeclared.py`, `scan_catch.py`), and two narrow pytest batches with a
logging plugin (`review_wire/replplug.py`) that recorded every `tool_replaced` / `tool_effect_refused` record per test:
2122 passed (90 tool/registry/api/executor/bootstrap/scene/entity/foundry/dm/orchestrator/sim/embodiment/reachy files)
plus 278 passed / 63 xfailed (the other generate_tools_for_entity users, test_exceptions, test_loop_setup,
integration/test_advisor_gate).

## Verdict

**SOUND WITH CONDITIONS.** No BLOCKER. The default ceiling is `ALL` at every production construction site, so no
production path can raise `ToolEffectRefused` today, and no production-shaped path in 2,400 tests logged a
`tool_replaced` WARNING (the only one came from test_tool_effect.py's own probe). Two SHOULD-FIX items: a stable-API
compatibility break in `Tool.__init_subclass__`, and a brief invariant ("a site that catches one logs it ... never
silently") that 19 generic `except Exception` registration sites outside the diff do not honour. Everything else is a
NIT.

## Producer → consumer table

| Contract | Producers (before → after) | Consumers (before → after) | What each consumer gets, before → after | Probe |
|---|---|---|---|---|
| Call: `ToolRegistry(ceiling=)` / `SimToolRegistry(*, ceiling=)` | none → constructor kwarg, default `ALL`. **Production sites passing `ceiling`: zero** (grep `ceiling=` over src/scripts). Construction sites: `bootstrap.build_tool_registry`, `tool_bridge.derive_capability_map`, `foundry` ×2, `orchestrator` (`SimToolRegistry()` for orch), `scripts/r2_drive_premise_probe.py`, `scripts/orient_backbone/exp53_*` | `EffectAdmission._admit` (every in-tree write); `SimToolRegistry.get`; `.ceiling` property read only by tests | unchanged at `ALL`; below ALL only in tests (`test_tool_effect.py`, `test_atonia_red_gates_effect_policy.py`) | 1 (capability-only today, stated in changelog), 2, 6 |
| Call: `ToolRegistry.register` / `register_scene_tools` raise `ToolEffectRefused` | never raised → raises on exceed, class-effect tamper, or any replacement below ALL | ~40 `registry.register(...)` sites in bootstrap, orchestrator, campaign_runner, agentic_runtime, foundry, api, lease `restore`; 3 sites catch it by name (api `_inject_registered_tools`, executor `_handle_entity_acquisition`, dm_runtime `_register_entity_tools`); 19 sites catch it under `except Exception` (scan_catch.py, listed in F2); unguarded sites (bootstrap stubs, filesystem/git/display tools, orchestrator orch_registry) propagate | unreachable under ALL → under a future ceiling: 3 sites log `tool_effect_refused`; 19 sites drop the tool at debug/warning/none with a generic message; unguarded sites abort registry build | 3, 4, 6 |
| Payload/log: `tool_replaced` WARNING (event, data `{tool,new,old}`) | none → `_admit` under ALL when `type(existing) is not type(tool)` | `_DisplayLoggingHandler` (sim display, WARNING+), MAXIM_LOG_FILE `StructuredFormatter`, any caplog test | Production-shaped replacement paths found: **only a user `register_tool` that shadows a built-in name** (repro: one WARNING per `run()` build). No built-in name collision across bootstrap vs orchestrator/narrative/introspection/response/discovery/tools_dm/sim tools (probe_names.py). Bootstrap stubs and real Reachy tools are mutually exclusive branches. Scene re-registration in orchestrator re-registers the SAME instances (silent). Lease `restore` re-registers baseline instances (warns only if a campaign swapped a class; correct). | 2, 9 |
| Payload/log: `tool_effect_refused` (via `log_effect_refusal`) | none → 3 catch sites | StructuredFormatter (serialises only `event`+`data`) | data has site/tool/effect/ceiling but **not the reason** (exceeds vs tamper vs replacement) | 7 |
| Call: `generate_tools_for_entity` registers via one `register_many` at the end; `entity_map.register(entity)` after | per-tool `registry.register` interleaved; `entity_map.register` first → batch at end; map after | Callers: `Executor.generate_entity_tools` (build_executor initial + Mechanism-B acquisition), `generative_runner._load_world_entities` / `_activate_phase_entities`, `dm_runtime.SceneState._register_entity_tools`, `foundry` T4 + gauntlet, `derive_capability_map`, scripts r2_drive_premise_probe + exp53. Every production registry passed is a `ToolRegistry` (aut_registry from `build_tool_registry` or `_adopted.registry`; dm/campaign pass aut_registry). The orch executor's `SimToolRegistry` never reaches it (orch AgentConfig has no `embodiment_ref`; acquisition needs `_entity_map`+`embodiment`, both None). Between tool construction and registration nothing reads the registry or map (`ModulatorAffordanceTool.__init__` only stores `entity_map`; names resolved against a local `existing` set). | identical result on success. On ANY exception mid-walk (incl. pre-existing `_resolve_tool_name` ValueError): before = partial tools + entity in map; after = nothing registered, entity not in map | 2, 3, 10 |
| Call: registry duck-type required by `generate_tools_for_entity` | `register` + `list_all` → `register_many` + `list_all` | `SimToolRegistry` has `register`/`list_all` but no `register_many` | before works; after `AttributeError` (repro). No production caller passes one today | 3 |
| Call: `create_stub_tool(..., *, effect)` required | optional → required kw | 4 calls in reachy_stubs.py, all pass it; zero callers in scripts/tests/docs | n/a | 3 |
| Call: `api.tool(fn=None, *, effect=None)`, `register_tool(tool, *, effect=None)` | positional-only `tool(fn)`, `register_tool(tool)` → optional effect | `maxim.__init__` `_API_VERBS`/`__all__` (unchanged names); docs/user/python-api.md, extension_api.md (bare `@maxim.tool` and `register_tool(X())` both still valid); no .pyi stubs; no scripts callers | bare forms unchanged; `@maxim.tool` without effect now writes `effect = Effect.ALL` into FunctionTool's own dict (declared ALL, not undeclared) | 3 |
| Vocabulary: `Effect` flags + `effect_names` serialised form | none → 7 flags, `ALL` | registry, `ToolEffectRefused.effect/ceiling`, exemptions fixture, tests | not persisted anywhere (no pickle/deepcopy of tools or classes in src; `copy.deepcopy` in base.py only copies schemas) | 5 |
| Gate: `Tool.__init_subclass__` record (`_EFFECT_RECORD`) | none → every Tool subclass at class creation; `declare_effect` (register_tool) | `effect_of`, `effect_mismatch` in `_admit` | AST scan: **every** Tool subclass in src/ and scripts/ declares an own `effect` (0 undeclared); no `type(...)`-built Tool classes in src/scripts; `create_stub_tool` and `api.tool` put `effect` in the class body. Third-party subclasses with a non-Effect `effect` attribute now fail at class creation (F1) | 6 |
| Lazy exports: `maxim.tools._LAZY` / `__all__`, `maxim/__init__` | — → no `Effect`/`ToolEffectRefused` export | users import `maxim.tools.base.Effect` (as documented in stable_api.md) | no change; acceptable (NIT F9) | 1 |
| `maxim.exceptions.__all__` + `ToolEffectRefused(MaximError)` | — → added | `test_exceptions.py` (passes), registry, api, executor, dm_runtime | not a `ToolError` subclass, so `except ToolError` handlers do not see it (none wrap registration) | 3 |
| Claims: changelog fragment, runtime-tools.md invariant, atonia notes §2 BUILT block, stable_api.md | — | the code | see F2 (brief "never silently"), F8 (exemption-file path), F9 (extension_api.md) | claims |

## Findings

### SHOULD-FIX

**F1. A third-party `Tool` subclass that has any non-`Effect` attribute named `effect` now raises `TypeError` at
class creation: a break of the stable 1.0 `Tool` ABC.** Fail-closed (loud), but breaks user code on upgrade.
- Where: `src/maxim/tools/base.py::Tool.__init_subclass__`.
- Repro (child python): `class PotionTool(Tool): name="drink_potion"; ...; effect = "heal"` →
  `TypeError: __main__.PotionTool.effect must be a non-empty maxim.tools.base.Effect, got 'heal'`. The same happens
  for `@property def effect(self)` or a method named `effect`.
- Contract: docs/user/extension_api.md §2 says the `Tool` ABC is "part of the 1.0 contract", and docs/user/stable_api.md
  rule 4 treats a new requirement on third-party tool authors as breaking. The changelog fragment mentions the
  TypeError, but as an addition, not a compatibility change.
- Fix (owner decision): either (a) state it as a named 1.x compatibility exception in stable_api.md and the changelog,
  with the rationale (an own non-Effect `effect` could otherwise slip a tool past a ceiling); or (b) for a value that
  is not an `Effect` at all, record `ALL` with one WARNING naming the class, and teach `effect_mismatch` to ignore a
  non-Effect own value whose record is `ALL`. Keep `Effect(0)` as a TypeError. Under (b) the class is still `ALL`, so
  nothing can pass a ceiling. (b) is just as fail-closed and breaks no one, so it is the strict and the compatible
  option at once.

**F2. The new brief invariant says a site that catches `ToolEffectRefused` "logs it with `log_effect_refusal`, never
silently", but 19 registration sites outside the diff catch it under `except Exception`.** Fail-closed (the tool is
dropped), silent or generic. Unreachable today (no ceiling), but it is the composition item 3 will turn on.
- Where (scan_catch.py, try-blocks that wrap a registry write and have a generic handler with no
  `ToolEffectRefused` clause):
  - `runtime/bootstrap.py::build_tool_registry`: 7 blocks (Reachy, mode switch, sleep, internet, sandbox, response,
    comms) logging a generic WARNING.
  - `simulation/orchestrator.py`: 959, 1091 (8 introspection tools in one try, DEBUG), 1155 (narrative, DEBUG), 1475
    (experiment tools, DEBUG), 887 (respond lease, WARNING).
  - `simulation/campaign_runner.py`: 108 (ERROR).
  - `simulation/foundry.py`: 484 (appends to errors), 562, 1283.
  - `embodied_runtime/agentic_runtime.py`: 663, 683, 716 (no log call in the handler).
- Also: a batch try (orchestrator 1091, agentic_runtime 716) drops every LATER tool in the block when one is refused.
- Fix: narrow the brief sentence to "the sites that expect a refusal (api injection, entity acquisition, dm scene)",
  and record the 19 sites as an item-3 prerequisite (or a mechanization-backlog row: an AST lint that a try around a
  registry write with an `Exception` handler has a `ToolEffectRefused` clause before it). Item 3, not this PR, should
  add the clauses, since no registry below ALL exists yet. The claim has to match the code now.

### NIT

**F3. A user tool that shadows a built-in name logs a `tool_replaced` WARNING on every `run()`.** Fail-closed
(diagnostic only).
- Where: `tools/registry.py::EffectAdmission._admit` with `api.py::_inject_registered_tools`.
- Repro: `register_tool(MyRead())` with name `read_file`, then two registry builds give two
  `tool 'read_file': MyRead replaces ReadFileTool` WARNINGs. In a sim they reach the display through
  `sim_logger._DisplayLoggingHandler`.
- Replacing a built-in is documented ("same name replaces"), and the warning is arguably the point. No built-in
  collision exists (probe_names.py), and orchestrator plus api double-injection hand over the same instance, so it
  stays silent.
- Fix: keep it, and add one line to extension_api.md §2 saying a shadowing replacement logs a WARNING. Optionally
  dedupe it per (name, new class) per process.

**F4. The `tool_replaced` message is ambiguous when two classes share a qualname.** Every `@maxim.tool` class is
`tool.<locals>.FunctionTool`, and every `create_stub_tool` class is `create_stub_tool.<locals>.StubTool`.
- Captured in the run: `{"new": "abc.T_dup", "old": "abc.T_dup"}`.
- Fix: include the tool name of the old class, or `id()`/`__qualname__` plus the defining function's name.

**F5. The `tool_effect_refused` JSONL loses the reason.** `StructuredFormatter` serialises only `event`+`data`, and
`data` has no reason field, so exceeds, tamper and replacement look the same in `MAXIM_LOG_FILE`.
- Repro output: `{"e":"tool_effect_refused","site":"probe","tool":"read_file","effect":["READ"],"ceiling":["READ"]}`.
  That record was a replacement refusal, but nothing in it says so.
- Fix: add a `reason` kind (`exceeds` | `reassigned` | `replacement`) to `ToolEffectRefused` and to the `data` that
  `log_effect_refusal` writes.

**F6. `generate_tools_for_entity` now needs `register_many`, and `SimToolRegistry` lacks it.**
- Repro: `generate_tools_for_entity(rusty_sword, SimToolRegistry())` gives
  `AttributeError: 'SimToolRegistry' object has no attribute 'register_many'`. It worked before the change.
- No production caller passes one today. If one ever did through dm_runtime, the `except Exception: log.debug` there
  would hide it.
- Fix: move `register_many` (batch admit plus write) onto `EffectAdmission`, or give `SimToolRegistry` one, so both
  registries share the batch contract.

**F7. `register_tool(tool, effect=)` changes the class before it validates the tool.** `declare_effect` runs before
`_record_registration`'s name check.
- Repro: a nameless instance with `effect=Effect.READ` is rejected with a TypeError, yet the class keeps
  `effect = READ` and its record afterwards.
- Fix: validate the name first (call `_record_registration`'s check, or move `declare_effect` after it).

**F8. Claim drift on the exemption list's path.** atonia notes §2.1 still says `tools/effect_exemptions.json`, but the
file is `tests/fixtures/effect_exemptions.json` (the brief has the right path). The BUILT block does not record the
move.
- Fix: add one BUILT-block line, or correct §2.1.

**F9. docs/user/extension_api.md §2, the stable custom-tools contract, is not updated.** It has no `effect`, no
`@maxim.tool(effect=...)` and no ALL default. stable_api.md is updated. `Effect` is also not exported from
`maxim.tools` (`_LAZY`) or from `maxim`; users import it from `maxim.tools.base`, which is consistent with the docs.
- Fix: add a short paragraph to extension_api.md §2 (it is the page third-party authors read), and mention F1 there.

**F10. All-or-nothing now applies to every exception, and Mechanism-B acquisition is not atomic.**
- `generate_tools_for_entity` now registers nothing, and skips `entity_map.register`, on any exception. That includes
  the pre-existing `_resolve_tool_name` ValueError, where before it registered some tools and the entity was already
  in the map. This is an improvement, but the changelog presents it as ceiling-only.
- In `executor.py::_handle_entity_acquisition` the entity is reparented and `transfer_to_self` runs BEFORE
  generation. A refusal there leaves an entity owned by the body with no tools. Unreachable under ALL.
- Fix: one changelog clause ("any failure registers none"). Note the acquisition ordering for item 3.

### Checked and clean

- No src/ or scripts/ path builds a registry below ALL.
- Every Tool subclass in src/ and scripts/ declares an own `effect` (AST closure).
- No `type()`-built Tool classes; no pickle or deepcopy of tools or classes.
- No other write path bypasses `_admit` (only the `_tools[...]` writes in register, register_many and
  `_register_scene_locked`).
- `ToolRegistry.__init__(tools=)` sets the ceiling before it registers.
- `create_stub_tool` has no outside callers.
- `@maxim.tool` and `@maxim.tool(effect=...)` both work, and bare `register_tool(X())` is unchanged.
- The orchestrator characterization golden and the `executor_import_closure_step0a` gate both pass.
- `SimToolRegistry.get` raising KeyError below ALL reaches `Executor.execute`'s existing KeyError handling
  (`find_similar` is present).
