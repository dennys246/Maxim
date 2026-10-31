# #1315 item 2 (`Tool.effect` + registry ceiling): Architecture lens

Diff: `git diff origin/main...HEAD` at e37f2ea1 (.worktrees/v-effect).

## Verdict: SOUND WITH FOLDS. No blockers. 3 SHOULD-FIX, 8 NITs.

None of the findings needs a new owner decision. S1 asks the owner to confirm, in one line, a default the build already took (the robot link counts as `ACTUATE`). N3 needs an owner decision only if the fix chosen is a top-level `maxim.Effect`.

### What checks out
- **Layering.** `tests/unit/test_architecture_audit.py` passes. That is 77 tests, run together with `test_tool_effect.py`. The new edges are:
  - `tools/registry → maxim.exceptions`, a root module that is in no layer;
  - `simulation/tools → tools.registry.EffectAdmission`;
  - `runtime/executor → exceptions` and `tools.registry`.

  None of them is in `LAYER_RULES`.
- **`ToolEffectRefused` in `maxim/exceptions.py` is the right home.** It sits beside the `ToolExecutionError` family and is a `MaximError`. It is deliberately not a `ToolExecutionError`: a refused registration is not a failed execution. `exceptions.py` imports only `pathlib` and `typing`, so the step-0a import-closure gate does not grow. Keeping it out of `tools/base` also keeps `base` free of exception plumbing.
- **The `EffectAdmission` mixin is the right shape for §2.2's "shared mixin".** It holds the state (`_ceiling`, which defaults to ALL), a read-only `ceiling` property and one `_admit`. Both registries inherit it. The test that classifies `_REGISTRY_READERS`/`_REGISTRY_WRITERS` was updated for `ceiling` and `register_many`.
- **The item 3 seam is clean.** Item 3 needs `effect_of`, `exceeds` and a registry `find()`, and all three exist. Item 2 adds nothing to `Executor._permission_denial` or `permits()`, and the policy-vs-registry ceiling split matches §3: a child registry's ceiling equals the child policy's. Nothing here pre-empts `ToolPolicy`, `check()` or the typed `ToolErrorKind`s.
- **§2.2 bullets.** Built: the declaration, the record (WeakKeyDictionary), `effect_of`, mutation visibility (own `__dict__` only), `exceeds`, a fixed ceiling, and `_admit` on `__init__(tools=)`, `register`, `_register_scene_locked` and `SimToolRegistry.register`. The replacement rule (refused below ALL, warned under ALL) is in. Entity generation is all-or-nothing, and the EntityMap is registered after it. `SimToolRegistry.get` raises `KeyError` below ALL. The refusal sites log in three places: api inject, dm_runtime, and Executor Mechanism-B. The declaration test and the sink test are both un-xfailed. The literal-name helper test and the two factories are covered. Only the roster bullet is not built (see N1).
- **V11–V14 are applied consistently.** I checked all 86 classes with an AST dump.
  - V11: `sense_presence` (READ|SELF_SCOPE), `design_entity`, `check_completion` and `analyze_results` (READ) carry no NETWORK.
  - V12: `send_message`, `extend_simulation` and `inject_pain` are ALL, and so are `actor` and `spawn_sub_simulation`.
  - V13: `focus_interests` is ALL, both the real tool and the stub.
  - V14: every stub, the factory-built and the hand-written ones (`NoOpMaximCommandTool`, `NoOpNoveltyTrackTool`), equals its real tool. `create_stub_tool(effect=)` is required keyword-only.
  - The §2.1 worked examples (bash, execute_file, http_fetch, respond, ModulatorAffordanceTool, the mode tools) match. No class contradicts a stated rule. The one wording conflict is S1.
- **The invariant text in `docs/agents/runtime-tools.md`** is accurate against the code. It correctly narrows §1A.9's "checked at one dispatch seam" to "checked at admission; dispatch is item 3's". Its `Regression guard:` line cites `file::symbol` entries that resolve, and `scripts/lint_claude_md_invariants.py` passes.
- **The changelog fragment** matches the code claim for claim. `changelog_fragments.py check` is clean. It honestly says no built-in path builds a narrower registry yet.
- **Rule T.** `scripts/lint_function_length.py` reports clean. The one pinned function near the diff is `ModulatorAffordanceTool.execute`. The edit there is a class-body line, not inside `execute`.
- **The `tool_replaced` WARNING is quiet on shipped paths.** Building `build_tool_registry` in both passive and active modes emitted none. The orchestrator re-registers the same class (respond/speak), and `restore()` deregisters before it registers. Neither fires the warning.

## SHOULD-FIX

**S1 (doc/definition; fail-open in wording, not in code): the NETWORK definition now contradicts two classifications.**
- Where: `tools/base.py::Effect` docstring ("`NETWORK`: any egress off the machine, DNS and GETs included") and the §2.1 table row.
- The contradictions:
  - V11 exempts the LLM router.
  - The "robot link is ACTUATE, not NETWORK" default, taken without asking, exempts the Reachy and bridge links. A wireless Reachy is literal off-machine egress.
- Why it matters: a future classifier reads the definition, not the as-built block, and would classify the next robot or LLM tool differently.
- Fix: amend the definition in both places to name the infrastructure carve-outs. For example: "egress a tool chooses or causes; the LLM router (V11) and an attached body's own link are infrastructure, governed by the lanes and the offline latch".
- Owner: the robot-link carve-out was not asked and contradicts the written text. Ask for a one-line confirmation, not a new V-row.

**S2 (fail-open, latent): "refusals are never silent" is a rule with no check, and existing broad handlers would catch one generically.**
- Where: the runtime-tools.md invariant ("a site that catches one logs it with `log_effect_refusal`, never silently"), the three sites, and `tools/registry.py::log_effect_refusal`.
- Why: only the three listed sites use `log_effect_refusal`. Other `register(...)` calls sit inside `except Exception` handlers that log a generic message, without the `tool_effect_refused` event. One example is `simulation/orchestrator.py`'s respond/speak lease onto the adopted registry.
- It is latent today: under ALL, only an effect mismatch can raise. It becomes live the day item 1 or item 3 builds a narrower registry.
- CLAUDE.md "Enforced, or on the backlog" applies. A rule with no check, the same as "read only through `effect_of`", needs a check or a backlog row.
- Fix, preferred because it is structural: emit the structured `tool_effect_refused` WARNING once, in `EffectAdmission._refusal` or `_admit`, at raise time. Catchers then cannot forget, and `log_effect_refusal` shrinks to a site tag or disappears. This follows "push silent-no-op invariants into types".
- Fix, alternative: an AST test that every `except ToolEffectRefused` body calls `log_effect_refusal`, plus a mechanization-backlog row for the broad-handler case.

**S3 (doc accuracy, stable API): `docs/user/stable_api.md` describes behaviour this release does not have.**
- What it says: "Only a narrower agent (offline, the observer, Talk) drops an undeclared tool, with a WARNING".
- What is true in this release:
  - No observer or Talk path exists.
  - Offline narrows at dispatch: V7 ANDs the ceiling in item 3's seam (§1A.2), and the registry never drops the tool. So the offline outcome will be a typed dispatch refusal, not a drop at registration.
- Fix: state what is true now ("`run()` runs it as before; no shipped path narrows the tool set yet"), and add the narrower-agent sentence when item 1 or item 3 lands. Or word it per mechanism.

## NITs

- **N1. Record that the roster bullet is not built.** §2.2's bullet "the prompt roster (`permits`) applies the ceiling too" is item 3's (the §1A.2 seam), and the as-built block does not say so. Add a line: a registry ceiling is already reflected, because refused tools never enter the registry. In the same block, note that §2.1 still names `tools/effect_exemptions.json`, while the build, and §1A.3, use `tests/fixtures/effect_exemptions.json`.
- **N2. Some of this is capability without a caller, and that is correctly disclosed.** `ToolRegistry(ceiling=)`, `SimToolRegistry(ceiling=)` and the below-ALL `KeyError` branch have no non-test caller in `src/` or `scripts/`. The red gates flip because the tests build narrower registries by hand. `declare_effect` (caller: `api.register_tool`), `register_many` (caller: `generate_tools_for_entity`) and `effect_names` (caller: the registry) do have callers. Keep the "Ten item-2 red gates flipped" sentence paired with "the composition gate `test_o2_o5_...` stays red", so nobody reads item 2 as closing the atonia defect.
- **N3. `Effect` is a parameter type of two stable verbs, but no public surface lists it.** It is not in `maxim/tools/__init__.py`'s `_LAZY` table or `__all__`, and not in stable_api.md's tables. Users must import it from `maxim.tools.base`. Fix: add `Effect` to `tools/_LAZY` and list it in the stable_api experimental table. A top-level `maxim.Effect` would be an owner call.
- **N4. Deregistering, then registering, gets around the "below ALL, any replacement refuses" rule.** The effect bound still holds, because the new tool is admitted against the ceiling. Identity, the reason the rule exists, is item 3's pinned policy. Add a row to §2.2's inputs table so this is a stated residual.
- **N5. `ToolRegistry.register_many` writes `self._tools` directly.**
  - It bypasses `register()`'s body, including the `kind=` override path, so a later change to `register` can diverge from it. Consider one `_store_locked(tool)` that both use.
  - `_admit_all` uses `staged.get(n) or self._tools.get(n)`, which relies on Tool truthiness. Use an explicit `is None` check.
- **N6. A design consequence to carry into Talk (#1431) and item 1.** `ModulatorAffordanceTool` is ALL and entity generation is all-or-nothing. So under any below-ALL registry, an entity with even one affordance yields zero tools, including its READ `sense_*` tools. It is as specified, and fail-closed. It may surprise the Talk ceiling design. Note it in §2.2.
- **N7. Two tools over-declare.** `damage_component` and `set_entity_sensor` declare `SELF_SCOPE`, but they mutate the in-memory sim body, which by the READ definition "changes nothing outside process memory". That is fail-closed and harmless. If it is deliberate (pain flows into persisted memory), say so in a comment, the way the ALL classes do.
- **N8. One `register_tool` combination always fails.** `register_tool(fn_tool, effect=X)` on a `@maxim.tool`-built tool always raises `TypeError`, because `FunctionTool` declares its own ALL. That is correct, since a declaration is never rewritten, but one docstring sentence in `register_tool` would save a user the surprise.
