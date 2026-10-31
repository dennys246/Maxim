# #1315 item 2: delta review of the fold (`git diff e37f2ea1 5fb640ac`)

Reviewer: delta (a reader who was not one of the three lenses). Covers Executor, Architecture and Wire on the delta only.
Probes: `/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/248fdc0d-2f2e-471a-b37d-dacb96a7ecce/scratchpad/review_delta/probe1.py`
and `/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/248fdc0d-2f2e-471a-b37d-dacb96a7ecce/scratchpad/review_delta/probe2.py`.
Narrow tests: `test_tool_effect.py`, `test_atonia_red_gates_effect_policy.py`, `test_register_tool_persistence.py`,
`test_dm_runtime.py` and `test_mode_switch_gate.py` give 83 passed and 14 xfailed. `ruff check` is clean on the touched files.

## Verdict: SOUND WITH ONE CONDITION

Every SHOULD-FIX from the three reviews is folded or explicitly deferred, and every fold does what it claims. The
metaclass, V15, logging at raise and `SimToolRegistry.register_many` add no new defect on 3.10, 3.12, 3.13 or 3.14.
One fail-open finding remains (D1). It is older than the fold and is in the same in-process mutation class as
executor S1, but it makes the fold's new claims false: "a reassignment never narrows a tool, and never leaves a stale
narrower record" (`effect_of`), and "a class whose value no longer matches its record is refused" (`declare_effect`).
The fix is two lines and needs a delta-scoped round.

## Finding → fold map

| Review finding | Fold | Closed? |
|---|---|---|
| Executor S1: `Effect.ALL` can be reassigned (fail-open) | `_EffectMeta` refuses setting and deleting `ALL`; test `test_effect_all_cannot_be_reassigned_or_deleted` | Yes. Reproduced on 3.10, 3.12, 3.13 and 3.14. Members still refuse reassignment, and other attributes still set. |
| Executor S2 (a) and (b) / Wire F1: the reserved `effect` attribute breaks user tools | V15: a non-`Effect` own value records `ALL` with one WARNING. A mismatch makes `effect_of` return `ALL`, which is admitted under `ALL` and refused below it. The changelog and docs are corrected. | Yes for both cases (probe1: `{"health": 5}`, post-hoc `Y.effect = READ`). |
| Architecture S1: the NETWORK definition contradicts V11 and the robot link | V16: the `Effect` docstring and the §2.1 row are amended | Yes. The wording matches in both places. |
| Architecture S2 / Wire F2: "refusals never silent" had no check, and 19 broad handlers exist | `EffectAdmission._refuse` logs `tool_effect_refused` (with `reason`) at raise time, and the three sites now log at debug. The batch-try prerequisite is recorded as owed by item 3. | Yes, structurally. Any broad `except Exception` now leaves a WARNING behind. |
| Architecture S3: stable_api.md described behaviour that does not exist | Reworded: "No shipped agent narrows its tools yet" | Yes. |
| Executor N1 (`declare_effect` equal-value early return) | Also checks `_EFFECT_RECORD.get(cls) == effect`, with a test | Yes, for reassignment. Not for deletion (D1). |
| Executor N2 / Architecture N5b (`or` truthiness, untested staging) | `is None` check, plus `test_a_batch_checks_a_repeated_name_against_its_own_earlier_tool` | Yes |
| Executor N3 (entity tests do not pin the type or the ordering) | `test_entity_generation_raises_the_refusal_and_maps_the_entity_only_after_registering` | Yes |
| Executor N4 / Wire F10 (acquisition reparent ordering, any-exception all-or-nothing) | Notes "Not built here" bullet; changelog "whatever stops it" | Deferred, stated |
| Executor N5 / Architecture N1 (comment drift, roster bullet, exemption path) | Comment rewritten; roster bullet deferred to item 3; §2.1 path corrected | Yes |
| Executor N6 (`autonomy_level` lacks COMMS) | COMMS added | Yes |
| Executor N7 / Wire F7 (declares before the name check) | `_registration_name` runs before `declare_effect`, with a test | Yes |
| Architecture N2 (capability without a caller) | The disclosure sentence is kept | Yes |
| Architecture N3 (`Effect` not exported from `maxim.tools` or listed in stable_api) | — | **Not folded and not deferred** (NIT, D4) |
| Architecture N4 (deregister plus register gets past the replacement rule) | Stated residual in the notes and the `_admit` docstring | Deferred, stated |
| Architecture N5a (`register_many` bypasses `register`'s body) | — | Not folded (NIT, accepted silently) |
| Architecture N6 (one `ALL` affordance keeps the whole entity out) | Notes bullet for Talk (#1431) | Deferred, stated |
| Architecture N7 (two tools over-declare) | Comments added | Yes |
| Architecture N8 (`register_tool(fn_tool, effect=)` always fails) | Docstrings in `register_tool` and `declare_effect` | Yes |
| Wire F3 (shadowing logs `tool_replaced`) | extension_api.md line | Yes |
| Wire F4 (ambiguous qualname in `tool_replaced`) | `new_effect` and `old_effect` added | Partly. `new` and `old` can still both read `abc.T_dup` (NIT) |
| Wire F5 (refusal JSONL loses the reason) | `reason` added to `data` | Yes |
| Wire F6 (`SimToolRegistry` lacks `register_many`) | `SimToolRegistry.register_many`, with a test | Yes |
| Wire F8 / F9 (exemption path, extension_api) | Both fixed | Yes |

## New findings

### SHOULD-FIX

**D1. `del cls.effect` followed by `maxim.register_tool(tool, effect=)` narrows an `ALL` tool. This is fail-open, older than the fold, and makes the fold's new claims false.**
- **Where:** `tools/base.py::effect_mismatch` (the `"effect" not in cls.__dict__` → `None` branch) and `tools/base.py::declare_effect` (the absent-value branch overwrites the record unconditionally).
- **Scenario** (`probe2.py`, reproduced): take `del ExecuteSandboxScriptTool.effect`, a code runner that declares `ALL`, then `maxim.register_tool(inst, effect=Effect.READ)`. The record becomes `READ`, `effect_of` reports `READ`, and `ToolRegistry(ceiling=Effect.READ).register(inst)` **admits it**.
- **Second scenario** (`probe1.py`, "del then inherit"): delete `C.effect` on a READ subclass of an `ALL` parent. `C.effect` now resolves to `ALL`, but `effect_of(C())` returns `READ`. That is narrower than the class now claims, and the class declares nothing in its own `__dict__`. By §2.2's rule ("undeclared means `ALL`"), it should be `ALL`.
- **Class:** in-process mutation of the declaration, the same adversary as executor S1 and §2.2's "Mutation is visible" row. The fold closed reassignment (`effect_of` → `ALL`) but not deletion. The new docstrings claim the general property. The trigger is in-process code (a plugin, a careless `monkeypatch.delattr`), not the model.
- **Fix** (two lines, both fail-closed):
  - In `effect_mismatch`, when `"effect"` is absent from `cls.__dict__`, return a mismatch whenever `recorded != Effect.ALL` (for example "its own effect was deleted; its record is [...]"). `effect_of` then returns `ALL`.
  - In `declare_effect`, refuse when `_EFFECT_RECORD.get(cls, Effect.ALL) != Effect.ALL`. A class that ever declared is never re-declared.
  - Add one test for each scenario above.

### NIT

**D2. A refusal at one of the 19 broad-handler sites is now logged twice** (the registry's `tool_effect_refused` WARNING, then the site's generic WARNING or ERROR). Fail-closed and diagnostic only. No shipped path builds a registry below `ALL`, so none of these can fire yet. `tools/registry.py::EffectAdmission._refuse`, with `runtime/bootstrap.py::build_tool_registry` and others. **Fix:** none needed now. The item 3 per-tool handlers (already owed in the notes) can catch `ToolEffectRefused` first and log at debug, as the three diff sites do.

**D3. `warn_invalid_effect_once` warns once per class for its lifetime.** A class reassigned a second time, to a different value, gets no second WARNING. Fail-closed: `effect_of` stays `ALL` and below `ALL` every attempt still logs `tool_effect_refused`. The WeakSet is safe: entries are removed on GC, so a reused `id()` cannot give a false hit, and a check-then-add race costs at most one duplicate WARNING. `tools/base.py::warn_invalid_effect_once`. **Fix:** optional. Key the set on `(cls, why)`.

**D4. Architecture N3 is neither folded nor deferred.** `Effect` is a parameter type of two stable verbs, but it is not in `maxim/tools/__init__.py`'s `_LAZY` or `__all__`, and stable_api.md does not list it. Fail-closed (documentation). **Fix:** add it, or record a "not now" line in the notes.

**D5. `api.py::_record_registration` keeps a dead `if name is not None:` branch.** `_registration_name` now always returns a non-empty str. **Fix:** drop the guard.

**D6. The §2.2 body text is stale against V15.** It still says "A non-`Effect` value or `Effect(0)` raises `TypeError`", "Only a present value that differs from the record refuses admission", and "Refusals are never silent. A structured WARNING event at `api._inject_registered_tools`, …". The status block's "where it differs from the text below" convention and the V15 row cover this, but a reader of §2.2 alone is misled. **Fix:** one line in the as-built block naming those three §2.2 sentences as superseded, or strike them.

**D7. extension_api.md does not say that the class-body route and `register_tool(effect=)` exclude each other, or that `register_tool(effect=)` declares on the CLASS** (every instance and every later registry). `register_tool`'s docstring says the first. **Fix:** one clause.

### Checked and clean
- **The `effect_of` → `effect_mismatch` call:** no recursion (`effect_mismatch` calls only `effect_names`). The cost is two dict lookups. A duck type, a mock or `SimpleNamespace` returns `ALL` before `effect_mismatch` runs.
- **`_EffectMeta`:** pickling, `deepcopy`, `Effect(1)`, `Effect(127) == ALL`, `Effect["READ"]`, `~Effect.READ` and member iteration (exactly seven) behave the same on 3.10.19, 3.12, 3.13.3 and 3.14.6. Member reassignment is still refused by EnumMeta. Other attributes still set and delete. No test monkeypatches `Effect.ALL`.
- **V15 below `ALL`:** a mismatched class is refused in `_admit` before `exceeds`, and `effect_of` returns `ALL` regardless, so neither path admits it. I found no narrowing path other than D1.
- **Logging at raise:** the three catch sites log at debug only, so a refusal is logged once there. No test in `tests/` asserts the absence of a WARNING around a registry below `ALL` (none build one outside `test_tool_effect.py` and the red gates). Under `ALL` no refusal is possible except a below-`ALL` mismatch.
- **`SimToolRegistry.register_many`:** all or nothing; it updates `_FallbackRedirectTool._registered_tools` like `register`; tested.
- **`api._registration_name` / `register_tool`:** the name is checked before `declare_effect`, and the test proves the class is left unmodified.
- **`api.tool` validation:** it is needed, because under V15 a non-`Effect` value in `FunctionTool`'s body would otherwise record `ALL` silently. `Effect(0)` is caught too.
- **Changelog, runtime-tools.md, stable_api.md and extension_api.md:** every claim matches the code, except D1's two docstring claims and D6.

## STOP decision

**Not STOP.** D1 is fail-open, so the stop rule does not apply. A delta-scoped round is owed, covering only D1's two-line fix and its tests. D2–D7 are all fail-closed, in already-covered classes (diagnostics, docs, dead code), and go to a follow-up or into the same small commit. If the owner instead rules `del cls.effect` out of the model as in-process code (§0, the same reasoning as `object.__setattr__` on the policy), the `effect_of` and `declare_effect` docstrings must say so. Then every remaining finding is fail-closed, and STOP applies.
