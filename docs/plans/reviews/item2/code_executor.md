# Executor lens: #1315 item 2 (`Tool.effect` and the registry ceiling)

Reviewed: `git diff origin/main...HEAD` at e37f2ea1 (worktree `.worktrees/v-effect`). Spec: atonia notes §2 (the BUILT
status block, §2.1, §2.2), §1A.3, §1A.4, V10–V14.

## Verdict: SOUND WITH CONDITIONS

No BLOCKER. The admission mechanism does what §2.2 says at every input I drove. The record works, as do the
own-`__dict__` rule, the mismatch check, `Effect(0)`/non-Effect refusal and the all-or-nothing batches (eviction is
admitted first). It holds on Python 3.10, 3.12, 3.13 and 3.14. I spot-checked 27 of the 86 classifications and none
is fail-open. Two SHOULD-FIX items need a decision before merge: one is a cheap fail-open hardening, the other a
stable-verb compatibility claim that the code does not keep.

What I ran:
- `tests/unit/test_tool_effect.py` + `tests/unit/test_atonia_red_gates_effect_policy.py`: 35 passed, 14 xfailed
  (item 3 and the O2/O5 gate, as expected).
- 11 neighbouring test files that call `generate_tools_for_entity` or build registries: 272 passed.
- `Effect` extracted and run under `/opt/homebrew/bin/python3.10/3.12/3.13/3.14 -I`: iteration yields exactly the seven
  flags on all four. `ALL` is not a member, `~ALL` is falsy, and pickle and deepcopy round-trip. `maxim.tools.base`
  imports on 3.10, and a `@dataclass(slots=True)` Tool subclass records its effect correctly.
- A default `build_tool_registry` (30 tools) emits zero `tool_replaced` warnings. The narrator registry
  (orchestrator.py) registers no duplicate names either, so the new WARNING is not noise on shipped paths.
- Probe scripts are in `.../scratchpad/review_exec/` (`probe1.py`, `probe2.py`, `effect_compat.py`,
  `test_replace_noise.py`).

---

## SHOULD-FIX

### S1. `Effect.ALL` is a plain, reassignable class attribute, and the fail-closed default reads it live (fail-open)
`tools/base.py::Effect`, `effect_of`, `Tool.__init_subclass__`, `tools/registry.py::EffectAdmission._ceiling` /
`_admit`, `simulation/tools.py::SimToolRegistry.get`

`EnumMeta.__setattr__` blocks reassigning members only. `ALL` is attached after the class and is not a member, so
`Effect.ALL = Effect.READ` succeeds on 3.10 through 3.14 (reproduced). Every fail-closed default reads it at call
time:
- `effect_of` returns `_EFFECT_RECORD.get(type(obj), Effect.ALL)`;
- `__init_subclass__` records `Effect.ALL` for an undeclared class;
- `_admit` and `SimToolRegistry.get` compare `self._ceiling != Effect.ALL`.

**Scenario** (`probe1.py`): after `Effect.ALL = Effect.READ`, an undeclared Tool subclass created afterwards and a
duck-typed tool (no record) are both **admitted** by `ToolRegistry(ceiling=Effect.READ)`. `effect_of` reports
`Effect.READ` for the undeclared class.

The trigger is an in-process assignment: a careless `monkeypatch.setattr(Effect, "ALL", ...)` in a test, or a plugin.
That is the same adversary class §2.2 already closes for `cls.effect = ...` ("Mutation is visible"), so leaving the
constant itself mutable is inconsistent.

**Fix:** compute `_ALL = functools.reduce(operator.or_, Effect)` once as a module-private constant, use it in
`effect_of`, `__init_subclass__`, `EffectAdmission` and `SimToolRegistry.get`, and keep `Effect.ALL = _ALL` as the
public alias. Alternatively, give `Effect` a metaclass (an `EnumMeta` subclass) whose `__setattr__` refuses `ALL`
after the first set. Add a one-line test that `Effect.ALL = Effect.READ` raises, or that `effect_of(duck)` stays at
the seven flags after a reassignment.

### S2. Reserving the `effect` class attribute breaks user tools under `maxim.run`, and the changelog says nothing changes (fail-closed)
`tools/base.py::Tool.__init_subclass__`, `tools/base.py::effect_mismatch`, `api.py::_inject_registered_tools`,
`changelog.d/1315-item2-tool-effect.added.md`

V10 says the stable verbs stay unchanged. The changelog says "nothing changes for `maxim.run` or any shipped run".
Two reproduced cases contradict that:
- **(a)** A user `Tool` subclass that already has a class attribute named `effect` with its own meaning, such as
  `effect = {"health": 5.0}` (plausible in this codebase: affordances carry an `effect: dict[str, float]`), now raises
  `TypeError` **at class creation**, so the user's module fails to import (`probe1.py` case 1).
- **(b)** A user who declares post hoc with `MyTool.effect = Effect.READ`, the natural idiom, or who assigns a domain
  value after the class is defined, gets the tool **dropped under ceiling ALL**. `_inject_registered_tools` catches
  the mismatch refusal and logs `tool 'lookup' refused: ... reassigned after class creation`, and `injected 0`
  (`probe2.py`). The comment at that catch site ("A ceiling below ALL ... never maxim.run") is also wrong: a mismatch
  refuses at ALL as well.
- Related: `register_tool(Lookup(), effect=Effect.READ)` does not heal (b). `declare_effect` sees
  `cls.__dict__["effect"] == effect` and returns early while the record still says ALL, so the class stays
  inadmissible with no error at the call (see N1).

Both cases are loud, so this is fail-closed, but it is a stable-API behaviour change that is not documented.

**Fix (owner call; the strict option first):**
- **(i)** Keep the behaviour and document the reserved name. `stable_api.md` and the changelog should state that a
  `Tool` subclass's class attribute `effect` is now reserved for `Effect`, and that `effect` must not be set after the
  class is created; use `register_tool(effect=)` instead. Make the mismatch message name that remedy, and correct
  the comment in `_inject_registered_tools`.
- **(ii)** Alternatively, under ceiling `ALL` only, downgrade a mismatch from a refusal to a WARNING. Admission there
  adds no safety, and dispatch in item 3 is authoritative. This keeps refusal below ALL.

Either way, the changelog's "nothing changes" line has to be corrected.

---

## NIT

### N1. `declare_effect` returns silently on an equal value even when the record disagrees (fail-closed)
`tools/base.py::declare_effect`

When `cls.__dict__["effect"] == effect`, the function returns without checking `_EFFECT_RECORD[cls]`. After a
post-hoc `cls.effect = X`, the call `declare_effect(cls, X)` therefore "succeeds", yet `effect_mismatch` still
refuses the class at admission.

**Fix:** if a record exists and differs from `effect`, raise the same `TypeError` as the "cannot rewrite" branch.
Do not update the record, which would launder a narrowed explicit declaration.

### N2. `_admit_all`'s intra-batch staging is untested, and it uses `or` on Tool truthiness (neutral)
`tools/registry.py::ToolRegistry._admit_all`

No test registers two same-name tools in one `register_many` or `register_scene_tools` batch. If the `staged` dict
were removed, every test would still pass. Below ALL that batch must refuse; under ALL with two classes it must log
`tool_replaced`. `staged.get(n) or self._tools.get(n)` would also misfire if a Tool ever defined `__bool__` or
`__len__` (none do today).

**Fix:** use `staged[n] if n in staged else self._tools.get(n)`, and add the two-same-name batch test.

### N3. Entity-generation tests do not pin the refusal type or the EntityMap ordering (would pass on broken code)
`tests/unit/test_atonia_red_gates_effect_policy.py::test_item2_entity_generation_is_all_or_nothing`

That gate swallows any `Exception`, so an implementation that raised `AttributeError` before registering anything
would pass. The changelog's claim that the entity is added to the `EntityMap` only after its tools register has no
test.

**Fix:** in `test_tool_effect.py`, call `generate_tools_for_entity(..., entity_map=EntityMap())` under a narrow
ceiling. Assert `pytest.raises(ToolEffectRefused)`, assert the entity does not resolve in the map, and assert that
it does resolve on the success path.

### N4. The `generate_tools_for_entity` reorder has side effects beyond refusals (fail-closed, behaviour change)
`embodiment/tool_bridge.py::generate_tools_for_entity`, `runtime/executor.py::Executor._handle_entity_acquisition`

- Any construction exception now leaves the entity out of the `EntityMap`, not only a refusal. Before this change,
  the map registration came first. The DM runtime logs that case at DEBUG.
- On acquisition, a refusal leaves the entity reparented and `transfer_to_self`'d with no tools. That was already
  true for other exceptions, but the item claims "no half set".
- I checked the consumers. Tool constructors do not read `entity_map`: `ModulatorAffordanceTool` uses it only in
  `execute`. So no consumer needed the earlier registration.

**Fix:** state the acquisition residual in the notes. Optionally, keep the map registration first for the
non-refusal path, or accept the stricter behaviour explicitly.

### N5. `api._inject_registered_tools` comment and §2.2 "prompt roster" bullet drift (doc)
- The catch comment covers the below-ALL case only (see S2).
- §2.2 lists "The prompt roster (`permits`) applies the ceiling too", but `Executor.permits` was not touched, and the
  BUILT status block does not list this as a deviation. Admission already keeps above-ceiling tools out of the
  registry, so the roster cannot offer one today. Either say "item 3" in the status block or drop the bullet.

### N6. Classification consistency: `autonomy_level` reaches a person without `COMMS` (judgement; fail-open only for a contrived ceiling)
`tools/mode_switch.py::AutonomyLevelTool`

`request_autonomy` calls the attached approval surface, which is a human prompt (`agents/autonomy.py::
AutonomyController.request_autonomy`). `request_interaction`, by contrast, declares `COMMS` for prompting the user.
The effect is only practical under a ceiling that admits `GOVERN` but not `COMMS`, so this is a NIT.

**Fix:** add `COMMS`, or record the "an approval prompt is not COMMS" default next to the other defaults in the
status block.

### N7. `register_tool(tool, effect=)` declares before `_record_registration` validates
`api.py::register_tool`

`_record_registration` can still raise `TypeError` for a nameless tool, and by then the class has already been
mutated. The window is tiny, since a `Tool` instance already passed `Tool.__init__`'s name check.

**Fix:** validate the name first. Concurrent `declare_effect` calls with different effects can interleave the
attribute and record writes, but the result is a loud mismatch (fail-closed), so a lock is optional.

---

## Classification spot-check (27 classes; no fail-open found)

Checked against `execute` and its collaborators:

| Tool | Declared | What execute and its collaborators do |
|---|---|---|
| `speak` | READ\|ACTUATE | piper in-process; device sink or `LocalSpeaker`; robot link = ACTUATE by default |
| `respond` | READ\|WRITE_LOCAL\|ACTUATE | |
| `search_code` | READ | pure Python, no subprocess |
| `run_tests` | ALL | |
| `track_target` | READ\|ACTUATE | |
| `maxim_command` | READ\|SELF_SCOPE\|ACTUATE\|GOVERN | training logger under the data home; motion; request_* |
| `focus_interests` | ALL | V13 |
| `display_mode` | READ\|ACTUATE | sim_logger display only |
| `set_scene` | ACTUATE | sim_logger display only |
| `request_interaction` | READ\|ACTUATE\|COMMS | |
| comms `send_message` / `call_user` | COMMS\|NETWORK | gateway; Twilio |
| `say` | READ | |
| `think` | READ\|SELF_SCOPE | |
| `math` | READ\|SELF_SCOPE | no eval or sympify in `maxim/math` |
| `generate_scenario` | READ | writes no file |
| `choose` | READ | |
| `design_entity` | READ | V11 |
| `emit_campaign` | READ\|WRITE_LOCAL | |
| `predict_outcome` | READ | NAc predict paths write nothing |
| `similarity_search` | READ | EC lookups read only |
| `sense` | READ | |
| `sense_presence` | READ\|SELF_SCOPE | imagination registers scene entities, observe-only; the trigger's `_tool_registry` is stored but never used |
| `sense_tools` | READ\|GOVERN | |
| `sleep` | READ\|GOVERN | `set_processing_state`, no actuator |
| `mode_switch` / `autonomy_level` | READ\|GOVERN | see N6 |
| `damage_component` / `set_entity_sensor` | READ\|SELF_SCOPE | pain-bus subscribers write memory and NAc only; no actuation |
| `inspect_aut` | READ\|SELF_SCOPE | `ALLOWED_QUERIES` only |
| `create_sandbox_script` | READ\|WRITE_LOCAL | writes, never runs |
| git tools | per exemptions | `core.hooksPath=/dev/null` hardening confirmed |
| reachy stubs | match the real tools | V14 |
