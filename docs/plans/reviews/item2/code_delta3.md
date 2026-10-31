# Item 2 micro-round 3: the D8 fold (`declare_effect` refuses an unrecorded class)

Scope: the staged hunk in `src/maxim/tools/base.py::declare_effect` (`if cls not in _EFFECT_RECORD: raise TypeError`)
and `tests/unit/test_tool_effect.py::test_an_unrecorded_class_cannot_be_declared`, against HEAD 5fb640ac.
Probe script: `/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/248fdc0d-2f2e-471a-b37d-dacb96a7ecce/scratchpad/review_delta3/probe.py`.
`tests/unit/test_tool_effect.py`: 29 passed.

## Verdict: D8 CLOSED; no legitimate normally-created Tool class is refused

### (1) Does it close D8?

Yes. The check runs before every other branch, so an unrecorded class cannot reach `type.__setattr__` /
`_EFFECT_RECORD[cls] = effect` whatever its `__dict__` holds. Probe: subclass `J` of a base whose
`__init_subclass__` skips `super()` gets `effect_of` = ALL, `ToolRegistry(ceiling=READ)` refuses it, and
`declare_effect(J, READ)` raises. The staged test covers the exact D8 shape (body ALL, `del`, declare READ).
The only way into `_EFFECT_RECORD` is now `Tool.__init_subclass__` or a `declare_effect` on an already-recorded
class, so the record can no longer be created by a late declaration.

### (2) Can it refuse a legitimate `register_tool(tool, effect=)`?

Not for any class created through `Tool.__init_subclass__`. Every case below was declared READ without error:

| Case | Result |
|---|---|
| plain subclass | ok |
| `@dataclass` | ok |
| `@dataclass(slots=True)` (dataclass rebuilds the class; the rebuilt class passes through the hook again) | ok |
| ABC intermediate with its own `@abstractmethod` | ok |
| `type("D", (Tool,), ns)` | ok |
| `types.new_class` | ok |
| custom metaclass derived from `ABCMeta` | ok |
| base `__init_subclass__` that calls `super()` | ok |
| mixin before `Tool` in the MRO whose hook calls `super()` | ok |
| the broken-hook base ITSELF (Tool's hook ran for it) | ok |
| same class, same effect twice | ok (idempotent) |

Refused (all with a loud `TypeError`, and `register_tool` raises before `_record_registration`, so nothing is
registered):

| Case | Note |
|---|---|
| subclass of a no-`super()` base, never declared (`J`) | see F1 |
| `class H2(MixBad, Tool)`, `MixBad.__init_subclass__` without `super()` | see F1; `Tool`'s hook never runs for H2 |
| virtual subclass via `Tool.register(V)` | see F2 |
| `Tool` itself | abstract, not instantiable; unreachable from `register_tool` |

## Findings

**F1 (fail-closed; already covered: it is D8's own refusal class).** A never-declared subclass of a no-`super()` base
or mixin (`J`, `H2`) was declarable through `register_tool(effect=)` at HEAD; now that call raises and the tool is not
registered. This is unavoidable: without a record, "never declared" and "declared then deleted" cannot be told apart,
and that is exactly D8. The user can still `register_tool(tool)` without `effect=` (counts as ALL) or fix the hook.
`docs/user/extension_api.md` documents only the "already declares one" refusal; adding half a sentence is optional.

**F2 (fail-closed, cosmetic; already covered: same refusal).** For a virtual subclass (`Tool.register(V)`) the message
blames "an `__init_subclass__` that does not call `super()`?", which is the wrong hint: V is not a real subclass.
The refusal itself is correct (V never passed the hook). A wording tweak such as "was not created through
`Tool.__init_subclass__`" would cover both causes. NIT.

Nothing else is new. The staged hunk adds no swallow, no env var and no new caller. `api.register_tool` is the only
non-test caller of `declare_effect`, and the approach-notes line for D8 matches the code.

## STOP decision

Both findings are fail-closed and in an already-covered class (D8's refusal of an unrecorded class).
**STOP: no further round owed.**
