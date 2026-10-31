# #1315 item 2: delta round 2 (D1 fix, `git diff --cached` vs 5fb640ac)

Probe: /private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/248fdc0d-2f2e-471a-b37d-dacb96a7ecce/scratchpad/review_delta2/probe.py
Narrow tests: test_tool_effect.py + test_profile_readers_v9.py = 65 passed, 1 xfailed (#1479, unrelated). The
TYPE_CHECKING mirror equals `_LAZY` for `maxim.tools` (`maxim.tools` is in LAZY_PACKAGES); `maxim.tools.Effect` resolves and is in `__all__`.

## Verdict: D1 CLOSED for every class that passes through `Tool.__init_subclass__`; one fail-open residual (D8)

| Attack | Result |
|---|---|
| ALL body, `del`, reassign READ | own READ != record ALL -> mismatch -> ALL, refused under READ; `declare_effect` TypeError |
| Same via `type.__delattr__` / `type.__setattr__` | identical (no metaclass hook to bypass; same `__dict__`) |
| `del` on an undeclared class | AttributeError (no own attr); a later `declare_effect` is the sanctioned V10 path |
| Undeclared -> declare READ -> READ again -> ALL | declared / no-op / TypeError |
| ...then `del` -> declare again | effect_of ALL (deleted, record READ), refused; `declare_effect` TypeError (in `_EFFECT_DECLARED`) |
| Base READ deleted; undeclared sub | sub ALL, refused (sub's `declare_effect` allowed: sub never declared, V10 path) |
| Base READ deleted; sub declares READ itself | sub READ (its own declaration, correct) |
| Sub declares ALL over a READ base, sub's effect deleted | ALL; `declare_effect` TypeError |
| V15 non-Effect value -> `declare_effect` | TypeError; after `del` the declare is allowed (by design: the value was never a declaration and counted ALL) |
| GC + recreate under same name (same `id()` reused, verified) | new class not in `_EFFECT_DECLARED`; WeakSet drops the dead ref at dealloc and dead refs never compare equal to live ones -> no stale carry-over either way |
| `api._record_registration` guard removal | `_registration_name` always returns a non-empty str or raises, so the removed `None` branch was dead; no behaviour change |

## New finding

**D8 (fail-open, NIT; already-covered class: in-process mutation of a declaration, executor S1 / D1).**
A `Tool` base that overrides `__init_subclass__` WITHOUT calling `super()` leaves its subclasses with no record and
out of `_EFFECT_DECLARED`. A subclass whose body declares `effect = Effect.ALL` is ALL (no record -> `effect_of` ALL,
fail-closed), but after `del Sub.effect` the class looks undeclared, so `register_tool(inst, effect=Effect.READ)`
declares it READ and `ToolRegistry(ceiling=READ)` ADMITS it (probe case 7). That is D1's exact shape, reached through
a broken hook chain instead of a recorded declaration. No such base exists in `src/` today (the only other
`__init_subclass__` is `utils/store_ownership.py`, not a Tool, and it calls super). Trigger needs third-party code
with a swallowing hook plus a deliberate `del`.
Fix (fail-closed, one line): in `declare_effect`, refuse a class with no `_EFFECT_RECORD` entry ("its class creation
bypassed Tool.__init_subclass__"), plus a test of case 7. Alternative: accept as a stated residual in the notes.

## STOP decision

Not earned under the stop rule: D8 is fail-open (though in an already-covered class, and trigger-gated on
third-party code). Either fold the one-line refusal + test (a micro-round over that delta only), or record D8 as an
owner-accepted residual; nothing else is owed.
