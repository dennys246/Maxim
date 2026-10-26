# Atonia gate red gates: second delta round, architecture lens

Scope: the staged fold (`git diff --cached`), and within it only the parts that answer `redgates_delta_architecture.md`
D1–D4. Those parts are notes v6.4 §1A.1 (`engage()`'s three steps), §1A.2 (the one-kind-per-layer table), §3 (the
"Refusals are typed" list), §6 (the `_reset_for_tests` bullet), the conftest fixture, and
`test_offline_latch.py` §7c. I read the code and ran one AST probe. No tests were run.

## Fold verdicts

- **D1: closed in substance.** `engage()` imports nothing from `maxim.utils`, and `utils/http.py` registers
  `close_all` through `on_engage`, so the dependency edge is `utils → _offline`. That edge respects layering:
  `_offline` is a top-level leaf, it has no `LAYER_RULES` key, and no rule forbids `utils` from importing it. The
  leaf's stdlib-only rule also holds, because `on_engage` stores callables and does not import their owners. The
  behavioural gate `test_a_client_pooled_before_engage_cannot_send_after` discriminates correctly: a pre-latch client
  is non-strict, so without `close_all` the listener sees a third request. §6 line "`engage()` calls `close_all()`" is
  now consistent with §1A.1 under the callback reading.
- **D3: closed.** The fixture snapshots and restores the five keys in a `finally` block, and it fails when the
  module is loaded but the reset is missing.
- **D4: closed.** The kind-to-layer table covers every member, and `INACTIVE_SCENE` is in §3. One remaining
  ambiguity: `POLICY_PERMISSIONS` maps to two layers (`forbid` and `permissions`), so "Each layer has exactly one
  kind" is true but "each kind has one layer" is not. The gates read `refusal_layer`, so this is harmless (NIT N3).

## BLOCKER

None.

## SHOULD-FIX

**S1 (fail-open, new in this fold). `test_no_caller_of_reset_for_tests_outside_tests` misses an aliased import.**
- `from maxim._offline import _reset_for_tests as r; r()` produces `ImportFrom` → `alias(name="_reset_for_tests")`
  plus `Name("r")`. The scan matches only `Attribute`, `Name` and `str` `Constant` nodes, so it never sees the name.
  I checked the AST node list.
- This is adversary 4 in §0 ("future code"), and `import … as` is an ordinary idiom, not evasion. It is the one guard
  that keeps invariant 3's "one-way" claim honest now that the reset lives in `src/`.
- **Fix:** also match `ast.alias` nodes whose `.name` is `_reset_for_tests` (or ends with `._reset_for_tests`).
  Prove it by deletion: a scratch `src/` file with the aliased form must fail the guard. Dynamic strings
  (`getattr(m, "_reset" + "_for_tests")`) are arbitrary-code-shaped and fall under §0's out-of-model row; say so in
  the docstring.
- The notes call this "a lint", but it is a test in the integration file, not a `scripts/lint_*.py` in the CI lint
  job, which is what §1A.10 says censuses ride on. It still runs in the fast suite. Either name it as a test in §6, or
  move it into a lint (NIT-level wording).

**S2 (fail-open, pre-existing, sharpened by this fold). Invariant 3's text overclaims for a mid-run raise.**
- "One-way" is honest given §0. In-process code is out of the model, and with S1 fixed the only production
  disengage path is guarded. The invariant still does not say so, and its regression guard does not cite the reset
  guard. **Fix:** "one-way: no non-test caller can disengage it (guard:
  `test_no_caller_of_reset_for_tests_outside_tests`)".
- "No socket" is wider than the design. §1A.1 step 3 closes only `utils/http.py`'s pool. A connection established
  before a *later* raise outside that pool (the reachy_mini SDK websocket in §6's defect list, and the allow-listed
  raw-socket sites) keeps sending, because `send`/`sendall` raise no audit event. The notes themselves give that
  reason for step 3.
- **Fix (docs only):** "no new connection, DNS lookup or spawn after engage; `utils/http.py`'s pool is closed; other
  pre-latch connections are not, so a mid-run raise is complete only for the HTTP pool." The other option is to name
  which established channels must register `on_engage`. Startup engage (CLI first statement, the observer) is
  unaffected.

## NIT

- **N1: `engage_callbacks()` is an incidental pin.** The `close_all_registered` assertion fails a correct
  implementation that registers a wrapper (`lambda: close_all()`, or a `_on_offline` that also rebuilds strict). The
  listener-port and refusal assertions already prove step 3 behaviourally. Keep `has_on_engage` if you want the seam
  pinned, and drop the `engage_callbacks()` identity check or relax it to "some callback is registered". It fails
  closed.
- **N2: "Exactly three things happen in `engage()`" is not complete.** `engage()` also sets the latch and installs
  the hook on its first run (§6). The notes should also state the order, so a callback exception cannot leave the
  process unlatched: latch and hook first, then steps 1–3. A callback that raises is re-raised after the rest run.
  As written this fails closed, because a raise out of `engage()` crashes `cli.main` or the observer loudly.
- **N3:** `POLICY_PERMISSIONS` maps to both `forbid` and `permissions`, as noted under D4 above. State that the
  layer, not the kind, distinguishes them.
- **N4: §1A.1 "Tests." contradicts §6.** It still says "The conftest latch reset touches `maxim._offline`'s private
  state", but v6.4 has the fixture call `_reset_for_tests()`.
- **N5: `_reset_for_tests` "keeps the `on_engage` callbacks" interacts with "runs `fn` at once if the latch is
  already up".** After a reset, a later `engage()` re-runs `close_all`. That is fine, but one sentence should say
  that callbacks run on every engage, not only once per process.

## CLAUDE.md

No contradiction. The fold does not touch `src/`. `utils/http.py` remains the only HTTP surface. The new `os.environ`
writes are paired with the conftest restore (the autouse-scrub lesson). Nothing is swallowed.

## Stop rule

There is no BLOCKER. There are two fail-open SHOULD-FIXes. S1 is new and tests-only, about three lines. S2 is
pre-existing and docs-only. **A further round is owed, scoped to S1 only:** the alias match plus its deletion probe.
S2 and the NITs are wording, and wording needs no round.
