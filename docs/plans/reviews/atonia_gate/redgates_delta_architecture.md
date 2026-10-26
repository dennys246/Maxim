# Atonia gate red gates: delta round, architecture lens

Diff: `git diff 0ac02acb 11c90ab1` (the staged fold, now committed; 12 files, no `src/`). This round reads that diff
against notes v6.3, the previous round's `redgates_architecture.md` (B1, S1–S14) and CLAUDE.md. Read only, nothing run.

## What the fold closed

- **B1:** `test_open_builds_all_three_layers` spies on the real `open()` and checks for one `_build_dispatch_stack`
  call with all three layers, `session._executor is built`, and pin identity. The composition hole is closed.
- **S1:** every nodeid that §1A.9 and §1A.1 cite now exists under exactly that path and name. I checked
  `test_item2_every_tool_declares_own_effect`, `test_item2_process_and_code_runners_declare_all`,
  `test_item3_policy_is_required_and_permissions_removed`, `test_no_effect_subprocess`,
  `test_offline_module_is_stdlib_only` (integration/), `test_per_layer_deletion_sweep_{ceiling,pin,sealed}`,
  `test_open_builds_all_three_layers`, `test_import_closure`, `test_env_read_set`, `test_no_effect`, and
  `test_stop_reasons.py::TestToolErrorKind`. Each one is present.
- **S3:** `test_cases_cover_every_gate_refusal_kind` adds the offline-ceiling and in-tool-body cases.
- **S4, S5, S8:** the gates now use `metadata["refusal_layer"]` with `"registry"`, `.value` serialisation, and the
  `pins` slot. These are consistent across `test_advisor_gate`, the log gates, the effect-policy gates, and the
  §1A.8 example.
- **S6:** the deadline gate is split into `deadline` (a valid completion) and `deadline_half`, and the notes now say
  the deadline wins.
- **S11:** a single autouse reset in conftest replaces the two per-file fixtures.
- **The closure:** `agents.autonomy` and `agents.llm_worker` are forbidden, and `agents.permissions` is allowed
  transitively. This matches §1A.3's `LAYER_RULES` (a fence on direct imports only) and §4.2.

## BLOCKER

None.

## SHOULD-FIX

**D1 (fail-open, incomplete fold). §1A.1 v6.3 says `engage()` "calls nothing in `maxim.utils`" and "its only writes
are to `os.environ`". §6 layer 2 still says `engage()` calls `close_all()` (`src/maxim/utils/http.py::close_all`),
sets `huggingface_hub.constants.HF_HUB_OFFLINE`, and sets `ultralytics.utils.ONLINE`.**
- §1A wins over §6, so an implementer who follows the notes drops `close_all()`.
- `test_import_closure` also forbids `maxim.utils.http` after an `open()` that engages, so even a lazy call would
  fail that test.
- **Failure scenario:** a run raises `MAXIM_OFFLINE` mid-run (§6 says "a later raise counts"). An httpx client
  pooled before the latch keeps its keep-alive socket. That client was built before the strict per-request
  `request.url` check, and `send`/`sendall` on a connected socket raise no audit event. Bytes leave after the latch
  is up.
- No red gate covers it: `test_offline_latch.py` never mentions a pool or `close_all`.
- **Fix, in the notes:** invert the edge. `utils/http.py` registers `close_all` with a stdlib-only
  `_offline.on_engage(callback)` (or an equivalent), so the leaf keeps no back-edge.
- **Fix, in the wording:** restate "only writes" as "five `os.environ` keys, plus the two already-imported library
  flags (§6), plus registered on-engage callbacks". The observer's env-write allow-list stays the five keys.
- **Fix, in the tests:** add a strict red gate. A pooled `get_client()` connection opened before `engage()` must not
  carry a request after it: the request raises `OfflineRefusal`, or the pool is closed.
- Separately, `HF_HUB_DISABLE_TELEMETRY` appears only in the new §1A.1 line. §6's lists (around "HF:" and the
  spawn-children inherit list) should name it too (NIT).

**D2 (fail-open, latent). `_registry_public_writers` decides what counts as a reader by name prefix:
`("get", "find", "list", "is_", "has_", ...)`.**
- A later `ToolRegistry` writer named `get_or_register` or `list_replace` would be classed as a reader and left out
  of the sweep. Today that hole is not exploitable.
- The old hand list failed closed when stale. This derivation fails open on exactly the stale case S7 was about.
- **Fix:** list the reader names exactly (or take them from the `ToolRegistryLike` Protocol's read methods). Treat
  every other public callable as a writer. A new reader then fails the gate loudly until someone adds it.
- The sweep also requires every `ToolRegistry` public writer to exist on the sealed class and raise. That makes the
  sealed class mirror `ToolRegistry`. It fails closed and is acceptable, but §4.2 should say it.

**D3 (fail-closed, interaction). The conftest reset is defeated by v6.3's own env write.**
- `engage()` now writes `MAXIM_OFFLINE` (and four more keys) into `os.environ`.
- §6 says `is_offline()` "engages on any call that sees `MAXIM_OFFLINE` truthy".
- After `_reset_for_tests()`, the next `is_offline()` in a later test therefore re-latches. The fixture only resets
  `_offline`'s private state and does not restore the environment.
- **Result:** an in-process `open()` in one test makes later, unrelated tests fail with `OfflineRefusal`, far from
  the cause. That is loud but order-dependent, the #1339 class.
- **Fix:** snapshot the five keys before `yield` and restore them after (or have the notes say `_reset_for_tests`
  restores them).
- Also, once `maxim._offline` exists, a missing `_reset_for_tests` should fail rather than be skipped. The
  `getattr(..., None)` plus `callable` check is a silent no-op in test infrastructure.
- The fixture is otherwise in line with ci-testing.md: it is autouse, reads `sys.modules` without importing, and
  costs nothing before step 0b.

**D4 (fail-closed, notes vocabulary incomplete).**
- (a) `OFFLINE`, which the seam maps from an `OfflineRefusal` inside a tool body, has no `refusal_layer` in the closed
  set `forbid|mode|registry|permissions|ceiling|pin`. Yet §3 says "every refused dispatch carries ... the layer".
- (b) The V7 offline-ceiling refusal's kind is not decided. The booking gate accepts `{"POLICY_CEILING", "OFFLINE"}`,
  a deliberately loose pin.
- (c) The `"forbid"` layer has no `ToolErrorKind` in `GATE_REFUSAL_KINDS`, so it is unclear whether a forbid refusal
  books NAc.
- (d) §3's "Refusals are typed" list (`ToolErrorKind` gains ...) omits `INACTIVE_SCENE`, which v6.3 added to
  `GATE_REFUSAL_KINDS` and which is not a member today.
- **Fix:** one table in §1A.2 that maps kind to layer for every gate kind.

## NIT

- **Call style in `test_open_builds_all_three_layers`.** `kwargs.get("layers", ())` and
  `args[0] if args else kwargs.get("propose_tool")` pin the call style. A correct `open()` that relies on a default
  `layers=` fails. Use `inspect.signature(real).bind(*args, **kwargs)` with `apply_defaults()`. The notes still do not
  name `_build_dispatch_stack` as a production-module test seam that can build an unsealed, ALL-ceiling executor. B1's
  docstring and census-listing half is unfolded.
- **§4.2's env-read rule is folded only in the v6.3 table, not in its body.** The body still says "any iteration or
  copy ... fails the test" and "a pinned allow-list". The gate adds two things the body does not mention:
  - `ENV_WILDCARD_ALLOW` (the one sanctioned scanner);
  - `ENV_READ_ALLOW_CALLERS`, which allows any read attributed to `numpy*` or `threadpoolctl*`. This makes the exact
    numpy names in `ENV_READ_ALLOW` decorative.

  Reads attributed to maxim stay pinned, so the threat the gate exists for (hostile `MAXIM_*`) is still covered.
  Record both in the §4.2 body.
- **Incidental pins.**
  - `_LAZY` as the lazy-table name. §1A.1 says only "the lazy table".
  - `ConfigurationError` and `ModelError` in O8, and `pool._llm_router`.
  - `maxim._offline.degraded()`. It is lenient, OR-ed with a WARNING.

  Each fails closed, and each is named in its file's SEAMS docstring. Either list them in the notes' seams or accept
  them as the implementing PR's to change.
- §1A.9 invariant 4 writes `::…_pin`, `::…_sealed`. Spell the nodeids out, so that the lift into a brief carries
  greppable names.
- File docstrings still cite "DRAFT v6.2" (for example `test_offline_latch.py`'s header) while their bodies say v6.3.
- The ddgs census note is dated 2026-10-26, which is after this review (2026-10-25).

## Scope

The diff touches no `src/` and no CI configuration. It contains tests, one conftest fixture, the notes, and three
review files. That is within a tests-only red-gate PR that also carries its plan fold.

## Stop rule

There is no BLOCKER. There are two fail-open SHOULD-FIXes: D1, which is notes plus one new red gate, and D2, a
one-line derivation change. **A further round is owed, scoped to the D1 and D2 folds only.** D3 and D4 fail closed
and can be folded without a round.
