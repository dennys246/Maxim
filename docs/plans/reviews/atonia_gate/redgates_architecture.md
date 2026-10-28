# Atonia gate red gates: architecture lens

Diff: `git diff origin/main...HEAD` on `test/atonia-red-gates` (seven new test files, tests only). Read against
`docs/plans/atonia_gate_approach_notes.md` v6.2 (§1A wins), `docs/CODE_REVIEW.md`, CLAUDE.md's working principles
and invariants. Sanity run: the five unit files give `10 passed, 50 xfailed` (all strict xfails fail today with
`AssertionError`). The two integration files were read, not run.

Overall: most gates pin observable behaviour at the level the notes name, use the existence-assertion then
behaviour pattern, and keep `ScenarioBroken` apart from the gate. The problems below are about where the gates
pin private seams, and about places where the notes and the gates disagree.

## BLOCKER

**B1. `tests/integration/test_advisor_gate.py::test_per_layer_deletion_sweep_{ceiling,pin,sealed}`: the
deletion probe checks a test-only builder, and no gate ties that builder to what `open()` builds.**
§1A.9 invariant 4 cites the per-layer sweep as its regression guard. The sweep drives
`_session._build_dispatch_stack(propose, layers=..., registry_tools=...)`. Nothing in the diff asserts that
`maxim.advisor.open()` builds its executor through that same function with all three layers.
`test_registration_sweep_b` only inspects `session._executor.registry`.
Failure scenario: the implementer writes `_build_dispatch_stack` for the tests, and `open()` builds its own
`Executor(sealed, policy=ToolPolicy.pinned(...))` inline, say with `ceiling=Effect.ALL` left in by mistake. All
three sweeps go green, and invariant 4's guard passes over a broken production composition. This is the
"shipped the pieces, not the composition" lesson (CLAUDE.md, D43).
Fix: add a gate that spies on `_session._build_dispatch_stack` during a real `open()` and asserts one call with
`layers == {"sealed","ceiling","pin"}`. The alternative is to assert that `session._executor` equals what the
full-layer build returns (same registry type, same policy ceiling and pin identity). Also, the test-only seam lives
in a production module, and with `layers=frozenset()` it builds an unsealed, unpinned, ALL-ceiling observer
executor. Its docstring should say it is test-only, and the closure or census lint should list it.

## SHOULD-FIX

**S1. §1A.9's guard nodeids do not exist under the names the notes give.** Checked one by one:
- Invariant 1: `tests/unit/test_tool_effect.py::test_every_tool_declares_own_effect` and
  `::test_process_and_code_runners_declare_all` are **missing**. The tests are
  `tests/unit/test_atonia_red_gates_effect_policy.py::test_item2_every_tool_declares_own_effect` and
  `::test_item2_process_and_code_runners_declare_all`.
- Invariant 2: `tests/unit/test_tool_policy.py::test_executor_refuses_missing_policy` is **missing**. The test is
  `test_atonia_red_gates_effect_policy.py::test_item3_policy_is_required_and_permissions_removed`.
- Invariant 3: `tests/integration/test_offline_latch.py::test_no_effect_subprocess` is **present**.
- Invariant 4: `::test_import_closure`, `::test_env_read_set` and `::test_no_effect` are **present** (parametrised by
  backend). `::test_per_layer_deletion_sweep` is **missing**: it is split into `_ceiling`, `_pin` and `_sealed`.
- §1A.1: the notes put `test_offline_module_is_stdlib_only` in `tests/unit/test_offline_latch.py`. It is in
  `tests/integration/test_offline_latch.py`, and the file's docstring admits the move.

`lint_claude_md_invariants.py` checks markdown links, not backticked nodeids. So when the stubs are lifted into
runtime-tools.md or llm-routing.md, they will cite guards that do not exist, and nothing will catch it.
Recommendation: rename the tests to the notes' subsystem-named permanent homes, `tests/unit/test_tool_effect.py` and
`tests/unit/test_tool_policy.py`. A red gate becomes the permanent guard once it flips, and "atonia_red_gates" is a
name for a moment in time, not for a subsystem. Fold the split sweep name and the stdlib-only test's path into the
notes.

**S2. Contradiction across files: `test_refusal_booking_red_gate.py::_rig` vs
`test_atonia_red_gates_effect_policy.py::test_item3_policy_is_required_and_permissions_removed`.**
The booking rig calls `build_executor(..., permissions=permissions, ...)`. The item-3 gate requires that the same call
raises `TypeError` (§1A.5, which lands in one commit with item 3). On the day the booking gate is meant to flip, its
rig raises `TypeError` instead of flipping, and strict `raises=AssertionError` reports that as a hard failure. The
implementer then has to re-plumb the gate's setup, which is the re-pointing the strict-red-gate rule warns against.
Fix: build the rig through one helper that uses `ToolPolicy.deny(...)` or `unrestricted(...)` when
`maxim.runtime.tool_policy` exists and `permissions=` otherwise. The docstring should name that helper as the one
line the item-3 PR touches.

**S3. The booking gate does not cover all the kinds the notes require.** §1A.2 says: "For each kind in
`GATE_REFUSAL_KINDS`, a refusal is driven..." and "The behavioural gate drives [the `OfflineRefusal` booking-order]
case too." `KINDS` is only {mode, permissions, unregistered, inactive_scene}. `POLICY_CEILING`, `POLICY_PIN` and
`OFFLINE` have no case. That includes an `OfflineRefusal` raised inside a tool body, which goes through `_fail_ran`,
the one path where the booking order matters. Add these as strict xfails that assert existence first. Also add an
assertion that, once `GATE_REFUSAL_KINDS` exists, the cases cover it, so a new kind joins automatically. The
`_tools_hallucinated` "kept" sink is not asserted (NIT).

**S4. `test_advisor_gate.py::_assert_refused` pins the layer vocabulary `{"sealed","ceiling","pin"}` in
`metadata["layer"]`, and maps `NOT_REGISTERED` to `"sealed"`.**
§1A.2's seam names the layers forbid, mode, permissions, ceiling and pin. `NOT_REGISTERED` comes from the registry
lookup, and the generic `Executor` cannot know that its registry is sealed. To flip this gate, the executor has to
special-case `SealedToolRegistry`, or label every unregistered name in every run (and in `actions.jsonl`) as
`"sealed"`. That is a misleading vocabulary.
Also, `metadata` is the untyped "caller-facing extras" dict on a frozen `ToolOutput`. A refusal's kind, layer and
effect form one record. Pinning `metadata["layer"]` makes a stringly-typed side channel into the contract before
the notes chose one.
Fix: the notes should fix the closed layer vocabulary, for example
`registry | mode | permissions | ceiling | pin | offline`, and the field that carries it (a typed refusal on
`ToolOutput`, or the `metadata` key). The gate then follows. The log test's `_refusal()` (`"layer": "pin"`) and
§1A.8's report example must use the same vocabulary.

**S5. Serialising `error_kind`: NAME in the report and log, `.value` in `actions.jsonl`.**
`test_report_schema_and_status`, `GATE_KIND_NAMES` and `test_advisor_log_red_gates._refusal` pin `"POLICY_PIN"`,
the member name. §1A.2 says `ActionRecord.error_kind` is "the enum `.value`", and today's values are lowercase
(`"permission_denied"`). One enum would then reach AvServ and pulse in two spellings. This contradiction is in the
notes, not between test files. Settle it before item 3 ships the `actions.jsonl` field, and say so in the gate
docstrings.

**S6. `test_report_status_rows[deadline]` contradicts §1A.8's status precedence.**
`FakeBackend("deadline")` returns half the JSON text with `deadline_hit=True`. That text is also schema-invalid, so
it meets the `error` row as well. §1A.8 says `error > abandoned`, so a faithful implementation reports `error`, and
the gate demands `abandoned`.
Fix: make the deadline case return a complete, valid completion with `deadline_hit=True`, which tests the row alone.
Alternatively, the notes should state that a deadline stop short-circuits parsing.

**S7. `test_registration_sweep_b` requires that `SealedToolRegistry` has `ToolRegistry`'s private writers
(`_register_scene_locked`, `_deactivate_scene_locked`) and that they raise.** §4.2 makes the sealed registry a
separate class that satisfies a `ToolRegistryLike` Protocol, "with no inherited writer". Requiring `hasattr` for
`ToolRegistry`'s private internals makes the sealed class mirror them. A writer added to `ToolRegistry` later is not
in the hand-written list, so the sweep goes stale without anyone noticing.
Fix: derive the writer set from `ToolRegistry` by introspection (public mutators, plus `_*_locked`). Require each to
be absent from the sealed class, or to raise `SealedRegistryError`. Require `hasattr` only for the Protocol's public
writers.

**S8. `test_item3_tool_policy_is_sealed` inherits a design that cannot be built: `ToolPolicy.pinned` is both a
factory and a slot.** §3 gives `ToolPolicy` a `pinned(instances, *, ceiling, reason)` factory and a
`pinned: Mapping` field, on a `__slots__` class. The gate calls `ToolPolicy.pinned(source, ...)` and then reads
`p.pinned["pin_a"]`. Python rejects this at class creation:
`ValueError: 'pinned' in __slots__ conflicts with class variable` (verified). The only way to satisfy it is a
custom descriptor that is a classmethod on the class and a slot on the instance, which is a worse shape for a
security type. `test_advisor_gate` (§4.2 text) and `test_item3_dispatch_checks_the_object_it_runs` assume the same
factory name. Fix the notes by renaming the factory (`pinned_to(...)` or `pin(...)`) or the field (`pins`), then
update both files.

**S9. `test_item3_dispatch_checks_the_object_it_runs` asserts `len(lookups) == 1`, which pins incidental
structure.** The behaviour the notes want ("the checked object is the object run") is already proven by
`not b_ran` and `ran == ["race"]`: the swap returns B on every later lookup. The count also fails a correct executor
that does a harmless second `find` after the run (for embodiment metadata or Mechanism-B). It also cannot tell an
executor that looks up through some other path (zero counted lookups) from a vacuous probe. Fix: drop the `== 1`,
and `require(lookups)` so that a probe nobody reached is `ScenarioBroken`.

**S10. The real-backend parametrisations (`[real]`, `test_special_token_defang*`) will hard-fail in the main suite
after the flip.** §1A.10 says "fail, never skip". The repo's mechanism for model-dependent tests is the
`requires_model_cache` marker plus the nightly lane's roster (`scripts/lane_rosters/model-cache.json`), which makes
a skip visible, and the main `tests` job does not install `llama_cpp`. Mark those parametrisations
`requires_model_cache` and add them to the roster in the item-1 PR, so "never silently skipped" is enforced by the
roster and not by a red main job. Separately, `_vocab_files()` passes with any one vocab. `PINNED_VOCABS` is
declared but never enforced, so §4.2's four-vocab sweep can shrink to one. Assert that all four are present.

**S11. Latch reset is opt-in per test, duplicated in two files (`_no_latch_leak` in `test_advisor_gate.py` and
`test_advisor_log_red_gates.py`).** §1A.1 puts the latch reset in conftest. An audit hook cannot be removed, so after
one in-process `open()` the pytest process keeps the hook. If a later test calls `open()` without requesting the
fixture, every later socket or spawn test in the session is refused, far from the cause. Fix: one autouse reset in
`tests/conftest.py`, landing with step 0b. Until then, keep a single shared fixture, not two copies.

**S12. The profile-reader findings: one belongs here, the other should be filed as its own defect.**
- `models.download LLM_MODELS` (`test_profile_readers_v9.py`) belongs here. It is the import-order hole that 0a's
  `llm_models()` accessor exists to close, and it flips with 0a.
- `doctor VRAM projection (alias)` should be filed separately. It is a pre-existing defect that has nothing to do
  with user profiles: `doctor/checks.py::_check_vram_context_pressure` does
  `_BUILTIN_PROFILES.get(str(profile_value))` with no normalisation, so the **built-in** alias `mistral` fails too.
  §1A.1 says "`get_profile(name)` does not normalise; callers normalise... so the doctor ... lookups are unchanged."
  Step 0a therefore does not flip this, and it becomes a strict xfail with no owner.
  Fix: file the doctor issue, give it its own red gate that uses a built-in alias, and cite the issue in its reason.
  Keep the V9 parameter on the canonical name only.

**S13. `test_atonia_open_items_other.py::test_o8_create_pool_wires_an_llm_or_raises` and
`::test_o8_run_turn_llm_failure_is_typed` treat `except Exception: return` as a pass.** Once the decorator comes off,
any `ImportError` or `TypeError` from a broken builder satisfies the guard for good. Narrow both to a typed
exception (an `LLMUnavailableError`-style class, or `ConfigurationError`), and pin it in the seams list.

**S14. Requirements in the notes with no gate (§1A.10 and the item sections):**
- V7: `policy.ceiling & OFFLINE_CEILING` at dispatch and in `permits()` when `is_offline()` (§1A.2).
- `GATE_REFUSAL_KINDS` pinned as a set, and the AST test that its members are constructed only in
  `runtime/executor.py` (§1A.2).
- `record_outcome(refused=)` keyword-only and required. The booking gate tests behaviour only.
- The literal-call-site check for the pinned helpers that `_Reach` skips (§2.1). Without it, skipping
  `require_optional_dependency` is unguarded.
- The §1A.11 name-set vs `Effect` agreement test.
- Entity generation all-or-nothing, the structured WARNINGs (undeclared tool dropped below ALL, same-name
  replacement under ALL), and the `permits()` roster applying the ceiling for a normal run.
- `session_internet_lowered()` returns True when offline. `scripts/_evidence_records.py::unknown` refuses
  `"unknown: offline"`.
- §1A.7's composed red gate ("an offline sim ... writes `report.json` with `executed_git_hash: "unknown: offline"`").
  `test_offline_code_stamp_says_offline` drives the private `_code_stamp` alone, a hand-composed piece and not the
  report writer.
- 0a's `__all__`-vs-lazy-table pin.
- §1A.10's in-process unit tests of every audit-hook branch and `engage()`. Every latch gate is in a subprocess,
  which coverage cannot see, so the implementing PR will hit the 80% diff gate unless these are owed explicitly.
- §1A.4's declared form (`@maxim.tool(effect=Effect.READ)` runs under `maxim.run`).

The censuses are PR-time lints by §1A.10, so their absence here is correct. Each item above should be added as an
existence-first strict xfail, or listed in the PR body as owed by the implementing PR.

## NIT

- `test_atonia_red_gates_effect_policy.py::_Reach` is a call-graph analyser of about 300 lines inside a test. Its only
  self-check is "found BashTool", and a resolver bug hides sinks silently. Give it a canary per sink family (a fixture
  module per family), or move it to a `scripts/lint_*.py` with its own unit test, in line with §1A.10's census
  pattern.
- `test_item3_composite_spawn_builds_narrowed_child` reads the child executor as `loop_calls[-1][5]` (a positional
  index), and the stub takes `*a` only. A switch to keyword arguments would give `IndexError`, a hard fail. Capture
  `**kw` too, and look up `executor` by name.
- `test_o1_*` reads `t._allowed_dirs` and `executor.policy`. Both are reasonable, but the notes should name
  `Executor.policy` as a public read-only attribute, since the gate makes it one.
- `test_special_token_defang*` adds `vocab_only=` to `open_closed_local_backend` and patches `_token_attr`. That
  widens the "exactly three calls" closed surface (§1A.10). State the test-only parameter in the notes.
- `hook_installed()` becomes public `_offline` surface so that a test can use it. That is acceptable, since it is
  backed by the behavioural audit probe.
- `test_advisor_gate.py::test_executor_import_closure_step0a` and `::test_language_config_import_does_not_read_profiles`
  are step 0a gates, a separate PR, and they sit in the item-1 file. Moving them to a 0a file keeps each PR's gates
  in that PR's file. Likewise, the AST-only and config tests in `test_offline_latch.py` are unit tests that live
  under integration.
- `test_o4_*` pins "default deny with no allowlist" for Session H's O4. That is a behaviour choice for every comms
  user, and the owning plan should make it. Confirm it with H, or keep it in H's test file.
- `ScenarioBroken` and `require` are defined seven times. A shared `tests/_redgate.py` would make the pattern
  lintable (every `xfail(strict=True)` has `raises=AssertionError`).
- The booking docstring says the parallel batch writes no autonomy audit entry today. §1A.2's "Today" column says
  "booked". Fold the correction into the notes.
- Child scripts use `except ImportError: pass` for optional pre-imports. The no-swallow lint covers `src/` only, but
  writing the miss into `OUT` would cost nothing.
