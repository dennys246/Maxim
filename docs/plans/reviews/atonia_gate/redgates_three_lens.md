# Three-lens review: the atonia gate's strict red gates (commit 0ac02acb)

**Scope.** `git diff origin/main...HEAD`: seven test files, 4,563 lines, tests only. The design they must match is
`docs/plans/atonia_gate_approach_notes.md` v6.2, where §1A wins over §2–§6. All three lenses (docs/CODE_REVIEW.md) were
applied, and each finding names its lens. Every finding marked *measured* was reproduced from a scratch directory
outside the worktree. Nothing in the worktree was edited except this file.

**Verdict.** 3 BLOCKER, 13 SHOULD-FIX, 12 NIT. The harness is mostly sound:
- preconditions raise `ScenarioBroken`;
- the children are isolated;
- the spies block egress, and are never the pass.

Every behavioural gate that fails today fails for the reason its docstring names (*measured*, with `--runxfail`).
The blockers are three gates that a build which follows the notes cannot flip. A gate like that would sit red as
"data" for the wrong reason.

Run as committed: 141 xfailed, 10 passed, 35 s of test time (43 s wall) on macOS.

---

## BLOCKER

### B1 [Executor + Architecture] `tests/integration/test_advisor_gate.py::test_env_read_set` (both backends) cannot flip on a build that follows the notes

`ENV_READ_ALLOW` holds only the refusal list and four numpy names. But the notes require the following:
- `maxim/__init__` reads `MAXIM_OFFLINE` on every import (§6 "Self-engage in children").
- `is_offline()` re-reads it on every call (§6, "Engaging the latch": "engages on any call that sees `MAXIM_OFFLINE`
  truthy").
- The observer always engages, and `engage()` sets `HF_HUB_OFFLINE` (§6 layer 2). Spawn children "inherit
  `MAXIM_OFFLINE=1`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `YOLO_OFFLINE=true`" (§6 sub-processes, V5).

So a correct observer reads `MAXIM_OFFLINE`, which falls outside `ENV_READ_ALLOW`. If `engage()` exports the latch
variables as §6 says, those writes fall outside `ENV_WRITE_ALLOW` too.

**Scenario:** item 1 and step 0b land exactly as specified, and the gate stays XFAIL indefinitely.

**Fix:**
- Add `MAXIM_OFFLINE` to the read allow-list.
- Settle the notes' internal contradiction: §1A.1 says "`engage()` writes nothing", while §6 says `engage()` sets
  `HF_HUB_OFFLINE` and children inherit the latch variables. Then pin the latch's own writes, or the absence of them,
  explicitly in `ENV_WRITE_ALLOW`.

### B2 [Executor] `tests/integration/test_advisor_gate.py::test_no_effect` (both backends) flags the notes' own atomic writer

`_outside()` treats a relative name as legal only if `report_dir/<name>` still exists after the run. §1A.3 makes the
report's writer `utils/atomic_io.py::atomic_write_json_at(dir_fd, name, payload)`, and §4.2 says "the log and the
report are written through that fd". An atomic write opens a temporary file under a `dir_fd`-relative name and then
renames it. That temporary name no longer exists after the run. So two checks fire on correct code:
- `open_write` is flagged as "outside report_dir";
- `os.rename` (whose first argument is the temporary name) is flagged as a "filesystem mutation outside report_dir".

**Scenario:** item 1 is correct, and the gate never flips.

**Fix:** a `dir_fd`-relative name cannot be resolved from the audit event. Two options:
- run the child with an unwritable CWD (`chmod 0555`), so any CWD-relative write fails loudly, and accept relative
  names when the CWD tree diff is clean;
- or record the `report_dir` fd in the child and allow relative names only through it.

### B3 [Architecture] `tests/integration/test_advisor_gate.py::test_o3_no_approval_surface` forbids every `maxim.agents.*` module

The notes' design puts `maxim.agents.permissions` in the observer's closure:
- `ToolPolicy` "holds `permissions: AgentPermissions`" (§3) and lives in `runtime/tool_policy.py` because it "holds
  `AgentPermissions` (`agents/`)" (§1A.3);
- the sealed executor is built with `ToolPolicy.pinned(...)` (§4.2).

§4.2's forbidden list names only `agents.llm_worker`, and O3 concerns `agents.autonomy`. The `LAYER_RULES` fence in
§1A.3 restricts what `maxim.advisor` imports directly, not its transitive closure.

**Scenario:** item 1 lands as designed, and the gate stays red indefinitely.

**Fix:** forbid `maxim.agents.autonomy` (plus `FORBIDDEN_MODULES`), not the whole `maxim.agents.` prefix.

---

## SHOULD-FIX

### S1 [Executor] O8 gates can XPASS for the wrong reason
- **Gates:** `tests/unit/test_atonia_open_items_other.py::test_o8_create_pool_wires_an_llm_or_raises` and
  `::test_o8_run_turn_llm_failure_is_typed`.
- **The problem:** both treat *any* `Exception` as "documents and raises" or "typed error".
- **Scenario:** step 0a's lazy inits are the notes' own named risk ("lazy inits can surface latent cycles"). If 0a
  makes `from maxim.mesh.bus import LocalMessageBus` raise `ImportError` or `AttributeError`, or a later refactor
  makes `AgentPool(...)` or the `SimpleNamespace` stand-in raise `TypeError` or `AttributeError`, the gate XPASSes.
  Strict mode then prompts removing the decorator, and the test passes vacuously from then on.
- **Fix:** catch only the typed error the fix introduces. At the least, re-raise
  `ImportError`/`AttributeError`/`TypeError`/`NameError`, and require the message to name the LLM.

### S2 [Executor] `tests/unit/test_atonia_open_items_other.py::test_o4_send_message_refuses_an_unlisted_recipient` is a bare negative assert
- **The problem:** `Tool.run` (`tools/base.py`) catches every exception into `ToolOutput(success=False)`.
- **Scenario:** a crash in `SendMessageTool.execute` or the gateway, or a refactor that refuses every send, leaves
  `channel.send` uncalled, and the gate XPASSes.
- **Fix:** add the positive control (an allow-listed recipient IS delivered), and assert that the result is a
  recipient refusal.

### S3 [Executor] `pytest.raises` inside `xfail(raises=AssertionError)` gates
- **The problem:** `DID NOT RAISE` raises `Failed`, whose MRO is `Failed → OutcomeException → BaseException`. It is
  not an `AssertionError`, so a partial build turns the suite **red** instead of leaving the gate as data. *Measured*
  with a one-test probe.
- **Inconsistency:** `test_atonia_red_gates_effect_policy.py` documents this trap and uses `_expect_raises`; the other
  files do not.
- **Sites:**
  - `test_advisor_gate.py`: `::test_builder_refusals_code_loading_env`, `::test_builder_refusals_model`,
    `::test_builder_refusals_dirs`, `::test_registration_sweep_b`;
  - `test_advisor_log_red_gates.py`: `::test_create_refuses_planted`, `::test_append_one_ascii_line_per_record_no_nan`,
    `::test_append_oserror_marks_broken`, `::test_read_log_rejects_malformed`.
- **Fix:** use `_expect_raises` in these files too.

### S4 [Executor] `tests/integration/test_offline_latch.py::test_engaged_latch_refuses_each_channel[af_unix_connect]` cannot flip on macOS
- **The problem:** the socket path is `tempfile.gettempdir()/no.sock`, under pytest's resolved tmp directory. On this
  Mac that is 107 characters, and macOS caps `sun_path` at 103 usable bytes. CPython's `sock_connect` runs
  `getsockaddrarg` **before** `PySys_Audit("socket.connect")`, so `OSError: AF_UNIX path too long` is raised before
  the latch can see the connect. *Measured.*
- **Scenario:** CI on Linux XPASSes while a Mac developer sees XFAIL on the same code, and the two disagree.
- **Fix:** use a short absolute path (for example `/tmp/mx-<pid>.sock`), or `chdir` and a relative name.

### S5 [Executor + Wire] The census gate over-specifies one site and under-covers the notes
`tests/integration/test_offline_latch.py::test_no_effect_subprocess` and `CENSUS_SITES` have two problems.

**The encoder site cannot flip as specified.** `similarity.encoder[missing model]` must raise `OfflineRefusal` or grow
the ledger. But §6 handles HF with `local_files_only`, `HF_HUB_OFFLINE` and a "degraded" record ("The fallbacks
become loud"), none of which is an `OfflineRefusal`. The encoder also swallows its load error: *measured*, it raises
nothing and logs "Failed to load encoder model". So a build that follows the notes stays red on this site. When
`sentence_transformers` is absent, the site is silently `not_applicable` instead.

**Sites the notes name are missing**, so the gate can flip while they are unrefused:
- the `ddgs` / `_search_duckduckgo_lite` urllib3 bypass;
- ultralytics (`YOLO(name)`, `attempt_download_asset`);
- faster-whisper;
- a direct HF hub call ("must raise `OfflineModeIsEnabled` with no socket event");
- twilio;
- `rtm_engine`;
- the Anthropic and OpenAI SDKs.

**Fix:** accept "degraded" or `OfflineModeIsEnabled` with zero socket events for HF-backed sites. Add drivers for the
named sites, or state in the docstring that the list is partial until the JSON site census drives it.

*Measured on the positive side:* with a no-op latch, every one of the 8 drivers reached the spy and was blocked, and
nothing egressed outside the driven windows. The harness itself is non-vacuous and safe.

### S6 [Wire] `tests/unit/test_refusal_booking_red_gate.py::test_a_gate_refusal_books_no_learning` omits three gate kinds
- **The problem:** §1A.2 requires the gate to cover "each kind in `GATE_REFUSAL_KINDS`", and says of the
  `OfflineRefusal`-in-`_fail_ran` ordering case: "The behavioural gate drives that case too". The file drives only
  mode, permissions, unregistered and inactive-scene.
- **Scenario:** `POLICY_CEILING`, `POLICY_PIN` and `OFFLINE` are never driven. So the item-3 branch can flip all 8
  cases green while `_fail_ran` still books a `ToolPainBridge` NAc negative for an `OfflineRefusal` raised inside a
  tool, which is the exact ordering bug §1A.2 names.
- **Fix:** add the three kinds behind an existence assert. Also name which `ToolErrorKind` the inactive-scene refusal
  maps to: it is not in `GATE_REFUSAL_KINDS` as written, yet the gate demands zero writes for it.

### S7 [CI fit] Two parametrised gates can never be accepted as release-gate evidence
- **Gates:** `tests/unit/test_refusal_booking_red_gate.py::test_a_gate_refusal_books_no_learning` and
  `tests/unit/test_profile_readers_v9.py::test_every_public_reader_resolves_a_user_profile`.
- **The problem:** their `xfail(strict=True)` is built inside helper defs (`_case`, `_reader_param`).
  `scripts/lint_release_gates.py::_is_strict_xfail` sees only the def's own decorators. *Measured:*
  `red_gate_problems(..., commit=0ac02acb)` returns "did not carry xfail(strict=True) — not a red gate" for both
  (`test_no_effect_subprocess` passes the same check).
- **Consequence:** if the gate's release gets a `release_gates.json` entry, these cannot be its `red_gate` evidence.
  Rule 3a's static check also passes them today, despite every case being xfailed. That is a lint blind spot that
  only rule 6's runtime check catches.
- **Fix:** give each case its own def with a literal decorator, or extend the lint to see through these helpers.

### S8 [Wire] `tests/unit/test_profile_readers_v9.py::READERS` leaves out lane resolution
- **Missing readers:** §1A.1 lists "lane resolution" among the public readers the outcome gate must cover, and v6.2
  says this gate, not the table, is the enforcement. Neither `lane_models`/`lane_backends` resolution nor `llm_agent`,
  `download_llm`/`delete_llm` or `cli_utils` (which reach the private `_normalize_profile`) is driven.
- **A weak assertion:** `ensure_available lookup` asserts only the *absence* of the substring
  `"is not in the LLM_MODELS"`, so a reworded message passes vacuously.
- **Fix:** add a lane-resolution reader. For `ensure_available`, assert a positive outcome: the profile is found and
  reported as not downloaded.

### S9 [Architecture] `tests/integration/test_advisor_gate.py::test_no_effect_c_level_disk_and_inbound` bundles item 1's framing with item 7's Metal cache
- **The problem:** §4.2 makes framing an item-1 obligation: each untrusted segment is "already framed by
  `frame_tool_output(..., external=True)`". The "inbound" that §4.2 leaves to item 4 is **ingress**
  (`bind`/`listen`), not framing.
- **Scenario:** item 1 ships correct framing, but the test stays red on the item-7 cache leg. A framing regression in
  item 1 is then hidden behind that unrelated red.
- **Fix:** move the framing assertion into an item-1 gate. Keep this one for the C-level cache only.

### S10 [Architecture: claims about the wiring] The notes' regression-guard nodeids do not exist
§1A.9's draft invariant stubs and §1A.1 cite these nodeids:
- `tests/unit/test_tool_effect.py::test_every_tool_declares_own_effect` and `::test_process_and_code_runners_declare_all`;
- `tests/unit/test_tool_policy.py::test_executor_refuses_missing_policy`;
- `tests/unit/test_offline_latch.py::test_offline_module_is_stdlib_only`;
- `tests/integration/test_advisor_gate.py::test_per_layer_deletion_sweep`.

None exists. The red gates live under:
- `test_atonia_red_gates_effect_policy.py::test_item2_*` and `::test_item3_policy_is_required_and_permissions_removed`;
- `tests/integration/test_offline_latch.py`;
- `::test_per_layer_deletion_sweep_{ceiling,pin,sealed}`.

When those stubs move into the briefs, every `Regression guard:` line will name a missing test.

**Fix:** fold the real nodeids into §1A.9 in this PR, or rename the tests to match.

### S11 [CI fit] Run time, and an unspied child
- **The two heavy tests:**
  - `tests/unit/test_atonia_open_items_other.py::test_o8_server_mode_not_ready_is_reported` takes 15–26 s in the
    unit fast suite. *Measured:* the child spawns `nvidia-smi`, `uname`, `fc-list` and `system_profiler` (matplotlib
    builds a fresh font cache under its temporary `MPLCONFIGDIR`), binds a socket, and carries **no** egress spy.
  - `tests/unit/test_atonia_red_gates_effect_policy.py::test_item2_every_tool_declares_own_effect` takes 14 s, mostly
    parsing every `src/maxim` file, and will be slower under coverage tracing.
- **Constraint:** marking either `slow` would bar it from release evidence (rule 3a denies `slow`).
- **Fix:**
  - in the O8 server child, stub `ensure_blackwell_guards` and the matplotlib import, and add a recording-and-blocking
    spy;
  - share one parsed index across both item-2 tests: `lru_cache` already does, so the cost is the first parse.

### S12 [Executor] `tests/integration/test_advisor_gate.py::test_special_token_defang` can flip on one vocab
- **The problem:** `_vocab_files()` accepts *any* non-empty set of `ggml-vocab-*.gguf`. §4.2 pins four vocabs:
  command-r, phi-3, qwen2 and llama-bpe.
- **Scenario:** a fixture commit adds one vocab, the gate XPASSes, and the other three are never checked.
- **Fix:** `require` every one of `PINNED_VOCABS` to be present.

### S13 [Executor] `tests/unit/test_atonia_red_gates_effect_policy.py::test_item3_dispatch_checks_the_object_it_runs` counts lookups
- **The problem:** it asserts `len(lookups) == 1` and `ran == ["race"]`. But §3's own order is "Resolve the alias.
  Get the tool once. Check. Run". Today's alias resolver (`resolve_tool_name`) and the `kind:` permission check
  (`_permission_denial`) each call `registry.find`.
- **Scenario:** a build that keeps `find` for alias resolution and `get`s once is safe: it checks and refuses the
  swapped-in B, and nothing above the ceiling runs. Yet it fails both asserts.
- **Fix:** the security property is "`b_ran` is empty, and the object that ran (if any) is the object that was
  checked". Assert that, and drop the lookup count.

---

## NIT

1. **[Executor]** `test_advisor_gate.py::test_executor_import_closure_step0a` checks only `agent_loop` and
   `llm_worker`. V9's finding also names `default_network`, `memory` and `utils.paths`, and §1A.10 names
   `runtime.tool_dispatch`, so a partial step 0a flips this gate. Add them.
2. **[Executor]** `_no_latch_leak` (in two files) resets the latch but not the process env. If `engage()` exports
   `MAXIM_OFFLINE=1` for children (V5), the variable outlives an in-process `open()`. Every later `is_offline()` then
   re-engages, and every later test runs offline. Snapshot and restore `os.environ` in that fixture, or assert the
   variable is absent afterwards.
3. **[Executor]** The advisor child's `_blocks` only *records* AF_UNIX connects and `os.fork`/`os.forkpty`, where the
   offline child blocks them. Align the two.
4. **[Executor]** `_caller()` skips `"_collections_abc"`, but that module's `__name__` is `collections.abc`. *Measured:*
   `setdefault` reads are attributed to `collections.abc.setdefault`, so the messages are wrong and the
   `ENV_WILDCARD_ALLOW` match can miss.
5. **[Executor]** `ENV_READ_ALLOW` lists numpy's reads by name, measured on the dev venv. §4.2 allows "library reads
   such as numpy's and threadpoolctl's". Allowing by caller-module prefix would survive CI's numpy version.
6. **[Executor]** `test_atonia_red_gates_effect_policy.py::test_o2_o5_ceiling_refuses_acting_tools_the_mode_gate_allows`
   sends `send_message` a `message=` argument, while its schema says `body`. `Tool.run`'s `_validate_input` refuses
   that before `execute`, so the "body never runs" half is vacuous for `send_message`. Use the schema's keys.
7. **[Executor: safety]** The router children (`_ROUTER_*_BODY`) build `LLMRouter` before `engage()` with no spy and
   no `MAXIM_SKIP_REMOTE_PROBE`. A real loopback probe to `127.0.0.1:9` is possible. It is harmless, but it should be
   recorded.
8. **[Executor]** `test_per_layer_deletion_sweep_pin` has no `require(reads)`. If item 2 classifies no default tool
   as READ, the loop is vacuous.
9. **[Executor]** `_posix_spawn`/`_popen` use `/bin/true`, which macOS does not have. The latch refuses first, so
   this does not matter today.
10. **[Executor]** `test_item3_composite_spawn_builds_narrowed_child` reads the executor as `loop_calls[-1][5]`, a
    positional index. A switch to keyword arguments raises `IndexError` (an error, not an xfail).
11. **[Architecture: claims]** §1A.2's table says the autonomy audit entry is "booked" today for every path. *Measured:*
    `execute_parallel_actions` writes 0 audit entries for every kind. The gate forces a new `log_action` in the batch,
    so the notes should name that work item.
12. **[Executor]** `test_item3_policy_is_required_and_permissions_removed`'s first `build_executor(reg, pain_bus=None)`
    check already raises `TypeError` today (`permissions=` is required). That sub-check does not discriminate. The
    gate as a whole does, because `Executor(reg)` is accepted today.

---

## Wire integrity: the spied sinks against production (the refusal-booking gate)

| Contract | Production producer | What the test spies | Verdict |
|---|---|---|---|
| NAc tool outcome | `tool_dispatch.py::record_outcome` → `nac.observe` (two sites) | `patch.object(rig.nac, "observe")`, the same instance `execute_and_learn`/`execute_parallel_actions` receive | Real sink |
| Goal credit / cluster reward | `record_outcome` → `nac.credit_goal`, `nac.update_cluster_reward` | Same instance | Real sink |
| Pain | `Executor._report_failure` → `PainDetector.record_tool_error` → `_emit_pain` → `PainBus.publish` → `_on_pain` → `nac.record_outcome_full` (*traced* for the unregistered case) | `patch.object(rig.bus, "publish")`; the detector reads the bus attribute at call time | Real sink; `record_outcome_full` is reached only through `publish` |
| Plan history | `execute_and_learn` → `bio_integration.record_plan_outcome` → `memory_hub.record_plan_outcome` | `MagicMock(spec_set=["record_plan_outcome"])` | Real seam; it is the only path. The batch path has no `memory_hub` |
| ToolPainBridge | `Executor._tool_pain_bridge.record_tool_*` | Instance attributes on the bridge `build_executor` built | Real sink |
| Kept sinks | `llm_worker.record_outcome`, `recent_outcomes`, `AutonomyController.audit_log` | Counted | Real. The batch writes 0 audit entries today (NIT 11) |

**Residual.** The spies are method-specific. A fix that routes a refusal through an *unspied* NAc writer would let
the gate flip with a write still happening. Unspied writers include `set_pending_operant_action` (operant cluster
path, not reached by this rig) and a direct `record_outcome_full`. A before/after equality check on NAc's state, beside
the spies, would close that gap.

## Safety and CI fit: what was checked and holds

**Children:**
- Every child runs as `sys.executable -I -B` with an explicit env: HOME, TMPDIR and XDG point at tmp, and no
  HF/MAXIM variables. Each asserts its `maxim` is this checkout's `src`.
- The advisor and offline children block INET connect and send, DNS, and spawn with `_SpyBlocked` (the advisor child
  only records AF_UNIX connects and `os.fork`, NIT 3). The offline children's spy is never the pass.
- The provenance children run in the repo with the spy installed first. *Measured:* git was blocked, and nothing was
  written.

**In-process tests:**
- They use conftest's per-session test HOME and monkeypatched env.
- No test-installed audit hook enters the pytest process today.
- No docker: the census `ensure_image` is blocked by the spy.
- No real `~/.maxim` or `~/.config`.

**Lints and determinism:**
- Clean: `ruff check`/`format`, `lint_release_gates` (fails only S7's red-gate rule 3b), `lint_orphan_modules`,
  `lint_multi_agent_marker`, `lint_secrets`, `lint_fix_touches_tests`, `lint_harness_provenance`.
- Coverage is neutral, since the change is tests only.
- `tests/integration` is in the fast suite, and these files carry no `slow` marks.
- Module- and class-level `xfail` is absent everywhere, so later evidence is not blocked.
- Determinism holds, apart from S4 (macOS path length) and NIT 5 (numpy version).
