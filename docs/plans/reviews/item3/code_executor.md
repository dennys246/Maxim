# #1315 item 3: Executor lens code review

Diff: `git diff origin/main...HEAD` on `feat/1315-item3-tool-policy` (e2f139af, 06af7bca). Reviewer: Executor lens
(docs/CODE_REVIEW.md). Probes: `/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/248fdc0d-2f2e-471a-b37d-dacb96a7ecce/scratchpad/review3_exec/p1.py`–`p4.py`.

Run on the branch: the four item-3 test files (`test_tool_policy.py`, `test_refusal_booking_red_gate.py`,
`test_tool_policy_census.py`, `test_atonia_red_gates_effect_policy.py`) pass, 87/87. Every test file the diff touches,
run together with `-m "not slow"`: **8 failed, 1749 passed, 15 xfailed** (all 8 failures are in
`test_agent_permissions.py`, E1).

## Findings

### E1. BLOCKER (red suite): `tests/unit/test_agent_permissions.py` still builds `Executor(..., permissions=)`

- **Where:** `tests/unit/test_agent_permissions.py::TestExecutorPermissionGate` (4 tests) and `::TestKindSelector`
  (4 tests), at lines 130, 136, 143, 156, 304, 319, 329 and 335.
- **What happens:** the diff edited this file (`TestPolicyFromToolsSettings`) but left eight executor constructions
  on the deleted keyword. They fail with `TypeError: Executor.__init__() got an unexpected keyword argument
  'permissions'`, plus one `missing 1 required keyword-only argument: 'policy'`.
- **Evidence:** the narrow run above lists all eight `FAILED` lines. This is the first file `-x` stops on, so CI is red.
- **Why it matters beyond CI:** these are the oldest tests of the name gate: deny, allow-list, alias-then-canonical,
  and the `kind:` selectors resolved from the registry. Until they are ported, nothing pins
  `Executor._permissions_refusal`'s alias and selector behaviour under the new seam except the newer tests.
- **Fix:** port them to `policy_for(perms)` / `TEST_POLICY` from `tests/policy_helpers.py`. Keep every assertion.
  `test_no_permissions_runs_tool` becomes "unrestricted runs the tool".

### E2. SHOULD-FIX, fail-OPEN (learning): a tool body can return a gate kind and skip dispatch's NAc booking

- **Where:** `runtime/tool_dispatch.py::gate_refusal_of`, which trusts `result.error_kind` alone, and
  `runtime/executor.py::Executor._run_started`, which passes a tool's own `ToolOutput` through untouched.
- **The claim this breaks:** `tools/base.py::GATE_REFUSAL_KINDS`'s docstring and §1A.2 ("A tool body returning `MODE`
  cannot exempt itself from NAc").
- **What the guard covers:** only the static AST test
  `test_atonia_red_gates_effect_policy.py::test_item3_gate_kinds_constructed_only_in_executor`, which scans
  `src/maxim`. It does not see:
  - a third-party or `pymaxim` extension tool;
  - a composite that returns a child executor's refused `ToolOutput` verbatim (no `ToolErrorKind.<GATE>` token
    appears in its source);
  - `ToolErrorKind(some_variable)`.
- **Scenario:** a tool whose `execute` returns `ToolOutput(success=False, error_kind=ToolErrorKind.MODE)` RAN. The
  executor treats it as a ran failure: `record_tool_start` runs, `_fail_ran` runs and the bridge books. Dispatch then
  sees `gate_refusal_of(result) == MODE`, so `record_outcome` drops NAc, goal credit and cluster reward. The guarantee
  is broken, and one failure is now booked half by the bridge and half not by dispatch.
- **Evidence (p1.py):**
  ```
  result kind: ToolErrorKind.MODE metadata: {} gate_refusal_of: ToolErrorKind.MODE
  bridge.record_tool_start called: True
  NAc observe calls for a tool that RAN and failed: 0 []
  ```
- **Fix:** enforce it at runtime, at the seam that knows. In `_run_started`, after `tool.run`, a returned result
  whose `error_kind in GATE_REFUSAL_KINDS` is re-typed (`EXTERNAL_FAILURE`, or `None` plus a WARNING). Only
  `_refused_output` can then produce a gate kind on the way out of the executor. A cheaper complement is to have
  `gate_refusal_of` also require the executor's own stamp. But `metadata["refusal_layer"]` alone is forgeable by the
  same body, so the re-typing is the real fix. Add a test: a body returning each gate kind still books NAc.

### E3. SHOULD-FIX, fail-OPEN: `Executor.policy` is a plain writable attribute, so V3's type check runs only once

- **Where:** `runtime/executor.py::Executor.__init__` (`self.policy: ToolPolicy = require_policy(...)`).
- **What happens:** `ToolPolicy` is sealed, but the executor's slot holding it is not. Ordinary code (no
  introspection) can widen a locked executor, or swap in a duck-typed stand-in that `require_policy` would have
  refused. The seam reads only `.permissions`, `.ceiling` and `.pins`. No production code assigns it today (grep:
  zero `\.policy =` hits on executors), so this is latent. But the module docstring promises "unforgeable from
  ordinary code", and the slot is the weaker link.
- **Evidence (p3.py):**
  ```
  before: ToolErrorKind.POLICY_PERMISSIONS
  after duck reassign: success= True net
  ```
- **Fix:** store the policy as `self._policy` behind a read-only `policy` property. Optionally also run
  `require_policy` in `effective_ceiling` and `_permissions_refusal`, so a stand-in fails closed at dispatch. Wrappers
  that use `__getattr__` keep reading it.

### E4. NIT, fail-closed: `Executor.check` says "never raises" but a raising forbid source or mode source propagates

- **Where:** `runtime/executor.py::Executor.check`.
- **Evidence (p2.py):** `check RAISED with a raising forbid source: forbid boom`.
- **Effect:** dispatch raises in the same case, and `auto_sense` wraps both call sites in `except Exception` (logged),
  so this is fail-closed. Only the docstring is wrong. Either drop "Never raises", or catch, log and return a refusal.

### E5. NIT, fail-closed: under a pinned policy `permits(alias)` is True, but dispatch refuses the alias (layer `pin`)

- **Where:** `runtime/executor.py::Executor.permits`. It runs `_object_refusal` only when `registry.find(name)` finds
  the name; an alias is not registered, so it falls through to True.
- **Evidence (p2.py):** `pinned: permits('recall') = True | execute recall -> ToolErrorKind.POLICY_PIN`.
- **Effect:** aliases are not advertised, so the roster does not offer one. But `permits()` is documented as "would
  let *tool_name* run". Fix: in `permits`, return False for a name that `resolve_tool_name` maps elsewhere when
  `policy.pins is not None`.

### E6. NIT, fail-closed: a pinned policy refuses an unregistered or inactive name as `registry`, not `pin`

- **Where:** `ToolPolicy.pinned`'s docstring and the as-built note both say "refuses aliases and every other name
  (layer `pin`)". An unregistered name under pins actually returns `NOT_REGISTERED` / `"registry"`, because
  `_object_refusal` runs only after `registry.get`.
- **Effect:** harmless, since both are gate kinds and book the same way. Align the docstring with the code (or the
  code with the docstring).

### E7. NIT, fail-closed: `narrowed(only=...)` over a `kind:` allow-list intersects strings, not meanings

- **Where:** `runtime/tool_policy.py::ToolPolicy.narrowed`.
- **Evidence (p2.py):** `ToolPolicy.only(["kind:sense"]).narrowed(only=["look"])` has allow `frozenset()`. That
  refuses everything, even when `look` is of kind `sense`.
- **Effect:** never wider, so it is safe, but surprising. The docstring already says "``kind:`` selectors are entries
  like any other". It could add one sentence saying that the result is the empty list.

### E8. NIT, fail-closed: `SpawnSubSimulationTool` hands a pinned parent's pins to a child whose tools are fresh instances

- **Where:** `simulation/tools.py::SpawnSubSimulationTool.execute`. It calls `parent_policy.narrowed()`, and
  `narrowed` keeps `pins`. `_run_sub_simulation` then builds a new registry.
- **Evidence (p2.py):** `child run of fresh memory_recall: ToolErrorKind.POLICY_PIN`.
- **Effect:** no shipped dispatcher of the spawn tool is pinned (the narrator and the AUT are unrestricted), so this
  is latent. A future pinned parent would get a child that can run nothing.
- **Related, pre-existing and out of the hunks:** the child does not inherit the parent's MODE or autonomy forbids.
  `sub_state.data["mode"] = "active"` is hard-coded, and `SupervisionPolicy` allows bash and write_file. A parent
  under a narrower mode spawns an "active" child, so "never wider" holds for the ToolPolicy only. Worth one line in
  §3's as-built block.

### E9. NIT: the census's direct-call scan misses `auto_sense`'s interoception call

- **Where:** `tests/unit/test_tool_policy_census.py::_tool_receiver` keys on a receiver name containing "tool".
  `runtime/loop_perception.py::auto_sense` calls `_sense.execute(entity_name=...)`, and `_sense` has no "tool" in it.
  So the census neither pins nor sees it (grep over `src/` for `.execute(` receivers: it is the only miss).
- **Effect:** the `executor.check(_sense)` guard exists and has its deletion probe, but a NEW direct call named like
  this would pass the census. Fix: also match receivers bound from `registry.get(...)` / `_tool_reg.get(...)`, or add
  `_sense` explicitly, and pin `("runtime/loop_perception.py", "auto_sense", "_sense.execute")`.

### E10. NIT: if the parallel batch's new audit write raises, the outcome is mislabelled

- **Where:** `runtime/tool_dispatch.py::execute_parallel_actions` / `_audit_parallel_action`.
- **What happens:** `_audit_parallel_action` runs after the result is appended but before `if not success:
  all_succeeded = False`. If `log_action` or `operational_mode` raises (p4.py, a mode accessor returning a
  non-str), the `except` logs "Parallel action x failed: ..." for an action that ran, and `all_succeeded` stays True
  for a failed action. `all_succeeded` is only logged, and the entry still books correctly, so the impact is a
  misleading log only.
- **Fix:** move `_audit_parallel_action` after the `all_succeeded` update, or give it its own try that logs.

### E11. NIT (unlisted sink, for the owner): gate refusals still reach `capture_loop_action` (Hippocampus episode)

- **Where:** `runtime/tool_dispatch.py::execute_and_learn`. It calls `capture_loop_action` for every returned
  result, refused ones included.
- **Why it might matter:** §1A.2's booking table does not list the episodic capture, so a refused action is stored as
  an episode with `success=False` and gets the `observe_episode` salience. It is not NAc, pain or credit, so it
  probably counts as "carryover". But it is a sink the red gate does not spy on. Confirm whether that is intended.

## Checked and found correct

- **Dispatch order:**
  - the pre-alias deny-only check (forbid + deny);
  - alias resolution; under pins, an alias is refused (`pin`);
  - the full name gate (forbid → mode → permissions);
  - inactive scene;
  - the ONE `registry.get`;
  - `_object_refusal` on that object (permissions on its `kind` → effective ceiling → pin);
  - then `record_tool_start`;
  - then `tool.run` inside `dispatching(policy)`.

  No refusal path calls `record_tool_start`, `_report_failure` or `_fail_ran`. The ceiling refusal probe shows only
  `pop_invocation_*` / `finish_invocation` on the bridge.
- **`permits()`** applies the object gates for registered names, the offline AND included (`permits(net)` is False
  under ceiling READ).
- **`OfflineRefusal`:**
  - `Tool.run` re-raises it;
  - it is the only caller of `Tool.run` in `src/`;
  - `_run_started` maps it to `OFFLINE` before any booking;
  - with the latch off and a bridge, the bridge gets only `record_tool_start` plus retirement, no failure;
  - with the latch on, `_suppress_nac` is set, so the start is skipped too.
- **`record_outcome(refused=)`:** `_learning_nac` zeroes NAc for every gate kind, and every learning sink in
  `record_outcome` (observe, goal credit, cluster reward, operant pending, energy) sits under `nac is not None`. The
  plan outcome is guarded at its one site. All six call sites pass `refused=`. `book_agent_fallback_failure` gets a
  per-action `result = None` (agent_loop resets it before the try), so it cannot see a stale earlier stamp.
- **`ToolPolicy` sealing (p1.py):**
  - set, delete, pickle, the public constructor and subclassing all refuse;
  - copy and deepcopy return self;
  - `AgentPermissions` is frozen and `pins` is a `MappingProxyType` snapshot.

  `ToolPolicy.unrestricted.__func__(FakeCls, ...)` leaks `_KEY` (p1.py: `key leak via classmethod __func__: True`).
  That is introspection, out of the §0 threat model as the docstring states, and noted only for completeness.
- **`narrowed`** never widens: ceilings are ANDed, allow-lists intersected, denies unioned.
- **`current_policy()`** is set only inside dispatch, reset after it, and None in a spawned thread (fail-closed, as
  documented). `SpawnSubSimulationTool` reads it synchronously and refuses when it is None.
- **`auto_sense`:** both direct calls are gated by `check()`. A refused auto-fire tool returns its refusal, which is
  not shown, and is not counted.

## Verdict

NOT READY. One BLOCKER (E1, eight red tests in a file the diff touched) and two fail-open SHOULD-FIXes (E2: a tool
body can type itself out of NAc; E3: a writable `Executor.policy` slot). The seam itself is otherwise correct, and I
found no path that runs a tool without the gates.
