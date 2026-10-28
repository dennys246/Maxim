# Atonia gate red gates: wire-integrity lens

Reviewed: `git diff origin/main...HEAD` on `test/atonia-red-gates` (0ac02acb), seven new test files, tests only. The question
for each behavioural gate is whether it observes the real producer and consumer on the shipped path, so that it flips
when, and only when, the shipped composition is fixed. Method: I read every gate. I traced every patched or spied
symbol to its production import site, and I ran probes from scratch (listed under "Probes run").

## Producer → consumer tables, per behavioural gate

Legend: **real** = the object production calls on that path. **test-only** = no production caller composes it this
way. **stand-in** = a fake that replaces the code under test.

### `test_refusal_booking_red_gate.py::test_a_gate_refusal_books_no_learning` (8 cases)

| Contract | Producer driven | Consumer / sink spied | Real on shipped path? | Probe |
|---|---|---|---|---|
| Dispatch entry | `tool_dispatch.execute_and_learn` (loop_setup `_bind_execute_and_learn`) and `execute_parallel_actions` (agent_loop batch) | — | real (both are the shipped entry points) | grep callers |
| Outcome recorder | `functools.partial(record_outcome, drive_relief_only=False)` | NAc.observe / credit_goal / update_cluster_reward | real function. **The binding is the `substrate-primary` one**: the default `aut_mode="llm-primary"` binds `drive_relief_only=True` (loop_setup.py `_drive_relief_only`) | loop_setup.py |
| Executor build | `build_executor(registry, pain_bus=None, permissions=…, nac=, pain_detector=detector)` | `_tool_pain_bridge` (four booking methods) | real builder, **test-only wiring**: no production `build_executor` call passes `pain_detector=`. `create_full_agent`, bio_stack and agentic_runtime all pass `pain_bus=` | grep `pain_detector=` |
| Pain publish | `Executor._report_failure` → `PainDetector.record_tool_error` → `PainBus.publish` | `rig.bus.publish` | **test-only producer**: under production wiring `_pain_detector is None`, so `_report_failure` publishes nothing | probe P1 |
| Plan bridge | — | `MemoryHub.record_plan_outcome` (MagicMock `spec_set`) | stand-in sink, acceptable: the spy is the method name the shipped code calls | — |
| Kept sinks | — | `llm_worker.record_outcome` (MagicMock), `recent_outcomes` (real list), `AutonomyController.audit_log` (real) | real or acceptable | — |
| Mode refusal | `executor.set_operational_override("passive")` | `effective_operational_mode` | real (the operator grant) | executor.py |

### `test_atonia_red_gates_effect_policy.py`

| Gate | Producer driven | Consumer / sink observed | Real? |
|---|---|---|---|
| item2 own-effect / process runners | AST closure over `src/maxim`, plus the runtime factories `create_stub_tool` and `api.tool` | class bodies | real (static). `api.tool` registers into the process-global `api._registered_tools` and is unregistered in `finally` |
| item2 READ sweep | `build_tool_registry(…all optional blocks…)` | `ToolRegistry(ceiling=READ).register` | real builder, a superset of `maxim.run`'s `build_tool_registry(internet_launch_enabled=False, maxim=_robot)` |
| item2 sim fallback | `SimToolRegistry().get` | `_FallbackRedirectTool` | real |
| item3 policy required | `Executor`, `build_executor`, `AgentConfig` | ctor signatures | real |
| item3 race | `Executor.execute` → `self.registry.find` | instance-patched `reg.get` / `reg.find` | real (executor.py `self.registry.find`) |
| item3 composite spawn | `SpawnSubSimulationTool.execute` (lazy imports, so the patches reach) | `run_agentic_loop` positional arg 5 = `sub_executor` | real producer. The loop is a stand-in, but it is not the code under test |
| item3 auto_sense | `loop_perception.auto_sense` (direct `.execute()` calls) | probe tools' `execute` | real |
| O1 | `api.run` end to end, with lazy imports, so `build_primary_router` / `LLMWorker` / `MaximAgent` / `run_agentic_loop` patches reach. `learning=True` runs the real `AgentFactory.create_full_agent` | `executor` kwarg handed to `run_agentic_loop` | real: it is the object that would dispatch |
| V10 guard (passes) | same as O1, plus `api.register_tool` / `api.tool` → `_inject_registered_tools` | `ex.execute` under `set_mode_source(lambda: "passive")` | mostly real. **The mode source is a hand-composed constant**: production installs `lambda: run_mode(state)`, which for `maxim.run`'s `build_state()` yields `observe`, not `passive` |
| O2/O5 | `Executor.execute` → `tool.run` | instance-wrapped `execute` | real. See S3 on the params |

### `test_offline_latch.py` (child processes; the spy blocks egress)

| Gate | Producer driven | Observed | Real? |
|---|---|---|---|
| channels / self-engage / no-hook | raw `socket` / `subprocess` / `os` calls plus `import maxim` | the latch ledger and `OfflineRefusal` | real (stdlib events) |
| provenance | `code_tree.git_out`, `simulation/report.py::_code_stamp`, `maxim.get_version_info` | return value, spy | real. All signatures bind (probe P3) |
| census | the eight drivers | `OfflineRefusal` in the chain or the ledger, with the spy installed FIRST | real. Every driver binds today (P3), `smollm-135m-instruct` is in `LLM_MODELS`, and `ensure_image` spawns `docker image inspect` first, so it is refused on a box without docker too |
| router build/dispatch | `LLMRouter(cfg)`, `generate_json` | `router._backends` / `._backend` fakes | real router. The fake is injected through private attrs, which exist today |
| config key/version | `LLMConfigSection`, the schema fixture | — | real |
| downgrade warning | `config_writer.downgrade_config` | `config_loader.unknown_config_entries`, wrapped | real. The import is lazy, inside the function, so the patch reaches |
| conftest pop | AST of `tests/conftest.py` | — | real |

### `test_advisor_gate.py` / `test_advisor_log_red_gates.py`

`maxim.advisor` does not exist yet, so every gate pins a seam the implementing PR must provide
(`_session._backend_factory`, `_build_dispatch_stack`, `ProposeActionTool`, `AdvisorLog.create`). Gates whose
production producer exists today:

* `test_executor_import_closure_step0a`: `import maxim.runtime.executor` in a `-I` child. Real.
* `test_language_config_import_does_not_read_profiles`: an `open` audit on the real import. Real.
* `_default_registry_tools`: the real `build_tool_registry`, with instances neutralised, which is correct for a
  deletion sweep.
* `test_per_layer_deletion_sweep_sealed` reads the process-global `TOOL_ALIASES` at test time. A leaked alias only
  adds names, and each must still refuse, so leakage cannot make the gate pass.

The FakeBackend replaces the LLM. That is the backend seam, not the gate under test. The real-backend
parametrisations exist and fail on the missing fixture, so the backend is not mocked away.

### `test_atonia_open_items_other.py`

| Gate | Producer driven | Observed | Real? |
|---|---|---|---|
| O8 pool | `maxim.create.pool()` | `AgentPool._llm_router` | real |
| O8 run_turn | `AgentPool(llm_router=_FailingRouter).run_turn` | `TurnResult` | real `run_turn` / `_generate_npc_response`. The instance is a `SimpleNamespace` stand-in for an AgentFactory instance |
| O8 server | `maxim.cli.main([])` in a child (lazy `build_primary_router` import, so the patch reaches) | rc and stdout | real. **No egress spy in this child**, unlike the other children |
| O4 | `SendMessageTool.run(channel, recipient, body)` → `CommunicationGateway.send` | `channel.send` | real. The params match the schema |
| O6 | `ExecAgent._build_llm_context(MagicMock self, ctx)` with a hand-built `StructuredContext.comms_messages` | prompt text | **mid-pipeline**: the producer `MemoryAgent` (memory_agent.py, the `comms:` percept → `_comms_messages`) is bypassed. See S2 |
| O9 | `create.agent(..., ephemeral=True)` | `HOME` tree | real. See N4 |

### `test_profile_readers_v9.py` (passing guard)

Every reader is driven in a fresh child against the real module. The doctor reader patches
`maxim.runtime.capabilities.detect_compute_resources`, and `_check_vram_context_pressure` imports it lazily, so the
patch reaches. The `ensure_available lookup` reader passes today only because
`from maxim.runtime.llm_server import profile_has_local_file` imports `models.language.config`, which runs the
merge. That import side effect is the coupling step 0a removes, so the guard goes red on an unmigrated reader. That
is its purpose.

## Findings

### BLOCKER

**B1. `test_refusal_booking_red_gate.py::test_a_gate_refusal_books_no_learning` (all 8): the gate cannot flip, because
the fix it waits for breaks its rig first.** `_rig` calls
`build_executor(registry, pain_bus=None, permissions=permissions, nac=nac, pain_detector=detector)` with no
`policy=`. Item 3 is the same item whose `refused=` branch flips this gate, and its own red gate,
`test_atonia_red_gates_effect_policy.py::test_item3_policy_is_required_and_permissions_removed`, requires
`build_executor(…, permissions=…)` and a `build_executor` without `policy=` both to raise `TypeError`.

* **Failure scenario:** the item-3 PR lands. `_rig` raises `TypeError` before any spy is installed. That is not
  `AssertionError`, so all 8 cases FAIL (they do not XPASS). The PR author then has to rewrite the rig: re-point the
  gate, which CLAUDE.md forbids. The `permissions` kind also has to move to a `ToolPolicy` deny, so the case pins an
  API that item 3 deletes.
* **Fold:** build the executor through a lazy helper that uses `ToolPolicy` when `maxim.runtime.tool_policy` exists
  (`ToolPolicy.only`, or a deny for the `permissions` kind) and today's `permissions=` otherwise, the same pattern as
  `_require_policy()`. Pin in the docstring that the `permissions` case becomes `POLICY_PERMISSIONS`.

### SHOULD-FIX

**S1. `test_refusal_booking_red_gate.py`: the `PainBus.publish` sink is fed by a test-only producer.** The rig wires
`pain_detector=detector`, and no production `build_executor` call does that. `create_full_agent` (`maxim.run`
learning), bio_stack and agentic_runtime all pass `pain_bus=` with no detector, so `Executor._report_failure` is a
no-op there.

* **Probe P1:** under the production-like wiring, the unregistered and inactive_scene cases publish **0** pain
  events. Under the rig they publish 1.
* **Effect:** the docstring's claim that "the unregistered and inactive-scene cases also book pain" holds only in a
  composition nobody ships.
* **Failure scenario:** a fix that zeroes every write production makes still leaves the gate red on the detector
  path. Conversely, the production bus wiring is never exercised for refusal pain.
* **Fold:** parametrise the wiring over (a) `create_full_agent`'s `pain_bus=bus` and (b) the detector path, labelled
  legacy. At minimum, correct the claim. Also parametrise `drive_relief_only` over `{True, False}`: the shipped
  default is `True`, and the gate covers only `substrate-primary`.

**S2. `test_atonia_open_items_other.py::test_o6_inbound_comms_reach_the_prompt_framed` starts mid-pipeline.** It
hand-builds `StructuredContext(comms_messages=[...])` and calls `_build_llm_context` with a MagicMock `self`.

* **Failure scenario:** item 4 frames inbound text where it enters, at the producer (`MemoryAgent`, the `comms:`
  percept → `_comms_messages` append), which is a natural home for it. The prompt is then correct in production, but
  this gate still feeds raw content into
  `_build_llm_context` and stays red. The other way it breaks: if the framing moves into a `self._…` helper, the
  MagicMock returns a Mock and the gate dies as `ScenarioBroken`, never flipping.
* **Fold:** drive a `comms:` Percept through a real MemoryAgent → `StructuredContext` → `_build_llm_context`, or a
  real `ExecAgent` built with minimal collaborators.

**S3. `test_atonia_red_gates_effect_policy.py::test_o2_o5_ceiling_refuses_acting_tools_the_mode_gate_allows`: the
"body never runs" half is vacuous for `send_message`.**

* **Cause:** the params `{"recipient", "message"}` miss `SendMessageTool`'s required `channel` and `body`, so
  `Tool.run` → `_validate_input` raises before `execute` whether or not a ceiling exists.
* **Effect:** only the `POLICY_CEILING` kind check pins `send_message`.
* **Fold:** pass `{"channel": "sms", "recipient": …, "body": …}`. `call_user`'s `{"recipient", "message"}` is
  correct.

**S4. `test_atonia_open_items_other.py::test_o8_create_pool_wires_an_llm_or_raises` accepts ANY exception as
"documents and raises".**

* **Failure scenario:** an unrelated break in `create.pool()` (an import error in `LocalMessageBus`, a renamed kwarg)
  XPASSes the gate as if O8 were fixed.
* **Fold:** pin the documented exception type (e.g. `ConfigurationError`) or a message substring.
  `test_o8_run_turn_llm_failure_is_typed` has the same `except Exception: return`. It is softer there, because
  `run_turn` itself converts every exception to `TurnResult(error=...)` and `require(router.calls == 1)` guards the
  pre-router crashes.

### NIT

* **N1.** `test_v10_registered_tools_dispatch_under_maxim_run`: the docstring says "under the passive mode source the
  loop installs". The loop installs `lambda: run_mode(state)`, which for `maxim.run` gives `observe`. Install the
  real source (`run_mode(build_state())`) or fix the wording.
* **N2.** `_sub_sim_rig` reads `loop_calls[-1][5]`, which depends on `run_agentic_loop` being called positionally
  (simulation/tools.py `_sub_aut`). A keyword refactor gives `IndexError`, which fails loudly. Capture
  `executor=kw.get("executor", a[5] if len(a) > 5 else None)`.
* **N3.** `test_o8_server_mode_not_ready_is_reported`'s child has no egress spy. The census, provenance and advisor
  children all block egress. `main([])` in server mode with `MAXIM_ROLE=solo` made no connect in my run, but nothing
  enforces that. Add the `_PRELUDE` spy pattern. Also, the regex `not[ -]ready` misses wording like "failed to become
  ready"; the rc path covers it only if the fix exits non-zero.
* **N4.** `test_o9_…` snapshots `Path(os.environ["HOME"])`, which is the session-wide conftest isolation HOME.
  Post-fix, any other in-process writer during the call shows up as a false failure; this matters under xdist (to be
  relanded per the memory notes). Monkeypatch `HOME` to a per-test dir.
* **N5.** `_no_latch_leak` is defined twice (test_advisor_gate.py, test_advisor_log_red_gates.py). §1A.1 puts the
  latch reset in conftest. An in-process `advisor.open()` installs an audit hook that cannot be removed for the rest
  of the pytest session, so `_reset_for_tests` must leave the hook inert, not only clear a flag. Pin that by
  asserting a synthetic `sys.audit("socket.getaddrinfo", …)` passes after the reset.
* **N6.** `_ROUTER_DISPATCH_BODY` builds a `maxim_peer` lane at `127.0.0.1:9` BEFORE engage, with no spy and without
  `MAXIM_SKIP_REMOTE_PROBE`. Any build-time probe makes a real loopback connect. That is harmless (port 9), but set
  the variable for hygiene.

## Order dependence

| Run | Result |
|---|---|
| Seven files, default order | 10 passed, 141 xfailed |
| Seven files, reversed | 10 passed, 141 xfailed |
| Seven files, shuffle seed 1 | 10 passed, 141 xfailed |
| Seven files, shuffle seed 2 | 10 passed, 141 xfailed |
| Five unit files + `test_api_core/expansion/surface.py` + `test_tool_dispatch.py`, default / reversed / seed 7 | 184 passed, 50 xfailed in all three |

No order dependence today.

The process-global state the gates touch: `api._registered_tools`, which is cleaned in `finally`; `TOOL_ALIASES`,
which is read only and leak-tolerant; the conftest HOME/XDG/HF forcing, which the children replace with fresh envs
and assert is absent; and `network_guard`, which only matters in-process. None changed an outcome. Today most gates
stop at an existence assertion, so the order-dependence surface opens up once the items land (N4, N5).

## Probes run

* **P1:** scratch `probe_refusal.py`: the rig vs `pain_bus=bus` with no detector, NAc.observe and PainBus.publish
  counts per (kind, path). The table is in S1.
* **P2:** grep of every `build_executor(` / `pain_detector=` / `set_mode_source(` call site across `src/`.
* **P3:** `inspect.signature` of every census and provenance driver target, plus the `LLM_MODELS` membership and the
  router private attrs.
* **P4:** the seven files in four orders, plus a neighbour slice, with a scratch `shuffle_plugin`.
