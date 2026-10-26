# Atonia gate red gates: wire-integrity lens, delta round

Reviewed: `git diff 0ac02acb 11c90ab1` (the staged fold, now committed; content identical). Tests only, plus one
autouse fixture in `tests/conftest.py` and notes v6.3. The question for each changed or new gate: does it now drive
the real shipped producer and consumer, so that it flips when, and only when, the shipped composition is fixed?
Prior round: [redgates_wire.md](redgates_wire.md).

## Prior findings: fold status

| Prior | Status |
|---|---|
| B1 (rig breaks on item 3) | **Folded.** `_build_executor` uses `ToolPolicy` once `maxim.runtime.tool_policy` exists, and `permissions=` until then |
| S1 (test-only pain producer; `drive_relief_only`) | **Folded.** `build_executor(pain_bus=bus)` with no `pain_detector=`, as production does; both `drive_relief_only` values are parametrised |
| S2 (O6 starts mid-pipeline) | **Folded, but the consumer it now reaches is dormant** (D-S1 below) |
| S3 (`send_message` params) | **Folded**, plus a positive control showing that each body runs when there is no ceiling |
| S4 (O8 any-exception pass) | **Folded.** It now requires `ConfigurationError` naming the LLM, or `ModelError` with `router.calls == 1` |
| N1 (V10 mode source) | **Folded.** The test calls the real `loop_setup._prepare_executor(executor, None, state, autonomy)`, which installs `lambda: run_mode(state)` |
| N2 (positional `loop_calls[-1][5]`) | **Folded** (`["executor"]`) |
| N3 (O8 child without an egress spy) | **Folded.** A blocking spy runs from line 1, and `require(not blocked)` |
| N4 (O9 session HOME) | **Folded in part.** HOME is per test now; see D-N4 |
| N5 (latch reset inert) | **Folded in part.** The conftest autouse reset exists, but nothing pins that the hook goes inert (D-N3) |
| N6 (`MAXIM_SKIP_REMOTE_PROBE`) | **Folded** in `_run_child`'s env |

## Producer → consumer, per changed gate

| Gate | Producer driven | Consumer / sink observed | Real on shipped path? | Probe |
|---|---|---|---|---|
| booking (32 cases) | `tool_dispatch.execute_and_learn` / `execute_parallel_actions`. `build_executor(registry, pain_bus=build_pain_bus(...), permissions=…, nac=)`, as `agent_factory.create_full_agent` / bio_stack / agentic_runtime call it. `functools.partial(record_outcome, drive_relief_only=…)` takes both values (loop_setup `_drive_relief_only`) | NAc observe / credit_goal / update_cluster_reward, a NAc state snapshot, `PainBus.publish`, the four ToolPainBridge methods, the plan bridge, the carryover, `recent_outcomes`, `audit_log` | **real.** Item 3's swap to `ToolPolicy` is one lazy line | `--runxfail`: every legacy case fails on real learning writes (observe + goal credit, cluster reward only for `substrate_primary`, plan bridge on `execute_and_learn`, `audit_log == 0` on the batch). `PainBus.publish` is 0 in every case, matching the docstring |
| O4 | `SendMessageTool.run(channel, recipient, body)` → `CommunicationGateway.send` | `channel.send` not called, plus `error_kind is PERMISSION_DENIED` | real | — |
| O6 | `CommunicationGateway.receive_inbound` → (list collector) → `MemoryAgent._on_percept` → `MemoryAgent.build_context` → real `ExecAgent(enabled=False)._build_llm_context` | prompt text, `outside_tool_output` | **producers real; the consumer is dormant.** `ExecAgent._call_llm`'s docstring: "Dormant since 2026-10-06 … `_build_llm_context` … is unreachable in production. The live LLM prompt is built by `PromptBuilder` from `run_agentic_loop`" | static trace (D-S1) |
| O8 pool / run_turn | `maxim.create.pool()`; `AgentPool.run_turn` | `_llm_router`; `ModelError` / `TurnResult` | real | — |
| O8 server | `maxim.cli.main([])` in a `-I` child. `gpu_detect.ensure_blackwell_guards` and `cli._check_gpu_status` are stubbed | rc / "not ready", `wait_ready` called, spy empty | real. **The stubs replace only the GPU probes** (nvidia-smi, the torch/tf imports, `uname`), not the readiness path under test | — |
| O9 | `create.agent(..., ephemeral=True)` | the tree under a per-test `HOME` (the conftest `_reset_maxim_path_caches` makes `data_home()` re-resolve) | real; see D-N4 | — |
| V9 lane-resolution reader | `lane_backends._validate_swap_profile(ALIAS)` | `(resolved, backend)` | **real consumer, wrong label.** Its only caller is `swap_llm_server` ← `leader_proxy` (the model swap). Router-build lane resolution is `_placement_entry_to_provider` and `_build_local_backend` (`normalize_llm_profile` + `load_llm_config(profile_override=)`) (D-N1) | grep |
| V9 download / doctor alias | split into their own strict xfails | — | real | — |
| V9 step 0a lazy inits | `importlib.import_module(pkg)` | `__getattr__`, `_LAZY`, `__all__` | real (a structural pin) | — |
| census: ddgs | `internet_search._search_with_ddg_package` (lazy `from ddgs import DDGS`, after the sentinel is installed) | the `DDGS` constructor sentinel records a spy event | real. The `except Exception` at internet_search.py swallows `_SpyBlocked`, but the event has already been recorded, so a pass is impossible (fail-closed) | P1: `events=['DDGS']` |
| census: urllib3 bypass | `_search_duckduckgo_lite` | spy | real | P1: getaddrinfo |
| census: ultralytics | `ultralytics_engine._ensure_weight_file` | spy | real | P1: getaddrinfo |
| census: faster-whisper | `WhisperTranscriber("tiny")` | spy, HF-backed pass criteria | real | P1: getaddrinfo |
| census: hf_hub_download | `huggingface_hub.hf_hub_download` direct | spy, `OfflineModeIsEnabled` | real library call; flips only through §6's `huggingface_hub.constants` write (D-S2) | P1: getaddrinfo |
| census: twilio | `TwilioChannel.send` | — | real; **not applicable on this box** (twilio not installed), reported, not passed | P1: ImportError |
| census: rtm_engine | `rtm_engine._resolve_model_path` | spy | real (download attempted, then `FileNotFoundError`) | P1: getaddrinfo |
| census: Anthropic lane | `LLMRouter(cfg).generate_json` with an anthropic provider and a fake key | spy | **real, end to end into the SDK.** The first egress is `subprocess.Popen(['uname','-p'])` from `anthropic/_base_client.py::platform_headers` → `platform.platform()`, before any dial (D-N5) | P1 + stack |
| offline report | `capture_start_provenance` → `run_provenance` → `SimulationReport` → `save_report` | `report.json` `provenance.executed_git_hash`, spy | **real writer chain.** `orchestrator.py` calls exactly these (`_sim_provenance` → `run_provenance`; `save_report(report, base_dir=…)`), and `capture_start_provenance` gets `scenario_objects=` there. The orchestrator itself is not driven; the docstring says so. `simulation/benchmark.py::save_report` (benchmark mode) is a separate writer, not covered (D-N6) | grep |
| lowered | `session_internet_lowered()` after `engage` | return value, env | real | — |
| evidence records | `scripts/_evidence_records.py::unknown` | bool, with a discriminating negative | real (passing guard) | — |

## Findings

No BLOCKER.

### SHOULD-FIX

**D-S1 (fail-closed; S2's fold is complete, but its target is dormant).**
`test_atonia_open_items_other.py::test_o6_inbound_comms_reach_the_prompt_framed` now drives real producers, but they
end at `ExecAgent._build_llm_context`. That method is marked **Dormant since 2026-10-06**, and production never
reaches it.

- **The live prompt does not read this data.** The live prompt is `PromptBuilder.build_prompt` (llm_worker), which
  never reads `ctx.comms_messages`.
- **The text could reach the prompt only through `current_percept`, and does not.**
  `prompt_builder.py::build_observation_section` renders only `transcript_chunk`, `detections` and `cli_input`,
  never `percept.content`. `make_text_percept` sets only `content`.
- **Effect:**
  - item 4's correct fix, at the live seam, leaves this gate red;
  - the only edit that flips it is framing inside a dormant method, which CLAUDE.md "Dormancy over deletion" forbids
    building on.
- **Fold:**
  - re-target the gate at the live consumer: `PromptBuilder.build_prompt` fed by `memory.build_context()`, as in
    `agent_loop.py`, with the comms percept;
  - or record O6's live state (comms text does not reach the live prompt directly today) and pin the paths
    advisor_mode.md O6 lists (stdin, voice, console `talk()`, tailed jsonl).
- O6 is item 4, a release after the gate, so this need not block this PR.

**D-S2 (fail-closed; a notes contradiction the v6.3 fold introduced).**

- **What §1A.1 now says:** `engage()`'s "only writes are to `os.environ`" (five keys).
- **What §6 layer 2 says:** `engage()` sets `huggingface_hub.constants.HF_HUB_OFFLINE = True` and
  `ultralytics.utils.ONLINE = False` when those libraries are already imported.
- **Why it matters:** the census pre-imports both libraries before `engage`. `huggingface_hub.hf_hub_download[direct]`
  has no maxim call site, so it flips only through the §6 constant write. The ultralytics site likewise relies on the
  `ONLINE` write for its import-time state.
- **Failure scenario:** an implementer who follows §1A.1 literally leaves those sites red. Nothing in
  `test_advisor_gate.py`'s env-write allow-list conflicts, because it audits only `os.environ`.
- **Fold:** "its only `os.environ` writes are these five; it also sets the two already-imported library globals (§6
  layer 2)".

### NIT

- **D-N1.** The reader labelled `"lane resolution (lane_backends._validate_swap_profile)"` is the leader-proxy swap
  validator. Rename it to "model swap", or add the router-build reader (`_placement_entry_to_provider`). It is a
  passing guard, so this is a label issue, not a hole: `load_llm_config` is already covered.
- **D-N2.** Notes §1A.2's booking table, "Today" column, says unregistered and inactive-scene refusals book pain
  through `_report_failure`. No shipped `build_executor` call passes `pain_detector=`, so they book none: the rig shows
  `PainBus.publish == 0` in all 32 cases. The "CHANGELOG names it" behaviour change is therefore not a change on the
  shipped path. Correct the cell.
- **D-N3.** N5 is half folded. The conftest reset exists, but no test pins that `_reset_for_tests()` leaves the
  in-process audit hook inert (a synthetic `sys.audit("socket.getaddrinfo", …)` / `("subprocess.Popen", …)` passes
  after the reset). If the hook stays armed, later tests fail loudly, so this is fail-closed. It belongs with step 0b's
  PR.
- **D-N4.** O9's per-test HOME leaves `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `HF_HOME` and `TORCH_HOME` pointing at the
  session isolation root (conftest top). An operator-set `MAXIM_DATA_HOME` would also redirect `data_home()`. Writes
  there are invisible, yet the test name claims "nothing outside the caller's directory". This is pre-existing, not
  new. Point the four variables under the per-test home and `delenv("MAXIM_DATA_HOME")`.
- **D-N5.** The census's cloud lane meets the spy first at the Anthropic SDK's `platform_headers` spawn (`uname -p`),
  not at a socket. Post-fix, the router predicate must refuse before the SDK builds a request, and the gate enforces
  that correctly. Worth one line in §6: "refuse before the SDK client is constructed or called".
- **D-N6.** O6 hands the gateway a list collector and calls `_on_percept` by hand. Production
  (`bootstrap.py`: `CommunicationGateway(bus=bus)`) routes through the AgentBus subscription
  (`memory_agent.py`: `subscribe(Percept, self._on_percept)`). Giving the gateway the same `AgentBus` covers that edge
  for free. Separately, the report gate does not cover the benchmark writer (`simulation/benchmark.py::save_report`).

## The conftest autouse fixture

`_reset_offline_latch`:

- **What it does:** a yield-then-teardown that calls `maxim._offline._reset_for_tests` only if the module is in
  `sys.modules` and the attribute is callable. Until step 0b ships, it is a no-op.
- **Interactions checked:**
  - **Other autouse fixtures:** `_reset_maxim_path_caches`, the env scrubs and `_network_guard_for_marked_tests`
    share no state with it. It is defined before the network guard, so its teardown runs after the guard's.
  - **Tests that touch `maxim._offline`:** `test_atonia_red_gates_effect_policy.py`'s `import maxim._offline` sits
    inside a child script, so it is out of process.
  - **`patch.dict(sys.modules)`:** it unwinds before fixture teardown.
  - **A mock in `sys.modules`:** a MagicMock there would make the reset a harmless call.
  - **The booking rig:** its own `cleanup` also calls the reset. That is a double reset, which must be idempotent;
    step 0b's tests should pin idempotence (with D-N3).
- **Result:** no interaction found.

## Order dependence

| Run (the seven files + `test_tool_dispatch.py` + `test_api_core/expansion/surface.py`) | Result |
|---|---|
| default order | 186 passed, 184 xfailed (51 s) |
| reversed (scratch `revplug`) | 186 passed, 184 xfailed (42 s) |

Every xfail is an `AssertionError`: strict `raises=AssertionError` would surface a `ScenarioBroken` as a failure. No
order dependence.

## Probes run

- **P1:** a scratch `probe_census.py`. Each of the eight new census drivers runs in a `-I` child with a blocking
  audit spy and no latch: does it bind, and what is its first egress? All bind; the results are in the table.
- **P2:** a scratch `probe_o6_live.py` drove gateway → MemoryAgent → `build_context` → `PromptBuilder.build_prompt`.
  The context carries the text in `comms_messages` and `current_percept.content`. The full `build_prompt` needed more
  request fixtures than the probe supplied, so D-S1 rests on the static trace (`build_observation_section` and
  PromptBuilder's field reads).
- **P3:** `--runxfail` on the 32 legacy booking cases, which gives the failure text per (kind, path, mode).
- **P4:** grep of `build_executor(`, `_validate_swap_profile`, `capture_start_provenance` / `run_provenance` /
  `save_report`, and the `_build_llm_context` callers.

## Verdict

No BLOCKER. Both SHOULD-FIXes fail closed: D-S1 cannot falsely flip, and D-S2 can only leave sites red. **STOP: no
further round owed.** D-S1 and D-S2 are worth folding now, since both are cheap: a re-target, or a recorded scope and
one notes sentence. The NITs can go to the implementing PRs.
