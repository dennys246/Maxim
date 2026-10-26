# Atonia gate approach notes v5.1: wire-integrity lens

**Reader:** the wire-integrity lens of the three-lens pre-merge round (docs/CODE_REVIEW.md), 2026-10-25, read-only.
**Object:** `git diff origin/main...HEAD`: `docs/plans/atonia_gate_approach_notes.md` v5.1 and `docs/plans/advisor_mode.md`.
The plan is paper. Each row maps a contract the plan **will** touch to its producers and consumers **outside** the
planned hunks, as of this worktree's base (`d6389cd4`, which merges origin/main `8eef529d`). `[T]` marks a test-only
site.

**Verdict:** 2 BLOCKER, 9 SHOULD-FIX, 9 NIT. Neither blocker is a hole in the gate itself. Both are places where the
plan claims a neighbour's behaviour that the neighbour's code will not produce.

---

## 1. Producer → consumer tables

### 1.1 Executor construction (`Executor.__init__`, `build_executor`, `AgentConfig`): permissions → required `policy`

| Contract | Producers (before → after) | Consumers (before → after) | What each consumer gets, before → after | Probe |
|---|---|---|---|---|
| `Executor(registry, pain_detector, tool_pain_bridge, permissions, embodiment, *, …)` → `Executor(registry, *, policy, …)` | src: `runtime/bootstrap.py::build_executor` (one physical `Executor(`). tests: 42 `[T]`, none positional past the registry (AST scan), so making the tail keyword-only breaks no existing call. scripts: `orient_backbone/exp53_cross_context_readout.py` uses a local `_Executor` class, not ours. | Every wrapper that forwards through `__getattr__`: `FearGatedExecutor`, `PainInterceptorExecutor`, `AnticipatoryPainExecutor`, `InstrumentedExecutor` (each has one); `ConcurrentExecutor` has none | Unchanged; `policy` lives on the inner object | 2, 3 |
| `build_executor(*, permissions=)` → `(*, policy=)` | 7 src calls: `api.py::run` (`learning=False`, `permissions=None`), `simulation/tools.py::SpawnSubSimulationTool` (`permissions=None`, plus its own `AutonomyController` that allows `bash`/`execute_file`), `runtime/agent_factory.py::create_full_agent` (`config.permissions`), `simulation/foundry.py`, `simulation/minecraft_harness.py`, `embodied_runtime/agentic_runtime.py` ×2. tests: 69 `[T]` (one `**splat`, which the census names as a blind spot). Docstring examples: `proprioception/pain_bus.py`, `runtime/pain_interceptor.py` module docstring, `runtime/bio_stack.py`. | — | Each call raises `TypeError` until it is edited (V3, intended) | 3 |
| `AgentConfig.permissions` → `AgentConfig.policy` (`kw_only`, no default) | 8 src: `create.py::agent` (`permissions=None`), `api.py::run`, `load.py`, `cli.py`, `agent_factory.py::create_npc_agent` (`permissions=None`, builds no executor), `simulation/orchestrator.py` (AUT and narrator), `console/handle.py` (`tool_permissions_from_settings(allow, deny)`). tests: 72 `[T]`. | `agent_factory.py::create_full_agent` reads it at the executor build; `create_agent` builds no executor | Today `None` gives a permissive executor; after, `None` is legal only where no executor is built | 2, 6 |
| `tools.allow`/`tools.deny` → `AgentPermissions` (`agents/permissions.py::tool_permissions_from_settings`) | `console/handle.py` only. The CLI and `api.run` pass `None` (D73 row, `docs/bugs/README.md`). | `Executor._permission_denial`, `permits` | After: wrapped in `deny(...)`/`only(...)`. Its `kind:` selectors read the **instance** `kind` (see S6) | 6 |
| `CampaignDef.permissions` (`simulation/dm_schema.py`, built from campaign YAML) | DM schema parser | **No reader** in src/ (grep `campaign.*permissions`) | A record that is never read today and is outside the census after the change (the D73 shape) | 1 |
| Claim surfaces | `create.py` error text `"…create_agent(AgentConfig(agent_id=..., persistence_dir=…))"` and `"…with AgentConfig.permissions"`; `agent_factory.py` docstrings (module, `:667`, `:1280`); `docs/agents/runtime-tools.md` builder table and its "five logical call sites" claim; `docs/proprioception.md` §example (`build_executor(registry, pain_bus=…, nac=…)`, which already lacks `permissions=` and is a `TypeError` today); `docs/bugs/README.md` D73 ("Unconfigured → None → byte-identical"); `docs/reference.md`; `docs/user/extension_api.md` | — | Each tells users to write a call that raises after V3 | claims |

### 1.2 `ToolErrorKind` gaining members, and how refusals are booked

| Contract | Producers (before → after) | Consumers (before → after) | What each consumer gets, before → after | Probe |
|---|---|---|---|---|
| `ToolErrorKind` (+`NOT_REGISTERED`, `POLICY_CEILING`, `POLICY_PERMISSIONS`, `POLICY_PIN`, `MODE`) | `tools/{code_tools,filesystem,git_tools}.py`, `simulation/tools.py`. Re-exported by `agents/bus.py`. | `proprioception/pain.py::record_tool_error` (`.get(kind, TOOL_FAILURE)`, so a new member maps to `TOOL_FAILURE`); `executor.py::_report_failure` (`or EXTERNAL_FAILURE`); `runtime/loop_state.py` → `ReplanContext.failure_type=str(error_kind)` → `planning/plan_document.py` prompt text and `to_dict` (persisted; `str(Enum)` gives `"ToolErrorKind.X"`, not `.value`); 13 test files `[T]` | New members reach the replanner prompt and plan documents as strings. Neither `actions.jsonl` (`ActionRecord`, `console/schemas.py::ActionRecord`) nor `report.json` carries `error_kind`. | 4 |
| Executor policy/mode denial (`Executor.execute`, both `_permission_denial` passes) | Executor | **Today:** it returns `ToolOutput(success=False)` *before* `record_tool_start` and `_report_failure`, so it books **no pain and no bridge NAc** already. It appends to `_tools_hallucinated` and increments `_consecutive_failures`. **Downstream:** `tool_dispatch.py::execute_and_learn` (two `rec_outcome` sites) and `execute_parallel_actions` call `record_outcome(success=False, error=…)` → `_observe_tool_result` → `nac.observe(...)` negative, goal credit, cluster reward, `llm_worker.record_outcome`, and `_record_plan_outcome` → `PlanHistoryBridge` | Today NAc books every executor refusal as a negative outcome through dispatch, not through the executor. After: the plan's skip sits in the executor (`_run_started`, `pain.py`), which refusals never reach | 4, 9 |
| `_tools_hallucinated` | Executor (both denial passes, and the unregistered `KeyError`); `agent_loop.py` parse-failure path (appends) | `agent_loop.py` → `failed_tools=` into `llm_worker.submit_context` under `MAXIM_TOOL_FAILURE_HINTS` | After: a refused tool no longer reaches the prompt's failed-tools hint | 4 |
| `_consecutive_failures` (executor) | Executor | **Only** `executor.py::_run_started`'s "Available tools" hint (`>= 2`) | It stops nothing. The loop stops on `AutonomyController.max_consecutive_failures` (`audit_log` "executed"+error, via `tool_dispatch.py`'s `log_action`). `MAX_REJECTION_FOLLOWUPS` counts `ctrl.rejection_followups` in `book_hard_rejection`, which is autonomy's path, not the executor's | 4 |
| `OfflineRefusal` raised inside a tool's `run` | `utils/offline` hook | `executor.py::_run_started` `except Exception` → `ToolOutput(error=…)` with no `error_kind` → `_fail_ran` → `ToolPainBridge.record_tool_failure` (NAc) + pain detector `EXTERNAL_FAILURE` → `TOOL_FAILURE`; then dispatch's `record_outcome` | A refused robot actuation (V7 says those "fail closed with `OfflineRefusal`") books pain and NAc aversion | 4 |

### 1.3 `ToolRegistry` admission, replacement, and `SimToolRegistry.get`

| Contract | Producers | Consumers | Before → after | Probe |
|---|---|---|---|---|
| Silent same-name replace (`ToolRegistry.register`) | `api.py::_inject_registered_tools` (each `run`/`imagine`/`campaign`; it catches `Exception` → WARNING); `register_tool`/`tool` (stable verbs, ✅ in `docs/user/stable_api.md`: "same-name re-registration replaces"); DM `_register_entity_tools`; `generate_tools_for_entity`; `Executor._handle_entity_acquisition` | Executor dispatch | Below `ALL` a replacement is refused, and a user's override of a built-in becomes a logged refusal | 2, 3 |
| `register(tool, kind=)` → `object.__setattr__(tool, "kind", kind)` | test harnesses, SEM wiring | `Executor._permission_denial` (`kind:` selectors via `registry.find(...).kind`) | The instance `kind` stays mutable at admission, and the allow-list judges on it | 6 |
| `ToolRegistry(` construction sites | `bootstrap.py::build_tool_registry`, `agent_factory.py` (empty fallback), `minecraft_harness.py`, `foundry.py` ×2, `embodiment/tool_bridge.py`; scripts `r2_drive_premise_probe.py`, `exp53` | — | Census pins them; ceiling `ALL` by default | 2 |
| `SimToolRegistry` (**not** a `ToolRegistry` subclass) | `simulation/orchestrator.py` (one src site); 10 total with tests | `get` is called by **`Executor._run_started`** (dispatch), not by `orchestrator.py`. It never raises, so `NOT_REGISTERED` is never produced in sims. `_FallbackRedirectTool` is built per call (`name="respond"`), and `_registered_tools` is a **class global** written by every `register` | After: under a pin, the fallback object fails as `POLICY_PIN`, not `NOT_REGISTERED`. Its `effect` must be declared for the declaration test | 2, 9 |

### 1.4 The prompt roster (`permits`) and the effect wire

| Contract | Producers | Consumers | Before → after | Probe |
|---|---|---|---|---|
| `Executor.permits(name)` gains the ceiling | Executor | `agent_loop.py::_advertisable_tools` (via wrappers' `__getattr__`); `executor._runnable_tools` (error suggestions); `find_similar` filtering; `simulation/narrator.py` text mentions it | A name has to be resolved to an object to get its effect; for an unregistered name, the effect is `"unknown"` | 9 |
| `Tool.effect` read by #1431 (OPEN, 1.3.4: `GET /api/tools?mode=talk`, MCP `annotations`) | the gate (later release) | #1431 decides read-only from a declared eligible set, not from `effect`; #1416 (closed as a duplicate) assumed a ladder | No serialisation of a `Flag` set is defined for the console/pulse wire | 9 |

### 1.5 `auto_sense`, the `current_policy` contextvar, composites

| Contract | Producers | Consumers | Before → after | Probe |
|---|---|---|---|---|
| `runtime/loop_perception.py::auto_sense` | — | It dispatches **two** paths outside the executor: the `get_auto_fire_tools()` loop, **and** `_tool_reg.get("sense").execute(entity_name=…)` per self-entity. Both sit inside an outer `except Exception` → `log_swallowed_exception` | V6 names one `executor.check(tool)`. If `check` raises (a wrapper without forwarding, or an attribute error), it is swallowed and looks like "nothing perceived" | 2, 4 |
| `current_policy` | `Executor.execute` | `SpawnSubSimulationTool.execute` (synchronous read). `ConcurrentExecutor` has **no production constructor** (exported from `runtime/__init__` only). `execute_parallel_actions` calls `executor.execute` itself | Sound as designed; the `ConcurrentExecutor` text is about an export-only class | 13 (exec ctx) |

### 1.6 Config: `CONFIG_FORMAT_VERSION`, `llm.offline`, `LaneDecisionRecord`, `MAXIM_OFFLINE`

| Contract | Producers | Consumers | Before → after | Probe |
|---|---|---|---|---|
| `CONFIG_FORMAT_VERSION` "1.3" → ? | `runtime/config_loader.py`; the #1431 `console.talk_tools` field (1.3.4, **ships first**) | `config_loader` version check: a same-version unknown key → `tolerate_unknown=False` → **refused**; a future minor → dropped with a WARNING. `config_writer` (refuses to write over a newer file; restore and downgrade, #974) | If #1431 takes "1.4" in 1.3.4 and the gate later adds `llm.offline` at "1.4", a 1.3.4 build **refuses** the gate's config. The plan states fail-open | 6 |
| `llm.offline` in `LLMConfigSection` (SHAPE-FROZEN; "any field bumps the minor") | `maxim config set` | CLI startup, router build, `api` verbs; `_NON_SECURITY_FIELDS` (kept off: flagged on restore, correct); console setup endpoints write only mesh/cloud (`apply_mesh_setup`, `apply_cloud_setup`), so there is no console writer | — | 6 |
| `LaneDecisionRecord.offline` | `decision_log.build_record` | `read_last_record` (`LaneDecisionRecord(**payload)`; `TypeError` → `None`); `doctor/cli.py`; `scripts/benchmark_cross_session.py` and `fail_loud_stage2.py` read the JSONL by key; `_capture_env_snapshot` does not list `MAXIM_OFFLINE` | As stated: a 1.3 doctor shows nothing for a 1.4 record | 7 |
| `MAXIM_OFFLINE` | user shell, harnesses, `engage()` | `maxim/__init__` self-engage; `tests/conftest.py`, **which imports `maxim` at line 8** (`from maxim.memory.encoding import …`) before any env handling; `tools/base.py::host_tool_env` allow-list | A shell export self-engages the pytest process before any autouse scrub runs. The `host_tool_env` addition has no reachable consumer under V8 | 1, 6 |

### 1.7 `OfflineRefusal(RuntimeError)` against existing catches, and the provenance stamp

| Contract | Producers | Consumers | Before → after | Probe |
|---|---|---|---|---|
| A spawn refused under offline | hook | `utils/code_tree.py::git_out` catches `(OSError, TimeoutExpired)` **only** → `head_commit`, `tree_dirty`, `code_tree_sha256` → `simulation/report.py::_code_stamp` (run start and end provenance) | **The refusal propagates and the report's provenance build raises in a checkout.** The plan says the report "says `executed_git_hash: "unknown"` and records the refusal", with no producer change named | 4, 7 |
| | hook | `maxim/__init__.py::get_version_info` (`except Exception`) ← `cli.py` banner, `leader_proxy` `/version`, `peer/cli.py`, and the advisor `banner` (advisor_mode.md) | Swallowed: the version is returned without `git_hash`, and a ledger entry is written | 4 |
| | hook | the RuntimeError catches in src (10; none on a fetch or spawn path except `hivemind/bundle.py`, which also catches `OSError`) | Fine, and covered by the widened swallow census | 4 |
| `report.json` code stamp → M1 lints | `report.py::_code_stamp` | `scripts/_evidence_records.py::unknown` (accepts `"unknown"` and an `"unknown:"` prefix), `scripts/_provenance.py`, `o19_verdict.py`, `grounding_census.py`, survival harnesses | `"unknown: offline"` is refused correctly, once the stamp is produced instead of raising | 7 |

### 1.8 `maxim.advisor` report and log against the draft that AvServ designs to

| Field | Notes v5.1 (producer) | advisor_mode.md draft JSON (the consumer's contract) | Probe |
|---|---|---|---|
| `open(...)` | `home_dir, model, report_dir, cursor, max_calls, turn_timeout_s` (+ `log_mode`) | `open(home_dir=, model=, cursor=)` | 3 |
| `calls` | in the closed completion schema; `would_do` too | absent; `would_do.summary` only | 7, 9 |
| `log_last_seq` | on every report | absent | 7 |
| `inputs[].rejected: "special_token"` | yes | `inputs[]` = `event_id, kind, payload_sha256` | 7 |
| `refusals[]` | log record: `name, error_kind, layer, effect, reason` | `{tool, effect, reason}` | 7 |
| session ↔ log join | log file `advisor_log_<session_id>.jsonl`, header `session_id` | no `session_id` on the report | 8 |
| refusal ↔ report join | `proposal` has `report_id`; **`refusal` has no `report_id` or `ts`** | — | 8 |
| `provenance` | the observer always engages offline; offline runs carry no M1 stamp (§6) | "the M1 report.json provenance stamp shape" | claims |
| `banner` | — | from `maxim.get_version_info()`, which spawns git in a checkout | 4 |
| status precedence | a table row per condition; "refused → `ok`", "degraded input → `degraded`" | "only `ok` carries a model judgement" | 4 |

### 1.9 Public surface and shared files

| Contract | Producers | Consumers | Before → after | Probe |
|---|---|---|---|---|
| `maxim.advisor` | new module | `maxim/__init__.py` has `_API_VERBS` (counted against the README by `tests/unit/test_api_surface.py`) and `_API_SUBMODULES`; `docs/user/stable_api.md` Experimental table | It belongs in `_API_SUBMODULES`; `maxim.api` is light at import (verified: no runtime modules load), so the import-closure test survives either way | 2 |
| `maxim.research` | `api.py::research` (Experimental) | `_API_VERBS`, `_API_TYPES["ResearchResult"]` | Correctly flagged; no collision with `advisor` | — |
| `tools/internet_search.py` urllib3 → `utils/http.py` (Session H's file) | — | the CI grep widened to `urllib3`, `requests`, `httpx.Client(`, `socket.socket(` | **Existing `socket.socket(` sites:** `doctor/platform_detect.py`, `simulation/minecraft_harness.py`, `utils/net.py`, `utils/reachy_diagnostics.py`; plus the `"urllib3"` string literal in `utils/sandbox_executor.py`. The widened grep is red on day one | 6 |
| `cli.py::main` "engage() first statement" | — | `llm.offline` must be read from config first; `ensure_blackwell_guards` is called at `cli.py` (one site); `python -m maxim` → `cli.main` (one path); the only other entry point is `maxim-diagnostics` | The order is config read → engage → guards | 10 |

---

## 2. Findings

### BLOCKER

**B1. `simulation/report.py::_code_stamp` raises under offline. The plan's stated outcome needs a producer change it does not name.**
- §6 makes `OfflineRefusal` a `RuntimeError` so that `utils/code_tree.py::git_out` does **not** absorb it, and lists `git_out → None` as a hazard. `git_out` catches `(OSError, subprocess.TimeoutExpired)` only.
- So under offline in a checkout, `head_commit` raises out of `_code_stamp`, at run start and at run end (`report.py` provenance, and `end_executed_git_hash`). The sim's `report.json` provenance build fails, and the run writes no stamp at all.
- The plan claims "an offline run's `report.json` says `executed_git_hash: "unknown"` and records the refusal".
- **Fix in the plan:** an explicit `is_offline()` branch in `_code_stamp` (and in the console identity, and in `scripts/_provenance.py` if a harness ever engages) that returns `"unknown: offline"` without spawning. `_evidence_records.py::unknown` already refuses that prefix.
- **Red gate:** an offline sim in a checkout writes `report.json` with that value.
- The same shape covers every other `git_out` caller in src (`head_hash`, `tree_dirty`, `code_tree_sha256`).

**B2. The stable verbs `maxim.register_tool` and `maxim.tool` change behaviour, and V3's rationale does not cover them.**
- Both are ✅ in `docs/user/stable_api.md`, whose contract says "same-name re-registration replaces". `api.py::tool(fn)` takes no effect argument, so every user tool is undeclared, which means `ALL`.
- Under §3's `api.run` policies (`only`/`pinned`, "workspace-only"), and under any registry ceiling below `ALL`:
  - every user tool is refused at admission or at dispatch;
  - a user's override of a built-in name is refused;
  - `api.py::_inject_registered_tools` turns each refusal into a WARNING inside `except Exception`.

  V3 says "None is in stable_api.md", but that is true only of `Executor`, `build_executor` and `AgentConfig`.
- **The plan must decide:**
  - an `effect=` parameter on `register_tool`/`@tool`, or documented `ALL`;
  - whether `api.run`'s allow-list includes registered tools;
  - the stable_api.md row and the CHANGELOG line.

  A red gate is owed: a registered READ tool runs under `maxim.run`.

### SHOULD-FIX

**S1. The learning-signal skip is placed where refusals never arrive** (`runtime/executor.py::execute`, `runtime/tool_dispatch.py::record_outcome`).
- Executor permission and mode denials already return before `record_tool_start`/`_report_failure`, so the planned skip in `_run_started`'s pain mapping and in `pain.py` is a no-op for them.
- The NAc outcome is booked downstream:
  - `execute_and_learn` (two `rec_outcome` sites) and `execute_parallel_actions` call `record_outcome(success=False, error=…)`;
  - that reaches `_observe_tool_result` → `nac.observe` (negative), goal credit, cluster reward, `llm_worker.record_outcome`, and `_record_plan_outcome` → `PlanHistoryBridge`.

  `record_outcome` receives no `error_kind`. "Books no NAc outcome" therefore needs `error_kind` (or a `refused` flag) threaded into `record_outcome` and its callers, and each sink there decided.
- §3 also says "how existing mode refusals are booked is unchanged" while adding a `MODE` member on the **same** `_permission_denial` path, and while removing `_tools_hallucinated` booking. That booking feeds `agent_loop.py`'s `failed_tools` prompt hint (`MAXIM_TOOL_FAILURE_HINTS`).
- **Fix:** a per-kind booking table (pain, bridge NAc, `record_outcome` NAc/goal/cluster/plan, `_tools_hallucinated`, the autonomy audit), with today's value and the new one per kind.
- **Failure scenario:** after the fold, a `POLICY_CEILING` refusal still books a negative `CausalLink` through `record_outcome`, and NAc learns aversion to a tool the policy refused. The executor-level test passes.

**S2. The plan names the wrong counter as the one that stops a model.**
- "It still counts toward `_consecutive_failures` and the rejection follow-up cap, so a model that keeps naming a refused tool still stops." But `Executor._consecutive_failures` only lengthens the error hint (`_run_started`, `>= 2`).
- `MAX_REJECTION_FOLLOWUPS` counts `ctrl.rejection_followups` in `book_hard_rejection`, which is autonomy's hard-rejection path. An executor refusal never enters it.
- The real stopper is `AutonomyController.max_consecutive_failures`, over `audit_log` entries that `tool_dispatch.py` writes with `log_action`. The red gate should drive that path.

**S3. `OfflineRefusal` inside a tool is booked as a tool failure** (`executor.py::_run_started` `except Exception` → `_fail_ran`).
- V7 relies on "an offline robot's tools fail closed with `OfflineRefusal`".
- Each such call becomes a failed `ToolOutput` with no `error_kind`. That books `ToolPainBridge.record_tool_failure` (NAc), pain `TOOL_FAILURE`, and dispatch's `record_outcome` negative.
- Offline sessions therefore teach aversion to every actuation and every lazy-download tool.
- **Fix:** map `OfflineRefusal` to a typed kind (`OFFLINE`, or a `POLICY_*`) in that `except`, under the S1 table.

**S4. The shared 1.4 config bump is stale since the 2026-10-24 release split** (§6 Config; §7 "Files owned elsewhere").
- #1431 (OPEN, 1.3.4) adds `console.talk_tools` to a SHAPE-FROZEN config ("any field bumps the minor"), so 1.3.4 takes "1.4". The gate ships later and needs "1.5".
- If both claim "1.4", a 1.3.4 build reads the gate's file as **same-version with an unknown key** and refuses it (`config_loader.py`, `tolerate_unknown=False`). That contradicts the stated "a 1.3 build tolerates the unknown key and runs online".
- The fixtures, the #974 downgrade warning and the docs need the real number.

**S5. The advisor report diverges from the draft schema AvServ designs against** (advisor_mode.md §The observer). The banner notes only that the notes win for items 1, 2, 3, 5 and 10. Six gaps:
- The draft lacks `calls`, `log_last_seq`, `inputs[].rejected`, and `session_id` (needed to find `advisor_log_<session_id>.jsonl`).
- The draft's `refusals[]` is `{tool, effect, reason}`; the log's refusal record is `{name, error_kind, layer, effect, reason}`.
- The draft's `open()` lacks `report_dir`, `max_calls`, `turn_timeout_s` and `log_mode`.
- `provenance: "the M1 stamp shape"` cannot be met, because the observer always engages offline, and §6 says offline runs carry no stamp.
- `banner` from `maxim.get_version_info()` spawns git in a checkout. The spawn is refused and swallowed, but it lands in the refusal ledger, and the observer's "nothing may spawn" red gate then goes red in any dev or CI checkout. Use `maxim.__version__`.
- `would_do.summary` and `calls[propose_action].summary` are two homes for the same judgement (probe 9).

**Fix:** update the draft JSON in this PR, or mark it superseded with a pointer, so AvServ's typed client is not built to a dead shape.

**S6. The log identity cannot be joined end to end** (item 5 records).
- `refusal` records carry no `report_id` and no `ts`, so a refusal cannot be tied to the report or the turn that produced it.
- Reports carry no `session_id`.
- Add both. `read_log`'s lost-tail check also needs the join to be per report.

**S7. The widened CI grep is red on existing code** (§6 layer 2).
- `socket.socket(` already appears outside `utils/http.py` in `doctor/platform_detect.py`, `simulation/minecraft_harness.py`, `utils/net.py` and `utils/reachy_diagnostics.py`.
- The string `"urllib3"` appears in `utils/sandbox_executor.py`'s blocklist.
- The plan needs a pinned allow-list with reasons, or these sites moved, in the same commit.

**S8. `auto_sense` has two dispatch paths, and a swallowing wrapper** (`runtime/loop_perception.py::auto_sense`).
- V6 names one `executor.check(tool)`. The interoception path (`_tool_reg.get("sense").execute(entity_name=…)`) is a second bypass, and it must be checked too.
- The whole block sits in `except Exception` → `log_swallowed_exception`. A `check` that raises (an unforwarded wrapper, a refactor) silently disables perception rather than refusing loudly.
- `check()` needs a return contract: bool, or a refusal object. A refusal should be logged as a refusal, not swallowed as an error.

**S9. The call-site list and the claim surfaces are incomplete for V3** (§3 "Call sites"; §1.1 rows above).
- **Name in the census list:**
  - `simulation/tools.py::SpawnSubSimulationTool`'s child `build_executor(…, permissions=None)`, with its own `AutonomyController` that allows `bash`/`execute_file`. The `FearGatedExecutor` wrap there is `except Exception: pass`, a grandfathered swallow that the policy change must not depend on;
  - `create_npc_agent`.
- **Edit these claims, which tell users to write a call that raises after V3:**
  - the user-facing error text in `create.py` (`AgentConfig(agent_id=..., persistence_dir=…)`, "with AgentConfig.permissions");
  - the docstring examples in `pain_bus.py`, `pain_interceptor.py` and `bio_stack.py`;
  - `docs/proprioception.md`'s example, which is already a `TypeError`;
  - `docs/agents/runtime-tools.md`'s "five logical call sites" and its builder table;
  - D73's "byte-identical when unconfigured".
- `simulation/dm_schema.py::CampaignDef.permissions` has no reader. Census it as a dormant field, or wire it.

### NIT

1. The plan says `SimToolRegistry` "through a shared mixin", and that "its one caller (`orchestrator.py`) keeps the fallback".
   - It is not a `ToolRegistry` and has no ceiling parameter. Its `get` is called by `Executor._run_started`.
   - `_FallbackRedirectTool` is a fresh object per call, so under a pin it fails as `POLICY_PIN`, not `NOT_REGISTERED`.
   - Its class-global `_registered_tools` is shared across registries.
2. `ToolRegistry.register(kind=)` re-kinds an instance with `object.__setattr__`, and `AgentPermissions` `kind:` selectors read that instance attribute. The §2.2 row "`kind` is not the gate" is true of the ceiling, but not of an `only(...)` policy built from `kind:` selectors. Snapshot `kind` at admission, or say so.
3. `ConcurrentExecutor` has no production constructor (it is exported from `runtime/__init__` only). Say that the contextvar text about it concerns an export-only class.
4. `maxim.advisor` goes in `_API_SUBMODULES`, not `_API_VERBS`. The README verb count is enforced by `tests/unit/test_api_surface.py`. Defining `open` inside the module shadows the builtin for its own code.
5. `tests/conftest.py` imports `maxim` at line 8, before any env handling, so `MAXIM_OFFLINE=1` in a developer shell self-engages the pytest process. The scrub must `os.environ.pop("MAXIM_OFFLINE")` at the top of the module, before that import, and not only in an autouse fixture.
6. The `host_tool_env` allow-list gaining the offline vars has no reachable consumer under V8: engaged means no spawn, and not engaged means the var is normally unset (probe 1). Keep it as defence in depth, but say so.
7. Typed refusals never reach `actions.jsonl`, `console/schemas.py::ActionRecord`, pulse, or `report.json`'s `blocked_actions`. A policy refusal reads as a tool failure. State it, or append an optional `error_kind` field; SHAPE-FROZEN `ActionRecord` allows an optional field at the end.
8. No wire serialisation is defined for `Effect` (a `Flag` set) for #1431's `/api/tools` and its MCP annotations. Define it (a sorted list of flag names) before any console reader appears.
9. `cli.py::main` cannot run `engage()` as its literal first statement, because `llm.offline` must be read first. State the order: config read (no spawn, no socket) → engage → `ensure_blackwell_guards`. Also, `_capture_env_snapshot` does not record `MAXIM_OFFLINE`; the new `offline` field covers it.

---

## 3. Knowledge-base note (for `docs/wiring/`)

A recurring shape, worth an entry when the gate lands: **a typed exception chosen so that narrow catches do not absorb it will crash every caller that relied on those catches for a fallback value.** `RuntimeError` was chosen over `OSError` so that `git_out` would not return `None`. That turns `git_out`'s callers, the provenance stamp first, from "unknown" into a raise. For each narrow catch the choice bypasses, list the caller and say what it must do instead.
