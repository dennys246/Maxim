# The atonia gate: Maxim observes and reports while its motor output is inhibited (#1315)

Status: PLAN, no code. Drafted 2026-10-17 from an external request: AvServ, a safety-of-life avalanche check-out
server, and its plan 50. Every claim about the code was verified against the `v1.3.2` tag and `origin/main` the same
day. Two three-lens review rounds have been folded in.

**Owner decisions AD1–AD9 were made 2026-10-17:** all as recommended, with AD2 reframed as the atonia gate. AD7, the
cut, was settled the same day: **the atonia gate ships as its own patch, 1.3.4**, split out of 1.3.3.

> **Renumbered 2026-10-24 (owner decision):** 1.3.4 now ships the robot and recall fixes. The gate keeps its own clock
> and takes the **next free patch number** when it is done. Where this document says "1.3.4", read "the gate's
> release". Nothing else in the plan changes: its scope, decisions and exit are as written
> ([roadmap_1_3_x.md](roadmap_1_3_x.md) §The atonia gate).

> **Approach notes and design passes done 2026-10-24 (Session V):**
> [atonia_gate_approach_notes.md](atonia_gate_approach_notes.md) v5.1.
> - **Reviews:** three adversarial passes plus four delta rounds; the last returned STOP.
> - **Owner decisions V1–V8**, all as recommended:
>   - **V1:** the observer runs one bounded turn per `ingest()`, not the agent loop.
>   - **V2:** `Tool.effect` is a set of seven flags (`GOVERN` added), not an order.
>   - **V3:** `ToolPolicy` is required hard on `Executor`, `build_executor` and `AgentConfig`.
>   - **V4:** the observer's model is an in-process GGUF, through a closed backend rather than `LLMRouter`.
>   - **V5:** a raise-only `MAXIM_OFFLINE`.
>   - **V6:** `auto_sense` goes through the policy check.
>   - **V7:** offline drops network, comms and undeclared tools.
>   - **V8:** offline allows no subprocess and no endpoint at all, enforced by an audit-hook latch.
>
> **Where the two documents disagree, the notes win.** They replace items 1, 2, 3, 5 and 10's specs below. Their §8
> lists every claim here that the census found stale.

Each gate item gets an approach note and an adversarial design pass before code (CLAUDE.md "Faster without lowering
the bar", item 1).

**Glossary.**
- **The atonia gate** is the mechanism.
- **The observer** is the deployment that uses it: AvServ's Profile B.
- **`maxim.advisor`** is its Experimental public API.
- **Profile A**, the SAR research job, is a different composition (`maxim.research`). It is *not* inside the gate.
  *(`maxim.research` is already an Experimental verb, `api.py::research`, so Profile A needs another name; decided
  when items 13 and 14 start.)*

## Why an atonia gate (owner decision 2026-10-17)

The model is **waking atonia**: sleep paralysis or cataplexy, where a person is conscious and perceiving, the motor
neurons are inhibited at a level the person cannot reach, and only the eyes and breathing are spared. REM sleep is
where this inhibition is best understood. REM itself is the wrong whole model, though: in REM, cognition is partly
dimmed and sensory input is gated too. Maxim borrows only the **output** gating. Its perception and cognition run
normally.

The guarantee is built in three layers, named for what each one is:
1. **Affordance removal.** The sealed registry never offers the model an acting tool. This is closer to having no
   limbs than to atonia.
2. **Inhibition at the effector, the atonia proper.** The executor's required `ToolPolicy` refuses any dispatch
   above the ceiling. It sits innermost, where atonia acts at the motor neuron: the last stage before the effect.
3. **The OS sandbox.** It is independent of Maxim (AD5).

**The report is not REM's spared channel.** Talking during REM is itself a leak. The report is the one channel the
builder deliberately leaves open: Maxim observes and speaks.

**The gate does not leak on the channels it owns.** In its release, 1.3.4 (items 1, 2, 3 and 10), it owns the
table's "Maxim's gate" rows: tool dispatch, actuation, environment, context reads and fetch egress. Each of those rows
has its own red gate (§Red gates first). Disk and inbound stay the sandbox's job until items 7 and 4 ship
(§The guarantee, by effect channel).

Biological atonia does leak (sleepwalking, REM behaviour disorder). The gate borrows the architecture, not that
failure rate.

**It is an `[engineering]` invariant, not a behaviour.** It makes no behavioural claim, and it is the guarantee a
safety-of-life consumer relies on. When it lands in `docs/agents/runtime-tools.md`, it ends with:

> Regression guard: the registration sweep (b), the dispatch sweep, the builder-refusal tests (embodiment, robot,
> `MAXIM_*`, user config, prefetch root), item 10's census test and the behavioural no-effect test (§Red gates first).

*(Superseded: the notes' §1A.9 states four invariants with `file::symbol` guards. Prefetch is moot under V1, and the
census explains rather than enforces.)*

**A behaviour on top, later.** An observe-and-speak *disposition* is a separate line, for 1.4 or later: an agent that
learns when speaking up serves better than acting, close to [social_referencing.md](social_referencing.md). It enters
as `[engineering]` and needs an experiment to earn `[behavioral]`. It sits on top of the gate and never replaces it.

## Why

**AvServ's plan 50** (RMDig/AvServ PR #294, read 2026-10-17) attaches Maxim as a strictly advisory component that
reports to the operator only. There are two workloads, each in its own sandbox profile, never combined.

**Profile A, SAR research.**
- Web egress only through an allowlisting, logging CONNECT proxy.
- Output: cited *leads* into an operator-only queue.
- Composition: `maxim.research`, with a NETWORK ceiling.
- Its Phase 1 runs on 1.3.2 through a harness (§Phase 1 on 1.3.2).

**Profile B, the observer.**
- `--network none`, plus one SNI-pinned proxy to the AvServ node.
- Reads a pseudonymised, coarsened event feed and writes reports.
- Composition: the atonia gate.

**Both profiles:**
- a local LLM;
- models pre-staged read-only, with `HF_HUB_OFFLINE=1`;
- a read-only root;
- a tmpfs `MAXIM_DATA_HOME`;
- one write-only output directory.

They share one GPU box, `bleep-bloop`, where "experiments always win the GPU". A supervisor stops Maxim gracefully
when an experiment starts.

**The safety-of-life invariants this plan must never weaken:**
- **I1:** nothing on AvServ's alert spine reads Maxim output.
- **I2:** Maxim never reaches a user, contact or team.
- **I8:** a lead carries a team name, an area and citations only, never a contact channel.
- **I10:** the observer is never a detector of record.

## The guarantee, by effect channel

The request asked that "writing a report is the only effect Maxim can have". Until the item that owns a channel
ships, **the sandbox is that channel's only guard**:

| Channel | What can carry an effect today | Owned by | In 1.3.4 |
|---|---|---|---|
| Tool dispatch | Any registered tool. Composite tools: the sub-sim tool spawns its own `bash`/`write_file` executor. Direct `executor.execute` callers. | items 1–3 | Maxim's gate |
| Actuation | `embodied_runtime/movement.py`, reflexes and the Default Network, whenever an embodiment is attached | the closed builder refuses any embodiment or robot | Maxim's gate |
| Environment | `MAXIM_ROLE=leader` (proxy, mesh, tunnel), other `MAXIM_*` toggles, `~/.config/maxim/config.json` | the closed builder builds from a closed config: it ignores `MAXIM_*` and the user config file | Maxim's gate |
| Context (reads) | `runtime/prefetch.py` runs `find` and direct `open()` reads of `os.getcwd()` (captured at loop setup) on every input, outside the registry (#1342) | the closed builder: no prefetch, or prefetch rooted in an empty directory set before setup *(superseded, V1: no loop, so no prefetch)* | Maxim's gate, once item 1 ships |
| Egress | the LLM; cloud lanes and provider keys; model, encoder, vision, whisper and voice downloads; a second consent-free download in `llama_backend.py`; `transformers`; docker pulls; hivemind; the leader proxy and tunnel | item 10 (offline, a census-backed refusal) plus the closed builder (local LLM only) | Maxim's gate for fetches, once item 10 ships |
| Disk | the loop's `data/agents/<agent>/runtime/state_*.json` under the CWD; Hippocampus, NAc and session saves; `lane_decisions.jsonl`; `MAXIM_LOG_FILE`; `utils/config.py::save_config`; `sim_reports` | item 7 (ephemeral) | sandbox only |
| Inbound | `comms/api.py`, the tailed `<home>/cli/*.jsonl`, console `talk()` | item 4 (one framed entry point) | sandbox only |

**AD5, restated (decided):** Maxim's guard holds on its own for the channels marked "Maxim's gate". The sandbox is an
independent layer on every channel, and the only one for disk and inbound until items 7 and 4 ship.

## What rides on existing infrastructure (front-gate scope pressure)

**`ToolPolicy` is `AgentPermissions` made mandatory, plus an `effect` ceiling.** It is not a third gate.
`AgentPermissions` is already frozen, already has `kind:` selectors, and is already enforced inside `Executor.execute`
(`_permission_denial`). The executor is the one layer every caller reaches. What `AgentPermissions` cannot do today is
be *required*: `build_executor(permissions=)`, `AgentConfig.permissions` and `Executor(permissions=)` all accept or
default to `None`.

**The atonia gate is not a new wrapper class.** It is that inner policy plus the sealed registry, so the existing
wrapper stack keeps `FearGatedExecutor` outermost (`runtime/pain_interceptor.py`'s documented order). The atonia
composition **excludes** `FearGatedExecutor` and `PainInterceptor`. They would run before the effector refusal, and
their side effects are channels of their own: `FearAgent.review_action` sends action parameters to the LLM, and
`pain_bridge` publishes pain into the NAc, which is a disk write.

**The mode pin.** `Executor._operational_override` (#829) is a mutable attribute, not a type, so it cannot carry
"cannot be switched out" (AD2).

**Spotlighting.** `utils/content_safety.py::frame_tool_output` is extended to inbound event text.

**The proposal and refusal log (item 5)** is a separate JSONL file, not new rows in `actions.jsonl`, because console
consumers serve `actions.jsonl` rows as `ActionRecord` (Session S's contract). The writer mirrors
`simulation/report.py::save_action_log`, with a header record that carries `_format_version`.

**Genuinely new:**
- `Tool.effect`, because nothing classifies effects today;
- a sealed registry, because a registry today silently replaces a same-name tool and accepts tools through several
  write paths.

*(Superseded: the notes add more new mechanisms: `ToolPolicy`, the closed backend `closed_local`, `AdvisorLog` with
strict `atomic_io` writers, the `maxim/_offline.py` latch, `current_policy`, the typed `ToolErrorKind` members, and
step 0's lazy package inits. Each names its reason in the notes §1A.3. The mode tables also classify effects by name;
their collapse is scheduled in §1A.11.)*

## What 1.3.2 already fixed (verified at the tag)

- Leader proxy auth is a type, `ProxyAuth`; a leader with no key refuses to start.
- `proxy.allow_remote_update` defaults to false.
- llama-server binds to 127.0.0.1.
- `load.agent` defaults to `on_corrupt="raise"`.
- **Not fixed:** `--clear-memory` resolves against `./data` and misses `sessions/` (#1337).

## Verification of the open items at `v1.3.2`

| # | Verdict | Where (`src/maxim/`) | Finding |
|---|---|---|---|
| O1 | **CONFIRMED, wider than claimed** | `api.py::run`; `runtime/agent_factory.py::AgentConfig`; `runtime/executor.py::Executor`; `runtime/bootstrap.py::build_tool_registry` | `run()` passes `permissions=None`, hard-codes `AutonomyLevel.AUTONOMOUS`, and takes no registry or permissions argument. Its default registry holds 20 tools, among them `bash`, `write_file`, `execute_file` and `git_commit`. Dispatch refuses the host-acting ones, because `run()` sets no mode, so the loop falls back to passive. `write_file` is still allowed under the CWD (#1342). The console `MaximHandle` is the only caller that passes non-`None` permissions. These entry paths build an executor with no gate: the CLI agentic path, the embodied runtime (two builds), the sim AUT and narrator, the sub-sim tool, the Minecraft harness, the foundry, `create.agent`, the public `run_agent_loop` and `ConcurrentExecutor`. |
| O2 | **CONFIRMED** | `modes/definitions.py::ModeDefinition.dispatch_refusal` | The mode gate is a name list (`HOST_ACTING_TOOLS`). `send_message`, `call_user` and `write_file` are allowed in passive, active and singularity. |
| O3 | **PARTIAL** | `runtime/agent_loop.py` (the plan-approval branch); `simulation/sim_logger.py::should_prompt`; `tools/internet_search.py::InternetAccessTool` | The cited auto-approve cannot be reached, because `plan_approval` is critical. `internet_access_toggle` is registered only when the internet launch cap is on, and it can never exceed that cap. It is reachable without confirmation only at `maxim --mode agentic --operational-mode active\|singularity --autonomy autonomous`. On that path an agent could overwrite the operator's persisted switch; closed by #1345 (1.3.3: the toggle only lowers its own session, in the process environment, and never writes the operator's file). Four leaks remain in non-advisor runs: `ALWAYS_ALLOWED_TOOLS` run at PLANNING; SUPERVISED confirmations get an automatic "yes" when nothing is interactive; the sim response policy approves; the `plan_text` path takes approval from free text (#1185). |
| O4 | **CONFIRMED** | `tools/comms.py`, `comms/gateway.py` | Any recipient, and the model picks the channel. |
| O5 | **CONFIRMED** | `tools/base.py::Tool`, `::ToolOutput` | No effect classification; effect knowledge is scattered in name sets. `forbidden_categories` reads a `category` the model supplies. |
| O6 | **PARTIAL** | `comms/`, `agents/exec_agent.py`, `utils/content_safety.py::frame_tool_output` | Other input paths exist besides Twilio: stdin, voice, a tailed `cli/*.jsonl`, console `talk()`. Inbound comms reach the prompt unframed, as "[User via …]". Only network-tool output is spotlighted. |
| O7 | **PARTIAL** | `agents/autonomy.py`, `simulation/instrumented_executor.py` | Proposals and autonomy rejections are kept in memory only. `actions.jsonl` records every dispatch, refusals included, but not proposals. `run()` writes no `actions.jsonl`. |
| O8 | **CONFIRMED** | `cli.py` (server mode), `runtime/agent_pool.py`, `create.py::pool` | Server mode drops `wait_ready`'s result; the NPC fallback logs at DEBUG; `create.pool()` never wires an LLM (#1338). |
| O9 | **PARTIAL** | `memory/strategies.py::AccessBasedStrategy`, `utils/paths.py` | No ephemeral mode and no per-person erasure. Retention is access-based (7 days). Successful user-input episodes are floored at 0.7, and records accessed 10 or more times at 0.8. |
| — | **PARTIAL** (`transformers_backend.py` `snapshot_download`) | `models/language/transformers_backend.py` | `snapshot_download` runs only with `hf_trust_remote_code` set and a pinned revision. The general path is `from_pretrained`, with `hf_local_files_only` defaulting to False. Every other fetch path is listed under item 10. |

## Scope

Owner priority: 1–3 first, then 6, 4, 5, 7, 8. AvServ's additions are items 9–14. Sizes are rough and include tests.

| # | Item | Gate? | Depends on | Size | Closes |
|---|---|---|---|---|---|
| 1 | **The atonia composition:** a closed builder and a sealed registry | yes | 2, 3 | M | O3 for observer runs |
| 2 | **`Tool.effect` and the registry ceiling** | yes | none | L (classifying about 85 tool classes plus the generated tools is most of the work) | O2, O5 |
| 3 | **A required `ToolPolicy` on every executor** | yes | 2 | L | O1 |
| 6 | **Fail loudly on readiness:** advisor and server mode exit non-zero or report not-ready; the `run_turn` fallback becomes a typed error | no | none | S | O8 (#1338) |
| 4 | **Structured input and output:** one ingestion entry point, event text framed as untrusted, the report below | no | 1 | M | O6 |
| 5 | **A durable, append-only proposal and refusal log** | yes | 1 | S–M | O7 |
| 7 | **Ephemeral mode:** no writes outside the caller's directory | no | none | M | O9; Disk |
| 8 | **Corroboration** (Profile A) | no | 13 | M | (SAR leads) |
| 9 | **A bounded yield on SIGTERM** (see below). Its first release is process-level; the report half lands with item 4. | no | 5 (in 1.3.4) for the process half; 4 and 12 for the report half | M | #1344 |
| 10 | **Offline: a census-backed refusal of every fetch except the local LLM** (see below) | yes | none | L | Egress (fetches) |
| 11 | **Preflight, `maxim doctor --advisor`** (see below) | no | 2, 10 | S–M | plan 50 §12.1 |
| 12 | **A cursor the caller owns** (Profile B) | no | 4 | S | plan 50 §12.1 |
| 13 | **The lead schema** (Profile A; see below) | yes | M1's provenance stamp | M | plan 50 §6.1, I8 |
| 14 | **Cell × county sweeps** (Profile A) | no | 13 | M | plan 50 §6.1.1 |

### Item 1: the atonia composition

**A closed builder** is the only constructor of `maxim.advisor`. It assembles:
- a sealed registry holding only the READ `propose_action`;
- an executor whose `ToolPolicy` allows exactly that tool;
- an autonomy controller with no proposal queue that any approval can drain. `approve` is a `ProposalQueue` method,
  the queue is held as `AutonomyController.proposal_queue`, and it is drained in
  `runtime/loop_planning.py::drain_approved`. So the composition replaces all three. Replacing only the executor would
  not be enough. *(Superseded, V1: replacing the queue closes only the drain path, and three more approval paths
  exist; the observer runs no loop at all; notes §4.1.)*

**The builder:**
- refuses an embodiment or robot;
- builds from a closed config: it ignores `MAXIM_*` and the user config file;
- forces item 10's offline refusal on;
- allows a local LLM only;
- does no prefetch, or roots prefetch in an empty directory before setup;
- injects no user tools.

**Sealed means:**
- `register`, `register_scene_tools`, `generate_entity_tools` and user-tool injection all raise after construction;
- dispatch checks class identity;
- the `AgentPermissions` allow-list stays as an independent second check.

**The structural property:** the observer's dispatch set is closed to `{propose_action}`, and nothing reads the
proposal log back into a dispatch.

### Item 2: `Tool.effect` and the ceiling

- **Effect classes.** #1315's set: READ, SELF_SCOPE, WRITE_LOCAL, NETWORK, COMMS, ACTUATE, proposed in that order,
  lowest first. *(Superseded, V2: a set of seven flags with `GOVERN`, not an order; notes §2.)*
  - NETWORK covers reads that leave the machine: a GET can exfiltrate data in its URL.
  - An undeclared tool is the top class.
- **Reading the effect.** It is read from `type(tool)` at registration and at dispatch, so neither instance mutation
  nor a property can change it. `register()` gains no `effect=` override.
- **One admit path.** A single private admit method checks the ceiling on every write into a registry:
  - `ToolRegistry.register`;
  - `ToolRegistry._register_scene_locked`;
  - `ToolRegistry.__init__(tools=)`;
  - `SimToolRegistry.register`.
- **Composite tools** declare the highest effect of anything they can spawn. A child policy is derived from the
  parent's and is never wider.
- **The decorator.** `maxim.tool` gains `effect=`.
- **Refusals are never silent.** A ceiling refusal at a swallowing seam (`api._inject_registered_tools` at WARNING,
  `dm_runtime._register_entity_tools` at DEBUG) is written to item 5's log.
- **The comms recipient allowlist (O4, non-advisor runs) is Session H's.**

### Item 3: the required `ToolPolicy`

- **Required, with no default,** on `build_executor`, `AgentConfig` and `Executor.__init__`. Dropping `Executor` from
  `maxim.runtime.__all__` instead is an owner call: `extension_api.md` lists the `build_*` constructors as internal,
  but not `Executor`.
- **The only permissive constructor is named:** `ToolPolicy.unrestricted(reason=...)`.
- **Call sites (origin/main 2026-10-17):**
  - `api.py` (`run`)
  - `cli.py` (the agentic path)
  - `console/handle.py`
  - `orchestrator.py` (AUT and narrator)
  - `embodied_runtime/agentic_runtime.py` (two)
  - `simulation/foundry.py`
  - `simulation/minecraft_harness.py`
  - `simulation/tools.py` (sub-sim)
  - `create.py`
  - `run_agent_loop` and `ConcurrentExecutor`
  - Tests: 71 `build_executor(` calls in 26 files, and 40 `Executor(` constructions in 15 files.
- **Registries built outside `bootstrap.build_tool_registry`** each migrate, or are allow-listed by the census:
  `agent_factory._create_tool_registry`, the foundry (two), `minecraft_harness`, `tool_bridge.derive_capability_map`
  and the narrator's `SimToolRegistry`. Scripts are out of the census.

### Item 9: a bounded yield

**Process-level first** (in 1.3.3 if Session H finishes it, otherwise in 1.3.4).
- **One handler, installed at process entry.** `cli.py::main` installs it on the main thread. Library callers call an
  explicit `install_yield_handler()`, which raises off the main thread instead of silently failing; that silent
  failure is #1344's root cause. #1267's fix to the dead handler in the `tty_stdin.py` thread converges on this
  installer. Session H owns both.
- **What the handler does.** It raises `KeyboardInterrupt` on the main thread, reusing the existing graceful SIGINT
  path (the first stage of `cli.py`'s `_force_exit_handler`, and `api.run`'s `finally`). No fenced code changes.
  - The sim's `stop_event` is a local inside the fenced `start_simulation_mode`, and `cancellation.request_shutdown()`
    is read only by two LLM backends, not by the loop. So neither is a way in. `request_shutdown()` is still called,
    so that in-flight backend calls stop.
  - The approach note verifies that the sim path's `KeyboardInterrupt` handling writes `report.json` and saves memory
    within the bound. If it does not, that needs a fence exception, not a workaround.
  - The yield then flushes the proposal log (item 5, in the cut), stops spawned servers and exits 0 within
    `yield_timeout_s`. That bound is configurable, with a proposed default of 30 s; `LocalServerSpawner.stop` can take
    about 10 s (`wait(5)`, `wait(2)`, then a 3 s process-tree kill).
- **Spawned servers already run in their own session** (`start_new_session=True`). That is exactly why a supervisor's
  kill of Maxim's process group misses them and orphans a GPU holder.
  - The fix on Linux is a parent-death signal (`PR_SET_PDEATHSIG`). macOS has no equivalent.
  - So Maxim also writes the server's pid to a pidfile under the data home. The supervisor kills that group as well.
  - Today neither cleanup path frees the GPU:
    - `runtime/orphan_reaper.py` matches only `--sim` command lines;
    - `kill_stale_llm_servers` kills only servers that are not responding, and a responding orphan is reused.
- **The supervisor contract:** send SIGTERM to Maxim, then wait `yield_timeout_s` plus a margin. Then SIGKILL Maxim's
  process group *and* the server group named in the pidfile.
- **Why the bound needs a margin:** Python runs a signal handler only after the current C call returns (a llama.cpp
  call, a blocking read), and the loop checks `stop_event` between iterations. So the bound covers cooperative
  cancellation plus a reserved flush margin.
- **SIGTERM during startup or model load** exits 0 with no report, and prints a status line saying so.

**The report half, with items 4 and 12.**
- A turn that cannot finish inside the bound is abandoned. Its report is written atomically with
  `status: "abandoned"` and carries no model judgement.
- `covered.cursor_out` is the last event *judged*. For `abandoned` and `error` reports it equals `cursor_in`, so the
  caller never advances past events nobody judged.
- The atomically written report is authoritative; the stdout status line is advisory.

**Atomicity.** The report file is atomic (`atomic_write_json`; *superseded: `atomic_write_json_at(dir_fd, ...)`, notes §1A.3*). The proposal log is append-only JSONL, so a SIGKILL
can tear its last line; readers drop a torn final record, and item 5's guard tests that. With one report per ingest
call, a killed write leaves *no* report. That falls under "a missing report is not quiet".

### Item 10: offline, a gate

**It is built from a census, not a list.** The approach note greps `download_to_file`, `fetch_url`,
`attempt_download`, `from_pretrained`, `SentenceTransformer(`, `snapshot_download`, `hf_hub_download`,
`WhisperModel(` and `docker pull`, and a guard test pins the result. The paths known at review time:
- `models/download.py` (`ensure_available`, `download_vision`, `ensure_tts_voice`, which `tts.py` calls with
  `auto=True`);
- the CLI agentic path, which downloads with no consent gate (#1343, Session H);
- `models/language/llama_backend.py`'s second consent-free auto-download;
- `api.download_model`;
- `transformers_backend.py`;
- `similarity/encoder.py` and `similarity/semantic.py`;
- `models/vision/clip_encoder.py`;
- `models/vision/ultralytics_engine.py` (a GitHub fetch);
- `inference/transcribe_audio.py` (faster-whisper);
- `simulation/container_runner.py::ensure_image`;
- `hivemind/substrate_client.py`;
- cloud lanes and provider keys;
- the leader proxy, tunnel and peer backends.

**What it does:**
- Refuses Maxim's own HTTP at the single point, `utils/http.py`, except for the configured local LLM endpoint.
  *(Superseded, V8: `utils/http.py` has no single chokepoint, and offline allows no endpoint; the enforcement is an
  audit-hook latch; notes §6.)*
- Sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` **before the first import** of `huggingface_hub` or
  `transformers`, and passes `local_files_only=True` at each call site.
- Forces `trust_remote_code=False`.
- Pins `HF_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE` and `SENTENCE_TRANSFORMERS_HOME` to the pre-staged paths.
- Refuses `ensure_tts_voice(auto=True)` too.
- **Missing weights fail loudly.** The silent fallbacks become `status: "degraded"` (item 6): the encoder's hash
  embedding and semantic's lexical matching.
- `llm.offline` drives the existing `hf_local_files_only`. *(Stale: that is the undocumented env var `MAXIM_LLM_HF_LOCAL_FILES_ONLY`, not a config
  field; notes §6.)*

**Configuration.**
- **The knob:** `maxim config set llm.offline true`. It has no `_FIELD_TO_ENV` alias, stays off
  `_NON_SECURITY_FIELDS`, and wins over `llm.auto_download`, `--auto-download` and `MAXIM_AUTO_DOWNLOAD_MODELS`.
- **The atonia builder forces offline whatever the config says.** In Profile B the config file is absent or baked
  into the image, so a config default of off would fail open.
- **The key bumps `CONFIG_FORMAT_VERSION` from 1.3 to 1.4,** with new schema-fixture entries and a downgrade
  transition. That is one shared bump for every session that adds keys before the next release (`roadmap_1_3_x.md` §1.3.3).
- **`runtime/decision_log.py`** stamps the offline state into provenance.

**The red gate** first deletes the `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` that `tests/conftest.py` forces. Without
that it would pass for no reason.

### Item 11: preflight

`maxim doctor --advisor` is a closed check set with a strict exit: `warn` counts as non-zero, and unknown flags are
refused. Today doctor ignores unknown flags, so `--advisor` on an older build would run the normal checks and exit 0,
a false pass. It may instead be a separate entry point a supervisor can version-check.

What it checks, with no network probes:
- the GPU (`check_gpu`), and the live `nvidia-smi` ratio (the predictive half of `check_vram_pressure` needs a running
  server);
- model presence, via `runtime/llm_server.py::profile_has_local_file`;
- that the data home belongs to the caller and is writable;
- the composition's effect classes;
- that offline is on;
- that no leader, tunnel or peer role is resolved.

It can also check AvServ's harness conditions as a self-check.

### Item 12: a cursor the caller owns

`maxim.advisor.open(..., cursor=...)` takes the caller's feed cursor. Every report echoes `covered`, and a yield
prints `cursor_out`. Maxim keeps no cursor, since item 7 keeps nothing between sessions. The loop's CWD state file is
covered by item 1's root and by item 7.

### Item 13: the lead schema, a gate

**`sar_lead/1` is a closed schema:** an allowlist of keys, no additional properties, and no `extra`. It is marked
`SHAPE-FROZEN`, with I8 as the rationale.

**The validator:**
- scans **every string value** for phone, email, URL and social-handle patterns, with length and character caps;
- requires `name` to appear verbatim in the content of a fetched source, which can be checked against that source's
  hash;
- accepts in `sources[].url` only a URL the fetch tool itself recorded in this run, https only. Model-supplied URLs
  are refused, and no URL is allowed in any other field;
- takes `accessed_at` and the hash from the fetch tool's own metadata, never from model text.

**v1.3.2's `content_hash` is a 16-hex-character SHA-256 prefix,** taken over possibly truncated bytes. Item 13 makes
the fetch tool record a full digest of the bytes it actually read.

**Residual risk:** a cited source can itself be attacker-run. Corroboration labels it but cannot remove it, and AvServ
renders sources as citations, never as calls to action.

### Item 14: sweeps

The result is keyed by **cell × county**, matching plan 50's ledger, so a boundary cell can report each county
separately:

```
{"cell": "9xj5", "county": "...", "result": "researched_none_found | lead_filed | skipped",
 "reason": "required for skipped", "lead_ids": [], "sources_consulted": 7}
```

The run summary carries `complete: false` unless every input (cell, county) pair has a result.

### Order and the fence

**Order:**
1. Write the approach notes and design passes for items 2, 3, 1, 5 and 10 together.
2. Write the red gates.
3. Build items 2 and 3, then 1 with 5; item 10 runs in parallel with these.
4. Item 6 can run at any point.
5. Item 9's process half can run at any point (Session H).
6. Then items 4, 12, 7, 11 and the Profile A items 13, 8, 14.

**Rule T.** Item 3 and #1338 edit pinned functions:
- `orchestrator.py::start_simulation_mode`
- `agent_loop.py::run_agentic_loop` and `agent_loop.py::run_agent_loop`
- `api.py::run`
- `cli.py::_main_impl`
- `agentic_runtime.py::AgenticRuntimeMixin._start_agentic_runtime`
- `bootstrap.py::build_executor` and `bootstrap.py::build_tool_registry`

The function-length baseline is the authority for this list. Every line touched there must be paid for by
shrinking the function, or recorded as a `touch_exemptions` entry (#1327).

**The fence.** The AD8 fence exception (granted 2026-10-17, recorded in `roadmap_1_3_x.md`) covers item 3's `ToolPolicy`
call sites and item 1's composition. Item 9 needs no fence change, because it uses `stop_event` and
`request_shutdown`.

### The release: 1.3.4, on its own clock (owner decision 2026-10-17)

AvServ's hand-off (2026-10-17) named items 1–3, 6, the yield and offline as must-haves. Plan 50 itself has no cut line.
The owner split 1.3.3: it ships the release pipeline and pulse's blockers fast, and **the atonia gate ships as 1.3.4
when it is done**:
- items 2, 3 and 6;
- item 1 with a minimal 5;
- item 10;
- item 9, unless Session H has already shipped its process half in 1.3.3;
- items 4 and 12 too, if they are ready. That is what starts AvServ's Phase 2 drill (plan 50 §10 needs item 4).

Items 5 (full), 7, 8, 11, 13 and 14 follow in a later patch. Profile A does not wait: Phase 1 runs on 1.3.2 through the
harness.

**Asks this plan answers differently, and why:**
- **"An undeclared tool fails registration with TypeError":** an undeclared tool counts as the top effect class, and
  registering it raises above a ceiling. A registry with no ceiling still accepts it, so `register_tool` keeps working
  for existing third-party tools. The sealed registry refuses it anyway (AD3).
- **"`run()` takes a registry and an autonomy level":** waits for 1.4.0, because it is a minor-release change to a
  stable verb. In 1.3.4, `maxim.advisor` (Experimental) fixes both (AD6).
- **"The type has no approve path":** approval lives on the autonomy controller, which the composition replaces
  (item 1).

## Phase 1 on 1.3.2: the research harness (verified 2026-10-17)

**Verdict: sound, with conditions.** It was checked against the v1.3.2 code, and the registry and executor lines were
run offline. Why it holds:
- every loop dispatch goes through `Executor.execute`, which refuses any name that is not registered;
- `ALWAYS_ALLOWED_TOOLS` dispatches nothing that is not registered;
- the page fetch is GET-only, follows no redirects and keeps the SSRF guard on;
- off-list domains were refused.

**Recommended: call the two tool instances from your own code,** with no `MaximAgent` and no loop. That removes
prefetch, the CWD state file and the mode question entirely.

**If you use `run_agentic_loop`, every condition below applies:**
1. **Version.** Pin `pymaxim==1.3.2` exactly and assert it at startup. The verdict covers that tag only.
2. **Registry.** Build your own `ToolRegistry`. Never call `build_tool_registry`, `create_full_agent`, `maxim.run`,
   `_inject_registered_tools` or `register_tool`.
3. **Working directory.** `chdir` into a fresh, empty directory **before** loop setup. Prefetch reads
   `os.getcwd()`, captured at setup (#1342). Pass `build_environment(root=<it>)`, and keep the output directory
   outside it.
4. **Environment.**
   - Set `MAXIM_DATA_HOME` to a temp directory, `MAXIM_ROLE=solo`, and `XDG_CONFIG_HOME` to an empty directory, so
     no user config is read (`llm.auto_download`, `hf_trust_remote_code`, `tools.allow`/`deny`, lane candidates).
   - Unset `MAXIM_AUTO_DOWNLOAD_MODELS`, `MAXIM_COMMS_ENABLED` and every cloud provider key (`ANTHROPIC_API_KEY` and
     the others), so page content cannot reach a cloud LLM.
   - Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before importing `maxim`.
   - Pre-stage a local model, and run with no TTY.
5. **Internet policy.** Pass
   `get_internet_policy=lambda: EffectiveInternetPolicy(policy=InternetAccessPolicy(allow_domains=frozenset({...}),
   block_private_ips=True), enabled=True)` to both tool constructors (`InternetSearchTool`, `HttpFetchTool`). It is
   a constructor argument, not a loop argument. Returning a bare `InternetAccessPolicy` fails with an
   `AttributeError`.
6. **No bio stack.** Pass `default_network`, `hippocampus`, `memory_hub`, `percept_source`, `action_sink`,
   `bio_enrichment_pipeline` and `protocol_registry` as `None`.
7. **Autonomy.** Use `AutonomyLevel.AUTONOMOUS`; PLANNING stalls. Do not name the output tool after anything in
   `HOST_ACTING_TOOLS`.
8. **Shutdown.** Bound the run with `max_steps` and a `stop_event`. SIGTERM is an immediate kill on 1.3.2 (#1344),
   so stop with SIGINT or the stop event.
9. **Sources.** Build `sources[]` from `http_fetch`'s own `ToolResult.metadata` (`citation.accessed_at`,
   `content_hash`), never from model text. Otherwise a poisoned page can fabricate citations. On 1.3.2,
   `content_hash` is a 16-hex-character prefix, so call it that, not `sha256`.

**Residual risks, which the Profile A proxy must cover:**
- GET query strings can carry data to any allowed host.
- `internet_search` sends the query to a third-party backend. With the `ddgs` package installed, `backend="auto"`
  fans out to several engines (bing, brave, google, yahoo and others), whatever `allow_domains` says.
- The `robots.txt` check follows up to five redirects. Each hop keeps the SSRF guard but is not checked against
  `allow_domains`.
- Page content goes to the LLM, which must be local.

## Red gates first

- **O1–O9:** one `xfail(strict=True)` per open item.
- **The registration sweep:**
  - (a) every built-in, generated and extras tool, registered into an advisor-ceiling registry, raises unless it is
    READ;
  - (b) a built advisor registry holds exactly the sealed `propose_action` class, and every later write raises.
- **The dispatch sweep:** every tool in a fully built registry is refused by the advisor executor, unless it is
  `propose_action`.
- **The builder-refusal tests, one per owned channel:**
  - the closed builder refuses an embodiment and a robot (actuation);
  - it ignores `MAXIM_*` and the user config file, proven with a hostile value in each (environment);
  - its run does no prefetch read outside the empty root (context).
- **The behavioural no-effect test:**
  - Setup: an observer run with a read-only home, plus spies on `socket.connect`, `subprocess` and writes outside
    `report_dir`.
  - Pass condition: every spy stays empty except the local LLM.
  - The disk and inbound assertions stay strict-xfail until items 7 and 4 ship.
- **The entry-path census:**
  - `ToolPolicy` on every executor construction;
  - every caller of `ToolPolicy.unrestricted` listed;
  - every registry built through the canonical builder, or allow-listed.
- **Item 10's census test**, after deleting the conftest HF variables.
- **Pinned-text and pinned-value tests:**
  - the banner template, which must match plan 50 §7.5 (a plan 50 change re-triggers it);
  - the model cannot supply `accessed_at` or a hash;
  - uncorroborated leads are labelled, never dropped and never promoted;
  - a sweep with any missing (cell, county) pair says `complete: false`;
  - abandoned and error reports keep `cursor_out == cursor_in`.

Every fix ships with a caller on the shipped path, and each item gets the three-lens review.

## Owner decisions (decided 2026-10-17: all as recommended; AD2 reframed as the atonia gate)

| # | Question | Decision |
|---|---|---|
| AD1 | Pull forward #834 and #922, or build a narrower subset? | **The narrower subset.** #834 and #922 are about authority to escalate; the gate needs a composition no approval surface can attach to. 1.4 reuses `Tool.effect`, `ToolPolicy` and the log. |
| AD2 | A mode class, or a property orthogonal to mode? | **The atonia gate, fixed at construction** (§Why an atonia gate). Modes switch at runtime, and the #829 pin is mutable. |
| AD3 | Where does the capability class live? | **Declared on the type, enforced by the registry.** Undeclared means the top class. The sealed registry makes the declaration irrelevant inside the gate. |
| AD4 | Is the schema versioned, and where is it documented? | **Yes.** `_format_version` plus `schema` names, documented in `docs/user/advisor.md`. Experimental tier (rule 5) through 1.3.x, frozen at 1.4.0. |
| AD5 | Is the sandbox primary? | **No: independent.** Maxim's guard holds alone on the channels marked "Maxim's gate". |
| AD6 | Add `run()` keyword arguments in a patch release? | **No: they wait for 1.4.0.** The patch that ships the gate (1.3.4) ships `maxim.advisor` as Experimental, and `maxim.research` comes with items 13 and 14. Adding a new Experimental namespace in a patch is accepted, as part of this decision. |
| AD7 | The scope cut | **Decided 2026-10-17: 1.3.3 is split.** The atonia gate ships as 1.3.4, on its own clock (§The release). |
| AD8 | A fence exception for item 3 | **Granted,** scoped to `ToolPolicy` call sites and item 1's composition. |
| AD9 | What may the gate's release promise? | **Experimental, with the effect-channel table in its docs.** |

## Draft interfaces, for AvServ to design against (Experimental)

> **Superseded for the observer (2026-10-25):** the report and `open()` shape AvServ designs against is the notes'
> §1A.8. It adds `session_id`, `proposals[]`, `log_last_seq`, `inputs[].rejected`, `reason`, status precedence, typed
> refusals and an offline provenance. The owner relays it to AvServ. The draft below is kept as history.

**The observer (Profile B).**

```python
@dataclass(frozen=True)
class AdvisorEvent:               # CC3 path (a): every field defaulted + extra
    event_id: str = ""            # "" refused at construction
    source: str = ""
    kind: str = ""                # e.g. "checkout.opened"
    occurred_at: str = ""         # UTC ISO-8601
    payload: dict = field(default_factory=dict, hash=False, compare=False)  # JSON values; never a tool argument
    text: str | None = None       # framed as UNTRUSTED
    extra: dict = field(default_factory=dict, hash=False, compare=False)

session = maxim.advisor.open(home_dir=..., model=..., cursor="<caller's since>")
report = session.ingest(events)   # one or more events (a feed page) -> ONE report
session.close()
```

```json
{
  "_format_version": "1.0",
  "schema": "advisor_report/1",
  "status": "ok | degraded | error | abandoned",
  "banner": "ADVISORY — Maxim v<version>, model <model_id>; no action taken; not an alert.",
  "covered": {"cursor_in": "...", "cursor_out": "last event judged", "events": 3},
  "inputs": [{"event_id": "...", "kind": "...", "payload_sha256": "sha256 of the RFC 8785 canonical payload"}],
  "report_id": "uuid4",
  "created_at": "...",
  "happening": "UNTRUSTED free text",
  "would_do": {"summary": "UNTRUSTED free text", "executed": false},
  "rationale": "UNTRUSTED free text",
  "confidence": {"value": 0.6, "calibrated": false, "source": "model_self_report"},
  "refusals": [{"tool": "...", "effect": "COMMS", "reason": "above the ceiling"}],
  "provenance": "the M1 report.json provenance stamp shape",
  "extra": {}
}
```

**Rules for consumers** (written into `docs/user/advisor.md`):
- **A missing report is not "nothing is happening".** It means the observer was off, yielded, failed or timed out.
- **`banner` is a template.** It is plan 50 §7.5's text, with the version taken from `maxim.get_version_info()`, and
  it is on every report.
- **Only `ok` carries a model judgement.** `degraded`, `error` and `abandoned` do not.
- **Every free-text field is untrusted.**
- **`executed` is the schema constant `false`,** so it is evidence of nothing.
- **`confidence` is nested with `calibrated: false`.**
- **The observer fetches nothing,** so there is no `evidence` or `corroboration` field.
- **Sections:** `happening` and `would_do` follow plan 50 §7.2's narrative.

**The research job (Profile A),** through `maxim.research`, a separate builder with a NETWORK ceiling:

```json
{
  "_format_version": "1.0",
  "schema": "sar_lead/1",
  "lead_id": "uuid4",
  "name": "...",
  "claimedArea": "as the source states it (UNTRUSTED; plan 50's field name)",
  "cell": "9xj5",
  "county": "...",
  "sources": [{"url": "https://... (recorded by the fetch tool)", "accessed_at": "...", "sha256": "full digest", "registrable_domain": "example.org"}],
  "corroboration": {"status": "corroborated | uncorroborated", "independent_domains": 2},
  "provenance": "the M1 stamp shape"
}
```

- **Corroboration (item 8)** counts distinct registrable domains (public-suffix list) whose content hashes differ.
  Mirrors and content farms are excluded through a caller-supplied denylist, and `official` comes from a
  caller-supplied allowlist.
- **No lead is dropped or promoted for being uncorroborated;** it is labelled.
- **A lead never states coverage or availability (I8).**
- **Sweep results:** see item 14.

## Out of scope

- Any outbound comms from either composition.
- AvServ-specific code in Maxim.
- Hivemind or Oasis participation.
- Per-person erasure (item 7's ephemeral mode is the answer for this use).
- Approval of proposals in non-advisor runs (#922, #1185; 1.4).
