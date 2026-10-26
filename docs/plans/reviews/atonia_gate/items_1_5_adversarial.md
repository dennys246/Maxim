# Adversarial design review: atonia gate items 1 and 5 (approach notes §4, §5)

Reviewer: adversarial design pass (a different reader), 2026-10-24, read-only against the worktree at `d9dffe6e`.
Scope: `docs/plans/atonia_gate_approach_notes.md` §0, §1, §4, §5, against `docs/plans/advisor_mode.md`. I checked every
claim against the code, and the main ones by running probes. The probes were scratch scripts run with
`env -i PATH=... PYTHONPATH=<worktree>/src HOME=<scratch>`, a `sys.addaudithook` registered first, and an `os.environ`
read spy. Each one built `LLMRouter(LLMConfig(enabled=True, model_path=<gguf>))` exactly as §4.2 prescribes, called
`generate_json` once, and drove an `Executor` over a duck-typed sealed registry. Probe evidence is marked **[probe]**.

Classes: **DO-NOT-BUILD** (the design is wrong as written), **SHOULD-FIX** (a real gap to fold before code), **NIT**.
Each finding is also tagged **fail-open** (an effect or a hostile input gets through) or **fail-closed** (a refusal,
a crash or a lost record, with no effect escaping).

---

1. **DO-NOT-BUILD, fail-open: `LLMRouter(LLMConfig(...))` is not a closed router.** It reads env, writes the data
   home, and writes again at interpreter exit. §4.2 rests on the claim that skipping `build_primary_router` keeps the
   builder clear of `MAXIM_*` and of disk. Skipping it removes `_apply_lane_config_to_env`, the `llm.json` load and
   the decision log, but the router's own constructor still does all of the following:
   - `models/language/router.py::LLMRouter.__init__` builds a `CostTracker` from `cloud_dispatch.py::load_cost_config`.
     That leaves `state_path=""`, so `cost_tracker.py::CostTrackerConfig.__post_init__` calls
     `utils/paths.py::resolve_user_state("util/cost_state.json")`.
     - `paths.py::data_home` reads **`MAXIM_DATA_HOME`** (else `Path.home()`, so `HOME`), `mkdir`s it, and caches the
       result process-wide (`_data_home_cache`).
     - `resolve_user_state` then `mkdir`s `util/`. **[probe]** `os.mkdir <HOME>/.maxim` and `<HOME>/.maxim/util` at
       build time.
   - `CostTracker._load_state` reads `cost_state.json`, which is state left by an earlier run (adversary 3). On a
     corrupt or wrong-version file, `_mark_corrupt` calls `os.rename`, a disk write triggered by hostile state.
   - `LLMRouter.__init__` runs `atexit.register(self._cost_tracker.flush)`. `flush` calls `_persist_state`, which
     calls `atomic_write_json(<data_home>/util/cost_state.json)` at interpreter exit, unconditionally, even for a
     router that made no call.
     - **[probe]** at exit: `open .../home/.maxim/util/cost_state.json.tmp 'w'` followed by `os.rename`.
     - A hostile `MAXIM_DATA_HOME` therefore chooses where Maxim creates directories and writes a file. That is the
       **Environment** channel, which §4.2 and advisor_mode's table mark "Maxim's gate".
   - `home_dir` (the `open()` argument) is wired to none of this. The router never sees it.
   - With a read-only home (advisor_mode's no-effect setup), `data_home()` calls **`sys.exit`** on `PermissionError`,
     or `resolve_user_state`'s `mkdir` raises. The router cannot be built.
   - **[probe]** `MAXIM_*` reads on the §4.2 path: `MAXIM_DATA_HOME`, `MAXIM_INFERENCE_LOCK_TIMEOUT_S`
     (`router.py::_inference_lock_timeout_s`), `MAXIM_LLM_MAX_COST_PER_{REQUEST,HOUR,DAY,MONTH}` and
     `MAXIM_LLM_MAX_SESSION_COST` (`cloud_dispatch.py::load_routing_policy` via `_routing_cost_cap`), and
     `MAXIM_AGENTIC_CONSOLE` and `MAXIM_AGENTIC_VERBOSITY` (`utils/structured_logging.py`). The cost caps can only
     deny service on a local lane (fail-closed), but the note's sentence "reads no `MAXIM_*` variable" is false for
     the object the builder constructs.
   - `utils/cloud_audit.py::CloudAuditLogger` defaults to the **CWD-relative** `data/logs/cloud_audit.jsonl`. It is
     unreachable while `treat_as_cloud` is false, but it is a CWD write one config slip away.

   **Fold:**
   - Give `LLMRouter` (or the `CostTracker` it builds) an explicit no-persistence mode: no `state_path`, no
     `_load_state`, no `atexit` registration, and no `data_home()` call.
   - Pass `MAXIM_INFERENCE_LOCK_TIMEOUT_S`, the cost caps and the structured-logging verbosity as constructor values,
     so the observer's router reads no env.
   - Restate the claim as "the builder and everything it constructs read this pinned set of env names: {}".
   - Pin all of this with the subprocess audit-hook test in finding 10b. Pointing `cost_state_path` into `report_dir`
     is **not** enough: the `atexit` write still fires after `close()`, and `_load_state` still reads.

2. **SHOULD-FIX, fail-open: a model that goes missing after `open()` is silently replaced from the data home, or
   downloaded.**
   - `config.py::LLMConfig` defaults `profile` and `model_base` to `"mistral-7b-instruct-v0.2"`. The builder checks
     that the file exists at `open()`, but `llama_backend.py::_LlamaCppBackend._ensure` loads lazily, at the first
     `complete()`.
   - If the GGUF is renamed or deleted in between (a TOCTOU, adversary 3), `_fetch_missing_model` calls
     `download.py::ensure_available(profile)`.
     - If `profile_has_local_file` is true, it returns True. `_ensure` then re-resolves
       `build_model_path(model_base, quant)` and **loads the data home's GGUF instead of the caller's**.
     - If it is false, it downloads when `MAXIM_AUTO_DOWNLOAD_MODELS` or `llm.auto_download` grants consent.
   - No offline latch exists yet (`grep MAXIM_OFFLINE src/` finds nothing). The table's "hits item 10's refusal
     before `ensure_available` consults consent" holds only if item 10 places its refusal **above**
     `profile_has_local_file`. The substitution is not a fetch, so an offline latch placed at the fetch site misses it.

   **Fold:** build the lane with `profile=""` and `model_base=""` (`_fetch_missing_model` then returns False before
   any import), pass a `realpath`-resolved `model_path`, and warm up synchronously in `open()` (see finding 3). Add a
   test that deletes the GGUF between `open()` and `ingest()` and asserts a typed error with no `ensure_available`
   call.

3. **SHOULD-FIX, fail-closed but silent: "the model said nothing" and "the model never ran" produce the same
   report.**
   - `LLMConfig.enabled` defaults to `False`, and then `generate_json` returns `None` (`router.py::_complete_text`).
   - A failed load returns `""` with only a WARN (`_LlamaCppBackend.complete`, `_ensure`).
   - Every provider failing logs `dispatch_exhausted` and returns `""`.
   - In each case `ingest` would write an `ok` report with zero proposals. That is exactly the silent no-op class
     CLAUDE.md says to push into types.

   **Fold:** `open()` calls the backend's `warmup()` synchronously and raises `AdvisorModelLoadError` on False.
   `ingest()` maps a `None` or empty completion to `status: "error"`, never to `ok`. Add a test with a stub backend
   that returns `""`.

4. **SHOULD-FIX, fail-open against adversary 4: the one completion imports about 200 maxim modules, the loop's
   among them.**
   - `router.py::_dispatch_locked` lazily imports `maxim.agents.llm_worker` (for `_normalize_request_context`) and
     `maxim.runtime.llm_call_registry`.
   - **[probe]** one `generate_json` call adds **203** `maxim.*` modules, including `runtime.agent_loop`,
     `runtime.loop_controller`, `runtime.bootstrap`, `tools.comms`, `default_network.*`, `memory.hippocampus*` and
     `decisions.nac`. It also pulls in numpy, scipy, httpx and threadpoolctl, which `os.putenv`
     `OPENBLAS_MAIN_FREE`, `GOTOBLAS_MAIN_FREE` and `KMP_DUPLICATE_LIB_OK`.
   - V1's "not constructed" still holds: nothing in the loop is instantiated, and I saw no import-time disk or
     socket effect. But any future module-level side effect anywhere in that closure becomes an observer effect, and
     the "`os.environ` writes" spy in §4.2 goes red from third-party imports (see finding 10b).

   **Fold:**
   - Move `_normalize_request_context` to a leaf module (for example `models/language/types.py`).
   - Add an **import-closure census**: the set of `maxim.*` modules loaded by `open()`, `ingest()` and `close()` in a
     fresh subprocess is a pinned allow-list, and a new import fails it.
   - Spy on writes to `MAXIM_*` keys only, or snapshot and diff `os.environ`, rather than "any env write".

5. **SHOULD-FIX, fail-open on containment (it widens no effect channel): `frame_tool_output` does not neutralise
   chat-template special tokens, and the llama path parses them.**
   - The llama lane takes path A in `router.py::_invoke_backend`: a single string from
     `prompt_formats.py::_build_prompt`.
   - llama-cpp-python's `Llama._create_completion` tokenizes that string with `special=True` whenever `suffix is
     None` (`site-packages/llama_cpp/llama.py`, the `prefix_tokens` block).
   - `utils/content_safety.py::frame_tool_output` applies NFKC, strips zero-width characters, deletes C0/C1
     controls and defangs `tool output` lookalikes. It does nothing to `</s>`, `[INST]`/`[/INST]`, `<|im_start|>`,
     `<|im_end|>` or `<|eot_id|>`.
   - Event text containing them becomes real role-boundary tokens. The spotlighting claim ("Framed") overstates the
     protection. The sealed registry still bounds the damage to report content.

   **Fold:**
   - Defang the loaded vocabulary's control-token strings inside framed bodies (read them from the GGUF once at
     warmup), or assemble the prompt at token level with framed segments tokenized `special=False`.
   - Specify the payload encoding: `frame_tool_output` takes a `str`, so use
     `json.dumps(payload, sort_keys=True, ensure_ascii=True)`.
   - Name the router entry point: `generate_json(prompt, system_override=...)` with the builder's own preamble first.
     **Never** use the `"TOOL_PROMPT|"` path. `_generate_tool_response` runs planning-mode detection, splits on
     `PROMPT_SEGMENT_DELIMITER`, and its salvage branches **synthesize a `respond` action the model never called**
     (`reasoning: "fallback_parse"` / `"salvaged_from_json"`).
   - Also guarantee that event text can never begin the prompt: `generate_json` dispatches on the `"ANSWER_ONLY|"`
     and `"TOOL_PROMPT|"` prefixes.

6. **NIT, fail-closed: there are no native tool calls on this backend, and parsing cannot reach a tool.**
   - `_LlamaCppBackend` has no `complete_with_usage`, so `tools=` is dropped. A "`propose_action` call" is JSON text
     parsed by `json_parser.py::_extract_json_object` (`json.loads`, then `json_repair`). I verified it is pure.
   - The note never defines the output schema for "at most N per turn". The router's existing shape is one `action`,
     so the observer needs its own list schema, with anything past N truncated and recorded as a refusal.

7. **SHOULD-FIX, fail-open on the audit property: proposals and refusals can be lost silently, and the refusal
   "layer" has no structured source.**
   - `tools/base.py::Tool.run` catches **every** exception into `ToolOutput(success=False, error=str(e))`. If the log
     append in `ProposeActionTool.execute` fails (ENOSPC, EIO, an fsync error, a `UnicodeEncodeError`), the record
     disappears and nothing else writes it.
   - `executor.py::_run_started` calls `tool.run(**params)` with the **model's keys as kwargs**, so a model-chosen
     key `"self"` raises `TypeError`, which `Tool.run` swallows.
   - The executor's refusals are plain strings with `error_kind=None`. **[probe]**
     `"Tool not registered: 'bash'. Only use tools from the Available Tools list."`
   - `Executor` has no log hook, so no code exists that would write a refusal record.
   - `_permission_denial` (policy) runs **before** `registry.get` (sealed). For any foreign name the policy layer
     refuses first, so "three independent refusals, each logged" reduces to one in practice.

   **Fold:**
   - One session-owned `AdvisorLog` writer (a single seq counter and lock).
   - The **turn**, not the tool, writes the proposal and refusal records after inspecting a **typed** refusal: a new
     `ToolErrorKind` such as `REFUSED_SEALED`/`REFUSED_POLICY`/`REFUSED_CLASS`, set by the executor.
   - A failed write raises out of `ingest` (`status: error`) and poisons the session.
   - `ProposeActionTool.execute(self, /, **kwargs)`, or the turn passes the model's arguments as one dict value.
   - The note's "no reference to any registry, executor or callable" then holds exactly, because the tool writes
     nothing.

8. **SHOULD-FIX, fail-closed: the three layers are not independent, and the class check has no home.**
   - Nothing in `Executor.execute` checks `type(tool) is ProposeActionTool`. Adding the check puts an
     observer-specific branch into the shared dispatch path, unless it lives in `ToolPolicy` or in the sealed
     registry.
   - The policy check (`ToolPolicy.only` by **name**) and a class check both consult the object the same registry
     returns, so they share a failure mode.
   - Alias resolution reads a **mutable module global** (`executor.py::TOOL_ALIASES`, written by
     `Executor.register_aliases`) before the canonical check.

   **Fold:** check identity against the instance captured at build (`tool is sealed_tool`) in a frozen policy field.
   Have the observer's executor resolve no aliases. Pin both in the dispatch sweep.

9. **SHOULD-FIX, fail-closed: the refusal record's `effect` is undefined for names the observer has never seen.**
   - The model names tools that do not exist in the sealed process (`bash`), so `effect_of(tool)` has nothing to
     read.
   - The tempting implementation, building the default registry to look the effect up, constructs every tool, and
     constructing a tool is itself a channel.

   **Fold:** `effect: "unknown"` for names absent from the sealed mapping, with the reason `not_registered`, and a
   census test asserting that the observer never imports `bootstrap.build_tool_registry`.

10. **SHOULD-FIX: the listed red gates can pass vacuously, or fail for the wrong reason.**
    - (a) **The dispatch sweep** passes on `KeyError` alone, because every foreign tool is simply unregistered. That
      proves sealing and nothing about the policy or class layers. **Fold:** one deletion probe per layer:
      - seed the sealed mapping (test backdoor) with a READ tool named `propose_action` of another class, and assert
        the class or identity check refuses it;
      - run with a permissive policy and assert sealing still refuses;
      - run with no seal and assert the policy still refuses.
    - (b) **The no-effect spies.** Monkeypatched `open`, `socket.connect` and `subprocess` miss the following:
      - `os.open`, `os.mkdir`, `os.rename` and `Path.mkdir` (the `CostTracker` effects in finding 1);
      - **`atexit`-time writes**: `cost_state.json` is written after pytest's spies are gone. **[probe]**
      - native-code effects and `ctypes.dlopen`;
      - `.pyc` writes. **[probe]** an `Executor.execute` run wrote `src/maxim/simulation/__pycache__/*.pyc` into the
        worktree.

      **Fold:** run `open()`, `ingest()` and `close()` and **interpreter exit** in a subprocess. Register
      `sys.addaudithook` first, and an `atexit` dumper first (so it runs last), with `PYTHONDONTWRITEBYTECODE=1`.
      Assert on these events: `open` with write or create flags, `os.mkdir`, `os.rename`, `os.replace`,
      `os.remove`, `os.chmod`, `socket.connect`, `socket.getaddrinfo`, `subprocess.Popen`, `os.posix_spawn`,
      `os.exec*`, `ctypes.dlopen` (beyond the llama library) and `os.putenv` (MAXIM_*). Use a read-only `HOME`
      **and** a read-only `MAXIM_DATA_HOME`. My probe shows this method catches what in-process spies miss.
    - (c) **The hostile-`MAXIM_*` test.** "Every `MAXIM_[A-Z_]+` name grepped from src" misses names built at
      runtime:
      - `doctor/checks.py`: `f"MAXIM_LANE_{tier.upper()}_REMOTE_URL"` and `_REMOTE_API_KEY`;
      - `api.py`: `f"MAXIM_{subsystem.upper()}_TRACE"`;
      - `cli_utils.py`: `f"MAXIM_CLOUD_LANE_{...}_MODEL"`.

      A hostile **value** also proves nothing for a variable whose misuse is not observable in the test. **Fold:**
      make the primary assertion the **read set**. Spy on `os.environ` access in the subprocess (as my probe did) and
      pin the `MAXIM_*` names read to an allow-list, which is empty after finding 1's fold. Keep hostile values as
      the secondary check.
    - (d) **"Nothing reads the log back."** Grepping `src/` for `advisor_log` misses a path held in a variable, a
      `*.jsonl` glob and generic JSONL readers. **Fold:** make it structural. Only `AdvisorLog` holds the fd and the
      path, no public object exposes the path, and the turn takes no file input. Keep the grep as a tripwire only.
    - (e) advisor_mode's read-only-home setup is currently red **for the wrong reason**: the router build dies in
      `paths.data_home` (`sys.exit`). Finding 1's fold has to land before this gate means anything.

11. **SHOULD-FIX, fail-open: non-`MAXIM_*` env that the path honours is not in the threat table.**
    - **[probe]** `LLAMA_CPP_LIB_PATH` is read (`llama_cpp/llama_cpp.py`, module top) and decides which shared
      library is `dlopen`ed at the first `ingest`. That is native code execution chosen by the environment.
    - `GGML_*` and backend variables are read by ggml through C `getenv`, which no Python spy sees.
    - `CUDA_PATH` and `HIP_PATH` add DLL directories (Windows).
    - `LD_PRELOAD`/`DYLD_*` and `PYTHONPATH` (module shadowing: the CLAUDE.md harness-provenance lesson).
    - `HOME` (`data_home`, `Path.home`) and `XDG_CONFIG_HOME` (read at import).
    - `HTTP(S)_PROXY`/`HF_*` matter only if egress happens, which finding 2's fold forecloses.

    §0 puts "other env vars" in scope, but §4.2's answer covers only `MAXIM_*`, the provider keys and
    `XDG_CONFIG_HOME`. **Fold:** state that code-loading env (`LD_*`, `DYLD_*`, `PYTHON*`) is the sandbox's job.
    Have the builder **refuse** (typed, fail-closed) when `LLAMA_CPP_LIB_PATH` or any `GGML_*`/`LLAMA_*` variable is
    set, and document "run with a scrubbed env".

12. **SHOULD-FIX, fail-open: the log is reopened by name for every record.**
    - `append_jsonl_record(path, record)` opens the path on every call. `O_EXCL` guards only the creation, and
      `O_NOFOLLOW` guards only the **final** path component.
    - A same-user process (or a later run in a shared `report_dir`) can rename the file or swap a parent component,
      and later records then land in another inode.

    **Fold:**
    - Open once in `open()`:
      `dfd = os.open(report_dir, O_RDONLY|O_DIRECTORY|O_NOFOLLOW)`, then
      `os.open(name, O_WRONLY|O_APPEND|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, 0o600, dir_fd=dfd)`. Both work on macOS
      and Linux.
    - Hold the fd for the session, record `fstat` (`st_dev`, `st_ino`), and never reopen.
    - The helper's signature becomes `append_jsonl_record(fd, record)`.

13. **SHOULD-FIX, fail-closed: short writes.** A single `os.write` on a regular local file is atomic against other
    `O_APPEND` writers, but it can return fewer bytes than asked (ENOSPC mid-write, a signal after partial progress).
    The next record then concatenates onto a torn line in the **middle** of the file, and the strict reader raises
    for the whole log. `O_APPEND` atomicity also does not hold on NFS. **Fold:** check that the return value equals
    the length, and on a short write poison the writer and raise (`ingest` reports `error`). Require a local
    filesystem in the docs. The reader's error names the byte offset.

14. **SHOULD-FIX, fail-open on durability: fsync must not be swallowed, and the directory must be fsynced.**
    - The existing helpers swallow fsync failures: `utils/atomic_io.py::_fsync_path` logs at DEBUG and
      `atomic_write_text` does `except OSError: pass`. A copy of that pattern would make "every record is fsync'd"
      false without anyone seeing it.
    - After an fsync EIO, a retried fsync can report success with the data lost.

    **Fold:**
    - fsync errors raise, and the writer is poisoned.
    - fsync the **directory** once after creating the file (without that, the directory entry, and so the whole log,
      can vanish on power loss).
    - NIT: on macOS `os.fsync` does not flush the drive cache, so use `fcntl(F_FULLFSYNC)` to back the word
      "durable", or narrow the claim.
    - On tmpfs fsync is a no-op, so say durability is not claimed there.

15. **SHOULD-FIX, fail-closed: the header has a crash window, and the reader accepts an empty log.** Between the
    `O_EXCL` create and the header write, a crash leaves an empty file or a torn header. "Drop a torn **final** line"
    turns that into a valid-looking log with zero records. **Fold:** the reader requires a complete header as line 1
    and raises otherwise. Write and fsync the header **in `open()`**, so a read-only, full or tmpfs `report_dir`
    fails at `open()` rather than at the first proposal.

16. **SHOULD-FIX, fail-open on completeness: losing the tail is undetectable.** A monotonic `seq` detects gaps, not
    a truncated tail, and deleting the last N lines leaves a valid log. **Fold:**
    - `close()` and every report carry `log_last_seq`, and the log ends with a `close` record.
    - Optionally chain records with `prev_sha256`, so a mid-file edit is detectable.
    - State plainly that a same-user writer can forge the whole file. The log has no integrity against the host,
      which is the sandbox's job.

17. **NIT: the log's name contradicts its keying.** §5 says `report_dir/advisor_log.jsonl` (one fixed name), "one
    file per session, keyed by `session_id`", and "an existing file with another `session_id` refuses". With
    `O_EXCL`, **every** existing file refuses, whatever its id, and a second session in the same `report_dir` cannot
    start. Pick `advisor_log.<session_id>.jsonl`, with `O_EXCL` always.

18. **NIT: encoding bounds.**
    - Use `json.dumps(..., ensure_ascii=True, allow_nan=False)`. `json.loads` accepts `NaN`/`Infinity` from the
      model, which then produces invalid strict JSON for AvServ, and `ensure_ascii=False` raises on lone surrogates.
    - Bound **`reason`** as well. Executor errors embed model-chosen names and "Available tools: ..." lists.
    - Bound by encoded bytes, truncate at a code-point boundary, and set the truncation flag.

19. **SHOULD-FIX, fail-open (minor): logging is an output channel missing from the effect table.**
    - The router logs model output: `logger.debug("LLM raw response ... %s", text[:200])`,
      `warn("LLM returned non-dict ... raw text was: %s", text[:100])`, and the `info` planning lines.
    - An embedding application's handlers (stderr, a `MAXIM_LOG_FILE` handler, a log shipper) carry that text
      outside `report_dir`.

    **Fold:** add a "Logs" row to `docs/user/advisor.md`'s table, and have the observer's turn hold its loggers at
    WARNING with model text redacted, or say plainly that the logs carry untrusted model text.

20. **NIT: the sealed registry needs a Protocol.**
    - I verified that `src/` has no `isinstance(..., ToolRegistry)`. **[probe]** `Executor` runs correctly over a
      duck-typed sealed object exposing `find`, `get`, `advertised`, `list_all`, `get_tool_scene`, `is_tool_active`
      and `find_similar`. `deregister` is reached only through `_handle_entity_acquisition`, when a
      `tool_pain_bridge` is set.
    - `Executor.__init__` is annotated `tool_registry: ToolRegistry`, so mypy (`run_mypy_public.sh`) needs a
      `ReadOnlyToolRegistry` Protocol.
    - Also list the remaining write names explicitly (`deactivate_scene`, `deactivate_tool`), rather than "every
      other".

21. **NIT: the single completion has no timeout.** `LLMWorker` owned the call timeout, and `_inference_lock`'s bound
    (600 s) covers only waiters. A wedged llama call blocks `ingest` indefinitely, and item 9's yield must cover it.

22. **NIT, factual: the CWD re-read sits in a different function than the note says.** The note says
    "`run_prefetch` re-reads `os.getcwd()` on every input". The re-read is in
    `runtime/loop_controller.py::LoopController`
    (`self.prefetcher.prefetch_for_input(cli_text, cwd=os.getcwd())`), and the setup-time capture is
    `runtime/loop_setup.py` (`init_prefetcher(..., base_path=os.getcwd())`). It is moot under V1; fix the symbol in
    the note.

23. **NIT, factual (accurate):** `simulation/report.py::save_action_log` writes once, with no fsync, at
    `_ACTIONS_JSONL_FORMAT_VERSION = "1.2"`. `runtime/decision_log.py::read_last_record` scans in reverse and returns
    the newest parseable record. The atomic-io ratchet counts hand-rolled **rename** sites, so an append helper does
    not need registering with it.

---

**What holds up:**
- V1 is a real closure at the construction level. The turn needs no `PromptBuilder`, `ExecAgent`, `LLMWorker` or
  `LoopController`, only the router and an `Executor`.
- Parsing the model's output cannot reach a tool.
- The non-subclass sealed registry works with `Executor` today.
- Avoiding `build_primary_router` does remove the `os.environ` writes and the decision log.

**Verdict:** DO-NOT-BUILD as written. The router §4.2 builds is not closed: it reads `MAXIM_DATA_HOME` and seven other
`MAXIM_*` names, creates directories in the data home, and writes `cost_state.json` at exit. Fold findings 1–5, 7 and
10 (no-persistence router, `profile=""`, eager warmup, typed refusals with turn-owned logging, and audit-hook gates run
in a subprocess) and finding 12 (the held fd), then run one delta-scoped round.
