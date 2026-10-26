# Atonia gate approach notes, DRAFT v2: delta-round adversarial review

Scope: only the mechanisms v2 introduced or changed (§6 latch, §4.2 closed backend, §3 `ToolPolicy`, §2 flags,
§5 log, V7). Checked against the worktree at `plan/atonia-approach-notes` and the installed libraries
(Python 3.12.12, llama-cpp-python 0.3.8, huggingface_hub 1.26.0, transformers 5.5.4, ultralytics 8.3.248,
uvloop 0.22.1, httpx 0.28.1, reachy_mini 1.8.3). Probes ran with `python -I` from a scratch directory outside the
worktree. "Covered class" means an earlier round raised that class of defect.

## Findings

### 1. The latch's event list misses UDP and reverse DNS. SHOULD-FIX, fail-open, new class

- The probe raised in a hook on exactly v2's events. Then:
  - `sock.sendto(b"x", ("192.0.2.1", 53))` went through with no refusal. CPython raises `socket.sendto`, and also
    `socket.sendmsg`, but neither is on v2's list.
  - `socket.gethostbyaddr`, `socket.getnameinfo` and `socket.getfqdn` all ran unrefused. They raise
    `socket.gethostbyaddr` and `socket.getnameinfo`, which are reverse-DNS queries that leave the machine.
  - `gethostbyname_ex` was refused, because it raises the `socket.gethostbyname` event. So that one is covered.
- Who sends UDP today: reachy_mini 1.8.3 depends on `zeroconf` (mDNS `sendto`), and `dnspython`-style resolvers send
  DNS over UDP straight to a literal IP. Hand-rolled UDP DNS to a literal address gets past "DNS refused at the hook".
- **Fold.** Add `socket.sendto`, `socket.sendmsg` (refuse any destination that is not the allowed literal),
  `socket.gethostbyaddr` and `socket.getnameinfo` (refuse unconditionally while engaged). Make the red gate drive one
  of each.

### 2. Some connect paths raise no audit event at all. SHOULD-FIX, fail-open, covered class (item 10 #1/#7), new instances

- **uvloop.** `uvloop.run(asyncio.open_connection("192.0.2.1", 80))` really tried to connect (it ended in a
  `TimeoutError`, not a refusal) and the hook saw no `socket.connect` or `getaddrinfo` event, because libuv does both
  in C.
  - uvicorn 0.x's `uvloop_loop_factory` uses uvloop for the server loop whenever uvloop is installed.
  - That covers `console/server.py::main`'s `uvicorn.run` and `comms/api.py`'s server thread. Any async egress from
    a handler on that loop is invisible to the latch.
- **multiprocessing `spawn`.** `get_context("spawn").Process(...).start()` raised no event. The call is
  `multiprocessing.util.spawnv_passfds` → `_posixsubprocess.fork_exec`, which is not audited.
  - `embodied_runtime/selfy.py` and `embodied_runtime/media_loop.py` use `spawn`.
  - The child is a fresh interpreter. Its hook is inert until something calls `is_offline()`, so a spawn child is
    unlatched even though it inherits `MAXIM_OFFLINE=1`.
- **Native networking** (gRPC, libzmq, libcurl, Rust clients) is invisible to the hook too. That is the same boundary.
- **Fold.**
  - (a) The hook engages itself at install time when `MAXIM_OFFLINE` is truthy, and `maxim/__init__` imports
    `utils/offline`. Every Python child is then latched from its first import.
  - (b) Add a census category, `latch-blind`, for event-loop implementations and native-network libraries. Under
    offline, force `uvicorn.run(..., loop="asyncio")`.
  - (c) §6 states the boundary: the latch sees the Python `socket`, `subprocess` and `os` modules only.

### 3. The subprocess allow-list is keyed on a pre-PATH name. SHOULD-FIX, fail-open, new class

- The `subprocess.Popen` audit args are `(executable, args, cwd, env)`. The probe got `('true', ['true'], None,
  {...})`: the bare name, before PATH resolution. `executable` is usually `None`, or equal to `args[0]`.
- An allow-list keyed on names like `"git"` is therefore satisfied by any `git` on a hostile `PATH`, and `PATH` is in
  the §0 env adversary.
- A caller that passes an explicit `env=` without the offline variables also gives the child no latch at all. §6
  says sub-processes inherit the offline vars, but that is only true through `host_tool_env`.
- **Fold.**
  - While engaged, refuse a non-absolute `executable`/`args[0]`.
  - Key the allow-list on `realpath`.
  - When `env` is not `None`, refuse unless it carries `MAXIM_OFFLINE=1` and the HF/YOLO offline vars.

### 4. The "spawner record" also covers a reused server, which may not be ours. SHOULD-FIX, fail-open, covered class (item 10 #8), incomplete fold

- §6 allows "the `127.0.0.1:port` of the llama-server this process spawned". The code also has a reuse path:
  - `runtime/lane_backends.py::_route_to_reused_server` and `_REUSED_SERVER_SOURCE = "reused_server"`;
  - the docstring of `runtime/local_server_spawner.py::verify_llm_server_exposure`, which describes reusing "a server
    an older maxim bound".
- Both expose the same `base_url` (`LocalServerSpawner.base_url` → `http://127.0.0.1:{port}/v1`).
- What could be on a reused port: whatever is listening there, including an `ssh -L 8100:…` forward made by
  `maxim tunnel`, which is exactly the "loopback that is remote" case.
- A port freed when our child dies can also be re-bound by another process while the allowance still stands.
- **Fold.** The allowed literal exists only while a `LocalServerSpawner` in this process holds a live child
  (`Popen.poll() is None`, checked at connect). `stop()`, or the child exiting, revokes it. Under offline, the reuse
  path refuses.

### 5. `trust_env=False` does not remove proxies here. SHOULD-FIX, fail-closed (the latch backstops), new class

- `utils/http.py::_checked_client` already passes `trust_env=False`, then re-adds proxies on purpose through
  `mounts=_proxy_mounts(...)`, which reads `httpx._utils.get_environment_proxies()` (#921, an owner decision:
  "the proxy wins for operator paths").
- The `policy == "any"` branch of `_HTTPRegistry.get_client` builds a plain `httpx.Client(...)` with no
  `trust_env=False`.
- So v2's mechanism, as written, does not do what its input table claims.
- **Fold.** Under the latch, every client is built `strict` (`mounts=None`) and the `any` branch gets
  `trust_env=False`. State whether online proxy behaviour changes. If it does, that reverses #921 and needs the
  owner's decision.

### 6. The special-token defang is undone by framing, and deletion recombines tokens. SHOULD-FIX, fail-open, covered class (1+5 #5), broken fold

- v2 defangs "before framing". But `utils/content_safety.py::frame_tool_output` then applies
  `unicodedata.normalize("NFKC", text)`.
- Probe: `'＜｜im＿end｜＞'` NFKC → `'<|im_end|>'`, and fullwidth `［ＩＮＳＴ］` → `'[INST]'`.
- Deleting a token string recombines: `'<|im<|im_end|>_end|>'` → `'<|im_end|>'`.
- llama-cpp tokenizes a `str` prompt with `special=True` (`llama.py::Llama._create_completion`, the
  `prefix_tokens` block; `llama_chat_format.py` L599). That is the premise of the defang, and it is confirmed.
- **Fold (root cause).** Do not sanitize strings. Build the prompt as token ids:
  - the trusted template goes through `tokenize(..., special=True)`;
  - each untrusted segment goes through `tokenize(..., special=False)`;
  - pass the `List[int]` to `create_completion`, which accepts it (`prompt: Union[str, List[int]]`), with `grammar=`
    for the schema.
- If string defanging is kept:
  - run it after NFKC (inside `frame_tool_output`);
  - repeat it until nothing changes, and escape rather than delete;
  - cover both CONTROL and USER_DEFINED token types.
- Why this is not DO-NOT-BUILD: the pinned READ executor bounds the effect. But report integrity is the product.

### 7. The deadline does not bound the whole turn, and its stop is unlabelled. SHOULD-FIX, fail-closed, covered class (1+5 #21)

- `stopping_criteria` does exist in 0.3.8 with the assumed shape: `StoppingCriteriaList`, called as
  `(input_ids, logits) -> bool`, at `llama.py` L934/L1513.
- But it is checked only per *sampled* token. Prompt prefill (`eval`) is not interruptible, so `turn_timeout_s` does
  not cover a large framed prompt.
- A stop it triggers sets `finish_reason = "stop"` (L1513–1517), the same as a natural stop. Truncated JSON then
  lands as `error`, not `abandoned`.
- **Fold.**
  - A prompt-token cap is checked before `eval` (refuse with its own status).
  - The criterion sets a flag, and the status comes from the flag.
  - The docs state that prefill is bounded by the token cap, not by the clock.

### 8. The env read-set test is blind to iteration and to C `getenv`; the refusal list misses `LLAVA_CPP_LIB`. SHOULD-FIX, fail-open, covered class (1+5 #10/#11), incomplete fold

- v2 wraps the getters, `get`, `__getitem__`, `__contains__` and `getenv`. Whole-env reads go through
  `__iter__`/`keys`/`items`/`copy`, which v2 does not wrap:
  - in-tree: `utils/code_tree.py` (`os.environ.items()`), `runtime/local_server_spawner.py::_build_subprocess_env`
    (`os.environ.copy()`), `tunnel/daemon_spawner.py`, `utils/plotting.py`;
  - libraries: httpx `get_environment_proxies` and urllib's `getproxies_environment`.
- A prefix scan reads names the test never records, which is exactly the runtime-built-name case v2 cites.
- The C libraries read env the wrapper cannot see. `strings` on `llama_cpp/lib/*.dylib` shows `GGML_BACKEND_PATH`
  (it loads code), `GGML_METAL_EMBED_LIBRARY`, `GGML_METAL_NO_RESIDENCY`, `GGML_SCHED_DEBUG`, `LLAMA_TRACE` and
  `LLAMAFILE`. "Reads no env" cannot be proven at the C level.
- `llama_cpp/llava_cpp.py` L38 reads `LLAVA_CPP_LIB`, a library path, so it loads code. It is imported lazily (only
  by `Llava15ChatHandler`), and it is not on the refusal list.
- **Fold.**
  - Iteration or copy records a wildcard read, and the test fails on it.
  - Add `LLAVA_CPP_LIB` and `LLAMA_*` to the refusal list.
  - Pin the C-level names with a `strings` scan of the shipped libraries (a census, not a proof), or state them as
    out of model.

### 9. `ToolPolicy` with `__slots__` is still mutable in place. SHOULD-FIX, fail-open (future-code adversary), new class

- `__slots__` blocks new attributes, not assignment: `policy.ceiling = Effect.ALL` works. v1 #11 closed
  *construction*; nothing closes *mutation*.
- `pinned: Mapping[str, Tool]` is held by reference, so a caller that keeps the dict can add pins after build.
- **Fold.**
  - `__setattr__` and `__delattr__` raise after `__init__`.
  - `pinned` is snapshotted into `MappingProxyType(dict(instances))`.
  - `__copy__`/`__deepcopy__` return `self`, and `__reduce__` raises.
  - A constructor test plus an assignment test.

### 10. `error_kind` is an Enum, and refusals today feed the failure counters. NIT, fail-closed, new class

- `tools/base.py::ToolErrorKind` is an `Enum` with no `policy_*`, `not_registered` or `mode` members.
- `runtime/executor.py::Executor.execute` books refusals into `_tools_hallucinated` and `_consecutive_failures`.
- `_run_started` (L640) maps `error_kind or EXTERNAL_FAILURE` into pain, and `proprioception/pain.py` (L665) maps
  kinds to `PainType`.
- **Fold.** Add the members, and decide and state whether a policy refusal books pain, NAc or the
  hallucination counters.

### 11. V7's narrowing and the `WRITE_LOCAL` definition conflict with §2/§3. SHOULD-FIX, fail-open, covered class (2+3 #4/#5), new text

- §3 says "a ceiling is never built by complement". V7's "drop NETWORK, COMMS" therefore has to be the enumerated
  `READ|SELF_SCOPE|WRITE_LOCAL|ACTUATE|GOVERN`. Say so: a future eighth flag is then excluded, which fails closed.
- `WRITE_LOCAL` includes "runs local processes". By §2.1's own collaborator rule, a process runner can cause
  anything. A tool classified WRITE_LOCAL that spawns a process survives V7 and Profile A, and its child is latched
  only by env (finding 3).
  - Today's in-tree spawners (`tools/git_tools.py`) use fixed argv plus `tools/base.py::GIT_HARDENING_ARGS`. So this
    is a definition hole, not a live exploit.
- **Fold.** Remove "runs local processes" from `WRITE_LOCAL`. A tool that spawns a process declares `ALL`, unless its
  argv is fixed and on a pinned exemption list with a reason (the git tools).

### 12. A required field cannot follow defaulted fields on `AgentConfig`. NIT, fail-closed

- `runtime/agent_factory.py::AgentConfig` is a mutable `@dataclass` whose fields are all defaulted, so a required
  `policy` needs `field(kw_only=True)`.
- `agent_factory.py` L863 builds a config with `permissions=None` because no Executor exists there. Say what `policy`
  is on paths that build no executor.

### 13. The `is_offline_mode()` verification is tautological. NIT, fail-closed

- `huggingface_hub/constants.py::is_offline_mode` returns the module global `HF_HUB_OFFLINE` (L221). Setting it and
  then calling it verifies nothing.
- In the installed transformers and huggingface_hub, no module holds its own copy of it. So there is no live hole,
  and the latch backstops anyway.
- **Fold.** Verify behaviourally (a hub call that must raise `OfflineModeIsEnabled` without a socket event), or drop
  the claim.

### 14. `YOLO_OFFLINE` after ultralytics is already imported. NIT, fail-closed

- `ultralytics/utils/__init__.py` L910 has `ONLINE = is_online()`, which runs `getaddrinfo` at import unless
  `YOLO_OFFLINE == "true"` (L778). That confirms the fold's premise.
- If ultralytics was imported before `engage()`, `ONLINE` is already `True`.
- **Fold.** Mirror the HF rule: set `ultralytics.utils.ONLINE = False` when the module is in `sys.modules`.

### 15. AF_UNIX connects are refused, unstated. NIT, fail-closed

- The probe refused `AF_UNIX` `connect` and broke multiprocessing `forkserver`. That also breaks `docker.sock`,
  syslog, and any local IPC.
- **Fold.** §6 names the AF_UNIX policy: refuse, or a pinned path list.

### 16. The log's directory fd. NIT, fail-closed

- The fd mechanics are verified on macOS:
  - `os.open in os.supports_dir_fd`;
  - `O_CREAT|O_EXCL|O_NOFOLLOW` with `dir_fd` refuses a planted symlink (`EEXIST`);
  - `fcntl(fd, F_FULLFSYNC)` succeeds on the file and on the directory fd.
  - Linux has no `F_FULLFSYNC`, so it falls back to `os.fsync`.
- What is left:
  - `report_dir_fd` is opened after a `realpath` overlap check, which is a TOCTOU window. Open it with
    `O_DIRECTORY|O_NOFOLLOW` first, and run the checks on the fd (`fstat`, compare `st_dev`/`st_ino`).
  - `log_mode=0o640` is mentioned but is not in `create()`'s signature.

### 17. The audit-hook proofs see only Python-level I/O. NIT, fail-closed

- The GGUF is opened by C (`llama_model_load_from_file`), so the hook raises no `open` event for it. Python opens
  `os.devnull` (`_utils.suppress_stdout_stderr`).
- Metal writes its shader cache from C into the per-user cache directory, outside a fresh `HOME`/`TMPDIR`.
- **Fold.** Restate the open-set pin and the no-effect test as "Python-level". The disk part stays item 7's xfail.
  Add `huggingface_hub`, `httpx` and `maxim.utils.http` to the import-closure ceiling.

### 18. `current_policy` across threads: sound, with one condition. NIT, fail-closed

- `runtime/concurrent_executor.py::ConcurrentExecutor.execute_batch` submits `self.executor.execute` itself, so the
  variable is set inside the pool thread.
- `simulation/tools.py::SpawnSubSimulationTool` builds its sub-executor (L1529) before starting its
  `threading.Thread` (L1575).
- The condition: the read must happen synchronously in `execute()` before any thread starts. In 3.12 a
  `threading.Thread` does not inherit context.
- `executor.check()` does not set the variable, so an auto-fire composite refuses. State that.

## Stop-rule verdict

**Not STOP. Another delta round is owed.**
- Fail-open: findings 1, 2, 3, 4, 6, 8, 9 and 11.
- New classes among them: 1, 3 and 9. In 4, 6 and 8 the fold of a covered class is incomplete or broken.
- None is DO-NOT-BUILD. Each has a mechanical fold, and the independent layers (the pinned READ executor, the
  dispatch check) bound the observer's effect meanwhile.

The next round should be scoped to the v3 text of those eight folds:
- the latch's event list, its self-engage and its boundary statement;
- the subprocess realpath/env rule;
- the live-child endpoint;
- the token-id prompt;
- the env-wrapper iteration and the refusal list;
- `ToolPolicy` immutability;
- `WRITE_LOCAL`.

Findings 5, 7, 10 and 12–18 are fail-closed. They can be folded in the same edit or go to a follow-up issue.
