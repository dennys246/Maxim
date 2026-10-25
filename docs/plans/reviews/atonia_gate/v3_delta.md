# Atonia gate approach notes: second delta round (v3)

**Scope.** Only the v3 text of the v2 delta round's eight fail-open folds (1, 2, 3, 4, 6, 8, 9, 11). I read them from
`git diff HEAD -- docs/plans/atonia_gate_approach_notes.md` on `plan/atonia-approach-notes` and checked them against
the code at `02d101c1` and the installed libraries: Python 3.12.12, llama-cpp-python 0.3.8 and uvicorn 0.40.0. The
llama.cpp source I checked is the vendored tree from the cached llama-cpp-python 0.3.36 sdist. Probes ran with
`python -I` from a scratch directory, and nothing was written into the worktree except this file.

**Verdict, up front: not STOP.** The round found seven fail-open findings (1–6 and 9), all SHOULD-FIX. Each one
leaves a v2-delta fold incomplete, in a class an earlier round already covered. None is DO-NOT-BUILD. There are five
NITs.

## What held (probe results)

- **`create_completion(List[int], grammar=)` works as v3 says.** `llama_cpp/llama.py::Llama._create_completion`
  (L1169–L1178) drops both `bos_tokens` and `eos_tokens` when `prompt` is a list. The list passes through verbatim
  (L1203–L1205). No chat template is applied, because that happens only in `create_chat_completion`. `grammar` goes
  to `generate`, whatever the prompt type. So the trusted template must supply BOS itself.
- **Audit-event arguments**, from a probe on 3.12.12:

  | Call | Event and arguments |
  |---|---|
  | `Popen(list, env=)` | `('subprocess.Popen', exe, args, cwd, env)`. `executable` is filled from `args[0]` when it is not given. |
  | `Popen(..., shell=True)` | `('/bin/sh', ['/bin/sh', '-c', cmd], …)` |
  | `Popen(executable='/bin/sh')` | Reports `/bin/sh`, so keying on `executable` is right. |
  | `os.system` | `(b'cmd',)` only |
  | `os.posix_spawn` / `posix_spawnp` | Both raise `os.posix_spawn` with `(path, argv, env)`, and env is always a mapping. `posix_spawnp` passes the bare name. |
  | multiprocessing `spawn` | Raises no spawn event, as v3 states. |
  | Socket calls | `sendto`, `sendmsg`, `connect_ex` (as `socket.connect`), `gethostbyname_ex` (as `socket.gethostbyname`) and `getfqdn` (as `socket.gethostbyaddr`) all raise events. |
- **The env wildcard rule is workable.** I reassigned `os.environ.__class__` to a recording subclass. That produced
  **zero** `__iter__`/`copy` events across:
  - `import maxim`, `maxim.utils`, `maxim.models.language`, `maxim.tools.base` and `maxim.runtime.executor`;
  - `numpy` and `llama_cpp`;
  - a `Llama(vocab_only=True)` load.

  So the rule does not false-positive on the observer's closure today. (httpx's `getproxies_environment` would trip
  it, but httpx is excluded from the closure.)
- **The live check cannot deadlock.** `Popen.poll()` inside the hook takes `_waitpid_lock` non-blocking.
- **`ToolPolicy` holds against the ordinary paths.** `__class__` reassignment goes through `__setattr__` and is
  refused. `copy` returns `self`, and pickling raises.

## Findings

### 1. The live-child endpoint does not prove that the listener is the child. SHOULD-FIX, fail-open

**Already-covered class:** delta #4, whose fold is incomplete.

**Evidence:**
- `runtime/local_server_spawner.py` (L40): "llama-cpp-server loads the model BEFORE binding HTTP", with up to
  `READINESS_TIMEOUT_S` = 120 s.
- `LocalServerSpawner._health_check` (L525–L540) connects to `127.0.0.1:<port>` every 0.5 s, carrying
  `Authorization: Bearer <key>`.

**The attack.** An `ssh -L 8100:…` forward, or any listener already on the port, is the very case delta #4 named.
1. The listener answers `/v1/models` 200 while our child is still loading. Our child is alive, so `poll() is None`
   and the latch allows the connect.
2. `verify_llm_server_exposure` passes, because the forward is bound to loopback.
3. `start()` returns `base_url`. Prompts and the key go to the remote end until the child fails to bind and exits.

Refusing the reuse path does not help. Under offline, the reuse probe is refused at the latch, and that steers
`_maybe_auto_spawn_server` straight into this spawn path.

**A second gap.** `subprocess.Popen._internal_poll` returns `None` ("we know nothing yet") whenever another thread
holds `_waitpid_lock`, so a busy lock reads as "alive".

**Fold:**
- Under offline, before the spawn, a connect to the port must fail with `ECONNREFUSED`; otherwise refuse the spawn.
- Readiness under offline additionally requires that the **listening socket belongs to the child's pid**:
  `psutil.Process(pid).net_connections()` shows `LISTEN` on the port. Under offline, `psutil` comes in through
  `require_optional_dependency`.
- The latch reads the spawner's own state, `_process is not None` under its lock, and does not rely on `poll()`
  alone.
- Name the residual local-race window in the docs.

### 2. An allow-list keyed on the executable alone pins the interpreter, and with it every Python argv. SHOULD-FIX, fail-open

**Already-covered class:** delta #3, whose fold is incomplete.

**Evidence.** The offline CLI path needs `sys.executable` on the allow-list, because
`LocalServerSpawner._build_cmd` (L455–L470) runs `[sys.executable, "-m", "llama_cpp.server", …]`. Once `realpath`
of the interpreter is pinned, every Python spawn passes. Those children do not import `maxim`, so they are
unlatched:
- `runtime/leader_proxy.py` L1949 / L1973 / L2270 / L2280 run `python -m pip install …`, which is network egress
  from pip;
- `utils/sandbox_executor.py` L604 runs `python -c wrapper script_content`, which is model code;
- `utils/plotting.py` L34 runs `-c code`.

The same holds for shells. The probe shows that `shell=True` reports `executable='/bin/sh'` with argv
`['/bin/sh', '-c', cmd]`, so a pinned shell or `/usr/bin/env` admits any command. The `os.system` event carries only
the command bytes, so v3's "`executable`, or else `args[0]`" has nothing to read. An implementer who splits the
string is spoofable: `/usr/bin/git; curl …` would pass.

The env rule also checks only an explicit `env`. When `env` is `None`, the child inherits `os.environ`, which could
have lost `MAXIM_OFFLINE` in-process.

**Fold:**
- Each allow-list entry is a `(realpath, argv-prefix)` pair, for example
  `(python, ["-m", "llama_cpp.server", "--model"])`.
- Interpreters, shells and `env` are never pinned bare.
- **`os.system` always refuses under offline.**
- The env check runs on the effective env: `env`, or `os.environ` when `env` is `None`.

### 3. `tokenize(special=False)` still emits USER_DEFINED tokens, and `add_bos` defaults to True. SHOULD-FIX, fail-open

**Already-covered class:** delta #6, whose fold is incomplete.

**Evidence:**
- In the vendored `src/llama-vocab.cpp` (L3261), `!parse_special` skips only `CONTROL | UNKNOWN` tokens, so
  `USER_DEFINED` tokens are still partitioned out of plain text.
- I probed llama-cpp-python 0.3.8 against the vendored vocab GGUFs. Each listed token's own string, tokenized with
  `special=False`, came back as that token's id:

  | Vocab | USER_DEFINED tokens that survive |
  |---|---|
  | `ggml-vocab-command-r` | 999 of 999, including `<|START_OF_TURN_TOKEN|>`, `<|USER_TOKEN|>` and `<|SYSTEM_TOKEN|>` |
  | `ggml-vocab-phi-3` | `</s>` |
  | `ggml-vocab-qwen2` | 290 of 290 |

  Real Qwen2.5 models mark `<tool_call>` the same way.
- `Llama.tokenize` (`llama.py` L574) defaults to `add_bos=True`. v3 does not say `add_bos=False`, so each untrusted
  segment would start with a BOS token. The llama-bpe, phi-3 and command-r probes added one.

**Fold:**
- Untrusted segments use `tokenize(seg, add_bos=False, special=False)`.
- Then **check every id's `llama_token_get_attr`**. Any id in an untrusted span that is not NORMAL (`CONTROL`,
  `USER_DEFINED` or `UNKNOWN`) refuses the event with a typed status, or is escaped and re-tokenized until nothing
  changes.
- The vocab test enumerates tokens **by attribute**, not by "special" strings.

### 4. `ToolPolicy` can be forged through `__new__` and through a subclass. SHOULD-FIX, fail-open

**Already-covered class:** delta #9, whose fold is incomplete.

**Evidence.** I prototyped the v3 design ("`__setattr__` raises once `__init__` has finished"):
- `p = ToolPolicy.__new__(ToolPolicy); p.ceiling = ALL; …` succeeds, because the instance is not sealed yet. That is
  a policy built without the private key, through ordinary assignment.
- `class Sub(ToolPolicy): __slots__ = (); def __init__(self): pass` with its own permissive `__setattr__` passes
  `isinstance`.
- `ToolPolicy.ceiling.__set__(p, ALL)`, the slot descriptor, is equivalent to `object.__setattr__`, so it is out of
  the model. State that.

**Fold:**
- `__new__` takes and checks the key.
- `__setattr__` and `__delattr__` raise **unconditionally**; `__init__` uses `object.__setattr__`.
- `__init_subclass__` raises.
- `Executor`, `build_executor` and `AgentConfig` check `type(policy) is ToolPolicy`, not `isinstance`.

### 5. The code-loading refusal list still misses loader variables. SHOULD-FIX, fail-open

**Already-covered class:** delta #8.

**Evidence.** v3's list (§4.2, the refusals at `open()`) has `LD_PRELOAD` and `DYLD_LIBRARY_PATH` but not:
- `LD_LIBRARY_PATH`. On Linux it is searched before `RUNPATH`, so it resolves the `libggml*` dependencies when
  `libllama` is dlopened at `open()`.
- `DYLD_FALLBACK_LIBRARY_PATH` or `DYLD_FRAMEWORK_PATH`.
- `LD_AUDIT`. Like `LD_PRELOAD` it acts at exec, so it matters only as a marker of a hostile env.

The only env reads in llama_cpp's Python code are `LLAMA_CPP_LIB_PATH` and `LLAVA_CPP_LIB`. `CUDA_PATH` and
`HIP_PATH` are read on Windows only, in `_ctypes_extensions.py`. So v3's Python-level names are right.

**Fold:** add these four names, and say which act at exec and which at `dlopen`.

### 6. The new `WRITE_LOCAL` rule has no check. SHOULD-FIX, fail-open

**Already-covered class:** delta #11.

**Evidence.** §2.2's declaration test checks only for "own non-zero effect". Nothing fails if a tool that spawns a
process, or that runs code in-process through `exec`, `eval` or `importlib` of a model-chosen name, declares
`WRITE_LOCAL`. That tool then passes V7's ceiling and any other ceiling that contains `WRITE_LOCAL`.

The offline hook is a backstop only under offline, and finding 2 weakens even that. Nothing in v3 names the "pinned
exemption list" or a test for it. CLAUDE.md's "Enforced, or on the backlog" applies.

**Fold:**
- Add an AST check over the Tool subclass closure. Any class whose methods reach `subprocess.*`, `os.exec*`,
  `os.system`, `os.posix_spawn*`, `exec` or `eval` must declare `ALL` or appear in a pinned JSON exemption list with
  a reason.
- Widen the definition from "runs processes" to "runs processes or code".

### 7. NIT, fail-closed: the git tools spawn a bare `git`

**Already-covered class:** delta #3 / #11.

**Evidence.** `tools/git_tools.py` (L101, L189, L203) and `tools/base.py::git_root_refusal` (L694) run `["git", …]`.
The hook refuses a bare name, so the git tools that V7 keeps through the exemption are refused under offline every
time.

"Fixed argv" is also not literally true: the `-m message` and `add -- f` arguments are chosen by the model.
`GIT_HARDENING_ARGS` also leaves config-driven commands alone, such as `filter.<x>.clean` on `git add`. That residue
is the #949 class.

**Fold:**
- Resolve git once to an absolute path from a pinned list.
- Reword the rule as "fixed verb + hardening + `--end-of-options`".
- Name the filter residue.

### 8. NIT, fail-closed: `sendmsg` on a connected socket reports the destination as `None`

**Evidence.** CPython raises `socket.sendmsg` with `None` when no address is passed. asyncio's selector transport
uses `sendmsg` for `writelines`. A rule that reads "refuse unless the destination is the allowed literal" therefore
breaks asyncio streams, including streams to the allowed endpoint.

**Fold:** a `None` destination defers to the `connect` that was already audited.

### 9. Address families outside INET, INET6 and UNIX are allowed by omission. SHOULD-FIX, fail-open, low severity

**Already-covered class:** delta #1 / #15.

**Evidence.** v3 refuses `connect` only for `AF_INET`/`AF_INET6` addresses other than the allowed literal, plus
`AF_UNIX`. A `connect` on `AF_VSOCK` (which reaches the host or other VMs), `AF_SYSTEM` or `AF_PACKET` (through
`bind` + `send`) is not refused. The probe shows that `socket.__new__` reports the family. src has no such
socket today.

**Fold:** default-deny. Refuse `socket.__new__` for any family other than INET, INET6 and UNIX, and refuse INET raw
sockets.

### 10. NIT, fail-closed: the self-engage at `maxim/__init__` makes `import maxim` heavier and leaves a hook in every process

**Evidence:**
- Today `import maxim` loads **1** module. Importing `maxim.utils.offline` runs `maxim/utils/__init__.py`, which
  eagerly imports `structured_logging`, `internet_access`, `web_cache`, `response_output` and `content_safety`:
  **51** modules, `socket`, `logging` and `inspect` among them.
- This does not literally break docs/agents/runtime-tools.md's rule ("must not load optional dependencies"), since
  every module is stdlib. It does break the rule's intent ("lazy-loaded to keep `import maxim` fast").
- It also puts an irremovable audit hook in every host application.
- In pytest:
  - `MAXIM_OFFLINE=1` set in a developer's shell engages the latch at collection, before any autouse scrub runs, and
    nothing can undo it;
  - any in-process `engage()` in a test latches that worker for good.

**Fold:**
- `maxim/__init__` reads the env itself, stdlib only, and imports and engages the latch only when it is truthy.
  Otherwise the hook installs at the first `engage()`.
- The latch module stays clear of the eager imports in `utils/__init__`.
- Every `engage()` test runs in a subprocess.
- `tests/conftest.py` asserts that `MAXIM_OFFLINE` is unset before anything imports `maxim`.

### 11. NIT, fail-closed: V7 keeps `ACTUATE`, but the latch refuses the robot

Reachy is driven through its daemon, by a connect to localhost or to a LAN address. The latch allows only the
spawner's child, so motion under offline is refused at the socket. `ACTUATE` survives V7's ceiling on paper only.

**Fold:** say so. A pinned robot endpoint would be the owner's decision.

### 12. NIT: the env test's blind spot

`os.environb` and `posix.environ` (a startup copy) bypass a wrapper on `os.environ`.

**Fold:** wrap or poison both in the env read-set test.

## Stop-rule verdict

**Not STOP.** Findings **1, 2, 3, 4, 5, 6 and 9** are fail-open:
- each sits in an already-covered class (delta #4, #3, #6, #9, #8, #11, #1/#15) whose v3 fold is incomplete;
- none is DO-NOT-BUILD;
- each has a concrete fold above.

A third delta round, scoped to the v4 text of exactly those seven folds, is owed. Findings 7, 8, 10, 11 and 12 are
fail-closed or NIT and can go to the follow-up issue, or be folded alongside the others without their own round.
