# Atonia gate approach notes: v5 delta round (fourth delta round)

**Scope:** only v4 → v5 (`git diff -- docs/plans/atonia_gate_approach_notes.md`): (a) the V8 deletion (§1 V8 row; §6
latch bullets, router predicate, CLI startup refusal, inputs table) checked against the whole note and the code;
(b) the widened AST check in §2.1; (c) folds #5 (BYTE bitmask), #8 (`DYLD_*` by prefix) and #10 (fd adoption), for
correctness only. Read-only at `02d101c1` + the unstaged v5 diff. Probes ran from a scratch dir with
`.venv/bin/python -I` (CPython 3.12.12, llama-cpp-python 0.3.8, torch 2.10.0, transformers 5.5.4, macOS).

**Probe results:**
- Under an audit hook, these emit **no** spawn, connect, `getaddrinfo` or `socket.__new__` event:
  - `import llama_cpp`, `import torch`, `import transformers`;
  - `uuid.uuid1()` and `uuid.getnode()` (3.12 uses `_uuid`), `getpass.getuser()`;
  - `platform.system()`, `.machine()`, `.node()`, `.release()`, `import logging.handlers`;
  - `ctypes.util.find_library('c')` on macOS;
  - uvicorn `Config.bind_socket()` on `127.0.0.1`, and asyncio `create_server('127.0.0.1')`.
- llama-cpp loads `libllama` with `ctypes.CDLL(<absolute path>)` (`llama_cpp/_ctypes_extensions.py` L67). It does not
  use `find_library` or a subprocess, so "refuse every spawn" does not break the in-process lane.
- Each of these **does** emit an event:
  - `platform.platform()` / `platform.processor()` spawn `uname -p` and `file -b`. Nothing in `src/maxim` calls them;
    the only call is `platform.mac_ver()` in `doctor/platform_detect.py`, which spawns nothing;
  - asyncio `create_server('localhost')` → `socket.getaddrinfo`;
  - `http.server.ThreadingHTTPServer(('127.0.0.1', 0), …)` → `socket.gethostbyaddr`, from `server_bind` →
    `getfqdn`;
  - `socket.socket(fileno=fd)` → `socket.__new__` with `family=-1`;
  - `socket.fromfd(inet_fd, AF_UNIX, …)` → `socket.__new__` with family **1**, the family the caller claimed.
- **The CLI can run an in-process lane today.** `models/language/router.py::LLMRouter._get_backend_for_provider`
  (L379–386) builds `_LlamaCppBackend` or `_PyTorchTransformersBackend` in-process. A local lane gets one from
  `runtime/lane_backends.py::LaneBackendManager._build_local_backend` (L1135) whenever
  `_maybe_auto_spawn_server` (L2552) declines, which happens with no GPU, with `MAXIM_AUTO_SPAWN_LLM_SERVER=0`, or
  when the spawn fails ("falling back to in-process inference", about L2780).

## Findings

### 1. On a GPU host the auto-spawn path is still entered under offline; "never reached, because they refuse first" names no check. SHOULD-FIX, fail-closed, covered class (v3 #1 / v4 #2)

Evidence: `runtime/lane_backends.py::_maybe_auto_spawn_server` (L2552). Its only early returns are the env opt-out,
`has_gpu` false, a `remote_url`, no profile, or a cloud profile. Once past them, it runs four steps in order:
- `check_existing_llm_server` probes `127.0.0.1:8100` (about L2694);
- `kill_stale_llm_servers`, whose `lsof` fallback is a subprocess, inside `except Exception: pass` (about L2716);
- `LocalServerSpawner.start` (`runtime/local_server_spawner.py::LocalServerSpawner.start`, L345);
- then the fallback to in-process.

Note L512 says these paths "are never reached, because they refuse first". In the code they are reached, and the
latch refuses partway through, inside broad catches.

The outcome depends on where `OfflineRefusal` lands (finding 6): either an in-process fallback after swallowed
refusals and a noisy warning, or a CLI that dies at startup on the RTX 5080 leader even though an in-process lane
would work. No egress happens either way.

**Fold:** `_maybe_auto_spawn_server` and `_route_to_reused_server` (L2503) return unchanged at their top when
`is_offline()`. The CLI startup predicate (L511) runs on the lane set **after** that build. State the cost: offline on
a GPU host loses the shared-server dedupe, so each local tier loads its own in-process copy of the model, and a VRAM
budget check should refuse loudly when the tiers do not fit. This is unverified here.

### 2. Engage ordering on the CLI is unstated, and the first thing `_main_impl` does is spawn `nvidia-smi`. SHOULD-FIX, fail-closed, covered class (swallow census, v2)

Evidence:
- `cli.py::_main_impl` (L884–891) calls `utils/gpu_detect.py::ensure_blackwell_guards` before anything else. That runs
  `nvidia-smi` (L79) inside `except Exception` → `logger.debug` (about L103).
- `utils/gpu_compat.py::detect_blackwell` (L46) does the same.

`MAXIM_OFFLINE=1` self-engages at `import maxim`, so in every offline child, and in the parent too if `llm.offline`
engages first, Blackwell detection is refused **silently**. The 5080 then runs without the GStreamer/CUDA guards.
That is a hardware-behaviour change hidden behind a debug log: no egress, but the CLAUDE.md swallow class. The note
says only "read at CLI startup" (L564) and "engage() runs before any router or tool is built" (L491).

**Fold:**
- Pin the engage point: the first statement of `main()`, ahead of `ensure_blackwell_guards` and `_early_leader_boot`.
- State that any child spawned before it is unlatched. Add that to the pre-engage residual beside the pre-engage
  sockets.
- Under offline, Blackwell detection reads the `MAXIM_BLACKWELL` cache env or `torch.cuda.get_device_name` (in
  process, no spawn) instead of `nvidia-smi`, or records `degraded` loudly. Add both detectors to the swallow census.

### 3. V8 makes offline solo-only, and the note does not say so. SHOULD-FIX, fail-closed (stated-consequence clarification), covered class (V8 itself)

Evidence:
- **Peer:** its large lane is a remote URL, so the startup predicate refuses it. That is correct.
- **Leader:** `cli.py::_early_leader_boot` (L827) starts `runtime/leader_proxy.py`'s `_ThreadingHTTPServer` (L2769).
  The probe shows that `http.server`'s `server_bind` calls `getfqdn` → `socket.gethostbyaddr`, which the latch
  refuses unconditionally (L497). So the proxy cannot start, even though L47 puts ingress out of scope. Its upstream
  `:8100` connect is refused as well.

The `except Exception` there warns loudly (`leader_boot_exception`), so this is fail-closed.

**Fold:** add one line under V8: "`llm.offline` implies role solo. A peer refuses at the lane predicate. A leader
refuses at startup, with a typed reason, before `_early_leader_boot`, not through a `gethostbyaddr` refusal deep in
`http.server`."

Also state in the latch bullets that `bind`, `listen` and `accept` are **not** refused, since inbound is out of scope
(L47). The console's uvicorn on `127.0.0.1` binds with no audited resolve (probe). Under offline, a `host="localhost"`
anywhere in a `create_server` call fails at `getaddrinfo`, which is fail-closed.

### 4. Leftover text that still assumes an allow-list or an endpoint. NIT to SHOULD-FIX, fail-closed (consistency), covered classes

- **L499–500, `AF_UNIX`:** "a pinned path allow-list that is **empty for the observer**" says nothing about the CLI.
  V8's row says "no allow-list". **Fold:** under offline every `AF_UNIX` connect is refused, CLI included, with no
  list. Nothing on the CLI path needs one: forkserver, `Manager`, `docker.sock` and syslog are already refused, and
  asyncio uses `socketpair`, which has no connect. With no list, the `fromfd` mislabel in finding 9 is moot.
- **L606**, red gate: "no connect, resolve or spawn **beyond the allow-list**" → "none at all, except the pinned
  pre-engage residuals".
- **L615**: "IP literals only" → "no name and no address; every forward and reverse lookup refused".
- **L619**: "self-engages at install" → "self-engages in `maxim/__init__` from `MAXIM_OFFLINE`" (v5 moved it).
- **L524–526**: "A process that was spawned before `engage()` … inherits `MAXIM_OFFLINE=1`" is false. A child spawned
  before the variable is set cannot inherit it. **Fold:** split the sentence. Spawn-mode multiprocessing children,
  which still run through the unaudited `fork_exec`, inherit the variable and self-engage. Children spawned before
  engage are an unlatched, stated residual (finding 2).
- **V8 row, "No subprocess":** multiprocessing spawn children still start, through `fork_exec`, as L503–505 and L518
  say. **Fold:** "no subprocess the hook can see; multiprocessing spawn children start but are latched by self-engage".
  The callers are `embodied_runtime/media_loop.py` and `data/camera/display.py`. Their robot path is refused anyway
  (V7, `ACTUATE`).

### 5. Under offline no run can carry a code-provenance stamp. SHOULD-FIX, fail-closed, covered class (M1 weak-evidence rule)

Evidence:
- `simulation/report.py::_code_stamp` (L300) reaches `utils/code_tree.py::git_out` (L69), a git subprocess caught by
  `except (OSError, TimeoutExpired)` (L82) → `"unknown"`, dirty `True`.
- `maxim/__init__.py::get_version_info` (L196, L208) and `console/server.py`'s identity `_run` (L972) are also git
  subprocesses, inside broad catches.

Under V8 an offline sim's report therefore says `executed_git_hash: "unknown"` (if `OfflineRefusal` is an `OSError`),
or the report write dies (if it is not; finding 6). Both are fail-closed. But the note never says that an offline run
**cannot be M1-stamped**, so by the weak-evidence rule it can never gate a claim.

**Fold:** state it. Optionally, compute the stamp from `.git` files without spawning (`HEAD`, the ref file,
`packed-refs`) and mark `code_tree_sha256` unavailable under offline. Do not compute it before engage: that is still a
spawn in an offline run, against V8's wording.

### 6. `OfflineRefusal`'s base class is unstated, and it decides between "silent plausible fallback" and "crash" at many `except OSError` sites. SHOULD-FIX, fail-closed, covered class (swallow census, v2)

Evidence: narrow catches the `except Exception` swallow census does not cover:
- `runtime/orphan_reaper.py::_find_orphans_ps` (L126–132): `except (FileNotFoundError, TimeoutExpired, OSError)` →
  `[]`;
- `utils/code_tree.py::git_out` (L82) → `None`;
- `tools/base.py::git_root_refusal` (L691–702) → a refusal string.

If `OfflineRefusal` subclasses `OSError` (natural for a socket refusal), each of these silently returns a plausible
value. If it does not, each crashes its caller. The ledger records the refusal either way, so this is not fail-open.

**Fold:**
- Name the base. The recommendation is a non-`OSError` (`RuntimeError` subclass), so no "connection error"
  fallback absorbs it.
- Widen the swallow census from `except Exception` to every handler, around a census site, that catches a base of
  `OfflineRefusal`.

### 7. AST check (§2.1, L80): implementable, with two false-positive sources to design out. SHOULD-FIX, fail-closed (over-flagging), covered class (v4 #9)

A function-level probe over `src/maxim` finds 38 modules with a sink by the v5 list, 33 of them with a non-literal
argument. A module-import closure (an upper bound) reaches one from 19 of 22 `tools/` modules. Most of that goes
away at function granularity:
- `utils/gpu_compat.py::env_flag`, imported by `filesystem`, `code_tools` and `git_tools`, reaches no sink, although
  its module has `subprocess`;
- `tools/base.py::git_root_refusal`, a real `subprocess` sink, is called only by `git_tools`.

So the reach the note specifies (functions, not modules) is usable. The two real false-positive sources:
- **Import sinks with a literal name.** `tools/reachy.py` has `__import__("logging")` ×4 (L1049, L1092, L1140,
  L1167). `decisions/nac.py` (L4213) has `importlib.import_module("maxim")`.
- **The canonical helper.** `utils/optional_deps.py` (L168, L181) has `importlib.import_module(import_name)`, reached
  by `discovery`, `internet_search`, `narrative` and `reachy`. `runtime/lane_backends.py::resolve_backend_class`
  (L79) has `__import__(module_path)`, reached by anything that builds a router.

Without a rule for these, implementers grow `effect_exemptions.json` or mark tools `ALL`. Marking them `ALL` then
**drops them under offline** (V7). That is fail-closed, but it is exactly the exemption sprawl the list exists to
prevent.

**Fold:**
- An import or compile sink counts only with a non-literal name argument.
- `require_optional_dependency` and `optional_dependency_available` go on a pinned list of literal-name helpers,
  with a test that every call site passes a string literal or an `EXTRA_FOR_IMPORT` key.
- `resolve_backend_class` takes only `BACKEND_CLASSES` values, so it is pinned the same way.
- `compile` matches the builtin `Name` only, never `re.compile`.

### 8. AST check: unstated blind spots. SHOULD-FIX, stated-residual clarification, covered class (v4 #9)

"Reach: … `self.` methods, module functions" does not resolve calls on **other receivers**: `self._sandbox.execute()`,
`self._controller.connect()`, a router or hub instance. Those are the dominant call shape in `tools/`. Two examples:
- `tools/sandbox.py` reaches `utils/sandbox_executor.py` only through instances;
- `hardware/reachy/controller.py` (L245) spawns `ssh -L` behind a controller object.

Inherited methods (the MRO outside the subclass's module), decorators, `functools.partial` and thread or executor
targets are also unresolved. These sit in the same class as the stated blind spots (future code, caught by review and
the census). Under offline the hook refuses every spawn regardless, so the gap matters only for online ceilings.

**Fold:** add them to the stated blind spots, or resolve attribute receivers through annotated types and fail on an
unannotated receiver that reaches a sink-bearing module.

### 9. Fold #10 (fd adoption): correct as written for `socket(fileno=fd)`; `fromfd` reports the claimed family. NIT, fail-closed

Evidence: from the probe above,
- `socket.socket(fileno=fd)` → `socket.__new__ (-1, -1, -1)`;
- `socket.fromfd(inet_fd, AF_UNIX, SOCK_STREAM)` → `(1, 1, 0)`, and the object reports `family=1`.

The mislabel cannot reach INET: a `str` address becomes a `sockaddr_un`, which the kernel rejects on an INET socket.
It also needs in-process code holding an fd (§0). `accept()` and `socketpair()` raise `__new__` with real families
(2, and 1), so default-deny does not break inbound or asyncio.

**Fold:** "fd adoption **with the family omitted** reports `family=-1`"; with finding 4's `AF_UNIX` change, a mislabel
has nothing to connect to.

### 10. Fold #5 (token attribute bitmask): correct; its second stage contradicts it. SHOULD-FIX, fail-closed

Evidence: in the vendored llama.cpp (`llama-cpp-python` 0.3.36 sdist, `src/llama-vocab.cpp`):
- the special-token cache is built from exactly `CONTROL|USER_DEFINED|UNKNOWN` (L3038);
- `special=False` skips only `CONTROL|UNKNOWN` (L3261).

So refusing those three bits matches the parser's set exactly, and BYTE being plain text is right. But the Refuse
branch (L329) says "if an id is **still not NORMAL**". That refuses BYTE ids in an escaped span, and any
`NORMAL|LSTRIP`/`NORMALIZED` combination by equality. It contradicts the bitmask rule one line earlier and rejects
events needlessly. That is fail-closed.

**Fold:**
- One predicate for both stages: an id is refused iff `attr & (CONTROL|USER_DEFINED|UNKNOWN)`. If the owner wants
  extra margin, use the allow-list form instead: accept only if `attr & (NORMAL|BYTE)` and
  `attr & ~(NORMAL|BYTE|NORMALIZED|LSTRIP|RSTRIP|SINGLE_WORD) == 0`, which also refuses UNDEFINED and UNUSED.
- NIT: `pyproject.toml` floors `llama-cpp-python>=0.3.8` with no ceiling, and the installed venv is 0.3.8. Run the
  vocab test against the **installed** library's tokenizer, not only the pinned-tag vocabs, so attribute semantics
  cannot drift unseen.

### 11. Fold #8 (`DYLD_*` by prefix): correct. NIT, fail-closed

dyld applies `DYLD_LIBRARY_PATH` and `DYLD_FALLBACK_LIBRARY_PATH` at `dlopen` by leaf name, and the uv-installed
python is not SIP-protected, so they are live. Refusing the whole prefix is right. It will refuse operators who export
`DYLD_FALLBACK_LIBRARY_PATH` (conda, Homebrew habits), so the docs should say that.

On Linux, only `LD_LIBRARY_PATH` acts at `dlopen`. `LD_PRELOAD` and `LD_AUDIT` act at exec, as listed. `LD_*` by
prefix would be symmetric but is not needed.

## Verdict

Every finding is fail-closed (the action is refused, sometimes noisily or with over-rejection) or a stated-residual
or consistency clarification. Each is in an already-covered class: the spawner/endpoint surface V8 deleted, the
swallow census, the AST check from v4 #9, the pre-engage residual from v4 #10, and M1 provenance. V8 holds against the
code:
- nothing on the offline CLI or observer path needs a loopback connect, a DNS lookup or a subprocess to **work**;
- the in-process llama/transformers lane exists and loads without `find_library` or a spawn;
- what breaks (auto-spawn on GPU hosts, the leader proxy, Blackwell detection, git provenance stamps) breaks closed.

**STOP: no further round owed.** Findings 1–11 go to a follow-up issue. Findings 1, 2, 4, 6 and 10 are worth folding
into the note before `src/` work starts. Any `src/` fold made after this round still gets a delta-scoped round.
