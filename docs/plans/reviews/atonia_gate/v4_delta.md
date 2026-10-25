# Atonia gate approach notes: v4 delta round (third delta)

**Scope:** only the v4 text of the seven v3 fail-open folds (v3 findings 1, 2, 3, 4, 5, 6, 9), i.e. the unstaged
`git diff -- docs/plans/atonia_gate_approach_notes.md` on top of `02d101c1`. Read-only; probes ran from a scratch
dir with `python -I` against the worktree's interpreter (`.venv`, CPython 3.12.12, llama-cpp-python 0.3.8,
psutil 7.2.0, macOS 26 / arm64). Vocab probes used llama.cpp's `models/ggml-vocab-*.gguf` (from the
llama-cpp-python 0.3.36 sdist in the uv cache; they are not in the 0.3.8 wheel or the repo).

**What held up (no finding):**
- `psutil.Process(child_pid).net_connections(kind="inet")` works without root on macOS for a same-user child
  (0.13 ms; returned `('127.0.0.1', port) LISTEN`). System-wide `psutil.net_connections()` raises `AccessDenied`,
  so the notes are right to name the per-process call. On Linux, `/proc/<pid>/fd` and `/proc/<pid>/net/tcp*` are
  readable for a same-uid process, so it works there too. `llama_cpp.server` runs uvicorn in the child's own pid
  (no fork, `workers=1`), so the LISTEN lands on `Popen.pid`.
- The ECONNREFUSED pre-check is racy (a squatter can bind during the up-to-120 s model load), but that race does
  not matter: the ownership proof catches it, and on Linux the child's own bind then fails (`EADDRINUSE`, child
  exits, fail-closed). It is defence in depth and cheap; keep it.
- `llama_token_get_attr` exists in 0.3.8 (`llama_cpp/llama_cpp.py::llama_token_get_attr`, deprecated, takes a
  `llama_vocab_p`; `_internals.py::LlamaModel.token_get_attr`). `llama_vocab_get_attr` is the non-deprecated name.
- None of the five vendored vocabs (llama-spm, phi-3, command-r, qwen2, llama-bpe) has a single-character
  CONTROL or USER_DEFINED token, so per-character escaping always clears them (phi-3 `</s>` → `<`,`/`,`s`,`>`).
- `ToolPolicy`: with the v4 shape, `copy.copy`, `copy.deepcopy`, `pickle.dumps` at every protocol and a direct
  `__reduce_ex__(2)` all hold (the raising `__reduce__` is reached through `object.__reduce_ex__`).
  `object.__new__(ToolPolicy)` skips the Python `__new__`, but yields an instance whose slots are unset: reading
  `ceiling` raises `AttributeError`, so it fails closed; filling it needs `object.__setattr__` or the slot
  descriptors, both already out of the model.
- `socket.__new__` is a real audit event with args `(sock, family, type, proto)`, and it fires for
  `socket.socket`, `_socket.socket` directly, `socket.socketpair()` (twice) and `socket.fromfd()` (probe).
- `subprocess.Popen`'s audit event is raised after shell rewriting: `shell=True` arrives as
  `('/bin/sh', ['/bin/sh','-c',...])`, so "no shell pinned" does refuse every `shell=True` call.

## Findings

### 1. SHOULD-FIX, fail-open (class covered: v3 #1). A child that exits does not revoke the endpoint

**Evidence.** `atonia_gate_approach_notes.md` §6 layer 1, "The latch reads the spawner's own state" (L516) names
the predicate as `_process is not None and ownership_proven`, and "`stop()`, or the child exiting, revokes the
allowance". Neither named field changes when the child exits. `_process` is cleared only by `stop()`
(`runtime/local_server_spawner.py::LocalServerSpawner.stop`, L421) and `is_running` is the only thing that polls
(L342). Nothing in the v4 text says what flips `ownership_proven` on exit. "Residual window" (L519) then argues a
re-bind "cannot exist, because the port is taken" and files kill-and-rebind under the sandbox. But a crash (OOM, a
GPU fault, a `llama_cpp.server` exception) frees the port with no kill power needed. Any local process, any uid,
can bind an unprivileged loopback port. The next connect then hands the API key and the prompt to it. (Also, on
macOS a `0.0.0.0:port` listener *can* coexist with the child's `127.0.0.1:port` one, as the probe showed. It does
not receive 127.0.0.1 traffic while the child lives, so "the port is taken" is true only for the exact address.)

**Fold.** Name the revocation mechanism, and make it work without `Popen.poll()`'s `_waitpid_lock`. Either:
- re-prove on every allowed connect with `psutil.Process(pid).net_connections()` (LISTEN on `127.0.0.1:port` by
  the pid; 0.13 ms measured). A dead child is a zombie with no sockets, so the check fails and the connect is
  refused. Guard the hook against re-entry, because psutil opens `/proc` files on Linux; or
- run a reaper thread that blocks in `proc.wait()` and clears `ownership_proven` under the lock before it returns.

`os.waitid(..., WNOWAIT)` is not an option on macOS (`hasattr(os, "waitid")` is False there). Restate the residual
as "a check-to-connect TOCTOU of microseconds after a crash". Drop "cannot exist".

### 2. SHOULD-FIX, fail-closed (class covered: v3 #1). The latch refuses the spawner's own pre-proof connects; key-less readiness cannot succeed; the hook can deadlock

**Evidence.**
- The allowance exists only for an endpoint "proven to own" (L506–L515). Every connect the spawner makes before
  the proof is therefore to a non-allowed literal, and the latch refuses it. That covers the ECONNREFUSED pre-check
  (L508), the key-less health probe (L515), and `verify_llm_server_exposure`
  (`local_server_spawner.py::verify_llm_server_exposure`, L222), with its UDP `connect(("192.0.2.1", 9))` in
  `utils/net.py::primary_ipv4_address` (L49) and the LAN `create_connection`. If `OfflineRefusal` is an `OSError`,
  both of the latter are swallowed as "not reachable" and the exposure leg is silently skipped.
- In llama-cpp-python 0.3.8, every route, `/v1/models` included, depends on `server/app.py::authenticate`, and
  there is no `/health`. A probe without `Authorization` gets 401 from our own child.
  `LocalServerSpawner._health_check` (L525) maps that 401 to `_key_rejected = True`, and `_wait_ready` (L511) then
  raises `LLMServerKeyRejected`. So "the health probe connects without the Authorization header" is never ready.
- The hook reads spawner state "under its lock". `self._lock` is a `threading.Lock` (L312), and `start()` holds it
  across `Popen` (L355). If the pre-check connect runs inside that critical section, which is where it must sit to
  be atomic with the spawn, the audit hook re-acquires the same non-reentrant lock on the same thread and deadlocks.

**Fold.**
- Readiness under offline becomes "the child's pid holds LISTEN on `127.0.0.1:port`". llama-cpp-server binds only
  after the model has loaded, so that is a sufficient readiness signal. Then make one authenticated probe. Drop the
  key-less HTTP probe.
- Give the pre-check an explicit scoped exemption: a one-shot, key-free raw connect to that port.
- Under offline, skip the LAN leg of `verify_llm_server_exposure` explicitly, with a reason. Ingress is out of item
  10's scope.
- The hook reads a lock-free snapshot (an atomic tuple attribute), or the spawner uses an `RLock`.

### 3. SHOULD-FIX, fail-open (class covered: v3 #2 / #6). The pinned git pair runs code the latch cannot see

**Evidence.** v4 pins `(<git>, <each git tool's fixed argv prefix>)` (L496) and keeps the git tools under the
offline ceiling via the WRITE_LOCAL exemption (L75 + V7). The repo's own comment above
`tools/base.py::GIT_HARDENING_ARGS` (L660–L670) says clean/smudge/process **filters** "NOT closable by any git
switch" run on `diff` and `add`. `tools/base.py::git_env` (L650) → `host_tool_env` (L506, `HOST_TOOL_ENV_ALLOW`
L484) passes `HOME` and `XDG_CONFIG_HOME`, both adversary 3 (§0). So a hostile `~/.gitconfig` or
`$XDG_CONFIG_HOME/git/config` defines `filter.x.clean = <anything>`, a model-writable `.gitattributes` (not under
`.git`, so `git_dir_write_refusal` allows it) selects it, and git's C-level child does arbitrary egress, which no
Python audit hook in our process sees. Separately, on macOS the first fixed location, `/usr/bin/git`, is an xcrun
trampoline (78 hard links; `xcrun --find git` → the Xcode git). Its target is resolved at run time (xcode-select
state, the `xcrun_db` cache under `TMPDIR`, which is also adversary 3), so the pinned realpath is not the code that
runs.

**Fold (strict option first).** Under offline the exemption list is **empty**: the git tools are `ALL` and V7 drops
them. The weaker alternative is to pin `GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_NOSYSTEM=1` and a fixed `HOME` in
the pair's required env, but a pre-existing repo `.git/config` filter still runs, so it does not close the gap. If
the git tools are kept anywhere, order the macOS list with the real binary
(`/Library/Developer/CommandLineTools/usr/bin/git`, the Xcode path) ahead of the trampoline, or drop
`/usr/bin/git` on darwin.

### 4. NIT, fail-open but out of the model (class covered: v3 #2). `_posixsubprocess.fork_exec` raises no audit event

**Evidence.** `multiprocessing/util.py::spawnv_passfds` (L450), the path for the spawn and forkserver start
methods, calls `_posixsubprocess.fork_exec` directly. The probe recorded no `subprocess.Popen` or `os.*` event for
it. Spawn-mode children are covered by self-engage instead (the child imports `maxim` with `MAXIM_OFFLINE` in its
inherited environ), so the gap is only for code that calls the private C entry point with another executable,
which is the "arbitrary code in-process" class (§0).

**Fold.** Add one sentence to the refusal list: `_posixsubprocess.fork_exec` is unaudited; multiprocessing children
are covered by self-engage; other callers are out of the model. Also note that an argv **prefix** admits any
trailing flags (e.g. `--hf_model_repo_id`). That is acceptable only because the required env carries the HF offline
vars, so say so, or pin the spawner's full argv shape.

### 5. SHOULD-FIX, fail-closed (class covered: v3 #3). "Any id that is not NORMAL" rejects ordinary text in SPM vocabs

**Evidence.** §4.2, "every id in an untrusted span…" (L314). On llama-spm and phi-3, `tokenize(seg, add_bos=False,
special=False)` emits **BYTE**-attribute ids for `\n` (id 13), `\t` (id 12) and every byte of an emoji
(243/162/155/131). Per-character re-tokenization yields the same BYTE ids, so every multi-line event is
`rejected: "special_token"`. The vocab test as written is unsatisfiable on phi-3, which is in its own list. The
attribute is also a **bitmask**: phi-3's `</s>` is `USER_DEFINED|RSTRIP` and 10 controls are `CONTROL|RSTRIP`, so
an implementation that tests `attr == USER_DEFINED` would pass them (fail-open), while `attr != NORMAL` over-rejects.

**Fold.** Forbidden iff `attr & (CONTROL | USER_DEFINED | UNKNOWN | UNUSED)` or `attr == UNDEFINED`. NORMAL and
BYTE are allowed: a byte token is that byte and cannot be a control token. Call `llama_vocab_get_attr`. The test
asserts by mask.

### 6. NIT, fidelity only (class covered: v3 #3). Per-character escape distorts SPM text; the refuse path is untested

**Evidence.** On llama-spm and phi-3, tokenizing each character separately adds SPM's space prefix per call:
`</s>` escapes to `' < / s >'` (probe). That is not a security issue. Because no vendored vocab has a
single-character special token, the "Refuse" branch (L318–L321) is never exercised by the attribute test.

**Fold.** Say that the escape may change whitespace. Add one synthetic test (a monkeypatched attribute lookup that
marks a single-char token USER_DEFINED) for the refuse branch. Name the vocab source: the llama.cpp `models/` dir
at a pinned tag, 0.7–10.9 MB each, not shipped by the wheel.

### 7. NIT, out of the model (class covered: v3 #4). `__class__` assignment forges a fully-formed `ToolPolicy`

**Evidence.** Probe: `class Fake: __slots__ = ("ceiling", "reason")`; `f = Fake(); f.ceiling = "ALL";
f.__class__ = ToolPolicy` passes `type(f) is ToolPolicy` with `ceiling == "ALL"`, with no key, no
`object.__setattr__` and no descriptor. The assignment is checked by `Fake`'s `__setattr__`, not ToolPolicy's,
because layouts with identical slots are compatible. The key is also reachable through
`ToolPolicy.__new__.__globals__`. Both are in-process code (§0), the same class as the slot descriptors (L154).

**Fold.** List `__class__` assignment from a same-layout class, and the key reachable by introspection, next to the
slot-descriptor sentence as out of the model. Adversary 4 (future code) would not write either by accident.

### 8. NIT, fail-closed (class covered: v3 #5). The loader-variable list misses `DYLD_*` names

**Evidence.** L274–L278 omit `DYLD_FALLBACK_FRAMEWORK_PATH`, `DYLD_VERSIONED_LIBRARY_PATH`,
`DYLD_VERSIONED_FRAMEWORK_PATH` and `DYLD_IMAGE_SUFFIX`, all of which act when `libllama`/`libggml*` load (dyld
applies `DYLD_LIBRARY_PATH` by leaf name even to the absolute `CDLL` path). §0 already puts `DYLD_*` out of the
model, and this list is an extra.

**Fold.** Refuse by prefix: `DYLD_*`, plus `LD_PRELOAD`, `LD_AUDIT` and `LD_LIBRARY_PATH`, the same way `GGML_*`
already is.

### 9. SHOULD-FIX, fail-open for adversary 4 (class covered: v3 #6). The AST check's reach and sink list are narrower than the rule it enforces

**Evidence.** L75: the rule names `importlib` of a model-chosen name, but the AST sink list omits it. The list also
omits `__import__`, `runpy`, `compile`, `os.popen`, `os.spawn*`, `pty.spawn`, `asyncio.create_subprocess_*`,
`multiprocessing` and `ctypes`. Reach is "same-module helper" only. Tools already cross modules for process work,
e.g. the git tools call `tools/base.py::git_root_refusal`, which runs `subprocess.run`. There are 27 `Tool`
subclasses outside `src/maxim/tools/`. Under offline the latch still refuses a misdeclared runner's `Popen` (fail
closed), but outside offline a misdeclared runner passes any ceiling that admits `WRITE_LOCAL` (e.g. #1431's Talk
ceiling), and that is the property the check exists for.

**Fold.**
- Resolve calls through `maxim.*` imports transitively, so the reach is a package-wide call graph, not one module.
- Make the sink list match the rule's text (the names above).
- Treat a non-literal `getattr` on `os`, `subprocess`, `importlib` or `builtins` as a sink.
- State the residual blind spots: calls through injected or attribute-held callables (`self._runner(...)`),
  third-party libraries that spawn, and C extensions.

### 10. NIT, fail-closed (class covered: v3 #9). `socket.__new__` sees `family=-1` on fd adoption

**Evidence.** The audit args are taken before fileno handling. `socket.socket(fileno=fd)` reports `family=-1`
(probe), so default-deny refuses adoption (fail-closed). An explicit `family=AF_INET` with the fd of another family
is trusted by CPython, but obtaining such an fd needs a pre-engage socket or C code.

**Fold.** Say that `-1` (adoption) is refused under offline, and add "sockets created before `engage()`" to the
stated residuals (it interacts with v4's install-on-engage, v3 #10).

## Verdict

**NOT STOP.** Every finding is in an already-covered class, and the rounds are converging: 12 → 10, with no new
class and no DO-NOT-BUILD. Three findings are fail-open inside the model and need another, delta-scoped round:
- **#1:** endpoint revocation on child exit;
- **#3:** the git pair or exemption under offline;
- **#9:** the AST check's reach and sinks.

The other seven are fail-closed, a stated residual or an out-of-model clarification, and go to the follow-up
issue: #2 and #5 are SHOULD-FIX but fail-closed (fold them now anyway, because each breaks the feature as written),
and #4, #6, #7, #8 and #10 are NITs.
The fold for #3 is one line under the strict option (exemption list empty under offline). If the owner takes it,
#3 needs only a confirmation in the next round.
