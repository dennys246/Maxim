# #1801: URLs reduced at the boundaries, not per call site (approach note)

> **Status:** v11 2026-11-25 (Session A). This is a security boundary, so the adversarial design pass comes before any
> code (CLAUDE.md "Faster without lowering the bar" (1)).
> - **Round 1 (red-team, a different reader):** 2 DO-NOT-BUILD and 6 SHOULD-FIX, all folded below and tagged R1–R8.
>   - FAIL-OPEN: R1 (the factory cannot see `extra=`), R2 (the sink map was missing family 4), R4 (a later factory
>     that does not chain), R7 (`redact_urls` gaps) and R8 (`sim_log` `data` leaves).
>   - FAIL-CLOSED or test-only: R3 (cost and fidelity of rendering at creation), R5 (the claim about children) and
>     R6 (the factory leaking across tests).
>   - Not converged yet, because a new class turned up (family 4), so a delta round is owed on v2.
> - **Round 2 (delta, a different reader):** 0 DO-NOT-BUILD, 6 SHOULD-FIX and 3 NITs, all folded and tagged S1–S6 and
>   N1–N3.
>   - FAIL-OPEN: S1 and S2 (family 4's map, class R2) and S3 (serialization, a NEW class).
>   - S5 (module placement and an import cycle) is a NEW class too, but FAIL-CLOSED.
>   - Converging (no DO-NOT-BUILD; most findings narrower, inside R2/R3/R7). A delta round 3 is owed on S1–S6 only.
> - **Round 3 (delta, a different reader):** 0 DO-NOT-BUILD, 3 SHOULD-FIX and 3 NITs, folded and tagged F1–F3 and
>   Na–Nc.
>   - FAIL-OPEN: F1 (class S3), F2 (cut before reduce, a NEW class), F3 (class R2) and Na (class R7).
>   - Clean: S5's module move.
>   - Converging: findings are narrower each round, and only one new class (F2) appeared. A delta round 4 is owed on
>     F1–F3 only.
> - **Round 4 (delta, a different reader):** 0 DO-NOT-BUILD, 4 SHOULD-FIX and 2 NITs, folded and tagged G1–G6.
>   - FAIL-OPEN: G1 (the F2 lint), G2 (class R2), G3 (class S3) and G5 (class R7, narrow).
>   - NO new class. Converging. A delta round 5 is owed on G1–G6 only.
> - **Round 5 (delta, a different reader):** 0 DO-NOT-BUILD, 4 SHOULD-FIX and 2 NITs, folded and tagged H1–H6.
>   - FAIL-OPEN: H1 (class F2), H2 (class S3), H4 (class R2) and H6 (class R7, narrow).
>   - No new class; every finding is now a site-list completeness question or a mechanic made precise. A delta
>     round 6 is owed on H1–H6 only.
> - **Round 6 (delta, a different reader):** 0 DO-NOT-BUILD, 3 SHOULD-FIX and 4 NITs, folded and tagged J1–J7.
>   - FAIL-OPEN: J1 (class S3), J2 (class R2) and J3 (class F2). J5 and J6 are latent.
>   - All findings are in covered classes, and each is one mechanic made precise.
> - **Owner, 2026-11-25:** keep the stop rule literal. Design rounds continue until one comes back all fail-closed.
> - **Round 7 (delta, a different reader):** 0 DO-NOT-BUILD, 3 SHOULD-FIX and 3 NITs, folded and tagged K1–K6.
>   - FAIL-OPEN: K1 (class S3, latent), K2 and K3 (class F2). No new class.
>   - A delta round 8 is owed on K1–K6 only.
> - **Round 8 (delta, a different reader):** 0 DO-NOT-BUILD, 2 SHOULD-FIX and 4 NITs, folded and tagged L1–L6.
>   - FAIL-OPEN: L1 (class F2), and L2 (class F2, latent: 0 such sites in src).
>   - K1 holds: no `__dict__`/pickling receiver in `src/`, and nothing reads `record.args`/`record.msg`.
>   - A delta round 9 is owed on L1–L5 only.
> - **Round 9 (delta, a different reader):** 0 DO-NOT-BUILD, 2 SHOULD-FIX and 3 NITs, folded and tagged M1–M5.
>   - FAIL-OPEN: M1 (class F2, latent: 0 sites) and M2 (class F2, how the allowlist is keyed).
>   - L1, L3, L4 and L5 hold. The allowlist key was the only fix site not yet specific enough to build; M2 settles
>     it.
>   - A delta round 10 is owed on M1–M5 only.
> - **Round 10 (delta, a different reader):** 0 DO-NOT-BUILD, 3 SHOULD-FIX (all FAIL-OPEN, all latent: 0 sites in
>   src) and 1 NIT, folded and tagged P1–P4. P1's logger cut is now fixed at the root, in the factory, not in the
>   lint. M1, M3, M4 and M5 hold.
> - **Owner decisions:**
>   - 2026-11-25: a separate PR that lands right after #1743, not widened into it.
>   - D1–D4 (2026-11-25, asked together; all took the recommended option):
>     - the factory is installed only by `configure_logging`;
>     - `sim_log` and `log_agentic` are reduced in this PR;
>     - `redact_urls`' whitespace-in-password gap is fixed here;
>     - children are covered by their entry points plus one two-process guard.
>   - D5 (2026-11-25): `ToolOutput.error` is reduced at construction (option a).

## Why

`src/` has 248 `logger.*("... %s", e)` sites that format an exception's text raw. httpx puts the request URL in
its errors, so one transport failure can carry `user:pw@`, a private host path or a `?key=` query. #997 and #1743
reduced URLs by hand at the sites they owned (`utils/http.py::redact_urls`). In #1743's rounds 3 to 6, each round
found one more raw site on the same error path. Per-site wraps do not converge: the boundary has to be where text
leaves the process, not where it is formatted.

**Front-gate question:** can this ride on existing infrastructure? Yes. `redact_urls` is the reducer: it is
idempotent and never raises. `simulation/report.py::_url_scrubbed` is the recursive walker. The URL text helpers
(`url_origin`, the URL regex, `redact_urls`, and `url_scrubbed` moved from report.py) move to the stdlib-only
`utils/net.py`, and `utils/http.py` re-exports them (S5). `http` imports `structured_logging` at module level, so
importing `url_scrubbed` from `http` would be a cycle; it would also load httpx (275 ms) into `utils.logging`,
`sim_logger` and `tools.base`, or, imported lazily inside `getMessage`, while a handler lock is held. The
`("report.py", "_url_scrubbed")` entry in `tests/unit/_orchestrator_source.py::NEW_PRE_SLICE_MODULE_FUNCTIONS` moves
with it. No new module. What the existing per-site wraps cannot
reach is text no call site controls: a third-party library's record, a traceback, a record from a code path
written later.

## Where exception text leaves the process (measured on origin/main 0f50fdc9)

1. **stdlib `logging`**, set up by `utils/logging.py::configure_logging`. `cli.py::main` calls it early for every
   subcommand, the leader and `serve` included, and `api.run` calls it too. Its sinks:
   - the stdout `StreamHandler` and the optional plain `log_file` handler;
   - **`leader_proxy._LogBuffer`**, at INFO+, which is network-served to any cluster-key holder via
     `GET /v1/debug/logs`, tracebacks included;
   - the display handler (`sim_logger._DisplayLoggingHandler`) and `IKWarningHandler`;
   - third-party records (httpx/httpcore at WARNING+);
   - the `MAXIM_LOG_FILE` JSONL handler. Its `StructuredFormatter` writes only `event`/`data`, never `getMessage()`.
     So plain-message text never lands there, but the `extra={"data": ...}` payloads of `log_structured` and the 11+
     `extra=` sites do, raw (R1).
2. **`sim_logger.sim_log`**, which bypasses `logging`. It writes to its sinks (the console `/ws` stream),
   `_log_records` (the report and telemetry), its own JSONL file and the terminal.
3. **`structured_logging.log_agentic`**, which bypasses `logging`. It writes to the `AbstractionBuffer`, whose
   `get_recent` feeds the memory agent's LLM context, to its callbacks, and to stdout under `MAXIM_AGENTIC_CONSOLE`.
4. **Wire and record payloads, which are not logs (R2):**
   - `console/server.py`: the run-failure `_event_hub.publish(..., {"error": f"{type(e).__name__}: {e}"})` goes to
     `/ws`, and two `HTTPException(detail=f"{type(e).__name__}: {e}")` replies go out over HTTP;
   - `runtime/leader_proxy.py`: the `{"error": f"pip upgrade failed: {e}"}` reply;
   - `simulation/report.py`: `llm_summary = f"Roundup failed: {e}"`, written raw into a report.json whose other
     fields are scrubbed;
   - `runtime/executor.py` and 4 other sites build `ToolOutput(error=f"... {e}")`, which reaches the planning LLM's
     prompt and so a cloud provider (D5).

## Shape

- **Family 1, part a: a chained `logging` record factory**, installed by `configure_logging` (D1). It leaves `msg`
  and `args` untouched (R3: rendering at creation costs up to 598 µs for one large dict argument, it would render
  all 423 `logger.debug` sites under `MAXIM_LOG_FILE`'s root=DEBUG, and it hollows out tests that read
  `record.args`). Instead it:
  - reduces lazily, when a handler calls `getMessage`, behind the prefilter, so the cost is paid only when a handler
    formats the record. It uses no closure and no bound method: a closure cannot be pickled (S3), and a bound method
    stored in `__dict__` drags the raw record into `SocketHandler.makePickle`'s blob, or fails on `exc_info` (F1).
    The choice is made by the record's own type, not by the chained factory's identity (F1):
    - a record whose `type(record) is logging.LogRecord` (the stock class, whoever built it) has its
      `record.__class__` set to the module-level `_RedactingLogRecord(logging.LogRecord)`. Its `getMessage` reduces
      BEFORE and AFTER formatting (P1):
      - each `str` or exception argument is wrapped in a module-level proxy whose `__str__`, `__repr__` and
        `__format__` return the reduced text, then `msg % args` runs. So a precision cut in the message
        (`%.50s`, `%.50r`, `%.50a`, `{:.50}`), whatever the logger's name, a message held in a variable, or
        `logger.log(level, …)`, truncates text that is already reduced;
      - the result is reduced again (a URL built across arguments, or one in `msg` itself);
      - other arguments (numbers, None) pass through untouched, so `%d`/`%f` behave as today. That keeps any foreign attributes and pickles, and `makePickle` is clean (probed);
    - a record of a real subclass is rendered eagerly, `msg = redact_urls(getMessage())` and `args = ()`. That pays
      R3's cost only on this rare path. The render is wrapped in `try`: a bad format (`warning("x %d", "notint")`)
      must not raise out of the `logger` call, where inside an `except` block it would mask the original exception.
      On a failure (`TypeError`/`ValueError`/`KeyError`) the record is left untouched: no class swap, which would
      drop a subclass's own `getMessage`, or raise `TypeError` on a subclass with `__slots__` (H3). `handleError`
      then reports the bad format at emit, as it does today. The `except` is recorded with the swallow lint, in the
      Stage-1 form (H3). The subclass path also reduces the traceback fields in place, OUTSIDE the message-render
      `try`, so a bad format string cannot leave them raw (K5). Guarded as the stdlib guards it: `if record.exc_info:
      exc_text = redact_urls(logging.Formatter().formatException(exc_info))` (it raises `TypeError` on `None`, L3),
      and `stack_info` is reduced only when it is set (J5; latent, since there is no `LogRecord`
      subclass in `src/` or the venv). A failed render's `getMessage` raises again at emit and goes to
      `handleError`'s stderr dump: the stated terminal-only gap;
  - makes `exc_text` and `stack_info` data descriptors on `_RedactingLogRecord`. Their `__set__` stores the REDUCED
    value (H2): a descriptor that reduces only on read leaves raw text in `__dict__`, which `SocketHandler.makePickle`
    copies (probed on 3.12–3.14). `LogRecord.__init__` writes `stack_info` (and a caller can pass `exc_text`) into
    `__dict__` BEFORE the factory swaps the class, so right after the swap both are written back through their
    descriptors (`r.stack_info = r.__dict__.get("stack_info")`) (J1). A precompute at creation would miss `logging.makeLogRecord(dict)`: the factory runs with
    `exc_info=None`, and the dict update then installs a raw `exc_text` (G3; latent, since there is no
    `makeLogRecord`/`QueueListener`/`SocketHandler` receiver in `src/`). Every stdlib `Formatter` reads `exc_text`
    after filling it, so the read is reduced. `pickle`, `copy` and `QueueHandler.prepare` come out clean. A gap,
    stated: `makeLogRecord`'s dict update bypasses `__set__`, so that record's `__dict__` stays raw (latent; no
    receiver in `src/`) (H2). The clean-through-`pickle`/`copy`/`QueueHandler.prepare` claim covers `exc_text` and
    `stack_info` ONLY (K1);
  - carries a marker attribute, but whether to install is decided by the behavioural check, not by the marker. A
    chaining wrapper placed over ours hides the marker, and the outcome is a harmless double wrap (N2).

  It is a factory rather than a handler filter because `basicConfig(force=True)` and `sim_logger`'s reconciliation
  rebuild handlers, while the factory survives both (probed, as did `dictConfig`). The bridge logger with
  `propagate=False` goes through it too.
- **Family 1, part b: the formatter and the network buffer.**
  - `StructuredFormatter.format` passes `data` through `url_scrubbed`, because the factory runs before `extra=` is
    attached and cannot see it (R1).
  - `_LogBuffer.emit` also runs `redact_urls` on its formatted string. This is belt-and-braces on the highest-reach
    sink, and it covers a handler that re-renders `exc_info` itself (R4).
- **The chain check is behavioural (R4).** At the end of `configure_logging` and on each `_LogBuffer` install:
  build a record with a credential URL through the current factory and check that it comes out reduced. On a
  failure, re-install on top and log one WARNING; do not raise.
- **Family 2:** `sim_log` passes `message` and `data` through `url_scrubbed` once, before `_dispatch_to_sinks`. It
  recurses and stringifies leaves such as an exception object in `data` (R8).
  - The `"://"`/`r":\/\/"` prefilter (a raw string: the non-raw form raises a SyntaxWarning on 3.12+, Nc) lives
    inside `redact_urls` itself, so `url_scrubbed` gets it on every string:
    5.1 ms drops to 0.43 ms on large data with no URLs (S6).
  - Shape changes that now reach live `sim_log` sinks: a tuple becomes a list, a set or bytes become a str, and a
    URL-keyed dict collapses its keys (S6).
- **Family 3:** `log_agentic` and `log_structured` pass `data` through `url_scrubbed` before the `LogRecord` is
  built.
- **Family 4 (R2), one choke point per surface:**
  - `console/server.py`: `_EventHub._fanout` scrubs, because it sees both `sink` and `publish` (S2). Every
    `HTTPException` detail goes through one `_error_detail(e)` helper: the two run-failure details, the two
    `ConfigurationError` `detail=str(e)` (from `apply_mesh_setup(leader_url, …)`) and `Bad cursor: {e}`;
  - the leader proxy: when `code >= 400`, `_send_json` runs `url_scrubbed` over every string leaf of the body. That
    covers the 500 reply's `"stderr"` key, not only `"error"`, and it keeps a dict-shaped OpenAI error's structure and
    its int `code`, which `maxim_peer_backend._context_overflow_counts` parses (Nb, G2). The scrub runs before
    `json.dumps`, because `Content-Length` is computed from that string (H5). `_send_bytes` parses an error JSON body,
    scrubs its leaves and re-encodes it, never running `redact_urls` over the raw bytes, where the userinfo branch can
    span compact-JSON field boundaries (H5). An error body that does not parse as JSON, or is not JSON content, has
    `redact_urls` run over its decoded text, decoded and re-encoded as `utf-8` with `surrogateescape` (a
    strict re-encode raises `UnicodeEncodeError` on a `\x80` byte, L4), and reduced before the `Content-Length` header
    is computed (J6, K6; latent, since every oasis error goes through `json_response`). That which covers `hivemind/oasis_endpoints.py`'s `{"error": str(exc)}` 400 path (G2); That covers its 25 4xx/5xx sites,
    including `Invalid JSON body: {e}` and the bare `str(e)` (S2);
  - **errors that bypass `ToolOutput` (S1, F3):**
    - `tool_dispatch.py::record_outcome` reduces `error`. It is the one choke point for `recent_outcomes` and
      `llm_worker`, which both feed the planning prompt. Its callers include the `executor.execute`-raised path
      (`rec_outcome(error=str(e))`) and the agent-fallback path;
    - the parallel path: reduce at the `"error": str(e)` construction in `tool_dispatch.py::_dispatch_parallel`.
      `combined_results`, the LLM context, is built from those dicts, which `record_outcome` never touches (F3);
    - the sub-simulation tool: `simulation/tools.py`'s `sub_report["error"] = str(e)`, returned inside a
      *successful* `ToolOutput`, so outside D5 (F3);
    - success-shaped tool payloads that carry an exception's text (G6): `tools/introspection.py`'s two
      `{"error": str(e)}` inside a successful `ToolResult`, `dm_runtime._deliver_and_wait`, and `simulation/minecraft.py::call_action`'s
      `f"bridge send failed: {exc}"` `detail` (M5). Each is reduced where it is built;
    - `roy_runner.py::_persist_result`: result.json's `"detail": f"{type(e).__name__}: {e}"` (the `health_check`
      probe and preflight), `priming["error"]` and a raw `"url"`. Roy artifacts are committed, so `url_scrubbed` runs
      over the whole payload there (H4). The committed `summary.md`, rendered by `_format_summary(result)` from the
      same raw fields, is written as `redact_urls(_format_summary(result))` (J2). `ToolOutput.output` as a whole is deliberately NOT reduced, because
      fetched content keeps its URLs;
    - the leader proxy's `_install_status` writes (`str(exc)` and pip stderr). They are served in a **2xx** body at
      `/v1/debug/install-status`, so `_send_json`'s 4xx/5xx reduction misses them (F3);
    - `loop_planning.py::_abort_drain`'s `detail` (the refusal message);
    - `state.py::RuntimeState.mark_failure` (`last_error`). It reaches only the already-scrubbed report.json, not a
      prompt (F3 checked); it is reduced anyway, as belt-and-braces;
  - `report.py` scrubs `llm_summary` with the other fields;
  - `ToolOutput.error` is reduced at construction (D5a), in `tools/base.py::ToolOutput`, so the prompt, the
    narrator, the console and memory all receive the reduced text. `ToolOutput` is `@dataclass(slots=True, frozen=True)`, so a
    `__post_init__` sets it with `object.__setattr__` (never zero-arg `super()`, which raises `TypeError` in a
    slotted dataclass) and turns a non-str error into a str. `dataclasses.replace`, pickle and copy all keep the
    reduced value (probed). A third-party tool's error is covered too. `http_fetch` failures also carry the raw URL in
    `metadata={"url": url}`; the code round checks whether that reaches a prompt (N1).
- **`redact_urls` (D3, R7):**
  - Whitespace in a password: `https://u:p w@h/p`, `\n` and `\t` all leak a tail today. The fix must not
    over-match prose (S4). Matching any whitespace up to an `@` turned `"see https://a.com and user:x@y"` into
    `https://y` and `"GET https://h failed\nmail bob@ex.com"` into `https://ex.com`, which is silent-wrong.
    So the userinfo is `user:`, with no whitespace before the colon, then a password of at most 64 characters, and a
    userinfo that held whitespace emits `<url>`, not the origin after its `@`. Both prose strings are pinned in tests.
    The scan stays linear (8,000 starts take 1.8 ms, against 1.1 ms today).
    - The userinfo has two alternatives: the existing quote-tolerant `[^\s/@]*@`, so `https://to"k@h` gives the origin
      and not `https://to`, and `user:` plus a password of up to 64 characters that may contain whitespace, which
      gives `<url>` (Na). The code round pins the exact regex in this note before merging.
    - The user part excludes `\`, so the JSON-escaped form cannot reach into the userinfo (G5).
    - Gaps, stated: a password longer than 64 characters that contains whitespace still leaks its tail; so does a
      colon-less userinfo with whitespace (`https://tok en@h/p` keeps `tok` and `en`) (G5); and a user part with a
      backslash and a password with whitespace (`https://D\u:p w@h/x` leaks `w@h/x`) (H6). It is
      narrow (both gaps), because httpx percent-encodes userinfo (`p w` becomes `p%20w`, probed). A remaining over-match is
      fail-closed: `https://h:8100 failed for bob@x.com` becomes `<url>`, which loses text and leaks nothing (Na).
  - Add the JSON-escaped form `https:\/\/`, both to the regex and to the prefilter.
  - The regex stays linear (checked).
- **The per-site wraps stay** (#997, #1743). They are belt-and-braces, and they cut AFTER reducing, which the
  boundary cannot do for them.

## Sites that cut before reducing (F2)

A boundary reducing text that a site has already cut reduces a cut URL. `https://ghp_TOKEN@github.com` cut to
`https://ghp_TOK` survives as a "host", and a tail cut (`stderr[-500:]`) can drop the scheme, so no URL is seen at
all. The boundary cannot repair this, so each such site reduces before it slices:
- `loop_planning.py::_abort_drain` (`f"…{exc}"[:300]`);
- the two `router.py` warmup `str(e)[:50]` sites;
- `leader_proxy`'s `_install_status["error"] = r.stderr[-500:]`;
- `tools/code_tools.py`'s `result.stderr[-1000:]`.

#1743 already fixes `loop_intake.py`'s `[:100]` and `[:60]`.

Round 4 (G1) found more sites than the grep had:
- `openai_backend.py::_sanitize_error_message` (`msg[:max_len]`) and `sanitized_error[:200]`, which reaches
  `TraceRecord.error` and then the JSONL `data`;
- `leader_proxy.py::_run_pip_upgrade` (`stdout[-500:]` and `"stderr": (result.stderr or "")[-500:]`);
- `_install_status["pip_output"]`;
- `tools/code_tools.py`'s `result.stdout[-2000:]` and `leader_proxy._run_pip_install`'s `stdout[-500:]` (H1);
- `bench/recovery_time.py::run_recovery_benchmark`'s `fix_hint[:500]`. Its value is `_classify_error`'s
  `str(exc)`, and it reaches `bench/cli.py`'s `--output` JSONL. It is reduced at `_classify_error`'s exit, so every
  branch is covered: `_hint` covers only some, and the `return "unhandled_" + type(exc).__name__, str(exc)` branch,
  the likeliest to carry a raw httpx URL, skips it (K2, L1). A test pins the `unhandled_*` branch.

**Guard:** an AST lint, `scripts/lint_cut_before_redact.py`, run over `src/maxim`. A line grep was not enough: it
passed `redact_urls(str(e)[:50])` (cut first, then reduced: the very bug) and missed `{e!r}`, single-quoted
f-strings, `{error}`, `.stdout[...]`, and a cut made after the text was bound to a variable (G1). The lint:
- flags a `Subscript` with a `Slice` whose value is **tainted**, and passes it only when the sliced value is itself
  a `redact_urls(...)` / `url_scrubbed(...)` call;
- taint sources: an `except … as` name; a parameter annotated as an exception type (`*Exception`, `BaseException`)
  or named `exc`/`err`/`e`; `.stderr`/`.stdout`; `traceback.format_exc()` (H1);
- taint propagates through `str()`, `repr()`, f-strings (any conversion), `.attr`, `x.method()`, `x[...]`, `BinOp`
  (`%`, `+`), `AugAssign` (`s += str(ex)`), list/tuple displays (`[str(ex)][0]`), `for` and comprehension targets
  over a tainted iterable, tuple-unpack targets (`a, b = str(e), 1`), `NamedExpr` (walrus), and any call with a
  tainted positional OR keyword argument (`" ".join(str(e).split())`, `json.dumps({..: str(e)})`,
  `dict(m=str(e))["m"]`), `BoolOp`/`IfExp` (`(result.stderr or "")[-500:]`), dict displays (`{"k": str(e)}`), plus
  assignments within the function (H1, J3, K3, L2, M1). Non-text callables (`len`, `hash`,
  `isinstance`, `bool`) propagate too: excluding them changes 0 results on src/maxim, so no exclusion is made (L6);
- besides a `Subscript` slice, a **precision cut** over tainted text is flagged (L2, M1):
  - a `FormattedValue` whose spec has a precision with no type, or with type `s`/`r`/`a` (`f"{e!s:.50}"`). A
    precision on a number type (`{duration:.1f}`) is not a cut (M3);
  - `%`-formatting with a precision (`"%.50s" % e`, `"%.*s" % (50, e)`);
  - `"{:.50}".format(e)` and `format(e, ".50")`;
  - `textwrap.shorten(<tainted>, …)`;
  - the logger form (`log.warning("failed: %.50s", e)`) is NOT a lint rule. The factory's argument proxies make it
    cut reduced text, whatever the logger is called (P1);
  - `%`-formatting matches `\.(\d+|\*)[sra]` (`%.50r` and `%.50a` cut too, P1); a non-literal format string or spec
    applied to tainted text fails the lint (P1).
  - `.ljust`/`.rjust`/`.center` only pad, so they are not cuts.

  There are 0 such sites in src/maxim, so this is a ratchet;
- **a name is tainted if ANY assignment to it in the function is tainted** (a monotonic union, not last-writer-wins)
  (J3, K4). With last-writer-wins, `if a: s = str(e)` / `else: s = redact_urls(str(e))` / `s[:50]` would pass. An
  assignment from `redact_urls(...)`/`url_scrubbed(...)` is an untainted assignment, so `s = redact_urls(str(e));
  s[:50]` passes and the branched form fails. Rebinding a tainted name, a parameter or a local, does not clear it
  (`s = str(e); s = redact_urls(s); s[:50]` fails), so the code binds a new name: `safe = redact_urls(str(e))`
  (K4, L5); On 0f50fdc9 the rule gives 23 unique lines (26 raw hits, less 3 reported twice
  for nested functions) (M4), every one a listed site or an allowlist entry (L5) (prototype: `scratchpad/rt1801f/lint_h1.py`, the rule as written here; `rt1801e/lint_proto.py`, which gave 12, had no
  parameter or `.attr` sources) (J3). The narrower rule round 4 proposed missed
  `_abort_drain`'s and `_sanitize_error_message`'s parameters;
- **attributes each hit to its innermost enclosing function only** (a hit in a nested function is not also reported
  under the outer one), and keys each allowlist entry as `file::innermost_function` plus the `ast.unparse` text of
  the sliced expression. Any hit that matches no entry fails. With a function-only key, a future `str(e)[:200]` in
  an allowlisted function would pass silently (M2). Each entry also carries a `count`; more hits than that under one
  key fail, so two identical `str(e)[:50]` in one function cannot share one entry (P2);
- **scope (P3):** the lint scans module level and class bodies too (keyed `<module>` and `Class.<body>`), and keys a
  lambda to its enclosing def, where a lambda's `e`/`exc`/`err` parameters are sources. It detects with the
  enclosing scope's taint (a closure over an outer `e` is tainted) and attributes the hit to the innermost def;
- keys survive `ruff format` (the AST is unchanged, probed). A `ruff check --fix` UP031/UP032 rewrite or a Python
  version change alters the unparse, so the hit goes unmatched and FAILS (fail-closed). The lint runs on CI's pinned
  Python (P4);
- allowlists sites that are terminal-only or carry no exception text, each with a reason:
  - `install_core.py`'s `e.response.text[:500]` and `doctor/checks.py::_child_maxim_dir`;
  - `prefetch.py::discover_files_with_find`, `system_metrics.py::_memory_macos` and `container_runner.py::launch`.
    These three are method-call propagation off a command's `.stdout`, holding file paths, `vm_stat` numbers or a
    container id (J4). "Named `e`" adds 0 false positives (J4);
  - the three `router._try_provider` tags (built from numbers; they come from `getattr(e, …)` propagation) and
    `llm_spawn_record::established_connections`'s lsof line (from the comprehension-target rule) (K3, L5).
  - With K3 the rule catches `sanitized_error[:200]` directly, so that site leaves the blind spot (K3).
- states its blind spot: a cut made in a helper function the taint does not follow. `sanitized_error[:200]` (a
  helper's return value) is in it, so a test pins that site instead (H1).

## Inputs, and how each could make the boundary lie

| Input | How it could lie | Answer |
|---|---|---|
| message text | a URL without a scheme, `Bearer sk-…`, a percent-encoded whole URL | out of scope (URL userinfo, path and query only), stated in the docstrings |
| whitespace or JSON escaping | `u:p w@`, `\n`, `\t`; `https:\/\/` | fixed in `redact_urls` (R7), with a test per form |
| `extra=` / `data` | attached after the factory runs (R1) | reduced in `StructuredFormatter`, `log_structured`, `log_agentic` and `sim_log` |
| `args` | a non-str value, or a `__str__` that raises | rendered lazily by the stdlib `getMessage`; if it raises, logging's `handleError` prints the raw `Arguments:` to stderr: a terminal-only gap, stated |
| `exc_info` / `stack_info` | the message is the traceback's last line | `exc_text` precomputed and reduced; the proxy buffer is also reduced after formatting |
| install order | records before `configure_logging`; Python's last-resort stderr handler; a later factory that does not chain | before setup there are no Maxim sinks except stderr (a gap, stated). The behavioural chain check catches a later factory at each install point; one set after the last check point is caught only at the next `configure_logging`, and nothing in the venv calls `setLogRecordFactory` (a gap, stated; N2). `logging.makeLogRecord` and `deepcopy` keep the reduced text (probed) |
| raw `__dict__` (K1) | `msg` as an exception object, `args`, `exc_info` and `extra=` keys stay raw in `__dict__`, so `vars()`, `pickle` and `copy` carry them; `extra=` keys survive even `makePickle` and `QueueHandler.prepare` | a gap, stated: the boundary reduces what a handler RENDERS, not a record's raw attributes. Latent, since there is no pickling or `__dict__` receiver in `src/` (no `SocketHandler`/`QueueListener`/`makeLogRecord`). The `StructuredFormatter` path reduces `data` on render (R1) |
| serialization (S3) | a closure `getMessage` cannot be pickled, so `SocketHandler` and a multiprocessing queue drop the record | a module-level `LogRecord` subclass, or a `MethodType` bound to a module-level function |
| embedding | a global factory rewrites the host app's records | installed only by `configure_logging` (D1) |
| children (R5) | the LLM server runs `python -m llama_cpp.server`, and `maxim-diagnostics` never calls `configure_logging` | both write only to the terminal or to DEVNULL (stated). Every `cli.py::main` child is covered |
| tests (R6) | 14 test files call `configure_logging` or `cli.main` and never restore the factory | an autouse `tests/conftest.py` fixture saves and restores `logging.getLogRecordFactory()` |
| double application | a per-site wrap followed by the boundary | `redact_urls` is idempotent |

## Guards (each proven by deleting the mechanism)

- A credential URL in `msg`, `args`, `exc_info` and `extra=data`, through every handler `configure_logging`
  installs, plus `_LogBuffer` and the display handler, after a `basicConfig(force=True)` rebuild: only the origin
  survives. `test_followup_parse_log_1751`'s `record.args` assertion still means what it says.
- `sim_log` with a URL in `message` and in `data`, including an exception leaf: all four destinations are reduced.
- `log_agentic` and `log_structured`: `get_recent` and the JSONL record are reduced.
- Family 4: a `ToolOutput(error=<credential URL>)` is reduced; the console run-failure event, each `HTTPException` detail, the proxy's pip reply and report.json's
  `llm_summary`.
- The chain: a non-chaining factory installed after setup is caught and re-wrapped, with one WARNING.
- Two processes (the red-team's split): a child `python -c "from maxim.cli import main; main([...])"` logs a
  credential URL, and the test asserts on the child's output. `/v1/debug/logs` is tested in-process through
  `_ensure_log_buffer`.
- `redact_urls`: one test per whitespace form and for the JSON-escaped form, plus the two over-match prose strings
  (S4).
- S1: an `executor.execute` that raises with a credential URL leaves `recent_outcomes`, the parallel results,
  `_abort_drain`'s refusal and `last_error` reduced.
- S2: a `_send_json(500, {"error": <credential URL>})` and each console `HTTPException` detail come out reduced.
- S3/F1: a record made by Maxim's factory pickles and unpickles, its `SocketHandler.makePickle` blob holds no
  credential, and it renders reduced after a `QueueHandler.prepare`, with and without `exc_info`. Run both with the
  stock chain and behind a chaining third-party factory that returns a plain `LogRecord`.
- F2/G1: `lint_cut_before_redact.py` reports 0 on the branch, and fails on each planted form: `str(e)[:50]`,
  `redact_urls(str(e)[:50])`, `f'{e!r}'[:50]`, `msg = str(e)` then `msg[:50]`, and `proc.stdout[-500:]`. Each fixed site has
  a test with a token placed so that cutting before reducing would leak it.
- H2: a `SocketHandler.makePickle` blob of an exception record holds no credential. H3: a subclass with
  `__slots__` and a bad format string neither raises out of the `logger` call nor loses its own `getMessage`.
- H4/J2: a `roy_runner` result.json and summary.md written after a failed health check hold no credential.
- J1: a `stack_info=True` record carrying a credential comes out clean through `makePickle`, `vars()`, `pickle` and
  `QueueHandler.prepare`.
- P1: `log.warning("failed: %.50s", e)`, `self.log.warning(...)`, `logger.log(30, "%.50r", e)` and a variable message,
  each with a credential URL in `e`, all render reduced; `%d`/`%f` arguments render unchanged. P2: a second identical
  hit beyond an entry's `count` fails. P3: a module-level `except … as e: str(e)[:50]`, a class-body one, a lambda
  `lambda e: str(e)[:50]` and a closure over an outer `e` each fail.
- M1: the lint fails on each planted form: `"{:.50}".format(e)`, `format(e, ".50")`, `"%.*s" % (50, e)`,
  `textwrap.shorten(str(e), 50)`, `(r.stderr or "")[-500:]` and
  `json.dumps({"k": str(e)})[:50]`. It passes `f"{duration:.1f}"`. M2: a new `str(e)[:200]` placed inside an
  allowlisted function fails.
- L1: `_classify_error`'s `unhandled_*` branch with a credential URL leaves `fix_hint` clean. L2: the lint fails on
  `dict(m=str(e))["m"][:50]`, `f"{e!s:.50}"` and `"%.50s" % e`. L3: a subclass record logged with no exception does
  not raise. L4: a non-JSON error body with a `\x80` byte is sent and reduced.
- K3: the lint also fails on a `for` target over `.stderr`, a tuple-unpack, a walrus, `" ".join(str(e).split())[:50]`
  and `json.dumps({"k": str(e)})[:200]`. K2: `fix_hint` holds no credential in the `--output` JSONL.
- J3: the lint fails on the branched assignment, `s += str(ex)`, `[str(ex)][0][:50]` and `("x %s" % ex)[:50]`.
- G3: a `logging.makeLogRecord({... "exc_text": <credential traceback>})` formats reduced. G4: a bad format string
  through a subclass record does not raise out of the `logger` call.
- G2: a `_send_json(500, {"error": ..., "stderr": <credential URL>})` and the oasis 400 body come out reduced, and
  an OpenAI-shaped error keeps its int `code`.
- G6: each success-shaped payload site comes out reduced.
- F3: the parallel path's `combined_results`, the sub-simulation's `sub_report["error"]` and `/v1/debug/install-status`
  all come out reduced.
- S5: `import maxim.utils.logging` (and `structured_logging`, `sim_logger`, `tools.base`) loads no httpx, checked
  with a subprocess and `sys.modules`.

## Decisions

- D1–D4: taken (status block).
- **D5 (taken 2026-11-25: option a).** `ToolOutput.error`, whose text reaches the planning LLM and so a cloud provider.
  - (a) Reduce it at `ToolOutput`'s construction. It is the boundary, but the model then sees `https://host` and not
    the path that failed. The model chose the URL, so the path is in its own action.
  - (b) Reduce it only where it is rendered into a prompt.
  - (c) Leave it, and file it.

## Not in scope (stated)

- Scheme-less URLs, bearer tokens and percent-encoded whole URLs.
- `print`, Rich output, `sys.excepthook`, `handleError`'s stderr dump, and the LLM server child's own output.
