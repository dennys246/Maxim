# The observer (`maxim.advisor`) — Experimental

`maxim.advisor` runs Maxim as an **observer**. You hand it a batch of events. It says what is happening, what it
would do and why, and it records proposals. **It cannot act.** Its executor can dispatch one tool, `propose_action`,
which records a proposal and touches nothing. Everything else the model names is refused at the gate and recorded.

This is the atonia gate (#1315), named for the sleep paralysis that keeps a dreaming body still while the brain runs.
It is **Experimental**: the API and the report shape may change in a minor release. The design is in
[atonia_gate_approach_notes.md](../plans/atonia_gate_approach_notes.md) §4, §5 and §1A.8.

## Quick start

```python
from maxim import advisor

session = advisor.open(
    home_dir="/srv/observer/home",       # an existing, writable directory (reserved; nothing is written there)
    model="/models/qwen2.5-7b-q4.gguf",  # an existing GGUF file: a path, never a profile name or URL
    report_dir="/srv/observer/reports",  # an existing, writable directory, not inside home_dir
)
try:
    report = session.ingest([
        {"event_id": "e1", "kind": "door", "text": "someone walked past the door", "payload": {"zone": "hall"}},
    ])
    print(report["status"], report["happening"], report["proposals"])
finally:
    session.close()
```

`open()` needs `llama-cpp-python` (`pip install "pymaxim[llm-llama]"`).

## `open()`

Every argument is keyword-only:

| Argument | Default | Meaning |
|---|---|---|
| `home_dir`, `model`, `report_dir` | required | See above. |
| `cursor` | `None` | Your opaque feed cursor; Maxim only echoes it (see Cursors). |
| `max_calls` | `8` | The most calls one turn may make (1–32). |
| `turn_timeout_s` | `120` | The per-turn sampling deadline. |
| `log_mode` | `0o600` | The mode of the log and the reports: `0o600` or `0o640`. |
| `n_ctx` | `None` | The context size. `None` uses the model's trained context, capped at 8192. |
| `max_prompt_tokens` | `None` | The prompt cap. `None` means `n_ctx - 1024`; the completion budget is fixed at 1024. |
| `n_gpu_layers` | `-1` | GPU offload; `-1` offloads every layer. |

`open()` raises `advisor.AdvisorCompositionError` before building anything when:
- a code-loading environment variable is set: `LD_PRELOAD`, `LD_AUDIT`, `LD_LIBRARY_PATH`, `LLAMA_CPP_LIB_PATH`,
  `LLAVA_CPP_LIB`, or any name starting `DYLD_`, `GGML_` or `LLAMA_`;
- `model` is not an existing regular file, cannot be read, or changes while `open()` loads it;
- `home_dir` or `report_dir` is missing, not writable, a symlink, or the same as or inside the other;
- a limit is out of range, or `max_prompt_tokens + 1024` exceeds `n_ctx`;
- the model fails to load;
- the platform is not POSIX (Windows): the offline latch is POSIX-only (#1687), so the observer refuses rather than
  run without it.

`open()` also engages Maxim's process-wide **offline latch**, the same one `MAXIM_OFFLINE=1` raises. It is one way:
nothing in the process can lower it again. The observer reads no Maxim config file and no data home, and of the
`MAXIM_*` settings only `MAXIM_OFFLINE`, the latch's own.

The model is hashed once at `open()`, before it loads, and the report's `provenance.model_sha256` and banner name that
hash. If the file at that path has changed by the time the load finishes, `open()` refuses. llama.cpp loads by path,
so a file swapped and swapped back during the load is not caught: keep the model where only you can write it.

`session.limits` reads back the values the session actually runs with: `n_ctx`, `max_prompt_tokens`,
`completion_tokens` and `n_gpu_layers`.

## `ingest(events, *, stop=None)`

Each event is a dict with `event_id`, `kind`, `text` and `payload` (anything JSON), plus optional `occurred_at` and
`cursor`. Unknown fields, a payload that is not JSON, a string that is not valid UTF-8 (a lone surrogate), or a
`stop=` without `is_set()` raise `advisor.AdvisorInputError` before any turn.

Every byte of every event reaches the model as **framed, untrusted data**. It is never part of the instructions. The
prompt is built as token ids, and any special token an event could smuggle in is escaped. An event that cannot be
escaped is listed in the report's `inputs` as `rejected: "special_token"`, so nothing is dropped silently.

One `ingest()` is one bounded turn. It returns the report and also writes it to `report_dir` as
`advisor_report_<report_id>.json`. If that write fails (a full disk), `ingest()` raises
`advisor.AdvisorReportWriteFailed` and the session's cursor does not advance.

### The report

The schema is `advisor_report/1` (notes §1A.8). The fields you will read most:

| Field | Meaning |
|---|---|
| `status` | `ok`, `error` or `abandoned`; `reason` is set unless `ok`. Never a false `ok`: a refused call leaves `ok` with `refusals` filled. |
| `stop_reason` | `null`, except on `abandoned`/`stopped`: your `request_stop` reason, `"close"`, or `"per_call"` for `ingest(stop=)`. |
| `happening`, `would_do.summary`, `rationale` | The model's own words: **untrusted text**. `would_do.executed` is always `false`. |
| `confidence` | `{"value", "calibrated": false, "source": "model_self_report"}`: not a probability. |
| `proposals` | The structured `propose_action` calls, each with its log `seq`. |
| `refusals` | Every other call, with `error_kind`, the `layer` that refused it, and `effect`. The layer is a gate layer (`registry`, `ceiling`, `pin`, …), or `tool` when `propose_action` ran and refused its own arguments. |
| `covered` | `cursor_in`, `cursor_out` and the event count. |
| `prompt` | `tokens` and `max_tokens`, so you can size your batches. |
| `log_last_seq` | The last log record this report's turn wrote. |
| `banner` | "ADVISORY — Maxim v…, model …; no action taken; not an alert." Show it with the report. |

The failure statuses:
- `error`: `prompt_too_long` (nothing was sampled), `prompt_build_error: …` (the tokenizer failed; nothing was
  sampled), `completion_error: …`, `completion_truncated` (the 1024-token budget ran out mid-answer),
  `schema_invalid: …` or `log_write_failed: …`;
- `abandoned`: `deadline` (the turn timed out; the partial output is never parsed) or `stopped`.

### Cursors

Cursors are opaque strings that you supply and Maxim echoes back:
- Once a cursor is in play, from `open(cursor=)` or on any event in the batch, every event must carry one.
- On an `ok` turn, `cursor_out` is the last event's cursor: every event was judged or listed as rejected. The session
  carries it into the next turn's `cursor_in`.
- On `error` or `abandoned`, `cursor_out` equals `cursor_in`, so you can re-feed the batch.

### Stopping

- `session.request_stop(deadline_s, reason="yield")` is safe to call from a signal handler: it sets three plain
  attributes and returns, with no lock, I/O or exception. Pass a plain `int` or `float` and a `str`; anything else
  reads as "now" and `"unnamed"`. The turn in flight ends at its next token or call as `abandoned`/`stopped`, with
  your reason as `stop_reason`. After that, `ingest()` raises `advisor.AdvisorStopped`, including a call that was
  already waiting for the turn.
- `session.close()` stops the session too. With a turn in flight it waits for that turn until the `request_stop`
  deadline (not at all if you never called it). A turn still running then, in an uninterruptible prefill, writes its
  own report and releases the log, the directories and the model itself when it ends. So stop, then close, is safe
  from any thread. `close()` releases the model's memory.
- `ingest(..., stop=event)` abandons that one call only.
- Maxim's process-wide stop signals do **not** stop the observer, so an embedding process's own shutdown cannot
  abandon a turn it did not start.
- Prefill, the model reading the prompt, cannot be interrupted. Its length is bounded by `max_prompt_tokens`, not by
  the clock.

### The log

`report_dir/advisor_log_<session_id>.jsonl` records every proposal and refusal, each fsynced before the turn goes
on. `close()` writes its close record. Read it with `maxim.advisor.log.read_log(path)`, which needs no session. A lost tail shows as the log's
`last_seq` falling behind a report's `log_last_seq`.

## What the guarantee covers, by effect channel

"Writing a report is the only effect Maxim can have" is enforced channel by channel. Where Maxim's own guard does not
yet reach, **your sandbox is the only guard**.

| Channel | In the observer | Guarded by |
|---|---|---|
| Tool dispatch | One tool, `propose_action`: a sealed registry, a READ ceiling and an identity pin, each refusing on its own. | Maxim |
| Actuation | None: `open()` takes no robot, embodiment, tools or registry. | Maxim |
| Environment | The code-loading refusals above, and a pinned Python-level env read set. C-level reads inside `libllama` are invisible to Python. | Maxim (Python level); sandbox (C level) |
| Context reads | None: no agent loop, no prefetch, no data home, no config file. | Maxim |
| Egress | No router or cloud lane is built and nothing is downloaded. (The router's modules are imported with `maxim.models`, and nothing in them runs.) The offline latch refuses Python-level connects, DNS and spawns. | Maxim (Python level); sandbox (native) |
| Disk | Python-level writes go to `report_dir` only. A C-level Metal shader cache write on macOS is not yet measured (#1566). | Maxim (Python level); sandbox (C level, until item 7) |
| Inbound | None: events arrive only as `ingest()` arguments. | Maxim |
| **The report and the log** | **They carry the model's text.** Treat `happening`, `rationale`, `would_do`, proposals and refusal reasons as untrusted input wherever you display or forward them. | You |

## Conformance with AvServ plan 50 §18 (1.3.5, #1783)

**Status: final for 1.3.5.** Checked against AvServ `origin/main` a59a0fe and first drafted on Maxim `origin/main`
f145e27b. It was re-checked on Maxim `origin/main` 77880606, after #1603 (item 10a, merged as 0f50fdc9) and #1735
(item 10b, merged as 77880606). Neither PR changed anything under `maxim/advisor/` or in
`models/language/closed_local.py`, and no §18 verdict below changed. What they added to the offline latch is listed
under its boundary.

AvServ's read-only observer (plan 50, steps S4a–S5b) builds on this package, and its safety case depends on the points
below. Each point names the code that implements it and the test that fails if it stops holding. "Holds" means the
released code does what §18 says. "Differs" means §18 says less than, or something other than, what the code does,
and AvServ's runner should follow the code.

| # | §18 says | Verdict | Code | Test |
|---|---|---|---|---|
| 1 | `open(n_ctx, max_prompt_tokens, n_gpu_layers)` feed `session.limits`; `n_ctx=None` is at most 8192, clamped by the model; `max_prompt_tokens` defaults to `n_ctx - 1024`, and a cap that does not fit is refused | Holds | `advisor/_session.py::_resolve_limits`, `models/language/closed_local.py::DEFAULT_CTX_CAP` | `tests/integration/test_advisor_gate.py::test_open_exposes_limits`, `::test_closed_local_default_context_is_clamped`, `tests/unit/test_advisor_session.py::test_open_refuses_out_of_range_limits` |
| 2 | Cursors are opaque and per event; once one is in play every event carries one (else `AdvisorInputError`); `cursor_out` is the last event's cursor on `ok` and equals `cursor_in` on `error`/`abandoned` | Holds | `advisor/_session.py::AdvisorSession.ingest`, `::_turn` | `test_advisor_gate.py::test_cursor_out_follows_the_last_event`, `::test_cursor_required_on_every_event_when_used`, `::test_cursor_carries_across_ingests` |
| 3 | `ok` means every event was judged or listed as rejected; there are no partial results | Holds for the **report**. See difference C below for the log | `advisor/_session.py::_judge` (only its last return is `ok`) | `test_advisor_gate.py::test_report_status_rows`, `tests/unit/test_advisor_session.py::test_a_stop_between_dispatches_abandons_the_turn` |
| 4 | `request_stop(deadline_s, reason="yield")` is signal-safe; the turn ends `abandoned` with `stop_reason` set; every later `ingest()` raises `AdvisorStopped` | Holds. The report's `reason` is `"stopped"`, and `stop_reason` carries your reason | `advisor/_session.py::AdvisorSession.request_stop` | `test_advisor_gate.py::test_session_stop_abandons_at_the_next_token`, `tests/unit/test_advisor_session.py::test_request_stop_reads_only_plain_values`, `::test_an_ingest_waiting_on_the_lock_runs_no_turn_after_a_stop` |
| 5 | A `prompt_too_long` report carries the token counts, and the model never runs | Holds. The counts are in the report's `prompt` (`tokens`, `max_tokens`), not in `extra` | `advisor/_session.py::_judge` (checked before any sampling) | `test_advisor_gate.py::test_prompt_too_long_reports_token_counts` |
| 6 | The library installs no signal handlers and never exits the process | Holds. No module the observer loads calls `signal.signal`. Before 1.3.5 nothing enforced this; this PR adds the guard, which was proven by planting a handler | `advisor/`, `closed_local.py` (no `signal` use) | `test_advisor_gate.py::test_observer_installs_no_signal_handler` (fake and real backends) |
| 7 | The offline latch refuses network and subprocess for the whole process once `open()` runs | Holds **at the Python level only**. See the boundary below | `_offline.py::engage`, called first in `advisor/_session.py::open_session` | `tests/unit/test_advisor_session.py::test_open_engages_the_offline_latch_before_it_builds_anything`, `tests/integration/test_offline_latch.py::test_engaged_latch_refuses_each_channel`, `::test_engaged_latch_allows_inbound_listen` |

### The offline latch's boundary

`open()` engages the latch before it builds anything, and the latch is one way: nothing in the process lowers it.
It is a `sys.addaudithook` hook, so it sees only what CPython's `socket`, `os` and `subprocess` modules announce as
audit events. While the latch is up, it refuses:
- every outbound `connect`/`connect_ex`, loopback and `AF_UNIX` included;
- every `sendto`/`sendmsg` to an explicit address;
- every forward and reverse DNS lookup;
- creating a socket of any family other than INET, INET6 and UNIX;
- `subprocess.Popen`, `os.system`, `os.exec*`, `os.posix_spawn*`, `os.spawn*`, `os.fork` and `os.forkpty`.

Inbound `bind`, `listen` and `accept` pass.

**What 10a and 10b added to the latch (1.3.5).** The set of refused events above is unchanged. New:
- each refusal-ledger entry (`_offline.Refusal`, a NamedTuple) gains a fourth field, `thread`, the ident of the
  refusing thread (default 0). A reader that unpacks entries **by position** into three names breaks; read the fields
  by name;
- a `Degraded` record, written by `mark_degraded(site, reason)` when the latch forces a fallback, with one WARNING per
  site, and read by `degraded()`;
- `local_files_only_kwargs()`, so the Hugging Face load sites pass `local_files_only=True` while the latch is up;
- `refuse_if_offline(site)`, the typed early refusal the agent's call sites put before a side effect;
- `maxim.exceptions.SECURITY_REFUSALS = (LeaderRefusal, OfflineRefusal)`, the errors a broad handler around building
  the LLM must re-raise.

The observer reads none of them (no module under `maxim/advisor/` and not `closed_local.py`), so none reaches a report
or the log.

**It does not see:**
- **native code.** Networking or process creation done in C, C++ or Rust raises no audit event. That includes
  libuv, gRPC, libzmq, libcurl, Rust HTTP clients, and `ctypes` calls into libc.
- **a connection opened before the latch went up.** `send`/`sendall` raise no audit event, and neither does a socket
  adopted from such an fd with an explicit family. In the observer, `open()` engages first, so the only such
  sockets are ones the embedding process opened before it called `open()`.
- **`_posixsubprocess.fork_exec`.** It is what multiprocessing's `"spawn"` start method calls, and it can be called
  directly. So §18.1's "every `Popen`/`exec`/`spawn`" holds for the Python APIs named above, not for every way a
  process can start one.
- **any platform other than POSIX.** There `open()` refuses with `AdvisorCompositionError` instead of running
  without the latch (#1687).

The OS sandbox covers everything in that list. Profile B2's `--network none` covers the network half. Nothing in
B2's row of §9 stops a process from executing a binary already in the image. If the safety case needs "no
subprocess" to hold below Python, that is a job for the container (for example a seccomp profile that refuses
`execve`), not for this library.

### Where the code says more than §18

- **A. An eighth refusal `layer`, `tool`.** §18.2 lists `forbid`, `mode`, `registry`, `permissions`, `ceiling`, `pin`
  and `offline`. A `propose_action` call that ran and refused its own arguments is recorded with layer `tool` (owner
  decision 2026-11-02). It is not a gate refusal, and the digest should count it as its own value.
- **B. `AdvisorReportWriteFailed`.** When the report cannot be written durably (a full disk), `ingest()` raises it
  instead of returning a report, and the session's cursor does not advance. Treat it as a turn that was not judged,
  like a crash: re-ingest the batch in a new session or record a gap. Guard:
  `tests/unit/test_advisor_session.py::test_a_failed_report_write_raises_typed_and_keeps_the_cursor`.
- **C. A non-`ok` turn can leave records in the log.** Each proposal and refusal is fsynced to the log before the
  next call runs. So a stop or a log failure that lands between two calls leaves the earlier records in the log under
  that `report_id`, while the report itself is `abandoned` (or `error`) with no proposals. The report is the
  judgement. A log record whose report is not `ok` is not a judgement and must not be counted as one. Guard: the
  `log_last_seq == 1` assertion in `test_a_stop_between_dispatches_abandons_the_turn`.
- **D. `prompt_build_error` reports `prompt.tokens = 0`.** When the tokenizer fails, the report is `error` with reason
  `prompt_build_error: …`, nothing is sampled, and `prompt.tokens` is 0. §18.3's batch sizing uses
  `prompt.tokens` per event. Size from `prompt_too_long` and `ok` reports only, never from this one.
- **E. Report fields that §18.2's table omits:** `stop_reason`, `created_at` and `extra` (an empty object). A reader
  that refuses only an unknown `schema` or major `_format_version`, as §18.2 says, accepts them.

### Smaller notes

- `log_mode` defaults to `0o600`, so the runner must pass `log_mode=0o640` for the `digest` job's group read, as §18.2
  shows. Only those two modes are accepted.
- §9's B2 row says the observer reads no `MAXIM_*` variable. It reads one, `MAXIM_OFFLINE` (the latch's own). And
  `engage()` writes five variables into its own process environment: `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
  `HF_HUB_DISABLE_TELEMETRY`, `YOLO_OFFLINE` and `MAXIM_OFFLINE`. Neither changes anything for B2.
- `degraded` is still never produced, as §18.4 expects. That stays true with item 10b: the offline latch's new
  `degraded()` record is not read by the observer and never reaches a report. An embedder that also loads Maxim's
  agent stack in the same process can see `maxim.offline` "degraded" WARNINGs in its own log; those are not observer
  output.
