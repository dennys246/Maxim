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
- `model` is not an existing regular file;
- `home_dir` or `report_dir` is missing, not writable, a symlink, or the same as or inside the other;
- a limit is out of range, or `max_prompt_tokens + 1024` exceeds `n_ctx`;
- the model fails to load.

`open()` also engages Maxim's process-wide **offline latch**, the same one `MAXIM_OFFLINE=1` raises. It is one way: nothing in the process can lower it again. The observer reads
no Maxim config file, no `MAXIM_*` setting and no data home.

`session.limits` reads back the values the session actually runs with: `n_ctx`, `max_prompt_tokens`,
`completion_tokens` and `n_gpu_layers`.

## `ingest(events, *, stop=None)`

Each event is a dict with `event_id`, `kind`, `text` and `payload` (anything JSON), plus optional `occurred_at` and
`cursor`. Unknown fields, or a payload that is not JSON, raise `advisor.AdvisorInputError` before any turn.

Every byte of every event reaches the model as **framed, untrusted data**. It is never part of the instructions. The
prompt is built as token ids, and any special token an event could smuggle in is escaped. An event that cannot be
escaped is listed in the report's `inputs` as `rejected: "special_token"`, so nothing is dropped silently.

One `ingest()` is one bounded turn. It returns the report and also writes it to `report_dir` as
`advisor_report_<report_id>.json`.

### The report

The schema is `advisor_report/1` (notes §1A.8). The fields you will read most:

| Field | Meaning |
|---|---|
| `status` | `ok`, `error` or `abandoned`; `reason` is set unless `ok`. Never a false `ok`: a refused call leaves `ok` with `refusals` filled. |
| `happening`, `would_do.summary`, `rationale` | The model's own words: **untrusted text**. `would_do.executed` is always `false`. |
| `confidence` | `{"value", "calibrated": false, "source": "model_self_report"}`: not a probability. |
| `proposals` | The structured `propose_action` calls, each with its log `seq`. |
| `refusals` | Every other call, with `error_kind`, the gate `layer` that refused it, and `effect`. |
| `covered` | `cursor_in`, `cursor_out` and the event count. |
| `prompt` | `tokens` and `max_tokens`, so you can size your batches. |
| `log_last_seq` | The last log record this report's turn wrote. |
| `banner` | "ADVISORY — Maxim v…, model …; no action taken; not an alert." Show it with the report. |

The failure statuses:
- `error`: `prompt_too_long` (nothing was sampled), `completion_error: …`, `schema_invalid: …` or `log_write_failed: …`;
- `abandoned`: `deadline` (the turn timed out; the partial output is never parsed) or `stopped`.

### Cursors

Cursors are opaque strings that you supply and Maxim echoes back:
- Once a cursor is in play, from `open(cursor=)` or on any event in the batch, every event must carry one.
- On an `ok` turn, `cursor_out` is the last event's cursor: every event was judged or listed as rejected. The session
  carries it into the next turn's `cursor_in`.
- On `error` or `abandoned`, `cursor_out` equals `cursor_in`, so you can re-feed the batch.

### Stopping

- `session.request_stop(deadline_s, reason="yield")` is safe to call from a signal handler: it sets an event and
  returns, with no lock, I/O or exception. The turn in flight ends at its next token or call as `abandoned`/`stopped`.
  After that, `ingest()` raises `advisor.AdvisorStopped`.
- `ingest(..., stop=event)` abandons that one call only.
- Maxim's process-wide stop signals do **not** stop the observer, so an embedding process's own shutdown cannot
  abandon a turn it did not start.
- Prefill, the model reading the prompt, cannot be interrupted. Its length is bounded by `max_prompt_tokens`, not by
  the clock.

### The log

`report_dir/advisor_log_<session_id>.jsonl` records every proposal and refusal, each fsynced before the turn goes
on. `close()` writes its close record. Read it with `maxim.advisor.log.read_log(path)`. A lost tail shows as the log's
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
| Egress | No router, no cloud lane, no download. The offline latch refuses Python-level connects, DNS and spawns. | Maxim (Python level); sandbox (native) |
| Disk | Python-level writes go to `report_dir` only. A C-level Metal shader cache write on macOS is not yet measured (#1566). | Maxim (Python level); sandbox (C level, until item 7) |
| Inbound | None: events arrive only as `ingest()` arguments. | Maxim |
| **The report and the log** | **They carry the model's text.** Treat `happening`, `rationale`, `would_do`, proposals and refusal reasons as untrusted input wherever you display or forward them. | You |
