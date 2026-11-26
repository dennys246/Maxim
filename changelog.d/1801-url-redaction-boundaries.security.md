- **URLs are reduced where text leaves the process, not per call site** (#1801): an exception's text can carry a
  request URL's `user:pw@`, a private path or a `?key=` query (httpx puts the URL in its errors).
  - `configure_logging` installs a chained record factory. Every stdlib record renders with its URLs reduced: in its
    arguments before a precision cut (`%.50s`), in the formatted message, and in `exc_text`/`stack_info`. It reaches
    every handler, the leader's network-served `/v1/debug/logs` buffer and third-party records included.
  - `MAXIM_LOG_FILE`'s `data` payloads, `log_structured`, the agentic `AbstractionBuffer` (every producer; it feeds
    the memory agent's LLM context) and `sim_log` (its sinks, trail, JSONL, log-file mirror and terminal) reduce
    their structured data too. The conversation is the exception at the sinks: the agent's reply (`RESPONSE`) and the
    person's input (`USER`) reach the console chat with their links whole, and are still reduced everywhere they are
    persisted or logged.
  - On the wire and in records: the console's `/ws` events (but the conversation) and `HTTPException` details, the leader proxy's 4xx/5xx
    bodies and `/v1/debug/install-status`, `ToolOutput.error` (it reaches the planning prompt), recent outcomes, the
    parallel results, Roy's committed `result.json`/`summary.md`, and report.json's `llm_summary`. Memory sees the
    reduced error: failures that differ only by URL path now dedupe as one episode, and NAc links keyed on an error
    that began with a URL relearn. Success summaries are unchanged.
  - Sites that cut exception text now reduce it first, so a cut cannot leave a token as a "host".
    `scripts/lint_cut_before_redact.py` keeps the known forms from coming back.
  - `redact_urls` (now in the stdlib-only `utils/net.py`, re-exported by `utils/http.py`) also reduces userinfo
    that holds whitespace, JSON-escaped (`https:\/\/`) and `Path`-collapsed (`https:/`) URLs, and listed schemes
    in the Windows backslash form. Its accepted residuals are owner exceptions E1801-1..4 in
    `docs/plans/log_url_redaction_1801_approach.md`.
  - Two `AgentOutputManager` defects found on the way (`utils/agent_output.py`): the first `write_to_stream` to a
    stream nobody had opened deadlocked (a non-reentrant lock re-entered by the auto-open), and an empty
    abstraction buffer passed in was silently swapped for the global one (`or` where `is None` was meant).
