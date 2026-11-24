- **Six log lines no longer carry a PRIVATE or SECRET tool's output** (#1751). Each of these used to copy a slice of
  the tool's text into a log record:
  - the WARNING for a follow-up input that cannot be parsed (its first 100 characters);
  - the safety gate's per-call sim summary (80 characters), which the sim logger mirrors into `MAXIM_LOG_FILE`;
  - its "Tool … failed" line (200 characters of the error, for example a git command's stderr);
  - the `[AGENTIC] tool_result` line the CLI prints to stdout (100 characters);
  - the `[AGENTIC] goal_failed` line (the whole error, for example a test run's stderr);
  - the WARNING when a `write_file` overwrite retry fails (the whole error).

  These reach `MAXIM_LOG_FILE`, the sim stream or stdout, and the INFO and WARNING ones also reached the leader's
  `/v1/debug/logs` buffer. A PRIVATE, SECRET or unclassified tool's text is now shown only by its length and a digest. A PUBLIC or
  INTERNAL tool's text is shown as before.

  Not changed: a sim's `actions.jsonl` still records up to 1,000 characters of each tool's output (the #1404
  decision), and the sim's result keeps 200 characters of each in its action history.
