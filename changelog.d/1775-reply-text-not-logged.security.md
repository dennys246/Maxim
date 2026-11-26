- **A model's reply no longer reaches the logs as text** (#1775): replies, and text taken from them, were copied into
  log records. The sites were the router's JSON, `ANSWER_ONLY` and `TOOL_PROMPT` paths (50–500 characters, at DEBUG,
  INFO and, on a parse failure, WARNING), an NPC's line, the narrator's preview, the scenario generator's parse failure,
  the TTS lines, the critic's issues and the agentic stream's reasoning and plan previews. `MAXIM_LOG_FILE` records
  DEBUG, and a reply can quote a PRIVATE or SECRET tool's output it was shown. Each `LLMRouter` reply is now logged once,
  at the router's one dispatch, as its length and a stable digest (`cloud_redaction.text_shape`), as #1751 does for tool text,
  and every other site logs the shape or a count (`exec_agent`'s `goal_proposed` description too). No other log line or backend
  trace keeps reply text, so seeing the reply behind a parse failure means reproducing the call. The sim stream's `RESPONSE`
  record, `DELIBERATION`'s reasoning, the `SCENE` record (the narrator's prose and the approval prompt's quoted
  reasoning) and the `PERCEPT` record (the agent's incoming text, the narrator's prose and the person's typing
  included) keep their text in the sim-session JSONL, the console's Talk view, the ThinkingPanel and the Adventure
  view, but their copy in `MAXIM_LOG_FILE` now carries only the shape (owner decision, 2026-11-24). To read a sim's
  text after the fact, read the sim-session JSONL in `sim_sandbox/` (`sim_agent_*.jsonl`, `sim_log_*.jsonl`).
