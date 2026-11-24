- **A model's reply no longer reaches the logs as text** (#1775): the router's JSON, `ANSWER_ONLY` and `TOOL_PROMPT`
  paths logged the first 200, 500 and 300 characters of every reply at DEBUG, and `AgentPool` the first 80 of each
  NPC reply. `MAXIM_LOG_FILE` records DEBUG, and a reply can quote a PRIVATE or SECRET tool's output it was shown.
  They now log the reply's length and a stable digest (`cloud_redaction.text_shape`), as #1751 does for tool text.
  Nothing logs reply text any more, so seeing the reply behind a parse failure means reproducing the call.
