- **The loop's LLM proposal intake loads no bio module** (#1352): `safe_agent_name` and `tool_to_run` moved verbatim
  from `runtime/tool_dispatch.py` to the bio-free `runtime/loop_state.py`, so `loop_controller` no longer imports
  `tool_dispatch` and `runtime/loop_intake.py` (with the `_llm_unavailable` fallback) stays bio-free: B2's
  prerequisite 1. Behaviour is unchanged; `maxim.runtime.tool_dispatch.safe_agent_name` is removed (import it from
  `maxim.runtime.loop_state`).
- **The prompt-text helpers leave the memory package** (#1352, with #1519's helpers): `bounded_words`,
  `RECALLED_WORDS_MAX_CHARS`, `OPERATOR_SOURCES`, `Provenance`, `provenance_of`, `frame_external`, `whole_frames`,
  `shown_text` and `stored_percept_text` moved verbatim from `maxim.memory.text` to the new `maxim.utils.prompt_text`,
  which imports no bio package, so `agents/prompt_builder.py` no longer loads the memory package (and the loop's
  intake stays bio-free). Behaviour is unchanged; import them from `maxim.utils.prompt_text`.
