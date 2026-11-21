`safe_agent_name` and `tool_to_run` moved verbatim from `runtime/tool_dispatch.py` to the bio-free
`runtime/loop_state.py`, so `loop_controller` no longer imports `tool_dispatch` and the agent loop's LLM proposal intake
(`runtime/loop_intake.py`, with the `_llm_unavailable` fallback) loads no bio module: B2's prerequisite 1 (#1352).
Behaviour is unchanged; `maxim.runtime.tool_dispatch.safe_agent_name` is removed (import it from `maxim.runtime.loop_state`).
