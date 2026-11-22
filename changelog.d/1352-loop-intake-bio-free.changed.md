- **The loop's LLM proposal intake loads no bio module** (#1352): `safe_agent_name` and `tool_to_run` moved verbatim
  from `runtime/tool_dispatch.py` to the bio-free `runtime/loop_state.py`, so `loop_controller` no longer imports
  `tool_dispatch` and `runtime/loop_intake.py` (with the `_llm_unavailable` fallback) stays bio-free: B2's
  prerequisite 1. Behaviour is unchanged; `maxim.runtime.tool_dispatch.safe_agent_name` is removed (import it from
  `maxim.runtime.loop_state`).
