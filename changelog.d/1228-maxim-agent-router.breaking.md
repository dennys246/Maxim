- **`MaximAgent(shared_router=)` is a required keyword** (#1228): every caller states which router the agent's
  ExecAgent runs on, so none builds a second, private one by default. Pass the router you already hold,
  `maxim.agents.exec_agent.NO_ROUTER` for an agent with no LLM (it then builds none), or `None` to keep the old lazy
  build from `load_llm_config()`. `maxim.run()`'s agent and the console's talk agent now use their own run's router.
  The `agent_profiles.exec_agent` override applies only to that lazy build; a handed-in router bypasses it, and a
  WARNING says so once when they differ. The robot runtime hands its late-built LLM over with the new
  `ExecAgent.adopt_router(router, worker)`.
