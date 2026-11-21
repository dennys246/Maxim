- **`MaximAgent(shared_router=)` and `ExecAgent(shared_router=)` are required keywords** (#1228): every caller states
  which router the agent's ExecAgent runs on, so none builds a second, private one by default. Pass the router you
  already hold, `None` for an agent with no LLM (it then builds none, also after `adopt_router(None, ...)`), or
  `maxim.agents.exec_agent.BUILD_OWN` to keep the old lazy build from `load_llm_config()` (the CLI's agentic path,
  which asks for download consent there). `maxim.run()`'s agent and the console's talk agent now use their own
  run's router. The `agent_profiles.exec_agent` override applies only to that lazy build; a handed-in router
  bypasses it, and a WARNING says so once when they differ. The robot runtime hands its late-built LLM over with
  the new `ExecAgent.adopt_router(router, worker)`.
