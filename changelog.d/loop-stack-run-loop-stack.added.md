- **`bootstrap.run_loop_stack`, the one sanctioned way to start the agent loop** (docs/plans/loop_stack_builder.md):
  every stack component (the six positionals, the autonomy controller, the LLM worker, the hippocampus, hub and pain
  bus, the enrichment pipeline and thought gate, the Default Network, imagination, evaluators, the substrate trio,
  the sim's percept source, sink and response policy, the state home and the consolidation flavour) is a required
  keyword, so a caller that forgets one gets a `TypeError` instead of a silently less capable loop. It checks the
  agent's memory hub against the one handed to the loop, and makes the caller state the worker's acting coach and
  embodiment. The console's talk loop is the first caller; the arguments its loop receives are unchanged.
  - `MaximAgent.memory_hub`, a read-only property: the hub `wire_memory_hub` connected, or `None`.
  - A new CI lint, `scripts/lint_loop_stack.py`, fails any other `run_agentic_loop` entry point outside its ledger
    of the 8 sites not yet migrated (plus two permanent entries), and any call of `run_loop_stack` through an alias,
    a `partial` or a splat.
