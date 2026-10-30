# Which session end each runtime actually reaches

**Established:** 2026-10-24, the T8 (b) design review of `docs/plans/latent_forward_model.md` (wiring lens, D1 and
D2; report [../plans/reviews/grounding_gl4/wiring.md](../plans/reviews/grounding_gl4/wiring.md)). Code read at
`d9dffe6e`. Filed as [#1449](https://github.com/dennys246/Maxim/issues/1449).

## The fact

There is no single "session end". Each runtime reaches a different subset of the teardown, on a different instance:

| Runtime | What runs at the end | Cerebellum saved | `distributor.cleanup_session` |
|---|---|---|---|
| `maxim --sim`, the AUT (`sim_aut`, a per-run `mkdtemp` home) | `runtime/agent_loop.py::run_agentic_loop` → `runtime/bio_integration.py::end_bio_session`: the capture flush, and `MemoryHub.on_session_end_lightweight`, which saves the Hippocampus (since #1438 PR 1a) | **Never** | **Never** |
| `maxim --sim`, the narrator (`sim_orchestrator`, write-but-don't-read) | `runtime/agent_factory.py::AgentInstance.shutdown`, called from `simulation/orchestrator.py::start_simulation_mode` | Yes (`save_cerebellum()`), but this Cerebellum never trains | No |
| The Minecraft harness and the survival scripts | `end_bio_session`, then `bio.on_session_end()` (`runtime/bio_stack.py::BioStack.on_session_end`) | Yes | Yes |
| The console HANDLE (an adopted persistent agent) | `instance.shutdown()` on the adopted agent | Yes (`save_cerebellum()`) | No |
| `api.py` and `maxim.create.agent()` homes | `AgentInstance.shutdown()` | Yes (`save_cerebellum()`) | No |
| `maxim --sim` with an adopted persistent agent as the AUT | the agent loop's `end_bio_session` inside the sim, then the HANDLE's `shutdown()` later | at the HANDLE's shutdown | No |

**Since #1438 PR 1a** every hub session end (`integration/session_end.py::end_session`) saves the Hippocampus with
EC, the embedding store and the cross-layer graph in one read section, before any consolidation. An
`AgentInstance.shutdown` that finds a running loop holding the session hands its after-steps (the cerebellum save, the
hub's workers) to that loop's session end (#1520), so the last row's cerebellum save happens at the loop's end when
the HANDLE stops mid-campaign.

Two rules learned building it (PR 1a's wire-integrity round):
- **An abandon is a session end too.** Anything attached to a generation (`SessionOwnership.defer_until_ended`) is
  reached by `abandon()` as well as by a normal end; an abandon drops handed-over steps (logged), because they persist
  and its bound is spent.
- **A prune is a gate over a LOAD OUTCOME, never over emptiness.** A target store that came up empty (missing,
  unreadable or refused file) looks exactly like "every reference dangles"; `MemoryHub._prune_dangling_refs` resolves
  only against a store that read its file (`store_file_loaded`).
- The hub's end also stops every capture worker and drains the embedder (phase a), each within its share of the
  deadline.

`AgentInstance.shutdown` calls `bio_stack.save_cerebellum()` directly, never `BioStack.on_session_end`, so only the
runtimes that call `on_session_end` themselves run the distributor's cleanup. In `src/`, that is only
`simulation/minecraft_harness.py`. A post-sim `/new` goal recurses into `start_simulation_mode`: it builds a fresh throwaway AUT, or, under
adoption, re-adopts the same persistent agent, so each level has its own in-sim session end.

`--resume-sim` (`_restore_aut_from_session`) restores the Hippocampus, NAc, EC and ATL only. It restores no Cerebellum
and nothing a later stage would add.

## Why it bites

A session-end behaviour wired at `AgentInstance.shutdown` (a save, a refit, a cleanup) passes a caller grep, because
`shutdown` IS called on `--sim`. It is called on the narrator, though, so the `--sim` AUT never runs it. That is the
D43 shape: the piece ships, but the composition the experiment uses never reaches it.

**Rule:** name the runtime AND the instance for every session-end behaviour. Prove it with a test that drives that
runtime's real teardown (a fixture runtime and a mocked LLM) and asserts that the AUT's own stack ran, then prove the
test by deleting the call. A caller grep is not evidence. **Enforced by:** nothing yet; the teardown test owed by
[#1449](https://github.com/dennys246/Maxim/issues/1449) is the first check, and the backlog row for the caller-grep
half of "a fix ships with a caller" is M3.

**Corollary (D1):** the per-invocation body record exists only after `tool.run` returns, in
`Executor._stamp_invocation` / `_interoceptive_outcome`. Anything that needs the consequence (a forward-model write,
an action context) belongs in the executor, not inside the tool (`embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`
runs before the consequence exists).
