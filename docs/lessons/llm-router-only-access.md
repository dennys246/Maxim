# LLM access goes through `models/language/router.py`

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

- **[engineering] LLM access goes through `models/language/router.py`**; concrete backends (anthropic/llama/openai/transformers) should not be imported directly from outside `models/language/`. `maxim_peer_backend` is the deliberate exception: `_MaximPeerBackend.for_url(...).health_check()` is the sanctioned cross-module PROBE surface (per the probe-entry-point invariant) — but inference DISPATCH stays router-only even for the peer backend (`bench/recovery_time.py` is the lone sanctioned router-bypass, a deliberate benchmark). Self-hosted peer routes go through `_MaximPeerBackend`; cloud routes stay on `_OpenAIBackend`. Selection is driven by `runtime/lane_backends.BACKEND_CLASSES` + `resolve_backend_class` — adding a new backend type is exactly one line in the dispatch table + one branch in `_classify_backend`, no router edit. The `"maxim_peer"` / `"maxim-peer"` spelling is normalised by `resolve_backend_class`. Regression guard: [src/maxim/runtime/lane_backends.py::BACKEND_CLASSES](src/maxim/runtime/lane_backends.py) (single dispatch table) + CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" step) blocking backend imports outside `models/language/` (allow-listed: `agents/llm_agent.py` — grandfathered pre-router standalone agent, migration to the router is tracked follow-up work; `agents/exec_agent.py` — imports the `PROPOSED_GOAL_TOOL` constant, not a backend class; `_MaximPeerBackend` imports are sanctioned via the probe-entry-point invariant).
