# Auth in health probes

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] Auth in health probes:** Any HTTP health check that probes an endpoint behind API key auth MUST include the auth header. The leader's `_probe_upstream_ready()` was silently getting 401s from an auth-gated llama-cpp-server, causing `llm_ready` to be permanently false. Always send auth in probes, and treat 401 as "server is up" (auth-gated but alive). Regression guard: [src/maxim/models/language/maxim_peer_backend.py](src/maxim/models/language/maxim_peer_backend.py) — `_MaximPeerBackend.health_check` is the canonical probe entry point and sends the `Authorization` header (built inline from `api_key`) at every request site, treating 401 as "server is up".
