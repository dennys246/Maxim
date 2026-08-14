# `_MaximPeerBackend.complete_with_usage()` makes EXACTLY one HTTP call — the load-bearing invariant for Plan 3

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

## `_MaximPeerBackend.complete_with_usage()` makes EXACTLY one HTTP call — the load-bearing invariant for Plan 3

**[engineering] `_MaximPeerBackend.complete_with_usage()` makes EXACTLY one HTTP call — the load-bearing invariant for Plan 3:** the whole point of Plan 3 was killing the ~52s fail-slow caused by `_OpenAIBackend`'s internal gateway-retry loop amplified by the per-lane `_inference_lock`. `_MaximPeerBackend` in `models/language/maxim_peer_backend.py` replaces that path for self-hosted peer traffic. It raises typed `BackendError` subclasses on failure and lets the router's provider-fallback loop handle failover. **Adding a `try: ... except: <call again>` block anywhere in this file re-introduces the incident.** CI grep enforces the rule: `grep -nE "retry|backoff|gateway" src/maxim/models/language/maxim_peer_backend.py | grep -vE "retry_after_s|retry_timeout_s"` must return zero matches. The two allowed parameter-name matches are `BackendOverloaded.retry_after_s` (Plan 2 R2b contract) and `health_check.retry_timeout_s` (inherited from the pre-R2.6 probe signature, used for the liveness two-attempt budget — that's a retry *budget*, not a retry loop). If you need per-provider cooldown, use `LLMRouter._note_provider_overload` / `_set_long_backoff` / `_set_short_backoff` — they apply at the router layer, not inside the backend. The router's `_try_provider` catches the typed exceptions in specific-before-general order (`BackendOverloaded` → `BackendAuthFailed` → `BackendModelMissing` → `BackendInferenceBroken` → `BackendTimeout` → `BackendDown` → `BackendError` → `Exception` safety net). Long-cooldown branches (auth 300s, model_missing 60s, inference_broken 15s) do NOT call `_note_provider_failure` — that would overwrite the hard value with the exponential ramp. Do NOT "helpfully" add a `_note_provider_failure` call for symmetry; it's load-bearing that those branches skip it. Regression guard: CI grep `grep -nE "retry|backoff|gateway" src/maxim/models/language/maxim_peer_backend.py | grep -vE "retry_after_s|retry_timeout_s"` must return zero matches; enforced in [.github/workflows/test.yml](.github/workflows/test.yml).

## `_MaximPeerBackend.complete_with_usage()` makes EXACTLY one HTTP call

- **[engineering] `_MaximPeerBackend.complete_with_usage()` makes EXACTLY one HTTP call.** See the Plan 3 lesson above. The CI grep `grep -nE "retry|backoff|gateway" src/maxim/models/language/maxim_peer_backend.py | grep -vE "retry_after_s|retry_timeout_s"` enforces this. Failover is the router's job, not the backend's. Regression guard: CI grep above, enforced in [.github/workflows/test.yml](.github/workflows/test.yml).

