# Probe entry point is `_MaximPeerBackend.health_check` — Plan 3 R2.6

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

## Probe entry point is `_MaximPeerBackend.health_check` — Plan 3 R2.6

**[engineering] Probe entry point is `_MaximPeerBackend.health_check` — Plan 3 R2.6:** any liveness or readiness probe against a peer URL MUST use `_MaximPeerBackend.for_url(url, api_key=k, model=m).health_check(enable_stage2=...)`, not the deprecated `runtime.llm_server.probe_llm_server` / `llm_server_responding_at` / `_probe_once`. The three historical shim names were REMOVED from production code in the v1 cleanup (C1); a zero-match CI grep in `.github/workflows/test.yml` blocks re-introduction of the two removed public names — `probe_llm_server` / `llm_server_responding_at` (the internal `_probe_once` helper survives inside `maxim_peer_backend.py` and is not matched by the grep). If you're writing new code that probes a peer URL, go through the backend's method directly. **`_MaximPeerBackend.for_url` is concurrency-safe via instance-level `_api_key_override` — it does NOT mutate `os.environ`.** The R2.5 original shipment wrote the probe key to `os.environ["MAXIM_PEER_PROBE_KEY"]` which races under concurrent probes; the pre-merge review round caught this as a critical finding and the fix stores the key on the returned instance. If you add a new backend factory, use the same instance-attribute pattern — do NOT mutate process-global state from a factory call. Regression guard: zero-match CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("deprecated probe shims" block) — any `probe_llm_server` / `llm_server_responding_at` reference in `src/maxim/` fails CI.

## `_MaximPeerBackend.for_url(url, api_key=k, model=m).health_check()` is the canonical probe entry point

- **[engineering] `_MaximPeerBackend.for_url(url, api_key=k, model=m).health_check()` is the canonical probe entry point** (Plan 3 R2.6). The `probe_llm_server` / `llm_server_responding_at` shims were REMOVED in the v1 cleanup (C1); do NOT re-introduce the names. Regression guard: zero-match CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("deprecated probe shims" block; any match fails CI).

## `_MaximPeerBackend.for_url` is concurrency-safe via instance-level `_api_key_override`

- **[engineering] `_MaximPeerBackend.for_url` is concurrency-safe via instance-level `_api_key_override`** — it does NOT mutate `os.environ`. If you add a new backend factory that needs to accept an override, store it on the instance, never on process-global state. Regression guard: [src/maxim/models/language/maxim_peer_backend.py::_MaximPeerBackend.for_url](src/maxim/models/language/maxim_peer_backend.py) — `_api_key_override` is an instance attribute, not a module-level variable; pre-merge review caught the env-var race and pinned the fix.

