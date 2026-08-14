# HTTP errors are typed, not string-matched

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

- **[engineering] HTTP errors are typed, not string-matched.** `maxim/utils/http.py` defines `HTTPError` + subclasses (`HTTPTimeout`, `HTTPConnectionError`, `HTTPAuthError`, `HTTPServerError`, `HTTPClientError`, `HTTPRateLimited`) with `.status` + `.fix_hint`. Callers branch on these instead of parsing exception messages. **Plan 2 R2b SHIPPED** the parallel `BackendError` hierarchy in `models/language/types.py` (`.status`, `.response`, `.fix_hint` — same three access patterns, no `raw_body` or parallel attributes). **Plan 3 R2.5 SHIPPED** the router bridge in `LLMRouter._try_provider` that catches each subclass in specific-before-general order. Backends convert HTTP errors to Backend errors via one-line `except HTTPRateLimited as e: raise BackendOverloaded(...) from e` pairs; the router branches on the typed Backend exceptions. Do NOT introduce a parallel exception type — extend the existing hierarchy in `types.py` + add the corresponding router branch in specific-before-general order. Order violation is the same class of bug the R2c stage-2 probe review round caught (auth mis-classified as inference_broken). `INFERENCE_BROKEN_BACKOFF_S = 15.0` is the single source of truth linking router backoff to probe cache TTL — import, don't duplicate. Regression guard: [src/maxim/utils/http.py](src/maxim/utils/http.py) + [src/maxim/models/language/types.py](src/maxim/models/language/types.py) (typed hierarchies) + `LLMRouter._try_provider` specific-before-general catch order in [src/maxim/models/language/router.py](src/maxim/models/language/router.py).
