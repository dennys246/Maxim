# `BackendError.fix_hint` is never user-controllable

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] `BackendError.fix_hint` is never user-controllable:** Plan 2 R2b added the typed `BackendError` hierarchy in `models/language/types.py` mirroring `utils/http.py::HTTPError`. Every subclass has a class-level `fix_hint`. Subclasses may interpolate validated identifiers (model names, URLs) into hint strings, but the format strings themselves are always static. Prevents log injection via user-controlled exception content. Access patterns are exactly three: `.status`, `.response`, `.fix_hint`. Do NOT add `raw_body` or any parallel attribute — Plan 3's router bridge counts on the shape matching `HTTPError`. The `INFERENCE_BROKEN_BACKOFF_S = 15.0` constant in the same module is the single source of truth linking router backoff to probe cache TTL; import it, don't duplicate. Regression guard: [src/maxim/models/language/types.py](src/maxim/models/language/types.py) — `BackendError` hierarchy with class-level `fix_hint` strings (static format strings, not user-controllable).
