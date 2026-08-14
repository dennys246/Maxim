# CWD-relative path resolution in public API verbs is documented per-verb

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

- **[engineering] CWD-relative path resolution in public API verbs is documented per-verb** (CC10, v1.0 freeze, 2026-04-29). Three verbs accept bare or relative paths that resolve against the current working directory: `maxim.benchmark(suite=...)`, `maxim.imagine(scenario=...)`, `maxim.campaign(path=...)`. **Pass absolute paths from async / pip-install / arbitrary-CWD callers** (e.g. `asyncio.to_thread(maxim.imagine, ...)` from a FastAPI handler). Other public verbs are CWD-independent (they go through `os.path.expanduser` / explicit `home_dir=` resolution). The CWD-relative behavior is preserved as a developer-checkout convenience — fixing it post-1.0 (e.g. resolving via package data or a `MAXIM_SCENARIOS_DIR` env var) is a non-breaking add. The `benchmark` failure mode includes the active CWD in the `ConfigurationError` message so the failure is obvious. Regression guard: [src/maxim/api.py](src/maxim/api.py) — three verbs documented per-verb in module docstrings; `benchmark` ConfigurationError surfaces CWD context.
