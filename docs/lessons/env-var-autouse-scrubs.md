# Opt-in env vars in hot startup paths need autouse scrubs

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] Opt-in env vars in hot startup paths need autouse scrubs:** When you wire a new `if os.environ.get("MAXIM_FOO"): do_side_effect()` branch into anything reachable from `build_primary_router` (auto-spawn, tier detection, ensure_available, ...), pair it in the same commit with a `@pytest.fixture(autouse=True)` env-scrub fixture in [tests/conftest.py](tests/conftest.py). Without it, ANY test that sets the env var (e.g., a `normalize_args` unit test asserting `--auto-download` populates the var) leaks into every later test that constructs the runtime — and the leaked side effect runs for real. P5 cost a 9-minute pytest hang on a real 1 GB GGUF download to `~/.maxim/` before this was caught. The two existing scrubs (`_isolate_maxim_llm_profile_env`, `_isolate_maxim_auto_download_env`) are the template. Regression guard: [tests/conftest.py](tests/conftest.py) — autouse env-scrub fixtures pattern; new env-var branches must add a matching scrub in the same commit.
