# #1338 (advisor item 6) — Architecture lens review of d362b481

Scope: `git show d362b481`. Read: docs/CODE_REVIEW.md, CLAUDE.md invariants, docs/agents/runtime-tools.md.
Checks run: `lint_no_silent_swallows.py` clean; `lint_function_length.py` clean (`_main_impl` pin 1548, note added); `ruff check` clean on the three src files.

## BLOCKER

None.

## SHOULD-FIX

1. **`create.py::router` still fails silent, so the documented recipe produces a misleading error.**
   `build_primary_router` returns `(None, mgr)` when the large tier resolves to nothing (no profile, a failed backend build, a disabled LLMConfig), and `create.router` returns that `None`.
   Failure scenario: a user without `ANTHROPIC_API_KEY` copies the new docs line `create.pool(llm_router=maxim.create.router("claude-sonnet"))` and gets `ConfigurationError: create.pool() needs an LLM router: pass llm_router=...`, although they did pass one.
   The root cause of this silent path is `create.router` (a stable verb) returning `None`, which is the same fail-loud class as #1338. Either make `create.router` raise `ConfigurationError` naming the profile, or, at minimum, have the pool's message say "llm_router was None (did create.router fail to build a backend?)". Ask the owner which: the raise is a second stable-verb behaviour change.

2. **`runtime/agent_pool.py::AgentPool._generate_npc_response`: an agent with no personality never reaches the router.**
   It returns `None` before the `self._llm_router` check, and `create.agent(personality=None)` is the default.
   Failure scenario: `pool = create.pool(llm_router=r); pool.add(create.agent("scout")); pool.run_turn("scout", ...)` gives a `TurnResult` with `response=None` and `error=None`. That is a successful-looking turn with no LLM call, the exact symptom #1338 targets. Either raise or return an error there, or file it in the follow-up issue alongside the bare-`AgentPool()` stub.

3. **The CHANGELOG entry is under the wrong heading.** The `create.pool` break sits under `### Fixed`, labelled "a behaviour change to a stable API".
   Precedent puts stable-surface breaks under `### Changed` with the label `**Breaking, to stable surfaces:**` (#939/#950, `create.hippocampus`, #1435).
   Failure scenario: a user scanning the release notes for "Breaking" misses that `create.pool()` now raises.
   Move the pool bullet to `### Changed` with that label, and keep the run_turn and server-mode bullets under Fixed. stable_api.md rule 2/3 says argument names and the input/output contract hold within 1.x, so the row should also say "breaking since 1.3.4 (owner decision 2026-10-26)" rather than read as if it had always been so.

4. **`cli.py::_run_server_mode`: the `finally` block's bare `except Exception: pass` around `stop_active_spawner`.**
   The lint credits it as a verbatim move, so it is not a new swallow by count. But the new CHANGELOG text claims "The spawned server is stopped on that exit", and this handler is the only thing standing behind that claim.
   Failure scenario: the stop raises, exit is 1, and an orphan llama server keeps holding :8100 with no log line.
   Replace it with `log_swallowed_exception()` (or a `logger.warning`), and record it in `scripts/swallow_moves.json` if the lint needs that.

## NIT

5. **`create.py::pool`: `llm_router: Any = None`, then a runtime raise.** The `ConfigurationError` type is right: "required configuration missing" is its documented use, and a new type would add surface for nothing.
   A required keyword-only `llm_router` would follow the CLAUDE.md "invariants into types" lesson. It would still need the `None` check because of SHOULD-FIX 1, and the current message is more helpful than a `TypeError`. Acceptable as is.

6. **`ModelInferenceError` is correct (`ModelError` → `MaximError`), but its type is lost at the boundary.** `run_turn` flattens it to `error=str(e)`, and `TurnResult` has no error type or code.
   Failure scenario: a caller cannot tell an LLM failure from a hippocampus failure without matching on the string. Consider an `error_type` field, which is CC3-safe if `TurnResult` has defaults.

7. **The `cli.py` extraction is clean.** It sits next to `_early_leader_boot`, the `LeaderRefusal` import moved with it, and the pin note matches. It now also stops the spawner on the `LeaderRefusal` (rc 2) path, which the old block did not do; that is an improvement worth one CHANGELOG word. Session A owns `cli.py`: coordinate the merge order on the server-mode hunk.

8. **docs/user/python-api.md and getting-started.md:** say that `create.router` needs the provider key (for example `ANTHROPIC_API_KEY`) or a configured local profile. This documents SHOULD-FIX 1 until that fix lands.
