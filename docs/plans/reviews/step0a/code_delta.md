# Step 0a: delta-scoped review of the review fold (Executor + Wire integrity)

Reviewer: a different reader (not one of the three first-round reviewers). Scope: `git diff 7bfbee71 24f185f2` (the
fold), plus how #1537 (merged in at `1f7a4c94`) interacts with it. Read-only, apart from this file. Narrow tests were
run in the worktree and a deletion probe in a scratch copy.

## Verdict

One SHOULD-FIX (D1). It is in W1's class, but the fold *claims* to close that class and does not, so the stop rule
does not apply: **a further delta round is owed after D1 is folded** (or after the owner explicitly narrows the claim
instead).

Checked and clean: the re-entry flag (probe 2), the conftest early merge (probe 3), the #1537 interaction (probe 4),
and the seam's own guard test, which fails when the seam is deleted (see "Verified").

## Findings

### D1: SHOULD-FIX, fail-open (degrades with a WARNING; not silent). "Every run stops" does not hold: three run paths swallow the seam's error, or never reach it

The fold puts the one fail-fast point in `runtime/lane_backends.py::build_primary_router`. Its docstring
(`_require_valid_user_profiles`), the changelog fragment ("Every run still stops on it up front, when it builds its
model"), the approach note's Status block and the doctor fix hint ("every `maxim` run stops with this error") all
claim that every run path stops there. Two things break that claim. Some run paths never call
`build_primary_router`, and some call it inside an `except Exception` that turns the `ConfigurationError` into "run
without the LLM":

| Path | Reaches the seam? | What happens with a malformed `profiles.yml` |
|---|---|---|
| `cli.py::_main_impl`, `mode == "agentic"` (`maxim --mode agentic`, and `maxim --sim <scenario.yaml>` for a non-campaign, non-curriculum YAML, which forces `args.mode = "agentic"`) with no `--llm` | **No.** The router comes from `agents/exec_agent.py::ExecAgent._ensure_router` (the docstring of `cli.py::_ensure_agentic_model` says so) | `_ensure_router` catches the error: `[WARN] Failed to load language model config: ... is not valid YAML`, then `None`. `llm_router is None`, so no `LLMWorker`, and the agentic loop runs with no LLM. **Reproduced** (fresh process, sandbox HOME, malformed file: `ExecAgent(AgentBus(), None)._ensure_router()` returns `None` after that WARN). |
| `embodied_runtime/agentic_runtime.py::_start_agentic_runtime` (the live/robot runtimes) | Yes, inside `try` | `except Exception` → `utils/logging.py::warn_unless_refusal`, which re-raises only `LeaderRefusal`. Logs "Failed to create LLM worker", `llm_worker = None`, carries on with fallback responses. |
| `simulation/research_orchestrator.py::_build_research_llm` | Yes, inside `try` | `except Exception` → WARNING "Writer/Reviewer will use heuristics", carries on. |
| `cli.py::_main_impl` `--foundry` / auto-curate (`load_llm_config` under `except Exception`) | No | Foundry prints a WARNING and falls back to templates. Curate logs at **DEBUG** and curates without the LLM. That one is effectively silent, but only when `--llm` is unset: with `--llm`, `cli_utils.normalize_args` raises first, unswallowed. |

Paths that do stop, by reading the code: `simulation/orchestrator.py::start_simulation_mode` (its primary
`build_primary_router` call is not in a `try`, so goal-string `--sim`, curricula and DM all stop), `api.run`,
`create.router`, and the console talk path (`console/handle.py`, which propagates). `cli.py::_start_server_llm`
catches the error but reports it as NOT READY: a solo server exits 1, and one behind a LeaderProxy stays up not ready
(#1338 design). That is fail-closed and acceptable. `--list-models`, `--delete-model` and `normalize_args` (with
`--llm`) call accessors with nothing catching them, so they raise. The orchestrator's AUT router swallow cannot be
reached: the primary build raises first.

Severity: on main, all four rows above refused at import. Now they run degraded. The wrong decision is "no model",
not W1's "silently smaller model", and the first three rows say why at WARNING. That is why this is SHOULD-FIX, not
BLOCKER. It is still the owner's decision, "a malformed profiles.yml stops every run", not being carried out, and the
shipped text says it is.

Proper fix (owner's call; do not special-case each handler):
- **(a) Add the CLI entry as the second caller of the same seam.** Call `_require_valid_user_profiles()` once in
  `cli.py::_main_impl`, after the early dispatches that read no profile (doctor, `model`, `--clear-*`), before the
  mode dispatch. That covers agentic, live/robot (the embodied runtime), foundry/curate and research launched from
  the CLI. Library entries keep `build_primary_router`. This is what the first-round architecture review recommended
  ("`cli.py::_main_impl` after the `model`, `doctor` and similar early subcommand dispatch"). It is still one seam
  function, with two callers.
- (b) Alternatively, raise a typed subclass (e.g. `UserProfilesError(ConfigurationError)`) and have
  `warn_unless_refusal` re-raise it like `LeaderRefusal`. That covers `agentic_runtime`, but not `ExecAgent` or
  research, which use their own handlers. So (b) alone is not enough. It is a good complement to (a) for library use
  of the embodied runtime.
- Pin it with a fresh-process test: malformed file + `ExecAgent`-path entry (or `_main_impl` with `--mode agentic`)
  → non-zero / `ConfigurationError`, not a `None` router.
- Or, if the owner prefers to narrow the claim, change the docstring, the changelog and the doctor hint to name the
  paths that stop. The strict option (a) is the default recommendation.

### D2: NIT, fail-closed. The re-entry guard raises `RuntimeError`, which the swallowing handlers would also eat

`models/language/config.py::_ensure_user_profiles`. Today nothing can re-enter: `profile_loader.apply_user_profiles`
calls no accessor (checked: its only `maxim` imports are `exceptions` and `utils.paths`). If a future edit made it
re-enter, every access would raise `RuntimeError`, and `finally` resets the flag, so it is never cached. Through the
same `except Exception` sites as D1, that becomes "no LLM" plus a WARNING. Consider `ConfigurationError` (or D1(b)'s
subclass), so it rides whatever re-raise D1 adds. Optional.

### D3: NIT. `maxim.models.__all__` lists `LLM_MODELS`, so `from maxim.models import *` runs the merge

`models/__init__.py::__getattr__`. A star-import on a malformed file raises `ConfigurationError` at import. That is
consistent with "raises on lookup", but it is an import-time profile read. Nothing in `src/` star-imports it
(intentional surface). Mention it in the docstring, or leave it.

## Probes asked for

1. **First profile read on every run path.** No, see D1. Paths that read a profile, behind a swallowing handler,
   *before* `build_primary_router`: `api.py::_resolve_model` (`normalize_llm_profile` under `except Exception` →
   DEBUG). Harmless: it only skips persisting the model choice, and `build_primary_router` raises right after.
   `cli._main_impl` foundry/curate (D1 row 4). `_ensure_agentic_model` runs only after `_ensure_router` returned a
   router, so with a malformed file it is never reached. `--list-models` raises, unswallowed. The lane env
   projection (`apply_lane_env_overrides` and the rest) runs *after* the seam inside `build_primary_router`.
2. **Re-entry flag and threads.** Sound. `_user_profiles_merging` is only true while the merging thread holds the
   RLock, and it is only read under that lock, so another thread blocks on the lock and never sees `True`. Only a
   same-thread nested call can, which is the case it exists for. A failed merge resets it in `finally` and leaves
   `_user_profiles_loaded` False, so the next access retries. The unlocked fast path reads `loaded` only after the
   tables were mutated (GIL ordering). `download.py` does no module-level work that could take the profiles lock
   during import, so the deferred `from maxim.models.download import LLM_MODELS` inside the lock cannot deadlock.
3. **conftest early ensure.** No test loses what it tests. Every in-process test that writes a `profiles.yml` either
   passes an explicit `path=` with throwaway tables (`test_user_profiles.py`, `test_model_cli.py`, and
   `doctor/checks.py::check_user_profiles` itself merges into `{}`), or runs in a fresh subprocess
   (`test_profile_readers_v9.py::_run`, `test_advisor_gate.py`, the fresh-import sweep, the new `build_executor`
   test). `test_unreadable_config_dir_1403.py` uses explicit paths. No in-process test asserts `models.language` /
   `models.download` is absent from `sys.modules`. The conftest ensure has the same effect the old import-time merge
   had at collection, so behaviour matches main.
4. **#1537 interaction.** Clean. `models/language/cancellation.py` registers its hook pair at import, and
   `utils/cancellation.py::register_stop_hook` runs `on_request` at once if a stop is already pending. So when lazy
   inits import the backend module later, it is born stopped, and a backend that was never imported has no
   in-flight work to abort. The importers (`anthropic_backend`, `openai_backend`, `maxim_peer_backend`, `cli`,
   `orchestrator`) import the submodule directly, not through a lazy package attribute. `from maxim.utils.cancellation
   import ...` is a submodule import and is unaffected by `maxim.utils`' PEP 562 table. #1537's
   `console/handle.py` still builds through `build_primary_router` and propagates.
5. **Other.** `_lane_profile_is_cloud`'s `logger is not None` check, the verbatim move of `_capabilities_or_detected`,
   and the 245 → 239 pin with its history line are correct. The first-round W-items are otherwise folded as
   described.

## Verified

- Deletion probe (scratch copy, `_require_valid_user_profiles()` call removed from `build_primary_router`):
  `test_a_malformed_profiles_file_stops_the_run_before_tier_detection` **fails** ("tier detection ran before the
  malformed profiles.yml was refused"). The guard guards.
- Narrow runs in the worktree: `test_profile_readers_v9.py`, the profile tests in `test_doctor.py`,
  `test_unreadable_config_dir_1403.py` and `test_conftest_isolation_order.py`: 32 passed, 1 xfailed (#1479).
  `test_build_executor.py -k without_a_pain_bridge`: 1 passed.
- D1 row 1 reproduced: fresh process, `HOME`/`XDG_CONFIG_HOME` = a sandbox with
  `profiles:\n  bad:\n    nonsense: [\n`. `ExecAgent(AgentBus(), None)._ensure_router()` → WARN + `None`.
