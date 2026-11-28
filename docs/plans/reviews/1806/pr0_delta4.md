# #1806 PR 0 (#1817): delta round 4, all three lenses

Scope: `git diff 3a7f386a..5b4c98db` in `.worktrees/pr0-1806-groundwork`: ef560cfe (`config/llm.json` on the
control-plane write deny, plus the scan nits) and 5b4c98db (the CWD candidates are dropped from
`_llm_config_candidates`, and both CPU fallbacks now go through `configured_profile_uses_llama_cpp()` /
`llm_config_file()`). I ran the probes in a `git archive` copy at 5b4c98db (`scratchpad/d4`), with a scratch HOME and an
absolute `PYTHONPATH` on its own line. `maxim.__file__` was confirmed to come from the copy. No network, robot or sim was used.

**Verdict: no DO-NOT-MERGE. Three SHOULD-FIX (S1 security residual mis-stated, S2 user docs now wrong, S3 the
fallback rewiring is unguarded) and two NITs. The round does not stop. S1 folded in `src/` owes a delta-scoped look.**

## Mechanical checks

- The 5 touched test files all pass in the copy (171 tests).
- `ruff check` is clean.
- `lint_no_silent_swallows`, `lint_function_length` and `lint_orphan_modules` are clean. The touch rule's diff-scoped
  half was skipped because the copy has no git. Both functions shrank, so it cannot fire.
- The delta removes two broad swallows (`except Exception: pass` in `configure_cpu_fallback_model`, and the
  debug-logged one in `_apply_llm_backend_fallback`). The new handler is narrow (`OSError, ValueError`) and logs.

Deletion probes (one mutation at a time):

| Mutation | Result |
|---|---|
| re-add the two CWD candidates | 1 fail (`test_an_llm_json_planted_in_the_launch_directory_is_not_read`) |
| drop `("config", "llm.json")` from `_CONTROL_PLANE_DATA_FILES` | 3 fail (`..._union_...`, `test_the_llm_config_is_refused[write_file/edit_file]`) |
| revert `cli_utils.py` + `agentic_runtime.py` to 3a7f386a (the old `<home_dir>/util/llm.json` readers) | **0 fail** (117 tests in the two 1806/10b files, 66 in every test file naming either fallback). See S3 |

## Executor

- **Deny location matches the reader.** `user_config()` is `data_home() / "config"`, and `data_home()` is
  `$MAXIM_DATA_HOME` or `~/.maxim`. The deny's data-file roots are that same data home plus the default `~/.maxim`, so
  `("config", "llm.json")` covers exactly the user-config candidate the loader reads. The child-`MAXIM_DATA_HOME` case
  is tested and was proven by the deletion probe.
- **No legitimate file-tool writer breaks.** `models/download.py::enable_llm_config` writes the first existing
  candidate, or `user_config()/llm.json`, directly through `atomic_write_json`, not through the agent file tools. No
  other `src/` writer of llm.json exists.
- **The helper's semantics match the old code**, with one exception (N1). The match covers: the explicit `profile`
  key only (no env, no `model` alias, no normalisation, as before); `profiles[profile].backend == "llama_cpp"`; a
  missing file is False; unreadable JSON is False. A non-dict entry used to raise AttributeError, which was swallowed
  into False; it is now False through `isinstance`.
- **The behaviour change from `<home_dir>/util/llm.json` to the loader's file is correct in every case I could build.**
  - `home_dir` comes from `args.home_dir` (default `"data"`, CWD-relative) or the runtime's `home_dir`.
  - Before, a run from a checkout root read the checkout's `data/util/llm.json`, even when `~/.maxim/config/llm.json`
    existed and was what the loader used. The fallback could then keep a llama.cpp profile the loader had not
    selected, or the reverse.
  - Now the fallback and the loader agree by construction.
  - A `--home-dir /x` with `/x/util/llm.json` was never read by the loader, so the old agreement there was accidental.
  - I found no scenario where the new file is the wrong one.

## Wire

- **`_llm_config_candidates` consumers**: `load_llm_config` (+ `_log_shadowed_llm_configs`), `list_llm_profiles`,
  `exec_agent_router._llm_json_identity`, `download.enable_llm_config`, and the new `llm_config_file`. All of them
  iterate "first existing wins", and none of them indexed a CWD entry by position. All are correct without the CWD
  entries. Doctor does not consume the list.
- **Other llm.json readers in `src/` + `scripts/`**: none. Every other hit is prose or reads through `load_llm_config`.
  `paths.resolve_config` is generic and not called for llm.json.
- **Same pattern, other files (outside the delta, informational):**
  - `utils/response_config.py::_get_config_candidates` still reads `./data/util/` and `./` for `key_responses.json`
    and `phrase_responses.json`.
  - `inference/transcribe_audio.py` still reads `./whisper.json`.
  - An agent in the launch directory can plant these. The impact is lower: a phrase cannot enter a code-executing mode
    (#828), and whisper.json only tunes VAD. Still, it is the class the owner decision named. See N3.
- **Docs**: see S2.

## Architecture / tests

- **Module placement is right.** `models/language/config.py` owns llm.json discovery. The router-only rule covers
  backend imports, not config reads, so both fallbacks importing a config helper is fine.
- **The planted-CWD test's `os.path.isfile` patch is sound.** It admits only paths under `work`, which hides the
  repo-root candidates (a stated residual) and the empty data home. Re-adding the CWD candidates makes it fail (probe
  above), so it does not mask the mechanism it guards. It does not exercise the fallbacks, though (S3).
- **Patching the reader in `test_offline_call_sites_10b.py` is acceptable.** It isolates those pin-semantics tests
  from the checkout's tracked `data/util/llm.json`, which those tests were not testing.
- **Doc claims vs code**: safety.md, the changelog fragment, the runtime-tools brief and the `tools/base.py` docstring
  all match the code, except the repo-root residual description (S1).

## Findings

### S1 SHOULD-FIX (security): the "repo-root" fallbacks are not a source checkout's files on a wheel install, and can sit under the launch directory

`repo_root` is four `..` up from `models/language/config.py`:

- checkout `/x/Maxim/src/maxim/models/language/config.py` gives `/x/Maxim` (as intended);
- wheel `/p/.venv/lib/python3.12/site-packages/maxim/models/language/config.py` gives `/p/.venv/lib/python3.12`
  (computed; verified with `os.path`).

So an installed Maxim with no `~/.maxim/config/llm.json` reads `<venv>/lib/python3.12/data/util/llm.json` and
`<venv>/lib/python3.12/llm.json`.

**Failure scenario:**

1. A user has a project directory with an in-project `.venv` (a common layout) and runs `maxim` from it.
2. The data-home llm.json does not exist yet; it is only created on download/enable.
3. An agent whose file tools reach the project writes `.venv/lib/python3.12/llm.json` with `cloud_enabled: true` and
   raised `routing` caps.
4. The next run loads it.

That file is under the launch directory and outside the deny. So the new changelog claim "Maxim no longer reads an
`llm.json` from the directory it was started from" is false here. The safety.md, `tools/base.py` and brief wording
("a source checkout's own `data/util/llm.json`") understates the residual.

**Fix (small, needs no new decision, implements the stated intent "a source checkout's tracked developer file"):** add
the repo-root pair only when `repo_root` is actually a checkout, for example when
`os.path.isfile(os.path.join(repo_root, "pyproject.toml"))` and the package dir is `repo_root/src/maxim`. Add a test
that a site-packages-shaped layout yields no repo-root candidate. Alternatively, correct all four residual statements
to name the wheel case. That is weaker, and still leaves the changelog sentence needing a qualifier.

### S2 SHOULD-FIX: user docs still tell users to put llm.json in the working directory, and a moved budget cap is now silently ignored

- `docs/user/llm-setup.md:270`: "Override limits via an `llm.json` file in the working directory (or pointed at by
  `MAXIM_LLM_CONFIG`)", followed by a `routing` example including `"fallback_on_budget_exceeded": "reject"`.
- `docs/user/llm-setup.md:294`: "A legacy `llm.json` file (found via `MAXIM_LLM_CONFIG` env var or in the working
  directory) is still supported".
- `ARCHITECTURE.md` §Config File Priority (lines ~266–271): still lists `./data/util/llm.json` and `./llm.json` as
  entries 3–4 of the "one shared list". Lines 213/280/294 describe `data/util/llm.json` as the install-created config.
- `docs/user/configuration.md:357`: says `~/.maxim/config/llm.json` "is no longer read by the current runtime". The
  code reads it first after the env var, and the new safety.md text tells users to keep their settings there, so the
  two pages now contradict each other. This line predates the delta, but the delta's guidance makes the
  contradiction load-bearing.

**Failure scenario:**

1. A user followed llm-setup.md and keeps `./llm.json` with `max_session_cost: 1.00` and
   `fallback_on_budget_exceeded: "reject"`.
2. After this PR the file is silently not read, and the defaults apply ($5 session ceiling, graceful fallback).
3. Their stricter cap is gone with no message.

The changelog fragment tells such users to move the file, but nothing at runtime does.

**Fix:**

- Update the three docs.
- Add a one-time WARNING in `load_llm_config` when `./llm.json` or `./data/util/llm.json` exists and is not the loaded
  file. Name it as "no longer read, move to `~/.maxim/config/llm.json`", and do not read it. This reuses the
  `_log_shadowed_llm_configs` once-per-process pattern and is a few lines.

### S3 SHOULD-FIX (tests): nothing pins that either CPU fallback goes through the loader's file

Reverting `cli_utils.configure_cpu_fallback_model` and `AgenticRuntimeMixin._apply_llm_backend_fallback` to their
3a7f386a bodies leaves every test green. The bodies in question are the private `<home_dir>/util/llm.json` readers,
the ones that with the default `home_dir="data"` read exactly the planted `./data/util/llm.json`.

- `test_the_cpu_fallbacks_read_the_loaders_llm_json` calls only `C.configured_profile_uses_llama_cpp`.
- The planted-CWD test's docstring says "neither the loader nor either CPU fallback reads it", but it calls neither
  fallback.
- The 10b tests stub the helper to False, which the old code also returns for an absent `tmp_path/util/llm.json`.

This is the "ships with a CALLER" rule: the helper's wiring into both fallbacks is unguarded.

**Fix:** in the planted test, call `cli_utils.configure_cpu_fallback_model(logger, home_dir=str(work / "data"))` and
`_apply_llm_backend_fallback` on a `SimpleNamespace(home_dir=str(work / "data"), ...)`, with `is_gpu_available`
patched False and the env scrubbed. Then assert that the CPU profile pin was applied, i.e. the planted
`llama_cpp` profile did not short-circuit. With the old bodies this fails.

### N1 NIT: a non-hashable `profile` value now raises out of the CPU fallback

With `{"profile": ["x"], ...}` in the loaded llm.json, `configured_profile_uses_llama_cpp()` raises
`TypeError: unhashable type: 'list'` (probed). The old code swallowed it into False. `load_llm_config` itself
tolerates the value. The result is a startup crash on a GPU-less box with a malformed hand edit. It is loud, not
silent. Fix: `name = raw.get("profile"); entry = profiles.get(name) if isinstance(name, str) else None`.

### N2 NIT: vacuous first assertion

In `test_the_cpu_fallbacks_read_the_loaders_llm_json`, the first assertion
`configured_profile_uses_llama_cpp() is False or llm_config_file() != str(cfg)` always holds while `cfg` does not
exist, so it checks nothing. It also lets the helper read whatever data home the conftest HOME holds. Either drop it,
or set `MAXIM_DATA_HOME` to an empty tmp dir and assert `llm_config_file() is None`.

### N3 NIT (out-of-delta neighbour, for the owner)

`response_config` (key/phrase responses) and `transcribe_audio` (whisper.json) keep CWD candidates, as listed under
Wire. Under "Filing is not fixing" this is another site of the pattern. Either fold it (each is a deletion of 1–2
`append` lines), or state it as a residual with `Deferred because: owner-decision`, since the owner decision scoped
itself to llm.json.
