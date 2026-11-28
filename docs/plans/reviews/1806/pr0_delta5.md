# #1806 PR 0 — delta review round 5 (round-4 fold, `git diff 5b4c98db f74a3b99`)

Lenses: Executor, Architecture, Wire integrity (one reader, all three). Probes ran in a `git archive` copy under the
session scratchpad with a scratch `HOME` and `MAXIM_DATA_HOME`. The real `~/.maxim` was not touched.

## SHOULD-FIX

### S1. The ignored-CWD warning fires in a source checkout run from the repo root (the case it must skip)
`src/maxim/models/language/config.py::_warn_ignored_cwd_llm_json` compares each CWD path only with the **winner**.
The repo tracks `data/util/llm.json`. So in a checkout, with cwd = repo root and a `~/.maxim/config/llm.json` present
(the normal developer setup), `./data/util/llm.json` is the repo-root candidate that the user file shadows, not a
launch-directory file. It still gets the warning:

```
INFO  llm.json loaded from <dh>/config/llm.json (shadowing: <repo>/data/util/llm.json)
WARNING <repo>/data/util/llm.json is NOT read (since #1806 Maxim reads no llm.json from the launch directory);
        using <dh>/config/llm.json. Move it to <dh>/config/llm.json to use it.
```

I reproduced this in the copy. The message is wrong on three counts:
- The file IS a candidate.
- #1806 is not why it is unread. It is shadowed, and it was shadowed before #1806 too.
- It tells the user to move the tracked repo file over their own existing `llm.json`, which would overwrite their config.

The same false premise applies to any CWD file when a data-home file exists. The old order was user > cwd > repo, so a
CWD file was never read in that case, and the warning's own rationale ("caps would drop to the defaults without a word")
does not hold.

The no-user-file case is correct: winner = repo `data/util/llm.json`, realpath equal, no warning.

Fix (small): skip any CWD path whose realpath is in `{realpath(c) for c in _llm_config_candidates()}`. Optionally warn
only when the old order would have read it, i.e. `winner` is None or ranks below the user config. Add a test with
cwd = the checkout root, a data-home `llm.json`, and the tracked repo file, asserting no "is NOT read" record. The
existing test monkeypatches `_source_checkout_root` to None, so it cannot see this.

### S2. `load_llm_config` now raises `FileNotFoundError` when the working directory has been deleted
`config.py::_warn_ignored_cwd_llm_json` calls `os.getcwd()` without a guard, and every `load_llm_config()` now reaches
it. In a process whose cwd was removed, the call raises:
- examples: a `--sandbox tmpdir` cleanup, a test's tmp dir, a shell `cd`'d into a dir that was then `rm -rf`'d
- traceback ends at `cwd = os.getcwd()` → `FileNotFoundError`

At 5b4c98db the same probe returns normally ("old OK"), so this is a regression introduced by the round-4 fold. Several
callers wrap the call in a broad `except Exception` and degrade silently. For example,
`agents/exec_agent_router.py::configured_override` treats it as "no override". Others, like the router build path,
crash. A best-effort diagnostic must not be able to fail the config load.

Fix: `try: cwd = os.getcwd() except OSError: return`. Use a narrow except with a debug log so it does not count as a
new silent swallow. Add a regression test that runs `load_llm_config()` from a removed cwd.

## NIT

### N1. `exec_agent_router.py::_llm_json_identity` still walks `_llm_config_candidates()` itself
This is a second copy of `llm_config_file()`'s "first existing candidate" loop. The two are consistent today, because
both get the new checkout gating from the shared candidate list. But the fold's stated goal is one resolver, and
`download.py::enable_llm_config` was moved onto `llm_config_file()` for that reason. Do the same here
(`path = llm_config_file()`, then stat). This is outside the diff and is not a defect.

### N2. Guarding the warning against the realpath of a re-resolved winner
`load_llm_config` passes `llm_config_file()`, which resolves the winner a second time instead of reusing the `path` its
own loop chose. The result is the same, since both pick the first existing candidate. Passing the loop's `path` removes
a TOCTOU-shaped duplicate and one extra stat pass. Cosmetic.

## Probed, no finding
- **Editable install / `PYTHONPATH=src`:** `__file__` is under `<root>/src/maxim/...` with `pyproject.toml`, so it is
  treated as a checkout. Intended.
- **Symlinked checkout, or site-packages `maxim` symlinked into a checkout:** `realpath` resolves to the real
  `src/maxim`, so it is a checkout. Intended. The test's expected value uses `.resolve()`, which is consistent.
- **Zipapp / zipimport:** `pyproject.toml` is not a real file beside the zip path, so the result is None. Safe.
- **Wheel whose site-packages parent is named `src`** (e.g. `pip install --target <proj>/src` with a
  `<proj>/pyproject.toml`): treated as a checkout, so `<proj>/llm.json` is read. I confirmed this with a probe.
  Not counted as a finding: if Maxim's code is installed inside a directory the agent can write, the agent can edit
  `config.py` itself. That is the same stated residual class as the source checkout (`docs/user/safety.md`), and a
  pyproject-name check would be just as plantable.
- **Global `_ignored_cwd_warning_emitted`:** the only test that asserts on it resets it via `monkeypatch.setattr`.
  Other tests can only flip it to True, which suppresses a WARNING and asserts nothing. This matches the existing
  `_shadow_log_emitted` pattern. Isolation-safe.
- **`configured_profile_uses_llama_cpp` hardening:** a non-dict raw, a non-dict profiles value and a non-string profile
  all return False. Covered by the test loop.
- **Wire, other llm.json readers:** the only consumers of the candidate list outside `config.py` are
  `download.py::enable_llm_config` (now `llm_config_file()`) and `exec_agent_router.py::_llm_json_identity` (N1).
  `utils/paths.py::resolve_config` reads user config, then bundled, with no CWD. `tools/base.py` deny entry
  `("config", "llm.json")` is unchanged. No reader in `scripts/` resolves its own path list. The response_config and
  whisper CWD reads are out of scope (#1825).
- **Docs:** `ARCHITECTURE.md`, `llm-setup.md` and `configuration.md` describe the new order correctly, except that
  their "a warning names it" wording will over-promise S1's false positive until S1 is fixed.

## Stop-rule verdict
Both SHOULD-FIXes are in the new warning helper, which this round introduced. S1 is a false positive and S2 is a
crash; both are fail-loud, neither is a silent-wrong config read, and the security property holds: no CWD `llm.json`
is read and the wheel gating is correct. They narrow (round 4 found read-path holes, round 5 finds only diagnostic-path
defects in the fold's own helper), and both are small known-cause fixes with obvious tests. After folding S1 and S2
with their tests, no further full round is owed. A delta-scoped glance at that fold is enough under the "any `src/`
fold after the last round" rule. N1 and N2 are optional.

## Round 5 fold check

Scope: exactly `git diff 15620981 144bc4f8` (3 files: `models/language/config.py`, `agents/exec_agent_router.py`,
`tests/unit/test_1806_pr0_writedeny_control_plane.py`). Lenses: Executor, Architecture, Wire integrity (one reader).
Probes ran in `git archive` copies (new = 144bc4f8, old = 15620981) and a local clone under the session scratchpad,
with a scratch `HOME` and `MAXIM_DATA_HOME`. The real `~/.maxim` and the worktree were not touched.

### Folds verified
- **S1 (fixed).** `_warn_ignored_cwd_llm_json` now skips any CWD path whose realpath is in the realpaths of the
  loader's own `candidates` list. The checkout-root case gets no warning. A CWD file that IS the env or user-config
  candidate (e.g. cwd = `<dh>/config`) is also correctly skipped.
- **S2 (fixed).** `os.getcwd()` is guarded with a narrow `except OSError` and a debug log, and it returns before any
  `realpath` call that could itself need the cwd.
- **N1 (taken).** `_llm_json_identity` now calls `llm_config_file()`. The cache key is the same as before: the
  path string of the first existing candidate, with no realpath, then `(st_mtime_ns, st_size)`. `configured_override`
  is unchanged, so its caching behaviour holds. Both existing identity tests in `test_agent_router_slots_1228.py`
  (`test_no_llm_json_has_no_identity`, `test_an_llm_json_that_vanishes_before_stat_has_no_identity`) still pass. They
  patch `C._llm_config_candidates`, which `llm_config_file` looks up through module globals, and the global
  `os.path.isfile`, which is the same `os` module object. Those tests now also cover the new path.
- **N2 (taken).** `load_llm_config` passes its loop's `winner` and the same `candidates` list, so there is no second
  resolution.
- **The tests pin the folds.** With the new tests run against the pre-fold code, both fail for the right reason:
  - the checkout test fails on the `"is NOT read"` WARNING record;
  - the deleted-cwd test fails with `FileNotFoundError` out of `load_llm_config`.

  On 144bc4f8, the PR 0, #1228 router and model-path-resolution test files all pass (102 tests).
- **Swallow lint.** `scripts/lint_no_silent_swallows.py` is clean, including its diff-scoped per-function check. I ran
  that check in a clone at 144bc4f8 with `origin/main` set to 15620981. The totals (380 / 123 / 659) are identical to
  the old copy's. The router's `except OSError: return None` is the same handler in the same function, so it counts
  the same. The new `except OSError` in config.py logs, so it is not a swallow.
- Ruff check and format are clean on the three files. `lint_function_length` is clean. Neither src file is in
  `run_mypy_public.sh`. A changelog fragment for #1806 PR 0 already exists.
- **Wire integrity.** The consumers of the candidate list are unchanged except N1. `download.py::enable_llm_config`
  and `configured_profile_uses_llama_cpp` were already on `llm_config_file()`. `list_llm_profiles` still walks the
  candidates itself. That loop is the same first-existing-candidate rule and is outside this diff, so it is not a
  finding.

### SHOULD-FIX
None.

### NIT
- **N1'. S1's optional half is not taken, so the warning can still give self-contradictory, harmful advice.** Probe:
  - Setup: a non-checkout (checkout gating patched off), cwd `proj/` containing `llm.json`, and an existing
    `<dh>/config/llm.json`.
  - Result: `proj/llm.json is NOT read (since #1806 ...); using <dh>/config/llm.json. Move it to
    <dh>/config/llm.json to use it.`
  - Why it is wrong: under the old order (env > user > cwd > repo), this file was never read in this setup, so "since
    #1806" is false. A user who follows the advice overwrites the config that is in use.
  - Fix (about 2 lines): warn only when the old order would have read the CWD file, i.e. when neither the
    `MAXIM_LLM_CONFIG` file nor the user-config file exists. Equivalently, `winner is None` or winner is a repo-root
    candidate. Optionally, extend the checkout test or add one more case for it.

  This is the same class as round 5 S1, which raised it as optional. It is not a new class.

### Stop-rule verdict
The folds fix what round 5 found and introduce nothing new. The two new tests fail on the pre-fold code. The router
change preserves `configured_override`'s caching and its tests. The swallow lint is clean and unchanged. There is no
SHOULD-FIX and no new class, so **no further round is owed**. N1' is optional. Folding it would be a small `src/` touch,
which would owe only a delta glance under the "any `src/` fold after the last round" rule.
