# #1806 PR 0 (#1817): selfy to hardware/config move, delta review

Scope: the STAGED diff in `.worktrees/pr0-1806-groundwork` on top of HEAD 1f743145 (`git diff --cached`; 4 files:
`selfy.py`, `hardware/config.py`, `test_1806_pr0_robots_config.py`, `coverage_floors.json`). Lenses: executor,
architecture, wire integrity (docs/CODE_REVIEW.md). Probes ran in a `git archive HEAD` copy with the patch applied,
under a scratch HOME, with an absolute `PYTHONPATH` to the copy's `src`.

**Verdict: no SHOULD-FIX. 2 NITs. No new finding class.**

## (1) Behaviour against HEAD's selfy versions

I compared the code side by side with `git show HEAD:src/maxim/embodied_runtime/selfy.py`.

- `robot_connect_timeout` is a verbatim move of `_robot_connect_timeout`. The only change is that `math` and
  `InvalidConfigError` are now imported at module level. The refusal text, the blank-means-default handling and the
  finite/positive check are all the same.
- The simulation path is unchanged. It still logs "Connecting to Reachy Mini '%s'..." first, then returns
  `(robot_id, "simulated", {"video_resolution": (640, 480), "simulate_delays": False})` with
  `_resolved_robot_config = None`.
- The live path is unchanged. The `robot_type` default is still `"reachy_mini"`, and the steps run in the same
  order: the "Connecting" info log, `load_robots_config()`, `resolve_connection_config` with the same defaults,
  `resolve_robot_entry`, the F1 warning (same text and arguments), and the type-override info log (same text,
  em dash kept). The logger is the same one too: `self.log` is passed in as `log=`.
- `RobotsConfigError` still propagates out of `load_robots_config` before any assignment, so
  `_resolved_robot_config` stays `None`, as at HEAD.
- One sequencing difference: HEAD set `self._resolved_robot_config = entry` before the warning and override logs, and
  the new code sets it after the helper returns. Nothing between those two points can raise, so this is not
  observable.
- Patch seams still work. HEAD did a call-time `from maxim.hardware.config import load_robots_config`, and the new
  helper reads its own module global. Both pick up `monkeypatch.setattr(maxim.hardware.config,
  "load_robots_config", ...)`, which is the pattern `tests/unit/test_multi_robot.py::TestResolveBodyWiring._stub` and
  `test_the_connect_and_the_body_wiring_read_one_entry` use.

## (2) Layering and import cycles

- `hardware/config.py` adds only `math`, `os` and `maxim.exceptions.InvalidConfigError`. It already imported
  `maxim.exceptions`. Nothing it imports reaches `embodied_runtime`, `tools`, `runtime`, `agents` or `planning`, so
  `utils/audit.py::LAYER_RULES` (ARCHITECTURE.md, hardware row) still holds.
- selfy's module-level import of `maxim.hardware.config` (`resolve_daemon_name`) already existed at HEAD, so adding
  `robot_connect_timeout` to it adds no new dependency between modules. In the copy, `import maxim.hardware.config`
  on its own succeeds.

## (3) Callers and patch sites outside the diff

I grepped `src/ tests/ scripts/ docs/ changelog.d/` for the old and new names.

- `_robot_connect_timeout` has zero remaining references.
- `Maxim._resolve_robot_connection` keeps its name and signature. Its callers are `selfy.py` `__init__`, the comment
  and docstring in `agentic_runtime.py`, and the stubs in `test_1806_pr0_robots_config.py` and
  `test_multi_robot.py`. All still match.
- No test patches `maxim.embodied_runtime.selfy.load_robots_config`, `resolve_connection_config`,
  `resolve_robot_entry`, `os` or `math`.
- `MAXIM_ROBOT_TIMEOUT` has no other `src/` reader (`cli.py:1001` is a docstring), and the conftest scrub at
  `tests/conftest.py:685` still covers it.

## (4) Do tests exercise the moved functions?

Yes. `coverage run` over `test_1806_pr0_robots_config.py` and `test_multi_robot.py` (59 passed) reports
`hardware/config.py` lines 323–381 fully covered. The missing lines are 42-49, 217, 312, 408-439 and 460-482, all
outside the moved code. So the diff-coverage rule (≥80% of added lines) is met for the new lines.

The simulation branch in the wrapper is not covered, which was also true at HEAD; it sits in the excluded file.

An environment artifact, not a finding: `pytest --cov=maxim.hardware.config` in the local `.venv` fails the four
`_selfy_stub` tests with a numpy `_NoValueType` TypeError inside a scipy import chain (`selfy` → `media_loop` →
`utils/audio` → scipy). The same tests pass under plain pytest and under `coverage run`. CI is the suite of record
(M64).

## (5) Docs and briefs naming the old location

- `changelog.d/1806-pr0-robot-config-groundwork.fixed.md` describes behaviour only, so nothing is stale.
- `docs/plans/deferred/live_audio_orient_wiring.md` cites old selfy line numbers from before this PR, unrelated.

**NIT-1.** `docs/agents/embodiment.md` §6, the `MAXIM_ROBOT_TIMEOUT` row, still names the reading site as
`embodied_runtime/selfy.py::Maxim.__init__`. The variable is now read in `hardware/config.py::robot_connect_timeout`,
and selfy no longer contains the string at all. A reader grepping from the brief lands in the wrong file. No lint
checks these rows. Fix: change the row's owner to `hardware/config.py::robot_connect_timeout`, called from
`Maxim.__init__`. Optionally say `InvalidConfigError` (a `ConfigurationError` subclass) to match the code.

## (6) Exclusion allowance (offline)

- `lint_coverage.statement_count` on selfy.py gives 519 at HEAD, 500 at origin/main and **496 with the patch**.
- The allowance is 500: `Exclusions.allowance` takes the last ledger entry, the `#1315` one, over the grandfathered
  514.
- `exclusion_head_rules(...)` returns `[]`, and the pragma count is 1 against an allowance of 1.
- The diff rule (no rise against the merge-base) passes: 496 < 500.
- The full `lint_coverage.py` fails closed offline because there is no `coverage.json`, as expected.

## Other checks

- `ruff check` and `ruff format --check` are clean on the three .py files.
- `scripts/lint_function_length.py`, run read-only in the worktree, prints `clean`. `Maxim.__init__` is pinned at 243,
  and against origin/main the PR shrinks it by a net 61 lines, so the touch rule holds.
- `hardware/config.py` is not on the `run_mypy_public.sh` list.
- The `maxim/hardware` package gains about 30 fully covered statements, roughly +0.1 pt against its 87.4 floor. That
  is well inside the 1.0 pt "rose, floor not raised" band. Confirm on the CI run anyway.
- I cannot check the `maxim/utils` re-pin (72.9 / 1888, from run 45901613161) offline. Confirm it matches that run's
  printed suggestion.

**NIT-2.** `Maxim._resolve_robot_connection` still uses a function-local
`from maxim.hardware.config import resolve_robot_connection`, although selfy now imports from that module at module
level. Folding it into the line-41 import is cosmetic and removes a statement from the excluded file.

## New class?

No. Both NITs are stale-pointer and style items in classes the round already covers.
