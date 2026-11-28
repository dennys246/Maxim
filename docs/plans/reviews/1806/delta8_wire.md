# Device registry (#1806): delta-8 wire check (W7-1)

Reviewer: wire-integrity lens, final W7-1 check, read-only, 2026-11-25, v8 staged on 352ea398

**Verdict: CLOSED. No new fail-open defects. STOP RULE MET.** v8's "Doctor too" sentence sends doctor's `check_robot_reachable`, and the no-name `maxim device check`, through the one resolver as a non-legacy walk. It also deletes `_check_one_robot`'s private `robot_name or "reachy_mini"` + `.local` default and makes an addressless unpinned entry report `warn: no address` for that entry alone. That closes the silent-wrong "reachable" report. There are three fail-closed NITs, all in the covered daemon-name class. The legacy UX change (N8-1) should be stated in the note.

## 1. Re-check

| # | v8 status | Note |
|---|---|---|
| W7-1 (doctor keeps a private copy of chain step 4) | **CLOSED** | The address now comes from the one resolver, so doctor can no longer resolve `reachy-mini.local` for an unpinned `bob` that has no `host:` and no `config.robot_name`. The private default is deleted, and the outcome is `warn: no address` for that entry only, never "reachable". The red row patches `gethostbyname` to succeed and asserts "no address", so it would fail against today's code. Pinned entries now take `host:`/`hostname:`/cache, the same as a run, which also removes the pinned-address drift W7-1 noted. **Callers mapped (all on the one function):** `doctor/checks.py::run_all_checks` ("Robots" section), `api.py::diagnose` (via `run_all_checks`), and `doctor/cli.py::_worst_robot_result` (`--retry robot`; a `warn` is still picked up as the worst result, so retry still surfaces it). `_check_one_robot` has no other caller in `src/`, `scripts/` or `tests/`. N7-1 (the `repin` hint) and N7-2 (per-entry report) are folded |

## 2. New findings

| # | Finding | Fail-open/closed | Covered class? |
|---|---|---|---|
| N8-1 | **Legacy doctor/runtime disagreement; the note should state it.** Take a legacy single-robot user whose `robots.yaml` entry has no `host:` and no `config.robot_name`, in a registry with no pinned entry. Today doctor probes `reachy-mini.local` and can report `ok`. Under v8 it reports `warn: no address`. A legacy run still connects that user through the legacy chain: `resolve_connection_config` merges an empty `config`, then `selfy.py::Maxim.__init__` falls back to `MAXIM_ROBOT_NAME` or `"reachy_mini"`, and `ReachyMiniController` dials `<name>.local`. The same holds for a `MAXIM_ROBOT_NAME=alice` user, where doctor warns and the run dials `alice.local`. Doctor now **under-reports**, and never over-reports. That direction is intended: doctor cannot know which legacy source a future run will use. Today's `ok` was also only correct by coincidence for the primary-fallback case (see W7-1). **Suggested text:** one clause, "a legacy run still mDNS-dials the chain name, so a working legacy setup now sees this warn". The warn's fix hint should name `maxim device add` / `repin <name> <host>` (or `config.robot_name`) so the user can clear it | fail-closed (a warning, doctor never fails, no actuation) | yes (daemon name: W3-5/W4-1/W5-1/W6-1/W7-1) |
| N8-2 | **Two existing tests pin the deleted default.** In `tests/unit/test_doctor.py`, `test_resolution_failure_macos_fix_mentions_local_network` and `test_resolution_failure_linux_fix_omits_local_network` both use `_robots_cfg(host=None)`, which is `config: {}` with no `robot_name`. They patch `gethostbyname` to raise and assert `status == "fail"` plus the Local-Network / `host:` fix. Under v8 that entry is addressless, so the result becomes `warn: no address` and `gethostbyname` is never called. Both tests must be re-pointed to an entry with `config.robot_name` set, so the mDNS-failure path (and its macOS hint) stays covered. The no-address test row is the one the note adds. Name this in the PR's test list so the re-point is not read as weakening a guard | fail-closed (the tests would go red, loudly) | yes |
| N8-3 | NIT, wording. In §3.1's v8 paragraph, the "Doctor too" sentences were inserted **before** "Otherwise `maxim connect bob` would verify `reachy_mini.local` and move `primary:` to `bob` (…refusal hint…)". So "Otherwise" now reads as the consequence of the doctor rule, and the old refusal hint (`maxim device add`) follows the new one (`repin <name> <host>`), so the paragraph gives two different hints. Move the doctor sentences after the `maxim connect` sentence, and keep a single hint | fail-closed | yes |

## 3. Evidence

- `src/maxim/doctor/checks.py::_check_one_robot`: `host = robot.config.get("host")`. If that is falsy, then `mdns_name = str(robot.config.get("robot_name", "reachy_mini")).replace("_", "-") + ".local"` and `socket.gethostbyname(mdns_name)` is called. On an `OSError` the result is `fail` with the `host:` hint (plus Local Network on macOS). This is the default v8 deletes.
- `src/maxim/doctor/checks.py::check_robot_reachable`: `[_check_one_robot(info, robot) for robot in cfg.robots]`. It is called by `run_all_checks` ("Robots").
- `src/maxim/doctor/cli.py::_worst_robot_result` (`--retry robot`) returns the first result that is not `ok`/`info`, so it includes `warn`.
- `src/maxim/api.py::diagnose` calls `run_all_checks` and adds no robot path of its own.
- `rg _check_one_robot|check_robot_reachable` over `src/ scripts/ tests/` finds only the sites above, plus `tests/unit/test_doctor.py`'s class (`_run`).
- Legacy runtime chain: `hardware/config.py::resolve_connection_config` (merges the entry `config`, here empty), `embodied_runtime/selfy.py::Maxim.__init__` (`robot_name or os.getenv("MAXIM_ROBOT_NAME", "reachy_mini")`), `hardware/reachy/controller.py` (`self._robot_name.replace("_", "-") + ".local"`) and `cli_parser.py` (the `--robot-name` default is `MAXIM_ROBOT_NAME` or `"reachy_mini"`). v8 leaves all of these unchanged for legacy selections.
- No other `.local` default for a doctor-style probe exists in `src/maxim/`. The controller's copy is the runtime's own, and it is reached only through a legacy selection or a resolved `robot_name`.
- Note v8: `git diff --cached 352ea398 -- docs/plans/device_registry_1806_approach.md`, §3.1 paragraph "The chain's steps 2–4 … Doctor too".

## 4. Stop-rule outcome

**STOP RULE MET.** W7-1 is CLOSED. N8-1, N8-2 and N8-3 all fail closed, all fall in the already-covered daemon-name class, and need no owner decision. They are text and test-list folds with no further round owed.
