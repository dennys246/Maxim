# #1806 device registry: wire-integrity review, delta round 2

Reviewer: wire-integrity lens (docs/CODE_REVIEW.md), delta round 2, read-only, 2026-11-25, v3 @ 8540cf19 (code at origin/main b7380ee7, incl. `scripts/`, `tests/`, `docs/`). Fresh reader. One stdlib-only probe ran under a fresh `HOME=$(mktemp -d)` with `python3 -I`. I contacted no robot or LAN host and ran no sim.

**Verdict: SHOULD-FIX, no DO-NOT-BUILD.** v3 closes N1, N3, N7, F1b and F11. N4 and N5 are PARTIAL. There are four new fail-open findings (W2-1 to W2-4) and three fail-closed NITs. All of them are in classes already covered: the writer gate (F7/N4), selection precedence and re-exec (F2/N5), and one resolver (N3). Each one needs a text fix to the note, and none changes an owner decision. The stop rule is **not** met, because W2-1 to W2-4 are fail-open.

## 1. Re-check of delta-round-1 findings

| # | v3 status | Note |
|---|---|---|
| N1 | CLOSED | §4.1: the "data_home() unresolvable" branch and a relative `MAXIM_DATA_HOME` both raise `RobotsConfigError`. The check is scoped to robots.yaml resolution and the writer. `data_home()` does not expand `~`, so `MAXIM_DATA_HOME=~/x` counts as relative and raises (fail-closed) |
| N3 | CLOSED | D2″ makes `primary:` the single marker, with no `default:` key. `_resolve_body_wiring` receives the resolved entry. The test fallout is W2-6 (NIT) |
| N4 | **PARTIAL** | Lints (a) and (b) are sound. **(c) is a silent no-op as written** (W2-1) |
| N5 | **PARTIAL** | The conflict between `--device` and `--robot-name` is refused. Three gaps remain: the env var outranks the explicit flag, the re-exec contract for non-device runs is undefined, and PR 0 ships with a broken re-exec in between (W2-2, W2-3, W2-4) |
| N7 | CLOSED | `--robot-name` defaults to `None`, the env var is read at resolution time, and the gate reads parsed explicit flags. The `--rob` ambiguity is documented. This only holds if the default change and the gate rewrite land in the same commit, see (a) |
| F1b | CLOSED (NIT W2-5) | Unknown entry-level keys, and `hardware_id`/`hostname` under `config:`, raise. The allow-list is not enumerated, and no red-gate test is named |
| F11 | CLOSED | `adopt(sdk)` is a stated §4.3 residual and an item in #1808 (§3c) |

## 2. Producer → consumer maps

### (a) `--robot-name` default → `None`, `MAXIM_ROBOT_NAME` read at resolution

| Reader (outside the change) | Today | With `None` |
|---|---|---|
| `cli.py::_main_impl` gate: `getattr(args,"robot_name","reachy_mini") != "reachy_mini"` | env/default → not an action | `None != "reachy_mini"` is **True**, so every bare `maxim` becomes a robot run unless the gate rewrite lands in the same commit. v3 puts both in PR 0 ✓ (say "same commit") |
| `cli.py::_main_impl` startup log `robot_name=%s` | name | prints `None`. NIT: log the resolved name and its source |
| `cli.py::_main_impl` → `Maxim(robot_name=args.robot_name)` | str | `selfy.py::Maxim.__init__`: `robot_name or os.getenv("MAXIM_ROBOT_NAME","reachy_mini")`, so the env var is read at resolution ✓. The signature is annotated `str` and should become `str \| None` (selfy and cli are not in `run_mypy_public.sh`) |
| `cli_utils.py::reexec_with_mode`: `str(getattr(args,"robot_name","reachy_mini"))` | a real name | **emits `--robot-name None`** (W2-4) |
| `tests/unit/test_cli_action_gate.py::_compute_has_action` (a copy of the gate) | passes | `test_bare_maxim_is_not_action` fails loudly, so the copy must go in favour of the real gate, as v3 says ✓ |
| `tests/unit/test_operational_mode_launch.py::_args()` (`parse_args([])`) | `robot_name="reachy_mini"` | `None`. The re-exec tests do not assert `--robot-name`, so nothing catches W2-4 |
| `api.py` | no reader | — |
| `scripts/reachy_streamer.py` | its own parser (`default="reachy_mini"`) | unaffected |
| `docs/user/cli-reference.md` (default `reachy_mini`), `configuration.md` (`MAXIM_ROBOT_NAME` row), `robot-setup.md` (`maxim --robot-name my_reachy`) | | update the cli-reference default text. `robot-setup.md`'s documented use is the case W2-3 breaks |
| §3.1 precedence list | | `MAXIM_ROBOT_NAME` is a selection source (it becomes `effective_robot_id` and matches an entry id exactly) but is **not listed** (W2-2) |

### (b) Re-exec with `--device <id>` alone

| Site | Finding |
|---|---|
| Child argv parser: `python -m maxim.cli` → `cli_parser.py::_build_parser` | parses `--device` ✓ |
| A parent run that resolved no entry (legacy mDNS, empty robots.yaml) | "`--device <resolved entry id>`" has no id. Dropping `--robot-name my_reachy` means the child dials `reachy_mini.local`, a **different, unpinned robot** (W2-3) |
| A parent on a legacy run that resolved an **unpinned** entry | the child gets `--device <id>` without `--unpinned`, so §3a refuses it. A mode switch kills a working run (fail-closed, W2-3) |
| A parent on a legacy run (D13′ headless fallback) | the child is now device-selected, so it exits non-zero instead of falling back to headless. The exit semantics change across the re-exec (W2-3) |
| Tests that pin the re-exec argv: `test_operational_mode_launch.py::test_every_runtime_switchable_name_survives_argparse`, `::test_a_run_mode_switch_keeps_the_operators_grant`, `::test_the_reexec_hands_over_child_env_not_a_pth_rewritten_pythonpath`; `test_bounded_stop_cli_1344.py` (monkeypatches `_reexec_with_mode`) | none asserts the robot flags. The new "keeps the device" gate needs rows for no-entry, unpinned and pinned |
| Phasing (§8) | the `--robot-name` default changes in PR 0, but the `--device` re-exec lands in PR 2b (W2-4) |

### (c) Strict entry-level keys vs every robots.yaml writer and fixture

| Producer | Entry-level keys | Refused under v3? |
|---|---|---|
| `scripts/exp49/exp49_common.py::robots_yaml_text` → `run_trials.py` | `type`, `primary`, `config` | no ✓ |
| `data/util/robots.yaml` template | `type`, `primary`, `config` (`timeout` sits under `config:`, so it is not an entry-level key; §5 removes it) | no ✓ |
| `scripts/survival_world/r3_run.py` | a comment only, no writer | — |
| `tests/unit/test_multi_robot.py`, `test_robots_yaml_connect_config.py`, `test_exp49_harness_seams.py`, `test_motion_finite_1496.py` (`from_dict` dicts); `test_exp49_harness_seams.py` (`robots: {}`) | `type`, `primary`, `config` only | no ✓ |
| `test_doa_feed.py` (`_FakeRobotConfig`) | not parsed | — |
| User docs | no documented robots.yaml with extra entry-level keys (grep of `docs/user`, `docs/agents`, `docs/embodiment`) | — |
| **The v3 writer itself:** §1 reserves `kind: robot` on entries, but §4.1's list of new fields is `hardware_id`, `hostname`, `launch` | if the allow-list is built from §4.1, the writer's own `kind:` is refused (W2-5, fail-closed) |
| `config.hostname` | not a `ReachyMiniController.__init__` parameter today (`host` is), so no live file carries it | ✓ |

### (d) `primary:` moved by `use` (D2″)

| Reader | Effect |
|---|---|
| `hardware/config.py::RobotsConfig.get_primary` (falls back to `robots[0]`) | reached only through `resolve_robot_entry`'s guard (one entry, or an explicit primary) ✓ |
| `hardware/config.py::resolve_robot_entry` (the no-name path and legacy `--robot-name`) | follows the moved marker ✓, and so do 1.3.x readers ✓ |
| `agentic_runtime.py::_resolve_body_wiring` (an inline copy of the rule) | removed by the v3 fold ✓ |
| `doctor/checks.py::check_robot_reachable` | iterates every entry and never reads `primary` ✓ |
| `hardware/config.py::connect_configured_robots` (`set_primary=robot_config.primary`) | dormant or routed through the pin (§5) ✓ |
| exp49 (one entry, `primary: true`) | unaffected ✓ |
| `tools/reachy.py`, `hardware/registry.py` `.primary` | the runtime registry's primary, a different concept ✓ |
| §7 "`forget` moves `primary:`" | moves it **to where?** If the primary is forgotten while ≥2 entries remain, `resolve_robot_entry` returns None on the no-name path, and `selfy.py::Maxim.__init__` dials the defaults (`reachy_mini.local`) unpinned, even when every remaining entry is pinned (W2-7) |

### (e) `agentic_runtime._resolve_body_wiring` receiving the resolved entry

| Site | Finding |
|---|---|
| Callers: `agentic_runtime.py::AgenticRuntimeMixin` (the start-up block, the one `src/` caller). The host is `selfy.py::Maxim`, the sole mixin consumer, built only from `cli.py::_main_impl` | the handoff is via `self._resolved_robot_config`, set in selfy where it resolves the entry. Name that attribute in the note |
| Other readers of `_resolved_robot_config`: the DoA-feed gate (`audio_localization`), the motor-binding gate (`motor_binding`, two sites), and the autonomic config (`aut_mode`), all in `agentic_runtime.py` | each must see the **same** entry that connect used. That is exactly the N3 close ✓ |
| `selfy.py::Maxim.__init__` resolves only under `if not self._simulation` | `simulation=True` gets no entry, so it runs bodiless. No `src/` caller passes it. State this as a residual |
| Tests: `test_multi_robot.py::TestResolveBodyWiring` (10 tests) and `test_motion_finite_1496.py::test_a_bad_yaml_orient_delta_runs_bodiless_with_one_error` all monkeypatch `hardware.config.load_robots_config` | once the fold lands, the patch is dead. The tests that assert ref == body fail loudly. The tests that assert bodiless, **including the F1 wrong-robot guard `test_multi_robot_no_primary_id_miss_is_bodiless`**, pass vacuously. Re-point them to the handoff; the match rule is already pinned in `test_exp49_harness_seams.py::test_resolve_robot_entry_*` (W2-6) |

### (f) The relative-`MAXIM_DATA_HOME` raise, scoped to robots.yaml resolution

| Harness (relative or unresolved data home) | Path taken | Resolves robots.yaml? |
|---|---|---|
| `scripts/benchmark_exp41_exploration.py` (`Path("data/sim_sandbox/exp41_runs")`) | `maxim --sim` | no |
| `benchmark_exp42_preference.py`, `benchmark_cradle_mother.py` (`Path(args.workdir)`), `benchmark_cross_session.py` (`--workdir`) | `maxim --sim` | no |
| `cli.py::_main_impl` sim branch | every path ends in `sys.exit` before the selfy import | — |
| `src/maxim/simulation/`, `src/maxim/interactive/` | no `hardware.config` or registry reader (grep) | — |
| The only `src/` robots.yaml readers: `selfy`, `agentic_runtime`, `doctor/checks.py::check_robot_reachable` (via `maxim doctor` or `api.py`'s `run_all_checks`) | none is on the `--sim` path | **confirmed** ✓ |

## 3. New findings

| # | Finding | Fail-open/closed | Covered class? |
|---|---|---|---|
| W2-1 | §3b(c) "`BLOCKED_PYTHON_IMPORTS` gains `maxim.devices`" is a **no-op**. `sandbox_executor.py::_restricted_import` compares only `name.split(".")[0]`, which is `"maxim"`, so a dotted entry never matches. And `import maxim; maxim.devices.add(...)` loads through `maxim/__init__.py::__getattr__` → `importlib.import_module`, which never calls `builtins.__import__` (probe below). Fix: block the top-level `"maxim"`, or delete (c) and state the residual next to the "not a security boundary" note. Either way, add a test that runs the sandbox and asserts the import fails | fail-open (silent no-op gate) | yes (F7/N4) |
| W2-2 | §3.1 precedence puts `MAXIM_DEVICE` (env, step 2) **above** an explicit legacy `--robot`/`--robot-name` (step 3), and only `--device` + `--robot-name` is refused. An exported `MAXIM_DEVICE` therefore silently overrides the robot the operator typed. `MAXIM_ROBOT_NAME` is also an unlisted selection source. "Only step 4 falls back to `primary:`" contradicts step 3's "today's matching", which includes the primary fallback. Fix: an explicit flag beats the env var, or `MAXIM_DEVICE` + explicit `--robot-name` is a refusal; list `MAXIM_ROBOT_NAME` as step 3b (legacy matching); define "conflicting" (the flag's value exactly matches a *different* entry id) | fail-open (wrong, if verified, robot) | yes (N5) |
| W2-3 | The re-exec "`--device <id>` alone, never `--robot-name`" is undefined for runs that were not device-selected. With no entry, an explicit `--robot-name my_reachy` (documented in `robot-setup.md`) is dropped, and the child dials `reachy_mini.local`: a **different, unpinned robot**. With an unpinned entry, the child lacks `--unpinned` and is refused. On a legacy run, the D13′ exit semantics flip. Fix: the re-exec replays the parent's **selection**: `--device <id>` (plus `--unpinned` iff the parent was granted it) when the parent was device-selected or resolved an entry; the parent's explicit `--robot-name` only when it resolved no entry; never a value that came from the env. Add test rows for each case | fail-open (no-entry case) / fail-closed (unpinned) | yes (F2/N5) |
| W2-4 | Phasing: PR 0 makes `--robot-name` default to `None`, but `reexec_with_mode` is untouched until PR 2b. It emits `--robot-name None`, and no test asserts it. In the child, the explicit name `"None"` beats `MAXIM_ROBOT_NAME`, falls to the primary, and becomes the daemon name. A parent that selected a non-primary entry through `MAXIM_ROBOT_NAME` therefore re-execs onto the primary robot. Fix: PR 0's re-exec omits `--robot-name` when it is `None` (or applies W2-3's rule then), with an assertion in `test_operational_mode_launch.py` | fail-open (wrong robot) | yes (N5/F4) |
| W2-5 | The strict allow-list is not enumerated. §1's reserved `kind:` is missing from §4.1's field list, so the writer's own file would be refused. F1b has no named red gate. Fix: list the allow-list `{type, primary, config, kind, hardware_id, hostname, launch}`. Add `test_robots_yaml_strict_keys` (a typo key, `hardware_id` under `config:`, `kind` accepted), proved by a deletion probe | fail-closed | yes (F1b) |
| W2-6 | After the N3 fold, the `TestResolveBodyWiring` tests and the 1496 test patch a dead `load_robots_config`. The ones that assert bodiless, including the F1 wrong-robot guard, pass vacuously. `simulation=True` gets no entry. Fix: re-point the tests to the `_resolved_robot_config` handoff, name the attribute, and state the `simulation=True` residual | fail-closed (the guard survives in `test_resolve_robot_entry_*`) | yes (F12 non-vacuous seams) |
| W2-7 | §7: when `forget` removes the primary entry, the note does not say where `primary:` moves. With ≥2 entries left and no primary, the no-name path resolves None, and selfy dials `reachy_mini.local` **unpinned** even though the registry is full of pinned entries. Fix: `forget` of the primary leaves no marker and says so. A no-name run then refuses whenever robots.yaml holds a pinned entry ("run `maxim connect <name>`") and never dials the default mDNS name. The legacy headless fallback stays only for a registry that holds no pinned entry | fail-open | yes (N3 one resolver / D13′) |

## 4. Evidence

- **W2-1.** `utils/sandbox_executor.py::_restricted_import`: `if name.split(".")[0] in blocked: raise ImportError`. `maxim/__init__.py::__getattr__`: `if name in _API_SUBMODULES: ... importlib.import_module(...)`. Probe (stdlib only, fresh `HOME`, `-I`), with a hook copied from that function and `blocked={"maxim.devices"}`: `importlib.import_module("json.decoder")` never reached the hook (`importlib hit hook: False`); `"maxim.devices".split(".")[0] in blocked` gave `False`.
- **W2-2.** v3 §3.1, the "Selection precedence" list and the sentence after it.
- **W2-3 / W2-4.** `cli_utils.py::reexec_with_mode` always appends `"--robot-name", str(getattr(args, "robot_name", "reachy_mini"))`. The attribute exists (argparse), so a `None` default stringifies to `"None"`. `selfy.py::Maxim.__init__` sets `self.name = robot_name or os.getenv("MAXIM_ROBOT_NAME", ...)`, then `effective_robot_id = self._robot_id or self.name`, then `resolve_robot_entry`'s exact match, else the primary. v3 §8 puts the `--robot-name` default in PR 0 and the "`--device` re-exec" in PR 2b.
- **W2-5.** v3 §1 ("Entries carry a reserved `kind: robot`") vs §4.1 ("entries gain `hardware_id:`, `hostname:` and `launch:`").
- **W2-6.** `test_multi_robot.py::TestResolveBodyWiring::_patch_robots` sets `cfg_mod.load_robots_config`, and `test_motion_finite_1496.py` does the same. `selfy.py::Maxim.__init__`'s resolution sits inside `if not self._simulation:`.
- **W2-7.** `hardware/config.py::resolve_robot_entry` returns None when there are several entries and none is primary. `selfy.py::Maxim.__init__` then gets `resolve_connection_config(...)`, which returns the defaults, and calls `connect_robot` with `{"robot_name": self.name, ...}`.

## 5. Stop-rule outcome

**Not met.** W2-1 (a silent no-op gate), W2-2, W2-3 (the no-entry case), W2-4 and W2-7 are fail-open. Every one falls in a covered class (the writer gate, selection precedence and re-exec, one resolver), and each needs a text fix to the note, not a new decision. W2-5 and W2-6 are fail-closed and covered, so they can be folded without a round. I recommend a delta round 3 for the wire lens, scoped to the v4 text for W2-1 to W2-4 and W2-7. If those are folded as written above, I expect that round to meet the stop rule.
