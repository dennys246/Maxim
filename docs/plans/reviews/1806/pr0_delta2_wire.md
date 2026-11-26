# #1806 PR 0 — delta round 2, Wire integrity lens

Scope: `git diff 3457f3ef..46e68801` in `.worktrees/pr0-1806-groundwork` (HEAD 46e68801). Probes ran in a copied tree
under the session scratchpad (`d2_wire/tree`, scratch HOME, absolute PYTHONPATH). The three touched test files pass
there: 100 passed.

**Verdict: no DO-NOT-MERGE and no SHOULD-FIX. Three NITs. The round may stop.**

## 1. `probe_daemon` / `connects_as_given` / `failure == "unsupported"`: every caller outside tests

| Caller | `explicit_host` | Handles `unsupported` | Agrees with the run? |
|---|---|---|---|
| `hardware/reachy/controller.py::_precheck_daemon` | `self._host is not None` | `connects_as_given` first → dial as given | it is the run |
| `doctor/checks.py::_check_one_robot` | `bool(host)` | `connects_as_given` first → `info` | yes, apart from `host: ""` (N2) |
| `utils/reachy_diagnostics.py::diagnose_reachy` | `True` (a typed `--host`) | `connects_as_given` first → warning, `all_passed=False` | n/a (the tool does not connect) |
| `hardware/reachy/probe.py::refusal` | — | `"not probed: …"` | reached only from doctor's tail and diagnostics' `reachable` branch, both after the `connects_as_given` branch, so it is unreachable for `unsupported` today (harmless) |

`scripts/check_reachy_connection.py` is the standalone copy for a machine with no Maxim install. It does not import
the probe. `embodiment/audio_localization.py` only mentions the probe in a docstring. **No caller still reads an
IPv6 or unsupported outcome as "daemon down".** `unsupported` now arises only for an IPv6 literal (which needs a typed
host, because a default target is always `<name>.local` or `127.0.0.1`) or for an IPv6-only name when `explicit_host`
is set. So `connects_as_given` dropping the old `and not mdns` changes nothing.

`reachy_diagnostics` exit code: `all_passed=False` for IPv6 makes `maxim-diagnostics` exit non-zero. Its only
consumer is the `pyproject` entry point (`main` returns it). No `src/` or `scripts/` code branches on it, so nothing
breaks.

## 2. `util/cost_state.json`: writers and the deny matcher

- The one writer is `models/language/cost_tracker.py::CostTracker` (`_save_state` → `atomic_write_json`), built only
  by `router.py` from `cloud_dispatch.py::load_cost_config`. Its default path is
  `CostTrackerConfig.__post_init__` → `paths.resolve_user_state("util/cost_state.json")` = `data_home()/util/cost_state.json`.
  That is the same `data_home()` the matcher's roots use, so `MAXIM_DATA_HOME` is honoured on both sides.
- Probed with `MAXIM_DATA_HOME=<tmp>/dh`. Each of these is refused: the writer's own path; a symlink to the file; the
  file reached through a symlinked parent directory; a case-folded spelling (`DH/UTIL/Cost_State.json`); a `..`
  spelling; and `<anywhere>/.maxim/util/cost_state.json` (the name rule). `cost_state.json.tmp` is writable, the same
  as every other row. That does not matter, because `atomic_write_json` uses its own tempfile.
- Maxim's own writer goes straight to `atomic_write_json` and never through the file tools, so the deny does not stop
  real persistence.

## 3. Config-dir env reads vs the new AST scan

A grep of `src/maxim` finds the env reads only in `utils/paths.py::maxim_config_dir` and
`tools/base.py::_global_git_config_paths`. Those are exactly the scan's allowlist. Line 764 in `tools/base.py` is a
name in a scrub list, not a read. No hand-built `Path.home()/".config"/"maxim"` exists outside the resolver (the
matcher's own `os.path.join(h, ".config", "maxim")` is a deny root, not a resolver). **Nothing is missed today.**

## Findings

**N1 (NIT): the D5 scan has blind spots, so it guards less than its docstring claims ("no new hand-copied
config-dir resolver can appear").** I planted a module with five forms and the test still passed:
`os.environ.get(_X)` with `_X = "XDG_CONFIG_HOME"` (a non-literal argument);
`env = os.environ; env["XDG_CONFIG_HOME"]` (the subscript check needs `environ` in the receiver's text);
`"XDG_CONFIG_HOME" in os.environ`; `os.getenv(key="APPDATA")`; and `Path.home() / ".config" / "maxim"`. Aliased
`.get` calls are caught, because the receiver is ignored. Scenario: a future helper builds
`Path.home()/".config"/"maxim"` by hand, and the deny set and the real path drift apart unseen.

Cheap tightening, if wanted: also flag any `ast.Constant` equal to the two names outside the allowlist, plus a
`".config"` constant next to a `"maxim"` constant. Otherwise narrow the docstring to "literal-argument env reads".

**N2 (NIT, mostly pre-existing): doctor and the controller disagree on `host: ""`.** Doctor uses `bool(host)` and
probes `<name>.local` as not explicit. The controller uses `self._host is not None` and probes `""` as explicit (the
registry passes the YAML value straight through). The delta's own claim, "doctor says connects as given exactly when
the controller connects as given" (`test_doctor_and_the_controller_agree_on_an_ipv6_only_target`), therefore holds
only for a non-empty host. The target divergence predates the delta. The delta newly encodes the two different
predicates into `explicit_host`. Scenario: `host: ""` makes `gethostbyname("")` return `0.0.0.0`, so the controller
probes and dials the local box while doctor reports on the `.local` robot. Fix shape: one shared `explicit host`
predicate (for example normalise `""` to `None` in the controller ctor), or record it for #1806 PR 1's pinned-entry
rule.

**N3 (NIT): there are two small leftovers in the delta.**
- `controller.py::_precheck_daemon` line 396 `if outcome.failure in ("resolve", "unsupported")`: the `"unsupported"`
  member can no longer be reached, because `connects_as_given` returns on every `unsupported` just above it. It is
  harmless, but it misleads.
- `routing.cost_state_path` (`cloud_dispatch.py::load_cost_config`) is an operator override that puts the spend file
  outside the denied set. `docs/user/safety.md` now says without qualification that "an agent cannot zero it". The
  knob itself lives in the protected `config.json` and has no user docs, so the exposure is small. Either add one
  residual clause to safety.md, or add the configured path to the deny roots.
