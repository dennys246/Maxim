# #1806 PR 0 (#1817): delta round 3, all three lenses

Scope: `git diff 46e68801..3a7f386a` in `.worktrees/pr0-1806-groundwork` (7 files, +164/−47), which folds round 2
(`pr0_delta2_*.md`). I ran the probes in a `git archive` copy at 3a7f386a (`scratchpad/d3/repo`), with a scratch HOME,
an absolute `PYTHONPATH` on its own line, and the import origin confirmed to be the copy. No network, robot or sim was used.

**Verdict: one DO-NOT-MERGE (D1, in the docs/claims the delta adds). The code changes are sound. Three NITs. The round
may stop once D1 is folded. If D1 is folded as the recommended `src/` deny-list entry, that fold owes a delta-scoped
look. If it is folded as a docs-only retraction, it does not.**

## Mechanical checks

- `test_1806_pr0_probe_daemon`, `test_1806_pr0_config_dir_agreement`, `test_reachy_connection_options`,
  `test_1806_pr0_writedeny_control_plane` and `test_doctor` all pass in the copy (208 tests).
- `ruff check` and `ruff format --check` are clean on the 5 changed .py files.
- `reachy/controller.py` has no entry in `function_length_baseline.json`, so the touch rule does not apply.

Deletion probes, one mutation at a time (each turns exactly one test red):

| Mutation | Result |
|---|---|
| `reachy_diagnostics`: `explicit_host=True` → `False` | 1 fail (`test_maxim_diagnostics_treats_its_host_as_typed`) |
| controller: `self._host = host or None` → `host` | 1 fail (`test_doctor_and_the_controller_agree_that_an_empty_host_is_unset`) |
| doctor: `if host` / `bool(host)` → `is not None` | 1 fail (same test) |

## Executor and Wire: `self._host = host or None`, `_precheck_daemon`

- **Every `self._host` use in controller.py** is listed below. Each one already read `""` as unset, or is now consistent:
  - the tunnel guard (`if not self._host`);
  - the ssh target and its log lines (reached only after that guard, so `_host` is truthy there);
  - `_precheck_daemon` (the `mdns`, target and `explicit_host` derivations all use `is None`, so `""` now probes
    `<name>.local` instead of `gethostbyname("")` → `0.0.0.0`, which was round 2's N2 scenario);
  - `_warn_if_media_already_released(self._host or probe_host)` (unchanged in effect).
- **No external reader exists.** `\._host\b` has no hit outside controller.py in `src/` or `scripts/`. In tests,
  `test_reachy_connection_options` asserts `"10.42.0.1"` and `None`, and both still hold. The controller has no
  `host` property.
- **Every constructor that passes `host=`:**
  - `hardware/registry.py` passes `controller_class(**ctor_config)`, the YAML `config:` dict straight through
    (`config.py::merge_robot_config`).
  - `api.py`, `selfy.py` and `hardware/__init__.py` only register the class.
  - `hardware/controller.py:516` passes no host.
  - The four `scripts/orient_backbone/*` scripts pass the result of `resolve_host`, and each exits when that result
    is `None`.
  - Nothing passes `""` meaning "bind locally" or anything else. A falsy non-string such as YAML `host: 0` now becomes
    `None` in both doctor and the controller, so they agree.
- **`outcome.failure == "resolve"` is behaviour-identical.** `connects_as_given` is exactly
  `failure == "unsupported"` (`probe.py::connects_as_given`), and it returns first, so no outcome that used to hit
  the tuple branch now falls through.

## Findings

| # | Site | Finding | Class |
|---|---|---|---|
| D1 | `docs/user/safety.md` (new "A moved spend file" bullet), `changelog.d/1806-pr0-control-plane-write-deny.security.md`, `tools/base.py::control_plane_write_refusal` docstring | **The new residual names the wrong file, and its reassurance is false.** `routing` is read from **`llm.json`**, not `config.json`: `models/language/config.py::load_llm_config` → `raw.get("routing")`, with `raw` from `_llm_config_candidates()` = `$MAXIM_LLM_CONFIG`, then `<data_home>/config/llm.json`, then the CWD/repo-root `data/util/llm.json` and `llm.json`. The same `routing` dict also carries the caps themselves (`cloud_dispatch.py::load_routing_policy`: `max_cost_per_request/hour/day/month`, `max_session_cost`). `llm.json` is **not** denied: probed with `MAXIM_DATA_HOME=<tmp>/dh`, `control_plane_write_refusal("<dh>/config/llm.json")` and `("~/.maxim/config/llm.json")` both return `None`, while `util/cost_state.json` and `~/.config/maxim/config.json` are refused. So "(`config.json` itself is protected, so the agent cannot set it.)" is false, and so is the fragment's "an agent cannot … reset the budget cap by editing a file". | **DO-NOT-MERGE** (a false security claim in a `.security.md` fragment; label class `security`) |

**D1 scenario.** A singularity agent uses `write_file` on `~/.maxim/config/llm.json`, creating it if absent, since
it is the first candidate after the env var. It either sets `routing.max_cost_per_month: 1e9`, or sets
`routing.cost_state_path` to a fresh file. After the next restart the budget is unbounded or reset, and the cost
state that the PR protects was never touched. The same file's `providers` entries feed the router's provider config
(`router.py` reads `provider_cfg.get("base_url")`), so it is likely a trust anchor beyond spend as well. I did not
trace the full dispatch.

**D1 fix.** This is the same root cause and well under 50 lines, so per "Filing is not fixing" it belongs in this PR:

1. Add `("config", "llm.json")` to `tools/base.py::_CONTROL_PLANE_DATA_FILES` and add the matching test rows. The name
   rule and roots already derive from the tuple.
2. Fix the safety.md bullet and the docstring to say `llm.json`, and list as residuals the `MAXIM_LLM_CONFIG`
   override and the CWD/repo-root fallbacks. The CWD fallback matters: with no data-home `llm.json`, an agent can
   create `./llm.json` in its cwd, and the next run launched from that cwd loads it.
3. Keep the `cost_state_path` residual, since an operator-moved spend file is still outside the deny.

**Fallback if the owner defers the `src/` change.** Retract the budget sentence in the fragment and safety.md, and
state the `llm.json` gap plainly.

**NITs**

| # | Site | Finding | Class |
|---|---|---|---|
| N1 | `tests/unit/test_1806_pr0_config_dir_agreement.py::_flagged` | **A path in a single constant is not flagged.** `os.path.expanduser("~/.config/maxim")`, `Path.home() / ".config/maxim"` and `f"{h}/.config/maxim"` each pass the scan in a planted module. Only a bare `".config"` piece is caught. The docstring's stated limit names only runtime assembly. None of these forms exists as a resolver today; the ~15 `~/.config/maxim` hits in `src/` are messages and docstrings, which is presumably why they are not flagged. Fix: add the single-constant form to the stated limit, or also flag constants containing `/.config/maxim` outside f-string messages. | NIT (test claim) |
| N2 | same file, `_CONFIG_SCAN_ALLOWED` `("tools/base.py", "<module>")` | **The module-scope allowance covers all of `tools/base.py`'s top level.** Today it covers only the env-passthrough list entry `"XDG_CONFIG_HOME"` and `_CONTROL_PLANE_CONFIG_PAIR = (".config", "maxim")`. A planted module-level `_XDG = os.environ.get("XDG_CONFIG_HOME")` in that file passes. The risk is bounded to the deny matcher's own file, where the round-1 drift concern lives. Also, only `maxim_config_dir` is asserted present, so a stale allowance would stay silently open. Fix: key the module allowance on the assignment target (for example `<module>:_CONTROL_PLANE_CONFIG_PAIR` plus the passthrough set's name), and assert that every allowed scope is hit. | NIT |
| N3 | `tests/unit/test_1806_pr0_probe_daemon.py::test_maxim_diagnostics_treats_its_host_as_typed` | **The test makes a real outbound attempt, which the in-process guard blocks.** It patches `socket.getaddrinfo` globally (via `probe.socket`), so `diagnose_reachy`'s Test 3 (`maxim_http.fetch_url("http://v6only.example:8000/docs")`) resolves to `fd00::5` and dials it. The run prints `network guard: blocked 1 outbound call(s) (connect)`; the IPv6-literal sibling test does not. The test passes, and the guard makes it harmless in-process. Fix: also stub `diag.maxim_http.fetch_url`, or patch `test_port(8443)` and fetch together. | NIT |

## Doc claims checked against code

- The controller comment ("doctor (`bool(host)`) and the tunnel check (`not self._host`) already read it") matches
  `doctor/checks.py::_check_one_robot` and `connect`.
- The `probe.py` docstring is accurate: a typed `x.local` pays one more lookup with no timeout, and a derived
  `<name>.local` does not (round 2 N4 and A5b closed).
- The `tools/base.py` docstring residual and the fragment's "at its default location" qualifier are accurate as far as
  they go. They are incomplete per D1.

## Round-2 fold verification

| Round-2 item | Status |
|---|---|
| A1 (`cost_state_path` residual) | Folded, with the error in D1 |
| A2/N3 (dead `"unsupported"`) | Folded |
| N1/A4 (scan blind spots) | Folded. All 8 planted forms are caught; N1 and N2 here are the leftovers |
| N2 (diagnostics flag guard) | Folded; the deletion probe bites |
| Wire N2 (`host: ""` split) | Folded; the deletion probe bites |
| N4/A5b (docstring) | Folded |
