# #1806 PR 0 (#1817): delta round 2, Executor lens

Scope: `git diff 3457f3ef..46e68801` in `.worktrees/pr0-1806-groundwork` (HEAD 46e68801). Probes ran in a copied tree
(`scratchpad/d2_exec/repo`, absolute `PYTHONPATH`, scratch `HOME`, with no network, robot or sim). Imports were
confirmed to come from the copy.

**Verdict: SHIP. No DO-NOT-MERGE and no SHOULD-FIX findings. Four NITs, all fail-closed or test/doc-only. The round may stop.**

## Doctor vs run agreement (round-1 D1)

`connects_as_given(outcome)` is `failure == "unsupported"`. `"unsupported"` arises only for an IPv6 literal, or for an
IPv6-only name when `explicit_host=True`. Doctor passes `explicit_host=bool(host)` and the controller passes
`self._host is not None`. Neither caller can hand an IPv6 literal with `explicit_host=False`: the untyped target is
always `<name>.local` or `127.0.0.1`. So the controller dropping `and not mdns` changes nothing.

| Input | probe failure | doctor | run (`_precheck_daemon`) | agree |
|---|---|---|---|---|
| IPv4 literal `host:` | none, or tcp/status | grades the outcome | dials it | yes |
| IPv6 literal `host:` (incl. `[fe80::1]`, `fe80::1%en0`) | unsupported | info "connects as given" | `(target, None)` | yes |
| typed name, IPv6 only | unsupported | info | `(target, None)` | yes (test) |
| typed name, both families | none (v4 wins) | grades the outcome | dials it | yes |
| typed name, unresolvable | resolve | fail | `(None, None)` | yes (probe) |
| no `host:`, `<name>.local` IPv6 only | resolve | fail | `(None, None)` | yes (test) |
| no `host:`, unresolvable | resolve | fail | `(None, None)` | yes |

A typed host and a `robots.yaml` host are the same input to both: each reads `config["host"]`, and neither treats a
CLI-typed host differently. `reachy_diagnostics` passes `explicit_host=True`, which is correct: its host is `--host`,
`$MAXIM_REACHY_HOST` or the IPv4 default `10.42.0.1`.

Pre-existing and outside the delta, noted only: `host: ""` is falsy for doctor, so it probes `<name>.local`, but it is
`not None` for the controller, so it probes `""`. That target split predates this delta; the new flag just follows
each caller's existing rule.

## Deletion probes (one mutation at a time; tests: probe_daemon + writedeny files, 91 tests)

| Mutation | Result |
|---|---|
| probe: drop the `explicit_host and` gate | 2 fail |
| doctor: `explicit_host=True` / `=False` | 1 fail / 1 fail |
| controller: `explicit_host=True` / `=False` | 1 fail / 1 fail |
| diagnostics: `connects_as_given` branch → `if False:` | 1 fail |
| diagnostics: `explicit_host=False` | **0 fail** (N2) |
| deny list: drop `("util", "cost_state.json")` | 5 fail |

The `cost_state.json` entry is correct. It sits in `_CONTROL_PLANE_DATA_FILES`, so all three tests (stat walk, string
compare, name rule) cover it under both the data home and `~/.maxim`. The docstring list now matches the tuple
(round-1 D1-W3 closed).

## Findings

| # | Site | Finding | Class |
|---|---|---|---|
| N1 | `tests/unit/test_1806_pr0_config_dir_agreement.py::test_no_other_module_reads_the_config_dir_env_vars` | **The scan catches the literal form it targets, but the docstring claims more ("no new hand-copied config-dir resolver can appear").** I planted each of these in a scratch module and the test still passed: `Path.home() / ".config" / "maxim"` (the likeliest hand copy; `tools/base.py` already carries an intended one at the deny-list roots), `from os import environ as env; env["XDG_CONFIG_HOME"]`, `os.environ.get(_K)` with a module constant, `os.path.expandvars("$XDG_CONFIG_HOME/maxim")`, and `os.environ.setdefault("APPDATA", ...)`. Scope is keyed by function name only, so `class C: def maxim_config_dir(self): os.environ.get("XDG_CONFIG_HOME")` added to `utils/paths.py` also passes. A planted `os.environ.get("XDG_CONFIG_HOME")` in a new module correctly fails it. **Scenario:** a future peer-side resolver written as `Path.home()/".config"/"maxim"` drifts from XDG without the guard noticing. Fix: narrow the docstring to "no other module reads XDG_CONFIG_HOME/APPDATA by literal name", or also flag adjacent `".config"`/`"maxim"` string constants with an allow-list. | NIT (test claim) |
| N2 | `utils/reachy_diagnostics.py::diagnose_reachy` | Nothing guards `explicit_host=True`. The test uses an IPv6 literal, which is `unsupported` whatever the flag. **Scenario:** a revert to the default `False` makes an AAAA-only `--host` print "does not resolve" plus "Ensure the daemon is running" (the D4 wrong advice), and no test fails. Add one IPv6-only-name row. | NIT |
| N3 | `hardware/reachy/controller.py::_precheck_daemon` | `outcome.failure in ("resolve", "unsupported")`: `"unsupported"` can no longer reach this line, because `connects_as_given` returns first. The dead member suggests a path that does not exist. | NIT |
| N4 | `hardware/reachy/probe.py::probe_daemon` / `_has_ipv6_address` | A typed `host: robot.local` that fails IPv4 still pays a second `getaddrinfo(AF_INET6)`. That call has no timeout (`setdefaulttimeout` does not bound name lookups), in doctor and at connect. The docstring's "never pays a second mDNS wait" is correctly limited to the untyped `<name>.local`. "`deadline_s` bounds EACH network stage" still reads wider than this second lookup. It fails closed (only a delay). | NIT |

## Stop rule

Every finding is fail-closed (N4 only delays) or test/doc-only (N1–N3), and each is in an already-covered class
(guard coverage, docstring precision, dead branch, the round-1 D4 lookup residual). **No findings require another
round; the round may stop.** N1–N3 are small enough to fold without a further round only if the fold is test/doc-only;
a `src/` fold of N3 would owe a delta-scoped look per CLAUDE.md.
