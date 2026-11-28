# #1806 PR 0 delta round 2: architecture lens

Reviewer: architecture lens, PR 0 delta round 2, 2026-11-25

Scope: `git diff 3457f3ef..46e68801` only (13 files, +161/−32), which folds round 1's D1–D5 (`pr0_delta1_*.md`).

**Verdict: SHIP after one docs-only fold.** There is no DO-NOT-MERGE. A1 is a SHOULD-FIX that needs only a docs change. A2–A5 are nits. Because the SHOULD-FIX fold is docs-only, it does not owe another round.

## Mechanical checks

I ran these in a scratch two-commit git copy (base 3457f3ef, head 46e68801), with a scratch HOME and `PYTHONPATH` set to the copy's `src`:

- `lint_function_length` (touch rule included) is clean. So are `lint_no_silent_swallows` (no new swallows), `ruff check` and `ruff format --check` on the changed .py files.
- 254 tests pass: `test_1806_pr0_probe_daemon`, `test_1806_pr0_config_dir_agreement`, `test_1806_pr0_writedeny_control_plane` and `test_doctor`.
- `changelog_fragments.py check` fails on this delta alone. That is expected and not a finding: the delta edits fragments that the PR added earlier, and on the whole PR against main they count as added.
- CC3: the delta adds no dataclass. `ProbeOutcome` is unchanged and already on the path-(a) roster.
- I ran two deletion probes on the D1 guard, and both turned red:
  - doctor passing `explicit_host=True` unconditionally fails `test_doctor_and_the_controller_agree_on_an_ipv6_only_target`;
  - dropping the `explicit_host and` gate in `probe_daemon` fails two tests.

## Round-1 fold verification

| Prior | Folded? | Notes |
|---|---|---|
| D1: one rule for "connects as given" | yes | See the seam assessment below. Doctor and the controller now agree on both paths: a typed IPv6-only `host:` gives `info` from doctor and a dial-as-given from the run, and a derived IPv6-only `.local` gives `fail` from doctor and a skip from the run. The guard is real (the deletion probes above). |
| D2: `util/cost_state.json` | yes | The tuple entry, the test rows in both data-home parametrizations, the union-roots assert, and all four doc lists. The name-rule suffixes are still derived from the tuple. One gap: A1. |
| D3: list drift | yes | There is no count left in the docstring, safety.md or the test docstring. The changelog's doubled "and" is gone. |
| D4: second mDNS wait | yes | The AAAA lookup runs only with `explicit_host`. The docstring now states the `setdefaulttimeout` residual (this also closes the old A8 nit). |
| D5: config-dir source scan | yes | It has a two-entry allow-list plus a non-vacuity assert. Maintainability is covered under A4. |

## Seam assessment (the question asked)

The seam is sound for PR 0. The rule now has exactly two parts, both in `hardware/reachy/probe.py`:

- `probe_daemon(..., explicit_host=)` decides whether an IPv6-only name can become `"unsupported"`;
- `connects_as_given(outcome)` is the one predicate both consumers branch on.

Neither consumer looks at `failure == "unsupported"` any more.

What each caller still derives for itself is `explicit_host`, in three spellings:

| Caller | `explicit_host` |
|---|---|
| doctor | `bool(host)` |
| controller | `self._host is not None` |
| diagnostics | `True` |

Each spelling matches that caller's own target derivation, so the probe gets a consistent answer from each. The remaining divergence (for example `host: ""`, which doctor treats as absent and the controller treats as typed) belongs to the target derivation, which existed before this delta. PR 1 deletes doctor's private address choice (approach §3.1, the "Doctor too" item), so it is not this delta's to fix.

## Findings

| # | file::symbol | finding | severity | scenario / fix |
|---|---|---|---|---|
| A1 | `docs/user/safety.md` §"Maxim's own control-plane files", `changelog.d/1806-pr0-control-plane-write-deny.security.md`, `tools/base.py::control_plane_write_refusal` docstring | **The new cost-state claim holds only for the default location.** `models/language/cloud_dispatch.py::load_cost_config` honours an operator `routing.cost_state_path` override, and `CostTrackerConfig.__post_init__` resolves `util/cost_state.json` only when that override is empty. The deny list matches a fixed relative name under the data-home roots, so an overridden file is writable by `write_file`/`edit_file`. The four denied files that came before it have no such override: the internet files and the hive sequence resolve only through `data_home()`. | SHOULD-FIX (docs-only) | **Scenario:** an operator sets `routing.cost_state_path: ~/work/maxim_costs.json`. A singularity agent edits it to zero spend. On restart `_budget_status` sees a fresh budget, while the security fragment says "an agent cannot … reset the budget cap by editing a file". **Fix (cheap, recommended for PR 0):** add the override to safety.md's "A non-default location you set yourself" residual ("…or `routing.cost_state_path`, which moves the cost state out of the data home"), and give the fragment and docstring the same qualifier. **Fix (derived, later):** let the deny set union `CostTracker`'s resolved `state_path`. That needs the tools layer to see the LLM config, so it belongs with A3's registry, not in PR 0. |
| A2 | `hardware/reachy/controller.py::ReachyMiniController._precheck_daemon` | **A dead re-derivation remains.** `if outcome.failure in ("resolve", "unsupported")` follows `connects_as_given`, which now returns on every `"unsupported"`, so that half of the tuple can never match. It suggests to a later reader that an `"unsupported"` outcome can reach the mDNS "could not resolve" log. | NIT | Change it to `outcome.failure == "resolve"`. |
| A3 | `tools/base.py::_CONTROL_PLANE_DATA_FILES` (the "derive the list?" question) | **A hand list is the right layer for PR 0, but nothing guards it.** Two rounds in a row grew it from a manual `util/` survey (hive, then cost). The remaining `resolve_user_state("util/…")` files are classified only in review prose. My spot-check found no live gap: `fear_learning.json` is Dormant (#840) and has no consumer, and `learned_bounds`/`workspace_bounds` do not feed the hard clamp (round 1). Deriving the list in `tools/` would import the owners (router, signer, internet), which is the wrong direction. | NIT | Add a guard test, not a derivation: AST-scan `src/maxim` for `resolve_user_state("…")` and `data_home() / …` literals, and require each to be either in `_CONTROL_PLANE_DATA_FILES` or in a test-side "not control plane" dict with a one-line reason. A new persisted limit file then fails CI until someone classifies it. It is under 50 test lines, so either fold it now or carry it on the PR 1 checklist. |
| A4 | `tests/unit/test_1806_pr0_config_dir_agreement.py::test_no_other_module_reads_the_config_dir_env_vars` | **The allow-list shape is maintainable.** It is keyed by `(relpath, innermost function name)`; a rename or move fails loudly, and the non-vacuity assert pins the real resolver. Its blind spot is the other drift form: a hand-built `Path.home() / ".config" / "maxim"` that reads no env variable, which is exactly an XDG-ignoring copy. I found none today. The only site is the deliberate union in `_control_plane_roots`. | NIT | Optional: also flag a `".config"` string constant followed by a `"maxim"` one in the same call outside the two allow-listed functions plus `tools/base.py`'s roots helper. The test can also be left as is, with the docstring saying it covers env reads only. |
| A5 | `hardware/reachy/probe.py::connects_as_given` / `probe_daemon` docstrings; approach note §8 | **Two forward-API notes for PR 1.** (a) `connects_as_given` is framed universally ("whether a robot run … dials this target as typed"), but it is true only of the legacy, unpinned path. For a pinned entry, §4.2 says IPv6 is out of v1 and `refusal()` already refuses `"unsupported"`. If PR 1's resolver passes `explicit_host=True` for a pinned `host:` name and a caller branches on `connects_as_given` first, an AAAA-only pinned host would be dialled without the identity check. (b) The docstring sentence "an unresolvable `<name>.local` … never pays a second mDNS wait" means the *derived* name; a typed `host: foo.local` does pay it. | NIT | (a) Add to the §8 PR 1 checklist: the one resolver returns `(target, explicit_host)` so the flag is derived once, and pinned entries go through `refusal()` and never through `connects_as_given`. Alternatively, give the predicate the entry, mirroring `refusal(outcome, entry)`. (b) Say "a derived `<robot_name>.local`". |

## Doc claims checked against code

- runtime-tools.md stub, modes-guide.md, safety.md list and stack item 4, the test docstring and the refusal docstring all match `_CONTROL_PLANE_DATA_FILES`. The only exception is A1's override.
- The `.fixed.md` cache bullet ("reads stay fresh either way because the cache checks file modification times") is backed: `runtime/fetch_cache.py::ToolResultCache.get` drops a `read_file` entry whose mtime changed.
- The `.fixed.md` doctor bullet ("an IPv6 `host:` (or a name with only an IPv6 address) … the robot run still connects to it as given") is now accurate, because it is scoped to `host:`. The `maxim-diagnostics` IPv6 message and the 8 s → 3 s per-stage deadline are not in a fragment. The deadline change is a small change (the TCP stage goes from 5 s to 3 s against main; the lookup was never bounded), so no fragment is needed.

## Stop-rule outcome

No DO-NOT-MERGE. A1 is SHOULD-FIX and fail-open, but the fix is docs-only, and per CLAUDE.md a docs-only touch-up does not owe another round. If the derived-path fix is chosen instead, that is a `src/` fold and needs a delta round. A2–A5 are nits. **The round may stop once A1's docs fold lands.**
