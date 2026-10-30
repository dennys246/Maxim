# Step 0a (#1315, V9): architecture lens

Reviewed: the staged diff on `feat/1315-step0a-lazy-inits` (27 files), checked against
`docs/plans/atonia_gate_approach_notes.md` §1A.1 and its as-built status block, `ARCHITECTURE.md`, CLAUDE.md and the
runtime-tools, llm-routing and ci-testing briefs. Reviewed 2026-10-29. I did not run mypy, because the box is busy.

## Verdict

The seam is the right one. Moving the merge to first use behind ensuring accessors, guarded by an AST scan over
`src/` and `scripts/`, fixes the problem at its root. It is not a special case. The lazy inits are mechanically
sound. I checked all four packages:

- `_LAZY` equals `__all__`.
- Each `_LAZY` target defines the name at module top level.
- No public name collides with a submodule name.
- Every name the old eager inits bound still resolves.
- Each `TYPE_CHECKING` mirror matches its `_LAZY` table exactly.
- Every lazy target lies inside its own package.
- `maxim --audit-architecture` reports nothing new.

These lints are clean: invariants, changelog fragments, function length, orphans, dead imports, silent swallows,
loop-mode reads and atomic I/O. The step 0a guard tests pass: 17 passed with 1 expected xfail (#1479) in
`test_profile_readers_v9.py`, and both `test_advisor_gate.py` gates pass.

Two findings need action before the PR: one BLOCKER (fail-closed) and one SHOULD-FIX (fail-open).

Module counts, measured fresh (`python -I`, empty `HOME`/`XDG_CONFIG_HOME`):

| import | HEAD: maxim / total `sys.modules` | branch: maxim / total |
|---|---|---|
| `maxim.runtime.executor` | 236 / 843 | 8 / 131 |
| `maxim.agents.permissions` | 237 / 844 | 3 / 116 |
| `maxim.models.language.config` | 27 / 190 | 20 / 150 |
| `maxim` (bare interpreter baseline is ~91) | 1 / 91 | 1 / 91 |

So the CHANGELOG's "about 130 modules instead of about 830" is accurate when it counts all of `sys.modules`.

## BLOCKER

**B1. The new slow test is not in the slow-lane roster, so the fast suite goes red.** Fail-closed (loud).
`tests/integration/test_fresh_import_sweep.py::test_every_module_imports_first_in_a_fresh_process` is
`@pytest.mark.slow`, but `scripts/lane_rosters/slow.json` is not in the diff. I ran
`tests/unit/test_lane_rosters.py::test_the_roster_is_exactly_what_collection_selects[slow]` on this branch, and it
FAILS: `not in the roster ['tests/integration/test_fresh_import_sweep.py::test_every_module_imports_first_in_a_fresh_process']`.
On CI, the PR's fast suite fails. If that were bypassed, the nightly's `check_lane_roster.py` step would fail instead.

Fix: run `python3 scripts/check_lane_roster.py --lane slow --generate` and commit the reviewed diff.

The roster also matters for the guard itself. The runtime-tools invariant cites this sweep as a regression guard, and
the slow lane (nightly, `--require-extras=console,sign`) is the only place it runs. Without the roster entry, that
guard never runs anywhere.

Budget note: the sweep starts 543 interpreters, one per module, inside a lane with `timeout-minutes: 60`. Time one
local run and record the wall time in the roster PR, so a nightly timeout cannot surprise anyone.

## SHOULD-FIX

**S1. A malformed `profiles.yml` no longer fails at startup. It now fails at the first reader that does not swallow
the error, and some readers that swallow it make a wrong decision.** Fail-open (narrow).

Before this change, any process that imported `models/language/config.py` raised `ConfigurationError` at import, so
no profile reader ever ran against a broken file. Now the first accessor raises, and several readers wrap that
accessor in `except Exception`. I probed the branch with a malformed file
(`profiles: {bad: {backend: nope}}`):

- `runtime/lane_backends.py::_lane_profile_is_cloud(cfg(model_profile="claude-sonnet-4-6"), ...)` returns **False**.
  The `ConfigurationError` lands in its `except Exception` and is logged at DEBUG. `_maybe_auto_spawn_server` then
  goes on to spawn a llama server for a cloud lane, which is the exact waste that block exists to prevent. On HEAD the
  same call cannot happen, because the import raises first.
- `LaneBackendManager` (the cost-tracking `has_cloud…` check) returns True through its `except`. That is
  conservative, but it is still a decision taken on a swallowed configuration error.

"Failure is never cached" holds: each access raises again. But a swallowing site swallows every time. Whether a run
stops on a broken file now depends on whether a non-swallowing reader happens to run first. Nothing guarantees that
on the `maxim --sim`/agentic path when no `--llm` is passed. Today the CLI's first profile touch is `normalize_args`,
and only when `--llm`, `--cloud-*` or `--foundry` is set.

Root-cause fix. It should not be a per-site patch, because the swallows are pre-existing and many. Add one explicit
early ensure at the process entry points that are not the observer: `cli.py::_main_impl` after the `model`, `doctor`
and similar early subcommand dispatch, and the stable verbs' builders (`api.run` and `create.*`). Use a public
`ensure_user_profiles()`, or call `_ensure_user_profiles()` and record that in the invariant. That restores
"a malformed file stops a normal run before anything reads a profile", keeps import side-effect-free (the observer
property V9 exists for), and leaves `maxim doctor` able to report the broken file, which is an improvement over HEAD.

Pin it with a fresh-process test: a malformed file plus the CLI entry with no `--llm` exits non-zero with the
`ConfigurationError` before any lane is built. Also update `doctor/checks.py`'s profiles-file check docstring, which
still says "the loader raises `ConfigurationError` from module import".

## SHOULD-FIX: docs that still describe import-time loading (wiring claims)

**S2. Stale "import time" claims remain in `src/` and `scripts/`.** These are honesty findings, not behaviour
findings (fail-open for readers). The wire delta round flagged the script docstring in advance
(`reviews/atonia_gate/wire_delta_v6_1.md`, row `benchmark_cross_session.py`).

- `src/maxim/models/language/profile_loader.py` module docstring: "Reads `~/.config/maxim/profiles.yml` at import
  time and merges…" and "Loading is intentionally synchronous + import-time."
- `src/maxim/doctor/checks.py` (profiles-file check docstring): "blocks `maxim` startup (the loader raises
  `ConfigurationError` from module import)". After S1 it should say "at the entry point's ensure"; without S1 it is
  plainly false.
- `scripts/benchmark_cross_session.py::_registered_profile_cloud_flag` docstring: "merges user profiles … into
  `_BUILTIN_PROFILES` at import".
- `docs/plans/atonia_gate_approach_notes.md` §7 table, row 0a, still says rule T bites `load_llm_config`. The status
  block corrects this, but the table row should point at `_maybe_auto_spawn_server` 271 → 259.
- `docs/user/cli-reference.md` ("merge into the bundled profile set at startup") is acceptable from the user's point
  of view if S1 lands. Otherwise it should say "on first use".

**S3. Nothing pins the `TYPE_CHECKING` mirror, and the no-eager-import rule is only partly enforced.** Fail-open,
for types and for the closure.

- The four mirrors match `_LAZY` today (I checked by AST). But `test_step0a_package_init_is_lazy` pins only
  `__all__` against `_LAZY`. A name added to `_LAZY` and `__all__` but not to the mirror makes mypy resolve it
  through the module `__getattr__ -> Any`. The type is silently lost, and `run_mypy_public.sh` stays green. The spec
  requires the mirror ("so mypy keeps real types"), so add an AST equality check, mirror == `_LAZY` (name → module),
  to that test.
- The same test explicitly tolerates eagerly bound names (`eager = {n for n in exported if n in vars(mod) …}`). So
  the brief's "Do NOT add an eager `from maxim.<pkg>.<module> import …` to these inits" is enforced only when the
  eager import happens to load one of the closure test's cut list (`agent_loop`, `llm_worker`, `tool_dispatch`,
  `default_network.*`, `memory.*`). Example: an eager `from maxim.utils.web_cache import WebCache` in
  `utils/__init__` would pull `web_cache`'s closure back into every `maxim.utils.*` import, and nothing would fail.
  Two options:
  - Assert that each init's top-level statements are only `__future__`/`importlib`/`typing` imports, the
    `TYPE_CHECKING` block, `_LAZY`, `__all__` and the two dunder functions.
  - Or narrow the brief's claim to what the guard enforces.

## NIT

- **N1. The `config.py::_user_profiles_lock` comment gives the wrong reason for re-entrancy.** It says "re-entrant:
  the merge itself may resolve a name through an accessor". If that ever happened, the inner call would still see
  `_user_profiles_loaded is False`, re-enter the RLock and run `apply_user_profiles` recursively until it hit a
  `RecursionError`. No re-entry exists today: `profile_loader` imports only `maxim.exceptions` and
  `maxim.utils.paths`. Either use a plain `Lock` and drop the claim, or add a `_merging` flag that makes re-entry
  raise a clear error.
- **N2. `load_llm_config` reads `_BUILTIN_PROFILES` safely only because `_normalize_profile` ensures first.** The
  status block records this coupling, and today it is unconditional: `profile = _normalize_profile(...)`, then
  `_BUILTIN_PROFILES.get(profile)`. A future refactor that short-circuits the normalise would silently skip the merge.
  The "build_model_path (load_llm_config)" reader test would catch it, so this is acceptable. A one-line comment at
  the `_BUILTIN_PROFILES.get` in `load_llm_config` would make it local.
- **N3. The `closed_local` constraint has no carrier yet.** The spec says "`closed_local` never calls
  `_ensure_user_profiles`". The module does not exist, so this is item 1's. But this PR makes `_normalize_profile`,
  and so `normalize_llm_profile`, ensuring. A future `closed_local` that normalises a profile name would parse
  `profiles.yml`. Add a line to the status block saying it is owed by item 1, with its env read-set test, so it is
  not lost.
- **N4. `maxim.models.LLM_MODELS` (module `__getattr__`) has no `TYPE_CHECKING` declaration,** so `from maxim.models
  import LLM_MODELS` types as `Any`. Add `if TYPE_CHECKING: LLM_MODELS: dict[str, dict[str, Any]]`.
  `maxim.models.download.LLM_MODELS` is still a public-looking name, and external code reading it before the first
  ensure misses user profiles. It is not in `stable_api.md`, so documenting `llm_models()` as the way to read it is
  enough.
- **N5. The runtime-tools invariant body carries volatile counts** ("about 830 modules instead of about 130").
  CLAUDE.md asks invariants to avoid volatile counts. Say "the loop and the LLM worker" and leave the numbers to the
  CHANGELOG.
- **N6. The CHANGELOG fragment should name the user-visible upside.** A command that never reads a profile, such as
  `maxim doctor`, no longer fails on a malformed `profiles.yml`. If S1 lands, the fragment can say that a run still
  stops on a malformed file. Also: "130 vs 830" counts all of `sys.modules`; Maxim's own modules go from 236 to 8.
- **N7. `_lane_profile_is_cloud` keeps the old block's `logger.debug(...)` in its `except`** while `logger` may be
  `None`. Moving it verbatim is right under rule T. The latent `AttributeError` predates this PR.
- **N8. `pkgutil.walk_packages` in the sweep skips a subpackage whose `__init__` raises `ImportError`**
  (`onerror=None`). Such a subpackage's modules drop out of the sweep instead of failing it. Pass an `onerror` that
  records the package as a failure.

## Deviation-by-deviation (status block)

| Deviation | Sound? |
|---|---|
| `load_llm_config` not edited; ensures through `_normalize_profile` | Yes. It is unconditional today (N2). |
| `build_model_path` does not ensure | Yes. It reads no table (verified). |
| AST scan instead of CI grep | Yes, and better. It runs in the fast suite, covers `scripts/`, has a non-vacuity probe, and an `ImportFrom` of `LLM_MODELS` from `maxim.models` is also caught. It does not see `getattr(mod, "_BUILTIN_PROFILES")` string reads (none exist). Acceptable. |
| `config.py` may name `LLM_MODELS` | Yes. It is the merge. |
| `maxim.models.LLM_MODELS` via `__getattr__` | Yes (N4 covers the type). |
| `GoalAgent` lazy in `agents/base.py` | Yes. It is the root-cause fix for a cycle the eager init hid. |
| Rule T: `_maybe_auto_spawn_server` 271 → 259 | Yes. The lint is clean and the baseline note is present. |
| Executor `sim_logger` import gated on the pain bridge | Yes. Every consumer of `suppress_nac` already guards on `_tool_pain_bridge is not None`, and the bridge is set only in `__init__`. No behaviour change. |
| Seven red gates flipped; #1479 stays red | Yes. I verified both: the strict xfail stays on #1479's own def. |

## Checked, no finding

- **Accessor coverage.** No `src/` or `scripts/` reader names a private table outside its owner (AST scan, plus my
  own grep, which turned up only docstrings). The readers the spec named all go through accessors: doctor's VRAM
  projections (both), lane resolution (`detect_tiers`, `_pick_infer_profile_with_ctx`, placement), `ensure_available`
  and `api.list_models`. In-module, every function that reads a table ensures first.
- **`run_mypy_public.sh`.** No new file needs listing: the inits only re-export, and the new test files are tests.
  `api.py`, `create.py` and `runtime/executor.py` are on the list and changed, so the author owes one run of
  `scripts/run_mypy_public.sh` (and `scripts/prepr.sh`) before the push.
- **Layering.** `download.llm_models()` imports `config._ensure_user_profiles`, and `config` names
  `download.LLM_MODELS`. That two-way coupling predates this PR, both modules are under `models/`, and the imports are
  deferred.
- **[engineering] tags and Regression guard lines** on both new invariants are valid, and
  `lint_claude_md_invariants.py` passes. The runtime-tools guard's slow-sweep leg only runs once B1 is fixed.
- **conftest.** It pins `XDG_CONFIG_HOME` to an isolation root before collection. The tests that exercise the merge
  call `apply_user_profiles` on throwaway dicts and do not rely on import-time loading.
