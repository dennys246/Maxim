# CLAUDE.md

## Project Overview

Maxim is a bio-inspired cognitive architecture for AI agents. It combines a 5-agent pipeline (Perception, Memory, Exec, Goal, Statistician) with biological memory systems (Hippocampus, ATL, Angular Gyrus, SCN, NAc) and a reactive Default Network. Works headless, in simulation, or connected to a robot.

## When making changes — required checks

Run these before considering any non-trivial task done:

```bash
# Lint + format
ruff check src/ tests/
ruff format src/ tests/

# Tests (fast suite)
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# If touching memory/, decisions/, integration/memory_hub.py:
python -m pytest tests/integration/test_memory_hub.py -q
```

Additional guardrails:
- **Test interactive changes with logging.** When touching interactive mode (display, prompts, stdin reader, orchestrator sim loop), capture a logged session and read the JSONL; check for `ACTION_FOLLOWUP` entries to confirm user responses reach the LLM. Command: [docs/agents/simulation-experiments.md](docs/agents/simulation-experiments.md) §4.
- **No band-aid fixes.** If you spot a bug while working on a task, determine whether the fix addresses the root cause or merely hides the symptom. If it's the latter — a special case, a swallowed exception, a flag that toggles around broken behavior, a fix that would need to be repeated elsewhere — stop, describe the root cause and the scope of the proper fix, and ask the user how to proceed. Never silently choose the smaller fix because it's easier.
- Prefer editing existing modules over creating new ones — this codebase favors many small files already
- Don't rename bio-system classes (Hippocampus, ATL, NAc, SCN, EC, AngularGyrus) — names are load-bearing for the mental model
- If you touch provenance, run a sim with `MAXIM_PROVENANCE_VERBOSITY=2` and eyeball the trace
- **Run `scripts/prepr.sh` before opening or pushing a PR** — CI's lint job + the mypy step, read from test.yml; only exit 0 (`prepr: PASS`) clears a push (details: [docs/agents/ci-testing.md](docs/agents/ci-testing.md) §4; mechanization backlog M66).
- **Run `scripts/run_mypy_public.sh`** after changing any file it lists — the script CI runs (mypy pinned at 1.20.0) and the one place the file list lives.
- **Run `ruff format`** after any changes: `ruff format src/ tests/`
- **Parallel sessions use worktrees, and worktrees live under `.worktrees/`.** When ≥2 Claude sessions run concurrently, each uses `git worktree add .worktrees/<branch-slug> -b <full-branch> origin/main` and absolute paths within it, never a sibling `../Maxim-wt-*`; single-session work stays in the main checkout. Tests there need `export PYTHONPATH="$PWD/src"` (absolute, its own line). `~/.maxim/` is shared across worktrees, so no concurrent sims. Why, and cleanup: [docs/agents/ci-testing.md](docs/agents/ci-testing.md) §4.

## Where the knowledge lives (routing table)

This file is the always-loaded core; everything subsystem-specific lives one hop away:

- **`docs/agents/<subsystem>.md`** — per-subsystem working briefs: mental model, key files, that subsystem's invariants (with their `Regression guard:` lines), gotchas, env vars. **Read the matching brief BEFORE editing in its area.**
- **`docs/lessons/<slug>.md`** — per-incident archives (full narratives, dates, PR numbers, dead ends; index: [docs/lessons/README.md](docs/lessons/README.md)). Follow a stub's "Full history" link only when its trigger fires. The complete pre-split CLAUDE.md is frozen at [docs/lessons/claude-md-2026-08-13-pre-diet.md](docs/lessons/claude-md-2026-08-13-pre-diet.md).

| Touching | Read first |
|---|---|
| `src/maxim/memory/`, `decisions/`, `similarity/`, `integration/memory_hub.py`, `hivemind/`, `time/`, `agents/bus.py` (tiers/valence), substrate encoding | [docs/agents/bio-memory.md](docs/agents/bio-memory.md) |
| `src/maxim/models/language/`, `runtime/lane_*.py`, `runtime/function_router.py`, `runtime/leader_proxy.py`, `runtime/llm_server.py`, `runtime/llm_call_registry.py`, `peer/`, `mesh/`, `tunnel/`, `doctor/`, `utils/http.py` | [docs/agents/llm-routing.md](docs/agents/llm-routing.md) |
| `src/maxim/embodiment/`, `proprioception/`, `bridges/`, `reactions/`, `default_network/`, `embodied_runtime/`, `motion/`, robot YAMLs, **anything commanding Reachy motion** | [docs/agents/embodiment.md](docs/agents/embodiment.md) — hardware-safety section is mandatory before motion code |
| `scripts/benchmark_*`, `scripts/exp*`, `scripts/orient_*`, `simulation/`, `interactive/`, `tests/behavioral/`, `docs/experiments/`, running any sim | [docs/agents/simulation-experiments.md](docs/agents/simulation-experiments.md) |
| `utils/atomic_io.py`, `utils/format_version.py`, `utils/seeding.py`, `utils/paths.py`, `runtime/config_loader.py`, `runtime/config_writer.py`, `runtime/role.py`, any persisted-JSON shape, any frozen dataclass | [docs/agents/persistence-config.md](docs/agents/persistence-config.md) |
| `runtime/agent_loop.py`, `runtime/loop_*.py`, `runtime/substrate_proposal.py`, `runtime/executor.py`, `runtime/bootstrap.py`, `runtime/bio_stack.py`, `runtime/agent_factory.py`, `runtime/agent_pool.py`, `runtime/tool_dispatch.py`, `tools/`, `agents/`, `cli.py`, `api.py` | [docs/agents/runtime-tools.md](docs/agents/runtime-tools.md) |
| `.github/workflows/`, `scripts/lint_*.py`, `scripts/suite_at_commit.sh`, `scripts/pr_merge_readiness.py`, `tests/conftest.py`, pytest/coverage config | [docs/agents/ci-testing.md](docs/agents/ci-testing.md) |

Decision records (why a contract is what it is) live in [DECISIONS.md](DECISIONS.md); the layer-ownership rules the architecture audit enforces live in [ARCHITECTURE.md](ARCHITECTURE.md). Multiple rows match → read all matched briefs. Adding an env var → add it to the owning brief's table (and pair it with a conftest scrub, see the lesson below). Project structure reference: [docs/reference.md](docs/reference.md).

## Lessons learned (bugs that bit us) — cross-cutting core

Subsystem lessons live in the owning brief. These fire in ANY area:

**[engineering] A harness that spawns `maxim` MUST assert the `maxim` its sub-sims import is its OWN repo — `git_hash` answers the wrong question.** Full rule: [docs/agents/simulation-experiments.md](docs/agents/simulation-experiments.md) §1. The cross-cutting half: **a result whose code-under-test cannot be established is not a validation** (the Exp 42b retraction), and the operator hygiene that causes it, `export PYTHONPATH="$PWD/src"` ABSOLUTE and on its OWN line, never chained after a `source` with `&&`. Full history: [docs/lessons/harness-provenance-assert-repo-interpreter.md](docs/lessons/harness-provenance-assert-repo-interpreter.md). Regression guard: [scripts/lint_harness_provenance.py](scripts/lint_harness_provenance.py) in CI.

**[engineering] Weak evidence never gates: a typed abort is not data, and a release waits rather than override.** From now on, a run that ends `planning_failed` or any other typed abort (D22) cannot move a ledger row or a release claim, and until M1 stamps them, a record whose code-under-test, model or context size is only operator-attested must say so and cannot be the sole evidence for a status change. If the only evidence is weak, the claim waits or is dropped. An owner override, if ever taken, is a committed machine-readable exception, and the strict option is the default recommendation when it is offered. (2026-09-27: the Exp 10 re-run gated 1.3.1 on one-turn typed aborts; the blind Codex card graded Research integrity three steps below the Claude card for it.) Regression guard: process invariant — mechanization backlog M1 (provenance stamps in `report.json` + a ledger lint refusing unstamped or aborted records).

**[engineering] Push silent-no-op invariants into types, not helpers.** Count silent failures, not loud ones: one silent-failure miss in a critical path → consider structural enforcement; three silent-failure misses in any path → no longer a question, push the invariant DOWN into the type/constructor signature so forgetting becomes a `TypeError`, not a silent no-op. Canonical example: `build_executor(pain_bus=...)` required keyword-only (see the canonical-builders entry in [docs/agents/runtime-tools.md](docs/agents/runtime-tools.md)). Full history: [docs/lessons/silent-noop-invariants-into-types.md](docs/lessons/silent-noop-invariants-into-types.md). Regression guard: [src/maxim/runtime/bootstrap.py::build_executor](src/maxim/runtime/bootstrap.py) (the required keyword-only `pain_bus=` parameter).

**[engineering] Opt-in env vars in hot startup paths need autouse scrubs.** Any new `if os.environ.get("MAXIM_FOO"): do_side_effect()` branch reachable from `build_primary_router` MUST be paired in the same commit with an `@pytest.fixture(autouse=True)` env-scrub in tests/conftest.py — a leaked var makes the side effect run for real in every later test. Full history: [docs/lessons/env-var-autouse-scrubs.md](docs/lessons/env-var-autouse-scrubs.md). Regression guard: [tests/conftest.py](tests/conftest.py) — autouse env-scrub fixtures pattern; new env-var branches must add a matching scrub in the same commit.

**[engineering] `utils/optional_deps.py` is the canonical optional-dependency surface — do NOT add any new `try: import X except ImportError:` variant (silent pass / non-deduped warning / swallowed return-None) anywhere in `src/maxim/`.** Pick by intent: `require_optional_dependency` (explicitly-requested feature → raises typed `OptionalDependencyError`), `optional_dependency_available` (capability probe → bool, never logs), `warn_optional_fallback` (real fallback exists → ONE deduped WARNING). Add new extras in `EXTRA_FOR_IMPORT`, not at call sites; `OptionalDependencyError` access patterns are exactly `.import_name`/`.extra`/`.fix_hint` — no parallel attributes. Full history: [docs/lessons/optional-deps-canonical-surface.md](docs/lessons/optional-deps-canonical-surface.md). Regression guard: [tests/unit/test_optional_deps.py](tests/unit/test_optional_deps.py).

**[engineering] HTTP call sites must use `maxim/utils/http.py`.** New outbound HTTP calls pick `http.get`/`http.post` (registered endpoint), `http.fetch_url` (arbitrary URL; a MODEL-chosen URL passes `public_only=True`, #824), or `http.download_to_file` (streaming); the `raw_proxy_forward` escape hatch is reserved for `leader_proxy._proxy_request` ONLY — do not use it elsewhere. Full history: [docs/lessons/http-via-utils-http.md](docs/lessons/http-via-utils-http.md). Regression guard: CI grep `grep -r "urllib.request.urlopen" src/maxim/ | grep -v "utils/http.py"` must return zero matches; enforced in [.github/workflows/test.yml](.github/workflows/test.yml).

**[engineering] A green PR may have run NO tests — confirm the expected checks are PRESENT, not just green.** A `CONFLICTING` PR never fires the `pull_request` Tests workflow, and a ruleset-required CodeQL gate is invisible to `gh pr checks`. Run `python scripts/pr_merge_readiness.py <N>` instead of diagnosing by hand (what it reports: the lesson). A reopen fixes neither — resolve the conflict and push. Read a failing check's OWN output before applying a remembered remedy, and never while its sub-jobs are still `in_progress`. Full history: [docs/lessons/green-pr-with-no-tests-run.md](docs/lessons/green-pr-with-no-tests-run.md). Regression guard: [scripts/pr_merge_readiness.py](scripts/pr_merge_readiness.py) + [tests/unit/test_pr_merge_readiness.py](tests/unit/test_pr_merge_readiness.py); the run-it-before-merging half is mechanization backlog M6.

**[engineering] A fix ships with a CALLER, or it has not shipped — and a red gate that does not flip is data.** Before declaring a defect fixed, grep the new symbols across `src/` + `scripts/` excluding tests: zero non-test callers means capability, not a fix. Say where a number was measured; a hand-composed sequence is not the shipped path. Write red gates `xfail(strict=True)`; when one fails to flip, do not remove or re-point it. A defect that lives in a COMPOSITION is fixed in the seam, as one callable thing (D43/#590). Full history: [docs/lessons/shipped-the-pieces-not-the-composition.md](docs/lessons/shipped-the-pieces-not-the-composition.md). Regression guard: [tests/unit/test_d44_merge_behavioural_delta.py](tests/unit/test_d44_merge_behavioural_delta.py) (the strict-red-gate pattern); the caller-grep half is mechanization backlog M3.

**[engineering] Touch it, shrink it: an edit inside a function pinned over 200 lines must shrink it by at least the lines touched** (owner decisions 2026-10-16). Move the code you touched out into a helper, a phase or a module, so the extraction carries the edit. An urgent fix that cannot do that records a `touch_exemptions` entry in `src/maxim/utils/function_length_baseline.json`, matching the lint's printed numbers. Never shrink by deleting comments or docstrings. Full rule: [docs/agents/ci-testing.md](docs/agents/ci-testing.md) §3. Regression guard: [scripts/lint_function_length.py](scripts/lint_function_length.py)::touch_rule + [tests/unit/test_lint_function_length.py](tests/unit/test_lint_function_length.py) (the `test_t_*` tests).

**[engineering] Dead code accumulates silently:** a module no import reaches fails CI. Regression guard: [scripts/lint_orphan_modules.py](scripts/lint_orphan_modules.py) in CI (lint job) + [tests/unit/test_lint_orphan_modules.py](tests/unit/test_lint_orphan_modules.py).

**[engineering] Every sub-plan and every `src/` change gets a pre-merge review round, and the merged diff must be the reviewed diff.** Spawn three parallel reviewers (Executor, Architecture, Wire integrity) and fold their findings into the SAME branch before the PR opens — never merge-then-fix. Wire integrity maps every producer and consumer OUTSIDE the diff of each contract it touches. Scope (incl. `scripts/`), contract types, probes: [docs/CODE_REVIEW.md](docs/CODE_REVIEW.md). A round covers the diff as it was when it ran: new sub-plans, experiments, `src/` changes or claims after it → another round (a docs-only touch-up does not). The value is a DIFFERENT reader, not a more careful one. A round is complete only when its folds are ON THE TARGET: verify with `git show origin/main:<file>`, never squash-merge a PR still receiving folds, and land a silent-failure fix's guard test in the same commit. An empty `gh pr list --state open` means merged. Full history: [docs/lessons/review-round-discipline.md](docs/lessons/review-round-discipline.md). Regression guard: process invariant — mechanization backlog M4 (reviewed-diff vs merged-diff) + M37 (wire-integrity scans).

**[engineering] A NEW experiment design gets a four-lens DESIGN review BEFORE the harness is built** — confounding, bio-faithful, wiring, environment — each returning DO-NOT-BUILD/SHOULD-FIX/NIT into `docs/experiments/rationale/<slug>/<lens>.md` and folded into the prereg, presented to the owner as one plan before building (all four for a new claim; confounding + bio-faithful for a derivative; none for a pure re-run). Pipeline: design review → build → code review → dry-run → run. Charters: [docs/experiments/DESIGN_REVIEW.md](docs/experiments/DESIGN_REVIEW.md). Regression guard: process invariant — mechanization backlog M5.

**[engineering] Faster without lowering the bar (adopted 2026-09-29).** (1) A GATE (provenance stamp, clean flag, ledger/prereg lint, allowance, security boundary) gets an adversarial DESIGN pass on a one-page approach note before code: enumerate every input it consults (config, env, index, refs, filesystem, caller) and how each could lie; when the REVIEWER classifies every finding in a gate's round as fail-closed and in an already-covered class, those go to a follow-up issue and no further round is owed, but any `src/` fold after the last round still gets a (delta-scoped) round. (2) A fold never invalidates a running suite: snapshot (`scripts/suite_at_commit.sh`) and keep working; one full suite per fold batch, and the last on the exact commit pushed. (3) Ask an issue's owner decisions together at its start. (4) While a PR awaits merge, start the next issue in its own worktree. (5) Follow-up rounds get the exact delta and the stop rule. Every check stays: a different reader, deletion probes, lint-then-push, the full suite. Full practices: [docs/lessons/development-flow-speed.md](docs/lessons/development-flow-speed.md). Regression guard: process invariant — mechanization backlog M20–M24.

## Working principles for new mechanisms

These six principles govern HOW new architectural commitments enter the codebase. They are upstream of the invariant surfaces (here and in the briefs) — apply them when *adding* invariants, not just when reading them.

- **Two-tier invariant tracking.** Tag each new invariant `[engineering]` (code breaks loudly without it) or `[behavioral]` (empirically validated via Roy or equivalent as carrying measurable behavioral weight). **New mechanisms enter as `[engineering]` only** and graduate to `[behavioral]` when an experiment earns them. Bio-inspired naming is load-bearing for the mental model but does NOT count as behavioral validation. Graduation tracking lives in [docs/plans/behavioral_graduation_candidates.md](docs/plans/behavioral_graduation_candidates.md); `Stale`/`Broken` entries block the next release.

- **Dormancy over deletion.** When a mechanism fails to earn behavioral weight, mark it `Dormant since <date>: <reason>` in its module docstring rather than deleting. Code stays wired, callers intact. But: no new features build on it, no new invariants accrue, tests beyond regression are not extended. Resurrection requires a new experiment that earns the weight, not "we have time now." This codebase is intimately wired by design — whim-deletion historically caused secondary breakage; dormancy is the middle path between deletion-cascade and monotonic accumulation.

- **Front-gate scope pressure at design time.** Before drafting any implementation plan for a new mechanism (bus, bridge, bio-system, annotation Wire, gating layer, builder), force the question: *"Does this need to be its own mechanism, or can it ride on existing infrastructure?"* If it needs to be its own, name the specific reason in the plan doc's motivation ("existing infrastructure X cannot do this because Y"). If it can ride on existing, choose that path even when less architecturally elegant.

- **Cycle convergence vs divergence.** Distinguish **convergence** (same kind of issue, narrowing each iteration → keep cycling) from **divergence** (new failure modes each iteration). **Two divergence iterations in a row → stop iterating on the mechanism and run a bird's-eye audit:** "what else changed?", "what's the actual independent variable?", "is the mechanism the cause or the messenger?", "have any non-code dependencies moved?", and **"did the action I commanded actually happen?"** **This trigger covers DEBUGGING, not only pre-registered cycles: if two explanations for a bad measurement have died, stop generating a third — audit the layer beneath.** The post-hoc-findings rule and the history: [docs/lessons/review-round-discipline.md](docs/lessons/review-round-discipline.md) ("Cycle convergence vs divergence").

- **Regression-guard / experiment citation per invariant.** Every `[engineering]` invariant — in this file or a brief — ends its body with `Regression guard: <path>`; every `[behavioral]` invariant with `Roy experiment: <path>`. Valid guard references: a test path, a CI grep pattern, or a co-located source file that structurally enforces the rule (typed constructor, frozen dataclass, `@abstractmethod`). **A missing line is a visible coverage gap by design** — surfacing the absence is the discipline's value. Cite `file::symbol`, never `file:line` (audited line numbers all drift; symbols hold), and avoid volatile counts. Enforced by `scripts/lint_claude_md_invariants.py`.

- **Enforced, or on the backlog.** The score cards credit only what a check enforces; a rule followed by attention scores nothing and is where slips land (2026-09-27: the Exp 10 override, a missed "signed bundle" column, a stale version line, a release gate firing inside its own nightly). Every new rule names the check that enforces it; a rule with no check gets a row in the mechanization backlog ([docs/plans/outstanding.md](docs/plans/outstanding.md) §Mechanization backlog) in the same commit, ranked by the score-card axis it moves, and its `Regression guard: process invariant` line cites that row. Mechanizing a row beats adding a rule.

## Architectural invariants — cross-cutting core

Subsystem invariants live in the owning brief (same stub format, same lint). These apply everywhere:

- **[engineering] Tool results flow through the agent bus**; don't call agents directly from tools. Regression guard: convention — mechanization backlog M7; [src/maxim/runtime/executor.py](src/maxim/runtime/executor.py) is the canonical dispatch site.
- **[engineering] Persistence uses `maxim.utils.atomic_io.atomic_write_json`** (fsync + tmp cleanup). Don't hand-roll `open().write()` + `os.replace()`. Regression guard: [src/maxim/utils/atomic_io.py](src/maxim/utils/atomic_io.py) is the canonical writer; [scripts/lint_atomic_io_ratchet.py](scripts/lint_atomic_io_ratchet.py) in CI (lint job) prints the per-file count of hand-rolled atomic-rename call sites every run and fails any branch that raises a file's count. **The number lives in that CI output, not here**; what it counts: [docs/agents/persistence-config.md](docs/agents/persistence-config.md) §4.
- **[engineering] Every `@dataclass(frozen=True)` that persists or crosses a wire MUST declare its forward-compat path in the class docstring before merge** (CC3): (a) escape-hatch — defaults on all fields + `extra: dict = field(default_factory=dict, hash=False, compare=False)`, or (b) a `SHAPE-FROZEN at 1.0 (CC3)` marker with the rejection rationale. The `extra` rules, typed exception hierarchies, scope and class rosters: [docs/agents/persistence-config.md](docs/agents/persistence-config.md) §3. Full history: [docs/lessons/frozen-dataclass-forward-compat.md](docs/lessons/frozen-dataclass-forward-compat.md). Regression guard: CC3 audit list + the `SHAPE-FROZEN at 1.0 (CC3)` docstring marker on each frozen-without-extra dataclass; new frozen dataclasses must pick path (a) or (b) before merge.
- **[engineering] Every persisted JSON file carries `"_format_version": "1.0"` at root.** Writers wrap via `with_format_version(payload)` + `atomic_write_json`; loaders call `check_format_version(data, "<file_type>", log=logger)`. Its coexistence with the envelope `schema_version` and the tombstoned legacy `"version"` strings: [docs/agents/persistence-config.md](docs/agents/persistence-config.md) §3. Full history: [docs/lessons/format-version-contract.md](docs/lessons/format-version-contract.md). Regression guard: [tests/integration/test_persistence_compat.py](tests/integration/test_persistence_compat.py).
- **[engineering] LLM access goes through `models/language/router.py`; concrete backends are not imported outside `models/language/`.** Sanctioned exceptions (the `_MaximPeerBackend` probe surface, `bench/recovery_time.py`) and how to add a backend type: [docs/lessons/llm-router-only-access.md](docs/lessons/llm-router-only-access.md). Regression guard: [src/maxim/runtime/lane_backends.py::BACKEND_CLASSES](src/maxim/runtime/lane_backends.py) (single dispatch table) + CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" step) blocking backend imports outside `models/language/`.
- **[engineering] No NEW silent exception swallows** — never add a bare `except Exception: pass`; narrow the exception type, or handle-and-log. A handler that assigns or returns a fallback without reporting is the same swallow. On the measurement path, handle-and-log means the Stage-1 form (`log_swallowed_exception()` or `site=`); the explicit `(e, operation=...)` form does NOT count. Existing sites are grandfathered at their per-FUNCTION count; a verbatim move is free, an edited move needs a `scripts/swallow_moves.json` record. The deleted `@resilient` decorator must not be cited or re-introduced. Checks, move credit and blind spots: [docs/agents/ci-testing.md](docs/agents/ci-testing.md) §4. Regression guard: [scripts/lint_no_silent_swallows.py](scripts/lint_no_silent_swallows.py) in CI, which prints the repo-wide total every run (the count lives in that output) + [tests/unit/test_lint_no_silent_swallows.py](tests/unit/test_lint_no_silent_swallows.py).
- **[engineering] Values that cross a persistence boundary MUST be hashed with `utils/seeding.py::stable_hash_32` / `stable_hash_64_signed`, never builtin `hash()`** (PYTHONHASHSEED randomization makes persisted hashes permanently unmatchable across processes; a seed PARAMETER routed through `hash()` only looks deterministic). The SIGNED variant and the `hash_scheme` stamp: [docs/agents/persistence-config.md](docs/agents/persistence-config.md) §3. A same-process test passes over this entire bug class — the guard MUST be two-process with differing PYTHONHASHSEED. Full history: [docs/lessons/stable-hash-persistence.md](docs/lessons/stable-hash-persistence.md). Regression guard: [tests/unit/test_stable_hash_two_process.py](tests/unit/test_stable_hash_two_process.py) (verified to fail 5/5 against the pre-fix code).
- **[engineering] `main` is ahead of PyPI: the version bump happens in the RELEASE TRANSACTION, not on the change that earns it.** Between releases `pyproject.toml` + `src/maxim/__init__.py` carry the last published version and `CHANGELOG.md` accumulates under `## [Unreleased]`. The three living sync lines (this file, `docs/plans/README.md`, `docs/index.md`) NAME the `pyproject` version and LINK PyPI — they never describe what PyPI serves. Full procedure: [docs/publication_guide.md](docs/publication_guide.md); the full rule: [docs/lessons/versioning-main-ahead-of-pypi.md](docs/lessons/versioning-main-ahead-of-pypi.md). Regression guard: [scripts/lint_version_sync.py](scripts/lint_version_sync.py) + [scripts/lint_unreleased_on_src_change.py](scripts/lint_unreleased_on_src_change.py) in CI (lint job); [tests/unit/test_lint_version_sync.py](tests/unit/test_lint_version_sync.py) + [tests/unit/test_lint_unreleased_on_src_change.py](tests/unit/test_lint_unreleased_on_src_change.py).
- **[engineering] Removed/renamed identifiers stay removed:** the class is `NAc`, never `NucleusAccumbens`; lane tiers are `"large"`/`"medium"`/`"small"`, never `"infer"`/`"review"`/`"record"`; `EnergyReactionBridge`/`MovementEnergyTracker` are deleted; the probe shims `probe_llm_server`/`llm_server_responding_at` are removed. Do not re-introduce any of them; grep after touching adjacent code. Regression guard: CI greps in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" + "deprecated probe shims" steps; the old tier literals are grepped in [src/maxim/runtime/lane_models.py](src/maxim/runtime/lane_models.py)).
- **[engineering] Reachy motion has TWO hard rules: head pose is WORLD-frame (a body turn with `head=None` COUNTER-ROTATES the head, so head-mounted sensors do not turn with it), and `ReachyMiniController.goto_target` is the single clamped+locked dispatch point — motors 2+3 were destroyed by an unclamped pose.** Every controller dispatch also starts from a fresh daemon readback (never a cache or a guessed 0) and samples the SDK's own path, splitting or refusing a move that would sweep the neck mid-way (#1499). Read [docs/agents/embodiment.md](docs/agents/embodiment.md) §1 before commanding any motion. Not robot-specific: **when a measurement disagrees with the model, verify the ACTUATION assumption — did the thing you are sensing with actually move? — BEFORE theorizing about the sensor**, and read the vendor's docs before reverse-engineering their kinematics. Regression guard: [tests/unit/test_reachy_head_frame.py](tests/unit/test_reachy_head_frame.py) + [tests/unit/test_reachy_workspace_safety.py](tests/unit/test_reachy_workspace_safety.py) + [tests/unit/test_reachy_retained_axes.py](tests/unit/test_reachy_retained_axes.py) + [tests/unit/test_motion_path_1499.py](tests/unit/test_motion_path_1499.py).

## Running simulations — keep them small

Simulations call a live LLM for every turn and burn cost + time. Full sim discipline: [docs/agents/simulation-experiments.md](docs/agents/simulation-experiments.md) §4. The three session-killing rules:

- **IMPORTANT: Use `--interactive false` when running sims from Claude Code or scripts.** Interactive mode is ON by default in CLI with a TTY; the raw terminal reader conflicts with non-human stdin.
- **Configure model + n_ctx via `maxim config`, not transient env/flags — single source of truth.** `maxim config set llm.profile <profile>` + `maxim config set llm.n_ctx <N>`, then verify with `maxim doctor 2>/dev/null | grep -i "n_ctx\|profile"`. The server's spawn n_ctx and the PromptBudgeter's belief resolve through DIFFERENT paths; if they drift, the sim silently takes 0 real actions behind HTTP 500s. Full three-leg bug history: [docs/lessons/sim-n-ctx-config-drift.md](docs/lessons/sim-n-ctx-config-drift.md).
- **Never co-locate a `maxim-leader`/experiment run with the sim on one box** — a second consumer of the :8100 server causes 500s under contention; the harness belongs on a different machine (see [docs/lessons/no-harness-on-leader-machine.md](docs/lessons/no-harness-on-leader-machine.md)).
- Set a narrow `--goal`, cap duration, prefer `--sandbox tmpdir`, watch `Cost:` in the report — details in the brief's §4.

## `maxim doctor` — environment diagnostics

`maxim doctor` (`--retry`, `--json`, `--as peer <url>|leader|solo`) and `maxim peer test`, plus the companion `maxim tunnel`: [docs/agents/llm-routing.md](docs/agents/llm-routing.md).

## Key Commands

```bash
maxim                                        # Rich menu: campaigns, chat, doctor, help
maxim --llm claude-sonnet                    # Claude (needs ANTHROPIC_API_KEY); --list-models for local ones
maxim --sim "test memory recall" --interactive false  # non-interactive sim (Claude Code, CI, scripts)
```

Full CLI reference, including every command this section used to list: [docs/user/cli-reference.md](docs/user/cli-reference.md)

## Remote Update Workflow

`maxim peer update && maxim peer restart` (`--dev` for git-checkout leaders, `--dry-run` first if unsure). Every one of these, dry runs included, needs the leader's opt-in (`maxim config set proxy.allow_remote_update true`, off by default since 2026-10-14). Flags and troubleshooting: [docs/troubleshooting/remote_update.md](docs/troubleshooting/remote_update.md).

**Important for Claude agents:** `maxim peer update --dry-run`, `maxim peer version`, `maxim peer logs`, `maxim peer llm --status`, and `maxim peer deps` are safe and read-only. `maxim peer update`, `maxim peer restart`, `maxim peer llm <model>`, and `maxim peer install <extras>` modify leader state — only run when explicitly asked by the user.

## Versioning

Policy and enforcement: the `main`-is-ahead-of-PyPI invariant above; a `src/` change does not bump on the spot (every `src/maxim/*.py` change must add an `[Unreleased]` line; `lint_unreleased_on_src_change.py` fails otherwise). Section text before the 2026-10-16 diet: [docs/lessons/versioning-main-ahead-of-pypi.md](docs/lessons/versioning-main-ahead-of-pypi.md).

## Environment Variables — session-critical core

Full per-subsystem tables (with the experiment/ablation toggles) live in the owning `docs/agents/` brief. The canonical truthy parser for MAXIM_* toggles is `cluster_bias_annotation.annotation_disabled_via_env` ("1"/"true"/"yes"/"on", case-insensitive).

```bash
ANTHROPIC_API_KEY          # Claude backend (7 more provider keys: see docs/agents/llm-routing.md)
MAXIM_ROLE=leader          # Explicit role: leader|peer|solo (exported at startup; downstream reads env)
MAXIM_LOG_FILE=/tmp/maxim.jsonl  # JSONL file handler; stdout stays human-readable
MAXIM_BACKEND_TRACE=1      # Per-call peer-backend JSONL (pair with MAXIM_LOG_FILE)
MAXIM_PROVENANCE_VERBOSITY=1     # Decision log at ~/.maxim/util/lane_decisions.jsonl (0/1/2)
MAXIM_SKIP_REMOTE_PROBE=1  # Bypass remote-URL probe — CI escape hatch
```

## Testing

**Run narrow first, then wide.** Test the specific module you changed (`python -m pytest tests/unit/test_<module>.py -v`) before the full suite (15,427 tests, ~38 min serial under coverage in CI, 2026-10-16; timings + pre-push flow: [docs/agents/ci-testing.md](docs/agents/ci-testing.md)). **Kill stale sims before running tests** (`pkill -f "maxim.*sim" 2>/dev/null; sleep 2`) — a running sim holds GPU + port resources and causes hangs. Use `threading.RLock`, not `Lock`, when a locked method calls another locked method (re-entry deadlocks; a "hanging" thread test is usually deadlocked). **Don't run sims from tests** — sims call real LLMs; tests mock them. Peer/tunnel guides: [docs/troubleshooting/](docs/troubleshooting/).

## Simulation Reports

Sim runs save to `~/.maxim/sim_reports/{session_id}/` and `maxim.create.agent()` homes to `~/.maxim/agents/{name}/` (both resolved by `utils/paths.py::resolve_run_dir`); contents: [docs/agents/simulation-experiments.md](docs/agents/simulation-experiments.md) §4.

## Python API (pymaxim)

Published to PyPI as `pymaxim` (import name `maxim`); maintenance rules: [docs/agents/runtime-tools.md](docs/agents/runtime-tools.md); build validation before any publish: [docs/publication_guide.md](docs/publication_guide.md).

## Active initiatives

Current version: **1.3.3** (`pyproject.toml` + `src/maxim/__init__.py`; PyPI: https://pypi.org/project/pymaxim/ — `main` is ahead of PyPI by policy, §Versioning). This line is checked by `scripts/lint_version_sync.py` — bump it in the release transaction. Active theme **1.4** (working title "Anticipation"; [roadmap_1_4.md](docs/plans/roadmap_1_4.md)). What each release shipped and claims, the roadmap index and the deferred plans: [docs/plans/README.md](docs/plans/README.md); this section's text before the 2026-10-16 diet: [docs/lessons/active-initiatives-history-2026.md](docs/lessons/active-initiatives-history-2026.md).

Since B1 (2026-10-14/15) T1-13/14/15 (Exp 60/61/62) are PARTIAL, floor matched under BF2 ([baseline_floors.md](docs/plans/baseline_floors.md)).
