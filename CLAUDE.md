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
- **Test interactive changes with logging.** When touching interactive mode (display, prompts, stdin reader, orchestrator sim loop), capture a session with `MAXIM_LOG_FILE=/tmp/maxim.jsonl maxim --sim "test basic recall" --interactive --sim-max-turns 3` and read the JSONL to verify percepts, tool calls, and followups flow correctly. Check for `ACTION_FOLLOWUP` entries to confirm user responses reach the LLM. Use `MAXIM_BACKEND_TRACE=1` for per-call token/latency data.
- **No band-aid fixes.** If you spot a bug while working on a task, determine whether the fix addresses the root cause or merely hides the symptom. If it's the latter — a special case, a swallowed exception, a flag that toggles around broken behavior, a fix that would need to be repeated elsewhere — stop, describe the root cause and the scope of the proper fix, and ask the user how to proceed. Never silently choose the smaller fix because it's easier.
- Prefer editing existing modules over creating new ones — this codebase favors many small files already
- Don't rename bio-system classes (Hippocampus, ATL, NAc, SCN, EC, AngularGyrus) — names are load-bearing for the mental model
- If you touch provenance, run a sim with `MAXIM_PROVENANCE_VERBOSITY=2` and eyeball the trace
- **Run `mypy` on public API files + hivemind/ + the composition layer** after changing api.py, session.py, create.py, load.py, __init__.py, `runtime/executor.py`, `runtime/agent_loop.py`, `runtime/loop_*.py`, `runtime/substrate_proposal.py`, or anything in `src/maxim/hivemind/`, `bridges/` or `planning/`: `mypy src/maxim/__init__.py src/maxim/api.py src/maxim/session.py src/maxim/create.py src/maxim/load.py src/maxim/hivemind/ src/maxim/runtime/executor.py src/maxim/runtime/agent_loop.py src/maxim/runtime/loop_setup.py src/maxim/runtime/loop_gates.py src/maxim/runtime/loop_controller.py src/maxim/runtime/loop_state.py src/maxim/runtime/tool_dispatch.py src/maxim/runtime/loop_types.py src/maxim/runtime/loop_substrate.py src/maxim/runtime/substrate_proposal.py src/maxim/bridges/ src/maxim/planning/ --ignore-missing-imports --follow-imports=silent --warn-unused-ignores` (same invocation CI runs — `--follow-imports=silent` scopes errors to the checked files; the rest of the codebase is not mypy-clean yet; hivemind/ added per 1.2 gate 8 — the bundle format is a wire boundary; the composition layer per 1.3.2 item 4 — the #840/#841 silent-seam class)
- **Run `ruff format`** after any changes: `ruff format src/ tests/`
- **Parallel sessions use worktrees, and worktrees live under `.worktrees/`.** When ≥2 Claude sessions run concurrently on independent work, each uses its own git worktree — `git worktree add .worktrees/<branch-slug> -b <full-branch> origin/main` — and works entirely in absolute paths within it. Never a sibling directory (`../Maxim-wt-*`): `.worktrees/` is gitignored (so nested checkouts are invisible to `git status` and to ruff, which honors `.gitignore`), `pytest` is pinned out of it (`testpaths` + `norecursedirs` in `pyproject.toml`), and a nested tree sits inside a Claude session's primary working directory, so no extra permission scope is needed. Single-session work stays in the main checkout. Tests in worktrees need `export PYTHONPATH="$PWD/src"` (absolute, its own line — the installed editable package shadows the worktree otherwise). `git worktree remove .worktrees/<slug>` when the PR merges; `git worktree prune` for stragglers. Note `~/.maxim/` is shared across worktrees — don't run sims from concurrent doc/code sessions or they'll collide on persisted state.

## Where the knowledge lives (routing table)

This file is the always-loaded core: commands, checks, hard safety rules, cross-cutting invariants, and this table. Everything subsystem-specific lives one hop away:

- **`docs/agents/<subsystem>.md`** — per-subsystem working briefs: mental model, key files, that subsystem's invariants (with their `Regression guard:` lines), gotchas, env vars. **Read the matching brief BEFORE editing in its area.**
- **`docs/lessons/<slug>.md`** — per-incident archives (full narratives, dates, PR numbers, dead ends). Follow a stub's "Full history" link only when its trigger fires. The complete pre-split CLAUDE.md is frozen at [docs/lessons/claude-md-2026-08-13-pre-diet.md](docs/lessons/claude-md-2026-08-13-pre-diet.md).

| Touching | Read first |
|---|---|
| `src/maxim/memory/`, `decisions/`, `similarity/`, `integration/memory_hub.py`, `hivemind/`, `time/`, `agents/bus.py` (tiers/valence), substrate encoding | [docs/agents/bio-memory.md](docs/agents/bio-memory.md) |
| `src/maxim/models/language/`, `runtime/lane_*.py`, `runtime/function_router.py`, `runtime/leader_proxy.py`, `runtime/llm_server.py`, `runtime/llm_call_registry.py`, `peer/`, `mesh/`, `tunnel/`, `doctor/`, `utils/http.py` | [docs/agents/llm-routing.md](docs/agents/llm-routing.md) |
| `src/maxim/embodiment/`, `proprioception/`, `bridges/`, `reactions/`, `default_network/`, `embodied_runtime/`, `motion/`, robot YAMLs, **anything commanding Reachy motion** | [docs/agents/embodiment.md](docs/agents/embodiment.md) — hardware-safety section is mandatory before motion code |
| `scripts/benchmark_*`, `scripts/exp*`, `scripts/orient_*`, `simulation/`, `interactive/`, `tests/behavioral/`, `docs/experiments/`, running any sim | [docs/agents/simulation-experiments.md](docs/agents/simulation-experiments.md) |
| `utils/atomic_io.py`, `utils/format_version.py`, `utils/seeding.py`, `utils/paths.py`, `runtime/config_loader.py`, `runtime/config_writer.py`, `runtime/role.py`, any persisted-JSON shape, any frozen dataclass | [docs/agents/persistence-config.md](docs/agents/persistence-config.md) |
| `runtime/agent_loop.py`, `runtime/loop_*.py`, `runtime/substrate_proposal.py`, `runtime/executor.py`, `runtime/bootstrap.py`, `runtime/bio_stack.py`, `runtime/agent_factory.py`, `runtime/agent_pool.py`, `runtime/tool_dispatch.py`, `tools/`, `agents/`, `cli.py`, `api.py` | [docs/agents/runtime-tools.md](docs/agents/runtime-tools.md) |

Decision records (why a contract is what it is) live in [DECISIONS.md](DECISIONS.md); the layer-ownership rules the architecture audit enforces live in [ARCHITECTURE.md](ARCHITECTURE.md). Multiple rows match → read all matched briefs. Adding an env var → add it to the owning brief's table (and pair it with a conftest scrub, see the lesson below). Project structure reference: [docs/reference.md](docs/reference.md).

## Lessons learned (bugs that bit us) — cross-cutting core

Subsystem-specific lessons live in the owning `docs/agents/` brief; full narratives in `docs/lessons/`. These fire in ANY area:

**[engineering] A harness that spawns `maxim` MUST assert the `maxim` its sub-sims import is its OWN repo — `git_hash` answers the wrong question** (it describes where the harness *lives*, not what the sub-sims *imported*). Full rule, the six spawning harnesses and the `# provenance-exempt:` escape hatch: [docs/agents/simulation-experiments.md](docs/agents/simulation-experiments.md) §1, which the routing table already makes mandatory before touching `scripts/exp*` or `docs/experiments/`. The cross-cutting half, which is why this stays here: **a result whose code-under-test cannot be established is not a validation** — do not argue it was probably fine (the Exp 42b retraction) — and the operator hygiene that causes it, `export PYTHONPATH="$PWD/src"` ABSOLUTE and on its OWN line, never chained after a `source` with `&&`. Full history: [docs/lessons/harness-provenance-assert-repo-interpreter.md](docs/lessons/harness-provenance-assert-repo-interpreter.md). Regression guard: [scripts/lint_harness_provenance.py](scripts/lint_harness_provenance.py) in CI.

**[engineering] Weak evidence never gates: a typed abort is not data, and a release waits rather than override.** From now on, a run that ends `planning_failed` or any other typed abort (D22) cannot move a ledger row or a release claim, and until M1 stamps them, a record whose code-under-test, model or context size is only operator-attested must say so and cannot be the sole evidence for a status change. If the only evidence is weak, the claim waits or is dropped. An owner override, if ever taken, is a committed machine-readable exception, and the strict option is the default recommendation when it is offered. (2026-09-27: the Exp 10 re-run gated 1.3.1 on one-turn typed aborts; the blind Codex card graded Research integrity three steps below the Claude card for it.) Regression guard: process invariant — mechanization backlog M1 (provenance stamps in `report.json` + a ledger lint refusing unstamped or aborted records).

**[engineering] Push silent-no-op invariants into types, not helpers.** Count silent failures, not loud ones: one silent-failure miss in a critical path → consider structural enforcement; three silent-failure misses in any path → no longer a question, push the invariant DOWN into the type/constructor signature so forgetting becomes a `TypeError`, not a silent no-op. Canonical example: `build_executor(pain_bus=...)` required keyword-only (see the canonical-builders entry in [docs/agents/runtime-tools.md](docs/agents/runtime-tools.md)). Full history: [docs/lessons/silent-noop-invariants-into-types.md](docs/lessons/silent-noop-invariants-into-types.md). Regression guard: [src/maxim/runtime/bootstrap.py::build_executor](src/maxim/runtime/bootstrap.py) — required keyword-only `pain_bus=` parameter is the canonical example; signature enforces the rule structurally so forgetting becomes a `TypeError`, not a silent no-op.

**[engineering] Opt-in env vars in hot startup paths need autouse scrubs.** Any new `if os.environ.get("MAXIM_FOO"): do_side_effect()` branch reachable from `build_primary_router` MUST be paired in the same commit with an `@pytest.fixture(autouse=True)` env-scrub in tests/conftest.py — a leaked var makes the side effect run for real in every later test (P5: 9-minute pytest hang on a real 1 GB GGUF download). Full history: [docs/lessons/env-var-autouse-scrubs.md](docs/lessons/env-var-autouse-scrubs.md). Regression guard: [tests/conftest.py](tests/conftest.py) — autouse env-scrub fixtures pattern; new env-var branches must add a matching scrub in the same commit.

**[engineering] `utils/optional_deps.py` is the canonical optional-dependency surface — do NOT add any new `try: import X except ImportError:` variant (silent pass / non-deduped warning / swallowed return-None) anywhere in `src/maxim/`.** Pick by intent: `require_optional_dependency` (explicitly-requested feature → raises typed `OptionalDependencyError`), `optional_dependency_available` (capability probe → bool, never logs), `warn_optional_fallback` (real fallback exists → ONE deduped WARNING). Add new extras in `EXTRA_FOR_IMPORT`, not at call sites; `OptionalDependencyError` access patterns are exactly `.import_name`/`.extra`/`.fix_hint` — no parallel attributes. Full history: [docs/lessons/optional-deps-canonical-surface.md](docs/lessons/optional-deps-canonical-surface.md). Regression guard: [tests/unit/test_optional_deps.py](tests/unit/test_optional_deps.py) — covers `require_optional_dependency` raise/return, `optional_dependency_available` bool, `warn_optional_fallback` dedup, `OptionalDependencyError` subclass shape, and LLM-router reraise behaviour.

**[engineering] HTTP call sites must use `maxim/utils/http.py`.** New outbound HTTP calls pick `http.get`/`http.post` (registered endpoint), `http.fetch_url` (arbitrary URL; a MODEL-chosen URL passes `public_only=True`, #824), or `http.download_to_file` (streaming); the `raw_proxy_forward` escape hatch is reserved for `leader_proxy._proxy_request` ONLY — do not use it elsewhere. (Origin: the 2026-04-12 Cloudflare Bot Fight Mode missing-User-Agent incident.) Full history: [docs/lessons/http-via-utils-http.md](docs/lessons/http-via-utils-http.md). Regression guard: CI grep `grep -r "urllib.request.urlopen" src/maxim/ | grep -v "utils/http.py"` must return zero matches; enforced in [.github/workflows/test.yml](.github/workflows/test.yml).

**[engineering] A green PR may have run NO tests — confirm the expected checks are PRESENT, not just green.** A `CONFLICTING` PR never fires the `pull_request` Tests workflow, and a ruleset-required CodeQL gate is invisible to `gh pr checks`. Run `python scripts/pr_merge_readiness.py <N>` instead of diagnosing by hand: it reports every gating surface (merge state; the required `unit-tests` + `lint` + `release-build` present; each failing check's own output; code-scanning alerts; ruleset gates), reports an unreadable surface as UNVERIFIED and never asserts an absence while anything is in flight. A reopen fixes neither — resolve the conflict and push. Read a failing check's OWN output before applying a remembered remedy, and never while its sub-jobs are still `in_progress`. Full history: [docs/lessons/green-pr-with-no-tests-run.md](docs/lessons/green-pr-with-no-tests-run.md). Regression guard: [scripts/pr_merge_readiness.py](scripts/pr_merge_readiness.py) + [tests/unit/test_pr_merge_readiness.py](tests/unit/test_pr_merge_readiness.py); the run-it-before-merging half is mechanization backlog M6.

**[engineering] A fix ships with a CALLER, or it has not shipped — and a red gate that does not flip is data.** Before declaring a defect fixed, grep the new symbols across `src/` + `scripts/` excluding tests: zero non-test callers means capability, not a fix. Say where a number was measured; a hand-composed sequence is not the shipped path. Write red gates `xfail(strict=True)`; when one fails to flip, do not remove or re-point it. A defect that lives in a COMPOSITION is fixed in the seam, as one callable thing (D43/#590). Full history: [docs/lessons/shipped-the-pieces-not-the-composition.md](docs/lessons/shipped-the-pieces-not-the-composition.md). Regression guard: [tests/unit/test_d44_merge_behavioural_delta.py](tests/unit/test_d44_merge_behavioural_delta.py) (the strict-red-gate pattern); the caller-grep half is mechanization backlog M3.

**[engineering] Dead code accumulates silently** (15 dead modules, ~8,500 LOC, once shipped in the wheel): a module no import reaches fails CI. Regression guard: [scripts/lint_orphan_modules.py](scripts/lint_orphan_modules.py) in CI (lint job) + [tests/unit/test_lint_orphan_modules.py](tests/unit/test_lint_orphan_modules.py).

**[engineering] Every sub-plan and every `src/` change gets a pre-merge review round, and the merged diff must be the reviewed diff.** Spawn three parallel reviewers (Executor, Architecture, Wire integrity) and fold their findings into the SAME branch before the PR opens — never merge-then-fix. Wire integrity maps every producer and consumer OUTSIDE the diff of each contract it touches: in the 2026-10-05 history audit, all eight sampled breaking changes with a recorded two-lens round missed the de-wiring in their scope (a sample, not a catch rate). Scope (incl. `scripts/`), contract types, probes: [docs/CODE_REVIEW.md](docs/CODE_REVIEW.md). A round covers the diff as it was when it ran: new sub-plans, experiments, `src/` changes or claims after it → another round (a docs-only touch-up does not). The value is a DIFFERENT reader, not a more careful one. A round is complete only when its folds are ON THE TARGET: verify with `git show origin/main:<file>`, never squash-merge a PR still receiving folds, and land a silent-failure fix's guard test in the same commit. An empty `gh pr list --state open` means merged. Full history: [docs/lessons/review-round-discipline.md](docs/lessons/review-round-discipline.md). Regression guard: process invariant — mechanization backlog M4 (reviewed-diff vs merged-diff) + M37 (wire-integrity scans).

**[engineering] A NEW experiment design gets a four-lens DESIGN review BEFORE the harness is built** — confounding, bio-faithful, wiring, environment — each returning DO-NOT-BUILD/SHOULD-FIX/NIT into `docs/experiments/rationale/<slug>/<lens>.md` and folded into the prereg, presented to the owner as one plan before building (the design review reads the PREREG, the code review the harness; all four for a new claim; confounding + bio-faithful for a derivative; none for a pure re-run). Pipeline: design review → build → code review → dry-run → run. Charters: [docs/experiments/DESIGN_REVIEW.md](docs/experiments/DESIGN_REVIEW.md). Regression guard: process invariant — mechanization backlog M5.

**[engineering] Faster without lowering the bar (adopted 2026-09-29).** (1) A GATE (provenance stamp, clean flag, ledger/prereg lint, allowance, security boundary) gets an adversarial DESIGN pass on a one-page approach note before code: enumerate every input it consults (config, env, index, refs, filesystem, caller) and how each could lie. When the REVIEWER classifies every finding in a gate's round as fail-closed and in an already-covered class, those go to a follow-up issue, not onto the branch, and no further round is owed; any `src/` fold after the last round still gets a (delta-scoped) round. (2) A fold never invalidates a running suite: snapshot (`scripts/suite_at_commit.sh`, on a commit or `git stash create`) and keep working; one full suite per fold batch, and the last on the exact commit pushed. (3) Owner decisions an issue needs are asked together at its start. (4) While a PR awaits merge, start the next issue in its own worktree. (5) Follow-up review rounds get the exact delta and the stop rule. Every check stays: a different reader, deletion probes, lint-then-push, the full suite. Full practices: [docs/lessons/development-flow-speed.md](docs/lessons/development-flow-speed.md). Regression guard: process invariant — mechanization backlog M20–M24.

## Working principles for new mechanisms

These six principles govern HOW new architectural commitments enter the codebase. They are upstream of the invariant surfaces (here and in the briefs) — apply them when *adding* invariants, not just when reading them.

- **Two-tier invariant tracking.** Tag each new invariant `[engineering]` (code breaks loudly without it) or `[behavioral]` (empirically validated via Roy or equivalent as carrying measurable behavioral weight). **New mechanisms enter as `[engineering]` only** and graduate to `[behavioral]` when an experiment earns them. Bio-inspired naming is load-bearing for the mental model but does NOT count as behavioral validation. Graduation tracking lives in [docs/plans/behavioral_graduation_candidates.md](docs/plans/behavioral_graduation_candidates.md) — a 1.0 gate AND the ongoing post-1.0 regression discipline; Earned entries carry **Re-run on:** triggers + **Regression guard:** experiment paths; `Stale`/`Broken` entries block the next release.

- **Dormancy over deletion.** When a mechanism fails to earn behavioral weight, mark it `Dormant since <date>: <reason>` in its module docstring rather than deleting. Code stays wired, callers intact. But: no new features build on it, no new invariants accrue, tests beyond regression are not extended. Resurrection requires a new experiment that earns the weight, not "we have time now." This codebase is intimately wired by design — whim-deletion historically caused secondary breakage; dormancy is the middle path between deletion-cascade and monotonic accumulation.

- **Front-gate scope pressure at design time.** Before drafting any implementation plan for a new mechanism (bus, bridge, bio-system, annotation Wire, gating layer, builder), force the question: *"Does this need to be its own mechanism, or can it ride on existing infrastructure?"* If it needs to be its own, name the specific reason in the plan doc's motivation ("existing infrastructure X cannot do this because Y"). If it can ride on existing, choose that path even when less architecturally elegant.

- **Cycle convergence vs divergence.** Cyclic experiment findings are signal — but distinguish **convergence** (same kind of issue, narrowing each iteration → keep cycling) from **divergence** (new failure modes each iteration → the mechanism is getting more complex faster than stabilizing). **Two divergence iterations in a row → stop iterating on the mechanism and run a bird's-eye audit:** "what else changed?", "what's the actual independent variable?", "is the mechanism the cause or the messenger?", "have any non-code dependencies moved (encoder model, library version, env var defaults)?", and — added after it was blown through three times in one hardware session — **"did the action I commanded actually happen?"** A wrong actuation assumption is indistinguishable from a broken sensor and manufactures unlimited plausible sensor theories. **This trigger covers DEBUGGING, not only pre-registered cycles: if two explanations for a bad measurement have died, stop generating a third — audit the layer beneath.** Post-hoc findings (from post-result investigation rather than planned measurements) don't directly count as divergence — they spawn new pre-registered iterations; the trigger then watches those. Sharpened form when post-hoc findings are present: two iterations in a row whose primary criterion fails AND whose post-hoc findings each spawn new follow-up plans. Full history incl. the Roy-3c bisect and the 2026-07-16 six-hypothesis actuation incident: [docs/lessons/review-round-discipline.md](docs/lessons/review-round-discipline.md) + [docs/lessons/reachy-head-world-frame.md](docs/lessons/reachy-head-world-frame.md).

- **Regression-guard / experiment citation per invariant.** Every `[engineering]` invariant — in this file or a brief — ends its body with `Regression guard: <path>`; every `[behavioral]` invariant with `Roy experiment: <path>`. Valid guard references: a test path, a CI grep pattern, or a co-located source file that structurally enforces the rule (typed constructor, frozen dataclass, `@abstractmethod`). **A missing line is a visible coverage gap by design** — surfacing the absence is the discipline's value. Cite `file::symbol`, never `file:line` (audited line numbers all drift; symbols hold), and avoid volatile counts. CI enforcement: `scripts/lint_claude_md_invariants.py` audits this file AND `docs/agents/*.md`, existence-checks lesson links, and holds this file under its token ceiling.

- **Enforced, or on the backlog.** The score cards credit only what a check enforces; a rule followed by attention scores nothing and is where slips land (2026-09-27: the Exp 10 override, a missed "signed bundle" column, a stale version line, a release gate firing inside its own nightly). Every new rule names the check that enforces it; a rule with no check gets a row in the mechanization backlog ([docs/plans/outstanding.md](docs/plans/outstanding.md) §Mechanization backlog) in the same commit, ranked by the score-card axis it moves, and its `Regression guard: process invariant` line cites that row. Mechanizing a row beats adding a rule.

## Architectural invariants — cross-cutting core

Subsystem invariants live in the owning `docs/agents/` brief (same stub format, same lint). These apply everywhere:

- **[engineering] Tool results flow through the agent bus**; don't call agents directly from tools. Regression guard: convention — mechanization backlog M7; [src/maxim/runtime/executor.py](src/maxim/runtime/executor.py) is the canonical dispatch site.
- **[engineering] Persistence uses `maxim.utils.atomic_io.atomic_write_json`** (fsync + tmp cleanup). Don't hand-roll `open().write()` + `os.replace()`. Regression guard: [src/maxim/utils/atomic_io.py](src/maxim/utils/atomic_io.py) is the canonical writer; [scripts/lint_atomic_io_ratchet.py](scripts/lint_atomic_io_ratchet.py) in CI (lint job) prints the per-file AST count of hand-rolled atomic-rename call sites — `os.replace`/`os.rename` (alias-resolved) and `Path.replace`/`Path.rename` — every run and fails any branch that raises a file's count. **The number lives in that CI output, not here** (the 2026-08-13/19 note's "17" came from a text grep that counted comments and saw one spelling). Read it precisely: it counts hand-rolled RENAMES, not JSON written without `atomic_write_json` — as of 2026-08-29 no counted site writes JSON; most duplicate `atomic_write_text`, and the two that write BYTES (`hivemind/bundle.py` zip, `models/download.py` GGUF) now have a canonical writer — `atomic_write_bytes` shipped 2026-09-06 (1.2 P2P Slice B) — but are still UNMIGRATED (those two sites remain hand-rolled; migrating them is its own burn-down task). The ratchet only lets the count fall.
- **[engineering] Every `@dataclass(frozen=True)` that persists or crosses a wire MUST declare its forward-compat path in the class docstring before merge** (CC3): (a) escape-hatch — defaults on all fields + `extra: dict = field(default_factory=dict, hash=False, compare=False)` (JSON-serializable values only; `__post_init__` rejects extra keys colliding with declared fields), or (b) a `SHAPE-FROZEN at 1.0 (CC3)` marker with the rejection rationale. Typed exception hierarchies follow the same spirit via explicit keyword-only `__init__`s — no `**kwargs`/`extra`. Runtime-ephemeral config dataclasses are out of scope. Class rosters: [docs/agents/persistence-config.md](docs/agents/persistence-config.md). Full history: [docs/lessons/frozen-dataclass-forward-compat.md](docs/lessons/frozen-dataclass-forward-compat.md). Regression guard: CC3 audit list + the `SHAPE-FROZEN at 1.0 (CC3)` docstring marker on each frozen-without-extra dataclass; new frozen dataclasses must pick path (a) or (b) before merge.
- **[engineering] Every persisted JSON file carries `"_format_version": "1.0"` at root.** Writers wrap via `with_format_version(payload)` + `atomic_write_json`; loaders call `check_format_version(data, "<file_type>", log=logger)` (missing → `"0.x"` sentinel + one warning per file_type; old files still load). Envelope `schema_version` (int) and `_format_version` (string) coexist by design; do NOT bump the tombstoned legacy payload-layer `"version": "1.0"` strings. Full history: [docs/lessons/format-version-contract.md](docs/lessons/format-version-contract.md). Regression guard: [tests/integration/test_persistence_compat.py](tests/integration/test_persistence_compat.py).
- **[engineering] LLM access goes through `models/language/router.py`; concrete backends are not imported outside `models/language/`.** Sanctioned exceptions: `_MaximPeerBackend.for_url(...).health_check()` as the cross-module PROBE surface (inference DISPATCH stays router-only) and `bench/recovery_time.py` (deliberate benchmark bypass). Adding a backend type = one line in `runtime/lane_backends.BACKEND_CLASSES` + one `_classify_backend` branch — no router edit. Full history: [docs/lessons/llm-router-only-access.md](docs/lessons/llm-router-only-access.md). Regression guard: [src/maxim/runtime/lane_backends.py::BACKEND_CLASSES](src/maxim/runtime/lane_backends.py) (single dispatch table) + CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" step) blocking backend imports outside `models/language/` (allow-listed: `agents/llm_agent.py` — grandfathered; `agents/exec_agent.py` — imports a constant, not a backend; `_MaximPeerBackend` sanctioned via the probe-entry-point invariant).
- **[engineering] No NEW silent exception swallows** — never add a bare `except Exception: pass`; narrow the exception type, or handle-and-log. A handler that assigns or returns a fallback without reporting is the same swallow (check 5, the silent-default shape). Existing sites predate the rule and are grandfathered at their per-FUNCTION count; a verbatim move is free, an edited move needs a `scripts/swallow_moves.json` record. The deleted `@resilient` decorator must not be cited or re-introduced. Regression guard: [scripts/lint_no_silent_swallows.py](scripts/lint_no_silent_swallows.py) in CI (shipped 2026-08-13, fail-loud Stage 4) — zero-total over the measurement-path files (its `MEASUREMENT_PATH` list) + diff-scoped no-count-increase repo-wide + no DE-INSTRUMENTATION (checks 3–4, 2026-09-23): a "report" is ONLY the Stage-1 form (`log_swallowed_exception()` or `site=`) — the explicit `(e, operation=...)` form is a DEBUG line with no event and does NOT count, so on the measurement path handle-and-log means Stage-1; compared PER FUNCTION (per-file counts let a burn-down mask a de-instrumentation elsewhere in the file), the unreported count may not rise on listed files and repo-wide reports may disappear only with their handlers — deleting a swallow is free, de-instrumenting one fails, including in a moved function; masking within ONE function remains a stated blind spot; [tests/unit/test_lint_no_silent_swallows.py](tests/unit/test_lint_no_silent_swallows.py) drives `main()` itself. And since 2026-08-29 it PRINTS the repo-wide total every run, so the count lives in the CI output rather than in a number here that rots; the ad-hoc review grep remains the belt for the evasion shapes the lint's docstring lists.
- **[engineering] Values that cross a persistence boundary MUST be hashed with `utils/seeding.py::stable_hash_32` / `stable_hash_64_signed`, never builtin `hash()`** (PYTHONHASHSEED randomization makes persisted hashes permanently unmatchable across processes; a seed PARAMETER routed through `hash()` only looks deterministic). Sum-then-branch-on-sign sites use the SIGNED 64-bit variant. Persisted files carry `hash_scheme: "stable-sha256-v1"`; loaders WARN when absent. A same-process test passes over this entire bug class — the guard MUST be two-process with differing PYTHONHASHSEED. Full history: [docs/lessons/stable-hash-persistence.md](docs/lessons/stable-hash-persistence.md). Regression guard: [tests/unit/test_stable_hash_two_process.py](tests/unit/test_stable_hash_two_process.py) (verified to fail 5/5 against the pre-fix code).
- **[engineering] `main` is ahead of PyPI: the version bump happens in the RELEASE TRANSACTION, not on the change that earns it.** Between releases `pyproject.toml` + `src/maxim/__init__.py` carry the last published version and `CHANGELOG.md` accumulates under `## [Unreleased]`; the release PR moves that section under `## [X.Y.Z] - <date>`, then build → publish → tag on the published commit → GitHub Release with the exact artifacts and ABSOLUTE links. The three living sync lines (this file, `docs/plans/README.md`, `docs/index.md`) NAME the `pyproject` version and LINK PyPI — they never describe what PyPI serves, because that prose drifted on every release. Written intent diverging from routine practice is the Codex card's definition of a D; this is the written intent. Full procedure: [docs/publication_guide.md](docs/publication_guide.md). Regression guard: [scripts/lint_version_sync.py](scripts/lint_version_sync.py) in CI (lint job, "Version sync" step) — `pyproject` == `__init__` == the NEWEST released `## [X.Y.Z]` CHANGELOG header == all three sync lines, none of which may carry "pending"/"rc"/"serves"/"published" prose in the version claim; [tests/unit/test_lint_version_sync.py](tests/unit/test_lint_version_sync.py). The `[Unreleased]`-accumulates half is ENFORCED too (roadmap 1.1.x item 16.10, CLOSED): [scripts/lint_unreleased_on_src_change.py](scripts/lint_unreleased_on_src_change.py) in CI (lint job) is diff-scoped — a `src/maxim/*.py` change that does not grow `## [Unreleased]` FAILS; [tests/unit/test_lint_unreleased_on_src_change.py](tests/unit/test_lint_unreleased_on_src_change.py).
- **[engineering] Removed/renamed identifiers stay removed:** the class is `NAc`, never `NucleusAccumbens`; lane tiers are `"large"`/`"medium"`/`"small"`, never `"infer"`/`"review"`/`"record"`; `EnergyReactionBridge`/`MovementEnergyTracker` are deleted; the probe shims `probe_llm_server`/`llm_server_responding_at` are removed. Do not re-introduce any of them; grep after touching adjacent code. Regression guard: CI greps in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" + "deprecated probe shims" steps) — `NucleusAccumbens`, `EnergyReactionBridge`/`MovementEnergyTracker`, and the probe shims are zero-match in `src/maxim/`; quoted `"infer"`/`"review"`/`"record"` literals in [src/maxim/runtime/lane_models.py](src/maxim/runtime/lane_models.py) fail CI.
- **[engineering] Reachy motion has TWO hard rules and they live in the brief, which the routing table already makes mandatory before motion code: head pose is WORLD-frame (a body turn with `head=None` COUNTER-ROTATES the head, so head-mounted sensors do not turn with it), and `ReachyMiniController.goto_target` is the single clamped+locked dispatch point — motors 2+3 were destroyed by an unclamped pose.** Read [docs/agents/embodiment.md](docs/agents/embodiment.md) §1 before commanding any motion; it carries both in full with their clamp/retained-axis/`note_external_head_motion` details. The half that is NOT robot-specific, and the reason this stub stays here: **when a measurement disagrees with the model, verify the ACTUATION assumption — did the thing you are sensing with actually move? — BEFORE theorizing about the sensor**, and read the vendor's docs before reverse-engineering their kinematics. Regression guard: [tests/unit/test_reachy_head_frame.py](tests/unit/test_reachy_head_frame.py) + [tests/unit/test_reachy_workspace_safety.py](tests/unit/test_reachy_workspace_safety.py) + [tests/unit/test_reachy_retained_axes.py](tests/unit/test_reachy_retained_axes.py), all cited with their pre-fix failure counts in the brief.

## Running simulations — keep them small

Simulations call a live LLM for every turn and burn cost + time. Full sim discipline (resume, debug flags, sandbox choice, cost calibration): [docs/agents/simulation-experiments.md](docs/agents/simulation-experiments.md). The three session-killing rules stay here:

- **IMPORTANT: Use `--interactive false` when running sims from Claude Code or scripts.** Interactive mode is ON by default in CLI with a TTY; the raw terminal reader conflicts with non-human stdin.
- **Configure model + n_ctx via `maxim config`, not transient env/flags — single source of truth.** `maxim config set llm.profile <profile>` + `maxim config set llm.n_ctx <N>`, then verify with `maxim doctor 2>/dev/null | grep -i "n_ctx\|profile"`. The server's spawn n_ctx and the PromptBudgeter's belief resolve through DIFFERENT paths; if they drift, the sim silently takes 0 real actions behind HTTP 500s. Full three-leg bug history: [docs/lessons/sim-n-ctx-config-drift.md](docs/lessons/sim-n-ctx-config-drift.md).
- **Never co-locate a `maxim-leader`/experiment run with the sim on one box** — a second consumer of the :8100 server causes 500s under contention; the harness belongs on a different machine (see [docs/lessons/no-harness-on-leader-machine.md](docs/lessons/no-harness-on-leader-machine.md)).
- Set a narrow `--goal`; cap duration (Ctrl+C after 30–90s — partial results still report); prefer `--sandbox tmpdir`; local models for loop-testing, Claude for final behavior; watch `Cost:` in the report ($0.05–$0.15 per short run is normal).

## `maxim doctor` — environment diagnostics

Platform-aware environment checks + fix hints with actual IPs filled in; lives in [src/maxim/doctor/](src/maxim/doctor/). `maxim doctor` (leader/solo), `maxim doctor --retry` (interactive fix loop), `--json`, `--as peer <url>` / `--as leader` / `--as solo` role override; `maxim peer test` runs the peer-side probes self-contained. Companion: `maxim tunnel` in [src/maxim/tunnel/](src/maxim/tunnel/). Check-authoring + retry-loop maintenance guide: [docs/agents/llm-routing.md](docs/agents/llm-routing.md).

## Key Commands

```bash
# Quick start — interactive menu (no args needed)
maxim                                        # Rich menu: campaigns, chat, doctor, help

# Agent runtime
maxim --llm mistral-7b                       # local LLM
maxim --llm claude-sonnet                    # Claude (needs ANTHROPIC_API_KEY)

# Model management
maxim --list-models                          # show models + download status
maxim --delete-model llama-2-13b-chat        # free disk space

# Simulation (interactive mode ON by default for CLI with TTY)
maxim --sim "test memory recall"             # generative campaign (interactive)
maxim --sim interactive                      # interactive chat (full generative sim stack)
maxim --sim scenarios/campaigns/heist_v1.yaml  # DM campaign
maxim --sim "test safety" --research         # with research report
maxim --sim benchmark --models mistral-7b,qwen2.5-14b      # benchmark
maxim --sim scenarios/substrate/P0_paraphrase_collapse.yaml --seed 42  # fixture-driven
# In-sim commands: /cancel /pause /resume /status /report /display clean|bio|debug
# /new <goal> — arrow keys scroll the log; DM campaigns: type choice number/name, or free-text to roleplay

# Non-interactive (for Claude Code, CI, scripting, or debugging)
maxim --sim "test memory recall" --interactive false  # raw output, no Rich panel

# Embodiment in sim — AUT gets SEM affordance tools + pain cascade
maxim --sim "test sword combat" --embodiment weapons/rusty_sword
maxim --sim cradle --embodiment bodies/infant_humanoid   # 4-act developmental sim

# Asset Foundry / auto-curation
maxim --foundry "cyberpunk weapons" --foundry-genre cyberpunk
maxim --sim "test combat" --embodiment weapons/rusty_sword --auto-curate

# Diagnostics + networking
maxim doctor                                 # environment check
maxim tunnel setup                           # Cloudflare tunnel
maxim peer update && maxim peer restart      # remote update (auto-detects pip/git mode)
maxim peer install semantic                  # install optional extra on leader

# Tests
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
```

Full CLI reference: [docs/user/cli-reference.md](docs/user/cli-reference.md)

## Remote Update Workflow

```bash
# Pip-installed leaders (auto-detected):
maxim peer update && maxim peer restart

# Git-checkout leaders (dev workflow):
git push origin main && maxim peer update --dev && maxim peer restart
```

Use `--dry-run` first if unsure; `--version X.Y.Z` pins a PyPI version; `--force` (dev mode) clears untracked-file blocks. Troubleshooting: [docs/troubleshooting/remote_update.md](docs/troubleshooting/remote_update.md).

**Important for Claude agents:** `maxim peer update --dry-run`, `maxim peer version`, `maxim peer logs`, `maxim peer llm --status`, and `maxim peer deps` are safe and read-only. `maxim peer update`, `maxim peer restart`, `maxim peer llm <model>`, and `maxim peer install <extras>` modify leader state — only run when explicitly asked by the user.

## Versioning

Policy and its enforcement live in the `main`-is-ahead-of-PyPI invariant above (bump in the release transaction; `[Unreleased]` accumulates; the three sync lines; `lint_version_sync.py` + `lint_unreleased_on_src_change.py`). A `src/` change still obliges a release cut eventually (runtime behavior, CLI interface, peer/leader protocol) but does not bump on the spot. Procedure: [docs/publication_guide.md](docs/publication_guide.md); history: [docs/lessons/versioning-main-ahead-of-pypi.md](docs/lessons/versioning-main-ahead-of-pypi.md). Check locally: `python -c "from maxim import get_version_info; print(get_version_info())"` or `maxim peer version` (mismatch → `maxim peer update && maxim peer restart`).

## Environment Variables — session-critical core

Full per-subsystem tables (with the experiment/ablation toggles) live in the owning `docs/agents/` brief. Adding a var → owning brief's table + autouse conftest scrub (lesson above). The canonical truthy parser for MAXIM_* toggles is `cluster_bias_annotation.annotation_disabled_via_env` ("1"/"true"/"yes"/"on", case-insensitive).

```bash
ANTHROPIC_API_KEY          # Claude backend (7 more provider keys: see docs/agents/llm-routing.md)
MAXIM_ROLE=leader          # Explicit role: leader|peer|solo (exported at startup; downstream reads env)
MAXIM_LLM_ENABLED=1        # Enable LLM inference
MAXIM_LLM_PROFILE=claude-sonnet  # Default model profile (prefer: maxim config set llm.profile)
MAXIM_LLM_N_CTX=4096       # Override llama.cpp n_ctx (prefer: maxim config set llm.n_ctx)
MAXIM_LOG_FILE=/tmp/maxim.jsonl  # JSONL file handler; stdout stays human-readable
MAXIM_BACKEND_TRACE=1      # Per-call peer-backend JSONL (pair with MAXIM_LOG_FILE)
MAXIM_HTTP_TRACE=1         # Log every outbound HTTP call at INFO
MAXIM_PROVENANCE_VERBOSITY=1     # Decision log at ~/.maxim/util/lane_decisions.jsonl (0/1/2)
MAXIM_SUBSTRATE_PATH=1     # Enable substrate encoding path (LinguisticEncoder → EC → ATL)
MAXIM_HEARTBEAT=1          # System health heartbeat every 10s + stall detection
MAXIM_SKIP_REMOTE_PROBE=1  # Bypass remote-URL probe — CI escape hatch
MAXIM_AUTO_DOWNLOAD_MODELS=1     # Skip the auto-download prompt
```

## Testing

```bash
# Full suite
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
# Just the module you changed (fast feedback)
python -m pytest tests/unit/test_lane_metrics.py -v
```

**Run narrow first, then wide.** Test the specific module you changed before the full suite (~12 min as of 2026-08 — measured 9,168 tests in 12:09). **Kill stale sims before running tests** (`pkill -f "maxim.*sim" 2>/dev/null; sleep 2`) — a running sim holds GPU + port resources and causes hangs. **Threading pitfalls:** use `threading.RLock` (not `Lock`) if a method acquires the lock then calls another method that also acquires it — regular `Lock` deadlocks on re-entry; thread-safety tests that appear to hang are usually deadlocked, not slow. **Don't run sims from tests** — sims call real LLMs; tests mock them. Peer/tunnel checks: `curl -si -H "Authorization: Bearer $KEY" https://maxim.yourdomain.com/v1/models`; guides in [docs/troubleshooting/](docs/troubleshooting/).

## Simulation Reports

Sim runs save to `~/.maxim/sim_reports/{session_id}/` (report.json, actions.jsonl, aut_hippocampus.json, aut_nac.json); `maxim.create.agent()` homes live in `~/.maxim/agents/{name}/` (both resolved by `utils/paths.py::resolve_run_dir`). Research protocol + campaign flow: `docs/simulation.md` and `docs/experiments/`.

## Python API (pymaxim)

Published to PyPI as `pymaxim` (import name `maxim`); verb-based facades lazy-loaded from `src/maxim/api.py`. Maintenance rules (verbs are facades not logic, lazy imports only, structured returns, package extras): [docs/agents/runtime-tools.md](docs/agents/runtime-tools.md). Build validation before any publish: `python -m build && twine check dist/*`; guide: [docs/publication_guide.md](docs/publication_guide.md).

## Active initiatives

Current version: **1.3.1** (`pyproject.toml` + `src/maxim/__init__.py`; PyPI: https://pypi.org/project/pymaxim/ — `main` is ahead of PyPI by policy, §Versioning). 1.3.1 is **"Hardening"** — fixes and their guards, no new behavioural claim (security, the public format freeze, public-API correctness, release integrity; Exp 10 re-run MAINTAINED (narrow), [#935](https://github.com/dennys246/Maxim/issues/935)). 1.3.0 is **"Oasis-2"** — the survival world: reward from the game (Paper 1.20.4, no LLM in the action path); **Exp 60 EARNED** (anticipatory drowning-avoidance) + **Exp 61 EARNED** (the survival fear transfers between agents) + the **R3 survival benchmark** (instrument + frozen baseline, nothing graduated) + Exp 56 re-baselined on 1.20.4. **Exp 62 rung A EARNED 2026-09-20** (after 1.3.0; not claimed by 1.3.1 until its different-reader pass is recorded; the shipped body carries the drowning-fear across pools — 1.4 Phase 1 closed; bounded to the sealed-shell frozen-day class; its "night miss" (0.799) is the `time_of_day` wrap just before dawn, not night — [#899](https://github.com/dennys246/Maxim/issues/899)). 1.2.x was **"Oasis"** (Exp 56 EARNED; 1.2.1 shipped the spoken-code pairing pieces, composition unwired — corrected in 1.3.0). This line is checked by `scripts/lint_version_sync.py` — bump it in the release transaction. Active theme **1.4** (working title "Anticipation"; [roadmap_1_4.md](docs/plans/roadmap_1_4.md)): generalization + multi-step credit on the survival world; perception fabric revives when GL3's registry+provenance stage ships and a 1.4 rung needs cross-modal binding ([grounding.md](docs/plans/grounding.md); owner 2026-10-07), microduck DEFERRED until a second physical body exists. The roadmap index is [docs/plans/README.md](docs/plans/README.md); the shipped-release history through 1.3 is archived at [docs/plans/archive/roadmap_1_1_to_1_3.md](docs/plans/archive/roadmap_1_1_to_1_3.md) (its 1.1.x item-16 block is CLOSED — verified 2026-09-20); behavioral-graduation gates live in [docs/plans/behavioral_graduation_candidates.md](docs/plans/behavioral_graduation_candidates.md). Deferred plans (revive on trigger): [docs/plans/deferred/](docs/plans/deferred/). Shipped-history through 2026-04 (the old "Recently shipped" ledger): [docs/lessons/active-initiatives-history-2026.md](docs/lessons/active-initiatives-history-2026.md).
