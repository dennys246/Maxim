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
- **Run `mypy` on public API files** after changing api.py, session.py, create.py, load.py, or __init__.py: `mypy src/maxim/__init__.py src/maxim/api.py src/maxim/session.py src/maxim/create.py src/maxim/load.py --ignore-missing-imports --follow-imports=silent` (same invocation CI runs — `--follow-imports=silent` scopes errors to the five files; the rest of the codebase is not mypy-clean yet)
- **Run `ruff format`** after any changes: `ruff format src/ tests/`
- **Parallel sessions use worktrees.** When ≥2 Claude sessions run concurrently on independent work, each uses its own git worktree (`git worktree add ../Maxim-wt-<branch> -b <full-branch>`) and works entirely in absolute paths within that worktree. Single-session work stays in the main checkout. Tests in worktrees need `PYTHONPATH=src`. Note `~/.maxim/` is shared across worktrees — don't run sims from concurrent doc/code sessions or they'll collide on persisted state.

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
| `runtime/agent_loop.py`, `runtime/executor.py`, `runtime/bootstrap.py`, `runtime/bio_stack.py`, `runtime/agent_factory.py`, `runtime/agent_pool.py`, `runtime/tool_dispatch.py`, `tools/`, `agents/`, `cli.py`, `api.py` | [docs/agents/runtime-tools.md](docs/agents/runtime-tools.md) |

Multiple rows match → read all matched briefs. Adding an env var → add it to the owning brief's table (and pair it with a conftest scrub, see the lesson below). Project structure reference: [docs/reference.md](docs/reference.md).

## Lessons learned (bugs that bit us) — cross-cutting core

Subsystem-specific lessons live in the owning `docs/agents/` brief; full narratives in `docs/lessons/`. These fire in ANY area:

**[engineering] An experiment harness MUST assert that the `maxim` its sub-sims import is its OWN repo — `git_hash` answers the wrong question.** Any harness that spawns `maxim` calls `scripts/_provenance.py::assert_repo_interpreter(repo_root, binary, exempt=<mock>)` before its first sub-sim (exit 3 on mismatch), and SHOULD stamp `executed_code_provenance(...)` into every run record. Trigger: stale editable `.pth` files / relative `PYTHONPATH` can silently run another checkout while every sub-sim "succeeds" (Exp 42b retraction — a result whose code-under-test cannot be established is not a validation; do not argue it was "probably fine"). Operator hygiene: `export PYTHONPATH="$PWD/src"` (absolute) on its OWN line, never chained after a `source` with `&&`. Full history: [docs/lessons/harness-provenance-assert-repo-interpreter.md](docs/lessons/harness-provenance-assert-repo-interpreter.md). Regression guard: [scripts/lint_harness_provenance.py](scripts/lint_harness_provenance.py) in CI (shipped 2026-08-13) — every `scripts/**/*.py` matching a spawn-of-maxim pattern must reference `assert_repo_interpreter` or carry a `# provenance-exempt: <reason>` marker; as of the ship date all six spawning harnesses (`benchmark_exp42_preference.py`, `benchmark_exp41_exploration.py`, `benchmark_cross_session.py`, `benchmark_cradle_mother.py`, `exp44/campaign.py`, `exp49/run_trials.py`) call it — plus [scripts/_provenance.py](scripts/_provenance.py) itself and the post-mortem in [docs/experiments/42b_drive_pain_fold_revalidation.md](docs/experiments/42b_drive_pain_fold_revalidation.md).

**[engineering] Push silent-no-op invariants into types, not helpers.** Count silent failures, not loud ones: one silent-failure miss in a critical path → consider structural enforcement; three silent-failure misses in any path → no longer a question, push the invariant DOWN into the type/constructor signature so forgetting becomes a `TypeError`, not a silent no-op. Canonical example: `build_executor(pain_bus=...)` required keyword-only (see the canonical-builders entry in [docs/agents/runtime-tools.md](docs/agents/runtime-tools.md)). Full history: [docs/lessons/silent-noop-invariants-into-types.md](docs/lessons/silent-noop-invariants-into-types.md). Regression guard: [src/maxim/runtime/bootstrap.py::build_executor](src/maxim/runtime/bootstrap.py) — required keyword-only `pain_bus=` parameter is the canonical example; signature enforces the rule structurally so forgetting becomes a `TypeError`, not a silent no-op.

**[engineering] Opt-in env vars in hot startup paths need autouse scrubs.** Any new `if os.environ.get("MAXIM_FOO"): do_side_effect()` branch reachable from `build_primary_router` MUST be paired in the same commit with an `@pytest.fixture(autouse=True)` env-scrub in tests/conftest.py — a leaked var makes the side effect run for real in every later test (P5: 9-minute pytest hang on a real 1 GB GGUF download). Full history: [docs/lessons/env-var-autouse-scrubs.md](docs/lessons/env-var-autouse-scrubs.md). Regression guard: [tests/conftest.py](tests/conftest.py) — autouse env-scrub fixtures pattern; new env-var branches must add a matching scrub in the same commit.

**[engineering] `utils/optional_deps.py` is the canonical optional-dependency surface — do NOT add any new `try: import X except ImportError:` variant (silent pass / non-deduped warning / swallowed return-None) anywhere in `src/maxim/`.** Pick by intent: `require_optional_dependency` (explicitly-requested feature → raises typed `OptionalDependencyError`), `optional_dependency_available` (capability probe → bool, never logs), `warn_optional_fallback` (real fallback exists → ONE deduped WARNING). Add new extras in `EXTRA_FOR_IMPORT`, not at call sites; `OptionalDependencyError` access patterns are exactly `.import_name`/`.extra`/`.fix_hint` — no parallel attributes. Full history: [docs/lessons/optional-deps-canonical-surface.md](docs/lessons/optional-deps-canonical-surface.md). Regression guard: [tests/unit/test_optional_deps.py](tests/unit/test_optional_deps.py) — covers `require_optional_dependency` raise/return, `optional_dependency_available` bool, `warn_optional_fallback` dedup, `OptionalDependencyError` subclass shape, and LLM-router reraise behaviour.

**[engineering] HTTP call sites must use `maxim/utils/http.py`.** New outbound HTTP calls pick `http.get`/`http.post` (registered endpoint), `http.fetch_url` (arbitrary URL), or `http.download_to_file` (streaming); the `raw_proxy_forward` escape hatch is reserved for `leader_proxy._proxy_request` ONLY — do not use it elsewhere. (Origin: the 2026-04-12 Cloudflare Bot Fight Mode missing-User-Agent incident.) Full history: [docs/lessons/http-via-utils-http.md](docs/lessons/http-via-utils-http.md). Regression guard: CI grep `grep -r "urllib.request.urlopen" src/maxim/ | grep -v "utils/http.py"` must return zero matches; enforced in [.github/workflows/test.yml](.github/workflows/test.yml).

**[engineering] Dead code accumulates silently:** Before publishing or after major refactors, grep for orphan modules: `.py` files whose basename doesn't appear in any `import` statement. We found 15 dead modules (~8,500 LOC) shipping in the wheel. Regression guard: process invariant — periodic grep before publish; no automated test enforces.

**[engineering] Plan review round runs BEFORE PR merge, not after** (refined 2026-04-12; validated repeatedly — full round-by-round history in [docs/lessons/review-round-discipline.md](docs/lessons/review-round-discipline.md)): every completed sub-plan on a `feat/<plan>` branch triggers a pre-merge review round. Spawn two parallel review Claudes (Executor lens + Architecture lens). Fold findings into the same branch via a follow-up commit BEFORE opening/merging the PR. Do NOT merge first and ship a `fix/<plan>-loose-ends` PR after — that splits the bisect surface and leaves known-buggy code on main. Tests catch few of these findings — they're correctness issues in input spaces tests don't cover; both reviewers catching the same finding independently is a strong trust signal. Review rounds are non-optional, not ceremony. **SCOPE TRIGGER: a round covers the diff AS IT EXISTED WHEN IT RAN. If the merged diff is not the reviewed diff, you did not review the merge — run another round.** Re-run if the branch gains new sub-plans, new experiments, new `src/` changes, or a new claim after the last round; a docs-only touch-up does not re-trigger. **The value of the round is a DIFFERENT reader, not a more careful one** — resolving to be careful does not substitute, which is why this is a process rule rather than an epistemics one. Regression guard: process invariant — review-round discipline enforced by author + reviewer attention; no automated test enforces (the reviewed-diff-vs-merged-diff comparison is mechanically checkable and is tracked follow-up work).

**A review round is not complete until its fold commits are ON THE MERGE TARGET** (2026-07-29, cost a broken `main` — full narrative in [docs/lessons/review-round-discipline.md](docs/lessons/review-round-discipline.md)). A PR was squash-merged with only its first commit, shipping a design its own review round had refuted, with green CI because the guard test was in the unmerged fold. **Rules:** (1) after a review fold, verify the fix is on the TARGET, not just the branch — `git show origin/main:<file> | grep <the-fix>`, or check the squash-merge's diff, not the PR's file list; (2) never squash-merge a PR that is still receiving fold commits — the reviewed diff and the merged diff must be the same diff; (3) when a fold fixes a silent failure, land the *guard test* in the same commit as the fix so a partial merge cannot produce green CI on broken code. Corollary: `gh pr list --state open` returning EMPTY means your PR is already merged — read that as an answer, not as a missing list. Regression guard: process invariant — no automated test; the mechanically checkable form is comparing the merge commit's diff against the last-reviewed diff, tracked as follow-up work.

## Working principles for new mechanisms

These five principles govern HOW new architectural commitments enter the codebase. They are upstream of the invariant surfaces (here and in the briefs) — apply them when *adding* invariants, not just when reading them.

- **Two-tier invariant tracking.** Tag each new invariant `[engineering]` (code breaks loudly without it) or `[behavioral]` (empirically validated via Roy or equivalent as carrying measurable behavioral weight). **New mechanisms enter as `[engineering]` only** and graduate to `[behavioral]` when an experiment earns them. Bio-inspired naming is load-bearing for the mental model but does NOT count as behavioral validation. Graduation tracking lives in [docs/plans/behavioral_graduation_candidates.md](docs/plans/behavioral_graduation_candidates.md) — a 1.0 gate AND the ongoing post-1.0 regression discipline; Earned entries carry **Re-run on:** triggers + **Regression guard:** experiment paths; `Stale`/`Broken` entries block the next release.

- **Dormancy over deletion.** When a mechanism fails to earn behavioral weight, mark it `Dormant since <date>: <reason>` in its module docstring rather than deleting. Code stays wired, callers intact. But: no new features build on it, no new invariants accrue, tests beyond regression are not extended. Resurrection requires a new experiment that earns the weight, not "we have time now." This codebase is intimately wired by design — whim-deletion historically caused secondary breakage; dormancy is the middle path between deletion-cascade and monotonic accumulation.

- **Front-gate scope pressure at design time.** Before drafting any implementation plan for a new mechanism (bus, bridge, bio-system, annotation Wire, gating layer, builder), force the question: *"Does this need to be its own mechanism, or can it ride on existing infrastructure?"* If it needs to be its own, name the specific reason in the plan doc's motivation ("existing infrastructure X cannot do this because Y"). If it can ride on existing, choose that path even when less architecturally elegant.

- **Cycle convergence vs divergence.** Cyclic experiment findings are signal — but distinguish **convergence** (same kind of issue, narrowing each iteration → keep cycling) from **divergence** (new failure modes each iteration → the mechanism is getting more complex faster than stabilizing). **Two divergence iterations in a row → stop iterating on the mechanism and run a bird's-eye audit:** "what else changed?", "what's the actual independent variable?", "is the mechanism the cause or the messenger?", "have any non-code dependencies moved (encoder model, library version, env var defaults)?", and — added after it was blown through three times in one hardware session — **"did the action I commanded actually happen?"** A wrong actuation assumption is indistinguishable from a broken sensor and manufactures unlimited plausible sensor theories. **This trigger covers DEBUGGING, not only pre-registered cycles: if two explanations for a bad measurement have died, stop generating a third — audit the layer beneath.** Post-hoc findings (from post-result investigation rather than planned measurements) don't directly count as divergence — they spawn new pre-registered iterations; the trigger then watches those. Sharpened form when post-hoc findings are present: two iterations in a row whose primary criterion fails AND whose post-hoc findings each spawn new follow-up plans. Full history incl. the Roy-3c bisect and the 2026-07-16 six-hypothesis actuation incident: [docs/lessons/review-round-discipline.md](docs/lessons/review-round-discipline.md) + [docs/lessons/reachy-head-world-frame.md](docs/lessons/reachy-head-world-frame.md).

- **Regression-guard / experiment citation per invariant.** Every `[engineering]` invariant — in this file or a brief — ends its body with `Regression guard: <path>`; every `[behavioral]` invariant with `Roy experiment: <path>`. Valid guard references: a test path, a CI grep pattern, or a co-located source file that structurally enforces the rule (typed constructor, frozen dataclass, `@abstractmethod`). **A missing line is a visible coverage gap by design** — surfacing the absence is the discipline's value. Cite `file::symbol`, never `file:line` (audited line numbers all drift; symbols hold), and avoid volatile counts. CI enforcement: `scripts/lint_claude_md_invariants.py` audits this file AND `docs/agents/*.md`, existence-checks lesson links, and holds this file under its token ceiling.

## Architectural invariants — cross-cutting core

Subsystem invariants live in the owning `docs/agents/` brief (same stub format, same lint). These apply everywhere:

- **[engineering] Tool results flow through the agent bus**; don't call agents directly from tools. Regression guard: convention — enforced by reviewer attention; [src/maxim/runtime/executor.py](src/maxim/runtime/executor.py) is the canonical dispatch site.
- **[engineering] Persistence uses `maxim.utils.atomic_io.atomic_write_json`** (fsync + tmp cleanup). Don't hand-roll `open().write()` + `os.replace()`. Regression guard: [src/maxim/utils/atomic_io.py](src/maxim/utils/atomic_io.py) is the canonical writer; ad-hoc `grep -rn "os.replace" src/maxim/ | grep -v atomic_io.py` surfaces violations. (KNOWN GAP 2026-08-13, recounted 2026-08-19: that grep currently surfaces 17 hand-rolled sites — detection-only, not enforced; flagged in the diet-plan fold, needs its own task.)
- **[engineering] Every `@dataclass(frozen=True)` that persists or crosses a wire MUST declare its forward-compat path in the class docstring before merge** (CC3): (a) escape-hatch — defaults on all fields + `extra: dict = field(default_factory=dict, hash=False, compare=False)` (JSON-serializable values only; `__post_init__` rejects extra keys colliding with declared fields), or (b) a `SHAPE-FROZEN at 1.0 (CC3)` marker with the rejection rationale. Typed exception hierarchies follow the same spirit via explicit keyword-only `__init__`s — no `**kwargs`/`extra`. Runtime-ephemeral config dataclasses are out of scope. Class rosters: [docs/agents/persistence-config.md](docs/agents/persistence-config.md). Full history: [docs/lessons/frozen-dataclass-forward-compat.md](docs/lessons/frozen-dataclass-forward-compat.md). Regression guard: CC3 audit list + the `SHAPE-FROZEN at 1.0 (CC3)` docstring marker on each frozen-without-extra dataclass; new frozen dataclasses must pick path (a) or (b) before merge.
- **[engineering] Every persisted JSON file carries `"_format_version": "1.0"` at root.** Writers wrap via `with_format_version(payload)` + `atomic_write_json`; loaders call `check_format_version(data, "<file_type>", log=logger)` (missing → `"0.x"` sentinel + one warning per file_type; old files still load). Envelope `schema_version` (int) and `_format_version` (string) coexist by design; do NOT bump the tombstoned legacy payload-layer `"version": "1.0"` strings. Full history: [docs/lessons/format-version-contract.md](docs/lessons/format-version-contract.md). Regression guard: [tests/integration/test_persistence_compat.py](tests/integration/test_persistence_compat.py).
- **[engineering] LLM access goes through `models/language/router.py`; concrete backends are not imported outside `models/language/`.** Sanctioned exceptions: `_MaximPeerBackend.for_url(...).health_check()` as the cross-module PROBE surface (inference DISPATCH stays router-only) and `bench/recovery_time.py` (deliberate benchmark bypass). Adding a backend type = one line in `runtime/lane_backends.BACKEND_CLASSES` + one `_classify_backend` branch — no router edit. Full history: [docs/lessons/llm-router-only-access.md](docs/lessons/llm-router-only-access.md). Regression guard: [src/maxim/runtime/lane_backends.py::BACKEND_CLASSES](src/maxim/runtime/lane_backends.py) (single dispatch table) + CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" step) blocking backend imports outside `models/language/` (allow-listed: `agents/llm_agent.py` — grandfathered; `agents/exec_agent.py` — imports a constant, not a backend; `_MaximPeerBackend` sanctioned via the probe-entry-point invariant).
- **[engineering] No NEW silent exception swallows** — never add a bare `except Exception: pass`; narrow the exception type, or handle-and-log. Existing sites (≈430 as of 2026-08) predate the rule and are grandfathered at their per-file count. The deleted `@resilient` decorator must not be cited or re-introduced. Regression guard: [scripts/lint_no_silent_swallows.py](scripts/lint_no_silent_swallows.py) in CI (shipped 2026-08-13, fail-loud Stage 4) — zero-total over the 16 measurement-path files + diff-scoped no-count-increase repo-wide; the ad-hoc review grep remains the belt for the evasion shapes the lint's docstring lists.
- **[engineering] Values that cross a persistence boundary MUST be hashed with `utils/seeding.py::stable_hash_32` / `stable_hash_64_signed`, never builtin `hash()`** (PYTHONHASHSEED randomization makes persisted hashes permanently unmatchable across processes; a seed PARAMETER routed through `hash()` only looks deterministic). Sum-then-branch-on-sign sites use the SIGNED 64-bit variant. Persisted files carry `hash_scheme: "stable-sha256-v1"`; loaders WARN when absent. A same-process test passes over this entire bug class — the guard MUST be two-process with differing PYTHONHASHSEED. Full history: [docs/lessons/stable-hash-persistence.md](docs/lessons/stable-hash-persistence.md). Regression guard: [tests/unit/test_stable_hash_two_process.py](tests/unit/test_stable_hash_two_process.py) (verified to fail 5/5 against the pre-fix code).
- **[engineering] Removed/renamed identifiers stay removed:** the class is `NAc`, never `NucleusAccumbens`; lane tiers are `"large"`/`"medium"`/`"small"`, never `"infer"`/`"review"`/`"record"`; `EnergyReactionBridge`/`MovementEnergyTracker` are deleted; the probe shims `probe_llm_server`/`llm_server_responding_at` are removed. Do not re-introduce any of them; grep after touching adjacent code. Regression guard: CI greps in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" + "deprecated probe shims" steps) — `NucleusAccumbens`, `EnergyReactionBridge`/`MovementEnergyTracker`, and the probe shims are zero-match in `src/maxim/`; quoted `"infer"`/`"review"`/`"record"` literals in [src/maxim/runtime/lane_models.py](src/maxim/runtime/lane_models.py) fail CI.
- **[engineering] Reachy head pose is WORLD-frame and sits ABOVE `body_yaw` — `goto_target(body_yaw=X)` with `head=None` COUNTER-ROTATES the head, so head-mounted sensors (mics, camera) DO NOT turn with the body.** Rule: any code that turns the body and then reads a head-mounted sensor MUST ship an explicit `head=` matrix with the body delta added to head yaw; call `set_automatic_body_yaw(False)` when your loop owns the yaw axis. `head=None` means "re-solve IK against the RETAINED world head target", NOT "leave the head alone". Generalized lesson: when a measurement disagrees with the model, verify the ACTUATION assumption — did the thing you are sensing with actually move? — BEFORE theorizing about the sensor; and read the vendor's docs before reverse-engineering their kinematics. Full history: [docs/lessons/reachy-head-world-frame.md](docs/lessons/reachy-head-world-frame.md). Regression guard: [tests/unit/test_reachy_head_frame.py](tests/unit/test_reachy_head_frame.py) — offline tests against a fake SDK pinning the production `ReachyMiniController` path (body-only ships a head matrix; head world-yaw tracks body; `head_yaw` is body-relative and composed onto the last COMMANDED body with the readback as one-shot seed; `get_current_pose()` exposes `body_yaw`); verified to fail on the pre-fix controller.
- **[engineering] `ReachyMiniController.goto_target` is the single clamped+locked motion dispatch point; `motion/movement.py::move_head` is the only other sanctioned SDK motion primitive. Do NOT hand-roll `mini.goto_target(...)` / `mini.set_target(...)` / `mini.look_at_image(...)` anywhere else** (motors 2+3 were destroyed by an unclamped pose). Head-yaw clamps apply in the BODY-RELATIVE frame under `_motion_lock`; callers reporting pose outcomes MUST read `last_clamped_axes` or the frame readback, never echo the commanded value. Retained axes fill from the per-axis last-COMMANDED stash (post-clamp), never live readback (positive-feedback ratchet); readback seeds an axis exactly once; any raw head mover MUST wire `controller.note_external_head_motion()` or the next command snaps the head to a stale pose. Full history: [docs/lessons/reachy-motion-dispatch-safety.md](docs/lessons/reachy-motion-dispatch-safety.md). Regression guard: [tests/unit/test_reachy_workspace_safety.py](tests/unit/test_reachy_workspace_safety.py) (verified to fail 10/14 on the pre-fold code) + [tests/unit/test_reachy_retained_axes.py](tests/unit/test_reachy_retained_axes.py) (biased-plant fake SDK; core ratchet tests verified to fail on the pre-fix controller) + CI grep in [.github/workflows/test.yml](.github/workflows/test.yml) ("1.0 guard promotion" step) restricting raw `mini.(goto_target|set_target|look_at_image)` to the sanctioned primitives.

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

Version is defined in two places that **must stay in sync**: `pyproject.toml` and `src/maxim/__init__.py`. Bump on any change affecting runtime behavior, CLI interface, or peer/leader protocol (docs-only or test-only changes don't). Check: `python -c "from maxim import get_version_info; print(get_version_info())"` or `maxim peer version` (mismatch → the leader needs `maxim peer update && maxim peer restart`).

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

Sim runs save to `~/.maxim/sessions/{session_id}/` (report.json, actions.jsonl, aut_hippocampus.json, aut_nac.json). Research protocol + campaign flow: `docs/simulation.md` and `docs/experiments/`.

## Python API (pymaxim)

Published to PyPI as `pymaxim` (import name `maxim`); verb-based facades lazy-loaded from `src/maxim/api.py`. Maintenance rules (verbs are facades not logic, lazy imports only, structured returns, package extras): [docs/agents/runtime-tools.md](docs/agents/runtime-tools.md). Build validation before any publish: `python -m build && twine check dist/*`; guide: [docs/publication_guide.md](docs/publication_guide.md).

## Active initiatives

Current version: **1.1.0** (`pyproject.toml` + `src/maxim/__init__.py`) — the 1.1 "Sensorimotor" cut, **release candidate; PyPI serves 1.0.9** (published 2026-08-23, tag `v1.0.9` at `5cb4413b`) until the 1.1.0 upload + `v1.1.0` tag land. This line is in the release procedure's sync table — update it with the version bump, not after. Active theme **1.1 "Sensorimotor"**. The roadmap index is [docs/plans/README.md](docs/plans/README.md); the roadmap through 1.3 is [docs/plans/roadmap_1_1_to_1_3.md](docs/plans/roadmap_1_1_to_1_3.md); behavioral-graduation gates live in [docs/plans/behavioral_graduation_candidates.md](docs/plans/behavioral_graduation_candidates.md). Deferred plans (revive on trigger): [docs/plans/deferred/](docs/plans/deferred/). Shipped-history through 2026-04 (the old "Recently shipped" ledger): [docs/lessons/active-initiatives-history-2026.md](docs/lessons/active-initiatives-history-2026.md).
