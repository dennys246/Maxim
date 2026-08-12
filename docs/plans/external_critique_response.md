# External critique response — living scorecard

**Status:** LIVING DOC. Update a row when its status changes; do not let this drift.
**Origin:** an external deep-dive code critique (2026-08-10), assessed **~85% correct**
after verifying its factual claims against the code. Its two misfires are recorded below
so they are not "fixed" by mistake.

**Why this doc exists:** the four response plans were written and then registered nowhere —
invisible from the index, tracked only in one session's memory. That is the same
missing-is-the-signal failure the four-lens review caught elsewhere. A backlog nobody can
find is a backlog that quietly becomes untrue.

## Scorecard

| # | Critique item | Status (2026-08-11) | Plan / artifact |
|---|---|---|---|
| 3 | NAc docstring claims TD learning; bio-naming asserts unearned isomorphism | **DONE** — docstring fixed (#486); full audit merged (#492): 4 MECHANISM (Kuramoto oscillators, Rescorla-Wagner in `CausalLink`, cerebellar delta rule, Hebbian episode binding), 7 FUNCTIONAL, 1 NAME-ONLY. **Open:** the CI claim-lint; and a bio-lens correction that EC's "pattern separation" is a DG/CA3 function, not entorhinal. | [bio_docstring_truth_pass.md](bio_docstring_truth_pass.md) |
| 4 | Exp 37's pre-registration was amended into something else | **STRUCTURALLY ADDRESSED, not closed.** Exp 44 relabeled EXPLORATORY; Exp 44b pre-registration written with a single frozen confirmatory test, frozen gates, and an amendment rule that demotes to exploratory. Closes when 44b runs at power. | [../experiments/protocols/exp44b_preregistration.md](../experiments/protocols/exp44b_preregistration.md) |
| 5 | n=5 with no statistics | **APPARATUS DONE, not closed.** `stats_counterfactual.py`: exact sign test on the run as unit (flips within a run are correlated), pooled binomial + Wilson CI, instrument-verified (planted effect p=3e-05; null rejection 2% at α=0.05 over 200 Monte Carlo campaigns). Closes when the confirmatory campaign runs. | `scripts/exp44/stats_counterfactual.py` |
| 2 | 1,749 `except Exception`, ~407 bare `pass` — a measurement-integrity problem | **1 of 4 stages.** Stage 1 merged (#487): 48 measurement-path sites instrumented, plus the fix that made the event actually reach the JSONL (it did not, and the Stage-2 protocol greps that JSONL). Stages 2 (measure which fire), 3 (root-cause + narrow), 4 (CI zero-swallow lock) open. Repo-wide bare swallows: **432** — deliberately out of scope. | [measurement_path_fail_loud.md](measurement_path_fail_loud.md) |
| 1 | God functions: `run_agentic_loop` 3,298 lines / 47 trys | **NOT STARTED.** Deliberately gated behind fail-loud Stage 2 (the swallow instrumentation is the behavior-preservation detector for extraction). Boundaries already exist: 56 numbered section banners + `LoopController`. | [god_function_decomposition.md](god_function_decomposition.md) |
| 6 | CLAUDE.md is the bottleneck (~62K tokens) | **NOT STARTED — and it GREW.** Still ~62.5K; this session added an env-table entry. Every session pays this before doing anything, which makes it the highest-compounding item on the list and the one most embarrassing to have skipped. | [claude_md_diet.md](claude_md_diet.md) |
| 7 | CI gates almost nothing; slow tests; PyPI drift; 182 env vars | **UNTOUCHED.** CI still runs `pytest tests/unit/` only (behavioral/integration/substrate/learning gate nothing); mypy on 5 files; no coverage gate; PyPI 1.0.0 vs repo 1.0.6; env vars 182 → 183. | *no plan yet* |

**Also fixed along the way (not in the critique):** `profile_has_local_file` fail-closed on
unknown profiles (#491) — a host-dependent-CI-green bug surfaced while running the suite.

## Where the critique was WRONG — do not "fix" these

- **Angular Gyrus is not "Hebbian binding = an outer product."** It is an exact-math engine
  with an honest mapping paragraph. The critique guessed from the name.
- **Three more modules earn mechanism-level claims** the critique implicitly denied:
  genuine Kuramoto dynamics, genuine Rescorla-Wagner value learning with a true
  single-trial RPE, and a cerebellar delta rule. One false claim existed (TD); four true
  ones were sitting unclaimed.

## The meta-finding

The critique's deepest point was *"adversarial-to-self isn't adversarial"* — every quality
signal in this repo is self-generated. That one is **partially answered**: the four-lens
review on 2026-08-11 caught a recommendation that would have broken a graduated result
(a ring code on a non-circular sensor) and found that a plan duplicated work shipped three
weeks earlier. Multi-lens review demonstrably catches what single-author review does not.
It is still not *external* review, which is the argument for shipping the paper.

## Maintenance

Update a row when status changes. An item moving to DONE requires its artifact linked and,
where applicable, its regression guard named — the same Principle-5 discipline CLAUDE.md's
invariants carry.