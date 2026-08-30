# External critique response — living scorecard

**Status:** LIVING DOC. Update a row when its status changes; do not let this drift.
**Last verified against the code:** 2026-08-29 (1.1.1 Cluster C truth pass — roadmap item 16.6). Rows 1, 2, 6 and 7 were stale: row 7 asserted a CI scope that had been false since #527, row 6 described a diet that had shipped, row 2 a swallow lock that had shipped. Every number below was re-measured on the date in its row.
**Origin:** an external deep-dive code critique (2026-08-10), assessed **~85% correct**
after verifying its factual claims against the code. Its two misfires are recorded below
so they are not "fixed" by mistake.

**Why this doc exists:** the four response plans were written and then registered nowhere —
invisible from the index, tracked only in one session's memory. That is the same
missing-is-the-signal failure the four-lens review caught elsewhere. A backlog nobody can
find is a backlog that quietly becomes untrue.

## Scorecard

| # | Critique item | Status (rows re-verified against the code 2026-08-29) | Plan / artifact |
|---|---|---|---|
| 3 | NAc docstring claims TD learning; bio-naming asserts unearned isomorphism | **DONE** — docstring fixed (#486); full audit merged (#492): 4 MECHANISM (Kuramoto oscillators, Rescorla-Wagner in `CausalLink`, cerebellar delta rule, Hebbian episode binding), 7 FUNCTIONAL, 1 NAME-ONLY. **Open:** the CI claim-lint. **Bio-lens correction LANDED 2026-08-30** (1.1.2 Cluster C): `similarity/ec.py`'s module docstring and both `ARCHITECTURE.md` EC rows now state that pattern separation is a **dentate gyrus** function and pattern completion a **CA3** attractor function — entorhinal cortex is the interface, not the separator — and that the names assert a FUNCTIONAL parallel only. The same pass corrected a second, unnamed staleness in those rows: they advertised EC as "LSH-based approximate nearest neighbour" without distinguishing the two surfaces, when substrate pattern routing (`pattern_complete_or_separate`, the path the substrate results rest on) is an **exact O(Nd) centroid scan** and only `find_similar` is LSH-backed. | [bio_docstring_truth_pass.md](archive/bio_docstring_truth_pass.md) |
| 4 | Exp 37's pre-registration was amended into something else | **STRUCTURALLY ADDRESSED, not closed.** Exp 44 relabeled EXPLORATORY; Exp 44b pre-registration written with a single frozen confirmatory test, frozen gates, and an amendment rule that demotes to exploratory. Closes when 44b runs at power. | [../experiments/protocols/exp44b_preregistration.md](../experiments/protocols/exp44b_preregistration.md) |
| 5 | n=5 with no statistics | **APPARATUS DONE, not closed.** `stats_counterfactual.py`: exact sign test on the run as unit (flips within a run are correlated), pooled binomial + Wilson CI, instrument-verified (planted effect p=3e-05; null rejection 2% at α=0.05 over 200 Monte Carlo campaigns). Closes when the confirmatory campaign runs. | `scripts/exp44/stats_counterfactual.py` |
| 2 | 1,749 `except Exception`, ~407 bare `pass` — a measurement-integrity problem | **2 of 4 stages.** Stage 1 merged (#487): 48 measurement-path sites instrumented, plus the fix that made the event actually reach the JSONL (the Stage-2 protocol greps that JSONL). **Stage 4 SHIPPED 2026-08-13 ahead of Stages 2–3** — `scripts/lint_no_silent_swallows.py` in CI holds the 16 measurement-path files at zero and grandfathers every other file at its `origin/main` count, so the total can no longer rise. Stages 2 (measure which fire) and 3 (root-cause + narrow) remain open. Measured 2026-08-29: `except Exception|BaseException` **1,793**; repo-wide bare swallows **431 in 116 files** (was 432) — deliberately out of scope for burn-down, but now ratcheted. | [measurement_path_fail_loud.md](measurement_path_fail_loud.md) |
| 1 | God functions: `run_agentic_loop` 3,298 lines / 47 trys | **NOT STARTED — and it GREW; a no-growth ratchet now holds it (2026-08-29).** Measured 2026-08-29: `run_agentic_loop` **3,546** lines / 49 `try`; `start_simulation_mode` 3,342 / 85; `_main_impl` 1,752 / 38. Decomposition is still gated behind fail-loud Stage 2 (the swallow instrumentation is the behavior-preservation detector for extraction) and is scheduled for **1.1.2**; boundaries exist (56 numbered section banners + `LoopController`). 1.1.1 item 16.4 (**#570, not yet merged as of this row's date**) adds `src/maxim/utils/function_length_baseline.json` + `tests/unit/test_function_length_baseline.py` to pin all three at their v1.1.0 spans in the fast suite; until that lands, nothing bounds their length. Re-verify this row when #570 merges. | [god_function_decomposition.md](god_function_decomposition.md) |
| 6 | CLAUDE.md is the bottleneck (~62K tokens) | **DONE 2026-08-13 (the diet), held by CI since.** ~62.5K → **~9.6K estimated tokens** measured 2026-08-29 (`len(text)//4`); the corpus split into always-loaded core + `docs/agents/<subsystem>.md` briefs + `docs/lessons/<slug>.md` archives. `scripts/lint_claude_md_invariants.py` enforces a 12K ceiling, so it cannot silently regrow — that lint is why this row can be trusted rather than re-measured by hand. | [claude_md_diet.md](archive/claude_md_diet.md) |
| 7 | CI gates almost nothing; slow tests; PyPI drift; 182 env vars | **PARTIAL — this row was FALSE and is the reason for the 2026-08-29 truth pass.** Re-verified 2026-08-29: CI runs the **whole fast suite** (`pytest tests/ -x -q -m "not slow"`, not `tests/unit/` — false since #527/D20) plus the MemoryHub integration gate, the architecture-audit gate, and eight lint/grep guards; mypy is **still 5 files**; there is **still no coverage gate**; the **PyPI drift is closed** (PyPI 1.1.0 == `pyproject` 1.1.0, and the ahead-of-PyPI policy + `scripts/lint_version_sync.py` land in 1.1.1); the **slow lane still runs nowhere** except the nightly `model-cache-tests` job, which was RED on every scheduled run from 2026-08-21 until the 1.1.1 fix (stale warm list). Env vars **189** (was 182 → 183). | 1.1.1 items 16.1/16.6 + score card [2026-08-27](../limits/score_cards/2026-08-27-claude.md) Test/CI truthfulness |

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

**Re-verify, do not re-assert (added 2026-08-29).** Four of seven rows were stale at the
1.1.1 truth pass, and the two worst described CI and doc facts that had CHANGED IN THIS
REPO — a living scorecard whose rows are not re-measured is a second honesty layer that
rots, which is precisely the failure class the 08-19 and 08-27 score cards both flagged.
Every row now carries the date its numbers were measured; a row without one is stale by
definition. Re-measure at each release cut, alongside the score card.