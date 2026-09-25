# Roadmap 1.3.x — the hardening line: 1.3.1 (fixes + guards) → 1.3.2 (decomposition)

**Drafted 2026-09-19**, the day 1.3.0 "Oasis-2" published, from the v1.3.0 blind re-score
([docs/limits/score_cards/2026-09-19-claude.md](../limits/score_cards/2026-09-19-claude.md) and its
Codex twin) plus the release-day different-reader pass. **Owner decision the same day:** ship two
patch releases before 1.4's experiments — 1.3.1 for the defects and the enforcement gaps, 1.3.2 for
the `agent_loop` decomposition and the typing scope — rather than carrying either into 1.4.

**The rule this line runs on:** the score card credits only what is ENFORCED. So every item here
ships with its guard in the same PR — a test, a lint, a ratchet, a CI lane or a required check.
An item whose guard cannot be named does not belong in these releases. Both releases are
**infrastructure only, no behavioural claim**; the 1.1.2 "Decomposition" release is the precedent.

**Why before 1.4, not during** (the decomposition half): 1.4's Phase 0 builds the trajectory
instrument and the per-step credit read directly on `agent_loop.py`; decomposing afterwards means
building that instrument twice. And a loop refactor fires the Exp 60 / Exp 61 re-run triggers by
their letter — discharging them is cheapest now, while the rig is set up and the classroom is built,
and before 1.4's campaigns exist. Refactoring *while* may-fail experiments run is what the
divergence rule warns against: a null then confounds mechanism with refactor.

---

> **The quality burndown is merged in (2026-09-20).** `quality_burndown.md` was a second,
> mutually-unaware list over this same territory; its live remainder now lives in §1.3.1 "Carried in
> from the quality burndown" below, each item re-verified against `docs/bugs/README.md` that day and
> given the guard this line's rule requires. The old file is archived at
> [archive/quality_burndown.md](archive/quality_burndown.md) as the record of Batches 0–2.

## 1.3.1 — the defects and the enforcement gaps

Grouped by the axis each item lifts; the "to reach" conditions come from the card.

### Test/CI truthfulness (C+ → B−) — the biggest lever

| item | guard that makes it count |
|---|---|
| **A gating lane that installs the `console` extra and the crypto dependency**, so the console, bundle-signing, hive-pull and Oasis-exchange tests run on every PR. Today **no lane installs fastapi or cryptography**, so those tests are skipped everywhere — and the 1.2 and 1.3 headlines both travel the signed-bundle path. | the lane itself, required in branch protection; a positive control asserting the previously-skipped modules now execute (count > 0), so the lane cannot go quietly vacuous. *Built 2026-09-25: `unit-tests` (already required) installs `console` + `sign` from `pyproject.toml`; `--require-extras=console,sign` fails any skip for a missing required extra (`tests/conftest.py`, pinned by `tests/unit/test_require_extras_lane.py`).* |
| **The nightly model-cache lane green** (red 16 nights running, 25 of the last 30 scheduled runs; new console modules missing from its skip allow-list). Fix by making a missing module FAIL rather than by extending the allow-list. | the lane's own red/green + a check that the allow-list cannot grow silently |
| **A slow lane that runs**: install `sentence-transformers` so the 24 substrate sweeps execute; replace "executed > 0" with a pinned minimum. | `scripts/check_slow_lane.py` asserting the minimum |
| **Network blocked in tests** (hermeticity is HOME/HF isolation + ~48 env scrubs today, with no block). | a conftest socket guard + a test that asserts an outbound call raises |
| **`release-build` required**, `enforce_admins` on, and a required-checks-present gate (`pr_merge_readiness.py` is manual today; the ruleset grants an always-bypass admin role). | branch-protection settings — **owner action**, not a PR |
| **The release procedure reads the nightlies**: refuse to publish while a nightly lane is red. | a step in `audit_release_build.py` or the release PR checklist, mechanized |

### Runtime correctness (C+ → B−)

| item | guard |
|---|---|
| **`AgentInstance.export_memories()` always reports 0** — it reads `self.hippocampus.memories`, which does not exist, and an `except Exception` turns the error into `0`; `AgentPool.export_all_memories` propagates it; the documented example in `docs/user/python-api.md` prints "0 memories" beside a hippocampus holding one. | a test asserting the COUNT (both current tests are vacuous: one checks the key exists, the other that it is a dict) |
| **`create.agent`'s docstring example crashes** — `capture(perception="dark cave ahead")` raises `AttributeError`; `capture` does not validate its argument. | argument validation + a doctest-style test that runs the documented example |
| **`maxim.diagnose()` and `maxim doctor --json` disagree** (diagnose reports all-passed while the CLI exits 1 on a probe diagnose never runs). | a test pinning one probe set for both entry points |
| **The silent-default swallow shape** — a handler that ASSIGNS a fallback instead of `pass`, which is what hid `export_memories` and which `lint_no_silent_swallows.py` cannot see. | extend the lint to that shape, as a ratchet on today's count (430 bare sites, 1,788 `except Exception` total) |

### Maintainability (C → C+, the cheap half)

| item | guard |
|---|---|
| **Extend the function-length ratchet to every function over 300 lines** (18 today, one of 921; the ratchet covers 3). Pin at current length, shrink-only. | `scripts/lint_function_length.py` + the baseline file |
| **A repo-wide mypy error-count ratchet** starting at today's measurement (1,050 errors in 141 files over all of `src/maxim`; CI's typed set stays at 18 files). | a new lint in CI, shrink-only |

### Research integrity + documentation honesty (both B+, cheap items)

- **Extend `lint_prereg_precedes_data.py` to `docs/experiments/*_prereg.md`** — it reads only
  `protocols/*preregistration*.md`, so **none of 1.3.0's own experiments** (Exp 60, Exp 61, R3) are
  covered; their ordering was verified by hand. Guard: the lint, with the three 1.3 experiments in
  its governed set.
- **Point the 1.2.1 surfaces at the correction** — the 1.2.1 CHANGELOG entry, `release_1_2_1.md`
  and the v1.2.1 GitHub Release body still say "end to end" with no pointer to the 1.3.0 correction.
- **Rewrite README.md for what ships** — it is the PyPI description, still calls substrate-driven
  action selection a "post-1.0 research direction", never mentions the 1.2/1.3 results, and says 16
  extras where there are 21.
- **Three smaller errors:** the release notes' `maxim substrate invalidate --drop-geometry`
  invocation is incomplete (needs `--session`, `--modality`, a tag value, `--apply`); the ledger's
  Exp 60 freeze hash names the wrong PR merge; the Exp 56 row calls amendments 3–4 pre-confirmatory
  while their headers say POST-DATA.

### Carried in from the quality burndown (merged 2026-09-20)

Re-verified against [docs/bugs/README.md](../bugs/README.md) on the merge date; anything already
closed was dropped rather than copied (**D19 was FIXED** — the architecture-audit gate exists — and
the burndown still listed it; **D84 → #796** is fixed by PR #804). Same rule as every item above:
it ships with its guard or it does not ship.

| item | guard that makes it count |
|---|---|
| **D40 remainder (was N1)** — thread `prompt_handler` through `start_simulation_mode` (the consumer, `bootstrap.build_tool_registry(prompt_handler=…)`, already exists; only the passthrough is missing). `npc_model` stays a loud `NotImplementedError` until party-mode NPC agents exist — that half is a mechanism, not a defect. | extend `tests/unit/test_api_expansion.py::TestCampaignParametersAreThreadedOrRejected`: a passed handler is the one the run's prompts reach |
| **D32** — load the foundational preamble from `CONSTITUTION.md` as package data (pip users get an empty preamble today) | a drift test: packaged copy == repo-root `CONSTITUTION.md`, and a wheel-install test that the preamble is non-empty |
| **D49** — benchmark honesty: apply-or-delete `weight`, fix the running half-mean, drop-or-ship the missing tier2/tier3 suite files (`simulation/benchmark.py`) | a unit test per promise: a weighted suite's aggregate moves with `weight` (or the key is rejected), and every suite file the format names loads |
| **D46 + D50** — delete the dead percept-transport reference (`simulation/sources.py`); warn on the inert `party_mode` / `choice_resolution` keys in `load_campaign` and drop the dead schema field | a test that loading a campaign carrying either key WARNS once |
| **D63** — a PR against a non-`main` base runs no required checks | a ruleset/branch-protection change (owner action) + `scripts/pr_merge_readiness.py` reporting it; the guard is the gate existing |
| **Fail-loud Stage 3** — narrow the measurement-path swallows; green-lit since Stage 2 measured **zero** firings ([deferred/measurement_path_fail_loud.md](deferred/measurement_path_fail_loud.md)). Must not land mid-walk on a branch a graduation run reads from. | `scripts/lint_no_silent_swallows.py`'s zero-total set grows to cover each narrowed file |
| **Sandbox defects #800 / #801 / #802** (found 2026-09-20 beside #796: Python scripts never run; raw-prefix containment; path executed instead of approved content) — latent, no in-repo caller wires the sandbox | each issue's own red gate |
| **L8 record-stamping** (stamp model / endpoint / n_ctx / quantization on every run record) — Exp 44b's prerequisite; status **not re-verified** on the merge date, check before starting | a test that a run record without those fields is refused by its writer |

Already covered above, so not duplicated: `mypy` scope (the ratchet in this section), god-function
decomposition (§1.3.2). Behavioral-suite thickening for Exp 52/53b/56 has no nameable guard as
stated and is left out by this line's rule, not forgotten.

### Not in 1.3.1

The decomposition beyond nothing, full mypy coverage, and any new mechanism. The ratchets above
stop both from getting worse, which is the point of doing them first.

---

## 1.3.2 — the decomposition

**One target: `agent_loop.py`** (5,348 lines; `run_agentic_loop` 3,484). `start_simulation_mode`
(3,324) and `_main_impl` (1,747) stay pinned by the ratchet for a later pass — naming them here
would repeat the "kicked down the road" pattern this release exists to end.

**Behaviour preservation is the gate, not an aspiration.** Every slice must keep green, in the same
PR: the byte-identical-selection provenance test, the encoder golden pin, and an offline
reproduction of the committed Exp 60, Exp 61 and R3 verdicts from their data. A slice that cannot
show all three does not merge.

**Typing rides along, scoped:** every module the decomposition creates enters CI's mypy set. The
repo-wide ratchet from 1.3.1 holds the rest. Full coverage is not promised.

**Close it honestly:** a trigger walk over the ledger (Exp 60 and Exp 61 both name
`run_agentic_loop`'s idle-gate and autonomy handling) and a live re-run of Exp 60 on the rig
(5 seeds per arm, ~1 h) to discharge them with a dated annotation, rather than an argument that a
pure extraction changes nothing.

**Sizing, honestly:** this is the largest item on the list — several sittings of work in reviewed
slices, plus the rig hour. If a slice stalls, the release ships the slices that landed; the ratchet
records the new ceiling either way.

---

## Sequence

```
1.3.0 (published 2026-09-19)
  → 1.3.1  defects + guards        (each item ships its guard; owner action: branch protection)
  → 1.3.2  agent_loop decomposition + typing scope, triggers discharged
  → 1.4    Phase 0 instrument on the decomposed loop, then Exp 62 → E1 → …
             ([roadmap_1_4.md](roadmap_1_4.md))
```

Exp 62 RAN and is EARNED (2026-09-20) — it never depended on either patch release, and its result is the 1.3.1-line content this file's §1.3.1 ships alongside. Any further rung likewise runs on the rig in parallel
(`docs/experiments/exp62_pressure_interoception_prereg.md`, decisions D1–D4 taken).

## Cadence

Re-score at the 1.4 cut, or when an axis's "to reach" condition is claimed complete — and the claim
is that the guard exists, not that the work was done.
