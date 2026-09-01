# pymaxim 1.1.2 — "Decomposition"

**Released 2026-08-31.** `pip install --upgrade pymaxim`

1.1.1 turned the repository's claims about itself into lints. 1.1.2 does the thing 1.1.1
deliberately did not — it splits a god function — and it closes four gates that **could not
fail**, three of which were discovered by running them for the first time.

## Fail-loud Stage 2 finally ran, and the answer is zero

[`docs/plans/god_function_decomposition.md`](https://github.com/dennys246/Maxim/blob/main/docs/plans/god_function_decomposition.md)
gates every extraction on *"zero new `swallowed_exception` firings vs the Stage-2 baseline."*
**Stage 2 had never been run, so that baseline did not exist and the gate could not fail.**

It exists now: **zero firings across 75,654 log records**, both modes, all 50 instrumented
sites silent. Artifact and both raw captures are committed at
[`docs/experiments/data/fail_loud_stage2/`](https://github.com/dennys246/Maxim/tree/main/docs/experiments/data/fail_loud_stage2).
Per the plan's own reading that is the good branch of its informative-either-way outcome: the
swallows are dead defensive weight, and **Stage 3 is green-lit**.

The instrument was verified live in both captures *before* the zero was trusted — which
mattered, because the plan's own instruction ("grep the JSONL for `swallowed_exception`") does
not work as written. The log handler emits through `to_compact()`, which keys the event `"e"`
and flattens the payload, so a parser written against the call site's shape reads a capture
full of firings as **zero** and prints a pass.

## The first extraction

`run_agentic_loop` went from **3,546 to 3,512 lines**: sections 7 (step callback) and 8.5
(NAc per-tick maintenance) are now module-level `_loop_step_callback` and
`_loop_bio_tick_maintenance`, exercised by tests that call them directly.

Mechanical only. The aliasing claim was checked by AST *before* the move rather than assumed,
and section 8 was deliberately left alone — it contains a `break`, so extracting it would be a
semantic transformation rather than a move. The no-growth baseline tightened in the same
commit, as its own guard requires.

## Four gates that could not fail

- **Nothing in CI had ever built a wheel.** On 2026-08-30 two defective wheels passed
  `twine check`: one with **zero** Console UI files (a worktree build — the bundle is
  gitignored and vendored at release time), and one carrying a stale **1.1.0** version. A new
  `release-build` job now builds wheel *and* sdist and asserts contents and version
  ([D47/D48](https://github.com/dennys246/Maxim/blob/main/docs/bugs/README.md)).
- **The D12 guard failed by hanging.** It asserts a bound *after* the call returns; against the
  pre-fix code the call never returns, so the assertion was never reached and the job died on
  its runner timeout — reported as infrastructure, not as the defect. It now fails by assertion
  in 30 seconds, naming the test.
- **41 `@pytest.mark.slow` tests ran nowhere.** The marker had been applied across ~10 files
  for months while the only mention of it in CI was the fast suite's `-m "not slow"` — a
  *deselection*. There is now a nightly lane, with an `executed > 0` assert, because
  `pytest -m slow` exits 0 both when it collects nothing and when everything skips.
- **A post-tag `src/` change must declare itself** (roadmap item 16.10) — bump the version or
  add an `[Unreleased]` line.

## Honesty work

Every Tier-3 row the 2026-08-27 score card named now carries a disposition — 2 Dormant,
12 Dropped, 2 moved to Tier 2, and **zero EARNED**. Three of the Dropped are stronger than
"never tested": their predicates ran and *failed*.

`README.md` advertised **17** Python-API verbs; the real surface is **21**, wrong through two
releases because the only place the two claims met was a human reading both. A test now checks
them against each other.

The `ARCHITECTURE.md` EC rows and the `similarity/ec.py` docstring are corrected at both ends:
pattern separation is a **dentate gyrus** function and completion a **CA3** function —
entorhinal cortex is the interface, not the separator — and substrate pattern routing is an
**exact O(Nd) centroid scan**, not the LSH path the docs advertised.

## The release rule changed

The ≥1-day tag wait was a proxy for *a second reading by someone not carrying the release*.
All three of 1.1.0's provenance failures were human-judgment failures over **honestly-stamped**
artifacts, so attaching artifacts more firmly to commits would have caught none of them. The
rule is now **structure OR time**: data PR, then a separate later interpretation PR reviewed by
a different reader — with the wait kept as the fallback, never as an extra hurdle on top.

## What the review round caught

Two lenses cross-confirmed the worst finding: the new Stage-2 tool wrote to a gated path with a
hand-rolled dirty check, so it *stamped* the flag and continued — the same shape as the Exp
53/53b incident, on the branch whose own process work rewrites that rule. Fixed at the root
(the tool now refuses) plus a third lint family keyed on **where** records land rather than how
a harness runs, because the script had escaped both existing families.

A second round on the fold found the first fix was **narrowed, not closed** — any well-formed
JSONL still passed — and that the headline fix had shipped without a guard test.

Full changelog:
[CHANGELOG.md](https://github.com/dennys246/Maxim/blob/main/CHANGELOG.md#112---2026-08-31--decomposition)
