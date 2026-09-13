# The four-lens experiment-design review (run BEFORE building the harness)

**Why.** Our two-lens *code* review catches implementation bugs, but it reads the harness *after*
the design is set. The deepest failures — the causal-link confound, cluster-generalization, the
floor-into-cluster contamination (R2 learned-bias, 2026-09-12) — were **design** flaws, understood
only *after* building v1, reviewing it twice, and running a dry-run. A **pre-build design review**
catches them before a line of harness. An experiment costs hours of compute and a possible
retraction; four cheap review agents up front is a great trade.

This is a DISTINCT gate from the code review. The full experiment pipeline:

    design review (4 lenses, reads the PREREG)  →  build harness  →
    code review (execution + architecture, reads the CODE)  →  dry-run / pilot  →
    confirmatory run  →  data PR

## The four lenses

Each is a SEPARATE parallel reviewer (the value is a different reader, as with the code review).
Each reads the pre-registration (+ the cited briefs) and returns findings ranked
**DO-NOT-BUILD / SHOULD-FIX / NIT**, plus what it verified. The lenses, and the failure each exists
to prevent:

- **Confounding lens** — *does the metric isolate the claimed cause?* Right controls/ablations, a
  statistic matched to the baseline, no alternative explanation (prior, causal link, repetition,
  drift, floor). Ask: "could a positive OR a null arise for a reason other than the claim?" *(Would
  have caught the causal-link + floor confounds.)* The single most valuable lens.
- **Bio-faithful lens** — *does it test the mechanism's REAL job, not a caricature?* The manipulation
  respects how the substrate/body/drives actually work; the metric measures the mechanism's actual
  purpose. Reads the owning `docs/agents/` brief. *(Would have caught "the credit's job is
  state-conditioned selection, not eat-sooner.")*
- **Wiring lens** — *real consumers + real credit path (D43), right encoding/seams, no hand-composed
  shortcut that passes while the loop fails.* Reads `docs/wiring/`. *(The shipped-the-pieces family.)*
- **Environment lens** — *does the world game-natively afford it (D1 — no synthetic sensor/reward),
  are the needed states/acts reachable + measurable, does the bridge/world behave?* *(Would have
  caught hunger-drains-too-slowly, food-caps-at-20, the eat-lag.)*

The design lenses are **fed by `docs/wiring/`** — that is the loop that stops us re-discovering: the
confounding lens now opens `substrate-learning-channels.md` and catches the causal-link trap on
sight. Every design review that discovers something new adds a `docs/wiring/` entry.

## When to run (proportional, like the code-review rule)

- **Full four** — a NEW experiment design or a NEW claim.
- **Floor (confounding + bio-faithful)** — a derivative variant / changed metric / new controls.
- **Skip** — a pure re-run with no design change (more seeds/N only).

## Per-experiment layout + the synthesis step

Each experiment is a directory `docs/experiments/<slug>/`:

    docs/experiments/<slug>/
      prereg.md                 # the (draft, then frozen) pre-registration
      rationale/
        confounding.md          # one file per lens — the reviewer's findings, PRESERVED
        bio-faithful.md
        wiring.md
        environment.md
      outcome.md                # dated, after the run

Each lens's review is written to `rationale/<lens>.md` verbatim (preserve the reasoning — it is why
the design is what it is; future experiments read it). **Then the main session** (this Claude) reads
all four, resolves/folds the findings into the prereg, and presents the **cohesive experiment plan**
— what changed, what was dismissed and why, the go/no-go. Only then do we build the harness.

A DO-NOT-BUILD from any lens blocks the build until folded or explicitly overruled with a recorded
reason (the same posture as a code-review DO-NOT-SHIP).

## Regression guard

Process invariant — enforced by author + reviewer attention; no automated test (like the plan
review-round discipline). The mechanically checkable part is "a new `docs/experiments/<slug>/` with a
frozen prereg has a populated `rationale/`"; tracked as convention. Motivated by the R2 learned-bias
v1→v2 cycle, where the absence of this gate cost a full build-review-dry-run before the design flaw
was understood.
