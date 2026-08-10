# Bio-Docstring Truth Pass — Say What the Algorithm Is

**Status:** DRAFT (2026-08-10); NAc already fixed in the kickoff session
**Motivation:** External critique (2026-08-10), point 3, verified: `decisions/nac.py` line 2
claimed "temporal difference learning" over an eligibility-trace proportional-credit scheme —
a ninety-second grep-and-dismiss for any reviewer. Bio-inspired naming is load-bearing for
the mental model (CLAUDE.md rule: don't rename), but a docstring asserting an algorithm the
code doesn't implement converts inspiration into a credibility liability. Note the critique
also MISFIRED once (accused Angular Gyrus of being "Hebbian binding = outer product"; it is
actually an exact-math engine with an honest mapping paragraph) — which is why this pass
audits against the code, not against the critique.

## The tag (module-level analog of the engineering/behavioral two-tier)

Every bio-named module docstring gains a `Bio-mapping:` block declaring one of three levels:

- **MECHANISM** — the algorithm matches the cited biological model (e.g. Cerebellum's
  "Rescorla-Wagner style" error-proportional delta rule, if verification holds).
- **FUNCTIONAL** — the *role* matches, the algorithm differs; the block names BOTH the
  biological story and the actual algorithm (NAc's new docstring is the template).
- **NAME-ONLY** — the name is a mnemonic for the mental model; no algorithmic claim.

Rule going forward (one line added to CLAUDE.md working principles): a bio-named module may
not cite a named algorithm ("TD learning", "pattern separation", "Hebbian") in its docstring
unless the implementation earns MECHANISM for that claim.

## Audit list

| Module | Current claim | Suspected level |
|---|---|---|
| decisions/nac.py | fixed 2026-08-10 | FUNCTIONAL (done) |
| embodiment/cerebellum.py | "Rescorla-Wagner style" | verify → MECHANISM or reword |
| similarity/ec.py | pattern_complete_or_separate | verify: threshold split vs DG/CA3 story → likely FUNCTIONAL |
| memory/hippocampus*.py | episodic capture/consolidation/replay | FUNCTIONAL |
| memory/episode.py::apply_hebbian_on_close | "Hebbian" | verify: co-activity-driven weight change may earn MECHANISM |
| memory/sleep_replay.py | replay consolidation | FUNCTIONAL |
| memory/atl.py | semantic integration | FUNCTIONAL (docstring already modest) |
| math/angular_gyrus.py | arithmetic-fact retrieval | already honest; add tag only |
| time/scn.py + time/oscillator.py | circadian pacemaker | already says "Inspired by"; tag NAME-ONLY/FUNCTIONAL |
| proprioception/pain_bus.py, attention/*, default_network/* | various | sweep + tag |

## Method

One PR, docstrings only, zero behavior. Per module: read the mechanism, write the tag, and
where a named algorithm is claimed, either verify it or replace with the actual algorithm's
name. Grep sweep for remaining named-algorithm claims:
`grep -rniE "temporal.difference|reward prediction error|hebbian|pattern separat|rescorla" src/maxim/`
— every hit must be inside a module whose tag earns it, or in a comment explicitly marked as
the biological *analogy*.

Tie-in: the graduation ledger already tracks which *behavioral* claims are Earned; this tag
tracks which *algorithmic* claims are earned. They are orthogonal (a FUNCTIONAL module can
hold an Earned behavioral claim — Exp 42's NAc discrimination doesn't require TD).

**Regression guard:** the grep sweep above lands as a CI step with an allow-list of
(file, claim) pairs that passed verification; new named-algorithm claims outside the
allow-list fail CI.
