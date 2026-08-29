# pymaxim 1.1.1 — "Enforcement"

**Released 2026-08-30.** `pip install --upgrade pymaxim`

Every item in this release is a lint, a CI step or a test. Nothing new was built; several
things the repository *claimed* about itself became mechanically checkable, and four defects
that had been invisible were found in the process.

The 1.1.0 repository re-score ([score card](https://github.com/dennys246/Maxim/blob/main/docs/limits/score_cards/2026-08-27-claude.md))
graded on one rule — **a grade moves only when the normal workflow enforces the improvement**.
This release answers that rule for six of its axes (roadmap [item 16.1–16.9](https://github.com/dennys246/Maxim/blob/main/docs/plans/roadmap_1_1_to_1_3.md)).

## Research integrity

- **Pre-registration precedes data, checked by CI.** [`scripts/lint_prereg_precedes_data.py`](https://github.com/dennys246/Maxim/blob/main/scripts/lint_prereg_precedes_data.py)
  asserts, for every `docs/experiments/data/` entry whose experiment has a pre-registration,
  that the prereg's (and each PRE-DATA amendment's) first commit on `main`'s first-parent
  chain precedes the data's first `ts`. Exp 53/53b landed a pre-registration in the same
  squash as its data on 1.1.0 release day; this is the check that would have failed it.
- **Harnesses refuse to write a gated record from a dirty tree.** Both families — the
  in-process robot harnesses and the sub-sim spawners — exit 3 unless `--allow-dirty`, which
  then stamps `allow_dirty: true` into every record so a write-up cannot omit it. Stamping
  was detection; refusing is enforcement.
- **EARNED ledger rows must cite committed data** or carry a dated data-lost annotation.
- The clean-tree rule surfaced **two dirty-tree records nobody had disclosed** (the H1 Part C
  block cited by the Exp 45 row, and Exp 54's Phase B target declaration). Both are
  disclosed on the rows that cite them, grandfathered by explicit list with reasons, and
  printed as still-failing on every CI run.

## Repository honesty

- **The `os.replace` confession was stale in both directions.** CLAUDE.md had admitted "17
  hand-rolled sites, detection-only" — a text grep that counted comments and saw one spelling
  of four. The real count is **12 call sites in 12 files**, now printed by
  [`scripts/lint_atomic_io_ratchet.py`](https://github.com/dennys246/Maxim/blob/main/scripts/lint_atomic_io_ratchet.py) on every CI run,
  with a per-file ratchet so it can only fall. CLAUDE.md cites the CI output, not a number.
- **Four rows of the living critique scorecard were false**, two of them about this repo's own
  CI and documentation. Every row now carries the date its numbers were measured.
- **The version policy is written down and enforced**: `main` is ahead of PyPI, the bump
  happens in the release transaction, and the three version lines name the version and link
  PyPI instead of describing what it serves.

## Release governance

- [`scripts/audit_release_tags.py --check-releases`](https://github.com/dennys246/Maxim/blob/main/scripts/audit_release_tags.py) audits
  the Release objects themselves: every version PyPI serves needs a tag, a Release, the exact
  wheel + sdist with sha256 matching PyPI, and notes with absolute links.
- The dead duplicate `docs/CHANGELOG.md` (frozen at 0.3.0) is gone.

## Test and CI truthfulness

- **Every diff-scoped lint had been a no-op on pull requests** ([D37](https://github.com/dennys246/Maxim/blob/main/docs/bugs/README.md)).
  The lint job checked out at depth 1, so no merge-base existed: the multi-agent marker lint
  and the no-silent-swallows ratchet had been skipping on *every* PR since they shipped. Full
  history is fetched now, and a missing base ref is a hard error on a pull request.
- **The nightly model-cache lane had failed every scheduled run since 2026-08-21** — the warm
  list was a hand-kept comment missing a model, and the cache key did not include the list.
  The list is derived from source, hashed into the key, and warmed unconditionally; an
  explicit skip allow-list replaces the "at least 12 executed" floor.
- **A `fix` that touches `src/` must ship a test**, checked against the PR title (the subject
  that actually lands on `main`) as well as the branch commits.
- **The three god functions cannot grow**: their v1.1.0 AST spans are pinned, and shrinkage
  must tighten the ceiling in the same commit so the baseline never overstates the debt.

## Runtime correctness

- **The public API's own shutdown produced unloadable state** ([D41](https://github.com/dennys246/Maxim/blob/main/docs/bugs/README.md)).
  `create.agent → mutate → shutdown → load.agent` wrote only `hippocampus.json` + `nac.json`
  and dropped EC, SCN and ATL, so the next load warned "Half-present NAc/EC pair" on the API's
  own output. The instance now opens the session it later closes — on the hub it actually
  keeps, which the bio-stack construction path swaps in.
- **The runtime never persisted SCN at all** ([D42](https://github.com/dennys246/Maxim/blob/main/docs/bugs/README.md)): `build_bio_stack`
  built a pathless `SCN()`, so every runtime agent silently lost its temporal signatures
  between sessions.
- **`api.campaign()` stops accepting arguments it ignores** ([D40](https://github.com/dennys246/Maxim/blob/main/docs/bugs/README.md)).
  `interactive=` is honoured and tri-state; `npc_model=` and `prompt_handler=` raise
  `NotImplementedError`. **This is a deliberate breaking change in a patch release** — code
  that passed either worked on 1.1.0 and raises now, because a documented argument with no
  observable effect is a worse contract than a loud refusal. See
  [stable_api.md](https://github.com/dennys246/Maxim/blob/main/docs/user/stable_api.md).
- **D6 and D9 are marked dormant** rather than half-implied: the mechanisms stay wired, no new
  work builds on them, and the resurrection trigger is named (the 1.3 fabric). The D9 drive
  emitter, malformed for its whole life behind a `log.debug` swallow, is repaired and loud.

## Upgrading

`pip install --upgrade pymaxim`. The only behaviour change a caller can hit is
`api.campaign()`'s two rejected parameters; everything else is repository machinery.

Full detail: [CHANGELOG.md](https://github.com/dennys246/Maxim/blob/main/CHANGELOG.md#111---2026-08-30--enforcement).
