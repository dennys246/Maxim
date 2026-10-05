# Wire integrity: why code review has a third lens

**Adopted 2026-10-05** (owner proposal, after #851, #873 and #908–#910 surfaced in one session). The charter is
[../CODE_REVIEW.md](../CODE_REVIEW.md). This file keeps the synthesis it was designed from. The source data, two
independent passes over the history with provenance per row (one through the lessons, bugs ledger and CHANGELOG, one
through issues, PRs and git, pickaxe-dated where marked **V**), is in
[wire-integrity-review-incidents.md](wire-integrity-review-incidents.md): 30 + 39 table rows (about a dozen incidents
appear in both, and several rows group related incidents).

## What the history shows

These are incidents, so every one is a de-wiring nobody caught in time; the record cannot show how many a round did
catch. Read the figures as "the gap exists and is expensive", not as a catch rate.

- **Detection latency:** median about 5 months, range about 1 day to 7.5 months.
- **All eight sampled breaking changes with a recorded executor + architecture round missed the de-wiring in their
  scope** (#411, 5a1dd499, 8a09ae85, 4619e941, 60702417, #1029, the #985 fixes #973/#983/#986, the 1.2.1 pairing
  PRs; 8a09ae85's round covered its results, not its code; #1029 fixed one site of N). In **#908 the review folds
  created the vacuous saves**
  ("Arch #2" and the executor lens each added a `save_cerebellum()` call that guarded a no-op). In **#851 both lenses
  saw the any-pending guard** and folded a doc note.
- **The catches that did happen came from readers looking outside the diff**, mostly the executor lens at code next to
  an unrelated change (D37, D77, D79, #972, #851, #873).
- **Fixes whose reader found an older defect next door** (the strongest evidence that looking outside the diff pays):
  #839 → #840/#841, #1083 → #1085, #863 → #870 → #871 → #873 → #1093, #971 → #982 (origin f22d9236); #1047's run
  exposed #1052 (origin April).
- **A fix that created the next de-wiring**, caught by the following PR's round: #973/#983/#986 → #985 (verified in
  the history pass).
- **Test-only callers masked zero production callers** (D58, D43, #908), and test fakes hid wrong field names
  (`FakeEpisode` has `.content`, so #845's readers passed).
- **A blind backtest** applied the charter, cold, to 4619e941 and 60702417 as of their dates. It found #908's
  load/save split (probe 5) and #851's leaked-entry guard (with a stronger trigger than the issue: no production
  executor had a pain detector, so every failed tool leaked an entry). Its gaps became probe 10 (paired lifecycle),
  probe 6's runtime state, probe 1's rule that a caller of a no-op mechanism is not a connection, probe 8's
  constructor list and the off-path claim rule.

## The shapes, ranked (both passes combined)

| Rank | Shape | Contract types | Examples |
|---|---|---|---|
| 1 | Producer or consumer never connected: zero production callers, a key read with no writer, a topic with no producer | call, payload, topic | 1.2.1 pairing announcer, #909, #817, #822, D43, D45, D9, #1052, #1084, D83, D73 |
| 2 | A builder collaborator or flag reaches 1 of N sites, or the wired object is swapped out after wiring | builder, gate, lifecycle | D77 → D79 → D86, D41, D42, #972, D28, PainBus (3 of 4 CLI paths), MemoryHub `.connect()`, `build_executor` ×6, #1042 (1–3), db5a8768 |
| 3 | Call-contract drift masked by `except` or a `getattr` default | call, payload | D58 (Cerebellum never trained, about 5 months), #840, #841, D60, #861, #991, #993, #845 |
| 4 | Persistence asymmetry: load/save path split, a gate at only some restore sites, restore then overwrite, write without read | persisted state | #908, #982, D42, D28, #972, #939, #985, D2 |
| 5 | Outcome channel collapsed: fallback or stub reports success, polarity squashed, return value ignored | call (return), learning signal | D53, D56(f), #870, #873, discarded `side_effects` |
| 6 | A default or constant flips a mechanism off; a graceful skip looks like success | gate | #1085 (`plan_approval` killed the only `approve` caller), #889 (switch on a dead path), #596, Exp 60's default autonomy, D37, the logging-before-dispatch lesson |
| 7 | Producer replaced or moved, consumer or guard left on the old path | call, gate | #889, #910 (annotators read the pre-clamp store), 7eb77e0f, the mutable-globals lesson |
| 8 | Identity key missing or mismatched | identity | D54/D57 (`Reaction` without `agent_id`), Exp 32 Bug A, per-agent stashes, #1042 tier namespace |
| 9 | Duplicated logic or two resolution paths drift | any | sim-n-ctx, D82, D85, D32, #1093, #1042 |
| 10 | Migration strands callers outside CI | call | D68 (`geometry` required; a script broke unseen) |
| 11 | Merged diff ≠ reviewed diff | process | PR #395, PR #435 (M4) |

**The bio-specific shape, stated generically:** a learning signal must reach its store. Every outcome a component
observes must reach every learner that claims to learn from it, with identity and polarity intact, through the
production entry point (D53, D54, #851, #889). The engram ladder generalises it to any state: written → keyed → read →
acts.

## Diff signals the lens keys on

A parameter added, removed, renamed, made required or re-defaulted; a return type or field changed; a symbol or
module moved or deleted; any builder, factory, `connect` or session/tick hook; a persistence path, key or format; an
env var, config key, default constant or set membership; a new or changed event type, topic or payload field;
`success=True` in an `except` or fallback branch; a new `except` or `getattr` default around an in-tree call; a new
public symbol (it needs a caller); string-key literals shared across modules; an object constructed and later
replaced; prompt, annotation or tool-schema text; an identity or key format; a release note saying "end to end".

## What can be mechanized (backlog M37)

M37 in [../plans/outstanding.md](../plans/outstanding.md) holds the rows: (a) a check that the lens ran at all (a
PR-body table or empty-table statement against the head SHA); (b) a caller diff for new or changed public symbols (it
subsumes M3's new-symbol half; M3's fixes-an-issue half stays on M3; #1106's AST caller scan is reusable); (c) an AST
check that every statically resolvable call site's keyword arguments exist in the callee's signature across `src/` +
`scripts/` (d916aea9's guard, generalised); (d) a check that a `getattr(x, "lit", d)` literal names an attribute some
class defines; (e) persistence symmetry (every store a builder loads has a save path, and no re-init after
`load_state`); (f) builder-kwarg completeness (a `None`-defaulted builder kwarg passed at some but not all non-test
call sites fails unless marked `# optional-collaborator:`; the structural fix stays a required keyword-only
collaborator, as #986 did).

Ideas the passes raised that are **not on the backlog** (no row yet; add one before relying on them): a string-key
producer/consumer cross-reference, ablation-flag reach (the flag guards a function with a production caller), and
widening mypy further (it caught #1083). Judgment stays with the reader: outcome polarity, object swaps, and whether a
constant change starves another branch.

## Sources

[wire-integrity-review-incidents.md](wire-integrity-review-incidents.md) holds both passes' tables, with V/I or
doc/inf provenance per row and issue links.
