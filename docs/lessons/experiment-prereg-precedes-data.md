# Gated experiment records come from a CLEAN tree at a commit REACHABLE from `main`, and the pre-registration is ON `main` before the first data record

**Written 2026-08-27** from the 1.1.0 repository re-score
([docs/limits/score_cards/2026-08-27-claude.md](../limits/score_cards/2026-08-27-claude.md)),
which dropped Research integrity from A− to B− on one finding. The enforced rule survives
as a stub in [docs/agents/simulation-experiments.md](../agents/simulation-experiments.md)
§3; this file preserves the full narrative, timeline, root cause, and the prevention set.

---

## The incident — Exp 53 / 53b (2026-08-26, release day)

Exp 53b is the 1.1.0 headline hardware result: nursery-taught `aut_nac.json` +
`aut_ec.json` from the sim drive the physical Reachy toward a speech source 36/36, ledger
row `behavioral_graduation_candidates.md` L193 **EARNED**. Its provenance, as stamped by
the harness itself:

- every `start` record in `docs/experiments/data/53_cross_context_readout.jsonl` (2) and
  `53b_cross_context_readout.jsonl` (3) carries `executed_git_hash: "68f9026e268f"` and
  `working_tree_dirty_src_scripts: true`;
- `git merge-base --is-ancestor 68f9026e v1.1.0` → **not an ancestor**. The commit is a
  branch object that the squash-merge `617b1625` (#551) made unreachable;
- neither `53_cross_context_readout.md` nor either pre-registration contains the word
  "dirty"; the write-up cites the hash as if it identified the code.

Timeline (UTC, from record `ts` fields vs `git show -s --format=%ci`):

| time | event | where |
|---|---|---|
| 13:36 | Exp 53 pre-registration commit `0570aa8e` | branch only (dangling after squash) |
| 14:13 | harness + amendment 1 commit `68f9026e` | branch only (dangling after squash) |
| 15:04 | first Exp 53 data record | dirty tree at `68f9026e` |
| 15:57 | first Exp 53b data record | dirty tree at `68f9026e` |
| 17:05 | Exp 53 pre-registration reaches `main` (`6e7530c6`, #550) | 2 h AFTER first data |
| 18:27 | squash `617b1625` (#551): amendment 2 + 53b pre-registration + 53 data + 53b data + RESULT | first and only appearance of the 53b prereg and amendment 2 on `main` |
| 18:47 | tag `v1.1.0` + PyPI upload + GitHub Release | same afternoon |

So: the harness told the truth, the write-up dropped it, the squash flattened the time
axis that was the evidence of freeze-before-data, and the ledger row went EARNED the same
afternoon the tag was placed. The 53b "one declared change" (δ 0.55 → 0.30 rad) was
chosen after seeing Exp 53's misses and justified against "the azimuths Exp 53 actually
delivered" — disclosed, principled, and still a same-session parameter change whose
pre-data status rests on self-attestation because no commit object for it precedes the
data anywhere.

Contrast **Exp 52** (the same week, a sim): pre-registration `b9398ded` → harness +
amendments `e367f526` → Phase A data `60195a29` → Phase B data `732386ef`, four separate
PRs, all on `main`, all 36 Phase B rows stamping a hash reachable from the tag. Exp 52
passes every rule below today.

## Root cause

1. **The provenance guard was scoped to the wrong door.**
   `scripts/lint_harness_provenance.py` matches harnesses that *spawn* `maxim` as a
   subprocess; `assert_repo_interpreter` exits 3 on an interpreter mismatch. The Exp 53
   harness (`scripts/orient_backbone/exp53_cross_context_readout.py`) imports `maxim`
   in-process and drives the robot directly — outside the regex, so nothing asserted a
   clean tree or refused to run. `executed_code_provenance(...)` *stamped* the dirty flag;
   stamping without refusing is detection, not enforcement (the same shape as the
   `os.replace` "KNOWN GAP").
2. **Hardware sessions run on a live branch.** A five-hour robot session with the branch
   evolving under it: a miss (Exp 53 APPARATUS, 27/36 = 0.75 < 0.80), a diagnosis, an
   edited pre-registration, a re-run — all before anything reached `main`. Sims don't
   have this problem because their pre-registrations land as their own PRs.
3. **Squash-merge destroyed the audit trail.** #551 folded prereg + amendment + harness +
   data + RESULT into one commit. The repo already has this lesson on the code side
   (2026-07-29, fold commits lost to a squash — [review-round-discipline.md](review-round-discipline.md));
   this is the research-side version: the reviewed diff and the merged diff differ in
   *when* things happened, and squash flattens time.
4. **Release-day pressure.** Exp 53 was roadmap item 19, a 1.1.0 gate, run on the day of
   the cut. Nobody re-opens provenance on the gating result on release day. The 08-19 A−
   was earned on Exp 42b and Exp 48 — retractions and held misses made *with time*.

The Exp 42b corollary applies unchanged: **a result whose code-under-test cannot be
established is not a validation, independent of whether it happens to be correct.** The
53b data may well be fine — `68f9026e` became `617b1625`, and the dirty changes were
probably harness edits that ended up in the squash — but "probably fine" is exactly the
argument the 42b lesson forbids.

## Prevention set (ranked by enforcement bought per unit of work)

Each rides existing infrastructure — no new mechanism. Tracked as roadmap 1.1.x item 16
sub-items 7–9 ([roadmap_1_1_to_1_3.md](../plans/roadmap_1_1_to_1_3.md)); the two
process rules live in the brief stub and the publication guide.

1. **Harnesses REFUSE on a dirty tree, not just stamp it.** `scripts/_provenance.py`
   already computes `working_tree_dirty_src_scripts`. When a harness writes a *gated*
   record (anything under `docs/experiments/data/`) from a dirty tree it exits 3 unless
   `--allow-dirty` was passed — and then writes `allow_dirty: true` into every record so
   the write-up cannot silently omit it. Extend `lint_harness_provenance.py`'s regex to
   `scripts/orient_*/` so in-process harnesses must call the helper. Guard: a unit test on
   the refusal. (Item 16.7)
2. **Pre-registration precedes data, checked by CI.** For every
   `docs/experiments/data/<N>*.jsonl`, find the pre-registration named in
   `docs/experiments/<N>_*.md` and assert `git log --format=%ct --diff-filter=A -- <prereg>`
   on `main` < the file's first `ts`. Amendments follow the same rule for the data they
   govern. Fails the PR that lands data before its prereg reached `main`. Exp 52 passes
   today; 53b would have failed at 18:27 on release day instead of getting an EARNED row.
   **If only one item ships, ship this one** — it turns "was the gate frozen" from a
   self-attestation into something CI answers. (Item 16.8)
3. **No squash-merge for PRs carrying `docs/experiments/data/` or `protocols/`.** Squash
   flattens time and time is the evidence. Merge-commit or rebase-merge so branch commit
   objects stay reachable. Mostly moot once #2 lands (the prereg is already on `main` as
   its own commit), so this is a process line in the brief, not a ruleset.
4. **EARNED ledger rows require a data citation.** Add
   `docs/plans/behavioral_graduation_candidates.md` to `lint_claude_md_invariants.py`'s
   doc set with one rule: any row whose status matches `EARNED` must carry a
   `Regression guard:` link that resolves and includes a path under
   `docs/experiments/data/` or a dated data-lost annotation. L185 (EC pattern completion)
   and L186 (SEM pain → NAc) fail today; that is the point. (Item 16.9)
5. **The tag waits — REVISED 2026-08-30 to "structure OR time", see below.** A gating
   experiment on hardware on the day of the cut means every judgment call is made under
   release pressure. The release gate is "the result is on `main` with its data and its
   pre-registration precedes it". Exp 54 Phase B/C is the next chance to practice:
   prereg PR → data PR → ledger PR, tag nowhere near it.
   (Publication guide §"Tag the released commit")
6. **Tier-3 dispositions** are a one-hour doc sweep already owned by 1.2 gate 8(d).

### The ≥1-day wait, reconsidered (decided 2026-08-30)

**The wait is a proxy, and naming what it is a proxy FOR gives a better rule.** The
original prevention item said "the tag waits at least a day". A day is not the thing that
was missing on 2026-08-26. What was missing was **a second reading of the interpretation by
someone not carrying the release**. Time was standing in for fresh eyes, and it is a crude
stand-in: a day of waiting in which nobody re-reads changes nothing, and a same-day review by
a different reader would have caught what a day of silence did not.

That matters because of what actually failed. **All three 1.1.0 provenance failures were
human-judgment failures over HONESTLY-STAMPED artifacts.** Exp 53/53b stamped
`working_tree_dirty_src_scripts: true` into every start record and the run continued; the
tag went on a `PUBLISH-PENDING` commit. Nothing lied. Somebody read a true stamp and did not
act on it, under release pressure. So a fix that only attaches artifacts more firmly to
commits — the obvious mechanical move — would miss this entire class: the artifacts were
already correctly attached.

**Replacement rule.** A gating result may be cited by a release only when:

- **(a)** its **DATA** landed in its own merged PR;
- **(b)** its **INTERPRETATION** — the write-up plus the ledger row — landed in a
  **separate, later** PR; and
- **(c)** that interpretation PR got a review pass from **a different reader**.

(a) and (b) are mechanically checkable in the shape the prereg lint already walks: two
distinct merge commits, data strictly before interpretation, for every gated result a
release cites. (c) is not mechanizable and is not pretended to be — it is the same
different-reader discipline the review-round rule already rests on, and for the same stated
reason: *the value of the round is a DIFFERENT reader, not a more careful one.*

**Time stays as the FALLBACK, not as a second requirement: structure OR time, never both.**
Where the split cannot be met — a genuinely single-PR result, a solo run with no second
reader available — the ≥1-day wait applies unchanged. Keeping both would make the honest
path (split the PRs, get a reader) *more* expensive than the sloppy one, which is how process
rules get routed around.

**What this deliberately does not claim:** the split is not a guarantee. A different reader
who rubber-stamps buys nothing, and (c) has no enforcement beyond the same attention the
review rounds rely on. The claim is narrower — it removes the release-day compression that
made the 1.1.0 misreads *likely*, and it replaces an unfalsifiable "we waited" with two
checkable facts.

What to avoid: a repo-wide "provenance framework", an experiment registry, or a new bus.
The pieces exist — `_provenance.py`, the lint, the invariants audit, the ledger's own
rules. The drop happened because they were scoped to sims while the hardware path grew a
parallel harness family that inherited the vocabulary but not the enforcement. Widening
scope is a smaller change than building anything new.

## The door that was still open after 16.7 (2026-08-30, 1.1.2)

**The widened lint was keyed on HOW a harness runs, not WHERE its records land —
and a new writer walked straight through.** `scripts/fail_loud_stage2.py` (the
fail-loud Stage-2 tool) writes its baseline into `docs/experiments/data/`, a
gated path. It matched **neither** existing family: it spawns no `maxim`
(Family 1's regex) and does not live under `scripts/orient_*/` (Family 2's glob).
So it hand-rolled its own dirty-tree check, **stamped** the flag, and wrote the
artifact anyway — detection, not enforcement, the exact shape this lesson
exists about. The committed `baseline.json` then read
`working_tree_dirty_src_scripts: true` while the commit message, the artifact
README and the plan all said "clean tree". An honest machine stamp under a human
misreading, on the same branch that rewrote the rule above.

The captures had genuinely been taken from a clean tree at `6f3f3b7d`; it was the
ARTIFACT that got regenerated minutes later with an uncommitted patch in the
tree. That distinction is exactly the kind a reader will not make on release day,
which is the whole argument of this lesson.

**Fixed two ways.** The tool routes through
`_provenance.py::preflight_gated_record_or_exit` and now REFUSES (exit 3) unless
`--allow-dirty` is passed, which stamps `allow_dirty: true` into the artifact.
And `lint_harness_provenance.py` gained a **Family 3** keyed on WHERE records
land: any `scripts/**/*.py` that names `docs/experiments/data` and writes records
must run one of the sanctioned guards. It accepts all three guard forms — the
first cut did not, and false-positived on `exp44/campaign.py`, which is correctly
guarded via `executed_code_provenance(..., out_path=)`.

**Generalisation worth keeping:** when a rule is about an OUTCOME (records
landing in a gated tree), a lint keyed on MECHANISM (how the harness runs) will
keep growing new doors as new mechanisms appear. Family 3 is the outcome-keyed
one, and it is the family that would have caught this without being widened.

## Status

- 2026-08-27: lesson written; brief stub + roadmap items 16.7–16.9 + guide line landed
  in the same PR. **None of items 1, 2, 4 is built yet** — the brief stub's
  `Regression guard:` line says so explicitly until they are.
- The Exp 53b row stays EARNED in the ledger pending the operator's call; the card's
  upgrade condition for Research integrity is items 1–4 above plus Tier 3.
- 2026-08-28: **R1 replication** at tag `v1.1.0` from a clean tree (#567 data, #568
  write-up) — PASS 1.00 / 0.00 / 0.50; the ledger row now rests on R1. R1 also found three
  harness gaps (bugs ledger D34–D36).
- 2026-08-29: **items 1, 2, 4 built** (1.1.1 Cluster A): `_provenance.py::preflight_gated_record`
  behind `live_common.JsonlLog` (exit 3 / `--allow-dirty` stamps every record), the harness
  lint widened to `scripts/orient_*/`, `lint_prereg_precedes_data.py` in CI (the original 53 /
  53b files, the 53 inputs and — a new finding — the Exp 44b pilot's `campaign_start`, 16 min
  before its prereg commit, are grandfathered by explicit list with reasons), and the EARNED-row
  rule in `lint_claude_md_invariants.py` (L185 / L186 got dated data-lost annotations + test
  guards; L188 got its Exp 42 data links). D34–D36 fixed in the same PR. The brief stub's
  `Regression guard:` line now names these artifacts. The review round (both lenses) caught
  that `git log --diff-filter=A` without `--first-parent` reports a merge-committed prereg at
  its BRANCH time — the lint would have passed the incident under the merge style rule (3)
  mandates; fixed + fixture-tested. The clean-tree rule then surfaced two more dirty-stamped
  gated records nobody had disclosed: `h1_partc_big_block.jsonl` (Exp 45 row) and
  `54_targets.json` (Exp 54 Phase B inputs) — grandfathered with reasons, disclosed on their
  rows/docs, re-runs owed. The refusal now covers the sub-sim spawner family too.
