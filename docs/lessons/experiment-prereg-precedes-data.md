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
5. **The tag waits.** A gating experiment on hardware on the day of the cut means every
   judgment call is made under release pressure. The release gate is "the result is on
   `main` with its data and its pre-registration precedes it"; the tag waits at least a
   day. Exp 54 Phase B/C is the next chance to practice: prereg PR → data PR → ledger PR,
   tag nowhere near it. (Publication guide §"Tag the released commit")
6. **Tier-3 dispositions** are a one-hour doc sweep already owned by 1.2 gate 8(d).

What to avoid: a repo-wide "provenance framework", an experiment registry, or a new bus.
The pieces exist — `_provenance.py`, the lint, the invariants audit, the ledger's own
rules. The drop happened because they were scoped to sims while the hardware path grew a
parallel harness family that inherited the vocabulary but not the enforcement. Widening
scope is a smaller change than building anything new.

## Status

- 2026-08-27: lesson written; brief stub + roadmap items 16.7–16.9 + guide line landed
  in the same PR. **None of items 1, 2, 4 is built yet** — the brief stub's
  `Regression guard:` line says so explicitly until they are.
- The Exp 53b row stays EARNED in the ledger pending the operator's call; the card's
  upgrade condition for Research integrity is items 1–4 above plus Tier 3.
