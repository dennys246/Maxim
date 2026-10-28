# M83 step 3: CHANGELOG fragments (approach note)

> **Status:** v4.1, design SETTLED 2026-10-28, process session. **Round 4: STOP, no further round owed**
> (4 NITs, all folded: a pinned F3 test, the freeze wording made consistent, the 80-character residual stated,
> named F1 tests). Any `scripts/` fold after this point still gets a delta-scoped code round.
> - **Round 3 (a third reader, delta-scoped):** 0 DO-NOT-BUILD, 3 SHOULD-FIX (F1 fail-closed; F2 and F3 fail-open,
>   narrow, inside the R1 and R2 classes) and 3 NITs, each folded and tagged. Converging. A one-line round 4
>   checks that F2, F3 and their named tests are in.
> - **Round 2 (a different reader, delta-scoped):** 0 DO-NOT-BUILD, 5 SHOULD-FIX (R1–R5; R1 and R2 are
>   fail-open) and NITs n1–n8, each folded below and tagged with its id. It found the same classes as round 1,
>   only narrower (converging), so one more delta round is owed on the R1/R2 folds.
> - **Round 1's claims, corrected:** `-C` uses only files modified in the same diff as copy sources, so a copy
>   of an untouched fragment still lists as `A` under `-M -C`. Rule 1 below uses `--find-copies-harder` plus a
>   near-duplicate refusal (R2). This is a gate change, so the adversarial design pass
> comes before any code (CLAUDE.md "Faster without lowering the bar" (1)).
> - **Backlog row:** `docs/plans/outstanding.md` M83.
> - **Owner decisions (2026-10-28, asked together):** leave the existing `[Unreleased]` text and fold it in at
>   the cut; write a small in-repo script, not towncrier; the kinds are Keep-a-Changelog's six plus `breaking`;
>   land before the 1.3.4 cut, so 1.3.4 is the first assembly.
> - **Round 1 (adversarial):** 0 DO-NOT-BUILD, 5 SHOULD-FIX (S1–S5, four of them fail-open) and NITs N1–N8.
>   All are folded below, each tagged with its id. The open questions are answered (§Resolved). S1 was
>   reproduced independently in a scratch repo: under `--no-renames` both a renamed and a copied fragment list
>   as `A`; under `-M -C` they list as `R100` and `C100`.

## Why

`CHANGELOG.md` is the file parallel PRs conflict on most. On 2026-10-27 alone, #1502, #1506 and step 0b all
conflicted on it, and `scripts/pr_overlap.py` flags it for nearly every pair of PRs. Every `src/` PR must add
an entry, and every one appends it at the top of `## [Unreleased]`. Fragments remove that conflict, and so the
manual resolve-and-push it forces. **They do not remove "BEHIND":** with up-to-date branches required, every
merge still restarts the other PRs' CI. That is M83 step 2, the merge train (N7).

**Front-gate question:** can this ride on existing infrastructure? Partly. The check reuses
`scripts/_lint_git.py` (`base_ref`, `must_not_skip`, `show`) as the two lints it replaces do. What those lints
cannot do is accept an entry anywhere except one shared section, and that shared section is the conflict.

## Shape

- **Fragments.** Each is a file `changelog.d/<slug>.<kind>.md`:
  - `<slug>` matches `[a-z0-9][a-z0-9-]*`. Two PRs that pick the same slug get an add/add conflict, which fails
    loudly.
  - `<kind>` is one of `added | changed | deprecated | removed | fixed | security | breaking`.
  - The body is the markdown entry: one or more top-level `- ` bullets, nested lists allowed, no `#` headings.
  - `changelog.d/README.md` holds the format and is never a fragment.
- **Assembly**, run in the bump PR:
  `python scripts/changelog_fragments.py assemble --version X.Y.Z --date YYYY-MM-DD --name "<Name>"`. It:
  1. writes `## [X.Y.Z] - <date> — "<Name>"` under `## [Unreleased]`. **If that section already exists, it
     merges into it** (R3): a re-run is the normal path, because the bump branch must take `main` after every
     merge, which brings new fragments. Text already in the section, including the reword pass's edits, is
     kept, and new fragments join it by kind. `check` refuses two headings for one version;
  2. moves **all** `[Unreleased]` content into that section on every run, not just the first (N2);
  3. **merges by kind (N3, n3):**
     - It maps each `###` heading in the moved text and the existing section to a kind by its first word.
     - A plain heading (`### Fixed`) merges into its kind.
     - A heading with a subtitle (`### Changed — release integrity`) is kept verbatim as its own block,
       directly after its kind's block.
     - `####` subheadings stay inside their block.
     - Text before the first `###` (the release summary) stays at the top of the section.
     - A heading that matches no kind is kept verbatim after the canonical blocks.
     - The canonical order is `### Breaking` first, then Security, Added, Changed, Deprecated, Removed, Fixed,
       so the 1.3.4 cut gets one plain `### Fixed`, not two.
     - A line-multiset round-trip test proves no line is lost or duplicated.
     - **Within each kind block, fragments go AFTER any unmarked text**, the moved `[Unreleased]` bullets
       included, and the same holds on a re-run (F3). A marker is therefore never followed by someone else's
       unmarked bullet.
  4. puts a `<!-- changelog.d/<file> -->` marker on its own line before each fragment's text (S3). Markers are
     HTML comments, so they don't render. The release notes come from `docs/announcements/`, not CHANGELOG;
  5. **orders the fragments within a kind (N1)** by the first-parent commit on `main` that added each one
     (`git log origin/main --first-parent --diff-merges=first-parent --diff-filter=A`, run on `origin/main`, not
     HEAD; n5), with the file name breaking ties. A fragment with no such commit sorts last. The list of
     fragments comes from `git ls-files changelog.d/`;
  6. deletes the fragments with `git rm`.

  The hand-written release summary stays hand-written, and the different-reader pass may reword any fragment's
  text inside the section. The markers make that safe (Q1).
- **One script, two subcommands** (`check`, `assemble`) sharing one parser. It replaces both
  `lint_unreleased_on_src_change.py` and `lint_unreleased_declared.py`, which enforce nearly the same rule as
  two CI steps with different release-transaction tests and different scopes (S5).

## The rules `check` enforces

`check` compares the diff against the merge-base with `origin/main`, through `base_ref`, in the `lint` job
(N6). Renames and copies are listed with `-M -C --find-copies-harder`, never `--no-renames` (S1, R2).

1. **A `src/` change carries a new fragment.** If the diff touches any path under `src/` (the wider scope of
   today's `lint_unreleased_declared::SRC_PREFIX`, so YAML and packaged data count; S5), it must ADD at least
   one valid fragment: a name-status line of exactly `A`. A rename (`R`) or copy (`C`) never counts.
   - **Near-duplicates are refused too (R2, F2).** An added fragment whose body is 0.9 or more similar to any other
     fragment's body in the tree fails, with both names in the message. This catches a copy with one word changed,
     which every copy-detection flag misses.
     - The similarity is `difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()` on normalized bodies:
       lowercased, with whitespace runs collapsed to one space and the ends stripped. With the default
       `autojunk=True`, a 2,257-character real bullet with one word changed scored 0.595 and passed.
     - The check applies only to bodies of 80 or more normalized characters. Short entries such as "Fixed a typo
       in X/Y" score 0.95 against each other and are not copies.
     - Named tests: a 200+ character fragment with one word changed is refused, and two short typo fixes pass.
     - `changelog.d/README.md` gives the revert template "Reverted #N: <why>", which scores 0.79 against its
       original, rather than "Revert: <original text>", which scores 0.93.
   - **Residual, stated:** a deliberately rewritten copy of another PR's entry is not detectable by any diff
     rule. The threat model is accidental reuse by cooperating sessions, not an adversary. Nor does the check
     catch a one-word-changed copy of a SHORT pending fragment (under the 80-character floor), a bounded trade
     made to avoid false refusals of short entries (round 4).
   - A version-bump diff is exempt from this rule (rule 3).
2. **No fragment leaves outside a release.** This applies to every diff, `src/` or not (S2). Unless the
   pyproject version changed, the diff may not delete or rename away any fragment. Modifying one, to reword
   it, is allowed. **Consequences (n4), documented in `changelog.d/README.md`:**
   - A wrong kind can be fixed only in the bump PR, where the entry moves under the right heading.
   - Reverting an unreleased `src/` PR keeps that PR's fragment and adds a "reverted" fragment of its own.
3. **A bump consumes every fragment exactly once.** This applies whenever the `pyproject.toml` version changed,
   and it is a set of constraints, not an alternative path (S4).
   - **Two sets (R4, F1).** The net diff alone isn't enough: a fragment added and assembled inside the bump
     branch is absent at both ends.
     - **Required:** the fragments present at the merge-base, plus every fragment that a first-parent unit of
       the range NET-adds (`git log --first-parent --diff-merges=first-parent --no-renames --diff-filter=A`).
     - **Permitted:** the required set plus every name added anywhere in the range (`--no-renames`), so a
       fragment renamed during review is still covered.
     - All of these calls pass `--no-renames` explicitly, so the user's `diff.renames` setting cannot change the
       answer.
   - **Markers** must cover the required set and stay inside the permitted one, each exactly once. Markers
     count only inside the new section.
   - **On a push to `main`,** rule 3 is evaluated for the bump's own first-parent unit, as `push_units` already
     does. A bump that lands in the same push range as a PR whose push run was cancelled therefore isn't red.
   - **Every marker must be followed by an entry (R1, F3):** its next non-blank line starts with `- `. A heading
     or the end of the section counts as no entry. Named test: a fragment's text deleted in the reword pass
     fails, where the fixture fragment is LAST in a kind block that also holds moved `[Unreleased]` bullets,
     which pins the F3 placement. Residual, stated: deleting some bullets of a multi-bullet fragment passes, because that is the
     reword licence. No `<!--`
     other than a marker line may appear in the section, so a stray comment can't hide entries.
   - No file may remain in `changelog.d/` except `README.md` (S3).
4. **`[Unreleased]` does not grow.** This applies outside a bump. Growth means the count of non-blank content
   lines rises; HTML-comment lines count as content (n8, matching the old helper's code rather than its
   docstring). Rewording without adding lines is allowed (N5). The failure message points to `changelog.d/`.
   After the first assembly `[Unreleased]` is empty, so in practice it then has to stay empty. Outside a
   bump the diff may also add no new `## [` heading, because a pre-written version section would bring the
   conflict back (round 3 NIT).
5. **Every fragment in the tree is valid:**
   - a regular file directly in `changelog.d/`, so no symlinks and no subdirectories (n6);
   - its name matches the pattern and its kind is known;
   - it has at least one content line and starts with `- `;
   - no line matches `^#{1,6}\s` (n6), and no `<!--` anywhere in the body, so a fragment can't carry a forged
     marker (n1).

**The in-flight path (c) is DROPPED (Q2).** Since 1.3.3 the guide tags the bump merge commit, so a `src/`
change merged during the freeze is not in `vX.Y.Z`. A fragment is the correct record: it goes into the next
release. `lint_unreleased_on_src_change.py` never had (c) and runs in the same job, so (c) was already dead in
practice, and dropping it also removes S4's ambiguity.

**At the tag (N4, R5):** `check_release_tag.py notes` fails if any fragment exists, but only when `v<pyproject
version>` is not yet tagged or the event is a tag push. An unreadable tag list counts as not yet tagged, the
strict path. A pre-bump rehearsal of `main` (pyproject still names the last, tagged release) is therefore not
failed by the cycle's pending fragments. A post-bump rehearsal or the tag itself is, which is when an entry
left over would be missing from the released section. "Tagged" is read with `git tag -l v<version>`,
which needs the tags fetched: the job's `fetch-depth: 0` does that, and a test pins it. A rehearsal of `main`
after the bump merge and before the tag goes red if a fragment merged during the freeze. That is a known false
red on rehearsals only. Such a fragment is the correct record (Q2) and goes into the next release, and the tag
itself sits on the bump merge commit, which has no fragments (round 4). A push range that spans a freeze-violating merge goes
red; that is the N6 class, and `push_base_accepts.json` is the documented way out (n2).

## Inputs, and how each could lie

| Input | How it could lie | Answer |
|---|---|---|
| merge-base (`base_ref`) | a shallow clone has no base, so the lint silently passes | `must_not_skip` makes that a hard error on a PR (D37) |
| diff name-status | a rename or copy counted as added (S1); a copy of an untouched fragment listed as `A` (R2); a deletion unchecked (S2) | `-M -C --find-copies-harder`, only `A` counts, a near-duplicate refusal, and rule 2 |
| the scope of "src change" | narrowed to `.py` (S5) | any path under `src/`, with a unit test |
| fragment content | empty, whitespace, only a heading | rule 5 |
| pyproject version | a bump without assembly; a bump that drops a fragment; a re-run that duplicates the section (R3) | rule 3, plus one heading per version |
| markers | hand-written, duplicated, empty (no entry after them), hidden by a stray comment, or forged inside a fragment body | each names an allowed file exactly once (R4's set); a `- ` entry must follow it (R1); no other `<!--` in the section; none in fragment bodies (n1) |
| `CHANGELOG.md` at base | a failed read returned as "" (#1098 class) | `show` raises; the error propagates to `must_not_skip` |
| push to `main` | one check covers several PRs | the same blind spot today's lints have; up-to-date branches mitigate it (N6) |
| merge-order commits | a merge-commit PR credited to its branch commit (N1); HEAD's history used instead of main's (n5) | `git log origin/main --first-parent --diff-merges=first-parent`; this only orders, never gates |
| tags (`notes` at rehearsal) | fragments always pending pre-bump (R5) | check only when the version is untagged or on a tag push; an unreadable tag list counts as untagged |
| a reverted bump | rule 3 run in reverse | fails closed; a revert PR records an exception by hand (N8) |

## Changes that come with it (one PR)

- **CI:**
  - the two `test.yml` lint steps become one, `python3 scripts/changelog_fragments.py check`;
  - `check_release_tag.py notes` gains the no-fragments-at-tag check.
- **Tests:** `tests/unit/test_changelog_fragments.py` builds fixture repos covering:
  - every rule, each with a positive control;
  - the rename/copy (S1), stray-deletion (S2), duplicate-first-line and dropped-fragment (S3), bump-scope (S4)
    and YAML-under-`src/` (S5) cases;
  - the #561/#562 shape as a regression case;
  - `assemble` round trips: heading merging with subtitles (N3, n3), a line-multiset check, and assemble → reword → new fragment → assemble again (N2, R3);
  - R1 (an empty marker, a stray comment), R2 (a one-word-changed copy) and R4 (a fragment added and assembled inside the bump branch) as named cases;
  - F1: a fragment renamed during review inside the bump branch, and a `src/` PR merging after the bump in the same push range (round 4).

  The old lints' tests move over, so no case is dropped.
- **Docs:**
  - `docs/publication_guide.md` step 2 (the CHANGELOG cut becomes the `assemble` command);
  - the `docs/agents/ci-testing.md` lint row;
  - `outstanding.md` M83 (step 3 shipped);
  - `pr_overlap.py`'s hot-file comment (CHANGELOG.md stays on the list, since bumps still edit it);
  - `changelog.d/README.md`, which shows the marker syntax only inside backticks, so it is never read as a marker (round 4).
- **CLAUDE.md, in this same PR** (n7: `lint_claude_md_invariants.py` fails once a cited guard path is gone), once the owner approves the exact text: the versioning invariant's "`CHANGELOG.md` accumulates
  under `## [Unreleased]`", its Regression guard line, and §Versioning's "must add an `[Unreleased]` line".
- **Other docs citing the old scripts** (n7): `docs/lessons/versioning-main-ahead-of-pypi.md`, `docs/agents/ci-testing.md`, `docs/plans/latent_forward_model.md`.
- **This PR's own fragment,** the first one.

## Resolved (round 1's open questions)

1. **Rewording in the bump PR.** The first-line check was too strict, and it could not tell a reworded
   fragment from a dropped one. Markers replace it: rule 3 checks identity (which files), not text, so
   rewording is free.
2. **The in-flight path (c).** Dropped, as above.
3. **`[Unreleased]` growth during the transition.** Fail on it. `main` requires up-to-date branches, so every
   open PR re-runs CI after M83 merges and moves its hunk into a fragment once. Anything that slips through is
   still moved by `assemble` (N2).
