# changelog.d: one file per change

Every change that touches `src/` adds **one new file** here, instead of a line under `CHANGELOG.md`'s
`## [Unreleased]`. Every PR used to append at the top of that one section, which made it the file parallel PRs
conflicted on most. The release PR folds every file into the new version section and deletes them.
`scripts/changelog_fragments.py check` enforces this in CI; the design and its review rounds are in
`docs/plans/changelog_fragments_m83_approach.md`.

## The file

`changelog.d/<slug>.<kind>.md`

- **`<slug>`**: lowercase letters, digits and hyphens, starting with a letter or digit. Use the branch slug or the
  issue number (`recall-1444`, `1496-reachy-nan`). Two PRs that pick the same slug conflict loudly, so pick a
  specific one.
- **`<kind>`**: one of `breaking`, `security`, `added`, `changed`, `deprecated`, `removed`, `fixed`. A `breaking`
  entry is assembled first, under `### Breaking`; say what breaks and how to migrate.
- **The body**: the entry exactly as it should read, in the CHANGELOG's usual style: one or more `- ` bullets,
  with nested lists allowed. No `#` headings (the kind supplies the heading) and no HTML comments.

```markdown
- **Talk remembers what you told it in an earlier session** (#1444): recalled episodes now render what was heard,
  framed as untrusted content.
```

## What CI checks

- A diff that touches anything under `src/` must **add** a fragment. Renaming another PR's fragment does not count,
  and a new fragment that repeats another one is refused: an exact copy at any length, or one 90% similar or more for
  entries of 80+ characters.
- CI reads the **committed** tree, so commit your fragment before running `python scripts/changelog_fragments.py
  check` locally: an uncommitted fragment is invisible to it.
- Outside a release, a fragment may be **reworded but not deleted or renamed**. If you chose the wrong kind, the
  release PR moves it.
- `## [Unreleased]` in `CHANGELOG.md` must not grow, and no `## [X.Y.Z]` section may be written ahead of the
  release.

## Open PRs from before fragments

A branch that still adds a line under `[Unreleased]` fails with "`[Unreleased]` grew" after this lands. Move the
line into `changelog.d/<slug>.<kind>.md` and restore `[Unreleased]` to `main`'s version.

## Reverting an unreleased change

Keep the reverted PR's fragment and add one of your own, in the form `- Reverted #N: <why>.`. Restating the
original text would be refused as a copy of it.

## At release time

The bump PR runs:

```bash
python scripts/changelog_fragments.py assemble --version X.Y.Z --date YYYY-MM-DD --name "<Name>"
```

`assemble` moves everything still under `[Unreleased]` into the new section, merges by kind, adds each fragment after
a marker line of the form `<!-- changelog.d/<file> -->`, and `git rm`s the fragments. Run it again after every merge
of `main` into the bump branch: it merges into the existing section, and wording edits made there are kept. CI then
checks that every fragment appears exactly once, each marker still has its entry, and no fragment is left. The
release PR may reword entries freely, but must keep each marker line directly above its entry.
