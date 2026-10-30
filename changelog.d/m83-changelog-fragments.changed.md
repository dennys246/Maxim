- **CHANGELOG entries are now one file per change, in `changelog.d/`** (M83 step 3). A `src/` change adds
  `changelog.d/<slug>.<kind>.md` instead of a line under `[Unreleased]`, which was the file parallel PRs conflicted
  on most; the release PR folds them in with `scripts/changelog_fragments.py assemble`. One check,
  `changelog_fragments.py check`, replaces `lint_unreleased_on_src_change.py` and `lint_unreleased_declared.py`, now
  covers every path under `src/` rather than only `.py` files, and refuses a fragment deleted, renamed or copied
  outside a release. The release workflow also refuses a fragment left at the tagged commit. Format:
  `changelog.d/README.md`.
