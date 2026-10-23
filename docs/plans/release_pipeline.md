# The release pipeline: tag-triggered trusted publishing (approach note, #1372 + #938)

Status: APPROACH NOTE, 2026-10-18, Session P. Its design pass is below. Owner decisions (2026-10-18):
- a GitHub `pypi` environment with the owner as **required reviewer**;
- a `v*` tag ruleset;
- #938 option 1: the nightly check runs in `release-build` on `pull_request` only, and the publish workflow runs it before uploading.

Standing owner rule (2026-10-16): no Test PyPI; the exact built wheel is smoke-tested in a clean venv, then published.

## Why existing infrastructure

The workflow composes scripts that already exist:
- `lint_version_sync.py`;
- `lint_release_gates.py`;
- `check_nightlies.py`;
- `vendor_console_ui.py` (R4: `--verify-pin`, `--from-pin`; see [r4_console_vendor_gate.md](r4_console_vendor_gate.md));
- `audit_release_build.py`;
- `audit_release_tags.py`;
- `check_repo_settings.py`.

It adds one new workflow file and one new script, `check_release_tag.py` (design pass SF3). Without it, publishing is a sequence of hand steps on a laptop, and the 1.3.2 cut showed a hand step gets skipped. Without trusted publishing, a long-lived PyPI token sits on that laptop.

## The workflow (`.github/workflows/release.yml`)

**Trigger.** `push` of tags `v*`, plus `workflow_dispatch` (a rehearsal: everything except `publish` and `github-release`). The top level sets `permissions: {}`, and each job asks for only what it uses. Every third-party action is pinned by full commit sha.

1. **`build`** (`contents: read`, `actions: read`):
   - The tag is `v` + pyproject's version and `__init__`'s version (`lint_version_sync.py`).
   - The tag is an **annotated, GitHub-verified-signed** tag object (`git/tags/<sha>` → `verification.verified`).
   - Its commit is an ancestor of `origin/main`.
   - `lint_release_gates.py` passes.
   - `check_nightlies.py --at <tag commit>`: a green nightly that tested **this commit**, not main's later tip. This is a new flag; the default behaviour is unchanged.
   - `vendor_console_ui.py --verify-pin`, then `--from-pin`.
   - `rm -rf build dist *.egg-info`, then `python -I -m build`.
   - `twine check`, then `audit_release_build.py`.
   - It records the wheel's and the sdist's sha256 as job outputs (immutable to later jobs, unlike a `SHA256SUMS`
     file inside the artifact it vouches for) and uploads `dist/` as an artifact.
2. **`smoke`** (`contents: read`). On a clean runner, outside the checkout, it creates a fresh venv, runs `pip install <the exact wheel>` and switches to `cd /tmp`, then with `python -I` asserts:
   - `__version__ == tag`;
   - `__all__` is non-empty;
   - `diagnose()` returns;
   - `maxim/console/ui_dist/index.html` is in the installed package;
   - `python -m maxim --help` exits 0;
   - `maxim.__file__` is inside the venv.

   It also re-verifies both files against the build job's digest outputs.
3. **`publish`** (`environment: pypi`, `id-token: write` only). It **pauses for the owner's approval**, then re-verifies both files against the build job's digests and uploads with `pypa/gh-action-pypi-publish` over OIDC. No token is stored anywhere. The job runs only on a tag push, never on dispatch.
4. **`github-release`** (`contents: write`). It runs `gh release create v<X> --verify-tag`, with notes from `docs/announcements/release_<X>.md` plus the audit's `vendored console:` line, and attaches the wheel and the sdist only (a third asset would fail the audit: "not a file PyPI serves"). That is what `audit_release_tags.py --check-releases` asserts.

**#938.** `release-build`'s nightly step gains `if: github.event_name == 'pull_request'`. Main's push and nightly runs therefore stop going red between a release merge and its tag. The job itself keeps no job-level `if:`, because it is a required check.

**Order of a release becomes:**
1. The bump PR merges.
2. Dispatch the nightly on that merge commit and wait for it to go green.
3. Push a signed `v<X>` tag.
4. The workflow builds and smoke-tests.
5. The owner approves, and the workflow publishes.

## Inputs and how each could lie (design pass)

| Input | How it lies | Defense |
|---|---|---|
| The tag name | `v1.3.3` on a tree whose pyproject says 1.3.2, or the reverse | Version-sync check against the tag; the audit's D48 checks |
| Who pushed the tag | Anyone with write access | `v*` ruleset (create, update and delete restricted; owner bypass) + a verified signature + the environment reviewer |
| The tag's commit | A branch commit not on main (unreviewed code) | Ancestor-of-`origin/main` check |
| A re-pointed tag | Force-moved after the build started | The ruleset blocks updates. The checkout uses `github.sha`, and every job reuses that sha |
| The artifact between jobs | Replaced in the artifact store | Built in one job; both digests, carried as job outputs, re-verified in `smoke` and `publish`. The artifact store is scoped to the run |
| The nightlies | A green run on another commit | `--at <tag commit>` |
| The nightlies | Unreadable | Exit 2 refuses |
| The console bundle | Stale, or not the pinned tag | R4 (`--verify-pin`, `--from-pin`, the audit's tree digest) |
| The smoke venv | Imports the checkout instead of the wheel | `cd` outside, `python -I`, assert `__file__` is in the venv |
| The environment | Reviewer removed, or the environment deleted (GitHub silently re-creates a named environment unprotected) | Pin `pypi` (its reviewers and its `v*` tag policy) and the `v*` ruleset in `repo_settings_expected.json`; the nightly drift check reads them |
| PyPI's trusted publisher | Mis-configured (wrong workflow or environment name) | Fails closed at upload with an OIDC error; nothing is published |
| A third-party action | A moved tag ships new code | Pinned by sha |
| `workflow_dispatch` | Used to publish | `publish` and `github-release` have `if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')` |
| The release notes file | Missing, or relative links | `gh release create` fails on a missing file; `audit_release_tags.py --check-releases` (non-PR runs) flags links |

**Accepted limits:**
- A failure after the tag (smoke, rejection) leaves a tag with no PyPI release. The fix is an owner-bypass delete and a re-tag. The tag has not shipped, and `audit_release_tags.py` checks only versions PyPI serves.
- PyPI is immutable: once approved, a bad upload can only be yanked.

## Guards

`tests/unit/test_release_workflow_shape.py` pins:
- the trigger;
- per-job permissions (`id-token` only in `publish`);
- `environment: pypi` on `publish`;
- the publish `if:`;
- sha-pinned actions;
- the step order (verify-pin → from-pin → build → audit → digests);
- the smoke assertions;
- that no step uses a `PYPI_TOKEN` or `password:`.

`test_check_nightlies.py` pins #938's `pull_request` guard on the nightly step and covers `--at`. Pinning the
environment and the ruleset in `check_repo_settings.py` is deferred to #1384; `check_release_tag.py environment` and
`ruleset` check both inside the workflow, at the moment they are relied on.

## Owner setup (one-time; listed in the PR), in this order (SF1)

1. **GitHub** → Settings → Environments → **New environment** `pypi`:
   - Required reviewers: you.
   - Deployment branches and tags: **Selected branches and tags** → add a rule of type **Tag**, pattern `v*`.
2. **GitHub** → Settings → Rules → Rulesets → **New tag ruleset** `release-tags`:
   - Enforcement: Active.
   - Target: include by pattern `v*`.
   - Rules: Restrict creations, Restrict updates, Restrict deletions.
   - Bypass list: Repository admin.
3. **PyPI** → pymaxim → Settings → Publishing → **Add a new publisher** (GitHub). This comes LAST: the publisher
   matches only the environment's name, so the protected environment has to exist before this step.
   - Owner `dennys246`.
   - Repository `Maxim`.
   - Workflow `release.yml`.
   - Environment `pypi`.

## Design pass (2026-10-18): BUILD WITH FOLDS, all folded (these override the text above)

**SHOULD-FIX:**
- **SF1, a missing environment is re-created unprotected (fail-open).** `build` runs `check_release_tag.py environment`
  before `publish` can be scheduled. It requires required reviewers and deployment policies of exactly `[v* tag]`,
  and an unreadable answer is exit 2. The owner setup order is now environment → ruleset → trusted publisher, last
  (§Owner setup). Pinning the environment and ruleset in the nightly drift check (`check_repo_settings.py`) is new code
  in that script, which reads only `main` and `settings-check` today, so it is deferred to [#1384](https://github.com/dennys246/Maxim/issues/1384). The
  in-workflow check is the gate.
- **SF2, a stale run waiting for approval could publish a superseded commit (fail-open).**
  - `concurrency: release-${{ github.ref }}`, with no cancel-in-progress.
  - After approval, `check_release_tag.py still` re-resolves the tag and requires the same tag object and commit.
  - Recovery: reject the old run before re-tagging.
- **SF3, nothing compared the tag with the version.** `check_release_tag.py tag` checks tag = `v` + pyproject =
  `__init__`, with tests. "No new script" was wrong: this is the one new script.
- **SF4, `--at`.** `check_nightlies.py --at <sha>` reads `gh run list --commit <sha>`. A run counts only if it is
  scheduled or dispatched, not cancelled, on `main`, with that head sha. A dispatch has to test the commit that gets
  tagged, so the order is: tag the commit a green nightly tested, at or after the bump. A later tag commit makes
  `lint_release_gates` read the version as released, and its gates CLOSE (print only). The bump PR already enforced
  them, so this is harmless.
- **SF5, staleness across the approval wait.** `publish` re-runs `check_nightlies --at` and `--verify-pin` after
  approval, as R4's fold 3 also requires.
- **SF6, circular checksums.** The digests are `build` job outputs, which later jobs cannot change. `smoke`, `publish`
  and `github-release` each run `sha256sum -c` against them.
- **SF7, partial upload.** The upload uses `skip-existing: true`, then a mandatory check that PyPI serves exactly the
  two built files with the output digests. Recovery is re-running the failed `publish` job of the SAME run, never
  re-tagging (artifact retention is 30 days).
- **SF8, checkable steps before the irreversible one.** `check_release_tag.py notes` runs in `build`: the notes file
  must exist with no relative links. The `vendored console:` line travels as a job output into the Release notes.
- **SF9, the tag object.** The ref must be `object.type == "tag"` (a lightweight tag is refused). The check reads
  `git/tags/<tag object sha>` and requires `verified` with reason `valid`, the pinned tagger email, and a commit
  equal to `github.sha`. The checkout must also satisfy `rev-parse $GITHUB_REF^{commit} == $GITHUB_SHA`.
- **SF10, pulse coupling.** The strict option was built (the default recommendation): `--verify-pin` runs in `build`
  and again after approval, so a pulse release in the window blocks publishing until the pin moves. **Owner decision 2026-10-19:
  strict.**
- **SF11, sequencing for 1.3.3.**
  1. `release.yml` merges at or before the bump.
  2. The owner does the setup (environment, ruleset, publisher).
  3. Bump, nightly, signed tag, approve.
  - The dispatch rehearsal does not cover the tag, signature or OIDC steps; the first real upload is
    their first run.
  - The hand path in `publication_guide.md` stays the documented fallback until 1.3.3 has gone through this workflow.

**NIT:**
- N1: the tag, not the upload, is now the "released" marker. Between the bump merge and the tag, PRs are still red
  on the nightly check, and that window now includes nightly and approval latency.
- N2: the `build` checkout uses `fetch-depth: 0`.
- N3: actions are sha-pinned. The publish action builds a container, so its pin is not hermetic (accepted residual).
- N4: `attestations: true` is pinned.
- N5: job names are prefixed `Publish:`, so they collide with no required check.
- N6: every artifact is sha-checked before use.
- N7: the sdist is audited, not installed (accepted).
- N8: `python -m maxim --help` exits through argparse.
- N9: `--verify-tag` and tag-typed deployment policies are confirmed real.

**From R4's architecture review:** inside a tag-push run the tag exists, so `releasing()` reads False. `release.yml`
therefore never passes `--only-when-releasing`, `--contract-drift-ok-unless-releasing` or `--force`;
`tests/unit/test_release_workflow_shape.py` pins that.

## Code review (2026-10-18, three lenses): two BLOCKERs, folded

- **B1, executor: no pipefail.** GitHub's default `bash -e {0}` has no pipefail, so `audit_release_build.py | tee`
  passed a failed audit and the upload would have followed. Fix: `defaults: run: shell: bash`
  (`-eo pipefail`), pinned by `test_every_step_runs_with_pipefail`.
- **B1, wire: missing pyyaml.** `build` installed no pyyaml, and `lint_release_gates.py` fails without it, so every
  run (rehearsal included) would have stopped at the gates. Fix: pyyaml is installed, pinned by
  `test_build_installs_what_its_scripts_import`.
- **S1: the ruleset had no check.** `check_release_tag.py ruleset` now requires an active tag ruleset on
  `refs/tags/v*` that restricts creation, update and deletion. It runs in `build`, so the release refuses until the
  ruleset exists.
- **S4 / SF-e: an approval more than 48 h after the nightly.** A dispatch tests main's tip, so once main has moved
  past the tag, no re-run can satisfy `--at`. The `--at` failure now prints the real recovery: approve within the
  window, or reject the run and cut the release again from a commit a green nightly tested. The same limit applies
  to an SF7 "re-run publish" more than 48 h later. **Approve promptly.**
- **The rehearsal now runs the environment, ruleset and nightly checks too**, so a `workflow_dispatch` on main
  exercises the owner setup before the first real tag.
- **Tests.** The self-referential API test was fixed, and the main exit codes and the `tag_object=` output are
  pinned. The tag-version clause, the target type, a re-tag to a lightweight tag and the `--at` sha filter each have
  their own test.
- **Deferred ([#1384](https://github.com/dennys246/Maxim/issues/1384)):** pinning the `pypi` environment and the `v*` ruleset in `check_repo_settings.py`'s
  nightly drift check. The in-workflow checks fail closed when they are relied on.

## Hand-off to the process session (`docs/publication_guide.md`, not edited here)

1. **§2b is stale.** After #938, push and schedule runs no longer run the nightly check, so "every later PR and
   push to main" no longer goes red. Release PRs still do.
2. **The publish path becomes `.github/workflows/release.yml`.** The new order:
   1. The bump PR merges, with `docs/announcements/release_<X>.md` written with absolute links.
   2. A green nightly runs on the commit to be tagged.
   3. The owner pushes a signed annotated `vX.Y.Z` tag on that commit.
   4. The workflow builds, audits and smoke-tests.
   5. The owner approves `pypi` **within 48 h of the nightly**.
   6. The workflow uploads over OIDC, verifies what PyPI serves, and creates the GitHub Release.

   The hand path (`twine upload` and a hand-made Release) stays as a documented fallback until 1.3.3 has gone
   through the workflow.
3. **The console steps (R4).**
   - Step 1b becomes `python scripts/vendor_console_ui.py --verify-pin && python scripts/vendor_console_ui.py --from-pin`,
     replacing the local-path vendor.
   - Local `apps/console/dist` vendoring is dev-only, and must be byte-identical to the pinned asset apart from
     `.vite/`.
   - Remove the `--allow-missing-ui-dist` paragraph (the flag is gone).
   - Step 4's "Asserts" list gains the R4 checks: the console digest and stamp against the pin, and the artifact's
     own contract.
   - The "Vendoring the Console UI" section's "Sources … no `v*` tag needed" and "`--force` overrides" lines are no
     longer true for a release.
4. **The owner setup, once, in order:**
   1. the `pypi` environment;
   2. the `release-tags` ruleset;
   3. the PyPI trusted publisher.

   (§Owner setup above.)
