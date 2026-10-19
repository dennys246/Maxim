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

It adds one new workflow file and no new script. Without it, publishing is a sequence of hand steps on a laptop, and the 1.3.2 cut showed a hand step gets skipped. Without trusted publishing, a long-lived PyPI token sits on that laptop.

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
   - It writes `SHA256SUMS` and uploads `dist/` as an artifact.
2. **`smoke`** (`contents: read`). On a clean runner, outside the checkout, it creates a fresh venv, runs `pip install <the exact wheel>` and switches to `cd /tmp`, then with `python -I` asserts:
   - `__version__ == tag`;
   - `__all__` is non-empty;
   - `diagnose()` returns;
   - `maxim/console/ui_dist/index.html` is in the installed package;
   - `python -m maxim --help` exits 0;
   - `maxim.__file__` is inside the venv.

   It also re-verifies `SHA256SUMS`.
3. **`publish`** (`environment: pypi`, `id-token: write` only). It **pauses for the owner's approval**, then re-verifies `SHA256SUMS` and uploads with `pypa/gh-action-pypi-publish` over OIDC. No token is stored anywhere. The job runs only on a tag push, never on dispatch.
4. **`github-release`** (`contents: write`). It runs `gh release create v<X> --verify-tag`, with notes from `docs/announcements/release_<X>.md` plus the audit's `vendored console:` line, and attaches the wheel, the sdist and `SHA256SUMS`. That is what `audit_release_tags.py --check-releases` asserts.

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
| The artifact between jobs | Replaced in the artifact store | Built in one job; `SHA256SUMS` re-verified in `smoke` and `publish`. The artifact store is scoped to the run |
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
- the step order (verify-pin → from-pin → build → audit → `SHA256SUMS`);
- the smoke assertions;
- that no step uses a `PYPI_TOKEN` or `password:`.

`test_ci_workflow_shape.py` pins #938's `pull_request` guard on the nightly step. `test_check_nightlies.py` covers `--at`. `test_check_repo_settings.py` covers the new pins.

## Owner setup (one-time; listed in the PR)

1. **PyPI** → pymaxim → Publishing → add a GitHub trusted publisher:
   - owner `dennys246`;
   - repository `Maxim`;
   - workflow `release.yml`;
   - environment `pypi`.
2. **GitHub** → Settings → Environments → `pypi`:
   - required reviewer: you;
   - deployment branches and tags: selected, tag pattern `v*`;
   - prevent self-review stays off (you are the only reviewer).
3. **GitHub** → Rulesets → new tag ruleset `release-tags`:
   - target `refs/tags/v*`;
   - restrict creations, updates and deletions;
   - bypass: Repository admin.
4. Then `python3 scripts/check_repo_settings.py --snapshot`, reviewed in a PR, re-pins the expected settings.
