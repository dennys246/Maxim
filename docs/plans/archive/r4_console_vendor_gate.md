# R4: the vendored-console gate (approach note, #1333)

> **ARCHIVED (2026-10-24 plans archive):** Built by PR #1383. The rule lives in docs/agents/ci-testing.md; the cut rule's authority is readiness_and_extras_standards.md §R4.

Status: APPROACH NOTE, 2026-10-18, Session P. Design pass run 2026-10-18 (BUILD WITH FOLDS); every finding is folded in §Folds, which **overrides** the sections above it where they disagree. Standard:
[readiness_and_extras_standards.md](../readiness_and_extras_standards.md) §R4. Owner decision 2026-10-18: the source of
truth is a **committed pin plus a live check**, failing closed.

## Why existing infrastructure

This rides on what is already there:
- `scripts/vendor_console_ui.py`, the only copy step;
- `scripts/audit_release_build.py`, which already opens the built wheel and sdist;
- the `release-build` CI job;
- pulse's signed `v*` tags and their `console-dist.tar.gz` release assets, whose sha256 GitHub publishes as `digest`.

The one new artifact is a data file, `scripts/console_ui_pin.json`.

## The failure (1.3.2)

The wheel vendored pulse v0.4.0 after v0.4.1 had been tagged. Two gaps let it through:
- `vendor_console_ui.py` takes any directory and refuses only a contract mismatch or a dirty build;
- nothing reads the vendored stamp at release.

## Design

**1. The pin.** `scripts/console_ui_pin.json` carries:
- `_format_version`;
- `repo` (`dennys246/Maxim-pulse`);
- `tag`;
- `commit` (the full sha the tag peels to);
- `contract_version`;
- `asset` (`console-dist.tar.gz`) and `asset_sha256`;
- `tree_sha256`: a digest over the extracted bundle, the sorted `"<sha256>  <relpath>\n"` lines of every file, **`maxim-ui.json` included**.

`--update-pin <tag>` writes it from live data (below), so nobody types a hash by hand. The release PR bumps it.

**2. Offline refusal (`vendor_console_ui.py`).** `validate(source, pin=<committed pin>)` keeps its shape checks, the `target` check and the `dirty` check, and adds:
- the pin's `contract_version` must equal the backend's `CONSOLE_CONTRACT_VERSION`. If they differ, the message says to bump the pin; this mechanizes R3's cut rule.
- the stamp's `describe` must equal `pin.tag`;
- the stamp's `commit` must be a hex prefix (at least 7 characters) of `pin.commit`;
- the bundle's tree digest must equal `pin.tree_sha256`. This is the gate; the stamp checks above exist only to give readable messages.

Because the digest covers the stamp file, a bundle with an edited, missing or unparseable provenance field cannot match. The three pinned "staleness is not refused" tests therefore reverse: a stale, untagged or provenance-less bundle is now refused. They were right when nothing defined "current"; the pin now does.

New mode `--from-pin`: it downloads the pinned asset from the pinned tag's release, verifies `asset_sha256` before extracting (`tarfile` `filter="data"`), validates the result, then vendors it. **This is the release path and CI's path.** Vendoring from a local directory still works, but it passes only if the result is byte-identical to the pinned asset. `--force` still exists and still prints every problem; the audit has no such waiver.

**3. Live check: `--verify-pin [--only-when-releasing]`.** It re-derives the pin and compares every field:
- it reads pulse's tags with `git ls-remote --tags`, peeling annotated tags;
- it takes the tags matching `^v\d+\.\d+\.\d+$`, ordered as integer tuples;
- it reads each tag's contract from `packages/kit/openapi.json` `info.version` at the **commit sha** (raw.githubusercontent), the same source pulse's `check_pymaxim_vendor.py` uses;
- "newest" is the highest such tag whose contract equals the backend's. It must be `pin.tag`, and its commit must be `pin.commit`;
- the release API's asset `digest` must equal `pin.asset_sha256`;
- the downloaded asset's tree digest must equal `pin.tree_sha256`.

Exit codes: 0 for current, 1 for a stale or mismatched pin, 2 for an unreadable source (fail closed, as `check_nightlies.py` does). With `--only-when-releasing` it is a no-op unless pyproject's version has no `v<version>` tag yet, reusing `check_nightlies.is_tagged`.

**4. Audit (`audit_release_build.py`).** For the wheel and the sdist, when `ui_dist` is present it checks:
- the tree digest equals `pin.tree_sha256`;
- the stamp's `describe` equals `pin.tag`, and the stamp's `commit` matches `pin.commit`;
- the stamp's contract equals the `CONSOLE_CONTRACT_VERSION` parsed out of **the artifact's own** `ui_bundle.py`, not the repo's;
- `dirty` is `false`.

On success it prints `vendored console: <tag> (<commit7>, contract <c>)`; the release notes copy that line (M71 stays backlog).

**5. CI.** The `release-build` job's hand-written stub manifest is replaced by `vendor_console_ui.py --from-pin`. Every PR then builds the wheel with the real pinned bundle and audits it on the exact release path. `--verify-pin --only-when-releasing` runs next to `check_nightlies` on `pull_request`, per #938's decision. The #1372 publish workflow runs `--verify-pin` unconditionally.

## Inputs and how each could lie (design pass)

| Input | How it lies | Defense |
|---|---|---|
| The pin file | Not bumped after a new pulse tag (the 1.3.2 failure, moved) | `--verify-pin` on a releasing PR and in the publish workflow |
| The pin file | Hand-edited so its fields disagree | `--verify-pin` re-derives every field |
| `maxim-ui.json` stamp | Self-reported; hand-edited to say v0.4.1 over v0.4.0 contents | The tree digest covers every file, the stamp included |
| A local `dist/` directory | A dev build, a build from another branch, or a build with a different toolchain | Must be byte-identical to the pinned asset, or it is refused |
| The release asset | Re-uploaded under the same tag | The API digest is compared with the pin, and sha256 is verified locally before extracting |
| A pulse tag | Moved by a force-push | The peeled commit must equal `pin.commit` |
| Pulse's tag list | A pre-release (`v0.5.0-rc1`) or a non-semver tag | The regex excludes it, as stated |
| Pulse's tag list | A tag newer than the pin whose `openapi.json` is unreadable | Fail closed (exit 2); never silently excluded |
| The network | Offline, rate-limited (60/h unauthenticated) or 5xx | Exit 2. `GH_TOKEN` is used when set |
| Backend contract (vendor) | `sys.path` imports another tree's `maxim` | It reads `REPO_ROOT/src` first, as today. The audit reads the artifact's own `ui_bundle.py` |
| Backend contract (audit) | The wheel was built from another tree | The artifact's `ui_bundle.py`, not the repo's |
| `--force` | Waives the vendor refusal | The audit has no waiver for the pin or the digest, so a forced bundle cannot pass `release-build` |
| `--allow-missing-ui-dist` | Also waives the pin | When `ui_dist` is present, the pin checks still run. The flag waives absence only |
| "Is this tree releasing" | A wrong answer skips the live check | Same source and same fail-closed exits as `check_nightlies`. The publish workflow runs the check unconditionally |
| Tag creator dates | Can be backdated | Not used. Ordering is by semver |

**Accepted limits:**
- A compromised pulse repo (same owner) is out of scope.
- The live check runs only when releasing, so between releases the pin may lag pulse. Only a release consumes it.
- Every PR's `release-build` now needs github.com release downloads, the same dependency as `actions/checkout`.

## Guards (each proven by deleting its mechanism)

- A stale bundle (an older tag's commit or stamp) is refused.
- An untagged describe is refused.
- Missing provenance is refused (the three reversed tests).
- An edited stamp over the right files is refused (digest).
- A contract and pin disagreement is refused.
- `--from-pin` refuses an asset with the wrong sha256.
- `--verify-pin` returns 1 for a newer same-contract tag, 1 for a moved tag, 1 for a digest mismatch, and 2 when offline or for a newer tag with an unreadable contract. It returns 0 when current and ignores pre-release and other-contract tags. These run against a fake ls-remote and a fake HTTP layer.
- The audit fails a wheel or sdist whose `ui_dist` digest differs from the pin, and when the artifact's `ui_bundle.py` contract differs from the stamp.
- A workflow-shape pin checks that `release-build` vendors `--from-pin` and that no stub manifest remains.

## Folds from the design pass (2026-10-18; these override the text above)

**DO-NOT-BUILD, folded.**

1. **Contract equality only where a release can ship.**
   - `TestContractSurfaceCannotDriftSilently` forces a contract bump on any console-schema change. Requiring `pin.contract == CONSOLE_CONTRACT_VERSION` on every PR would block that bump until pulse tagged a bundle built against unmerged pymaxim. That is a deadlock, and stricter than R3's cut rule, which applies at release.
   - So contract equality is enforced on the releasing paths: `--verify-pin`, the publish workflow, and `vendor --from-pin` / the audit whenever the tree is releasing.
   - On a non-releasing tree they print `CONTRACT DRIFT: a release is blocked until the pin moves to a <c>-contract pulse tag` and pass. The digest, commit and dirty checks still run.
   - "Releasing" means `check_nightlies.is_tagged` says no `v<version>` tag exists. An unreadable answer counts as releasing, so it fails closed.
   - Mechanism: a `--contract-drift-ok-unless-releasing` flag, passed only by CI's `release-build`. Default behaviour is strict.
   - Guard: a non-releasing tree with drift passes with the banner; a releasing one is refused.

**SHOULD-FIX, folded.**

2. **`repo` and `asset` are constants in the script.** The pin echoes them, but they are only ever compared, never used as inputs. A pin naming another repo is exit 1. This closes a fork-substitution path that would fail open.
3. **Re-verify after approval.** The publish job re-runs `--verify-pin` after the environment approval and immediately before the upload. Otherwise a pulse tag pushed during the pause would ship the 1.3.2 failure shape.
4. **One definition of "newest".** It is stable semver (`^v\d+\.\d+\.\d+$`), ordered by integer tuple, and dates are not used.
   - Pulse's `check_pymaxim_vendor.py` uses `refs/tags/v*` ordered by creatordate. The two diverge on a later pre-release or a backport tag.
   - The 1.3.3 note to pulse proposes stable semver for its checker too, and §R4 states the definition.
5. **One canonical digest function.** `tree_digest(iterable of (relpath, bytes))` is fed directly from:
   - the tar members (never the extracted filesystem);
   - the wheel entries with `maxim/console/ui_dist/` stripped;
   - the sdist entries with `pymaxim-<v>/src/maxim/console/ui_dist/` stripped;
   - `DEST` after the copy (it must re-digest equal).

   Rules:
   - strip a leading `./`, skip directory entries, use bytes only and ignore modes;
   - **refuse** any non-regular member, `..`, an absolute path, a backslash, a `\n` in a name, a case-insensitive collision, and any dotfile, because the `console/ui_dist/**/*` package-data glob drops dotfiles and the wheel could then never match.

   Guard: the real v0.4.1 tarball (about 140 KB), committed as a fixture, gives one digest via the tar, a built wheel and a built sdist.
6. **Hermetic and typed.**
   - `ls-remote`, HTTP and download are injected callables.
   - An autouse fixture in the new tests makes `subprocess.run` and `urlopen` raise inside the module, so a test that forgets its fake fails rather than reaching the network.
   - `pin` is a **required keyword-only** argument of `validate()`, so no caller silently uses the repo's pin.
7. **The download path survives a flaky network.**
   - `--from-pin` builds `https://github.com/<repo>/releases/download/<tag>/<asset>` directly and does not call the API. It retries 3 times with backoff, and the sha256 is checked before anything is used.
   - CI caches the asset with `actions/cache` keyed on `asset_sha256`; the sha is re-checked after restore.
   - Only `--verify-pin` calls the API, with `GH_TOKEN` when set.

**NIT, folded.**

8. **Waivers.**
   - `--allow-missing-ui-dist` loses its only caller once CI vendors for real, so it is **deleted**.
   - `--force` never reaches the sha256 check of `--from-pin`.
   - "ui_dist present" means any file under the path, not only `index.html`.
9. **`--verify-pin` edge cases.**
   - It reads contracts only for tags ≥ the pin.
   - A null `digest`, or a newer tag with no asset yet, is exit 2 with the cause named.
   - Lightweight tags (no `^{}` line) are handled.
   - The local sha256 is verified before extracting.
10. **Commit, not describe.** `describe == pin.tag` is demoted to a printed note. The commit-prefix match and the digest are the gates; two tags on one commit, or pulse's tolerated `vX-N-g<sha>`, would otherwise refuse a correct bundle.
11. **Contract read as text.** The backend contract is regexed out of `REPO_ROOT/src/maxim/console/ui_bundle.py`, never imported. An editable install could otherwise answer for another tree. The audit reads the artifact's own copy the same way.
12. **Old Pythons.** Without `tarfile.data_filter`, extraction is refused cleanly.
13. **The pinned tests are rewritten, not flipped.**
    - The rule is "anything not matching the pin is refused, whatever its date or describe".
    - An old-dated bundle that matches the pin passes.
    - `--max-age-days` is removed: the pin supersedes it, and nothing calls it.
14. **`--update-pin <tag>`** refuses a tag whose contract is not the backend's, and warns when it is not the newest same-contract tag.
15. **The guide.** `publication_guide.md`'s vendoring section belongs to the process session. Its replacement text (`--from-pin` is the release path; local paths are dev-only and must be byte-identical) is handed over in the PR.

**Inputs added to the table:**
- the editable-install import path (11);
- subprocess network in tests (6);
- the approval pause (3);
- `repo` and `asset` as derivation inputs (2);
- filesystem case-insensitivity and dotfiles (5);
- pulse's checker's own definition (4).

## Code review (2026-10-18, three lenses): no BLOCKER; folds

- **A path flag on the audit was a waiver.** `--pin` is removed, so the audit reads the committed pin only.
- **The manual publish path.** Until #1372's `release.yml` lands, the hand procedure in `publication_guide.md` is the
  release path, and on it "newest" was proven only on the release PR's CI. Accepted limit until #1372: the guide's
  handoff (below) runs `--verify-pin` and `--from-pin` immediately before the build and the upload.
- **Local Vite builds.** Every local pulse build carries `.vite/manifest.json`, and pulse's `pack-dist.mjs` excludes
  it from the release asset. A directory bundle therefore skips that one top-level directory, and only it
  (`BUILD_ONLY_DIRS`). The tar path still refuses any dotfile, so the digest keeps one definition.
- **One definition of "releasing".** It now lives in `check_nightlies.releasing()`. That version has a `git ls-remote`
  timeout, never prompts for credentials, and treats any unreadable input as releasing. `_console_pin.releasing()`
  delegates to it.
- **Fold 1 consequence (the cut rule).** A contract bump merges, and `main` is then unreleasable until the pin moves.
  The drift is a `::warning::` in CI, not only a log line. The cut-rule text in §R4 and the roadmap says so.
- **Corrections to the folds above.**
  - Fold 5's guard is a synthetic tar/wheel/sdist/directory equivalence test plus CI's live `--from-pin` on every PR,
    not a committed binary fixture (about 150 KB, rejected). The real asset's equivalence across tar, built wheel
    and built sdist was verified by hand on 2026-10-18, in this session and independently by the executor lens.
  - Fold 9: a newer same-contract tag whose release has no asset yet is exit 1 ("move the pin"), not exit 2.
    `resolve_tag` reads only the pinned tag's release.
  - Fold 7: the CI cache key is `hashFiles(scripts/console_ui_pin.json)`, and the asset's sha256 is re-checked after
    every restore.
- **Executor-lens hardening.**
  - `main()` threads `releasing` through both scripts, and tests drive it.
  - Flags that would be silently ignored are refused.
  - Unreadable inputs are exit 2 throughout: the backend contract, a release body that is not UTF-8, an asset that
    is not a tar, `http.client` errors.
  - The audit treats a stamp that is not UTF-8 as a problem.
  - Vendoring `DEST` into itself is refused.
  - Every guard the lens found surviving deletion now has a test that fails without it.
- **Owed outside this PR.**
  - **Pulse** (the 1.3.3 note):
    - adopt the stable-semver definition of "newest" in `check_pymaxim_vendor.py`;
    - stop `gh release upload --clobber` from replacing a tagged asset, whose bytes are not reproducible.
  - **`publication_guide.md`** (the process session's): its two console sections and step 4's "Asserts" list are
    stale. The replacement text is in the PR body.
