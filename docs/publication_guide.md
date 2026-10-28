# PyPI Publication Guide

Step-by-step guide for publishing pymaxim to PyPI.

**Current release:** the newest `## [X.Y.Z]` section of `CHANGELOG.md`, and https://pypi.org/project/pymaxim/ for what PyPI serves. *(This line named a version until 2026-09-27 and still said 1.1.0 at the v1.3.1 tag — a hand-kept version line rots, which is why the living sync lines are linted and this one no longer names one.)*
**Package name:** pymaxim (import name: `maxim`)
**Build system:** setuptools + wheel

---

## Pre-Publication Checklist

Start in a fresh shell and reserve one unique directory for the exact candidate:

```bash
export MAXIM_RELEASE_DIR="$(mktemp -d)"
```

### 1. Verify simulations work

Run the core scenarios end-to-end to verify the new infrastructure doesn't break existing behavior:

```bash
# DM campaign (single-agent)
maxim --sim scenarios/campaigns/heist_v1.yaml

# Generative campaign
maxim --sim "test memory recall under interference"

# Research protocol
maxim --sim "test memory recall" --research

# Benchmark (quick check)
maxim --sim benchmark --models mistral-7b --campaign scenarios/benchmarks/quick_check.yaml
```

### 2. Verify tests pass

```bash
# Required fast suite (offline and hermetic)
python -m pytest tests/ -x -q -m "not slow" \
  --ignore=tests/integration/test_memory_hub.py

# MemoryHub integration is a separate required gate when its surface changed
python -m pytest tests/integration/test_memory_hub.py -q
```

Expected: zero failures, no network/model downloads, no hardware access, and no
writes outside the test-owned temporary root. The network half is ENFORCED: `tests/network_guard.py`
(installed by `tests/conftest.py`) makes any outbound TCP connect or non-loopback DNS lookup raise;
`@pytest.mark.allow_network` is the reviewed exception. So is the Docker half (#1103): `tests/docker_guard.py`
fails any non-`slow` test that spawns the `docker` CLI, and the real-container tests run in the nightly slow lane
(`scripts/lane_rosters/slow.json`). Do not waive an ordering failure as
"pre-existing." Do NOT treat a specific pass/skip count as the expectation: skip counts are
environment-dependent by construction (installed extras, platform, model
cache), and the totals move with every added test. Judge the run by
**exit code 0 with zero failures**, plus a clean `git status` afterwards.

Pretrained model/dataset checks are a separate, cache-backed opt-in lane:

```bash
MAXIM_RUN_MODEL_TESTS=1 HF_HOME=/path/to/preloaded/huggingface \
  python -m pytest tests/ -q -m requires_model_cache
```

They remain offline unless the operator separately overrides the standard model-
hub offline variables; they are not part of the correction-release gate.

### 2b. Verify the nightly lanes are green

```bash
python3 scripts/check_nightlies.py
```

The model-cache and slow lanes run only on schedule (or on dispatch), never on a PR, so a release PR
cannot show them red. This reads the latest nightly run on `main` and refuses (exit 1) unless every
`(nightly)` job succeeded within the last 48 hours AND that run tested the commit `main` is at now;
exit 2 means it could not read them, and that is not a pass.

Where the same check runs (#938, 1.3.3):
- **The release PR's required `Release build` check** runs it with `--only-when-releasing` (pyproject's version has
  no `v<version>` tag yet), so the bump PR cannot go green until a nightly passes on `main`'s current tip. Main's
  push and schedule runs no longer run it, so `main` no longer reads red between the release merge and the tag. Any
  OTHER PR opened in that window also runs it and may show red (it did, #1434); that is the gate, not a defect.
- **`release.yml`'s `build` job** runs it with `--at <tag commit>`: the green nightly must have tested the exact
  commit being tagged, not main's later tip. `publish` re-checks it, and the 48 h window, after approval.

When the latest nightly is red, stale, or tested an older commit: fix what is red on `main`, run
`gh workflow run test.yml --ref main`, and re-check once it completes. A re-run of an old run re-tests the OLD
commit and does not count. A dispatched run on `main` queues behind any push run of the same commit (one
concurrency group; `main` never cancels), so allow for both. There is no override.

### 3. Verify clean import

```bash
# No side effects, no subprocess calls, no crashes
python -c "import maxim; print(f'v{maxim.__version__}'); print('Import OK')"

# python -m maxim works
python -m maxim --help

# Diagnose works
python -c "import maxim; r = maxim.diagnose(); print(r)"
```

### 4. Check for remaining blockers

**Ledger rows (CLAUDE.md Principle 1; M25 mechanizes this step).** Every `STALE` or `BROKEN` row in
[behavioral_graduation_candidates.md](plans/behavioral_graduation_candidates.md) blocks the release unless a
`kind: "release"` clause in [evidence_exceptions.json](experiments/evidence_exceptions.json) names that row, that
status and this exact version. Such a clause is an owner decision; it moves no status, and the release notes name
it. Example: `release-1.3.2-T1-2`.

Known items from Phase 12b that may be blocking:

| Item | Status | Blocking? |
|------|--------|-----------|
| `maxim.run()` TypeError (api.py passes wrong kwarg to LLMWorker) | **FIXED** (2026-04-08) | No |
| cv2 module-level imports | **FIXED** | No |
| Error hierarchy exported | **FIXED** (2026-04-08, 7 category-level + RobotController) | No |
| Security hardening (12a) | **FIXED** | No |
| Error honesty audit (API surface: 14 silent catches) | **FIXED** (2026-04-08, security inversion + 13 warnings added) | No |
| `--list-models` CLI flag | Not started | No — nice-to-have |
| campaign()/research()/benchmark() return stubs | **FIXED** (2026-04-08, emit UserWarning with CLI guidance) | No |
| Hippocampus threading (queue race + flush polling) | **FIXED** (2026-04-08) | No |
| Composable API (create/load/Session/Report) | **SHIPPED** (2026-04-08, 83 tests) | No |
| Persistence fixes (NAc/SCN/AG atomic writes, Entity serialization) | **FIXED** (2026-04-08, 19 tests) | No |
| Phase 1 code quality (atomic writes, rate limits, env parsing, blocklist, type annotations, error logging) | **FIXED** (2026-04-08, 19 tests) | No |
| Store protocol wiring (1m) | **SHIPPED** (store protocol wiring complete, see substrate_binding_persistence.md) | No |

### 4b. Run the release gates on the bumped tree

[`scripts/release_gates.json`](../scripts/release_gates.json) gates whole versions (owner decision 2026-10-15:
no version >= 1.4.0 until the `agent_loop`/orchestrator decomposition and its listed items are done). CI's lint
job runs the check on every PR, but a build from a local tree would bypass CI. Run it on the exact tree you are
about to build, AFTER the version bump and BEFORE `python -m build`:

```bash
python3 scripts/lint_release_gates.py
# Expected: "release gates: clean". At a gated version, every gate at or below it prints ACTIVE (on the bump PR,
# judged against the pre-release main) or CLOSED (on main after the bump merges, where the merge-base is HEAD),
# with "0/N items outstanding, 0 functions not done", all arms satisfied and 0 unaccepted new debt.
```

A failure lists every outstanding entry, and the release waits until they are done. The bump diff may not edit
the gate file: retiring an entry or accepting new debt goes in its own reviewed PR before the bump (the lint
refuses it otherwise). The lint's docstring states every rule and residual.

**The bump PR also updates the checkout tests in `tests/unit/test_lint_release_gates.py`.** They are a deliberate
tripwire: `test_this_checkout_passes_below_the_gate` fails once the checkout's version reaches a gate, and
`test_this_checkout_at_1_4_0_fails_listing_every_outstanding_item` fails once nothing is outstanding. In the same
PR as the bump:

- rename the first to `test_this_checkout_passes_at_1_4_0` and assert exit 0, `0/N items outstanding`, and that the
  gate reads EITHER `release gate 1.4.0: ACTIVE` OR `release gate 1.4.0: CLOSED`. Both occur: the test clears the
  CI event, so it judges against the merge-base with origin/main, which is the pre-release main on the bump PR
  (ACTIVE) and HEAD itself on main's push run of the bump commit (CLOSED);
- delete the second, or point its forced version at the next gate in `scripts/release_gates.json` if one exists.

Once 1.4.0 is released (a `## [1.4.0]` header and pyproject version at the diff base), the gate prints `CLOSED`:
its item, arm, function and new-debt rules only print from then on, in the lint job and the runtime twin alike,
while its shape and history rules stay enforced.

### 5. Verify version consistency

```bash
# Both must show the version you are releasing
grep 'version = ' pyproject.toml
python -c "import maxim; print(maxim.__version__)"
```

### 6. Audit the canonical website and PyPI project links

The 1.0.9 correction release makes [pymaxim.bio](https://pymaxim.bio) the
canonical site and
[pymaxim.bio/getting-started](https://pymaxim.bio/getting-started/) the
documentation entry point. The historical `dennyschaedig.com/maxim` guides
remain a migration source, not a second authority. `docs.pymaxim.bio` currently
serves a duplicate homepage; before publication it must redirect every path to
the corresponding canonical `pymaxim.bio` path, with `/` landing on
`/getting-started/`.

Complete this audit against the **exact release candidate**, not a moving branch:

- Inventory every public page and map each historical
  `dennyschaedig.com/maxim/*` guide to a migrated page, an intentional archive,
  or an HTTP redirect. Do not silently strand deep links already present in the
  README, repository docs, search results, or external articles.
- Re-run every installation command and quickstart against a clean install of
  the candidate wheel. Reconcile version numbers, supported Python versions,
  extras, CLI flags, stable API behavior, robot/headless semantics, persistence
  paths, and feature availability with the repository.
- Reconcile every behavioral or biological claim with the graduation ledger,
  bug ledger, limits/scorecards, and current benchmark evidence. Preserve the
  existing honesty rules: influenced is not controlled; bio-inspired is not a
  neuroscience simulation; Oasis/Hivemind must remain future work until shipped.
- Check navigation, search, mobile layout, accessibility basics, canonical URLs,
  Open Graph metadata, page titles/descriptions, and the complete internal and
  external link graph. Record or fix every 404, redirect loop, mixed canonical,
  and stale GitHub/PyPI link.
- Confirm both canonical domains serve HTTPS successfully and that any `www` or
  legacy variants redirect to one canonical URL rather than serving duplicate
  content.
- Review the existing copy specification in
  [pymaxim_bio_update_suggestions.md](announcements/pymaxim_bio_update_suggestions.md),
  updating it wherever later experiments or release-truth work changed the
  evidence.

The package metadata must contain these exact destinations:

```toml
[project.urls]
Homepage = "https://pymaxim.bio"
Documentation = "https://pymaxim.bio/getting-started/"
```

After building, inspect the wheel metadata rather than assuming `pyproject.toml`
was carried through:

```bash
V="$(python -c 'import maxim; print(maxim.__version__)')"
unzip -p "$MAXIM_RELEASE_DIR/pymaxim-$V-py3-none-any.whl" \
  "pymaxim-$V.dist-info/METADATA" | grep '^Project-URL:'
```

After the PyPI upload, open the rendered project page and
click both links. Publication is incomplete if PyPI still exposes the legacy
site, a dead documentation URL, or metadata from a different artifact.

---

## The release: tag-triggered, CI-published (since 1.3.3)

pymaxim is published by [`.github/workflows/release.yml`](../.github/workflows/release.yml) from a signed tag,
through PyPI trusted publishing (OIDC, no stored token). 1.3.3 was the first release published this way
(2026-10-24). The design and its threat table: [release_pipeline.md](plans/archive/release_pipeline.md).

### Owner setup (once, in this order)

1. **The `pypi` environment:** a required reviewer (the owner), deployments only from `v*` tags.
2. **The `release-tags` ruleset:** `v*` tags restricted on create, update and delete, with the owner as the bypass
   actor.
3. **The PyPI trusted publisher** for pymaxim: owner `dennys246`, repo `Maxim`, workflow `release.yml`, environment
   `pypi`.

`check_release_tag.py environment` and `ruleset` check the first two inside every release build. **No script can
read the third** (it lives on PyPI), so confirm it on PyPI → pymaxim → Publishing before each tag; a wrong entry
fails closed at upload, after the approval.

### The order

1. **Exceptions first, in their own PR.** Any `kind: "release"` clause the release needs (§4, for example
   `release-1.3.3-T1-2`) merges BEFORE the bump: a release PR may not add its own exception (M25's rule).
2. **The bump PR.** It holds:
   - pyproject + `__init__.py`;
   - the CHANGELOG cut, made by `python scripts/changelog_fragments.py assemble --version X.Y.Z --date YYYY-MM-DD
     --name "<Name>"`. It folds `[Unreleased]` and every `changelog.d/` fragment into `## [X.Y.Z] - YYYY-MM-DD —
     "<Name>"`, merged by kind, and `git rm`s the fragments. **Re-run it after each merge of `main` into the bump
     branch**: it merges into the existing section and keeps wording edits. The release summary paragraph is
     still written by hand, and the different-reader pass may reword any entry but must leave each
     `<!-- changelog.d/... -->` marker directly above its entry. CI (`changelog_fragments.py check`) refuses a
     bump that drops, duplicates or leaves a fragment, and `check_release_tag.py notes` refuses a fragment at the
     tagged commit (M83 step 3; format: `changelog.d/README.md`). So `check_release_tag.py notes` prints `ok`
     only after `assemble` has run, because before it the cycle's fragments are still pending;
   - the sync lines (below);
   - the release-gate checkout tests, at a gated version (§4b);
   - the roadmap's release block;
   - `docs/announcements/release_<X_Y_Z>.md`, with **absolute links only** (`check_release_tag.py notes
     --version X.Y.Z` must print `ok`; the notes are frozen into the tag and the Release body).

   It gets the review rounds (the `__init__.py` change is a `src/` change) and a different-reader pass on the
   CHANGELOG and notes. Its required `Release build` check needs a green nightly on `main`'s current tip (§2b),
   so dispatch one after any merge that moves `main`.
3. **Freeze `main`** from the bump merge until the tag. Any merge in between invalidates the nightly the tag
   relies on.
4. **A nightly on the merge commit:** `gh workflow run test.yml --ref main` (about 80 minutes, after the push run of
   the same commit). Confirm with `python scripts/check_nightlies.py --at <merge sha>`.
5. **The owner tags that commit, signed and annotated, and pushes only the tag:**

   ```bash
   git fetch origin
   git tag -s -a vX.Y.Z <merge sha> -m "pymaxim X.Y.Z"
   git push origin vX.Y.Z
   ```

   It must be signed by the pinned tagger email (`RELEASE_TAGGER_EMAIL` in `release.yml`), GitHub-verified, on
   `origin/main`, with the tag name equal to `v` + pyproject's version.
6. **`release.yml` builds, then smoke-tests the exact wheel**:
   - `build`: the tag checks; environment and ruleset drift; the notes; the nightly `--at` the commit; the console
     pin; `vendor_console_ui.py --verify-pin` then `--from-pin`; `python -I -m build`; `twine check`;
     `audit_release_build.py`; the digests as job outputs.
   - `smoke`: a clean venv outside the checkout; version, `__all__`, `diagnose()`, `ui_dist/index.html`,
     `python -m maxim --help`, and that the import is from the venv.

   Then it **waits for approval** at the `pypi` environment.
7. **The owner approves `pypi` within 48 h of that nightly.** `publish` re-checks the tag, the nightly window and the
   pin, uploads over OIDC with PEP 740 attestations requested, and confirms PyPI serves exactly the two files and
   digests it built. `github-release` creates the Release from the notes file plus the audit's
   `vendored console:` line, with only the wheel and sdist attached (a third asset would fail the audit).

**Date rule.** The CHANGELOG date and the notes' "Released" line are the UTC date of PyPI's `upload_time`, which is
set at approval. They are written before the tag, so merge, nightly, tag and approval should fall on one UTC day,
or approve after midnight UTC if the files name the next day (1.3.3 did: upload 2026-10-24T00:00:10Z). If they
drift, correct the CHANGELOG in a docs PR and `gh release edit` the body; never re-tag. The "Released" line carries
no clock time.

**A rehearsal** (`gh workflow run release.yml --ref main`) runs `build` and `smoke` and never publishes. It is how
a candidate wheel is built for a downstream check (pulse's release acceptance ran on rehearsal candidates of 1.3.3).
It needs a green nightly on that exact commit too.

### After it publishes

```bash
python scripts/audit_release_tags.py --check-releases   # also runs in CI on push and nightly
d=$(mktemp -d); python3 -m venv "$d" && "$d/bin/pip" install --no-cache-dir "pymaxim==X.Y.Z" \
  && (cd /tmp && "$d/bin/python" -I -c "import maxim; print(maxim.__version__)")
```

Then: send the downstream notes (pulse, M70: the vendored console tag, any Python-API or untyped-`data` changes);
close the issues the release resolves, with evidence; and refresh the website (pymaxim.bio) from the notes.

### The hand path (retired; emergency only, by owner decision)

Until 1.3.3 the release was a hand `twine upload` plus a hand-made tag and Release. SF11 kept that as the fallback
until a release had gone through the workflow, and 1.3.3 did. It is now retired: there is no stored PyPI token,
and a hand upload skips every check above. If the workflow itself is broken at a release, fix the workflow. A hand
upload needs an explicit owner decision recorded in the release PR, a short-lived token, and the same checks run
by hand: the tag, the nightly `--at` the commit, `--verify-pin` + `--from-pin`, the audit, a clean-venv smoke, and
`--check-releases` after.

### Two release-writing rules (from 1.3.0)

- **Do not tag the same day as a gating experiment** unless its DATA and its INTERPRETATION landed in separate PRs
  and the interpretation had a review from a different reader. Otherwise wait at least a day. Full reasoning:
  [the lesson](lessons/experiment-prereg-precedes-data.md).
- **State a direction in claim terms:** on latency, pain-seconds, damage or error counts, "up" and "down" invert
  against "supports" and "undercuts". Write "0.10 s faster, in the claim's direction".

### The tag chain

Any version with a `## [X.Y.Z]` CHANGELOG section must have a matching `vX.Y.Z` tag, and every version PyPI serves
must have a Release with its exact wheel and sdist (sha256 matching PyPI). `scripts/audit_release_tags.py
--check-releases` (the `release-audit` CI job) asserts it; historical failures are grandfathered by explicit list
inside the script and printed on every run. The reconstructed `v1.0.1`–`v1.0.8` tags deliberately have no Release.

---

## Building a candidate locally (dev only)

The workflow is the release path. A local build is for debugging the build itself:

```bash
export MAXIM_RELEASE_DIR="$(mktemp -d)"
python scripts/vendor_console_ui.py --verify-pin && python scripts/vendor_console_ui.py --from-pin
# setuptools' build/lib is not pruned between builds and ships whatever is in it (measured 2026-08-20):
# clean first, whatever build mode you use.
rm -rf build/ *.egg-info
python -I -m build --outdir "$MAXIM_RELEASE_DIR"
twine check "$MAXIM_RELEASE_DIR"/pymaxim-*
python3 scripts/audit_release_build.py --dist-dir "$MAXIM_RELEASE_DIR"
# Expected: "release-build audit: clean". It asserts the filename, METADATA and pyproject versions agree (D48),
# console/ui_dist/index.html is present (D47), py.typed, __main__.py and the _data/ files are present, and the
# R4 checks: the console digest and stamp match scripts/console_ui_pin.json, and the bundle's own contract.
```

---

## Vendoring the Console UI

The Console web UI is built in the **maxim-pulse** repo and *vendored* into this package at release time, so
`pip install pymaxim[console] && maxim serve` serves a working Console with no flag and no config.

- **Destination:** `src/maxim/console/ui_dist/`, shipped as package data and `.gitignore`'d. Vendoring is a
  release step, **never a commit**; a source checkout has no bundle and falls back to the "no UI installed" page.
- **For a release, only the pin** (`scripts/console_ui_pin.json`: tag, commit, asset sha256, tree digest; R4).
  `--verify-pin` proves the pin is the newest stable pulse tag for the backend contract, and `--from-pin` vendors
  exactly that asset. The release build refuses any other bundle.
- **Local vendoring is dev-only:** a pulse checkout's `apps/console/dist` must be byte-identical to the pinned asset
  apart from `.vite/`. The script refuses a bundle whose `maxim-ui.json` names a different `target` or a
  `contract_version` that disagrees with `ui_bundle.CONSOLE_CONTRACT_VERSION`. `--force` vendors it anyway, for a
  dev build only: the release audit refuses the result.
- **At runtime:** a mismatched bundle still boots but logs a loud WARNING naming both versions.

Resolution order is `--ui-dist` > `config.json::console.ui_dist` > packaged.

```bash
python scripts/vendor_console_ui.py --verify-pin && python scripts/vendor_console_ui.py --from-pin   # a release
python scripts/vendor_console_ui.py --check                   # is one vendored?
python scripts/vendor_console_ui.py --clean                   # back to checkout state
```

**Bump `CONSOLE_CONTRACT_VERSION`** ([ui_bundle.py](../src/maxim/console/ui_bundle.py)) when the wire contract
changes in a way a stale bundle would notice: an endpoint removed or renamed, a required field added, an envelope
reshaped. It is also the FastAPI app `version`. **The cut rule (R4):** a bump leaves `main` unreleasable until the
pin moves to a pulse tag built for the new contract. Pulse tags first, pymaxim vendors, then pymaxim releases
([r4_console_vendor_gate.md](plans/archive/r4_console_vendor_gate.md)).

---

## Post-Publication Priorities

### Week 1-2

4. **Substrate-primary harness (B5)** (~1,360 LOC) — Phase -1 prototype + Phase 0 cradle-prelinguistic harness + Hivemind shareability infrastructure. See [substrate_primary.md](substrate_primary.md), [hivemind.md](hivemind.md), and [docs/plans/grounded_language_acquisition.md](plans/grounded_language_acquisition.md). NOTE: roadmap pivoted 2026-05-09; the original "Mother Maxim MVP + deidentification" track was superseded by the federated peer-to-peer Hivemind + Oasis architecture (~55% the LOC, more capable). Phase -1 already shipped.
5. **`--list-models` CLI flag** — model discovery for new users

### Month 1

6. **Maxim Oasis MVP (1.2, gated)** — single-Oasis instance hostable on Mac Mini class hardware. CLI: `maxim oasis serve`. LLM-AUT users opt in to contribute via `maxim contribute --to oasis://...`. ~800 LOC. Requires the provenance/compatibility/threat-model gates in the active roadmap.
7. **Substrate-primary AUT mode (1.1)** — Phase 0 validation runs (raw substrate, no Hivemind); Phase 1 (vocabulary-constrained) starts.
8. **Maxim Hivemind P2P protocol (1.2)** — peer discovery, substrate-snapshot exchange, conflict-resolution semantics, poison-resistance defenses. ~600 LOC.

---

## Version Strategy

| Version | What |
|---------|------|
| 1.0.0 | Stable API; substrate-primary harness (B5: Phase -1 + Phase 0 + Hivemind shareability infrastructure, all behind experimental flag); D1-D3 docs complete |
| 1.1.x | Release hardening and the post-cut experiment line; no Oasis implementation |
| 1.2.x | First hostable Maxim Oasis + full Hivemind P2P protocol; multi-Oasis federation; substrate-primary Maxims pull bootstrap after compatibility gates pass |
| 1.3+ | Phase 3 from-scratch sequence model; substrate-primary becomes default-eligible for end users |

---

## Rollback Plan

If a critical issue is discovered after publishing:

```bash
# Yank the release (removes from PyPI, users can't install it)
# Only use for security issues or data-corrupting bugs
pip install twine
# Contact PyPI support or use the web interface to yank

# For non-critical issues, publish a patch instead:
# Bump to X.Y.(Z+1) through the same pipeline: bump PR, nightly, signed tag, approval
```

---

## Files That Must Stay in Sync

| File | Field | Must Match |
|------|-------|------------|
| `pyproject.toml` | `version = "X.Y.Z"` | All three must be the same version |
| `src/maxim/__init__.py` | `__version__ = "X.Y.Z"` | |
| `CHANGELOG.md` | `## [X.Y.Z]` header | |
| git tag | `vX.Y.Z`, signed and annotated, on the bump's merge commit | pushed by the owner; it starts the release |
| `CLAUDE.md` | "Current version:" line under *Active initiatives* | the pyproject version + a link to PyPI; NO PyPI-state prose |
| `docs/plans/README.md` | "Current version:" line at the top | same |
| `docs/index.md` | "**Package version:**" line (and its "Release notes" link) | same |

Those three lines used to describe PyPI in prose ("PyPI serves X", "pending") and
drifted on every release — corrected in the 1.1 release-truth pass and stale again
five commits later. Since 2026-08-29 they name the pyproject version and link
https://pypi.org/project/pymaxim/ for what is served, and
`scripts/lint_version_sync.py` (CI) fails the build if they name a different version,
carry "pending"/"rc"/"serves" prose, or if `CHANGELOG.md` lacks the `## [version]`
section. Policy: `main` is ahead of PyPI; the bump is the release transaction.
