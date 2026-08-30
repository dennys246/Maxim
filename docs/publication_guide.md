# PyPI Publication Guide

Step-by-step guide for publishing pymaxim to PyPI.

**Current release:** 1.1.0 — published to PyPI 2026-08-26 (tag `v1.1.0` at `df881b87`; GitHub Release with the exact wheel + sdist attached; `pip install pymaxim`).
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

# DM campaign with party mode (multi-agent)
# Requires: party_mode: true in campaign YAML
maxim --sim scenarios/campaigns/heist_v1.yaml  # (add party_mode to YAML first)

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
writes outside the test-owned temporary root. Do not waive an ordering failure as
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

### 5. Verify version consistency

```bash
# Both must show the same version (1.1.0 for this release)
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
unzip -p "$MAXIM_RELEASE_DIR/pymaxim-1.1.0-py3-none-any.whl" \
  'pymaxim-1.1.0.dist-info/METADATA' | grep '^Project-URL:'
```

After the TestPyPI and real PyPI uploads, open the rendered project page and
click both links. Publication is incomplete if PyPI still exposes the legacy
site, a dead documentation URL, or metadata from a different artifact.

---

## Build Steps

```bash
# 1. Keep using the candidate directory reserved above.
test -n "$MAXIM_RELEASE_DIR"

# 1b. VENDOR THE CONSOLE UI (release-only step — see below)
python scripts/vendor_console_ui.py <path-to>/Maxim-pulse/apps/console/dist
python scripts/vendor_console_ui.py --check     # confirms it took

# 2. Build wheel + sdist
# setuptools' scratch tree is NOT pruned between builds and is NOT what
# --outdir controls: build_py copies the package into ./build/lib and ships
# whatever it finds there. A file deleted from src/ but still sitting in
# build/lib lands in the wheel. That is the documented mechanism behind
# "15 dead modules (~8,500 LOC) shipping in the wheel" — and a repo-root
# build/ tree from an earlier run is the normal state, not the exception.
# Measured 2026-08-20: the DEFAULT command below (sdist -> wheel) is protected
# by the sdist round-trip and did NOT pick up a planted build/lib/maxim/
# module. But `--wheel` and `--no-isolation` build in place and BOTH shipped
# it. Those are one habit away, so clean first rather than depending on which
# build mode someone reaches for.
rm -rf build/ *.egg-info

python -I -m build --outdir "$MAXIM_RELEASE_DIR"

# 3. Validate package metadata
twine check "$MAXIM_RELEASE_DIR"/pymaxim-*
# Expected: PASSED for both .whl and .tar.gz

# 4. Audit the BUILT wheel — contents AND version (bugs ledger D47 + D48)
#
# This replaces the three hand-rolled `python -c` checks that used to live here.
# They were correct and nobody ran them: on 2026-08-30 a worktree build shipped
# ZERO Console UI files and a stale-branch build produced a 1.1.0 wheel, and
# `twine check` PASSED both. Step 3 above validates that metadata RENDERS; it
# never asks whether the artifact contains what it should or is the version you
# meant. This step asks, and it is the same script the `release-build` CI job
# runs on every PR.
#
# Do NOT pass --allow-missing-ui-dist here. That flag exists for CI, which has
# no vendored bundle; on the release path a missing bundle is the defect.
python3 scripts/audit_release_build.py --dist-dir "$MAXIM_RELEASE_DIR"
# Expected: "release-build audit: clean"
# Asserts: filename version == METADATA version == pyproject version (D48);
#          console/ui_dist/index.html present (D47); py.typed, __main__.py and
#          25+ files under _data/ present.
```

---

## Vendoring the Console UI

The Console web UI is built in the **maxim-pulse** repo and *vendored* into
this package at release time, so `pip install pymaxim[console] && maxim serve`
serves a working Console with no flag and no config.

- **Destination:** `src/maxim/console/ui_dist/` — shipped as package data,
  `.gitignore`'d. Vendoring is a release step, **never a commit**; a source
  checkout has no bundle and falls back to the "no UI installed" page.
- **Sources:** a local pulse checkout (`apps/console/dist`) or an unzipped
  `ui-dist` CI artifact from a maxim-pulse run on `main` — no `v*` tag needed.
- **Validation:** the script refuses a bundle whose `maxim-ui.json` names a
  different `target` (pointing at the *reachy* build is the easy slip) or a
  `contract_version` that disagrees with `ui_bundle.CONSOLE_CONTRACT_VERSION`.
  `--force` overrides; you probably don't want it.
- **At runtime:** a mismatched bundle still boots but logs a loud WARNING
  naming both versions — refusing to start over a version string would be
  worse for a local tool, and mismatches are often benign.

Resolution order is `--ui-dist` > `config.json::console.ui_dist` > packaged.

```bash
python scripts/vendor_console_ui.py <path>   # vendor
python scripts/vendor_console_ui.py --check  # is one vendored?
python scripts/vendor_console_ui.py --clean  # back to checkout state
```

**Bump `CONSOLE_CONTRACT_VERSION`** ([ui_bundle.py](../src/maxim/console/ui_bundle.py))
when the wire contract changes in a way a stale bundle would notice — an
endpoint removed or renamed, a required field added, an envelope reshaped.
It is also the FastAPI app `version`, so the OpenAPI schema and the check
cannot drift.

## Publish to Test PyPI (safe, reversible)

```bash
# 1. Upload to Test PyPI
twine upload --repository testpypi "$MAXIM_RELEASE_DIR"/pymaxim-*

# 2. Test install in clean venv
python -m venv /tmp/test-maxim-install
source /tmp/test-maxim-install/bin/activate

pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pymaxim

# 3. Verify
python -c "
import maxim
print(f'Version: {maxim.__version__}')
print(f'Verbs: {sorted(maxim.__all__)}')
r = maxim.diagnose()
print(f'Diagnose: {type(r).__name__}')
print('Test PyPI install OK')
"

# 4. Verify python -m maxim
python -m maxim --help

# 5. Clean up
deactivate
rm -rf /tmp/test-maxim-install
```

---

## Publish to Real PyPI

Only after Test PyPI verification passes:

```bash
# The big moment
twine upload "$MAXIM_RELEASE_DIR"/pymaxim-*

# Verify real install
pip install pymaxim
python -c "import maxim; print(maxim.__version__)"
```

### Tag the released commit (do NOT defer this)

PyPI is immutable but carries no git history; without a tag, nothing records
*which commit* produced the artifact. Versions 1.0.1–1.0.6 skipped this step and
had to be reconstructed months later from version-bump commits — the incident
behind the 1.1 release-truth pass. Tag before you close the terminal.

**But not the same day as a gating experiment.** 1.1.0 was tagged the afternoon its
gating hardware result (Exp 53b) ran — from a dirty tree at a commit the squash-merge
made unreachable, with the pre-registration landing on `main` in the same commit as
the data ([lesson](lessons/experiment-prereg-precedes-data.md)). The release gate is
"the result is on `main` with its data and its pre-registration precedes it".

**Then: structure OR time — not both (revised 2026-08-30).** The ≥1-day wait was a proxy
for the thing actually missing on release day: *a second reading of the interpretation by
someone not carrying the release*. All three 1.1.0 provenance failures were human-judgment
failures over honestly-stamped artifacts, so attaching artifacts more firmly to commits
would not have caught any of them. A release may cite a gating result when:

- **(a)** its **DATA** landed in its own merged PR;
- **(b)** its **INTERPRETATION** (write-up + ledger row) landed in a **separate, later** PR;
- **(c)** that interpretation PR got a review pass from **a different reader**.

(a) and (b) are mechanically checkable — two distinct merge commits, data strictly earlier.
(c) is the same different-reader discipline the review rounds rest on and is not mechanized.

**Where that split cannot be met, the ≥1-day wait applies unchanged.** It is the fallback,
never an additional hurdle on top of the split — stacking both would make the honest path
cost more than the sloppy one. Full reasoning and the limits of the claim:
[the lesson](lessons/experiment-prereg-precedes-data.md) §"The ≥1-day wait, reconsidered".

```bash
git tag -a "v$(python -c 'import maxim; print(maxim.__version__)')" \
  -m "pymaxim $(python -c 'import maxim; print(maxim.__version__)')"
git push origin --tags

# Verify: the tag exists, points at the published commit, and matches PyPI
git describe --exact-match --tags HEAD
```

Any version with a `## [X.Y.Z]` CHANGELOG section must have a matching
`vX.Y.Z` tag. Check for drift with:

```bash
comm -23 \
  <(grep -oE '^## \[[0-9]+\.[0-9]+\.[0-9]+\]' CHANGELOG.md | tr -d '## []' | sort -u) \
  <(git tag -l 'v*' | sed 's/^v//' | sort -u)
# Any output = a released version with no tag.
```

The historical backlog this check found was reconstructed and pushed on
2026-08-20 (`scripts/audit_release_tags.py --write-tags`), so the tag chain is
now unbroken through v1.0.8. The check should therefore report only the
in-development version until it is published — any OTHER output means a
release shipped without a tag.

### Create the GitHub Release on the tag (same terminal, same artifacts)

A tag says *which commit*; a Release says *what it was* and hands the reader the
artifact. Until 2026-08-26 the repo's "Latest release" on GitHub was 1.0.0 while
PyPI served 1.0.9 — the same truth-drift class as the tag backlog, one surface
over. Attach the **exact** wheel + sdist you uploaded (same `$MAXIM_RELEASE_DIR`),
and take the notes from a versioned file, never the web form:

```bash
V="$(python -c 'import maxim; print(maxim.__version__)')"
gh release create "v$V" "$MAXIM_RELEASE_DIR"/pymaxim-"$V"-py3-none-any.whl "$MAXIM_RELEASE_DIR"/pymaxim-"$V".tar.gz \
  --title "pymaxim $V" \
  --notes-file "docs/announcements/release_${V//./_}.md" \
  --verify-tag
# Pre-releases add --prerelease. Verify the attached checksums match PyPI:
sha256sum "$MAXIM_RELEASE_DIR"/pymaxim-"$V"-py3-none-any.whl "$MAXIM_RELEASE_DIR"/pymaxim-"$V".tar.gz
# Name the two files explicitly: a bare pymaxim-1.1.0* glob also matches 1.1.0rc1
# artifacts sitting in the same directory (it did, 2026-08-26 — moved by hand).
gh release view "v$V" --json assets --jq '.assets[] | "\(.name) \(.size)"'
```

The notes file (`docs/announcements/release_<version>.md`) is written from the
CHANGELOG section before the publish and is part of the release PR — facts only;
the social announcement is a separate document. **Use ABSOLUTE `https://github.com/...`
links in it**: the notes render on the Releases page, where every `../` link 404s
(v1.1.0 shipped with seven; the source was rewritten 2026-08-29).
`scripts/audit_release_tags.py --check-releases` — the `release-audit` CI job, on
push to main and nightly — asserts for every version PyPI serves that a tag exists,
a Release exists on it, the exact wheel + sdist are attached with sha256 matching
PyPI, and neither the published notes nor the notes source carry relative links.
Historical failures are grandfathered by explicit list with reasons inside the
script and printed as still-failing on every run. Every tag on PyPI must have a
Release object; the 2026-08-26 backfill added `v1.1.0rc1` (pre-release) and
`v1.0.9`; the reconstructed `v1.0.1`–`v1.0.8` tags deliberately have none.

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
# Bump to X.Y.(Z+1), fix the issue, rebuild, upload
```

---

## Files That Must Stay in Sync

| File | Field | Must Match |
|------|-------|------------|
| `pyproject.toml` | `version = "X.Y.Z"` | All three must be the same version |
| `src/maxim/__init__.py` | `__version__ = "X.Y.Z"` | |
| `CHANGELOG.md` | `## [X.Y.Z]` header | |
| git tag | `vX.Y.Z` on the published commit | created at publish time, never deferred |
| `CLAUDE.md` | "Current version:" line under *Active initiatives* | the pyproject version + a link to PyPI; NO PyPI-state prose |
| `docs/plans/README.md` | "Current version:" line at the top | same |
| `docs/index.md` | "**Version:**" line | same |

Those three lines used to describe PyPI in prose ("PyPI serves X", "pending") and
drifted on every release — corrected in the 1.1 release-truth pass and stale again
five commits later. Since 2026-08-29 they name the pyproject version and link
https://pypi.org/project/pymaxim/ for what is served, and
`scripts/lint_version_sync.py` (CI) fails the build if they name a different version,
carry "pending"/"rc"/"serves" prose, or if `CHANGELOG.md` lacks the `## [version]`
section. Policy: `main` is ahead of PyPI; the bump is the release transaction.
