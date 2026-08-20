# PyPI Maintenance Guide

Long-term maintenance reference for the `pymaxim` package on PyPI. For first-time publication setup, see [publication_guide.md](publication_guide.md).

---

## Versioning policy

- **Semver,** `MAJOR.MINOR.PATCH`. The current line is `0.x` — research preview, API may move between minor versions.
- **Bump rules:**
  - **Patch** (`0.2.1 → 0.2.2`) — bug fixes, doc-only releases, packaging hotfixes, no API changes.
  - **Minor** (`0.2.x → 0.3.0`) — new features, additive API changes, may include breaking changes while in `0.x`.
  - **Major** (`0.x → 1.0.0`) — reserved for the version that demonstrably improves on a task across sessions without fine-tuning the underlying LLM, with a test that proves it. See [plans/substrate_p0_pilot.md](plans/archive/substrate_p0_pilot.md) and the substrate plan series.
- **Two version files must stay in sync:** [pyproject.toml](../pyproject.toml) and [src/maxim/__init__.py](../src/maxim/__init__.py). Mismatch = release bug.
- **Verify after bump:**
  ```bash
  python -c "from maxim import get_version_info; print(get_version_info())"
  grep -n version pyproject.toml | head -3
  ```

## Pre-publish checklist

Run before every release, in order:

```bash
# 1. Lint + format clean
ruff check src/ tests/
ruff format src/ tests/

# 2. Fast test suite green
python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py

# 3. Public API mypy clean (if api.py / session.py / __init__.py changed)
mypy src/maxim/__init__.py src/maxim/api.py src/maxim/session.py \
     src/maxim/create.py src/maxim/load.py --ignore-missing-imports

# 4. Version sync confirmed
grep -n '^version' pyproject.toml
grep -n '__version__' src/maxim/__init__.py

# 5. Canonical website metadata is release-ready
grep -A5 '^\[project.urls\]' pyproject.toml
# Then complete publication_guide.md's pymaxim.bio content/link audit.

# 6. CHANGELOG entry exists for the new version
head -20 CHANGELOG.md
```

If any step fails, do not proceed.

## Build + check + upload

```bash
# Always start clean — local build/ shadows the installed `build` package
rm -rf build/ dist/pymaxim-<OLD-VERSION>*

# Build wheel + sdist
python -m build

# Validate metadata renders + license file is included
twine check dist/*

# Smoke check: confirm the wheel does NOT contain repo-root junk
python -m zipfile -l dist/pymaxim-*.whl | grep -iE "htmls-guides|outputs|^[^/]*sandbox" \
  && echo "FAIL: wheel contains repo-root files" || echo "OK: wheel clean"

# Upload (always with --verbose so 4xx errors are diagnosable)
twine upload --verbose dist/pymaxim-<NEW-VERSION>*
```

After successful upload:

```bash
# Tag the release in git
git tag -a v<NEW-VERSION> -m "Release notes summary"
git push origin v<NEW-VERSION>

# Smoke test the published wheel from a clean venv
python -m venv /tmp/pymaxim-verify && source /tmp/pymaxim-verify/bin/activate
pip install pymaxim==<NEW-VERSION>
python -c "import maxim; print(maxim.__version__)"
deactivate && rm -rf /tmp/pymaxim-verify
```

## Token management

- **Account-scoped token** lives in `~/.pypirc` under `[pypi]` and is the default for any project.
  - Required for first-ever publish of a new project name (project-scoped tokens cannot create projects).
  - Larger blast radius if leaked — rotate annually or after any suspected exposure.
- **Project-scoped tokens** are stricter: locked to one existing project.
  - Useful for CI or shared maintainer setups.
  - Generate at https://pypi.org/manage/account/token/ → "Add API token" → scope to project.
- **Rotation:** generate new token, replace in `~/.pypirc`, delete old token at https://pypi.org/manage/account/token/.
- `~/.pypirc` template supporting multiple projects:
  ```ini
  [distutils]
    index-servers =
      pypi
      <other-project>

  [pypi]
    username = __token__
    password = pypi-<account-scoped-token>

  [<other-project>]
    repository = https://upload.pypi.org/legacy/
    username = __token__
    password = pypi-<project-scoped-token>
  ```

## Yanking vs deleting

- **Yank** (preferred for bad releases): marks a version as "do not use" but leaves it installable for anyone who pinned it. Reversible.
  ```bash
  # Via PyPI web UI: project page → Manage → Releases → Yank
  ```
- **Delete** is irreversible AND **does not free the version slot**. Once a version string has been used, it is permanently reserved — even after deletion, you cannot re-upload `0.2.1` after deleting `0.2.1`. You must bump to `0.2.2`.
- Default to yanking. Only delete if the release contains secrets, malware, or licensed content that must be removed.

## TestPyPI for risky releases

Use https://test.pypi.org for any release where the metadata or build process changed materially — major version bumps, dependency reshuffles, build-system migrations. TestPyPI's version slots are independent from real PyPI, so you can burn them freely.

```bash
# Add TestPyPI to ~/.pypirc
[testpypi]
  repository = https://test.pypi.org/legacy/
  username = __token__
  password = pypi-<testpypi-token>

# Upload + verify
twine upload --repository testpypi dist/pymaxim-*
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            pymaxim==<VERSION>
```

If TestPyPI upload + install both work, real PyPI will too.

## Diagnosing upload failures

**Always run `twine upload --verbose dist/*` on the first failure.** Without `--verbose`, twine prints only "400 Bad Request" with no body. The verbose output contains the actual server error.

Common 400/403 causes and fixes:

| Symptom | Cause | Fix |
|---|---|---|
| `400 'Topic :: X :: Y' is not a valid classifier` | Classifier not on PyPI's trove list. `twine check` does not validate classifiers — only README rendering. | Remove the classifier or check against the [`trove-classifiers`](https://pypi.org/project/trove-classifiers/) package. |
| `400` with no specific reason, `License:` field contains full text | PyPI Metadata 2.4 rejects embedded license text. | Use `license = "Apache-2.0"` (SPDX) + `license-files = ["LICENSE"]` in pyproject.toml. Remove the legacy `License :: OSI Approved :: ...` classifier. |
| `400 File already exists` | Version slot was previously used (even partial uploads sometimes count). | Bump version. PyPI version slots are immutable. |
| `403 Invalid API Token: project-scoped token is not valid for project: 'X'` | Token is scoped to a different project, or to no project at all. | Use an account-scoped token, or generate a new project-scoped token for the correct project. |
| `403` after recent token rotation | Old token still cached. | Verify `~/.pypirc` was actually saved; check there's no stray comment after `password = pypi-...` (some INI parsers append the comment to the value). |
| `python -m build` fails with `No module named build.__main__` | Local `build/` directory at repo root shadows the installed `build` package. | `rm -rf build/` (it's gitignored, regenerated each build). |

## Post-release housekeeping

- Update [CHANGELOG.md](../CHANGELOG.md) — every release needs an entry, even patch releases. Format follows [Keep a Changelog](https://keepachangelog.com/).
- Tag the commit (`git tag -a vX.Y.Z`) so the release is reproducible from git.
- Update https://pypi.org/manage/projects/pymaxim/ if any project URLs changed.
- Announce in CHANGELOG-driven channels (GitHub releases page, social, etc.) — the release isn't really "out" until users know it exists.
- Smoke install from a clean venv on a different machine when possible — catches "works on my dev box" failures that the wheel inspection misses.

## When in doubt

- **Don't infer error causes from indirect evidence.** Get the actual server response with `--verbose`. See `~/.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_verify_before_inferring.md` for the cautionary tale.
- **Don't burn version slots speculatively.** Use TestPyPI for dry runs.
- **Don't delete releases unless legally required.** Yank instead.
- **Don't ship without smoke-installing the published artifact.** A green local build is necessary but not sufficient.
