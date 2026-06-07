# PyPI Publication Guide

Step-by-step guide for publishing pymaxim to PyPI.

**Current version:** 0.9.3 (preparing for 1.0.0)
**Package name:** pymaxim (import name: `maxim`)
**Build system:** setuptools + wheel

---

## Pre-Publication Checklist

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
# Full suite (exclude known slow integration test)
python -m pytest tests/ -q --ignore=tests/integration/test_memory_hub.py

# Expected: 7800+ passed, 0 failed (1 pre-existing flaky ordering issue in test_lane_backends passes in isolation)
```

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
# Both must show the same version (e.g. 1.0.0)
grep 'version = ' pyproject.toml
python -c "import maxim; print(maxim.__version__)"
```

---

## Build Steps

```bash
# 1. Clean old builds
rm -rf dist/ build/ *.egg-info

# 2. Build wheel + sdist
python -m build

# 3. Validate package metadata
twine check dist/pymaxim-*
# Expected: PASSED for both .whl and .tar.gz

# 4. Verify bundled data is in the wheel
python -c "
import zipfile, glob
whl = sorted(glob.glob('dist/pymaxim-*-py3-none-any.whl'))[-1]
with zipfile.ZipFile(whl) as z:
    data = [f for f in z.namelist() if '_data/' in f]
    print(f'{len(data)} bundled data files')
    assert len(data) >= 25, 'Expected 25+ bundled files'
    # Check key files
    names = z.namelist()
    assert 'maxim/py.typed' in names, 'Missing py.typed'
    assert 'maxim/__main__.py' in names, 'Missing __main__.py'
    print('All checks passed')
"
```

---

## Publish to Test PyPI (safe, reversible)

```bash
# 1. Upload to Test PyPI
twine upload --repository testpypi dist/pymaxim-*

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
twine upload dist/pymaxim-*

# Verify real install
pip install pymaxim
python -c "import maxim; print(maxim.__version__)"
```

---

## Post-Publication Priorities

### Week 1-2

4. **Substrate-primary harness (B5)** (~1,360 LOC) — Phase -1 prototype + Phase 0 cradle-prelinguistic harness + Hivemind shareability infrastructure. See [substrate_primary.md](substrate_primary.md), [hivemind.md](hivemind.md), and [docs/plans/grounded_language_acquisition.md](plans/grounded_language_acquisition.md). NOTE: roadmap pivoted 2026-05-09; the original "Mother Maxim MVP + deidentification" track was superseded by the federated peer-to-peer Hivemind + Oasis architecture (~55% the LOC, more capable). Phase -1 already shipped.
5. **`--list-models` CLI flag** — model discovery for new users

### Month 1

6. **Maxim Oasis MVP (1.1)** — single-Oasis instance hostable on Mac Mini class hardware. CLI: `maxim oasis serve`. LLM-AUT users opt in to contribute via `maxim contribute --to oasis://...`. ~800 LOC.
7. **Substrate-primary AUT mode (1.1)** — Phase 0 validation runs (raw substrate, no Hivemind); Phase 1 (vocabulary-constrained) starts.
8. **Maxim Hivemind P2P protocol (1.2)** — peer discovery, substrate-snapshot exchange, conflict-resolution semantics, poison-resistance defenses. ~600 LOC.

---

## Version Strategy

| Version | What |
|---------|------|
| 1.0.0 | Stable API; substrate-primary harness (B5: Phase -1 + Phase 0 + Hivemind shareability infrastructure, all behind experimental flag); D1-D3 docs complete |
| 1.1.x | Substrate-primary AUT mode lands (`--aut-mode substrate-primary`); first hostable Maxim Oasis; Phase 1 grounded-language work begins |
| 1.2.x | Full Maxim Hivemind P2P protocol; multi-Oasis federation; substrate-primary Maxims pull bootstrap from Hivemind; Phase 2 (symbol binding) starts |
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
