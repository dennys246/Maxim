# PyPI Publication Plan

> **Status:** SUPERSEDED. Original packaging plan — problems identified here (name collision, hard deps, no public API) have all been resolved. Current publication tracking is in publication_refinement_plan.md.
> **Scope:** Packaging, dependency restructuring, metadata, CI, multi-robot generalization.
> **Depends on:** [Python API Plan](../archive/python_api_plan.md) (Phase 2 of this plan requires the verb-based API surface).
> **Blocks:** Community adoption, external contributors, downstream integrations.

---

## Problem

Maxim is architecturally ready for external use — the `RobotController` ABC, `SimulatedController`, headless mode, and simulation system all work without hardware. But the packaging makes it impossible for anyone else to install or use:

1. **`reachy-mini[gstreamer]` is a hard core dependency** despite being optional at runtime (lazy import + `ImportError` catch in `ReachyMiniController.connect()`). `pip install maxim` fails without GStreamer system libraries.
2. **No public API.** `src/maxim/__init__.py` exports only `__version__` and `get_version_info()`. There's no documented way to use Maxim as a library.
3. **The name "maxim" is taken on PyPI** (maxim-py, an OpenAI observability tool).
4. **Missing pyproject.toml metadata:** no `authors`, no `[project.urls]`, no `[tool.setuptools.package-data]`.
5. **No CI/CD.** `.github/workflows/` is empty. No automated test runs, no build validation.
6. **Robot discovery is Reachy-specific.** `hardware/__init__.py` hardcodes `reachy_mini` registration. Other robots can't be discovered without modifying Maxim's source.

## Solution

Ship Maxim as a lightweight cognitive framework with optional hardware backends. The core install should work headless/simulation-only with minimal dependencies. Robot support (Reachy or others) is opt-in via extras.

---

## Phase 0: Package Name + Metadata (~30 min)

**Goal:** Fix the metadata so the package can be built and uploaded.

### 0a. Choose a PyPI name

`maxim` is taken. Candidates (check availability on pypi.org before committing):

| Name | Pros | Cons |
|------|------|------|
| `pymaxim` | Describes the unique value | Long |
| `maxim-agent` | Short, descriptive | Generic |
| `maxim-robotics` | Clear domain | Undersells the sim/headless story |
| `maxim-mind` | Memorable, bio-flavored | Slightly whimsical |

Decision: pick one and update `pyproject.toml` `name` field. The import name stays `maxim` (users do `pip install pymaxim` then `import maxim`).

### 0b. Fill metadata gaps

```toml
# pyproject.toml additions
authors = [{name = "Denny Schaedig"}]

[project.urls]
Homepage = "https://github.com/dennys246/Maxim"
Repository = "https://github.com/dennys246/Maxim.git"
Issues = "https://github.com/dennys246/Maxim/issues"
Documentation = "https://github.com/dennys246/Maxim/wiki"

[tool.setuptools.package-data]
maxim = ["configs/templates/*.json", "configs/templates/*.yaml"]
```

### 0c. Update description + keywords

```toml
description = "Bio-inspired cognitive architecture with adaptive planning, biological memory systems, and local LLM inference. Works headless, with simulation, or connected to robots."
keywords = ["cognitive-architecture", "llm", "agentic", "planning", "embodied-ai", "robotics", "memory", "simulation"]
```

### 0d. Validate build

```bash
pip install build twine
python -m build
twine check dist/maxim-*.tar.gz dist/maxim-*.whl
```

**Files touched:** `pyproject.toml`

---

## Phase 1: Dependency Restructuring (~200 LOC)

**Goal:** Make the base install lightweight. Only ship what headless + simulation mode actually needs.

### Current core dependencies (13 packages, ~1+ GB installed)

All 13 are required today. Many are only used by specific subsystems.

### Proposed tiers

**Core (headless + simulation — the stuff everyone needs):**
```toml
dependencies = [
    "numpy>=2.2,<3.0",
    "pyyaml>=6.0",
    "json-repair>=0.30",
]
```

**Rationale:** The agent loop, memory systems, planning, and simulation orchestrator need only numpy (math), pyyaml (scenario YAML), and json-repair (LLM output parsing). Everything else is gated behind lazy imports.

**Optional extras (install what you use):**
```toml
[project.optional-dependencies]
# LLM backends
llm-local = ["llama-cpp-python>=0.3.8"]
llm-server = [
    "llama-cpp-python>=0.3.8",
    "sse-starlette>=1.6.0", "uvicorn>=0.22.0",
    "fastapi>=0.100.0", "pydantic-settings>=2.0.0",
    "starlette-context>=0.3.6", "openai>=1.0.0", "tiktoken>=0.7.0",
]
llm-anthropic = ["anthropic>=0.40.0"]
llm-openai = ["openai>=1.0.0", "tiktoken>=0.7.0"]
llm-torch = [
    "torch>=2.7", "transformers>=4.40.0", "accelerate>=0.27.0",
    "bitsandbytes>=0.43.0", "sentencepiece>=0.2.0", "huggingface-hub>=0.24.0",
]

# Perception
vision = ["opencv-python>=4.12,<5.0", "onnxruntime>=1.20,<2.0"]
audio = ["faster-whisper>=1.1.1", "ctranslate2>=4.6.0", "av>=14.0.0"]
yolo = ["ultralytics==8.3.248", "lap==0.5.12"]

# Robot hardware
reachy = ["reachy-mini[gstreamer]==1.2.6"]

# Communication
comms = ["twilio>=9.0.0", "fastapi>=0.100.0", "uvicorn>=0.24.0"]

# Research / ML
semantic = ["sentence-transformers>=2.2.0", "torch>=2.1"]
training = ["tensorflow>=2.20,<3.0", "keras>=3.13,<4.0"]
temporal = ["dateparser>=1.2.0"]
search = ["ddgs>=6.0.0"]

# Everything
all = [
    "pymaxim[llm-local,llm-anthropic,llm-openai,vision,audio,reachy,comms,search,temporal]",
]
```

### What changes in the code

Most lazy imports already exist. Audit and add guards for:

| Module | Import | Guard needed? |
|--------|--------|---------------|
| `models/language/llama_backend.py` | `llama_cpp` | Already guarded |
| `models/language/anthropic_backend.py` | `anthropic` | Already guarded |
| `models/language/openai_backend.py` | `openai` | Needs guard (currently core dep) |
| `conscience/media_loop.py` | `cv2`, `av` | Needs guard |
| `hardware/reachy/controller.py` | `reachy_mini` | Already guarded |
| `tools/introspection.py` | none | Clean |
| `integration/memory_hub.py` | `COCO_CLASSES` from `tools/reachy` | Needs guard or move |
| `environment/reachy_env.py` | reachy types | Needs guard |

**Key rule:** Any `import` of an optional dependency must be inside a function body or behind `try/except ImportError`. Module-level imports of optional deps are not allowed.

**Files touched:** `pyproject.toml`, ~10 source files for import guards

---

## Phase 2: Public API Surface

**Defined in:** [Python API Plan](../archive/python_api_plan.md)

**Summary:** Six verb-based top-level functions (`maxim.run()`, `maxim.imagine()`, `maxim.connect()`, `maxim.diagnose()`, `maxim.observe()`, `maxim.configure()`) implemented as thin facades over existing internals. Includes renaming `AUTIntrospector` -> `Observer` for LLM-friendliness, and registering `introspect` as an alias for `observe` via the existing alias pattern.

**This phase is complete when:** `import maxim; maxim.diagnose()` works from a clean install with no CLI involvement.

**Examples directory** created as part of the API plan (Phase 7):
```
examples/
  01_hello_headless.py      # maxim.run() with simulated model
  02_run_simulation.py      # maxim.imagine() with a YAML scenario
  03_observe_memories.py    # maxim.observe("memory") post-session
  04_custom_robot.py        # maxim.connect() with a custom RobotController
```

---

## Phase 3: Multi-Robot Generalization (~250 LOC)

**Goal:** Make robot discovery pluggable so third-party robots work without forking.

### 3a. Entry point-based robot discovery

Use Python entry points (the standard plugin mechanism) so robot packages register themselves:

```toml
# In a hypothetical ur5-maxim package's pyproject.toml:
[project.entry-points."maxim.robots"]
ur5 = "ur5_maxim:UR5Controller"
```

```toml
# In maxim's pyproject.toml (reachy becomes a self-registering plugin too):
[project.entry-points."maxim.robots"]
reachy_mini = "maxim.hardware.reachy:ReachyMiniController"
simulated = "maxim.hardware.simulation:SimulatedController"
```

### 3b. Auto-discovery in RobotRegistry

```python
# src/maxim/hardware/registry.py — add to __init__
def _discover_robot_plugins(self) -> None:
    """Auto-discover robot controllers from installed packages."""
    from importlib.metadata import entry_points

    for ep in entry_points(group="maxim.robots"):
        try:
            controller_cls = ep.load()
            self.register_controller_type(ep.name, controller_cls)
        except Exception:
            logger.debug("Failed to load robot plugin: %s", ep.name)
```

### 3c. Remove hardcoded registration

Current code in `hardware/__init__.py` and `conscience/selfy.py` hardcodes:
```python
_robot_registry.register_controller_type("reachy_mini", ReachyMiniController)
```

Replace with auto-discovery. The `reachy_mini` entry point handles registration. If `reachy-mini` isn't installed, the entry point isn't there, and that's fine.

### 3d. Generic tool stubs

Current `tools/reachy.py` has Reachy-specific tools (MoveTool, FocusInterestsTool, etc.). Refactor to:
- `tools/robot.py` — generic robot tools that work through `RobotController` ABC
- `tools/reachy.py` — Reachy-specific extensions (if any exist beyond the ABC)

This is the bigger refactor. The ABC already defines `goto_target()`, `goto_pixel()`, `wake_up()`, `goto_sleep()` — the tools should dispatch through these, not through Reachy SDK calls directly.

**Files touched:** `hardware/registry.py`, `hardware/__init__.py`, `conscience/selfy.py`, `tools/reachy.py` (split), `pyproject.toml` (entry points)

---

## Phase 4: CI/CD + Build Validation (~2 files)

**Goal:** Prove the package builds and tests pass before every merge.

### 4a. GitHub Actions workflow

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install ruff
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[test]"
      - run: python -m pytest tests/ -x -q --ignore=tests/integration/test_memory_hub.py

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install build twine
      - run: python -m build
      - run: twine check dist/*
```

### 4b. Community files

- `CONTRIBUTING.md` — how to run tests, submit PRs, code style (ruff)
- `CHANGELOG.md` — start tracking from 0.1.0
- `.github/ISSUE_TEMPLATE/bug_report.md` — basic bug template

**Files touched:** `.github/workflows/ci.yml`, `CONTRIBUTING.md`, `CHANGELOG.md`

---

## Phase 5: README + Description Rewrite (~1 file)

**Goal:** Make it clear what Maxim is, who it's for, and what you get within 30 seconds.

### New opening (replace first ~40 lines)

The README should answer these questions in order:
1. **What is it?** Bio-inspired cognitive architecture for autonomous agents
2. **Who is it for?** Roboticists, AI researchers, anyone building agents with memory/planning/safety
3. **What makes it different?** Biological memory systems (not just vector DBs), causal learning, safety gating, simulation-first testing
4. **How do I try it?** `pip install pymaxim && python examples/01_hello_headless.py`

### Dependency installation table

```markdown
## Installation

```bash
pip install pymaxim                        # Core (headless + simulation)
pip install pymaxim[llm-local]              # + local LLM via llama.cpp
pip install pymaxim[llm-local,vision,audio]  # + perception
pip install pymaxim[reachy]                  # + Reachy Mini robot
pip install pymaxim[all]                     # Everything
```\```

### Remove aspirational features from the feature table

Only list what works today. Move planned features to a "Roadmap" section with links to plan docs.

**Files touched:** `README.md`

---

## Phase 6: Pre-Release + Test PyPI (~30 min)

**Goal:** Validate the full publish flow before going live.

### Steps

1. Bump version to `0.1.0a1` (alpha pre-release)
2. Build: `python -m build`
3. Upload to Test PyPI: `twine upload --repository testpypi dist/*`
4. Test install in a clean venv: `pip install --index-url https://test.pypi.org/simple/ pymaxim`
5. Verify: `python -c "import maxim; print(maxim.__version__)"`
6. Run a headless example: `python examples/01_hello_headless.py`
7. If all passes, upload to real PyPI: `twine upload dist/*`

### Post-publish verification

```bash
pip install pymaxim
python -c "from maxim import Hippocampus, NucleusAccumbens, FearAgent; print('OK')"
python -c "from maxim.hardware.registry import RobotRegistry; print(RobotRegistry().get_controller_types())"
```

---

## Implementation Sequence

| # | Phase | LOC | Time estimate | Can parallel? |
|---|-------|-----|--------------|---------------|
| 0 | Metadata + name | ~20 | Quick | Yes (independent) |
| 1 | Dependency restructuring | ~200 | Medium | No (blocks Phase 6) |
| 2 | Public API surface | ~300 | Medium | Yes (with Phase 1) |
| 3 | Multi-robot generalization | ~250 | Medium | Yes (with Phase 1-2) |
| 4 | CI/CD | ~2 files | Quick | Yes (independent) |
| 5 | README rewrite | ~1 file | Quick | Yes (after Phase 1) |
| 6 | Test PyPI publish | 0 | Quick | No (last) |

**Critical path:** Phase 1 (deps) -> Phase 6 (publish). Everything else is parallel.

---

## Open Questions

1. **Package name** — needs availability check on pypi.org. `pymaxim`? `maxim-mind`? Something else?
2. **Minimum Python version** — currently 3.12. This excludes users on 3.10/3.11 (Ubuntu 22.04 ships 3.10). Worth lowering? The `X | None` syntax requires 3.10+, `match` statements require 3.10+. Check if anything requires 3.12 specifically.
3. **scipy as core dep?** — Used by IPS stats + Angular Gyrus. If these are core to the cognitive architecture, scipy stays in core deps. If they're optional subsystems, move to an extra.
4. **openai as core dep?** — Currently core but only used by cloud backends. Should move to `llm-openai` extra if cloud is opt-in.
5. **Version strategy** — Ship 0.1.0? Or start at 0.0.1 to signal "API will change"? Given active development, 0.0.1 is more honest.
6. **License scope** — Apache-2.0 is great for adoption. YOLO extra is AGPL-3.0 (already isolated). Verify no other AGPL/GPL deps leak into core.

---

## What this plan does NOT cover

- **Extracting subsystems as separate packages** (e.g., `maxim-memory`, `maxim-safety`). Premature — ship the monolith first, split if adoption demands it.
- **Documentation site** (ReadTheDocs/MkDocs). Worth doing but not a publish blocker.
- **Versioning automation** (bump2version, commitizen). Nice-to-have after first release.
- **Conda packaging.** PyPI first. Conda if scientific users request it.
