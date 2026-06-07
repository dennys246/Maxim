# Contributing to Maxim

Thank you for your interest in contributing to Maxim! This document provides guidelines for contributing code, documentation, and bug reports.

## Getting Started

```bash
# Clone and install in development mode
git clone https://github.com/dennys246/Maxim.git
cd Maxim
pip install -e ".[test]"

# Run tests
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

## Code Style

- **Linter:** ruff (configured in `pyproject.toml`)
- **Line length:** 120 characters
- **Imports:** Use lazy/deferred imports inside function bodies for optional dependencies. Module-level imports only for core deps (numpy, scipy, pyyaml, json-repair, rich).
- **Naming:** Don't rename bio-system classes (Hippocampus, ATL, NAc, SCN, EC, AngularGyrus) — names are load-bearing for the mental model.

## Architecture Rules

Maxim has a strict layer dependency graph. See `ARCHITECTURE.md` for full details.

**Key rules:**
- Agents never call tools directly — they propose intents, the executor dispatches.
- Memory tier progression is one-way: FORMING → SHORT_TERM → LONG_TERM. Active-reference context is in `WorkingMemorySet` (Exec-owned), not a tier.
- LLM access goes through `models/language/router.py` — don't import backends directly.
- Persistence uses `maxim.utils.atomic_io.atomic_write_json` — don't hand-roll `open().write()` + `os.replace()`.
- User data lives in `~/.maxim/`, bundled defaults in `src/maxim/_data/`.

## Testing

```bash
# Fast: just the module you changed
python -m pytest tests/unit/test_your_module.py -v

# Full suite (exclude known slow integration test and slow-marked tests)
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py

# If touching memory/, decisions/, integration/:
python -m pytest tests/integration/test_memory_hub.py -q
```

**Test markers:**
- `@pytest.mark.slow` — deselect with `-m "not slow"`
- `@pytest.mark.integration` — requires multiple subsystems
- `@pytest.mark.robot` — requires Reachy hardware
- `@pytest.mark.learning` — convergence/ML tests

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes with tests
4. Run `ruff check src/ tests/` and `python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py`
5. Commit with a descriptive message (see format below)
6. Push to your fork and open a PR

## Commit Message Format

```
type: short description

Longer description if needed.

Co-Authored-By: Your Name <email>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## Adding New Features

### Adding a tool
1. Create in `src/maxim/tools/` or extend an existing tool file
2. Inherit from `Tool` base class
3. Register in the tool registry
4. Add tests in `tests/unit/`

### Adding a SEM component
1. Create YAML in `src/maxim/_data/components/{category}/`
2. Follow the component format (see existing files for examples)
3. Test via `ComponentRegistry.get()` and `instantiate()`

### Adding an encounter template
1. Create YAML in `src/maxim/_data/encounters/{category}/`
2. Follow the encounter format (see existing files)
3. Test via `EncounterLibrary.get()`

### Adding a simulation persona
1. Add entry to `SIMULATION_PERSONAS` in `src/maxim/simulation/personas.py`
2. Or use `maxim.register_persona()` at runtime

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

**Exception:** The `[yolo]` extra uses AGPL-3.0 licensed code (ultralytics). Contributions to YOLO-related code are licensed under AGPL-3.0.

## Questions?

Open an issue at https://github.com/dennys246/Maxim/issues
