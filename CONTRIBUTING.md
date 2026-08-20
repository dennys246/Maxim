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
- **Imports:** Use lazy/deferred imports inside function bodies for optional dependencies. Module-level imports only for the seven core deps (numpy, scipy, pyyaml, json-repair, httpx, rich, filelock — `pyproject.toml` is the authority).
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

## Pre-merge two-lens review

**For any change that touches ≥~500 lines of `src/` (a new mechanism, a bus/bridge/builder, a bio-system, a routing layer, a CLI subcommand), run a two-lens review *before* opening the PR — not after.**

Spawn two independent reviews of the branch:

- **Executor lens** — does each code path do what it claims under the inputs it will actually see? Checks for silent no-ops, swallowed exceptions, env-var races, wrong-direction comparisons, and call sites that diverge from the contract they advertise.
- **Architecture lens** — does the change fit the layer graph and the existing invariants? Checks for band-aids (special cases, flags that toggle around broken behavior), duplicated source-of-truth, and mechanisms that should ride on existing infrastructure instead of being their own thing.

Fold the findings into the **same branch** with a follow-up commit before the PR opens. **Fold cross-confirmed findings first** — when both lenses flag the same issue independently, that is the strongest signal the finding is real and should be trusted even when it feels minor.

Why this is non-optional and not ceremony: across the LLM-path refinement rounds (R1/R2/R3) the review surfaced correctness bugs (an env-var race, a probe mis-classification, a canonical-shim bypass) that **the test suite caught zero of** — they were correctness issues in input spaces tests didn't cover. Do *not* merge first and ship a `fix/<feature>-loose-ends` PR after; that splits the bisect surface and leaves known-buggy code on `main`.

## Adding an env var

Every `MAXIM_*` env var read in a startup or hot path must land with its scrub in the **same commit**:

1. **Add it to the env-var table in `CLAUDE.md`** (the authoritative registry) with a one-line description of what it does, its default, and the file that reads it.
2. **Add an autouse `_isolate_*` scrub fixture in `tests/conftest.py`** that pops the var before each test and restores it after. Copy the shape of an existing one (e.g. `_isolate_maxim_nac_reward_bias_disabled`): `saved = os.environ.pop("MAXIM_FOO", None)` → `yield` → restore in `finally`.

The scrub is load-bearing whenever the var gates a side effect reachable from runtime construction (`build_primary_router`, auto-spawn, tier detection, `ensure_available`, ...). Without it, **any** test that sets the var leaks it into every later test that builds the runtime, and the leaked side effect runs for real. This is the rule earned by the **P5 9-minute pytest hang** on a real 1 GB GGUF download to `~/.maxim/`: a `normalize_args` unit test set `MAXIM_AUTO_DOWNLOAD_MODELS=1`, it leaked, and a later test's runtime construction acted on it. `_isolate_maxim_llm_profile_env` and `_isolate_maxim_auto_download_env` are the template fixtures.

## CI invariant gates

Several architectural invariants are enforced by `grep` allow-lists in [`.github/workflows/test.yml`](.github/workflows/test.yml) — a `grep` over `src/` (and sometimes `tests/`) that fails CI on a forbidden pattern, with an explicit allow-list of the file(s) permitted to match. Examples currently enforced:

- `urllib.request.urlopen` is banned everywhere except `utils/http.py` (all outbound HTTP routes through the registry-backed module).
- `maxim_peer_backend.py` may not contain `retry`/`backoff`/`gateway` (the one-HTTP-call, no-internal-retry invariant) except the two whitelisted parameter names.
- `write_config` / `mutate_config` / `set_field` callers are restricted to `config_writer.py` + `config_cli.py` + a short allow-list (single-writer for `config.json`).
- `write_mesh_config` callers are restricted to `mesh_setup.py` (single sanctioned writer for `mesh.yml`).
- `yaml.load(` is banned in favor of `yaml.safe_load`; deprecated probe shims and dead mesh modules are blocked from new callers.

**When you add a new gate, or legitimately add a new allowed call site, update the allow-list in `test.yml` and the matching `[engineering]` invariant in `CLAUDE.md` in the same commit.** The grep pattern and the CLAUDE.md `Regression guard:` line are a pair — one without the other drifts. If you find yourself wanting to add many call sites to a strict allow-list across successive PRs, that's a signal the rule needs revisiting rather than another entry; raise it instead of silently widening.

## Adding a bio-system

Bio-systems (buses, the memory hub, the default network) are constructed through **canonical `build_X` builders**, never by raw construction in production code. The rules:

- **Use the canonical builder.** PainBus → `proprioception/pain_bus.py::build_pain_bus`; ReactionBus → `reactions/bus.py::build_reaction_bus`; MemoryHub → `integration/memory_hub.py::build_memory_hub`; DefaultNetwork → `runtime/bootstrap.py::build_default_network`; the full pipeline → `runtime/bio_stack.py::build_bio_stack`.
- **Required keyword-only learning subjects.** Builders push silent-no-op invariants into the signature so forgetting is a `TypeError`, not a missing-learning bug. `build_pain_bus` requires `hippocampus=` and `nac=` (pass `None` to *explicitly* opt out). `build_memory_hub` requires the four core bio-systems `hippocampus=`, `scn=`, `nac=`, `ec=`. `build_default_network` requires `nac=`.
- **`agent_id=` is required.** `build_bio_stack` and `build_memory_hub` take a required keyword-only `agent_id=` — the old `default_agent` default silently split the substrate-write and substrate-read paths in multi-agent runs. Test sites that don't care pass the literal `"default_agent"` explicitly.
- **`_allow_raw=True` is for tests only.** Raw `PainBus()` / `ReactionBus()` / `MemoryHub()` construction raises `TypeError` (the C6 hard-error flip). Tests that genuinely need a bare instance pass `_allow_raw=True`; production code never does. The bio-system *classes* (`Hippocampus`, `NAc`, `SCN`, `EC`, `ATL`, `AngularGyrus`) are intentionally constructible directly — only the buses and the hub are gated.
- **Wire into `build_bio_stack` in dependency order.** A new bio-system that the full pipeline owns gets composed in `build_bio_stack`, which builds the Wave 1+2 builders in the correct order (ReactionBus before PainBus, etc.). Don't hand-roll the construction in an entry point; add it to the stack so every agent path picks it up.

Don't rename the bio-system classes — the names are load-bearing for the mental model.

See [`docs/architecture/structural_enforcement.md`](docs/architecture/structural_enforcement.md) for the full catalog of which invariants are pushed into types vs. enforced by review.

## Adding a dependency-gated feature

Maxim ships a small core (`numpy`, `scipy`, `pyyaml`, `json-repair`, `httpx`, `rich`, `filelock` — seven packages; `pyproject.toml` is the authority) and gates everything heavier behind optional extras. **Do not write a bare `try: import X except ImportError: ... return None` block** — route through `maxim/utils/optional_deps.py`, which makes the behavior uniform and intentional, split by intent:

- **`require_optional_dependency(import_name, *, extra=None, feature=None)`** — for a feature the caller *explicitly* requested (a configured cloud backend, a `--vision` flag, a selected TTS voice). A missing dependency here is a setup error, so this **raises** `OptionalDependencyError` with an actionable `pip install pymaxim[<extra>]` hint. `OptionalDependencyError` subclasses `ImportError`, so legacy `except ImportError` sites still catch it, but fail-soft layers (the LLM router) can distinguish "requested feature not installed → abort loudly" from "transient runtime failure → try next provider".
- **`optional_dependency_available(import_name)`** — for a capability *probe* ("use X if it happens to be present", GPU detection). Absence is normal here, so this returns a `bool` and never logs.
- **`warn_optional_fallback(import_name, *, fallback, extra=None, feature=None)`** — for a deliberate degradation path that has a real working fallback (`sentence-transformers` → bag-of-words embeddings, `spacy` → identity decomposition, `ddgs` → regex scraping). The caller keeps the fallback; this emits exactly **one** deduped log so the degradation is visible instead of silent.

The motivating failure (2026-06-05 cloud-dispatch incident): `anthropic` was not installed, the backend logged a single swallowed WARNING and returned `None`, the router treated the empty response as a transient hiccup, and the whole sim completed at `cost=$0` with every action a `_llm_unavailable` fallback — the missing backbone was invisible. The `EXTRA_FOR_IMPORT` table in `optional_deps.py` maps import names to extras; add new optional deps there so the fix hint is accurate.

## Key architecture docs

| Doc | What it covers |
|---|---|
| [`docs/architecture/llm_routing.md`](docs/architecture/llm_routing.md) | LLM router, provider fallback, lane tiers (`large`/`medium`/`small`), `_MaximPeerBackend` vs `_OpenAIBackend`, backend dispatch table |
| [`docs/architecture/structural_enforcement.md`](docs/architecture/structural_enforcement.md) | Which invariants are pushed into types (required kwargs, frozen dataclasses, `_allow_raw` gates) vs. enforced by CI grep vs. by review |
| [`docs/user/tool_side_effects.md`](docs/user/tool_side_effects.md) | Authoritative registry of well-known `ToolOutput.side_effects` keys (`embodiment_failures`, `entity_acquired`/`entity_released`, `affordance_blocked`) |

## Commit Message Format

```
type: short description

Longer description if needed.

Co-Authored-By: Your Name <email>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## Adding New Features

### Adding a tool
1. Create in `src/maxim/tools/` or extend an existing tool file.
2. Inherit from the `Tool` base class (`tools/base.py`).
3. Register in the tool registry.
4. Add tests in `tests/unit/`.

When implementing a non-trivial tool, mind these `Tool` contract surfaces:

- **`input_schema` is dual-format; export via `to_json_schema()`.** Authors may write either the legacy custom format (`{NAME: type}` / `{NAME: (type, default)}` / `{NAME: "description"}`) or JSONSchema 2020-12 (`{"type": "object", "properties": {...}, "required": [...]}`). `Tool.__init__` does *not* normalize — it leaves `input_schema` exactly as authored. Any consumer that reads the schema (prompt rendering, MCP export) **must route through `tool.to_json_schema()`** rather than iterating `input_schema.items()` directly, or `@maxim.tool` JSONSchema tools silently render with empty params. The custom format is strictly less expressive (no `enum`, `pattern`, `oneOf`, nested objects) — author JSONSchema directly when you need those.
- **`side_effects` is the typed bio-pipeline channel.** `ToolOutput.side_effects: dict[str, Any] | None` carries well-known keys (`embodiment_failures`, `entity_acquired` / `entity_released`, `affordance_blocked`) consumed by the bio pipeline. The append-only registry of keys is authoritative in [`docs/user/tool_side_effects.md`](docs/user/tool_side_effects.md) — add a new key there *and* wire its consumer (or mark it `informational`) in the same PR. Don't hijack `output` (the main result) or `metadata` (caller-facing extras) for bio signals.
- **`cancel()` is a thread-safe no-op by default.** The base `Tool.cancel()` returns `None` so existing tools work unchanged. Tools that do heavy work (HTTP, subprocess, large reads) override it to set a `threading.Event` the executing thread checks at safe points. `cancel()` is called from a *different thread* than `execute()`, must be thread-safe, and must never raise. No 1.0 dispatch path calls it — it's forward-compat for 1.1+ cancellation work; don't make it `@abstractmethod`.
- **`auto_fire=True`** marks a tool the agent loop dispatches every tick for passive perception (e.g. the SEM sense tools), rather than only when the agent proposes it. Set it on the dataclass; the registry exposes them via `get_auto_fire_tools()`.

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
