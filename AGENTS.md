# AGENTS.md — provider-neutral entrypoint (thin adapter)

> **Rewritten 2026-08-19 (operator-ratified).** The previous AGENTS.md (last
> substantive update 2026-06-06) had diverged badly from reality — wrong Python
> floor, pointers into a document structure that no longer exists, dead config
> notes. **The canonical instruction corpus is [CLAUDE.md](CLAUDE.md)** — the
> always-loaded core (commands, required checks, hard safety rules, cross-cutting
> invariants) — plus its satellite layers. This file is a deliberately thin
> adapter for agents that auto-load `AGENTS.md`; do NOT accumulate rules here.
> Substantive rules belong in CLAUDE.md (cross-cutting) or a `docs/agents/`
> brief (subsystem-scoped); this file should stay under ~80 lines.

## Read this first

1. **[CLAUDE.md](CLAUDE.md)** — the canonical core. Required checks, sim
   discipline, hard safety rules (Reachy motion safety is NOT optional reading),
   the cross-cutting `[engineering]` invariants with their regression guards,
   and the routing table below in authoritative form.
2. **The routing table** — before editing in an area, read its brief in full:

| Touching | Read first |
|---|---|
| `src/maxim/memory/`, `decisions/`, `similarity/`, `integration/memory_hub.py`, `hivemind/`, `time/`, substrate encoding | [docs/agents/bio-memory.md](docs/agents/bio-memory.md) |
| `src/maxim/models/language/`, `runtime/lane_*.py`, `runtime/function_router.py`, `runtime/leader_proxy.py`, `runtime/llm_server.py`, `peer/`, `mesh/`, `tunnel/`, `doctor/`, `utils/http.py` | [docs/agents/llm-routing.md](docs/agents/llm-routing.md) |
| `src/maxim/embodiment/`, `proprioception/`, `bridges/`, `reactions/`, `default_network/`, `embodied_runtime/`, `motion/`, robot YAMLs, **anything commanding Reachy motion** | [docs/agents/embodiment.md](docs/agents/embodiment.md) — hardware-safety section mandatory before motion code |
| `scripts/benchmark_*`, `scripts/exp*`, `scripts/orient_*`, `simulation/`, `interactive/`, `tests/behavioral/`, `docs/experiments/`, running any sim | [docs/agents/simulation-experiments.md](docs/agents/simulation-experiments.md) |
| `utils/atomic_io.py`, `utils/format_version.py`, `utils/seeding.py`, `utils/paths.py`, `runtime/config_loader.py`, `runtime/config_writer.py`, `runtime/role.py`, persisted-JSON shapes, frozen dataclasses | [docs/agents/persistence-config.md](docs/agents/persistence-config.md) |
| `runtime/agent_loop.py`, `runtime/executor.py`, `runtime/bootstrap.py`, `runtime/bio_stack.py`, `runtime/tool_dispatch.py`, `tools/`, `agents/`, `cli.py`, `api.py` | [docs/agents/runtime-tools.md](docs/agents/runtime-tools.md) |

3. **The ledgers** (read before claiming, fixing, or measuring anything):
   [docs/bugs/README.md](docs/bugs/README.md) (verified defects) ·
   [docs/limits/README.md](docs/limits/README.md) (measured instrument limits) ·
   [docs/plans/behavioral_graduation_candidates.md](docs/plans/behavioral_graduation_candidates.md)
   (behavioral claims lifecycle) · [docs/lessons/](docs/lessons/) (incident archives).

## Required checks (mirror of CLAUDE.md — that copy is authoritative)

```bash
ruff check src/ tests/ && ruff format src/ tests/
python -m pytest tests/ -x -q -m "not slow" --ignore=tests/integration/test_memory_hub.py
# touching memory/, decisions/, integration/: also run tests/integration/test_memory_hub.py
# touching api.py/session.py/create.py/load.py/__init__.py: run the 5-file mypy check (see CLAUDE.md)
```

## Hard rules that fire in ANY area (details + guards in CLAUDE.md)

- Sims from agents/scripts: **always `--interactive false`**; configure model/n_ctx
  via `maxim config`, never transient env; never co-locate a harness with a leader's
  LLM consumer; one harness at a time.
- Any harness spawning `maxim` calls `scripts/_provenance.py::assert_repo_interpreter`
  first — a result whose code-under-test cannot be established is not a result.
- No new bare `except Exception: pass`; persistence via `atomic_io`; persisted hashes
  via `utils/seeding.py` stable hashes, never builtin `hash()`.
- Reachy motion ONLY through `ReachyMiniController.goto_target` / `motion/movement.py::move_head`
  — motors have been physically destroyed by bypasses. Read the embodiment brief first.
- Cross-system naming conventions: see docs/agents/bio-memory.md §4b (rehomed from
  the old AGENTS.md).
- ≥2 concurrent agent sessions → each takes its own git worktree at the START.

## Repo etiquette

- PR open = branch frozen; follow-up work on a new branch. Review rounds run
  BEFORE merge; fold findings into the same branch.
- New invariants enter as `[engineering]` with a `Regression guard:` line, in
  CLAUDE.md or the owning brief — never in this file.
