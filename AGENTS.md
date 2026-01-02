# Agent Rules (Maxim)

Maxim is a Reachy Mini project for capturing audio/video, running perception + motor learning, and controlling the robot in real time.

## Standards (Project Defaults)
- Prefer small, surgical changes; minimize refactors unless explicitly requested.
- Preserve behavior unless the user asks for functional changes.
- Keep runtime loops responsive: push heavy I/O and compute off the control loop (threads/processes + queues).
- Make logging human-friendly; use `src/utils/logging.py` and respect `--verbosity`.
- Any public API or CLI change must be reflected in `DECISIONS.md` and (when user-facing) `README.md`.

## Allowed Actions
- Modify code under `src/`.
- Add/modify smoke tests under `src/tests/` (offline-by-default; provide explicit opt-in for robot/network).
- Update documentation (`README.md`, `DECISIONS.md`, `src/ARCHITECTURE.md`).

## Forbidden / Avoid
- Avoid excessively long additions or extensive refactoring
- Large refactors without explicit task instruction.
- Breaking public imports/entrypoints without providing a compatibility layer.
- Adding network-requiring steps to default tests (e.g., model downloads) without an opt-in flag.

## Coding Guidelines
- Python `>=3.12` (see `pyproject.toml`).
- Follow existing repo style; keep code straightforward and readable.
- Add type hints where they improve clarity; prioritize stable interfaces for cross-module use.

## When Uncertain
- Ask for clarification about desired runtime behavior (e.g., “record everything” vs “latest snapshot”).
- Don’t guess domain logic (robot kinematics, label semantics, training targets); prefer small instrumentation/logging to validate assumptions.
