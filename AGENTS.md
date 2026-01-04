# Agent Rules (Maxim)

Maxim is a Reachy Mini project for capturing audio/video, running perception + motor learning, and controlling the robot in real time.

## Standards (Project Defaults)
- Prefer small, surgical changes; minimize refactors unless explicitly requested.
- When possible reduce the size of code to simplified systems.
- Preserve behavior unless the user asks for functional changes.
- Keep runtime loops responsive: push heavy I/O and compute off the control loop (threads/processes + queues).
- Make logging human-friendly; use `src/maxim/utils/logging.py` and respect `--verbosity`.
- Keep importable Python code under `src/maxim/` (avoid new top-level packages under `src/` unless packaging is updated).
- Avoid reusable nested functions: if a helper could be reused, define it at module scope (or under `src/maxim/utils/`) instead of inside another function/method. Nested defs are OK when they must capture closure state (e.g., worker threads) or are truly one-off.
- Any public API or CLI change must be reflected in `DECISIONS.md` and (when user-facing) `README.md`.
- Add in concise commenting about import code functionality or nuanced.
- Build modules for scalability to be applied to multiple sensory modalities (e.g., diffusion models applied to both images and audio).
- Run additional analysis on the security of code. If an insecurity is identified within the repo, analyze the bug for potential fixes and notify the user alongside the fix.
- If files or data are created in the `sandbox/`, delete old and un-necessary files and data if no longer being used.

## Allowed Actions
- Modify code under `src/`.
- Add/modify smoke tests under `src/tests/` (offline-by-default; provide explicit opt-in for robot/network).
- Update documentation (`README.md`, `DECISIONS.md`, `ARCHITECTURE.md`).
- Creating new file within a `src/` folder if another file would better seperate module functionality, always request approval first.
- Create files and data in the `sandbox/` folder for creating experimental functionality and to be added directly into the repo if useful.

## Forbidden / Avoid
- Avoid excessively long additions or extensive refactoring without explicit requests.
- Large refactors without explicit task instruction.
- Breaking public imports/entrypoints without providing a compatibility layer.
- Adding network-requiring steps to default tests (e.g., model downloads) without an opt-in flag.
- Adding code outside of pre-existing architectures when a pre-existing build component could be
- Adding in functions within functions that could be used in other functions or libraries. Attempt to add to their respective src/maxim/ folders and scripts.
- Using insecure libraries, configurations, or practices even with explicit permission from the user if forbidden.

## Coding Guidelines
- Python `>=3.12` (see `pyproject.toml`).
- Follow existing repo style; keep code straightforward and readable.
- Add type hints where they improve clarity; prioritize stable interfaces for cross-module use.

## When Uncertain
- Ask for clarification about desired runtime behavior (e.g., “record everything” vs “latest snapshot”).
- Don’t guess domain logic (robot kinematics, label semantics, training targets); prefer small instrumentation/logging to validate assumptions.
