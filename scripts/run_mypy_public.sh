#!/usr/bin/env bash
# The targeted mypy check: public API surface + hivemind/ + the composition layer. The ONE place its file list
# lives (backlog M65): CLAUDE.md's required check and the unit-tests job's "mypy public API surface" step both run
# this script, and tests/unit/test_ci_workflow_shape.py fails when either carries the list again.
#
#   scripts/run_mypy_public.sh            # from anywhere inside the checkout
#
# Why these files: hivemind/ per 1.2 gate 8 (the bundle format is a wire boundary); the composition layer
# (executor, agent_loop and its loop_* slices, bridges/, planning/) per roadmap 1.3.2 item 4 (the #840/#841 silent
# seams were argument mismatches swallowed by a broad except, which mypy catches). --follow-imports=silent scopes
# error reporting to the checked files (the rest of the codebase is not mypy-clean yet); --warn-unused-ignores keeps
# #840's line-ignores from going stale silently. CI pins mypy (test.yml: "mypy==1.20.0"); a different local version
# can disagree. NOT the lint job's "mypy per-file ratchet" (scripts/lint_mypy_ratchet.py): neither green implies the
# other.
set -euo pipefail

FILES=(
    src/maxim/__init__.py
    src/maxim/api.py
    src/maxim/session.py
    src/maxim/create.py
    src/maxim/load.py
    src/maxim/hivemind/
    src/maxim/runtime/executor.py
    src/maxim/runtime/agent_loop.py
    src/maxim/runtime/loop_setup.py
    src/maxim/runtime/loop_external.py
    src/maxim/runtime/loop_gates.py
    src/maxim/runtime/loop_substrate.py
    src/maxim/runtime/substrate_proposal.py
    src/maxim/runtime/loop_planning.py
    src/maxim/runtime/loop_perception.py
    src/maxim/runtime/loop_controller.py
    src/maxim/runtime/loop_state.py
    src/maxim/runtime/tool_dispatch.py
    src/maxim/runtime/loop_types.py
    src/maxim/simulation/tty_stdin.py
    src/maxim/bridges/
    src/maxim/planning/
)

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
exec mypy "${FILES[@]}" --ignore-missing-imports --follow-imports=silent --warn-unused-ignores
