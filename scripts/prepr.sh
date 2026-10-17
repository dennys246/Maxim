#!/usr/bin/env bash
# Thin wrapper: the CI lint job's steps + the targeted mypy check, read from .github/workflows/test.yml (backlog M59).
# Everything lives in scripts/prepr.py (its docstring has the inputs-and-how-each-could-lie table and the skip rule).
set -euo pipefail
if ! python3 -c "import yaml" 2>/dev/null; then
    echo "ERROR: prepr needs PyYAML in the python3 on PATH ($(command -v python3 || echo 'no python3')):" >&2
    echo "       activate the repo's venv, or: python3 -m pip install 'pyyaml>=6.0,<7.0'" >&2
    echo "prepr: REFUSED (exit 2)"
    exit 2
fi
exec python3 "$(dirname "${BASH_SOURCE[0]}")/prepr.py" "$@"
