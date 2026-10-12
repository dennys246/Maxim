#!/usr/bin/env bash
# Run the fast suite (and test_memory_hub.py) on a COMMIT, in a throwaway worktree, so edits in the working
# tree cannot invalidate the run and the result names the commit it tested.
# docs/lessons/development-flow-speed.md ("A fold never invalidates a running suite").
#
#   scripts/suite_at_commit.sh [--no-cov] [<commit>]            # default: HEAD
#   scripts/suite_at_commit.sh "$(git stash create)"  # tracked changes, no commit (and no signature)
#
# By default the fast suite runs with CI's exact coverage command and is followed by
# `scripts/lint_coverage.py --local`, so the coverage ratchet's diff-coverage gate and the floors of the packages
# the diff touches fail HERE, on the exact commit about to be pushed, not in CI (#1201, #1213, #1232 and #1240 each
# failed CI on it). The untouched packages' floors are only noted: they are measured in CI's environment.
# `--no-cov` skips the measurement for a quicker run that is not the pre-push check.
#
# No -x, deliberately: a full failure list is worth more than the first one.
set -euo pipefail

cov=1
if [ "${1:-}" = "--no-cov" ]; then
    cov=0
    shift
fi
if [ "$#" -gt 1 ]; then
    echo "usage: scripts/suite_at_commit.sh [--no-cov] [<commit>]" >&2
    exit 2
fi
here="$(git rev-parse --show-toplevel)"
commit="$(git -C "$here" rev-parse --verify "${1:-HEAD}^{commit}")"
short="${commit:0:12}"
# Anchor every snapshot at the MAIN checkout (not whichever worktree we were run from), under the
# gitignored .worktrees/, with a unique name so two runs on one commit never share (or delete) a tree.
main="$(cd "$(git -C "$here" rev-parse --git-common-dir)/.." && pwd)"
mkdir -p "$main/.worktrees"
git -C "$main" worktree prune
tree="$(mktemp -d "$main/.worktrees/suite-$short.XXXXXX")"
rmdir "$tree"  # worktree add wants to create it

git -C "$main" worktree add --detach --quiet "$tree" "$commit"
cleanup() {
    cd "$main"
    git -C "$main" worktree remove --force "$tree" >/dev/null 2>&1 || true
}
trap cleanup EXIT  # armed only once the tree is ours

cd "$tree"
export PYTHONPATH="$tree/src"

echo "suite at $short ($(git log -1 --format=%s "$commit"))"
status=0
if [ "$cov" -eq 1 ]; then
    # The ratchet refuses a coverage.json from any other coverage version: say so now, not after the suite.
    python - <<'PY' || exit 2
import importlib.util, sys
spec = importlib.util.spec_from_file_location("lc", "scripts/lint_coverage.py")
lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)
import coverage
if coverage.__version__ != lc.COVERAGE_VERSION:
    sys.exit(f"coverage {coverage.__version__} installed; the ratchet needs {lc.COVERAGE_VERSION} (pip install 'coverage=={lc.COVERAGE_VERSION}' 'pytest-cov==7.0.0'), or pass --no-cov")
PY
    # CI's measuring step (lint_coverage.py: CI_CLEAN + CI_COV_ARGS), so the numbers are the gate's own.
    rm -f .coverage coverage.json coverage.xml
    python -m pytest tests/ -q -m "not slow" --ignore=tests/integration/test_memory_hub.py -p no:cacheprovider \
        --cov=src/maxim --cov-branch --cov-report=json:coverage.json --cov-report= || status=$?
else
    python -m pytest tests/ -q -m "not slow" --ignore=tests/integration/test_memory_hub.py -p no:cacheprovider || status=$?
fi
python -m pytest tests/integration/test_memory_hub.py -q -p no:cacheprovider || status=$?
if [ "$cov" -eq 1 ]; then
    python scripts/lint_coverage.py --local || status=$?
fi
echo "suite at $short: $([ "$status" -eq 0 ] && echo PASSED || echo "FAILED (exit $status)")"
exit "$status"
