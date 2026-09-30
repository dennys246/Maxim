#!/usr/bin/env bash
# Run the fast suite (and test_memory_hub.py) on a COMMIT, in a throwaway worktree, so edits in the working
# tree cannot invalidate the run and the result names the commit it tested.
# docs/lessons/development-flow-speed.md ("A fold never invalidates a running suite").
#
#   scripts/suite_at_commit.sh [<commit>]            # default: HEAD
#   scripts/suite_at_commit.sh "$(git stash create)"  # tracked changes, no commit (and no signature)
#
# No -x, deliberately: a full failure list is worth more than the first one.
set -euo pipefail

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
python -m pytest tests/ -q -m "not slow" --ignore=tests/integration/test_memory_hub.py -p no:cacheprovider || status=$?
python -m pytest tests/integration/test_memory_hub.py -q -p no:cacheprovider || status=$?
echo "suite at $short: $([ "$status" -eq 0 ] && echo PASSED || echo "FAILED (exit $status)")"
exit "$status"
