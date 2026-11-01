#!/usr/bin/env bash
# Run scripts/suite_at_commit.sh for a local commit on a second machine (the "rig"), so a full suite does not
# compete with the two-suite cap on the development box (owner-approved 2026-10-27).
#
#   RIG_SUITE_HOST=user@host scripts/rig_suite.sh <local-worktree> <label>
#   RIG_SUITE_HOST=user@host scripts/rig_suite.sh --attach <label>     # re-attach to a run already going
#
# Setup on the rig, once: a bare-enough clone at $RIG_SUITE_DIR/repo (local disk, never a network share), fed only
# by these pushes, so the rig needs no GitHub credentials; and a venv at $RIG_SUITE_DIR/venv with CI's exact pins
# (coverage and pytest-cov as lint_coverage.py's COVERAGE_VERSION demands, the ci-backends constraints). conftest.py
# isolates HOME/XDG/HF before import, so the rig's own ~/.maxim is never touched.
#
# The remote run is detached (nohup) and logs to $RIG_SUITE_DIR/logs/<label>.log, ending with `suite-exit=N`; this
# side polls that log, so a dropped SSH link loses nothing. If this script dies, re-attach with --attach <label>.
#
# Exit: the suite's own exit code, except 3 = CALIBRATED: the suite failed, but only on the rig's known cadence tests
# (RIG_KNOWN_FAILURES below) and the coverage ratchet was clean. 3 is not a pass; read the log and decide, the way
# prepr.sh's TAINTED-PASS works.
set -euo pipefail

host="${RIG_SUITE_HOST:?set RIG_SUITE_HOST=user@host (the SSH login of the rig)}"
dir="${RIG_SUITE_DIR:-maxim-suites}"  # relative to the rig user's home

# Tests that fail on the rig even alone and pass on the development box and in CI: real-time cadence assertions
# under macOS timer coalescing over SSH (calibrated 2026-10-27; the water-trial pair is also #954's wall-time class).
RIG_KNOWN_FAILURES=(
    "tests/unit/test_b1_floor_run.py::test_run_arm_seed_end_to_end_writes_a_serialisable_row_and_tears_down[q_explore_prior]"
    "tests/unit/test_b1_floor_run.py::test_run_arm_seed_end_to_end_writes_a_serialisable_row_and_tears_down[random]"
    "tests/unit/test_leader_proxy.py::TestKeepaliveEmitter::test_emits_at_configured_cadence"
    "tests/unit/test_water_trial_smoke.py::test_water_trial_ticks_acts_and_the_staging_close_persists_fear"
    "tests/unit/test_water_trial_smoke.py::test_donor_sequence_over_the_scripted_bridge_passes_the_staged_sanity"
)

rig() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$host" "$@"; }

verdict() {  # verdict <log-file> <suite-exit>: the exit code this script returns
    local log="$1" code="$2" failed unknown
    [ "$code" -eq 0 ] && return 0
    failed="$(grep -aoE '^FAILED [^ ]+' "$log" | cut -d' ' -f2 | sort -u || true)"
    unknown="$(printf '%s\n' "$failed" | grep -v '^$' | grep -vxF -f <(printf '%s\n' "${RIG_KNOWN_FAILURES[@]}") || true)"
    if [ -n "$failed" ] && [ -z "$unknown" ] && grep -aq '^coverage ratchet: clean' "$log" \
        && ! grep -aqE '^ERROR |^E +(ImportError|ModuleNotFoundError)' "$log"; then
        echo "rig suite: CALIBRATED (exit 3): every failure is a known rig cadence test, and the ratchet is clean:"
        printf '  %s\n' $failed
        return 3
    fi
    [ -n "$unknown" ] && printf 'rig suite: failures outside the calibrated list:\n%s\n' "$unknown"
    return "$code"
}

poll() {  # poll <label>: wait for the detached run's `suite-exit=N`, print its log, exit with the verdict
    local remote_log="$dir/logs/$1.log" done_line="" local_log
    while :; do
        sleep 60
        done_line="$(rig "grep -a -m1 '^suite-exit=[0-9]' $remote_log" 2>/dev/null || true)"
        if [ -n "$done_line" ]; then
            local_log="$(mktemp)"
            rig "cat $remote_log" | tee "$local_log"
            local code=0
            verdict "$local_log" "${done_line#suite-exit=}" || code=$?
            rm -f "$local_log"
            exit "$code"
        fi
    done
}

if [ "${1:-}" = "--attach" ]; then poll "${2:?usage: rig_suite.sh --attach <label>}"; fi
wt="${1:?usage: rig_suite.sh <local-worktree> <label>}"
label="${2:?usage: rig_suite.sh <local-worktree> <label>}"
case "$label" in *[!A-Za-z0-9._-]*) echo "rig suite: label must be [A-Za-z0-9._-]" >&2; exit 2 ;; esac
remote="ssh://$host/~/$dir/repo"
sha="$(git -C "$wt" rev-parse HEAD)"
git -C "$wt" fetch -q origin main
main_sha="$(git -C "$wt" rev-parse origin/main)"

# The two main refs move FORWARD only. A forced push of a stale worktree's origin/main used to move them backwards
# under another run in flight; that run's coverage ratchet diffs against the merge-base with origin/main, so its
# base slid back and other branches' changes read as its own. A non-forced push fails when the rig is already ahead:
# keep the rig's main then, provided it contains this commit's base (checked on the rig below).
git -C "$wt" push -q "$remote" "$main_sha:refs/heads/main" "$main_sha:refs/remotes/origin/main" 2>/dev/null \
    || echo "rig suite: the rig's main is ahead of this worktree's origin/main; keeping the rig's"
git -C "$wt" push -q -f "$remote" "HEAD:refs/heads/suite/$label"
echo "rig suite: $label at ${sha:0:12} (log on the rig: ~/$dir/logs/$label.log)"
rig "
  set -e
  mkdir -p ~/$dir/logs
  cd ~/$dir/repo
  git merge-base --is-ancestor $main_sha refs/remotes/origin/main \
    || { echo 'rig suite: the rig main does not contain this worktree origin/main; refusing' >&2; exit 2; }
  git checkout -q --detach refs/remotes/origin/main  # the runner script itself; it tests $sha in its own worktree
  export PATH=\$HOME/$dir/venv/bin:\$PATH
  nohup bash -c 'scripts/suite_at_commit.sh $sha; echo suite-exit=\$?' > ~/$dir/logs/$label.log 2>&1 < /dev/null &
"
poll "$label"
