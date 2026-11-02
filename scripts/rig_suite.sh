#!/usr/bin/env bash
# Run scripts/suite_at_commit.sh for a local commit on a second machine (the "rig"), so a full suite does not
# compete with the two-suite cap on the development box (owner-approved 2026-10-27).
#
#   RIG_SUITE_HOST=user@host scripts/rig_suite.sh <local-worktree> <label>
#   RIG_SUITE_HOST=user@host scripts/rig_suite.sh --attach <label>     # re-attach to a run already going
#
# Setup on the rig, once: a clone at ~/$RIG_SUITE_DIR/repo (local disk, never a network share), fed only by these
# pushes, so the rig needs no GitHub credentials; and a venv at ~/$RIG_SUITE_DIR/venv with CI's exact pins
# (coverage and pytest-cov as lint_coverage.py's COVERAGE_VERSION demands, the ci-backends constraints). conftest.py
# isolates HOME/XDG/HF before import, so the rig's own ~/.maxim is never touched. The rig always runs ITS main's
# suite_at_commit.sh (a PR that changes that script is not run by its own copy there).
#
# The remote run is detached (nohup), logs to ~/$RIG_SUITE_DIR/logs/<label>.log ending with `suite-exit=N`, and
# records its pid beside it; this side polls the log, so a dropped SSH link loses nothing.
#
# Exit: 0 = PASS, either outright or CALIBRATED with the listed tests re-run locally on the same commit and passing
# (scripts/rig_suite_verdict.py: every stage passed except the fast suite, whose only failures are the known rig
# cadence tests). 10 = CALIBRATED with no local re-run (--attach knows no worktree): run the printed command on that
# commit before counting it. Otherwise the suite's own exit code. Only 0 counts as the full suite on that commit.
#
# The whole body is one function, called on the last line (as suite_at_commit.sh does): bash reads a script as it
# runs, so an edit during a run cannot shift what the tail parses.
set -euo pipefail

main() {
    local host dir here
    host="${RIG_SUITE_HOST:?set RIG_SUITE_HOST=user@host (the SSH login of the rig)}"
    dir="${RIG_SUITE_DIR:-maxim-suites}"
    # dir is relative to the rig user's home and reaches remote shells: plain path characters only.
    case "$dir" in /* | *..* | *[!A-Za-z0-9._/-]* | "") echo "rig suite: RIG_SUITE_DIR must be a relative path of [A-Za-z0-9._/-]" >&2; exit 2 ;; esac
    here="$(cd "$(dirname "$0")" && pwd)"

    if [ "${1:-}" = "--attach" ]; then
        check_label "${2:-}"
        poll "$2" "" ""
    fi
    local wt="${1:?usage: rig_suite.sh <local-worktree> <label>}" label="${2:-}"
    check_label "$label"
    local sha main_sha
    sha="$(git -C "$wt" rev-parse --verify HEAD^{commit})"
    git -C "$wt" fetch -q origin main
    main_sha="$(git -C "$wt" rev-parse --verify origin/main^{commit})"

    # A label still running is refused: a second launch would truncate its log and both pollers would read one mix.
    if rig "test -f ~/$dir/logs/$label.pid && kill -0 \$(cat ~/$dir/logs/$label.pid) 2>/dev/null"; then
        echo "rig suite: a run labelled $label is still going (attach with --attach $label, or pick a new label)" >&2
        exit 2
    fi
    # The rig's origin/main moves FORWARD only. A forced push of a stale worktree's origin/main used to move it back
    # under a run in flight, whose coverage ratchet diffs against the merge-base with origin/main (other branches'
    # changes read as its own). Only refs/remotes/origin/main is pushed: nothing on the rig reads refs/heads/main.
    local push_err
    if ! push_err="$(git -C "$wt" push -q "ssh://$host/~/$dir/repo" "$main_sha:refs/remotes/origin/main" 2>&1)"; then
        case "$push_err" in
            *non-fast-forward* | *"fetch first"* | *rejected*) echo "rig suite: the rig's origin/main is ahead of this worktree's; keeping the rig's" ;;
            *) echo "rig suite: pushing origin/main to the rig failed: $push_err" >&2; exit 2 ;;
        esac
    fi
    git -C "$wt" push -q -f "ssh://$host/~/$dir/repo" "$sha:refs/heads/suite/$label"
    echo "rig suite: $label at ${sha:0:12} (log on the rig: ~/$dir/logs/$label.log)"
    rig "
      set -e
      mkdir -p ~/$dir/logs
      cd ~/$dir/repo
      git merge-base --is-ancestor $main_sha refs/remotes/origin/main \
        || { echo 'rig suite: the rig origin/main does not contain this worktree origin/main; refusing' >&2; exit 2; }
      git checkout -q --detach refs/remotes/origin/main  # the runner script; it tests $sha in its own worktree
      export PATH=\$HOME/$dir/venv/bin:\$PATH
      nohup bash -c 'scripts/suite_at_commit.sh $sha; echo suite-exit=\$?' > ~/$dir/logs/$label.log 2>&1 < /dev/null &
      echo \$! > ~/$dir/logs/$label.pid
    "
    poll "$label" "$wt" "$sha"
}

check_label() {  # labels reach remote shells and file names
    case "$1" in "" | *[!A-Za-z0-9._-]*) echo "rig suite: label must be non-empty [A-Za-z0-9._-]" >&2; exit 2 ;; esac
}

rig() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$host" "$@"; }

poll() {  # poll <label> <worktree|""> <sha|"">: wait for `suite-exit=N`, print the log, exit with the verdict
    local label="$1" wt="$2" sha="$3" log="~/$dir/logs/$1.log" pidf="~/$dir/logs/$1.pid"
    local done_line="" fails=0 started="$SECONDS" local_log noted=""
    while :; do
        # The exit line is the log's LAST line (the runner writes it after the suite): a match anywhere else could
        # be a test's own output and would end the poll mid-run.
        if done_line="$(rig "tail -n 1 $log 2>/dev/null | grep -a '^suite-exit=[0-9]' || true")"; then
            [ -n "$done_line" ] && break
            # No exit line yet: is the run still alive? (a reboot or an OOM kill never writes one.) The remote
            # prints alive/dead/nopid and exits 0, so a nonzero rc is ssh itself (255), counted as unreachable.
            local state=""
            if state="$(rig "if [ ! -f $log ]; then echo nolog; elif [ ! -f $pidf ]; then echo nopid; elif kill -0 \$(cat $pidf) 2>/dev/null; then echo alive; else echo dead; fi")"; then
                fails=0  # reset only when both checks reached the rig
                case "$state" in
                    nolog) echo "rig suite: no run labelled $label on the rig" >&2; exit 2 ;;
                    dead)
                        rig "tail -n 1 $log | grep -aq '^suite-exit=[0-9]'" && continue  # it ended just now
                        echo "rig suite: the run labelled $label died without finishing (log: $log on the rig)" >&2
                        exit 2 ;;
                    nopid)
                        [ -n "$noted" ] || echo "rig suite: no pid file for $label (a run from before pid files?); polling the log only"
                        noted=1 ;;
                esac
            else
                fails=$((fails + 1))
            fi
        else
            fails=$((fails + 1))
            [ "$fails" -ge 30 ] && { echo "rig suite: the rig is unreachable (30 attempts); re-attach with --attach $label" >&2; exit 2; }
        fi
        echo "rig suite: $label running ($(( (SECONDS - started) / 60 )) min)"
        sleep 60
    done
    local_log="$(mktemp)"
    trap "rm -f $(printf %q "$local_log")" EXIT
    rig "cat $log" | tee "$local_log"
    local code=0
    python3 "$here/rig_suite_verdict.py" "$local_log" "${done_line#suite-exit=}" || code=$?
    [ "$code" -eq 10 ] || exit "$code"
    # CALIBRATED is not a pass: re-run exactly those tests here, on the same commit, and only their pass counts.
    local ids=() id
    while IFS= read -r id; do ids+=("$id"); done < <(python3 "$here/rig_suite_verdict.py" "$local_log" "${done_line#suite-exit=}" | sed -n 's/^  //p')
    if [ -z "$wt" ]; then
        echo "rig suite: CALIBRATED; before counting it, run these on that commit:"
        printf '  python3 -m pytest -p no:cacheprovider'; printf " '%s'" "${ids[@]}"; echo
        exit 10
    fi
    rerun "$wt" "$sha" "${ids[@]}"
}

rerun() {  # rerun <worktree> <sha> <nodeid>...: the calibrated tests, locally, in a throwaway worktree of <sha>
    local wt="$1" sha="$2" tree code=0 main_checkout
    shift 2
    # Under the main checkout's .worktrees/, as suite_at_commit.sh does (CLAUDE.md's worktree rule).
    main_checkout="$(cd "$(git -C "$wt" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)"
    mkdir -p "$main_checkout/.worktrees"
    tree="$(mktemp -d "$main_checkout/.worktrees/rig-rerun-${sha:0:12}.XXXXXX")"
    rmdir "$tree"
    git -C "$wt" worktree add --detach --quiet "$tree" "$sha"
    # Expanded now: an EXIT trap runs after this function's locals are gone.
    trap "git -C $(printf %q "$wt") worktree remove --force $(printf %q "$tree") >/dev/null 2>&1 || true; rm -f $(printf %q "${local_log:-/nonexistent}")" EXIT
    (cd "$tree" && PYTHONPATH="$tree/src" python3 -m pytest -q -p no:cacheprovider "$@") || code=$?
    if [ "$code" -eq 0 ]; then
        echo "rig suite: PASS (the rig's cadence failures re-ran locally on ${sha:0:12} and passed)"
        exit 0
    fi
    echo "rig suite: FAILED: the calibrated tests also fail locally on ${sha:0:12} (exit $code)" >&2
    exit 1
}

main "$@"; exit "$?"
