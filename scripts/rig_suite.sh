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
# records its pid beside it; this side polls the log, so a dropped SSH link loses nothing. The log's first line is
# `rig-suite-run=<nonce>`, written before the run starts; the run's argv carries `rig-suite-<nonce>` too. A launch
# accepts a verdict only from a log that starts with its own nonce, and a pid counts as the run only while its argv
# carries that nonce (a reused pid is not the run).
#
# No decision reads ssh's own exit status: the rig's Tailscale SSH reports 0 for every remote command (found 2026-11-03).
# rig() reads each command's status from a sentinel line the remote prints after it, and a missing sentinel is a
# failure (255). RIG_SUITE_POLL_S (default 60) is the poll interval. Residual: two machines launching one label at
# the same moment can both pass the running-label check (only one machine launches today; follow-up issue owed).
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
        poll "$2" "" "" ""
    fi
    local wt="${1:?usage: rig_suite.sh <local-worktree> <label>}" label="${2:-}"
    check_label "$label"
    local sha main_sha
    sha="$(git -C "$wt" rev-parse --verify HEAD^{commit})"
    git -C "$wt" fetch -q origin main
    main_sha="$(git -C "$wt" rev-parse --verify origin/main^{commit})"

    # A label still running is refused: a second launch would truncate its log and both pollers would read one mix.
    # The remote prints `running` or `idle`; anything else (an unreachable rig, a lost sentinel) refuses too.
    local guard
    guard="$(rig "$(liveness_cmd "$label"); case \$s in alive | oldrun) echo \$s ;; *) echo idle ;; esac")" || guard="(rig exit $?) $guard"
    case "$guard" in
        idle) ;;
        alive) echo "rig suite: a run labelled $label is still going (attach with --attach $label, or pick a new label)" >&2; exit 2 ;;
        oldrun) echo "rig suite: $label holds an older runner's unfinished log (no run tag, not attachable); pick a new label" >&2; exit 2 ;;
        *) echo "rig suite: could not tell whether a run labelled $label is still going: $guard" >&2; exit 2 ;;
    esac
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
    local nonce launched rc=0
    nonce="${sha:0:12}-$(date +%s)-$$-$RANDOM$RANDOM"
    echo "rig suite: $label at ${sha:0:12} (log on the rig: ~/$dir/logs/$label.log, run rig-suite-run=$nonce)"
    # The nonce line is written synchronously, before the run starts and before this call returns: the run appends.
    launched="$(rig "
      set -e
      mkdir -p ~/$dir/logs
      cd ~/$dir/repo
      git merge-base --is-ancestor $main_sha refs/remotes/origin/main \
        || { echo 'rig suite: the rig origin/main does not contain this worktree origin/main; refusing' >&2; exit 2; }
      git checkout -q --detach refs/remotes/origin/main  # the runner script; it tests $sha in its own worktree
      export PATH=\$HOME/$dir/venv/bin:\$PATH
      rm -f ~/$dir/logs/$label.pid
      printf 'rig-suite-run=%s\\n' '$nonce' > ~/$dir/logs/$label.log
      nohup bash -c 'scripts/suite_at_commit.sh $sha; echo suite-exit=\$?' 'rig-suite-$nonce' >> ~/$dir/logs/$label.log 2>&1 < /dev/null &
      echo \$! > ~/$dir/logs/$label.pid
      head -n 1 ~/$dir/logs/$label.log
    ")" || rc=$?
    if [ "$rc" -eq 255 ]; then  # no reply: the run may have started before the link dropped
        echo "rig suite: the launch reply was lost; the run may be going: check with --attach $label" >&2
        exit 2
    fi
    if [ "$rc" -ne 0 ] || [ "${launched##*$'\n'}" != "rig-suite-run=$nonce" ]; then
        echo "rig suite: the launch on the rig failed (exit $rc); nothing of $label is this run's" >&2
        exit 2
    fi
    poll "$label" "$wt" "$sha" "$nonce"
}

liveness_cmd() {  # liveness_cmd <label>: remote shell text that sets s to nolog, nopid, notag, alive or dead
    # Sets h (the log's first line) and s. alive: the pid file's process carries this log's nonce in its argv (a
    # reused pid does not). A log with no nonce (an older runner's, which may write no pid file) is oldrun until it
    # ends with its exit line, so it is never launched over; notag once it has.
    local log="~/$dir/logs/$1.log" pidf="~/$dir/logs/$1.pid"
    printf '%s' 'h=$(head -n 1 '"$log"' 2>/dev/null || true); n=${h#rig-suite-run=}; p=$(cat '"$pidf"' 2>/dev/null || true)
if [ ! -f '"$log"' ]; then s=nolog
elif [ "$n" = "$h" ] || [ -z "$n" ]; then
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then s=alive
  elif tail -n 1 '"$log"' | grep -q "^suite-exit=[0-9]"; then s=notag; else s=oldrun; fi
elif [ -z "$p" ]; then s=nopid
elif ps -p "$p" -o args= 2>/dev/null | grep -qF -- "rig-suite-$n"; then s=alive; else s=dead; fi
'
}

check_label() {  # labels reach remote shells and file names
    case "$1" in "" | *[!A-Za-z0-9._-]*) echo "rig suite: label must be non-empty [A-Za-z0-9._-]" >&2; exit 2 ;; esac
}

rig() {  # rig <remote command>: run it on the rig, print its stdout, return ITS status (never ssh's)
    [ "$#" -eq 1 ] || { echo "rig suite: rig() takes one command string" >&2; exit 2; }
    local tag out rc
    tag="__rig_rc_$RANDOM$RANDOM$RANDOM"
    # A subshell, so the command's own `exit` or `set -e` ends only it and the sentinel still prints.
    out="$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "$host" \
        "( $1
) ; printf '\\n$tag=%d\\n' \"\$?\"")" || :
    case "$out" in
        *"
$tag="*) ;;
        *) return 255 ;;  # no sentinel: the link dropped or the command was killed; nothing it printed counts
    esac
    rc="${out##*
$tag=}"
    case "$rc" in "" | *[!0-9]*) return 255 ;; esac  # anything after the sentinel, or a mangled one
    printf '%s' "${out%
$tag=*}"
    return "$rc"
}

poll() {  # poll <label> <worktree|""> <sha|""> <nonce|"">: wait for `suite-exit=N`, then judge ONE read of the log
    local label="$1" wt="$2" sha="$3" nonce="${4:-}" log="~/$dir/logs/$1.log"
    local poll_s="${RIG_SUITE_POLL_S:-60}" fails=0 started="$SECONDS" dead=0 out state head last
    case "$poll_s" in "" | 0 | *[!0-9]*) echo "rig suite: RIG_SUITE_POLL_S must be a whole number of seconds >= 1" >&2; exit 2 ;; esac
    while :; do
        # One remote read per poll: the run's state, the log's first line (its nonce) and its LAST line. The runner
        # writes `suite-exit=N` after the suite, so a match anywhere else could be a test's own output.
        if out="$(rig "$(liveness_cmd "$label"); printf '%s\n%s\n%s\n' \"\$s\" \"\$h\" \"\$(tail -n 1 $log 2>/dev/null)\"")"; then
            fails=0
            state="$(printf '%s\n' "$out" | sed -n 1p)"
            head="$(printf '%s\n' "$out" | sed -n 2p)"
            last="$(printf '%s\n' "$out" | sed -n 3p)"
            [ "$state" != nolog ] || { echo "rig suite: no run labelled $label on the rig" >&2; exit 2; }
            if [ -z "$nonce" ]; then  # --attach: the run is whichever one this log is
                case "$head" in
                    rig-suite-run=?*) nonce="${head#rig-suite-run=}"; echo "rig suite: attached to $label, run rig-suite-run=$nonce" ;;
                    *) echo "rig suite: the log of $label carries no run tag (a run from before run tags?); not attachable" >&2; exit 2 ;;
                esac
            fi
            if [ "$head" != "rig-suite-run=$nonce" ]; then
                echo "rig suite: the log of $label on the rig is not this run's (its first line: ${head:-none}); no verdict" >&2
                exit 2
            fi
            # The exit line counts once the run is gone: a test's own output could print one mid-run.
            case "$last" in suite-exit=[0-9]*) [ "$state" = alive ] || break ;; esac
            if [ "$state" != alive ]; then  # a reboot or an OOM kill never writes the exit line
                [ "$dead" -eq 0 ] || { echo "rig suite: the run labelled $label died without finishing (log: $log on the rig)" >&2; exit 2; }
                dead=1  # read once more: it may have ended between this read's lines
                continue
            fi
            dead=0
        else
            fails=$((fails + 1))
            [ "$fails" -ge 30 ] && { echo "rig suite: the rig is unreachable (30 attempts); re-attach with --attach $label" >&2; exit 2; }
        fi
        echo "rig suite: $label running ($(( (SECONDS - started) / 60 )) min)"
        sleep "$poll_s"
    done
    # The verdict reads ONE copy of the log: its first line must still be this run's nonce, and the exit is its last.
    local text rc=0 first done_line
    text="$(rig "cat $log")" || rc=$?
    [ "$rc" -eq 0 ] || { echo "rig suite: reading the log of $label failed (exit $rc); no verdict" >&2; exit 2; }
    first="${text%%$'\n'*}"
    done_line="${text##*$'\n'}"
    [ "$first" = "rig-suite-run=$nonce" ] || { echo "rig suite: the log of $label changed under this poll; no verdict" >&2; exit 2; }
    case "$done_line" in suite-exit=[0-9]*) ;; *) echo "rig suite: the log of $label does not end with suite-exit; no verdict" >&2; exit 2 ;; esac
    local_log="$(mktemp)"
    trap "rm -f $(printf %q "$local_log")" EXIT
    printf '%s\n' "$text" | tee "$local_log"
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
