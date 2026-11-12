#!/usr/bin/env bash
# Run a test command inside a loopback-only Linux network namespace (roadmap 1.3.2 item 7, #940; owner decision
# 2026-10-04: no OS-level network exception). The in-process guard (tests/network_guard.py) cannot see a SUBPROCESS
# a test spawns; the kernel can.
#
#   bash scripts/ci_netns.sh [--keep-docker] '<commands, using "$PY" for the interpreter>'
#
# Everything is captured HERE, in the runner's shell: sudo's secure_path would replace PATH (losing setup-python's
# interpreter) and strip LD_LIBRARY_PATH, so both go in explicitly, with the interpreter as $PY. Root only brings `lo`
# up (and masks sockets, below); setpriv then drops back to the caller's uid/gid with the environment intact.
# MAXIM_EXPECT_NETNS=1 arms the positive control, tests/unit/test_network_boundary.py. Linux CI only (needs
# passwordless sudo, unshare, setpriv, ip, mount, find).
#
# The caller's environment (the job's `env:`, e.g. MAXIM_TEST_TTS_VOICE_DIR, HF_HOME, MAXIM_RUN_MODEL_TESTS) crosses
# sudo by NAME: `--preserve-env=<the exported names>`, never `-E` and never NAME=VALUE arguments. Ubuntu 26's sudo-rs
# ignores `-E` ("preserving the entire environment is not supported"), which silently emptied every netns lane's
# job variables from 2026-10-20 (O20); values on argv would put secrets on sudo's long-lived command line. Secret-
# shaped names stay out (the suite needs none). Inside, before the suite, every preserved name must be set, or
# the run fails (exit 3) naming the ones that were lost: a sudo that drops the caller's environment fails every
# lane loudly instead of one lane by symptom. Not carried: exported bash functions and non-identifier names, which
# `compgen -e` does not list (nothing here uses them). Classic sudo's env_check drops a TZ/TERM/LANG/LC_* value that
# looks unsafe (one containing `/` or `%` or `..`, e.g. TZ=/etc/passwd); exit 3 is then the correct answer.
#
# Nothing inside regains privilege or reaches a daemon outside the namespace (#1426, owner decision 2026-11-10; the
# runner facts were measured on ubuntu-latest, draft PR #1637):
#   - setpriv runs with --no-new-privs, so a setuid binary (sudo, sudo-rs, su) cannot regain root: the runner has
#     passwordless sudo, and `sudo nsenter` would otherwise leave the namespace itself;
#   - the docker group is dropped from the supplementary groups (the socket is 0660 root:docker, no ACL), unless the
#     lane passes --keep-docker (only slow-tests, whose real-Docker tests need it; its containers have the host's
#     network by design);
#   - in every lane but --keep-docker, every socket under the runner's runtime dir /run/user/<uid> except
#     pulseaudio's (pulse/, the tts cell's) is covered with /dev/null in a private mount namespace: the user's
#     systemd manager, dbus, snapd and packagekit agents there run in the host's network (a keep lane leaves them,
#     as its containers have the host's network anyway);
#   - then, before the suite, scripts/ci_netns_probe.py checks it from inside (exit 4 on a breach): no interface but
#     lo, sudo -n refused, the docker and containerd sockets refused (or reachable only with --keep-docker, which
#     on CI must reach them), no unmasked runtime socket answers.
# The boundary is against accidental egress by trusted test code, not a hostile test. Not covered: rootless or
# self-hosted Docker at other paths, podman, the system dbus (reaching systemd there needs polkit auth_admin),
# pulseaudio's socket (left for the tts cell; a tunnel module could reach the network through it), and, in the
# --keep-docker lane, the session sockets (systemd-run --user stays open there, as Docker does).
#
# The phases re-enter this file: caller (the runner's shell) -> root (sudo + unshare) -> user (setpriv). The phase
# rides in CI_NETNS_PHASE; a caller that sets it itself fails closed: the root phase demands uid 0, and the user
# phase's probe exits 4 outside a fresh network namespace (any interface but lo).
set -euo pipefail

usage() {
    echo "usage: ci_netns.sh [--keep-docker] '<commands>'" >&2
    exit 2
}

case "${CI_NETNS_PHASE:-}" in
"")
    KEEP_DOCKER=0
    if [ "$#" -eq 2 ] && [ "$1" = --keep-docker ]; then
        KEEP_DOCKER=1
        shift
    elif [ "$#" -ne 1 ]; then
        usage
    fi
    [ -n "$1" ] || usage
    # The supplementary groups setpriv hands the namespace: the caller's, minus docker's gid unless --keep-docker. A
    # docker PRIMARY group cannot be dropped this way, so closed mode refuses it rather than guess another group. No
    # docker group at all means nothing to drop. An empty list is --clear-groups, never `--groups=`.
    DOCKER_GID=$(getent group docker | cut -d: -f3) || DOCKER_GID=""
    if [ "$KEEP_DOCKER" = 1 ]; then
        SG="--init-groups"
    else
        if [ -n "$DOCKER_GID" ] && [ "$(id -g)" = "$DOCKER_GID" ]; then
            echo "ci_netns: the primary group is docker ($DOCKER_GID), which the namespace cannot drop; only a lane" \
                "meant to reach Docker may run here, with --keep-docker" >&2
            exit 2
        fi
        SG_LIST=$(id -G | tr ' ' '\n' | { grep -vxF "${DOCKER_GID:-no-docker-group}" || [ $? -eq 1 ]; } | paste -sd, -)
        if [ -n "$SG_LIST" ]; then SG="--groups=$SG_LIST"; else SG="--clear-groups"; fi
    fi
    # Exported names, minus: the ones passed explicitly below (later env assignments win anyway); LD_* (sudo scrubs
    # them; LD_LIBRARY_PATH is explicit); anything token-, key-, secret- or password-shaped (the suite needs none, and
    # a secret added to a netns job later must not cross by default); names bash or sudo recreate on their own (PWD,
    # SHLVL, _, OLDPWD, SUDO_*), which would make the arrival check count variables nobody carried; and this script's
    # own phase and mode names, which a caller's export must not set. Only grep's "no match" (exit 1) is tolerated: a
    # failing compgen or paste must not become an empty list and a vacuous check.
    NETNS_ENV_NAMES=$(compgen -e |
        { grep -vxE 'PATH|LD_.*|HOME|PY|USER|LOGNAME|R|G|SG|KD|SUITE|NETNS_SELF|CI_NETNS_PHASE|MAXIM_NETNS_ENV_NAMES|MAXIM_EXPECT_NETNS|MAXIM_EXPECT_NO_DOCKER|.*TOKEN.*|.*_KEY|.*SECRET.*|.*PASSWORD.*|PWD|SHLVL|_|OLDPWD|SUDO_.*' || [ $? -eq 1 ]; } |
        paste -sd, -)
    SELF=$(readlink -f "$0")
    exec sudo --preserve-env="$NETNS_ENV_NAMES" env "PATH=$PATH" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "HOME=$HOME" \
        "PY=$(command -v python)" "USER=$(id -un)" "LOGNAME=$(id -un)" \
        "R=$(id -u)" "G=$(id -g)" "SG=$SG" "KD=$KEEP_DOCKER" "SUITE=$1" "MAXIM_NETNS_ENV_NAMES=$NETNS_ENV_NAMES" \
        "NETNS_SELF=$SELF" "CI_NETNS_PHASE=root" \
        unshare --net --mount -- bash "$SELF"
    ;;
root)
    [ "$(id -u)" = 0 ] || {
        echo "ci_netns: the root phase runs only under sudo" >&2
        exit 2
    }
    ip link set lo up
    if [ "$KD" = 0 ] && [ -d "/run/user/$R" ]; then
        # unshare --mount makes this mount namespace private, so these masks never reach the host. The list goes
        # through a file so find's exit status is checked (a process substitution's is not, under set -e): a subtree
        # it cannot read would otherwise leave sockets unmasked in silence. -xdev keeps it out of a user's FUSE mount.
        # Only the sockets that exist now are covered; the probe, right after, checks the tree again.
        SOCKS=$(mktemp)
        if ! find "/run/user/$R" -xdev -path "/run/user/$R/pulse" -prune -o -type s -print0 > "$SOCKS"; then
            echo "ci_netns: cannot list /run/user/$R to cover its sockets" >&2
            rm -f "$SOCKS"
            exit 4
        fi
        while IFS= read -r -d '' sock; do
            mount --bind /dev/null "$sock"
        done < "$SOCKS"
        rm -f "$SOCKS"
    fi
    if [ "$KD" = 0 ]; then
        UNSET=()
        SETS=(MAXIM_EXPECT_NO_DOCKER=1)
    else
        UNSET=(-u MAXIM_EXPECT_NO_DOCKER)
        SETS=()
    fi
    # $SG is one token (--groups=<gids> or --clear-groups or --init-groups).
    exec setpriv --reuid="$R" --regid="$G" $SG --no-new-privs \
        env "${UNSET[@]}" CI_NETNS_PHASE=user MAXIM_EXPECT_NETNS=1 "${SETS[@]}" bash "$NETNS_SELF"
    ;;
user)
    lost=""
    IFS=, read -ra names <<< "$MAXIM_NETNS_ENV_NAMES"
    for n in "${names[@]}"; do [ -n "${!n+x}" ] || lost="$lost $n"; done
    if [ -n "$lost" ]; then
        echo "ci_netns: the caller's environment did not reach the namespace; lost:$lost" >&2
        exit 3
    fi
    echo "netns environment: ${#names[@]} caller variable(s) preserved"
    "$PY" -c "import ssl, sqlite3, sys; print('netns interpreter:', sys.executable)"
    if [ "$KD" = 1 ]; then MODE=keep; else MODE=closed; fi
    "$PY" "$(dirname "$NETNS_SELF")/ci_netns_probe.py" --mode "$MODE" || exit 4
    unset CI_NETNS_PHASE
    # `--` ends bash's options, so a suite that starts with "-" is a command, never an option.
    exec bash -eo pipefail -c -- "$SUITE"
    ;;
*)
    echo "ci_netns: unknown CI_NETNS_PHASE=${CI_NETNS_PHASE}" >&2
    exit 2
    ;;
esac
