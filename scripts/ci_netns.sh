#!/usr/bin/env bash
# Run a test command inside a loopback-only Linux network namespace (roadmap 1.3.2 item 7, #940; owner decision
# 2026-10-04: no OS-level network exception). The in-process guard (tests/network_guard.py) cannot see a SUBPROCESS
# a test spawns; the kernel can.
#
#   bash scripts/ci_netns.sh '<commands, using "$PY" for the interpreter>'
#
# Everything is captured HERE, in the runner's shell: sudo's secure_path would replace PATH (losing setup-python's
# interpreter) and strip LD_LIBRARY_PATH, so both go in explicitly, with the interpreter as $PY. Root only brings `lo`
# up; setpriv then drops back to the caller's uid/gid with the environment intact. MAXIM_EXPECT_NETNS=1 arms the
# positive control, tests/unit/test_network_boundary.py. Linux CI only (needs passwordless sudo, unshare, setpriv, ip).
set -euo pipefail
if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo "usage: ci_netns.sh '<commands>'" >&2
    exit 2
fi
sudo -E env "PATH=$PATH" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "HOME=$HOME" "PY=$(command -v python)" \
    "USER=$(id -un)" "LOGNAME=$(id -un)" \
    "R=$(id -u)" "G=$(id -g)" "SUITE=$1" \
    unshare --net -- sh -c 'ip link set lo up && exec setpriv --reuid="$R" --regid="$G" --init-groups \
        env MAXIM_EXPECT_NETNS=1 bash -eo pipefail -c "\"\$PY\" -c \"import ssl, sqlite3, sys; print(\\\"netns interpreter:\\\", sys.executable)\"
$SUITE"'
