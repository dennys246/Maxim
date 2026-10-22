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
set -euo pipefail
if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo "usage: ci_netns.sh '<commands>'" >&2
    exit 2
fi
# Exported names, minus: the ones passed explicitly below (later env assignments win anyway); LD_* (sudo scrubs them;
# LD_LIBRARY_PATH is explicit); anything token-, key-, secret- or password-shaped (the suite needs none, and a
# secret added to a netns job later must not cross by default); and names bash or sudo recreate on their own (PWD, SHLVL, _,
# OLDPWD, SUDO_*), which would make the arrival check count variables nobody carried. Only grep's "no match" (exit 1)
# is tolerated: a failing compgen or paste must not become an empty list and a vacuous check.
NETNS_ENV_NAMES=$(compgen -e |
    { grep -vxE 'PATH|LD_.*|HOME|PY|USER|LOGNAME|R|G|SUITE|MAXIM_NETNS_ENV_NAMES|.*TOKEN.*|.*_KEY|.*SECRET.*|.*PASSWORD.*|PWD|SHLVL|_|OLDPWD|SUDO_.*' || [ $? -eq 1 ]; } |
    paste -sd, -)
sudo --preserve-env="$NETNS_ENV_NAMES" env "PATH=$PATH" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "HOME=$HOME" \
    "PY=$(command -v python)" "USER=$(id -un)" "LOGNAME=$(id -un)" \
    "R=$(id -u)" "G=$(id -g)" "SUITE=$1" "MAXIM_NETNS_ENV_NAMES=$NETNS_ENV_NAMES" \
    unshare --net -- sh -c 'ip link set lo up && exec setpriv --reuid="$R" --regid="$G" --init-groups \
        env MAXIM_EXPECT_NETNS=1 bash -eo pipefail -c "lost=\"\"; IFS=, read -ra names <<< \"\$MAXIM_NETNS_ENV_NAMES\"
for n in \"\${names[@]}\"; do [ -n \"\${!n+x}\" ] || lost=\"\$lost \$n\"; done
if [ -n \"\$lost\" ]; then echo \"ci_netns: the caller'"'"'s environment did not reach the namespace; lost:\$lost\" >&2; exit 3; fi
echo \"netns environment: \${#names[@]} caller variable(s) preserved\"
\"\$PY\" -c \"import ssl, sqlite3, sys; print(\\\"netns interpreter:\\\", sys.executable)\"
$SUITE"'
