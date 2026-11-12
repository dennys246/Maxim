#!/usr/bin/env python3
"""The in-namespace check scripts/ci_netns.sh runs before the suite, in every lane (#1426; owner decision 2026-11-10).

Stdlib only (it runs before anything is installed into the namespace's view). Exit 0 when the namespace holds, 4 with
one line per breach otherwise:

  - no network interface but ``lo`` (also catches a run that never entered a namespace);
  - ``sudo -n true`` is refused (setpriv's no_new_privs; a missing sudo counts as refused);
  - the containerd socket is refused (root-only on hosted runners);
  - closed mode (every lane but slow-tests): each docker socket path is refused (PermissionError) or absent, and on CI
    (``GITHUB_ACTIONS=true``) at least one exists, or the check is vacuous; and no socket left under the runtime dir
    (outside ``pulse/``) answers a connect, since ci_netns.sh covered each with /dev/null;
  - keep mode (``--keep-docker``): on CI a docker socket must ANSWER, which proves this probe works against a live
    daemon; off CI an absent socket is a printed skip.

A connect only opens and closes the socket: nothing is sent. Every function takes its paths as arguments, so the rules
are tested on temporary sockets (tests/unit/test_ci_netns_probe.py); ``main`` uses the real paths.
"""

from __future__ import annotations

import argparse
import os
import socket
import stat
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

DOCKER_PATHS = ("/var/run/docker.sock", "/run/docker.sock")
CONTAINERD_PATH = "/run/containerd/containerd.sock"


def connect_result(path: str, timeout_s: float = 2.0) -> str:
    """``"answered"``, ``"refused"`` (not a socket, or nothing listening), ``"denied"``, ``"absent"`` or
    ``"error:<errno>"``."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    try:
        s.connect(path)
        return "answered"
    except FileNotFoundError:
        return "absent"
    except PermissionError:
        return "denied"
    except ConnectionRefusedError:
        return "refused"
    except OSError as e:
        return f"error:{e.errno}"
    finally:
        s.close()


def docker_paths(env: dict[str, str]) -> list[str]:
    paths = list(DOCKER_PATHS)
    host = env.get("DOCKER_HOST", "")
    if host.startswith("unix://") and host[len("unix://") :] not in paths:
        paths.append(host[len("unix://") :])
    return paths


def runtime_sockets(runtime_dir: Path, errors: list[str] | None = None) -> list[str]:
    """Every socket still a socket under ``runtime_dir``, outside its ``pulse/`` (a covered one is now /dev/null).

    A directory this process can SEARCH but not list is appended to ``errors``: a socket in it is reachable by name,
    so an unscanned subtree there is not a clean one. One it cannot search at all (systemd's mode-000
    ``systemd/inaccessible/dir``) hides nothing: connecting to a socket needs search permission on every directory on
    its path."""
    found: list[str] = []
    if not runtime_dir.is_dir():
        return found

    def _unreadable(e: OSError) -> None:
        if errors is not None and e.filename and os.access(e.filename, os.X_OK):
            errors.append(f"{e.filename}: cannot scan ({e.strerror})")

    for root, dirs, files in os.walk(runtime_dir, onerror=_unreadable):
        if Path(root) == runtime_dir and "pulse" in dirs:
            dirs.remove("pulse")
        for name in files + dirs:
            p = Path(root) / name
            try:
                if stat.S_ISSOCK(p.lstat().st_mode):
                    found.append(str(p))
            except OSError:
                continue
    return found


def interfaces() -> list[str]:
    """The interfaces of THIS process's network namespace, asked of the kernel (``if_nameindex``). Not
    ``/sys/class/net``: sysfs shows the namespace that mounted it, so inside ``unshare --net`` without a sysfs remount
    it lists the host's interfaces. An error reads as an unknown interface, so it fails the check."""
    try:
        return sorted(name for _index, name in socket.if_nameindex())
    except OSError as e:
        return [f"<if_nameindex failed: {e.errno}>"]


def sudo_refused() -> bool:
    try:
        r = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=10, check=False)
    except FileNotFoundError:
        return True  # no sudo at all: nothing to regain root through
    except subprocess.TimeoutExpired:
        return False  # sudo -n never prompts, so a hang is not a refusal: report it
    return r.returncode != 0


def breaches(
    *,
    mode: str,
    on_ci: bool,
    docker: Iterable[str],
    containerd: str,
    runtime: Iterable[str],
    links: Iterable[str],
    sudo_is_refused: bool,
    scan_errors: Iterable[str] = (),
) -> list[str]:
    out: list[str] = list(scan_errors)
    links = list(links)
    if links != ["lo"]:
        out.append(f"network interfaces {links}: only lo may exist inside the namespace")
    if not sudo_is_refused:
        out.append("sudo -n succeeded: no_new_privs is not in force")
    c = connect_result(containerd)
    if c not in ("denied", "absent"):
        out.append(f"{containerd}: {c}")
    results = {p: connect_result(p) for p in docker}
    if mode == "closed":
        for p, r in results.items():
            if r not in ("denied", "absent"):
                out.append(f"{p}: {r} (closed mode: the docker socket must be refused)")
        if on_ci and all(r == "absent" for r in results.values()):
            out.append("probe vacuous: no docker socket exists on a CI runner, so nothing was checked")
        for p in runtime:
            r = connect_result(p)
            if r == "answered":
                out.append(f"{p}: answered (a runtime socket the namespace should have covered)")
    elif mode == "keep":
        if on_ci and not any(r == "answered" for r in results.values()):
            out.append(f"keep mode on CI: no docker socket answered ({results}): the probe cannot be trusted")
    else:
        out.append(f"unknown mode {mode!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("closed", "keep"), required=True)
    args = ap.parse_args(argv)
    env = dict(os.environ)
    on_ci = env.get("GITHUB_ACTIONS") == "true"
    # The directory ci_netns.sh masked: /run/user/<uid>, never $XDG_RUNTIME_DIR (which crosses sudo by name and
    # may point elsewhere on a self-hosted box, where the probe would then scan an unmasked tree).
    runtime_dir = Path(f"/run/user/{os.getuid()}")
    scan_errors: list[str] = []
    runtime = runtime_sockets(runtime_dir, scan_errors) if args.mode == "closed" else []
    found = breaches(
        mode=args.mode,
        on_ci=on_ci,
        docker=docker_paths(env),
        containerd=CONTAINERD_PATH,
        runtime=runtime,
        scan_errors=scan_errors,
        links=interfaces(),
        sudo_is_refused=sudo_refused(),
    )
    if found:
        for line in found:
            print(f"ci_netns probe: BREACH: {line}", file=sys.stderr)
        return 4
    if args.mode == "keep" and not on_ci:
        print("ci_netns probe: keep mode off CI: the docker side was not required to answer")
    print(f"ci_netns probe: namespace holds ({args.mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
