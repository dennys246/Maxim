#!/usr/bin/env python3
"""PKG seam — prove the lean Pi extra resolves on aarch64 with no heavy backends.

Cross-resolves an extras combination for **linux aarch64 / CPython 3.11**
(Raspberry Pi OS bookworm = glibc 2.36) and asserts no heavy backend leaked in
(torch / CUDA / llama-cpp / tensorflow / triton) — the ``PKG`` regression guard.

Runs in seconds on an ordinary x86 runner, so it belongs on every PR. It is the
*resolution* half of the guard; the real-install half (which is the only thing
that can catch a missing apt build-dep) is the ``aarch64-install`` CI job.

WHY uv AND NOT ``pip --platform`` — this is load-bearing, not a preference.
``pip --platform`` does **not** evaluate environment markers against the target;
it uses the RUNNING interpreter for ``sys_platform``, ``platform_machine`` and
even ``python_version``. Verified empirically both ways:

* resolving an x86_64 target from an arm64 host, pip reported **zero**
  ``nvidia-*`` packages — torch gates CUDA on ``platform_machine == 'x86_64'``,
  so pip cheerfully certifies "no CUDA" for a target where CUDA absolutely
  installs. That is exactly the assertion this script exists to make, so pip's
  answer cannot be trusted for it (pypa/pip#6117).
* on macOS, ``gstreamer-bundle; sys_platform != "linux"`` resolved as if a Pi
  needed it, failing on a dependency the target never installs.

``uv pip compile --python-platform`` performs true cross-platform marker
resolution: same experiment yields 13 ``nvidia-*`` for x86_64 and 0 for
aarch64. So uv is required; there is no sound pip fallback for this check.

A second, separate trap (kept documented because it bites the pip path too):
``--platform`` is **exact-match**, not "compatible with" — requesting only
``manylinux2014_aarch64`` rejects a wheel tagged solely ``manylinux_2_28``
and reports an installable package as missing.

WHAT THIS CANNOT PROVE: sdist-only dependencies satisfied by SYSTEM packages.
``reachy-mini`` needs PyGObject and pycairo, which publish **no wheels on any
platform**; a Pi compiles them against apt-installed headers. They are declared
in ``_SOURCE_OK`` and allowed to build, and the required apt line is printed —
a dry resolve can never catch a missing system package, which is precisely why
the real-install job exists.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Pi 5 / Raspberry Pi OS bookworm: aarch64, glibc 2.36, CPython 3.11.
DEFAULT_PLATFORM = "aarch64-manylinux_2_36"
DEFAULT_PYTHON = "3.11"

#: Anchored PEP 503-normalized patterns. Anchored on purpose: a bare "torch"
#: substring would also flag torchvision, and we want the failure to name the
#: real offender.
_FORBIDDEN = (
    r"^torch$",
    r"^nvidia-",
    r"^llama-cpp-python$",
    r"^triton$",
    r"^tensorflow",
)
_RX = tuple(re.compile(p) for p in _FORBIDDEN)

#: No wheels on ANY platform; the target builds them against system headers.
_SOURCE_OK = {
    "pygobject": "libgirepository1.0-dev libcairo2-dev pkg-config python3-dev gcc",
    "pycairo": "libcairo2-dev pkg-config",
}


def normalize(name: str) -> str:
    """PEP 503 normalization — nvidia_cublas_cu12 and nvidia-cublas-cu12 are one."""
    return re.sub(r"[-_.]+", "-", name).lower()


def resolve(extra: str, platform: str, python_version: str) -> tuple[list[str], str]:
    """Cross-resolve ``pymaxim[extra]``; return (package names, raw output)."""
    if shutil.which("uv") is None:
        raise SystemExit(
            "uv is required: pip --platform evaluates markers against the HOST, so its\n"
            "absence assertion is unsound (see this file's docstring / pypa/pip#6117).\n"
            "Install: pipx install uv   — or in CI: astral-sh/setup-uv@v5"
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "resolved.txt"
        cmd = [
            "uv",
            "pip",
            "compile",
            str(REPO_ROOT / "pyproject.toml"),
            "--extra",
            extra,
            "--python-platform",
            platform,
            "--python-version",
            python_version,
            "--only-binary",
            ":all:",
            "-o",
            str(out),
        ]
        for pkg in _SOURCE_OK:
            cmd += ["--no-binary", pkg]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            for name, apt in _SOURCE_OK.items():
                if name in stderr.lower():
                    stderr += f"\n\nNOTE: {name} ships no wheels anywhere; a Pi builds it. Needs: apt install {apt}"
            raise SystemExit(f"RESOLUTION FAILED for {platform} / cp{python_version}:\n\n{stderr}")
        text = out.read_text()

    names: list[str] = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            if m := re.match(r"^([A-Za-z0-9._-]+)", line):
                names.append(m.group(1))
    return names, text


def assert_no_heavy(names: list[str]) -> list[str]:
    """Return the forbidden packages present (empty == clean)."""
    return sorted({n for n in names if any(rx.match(normalize(n)) for rx in _RX)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", default="pi", help="Extras name to check (default: pi)")
    ap.add_argument("--platform", default=DEFAULT_PLATFORM, help=f"uv --python-platform (default: {DEFAULT_PLATFORM})")
    ap.add_argument("--python-version", default=DEFAULT_PYTHON, help=f"Target CPython (default: {DEFAULT_PYTHON})")
    ap.add_argument("--from-json", help="Skip resolution; assert over a pip list --format=json / pip --report file")
    ap.add_argument("--print-resolved", action="store_true", help="Print the full resolved set")
    args = ap.parse_args(argv)

    if args.from_json:
        data = json.loads(Path(args.from_json).read_text())
        names = (
            [i["name"] for i in data]
            if isinstance(data, list)
            else [i["metadata"]["name"] for i in data.get("install", [])]
        )
        source = args.from_json
    else:
        names, text = resolve(args.extra, args.platform, args.python_version)
        source = f"pymaxim[{args.extra}] @ {args.platform} / cp{args.python_version}"
        if args.print_resolved:
            print(text)

    if not names:
        print(f"::error::no packages parsed from {source}", file=sys.stderr)
        return 1

    print(f"resolved {len(names)} packages for {source}")
    leaked = assert_no_heavy(names)
    for bad in leaked:
        print(f"::error::forbidden heavy backend in the lean aarch64 install: {bad}", file=sys.stderr)
    if leaked:
        print(
            "\nThe Pi extra must stay torch/CUDA/llama-cpp free — the encoder belongs on the\n"
            "LEADER (FIT: torch is a ~450MB runtime floor on a Pi).",
            file=sys.stderr,
        )
        return 1
    print("PASS — no torch / nvidia-* / llama-cpp-python / triton / tensorflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
