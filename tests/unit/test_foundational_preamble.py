"""D32: the foundational preamble reaches pip users (roadmap 1.3.1).

The loader walked up from its own file looking for a repo-root CONSTITUTION.md -- present in a
checkout, absent in every installed wheel -- so every pip user's agent ran with an EMPTY preamble.
It now gates on the constitution shipped as package data (maxim/_data/CONSTITUTION.md)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_the_packaged_constitution_is_the_repo_constitution() -> None:
    """The repo-root file is the source; the packaged copy must not drift from it."""
    packaged = REPO / "src" / "maxim" / "_data" / "CONSTITUTION.md"
    assert packaged.read_bytes() == (REPO / "CONSTITUTION.md").read_bytes(), (
        "src/maxim/_data/CONSTITUTION.md drifted from CONSTITUTION.md -- copy the repo-root file over it"
    )


def test_an_installed_package_gets_the_preamble(tmp_path) -> None:
    """Import maxim from a copy with NO repository above it -- the layout of a wheel install."""
    site = tmp_path / "site-packages"
    shutil.copytree(REPO / "src" / "maxim", site / "maxim", ignore=shutil.ignore_patterns("__pycache__"))
    assert not any((parent / "CONSTITUTION.md").exists() for parent in site.parents), "a repo file leaked in"
    probe = (
        "import maxim, pathlib; from maxim.agents.llm_context import _load_foundational_context as f; "
        f"assert pathlib.Path(maxim.__file__).is_relative_to({str(site)!r}), maxim.__file__; "
        "print(len(f()))"
    )
    env = {"PYTHONPATH": str(site), "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-600:]
    assert int(out.stdout.strip().splitlines()[-1]) > 0, "an installed maxim has an EMPTY foundational preamble"


def test_the_wheel_audit_requires_the_constitution() -> None:
    from scripts import audit_release_build

    assert "maxim/_data/CONSTITUTION.md" in audit_release_build.REQUIRED_FILES
