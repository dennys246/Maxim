"""D32: the foundational preamble IS the Constitution's text, and reaches pip users (roadmap 1.3.1).

It was a hard-coded paraphrase gated on a repo-root CONSTITUTION.md existing -- the document and the
prompt could drift, and every installed wheel got an EMPTY preamble. The prompt now reads the marked
"Runtime Preamble" block verbatim from the Constitution shipped as package data."""

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


def test_the_prompt_preamble_is_the_constitutions_block() -> None:
    """Read from the document, not paraphrased beside it: the drift D32 names is impossible."""
    import maxim.agents.llm_context as ctx

    block = ctx.extract_runtime_preamble((REPO / "CONSTITUTION.md").read_text(encoding="utf-8"))
    assert block and "Never attempt to prevent being powered off" in block
    ctx._foundational_context_cache = None
    try:
        assert ctx._load_foundational_context() == block
    finally:
        ctx._foundational_context_cache = None


def test_every_hard_constraint_reaches_the_prompt_verbatim() -> None:
    """The block is a summary kept by hand; §1's Hard Constraints may not fall out of it. The
    actuator-speed constraint had been missing from the prompt since before D32 (review 2026-09-27)."""
    from maxim.agents.llm_context import extract_runtime_preamble

    doc = (REPO / "CONSTITUTION.md").read_text(encoding="utf-8")
    section = doc.split("### Hard Constraints (Never Violate)\n", 1)[1].split("\n\n", 1)[0]
    bullets = [line for line in section.splitlines() if line.startswith("- ")]
    assert len(bullets) >= 4, "the §1 Hard Constraints list was not found where this test reads it"
    block = extract_runtime_preamble(doc)
    missing = [b for b in bullets if b not in block.splitlines()]
    assert not missing, f"§1 hard constraints missing from the Runtime Preamble block: {missing}"


def test_a_constitution_without_the_block_yields_no_preamble() -> None:
    from maxim.agents.llm_context import extract_runtime_preamble

    assert extract_runtime_preamble("# Constitution\nno markers here\n") == ""


def test_a_lost_end_marker_yields_no_preamble_not_the_rest_of_the_document() -> None:
    from maxim.agents.llm_context import _PREAMBLE_START, extract_runtime_preamble

    doc = f"# Constitution\n{_PREAMBLE_START}\nthe block\n# Article I\nthe rest of the document\n"
    assert extract_runtime_preamble(doc) == ""


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
