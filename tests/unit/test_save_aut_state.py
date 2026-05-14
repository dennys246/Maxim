"""Tests for simulation.report.save_aut_state.

Covers the persistence-wiring extension that adds ``ec`` and ``atl`` to
the hippocampus/NAc dump produced at session end. Without these dumps,
the Roy-5a cosine-localization analyzer (and ``analysis/substrate_diff.py``'s
``ec_diff``) have no on-disk centroid state to read.

Tests are scoped narrowly to the file-write behavior — they pass a
stub bio-system object whose ``save(path)`` method writes a sentinel
JSON. The orchestrator-side call site (passing ``aut_memory_hub.ec`` /
``.atl``) is exercised end-to-end by Roy harness runs, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

from maxim.simulation.report import save_aut_state


class _StubBio:
    """Minimal stand-in for hippocampus / NAc / EC / ATL.

    ``save(path)`` writes a sentinel JSON so tests can confirm the
    exact target path. ``__len__`` returns a small int so the log line
    in ``save_aut_state`` (`len(hippocampus)`) doesn't blow up for the
    hippocampus stub.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.saved_to: str | None = None

    def __len__(self) -> int:
        return 0

    def save(self, path: str) -> None:
        self.saved_to = path
        Path(path).write_text(json.dumps({"stub": self.label}))


def test_save_aut_state_writes_ec_and_atl(tmp_path: Path) -> None:
    """All four dumps land in session_dir when all four args are passed."""
    hippo = _StubBio("hippocampus")
    nac = _StubBio("nac")
    ec = _StubBio("ec")
    atl = _StubBio("atl")

    save_aut_state(
        hippocampus=hippo,
        nac=nac,
        base_dir=str(tmp_path),
        session_id="20260513_999999",
        ec=ec,
        atl=atl,
    )

    session_dir = tmp_path / "20260513_999999"
    assert (session_dir / "aut_hippocampus.json").is_file()
    assert (session_dir / "aut_nac.json").is_file()
    assert (session_dir / "aut_ec.json").is_file()
    assert (session_dir / "aut_atl.json").is_file()

    assert json.loads((session_dir / "aut_ec.json").read_text())["stub"] == "ec"
    assert json.loads((session_dir / "aut_atl.json").read_text())["stub"] == "atl"


def test_save_aut_state_ec_and_atl_optional(tmp_path: Path) -> None:
    """Back-compat: existing callers that pass only hippocampus + NAc still work."""
    hippo = _StubBio("hippocampus")
    nac = _StubBio("nac")

    save_aut_state(
        hippocampus=hippo,
        nac=nac,
        base_dir=str(tmp_path),
        session_id="20260513_888888",
    )

    session_dir = tmp_path / "20260513_888888"
    assert (session_dir / "aut_hippocampus.json").is_file()
    assert (session_dir / "aut_nac.json").is_file()
    assert not (session_dir / "aut_ec.json").exists()
    assert not (session_dir / "aut_atl.json").exists()


def test_save_aut_state_skips_missing_subjects(tmp_path: Path) -> None:
    """``None`` for any subject silently skips that file; never raises."""
    save_aut_state(
        hippocampus=None,
        nac=None,
        base_dir=str(tmp_path),
        session_id="20260513_777777",
        ec=None,
        atl=None,
    )

    session_dir = tmp_path / "20260513_777777"
    # session_dir is still created (mkdir is unconditional), but no dumps land
    assert session_dir.is_dir()
    assert list(session_dir.iterdir()) == []


def test_save_aut_state_swallows_save_failure(tmp_path: Path) -> None:
    """A subject whose ``save()`` raises does not abort the other dumps."""

    class _BrokenBio(_StubBio):
        def save(self, path: str) -> None:
            raise RuntimeError("simulated save failure")

    hippo = _StubBio("hippocampus")
    nac = _StubBio("nac")
    broken_ec = _BrokenBio("ec")
    atl = _StubBio("atl")

    save_aut_state(
        hippocampus=hippo,
        nac=nac,
        base_dir=str(tmp_path),
        session_id="20260513_666666",
        ec=broken_ec,
        atl=atl,
    )

    session_dir = tmp_path / "20260513_666666"
    assert (session_dir / "aut_hippocampus.json").is_file()
    assert (session_dir / "aut_nac.json").is_file()
    assert not (session_dir / "aut_ec.json").exists()
    assert (session_dir / "aut_atl.json").is_file()
