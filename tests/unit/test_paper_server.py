"""scripts/_paper_server.py — the shared Paper-server setup helpers both world builders use.

Extracted 2026-09-18 during the Exp 56 re-baseline port; these pin the behaviours the two builders
had already diverged on (sha256 verify + unlink-on-failure; the Java major parse; the stale-world
guard's three refusal shapes; the atomic, format-versioned stamp; the measured server version).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _paper_server as PS  # noqa: E402


@pytest.mark.parametrize(
    ("line", "major"),
    [
        ('openjdk version "1.8.0_392"', 8),
        ('openjdk version "11.0.21" 2023-10-17', 11),
        ('openjdk version "16" 2021-03-16', 16),
        ('openjdk version "17.0.9" 2023-10-17 LTS', 17),
        ('openjdk version "21.0.1" 2023-10-17', 21),
        ('openjdk version "15-ea" 2020-09-15', 15),
        ('java version "9"', 9),
        ("The operation couldn't be completed. Unable to locate a Java Runtime.", None),
        (None, None),
        ("unknown", None),
    ],
)
def test_java_major_parses_every_scheme(line, major) -> None:
    assert PS.java_major(line) == major


def test_java_start_command_pins_the_binary_per_invocation(tmp_path: Path) -> None:
    cmd = PS.java_start_command(tmp_path, "paper-1.20.4.jar")
    assert "java_home -v 17" in cmd and "paper-1.20.4.jar nogui" in cmd and str(tmp_path) in cmd


def test_stale_world_guard_three_refusals_and_two_passes(tmp_path: Path) -> None:
    assert PS.refuse_stale_world(tmp_path, "w", "1.20.4", force=False) is None  # no world yet
    (tmp_path / "w").mkdir()
    msg = PS.refuse_stale_world(tmp_path, "w", "1.20.4", force=False)
    assert msg and "no version stamp" in msg
    (tmp_path / PS.STAMP_NAME).write_text(json.dumps({"mc_version": "1.16.5"}))
    msg = PS.refuse_stale_world(tmp_path, "w", "1.20.4", force=False)
    assert msg and "stamped MC 1.16.5" in msg
    (tmp_path / PS.STAMP_NAME).write_text("{not json")
    msg = PS.refuse_stale_world(tmp_path, "w", "1.20.4", force=False)
    assert msg and "unreadable" in msg
    assert PS.refuse_stale_world(tmp_path, "w", "1.20.4", force=True) is None  # deliberate override
    PS.write_world_stamp(tmp_path, "1.20.4", stamped_by="test")
    assert PS.refuse_stale_world(tmp_path, "w", "1.20.4", force=False) is None  # matching stamp


def test_world_stamp_is_format_versioned(tmp_path: Path) -> None:
    p = PS.write_world_stamp(tmp_path, "1.20.4", stamped_by="scripts/x.py")
    data = json.loads(p.read_text())
    assert data["_format_version"] == "1.0" and data["mc_version"] == "1.20.4" and data["stamped_by"] == "scripts/x.py"


def test_download_failure_unlinks_the_partial_jar(tmp_path: Path, monkeypatch, capsys) -> None:
    import maxim.utils.http as http

    def fake_fetch(url, timeout=30):
        class R:
            content = json.dumps(
                {"id": 1, "downloads": {"server:default": {"url": "u", "size": 3, "checksums": {"sha256": "0" * 64}}}}
            ).encode()

        return R()

    def fake_download(url, dest, expected_bytes=None):
        Path(dest).write_bytes(b"abc")  # lands, then the sha256 check fails

    monkeypatch.setattr(http, "fetch_url", fake_fetch)
    monkeypatch.setattr(http, "download_to_file", fake_download)
    jar = tmp_path / "paper-1.20.4.jar"
    assert PS.download_paper("1.20.4", jar) is False
    assert not jar.exists()
    assert "sha256 mismatch" in capsys.readouterr().out
    jar.write_bytes(b"present")
    assert PS.download_paper("1.20.4", jar) is True  # an existing jar is not re-downloaded


@pytest.mark.parametrize(
    ("reply", "ok"),
    [
        (
            "This server is running Paper version git-Paper-499 (MC: 1.20.4) (Implementing API version 1.20.4-R0.1-SNAPSHOT)",
            True,
        ),
        ("This server is running Paper version git-Paper-794 (MC: 1.16.5)", False),
        ("", False),
        (None, False),
        ("Checking version, please wait...", False),
    ],
)
def test_server_version_matches_fails_closed(reply, ok) -> None:
    assert PS.server_version_matches(reply, "1.20.4") is ok
