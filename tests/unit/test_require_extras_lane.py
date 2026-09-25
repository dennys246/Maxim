"""The extras lane's positive control (roadmap 1.3.x "Test/CI truthfulness"; public_oasis Phase 0 item 1).

``--require-extras=console,sign`` turns a skip for a missing REQUIRED extra into a failure, so the lane
that installs them cannot go quietly vacuous -- a lane that installs the extras and still skips the
tests looks exactly like one that ran them.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import required_extra_skip

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("reason", "required", "expected"),
    [
        ("requires the `console` extra (fastapi/uvicorn)", {"console", "sign"}, "console"),
        ("requires the `console` extra", {"console"}, "console"),
        ("signed bundles need the [sign] extra (cryptography)", {"console", "sign"}, "sign"),
        ("signed bundles need the [sign] extra (cryptography)", {"console"}, None),  # not required here
        ("requires pretrained model/dataset assets", {"console", "sign"}, None),  # a different skip
    ],
)
def test_the_reasons_the_lane_reads(reason, required, expected):
    assert required_extra_skip(reason, required) == expected


def _run(tmp_path: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    test = tmp_path / "test_needs_sign.py"
    test.write_text(
        textwrap.dedent(
            """
            import pytest
            pytest.importorskip("maxim_no_such_module_xyz", reason="signed bundles need the [sign] extra (cryptography)")

            def test_signed():
                pass
            """
        )
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "tests.conftest", "-p", "no:cacheprovider", str(test), *flags],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_required_extra_skip_fails_the_lane(tmp_path):
    required = _run(tmp_path, "--require-extras=console,sign")
    assert required.returncode != 0, required.stdout
    assert "the 'sign' extra is required on this lane" in required.stdout
    # the control: without the flag the same module is just skipped
    optional = _run(tmp_path)
    # a module-level skip collects nothing: pytest exits 5 ("no tests collected"), never 1 ("failed")
    assert optional.returncode in (0, 5), optional.stdout
    assert "1 skipped" in optional.stdout and "required on this lane" not in optional.stdout


def test_a_skipif_marker_for_a_required_extra_fails_the_lane_too(tmp_path):
    """The other skip path: ``skipif`` markers (test_hive_cli, test_oasis_store) skip at SETUP, not collection."""
    test = tmp_path / "test_marker_sign.py"
    test.write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.mark.skipif(True, reason="signed bundles need the [sign] extra (cryptography)")
            def test_signed():
                pass
            """
        )
    )
    args = [sys.executable, "-m", "pytest", "-q", "-p", "tests.conftest", "-p", "no:cacheprovider", str(test)]
    required = subprocess.run([*args, "--require-extras=sign"], cwd=REPO, capture_output=True, text=True, check=False)
    assert required.returncode == 1 and "the 'sign' extra is required on this lane" in required.stdout, required.stdout
    optional = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False)
    assert optional.returncode == 0 and "1 skipped" in optional.stdout, optional.stdout
