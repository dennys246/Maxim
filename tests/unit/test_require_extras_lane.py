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

from tests.conftest import extra_import_names, required_extra_skip

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("reason", "required", "expected"),
    [
        ("requires the `console` extra (fastapi/uvicorn)", {"console", "sign"}, "console"),
        ("requires the `console` extra", {"console"}, "console"),
        ("signed bundles need the [sign] extra (cryptography)", {"console", "sign"}, "sign"),
        ("signed bundles need the [sign] extra (cryptography)", {"console"}, None),  # not required here
        ("requires pretrained model/dataset assets", {"console", "sign"}, None),  # a different skip
        # pytest's own importorskip message, for a call with no reason= (#940 item 4)
        ("could not import 'cryptography': No module named 'cryptography'", {"sign"}, "sign"),
        ("could not import 'cryptography.hazmat': No module named 'cryptography'", {"sign"}, "sign"),
        ("could not import 'fastapi': No module named 'fastapi'", {"console", "sign"}, "console"),
        ("could not import 'cryptography': No module named 'cryptography'", {"console"}, None),
        ("could not import 'cryptographyx': No module named 'cryptographyx'", {"sign"}, None),
    ],
)
def test_the_reasons_the_lane_reads(reason, required, expected):
    assert required_extra_skip(reason, required) == expected


def _extras_importable(tmp_path: Path) -> dict[str, str]:
    """An env where every console/sign module imports, whatever this box has installed.

    ``--require-extras`` refuses to start when an extra's modules do not import, so the skip-matcher tests
    below would otherwise test the box (exit 4 without the extras) instead of the matcher."""
    import os

    stubs = tmp_path / "extra_stubs"
    stubs.mkdir(exist_ok=True)
    for name in extra_import_names("console") + extra_import_names("sign"):
        (stubs / f"{name}.py").write_text("")
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(stubs), os.environ.get("PYTHONPATH", "")])}


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
        env=_extras_importable(tmp_path),
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
    env = _extras_importable(tmp_path)
    required = subprocess.run(
        [*args, "--require-extras=sign"], cwd=REPO, capture_output=True, text=True, check=False, env=env
    )
    assert required.returncode == 1 and "the 'sign' extra is required on this lane" in required.stdout, required.stdout
    optional = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False, env=env)
    assert optional.returncode == 0 and "1 skipped" in optional.stdout, optional.stdout


def test_the_extras_import_names_come_from_pyproject():
    """Known answer: today's console/sign requirements. A distribution whose import name differs from its
    requirement name would need a mapping, and installed ones are checked against their real top-level modules."""
    from importlib.metadata import packages_distributions

    assert extra_import_names("console") == ("fastapi", "uvicorn")
    assert extra_import_names("sign") == ("cryptography", "rfc8785")
    modules_of: dict[str, set[str]] = {}
    for module, dists in packages_distributions().items():
        for dist in dists:
            modules_of.setdefault(dist.lower().replace("-", "_"), set()).add(module)
    for name in extra_import_names("console") + extra_import_names("sign"):
        if name in modules_of:  # installed: the requirement's own distribution provides the derived module
            assert name in modules_of[name], (name, modules_of[name])


def test_a_reasonless_importorskip_of_a_required_extra_fails_the_lane(tmp_path):
    """The escape #940 item 4 named: ``pytest.importorskip("cryptography")`` with no reason= skipped with
    pytest's own message and slipped past ``--require-extras``. ``sys.modules[...] = None`` makes the import
    fail here whether or not the extra is installed."""
    test = tmp_path / "test_reasonless.py"
    test.write_text(
        textwrap.dedent(
            """
            import sys
            import pytest

            sys.modules["rfc8785"] = None

            def test_signed():
                pytest.importorskip("rfc8785")
            """
        )
    )
    args = [sys.executable, "-m", "pytest", "-q", "-p", "tests.conftest", "-p", "no:cacheprovider", str(test)]
    env = _extras_importable(tmp_path)
    required = subprocess.run(
        [*args, "--require-extras=sign"], cwd=REPO, capture_output=True, text=True, check=False, env=env
    )
    assert required.returncode == 1 and "the 'sign' extra is required on this lane" in required.stdout, required.stdout
    optional = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False, env=env)
    assert optional.returncode == 0 and "1 skipped" in optional.stdout, optional.stdout


def _configure_only(tmp_path: Path, *flags: str, block: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run pytest on a trivial test; ``block`` makes that module unimportable before conftest configures."""
    (tmp_path / "test_trivial.py").write_text("def test_ok():\n    pass\n")
    plugins: list[str] = []
    env = None
    if block is not None:
        (tmp_path / "block_extra_module.py").write_text(f"import sys\nsys.modules[{block!r}] = None\n")
        plugins = ["-p", "block_extra_module"]
        import os

        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *plugins, "-p", "tests.conftest", "-p", "no:cacheprovider"]
        + [str(tmp_path / "test_trivial.py"), *flags],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_a_lane_whose_required_extra_does_not_import_refuses_to_start(tmp_path):
    """The positive half: whatever shape a test's skip takes, the lane cannot run without the extra."""
    refused = _configure_only(tmp_path, "--require-extras=sign", block="rfc8785")
    assert refused.returncode == 4, refused.stdout + refused.stderr  # pytest's UsageError exit
    assert "the 'sign' extra is required on this lane, but 'rfc8785' does not import" in refused.stderr
    # the control: the same blocked module without the flag is not the lane's business
    assert _configure_only(tmp_path, block="rfc8785").returncode == 0


def test_an_undeclared_extra_is_refused(tmp_path):
    refused = _configure_only(tmp_path, "--require-extras=sing")
    assert refused.returncode == 4 and "pyproject.toml declares no 'sing' extra" in refused.stderr, refused.stderr
