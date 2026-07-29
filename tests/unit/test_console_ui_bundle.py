"""Guards for Console UI bundle vendoring + the cross-repo contract check.

What makes `pip install pymaxim[console] && maxim serve` show a working
Console: the bundle is built in maxim-pulse, vendored into package data at
release time, and resolved as the last fallback behind --ui-dist and config.
"""

from __future__ import annotations

import json

import pytest

from maxim.console.ui_bundle import (
    CONSOLE_CONTRACT_VERSION,
    UI_MANIFEST_NAME,
    check_ui_contract,
    read_ui_manifest,
    resolve_ui_dist,
)


def _bundle(root, *, contract=CONSOLE_CONTRACT_VERSION, target="console", manifest=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text("<html></html>")
    (root / "assets").mkdir(exist_ok=True)
    if manifest:
        (root / UI_MANIFEST_NAME).write_text(
            json.dumps({"target": target, "app_version": "0.0.1", "contract_version": contract, "commit": "abc"})
        )
    return root


class TestResolution:
    def test_precedence_cli_over_config_over_packaged(self, tmp_path, monkeypatch):
        import maxim.console.ui_bundle as ub

        packaged = _bundle(tmp_path / "packaged")
        monkeypatch.setattr(ub, "packaged_ui_dist", lambda: packaged)
        assert resolve_ui_dist("/cli/path", "/config/path") == __import__("pathlib").Path("/cli/path")
        assert resolve_ui_dist(None, "/config/path") == __import__("pathlib").Path("/config/path")
        assert resolve_ui_dist(None, None) == packaged

    def test_empty_strings_are_not_paths(self, tmp_path, monkeypatch):
        # `maxim config set console.ui_dist ""` must fall through to packaged,
        # not resolve to Path("") (which is the CWD).
        import maxim.console.ui_bundle as ub

        packaged = _bundle(tmp_path / "packaged")
        monkeypatch.setattr(ub, "packaged_ui_dist", lambda: packaged)
        assert resolve_ui_dist("", "") == packaged

    def test_no_bundle_anywhere_resolves_to_none(self, monkeypatch):
        import maxim.console.ui_bundle as ub

        monkeypatch.setattr(ub, "packaged_ui_dist", lambda: None)
        assert resolve_ui_dist(None, None) is None

    def test_half_copied_packaged_dir_does_not_shadow_the_no_ui_page(self, tmp_path, monkeypatch):
        # A directory without index.html must NOT count as installed —
        # otherwise a broken vendoring serves a blank page instead of the
        # clearer "no UI installed" explanation.
        import maxim.console.ui_bundle as ub

        empty = tmp_path / "ui_dist"
        empty.mkdir()
        monkeypatch.setattr(ub, "_PACKAGED_UI_DIST", empty)
        assert ub.packaged_ui_dist() is None


class TestContractCheck:
    def test_matching_contract_is_silent(self, tmp_path):
        assert check_ui_contract(_bundle(tmp_path / "b")) is None

    def test_mismatch_warns_with_both_versions(self, tmp_path, caplog):
        import logging

        bundle = _bundle(tmp_path / "b", contract="9.9.9")
        with caplog.at_level(logging.WARNING):
            msg = check_ui_contract(bundle)
        assert msg and "9.9.9" in msg and CONSOLE_CONTRACT_VERSION in msg
        assert any("contract mismatch" in r.getMessage() for r in caplog.records)

    def test_missing_manifest_is_not_an_error(self, tmp_path):
        # Hand-built / pre-manifest bundles simply cannot be checked.
        assert check_ui_contract(_bundle(tmp_path / "b", manifest=False)) is None
        assert read_ui_manifest(tmp_path / "b") is None

    def test_unreadable_manifest_never_raises(self, tmp_path):
        bundle = _bundle(tmp_path / "b")
        (bundle / UI_MANIFEST_NAME).write_text("{not json")
        assert check_ui_contract(bundle) is None  # warns internally, does not raise

    def test_absent_bundle_is_a_noop(self):
        assert check_ui_contract(None) is None

    def test_contract_version_is_the_app_version(self):
        # One source of truth: a bundle compares itself against the same
        # string the OpenAPI schema advertises. Needs the `console` extra
        # (fastapi) — CI's unit-test job installs a lean set.
        pytest.importorskip("fastapi", reason="requires the `console` extra")
        from maxim.console.server import build_app

        assert build_app(None).openapi()["info"]["version"] == CONSOLE_CONTRACT_VERSION


class TestVendorScript:
    def _script(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "scripts" / "vendor_console_ui.py"
        spec = importlib.util.spec_from_file_location("vendor_console_ui", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    def test_validates_a_good_bundle(self, tmp_path):
        ok, problems = self._script().validate(_bundle(tmp_path / "dist"))
        assert ok and problems == []

    def test_rejects_the_wrong_target(self, tmp_path):
        # Pointing at apps/reachy/ui/dist by mistake is the easy slip.
        ok, problems = self._script().validate(_bundle(tmp_path / "dist", target="reachy"))
        assert not ok
        assert any("target" in p for p in problems)

    def test_rejects_a_contract_mismatch(self, tmp_path):
        ok, problems = self._script().validate(_bundle(tmp_path / "dist", contract="9.9.9"))
        assert not ok
        assert any("contract mismatch" in p for p in problems)

    def test_rejects_a_non_bundle_directory(self, tmp_path):
        empty = tmp_path / "nope"
        empty.mkdir()
        ok, problems = self._script().validate(empty)
        assert not ok
        assert any("index.html" in p for p in problems)


@pytest.mark.parametrize("pattern", ["console/ui_dist/**/*"])
def test_pyproject_ships_the_bundle_as_package_data(pattern):
    # Without this the wheel silently omits the vendored bundle and every
    # pip-installed user gets the "no UI installed" page.
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    assert pattern in data["tool"]["setuptools"]["package-data"]["maxim"]


class TestContractVersionIsNotDecorative:
    """A stamp that never changes cannot detect drift.

    #438 added /api/campaigns + /api/events/subscribe-frame, made
    ConsoleEvent.tier/seq/message REQUIRED, and added RunAccepted.reply plus a
    "completed" status — all things a 0.1.0 bundle predates — yet the version
    stayed 0.1.0, so the stamp could not have flagged any of it.
    """

    def test_version_moved_past_the_initial_facade(self):
        assert CONSOLE_CONTRACT_VERSION != "0.1.0"

    def test_committed_openapi_snapshot_carries_the_same_version(self):
        # The snapshot IS what maxim-pulse generates its client from; if it
        # drifts from the constant the stamp compares against, the check lies.
        import json
        from pathlib import Path

        snapshot = Path(__file__).resolve().parents[2] / "src" / "maxim" / "console" / "openapi.json"
        schema = json.loads(snapshot.read_text())
        assert schema["info"]["version"] == CONSOLE_CONTRACT_VERSION

    def test_a_stale_bundle_is_now_flagged(self, tmp_path):
        # The concrete regression: a bundle built against the pre-#438 contract
        # must warn rather than pass silently.
        stale = _bundle(tmp_path / "stale", contract="0.1.0")
        msg = check_ui_contract(stale)
        assert msg and "0.1.0" in msg and CONSOLE_CONTRACT_VERSION in msg
