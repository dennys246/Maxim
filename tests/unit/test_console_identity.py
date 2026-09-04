"""Backend identity — "which build am I actually talking to?"

Exists because that question was unanswerable, and the answer changes
SILENTLY: pymaxim is typically installed editable, so `maxim serve` follows
the checked-out git branch (switch to a branch predating a seam and that
seam vanishes from the UI), and a stale serve process can outlive the code
on disk. Both bit this session's debugging.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="requires the `console` extra")

from fastapi.testclient import TestClient  # noqa: E402

from maxim.console.server import build_app, build_identity  # noqa: E402

_TOKEN = "mxc_" + "t" * 43
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


class TestIdentityContent:
    def test_reports_version_and_contract(self):
        import maxim
        from maxim.console.ui_bundle import CONSOLE_CONTRACT_VERSION

        ident = build_identity()
        assert ident.package_version == maxim.__version__
        assert ident.contract_version == CONSOLE_CONTRACT_VERSION

    def test_reports_git_identity_in_a_checkout(self):
        # The load-bearing field for an editable install: the branch IS the
        # capability set. (None is legitimate for a wheel install, so only
        # assert the pair is coherent.)
        ident = build_identity()
        assert (ident.git_sha is None) == (ident.git_branch is None)

    def test_seam_liveness_matches_reality(self):
        seams = {s.name: s.live for s in build_identity().seams}
        # If a seam flips live/501, this must be updated in the same commit —
        # an identity surface that lies is worse than none.
        assert seams["talk"] is True
        assert seams["adventure"] is True
        assert seams["rest"] is True
        assert seams["sim"] is False
        assert seams["event"] is True

    def test_sim_explains_where_to_go(self):
        sim = next(s for s in build_identity().seams if s.name == "sim")
        assert sim.detail and "maxim --sim" in sim.detail

    def test_ui_source_is_none_without_a_bundle(self):
        ident = build_identity(None, "none")
        assert ident.ui_source == "none"
        assert ident.ui_dist is None
        assert ident.ui_manifest == {}

    def test_ui_manifest_is_surfaced_when_present(self, tmp_path):
        import json

        from maxim.console.ui_bundle import UI_MANIFEST_NAME

        bundle = tmp_path / "dist"
        bundle.mkdir()
        (bundle / "index.html").write_text("<html></html>")
        (bundle / UI_MANIFEST_NAME).write_text(json.dumps({"target": "console", "contract_version": "9.9.9"}))
        ident = build_identity(bundle, "flag")
        # The OTHER half of a contract mismatch: what the bundle claims.
        assert ident.ui_source == "flag"
        assert ident.ui_manifest["contract_version"] == "9.9.9"


class TestIdentityEndpoints:
    def test_http_endpoint(self):
        with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
            body = c.get("/api/identity").json()
        assert body["package_version"] and body["contract_version"]
        assert isinstance(body["seams"], list) and body["seams"]

    def test_identity_is_the_FIRST_ws_frame(self):
        # A client must know what it is attached to BEFORE it starts
        # interpreting what that thing says.
        with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
            with c.websocket_connect("/ws", headers=_AUTH) as ws:
                first = ws.receive_json()
        assert first["kind"] == "identity"
        assert first["tier"] == "clean"
        assert first["data"]["package_version"]

    def test_identity_bypasses_subscribe_filters(self):
        # It is a meta-kind: a client that filters to one channel must still
        # learn which backend it is talking to.
        from maxim.console.server import _META_KINDS, _WsConn
        from maxim.console.schemas import SubscribeFrame

        assert "identity" in _META_KINDS
        conn = _WsConn()
        conn.apply_frame(SubscribeFrame(channels=["memory"], tier="clean"))
        assert conn.matches("identity", "clean", "") is True

    def test_build_app_records_the_served_bundle(self, tmp_path):
        import maxim.console.server as srv

        bundle = tmp_path / "dist"
        bundle.mkdir()
        (bundle / "index.html").write_text("<html></html>")
        build_app(bundle, "config", auth_token=_TOKEN)
        assert srv._SERVED_UI_DIST[0] == bundle
        assert srv._SERVED_UI_DIST[1] == "config"
        build_app(None, auth_token=_TOKEN)  # reset for other tests
        assert srv._SERVED_UI_DIST == (None, "none")


class TestDiagnoseSurfacesRealChecks:
    """`report.sections` is a list of (group, [CheckResult]) TUPLES — the old
    mapping treated them as dataclasses, so it emitted one BLANK row per group
    and dropped every actual check."""

    def test_checks_are_flattened_with_names_and_status(self):
        with TestClient(build_app(None, auth_token=_TOKEN), headers=_AUTH) as c:
            body = c.get("/api/diagnose").json()
        sections = body["sections"]
        assert len(sections) > 10, "should be one row per CHECK, not per group"
        named = [s for s in sections if s["name"]]
        assert len(named) == len(sections), "every row must have a name"
        assert any(s["status"] in {"ok", "warn", "fail", "info"} for s in sections)
        assert any(s["extra"].get("group") for s in sections), "the group must survive as context"
