"""`maxim serve` skeleton — the Console backend + OpenAPI facade contract.

Pins the contract-completeness invariant: the OpenAPI schema carries every seam
shape even while the seam bodies are 501 stubs, so the maxim-pulse kit can generate
its full FacadeClient now. Skips cleanly when the `console` extra is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="requires the `console` extra (fastapi/uvicorn)")

from fastapi.testclient import TestClient  # noqa: E402

from maxim.console.server import build_app  # noqa: E402

_TOKEN = "mxc_" + "t" * 43
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


def _client(app):
    return TestClient(app, headers=_AUTH)


# Every path the frontend generates against — live verbs + typed seam stubs.
_EXPECTED_PATHS = {
    "/api/models",
    "/api/diagnose",
    "/api/probe",
    "/api/setup/mesh",
    "/api/setup/cloud",
    "/api/recall",
    "/api/run",
}

# Seam shapes that MUST be in OpenAPI for type-gen (no bare dict/Any on a seam).
_EXPECTED_SCHEMAS = {
    "ModelsResponse",
    "ProbeRequest",
    "ProbeResult",
    "MeshSetupRequest",
    "CloudSetupRequest",
    "SetupResult",
    "RecallResponse",
    "RunRequest",
    "RunAccepted",
    "ConsoleEvent",
}


@pytest.fixture(scope="module")
def app():
    return build_app(None, auth_token=_TOKEN)


def test_openapi_paths_complete(app):
    spec = app.openapi()
    assert _EXPECTED_PATHS <= set(spec["paths"].keys())


def test_openapi_seam_schemas_complete(app):
    """The contract is complete before the seams are — this is the load-bearing invariant."""
    spec = app.openapi()
    schemas = set(spec["components"]["schemas"].keys())
    missing = _EXPECTED_SCHEMAS - schemas
    assert not missing, f"seam shapes missing from OpenAPI (breaks type-gen): {missing}"


def test_live_verb_models_ok(app):
    r = _client(app).get("/api/models")
    assert r.status_code == 200
    assert "groups" in r.json()


@pytest.mark.parametrize(
    "method,path,body",
    [
        # recall + setup/cloud + adventure + TALK are all LIVE now (#425, the
        # SETUP seam, HANDLE (a), and talk mode). Only sim/rest remain 501.
        # rest is LIVE (consolidation without teardown); only sim stays 501,
        # and deliberately so — it points at the CLI.
        ("post", "/api/run", {"mode": "sim"}),
    ],
)
def test_seam_stubs_are_501(app, method, path, body):
    c = _client(app)
    r = c.get(path) if method == "get" else c.post(path, json=body)
    assert r.status_code == 501


def test_setup_cloud_writes_placement_ref(app, tmp_path, monkeypatch):
    """SETUP cloud is live: writes a resolvable large-tier CLOUD placement with
    the key as a 0600 ref (never inline) + cloud.enabled + budget."""
    import stat
    from pathlib import Path

    cfg = tmp_path / "config.json"
    monkeypatch.setattr("maxim.runtime.config_writer.config_path", lambda: cfg)
    r = _client(app).post(
        "/api/setup/cloud",
        json={"provider": "anthropic", "profile": "claude-sonnet", "api_key": "sk-cl", "monthly_budget_usd": 20.0},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True and j["placement"] == "cloud"
    from maxim.runtime.config_loader import load_config

    conf = load_config(cfg)
    pl = conf.lanes.large.placement
    assert len(pl) == 1 and pl[0].origin == "cloud" and pl[0].model == "claude-sonnet"
    assert pl[0].api_key_ref and "sk-cl" not in cfg.read_text()  # ref, never inline
    assert stat.S_IMODE(Path(pl[0].api_key_ref).stat().st_mode) == 0o600
    assert conf.cloud.enabled and conf.cloud.max_lanes >= 1 and conf.cloud.session_budget_usd == 20.0


def test_recall_maps_curated_blend_to_wire(app, monkeypatch):
    """RECALL wraps api.recall()'s curated blend → the wire shape (story summaries
    + traits + preferences), never raw episodes."""
    from maxim.integration.recall import CuratedRecall, RecalledItem

    blend = CuratedRecall(
        name="Ada",
        player_model=["gravitates toward diplomacy"],
        story_memories=[RecalledItem(text="your rogue betrayed the party", kind="story", salience=0.9)],
        preferences=[RecalledItem(text="prefers stealth", kind="preference", salience=0.7, learned_from="play")],
    )
    monkeypatch.setattr("maxim.recall", lambda **kw: blend)
    j = _client(app).get("/api/recall").json()
    assert j["name"] == "Ada"
    assert j["player_model"] == ["gravitates toward diplomacy"]
    assert j["story_memories"][0]["summary"] == "your rogue betrayed the party"
    assert j["story_memories"][0]["salience"] == 0.9
    assert j["preferences"][0]["about"] == "prefers stealth"
    assert j["preferences"][0]["learned_from"] == "play"


def test_setup_mesh_writes_ref_config(app, tmp_path, monkeypatch):
    """SETUP mesh is live: writes a resolvable peer placement with the key as a
    0600 ref (never inline). Routed to a temp config so no real state is touched."""
    import stat

    cfg = tmp_path / "config.json"
    # config_writer imports config_path by name, so patch ITS binding (the
    # by-name-import gotcha), which apply_mesh_setup + mutate_config both use.
    monkeypatch.setattr("maxim.runtime.config_writer.config_path", lambda: cfg)
    r = _client(app).post("/api/setup/mesh", json={"leader_url": "https://leader.example", "api_key": "sk-xyz"})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True and j["placement"] == "mesh"
    # key is a ref, never inline; secret is 0600
    from maxim.runtime.config_loader import load_config

    conf = load_config(cfg)
    ref = conf.lanes.large.remote_api_key_ref
    assert ref and "sk-xyz" not in cfg.read_text()
    from pathlib import Path

    assert stat.S_IMODE(Path(ref).stat().st_mode) == 0o600
    assert conf.role == "peer" and conf.lanes.large.remote_url == "https://leader.example"


def test_probe_cloud_shape_dispatches(app):
    """PROBE now accepts the cloud shape (provider, no url): a missing key fails,
    a present key warns (no false-green — a live round-trip isn't faked)."""
    c = _client(app)
    r_missing = c.post("/api/probe", json={"provider": "anthropic"})
    assert r_missing.status_code == 200 and r_missing.json()["status"] == "fail"
    r_key = c.post("/api/probe", json={"provider": "anthropic", "api_key": "sk-x"})
    assert r_key.status_code == 200 and r_key.json()["status"] == "warn"
    # neither url nor provider → typed 422
    assert c.post("/api/probe", json={}).status_code == 422


def test_diagnose_platform_is_structured(app):
    """DiagnoseResponse.platform is now a structured object (os/arch/...), not a
    stringified PlatformInfo repr."""
    j = _client(app).get("/api/diagnose").json()
    assert isinstance(j["platform"], dict)
    assert "os" in j["platform"] and "arch" in j["platform"]


def test_models_carry_curated_marker(app):
    """ModelInfoWire exposes the curated marker so the wizard picks by intent."""
    groups = _client(app).get("/api/models").json()["groups"]
    everything = [m for members in groups.values() for m in members]
    assert everything and all("curated" in m for m in everything)


def test_probe_wires_classifier(app, monkeypatch):
    """PROBE maps the peer health_check outcome through the shared classifier to the
    wire shape — mocked so no network. Same classifier path `maxim doctor` uses."""
    from types import SimpleNamespace

    fake = SimpleNamespace(url="http://leader", outcome="auth_rejected", detail="HTTP 401", latency_ms=12.3)

    class _FakeBackend:
        def health_check(self, **kw):
            return fake

    monkeypatch.setattr(
        "maxim.models.language.maxim_peer_backend._MaximPeerBackend.for_url",
        classmethod(lambda cls, url, **kw: _FakeBackend()),
    )
    r = _client(app).post("/api/probe", json={"url": "http://leader", "api_key": "k"})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "fail"  # auth_rejected → fail
    assert j["outcome"] == "auth_rejected"
    assert j["latency_ms"] == 12.3
    assert "auth rejected" in j["message"]
    assert j["fix_hint"]  # classifier supplies an actionable fix


def test_seam_request_validation_is_typed(app):
    """A malformed seam body is a 422 (typed), proving the schema is enforced."""
    r = _client(app).post("/api/run", json={"mode": "not-a-valid-mode"})
    assert r.status_code == 422


def test_console_port_default_is_8765():
    from maxim.runtime.config_loader import resolve_setting

    value, _source = resolve_setting("console.port", cli_value=None)
    assert value == 8765


def test_openapi_snapshot_is_fresh():
    """The committed openapi.json must match the live schema — maxim-pulse generates
    its FacadeClient from this file, so drift here is silent cross-repo drift.
    Regenerate with: ``maxim serve --dump-openapi``."""
    import json

    from maxim.console.server import _OPENAPI_SNAPSHOT, openapi_schema

    assert _OPENAPI_SNAPSHOT.exists(), "snapshot missing — run: maxim serve --dump-openapi"
    committed = json.loads(_OPENAPI_SNAPSHOT.read_text())
    assert committed == openapi_schema(), "OpenAPI snapshot stale — run: maxim serve --dump-openapi"
