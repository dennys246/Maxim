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
    return build_app(None)


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
    r = TestClient(app).get("/api/models")
    assert r.status_code == 200
    assert "groups" in r.json()


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/api/probe", {"url": "http://x"}),
        ("post", "/api/setup/mesh", {"leader_url": "http://x", "api_key": "k"}),
        ("get", "/api/recall", None),
        ("post", "/api/run", {"mode": "talk"}),
    ],
)
def test_seam_stubs_are_501(app, method, path, body):
    c = TestClient(app)
    r = c.get(path) if method == "get" else c.post(path, json=body)
    assert r.status_code == 501


def test_seam_request_validation_is_typed(app):
    """A malformed seam body is a 422 (typed), proving the schema is enforced."""
    r = TestClient(app).post("/api/run", json={"mode": "not-a-valid-mode"})
    assert r.status_code == 422


def test_console_port_default_is_8765():
    from maxim.runtime.config_loader import resolve_setting

    value, _source = resolve_setting("console.port", cli_value=None)
    assert value == 8765
