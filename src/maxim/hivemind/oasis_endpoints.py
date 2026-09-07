"""Substrate-exchange endpoint handlers — the protocol, independent of transport.

1.2 P2P Slice B. These functions turn a parsed request + an :class:`OasisStore`
into an :class:`EndpointResponse` (status + content-type + body bytes). They do
NOT touch an HTTP framework: ``leader_proxy`` reads the request, calls one of
these, and writes the response back. Keeping the protocol here (next to the
store and the merge/ingest math) means the substrate wire logic is unit-testable
without spinning a server, and the review round reads one hivemind module rather
than diffing a 2,600-line proxy.

Three endpoints, mounted at ``/v1/substrate/`` on the authenticated Oasis
server:

- ``GET  /v1/substrate/releases``       → :func:`handle_list_releases`
- ``GET  /v1/substrate/bundle/<id>``    → :func:`handle_get_bundle`
- ``POST /v1/substrate/contribute``     → :func:`handle_contribute`

Auth, rate-limiting, and admission are the server's job (reused from the proxy);
these handlers assume the caller is already authorized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from maxim.hivemind.store import OasisStore, OasisStoreError

CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_BUNDLE = "application/zip"

# Path prefix the server matches to route here.
SUBSTRATE_PREFIX = "/v1/substrate"


@dataclass(frozen=True)
class EndpointResponse:
    """A transport-agnostic HTTP response: status, content-type, body bytes."""

    status: int
    content_type: str
    body: bytes

    @classmethod
    def json_response(cls, status: int, payload: object) -> EndpointResponse:
        """Build a JSON response. Named to not shadow the module-level ``json``."""
        return cls(status, CONTENT_TYPE_JSON, json.dumps(payload).encode("utf-8"))


def handle_list_releases(store: OasisStore) -> EndpointResponse:
    """``GET /v1/substrate/releases`` → the release summaries (manifests only)."""
    return EndpointResponse.json_response(200, {"releases": store.list_releases()})


def handle_get_bundle(store: OasisStore, release_id: str) -> EndpointResponse:
    """``GET /v1/substrate/bundle/<id>`` → the signed bundle bytes, or 404."""
    try:
        raw = store.open_release(release_id)
    except OasisStoreError as exc:
        return EndpointResponse.json_response(400, {"error": str(exc)})
    if raw is None:
        return EndpointResponse.json_response(404, {"error": "no such release", "id": release_id})
    return EndpointResponse(200, CONTENT_TYPE_BUNDLE, raw)


def handle_contribute(store: OasisStore, raw_body: bytes, *, source: str) -> EndpointResponse:
    """``POST /v1/substrate/contribute`` → land it in the experimental tier.

    Returns 200 with the accept/duplicate receipt, or 400 if the body is not a
    valid substrate bundle. It never promotes to the release tier — that is a
    separate gated operation, never a side effect of receipt.
    """
    if not raw_body:
        return EndpointResponse.json_response(400, {"error": "empty contribution body"})
    try:
        receipt = store.accept_contribution(raw_body, source=source)
    except OasisStoreError as exc:
        return EndpointResponse.json_response(400, {"error": str(exc)})
    return EndpointResponse.json_response(200, receipt)
