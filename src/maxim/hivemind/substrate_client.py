"""Substrate-exchange client — the peer-side transport for bundle pull/contribute.

1.2 P2P Slice B. The client half of the substrate endpoints served by an Oasis
(:mod:`maxim.hivemind.oasis_endpoints`). It lives in its own file next to the
bundle domain, NOT on ``_MaximPeerBackend`` — per the typed-peer-transports
invariant, the LLM-inference transport's "exactly one HTTP call, router-handles-
retry" rules are LLM-specific, and a dropped bundle fetch is not a failover
event. What it shares with that transport is the playbook: a single-purpose
surface, a typed error with a ``fix_hint``, and no internal retry.

All calls go through ``maxim.utils.http`` (the sanctioned client): ``fetch_url``
for JSON (releases list, contribute receipt) and ``download_to_file`` for the
streamed bundle bytes. Auth is the same Bearer token the Oasis server checks.

Consumed by the ``maxim hive`` CLI (Slice C); ``ingest_bundle`` (Slice A/D) is
what a puller runs on a fetched release before merging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from maxim.utils import http

logger = logging.getLogger(__name__)

_ZIP_CONTENT_TYPE = "application/zip"


class SubstrateExchangeError(Exception):
    """A substrate-exchange call failed at the protocol level (not transport).

    Transport failures (connection, timeout, auth, 5xx) surface as the typed
    ``maxim.utils.http.HTTPError`` subclasses with their own ``.fix_hint``; this
    is for the protocol-level outcomes the Oasis reports in a JSON body (a
    refused contribution, a malformed response).
    """

    def __init__(self, message: str, *, fix_hint: str = "") -> None:
        super().__init__(message)
        self.fix_hint = fix_hint


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _substrate_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover — best-effort cleanup
        logger.warning("could not remove partial download %s: %s", path, exc)


def list_releases(base_url: str, *, api_key: str | None = None, timeout: float | None = None) -> list[dict[str, Any]]:
    """Fetch the Oasis's published Queen-tier release summaries."""
    resp = http.fetch_url(
        _substrate_url(base_url, "/v1/substrate/releases"),
        headers=_auth_headers(api_key),
        timeout=timeout,
    )
    try:
        payload = resp.json()
    except ValueError as exc:
        raise SubstrateExchangeError(f"releases response was not JSON: {exc}") from exc
    releases = payload.get("releases") if isinstance(payload, dict) else None
    if not isinstance(releases, list):
        raise SubstrateExchangeError("releases response missing a 'releases' list")
    return releases


def fetch_bundle(
    base_url: str,
    release_id: str,
    dest_path: str | Path,
    *,
    api_key: str | None = None,
    timeout: float | None = None,
) -> Path:
    """Stream one signed release bundle to ``dest_path``; return the path.

    Raises :class:`SubstrateExchangeError` if the release is unknown (404) and
    the typed ``HTTPError`` subclasses for transport failures. The caller then
    runs ``ingest_bundle(require_signed=True, trusted_keys=...)`` to verify the
    signature before merging — fetching is not trusting.
    """
    dest = Path(dest_path)
    try:
        http.download_to_file(
            _substrate_url(base_url, f"/v1/substrate/bundle/{release_id}"),
            dest,
            headers=_auth_headers(api_key),
            timeout=timeout,
        )
    except http.HTTPClientError as exc:
        # 4xx — a 404 means the id is unknown; other 4xx (e.g. a 400 from an
        # invalid id shape) are a caller error worth naming distinctly. Nothing
        # is written on a 4xx (download_to_file's status check precedes the file
        # open), so there is no partial file to clean up here.
        if exc.status == 404:
            raise SubstrateExchangeError(
                f"release {release_id} not available (unknown id)",
                fix_hint="check `maxim hive pull` listed this id from this Oasis",
            ) from exc
        raise SubstrateExchangeError(f"release {release_id} refused ({exc.status})") from exc
    except http.HTTPError:
        # A mid-stream transport failure (timeout / connection drop) can leave a
        # truncated file at dest — remove it so a caller never mistakes a partial
        # download for a whole bundle (ingest would reject it, but no stale blob
        # should linger). Re-raise the typed transport error unchanged.
        _remove_if_present(dest)
        raise
    return dest


def contribute(
    base_url: str,
    bundle_path: str | Path,
    *,
    api_key: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """POST a bundle to the Oasis's experimental tier; return the accept receipt.

    A contribution lands in the experimental tier only — it is never promoted to
    a signed release by the act of contributing (that is a gated Oasis-side
    operation). The receipt carries the content digest and ``status``
    (``accepted`` / ``duplicate``).
    """
    raw = Path(bundle_path).read_bytes()
    headers = {**_auth_headers(api_key), "Content-Type": _ZIP_CONTENT_TYPE}
    try:
        resp = http.fetch_url(
            _substrate_url(base_url, "/v1/substrate/contribute"),
            method="POST",
            headers=headers,
            content=raw,
            timeout=timeout,
        )
    except http.HTTPClientError as exc:
        # The Oasis refuses a malformed contribution with 400 + a JSON error
        # body; fetch_url raises before returning it. Surface it as the
        # protocol-level error this module documents, carrying the server's
        # reason so the caller sees WHY it was refused (not a bare HTTP code).
        raise SubstrateExchangeError(
            f"contribution refused: {_server_error(exc)}",
            fix_hint="the Oasis rejected the bundle as invalid",
        ) from exc
    try:
        receipt = resp.json()
    except ValueError as exc:
        raise SubstrateExchangeError(f"contribute response was not JSON: {exc}") from exc
    if not isinstance(receipt, dict) or "digest" not in receipt:
        raise SubstrateExchangeError("contribute response missing a digest receipt")
    return receipt


def _server_error(exc: http.HTTPError) -> str:
    """Best-effort extraction of an Oasis JSON ``{"error": ...}`` body from a 4xx."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = resp.json()
            if isinstance(body, dict) and body.get("error"):
                return str(body["error"])
        except ValueError:
            pass
    return f"HTTP {getattr(exc, 'status', '4xx')}"
