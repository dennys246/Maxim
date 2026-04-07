"""PeerRegistry — registry of known mesh peers.

Initially populated from peer/config.py (tunnel URLs) and manual
registration. Phase 0a adds mDNS as an additional discovery source.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from maxim.mesh.peer_info import PeerInfo

logger = logging.getLogger(__name__)


class PeerRegistry:
    """Thread-safe registry of known mesh peers.

    Peers are keyed by peer_id. The registry can be bootstrapped from
    the existing peer config file (tunnel URL + API key) or populated
    manually / via mDNS discovery (Phase 0a).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._peers: dict[str, PeerInfo] = {}

    def register(
        self,
        peer_id: str,
        base_url: str,
        api_key: str = "",
        trust_level: str = "unknown",
    ) -> PeerInfo:
        """Register or update a peer. Returns the PeerInfo."""
        with self._lock:
            existing = self._peers.get(peer_id)
            if existing is not None:
                existing.base_url = base_url
                if api_key:
                    existing.api_key = api_key
                existing.trust_level = trust_level
                existing.touch()
                return existing
            info = PeerInfo(
                peer_id=peer_id,
                base_url=base_url,
                api_key=api_key,
                trust_level=trust_level,
            )
            self._peers[peer_id] = info
            logger.info("mesh: registered peer %s at %s (trust=%s)", peer_id, base_url, trust_level)
            return info

    def unregister(self, peer_id: str) -> bool:
        """Remove a peer. Returns True if it was present."""
        with self._lock:
            return self._peers.pop(peer_id, None) is not None

    def get_peer(self, peer_id: str) -> PeerInfo | None:
        """Look up a peer by ID."""
        with self._lock:
            return self._peers.get(peer_id)

    def all_peers(self) -> list[PeerInfo]:
        """Return a snapshot of all registered peers."""
        with self._lock:
            return list(self._peers.values())

    def alive_peers(self) -> list[PeerInfo]:
        """Return peers that are still alive (within heartbeat timeout)."""
        with self._lock:
            return [p for p in self._peers.values() if p.check_alive()]

    def peer_count(self) -> int:
        with self._lock:
            return len(self._peers)

    @classmethod
    def from_peer_config(cls) -> PeerRegistry:
        """Bootstrap a registry from the existing peer/config.py file.

        Uses the tunnel URL and API key. The peer_id is derived from the URL
        (since we don't yet know the remote's node_id — it will be updated
        on the first HEARTBEAT exchange).
        """
        registry = cls()
        try:
            from maxim.peer.config import read_peer_config

            cfg = read_peer_config()
            if cfg is None:
                return registry

            # Derive a stable peer_id from the URL
            peer_id = _url_to_peer_id(cfg.url)
            trust = "remote" if cfg.is_cloud else "verified"
            # Ensure URL ends with /v1 for mesh endpoint mounting
            base_url = cfg.url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1" if not base_url.endswith("/") else "v1"

            registry.register(
                peer_id=peer_id,
                base_url=base_url,
                api_key=cfg.api_key,
                trust_level=trust,
            )
            logger.info("mesh: bootstrapped peer %s from peer config", peer_id)
        except Exception:
            logger.debug("mesh: no peer config found, starting with empty registry")
        return registry

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize all peers (for /v1/mesh/peers endpoint)."""
        with self._lock:
            return [p.to_dict() for p in self._peers.values()]


def _url_to_peer_id(url: str) -> str:
    """Derive a stable short ID from a URL.

    Strips protocol and common suffixes, keeps the hostname.
    e.g. "https://maxim.example.com/v1" -> "maxim.example.com"
    """
    url = url.rstrip("/")
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix) :]
            break
    for suffix in ("/v1", "/v1/"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    # Remove port for common ports
    if url.endswith(":443") or url.endswith(":80"):
        url = url.rsplit(":", 1)[0]
    return url
