"""PeerChannel — network transport for mesh messages.

Implements the Channel interface so it plugs into CommunicationGateway.
Sends are queued to a background thread — callers never block on network I/O.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any
from uuid import uuid4

from maxim.comms.channels.base import Channel
from maxim.mesh.message import MESH_PROTOCOL_VERSION, MeshMessage, MeshMessageType
from maxim.mesh.peer_registry import PeerRegistry

logger = logging.getLogger(__name__)

# Sentinel object to signal the drain thread to stop
_STOP = object()


class PeerChannel(Channel):
    """Communication channel for peer-to-peer agent mesh.

    Plugs into CommunicationGateway using the existing Channel interface.
    Each peer is an HTTP endpoint. Mount under /v1/mesh/* to inherit
    LeaderProxy auth, rate limiting, and GPU metrics.

    Sends are queued — the caller never blocks on network I/O.
    """

    MAX_QUEUE_SIZE = 1000
    MAX_RETRIES = 3
    HTTP_TIMEOUT_S = 10.0

    def __init__(
        self,
        peer_registry: PeerRegistry,
        local_agent_id: str,
        *,
        bus: Any | None = None,
    ) -> None:
        self._peer_registry = peer_registry
        self._local_agent_id = local_agent_id
        self._bus = bus
        self._send_queue: queue.Queue[MeshMessage | object] = queue.Queue(
            maxsize=self.MAX_QUEUE_SIZE,
        )
        self._stopped = False
        self._sender_thread = threading.Thread(
            target=self._drain_queue,
            name="mesh-send",
            daemon=True,
        )
        self._sender_thread.start()
        # Stats
        self._messages_sent = 0
        self._messages_failed = 0
        self._messages_dropped = 0

    # ── Channel interface (CommunicationGateway compatible) ─────────────────

    def send(self, recipient: str, body: str) -> bool:
        """Send a mesh message to a peer (Channel interface).

        The body should be a JSON string. msg_type defaults to REQUEST.
        For mesh-specific sends with metadata, use send_mesh() instead.
        """
        return self.send_mesh(
            recipient=recipient,
            payload=json.loads(body) if body else {},
            msg_type=MeshMessageType.REQUEST,
        )

    # ── Mesh-specific send (with full control) ─────────────────────────────

    def send_mesh(
        self,
        recipient: str,
        payload: dict[str, Any],
        msg_type: MeshMessageType = MeshMessageType.REQUEST,
        correlation_id: str = "",
        in_reply_to: str | None = None,
    ) -> bool:
        """Enqueue a mesh message for async delivery to a peer.

        Returns True if enqueued, False if peer unknown or queue full.
        """
        if self._stopped:
            return False
        peer = self._peer_registry.get_peer(recipient)
        if peer is None:
            return False
        msg = MeshMessage(
            sender=self._local_agent_id,
            recipient=recipient,
            msg_type=msg_type,
            payload=payload,
            correlation_id=correlation_id or uuid4().hex[:12],
            in_reply_to=in_reply_to,
            protocol_version=MESH_PROTOCOL_VERSION,
        )
        try:
            self._send_queue.put_nowait(msg)
            return True
        except queue.Full:
            self._messages_dropped += 1
            logger.warning("mesh send queue full, dropping message to %s", recipient)
            return False

    # ── Inbound message handling ───────────────────────────────────────────

    def receive(self, msg: MeshMessage) -> None:
        """Handle an incoming mesh message (called by API endpoint).

        Publishes as a Percept on the agent bus so the agent pipeline
        sees peer messages the same way it sees sensor input.
        """
        if msg.protocol_version > MESH_PROTOCOL_VERSION:
            logger.warning(
                "rejecting mesh message with protocol_version=%d (local=%d)",
                msg.protocol_version,
                MESH_PROTOCOL_VERSION,
            )
            return

        if self._bus is None:
            logger.debug("mesh: received message but no bus attached, discarding")
            return

        from maxim.agents.modality import SensoryModality, SensoryTag
        from maxim.agents.percept_factory import make_text_percept

        percept = make_text_percept(
            json.dumps(msg.payload),
            source=f"mesh:{msg.sender}",
            salience=0.7,
            metadata={
                "msg_type": msg.msg_type.name,
                "sender": msg.sender,
                "correlation_id": msg.correlation_id,
                "external": True,
            },
            sensory=SensoryTag(modality=SensoryModality.ABSTRACT, submodality="mesh"),
        )
        self._bus.publish(percept)

        # Touch the peer in the registry so it stays alive
        peer = self._peer_registry.get_peer(msg.sender)
        if peer is not None:
            peer.touch()

    # ── Background send thread ─────────────────────────────────────────────

    def _drain_queue(self) -> None:
        """Background thread: drain send queue, POST to peers with retries."""
        while True:
            item = self._send_queue.get()
            if item is _STOP:
                break
            msg = item  # type: MeshMessage
            peer = self._peer_registry.get_peer(msg.recipient)
            if peer is None:
                self._messages_failed += 1
                continue

            success = False
            for attempt in range(self.MAX_RETRIES):
                try:
                    success = self._post_message(peer, msg)
                    if success:
                        break
                except Exception as exc:
                    logger.debug(
                        "mesh: send attempt %d/%d to %s failed: %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        peer.peer_id,
                        exc,
                    )
                # Exponential backoff: 1s, 2s, 4s (capped at 30s)
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(min(1 * 2**attempt, 30))

            if success:
                self._messages_sent += 1
            else:
                self._messages_failed += 1
                logger.warning(
                    "mesh: failed to send %s to %s after %d attempts",
                    msg.msg_type.name,
                    peer.peer_id,
                    self.MAX_RETRIES,
                )

    def _post_message(self, peer: Any, msg: MeshMessage) -> bool:
        """POST a message to a peer's mesh endpoint. Returns True on success."""
        try:
            import httpx
        except ImportError:
            logger.warning("mesh: httpx not installed, cannot send mesh messages")
            return False

        url = f"{peer.base_url}/v1/mesh/message"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if peer.api_key:
            headers["Authorization"] = f"Bearer {peer.api_key}"

        resp = httpx.post(
            url,
            json=msg.to_dict(),
            headers=headers,
            timeout=self.HTTP_TIMEOUT_S,
        )
        return resp.status_code == 200

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stopped = True
        try:
            self._send_queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._sender_thread.join(timeout=5.0)

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "messages_sent": self._messages_sent,
            "messages_failed": self._messages_failed,
            "messages_dropped": self._messages_dropped,
            "queue_depth": self._send_queue.qsize(),
            "stopped": self._stopped,
        }
