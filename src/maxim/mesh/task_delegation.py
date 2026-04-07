"""Task delegation — delegate goals to peers and evaluate incoming proposals.

TaskDelegator picks the best peer for a goal, sends a GOAL_PROPOSAL,
and blocks until a response arrives (or timeout). TaskReceiver evaluates
incoming proposals using local NAc predictions and queue depth limits.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from uuid import uuid4

from maxim.mesh.message import MeshMessageType
from maxim.mesh.peer_info import PeerInfo

logger = logging.getLogger(__name__)


class TaskDelegator:
    """Delegate goals to peer agents based on capabilities.

    The sender picks the best peer for a goal based on:
    1. Does the peer have the required tool?
    2. Does the peer's NAc predict success for this tool?
    3. Is the peer alive and not gated?

    Uses threading.Event for synchronous waiting — the agent loop is sync.
    """

    def __init__(
        self,
        peer_registry: Any,
        peer_channel: Any,
        local_agent_id: str,
        *,
        admission: Any | None = None,
    ) -> None:
        self._peers = peer_registry
        self._channel = peer_channel
        self._local_agent_id = local_agent_id
        self._admission = admission
        self._pending: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def find_best_peer(self, goal: dict[str, Any]) -> PeerInfo | None:
        """Find the peer best suited for a goal.

        Skips dead peers, peers without identity, and gated peers.
        Ranks by tool success rate from identity broadcast.
        """
        required_tool = goal.get("tool_name")
        if not required_tool:
            return None

        candidates: list[tuple[PeerInfo, float]] = []
        for peer in self._peers.all_peers():
            if not peer.is_alive:
                continue
            if self._admission and self._admission.is_peer_gated(peer.peer_id):
                continue
            identity = peer.identity
            if identity is None:
                continue
            if required_tool not in identity.available_tools:
                continue

            # Look up success rate from identity broadcast
            success_rate = 0.5
            for t in identity.top_tools:
                if t.get("name") == required_tool:
                    success_rate = t.get("success_rate", 0.5)
                    break
            candidates.append((peer, success_rate))

        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    def delegate(
        self,
        goal: dict[str, Any],
        peer: PeerInfo,
        timeout_s: float = 60.0,
    ) -> dict[str, Any] | None:
        """Send a goal to a peer and block until result or timeout.

        Injects delegation_depth and delegation_chain for loop detection.
        Returns the result payload dict, or None on timeout/failure.
        """
        correlation_id = uuid4().hex[:12]

        # Inject delegation tracking
        goal = dict(goal)  # Don't mutate caller's dict
        goal.setdefault("delegation_depth", 0)
        goal["delegation_depth"] += 1
        goal.setdefault("delegation_chain", [])
        goal["delegation_chain"] = [*goal["delegation_chain"], self._local_agent_id]
        goal.setdefault("timeout_s", timeout_s)

        event = threading.Event()
        with self._lock:
            self._pending[correlation_id] = {"event": event, "result": None}

        success = self._channel.send_mesh(
            recipient=peer.peer_id,
            payload=goal,
            msg_type=MeshMessageType.GOAL_PROPOSAL,
            correlation_id=correlation_id,
        )
        if not success:
            with self._lock:
                self._pending.pop(correlation_id, None)
            return None

        # Block until response or timeout
        event.wait(timeout=timeout_s)

        with self._lock:
            pending = self._pending.pop(correlation_id, {})
        return pending.get("result")

    def handle_response(self, correlation_id: str, payload: dict[str, Any]) -> None:
        """Called when a GOAL_RESULT/GOAL_ACCEPTED/GOAL_REJECTED arrives.

        Wakes the blocked delegate() call for this correlation_id.
        """
        with self._lock:
            pending = self._pending.get(correlation_id)
            if pending is None:
                return
            pending["result"] = payload
            pending["event"].set()

    @property
    def pending_count(self) -> int:
        """Number of delegations waiting for responses."""
        with self._lock:
            return len(self._pending)


class TaskReceiver:
    """Evaluates and optionally executes delegated goals from peers.

    Uses local NAc predictions and queue depth to decide whether to
    accept a proposal. Delegation loop detection via depth and chain.
    """

    MAX_DELEGATION_QUEUE = 5
    MAX_DELEGATION_DEPTH = 2

    def __init__(
        self,
        local_agent_id: str,
        *,
        executor: Any | None = None,
        nac: Any | None = None,
    ) -> None:
        self._local_agent_id = local_agent_id
        self._executor = executor
        self._nac = nac
        self._active_delegations = 0
        self._lock = threading.Lock()

    def evaluate_proposal(
        self,
        goal: dict[str, Any],
        sender_id: str,
    ) -> tuple[bool, str]:
        """Decide whether to accept a delegated goal.

        Checks (in order):
        1. Delegation loop detection (depth > 2 or cycle)
        2. Queue depth (hard cap on concurrent delegations)
        3. Tool availability (hard requirement)
        4. NAc prediction (soft — reject if high-confidence negative)
        """
        # 1. Loop detection
        depth = goal.get("delegation_depth", 0)
        chain = goal.get("delegation_chain", [])
        if depth > self.MAX_DELEGATION_DEPTH:
            return False, f"delegation too deep (depth={depth})"
        if self._local_agent_id in chain:
            return False, "delegation cycle detected"

        # 2. Queue depth
        with self._lock:
            if self._active_delegations >= self.MAX_DELEGATION_QUEUE:
                return False, f"overloaded ({self._active_delegations} active)"

        # 3. Tool availability
        tool_name = goal.get("tool_name")
        if not tool_name:
            return False, "no tool specified"
        if self._executor is not None:
            try:
                self._executor.registry.get(tool_name)
            except (KeyError, AttributeError):
                return False, f"tool {tool_name} not available"

        # 4. NAc prediction
        if self._nac is not None:
            try:
                prediction = self._nac.predict("tool", f"tool:{tool_name}")
                if prediction and prediction.predicted_valence.value == "negative" and prediction.confidence > 0.7:
                    return False, f"NAc predicts failure (confidence={prediction.confidence:.2f})"
            except Exception:
                pass  # NAc errors are non-fatal

        return True, "accepted"

    def begin_delegation(self) -> None:
        """Mark a delegation as started (increments active count)."""
        with self._lock:
            self._active_delegations += 1

    def end_delegation(self) -> None:
        """Mark a delegation as completed (decrements active count)."""
        with self._lock:
            self._active_delegations = max(0, self._active_delegations - 1)

    @property
    def active_delegations(self) -> int:
        with self._lock:
            return self._active_delegations
