"""Shared probe-outcome → (status, message, fix_hint) classifier.

Plan 4 Stage C1 pre-merge review round 2 extracted this module out of
``peer/mesh_config.py``. Zero imports from the parser / schema layer:
callers that want the classifier without the parser (test harnesses,
the future C3 admin-API dashboard) can depend on this module alone.

**Load-bearing invariants:**

1. **Single source of truth.** Every caller that probes a peer URL via
   :meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.health_check`
   routes its classification through :func:`classify_probe_outcome`.
   Current callers:

   - :func:`maxim.peer.mesh_cli._probe_node` (``maxim peer list-nodes``)
   - :func:`maxim.doctor.checks._probe_mesh_node_to_check` (``maxim doctor``)

   Future C2 ``--node install`` and C3 admin-API dashboard code MUST
   call this helper rather than re-implementing the mapping inline.

2. **Specific-before-general ordering** (Plan 2 R2c). ``auth_rejected``
   and ``inference_broken`` classify first; everything else falls to the
   generic network-down bucket. Inverting the order silently re-classifies
   auth errors as "inference broken".

3. **Callers do NOT override the returned message/fix.** Round 2 review
   (A1R2) flagged the "classifier returns X, caller rewrites to Y"
   pattern as a leak — the shared-helper promise requires that all
   presentation decisions happen inside the helper. If your caller
   wants a richer message, pass a richer ``detail`` string through and
   let the classifier compose the output. The ``detail`` parameter is
   the knob for caller-specific context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class ProbeClassification:
    """Structured probe-classification result.

    A frozen dataclass rather than a primitive tuple so C2/C3 can
    add fields (``severity``, ``retry_recommended``, ``cooldown_s``)
    without touching every call site.
    """

    status: Status
    message: str
    fix: str | None


def classify_probe_outcome(
    outcome: str,
    detail: str,
    latency_ms: float | None,
    url: str,
) -> ProbeClassification:
    """Map a ``ProbeResult.outcome`` literal to a :class:`ProbeClassification`.

    Parameters
    ----------
    outcome
        The ``ProbeResult.outcome`` literal from
        :meth:`_MaximPeerBackend.health_check`. Known values:
        ``ok``, ``auth_rejected``, ``inference_broken``, ``timeout``,
        ``connection_refused``, ``dns_fail``, ``tls_error``,
        ``http_5xx``, ``other``. Unknown values fall through to a
        ``warn`` with ``unknown outcome`` in the message.
    detail
        Caller-provided context string. Callers that want a richer
        "reachable" message (e.g., doctor wants role + URL) pass that
        context here instead of post-processing the returned message.
    latency_ms
        Optional probe latency in milliseconds. Rendered as
        ``[Xms]`` suffix when present.
    url
        The probed URL. Used in warn/fail messages + the generic
        "curl to diagnose" fix hint.
    """
    latency_tag = f" [{latency_ms:.0f}ms]" if latency_ms is not None else ""

    if outcome == "ok":
        return ProbeClassification(
            status="ok",
            message=f"reachable ({detail}){latency_tag}",
            fix=None,
        )
    if outcome == "auth_rejected":
        # Key-drift detection (post-config-unification UX fold,
        # 2026-06-03): the canonical 401 path on a peer-leader setup
        # is "your stored key doesn't match the leader's current key,"
        # not a generic "auth failed." Surface the rotate-and-re-paste
        # path as primary because the operator's stored key is the
        # thing they can fix; the leader's key only needs rotating if
        # THEY've decided to rotate. The mesh.yml::cluster_key path
        # stays as a secondary hint for cluster setups.
        #
        # Larger drift-detection design (option 3 from the discussion)
        # is tracked in docs/plans/deferred/key_drift_detection.md — that
        # follow-up adds a /v1/admin/key-fingerprint endpoint so peers
        # can detect drift PROACTIVELY without ever waiting for a 401.
        return ProbeClassification(
            status="fail",
            message=f"auth rejected ({url}): {detail} — your stored API key may be stale (leader's current key differs from yours)",
            fix=(
                "Your stored API key likely doesn't match the leader's current key.\n"
                "On the leader, print the current key (or rotate if you want a fresh one):\n"
                "  maxim tunnel key show       # canonical case: leader rotated, peer is stale\n"
                "  maxim tunnel key rotate     # security case: invalidate every peer + start fresh\n"
                "Then re-pair this peer with the fresh key:\n"
                "  maxim peer connect <leader-url>   # paste the key at the prompt\n"
                "(For mesh.yml-based cluster setups: update mesh.yml::cluster_key on this node instead.)"
            ),
        )
    if outcome == "inference_broken":
        return ProbeClassification(
            status="fail",
            message=f"chat endpoint broken ({url}): {detail}",
            fix=(
                "Leader is reachable but chat/completions is failing. Check:\n"
                "  1. A model is loaded: maxim peer llm --status\n"
                "  2. Leader logs: maxim peer logs\n"
                "  3. VRAM pressure: maxim doctor --as leader"
            ),
        )
    # All remaining network-layer outcomes (timeout / connection_refused
    # / dns_fail / tls_error / http_5xx / other) get the same generic
    # "is it reachable?" fix.
    known_network = ("timeout", "connection_refused", "dns_fail", "tls_error", "http_5xx", "other")
    if outcome in known_network:
        return ProbeClassification(
            status="warn",
            message=f"{outcome.replace('_', ' ')} ({url}): {detail}",
            fix=f"Is {url} reachable from here? Try: curl -v {url}/models",
        )
    return ProbeClassification(
        status="warn",
        message=f"unknown outcome {outcome!r} ({url}): {detail}",
        fix=None,
    )


__all__ = ["ProbeClassification", "Status", "classify_probe_outcome"]
