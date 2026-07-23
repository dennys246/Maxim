"""Declarative per-modality substrate channels (the extero/intero seam).

The substrate is already N-modality-ready — :class:`~maxim.similarity.ec.
EntorhinalCortex` scans within-modality only, ``encode_sensors`` takes a
``modality=`` tag with a per-``(agent_id, modality)`` delta stash, and
``"audio"`` is already a frozen-centroid modality. What was missing was the
CALLER honoring the seam: ``propose_via_substrate`` merged exteroceptive
sensors (azimuth) into the interoception encode, so direction was one term in
a text-embed sum dominated by the drives and left/right collapsed onto one EC
cluster (docs/plans/exteroception_interoception_seam.md — the root cause of
the embodied cradle orient sim measuring at chance).

:class:`ModalityChannel` is the declarative unit of that seam: one channel =
one sensor stream = one ``encode_sensors(modality=tag)`` call = one entry in
the ``{modality: cluster_id}`` set that flows through ``recommend_action``
(additive bias sum) and ``record_outcome`` (credit routing). Adding a future
modality (vision, touch) is one tuple entry at the registry site in
``runtime/agent_loop.py`` — NOT a new class, ABC, or percept manifest; the
modality string tag is the extensibility seam.

Bio framing: per-modality thalamic relays (LGN/MGN) → within-modality maps →
late convergence at association cortex (NAc's additive sum). Interoception
(hypothalamus/drives) is represented apart and only converges at selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# The interoceptive channel tag. Interoception is special-cased by consumers
# by NAME: it feeds ``current_drives`` (the drive-affinity heuristic), it is
# the legacy single-cluster alias (``current_cluster_id`` /
# ``LLMProposal.cluster_id``), and it is the ONLY channel generic
# tool-success credit may write to (the write-side complement of
# de-dilution — see tool_dispatch.record_outcome).
INTEROCEPTION_TAG = "interoception"


@dataclass(frozen=True)
class ModalityChannel:
    """One per-modality sensor stream feeding the substrate encode.

    Runtime-ephemeral configuration (never persisted, never crosses a wire) —
    out of scope for the CC3 frozen-dataclass forward-compat audit.

    Attributes:
        tag: EC substrate modality tag (``"interoception"``, ``"audio"``, …).
            Must be non-empty; it namespaces the EC cluster space, the
            encoder delta stash, and the ``{modality: cluster_id}`` set.
        read_values: ``(executor) -> {sensor_name: float}`` snapshot of the
            channel's sensors. An empty dict means the channel is inactive
            for this body — no encode call is made for it.
        read_ranges: ``(executor) -> {sensor_name: (lo, hi)}`` declared value
            ranges so signed sensors normalize monotonically (P1). Sensors
            absent from the mapping fall back to the legacy [0,1] map.
    """

    tag: str
    read_values: Callable[[Any], "dict[str, float]"]
    read_ranges: Callable[[Any], "dict[str, tuple[float, float]]"]

    def __post_init__(self) -> None:
        if not self.tag:
            raise ValueError("ModalityChannel.tag must be a non-empty modality string")
