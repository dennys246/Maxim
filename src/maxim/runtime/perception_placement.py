"""Per-stage placement for the perception pipeline.

The perception pipeline is an ordered sequence of stages (capture → DSP →
segmentation → sensor-encode → substrate → cognition) with sharply
different resource demands. *Where each stage runs* is a placement
decision per stage — the capability/placement orthogonality that
``runtime/worker_pool.py`` established for LLM lanes, extended to
perception.

This module is the perception **sibling** of lane placement. It
deliberately **borrows the type/config idioms** of
:class:`maxim.runtime.worker_pool.ProviderPlacement` (frozen value type,
CC3 ``extra`` escape hatch with ``hash=False, compare=False``,
``__post_init__`` collision guard, coherence validated at the *producer
boundary* rather than in ``__post_init__``) but **NOT its runtime
composition**: a lane placement tuple is a *failover list* (first healthy
provider wins, compiled onto ``LLMRouter.provider_priority``); a
perception pipeline is a *sequence of stages run across nodes*. There is
no failover and no router to compile onto, so this is a typed sibling,
not a shared runtime.

This module ships the value type + target enum + coherence validator
(commit 1). The stage DAG and the pinned/placeable model live alongside
it in a follow-up (commit 2); pinned-ness is a property of a *stage*, not
of a placement, so it is intentionally absent here.

See ``docs/plans/perception_pipeline_placement.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageOrigin(str, Enum):
    """Where a perception stage runs (the placement target axis).

    Three of the four values are **symbolic roles** resolved to a concrete
    node at pipeline-construction time; ``NODE`` names a concrete mesh node
    directly (via :attr:`PerceptionStagePlacement.node`).

    - ``SELF`` — this process / node (the default for an all-local
      pipeline; the self-contained Reachy case where one node runs the
      whole pipeline).
    - ``SENSOR`` — the node physically holding the sensor. Stages pinned
      here by physics (raw high-rate capture, sub-millisecond ITD/TDOA
      DSP) cannot run anywhere else; the raw stream lives at the sensor.
    - ``SUBSTRATE_OWNER`` — the single substrate-owning node (the node
      that owns EC/ATL and cognition). In the self-contained Reachy case
      this resolves to the Reachy itself — ``SUBSTRATE_OWNER`` does NOT
      mean "the leader", it means "wherever the single substrate owner
      is".
    - ``NODE`` — a concrete mesh node addressed by name (the *placeable*
      case: move GPU-heavy segmentation to a node that has a GPU). The
      node name lives in :attr:`PerceptionStagePlacement.node`.

    ``str`` subclass so a value serializes to its plain string
    (``"node"``) for JSON persistence and compares equal to its own value
    (``StageOrigin.NODE == "node"``). ``__str__`` is overridden to return
    the bare value so ``f"{origin}"`` yields ``"node"`` rather than
    ``"StageOrigin.NODE"`` (mirrors ``worker_pool.Origin``, closing the
    same f-string trap).
    """

    SELF = "self"
    SENSOR = "sensor"
    SUBSTRATE_OWNER = "substrate_owner"
    NODE = "node"

    def __str__(self) -> str:
        return self.value


# Declared (non-``extra``) field names — used by
# PerceptionStagePlacement.__post_init__ to reject colliding ``extra``
# keys per the CC3 path-(a) rule (mirrors _PROVIDER_PLACEMENT_DECLARED_FIELDS).
_PERCEPTION_PLACEMENT_DECLARED_FIELDS: frozenset[str] = frozenset({"stage", "origin", "node"})


@dataclass(frozen=True)
class PerceptionStagePlacement:
    """One perception pipeline stage and the node it runs on.

    ESCAPE-HATCH at 1.0 (CC3) — path (a). ``extra`` carries genuinely
    additive, JSON-serializable metadata placement is likely to grow
    (e.g. ``routing_weight``, hardware hints); the ``hash=False,
    compare=False`` spec is load-bearing so the ``dict`` field doesn't
    break ``__hash__`` / inflate ``__eq__`` on this frozen type. Producers
    prefer declared fields; ``extra`` values must be JSON-serializable.

    Per the CC3 path-(a) rule, ``__post_init__`` rejects ``extra`` keys
    that collide with declared field names (mirrors ``ProviderPlacement``)
    so a colliding key cannot silently shadow a declared field on a future
    round-trip.

    BOUNDARY (commit 1): this is a permissive runtime value type. Coherence
    (``NODE`` requires a ``node``; symbolic origins forbid a ``node``) is
    enforced by :func:`validate_perception_placement_coherence` at the
    config/CLI producer boundary — NOT in ``__post_init__`` — so
    internal/test code can build partial placements and derived placements
    are coherent by construction.

    ``pinned`` is intentionally NOT a field here: pinned-ness is a property
    of a *stage* in the pipeline DAG (commit 2), not of an individual
    placement value. A placement answers "where does this stage run"; the
    DAG answers "may this stage's placement be overridden".
    """

    stage: str
    origin: StageOrigin
    node: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        collisions = _PERCEPTION_PLACEMENT_DECLARED_FIELDS & set(self.extra.keys())
        if collisions:
            raise ValueError(
                f"PerceptionStagePlacement: extra dict contains key(s) that "
                f"collide with declared fields: {sorted(collisions)}. extra is "
                f"for forward-growth additive metadata only — declared fields "
                f"go in their own slots."
            )


def validate_perception_placement_coherence(p: PerceptionStagePlacement, *, where: str = "placement") -> None:
    """Raise ``ValueError`` if a placement entry can't resolve to a node.

    Coherence rules — what each origin minimally needs:

    - ``NODE`` requires a non-empty ``node`` (the concrete mesh node name).
    - ``SELF`` / ``SENSOR`` / ``SUBSTRATE_OWNER`` are symbolic roles
      resolved at construction time; naming a ``node`` for one is
      incoherent (it would silently shadow the role) and is rejected.

    Called at the **config-loader / CLI boundary** where untrusted
    (operator) placements enter — NOT inside ``__post_init__``: the
    runtime resolution path always emits coherent placements, and keeping
    the dataclass permissive lets internal/test code build partial values.
    ``where`` is a caller-supplied label for the message (e.g.
    ``"config.json: perception.audio.stages[0]"``).

    Per Q3 of the plan, this fails loud at the producer boundary rather
    than warning-and-ignoring, so a malformed config cannot silently route
    a stage to the wrong place.
    """
    if p.origin is StageOrigin.NODE:
        if not p.node:
            raise ValueError(f"{where}: a 'node' placement requires a non-empty 'node' (the concrete mesh node name).")
    elif p.node is not None:
        raise ValueError(
            f"{where}: a symbolic placement ('{p.origin}') must not name a "
            f"'node' — symbolic roles resolve to a node at construction time. "
            f"Use origin='node' to address a concrete node by name."
        )
