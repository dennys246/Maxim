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
(commit 1) and the stage DAG + pinned/placeable model + resolver
(commit 2). Pinned-ness is a property of a *stage* (``PerceptionStage``),
not of an individual placement value.

See ``docs/plans/perception_pipeline_placement.md``.

Dormant since 2026-08-07: zero production callers — the type layer merged
(PRs #382-#385 era) but nothing in ``src/maxim/`` ever constructs or
resolves a perception placement; only its unit tests exercise it. Found by
the 1.1 four-lens roadmap review ("claimed shipped" vs reality) and marked
per CLAUDE.md Principle 2 (dormancy over deletion): tests stay as the
regression floor, no new features build on top. Resurrection trigger: the
1.3 cross-modal perception fabric actually placing pipeline stages across
nodes (its plan self-targets exactly this seam), validated by an
experiment, not just re-wired.
"""

from __future__ import annotations

from collections.abc import Mapping
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


# ---------------------------------------------------------------------------
# Stage DAG + pinned/placeable model (commit 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionStage:
    """One stage in the perception pipeline DAG.

    ``pinned`` is the genuinely new concept lane placement did not have:
    a stage pinned by physics (raw capture + sub-ms DSP at the sensor;
    substrate/cognition at the single substrate owner) cannot have its
    placement overridden by config. ``placeable`` stages (the GPU-heavy
    movable middle — segmentation, sensor-encode) can.

    ``default_origin`` is the placement a stage takes when nothing
    overrides it. It is always a **symbolic** role (never ``NODE``):
    ``PerceptionStage`` carries no ``node`` field because a default cannot
    hardcode a concrete node name — the symbolic role resolves to a
    concrete node at construction time. ``__post_init__`` enforces this.

    SHAPE-FROZEN at 1.0 (CC3). The DAG is defined in code (not operator-
    authored), so the escape-hatch ``extra`` dict is deliberately rejected:
    a stage definition is a fixed contract, and an ``extra`` dict would
    invite per-stage logic the resolver could not honour. Adding a field
    post-1.0 is a review-gated change.
    """

    name: str
    pinned: bool
    default_origin: StageOrigin

    def __post_init__(self) -> None:
        if self.default_origin is StageOrigin.NODE:
            raise ValueError(
                f"PerceptionStage {self.name!r}: default_origin must be a "
                f"symbolic role (self/sensor/substrate_owner), not 'node' — a "
                f"default cannot hardcode a concrete node (no node field on a "
                f"stage). Concrete-node placement comes from a config override."
            )


# The canonical perception pipeline. Order is the data-flow order; pinned
# stages are fixed by physics (see the stage model in
# docs/plans/perception_pipeline_placement.md). An unconfigured pipeline
# resolves every stage to its default_origin — which is all-symbolic and,
# in the self-contained single-node case, all-local: byte-identical to
# today's behaviour (no distribution).
CANONICAL_PERCEPTION_PIPELINE: tuple[PerceptionStage, ...] = (
    PerceptionStage("capture", pinned=True, default_origin=StageOrigin.SENSOR),
    PerceptionStage("dsp", pinned=True, default_origin=StageOrigin.SENSOR),
    PerceptionStage("segmentation", pinned=False, default_origin=StageOrigin.SELF),
    PerceptionStage("sensor_encode", pinned=False, default_origin=StageOrigin.SELF),
    PerceptionStage("substrate", pinned=True, default_origin=StageOrigin.SUBSTRATE_OWNER),
    PerceptionStage("cognition", pinned=True, default_origin=StageOrigin.SUBSTRATE_OWNER),
)


def resolve_pipeline_placements(
    dag: tuple[PerceptionStage, ...] = CANONICAL_PERCEPTION_PIPELINE,
    overrides: Mapping[str, PerceptionStagePlacement] | None = None,
    *,
    where: str = "perception pipeline",
) -> tuple[PerceptionStagePlacement, ...]:
    """Resolve a stage DAG (+ optional config overrides) to ordered placements.

    Mirrors the spirit of ``lane_backends.derive_placement``'s
    empty-means-legacy back-compat: with **no overrides**, every stage
    resolves to its ``default_origin`` — the all-local, no-distribution
    behaviour that is byte-identical to today. Overrides (keyed by stage
    name) move *placeable* stages only.

    Fails loud at this producer boundary (Q3) on:

    - an override naming an unknown stage,
    - an override targeting a **pinned** stage (placement is fixed by
      physics — config cannot move it),
    - an override whose ``stage`` field disagrees with its dict key,
    - an incoherent override (delegated to
      :func:`validate_perception_placement_coherence`).
    """
    overrides = overrides or {}
    known = {s.name for s in dag}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(
            f"{where}: override(s) for unknown stage(s) {sorted(unknown)}; known stages are {sorted(known)}."
        )

    resolved: list[PerceptionStagePlacement] = []
    for stage in dag:
        override = overrides.get(stage.name)
        if override is None:
            resolved.append(PerceptionStagePlacement(stage=stage.name, origin=stage.default_origin))
            continue
        if stage.pinned:
            raise ValueError(
                f"{where}: stage {stage.name!r} is pinned by physics and cannot "
                f"be placed by config (attempted origin '{override.origin}'). "
                f"Only placeable stages may be overridden."
            )
        if override.stage != stage.name:
            raise ValueError(
                f"{where}: override for key {stage.name!r} carries mismatched stage field {override.stage!r}."
            )
        validate_perception_placement_coherence(override, where=f"{where}: {stage.name}")
        resolved.append(override)
    return tuple(resolved)
