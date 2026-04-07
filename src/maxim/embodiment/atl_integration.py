"""ATL integration — auto-register body_part concepts from Entity trees.

When an entity tree loads, each entity registers an ATL ``body_part``
concept with:
- Sensor ranges → IPS statistics (min/max/expected)
- Modulator affordances → parameter space descriptions
- Failure modes → pain threshold associations

This provides the ATL grounding that makes LLM predictions consistent.
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.embodiment.sem import Entity

log = logging.getLogger(__name__)


def register_body_concepts(
    entity: Entity,
    atl: Any,
    *,
    category: str = "body_part",
) -> list[str]:
    """Register ATL concepts for every entity in the tree.

    Parameters
    ----------
    entity : Entity
        Root of the entity tree.
    atl : ATL
        The ATL instance to register concepts in.
    category : str
        Concept category to use (default ``"body_part"``).

    Returns
    -------
    list[str]
        List of created concept IDs.
    """
    concept_ids: list[str] = []

    for ent in entity.walk():
        concept_id = _register_entity_concept(ent, atl, category)
        if concept_id:
            concept_ids.append(concept_id)

    if concept_ids:
        # Register relationships between parent and child concepts
        _register_relationships(entity, atl)

    log.info(
        "Registered %d body concepts from entity tree '%s'",
        len(concept_ids),
        entity.name,
    )
    return concept_ids


def _register_entity_concept(
    entity: Entity,
    atl: Any,
    category: str,
) -> str | None:
    """Register a single entity as an ATL concept."""
    try:
        from maxim.memory.semantic_types import ConceptProvenance
    except ImportError:
        log.debug("ATL semantic types not available — skipping concept registration")
        return None

    # Build definition from entity metadata
    definition_parts: list[str] = [
        f"{entity.name} is a {entity.entity_type}.",
    ]

    # Sensor ranges
    for sname, sensor in entity.sensors.items():
        schema = sensor.reading_schema
        stype = schema.get("type", "unknown")
        if stype in ("float", "int") and "range" in schema:
            lo, hi = schema["range"]
            definition_parts.append(f"Sensor '{sname}': {sensor.unit}, range [{lo}, {hi}].")
        elif stype == "ndarray" and "shape" in schema:
            definition_parts.append(f"Sensor '{sname}': {sensor.unit}, shape {schema['shape']}.")

    # Affordances
    for mname, modulator in entity.modulators.items():
        for aff_name, aff_schema in modulator.affordances.items():
            param_desc = ", ".join(
                f"{p}: {t.__name__ if isinstance(t, type) else t}" for p, t in aff_schema.params.items()
            )
            desc = aff_schema.description or aff_name
            definition_parts.append(f"Affordance '{aff_name}' (via {mname}): {desc}. Params: ({param_desc}).")

    # Failure modes
    for fm in entity.failure_modes:
        trigger_desc = ", ".join(f"{t.field} {t.op} {t.value}" for t in fm.triggers)
        definition_parts.append(
            f"Failure '{fm.name}': triggers when {trigger_desc}. Pain intensity: {fm.pain_intensity}."
        )

    definition = " ".join(definition_parts)

    # Build properties dict
    properties: dict[str, Any] = {
        "entity_type": entity.entity_type,
        "entity_path": entity.full_path,
        "sensor_count": len(entity.sensors),
        "modulator_count": len(entity.modulators),
        "affordance_count": sum(len(m.affordances) for m in entity.modulators.values()),
    }

    # Add sensor ranges as properties
    for sname, sensor in entity.sensors.items():
        schema = sensor.reading_schema
        if "range" in schema:
            properties[f"sensor_{sname}_range"] = schema["range"]
        if "initial" in schema:
            properties[f"sensor_{sname}_initial"] = schema["initial"]

    try:
        concept_id, created = atl.find_or_create(
            name=entity.full_path,
            category=category,
            definition=definition,
            provenance=ConceptProvenance.DIRECT_INGESTION,
        )
        if created:
            log.debug("Created ATL concept: %s (%s)", entity.full_path, concept_id)
        return concept_id
    except Exception as e:
        log.warning("Failed to register ATL concept for %s: %s", entity.full_path, e)
        return None


def _register_relationships(entity: Entity, atl: Any) -> None:
    """Register parent-child relationships between entity concepts."""
    for ent in entity.walk():
        for child in ent.children:
            try:
                atl.define_relationship(
                    source_id=ent.full_path,
                    target_id=child.full_path,
                    rel_type="has_part",
                    weight=1.0,
                    confidence=1.0,
                )
            except Exception:
                pass  # relationship registration is best-effort
