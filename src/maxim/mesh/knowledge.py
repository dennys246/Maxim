"""Knowledge sharing protocol — provider/receiver pattern + broker.

Any subsystem can participate in mesh knowledge exchange by implementing
KnowledgeProvider (export) and/or KnowledgeReceiver (import). The
ExperienceBroker routes between them without knowing subsystem specifics.

Built-in adapters:
- CausalLinkProvider / CausalLinkReceiver (wraps NAc)
- ReflectionProvider / ReflectionReceiver (wraps Hippocampus)

Future adapters register when their subsystems ship (DN thresholds,
motor programs, forward models, etc.) — no broker changes needed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Protocol: any subsystem can implement these
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class KnowledgeProvider(Protocol):
    """Subsystem that can export shareable knowledge."""

    def knowledge_type(self) -> str:
        """Unique key: 'causal_link', 'reflection', 'motor_program', etc."""
        ...

    def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict[str, Any]]:
        """Return serialized knowledge items suitable for sharing.

        Each item is a plain dict (ready for JSON wire format).
        Filters are subsystem-specific (e.g. tool_name for NAc).
        """
        ...


@runtime_checkable
class KnowledgeReceiver(Protocol):
    """Subsystem that can import knowledge from peers."""

    def knowledge_type(self) -> str:
        """Must match the provider's knowledge_type."""
        ...

    def import_knowledge(self, data: dict[str, Any], source: str, trust: str) -> bool:
        """Import a single knowledge item.

        The receiver owns filtering, deduplication, and discount logic.
        Returns True if imported (novel), False if rejected/duplicate.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# ExperienceBroker — generic registry + router
# ─────────────────────────────────────────────────────────────────────────────


class ExperienceBroker:
    """Routes knowledge between subsystems and peers.

    Subsystems register as providers and/or receivers. The broker
    dispatches EXPERIENCE_* messages to the right subsystem without
    knowing the specifics of each knowledge type.
    """

    def __init__(self) -> None:
        self._providers: dict[str, KnowledgeProvider] = {}
        self._receivers: dict[str, KnowledgeReceiver] = {}

    def register_provider(self, provider: KnowledgeProvider) -> None:
        kt = provider.knowledge_type()
        self._providers[kt] = provider
        logger.debug("knowledge broker: registered provider for %r", kt)

    def register_receiver(self, receiver: KnowledgeReceiver) -> None:
        kt = receiver.knowledge_type()
        self._receivers[kt] = receiver
        logger.debug("knowledge broker: registered receiver for %r", kt)

    def get_shareable(
        self,
        knowledge_type: str | None = None,
        limit: int = 10,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Get shareable knowledge, optionally filtered by type.

        Each returned item includes a 'knowledge_type' key so the peer
        knows which receiver to route to.
        """
        if knowledge_type is not None:
            provider = self._providers.get(knowledge_type)
            if provider is None:
                return []
            items = provider.get_shareable(limit=limit, **filters)
            return [{"knowledge_type": knowledge_type, **item} for item in items]

        result: list[dict[str, Any]] = []
        for kt, provider in self._providers.items():
            items = provider.get_shareable(limit=limit, **filters)
            result.extend({"knowledge_type": kt, **item} for item in items)
        return result[:limit]

    def import_knowledge(
        self,
        knowledge_type: str,
        data: dict[str, Any],
        source: str,
        trust: str,
    ) -> bool:
        """Route imported knowledge to the right receiver.

        Returns True if accepted, False if no receiver or rejected.
        """
        receiver = self._receivers.get(knowledge_type)
        if receiver is None:
            logger.debug("knowledge broker: no receiver for type %r, dropping", knowledge_type)
            return False
        return receiver.import_knowledge(data, source, trust)

    @property
    def registered_types(self) -> list[str]:
        """Knowledge types that can be shared and/or received."""
        return sorted(set(self._providers) | set(self._receivers))

    @property
    def provider_types(self) -> list[str]:
        return sorted(self._providers)

    @property
    def receiver_types(self) -> list[str]:
        return sorted(self._receivers)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in adapter: CausalLink (wraps NAc)
# ─────────────────────────────────────────────────────────────────────────────


class CausalLinkProvider:
    """Exports high-confidence causal links from NAc."""

    SHARE_MIN_CONFIDENCE = 0.6
    SHARE_MIN_OBSERVATIONS = 3

    def __init__(self, nac: Any) -> None:
        self._nac = nac

    def knowledge_type(self) -> str:
        return "causal_link"

    def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict[str, Any]]:
        tool_name = filters.get("tool_name")
        if tool_name:
            links = self._nac.get_links_for_event(f"tool:{tool_name}")
            return [
                link.to_dict()
                for link in links
                if link.confidence >= self.SHARE_MIN_CONFIDENCE
                and link.observation_count >= self.SHARE_MIN_OBSERVATIONS
            ][:limit]

        # Scan all links directly — get_promotion_candidates returns
        # PromotionCandidate (for ATL), not the raw CausalLink we need.
        result: list[dict[str, Any]] = []
        for links in self._nac._links.values():
            for link in links:
                if (
                    link.confidence >= self.SHARE_MIN_CONFIDENCE
                    and link.observation_count >= self.SHARE_MIN_OBSERVATIONS
                ):
                    result.append(link.to_dict())
                    if len(result) >= limit:
                        return result
        return result


class CausalLinkReceiver:
    """Imports causal links from peers into local NAc with transfer discount."""

    TRANSFER_DISCOUNTS: dict[str, float] = {
        "verified": 0.5,
        "discovered": 0.3,
        "remote": 0.3,
        "unknown": 0.1,
    }

    def __init__(self, nac: Any) -> None:
        self._nac = nac

    def knowledge_type(self) -> str:
        return "causal_link"

    def import_knowledge(self, data: dict[str, Any], source: str, trust: str) -> bool:
        from maxim.decisions.causal_link import CausalLink

        link = CausalLink.from_dict(data)

        # Deduplicate: skip if we already know this event→outcome
        existing = self._nac.get_links_for_event(link.event_signature)
        for ex in existing:
            if ex.outcome_signature == link.outcome_signature:
                return False

        # Apply trust-level transfer discount
        discount = self.TRANSFER_DISCOUNTS.get(trust, 0.1)
        link.confidence *= discount
        link.predicted_value = 0.5  # Reset R-W — local experience refines
        link.observation_count = 1

        # Tag provenance
        link.event_context["_imported_from"] = source
        link.event_context["_import_timestamp"] = time.time()
        link.event_context["_import_trust"] = trust

        self._nac._register_imported_link(link)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Built-in adapter: Reflection (wraps Hippocampus)
# ─────────────────────────────────────────────────────────────────────────────


class ReflectionProvider:
    """Exports reflections (episodic memories with reflection text) from Hippocampus."""

    def __init__(self, hippocampus: Any) -> None:
        self._hippocampus = hippocampus

    def knowledge_type(self) -> str:
        return "reflection"

    def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict[str, Any]]:
        memories = self._hippocampus.recall(limit=limit, tool=filters.get("tool_name"))
        result: list[dict[str, Any]] = []
        for m in memories:
            outcome = getattr(m, "outcome", None)
            if hasattr(outcome, "result") and isinstance(outcome.result, dict):
                if outcome.result.get("reflection"):
                    result.append(m.to_dict())
        return result


class ReflectionReceiver:
    """Imports peer reflections as low-salience episodic memories."""

    SALIENCE_CAPS: dict[str, float] = {
        "verified": 0.5,
        "discovered": 0.35,
        "remote": 0.35,
        "unknown": 0.15,
    }

    def __init__(self, hippocampus: Any) -> None:
        self._hippocampus = hippocampus

    def knowledge_type(self) -> str:
        return "reflection"

    def import_knowledge(self, data: dict[str, Any], source: str, trust: str) -> bool:
        from maxim.memory.types import EpisodicMemory

        memory = EpisodicMemory.from_dict(data)

        cap = self.SALIENCE_CAPS.get(trust, 0.15)
        memory.perception.salience = min(memory.perception.salience, cap)
        memory.perception.observations["_imported_from"] = source
        memory.perception.observations["_import_trust"] = trust

        return self._hippocampus.capture(record=memory) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Built-in adapter: MotorProgram (wraps ProgramRegistry + Embodiment)
# ─────────────────────────────────────────────────────────────────────────────


class MotorProgramProvider:
    """Exports learned motor programs from ProgramRegistry.

    Only shares programs with enough executions and a reasonable success rate
    so peers receive battle-tested sequences, not one-off accidents.
    """

    MIN_EXECUTIONS = 3
    MIN_SUCCESS_RATE = 0.5

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def knowledge_type(self) -> str:
        return "motor_program"

    def get_shareable(self, limit: int = 10, **filters: Any) -> list[dict[str, Any]]:
        state = self._registry.export_state()
        result: list[dict[str, Any]] = []
        for prog_data in state.get("programs", {}).values():
            executions = prog_data.get("total_executions", 0)
            successes = prog_data.get("total_successes", 0)
            if executions < self.MIN_EXECUTIONS:
                continue
            if executions > 0 and successes / executions < self.MIN_SUCCESS_RATE:
                continue
            # Optional goal filter
            goal = filters.get("goal")
            if goal and prog_data.get("goal_signature") != goal:
                continue
            result.append(prog_data)
            if len(result) >= limit:
                break
        return result


class MotorProgramReceiver:
    """Imports motor programs from peers with entity-path validation.

    Before accepting a program, every step's entity_path must exist in
    the local embodiment spec. This prevents importing sequences that
    reference hardware the local robot doesn't have.

    Requires at least "discovered" trust (motor programs touch hardware).
    """

    TRUST_FLOOR = "discovered"  # Minimum trust to accept motor programs
    _TRUSTED = {"verified", "discovered"}

    CONFIDENCE_DISCOUNT = 0.5  # Imported programs start at half confidence

    def __init__(self, registry: Any, embodiment: Any | None = None) -> None:
        self._registry = registry
        self._embodiment = embodiment

    def knowledge_type(self) -> str:
        return "motor_program"

    def import_knowledge(self, data: dict[str, Any], source: str, trust: str) -> bool:
        # Trust gate — motor programs touch hardware
        if trust not in self._TRUSTED:
            logger.debug(
                "motor_program: rejecting from %s (trust=%s, need %s)",
                source,
                trust,
                self._TRUSTED,
            )
            return False

        from maxim.embodiment.motor import MotorProgram

        program = MotorProgram.from_dict(data)

        # Deduplicate — skip if we already have a program with this name
        if self._registry.get(program.name) is not None:
            return False

        # Entity-path validation — every step must reference local hardware
        if self._embodiment is not None:
            for step in program.steps:
                if self._embodiment.find(step.entity_path) is None:
                    logger.debug(
                        "motor_program: rejecting %r — entity %r not in local spec",
                        program.name,
                        step.entity_path,
                    )
                    return False

        # Apply confidence discount + tag provenance
        program.confidence *= self.CONFIDENCE_DISCOUNT
        program.total_executions = 0  # Reset — local experience builds trust
        program.total_successes = 0
        program.source = f"peer:{source}"

        self._registry.register(program)
        return True
