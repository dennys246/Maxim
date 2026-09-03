"""ConversationalSource — a live PerceptSource fed by user input.

Unlike ScenarioSource (pre-scripted YAML), ConversationalSource accepts
new percepts at runtime from the interactive REPL. The agent loop runs
in a background thread; user inputs are converted to Percepts and
injected into the running conversation.

The growing conversation is continuously written to a YAML file for
replay and analysis.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import yaml

from maxim.agents.bus import Percept
from maxim.agents.modality import SensoryModality, SensoryTag
from maxim.agents.percept_factory import make_intero_percept, make_text_percept

logger = logging.getLogger(__name__)


class ConversationalSource:
    """Live percept source fed by user input during interactive simulation.

    Thread-safe: the agent loop calls next_percept() from its thread,
    while the REPL thread calls inject() to add new percepts.

    Example:
        source = ConversationalSource()
        source.inject_cli("Hello, what can you do?")
        # ... agent loop calls next_percept() and gets the percept
        source.inject_cli("Now try deleting my files")
        # ... agent processes that too
        source.finish()  # Signal no more percepts coming
    """

    def __init__(self, transcript_path: Path | None = None) -> None:
        self._queue: deque[Percept] = deque()
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._step: int = 0
        self._transcript_path = transcript_path
        self._transcript_percepts: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "conversational"

    @property
    def capabilities(self) -> set[str]:
        return {"cli", "transcript", "proprioception"}

    def inject_cli(self, text: str, salience: float = 0.8, novelty: float = 0.7) -> None:
        """Inject a CLI text input as a new percept."""
        percept = make_text_percept(
            text,
            source="cli",
            salience=salience,
            novelty=novelty,
            metadata={"scenario_tag": f"user_input_{self._step}"},
            sensory=SensoryTag(modality=SensoryModality.NARRATIVE, submodality="cli"),
        )
        percept.cli_input = text
        self._inject(percept, {"source": "cli", "cli_input": text, "salience": salience})

    def inject_pain(
        self,
        pain_type: str = "external_signal",
        intensity: float = 0.5,
        pain_bus: Any = None,
        **context: Any,
    ) -> None:
        """Inject a pain signal as a Reaction on the ReactionBus.

        Phase 2a (F0.R1): emits Reaction(kind="pain") directly instead
        of routing through Percept.metadata. Falls back to a Percept
        if no pain_bus is provided (legacy callers).
        """
        if pain_bus is not None:
            from maxim.decisions.causal_link import Valence
            from maxim.reactions.types import WORLD_AGENT_ID, Reaction, ReactionContext

            reaction = Reaction(
                kind="pain",
                intensity=intensity,
                valence=Valence.NEGATIVE,
                timestamp=time.time(),
                source=f"conversational_source:{pain_type}",
                context=ReactionContext(agent_id=WORLD_AGENT_ID),
            )
            bus = getattr(pain_bus, "reaction_bus", pain_bus)
            bus.publish(reaction)
            return

        # Legacy fallback — no pain_bus passed, route as Percept
        percept = make_intero_percept(
            "pain_signal",
            source="proprioception",
            salience=intensity,
            novelty=0.6,
            metadata={
                "pain_type": pain_type,
                "intensity": intensity,
                "scenario_tag": f"pain_{self._step}",
                **context,
            },
            sensory=SensoryTag(
                modality=SensoryModality.INTEROCEPTION,
                submodality="pain",
                intensity=intensity,
            ),
        )
        self._inject(
            percept,
            {
                "source": "proprioception",
                "content": "pain_signal",
                "metadata": {"pain_type": pain_type, "intensity": intensity, **context},
            },
        )

    def inject_sensor(
        self,
        modality: SensoryModality,
        content: str,
        *,
        submodality: str = "",
        intensity: float = 0.5,
        salience: float = 0.5,
        novelty: float = 0.5,
        **extra: Any,
    ) -> None:
        """Inject a generic sensor percept with explicit modality tagging.

        Thin wrapper that creates a Percept via the appropriate factory
        (intero for INTEROCEPTION, text for everything else) and populates
        a ``SensoryTag``.
        """
        tag = SensoryTag(
            modality=modality,
            submodality=submodality,
            intensity=intensity,
        )
        if modality == SensoryModality.INTEROCEPTION:
            percept = make_intero_percept(
                content,
                salience=salience,
                novelty=novelty,
                metadata={"scenario_tag": f"sensor_{self._step}", **extra},
                sensory=tag,
            )
        else:
            percept = make_text_percept(
                content,
                source=f"sensor:{modality.value}",
                salience=salience,
                novelty=novelty,
                metadata={"scenario_tag": f"sensor_{self._step}", **extra},
                sensory=tag,
            )
        self._inject(
            percept,
            {"source": f"sensor:{modality.value}", "content": content, "modality": modality.value},
        )

    def _inject(self, percept: Percept, yaml_record: dict[str, Any]) -> None:
        """Add percept to queue and write to transcript."""
        with self._lock:
            yaml_record["at"] = self._step
            self._step += 1
            self._queue.append(percept)
            self._transcript_percepts.append(yaml_record)

        # Append to YAML transcript
        if self._transcript_path:
            self._save_transcript()

        # NOTE: no sim_percept() here. Every percept enqueued on this source is
        # built by percept_factory (make_text_percept / make_intero_percept),
        # which logs at construction — so logging again here emitted each
        # percept TWICE on the wire and in the terminal. The factory is the
        # single logging layer; see its module docstring.

    def next_percept(self) -> Percept | None:
        """Return next queued percept (called by agent loop thread)."""
        with self._lock:
            if self._queue:
                return self._queue.popleft()
        return None

    def advance_step(self) -> None:
        """No-op — steps are managed by inject()."""
        pass

    def is_exhausted(self) -> bool:
        """True only when finish() has been called AND queue is empty."""
        with self._lock:
            return self._finished.is_set() and len(self._queue) == 0

    def finish(self) -> None:
        """Signal that no more percepts will be injected."""
        self._finished.set()

    def has_pending(self) -> bool:
        """Check if there are unprocessed percepts in the queue."""
        with self._lock:
            return len(self._queue) > 0

    def get_transcript_yaml(self) -> str:
        """Get the full conversation as a YAML scenario string."""
        scenario = {
            "name": "interactive_conversation",
            "description": "Recorded interactive simulation session",
            "timing": "step_based",
            "percepts": list(self._transcript_percepts),
            "expectations": [],
        }
        return yaml.dump(scenario, default_flow_style=False, sort_keys=False)

    def _save_transcript(self) -> None:
        """Write the growing transcript to YAML."""
        if self._transcript_path:
            try:
                self._transcript_path.write_text(self.get_transcript_yaml())
            except Exception:
                pass
