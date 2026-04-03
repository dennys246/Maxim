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
        percept = Percept(
            timestamp=time.time(),
            source="cli",
            cli_input=text,
            salience=salience,
            novelty=novelty,
            metadata={"scenario_tag": f"user_input_{self._step}"},
        )
        self._inject(percept, {"source": "cli", "cli_input": text, "salience": salience})

    def inject_pain(
        self, pain_type: str = "external_signal", intensity: float = 0.5, **context: Any
    ) -> None:
        """Inject a pain signal percept."""
        percept = Percept(
            timestamp=time.time(),
            source="proprioception",
            content="pain_signal",
            salience=intensity,
            novelty=0.6,
            metadata={
                "pain_type": pain_type,
                "intensity": intensity,
                "scenario_tag": f"pain_{self._step}",
                **context,
            },
        )
        self._inject(percept, {
            "source": "proprioception",
            "content": "pain_signal",
            "metadata": {"pain_type": pain_type, "intensity": intensity, **context},
        })

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

        # Simulation verbosity
        try:
            from maxim.simulation.sim_logger import sim_percept
            summary = yaml_record.get("cli_input") or yaml_record.get("content") or "signal"
            sim_percept(percept.source, str(summary)[:80], step=yaml_record["at"])
        except Exception:
            pass

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
