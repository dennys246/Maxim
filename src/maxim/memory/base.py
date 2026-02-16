from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
import time
import uuid

if TYPE_CHECKING:
    from maxim.agents.bus import StructuredContext

class SimpleRecord:
    """
    A single unit of memory (simple content wrapper).

    Used by InMemoryMemory for lightweight storage.
    Not to be confused with the MemoryRecord ABC in types.py
    which is the base class for EpisodicMemory, MathMemory, etc.
    """
    def __init__(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        record_id: Optional[str] = None,
    ):
        self.id = record_id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
    


class Memory(ABC):
    """
    Abstract base class for all memory systems.
    """

    @abstractmethod
    def store(self, record: SimpleRecord) -> None:
        """
        Persist a memory record.
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: Optional[Any] = None,
        *,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SimpleRecord]:
        """
        Retrieve relevant memory records.
        """
        pass

    @abstractmethod
    def forget(self, record_id: str) -> None:
        """
        Remove a specific memory record.
        """
        pass

    def store_raw(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = SimpleRecord(content=content, metadata=metadata)
        self.store(record)

    def retrieve_recent(self, limit: int = 10) -> List[SimpleRecord]:
        return self.retrieve(limit=limit)

    def clear(self) -> None:
        """
        Forget all memories.
        """
        records = self.retrieve(limit=10_000)
        for r in records:
            self.forget(r.id)


class InMemoryMemory(Memory):
    def __init__(self) -> None:
        self._records: list[SimpleRecord] = []
        self._cli_inputs: deque[str] = deque(maxlen=20)
        self._transcripts: deque[str] = deque(maxlen=20)

    def store(self, record: SimpleRecord) -> None:
        self._records.append(record)

    def retrieve(
        self,
        query: Optional[Any] = None,
        *,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SimpleRecord]:
        items = list(self._records)
        if filters:
            for key, value in filters.items():
                items = [r for r in items if r.metadata.get(key) == value]
        if query is not None:
            q = str(query).lower()
            items = [r for r in items if q in str(r.content).lower()]
        return items[-int(limit) :]

    def forget(self, record_id: str) -> None:
        self._records = [r for r in self._records if r.id != record_id]

    def record_command(self, command: str) -> None:
        """Record a CLI command for context building."""
        if command:
            self._cli_inputs.append(str(command))

    def record_transcript(self, text: str) -> None:
        """Record a transcript line for context building."""
        if text:
            self._transcripts.append(str(text))

    def build_context(self) -> "StructuredContext":
        """Build structured context for LLM goal proposal."""
        from maxim.agents.bus import StructuredContext

        return StructuredContext(
            timestamp=time.time(),
            current_percept=None,
            active_goal=None,
            mode="observe",
            detected_speech=list(self._transcripts),
            cli_inputs=list(self._cli_inputs),
            available_environments=["focus_interests", "track_target", "maxim_command", "respond", "speak"],
            detected_objects=[],
            detected_people=[],
            root_goal="Understand reality and help people.",
        )
