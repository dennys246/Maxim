from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

class MemoryRecord:
    """
    A single unit of memory.
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
    def store(self, record: MemoryRecord) -> None:
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
    ) -> List[MemoryRecord]:
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
        record = MemoryRecord(content=content, metadata=metadata)
        self.store(record)

    def retrieve_recent(self, limit: int = 10) -> List[MemoryRecord]:
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
        self._records: list[MemoryRecord] = []

    def store(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def retrieve(
        self,
        query: Optional[Any] = None,
        *,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
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
