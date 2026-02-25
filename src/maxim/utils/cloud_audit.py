"""Cloud audit logging for LLM usage."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from maxim.utils.structured_logging import log_structured


@dataclass(slots=True)
class CloudAuditEntry:
    timestamp: float
    provider: str
    model: str
    data_categories_sent: list[str]
    data_categories_redacted: list[str]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    request_id: str
    agent: str
    redaction_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "data_categories_sent": self.data_categories_sent,
            "data_categories_redacted": self.data_categories_redacted,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "request_id": self.request_id,
            "agent": self.agent,
            "redaction_policy": self.redaction_policy,
        }


class CloudAuditLogger:
    """Append-only JSONL audit log for cloud API calls."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path or os.path.join("data", "logs", "cloud_audit.jsonl")
        self._lock = threading.Lock()
        self._logger = logging.getLogger("cloud_audit")

    def write(self, entry: CloudAuditEntry) -> None:
        record = entry.to_dict()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

        log_structured(
            self._logger,
            logging.INFO,
            "cloud_audit",
            data=record,
            msg="cloud_audit",
        )


__all__ = ["CloudAuditEntry", "CloudAuditLogger"]
