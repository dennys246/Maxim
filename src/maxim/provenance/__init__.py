"""Provenance — lightweight, opt-in traceability for Maxim's bio systems.

Two-tier collection:
- Tier 1 (Cycle Traces): tied to run_id, tracks perception->outcome
- Tier 2 (Activity Log): background operations, no run_id required

Session-aware: all traces tagged with session_id for cross-run queries.
"""

from maxim.provenance.types import (
    PipelineStage,
    ProvenanceEntry,
    ProvenanceRef,
    ProvenanceTrace,
    ProvenanceVerbosity,
)

__all__ = [
    "PipelineStage",
    "ProvenanceEntry",
    "ProvenanceRef",
    "ProvenanceTrace",
    "ProvenanceVerbosity",
]
