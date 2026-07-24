"""Pydantic **wire-models** for ``maxim serve`` — the cross-repo facade contract.

These are the WIRE types (Pydantic, OpenAPI-emitting), deliberately distinct from
the internal CC3-frozen dataclasses (the same wire-vs-internal split as
``Percept.to_wire_dict``). FastAPI generates the OpenAPI schema from these, and the
maxim-pulse kit generates its TypeScript ``FacadeClient`` from that schema. **So
these shapes ARE the contract** — change them deliberately as a coordinated
cross-repo change, never casually.

The seam models (Probe / Setup / Recall / Run) are **schema-complete even though the
seams are Phase-1 stubs**, so the frontend can generate the full client now and the
seam bodies fill in later without touching the schema. No bare ``dict``/``Any`` on a
seam surface — that would defeat the generated TypeScript type.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ── existing structured verbs (wrap the api.py dataclasses) ──────────────────


class ModelInfoWire(BaseModel):
    name: str
    backend: str
    cloud: bool
    api_key_env: str | None = None
    context_length: int | None = None
    downloaded: bool = False
    ready: bool = False


class ModelsResponse(BaseModel):
    groups: dict[str, list[ModelInfoWire]]


class DiagnoseSection(BaseModel):
    name: str
    status: str = ""
    detail: str | None = None
    # Observability payload — CC3-style escape hatch (diagnostics are not a strict
    # binding contract like the seams below); kept loose on purpose.
    extra: dict[str, Any] = Field(default_factory=dict)


class DiagnoseResponse(BaseModel):
    platform: str
    sections: list[DiagnoseSection] = Field(default_factory=list)


# ── PROBE seam — structured connection test ─────────────────────────────────


class ProbeRequest(BaseModel):
    url: str
    api_key_ref: str | None = None
    model: str | None = None


class ProbeResult(BaseModel):
    status: Literal["ok", "auth_failed", "unreachable", "model_missing", "error"]
    message: str
    fix_hint: str | None = None
    latency_ms: float | None = None


# ── SETUP seam — write mesh / cloud config (key stored as a ref, never inline) ─


class MeshSetupRequest(BaseModel):
    leader_url: str
    api_key: str  # server writes this to a keyed file; only the ref lands in config


class CloudSetupRequest(BaseModel):
    provider: str
    profile: str
    api_key: str
    monthly_budget_usd: float | None = None


class SetupResult(BaseModel):
    ok: bool
    placement: Literal["mesh", "cloud"]
    detail: str


# ── RECALL seam — "what Maxim remembers about you" (curated, provenance-filtered) ─


class StoryMemory(BaseModel):
    summary: str
    when: str | None = None
    salience: float | None = None


class Preference(BaseModel):
    about: str
    learned_from: str | None = None


class RecallResponse(BaseModel):
    name: str | None = None
    player_model: list[str] = Field(default_factory=list)
    story_memories: list[StoryMemory] = Field(default_factory=list)
    preferences: list[Preference] = Field(default_factory=list)


# ── HANDLE seam — run a mode (talk / adventure / sim / rest) ─────────────────


class RunRequest(BaseModel):
    mode: Literal["talk", "adventure", "sim", "rest"]
    input: str | None = None
    campaign: str | None = None  # for mode="adventure"


class RunAccepted(BaseModel):
    session_id: str
    mode: str
    status: Literal["started", "rejected"]
    detail: str | None = None


# ── /ws event envelope — the EventClient stream contract ────────────────────


class ConsoleEvent(BaseModel):
    kind: str  # "heartbeat" | "log" | "thinking" | "status" | ...
    ts: float
    agent_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
