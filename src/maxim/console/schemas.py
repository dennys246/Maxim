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
    # Contract fix (maxim-pulse): a curation marker so the wizard's picker can
    # show "2-3 curated ▾ / Advanced" by INTENT, not collapse by count (first 3).
    # Set server-side from a small editable allow-list (server._CURATED_PROFILES).
    curated: bool = False


class ModelsResponse(BaseModel):
    groups: dict[str, list[ModelInfoWire]]


class DiagnoseSection(BaseModel):
    name: str
    status: str = ""
    detail: str | None = None
    # Observability payload — CC3-style escape hatch (diagnostics are not a strict
    # binding contract like the seams below); kept loose on purpose.
    extra: dict[str, Any] = Field(default_factory=dict)


class PlatformWire(BaseModel):
    """Structured platform identity for the StatusChip.

    Contract fix (maxim-pulse): ``DiagnoseResponse.platform`` used to be a
    stringified ``PlatformInfo`` repr (``"PlatformInfo(os='macos', …)"``) —
    useless for display. These are the display-relevant fields, mapped from
    ``doctor.platform_detect.PlatformInfo``.
    """

    os: str = ""
    arch: str = ""
    os_release: str = ""
    runtime: str = ""


class DiagnoseResponse(BaseModel):
    platform: PlatformWire = Field(default_factory=PlatformWire)
    sections: list[DiagnoseSection] = Field(default_factory=list)


# ── PROBE seam — structured connection test ─────────────────────────────────


class ProbeRequest(BaseModel):
    # Contract fix (maxim-pulse): ``url`` is now OPTIONAL so the SAME endpoint
    # serves both probe shapes (the seams plan: "covers both the mesh probe and a
    # cheap cloud-key probe"):
    #   * MESH probe  — ``url`` (+ optional ``api_key`` / ``model``): peer/leader
    #     reachability + auth via ``_MaximPeerBackend.health_check``.
    #   * CLOUD probe — ``provider`` (+ ``api_key``): a cheap pre-save key check
    #     against a cloud provider, no ``url``.
    # Exactly one of ``url`` / ``provider`` should be set; the handler dispatches
    # on which is present.
    url: str | None = None
    provider: str | None = None
    # Raw key to TEST (transient, localhost-only, NOT stored) — you probe a key
    # before saving it as a ref, so a ref can't exist yet. Mirrors the setup requests.
    api_key: str | None = None
    model: str | None = None


class ProbeResult(BaseModel):
    # Traffic-light — mirrors ProbeClassification.Status (the internal classifier).
    status: Literal["ok", "warn", "fail"]
    # Granular probe outcome: ok / auth_rejected / inference_broken / timeout /
    # connection_refused / dns_fail / tls_error / http_5xx / other.
    outcome: str
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
    # session_id is the CONSOLE-side run id minted at accept time. The sim
    # generates its own internal session_id after boot, so the two do NOT
    # match; correlating a run with ~/.maxim/sessions/{id} needs the future
    # run-status/ws surface (Phase 3). Documented so the generated TS client
    # doesn't assume filesystem correlation.
    session_id: str = Field(description="Console-side run id (minted at accept; not the sim's internal session_id).")
    mode: str
    status: Literal["started", "rejected"]
    detail: str | None = None


# ── /ws event envelope — the EventClient stream contract ────────────────────


class ConsoleEvent(BaseModel):
    # Skeleton envelope. The v2 shape is spec'd as the EVENT seam
    # (reachy_app_maxim_seams.md § EVENT): kind = lowercased sim_log subsystem
    # (open string — a closed Literal would fight the _SUBSYSTEM_TIERS
    # unknown→BIO opt-out invariant) + server-computed tier, seq, run_id,
    # epoch ts. `data` is the documented per-producer escape hatch (the
    # DiagnoseSection.extra precedent) — the envelope fields are the contract.
    kind: str  # lowercased sim_log subsystem, plus meta-kinds: "heartbeat" | "run" | "dropped" | "display"
    ts: float
    agent_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
