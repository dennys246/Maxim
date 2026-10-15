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


# ── Identity — "which backend am I actually talking to?" ────────────────────


class SeamStatus(BaseModel):
    """One console surface and whether it is live here."""

    name: str
    live: bool
    detail: str | None = None


class HelloResponse(BaseModel):
    """The ONE unauthenticated API surface (hardening design A6).

    Just enough for a tokenless client to detect contract skew and render the
    right login screen: the contract version and the auth scheme this server
    demands ("bearer", or "none" under sandbox mode where the proxy owns the
    edge). Everything richer — identity, seams, git — stays behind auth.
    """

    contract_version: str
    auth: Literal["bearer", "none"]
    # 0.5.0 (A9.1): whether this deployment can pair by spoken code — the
    # device path where a robot announces a short-lived code in the room.
    # Additive with a default so a 0.4.0 client never notices.
    pairing: Literal["available", "none"] = "none"


class PairRequestAccepted(BaseModel):
    """POST /api/pair/request accepted — the code is ANNOUNCED, never returned."""

    detail: str


class PairClaimRequest(BaseModel):
    """The spoken code, typed by the owner into the paste screen.

    ``max_length`` bounds the one pre-auth POST body on the server (review
    fold); real codes are exactly six digits, the slack forgives pasted
    whitespace.
    """

    code: str = Field(max_length=16)


class PairClaimResult(BaseModel):
    """A successful claim: the console bearer token — handled and stored by
    the client exactly as a ``/#token=`` fragment bootstrap would (the trust
    statement itself is A9.1's, spoken-code pairing)."""

    token: str


class IdentityResponse(BaseModel):
    """Self-describing backend identity.

    Exists because a console could not previously answer "which build is this?"
    — and the answer changes silently. pymaxim is typically installed EDITABLE,
    so `maxim serve` follows whatever git branch is checked out: switch to a
    branch predating a seam and that seam quietly vanishes from the UI. Add a
    stale `serve` process still holding the port and the console can be talking
    to code that no longer exists on disk.

    Every field here is something a debugging session would otherwise have to
    guess. Also emitted as the FIRST /ws event on connect (kind="identity"), so
    a client knows what it is attached to before any other event arrives.
    """

    package_version: str
    contract_version: str
    git_sha: str | None = None
    git_branch: str | None = None
    python_version: str = ""
    # Where the served UI bundle came from, and what it claims to be built
    # against — the other half of a contract mismatch.
    ui_source: Literal["flag", "config", "packaged", "none"] = "none"
    ui_dist: str | None = None
    ui_manifest: dict[str, Any] = Field(default_factory=dict)
    # Which console surfaces are live in THIS build.
    seams: list[SeamStatus] = Field(default_factory=list)


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
    # talk      — one conversational turn against the live loop (blocking).
    # adventure — a campaign, authored (`campaign`) or imagined (`input`).
    # rest      — consolidate memory WITHOUT teardown; the agent stays usable.
    # sim       — deliberately NOT a console surface (developer/research); the
    #             501 points at the CLI rather than reading as an unfinished
    #             stub. Kept in the enum because removing a value is a breaking
    #             wire change for the generated TS client.
    mode: Literal["talk", "adventure", "sim", "rest"]
    # mode="talk": the user's utterance.
    # mode="adventure": a free-text premise — "describe an adventure and let
    #   Maxim imagine it" (generative campaign). Exactly one of input/campaign.
    input: str | None = None
    campaign: str | None = None  # for mode="adventure": a campaign YAML path


class CampaignInfo(BaseModel):
    """One discoverable campaign — backs the launcher's picker dropdown.

    ``path`` is what you hand back to ``POST /api/run`` as ``campaign``; the
    rest is display metadata read cheaply from the YAML head.
    """

    name: str
    path: str
    goal: str | None = None
    source: Literal["user", "repo"] = "user"


class CampaignsResponse(BaseModel):
    campaigns: list[CampaignInfo] = Field(default_factory=list)
    # Where discovery looked — surfaced so an empty list is explainable in the
    # UI ("no campaigns in ~/.maxim/campaigns") rather than mystifying.
    searched: list[str] = Field(default_factory=list)


class RunAccepted(BaseModel):
    # session_id is the CONSOLE-side run id minted at accept time. The sim
    # generates its own internal session_id after boot, so the two do NOT
    # match; correlating a run with ~/.maxim/sessions/{id} needs the future
    # run-status/ws surface (Phase 3). Documented so the generated TS client
    # doesn't assume filesystem correlation.
    session_id: str = Field(description="Console-side run id (minted at accept; not the sim's internal session_id).")
    mode: str
    # "started" = accepted, running in the background (adventure).
    # "completed" = the call BLOCKED and the work is done (talk turn) — the
    #   client should not wait for a background finish.
    # "rejected" = not accepted.
    status: Literal["started", "completed", "rejected"]
    detail: str | None = None
    # Talk turns only: the agent's words. Also delivered on /ws as
    # kind="response" (the live channel the chat renders from); this field
    # makes delivery robust when the client connected late or the stream
    # dropped events under backpressure. None for background modes.
    reply: str | None = None


# ── /ws event envelope — the EventClient stream contract (EVENT seam) ────────


class ConsoleEvent(BaseModel):
    """v2 envelope (reachy_app_maxim_seams.md § EVENT) — the wire event IS the
    ``sim_log`` record, bridged via ``register_sim_sink``.

    ``kind`` stays an OPEN string (lowercased sim_log subsystem) — a closed
    Literal would fight the ``_SUBSYSTEM_TIERS`` unknown→BIO opt-out invariant
    (new subsystems must surface by default). The typed axis clients filter on
    is ``tier`` (server-computed per event, unknown subsystem → "bio").

    ``data`` is the documented per-producer escape hatch (the
    ``DiagnoseSection.extra`` precedent): the payload varies across 30+
    subsystems; the envelope fields are the contract. Per-kind typed models for
    headline kinds are a later additive step if a panel needs them.
    """

    # Lowercased sim_log subsystem ("hippocampus", "nac", "deliberation", …)
    # plus the meta-kinds: "heartbeat" | "run" | "dropped" | "display" | "identity",
    # plus the OPT-IN kinds (0.6.0, delivered only when a SubscribeFrame names
    # them): "health" | "health_stall". The name `health` is RESERVED: a future
    # sim_log HEALTH subsystem would collide with it on the wire.
    # "run" data carries status: "started" | "ended" | "failed" (+ run_id,
    # sim_session_id, report_path on "ended").
    kind: str
    # Server-computed from _SUBSYSTEM_TIERS (unknown subsystem → "bio").
    # Console-published meta-kinds (heartbeat/run/dropped) are "clean";
    # "display" arrives as a real DISPLAY sim_log record and carries that
    # subsystem's tier ("debug").
    # REQUIRED (no default): the server always populates it — a default here
    # would mark the generated TS field optional and force null-guards on
    # every consumer path (review fold).
    tier: Literal["clean", "bio", "debug"]
    # Per-connection monotonic, assigned at enqueue — a gap means drops.
    # REQUIRED for the same TS-optionality reason.
    seq: int
    # Console-side run id (RunAccepted.session_id); None outside a run. The
    # "run" meta-kind's data binds this to the sim's internal session id.
    run_id: str | None = None
    # Epoch seconds, stamped at bridge time. sim_log's own `t` is SIM-ELAPSED
    # and travels as elapsed_s — the definition travels with the data.
    ts: float
    elapsed_s: float | None = None
    agent_id: str | None = None
    agent: str | None = None  # display nickname, when registered
    # REQUIRED (may be empty string) — same TS-optionality reason.
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class SubscribeFrame(BaseModel):
    """Client→server ``/ws`` filter frame — the terminal's ``_show_channels`` /
    ``DisplayTier`` model lifted to the socket (a thin UI subscribes to less).

    Semantics: axes AND together; within the subsystem axis, ``channels`` and
    ``kinds`` union. ``tier`` passes events whose tier ≤ the requested tier
    (requesting "clean" = headline only; "debug" = everything). No frame (or
    all-None) = every ordinary event. Meta-kinds (heartbeat/run/dropped and the
    agent's "display" suggestions) bypass filtering — they carry stream/UI
    state, not subsystem traffic.

    OPT-IN kinds (0.6.0: ``health``, ``health_stall``) are never sent unless
    named in ``kinds``, and naming them ADDS them rather than filtering:
    ``{"kinds": ["health"]}`` = every ordinary event plus health;
    ``{"tier": "clean", "kinds": ["health"]}`` = the clean tier plus health.
    Only the ordinary kinds in ``kinds`` narrow the stream.

    OpenAPI does not model WS payloads, so ``GET /events/subscribe-frame``
    documents this shape for type-gen (same trick as ``/events/envelope``).
    """

    channels: list[str] | None = None  # _CHANNEL_MAP names ("bio", "memory", …) or raw subsystem names
    tier: Literal["clean", "bio", "debug"] | None = None
    kinds: list[str] | None = None  # exact kinds, case-insensitive


# ── run history + health (contract 0.6.0) ────────────────────────────────────
#
# Read-only views over `sim_reports/{id}/` and the console process's health
# log. Every field a reader cannot rely on is Optional: the on-disk shapes
# predate this contract (most local report.json files predate `ts`; credit
# fields exist only on actions written by a credit-recording (1.2) writer), and
# a missing value must read as "not recorded", never as a fake zero.


class RunSummary(BaseModel):
    """One row of the run list. ``has_report`` False = an aborted run dir."""

    session_id: str
    started_at: float | None = Field(
        default=None,
        description="Start epoch: report.ts if present, else parsed from a %Y%m%d_%H%M%S id, else null.",
    )
    has_report: bool
    mode: str | None = None
    goal: str | None = None
    finish_reason: str | None = None
    turns: int | None = None
    duration_s: float | None = None
    cost_usd: float | None = None
    total_actions: int | None = None


class RunPage(BaseModel):
    runs: list[RunSummary]
    next_cursor: str | None = None


class RunReport(BaseModel):
    """A named subset of ``SimulationReport`` — never the substrate (``aut_*``) files."""

    session_id: str
    started_at: float | None = None
    ended_at: float | None = Field(
        default=None, description="End epoch, from report.timestamp (the build time, written as local time)."
    )
    format_version: str | None = None
    goal: str | None = None
    mode: str | None = None
    finish_reason: str | None = None
    duration_s: float | None = None
    turns: int | None = None
    total_actions: int | None = None
    blocked_actions: int | None = None
    cost_usd: float | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    latency_p50_ms: float | None = None
    latency_p99_ms: float | None = None
    language_model: str | None = None
    aut_model: str | None = None
    tool_usage: dict[str, int] = Field(default_factory=dict)
    tool_success_rates: dict[str, float] = Field(default_factory=dict)
    pain_events_count: int | None = None
    fear_blocks_count: int | None = None
    learn_events_count: int | None = None
    llm_finish_status: str | None = None
    llm_finish_summary: str | None = None
    llm_summary: str | None = None
    response_texts: list[str] = Field(default_factory=list)


class ActionRecord(BaseModel):
    """One ``actions.jsonl`` line. Credit fields are null when the line predates them."""

    timestamp: float | None = None
    tool: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    success: bool | None = None
    blocked: bool | None = None
    block_reason: str | None = None
    error: str | None = None
    output: str | None = None
    agent_id: str | None = None
    entity_class: str | None = None
    rpe: float | None = Field(default=None, description="ToolOutput.rpe — the surprise.")
    pain: float | None = None
    nac_failure_booked: str | None = Field(
        default=None,
        description="The signature the tool-pain bridge booked this failure under (dispatch then skipped its own).",
    )
    credit_recorded: bool = Field(
        default=False,
        description=(
            "True when the run's actions file records per-action credit (format 1.2+). False = unknown, "
            "not zero: runs written before the 1.2 writer read False."
        ),
    )


class ActionPage(BaseModel):
    session_id: str
    available: bool = Field(description="False = this run recorded no actions file (aborted, or a talk/rest run).")
    reason: str | None = None
    format_version: str | None = Field(
        default=None, description="actions.jsonl header version; per-action credit exists from 1.2."
    )
    actions: list[ActionRecord]
    next_cursor: str | None = None
    skipped: int = Field(default=0, description="Malformed or oversized lines skipped on this page.")


class HealthRecord(BaseModel):
    kind: Literal["health", "health_stall"]
    ts: float
    data: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """The console process's health: latest snapshot, stalls and a downsampled series.

    Snapshots are allow-listed (gpu, cpu, memory, disk, lanes, loop): never
    network interfaces, Wi-Fi identity, hostnames or paths.
    """

    monitoring: bool = Field(description="False = this server is not sampling health (build_app without the monitor).")
    stall_detection: bool = Field(
        description=(
            "False = stalls CANNOT be detected here (no agent loop is running in this process), so an empty "
            "`stalls` means unknown, not healthy. A stall is a planning turn stuck past the threshold."
        )
    )
    recorded_since: float | None = Field(
        default=None, description="Earliest record this view covers; null = nothing recorded yet."
    )
    dropped_records: int = Field(
        default=0,
        description="Records this server could not persist (a full queue, a busy lock, a failed write). Nonzero = gaps.",
    )
    latest: dict[str, Any] | None = None
    stalls: list[HealthRecord] = Field(default_factory=list)
    series: list[HealthRecord] = Field(default_factory=list)


class DegradedSite(BaseModel):
    """``swallowed_exception`` fires grouped by site — the "degraded silently" signal."""

    site: str
    count: int
    first_seen: float
    last_seen: float
    exc_type: str | None = None
    sample: str | None = Field(default=None, description="One exception message, truncated to 300 chars.")


class DegradedResponse(BaseModel):
    """Stage-1 sites only (the explicit `exc=`/`operation=` form carries no event)."""

    since: float
    counted_since: float | None = Field(
        default=None, description="Counts cover fires from this time on (the log's retained window)."
    )
    sites: list[DegradedSite]
