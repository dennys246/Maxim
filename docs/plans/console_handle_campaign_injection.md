# HANDLE seam — part (a): persistent-agent campaign injection

**Status:** design (dedicated PR home). Part (b), the explicit `consolidation` stop
contract, shipped in #427 and this refactor consumes it.

**Parent spec:** [reachy_app_maxim_seams.md](reachy_app_maxim_seams.md) § HANDLE (a);
console framing in [maxim_console.md](maxim_console.md) § "HANDLE has two flavors".

---

## The problem (confirmed)

[`simulation/orchestrator.py:533`](../../src/maxim/simulation/orchestrator.py#L533)
builds `AgentConfig(agent_id="sim_aut", persistence_dir=sim_tmpdir)` and calls
`create_full_agent` → a **throwaway** AUT with session-scoped
Hippocampus/NAc (`~/.maxim/sessions/{id}/aut_hippocampus.json`). An Adventure run
on that plumbing learns into a throwaway the **Talk** agent never reads — the
"Adventure teaches Talk" promise fails **silently**. This is the single most
learning-safety-sensitive path in the console; the failure mode is silent
mis-routing of learning, not a crash.

## Front-gate — ride existing infra?

**Mostly.** `campaign_runner.run_dm_campaign` **already accepts**
`aut_hippocampus` / `aut_nac` / `aut_memory_hub` (and the generative runner takes
`nac` + `agent_id`). `api.campaign` already supports `party_mode`. The throwaway
exists *only* because `start_simulation_mode` **constructs** `sim_aut` internally.
No new runtime, no public-API change, no fork of `DMRuntime`.

## The refactor — inject at `start_simulation_mode`

Add one param: `persistent_agent: AgentInstance | None = None`. When provided,
the orchestrator **adopts** the live instance instead of constructing `sim_aut`.
Rejected alternative: rebuilding `run_dm_campaign`'s bridge/router/registry/
executor plumbing *inside the handle* — duplicates orchestrator setup and drifts.

### Branches that must change (each is a silent-mis-route risk if missed)

1. **Skip construction.** When `persistent_agent` is set, do NOT call
   `_aut_factory.create_full_agent`; set `_aut_instance = persistent_agent` and
   derive `aut_hippocampus/nac/memory_hub/executor/bio_stack/agent_id` from it.
   The `agent_id` is the persistent one (e.g. `"console_agent"`), **never**
   `"sim_aut"` — that is what routes episodes to the persistent home.
2. **Skip the resume-session file-load** (orchestrator.py:577-605). The persistent
   agent already restored state via `auto_load=True`; re-loading
   `sessions/{id}/aut_hippocampus.json` would **clobber** live state. Gate the
   whole `resume_session` block on `persistent_agent is None`.
3. **Persistence dir.** `persistence_dir` must be the persistent agent's home,
   not `sim_tmpdir`. Confirm no downstream code writes AUT state to `sim_tmpdir`
   when injected (tracers, session save, embodiment).
4. **Tool registry reconciliation.** `aut_registry` is built with sim tools
   *before* the factory call and passed as `tool_registry=`. A persistent agent
   carries its own registry. Decide: register the sim/DM tools (`ChooseTool`,
   memory/introspection tools) onto the **persistent** agent's registry for the
   campaign's duration, then deactivate on stop — do NOT swap the agent's
   registry wholesale.
5. **Session-end consolidation = "full".** Route the AUT loop's `run_agentic_loop`
   through the `consolidation="full"` override from #427 for the persistent agent
   (a session-scoped `sim_aut` gets `"lightweight"`; a persistent one must not).
6. **Fiction provenance.** Point `tag_imagined_links` / `decay_imagined_links` at
   the **persistent** agent so in-fiction facts decay while the shared episode +
   player-model persist as real (orchestrator.py:2740 area).
   **As-shipped honesty note (three-lens review, cross-confirmed):** the block
   is correctly pointed at the persistent NAc *but is structurally inert on the
   injected DM path* — the imagination trigger requires `entity_ref`, which
   injection forbids, so **campaign-declared fiction persists as real
   learning**. Episodes-as-real matches the intent above; provenance tagging
   for campaign-DECLARED entities (not just LLM-imagined ones) is a tracked
   follow-up. Design constraint for that follow-up: `decay_imagined_links`
   decays ALL `imagined=True` links per call — on a persistent NAc, repeated
   campaigns compound decay across sessions, so the tag set must be
   session-scoped.

## The handle wrapper (headless flavor)

A small `MaximHandle` (console) over one persistent agent built via
`create_full_agent(config, auto_load=True)` with a `~/.maxim` home:

- `handle.play_campaign(path)` → `start_simulation_mode(dm_campaign=…, persistent_agent=self.instance)`.
- `handle.stop(consolidation="full")` → `self.instance.shutdown()` (already does
  full `on_session_end` + hippocampus/NAc save + `save_cerebellum`).
- `talk(...)` / `rest(...)` live-loop modes + the `/ws` `api.on()` stream are
  **Phase 3** — out of scope here. This PR delivers the injection + clean stop.

Embodied (Reachy) flavor is the same interface with `body=bodies/reachy_mini`;
keep the constructor body-agnostic so `RunSurface` drives a HANDLE, not "a robot".

## `/api/run` wiring (console)

`POST /api/run` `mode="adventure"` → `handle.play_campaign(body.campaign)`, returns
`RunAccepted(session_id=…, mode="adventure", status="started")`. `talk`/`sim`/`rest`
stay 501 until Phase 3 streaming lands. No schema change → `openapi.json` unchanged
(regen + diff-check to confirm).

## Regression guards (the spec's bar — integration, real bio-stack)

1. `handle.play_campaign(...)` for one turn against a **persistent** agent → the
   resulting episode is **recallable from that agent's Hippocampus** (same
   `agent_id` + home), not merely a session-scoped AUT file.
2. An `imagined=True` in-fiction fact **decays** while a real episodic memory does
   **not** (fiction-vs-fact provenance survives the injection).
3. `persistent_agent is None` path is **byte-identical** to today (every existing
   sim/campaign run unchanged) — pin with the existing orchestrator tests.
4. Stop invokes **full** consolidation for the persistent agent (CC8 + #427).

## Review round (2026-07-25, pre-merge, three lenses) — folded

Executor + Architecture + bio-fidelity lenses ran on the implementation
commit; cross-confirmed findings folded in the same branch:

- **respond/speak leased onto the adopted registry** (Exec-B1/Arch-S5): the
  handle registry lacks them; without the lease every DM turn reads back None.
- **Lease exception-safety** (Exec-B2/Arch-B2): `_CampaignToolLease` is now an
  object-identity snapshot with idempotent `restore()` (also closes the
  replaced-tool blindness, Exec-S7); `MaximHandle.play_campaign` carries a
  `finally` safety net for mid-sim raises. Residual: direct
  `start_simulation_mode(persistent_agent=…)` callers bypassing the handle get
  normal-path restore only (a full try/finally inside the monolithic
  orchestrator body would re-indent ~2,700 lines — deferred to an
  orchestrator-decomposition refactor).
- **DN PainCircuitBridge unsubscribed from the persistent bus at teardown**
  (Exec-B3/bio-F2): otherwise each campaign accumulates a dead subscriber — a
  latent duplicate NAc pain learner.
- **ConceptExtractor revival** (Arch-B1/Exec-S4): `restart_worker()` +
  `MemoryHub.on_session_start` revive — root-cause fix; without it campaign #2
  on one handle captures episodes while ATL extraction is silently dead.
- **Pre-existing `hippocampus.sleep()` self-deadlock fixed** (surfaced by the
  revival guard, not by the lenses): `_sleep` ran `auto_save_after_sleep`'s
  `save_with_backup` UNDER the consolidation write lock, and save → dump takes
  a read lock on the same non-reentrant RWLock — full consolidation with a
  live persistence_path (the default config!) hung forever. Masked in CI
  because conftest sets `auto_save_after_sleep=False`; every persistent
  agent's campaign-end hits exactly this combination. Fix hoists the auto-save
  into the public `sleep()`/`sleep_with_clustering()` wrappers after lock
  release; guard runs sleep in a thread with a bounded join so a regression
  fails instead of hanging the suite.
- **Adoption requires the full bio surface** (Exec-S6/bio-F5): pain_bus/
  hippocampus/nac now validated (a missing bus silently orphans pain learning).
- **Branch-6 honesty** (all three lenses): see the note in branch 6 above.
- Also folded: `MAXIM_ALLOW_BASH` not armed for adopted agents (Arch-S6),
  adopted AUT join 300s so full consolidation completes before restore/return
  (Exec-S5), `/new` recursion keeps the injection (Arch-S3), `Literal`
  consolidation typing (Arch-S8), server shutdown hook flushes the handle
  (bio-F9), `RunAccepted.session_id` semantics documented in the schema
  (Arch-S7), interactive-mode restore in `play_campaign`.

**Verified clean by ≥2 lenses:** double consolidation (loop-end full +
`handle.stop()`) is idempotent via the `_session_active` guard — the second
`on_session_end` no-ops and `stop()` doubles as the consolidation safety net
if the loop dies early; exactly ONE hippocampus + ONE NAc pain learner on the
adopted bus (topology mirrors the throwaway sim); executor wrappers stay
sim-local (persistent executor unwrapped after the run); resume-gate/
save-skip/agent_id sweep complete.

**Deferred follow-ups (recorded, not shipped):** campaign-declared fiction
provenance (branch-6 note); a no-LLM stub-campaign end-to-end integration run;
promoting the lease onto `ToolRegistry` as a context manager when a second
consumer appears; a run-status/ws surface correlating the console run id with
the sim's internal session_id; `app.state` for the server's handle singleton.

## Review + discipline

- **Two-lens pre-merge review** (Executor + Architecture, plus the bio-fidelity
  lens) — this touches the learning-attribution path; cross-confirmed findings are
  trusted. Re-run the round if the branch grows after review.
- **No band-aids:** if the tool-registry reconciliation (branch 4) gets ugly,
  surface the root cause rather than special-casing — the registry-ownership
  question may itself need a small refactor.
- `PYTHONPATH=src` in the worktree; `ruff format` + targeted pytest before commit.
