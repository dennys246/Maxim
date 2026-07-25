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

## Review + discipline

- **Two-lens pre-merge review** (Executor + Architecture, plus the bio-fidelity
  lens) — this touches the learning-attribution path; cross-confirmed findings are
  trusted. Re-run the round if the branch grows after review.
- **No band-aids:** if the tool-registry reconciliation (branch 4) gets ugly,
  surface the root cause rather than special-casing — the registry-ownership
  question may itself need a small refactor.
- `PYTHONPATH=src` in the worktree; `ruff format` + targeted pytest before commit.
