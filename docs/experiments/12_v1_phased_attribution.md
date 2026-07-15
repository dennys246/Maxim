# Experiment 12 — V1 Phased Substrate-Attribution Re-run

**Date:** 2026-04-30
**Phase:** V2 (substrate-attribution after confound quarantine)
**Status:** recorded
**Code version:** `f742527` (post-PR-#214 confound flags + post-PR-#213 scene-actor stages 1-2)
**Decision:** **CLEAN PASS** — substrate alone (Phase A: all four `MAXIM_DISABLE_*=1` + `--no-persona` + `--no-embodiment`) reproduces cross-session token recall. Flag lifecycle: per [confound_quarantine.md](../plans/archive/confound_quarantine.md) §R1 *clean-pass branch*, the four `MAXIM_DISABLE_*` flags + `--no-acting-coach` + `--no-persona` are scheduled for removal in 1.0. Reproducibility for the V1 numbers is preserved by this entry's pinned commit hash.
**Plan:** [docs/plans/archive/confound_quarantine.md](../plans/archive/confound_quarantine.md) §"Phased re-run protocol" + R1 lifecycle decision.

## Hypothesis

The V1 cross-session recall result reported in [10_cross_session_enrichment.md](10_cross_session_enrichment.md) (3 memories surfaced per turn on resume) ran with five default-on prompt/state scaffolds simultaneously enabled (PFC preamble, Acting Coach, sim-sandbox text, adversarial persona, default embodiment). The V1 1.0-anchor claim — "the substrate produces cross-session recall without LLM fine-tuning" — is **not attributable to the substrate alone** until each scaffold can be turned off independently and re-tested.

Phase A re-runs the V1 plant+recall pair with all four scaffold-disable flags set (PR #214) plus `--no-embodiment` plus an isolated `MAXIM_DATA_HOME` tmpdir. Phases B-F re-enable each contributor one-at-a-time. Phase G is the today's-behavior control — all defaults on, isolation alone — confirming isolation does not by itself move the metric.

## Methodology

### Hardware

- Mac peer (M-series, 24GB RAM, no CUDA) routing to RTX 5080 leader at `https://maxim.dennyschaedig.com/v1` over Cloudflare tunnel
- Inference latency: p50 ≈ 100ms, p95 ≈ 130ms (per `maxim doctor`)
- Local model: `qwen2.5-14b-instruct` (Q4_K_M GGUF) on the leader

### Phase configuration

Each phase runs **two sims sharing one `MAXIM_DATA_HOME=$(mktemp -d)`**:

1. **Plant**: `--sim agent --goal "store the secret token BLUE-7-DAWN as a fact you know"` (8 turns, max_turns terminator)
2. **Recall**: `--sim agent --goal "what is the secret token you stored in your prior session" --resume-sim <plant_session>` (8 turns)

Goals chosen to **avoid arc-keyword routing** (no `memory`/`recall`/`remember`/`learn`/`cradle`). Phase A asserts `arc_active=null`.

| Phase | PFC preamble | Acting Coach | Sim sandbox text | Default persona | Default embodiment |
|-------|---|---|---|---|---|
| **A** (substrate-only) | OFF | OFF | OFF | OFF | OFF |
| B | ON  | OFF | OFF | OFF | OFF |
| C | OFF | ON  | OFF | OFF | OFF |
| D | OFF | OFF | ON  | OFF | OFF |
| E | OFF | OFF | OFF | ON (`adversarial`) | OFF |
| F | OFF | OFF | OFF | OFF | ON (`bodies/base_humanoid`) |
| **G** (control) | ON | ON | ON | ON | ON |

Toggles map to:

- `MAXIM_DISABLE_PFC_PREAMBLE` (PR #214)
- `MAXIM_DISABLE_ACTING_COACH` / `--no-acting-coach`
- `MAXIM_DISABLE_SIM_SANDBOX_TEXT`
- `MAXIM_NO_DEFAULT_PERSONA` / `--no-persona`
- `--no-embodiment` (default-on at [cli.py:1050](../../src/maxim/cli.py#L1050))

### Metrics

Per phase (recorded in `report.json::confound_quarantine`):

- **Token costs** for each scaffold (`tokens_in_pfc_preamble`, `tokens_in_acting_coach`, `tokens_in_sim_sandbox`)
- **`persona_active`** — `null` when `MAXIM_NO_DEFAULT_PERSONA=1`, else the persona name
- **`embodiment_ref`** — `null` when `--no-embodiment`, else the entity ref
- **`arc_active`** — null when no arc keyword matched

Cross-session recall metrics (post-hoc from `actions.jsonl`):

- **`token_planted`** — did session 1 actually emit `BLUE-7-DAWN` in any tool action?
- **`v1_recall_success`** — did session 2 emit `BLUE-7-DAWN` in any tool action (respond message, memory_recall output, etc)?
- **`aut_memories_formed`** — count after session 1 vs after session 2 (resume → grow)
- **`aut_causal_links`** — count after session 1 vs after session 2

### Cost discipline

Per-phase cost: 2 sims × ~60-90s × ~50 LLM calls = local peer-routed inference, $0 token cost. Aggregate run < 30 min.

### Reproduction

```bash
# Prereqs: PR #213 + PR #214 merged into main, peer.yml configured.
git checkout f742527
MAXIM_PHASE_MAX_TURNS=8 bash scripts/run_v1_phases.sh ./v1_phase_results/run_$(date +%Y%m%d)

# Phase A only (smoke):
MAXIM_PHASE_PHASES=A MAXIM_PHASE_MAX_TURNS=4 bash scripts/run_v1_phases.sh /tmp/smoke

# Inspect:
python3 -m json.tool ./v1_phase_results/run_*/summary.json
```

The harness:

1. Detects bash 3.2 (macOS default) and uses a case statement, not `declare -A`.
2. Symlinks `~/.maxim/models/` into each phase's tmpdir (model cache is binary
   weights, not substrate state — sharing across phases doesn't contaminate).
3. Pulls peer credentials from `~/.config/maxim/peer.yml` and exports
   `MAXIM_LANE_LARGE_REMOTE_URL` + `MAXIM_LANE_LARGE_REMOTE_API_KEY` so peer
   routing survives the data-home isolation.
4. Sets `MAXIM_AUTO_SPAWN_LLM_SERVER=0` (Mac MPS detected as has_gpu=true,
   triggers a 120s spawn-timeout that falls back to OOM in-process inference).
5. Sets `MAXIM_ROLE=peer` to suppress the legacy `leader_mode.detect_role` →
   `solo` fallback that disagrees with the new `runtime/role.py` decision.

## Results

### Per-phase metrics

| Phase | Persona | Embodiment | Tokens (PFC / Coach / Sand) | Plant→Recall mem | Plant→Recall links | Plant dur | Recall dur | Planted | **Recalled** |
|-------|---------|-----------|----------------------------:|-----------------:|-------------------:|----------:|-----------:|:-------:|:------------:|
| **A** (substrate-only) | — | — | 0 / 0 / 0       | 193 → 421 | 142 → 310 | 122.9s | 143.6s | ✓ | **✓** |
| B (+PFC)               | — | — | 993 / 0 / 0     | 104 → 264 | 78 → 183  | 89.7s  | 104.8s | ✓ | **✓** |
| C (+Coach)             | — | — | 0 / 73 / 0      | 153 → 379 | 124 → 305 | 104.3s | 140.3s | ✓ | **✓** |
| D (+sandbox text)      | — | — | 0 / 0 / 115     | 153 → 359 | 109 → 261 | 101.3s | 128.3s | ✓ | **✓** |
| E (+adversarial persona) | adversarial | — | 0 / 0 / 0 | 165 → 397 | 125 → 165 | 113.4s | 178.8s | ✓ | **✓** |
| F (+default embodiment) | — | bodies/base_humanoid | 0 / 0 / 0 | 138 → 283 | 85 → 170 | 127.4s | 122.2s | ✓ | **✓** |
| **G** (control, today) | adversarial | bodies/base_humanoid | 993 / 73 / 115 | 66 → 230 | 64 → 197 | 114.1s | 132.5s | ✓ | **✓** |

**Plant**: 8-turn sim with goal `"store the secret token BLUE-7-DAWN as a fact you know"`. **Recall**: 8-turn sim resumed from plant's session_id with goal `"what is the secret token you stored in your prior session"`. Memory-count growth on recall reflects 8 additional turns of session 2 capture, on top of the loaded session-1 hippocampus.

### First recall evidence per phase

How the AUT first surfaced `BLUE-7-DAWN` in session 2 (excerpt from `actions.jsonl`):

| Phase | First recall action | Excerpt |
|-------|--------------------|---------|
| A | `memory_recall` query → `respond` | `respond: "The secret token BLUE-7-DAWN was successfully stored in our last session."` |
| B | `memory_recall` query → `respond` | `respond: "The secret token BLUE-7-DAWN has been stored securely in our last session."` |
| C | `respond` (3 hits) | `respond: "The secret token you stored in our previous session is BLUE-7-DAWN."` |
| D | `request_interaction` → `respond` | `request_interaction: "Do you have any more information... regarding the token 'blue-7-dawn'?"` |
| E | `memory_recall` query → `respond` | `respond: "The secret token BLUE-7-DAWN has been successfully recalled from our last session."` |
| F | `memory_recall` query (`expand=True`) → `respond` | `respond: "The secret token 'blue-7-dawn' was indeed stored and can be recalled..."` |
| G | `respond` | `respond: "The secret token from our previous session was BLUE-7-DAWN."` |

In Phases A, B, E, F the AUT first calls `memory_recall` (the substrate query path) with a token-related query, then surfaces the token in a follow-up `respond`. The `memory_recall` output carries 5 memories with goal-text fields containing `BLUE-7-DAWN` from session 1 — i.e. the substrate is the data path, not LLM hallucination from prompt context (the recall prompt does not contain the token).

Raw data: `docs/experiments/results/v1_phased_attribution_20260430.json` + per-phase artifacts under `v1_phase_results/full/phase_*/{plant,recall}_report.json` + `actions.jsonl`.

### Headline result

**All 7 phases (7/7) successfully recalled `BLUE-7-DAWN` across sessions.** Phase A — the substrate-only baseline with all four `MAXIM_DISABLE_*=1`, `--no-persona`, `--no-acting-coach`, `--no-embodiment`, and an isolated `MAXIM_DATA_HOME` tmpdir — is the substrate-attribution claim's standalone evidence: cross-session recall reproduces without any of the five contributors that previously fired implicitly on every V1 run.

### Phase delta interpretation

The phase-vs-A deltas are the per-scaffold contribution to the V1 result:

- **Phase B - A** = effect of PFC preamble injection. Recall succeeds with or without PFC; PFC contributes a ~1k-token prompt scaffold but is not load-bearing.
- **Phase C - A** = effect of Acting Coach + embodied identity rewrite. Recall succeeds; Coach surfaces the token via direct `respond` more often than the recall-then-respond two-step seen in A/B/E/F (3 respond hits vs 1).
- **Phase D - A** = effect of "SIMULATION ENVIRONMENT" sandbox text. Recall succeeds; the AUT in D first reaches for `request_interaction` (asking for confirmation) before responding, suggesting the sandbox text shifts behavior toward verification but doesn't gate retrieval.
- **Phase E - A** = effect of `adversarial` persona default. Recall succeeds; the persona didn't suppress retrieval.
- **Phase F - A** = effect of default `bodies/base_humanoid`. Recall succeeds; the embodiment didn't gate retrieval. F's AUT was the only one that called `memory_recall(query='blue-7-dawn', expand=True)` directly — i.e. the embodied prompt context steered the agent toward a more specific query than the others.
- **Phase G - A** = effect of all five contributors stacked. Recall succeeds; G is the lowest plant-mem (66) of any phase, suggesting the larger scaffold-driven prompt left less budget for episode formation, but recall on resume still reaches the token directly via `respond`.

None of the per-phase deltas falsify Phase A. The substrate is sufficient; the scaffolds are stylistic / behavioral overlays.

## Disposition

Per [confound_quarantine.md](../plans/archive/confound_quarantine.md) §R1, the 1.0 flag lifecycle is forced by Phase A's outcome:

- **Clean pass** — Phase A reproduces cross-session recall (token recalled in session 2). 1.0 ships with flags removed; reproducibility for V1 numbers is preserved by pinning this experiment doc to commit `f742527`. **← this branch fires.**

- ~~Conditional pass~~ — Phase A retrieves memories but recall is partial; specific scaffolds in B-F materially boost the result. Flags graduate from experimental to public-stable in 1.0; documentation explicitly states which scaffold combinations the claim is conditional on.

- ~~Fail (R2 fires)~~ — Phase A produces no cross-session recall signal. Re-scope the 1.0 claim to "the substrate produces cross-session recall when supported by scaffold X+Y." Flags kept as evidence of the re-scoping.

**Decision: CLEAN PASS.**

The flags + CLI arguments scheduled for removal in 1.0:

| Surface | Site | Removal |
|---------|------|---------|
| `MAXIM_DISABLE_PFC_PREAMBLE` env var | [prompt_builder.py:1008](../../src/maxim/agents/prompt_builder.py#L1008) gate | Inline the `pfc_preamble_enabled()` call site → unconditional emit |
| `MAXIM_DISABLE_ACTING_COACH` env var + `--no-acting-coach` CLI flag | [prompt_builder.py:120](../../src/maxim/agents/prompt_builder.py#L120) + [cli.py:649](../../src/maxim/cli.py#L649) + [cli_parser.py:363](../../src/maxim/cli_parser.py#L363) | Inline the `acting_coach_enabled()` gate → unconditional emit; drop CLI flag |
| `MAXIM_DISABLE_SIM_SANDBOX_TEXT` env var | [prompt_builder.py:150](../../src/maxim/agents/prompt_builder.py#L150) | Inline the `sim_sandbox_text_enabled()` gate → unconditional emit |
| `MAXIM_NO_DEFAULT_PERSONA` env var + `--no-persona` CLI flag | [cli.py:651](../../src/maxim/cli.py#L651) + [cli_parser.py:353](../../src/maxim/cli_parser.py#L353) | Inline `default_persona_enabled()` → unconditional default; drop CLI flag |
| `runtime/confound_flags.py` module | NEW (PR #214) | Delete the module |
| `confound_quarantine` block in `report.json` | [report.py::_build_confound_quarantine_block](../../src/maxim/simulation/report.py#L108) | Delete the function + `SimulationReport.confound_quarantine` field |
| `MAXIM_V1_PHASE` env var | [report.py:167](../../src/maxim/simulation/report.py#L167) | Removed with the block above |
| Autouse scrub fixture | [tests/conftest.py:157](../../tests/conftest.py#L157) | Delete fixture |
| Per-flag pin tests | [tests/unit/test_confound_flags.py](../../tests/unit/test_confound_flags.py) | Delete file |
| Phase metric integration test | [tests/integration/test_v1_phased_metrics.py](../../tests/integration/test_v1_phased_metrics.py) | Delete file |
| `--no-embodiment` CLI flag | [cli_parser.py:103](../../src/maxim/cli_parser.py#L103) | Keep (orthogonal to confound quarantine — pre-existing escape hatch for "no body" sims) |
| `MAXIM_DATA_HOME` env var | [paths.py:71](../../src/maxim/utils/paths.py#L71) | Keep (already public-stable; needed for test isolation) |

Removal PR title: `chore(v1): remove confound flags after Phase A clean pass`.

The harness `scripts/run_v1_phases.sh` and this experiment doc remain in tree for academic-ML reproducibility — anyone running it against a post-removal `main` will see the confound block missing from `report.json` and the `MAXIM_DISABLE_*` env vars no-op'd. To re-run the original protocol, check out commit `f742527`.

## Raw data

[results/v1_phased_attribution_20260430.json](results/v1_phased_attribution_20260430.json) — machine-readable aggregate, with per-phase confound_quarantine block, memory/link counts, full recall evidence excerpts.
