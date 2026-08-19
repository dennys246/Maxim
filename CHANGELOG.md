# Changelog

All notable changes to pymaxim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note (2026-05-11, updated 2026-06-06):** versions 0.6.0 → 0.8.1 shipped to PyPI without
> matching CHANGELOG entries. The summaries for that window live in
> [docs/plans/v1_refinement.md](docs/plans/v1_refinement.md) and the
> [HTML roadmap](html-guides/maxim-roadmap.html). The 0.9.0 entry below
> picks up the convention again.

> **Note (2026-08-07, corrected 2026-08-19):** versions 1.0.1 → 1.0.6 were version
> bumps in-repo WITHOUT CHANGELOG entries, git tags, **or PyPI publication — PyPI's
> latest `pymaxim` release is 1.0.0**; the original form of this note ("shipped to
> PyPI") was itself false. The entries below were reconstructed from the git history
> during the 1.1 release-truth pass (roadmap item 2); annotated tags
> `v1.0.1`–`v1.0.6` were created at the verified version-bump commits at the same
> time. Publishing a current release (or explicitly deciding not to) is part of
> roadmap item 16.

## [Unreleased]

### Added

- **1.1 release-readiness ledgers** — repository scorecard plus a verified
  stable-API/architecture/test defect cluster (D15–D20), with explicit 1.1 versus
  1.1.x dispositions.
- **Exp 49 two-joint centering** — harness + three arms complete: H1 supported
  (head-only 0/10 vs two-joint 4/10), H2 1.00 dense AND sparse (`--speech-density`
  knob for the pre-registered sparse arm), H3 1.00 (B) / 0.969 (C after the
  timestamp-tie credit-attribution correction — gates clean). Key fold finding:
  the 180° false equilibrium traps a credit-following substrate on rear sources.
  (#463, #464, #466)
- **Roadmap 1.1 → 1.3** from the four-lens review — 1.1 "Sensorimotor" cut line
  (zero new mechanisms), 1.2 Oasis + Hivemind, 1.3 perception fabric + reflex
  tier; hardware note: motors 2+3 were broken for the entire 1.0+ era. (#471)
- **Cross-modal perception fabric (1.3) + three-factor credit assignment** design
  plans. (#468)

### Changed

- **1.1 scope reconciled around release closure** — D13/D14 liveness, stable API
  contracts, hermetic tests, atomic NAc+EC invalidation, architecture-audit
  enforcement, remaining heartbeat evidence, and release truth are the cut. Oasis
  and Hivemind moved to gated 1.2 work. Agent guidance ratified in the INVERSE of
  the first draft: `CLAUDE.md` stays the canonical (CI-linted) core; `AGENTS.md`
  rewritten as a thin provider-neutral adapter (see DECISIONS.md 2026-08-19).

### Fixed

- **EC performance-contract documentation** — `EntorhinalCortex` now distinguishes
  LSH-indexed signature queries from the substrate hot path's exact O(Nd) centroid
  scan; no runtime mechanism changed.
- **`ec_merge` silent cross-space corruption** — vectors from different encoder
  spaces (384-dim vs 768-dim, same modality tag) were compared over the shorter
  prefix by `zip` truncation and MERGED when the partial cosine cleared the
  threshold. Dimension mismatch now reads as similarity 0.0 (right-side node
  inserts as its own node — non-destructive). Blocker #1 for Oasis ingestion.
  (#467)
- Confirmation-branch `UnboundLocalError` crash; fell-short orient note now
  names real registered actions. (#462)
- CI queue starvation — in-flight runs per ref are superseded instead of
  competing for runners (jobs previously sat unassigned and died at 15m01s as
  false-red `failure`). (#470)

## [1.0.7] - 2026-08-19

### Fixed

- **Simulation planning liveness (D13/D14)** — failed or non-executable
  planning turns now retry boundedly or terminate with a typed status instead
  of idling forever. Recovery follows the exact `LLMWorker` job lifecycle,
  separates planning failures from worker-transport failures, corrects
  unregistered-tool retries, and keeps spinner state truthful.

## [1.0.6] - 2026-08-04

### Added

- **SEM motor binding Phase 2** — measured relief credit routed to the AUDIO
  cluster: the motor backend measures the azimuth transition across each real
  body turn (frame-corrected, staleness-gated, never fabricated) and the bio
  layer credits drive relief from the measurement, not from execution. (#461)

## [1.0.5] - 2026-08-04

### Added

- **NAc cross-session persistence** — save + load + decay-on-load land together,
  with the stable-hashing prerequisite (5 sites moved off randomized builtin
  `hash()`, two-process regression guard; persisted files carry `hash_scheme`
  markers). NAc + EC persist as a pair; `apply_decay=False` at `--resume-sim`
  and read-only observers; the orchestrator NPC is write-but-don't-read. (#446)
- **Live audio-orient wiring (Stages 0b–4b)** — DoA feed into the standard
  embodied runtime as a percept lane; phantom credit mill found and guarded
  (relief credit deferred to the motor bridge). (#447)
- **SEM motor binding Phase 1** — orient affordances become REAL body turns on
  live hardware via the pre-existing `attach_backends` seam; Phase 1 withholds
  relief credit (`drive_credit_withheld`), refuses on unreadable pose. (#460)
- **`focus_on_sound`** — zero-numeric closed-loop audio orienting: the LLM
  decides WHETHER to attend, the tool owns HOW FAR, reading the live DoA at
  execution time; honest per-situation outcome notes name real tools. (#455)
- `robots.yaml` `audio_salience` / `audio_novelty` operator escalation knobs
  (#451); `ear_map.py` DoA characterization sweep (#457).

### Fixed

- **`body_yaw` joint-index phantom frame** — the SDK >= 1.5 joint vector is
  `[body_yaw, *stewart_legs]`; a zenoh-era index-6 read returned a stewart LEG
  angle, corrupting every body-relative yaw consumer (DoA capture stamps,
  focus_on_sound aim, bounds-learner coordinates). Plus honest `focus_on_sound`
  readback (measured, not promised). (#459)
- **Daemon `automatic_body_yaw` off by default** — the daemon silently rotated
  the body behind the runtime (−25° observed), rotating the frame every yaw
  computation lives in. Maxim's loops own the yaw axis. (#456)
- **MoveTool gaze interface** — hardware-verified sign conventions; `target_x`
  now TURNS the head (pre-fix it translated mm-scale, so the LLM fell back to
  raw yaw whose sign it guessed — mirror-image orienting). (#454)
- Record-time cost metering keys on `treat_as_cloud` (was the provider
  `pricing_required` flag — False on cloud-URL lanes → silent $0 spend);
  fail-closed local-only banner; canonical model-path resolution. (#458)
- `robots.yaml` actually drives the live robot connection (#448); capability
  truth for media flags under `no_media` (#452); httpx INFO spam at the ~7 Hz
  DoA poll silenced (#450); live deep-dive fold — wake on sound, kill phantom
  gaze, embodied identity (#453).

## [1.0.4] - 2026-07-29

### Fixed

- **Recovered the broken #435 squash-merge** — `main` shipped only the first
  commit of the transition-based drive-pain PR, a design its own review round
  had refuted: a boolean band latch silenced repeat harmful affordances →
  `learn_success` flipped True → **positive** cluster reward for the harmful
  source, inverting the Exp 42 safe-vs-harm mechanism — with GREEN CI, because
  the guard test was in the unmerged fold. Recovery landed the channel-split
  design (direct FailureEvent channel state-based; PainBus channel
  severity-latched on the entity). (#443)
- Recovered 6 commits #442 left behind + console backend identity + diagnose
  fix + gate-log truth. (#445)

### Added

- Console rest mode + actionable sim pointer (#442); lean aarch64 Pi extra
  (`pi`) with uv-based sound-resolution guard, recovering orphaned #440 (#441).
- Process rule: **a review round is not complete until its fold commits are ON
  the merge target** (#444) — the lesson from #435/#440/#442.

## [1.0.3] - 2026-07-28

### Added

- **Substrate learns from lived experience, Phase 1** — the cluster-reward
  WRITE path closes at the `record_outcome` choke point in llm-primary:
  intero cluster encoded at action time, credit is DRIVE-RELIEF-ONLY
  (tool-success floor suppressed; reinforcement beats decay ~45×). (#436, #437)
- Console talk mode + launcher-completion seams. (#438)

### Fixed

- Transition-based drive-pain — fire on band entry, latched per entity/drive
  (#435; **note: this squash-merge was broken and shipped a refuted design —
  recovered in 1.0.4 by #443**).

## [1.0.2] - 2026-07-28

The largest window of the 1.0.x line (~56 PRs, #381–#434): the operant
orienting arc from scripted probes to live-hardware learning, plus the
maxim-pulse Console seam set.

### Added

- **Orient line (Exp 43 → 48):** operant gaze + substrate-generalization probes
  (#386); shared orient-to-center backbone (#387); Exp 45 Layer 1 live learning
  loop — all arms EARNED, direction + magnitude on real hardware (#395); Exp 45d
  magnitude replication + full-policy cross-session transfer (#413); Exp 45e
  population-vector readout resolves far-bin magnitude starvation (#431);
  productive orienting — hear, localize, turn toward sound (#403); operant
  orient — a mother teaches, a crèche pools, the system habituates (#410);
  **Exp 48 GRADUATE** — the extero/intero seam lifts the embodied cradle_mother
  sim off chance (taught 0.875 vs 0.448) (#411, #412).
- **Perception:** pipeline-placement type layer (#382–#383; later marked
  Dormant — zero production callers); DoA-consumption audio front-end (#385);
  thalamic relay — composite + side-channel + audio/DoA recognition in the
  loop (#402); percept-ingress organizing frame (thalamus & hypothalamus)
  (#401).
- **Live Reachy:** WS-era migration + embodiment docs suite (#392); Track 1 —
  SEM body wired into the live runtime (opt-in body + drift tick + body_state
  prompt) (#400); off-robot REST DoA reader (#399); standalone RTSP streamer
  (#388).
- **maxim-pulse Console seams:** `maxim serve` skeleton + OpenAPI facade
  contract (#416); FIT substrate footprint measurement + Pi runbook (#415);
  PROBE (#419), cloud + mesh SETUP (#426, #424), RECALL (#425), EVENT /ws
  bridge (#434), HANDLE persistent-agent campaign injection + consolidation
  stop contract (#428, #427); local-first distribution model pinned (#421).
- **Exp 44 counterfactual harness** — trajectory-matched ablation (#429).
- 2026-07-15 deep plans audit — 12 concluded plans archived, 19 deferred with
  revive triggers, 1000+ stale links fixed (#394).

### Fixed

- **P1 range-aware `_normalize_value`** — the signed-sensor FOLD (center 0.0
  aliased hard-left −1.0): orient full-range 0.847 → 1.000. (#409)
- **Orient motor-credit re-landed with the value-progress fix** — credit by the
  SIGN of drive value-progress toward comfort, not tool-execution success; the
  original #405 pain-based version floored Exp 42 (60→8 warmth contacts) and
  was reverted (#407) before the corrected re-land (#408).
- **Post-merge orient review fold — 7 blocking, including a production
  head-frame bug** (`ReachyMiniController.goto_target` passing `head=None`)
  (#397); controller passed `minimum_jerk` where the SDK only accepts
  `minjerk` — every controller-driven motion crashed on SDK >= 1.5 (#398).
- 1.0 regression-guard promotion — CI enforcement + live-violation fixes +
  stale citation repair (#389); embodiment truth restoration (tick invariant
  corrected, dormancy markers, dead-code deletion) (#390).

## [1.0.1] - 2026-06-23

### Added

- **Exp 42 substrate-primary preference setup** — drive-gating,
  delta-attribution (B8), introspection filter; the run that led to the
  substrate-primary safe-vs-harm GRADUATE (#6). (#380; the review fold +
  frozen GRADUATE results were stranded by the squash-merge and landed at the
  top of the 1.0.2 window as #381.)
- Exp 41 substrate-primary exploration + cradle harm-wiring + harness
  (result: VOID). (#379)

## [1.0.0] - 2026-06-17

The 1.0 release: **a bio-inspired cognitive harness for LLM agents** with
cross-session learning without fine-tuning.

### Added

- **Lane capability/placement split** — placement (`Origin{LOCAL,CLOUD,PEER}`)
  becomes a first-class typed axis orthogonal to capability tier, riding
  `LLMRouter`'s existing failover; declarative `config.json::lanes.<tier>.placement`;
  legacy configs byte-identical via the derive oracle. (#356–#362)
- **Exp 37/38/40 cross-model campaign** — the Goldilocks result: substrate
  signal lands only where LLM priors leave headroom; Exp 40 counter-prior
  dominance at Qwen-32B folded into the 1.0 announcement. (#363–#377)
- Prompt caching for cloud backends (byte-stable system prefix, ~38% ITPM
  reduction) (#349–#350); optional-dependency loud failures via
  `utils/optional_deps.py` (#341); CC13 auth format freeze (#375);
  reasoning-model timeout support (#369).

### Changed

- **Exp 37 row 1 reframe (2026-06-07, PR #344).** The 2026-06-06 Exp 37 Qwen14B fire
  completed 60/60 records with verdict `PARTIAL — investigation gate`. The substrate-
  transfer behavioral claim under LLM-AUT is not statistically detectable at the
  pre-registered N=5 threshold (fire_pit primary Δ = −0.06 SD vs threshold +1.0;
  ablations 0/3 PASS with 2 of 3 OVERSHOOTING past Arm A — the diagnostic signature
  of LLM prior dominance over substrate signal). [Tier 1 row 1](docs/plans/behavioral_graduation_candidates.md)
  splits into 1a "Cross-session memory persistence" (EARNED via [Exp 10](docs/experiments/10_cross_session_enrichment.md))
  and 1b "Cross-session behavioral delta under LLM-AUT" (PARTIAL — investigation gate).
- **1.0 framing pulled back from "substrate drives action selection" to "bio-inspired
  cognitive harness for LLM agents."** Per the 1.0 commitment line's explicit-
  retraction discipline in `behavioral_graduation_candidates.md`, the strong substrate-
  drives-behavior claim is pulled. The substrate provides cross-session infrastructure
  (memory, valence, causal links, drives) that LLM-driven agents use; substrate-driven
  action selection independent of the LLM is post-1.0 research direction via Exp 38
  substrate-primary work / Oasis architecture. README + Tier 1 row 1 + the Exp 37
  results doc updated together to honor the retraction discipline.
- **README "What Makes This Different" table** rewords the "Behavior emerges from
  learned experience" row to "LLM action selection augmented by substrate-derived
  context (memory recall, causal predictions, valence, drives)" — matches the empirical
  reality of how Maxim agents actually use the substrate, anchored to Exp 37 evidence.

## [0.9.3] - 2026-06-06

### Added

- **Loud optional-dependency failures** (`src/maxim/utils/optional_deps.py`). An audit
  found 45+ optional-import sites using four inconsistent behaviours (raise, warn-and-continue,
  warn-and-fallback, or fully silent). The new `optional_deps` module centralises this with
  two functions: `require_optional_dependency(import_name, extra=, feature=)` raises a typed
  `OptionalDependencyError` (subclass of `ImportError`) with an actionable
  `pip install pymaxim[...]` message for explicitly-requested-but-missing backends;
  `warn_optional_fallback(import_name, extra=, feature=)` logs a one-time WARNING and returns
  `None` for graceful-degradation paths. Motivated by a 2026-06-05 incident where the
  `anthropic` package was simply not installed: `_AnthropicBackend` swallowed the import
  error and returned None, the router treated every response as a transient hiccup, and the
  entire sim completed with `cost=$0` and every action an `_llm_unavailable` fallback — the
  missing backbone was invisible. Now all four backends (`anthropic`, `openai`-compatible,
  `llama_cpp`, `transformers`) raise on startup if their required SDK is absent and the
  profile was explicitly requested. 229 LOC core + 219 LOC tests.

### Changed

- `LLMWorker` (via `agents/llm_worker.py`) calls `require_optional_dependency` at
  construction time for the resolved backend profile, so the failure surfaces during agent
  startup rather than on the first inference call.

## [0.9.2] - 2026-06-05

### Added

- **Config-unification: `~/.config/maxim/config.json`** (PR #318). Single-source operator
  config absorbs ~22 daily-use `MAXIM_*` env vars onto one file. Precedence chain:
  CLI > env > config.json > builtin defaults, surfaced by
  `runtime/config_loader.py::resolve_setting`. New `maxim config` subcommand verbs: `get`,
  `get <field-path>`, `set <field-path> <val>`, `list`, `path`, `edit`. Role detection
  unified to a seven-rank single source of truth in `runtime/role.py` (env var → config.json
  → mesh.yml → cloudflared → peer.yml → `--llm` local → default leader). Shadow/convergence
  override logging surfaces mismatches. API-key references in config accept file paths or
  `keyring:<service>:<account>` URIs; inline plaintext keys are rejected at load time.
  peer.yml auto-migrates to config.json on first startup when cloudflared is absent.
- **`maxim doctor` "Resolved Config" section** (PR #318 C5, PR #322, PR #327). New section
  in `maxim doctor` output shows all operator-configurable fields with their resolved value
  and source (CLI / env / config.json / default). Includes cross-platform VRAM / context-fit
  row and aggregated legacy env-var migration row.
- **`maxim model add|remove|list` CLI verbs** (PR #314). `src/maxim/models/model_cli.py`.
  `add` appends an entry to `~/.config/maxim/profiles.yml`; `remove` deletes by name;
  `list` prints all known profiles (bundled + user). Doctor check surfaces profiles.yml
  parse errors at startup.
- **User-profile YAML loader** (`~/.config/maxim/profiles.yml`, PR #314 L2). Operator-
  authored profiles merge into the bundled profile table at load time.
- **Three new bundled profiles** (PR #314 L1): `qwen2.5-32b`, `llama-3.1-70b`,
  `mixtral-8x7b` with auto-detect VRAM thresholds.
- **Hivemind shareability infrastructure** (`src/maxim/hivemind/`, PRs #305–#311).
  Four-PR track: (A) provenance + substrate-domain + fan-in-contributors fields on
  `CausalLink` and EC nodes; (B) `nac_merge` / `ec_merge` Bayesian-aggregation pure
  functions in `hivemind/merge.py`; (C) identity-bearing concept detection in
  `hivemind/identity.py`; (D) substrate snapshot bundle format (ZIP + manifest, no episodes
  by design) and `maxim substrate export <out.zip> | import <in.zip> | inspect <in.zip>`
  CLI verbs. Merged-link/node provenance uses the `_consensus` reserved namespace; identity-
  quarantine uses `_identity`. Source contributor IDs in the `_*` namespace are rejected.
  `ec_merge` respects `frozen_centroid_modalities` to prevent cross-contributor centroid
  drift. The `import` verb extracts only — no auto-merge into a live system.
- **`MAXIM_NAC_REWARD_BIAS_DISABLED` env var** (PR #307). Gates
  `NAc.distribute_reward`, `decay_reward_biases`, and `get_agent_tool_biases` as no-ops.
  Used for Exp 37 ablation arm 3 (does cross-session behavioral delta come from bio-learning
  or from LLM in-context recall?). Set at NAc construction time; changes after construction
  are not picked up.
- **LLM timeout scalability** (PRs #320–#323):
  - **TTFT keepalive emitter** (PR #320). Leader proxy emits `: keepalive\n\n` SSE comment
    frames every `MAXIM_PROXY_KEEPALIVE_INTERVAL_S` (default 30s, clamped 5–90s) during
    upstream time-to-first-token. Prevents cloudflared tunnel idle-timeout (≈100s) from
    closing the stream before the first token arrives on 30B+ models.
  - **Per-tier inference timeout** (PR #321). `MAXIM_LANE_<TIER>_TIMEOUT_S` env var family
    (tier names: `LARGE`, `MEDIUM`, `SMALL`) and matching `lanes.<tier>.timeout_s`
    config.json field. Resolved via `resolve_setting`; threaded into the backend via
    `LaneConfig.remote_timeout_s`.
  - **Context-overflow admission gate** (PR #321). Leader proxy returns HTTP 413 with a
    typed `{"error": {"code": "context_overflow", ...}}` body for requests whose estimated
    prompt + `max_tokens` + overhead exceed the upstream context window. Character-based
    estimator (`len / 3.5` chars/token). Safety margin via
    `MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS` (default 256, clamped 0–4096). Gate controlled by
    `MAXIM_PROXY_CONTEXT_ADMISSION` (default on when `MAXIM_LLM_N_CTX` is resolvable).
  - **`resolve_setting` auto-load fix** (PR #323). `resolve_setting` was silently returning
    the builtin default when called without an explicit `config=` kwarg. Now auto-loads the
    canonical config on every call, so caller discipline is no longer required.
- **Stall detector timeout-awareness** (PR #324). `runtime/llm_call_registry.py`
  in-flight call registry tracks active calls + byte-arrival times per tier.
  `runtime/stall_threshold.py::compute_stall_threshold` derives per-tier thresholds from
  lane timeout config. Orchestrator stall detector suppresses nudges during legitimate
  inference; `oldest_byte_silence_s(tier=...)` gates the TTFT window.
- **EC + ATL state persistence in sim reports** (PR #248). `aut_ec.json` and
  `aut_atl.json` written to `~/.maxim/sim_reports/{session_id}/` at session end. Used for
  cross-session substrate resume.
- **`cluster_reward_bias_decay_tau` split from `reward_bias_decay_tau`** (PR #267).
  Wire-A's `_cluster_reward_bias` now decays with its own tau (`NACConfig.
  cluster_reward_bias_decay_tau = 300.0`) independent of `reward_bias_decay_tau = 50.0`.
  The 50.0 default was sized for EC threshold modulation, not multi-turn substrate-voice
  annotation.
- **Exp 37 cross-session graduation infrastructure** (PRs #304, #313, #315). Pre-
  registered experiment; cross-session benchmark harness (`scripts/benchmark_cross_session.py`,
  945 LOC) + analyzer (`scripts/analyze_exp37.py`, 920 LOC). 6 arms × 2 scenarios ×
  5 trials. Singleton spawn guard + preflight check so the harness can run from the leader
  machine using local Qwen14B.
- **Mesh perception transport 1.0 prep** (PR #329, C10). `Percept.to_wire_dict` /
  `from_wire_dict` wire format; substrate fields (`embedding`, `substrate_node_id`) excluded
  from wire; non-blocking `PerceptSource` protocol contract reserved. Full transport ships
  in 1.1.

### Fixed

- **SEM `self_effect` / `target_effect` applied before `evaluate_failures`** (PR #316).
  Pre-fix, `ModulatorAffordanceTool.execute` ran `evaluate_failures` before applying sensor
  deltas, so affordances that write to sensors and then fail (fire-pit touch → arms.thermal
  spike → burn failure) never triggered the pain cascade.
- **`get_version_info` recognises worktree `.git` file form** (PR #306). Worktrees store
  `.git` as a file, not a directory; the old code assumed a directory and returned an empty
  version string.
- **`maxim peer` reports now surface served model name and backend routing** (PR #325).
  Previously, peer status reports showed only the URL; now they include the active model
  and whether inference is routing through llama-cpp or a cloud provider.
- **`size_gb=None` in user profiles no longer crashes `ensure_available`** (PR #318 fold).

### Backward compatibility

- `NACConfig.cluster_reward_bias_decay_tau` is a new optional field with default `300.0`;
  existing serialized NAc state loads without change (missing field → default).
- `Percept.to_wire_dict` / `from_wire_dict` are additive; existing `to_dict` / `from_dict`
  paths are unchanged.
- All `maxim config`, `maxim model`, and `maxim substrate` subcommands are additive; no
  existing flags renamed or removed.

## [0.9.1] - 2026-05-25

### Added

- **Stage 0b + 0c telemetry** (PR #254). `agent_id` and `session_id` threaded into every
  action JSONL record via `RequestContext`. `entity_class` field added to MOTOR/PERCEPT sim
  events. NAc snapshots written at session boundary (not only at final shutdown). `_format_version`
  bump on action JSONL per the CC1 contract.
- **Stage 0d: `MAXIM_EC_TRACE_ACTIVATIONS` per-tick instrumentation** (PR #246). Gated
  `sim_ec_activation` JSONL events from `EntorhinalCortex.pattern_complete_or_separate`.
  Fields: `agent_id`, `tick`, `active_node_id`, `activation_strength`, `modality_tag`,
  `modality`, `is_new`. Off by default.
- **Wire-A: cluster-bias annotation prompt section** (PR #253). Agent loop now renders a
  `cluster_bias_annotations` section in the LLM prompt when `NAc._cluster_reward_bias` has
  non-zero entries. Controlled by `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION` env var (default on).
- **Wire 3: embodiment-state → tool filter** (PR #255). Active sensor readings gate which
  SEM affordance tools are offered to the LLM. Tools whose `requires` precondition is unmet
  by the current body state are suppressed from the tool window.
- **Wire 2: Pavlovian percept aversion** (PR #256). New per-percept aversion accumulator on
  NAc (`_percept_aversion`), keyed by percept identity. Negative-valence episodes from a
  percept type suppress its future tool-window visibility. New JSON key on NAc serialisation;
  `GatingContext.learned_aversions` field reserved.
- **Wire 1: risk-sensitive action annotation** (PR #257). Variance-band felt-sensation
  annotation on tool descriptions in the LLM prompt. Reads `CausalLink.variance_estimate`
  (Welford online algorithm on the binary reward signal) via `NAc.get_action_risk_profile`.
  Controlled by `MAXIM_DISABLE_VARIANCE_ANNOTATION` env var (default on).
- **EC text-modality centroid drift fix** (PRs #259–#264). Five-phase fix:
  `ECConfig.pattern_complete_threshold` raised from 0.40 to 0.44 for the `text` modality,
  eliminating progressive centroid drift under sequential streaming. Roy-2c rerun: behavioral
  signal unchanged, structural fragmentation reduced 79% (`cluster_reward_bias_l2` 2.566 →
  0.535).
- **`NAc.get_threshold_overrides` base parameterized** (PR #263, Phase 3.5). Callers pass
  the live EC threshold so the override tracks the active EC instance rather than a hardcoded
  0.44 fallback.
- **`cluster_reward_bias_decay_tau` split** (PR #267). See 0.9.2 entry; shipped into the
  0.9.1 line after the tag.

### Fixed

- **Roy-5 H1C boundary tracks EC default** (PR #262). `H1C_LOWER_BOUND` in
  `tests/unit/test_roy_5_cosine_localization.py` was hardcoded to the old 0.40 default;
  now tracks `ECConfig.pattern_complete_threshold` so the test doesn't silently pass at the
  wrong boundary after any future threshold change.

### Backward compatibility

- **`ECConfig.pattern_complete_threshold`** changed default 0.40 → 0.44. This is a behaviour
  change for existing EC instances that relied on the default. Sessions that persist EC state
  and resume will use the new threshold from the first resumed turn.
- **`NAc._percept_aversion`** is a new optional JSON key. Pre-0.9.1 snapshots load to an
  empty dict with no warning.
- **Wire-A, Wire 1, Wire 3** are on by default; Wire 2 is on by default. All four wires can
  be disabled via their respective env vars for ablation experiments.

## [0.9.0] - 2026-05-11

### Added

- **Roy three-arm iteration runner (R1–R5).** Long-horizon persona-convergence harness. `maxim roy run <spec.yaml>` primes substrate via a multi-stage curriculum, runs the same held-out test across three arms (substrate-primed neutral / blank persona-injected / blank neutral), and reports pairwise substrate divergence (`reward_bias L2`, `cluster_reward_bias L2`, hippocampus episode + valence deltas, ATL concept Jaccard). Sister subcommands `maxim roy diff <a> <b>` and `maxim roy log <iter>` provide ad-hoc diffs and idempotent protocol/iteration-log regeneration. New modules: `src/maxim/simulation/roy_runner.py`, `src/maxim/simulation/curriculum_runner.py`, `src/maxim/analysis/substrate_diff.py`, `src/maxim/analysis/roy_log.py`, `src/maxim/roy/cli.py`. Methodology: [docs/plans/persona_convergence_crucible.md](docs/plans/persona_convergence_crucible.md).
- **G3 — fail-fast LLM pre-flight probe** in `run_roy_iteration` (PRs #235, #238). Roy iterations chain ~5 sims back-to-back; if the configured `large` lane is unreachable the iteration grinds out ~10 min of static-fallback narration with `dispatch_exhausted` on every call. Probe resolves `MAXIM_LANE_LARGE_REMOTE_URL` env var first, then `~/.config/maxim/peer.yml`, then skips for local/cloud setups. One HTTP call only (Plan 3 R2.5 invariant); aborts with `result.aborted_at = "preflight"` + persisted `result.json` + `summary.md` in ≤3.3s when the leader is dead. 10 regression tests.
- **G4 — substrate-primary cluster_id reward-feedback wire** (PRs #236, #237). Closes the wire Track 2 (commit `6d0e4a7`) deliberately deferred: `LLMProposal.cluster_id` field carries the active EC interoception cluster from `propose_via_substrate` to outcome recording; `record_outcome(..., cluster_id=...)` calls `NAc.update_cluster_reward(agent_id, cluster_id, sig, ±1.0)`; all 6 `_record_outcome` call sites + `execute_parallel_actions` thread it through; `NAc.dump`/`load_state` serialise `_cluster_reward_bias` under a new JSON key; `substrate_diff.NacDiff` surfaces `cluster_reward_bias_{available,l2,top_deltas}` so Roy `result.json` carries the headline metric. Empirically validated: live Roy-0 re-run produced `cluster_reward_bias_l2 = 2.4587` on A-vs-blank pairs (≈11.6× blank-vs-blank noise floor). 6 regression tests.
- **Roy summary rendering of `cluster_reward_bias`.** `roy_runner._format_summary` now emits `NAc cluster_reward_bias L2=… (N keys differ)` on its own line so `summary.md` carries the headline metric, not just the (necessarily 0) `reward_bias_l2`.
- **CLI documentation refresh** (PR #239). `docs/user/cli-reference.md` gains `--aut-mode` flag entry plus a full "Roy Harness" section documenting `run`/`diff`/`log` subcommands, spec shape, preflight semantics, and examples. `docs/reference.md` adds `src/maxim/analysis/` and `src/maxim/roy/` module entries. `docs/index.md` adds a Roy Harness link to the Architecture & Modes table.
- **HTML guide refresh** (PR #239). `html-guides/maxim-overview.html` nav now links the 10 previously-orphaned guides (semantic-memory, component-library, deliberation, concept-decomposition, tools, prompt-system, agent-mesh, dm-campaigns, experiments, benchmarks) in a three-row layout. `html-guides/maxim-roadmap.html` "Path to 1.0" ASCII updated through v0.8.0 + Roy + G3/G4 with empirical numbers. `html-guides/maxim-substrate-primary.html` gains a new "Roy harness — how we measure substrate convergence" section with the three-arm methodology table and the Roy-0 empirical results.

### Backward compatibility

- **`LLMProposal.cluster_id: str | None = None`** is an optional dataclass field added at the end of the frozen `LLMProposal` — CC3-compatible non-breaking. LLM-primary proposals leave it `None`.
- **`aut_nac.json::cluster_reward_bias`** is a new optional JSON key. Pre-G4 snapshots (no field) load to an empty dict; the loader emits no warning. Field key format joins `(agent_id, cluster_id, tool_signature)` with `\x1f` (ASCII unit separator) so tool signatures containing `:` round-trip cleanly.
- **`record_outcome(..., cluster_id=None)`** is a no-op for the cluster-update path — the LLM-primary tool-outcome path stays bit-identical.
- **No breaking changes to public API surface.** No deprecation warnings added in this release; C4/C5/C6 deprecation cycle (per `docs/plans/v1_refinement.md`) remains scoped to 1.0.

### Why bump minor

This is a feature release (Roy harness + substrate-primary reward feedback) that's strictly additive to a working v0.8.x install. Per semver and the project's "any change that affects runtime behavior, CLI interface, or peer/leader protocol" guidance, this earns a minor bump rather than a patch.

## [0.5.0] - 2026-04-19

### Added

- **B4 Replanning — all 3 stages shipped (1.0 gate closed).** Failure diagnosis with prior-attempt retrieval via hippocampus episodes, Jaccard distance metric for structural novelty, anti-repetition prompt constraint. Blind A/B validation: treatment (replanning) 100% vs control (no replanning) 0%, mean Jaccard 0.894. 48 tests across 3 test files.
- **P6 Extinction.** `DependencyGraph.decay_edges()` — multiplicative Hebbian decay with pruning. Beats LRU baseline across 10 seeds. 9 tests.
- **P8 Sleep Replay.** `memory/sleep_replay.py` — offline memory consolidation. Episode ranking by NAc reward_bias + valence. Replay re-fires `apply_hebbian_on_close` with consolidation multiplier. F1 improves vs no-replay control across 10 seeds. 13 tests.
- **F2 AgentFactory CLI migration.** `AgentFactory.create_full_agent()` composes `build_bio_stack` + `build_executor` + `FearGatedExecutor`. CLI non-sim bootstrap (~100 lines) replaced with one factory call. `AgentConfig` extended with `with_bio_stack`, `with_executor`, `with_pain_bridge`, `with_fear_gate`, `embodiment_ref`. `AgentInstance` extended with `bio_stack`, `pain_bus`, `embodiment`. 10 new tests.
- **`planning/structural_diff.py`** — Jaccard distance on action sequences for plan comparison. Pure utility, no agent/memory/runtime imports.
- **`AgentInstance.shutdown()` saves cerebellum** — learned forward models no longer lost on session end.
- **Experiment results:** `b4_replanning_results.md`, `p6_extinction_results.md`, `p8_sleep_replay_results.md`.

### Fixed

- `executor.embodiment` attribute lookup was using `_embodiment` (wrong name) — always returned None. Fixed in both factory and CLI.
- Sim path was building a second PainBus on the same hippocampus/nac, causing double-subscription of learning callbacks. Fixed to reuse bio-stack's bus.
- Bio-stack construction failure now propagates instead of silently degrading to a partial agent.

## [0.4.0] - 2026-04-19

### Added

- **Input standardization.** Unified input handling across all simulation modes (generative, DM, interactive, fixture). `PerceptSource` protocol with 4 implementations.
- **DM interactive mode.** Free-text roleplay between choices. Campaign runs on thread so stdin reader accepts input.
- **Rich menu system.** `maxim` (no args) launches interactive menu with campaigns, chat, doctor, help.
- **NAc suppression in interactive mode.** Tool-outcome learning gated on `get_interactive_mode()` to prevent human-directed actions from corrupting causal models.
- **Scale validation.** 20/20 seeds, p = 3.87e-6. Cross-session learning is not a fluke.

### Fixed

- Display/interactive globals now reset between menu sims (`reset_sim_display_state()`).
- DM campaign thread ordering: campaign runs AFTER stdin reader starts.
- Stall detector disabled in interactive mode (nudge prompts contained adversarial probes).

## [0.3.2] - 2026-04-18

### Added

- **Bidirectional interactive mode.** Raw terminal input with in-panel rendering, `request_interaction` agent-to-user prompting, `set_scene` dynamic scene header, `/pause` `/resume` `/display` commands, scrollable log with bio trace dimming, end-of-sim review prompt.

### Fixed

- Display corruption from `print()` calls during Rich Live panels.
- Stdin contention between display thread and input reader.
- Tool schema validation for JSON schema vs flat tool formats.
- LLM prompt context truncation for long conversations.

## [0.3.1] - 2026-04-18

### Added

- **Interactive UX fixes.** `RequestInteractionTool` honest reporting, narrator fallback immersion, handler logging, story context truncation, `MaximDisplay` → `sim_logger` wiring, prompt cleanup.
- **4 introspection tools.** `nac_stats`, `memory_pressure`, `loop_stats`, `pain_triggers_active` — agent can reason about its own learning state.

## [0.3.0] - 2026-04-17

### Added

- **Cross-session learning without fine-tuning — demonstrated across all 3 tiers.** 41/41 hypotheses confirmed across 4 experiments.
- **SEM Learning Loop (5 stages).** Cerebellum activation in BioStack, distribute_reward wiring, success reactions, pain spike episode boundary.
- **Valence Annotation (Stages 1-3).** Episode.valence, Edge.metadata["valence"], spreading_activation(propagate_valence), retrieve_on_cue(include_valence).
- **Behavioral Convergence Wiring (4 stages).** Valence in PromptAssembler, observe_episode_event in agent loop, energy→Reaction bridge, food/water/poison SEM specs.
- **Bio-Stack Unification (Waves 0-3).** `build_bio_stack`, `build_pain_bus`, `build_memory_hub`, `build_default_network`, `build_executor` — all canonical construction sites with structural enforcement.
- **Substrate P0-P4 complete.** Recognition, reward modulation, episode binding, channel integration, persistence/snapshot, cross-modal binding — all shipped.
- **LLM Path Refinement (Plans 1-4).** Typed errors, fast failover, `_MaximPeerBackend`, reactive peer mesh with auto-drain.

## [0.2.1] - 2026-04-10

### Changed

- **Re-publish of 0.2.0 contents.** No functional changes from the 0.2.0 draft. The 0.2.0 version slot on PyPI was burned by an earlier upload+delete cycle (PyPI version numbers are immutable even after deletion), so this patch bump is the smallest version that could be published.

## [0.2.0] - 2026-04-10 — Research Preview

**Versioning note:** This release was originally drafted as 1.0.0 and the entries below describe that work. After a deep architectural review on 2026-04-10, the 1.0 label was pulled and reissued as a 0.2.0 research preview. The reasoning: the bio-inspired stack is currently half-earned (NAc and Cerebellum implement genuine analogs of their brain namesakes; ATL, Angular Gyrus, SCN, and Default Network use the vocabulary without the cross-region mechanisms), and shipping 1.0 would lock in stability promises before the percept-substrate refactor that closes that gap. The 1.0 label is now reserved for the version that demonstrably improves on a task across sessions without fine-tuning the underlying LLM. See [docs/plans/archive/substrate_plan.md](docs/plans/archive/substrate_plan.md) for the original 0.3 → 0.4 → 0.5 → 1.0 narrative (the plan has since been split into focused sub-plans under [docs/plans/](docs/plans/README.md)).

The work documented below is real and shipping in 0.2.0 — only the label changed.

## [Original 1.0.0 draft — shipped as 0.2.0] - 2026-04-09

### Added

- **Expanded SEM component library** — 54 seed components across 7 categories (bodies, creatures, environments, items, npcs, vehicles, weapons). Covers fantasy, cyberpunk, sci-fi, horror, historical, modern, and devops genres.
- **SEM entity wiring in DM campaigns** — campaigns now instantiate live SEM entities at startup. NPCs and world objects have real sensors, affordances, and failure modes. Registry refs (`ref: npcs/guard`) with optional overrides.
- **Event subscription system** — `maxim.on("tool_call", callback)` with typed payloads (`ToolCallEvent`, `PainSignalEvent`, `MemoryCaptureEvent`, `PromptEvent`). Bridged to internal AgentBus.
- **Custom tool registration** — `@maxim.tool` decorator and `maxim.register_tool()` now wired into runtime. Tools injected into all `run()`/`imagine()`/`campaign()` calls.
- **4 new DM campaigns** — Wizard's Tower (fantasy), Server Breach (devops), Haunted Manor (horror), Space Station Crisis (sci-fi).
- **Context manager protocol** — `AgentPool` and `Session` support `with` statements for automatic cleanup.
- **Return type annotations** — all `maxim.create.*` and `maxim.load.*` functions return typed objects instead of `Any`.
- **`ComponentNotFoundError`** — custom exception for missing SEM templates (was generic `KeyError`).
- **Exception renames** — `MaximConnectionError`, `MaximMemoryError`, `MaximRuntimeError` no longer shadow Python builtins. Old aliases removed.

### Changed

- **CLI restructured** — 70+ flags organized into 11 argparse groups (core, cloud, autonomy, memory, hardware, agentic, exploration, simulation, debug, benchmark, utilities). `--internet-access`/`--no-internet` now mutually exclusive.
- **`configure()` validates inputs** — warns on out-of-range verbosity, unknown show channels, unknown debug subsystems.
- **`model=` parameter now overrides env vars** — `setdefault` replaced with direct assignment in `run()`/`imagine()`/`campaign()`.
- **Observe dispatch deduplicated** — shared `query_observer()` between `Session.observe()` and `maxim.observe()`.
- **Atomic writes** — model persistence and markdown report saves now use tmp+fsync+replace.
- **Deferred numpy/scipy imports** — `response_output.py` no longer eagerly loads scipy on `import maxim.utils`.
- **`load.nac()`/`load.atl()` standardized** — both now check file existence before loading (was inconsistent).

### Fixed

- `maxim.on()` and `maxim.register_tool()` were no-ops — callbacks/tools never reached the runtime.
- `@maxim.tool` decorator missing thread lock on `_pending_tools` append.
- `_inject_pending_tools()` didn't clear the list — tools double-registered on subsequent calls.
- Silent `except Exception: pass` in `session.research()` — now logs warning.
- `--sim-report` silently ignored without `--sim` — now validated.
- All `llm-local` references updated to `llm-llama` (7 locations across docs and source).
- Stale CLI defaults in docs (wrong `--mode` default, wrong `--persona` default, non-existent `--prompt-profile`).
- Deprecated `--sim agent` syntax updated to `--sim "goal"` across all active docs.
- Broken `</span>` tags in maxim-operating-modes.html.

## [0.2.0] - 2026-04-08

### Added

- **SEM Component Registry** — reusable entity templates with inheritance and deep merge. Components stored as YAML specs in `~/.maxim/components/` (user) and bundled in the package. 9 seed components across NPCs, weapons, creatures, and environments.
- **DM Encounter Library** — reusable scene + choice templates for campaigns. 8 seed encounters across combat, social, exploration, and puzzle categories. Campaigns reference templates via `template:` key with campaign-specific wiring.
- **Agent Factory + Pool** — multi-agent runtime infrastructure. Independent agents with isolated Hippocampus, NAc, ATL, MemoryHub, and ToolRegistry. Concurrent execution via ThreadPoolExecutor. Thread-safe ToolRegistry.
- **Party DM Runtime** — NPC agents with real memory and learning in DM campaigns. NPCs react to scenes, their dialogue is folded into PC's stimulus, and outcomes are broadcast for hippocampus capture. Per-NPC memory export.
- **Hippocampus recall improvements** — relevance ranking via keyword overlap, observation dedup window (30s default), lightweight `store_observation()` for NPC agents.
- **Interactive Runtime** — universal prompt protocol (8 prompt types, 5 handlers) + Rich terminal display with structured panels. DM campaign extension with character sheet, inventory, encounter info, NPC relationships, and user notes.
- **Generative Architect** — LLM-driven campaign creation with Entity Designer (generates valid SEM specs from natural language), character templates (5 PC archetypes, 6 NPC roles), and architect tools (browse components/encounters, design entities, emit campaigns).
- **Expanded Python API** — 7 new verbs: `campaign()`, `benchmark()`, `research()`, `on()`, `register_tool()`, `register_persona()`, `@tool` decorator. Structured result types: `CampaignResult`, `BenchmarkResult`, `ResearchResult`, `EventHandle`.
- **Split persistence protocols** — `EpisodicStore`, `CausalStore`, `SemanticStore` protocol classes with `File*Store` defaults. Foundation for Mother Maxim's database backend.
- **Cloud provider profiles** — 10 new builtin LLM profiles: Gemini (2), Groq (2), Together (1), Fireworks (1), Mistral (2), DeepSeek (2). Zero new backend code — all use existing OpenAI-compatible endpoint.
- **`metadata: dict` field** on `EpisodicMemory` for extensible per-memory metadata (domain tags, contribution source, tenant ID).
- **Package infrastructure** — `py.typed` (PEP 561), `__main__.py` (`python -m maxim`), bundled data in `src/maxim/_data/`.

### Changed

- **Data paths** — all user data now writes to `~/.maxim/` (override via `$MAXIM_DATA_HOME`). Bundled defaults ship in `src/maxim/_data/`. 28+ source files migrated from CWD-relative `data/` paths.
- **GPU detection** — moved from import-time subprocess to lazy `gpu_detect.py` called from `cli.main()`. `import maxim` no longer has side effects.
- **Import hygiene** — `selfy.py` import-time side effects (`mp.set_start_method`, `PYOPENGL_PLATFORM`, GPU detection) moved to lazy `_setup_hardware_env()`.
- **Persistence safety** — 13 hand-rolled JSON persistence patterns replaced with `atomic_write_json()`.
- **Thread safety** — locks added to `ToolRegistry`, `narrative_transcriber._class_registry`, `lane_backends._active_routers`, `gpu_detect._detected`.
- **Version constraints relaxed** — `requires-python >=3.10` (was 3.12), `numpy >=1.26` (was 2.2), `scipy >=1.11` (was 1.15).
- **`rich` moved to core dependency** (was optional `[ui]` extra).

### Fixed

- Unclosed file handles in `provenance/store.py` and `sim_logger.py` (added `__del__`/atexit cleanup).
- `print()` in library code (`mesh_trace.py`) replaced with `logger.warning()`.
- Duplicate `llm-local` / `llm-llama` dependency removed.
- Pre-existing persona count test updated for `dungeon_master` + `adventure_architect`.

## [0.1.0] - 2026-04-06

### Added

- Initial release with bio-inspired cognitive architecture.
- 5-agent pipeline (Perception, Memory, Exec, Goal, Statistician).
- Biological memory systems (Hippocampus, ATL, NAc, SCN, Angular Gyrus).
- SEM protocol for embodiment (Entity, Sensor, Modulator).
- DM campaign system with 4 hand-authored campaigns.
- Simulation benchmarking with multi-model comparison.
- Verb-based Python API (run, imagine, connect, diagnose, observe, configure).
- Remote peer management (update, restart, LLM hot-swap).
- Doctor diagnostics with platform detection.
