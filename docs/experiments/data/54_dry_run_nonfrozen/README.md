# Exp 54 — harness verification (NOT a result, nothing here is frozen or gated)

Offline dry-rig runs made while building the Exp 54 harness (pre-data, 2026-08-26),
kept so the pre-registration's amendment 1 is auditable:

- `sweep_procedure_check_exp53_agents.json` — the `sweep` subcommand run over the **Exp 53
  agents** (infant body, δ map): the declared target procedure reproduces Exp 53
  amendment 1's placements exactly — gated `[-0.3, -0.2, 0.5, 0.6]`, exploratory `[0.2]`.
- `targets_rekeyed_exp53_agents.json`, `phase1_phase2_rekeyed_exp53_agents.jsonl`,
  `phaseC_rekeyed_exp53_agents.jsonl` — the Reachy nursery body (`bodies/reachy_mini_infant`)
  through the production `make_reachy_orient_factory` on the dry rig, driven by **copies of
  the Exp 53 agents with their bias keys re-written** `infant_operant_turn_*` →
  `reachy_mini_turn_*` (a scratch archive; the "taught_seed45" exploratory agent is Exp 53's
  seed 48 re-keyed). They exercise the 4-tool repertoire, the `--targets` path, Gate C (the Phase C file is the review-fold run in the USER's full tool space — no
  whitelist; `start.tool_whitelist` is null; Gate C PASS on the dry rig) and the verdict shape. No nursery on the Reachy body has run; the Phase A record is
  `../54_phaseA_nursery.jsonl` when it exists.

Provenance: these artifacts were produced from the **pre-commit working tree** of the harness
commit (their `provenance` stamps read `abdb3705` + `working_tree_dirty_src_scripts: true`, the
parent commit plus the uncommitted harness); the code-under-test is what the harness commit
then froze. Absolute `/Users/...` paths follow the Exp 53 records' convention; no secrets.
