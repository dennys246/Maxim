# Subcommand dispatch in `cli.py::main` bypasses logging setup by default

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] Subcommand dispatch in `cli.py::main` bypasses logging setup by default:** `cli.py::main` short-circuits to `run_doctor_subcommand` / `run_peer_connect_subcommand` / `run_tunnel_subcommand` before reaching the sim loop that previously was the only caller of `configure_logging`. Any feature that depends on early logging setup (MAXIM_LOG_FILE JSONL handler, future structured event emission, Plan 2 R2a's `detect_role` log event) needs `configure_logging` called at the TOP of `main()` before subcommand dispatch, not at the sim-loop entry. This was a real bug during Plan 1 R1 — MAXIM_LOG_FILE silently did nothing for `maxim doctor` until commit `c8a07e9` added the early call. The sim loop's later `configure_logging(force=True)` call dedupes JSONL handlers by absolute path, so the early call + late call is safe. **Plan 2 R2 re-encountered this class of bug in a different form:** `_has_local_llm_flag` scanned raw `argv` including subcommand names, so `maxim tunnel --llm X` mis-detected role as `solo`. Any code that runs early in `main()` and consumes `argv` must explicitly handle subcommand entry paths — either skip the scan when `argv[0]` is in `{doctor, peer, tunnel, ...}` or only apply the logic to the sim/agent entry path. Regression guard: [src/maxim/cli.py::main](src/maxim/cli.py) — `configure_logging` is called at the top of `main()` before subcommand dispatch; co-located code structure enforces ordering.
