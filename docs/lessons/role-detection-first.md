# Role detection is the first runtime action

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] Role detection is the first runtime action:** Plan 2 R2a made role explicit. `cli.py::main()` calls `runtime/role.py::detect_and_apply_role(raw_argv)` immediately after `configure_logging`, BEFORE subcommand dispatch. It exports `MAXIM_ROLE` + emits `role_detected` as the first structured log event. Downstream code (`runtime/llm_server.py::_model_state_file`) reads `MAXIM_ROLE` from env — never re-detects, never infers from `peer.yml` existence. If you're adding a new feature whose behavior depends on role, read `os.environ["MAXIM_ROLE"]`; never call `detect_role()` a second time. Decision order: env var → mesh.yml → peer.yml → `--llm` flag + no peer config → default leader. Persisted state is split per role (`active_llm_model.{role}.txt`). The call site is co-located with `configure_logging` at the top of `main()` — if you move it downstream you'll re-encounter the subcommand-dispatch logging gap described below. Regression guard: [src/maxim/runtime/role.py::detect_and_apply_role](src/maxim/runtime/role.py) + [src/maxim/cli.py::main](src/maxim/cli.py) — call site structurally precedes subcommand dispatch.
