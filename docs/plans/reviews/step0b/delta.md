# Step 0b (offline latch): delta round, all three lenses

Scope: the staged fold, `git diff --cached` on top of 3b7f775a, which answers executor.md, architecture.md and wire.md. Read-only, 2026-10-26. The box rule held: no suite, prepr or mypy. Probes ran as `python3 -I` children from a scratch dir, loading `_offline.py` by path.

**Verdict: no BLOCKER and no fail-open SHOULD-FIX. STOP: no further round owed.** There is one fail-closed SHOULD-FIX, which is a docs correction, and five NITs.

## Executor

Child-process probes. Each was run engaged and not engaged, against the same pre-built sockets.

| Channel | Not engaged | Engaged |
|---|---|---|
| `os.fork` | allowed | REFUSED, in the parent |
| `os.forkpty` | allowed | REFUSED |
| `os.spawnv(P_WAIT, /bin/true)` | allowed | REFUSED in the parent. No duplicate child unwinds, and it is booked |
| `AF_UNIX` `SOCK_DGRAM` `sendto(path)` / `sendmsg(..., path)` | allowed | REFUSED |
| Post-latch connect, then `send()` | allowed | REFUSED at connect, so `send` is never reached |
| Pre-latch socketpair `send()`, `sendmsg()` without an address (None dest), pre-latch TCP `sendall` | allowed | allowed. This is the stated residual |
| Inbound `bind` + `listen` | allowed | allowed |
| Ledger | 0 entries | 6 entries |

multiprocessing, with the latch engaged in the parent: `spawn` works (exit 0), `fork` is refused at `os.fork`, and **`forkserver` is refused at `socket.connect`** (see S1).

Tests:
- `tests/unit/test_offline_latch_unit.py` drives real branches. It calls `_refused_detail` and `_hook` directly, and these are the hook's actual decision functions; it does not restate a table. It also drives the real read sites in-process: `get_version_info`, `git_out`, `_code_stamp`, `load_internet_policy`, `live_internet_policy_getter`, the toggle tool, and the four former replicas checked against `truthy`. The event-argument shapes match CPython: `sendto`/`sendmsg` pass `(sock, addr)`, and `fork` passes `()`. The integration child (`REFUSED_CHANNELS`, 8 additions) confirms them end to end.
- Narrow run: the 5 touched test files gave 247 passed and 5 xfailed. All 5 xfails are the expected strict item-10 gates.
- Diff coverage, using `lint_coverage.diff_coverage` over that run:
  - The fold, base `HEAD`: **93.8%** (30/32).
  - The branch, base the merge-base 68724167: **96.0%** (169/176).
  - Both clear the 80% gate. The uncovered lines are the `session-lowered` elif in `internet_access.py` and in `internet_search.py`, and some `utils/http.py` lines; the session-lowered elifs belong to their own test files, which this narrow run did not include.
- Lints on the tree: ruff check and format are clean, and harness-provenance, orphan-module, function-length, no-silent-swallows, unreleased-on-src and claude-md-invariants are all clean.

## Architecture

- **Parser delegations.** `memory_hub`, `nac`, `models/download`, `utils/plotting` and `cluster_bias_annotation` now import the top-level stdlib leaf. `audit_architecture()` reports no `_offline` violation, since the leaf is in no `LAYER_RULES` layer. Importing `truthy` never adds the hook, because only `engage()` does. The NAc change is only `_read_nac_reward_bias_disabled_env`, called from `NAc.__init__`, and touches no selection logic. The Exp 44 harness loads `_offline.py` by path from its own tree (the `_provenance` pattern), and `lint_harness_provenance` is clean.
- **`MAXIM_AUTO_DOWNLOAD_MODELS`.** A value of `t` or `y` now means no consent. Because the variable is non-empty, it also masks `llm.auto_download` from config (N3). Both effects are fail-closed. The CHANGELOG names the toggle. The only writers set it to `"1"` (`cli_utils.py:157`, `benchmark_cross_session.py:562`).
- **Refusing `os.fork`.** Maxim's own process starts use `spawn`: `media_loop` (the whisper transcriber) and `camera/display`. They pass unaudited through `fork_exec`, and their targets import maxim, so the child self-engages. There is no `os.fork`, `pty`, `ProcessPoolExecutor` or `Pool` anywhere in `src/` or `scripts/`. Subprocess was already refused before this fold. So refusing fork breaks nothing Maxim itself does offline.
- **Invariant 3 (notes §1A.9).** Every clause matches the probes, but one consequence is incomplete (S1). All five guard tests exist. The guard list's ordering is wrong (N1).

## Wire integrity

- **The `"offline"` source.** Its consumers are:
  - `disabled_reason()` and the toggle tool's remedy: these are offline-aware, and the unit test covers them.
  - The `live_internet_policy_getter` startup warning: offline-aware.
  - `effective_internet_enabled`, read by `cli`, `agentic_runtime` and the prompt roster through `state.data["internet_access"]`: these see only a bool, which is False offline. That is correct.

  No console payload serialises the effective source. `InternetAccessState.to_dict` serialises the toggle file's own source, which is unchanged. A source string outside the known set falls through to the generic "disabled" message, so it fails closed.
- **The five parsers.** Their readers are `agent_factory`/`agentic_runtime` (body-state), `NAc.__init__`, download consent at two sites, and the matplotlib preflight/preload skips. No script, YAML, doc or workflow sets any of these toggles to `t` or `y`; the only hits are the review files.
- **Changed test pins.** `test_nac` adds `t`/`y`→False. `test_body_state_wiring` moves `t`/`y` to falsy. `test_exp42_pipeline` uses `yes`/`on` and adds the harness–runtime agreement test that closes Wire F1. Each pin now matches `truthy`.

## Findings

**S1 (SHOULD-FIX, fail-closed, docs only).** "multiprocessing's `fork` start method does not work offline" leaves something out: `forkserver` also fails. Its `AF_UNIX` connect to the server is refused, which I confirmed by probe. Python 3.14 makes `forkserver` the default on Linux, and `requires-python` is `>=3.10`. So any library that uses the default context fails offline under 3.14 on Linux, and under ≤3.13 on Linux, where the default is `fork`. Only `"spawn"` works. State this in `_offline.py`'s docstring, in notes §1A.9 invariant 3, in the CHANGELOG and in the llm-routing.md row.

**N1 (NIT).** In the guard list for invariant 3, `tests/unit/test_offline_latch_unit.py` sits between `::test_engaged_latch_refuses_each_channel` and `::test_a_client_pooled_before_engage_cannot_send_after`. Under the `::` shorthand, the next two nodeids now read as belonging to the unit file, where they do not exist. Move the unit file to the end of the list.

**N2 (NIT).** `scripts/benchmark_exp42_preference.py` still carries its old comment above `_load_offline_leaf`, which says "Same six-value truthy set … Replicated here". That is now false.

**N3 (NIT, fail-closed).** `_auto_download_enabled`: a non-empty, non-canonical value such as `y` now hides a config grant of `llm.auto_download=true`. The behaviour is unchanged for garbage values, but it is a new case for `y`. Consider one line in the CHANGELOG.

**N4 (NIT).** Six-value replicas remain in `cli_utils.py` (audio, interactive), `simulation/orchestrator.py::_env_trace`, `utils/gpu_compat.py` and `models/language/config.py`. CLAUDE.md §Environment Variables still names `annotation_disabled_via_env` as the canonical parser. That is still true through delegation, but the owner is `maxim._offline.truthy` (Arch S2 is unaddressed). A follow-up issue is enough.

**N5 (NIT).** The comment "checked first" in `load_internet_policy` is not accurate. An unreadable policy file is checked before offline, so an offline process with a bad policy file reports `policy-unreadable`. It is still disabled, so this fails closed. Separately, the messages name `llm.offline`, which this step does not ship. An operator who looks for it will not find it. The CHANGELOG line that ends "`maxim doctor` crashes)." is also over-wide.
