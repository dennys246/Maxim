# Step 0b (offline latch): Architecture lens

Reviewed: `git diff origin/main...HEAD` (63f196bb + merge 3b7f775a), read-only, 2026-10-26. No suite, prepr or mypy run (box rule).

**Verdict: no BLOCKER. Three SHOULD-FIX, five NIT.**

## What holds

- **Leaf.** `maxim/_offline.py` imports `os, socket, sys, threading, typing` only. It sits at top level, outside every `utils/audit.py::LAYER_RULES` layer, so `prompts/`, `utils/` and `simulation/` may import it. It is reached by `maxim/__init__`, `utils/http`, `utils/internet_access`, `prompts/cluster_bias_annotation` and `simulation/report`, so the orphan lint passes. `maxim/__init__::_self_engage_offline` costs one `os.environ.get`. It imports the leaf only when the variable is set, so a plain `import maxim` loads nothing heavy and adds no hook (`test_no_hook_without_the_variable` is flipped). `get_version_info` now imports the leaf unconditionally, but that adds only `socket`, which is cheap.
- **Hook.** An audit hook that can never be removed, and that is inert behind a single `_engaged` read, is the right mechanism. Python offers no other process-wide interception point that a child also inherits. The module docstring states the §0/§6 boundary honestly: Python level only; native stacks, a `send`/`sendall` on a connection opened before the latch, and the unaudited `fork_exec` are outside it.
- **Parser.** There is one canonical copy. `TRUTHY_DISABLE_VALUES` is an alias of `TRUTHY_VALUES`, and `annotation_disabled_via_env` delegates to `truthy`.
- **CHANGELOG.** It states the t/y change and `MAXIM_OFFLINE`. The brief's table has the row. The conftest pops the variable before `import maxim`, and the autouse `_reset_offline_latch` restores all five keys. Both are correct per the env-scrub lesson.
- **utils/http.** When the latch is not engaged, behaviour is unchanged. `strict or is_offline()` and `trust_env=not is_offline()` are both no-ops online, so #921 holds (the test asserts the precondition). Every client is built at one of two sites, and both carry `_OFFLINE_EVENT_HOOKS`, so a per-request `request` hook runs on every hop, including on clients pooled before the latch. That is the right layer, and `close_all` is the second line of defence.
- **Swallows.** `_run_callbacks` re-raises the first error, so nothing is swallowed. There is no new try-import.
- **Seams.** `refuse()` is justified: `utils/http::_refuse_offline_request` calls it, and it is §6 layer 2's typed, booked refusal.

## SHOULD-FIX

1. **The replica parsers now diverge, and their comments are false.** Where: `integration/memory_hub.py::body_state_prompt_enabled`, `decisions/nac.py::_REWARD_BIAS_DISABLE_TRUTHY` / `_read_nac_reward_bias_disabled_env`.
   - Scenario: an operator runs Exp 44's paired arms with `MAXIM_ENABLE_BODY_STATE_PROMPT=y` and `MAXIM_DISABLE_COACH_BODY_LAYERS=y`. The first toggle still turns its layer ON, but the second no longer turns its layer OFF. The run is confounded with no warning.
   - Both comments claim they mirror the canonical set ("divergence … fails loudly"). The pin tests (`tests/unit/test_body_state_wiring.py::test_truthy_values`, `test_nac.py::test_truthy_parser_handles_canonical_values`) now pin the divergence instead of catching it.
   - Root cause: the copies were made only to avoid importing `prompts/` from `decisions/` and `integration/`. The new stdlib leaf removes that reason.
   - Proper fix: both delegate to `maxim._offline.truthy`, in this PR or in a filed issue that the CHANGELOG names. The ad-hoc six-value literals (`simulation/orchestrator.py::_env_trace`, `models/download.py`, `utils/gpu_compat.py`, `utils/plotting.py`, `cli_utils.py`, `models/language/config.py`) belong in the same issue.
2. **CLAUDE.md names the old owner of the parser.** Where: `CLAUDE.md` §Environment Variables still says the canonical parser is `cluster_bias_annotation.annotation_disabled_via_env`. It should say `maxim._offline.truthy`, or the delegating name should stay but the owner be stated. Scenario: the next env var a session adds imports `prompts/` from a layer that cannot take it, and the session copies the set again, which is exactly fix 1's root cause.
3. **Invariant 3 claims "no send to a new destination", but the code refuses only INET sends.** Where: notes §1A.9 invariant 3 vs `_offline.py::_refused_detail`. An `AF_UNIX` datagram `sendto`/`sendmsg` to a new path passes, even though `connect` refuses `AF_UNIX` ("no allowed endpoint", V8). Either refuse `sendto`/`sendmsg` with a destination for every family, or narrow the invariant to "no INET send" and list `AF_UNIX` datagrams as a residual. Also, its first guard, `test_no_effect_subprocess`, is still `xfail(strict=True)` (the item 10 census). The stub must not land in a brief until that test flips. It is not landed yet, which is correct.

## NIT

1. `_offline.py::engaged_reason` has no caller in `src/`, `scripts/` or `tests/` ("a fix ships with a caller"). Delete it, or have `refuse()` use it.
2. `_offline.py::_set_imported_library_flags` sets `ultralytics.utils.ONLINE`, but modules that ran `from ultralytics.utils import ONLINE` keep their own copy. The docstring should say this is best effort, since item 10's call sites own the real refusal.
3. `utils/code_tree.py::git_out`: when loaded by path (scripts), the probe is always False, even with `MAXIM_OFFLINE` set. That is consistent, because no hook exists there, but the docstring could say "a process that never imported maxim is not offline".
4. `_offline.py::_SPAWN_EVENTS` has no `_winapi.CreateProcess`, so it does not cover Windows. State this in the boundary paragraph, or say POSIX only.
5. Placing a private module at top level is a new pattern, and neither ARCHITECTURE.md nor `docs/reference.md` mentions it. Add one line saying why it lives there: it avoids `maxim.utils`'s package init.
