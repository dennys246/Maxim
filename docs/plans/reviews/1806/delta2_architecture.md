# #1806 device registry: architecture, persistence and API-shape review, delta round 2

Reviewer: architecture/persistence/API lens, delta round 2, read-only, 2026-11-25, v3 @ 8540cf19. This is a fresh reader. Code was read at origin/main `b7380ee7` in the plan worktree. The only things run were two import probes, under a fresh `HOME=$(mktemp -d)` with `python3 -I` and the worktree's `src` on `sys.path`. No robot or LAN host was contacted. Scope: delta-1 architecture N1–N5, N7 and N12, plus any new fail-open defect from the v3 folds.

**Verdict: SHOULD-FIX before PR 1 and PR 2. No DO-NOT-BUILD. PR 0 is not blocked by this lens.** v3 closes N3, N5 and N12, and closes N4's main defect. Two of the §3b enforcement levers that v3 chose do not do what the note claims:

- (c) `BLOCKED_PYTHON_IMPORTS` gains `maxim.devices`. This entry can never match anything.
- (b) is the AST lint on attribute calls. A plain `from maxim.devices import add` gets past it.

Two smaller gaps also remain. The `devices` LAYER_RULES entry has no content, and an empty entry is skipped. The single `primary:` marker is still parsed loosely, so several `primary: true` entries are accepted and `"false"` reads as true.

## Re-check of delta-1 architecture findings

| N# | v3 status | Note |
|---|---|---|
| N1 | **PARTIAL** | The package `src/maxim/devices/` fixes the skip: `audit_architecture` now finds a directory. But v3 never says what the entry forbids. `audit_architecture` also does `if not forbidden: continue`, so an entry with an empty `must_not_import` is skipped. "The entry permits `peer.mesh_config` reads" describes nothing: `LAYER_RULES` is a deny-list, so whatever is not listed is allowed. Reading `peer` is feasible and breaks no layer rule. The probe imports only `maxim.peer`, `peer.cli`, `peer.config`, `peer.install_core`, `peer.mesh_config`, `utils` and `utils.atomic_io`, so there is no runtime or CLI pull and no cycle. See F3 |
| N2 | **PARTIAL** | Closed: the `skills/` vacuity, the overclaim (§3b "Scope of the claim"), the types moving to `hardware/`, and the per-symbol problem (`_write.py`). Still open: lever (c) is vacuous (F1), and lever (b) is porous to from-imports and bound references (F2) |
| N3 | **CLOSED** | Refuse-newer is in the loader and the writer, modelled on `config_writer._refuse_if_newer`. A present non-string value is an error. Feasible as written. NIT: "absent key without the re-save nag" means the loader must not call `check_format_version` on an absent key, because that function always WARNs then (`format_version.py::check_format_version`). Say so, or add a quiet variant |
| N4 | **CLOSED (main defect); residue in F4** | D2″ makes `primary:` the only marker, and there is no `default:` key. `resolve_robot_entry`, `agentic_runtime`'s mirror and old builds all read `primary:`, so they agree once `use` moves it. Residue: the marker's own parse (F4) and the meaning of `forget` on the primary (F5) |
| N5 | **CLOSED** | Editing the raw `safe_load` mapping under the lock, validating it through the strict parser, then `safe_dump` + `atomic_write_text` is coherent. Comments are lost, which is already accepted via the `.bak`. `safe_dump` quotes strings that would otherwise resolve as int, bool or float, so the writer's own output round-trips. The strict parser refuses a key it does not know instead of dropping it, so data loss becomes a loud refusal. NIT: the caller allow-list for `write_robots_config(` must also allow-list the round-trip test file, as `MESH_WRITER_MATCHES` does for `tests/unit/test_mesh_config.py`. Otherwise it fails CI (fail-closed) |
| N7 | **PARTIAL (fail-closed)** | Closed: `devices.open` versus `use`/`maxim connect`, and the `robot_id`/`robot_name` split (§3.1). Still open: PR 1 still lists `open` among the "read verbs", and lint (b)'s verb list does not include `.open`. `open` connects a daemon session, sends `SetAutomaticBodyYaw(False)` and may release media (#1704). The pin still holds, so this is fail-closed on identity. See F6 |
| N12 | **CLOSED** | D12′: `run_device` stays in `console/`, with at most a lazy re-export. Keep any re-export lazy (a module `__getattr__`), so that `import maxim.devices` never imports `console`/FastAPI |

Other feasibility checks from the brief:
- **`hardware/devices_types.py`.** This home is legal. `hardware` must not import `embodied_runtime`/`tools`/`runtime`/`agents`/`planning`. `embodied_runtime` has no LAYER_RULES entry, and the rule does not forbid `embodied_runtime → hardware`. `LaunchPlan`'s mode type, `modes/definitions.py::OperationalMode`, lives in `modes/`, which `hardware` may import, and `modes` imports only `utils.prompts`. The error bases `MaximConnectionError`/`ConfigurationError` are in `maxim/exceptions.py`, which is fine.
- **`maxim.devices` in `_API_SUBMODULES`.** This works through `__init__.py::__getattr__` → `importlib.import_module`.

## New findings from the v3 changes

| # | Finding | Class | Fail-open/closed | Covered class? |
|---|---|---|---|---|
| F1 | §3b(c) `BLOCKED_PYTHON_IMPORTS` gains `maxim.devices`, but this can never match. The sandbox hook tests `name.split(".")[0] in blocked`, so only top-level names count. `from maxim import devices` imports the name `maxim`. `import maxim; maxim.devices` loads it through `__init__.__getattr__` → `importlib.import_module`, which never calls `builtins.__import__` (probe below). The module's own docstring also says the list is "NOT a security boundary" | SHOULD-FIX | fail-open (vacuous guard) | yes (vacuous guard, as N1/N2), but it is a gate input |
| F2 | §3b(b) only refuses *attribute calls* `.add/.rename/.forget/.relocate/.repin/.use` on a `maxim.devices` reference. These get past it: `from maxim.devices import add` (or `from maxim.devices._write import add as a`) followed by a bare-name call; `f = maxim.devices.use; f(...)`; and `import maxim.devices._write`. Lint (a) does not catch these either, because the call into `write_robots_config` still comes from `_write.py`, which (a) allows | SHOULD-FIX | fail-open | yes (porous guard; N2's class) |
| F3 | The `devices` LAYER_RULES entry's content is unspecified (N1). It needs a non-empty `must_not_import`. Proposed: `["runtime", "tools", "agents", "planning", "console", "cli", "cli_parser", "cli_utils"]`. `target_layer("maxim.cli")` returns `"cli"`, so this one entry mechanically enforces §3c's goal that "the library does not depend on the CLI". Leave out `embodied_runtime`, because #1808's `devices.embody` will need it. Prove the entry by deletion: plant a violating import in a `tmp_path` tree, as `tests/unit/test_architecture_audit.py` does | SHOULD-FIX | fail-open (an empty entry is skipped) | yes (vacuous guard) |
| F4 | The marker's parse is lax. (i) `RobotConfig.from_dict` uses `primary=bool(data.get("primary", False))`, so the hand-written `primary: "false"` and `primary: "no"` read as **true**. (ii) If several entries say `primary: true`, `get_primary()` silently returns the first in YAML order. Precedence step 4 and the legacy fallback then choose an entry the operator may not have meant. The writer never produces either state, but hand-written files and old builds can. Fold this into §4.1's strict entries: `primary` must be a YAML bool, and more than one `true` is a `RobotsConfigError` | SHOULD-FIX | fail-open (the wrong *registered* entry is chosen; that entry's pin still verifies it) | yes (strict typed parse, F1b/N4) |
| F5 | §7 says "`rename`/`forget` move `primary:` within the same write". For `forget` of the primary entry, the note does not say where the marker goes. Reassigning it to another entry would silently pick a new default. State that `forget` **clears** it and never reassigns. With two or more entries left and no primary, `resolve_robot_entry` then returns `None`, which fails closed | NIT | fail-open if it is implemented as "reassign" | yes (N4) |
| F6 | `devices.open` is classed as a read verb (PR 1), and lint (b) does not list `.open`. State its side effects in the API docs (a daemon session, `SetAutomaticBodyYaw(False)`, a possible media release). Decide whether `tools/` may call it, and if not, add `.open` to (b). Also, defining `open` in `devices/__init__.py` shadows the builtin there: define it in a submodule and re-export it | NIT | fail-closed (the pin holds) | yes |
| F7 | YAML 1.1 types for `hardware_id`. An unquoted all-digit id is an `int`, and one with a leading `0` is octal, so `str()` would change its value. State that a non-`str` `hardware_id` is a `RobotsConfigError` with a "quote it" hint, and never coerce it | NIT | fail-closed (it would fail the hex regex, or mismatch) | yes |
| F8 | The docs list in §8 leaves out `ARCHITECTURE.md`, the layer-ownership file the audit cites, which needs a `src/maxim/devices/` row. "The new module goes into `run_mypy_public.sh`" should say every file in the package | NIT | fail-closed | yes |

**Proposed fold for F1 and F2:**
- Use `LAYER_RULES` for the inbound edge. Add `"devices"` to the `must_not_import` of `runtime`, `tools`, `agents`, `planning`, `memory`, `bridges` and `hardware`, plus a new `embodied_runtime` key. The audit already resolves every static spelling: `import maxim.devices[...]`, `from maxim.devices[._write] import x`, and `from maxim import devices [as d]` through the per-name candidate path. None of these trees needs `maxim.devices`, because the types are in `hardware/`. This also covers the read verbs and `open`.
- Keep (b) only for the `import maxim` + attribute path, which the audit records as target `None`. Make it reference-based (any `Attribute` load that resolves to `maxim.devices` through a per-module alias table for `import maxim [as m]`), not call-based.
- Say explicitly that `getattr` and `importlib` spellings are out of scope, matching the audit's docstring.
- Either drop (c), or replace it with top-level `"maxim"` in `BLOCKED_PYTHON_IMPORTS`, labelled defense in depth only.

## Evidence

- `src/maxim/utils/audit.py::audit_architecture`: `if not layer_dir.is_dir(): continue`, then `if not forbidden: continue`. `LAYER_RULES` has only `must_not_import` lists, and `runtime` forbids only `conscience`. There is no `embodied_runtime` key.
- `src/maxim/utils/audit.py::_check_file`: it handles `Import`, `ImportFrom` and per-name `from maxim import X` candidates, so the inbound-edge ban in the fold covers aliasing by `as`.
- `src/maxim/utils/sandbox_executor.py::_create_restricted_python_wrapper`: `if name.split(".")[0] in blocked`. Note that this module is `utils/sandbox_executor.py`, not `tools/`.
- Probe 1 (fresh HOME): I installed an `__import__` hook that records names, ran `import maxim`, then accessed `maxim.create`. The result was "lazy attr ok: maxim.create | hook saw maxim.create? False". Lazy `_API_SUBMODULES` loads bypass the import hook.
- Probe 2 (fresh HOME): the closure of `import maxim.peer.mesh_config` is `maxim, maxim.peer, maxim.peer.cli, maxim.peer.config, maxim.peer.install_core, maxim.peer.mesh_config, maxim.utils, maxim.utils.atomic_io`.
- `src/maxim/hardware/config.py`:
  - `RobotConfig.from_dict` does `bool(data.get("primary", False))`;
  - `RobotsConfig.get_primary` returns the first `primary`, else `robots[0]`;
  - `resolve_robot_entry` falls back to `get_primary()` only when there is one robot or an explicit primary.
- `src/maxim/utils/format_version.py::check_format_version`: it WARNs once per file type whenever the field is absent or not a string, and it never raises.
- `.github/workflows/test.yml` `MESH_WRITER_MATCHES`: a grep that covers both `write_mesh_config(` and its from-import spelling, with test files allow-listed. That is the template for lint (a).

## Stop-rule outcome

**Not met.** F1, F2 and F3 are fail-open guard defects, and F4 is a fail-open parse. All four are in already-covered classes (vacuous or porous guard; strict typed parse), but F1–F3 are the gate's own enforcement inputs, so they cannot go to a follow-up issue. Each fold is small and mechanical, as above. The next round can be delta-scoped to §3b's enforcement text, the `devices` LAYER_RULES entry and §4.1's `primary` typing. The NITs (F5–F8) can be folded without a round.
