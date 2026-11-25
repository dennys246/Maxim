# #1806 device registry: architecture, persistence and API-shape review, delta round 1

Reviewer: architecture/persistence/API lens, delta round 1, read-only, 2026-11-25, v2 @ f0369379. This is a fresh reader; round 1 was a different reviewer. Code was read at origin/main `b7380ee7` in the plan worktree. No code was run, and no robot or LAN host was contacted. Scope: the disposition of round-1 findings R1–R17, plus defects that v2 introduced. Unchanged v1 material was not re-reviewed.

**Verdict:** **SHOULD-FIX before PR 1 and PR 2. No DO-NOT-BUILD. PR 0 is not blocked by this lens.** v2 folds most of round 1. Five new defects are fail-open, though, and four of them sit in the guards v2 relies on:

- the `devices` LAYER_RULES entry cannot work;
- the import ban has holes: `skills/` does not exist, `embodied_runtime` is missing from the list, `import maxim` + attribute access gets past it, and the agent has a bash tool;
- `check_format_version` does not refuse newer files;
- the robots.yaml writer can drop keys the parser does not model.

The fifth is a second default marker, `default:` next to `primary:`, which the legacy launch path never reads.

## Disposition of round-1 findings

| R# | Status | Note |
|---|---|---|
| R1 | FOLDED (moved) | D9 plan-only. The in-process launch moves to #1808 (`embodied_runtime/launch.py::launch`). Correct home |
| R2 | PARTIAL | The embody contradiction is resolved by D9. The **entry-id / daemon-name split was not carried**: v2 never says `--device` passes `robot_id=<entry>` with `robot_name=<entry's daemon name>`. #1808 doesn't mention it either, so "moved to §3c" loses it. See N7 |
| R3 | FOLDED | §4.1: `from_yaml` raises `RobotsConfigError`; selfy and agentic_runtime stop swallowing it |
| R4 | FOLDED | §4.1: the registry enforces the pin and checks `supports_identity_pin`. The pin no longer goes through the constructor filter |
| R5 | PARTIAL | The default left config.json (D2′), which is right. v2 picked a new top-level `default:` key instead of writing `primary:`, and keeps `primary:` in the precedence chain. That gives two markers, and `resolve_robot_entry` reads neither consistently. See N4 |
| R6 | FOLDED | The stable `maxim.connect` is untouched and labelled unpinned. Pinned connection lives in `maxim.devices` |
| R7 | FOLDED | No `DEVICE_KINDS`; `kind: robot` is reserved and the router has one branch |
| R8 | PARTIAL | Homes are named (§7). But the "new `LAYER_RULES` entry like `api.py`" for `maxim/devices.py` is vacuous: `api.py` has none, and the audit only scans directories (N1). The home of `ResolvedDevice` and `LaunchPlan` relative to the import ban is not stated (N2) |
| R9 | PARTIAL | D10 takes the C2 amendment and the import ban. The ban as written is partly unenforceable, and "the agent can never call them" overclaims (N2) |
| R10 | FOLDED (stricter) | D11 removes the cwd fallback; the writer always targets `<data_home>`. The scope of "requires an absolute `MAXIM_DATA_HOME`" is unstated (N8, NIT) |
| R11 | PARTIAL | `_format_version`, `FileLock`, `atomic_write_text`, `.bak` and the downgrade note are all taken. But "refuse-newer via `utils/format_version`" names a helper that does not refuse (N3), the writer's round-trip fidelity is unspecified (N5), and the `test_persistence_compat.py` row and the "pinned (enforced by ≥…)" `check` line are not carried |
| R12 | FOLDED | `launch:` is an entry-level sibling of `config:`, `operational_mode` is refused, and `_validate_operational_grant` runs on the typed flags |
| R13 | FOLDED | §4.3 step 2 dials the verified IPv4 literal with `connection_mode="network"` forced |
| R14 | FOLDED | The exemption and the not-`probe_cache` reason are stated, and the file is keyed by identity. §7's "rename updates `device_state.json`" is then unnecessary (N9, NIT) |
| R15 | PARTIAL | Read-time plus write-time checks and the solo-box warning are folded. The charset, casefold and no-leading-`-` rule are not stated. "The one validator" is in `maxim/devices.py`, so the CLI `--device` path must go through `maxim.devices` to get the read-time check (N10, NIT) |
| R16 | PARTIAL | `entries()`, `DiagnosticReport`/`CheckResult`, the named typed errors, `MAXIM_DEVICE` with a scrub, and the `peer connect` naming are all folded. Three things are not. "`_API_VERBS` and the README verb count update" is now wrong (N6). "`DeviceEntry` a frozen view over `RobotConfig`, not a copy" contradicts itself (N11). CC3 says "only those that persist or cross a wire" without naming which ones; round 1 recommended path (a) for all five |
| R17 | FOLDED (no change) | No new input to `detect_role`. Still true in v2 |

## New findings introduced by v2

| # | Finding | Class | Fail-open/closed | Covered class? |
|---|---|---|---|---|
| N1 | A `LAYER_RULES["devices"]` entry for the single file `maxim/devices.py` is silently skipped: `audit_architecture` does `if not layer_dir.is_dir(): continue`. `api.py` has no entry, so the "like api.py" precedent does not exist | SHOULD-FIX | fail-open (guard checks nothing) | yes: vacuous guard (CLAUDE.md "verify the guard by deletion") |
| N2 | D10's import ban: `src/maxim/skills/` does not exist (the rule is vacuous); `embodied_runtime` was dropped from the list and has no `LAYER_RULES` key; `import maxim; maxim.devices.add(…)` gets past it; it bans whole modules, not the write verbs; and "the agent can never call them" is false while `BashTool` can run `maxim device add` | SHOULD-FIX | fail-open | partly: vacuous guard; the overclaim is new |
| N3 | `utils/format_version.py::check_format_version` only *returns* the version and never refuses newer. A hand-edited unquoted `_format_version: 2.0` is a YAML float and is read as legacy `"0.x"` | SHOULD-FIX | fail-open | no (new: YAML-root use of a JSON helper) |
| N4 | Two default markers: top-level `default:` and per-entry `primary:`. The legacy `--robot`/`--robot-name` path (`selfy.py::Maxim.__init__` → `resolve_robot_entry(cfg, self.name)`) falls back to `primary:` and never reads `default:`. Old builds sharing `~/.maxim` do the same. No precedence is given for when they disagree, or for a `default:` that names a missing entry | SHOULD-FIX | fail-open (connects a robot other than the operator's chosen default) | no |
| N5 | The writer's round-trip is unspecified. `RobotsConfig.from_dict` / `RobotConfig.from_dict` drop entry-level and top-level keys they don't model, so a load → mutate → dump writer silently deletes them. There is also no refuse-to-write-newer (`runtime/config_writer.py::_refuse_if_newer` is the precedent) | SHOULD-FIX | fail-open (operator data loss) | partly: persistence-config writer discipline |
| N6 | "`_API_VERBS` and the README verb count update" carries a stale R16 line. `maxim.devices` is a sub-module: it belongs in `__init__.py::_API_SUBMODULES` (next to `create`/`load`) and `__all__`. Putting it in `_API_VERBS` would make `getattr(api, "devices")` raise | NIT | fail-closed (loud) | yes |
| N7 | The verb named `connect` means two things. CLI `maxim connect` resolves, verifies and sets the default (= `devices.use`); Python `maxim.devices.connect` opens a live `RobotController`. PR 1 lists that `connect` among the "read verbs", but it opens a daemon session (and may release media, #1704). It also does not state the `robot_id`/`robot_name` split (R2) | SHOULD-FIX | n/a (naming); fail-closed for pinned entries | no |
| N8 | The scope of "requires an absolute `MAXIM_DATA_HOME`" is unstated. Today `utils/paths.py::data_home` accepts and caches a relative path | NIT | fail-closed | yes |
| N9 | "rename/forget update `default:` and `device_state.json` in the same locked transaction" is not atomic across two files. Because the state file is keyed by identity, `rename` doesn't need to touch it | NIT | fail-closed (a stale cache entry is re-verified) | yes |
| N10 | The read-time mesh-name check lives in `maxim/devices.py`, but the resolver lives in `hardware/`. Nothing states that CLI `--device` goes through `maxim.devices` | NIT | fail-closed | yes (R15) |
| N11 | "`DeviceEntry` (frozen) is a view over `RobotConfig`, not a copy" contradicts itself. `RobotConfig` is a mutable `@dataclass` with `eq=True`, so it is unhashable, and it holds a mutable `config` dict. A frozen dataclass that wraps it is neither frozen in effect nor hashable | NIT | fail-closed (TypeError on hash) | yes (R16) |
| N12 | D12 folds `embodied_device.md`'s `maxim.device.run_device`, which lives in `console/device.py` (it builds the device app). That is a *node/host* concern by §1's own table, so the fold puts Node and Device in one namespace and could pull `console`/FastAPI into `maxim.devices` | SHOULD-FIX (owner re-confirm) | n/a | no |

---

## N1. The `devices` LAYER_RULES entry is skipped by construction

`src/maxim/utils/audit.py::audit_architecture` loops over `LAYER_RULES` with `layer_dir = src_root / layer`, then `if not layer_dir.is_dir(): continue`. A key `"devices"` for the file `src/maxim/devices.py` is therefore never checked. Nothing fails, and the rule enforces nothing. The "like `api.py`" precedent does not exist either: `api.py`, `cli.py`, `create.py` and `load.py` have no entries. Round 1's R8 proposed this entry; v2 copied it without checking.

**Fix.** Pick one:
- (a) Make it a package, `src/maxim/devices/__init__.py`, with the write verbs in `devices/_write.py`. The `LAYER_RULES["devices"]` entry then works, and N2's per-symbol problem goes away, because the ban can target the writer module (see N2). `run_mypy_public.sh` lists the package's files.
- (b) Drop the claim.

The edges that matter run *into* `devices`, not out of it (N2), so (b) loses little. Whichever is chosen, prove the rule by deletion: plant a violating import in a test tree and see the audit fail.

## N2. The import ban: what `LAYER_RULES` can and cannot enforce

Banning `maxim.devices` from other layers is feasible for directory layers. `audit.py::target_layer("maxim.devices.x")` returns `"devices"`, and `_check_file` handles `from maxim import devices` through the per-name candidate path. Adding `"devices"` to the existing `must_not_import` lists of `tools`, `agents`, `runtime`, `planning`, `memory` and `bridges` works and creates no baseline debt. The holes in the ban as v2 writes it:

- **`skills/` is not a package.** `src/maxim/skills/` does not exist, so `LAYER_RULES["skills"]` is already skipped. Skills are data that `runtime/skill_matcher.py` loads from `skills_dir`. Listing skills gives assurance that nothing backs.
- **`embodied_runtime` was dropped.** R9 named it, but v2's list ("runtime, tools, agents and skills") does not. The audit has no `embodied_runtime` key, so one has to be added (`"embodied_runtime": {"must_not_import": ["devices"]}`). Then `ResolvedDevice`, `LaunchPlan` and `DevicePin` must live in `hardware/` (or `embodied_runtime/`), not in `maxim/devices.py`. Otherwise #1808's `embodied_runtime/launch.py::launch(plan, …)` cannot import the type it takes. v2 §7 puts "the entry types, the pin, the resolver" in `hardware/` but does not place `LaunchPlan` or `ResolvedDevice`. State it. Also add `"devices"` to `hardware`'s list, so the resolver never calls up into the validator.
- **`import maxim` + attribute access.** `maxim/__init__.py::__getattr__` lazy-loads every `_API_SUBMODULES` entry. `import maxim; maxim.devices.add(...)` records target `None` in the audit, and dynamic imports are out of scope by design (audit docstring). No `import maxim` exists today in `tools/`, `agents/`, `runtime/` or `embodied_runtime/`, so a CI grep for `\bmaxim\.devices\b` in those trees is cheap and closes the gap.
- **The ban is per module, not per symbol.** "bans importing them" (the write verbs) cannot be expressed in `LAYER_RULES`. Either ban the whole namespace (the runtime then uses only `hardware/` types and resolver, which is fine), or split the writes into their own module (N1 (a)) and ban that.
- **Overclaim.** §3b says "so the agent can never call them". `tools/` ships a `BashTool` (`tools/filesystem.py::_rebase_relative` references it) and file-write tools. An agent with shell access can run `maxim device add`, or write `<data_home>/robots.yaml` directly. The import ban is defense in depth against *code* written into the tool and runtime trees. The real boundary is the tool sandbox (`allowed_dirs`), and the C2 amendment should say so. That is the same honesty rule the note applies to `hardware_id` ("identity, not authentication").

**C2 amendment text.** C2 is stated in `docs/agents/persistence-config.md` §1 ("ONLY by operator-explicit one-shot CLI verbs … never by admin APIs"), not in DECISIONS.md. The DECISIONS entry must also amend the brief's §1 wording and its writer table (add a robots.yaml row: writer `hardware/…::write_robots_config`, a CI caller allow-list). Otherwise the brief keeps saying "CLI verbs only". The caller allow-list (`write_robots_config(` callers limited to the writer module and the `maxim.devices` write module) follows the existing `MESH_WRITER_MATCHES` grep in `.github/workflows/test.yml`, so it is feasible as specified.

## N3. `check_format_version` does not refuse newer files, and YAML types differ

`utils/format_version.py::check_format_version` returns the version string. Its docstring says a loader "can compare the returned version against an expected set and raise". It never raises, and a missing field is legal by contract ("Loaders MUST NOT raise on missing"). It is right that the helper works on a `yaml.safe_load` dict (`isinstance(data, dict)`). Two gaps:

- **Refuse-newer is loader code the note doesn't name.** Compare major.minor against the build's supported version in `RobotsConfig.from_yaml` and raise `RobotsConfigError`. Mirror `runtime/config_writer.py::_is_newer` / `_refuse_if_newer`, and give the writer the same refusal.
- **YAML scalars.** `_format_version: 1.0`, unquoted in a hand-edit, parses as `float`, so `isinstance(raw, str)` fails and the file reads as legacy `"0.x"`. Unquoted `2.0` would then load as legacy instead of being refused. The loader must treat a present-but-non-string `_format_version` as `RobotsConfigError`. `yaml.safe_dump` quotes the string `"1.0"`, so the writer's own output is safe.
- **Legacy noise (NIT).** Every hand-written file logs "pre-1.0 robots_yaml file … Re-save with this build" once per process. The only re-save is the writer, which strips comments (D5). Word the hint for robots.yaml, or keep it at INFO.

Add the `tests/integration/test_persistence_compat.py` row that round 1 asked for.

## N4. `default:` and `primary:`: two ways to mark the default

v2's precedence is `--device` > `MAXIM_DEVICE` > `default:` > single entry / `primary:`. The code paths disagree:

- The legacy path, which D3′ keeps with "today's exact behaviour", runs `selfy.py::Maxim.__init__` → `effective_robot_id = self._robot_id or self.name` → `hardware/config.py::resolve_robot_entry`. When the daemon name does not match an entry, it falls back to `get_primary()`, which reads `primary:` only. After `maxim connect rosie` writes `default: rosie`, `maxim --robot reachy_mini --mode live` connects the `primary:` entry. Two commands pick two robots.
- If `resolve_robot_entry` is changed to read `default:`, the legacy path's behaviour changes, against D3′.
- Old builds sharing `~/.maxim` ignore top-level keys (`RobotsConfig.from_dict` reads only `robots`), so they follow `primary:`.
- Unstated cases: `default:` naming a missing entry (left stale by a hand edit), and `default:` disagreeing with a `primary: true` on another entry.

**Fix (recommended, the strict and smaller option).** Drop `default:`. `use`/`connect` write `primary: true` on the chosen entry and clear it on the others, all through the one writer. One marker means every resolver, the legacy path and old builds agree, and `rename`/`forget` need no cross-key fix-up. The cost is a per-entry boolean as the store instead of a top-level pointer, which is what round 1's R5 proposed. **If `default:` is kept**, the loader must raise `RobotsConfigError` when `default:` names no entry or when it disagrees with an explicit `primary:`. The writer must keep both in sync. `resolve_robot_entry` must read `default:` before `primary:`, and the note must accept and document the change to the legacy path. Either way, this is an owner re-ask on D2′'s mechanism, not its meaning.

## N5. The writer's round-trip fidelity

`hardware/config.py::RobotConfig.from_dict` keeps only `type`, `primary` and `config`. `RobotsConfig.from_dict` keeps only `robots`, and skips non-dict entries. A writer that loads into these types, mutates them and dumps them deletes every key they do not model. That includes keys written by a newer build of the same `_format_version` minor, and an operator's custom entry-level keys. D5 accepts losing comments, not data.

**Fix.** Pick one:
- (a) The writer edits the raw `yaml.safe_load` mapping under the lock and validates it through the typed parser before the atomic write.
- (b) The typed parser is strict and refuses unknown entry-level and top-level keys, so nothing can be dropped silently. That matches the threat table's "strict typed fields" and makes the version bump the only way to add a key.

Also refuse to write a file whose `_format_version` is newer (N3). Add a round-trip test: load, then write, then load again, and check the mapping is equal for a fixture with every field.

**Home.** `hardware/robots_yaml_writer.py` is legal (it needs only `utils/atomic_io.atomic_write_text`, which exists, plus `filelock` and `yaml`). But the cited precedent co-locates the reader and writer (`peer/mesh_config.py::read_mesh_config` + `write_mesh_config`). `hardware/config.py` is 328 lines, so putting `write_robots_config` there keeps the parser/serializer pair and the round-trip test in one file, and adds no module. This is a preference, not a defect.

## N6. `maxim.devices` is a sub-module, not a verb

`src/maxim/__init__.py` has `_API_VERBS` (resolved via `getattr(api, name)`) and `_API_SUBMODULES` (`{"create": "maxim.create", "load": "maxim.load"}`). The namespace goes in `_API_SUBMODULES` and `__all__`. `_API_VERBS` and README's "21 verb-based functions" (`tests/unit/test_api_surface.py` parses that number) do **not** change. That §3b line is a leftover from R16's top-level `maxim.embody`, which D9 removed. Add `src/maxim/devices.py` (or the package files) to `scripts/run_mypy_public.sh`'s `FILES` next to `create.py`/`load.py`. That part is correct.

## N7. One verb name, two meanings, and the identity split

- §3.1: CLI `maxim connect <name>` = resolve + verify + refresh + set the default. §3b: `maxim.devices.use` is "what `maxim connect` calls", while `maxim.devices.connect` opens a live `RobotController`. The CLI is meant to be a "thin shell" over a same-named library verb, and here the names are crossed. Counting `maxim.connect` (stable, unpinned) and `maxim peer connect`, there are four `connect`s.
- **Fix.** Rename the Python controller opener (for example `maxim.devices.open(name) -> RobotController`). `maxim.devices.connect` can then be the CLI's verb, or the CLI's verb stays `use`. Move the controller opener out of PR 1's "read verbs": it opens a daemon session and can release media (#1704).
- The R2 remainder: state that `--device <entry>` / `plan_launch` pass `robot_id=<entry id>` and `robot_name=<entry's daemon name>` to `selfy.Maxim`. Today the CLI passes only `robot_name`, and `effective_robot_id = self._robot_id or self.name`. If the entry name were carried as `robot_name`, it would become the mDNS name (`rosie.local`). A pinned entry fails closed on that; an unpinned one dials the wrong name.

## N8–N11 (NITs)

- **N8.** Say whether "absolute `MAXIM_DATA_HOME` required" is a check local to robots.yaml (in `find_config_file` and the writer) or a change to `utils/paths.py::data_home`. `data_home` caches a relative `Path` unresolved, which every persisted surface uses. Also, `data_home()` calls `sys.exit` on `PermissionError`, which a library `add()` would hit. That is #1808's `sys.exit` ratchet class, already covered.
- **N9.** Cross-file "same locked transaction": name the lock order and the crash outcome. A leftover identity-keyed cache entry is harmless because it is re-verified. `rename` should not touch `device_state.json`.
- **N10.** State that CLI `--device`/`embody`/`connect` resolve through `maxim.devices` (resolver + validator), so the read-time collision check is on the CLI path too.
- **N11.** `DeviceEntry`: one constructor (`DeviceEntry.from_robot_config`) that *copies* into frozen fields, with `config` as a `MappingProxyType` or a tuple of items. "Not hand-maintained" is the real requirement; "not a copy" is not achievable with a frozen type over a mutable one. CC3: name the types. `DeviceEntry` and any `device_state.json` record type persist. Give all five path (a) (defaults + `extra`), since the namespace is experimental.

## N12. D12 folds a node concern into the device namespace

`docs/plans/embodied_device.md` row "`maxim.device.run_device(...)` (§2)" places `run_device` in `console/device.py` (F4, `console.device.build_device_app`). It runs Maxim *as* a device app with an injected `MaximHandle`, which is a **Node** in §1's own table, not a **Device**. Folding it into `maxim.devices` puts the two in one namespace, against §1's "one name namespace, separate stores" model. A re-export would also have to stay lazy so that `import maxim.devices` does not import `console`/FastAPI. #1808's body repeats the fold ("Proposed for #1806 (D12)"). **Ask the owner to re-confirm D12 with this framing.** If it is kept, `run_device` stays in `console/device.py` and `maxim.devices` re-exports it lazily, with that recorded in both plans.

## Stop-rule outcome

**Not met.** N1, N2, N3, N4 and N5 are fail-open. N3 (a JSON helper used on a YAML root), N4 (the dual default marker) and N7/N12 (namespace and verb shape) are not in an already-covered class. N1 and N2 are in the vacuous-guard class, but the guard is a gate input, so they cannot go to a follow-up issue. A delta round 2 on v3 is owed, scoped to N1–N5, N7 and N12. The NITs (N6, N8–N11) can be folded without another round.
