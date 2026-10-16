# Upgrade safety: advisory check + upgrade manifest (approach note, #1295)

Status: APPROACH NOTE, revised after the gate design pass of 2026-10-15 (three DO-NOT-BUILD findings folded in; see "Design pass" below), owner decisions recorded 2026-10-15. No code yet. Target: **1.3.3** (owner).

## Why existing infrastructure, and nothing new

This rides on:
- doctor (`doctor/checks.py` `CheckResult`, including the `info` status);
- `utils/http`;
- package data (`src/maxim/_data/`, already shipped via `[tool.setuptools.package-data]`).

The one new artifact is a data file, kept append-only like the config schema fixtures. There is no new persisted state beyond one small "seen at startup" record.

## 1. Doctor: published advisories and yanks for the installed version

- **Running version.** The in-memory `maxim.__version__`, never the installed metadata: `_get_current_version` reads importlib metadata, which an editable install leaves stale. Dev detection goes through ONE function. Today there are three: `leader_proxy._detect_install_mode`, which accepts a `.git` directory only; `__init__._has_git_metadata`, which accepts a directory or file and so handles worktrees; and `simulation/report.py`. They disagree for worktrees.
- **Source.** PyPI's JSON API for `pymaxim`:
  - the per-release `vulnerabilities` list (`id`, `aliases`, `fixed_in`, `withdrawn`; populated from OSV);
  - the per-file `yanked` / `yanked_reason`.
  PyPI is an origin pip already contacts, so this adds no new third party. A yank with a reason is a maintainer-controlled channel that takes effect at once, ahead of the GHSA → OSV import lag. *(Field shapes INFERRED; verify against the live API before building.)*
- **Response hygiene.**
  - Drop `withdrawn` entries.
  - Dedupe GHSA and PYSEC entries by `aliases`.
  - Choose the fix as the smallest `fixed_in` greater than the installed version, or say "no fixed release yet".
- **Rows.** "Security":
  - **FAIL** per advisory (ID, summary, fix, exact `pip install -U` command), and for a yanked installed version (with the reason).
  - **OK**, worded "no published advisory or yank for <version>", never "secure".
  - **INFO**, "could not check (offline / timeout / unparseable)". Never OK without an answer.
  - **Dev checkouts** (git metadata present): matches show as INFO, "a dev checkout at <hash> may already contain the fix".
- **Network.** Only the `maxim doctor` CLI fetches. It goes through a registered `pypi` endpoint in `utils/http` (`internal=False`), the same host pip uses. `maxim.diagnose()` and the console's `/api/diagnose` read the cache only and never fetch, so their latency and outbound behaviour do not change (CHANGELOG line). The CLI fetches within doctor's latency budget (about 2 s per probe, llm-routing brief §4). The result is cached in `~/.maxim/util/advisory_cache.json` (`_format_version`, `atomic_write_json`), keyed by version, with a TTL of about 24 h. A cache stamped in the future counts as stale. The cache is shared across worktrees and venvs, which is fine because it is keyed by version.
- **Startup.** A leader **reads the cache only and never touches the network**, then logs one WARNING per process if the cache shows an advisory. It never refuses to start on an advisory: a feed that could stop leaders is a remote kill switch. The conftest network guard keeps tests off the network.
- **Opt-out.** A `doctor.advisory_check` config key (owner decision 2026-10-15). When it is off, nothing fetches, and every cache reader (startup, `diagnose()`, `/api/diagnose`) checks the key and the 24 h TTL. So an older cache neither warns nor lingers.
  - **Cost:** a new `doctor` section means a `CONFIG_FORMAT_VERSION` minor bump, sequenced after any pending format change, plus entries in both config fixtures.
  - **Downgrade consequence:** rolling back to an older build hits that build's refusal to rewrite a newer config. That is the same rollback an advisory or yank might prompt, so the doctor row's fix text must say it (`maxim config downgrade`).

## 2. Upgrade manifest: what changed, diffed rather than counted

- **File.** `src/maxim/_data/upgrade_manifest.json`, under `_data/` so it ships. Its presence is asserted by extending `scripts/audit_release_build.py`, which already audits `_data/` in the `release-build` CI job; no separate test.
- **It is a cross-version wire format.** The OLD running process parses the NEW on-disk file. It carries a root `_format_version`.
  - Unknown keys are ignored.
  - A higher MINOR is read normally.
  - An unknown MAJOR treats every new id as `requires_cold_restart`, which `acknowledge` can still override. If the entry ids themselves cannot be extracted, it falls to the corrupt path: fail closed, recover with `manifest-unreadable`.
  - So a later shape change can never brick remote restart on older leaders.
- **Entries are keyed by a stable id** (an issue or PR slug), with:
  - `requires_cold_restart`;
  - `on_machine_steps`;
  - `checks` (doctor check ids);
  - an optional `supersedes`, which corrects a wrong entry without editing it.
- **Append-only** by lint (extend `lint_config_schema_append_only.py`), plus a test that every `checks` id resolves in the check registry (below). There is no release-filled field: the id diff never needs a version, and "the first release that ships the file" is a constant in code.
- **Check registry (new work).** Doctor has no addressable check ids today: `CheckResult` has `name` and an optional, shared `retry_id`, and `run_all_checks` runs everything. Add a `CHECKS: dict[str, Callable[[], CheckResult]]` registry with unique ids, used by the manifest, `doctor --upgrade` and the startup check.
- **No version arithmetic and no marker file.** At startup the process loads the manifest's entry ids into memory. "What's new" is always the set of ids **on disk now** minus the ids **this process loaded**. pip upgrades, `git pull` + editable installs, hand upgrades on the machine, and back-to-back updates all land in the same file. That one rule covers every path, including the dev path, where `__version__` does not move.
- **Reading the target manifest** is plain JSON from `Path(maxim.__file__).parent / "_data/upgrade_manifest.json"`, falling back to `importlib.metadata.distribution("pymaxim").locate_file(...)` if the install moved. It is never a subprocess that imports the new package with the leader's environment and secrets.
- **Missing or corrupt manifest.**
  - Corrupt: fail CLOSED (refuse the restart, say why).
  - Missing while the INSTALLED version is at or above the first release that ships the file: also fail CLOSED (it means a partial install, e.g. the pip or git "rollback INCOMPLETE" paths). The installed version comes from importlib metadata, which is fresh after a pip upgrade; it is not `__version__`. **Bounded on the dev path:** an editable install's metadata can be stale, so a dev checkout left incomplete by a git rollback may read as "unknown, warn" rather than failing closed. Accepted, because dev checkouts are operator-run, and it is stated here.
  - Recovery for both, from a peer: re-run `maxim peer update`, which reinstalls the files. Or override the restart with the reserved id `manifest-unreadable` in `acknowledge`.
  - Missing below that version: "unknown" (a downgrade), warn but do not block.
- **Enforcement, stateless.** `/v1/admin/restart` diffs the in-memory and on-disk manifests at request time. If any new entry `requires_cold_restart`, it refuses (409), and the steps go **inside the `error` string**, because older peer clients print only `error` for a 409.
  - **Override:** `{"acknowledge": [<ids>]}` must list exactly the outstanding ids (no blanket force). The request is logged with the peer IP. So a leader reachable only through the tunnel is never locked out.
  - **Recovery is automatic:** a cold restart loads the new manifest, and the diff empties.
- **One admin operation at a time.** A non-blocking process-wide admin lock covers update, install, restart and llm-swap.
  - A concurrent request gets **423 Locked**, with the reason in `error`. Old `update_on_target` clients mis-render a 409 as "dirty working tree" but print `error` for unhandled statuses.
  - **Lock order:** the admin lock is outermost; the swap's own `_swap_lock` nests inside it.
  - It is held until `install`'s background pip thread finishes, not merely until the 202 is returned, and it is held through `execv` on restart.
- **Wire changes.**
  - `/v1/admin/update` responses (both the pip and the dev paths) gain `upgrade_entries` (id, `requires_cold_restart`, steps). `maxim peer update` prints them and never chains a restart across a cold-restart entry. Old clients ignore the field.
  - `restart_on_target` gains `acknowledge=`, surfaced as `maxim peer restart --acknowledge <id,...>`, threaded through both callers (`peer/cli.py` and `peer/mesh_cli.py`).
  - On a refused restart, the mesh restart flow keeps its existing "STILL DRAINED … run resume" message (`mesh_cli.py`). No undrain logic is needed; the refusal's `error` text is added to that message.
- **Limit.** This protects upgrades only once the RUNNING version contains this logic.

## 3. Startup check and `maxim doctor --upgrade`

- **Seen at startup.** A leader start runs the `checks` of every manifest entry not yet recorded as satisfied, and WARNs on each failure, because "startup succeeded" does not prove the on-machine steps were done.
  - **Advancing:** an entry is added to `~/.maxim/util/manifest_seen.{role}.json` only once ALL its checks pass. A failing check therefore keeps warning on every start instead of being silenced after one.
  - **Format:** written with `_format_version` and `atomic_write_json`.
  - **Scope:** only leaders run this at startup. Solo and peer installs get the same checks through `maxim doctor --upgrade`, and a one-line startup INFO points them there when unseen entries exist.
- **`maxim doctor --upgrade`** runs the same checks on demand. The first entries reference doctor checks that already exist; a new release adds an entry, not a new code path.

## Stated assumptions

- `/v1/admin/install` cannot pin a pymaxim version (`_SAFE_PKG_RE` rejects specifiers), so it is not a bypass of the restart gate.
- The admin `restart` re-execs in place (`os.execv`, same environment). That is why a cold restart must mean a restart outside the process.

## Design pass (2026-10-15), folded

DO-NOT-BUILD, all folded:
1. `(running, target]` was empty in dev mode, and version-keyed entries clashed with append-only → entries keyed by id; diff the startup set against the on-disk set.
2. The marker was bypassable and could go stale (hand upgrades, back-to-back updates, concurrent updates, downgrades) → stateless diff plus an admin lock.
3. "Cleared on startup" proved nothing → startup runs the new entries' checks.

SHOULD-FIX, all folded:
- no subprocess;
- missing manifest fails closed at or above the first shipping version;
- acknowledge-by-id override, with the steps in `error`;
- no network on startup;
- a feed query that does not leak the host's version to a third party, using PyPI and yanks;
- response hygiene;
- packaging under `_data/`, with the wheel test being the `audit_release_build.py` assertion;
- the in-memory version.

The NITs are folded too.

## Owner decisions (2026-10-15)

1. **Feed:** PyPI JSON (per-release vulnerabilities and yanks). No new third party.
2. **Opt-out:** the `doctor.advisory_check` config key, not an env var.
3. **A known advisory NEVER blocks startup.** Warn only; a feed must not be a remote kill switch.
4. **Target:** 1.3.3.

## Guards to build with it

Each proven by deleting its mechanism:
- doctor rows against a fake PyPI (advisory / withdrawn / yanked / none / offline / garbage);
- the stateless restart refusal across a hand upgrade and across two back-to-back updates;
- the acknowledge override (exact ids only);
- the admin lock;
- a missing manifest at or above the shipping version;
- the append-only lint;
- the `audit_release_build.py` manifest assertion;
- startup never touches the network (empty cache, fetch deleted → no outbound call);
- an advisory never blocks startup;
- a future-stamped cache counts as stale;
- the seen-set advances only when an entry's checks pass;
- the steps appear inside `error` on the restart refusal;
- 423 on a concurrent admin request, and the lock held across the install thread;
- an unknown manifest MAJOR is treated as cold-restart and stays overridable.

The rule "a release that needs on-machine steps adds a manifest entry" cannot be mechanized directly, so it is backlog row M55 (`docs/plans/outstanding.md`).
