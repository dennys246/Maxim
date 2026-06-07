# Upgrading Maxim

Upgrade contract for users moving between Maxim versions. Covers what survives a `pip install --upgrade pymaxim`, what may warn but still work, and what (if anything) requires manual action.

## TL;DR — 0.8 → 1.0

Run the upgrade. Your existing state in `~/.maxim/` will load. If you previously saved bio-system files with 0.8, you'll see one warning per file type the first time 1.0 reads them. The warning is informational — the file loads correctly, and the next save re-writes it in the 1.0 format.

```bash
pip install --upgrade pymaxim
maxim doctor              # sanity check: should report all green
maxim --sim "test recall" --interactive false --sim-max-turns 3
```

If `maxim doctor` is green and the smoke sim runs, the upgrade is complete. No manual migration step is needed today.

## What survives the upgrade

The 1.0 upgrade is **state-preserving**. Every file Maxim writes under `~/.maxim/` continues to load.

| Path | Contents | 0.8 → 1.0 behavior |
|---|---|---|
| `~/.maxim/agents/<name>/hippocampus.json` | Episodic memory + associative graph + episodes | Loads. Re-save adds `_format_version: "1.0"`. |
| `~/.maxim/agents/<name>/nac.json` | Causal links + reward biases + outcome index | Loads. Re-save adds `_format_version: "1.0"`. |
| `~/.maxim/agents/<name>/atl.json` | Semantic concepts + concept graph | Loads. Re-save adds `_format_version: "1.0"`. |
| `~/.maxim/agents/<name>/scn.json` | Temporal signatures + circadian phase priors | Loads. Re-save adds `_format_version: "1.0"`. |
| `~/.maxim/sessions/<id>/` | Session reports, action JSONL, snapshot envelopes | Loads. Envelope-bearing files carry both `schema_version` (int) and `_format_version` (string) at root. |
| `~/.maxim/util/last_probe_status.json` | Peer probe outcomes | Loads. Pre-1.0 dict-keyed-by-URL shape is auto-wrapped under `entries`. |
| `~/.maxim/util/active_llm_model.{role}.txt` | Persisted LLM model selection | Plain text — unchanged. |
| `~/.maxim/util/drained_nodes.{role}.txt` | Drain state | Plain text — unchanged. |
| `~/.maxim/foundry/` | Generated SEM components and batch metadata | Loads. |
| `~/.maxim/models/` | Downloaded GGUF / safetensors weights | Untouched. |
| `~/.maxim/components/` | User-authored SEM components | YAML — unchanged. |

The contract is **append-only on the wire format**: 1.0 reads anything 0.8 wrote, and 1.0+ readers default unknown root fields to "0.x" with a single warning per file type.

## What might warn but still work

When 1.0 first loads a file 0.8 wrote, you may see one warning per file type. The logger name varies per loader — NAc emits under `maxim.decisions.nac`, ATL under `maxim.memory.atl`, hippocampus under `maxim.memory.hippocampus_persistence`, SCN under `maxim.time.scn`. The message text always contains the literal string `pre-1.0 <file_type>` — grep on that, not on the logger name, when scanning for upgrade warnings:

```
WARNING maxim.decisions.nac: Loading pre-1.0 nac file (no _format_version
at root); assuming 0.x. Re-save with this build to upgrade the file format.
Future warnings for nac suppressed.
```

This is informational. The file loads. The warning is one-shot per file type per process — loading 12 NAc snapshots only warns once. Re-saving the file (any normal session that touches the bio-system) stamps the 1.0 version field at root, and future loads no longer warn.

If you want to silence warnings without running a session, you can re-save in place. Each bio-system has the same `load(path)` / `save(path)` pair; the constructor signatures differ slightly:

```python
# Optional: silence the one-time warning by re-saving in place.
from maxim.memory.atl import ATL
from maxim.decisions.nac import NAc, NACConfig
from maxim.memory.hippocampus import Hippocampus
from maxim.time.scn import SCN

agent_dir = "/Users/you/.maxim/agents/scout"

atl = ATL()
atl.load(f"{agent_dir}/atl.json")
atl.save(f"{agent_dir}/atl.json")

nac = NAc(NACConfig())
nac.load(f"{agent_dir}/nac.json")
nac.save(f"{agent_dir}/nac.json")

hippo = Hippocampus()
hippo.load(f"{agent_dir}/hippocampus.json")
hippo.save(f"{agent_dir}/hippocampus.json")

scn = SCN()
scn.load(f"{agent_dir}/scn.json")
scn.save(f"{agent_dir}/scn.json")
```

## peer.yml → config.json migration (0.9.3+)

If you ran Maxim as a peer node before 0.9.3, your peer configuration lived in
`~/.config/maxim/peer.yml`. Starting with 0.9.3, the primary config layer is
`~/.config/maxim/config.json`. Maxim migrates the two automatically.

### What the shim does

On the first startup after upgrade, `load_config()` checks three conditions before
writing anything:

1. `~/.config/maxim/config.json` does **not** yet exist.
2. `~/.config/maxim/peer.yml` **does** exist.
3. No cloudflared config is found at `~/.cloudflared/config.yml` (or `.yaml`) or
   `/etc/cloudflared/config.yml` (or `.yaml`) — this prevents a machine that was
   previously both leader and peer from having its role silently flipped.

When all three conditions hold, the shim:

- Creates `~/.config/maxim/config.json` with `role: peer` and
  `lanes.large.{remote_url, remote_model, remote_api_key_ref}` copied from your
  peer.yml.
- Writes your inline API key to `~/.config/maxim/api_key` (mode `0600`), then stores
  the file path as the `remote_api_key_ref` reference in `config.json`. The key is
  never written inline to `config.json`.
- Leaves `peer.yml` in place — it is never deleted. Maxim continues to read it as a
  fallback on machines where the migration is skipped (e.g., cloudflared-present
  leaders).

If a write error occurs (permission denied, disk full), the shim logs a WARNING and
returns without setting the "done" flag, so the next `load_config()` call retries
automatically.

### Verifying the migration

After the first startup following the upgrade:

```bash
maxim config get lanes.large.remote_url
```

Expected output (value will be your peer's actual URL):

```
lanes.large.remote_url: https://maxim.yourdomain.com   [source=config]
```

The `[source=config]` marker confirms Maxim is reading from `config.json`. If you
see `[source=env]` instead, a `MAXIM_LANE_LARGE_REMOTE_URL` environment variable
is shadowing the config file — the migration still ran, but the env var takes
precedence. Unset it to let `config.json` be the sole source.

You can also inspect the full migrated state:

```bash
maxim config get
```

### If the migration did not run

The shim skips when any of the three trigger conditions are unmet:

- **cloudflared is installed:** the machine is treated as a potential leader and the
  shim intentionally does nothing. Manually set your role in `config.json` if needed:
  `maxim config set role peer`.
- **peer.yml is missing or malformed:** run `maxim peer test <url>` to confirm your
  peer is reachable, then create `~/.config/maxim/peer.yml` manually and restart.
- **config.json already exists:** the migration only runs when the file is absent.
  Use `maxim config get` to inspect the existing config.

### If the API key file already existed with a different key

The shim does not clobber an existing `~/.config/maxim/api_key` file when it
contains a different key. In that case `remote_api_key_ref` is left blank in
`config.json` and a WARNING is logged. Wire it manually:

```bash
maxim config set lanes.large.remote_api_key_ref ~/.config/maxim/api_key
```

## What requires manual action

Nothing today. The 1.0 upgrade is designed to be a no-op for users with existing state.

If a future minor release (1.1+) introduces a structural change a previous build cannot read cleanly, that release will:

1. Bump `_format_version` to `"1.1"` (or whatever) for the new shape.
2. Ship a migration in `maxim.memory.snapshot`'s envelope migration registry for envelope-conformant files (Hippocampus, NAc, ATL, SCN, PerceptTraceBuffer, CrossLayerGraph).
3. Document the change in this file under a new section.

Until that happens, no migration step is required.

## Future: `maxim migrate`

A `maxim migrate` CLI verb is reserved for a future release. The verb will:

- Scan `~/.maxim/` for state files that pre-date the current build.
- Re-save them through the appropriate loader/saver pair to upgrade the format version field (idempotent on already-upgraded files).
- Print a per-file summary of what was upgraded.

It is **not** required for 0.8 → 1.0. The verb is a convenience, not a correctness gate. Until it ships, normal use of the affected agent re-saves the file naturally.

## Downgrade (1.0 → 0.x)

Downgrading is **not supported as a contract**. 0.8 cannot read files that carry the new envelope fields and may either ignore them silently or raise depending on the loader. If you need to roll back, restore `~/.maxim/` from a backup taken before the 1.0 upgrade.

If you anticipate a possible downgrade, copy `~/.maxim/agents/` and `~/.maxim/sessions/` somewhere safe before running the upgrade.

## Reporting upgrade issues

If `maxim doctor` reports a fail or a smoke sim raises after the upgrade:

1. Capture the full output: `MAXIM_LOG_FILE=/tmp/maxim_upgrade.jsonl maxim doctor 2>&1 | tee /tmp/maxim_doctor.log`
2. File at <https://github.com/dennys246/Maxim/issues> with the version (`maxim peer version` or `python -c "from maxim import get_version_info; print(get_version_info())"`), the doctor output, and a redacted copy of the failing file from `~/.maxim/`.

## Reference

- Persistence-format contract: [`src/maxim/utils/format_version.py`](../../src/maxim/utils/format_version.py)
- Envelope migration registry: [`src/maxim/memory/snapshot.py`](../../src/maxim/memory/snapshot.py)
- Regression guard test: [`tests/integration/test_persistence_compat.py`](../../tests/integration/test_persistence_compat.py)
