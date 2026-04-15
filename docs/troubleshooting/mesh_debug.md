# Mesh Debug — operator runbook for the Plan 4 Stage C surface

**Audience:** operators running a Maxim mesh (one leader + N peers, or N peers without a single leader).
**Scope:** the `mesh.yml` declarative config, the `init-mesh` / `add-node` / `remove-node` setup verbs, and the `drain` / `resume` / `list-drained` runtime state surface shipped across PRs #112 (C1), #113 (C2), and #118 (C3.1) plus the C3.2 follow-up.

If you are debugging a network-layer problem (DNS, TCP, TLS, Cloudflare tunnel) start in [peer_leader_connectivity.md](peer_leader_connectivity.md) instead. This doc covers symptoms above the network layer — the cluster is reachable, but routing is doing the wrong thing.

---

## The two-config-file split

Maxim's per-host config lives in two files. Knowing which one to look at is the first half of debugging anything in this doc.

| File | Path | Purpose | Who writes it |
|---|---|---|---|
| `peer.yml` | `~/.config/maxim/peer.yml` | Per-host credentials + leader URL + the **role-detection signal** (`runtime/role.py` reads its existence). | `maxim peer connect` |
| `mesh.yml` | `~/.config/maxim/mesh.yml` | Declarative cluster topology — `cluster_key` + `self` + `nodes[]`. The source of truth for `--node X` routing, drain/resume, and `list-nodes`. | `mesh_setup.py` only (CI grep allow-list) |

**Why two files?** `peer.yml` predates the multi-node mesh. It is the simple-single-leader config and the role-detection signal. `mesh.yml` is the multi-node topology surface. They coexist by design — Stage C verbs leave `peer.yml` in place because deleting it would silently break role detection.

Mutable runtime state lives in a third location and is **never** in `mesh.yml`:

| File | Path | Purpose | Who writes it |
|---|---|---|---|
| Drained-nodes list | `~/.maxim/util/drained_nodes.{role}.txt` | Names of nodes currently drained (one per line). Serialized via `filelock.FileLock` for cross-process safety. | `drain` / `resume` runtime verbs |

The Plan 4 Stage C invariant is **"`mesh.yml` is declarative; mutable state lives in `~/.maxim/util/`"** (Kubernetes spec-vs-status pattern). If you find yourself wanting to write to `mesh.yml` from a runtime code path, you are about to violate the invariant — write to `~/.maxim/util/` instead.

---

## The setup verbs (one-shot, operator-explicit)

All three of these mutate `mesh.yml` and live in [src/maxim/peer/mesh_setup.py](../../src/maxim/peer/mesh_setup.py). They are the **only** sanctioned writers of `mesh.yml` — the CI grep allow-list in `.github/workflows/test.yml` blocks any other caller of `write_mesh_config`.

| Verb | What it does | Exit codes |
|---|---|---|
| `maxim peer init-mesh [--force]` | Synthesize `mesh.yml` from existing `peer.yml`. `--force` overwrites with `mesh.yml.bak` backup. | 0 ok / 1 nothing-to-convert / 2 refused |
| `maxim peer add-node <name> --url <url> [--role peer\|leader] [--force]` | Append a node to `mesh.yml::nodes`. `--force` replaces an existing entry (does not duplicate). | 0 ok / 2 refused / 3 invalid args |
| `maxim peer remove-node <name>` | Remove a node from `mesh.yml::nodes`. Side effect: clears any drain state for that node. | 0 ok / 2 refused (incl. self-removal) |

The `--node <name> install <extras>` verb from Stage C3.3 is **not** a setup verb (it does NOT mutate `mesh.yml`). It composes drain → install → resume and lives in [src/maxim/peer/mesh_cli.py](../../src/maxim/peer/mesh_cli.py), delegating the HTTP body to [src/maxim/peer/cli.py](../../src/maxim/peer/cli.py)'s shared `_install_on_target` core — same core the positional-URL `maxim peer install` verb uses.

Pass `--help` to any of them for the full usage string.

### Walkthrough: cold-start a 3-node mesh

```bash
# On node-A (leader):
maxim peer init-mesh                                            # synthesize mesh.yml from peer.yml
maxim peer add-node node-b --url https://node-b.example.com    # add peer
maxim peer add-node node-c --url https://node-c.example.com    # add peer
maxim peer list-nodes                                           # verify topology
```

### Walkthrough: tear out a flaking peer

```bash
maxim peer --node node-c drain        # stop routing to node-c
maxim peer list-drained               # confirm it's drained
maxim peer remove-node node-c         # remove from topology (clears drain state automatically)
```

The `remove-node` verb intentionally clears the drain entry for `node-c` so you don't leave an orphan in the drain state file. If the filelock acquisition warns ("could not acquire drain lock within 5s"), the removal **still proceeds** — the warning means the mesh.yml mutation succeeded but the drain state file may briefly contain a stale entry. Resolve by hand-editing `~/.maxim/util/drained_nodes.{role}.txt` or by running `maxim peer --node node-c resume` followed by `remove-node` again.

### Walkthrough: install an extra on a named mesh node (Plan 4 C3.3)

```bash
maxim peer --node mac-studio install semantic          # drain → install → resume
maxim peer --node mac-studio install semantic,llm-torch
maxim peer --node mac-studio install sentence-transformers  # raw pip package also works
```

The `--node X install` verb composes **drain → install → resume** around the shared `_install_on_target` HTTP core that the existing positional-URL `maxim peer install` verb also uses. The URL and cluster key are resolved from `mesh.yml::nodes` by name, so you never type a URL.

**Was-drained sticky semantics:** if `mac-studio` was already drained by the operator before the install (for an unrelated reason — e.g. you drained it manually to debug a different issue), the install verb skips BOTH the drain step AND the auto-resume. Your prior drain stays in place after the install completes. This matches the "operator intent is sticky" rule from C2.

**Failure mode:** if the drain succeeds but the install itself fails (network error, 403 disabled, pip resolution error, etc.), the node is **left drained** with a loud stderr message pointing at the manual resume command:

```
✗ Install failed. Node 'mac-studio' is STILL DRAINED.
  → Debug the failure above, then run:
      maxim peer --node mac-studio resume
```

This is deliberate. Auto-resume-on-failure would mask install failures behind a clean exit — exactly the silent-no-op shape CLAUDE.md's structural-enforcement lesson warns against. If you see this message, investigate the install failure first, then resume by hand when the node is ready.

**Self-install is refused.** `maxim peer --node <self> install <extras>` exits 2 and points you at `pip install pymaxim[<extras>]` directly. Remote-installing on yourself is a round-trip through your own admin endpoint; there's no reason not to use the local pip invocation.

**Positional URL is refused** in the `--node` form. If you want the no-mesh.yml fallback (e.g. you haven't run `init-mesh` yet and know the target's URL directly), use the positional verb: `maxim peer install <extras> https://target.example.com/v1`.

### "I ran `--node X install` and the node is still drained after"

Expected if the install **failed**. Check the exit code (non-zero) and the stderr "STILL DRAINED" message. After you've debugged and fixed the underlying install failure, run `maxim peer --node <name> resume` by hand to clear the drain.

If the install **succeeded** (exit 0, "Resumed 'X' after install" in stdout) but the node is still drained, check whether you pre-drained it before the install — the was-drained sticky rule means pre-drained nodes never get auto-resumed. Run `maxim peer --node <name> resume` to clear it.

---

## Symptoms → first place to look

### "I added a node and it isn't routing"

1. `maxim peer list-nodes` — does the node appear? If not, the `add-node` failed silently (it shouldn't — file an issue with the exit code).
2. `cat ~/.config/maxim/mesh.yml` — is the node in the file? If yes but `list-nodes` doesn't show it, the running daemon is reading a stale in-memory copy. Restart with `maxim peer restart`.
3. `maxim peer --node <name> status` — is the URL reachable? `add-node` does **syntax-only URL validation** (no DNS at parse time, matching the C1 rule). The first reachability check is at probe time.
4. Check `~/.maxim/util/drained_nodes.{role}.txt` — is the node accidentally in the drained list from a previous session? Run `maxim peer --node <name> resume`.

### "I drained a peer and it didn't come back after `resume`"

1. `maxim peer list-drained` — is the node still listed? If yes, `resume` failed; check the exit code.
2. `cat ~/.maxim/util/drained_nodes.{role}.txt` — is the entry present in the file? If the file says drained but `list-drained` doesn't, you have a per-role file mismatch — see "role detection picked the wrong file" below.
3. Restart the daemon — drain state is read at request-routing time, but cached state in long-lived router objects can lag the file by up to one request cycle on first call.

### "`init-mesh` says nothing-to-convert (exit 1)"

You don't have a `peer.yml` to synthesize from. Run `maxim peer connect <leader-url>` first to create one, then re-run `init-mesh`.

### "`init-mesh --force` says refused with a `.bak` already exists (exit 2)"

A previous `--force` run created `~/.config/maxim/mesh.yml.bak`. The C3.1 A2 fold added this guard to prevent a double-`--force` from overwriting your only good backup. Resolve by:

```bash
# Option 1: the .bak is the one you want to keep — restore it
mv ~/.config/maxim/mesh.yml.bak ~/.config/maxim/mesh.yml

# Option 2: the .bak is stale, throw it away
rm ~/.config/maxim/mesh.yml.bak
maxim peer init-mesh --force
```

### "`add-node` or `remove-node` rejected my YAML and I don't know why"

The `MeshConfig.__post_init__` validation rejects:

- Empty `nodes` (C3.1 E7 fold)
- A `self` value that doesn't match any node name (C3.2 A1 fold)
- Non-yaml-safe characters in `cluster_key`, node names, or URLs — currently any newline or `<whitespace>#` (the latter would be silently truncated by the parser otherwise)

The error message names the offending field. If the message is unclear, file an issue — these validators exist to make round-trip corruption impossible, so any unclear rejection is a doc bug.

### "Role detection picked the wrong file"

Mutable state is stored per role: `drained_nodes.leader.txt` vs `drained_nodes.peer.txt` vs `drained_nodes.solo.txt`. If a node's effective role flips (e.g. you renamed a leader to a peer), the old file is stranded. Check both:

```bash
ls -la ~/.maxim/util/drained_nodes.*.txt
echo "MAXIM_ROLE=$MAXIM_ROLE"
maxim doctor          # role_divergence will appear in the report if config + env disagree
```

The role-detection decision order (Plan 2 R2a) is: `MAXIM_ROLE` env var → `mesh.yml::self` role → `peer.yml` existence → `--llm` flag without peer config → default leader. If you need to override, set `MAXIM_ROLE=peer` (or `leader` / `solo`) in the environment before starting the daemon.

---

## Concurrency model — what's locked, what isn't

| Surface | Concurrency safety | If two operators race |
|---|---|---|
| `mesh.yml` setup verbs (`init-mesh` / `add-node` / `remove-node`) | **No filelock** — operator-serial by construction | Last writer wins, silently drops the other's mutation |
| `drained_nodes.{role}.txt` (`drain` / `resume`) | **`filelock.FileLock` with 5s timeout** | One waits, both succeed |
| `mesh.yml` reads (router, `list-nodes`) | Read-only, no lock needed | N/A |

This is a **conscious trade-off** (E5/A3 fold from the C3.2 pre-merge review). Setup verbs are operator-explicit one-shots — adding a filelock would punish the common case. Drain/resume are runtime verbs that can fire from automation, so they are locked. **If a future Stage C feature wires automatic writes to `mesh.yml`** the lock gap becomes a correctness bug — the right resolution is "don't write to `mesh.yml` from automatic paths," not "add a filelock to setup verbs." See `mesh_setup.py` module docstring for the full discussion.

---

## File layout cheatsheet

```
~/.config/maxim/
├── peer.yml          # leader URL + API key (left in place by every Stage C verb)
├── mesh.yml          # declarative topology (writers: mesh_setup.py only)
└── mesh.yml.bak      # only present if init-mesh --force has been run (max 1)

~/.maxim/util/
├── drained_nodes.leader.txt    # one drained node per line
├── drained_nodes.peer.txt
├── drained_nodes.solo.txt
└── ...other runtime state files
```

---

## Related runbooks

- [peer_leader_connectivity.md](peer_leader_connectivity.md) — network-layer diagnosis (DNS, TCP, TLS, Cloudflare tunnel)
- [peer_diagnosis_runbook.md](peer_diagnosis_runbook.md) — "my peer isn't talking to the leader" playbook
- [http_debugging.md](http_debugging.md) — unified HTTP client + structured event surface
- [leader_proxy_debug.md](leader_proxy_debug.md) — admission control, request forwarding, GPU header injection
- [remote_update.md](remote_update.md) — `maxim peer update` / `maxim peer restart` troubleshooting
- [mesh.md](mesh.md) — historical post-mortem of the deleted R0 mesh scaffolding (not the current surface)

For the architectural rationale of the two-file split, the structural enforcement around `write_mesh_config`, and the load-bearing invariants, see [../architecture/structural_enforcement.md](../architecture/structural_enforcement.md).
