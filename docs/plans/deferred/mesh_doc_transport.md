# Mesh Doc Transport — standardized small-document exchange between peer-mesh nodes

**Status:** Shell plan — tracking only, no active work. Filed 2026-04-15 after the Plan 4 C3.3 ship surfaced the question "how do agents on different mesh nodes actually communicate?"

**Scope:** ~400-600 LOC production code + tests. New endpoint family (`/v1/mesh/docs/*`), new state-layer namespace (`~/.maxim/util/mesh_docs/`), new CLI verb (`maxim peer docs put|get|ls|rm`), new CI grep allow-list entry for the admin endpoint.

**Target version:** Roadmap stage C9 (see [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) §4). No hard release gate — parallel to C4/C5/C6, probably slots into 0.5 or 0.6 depending on when multi-agent coordination becomes a user-visible blocker.

**Gates:** None. This is a new primitive for the "mesh as coordination substrate" thesis, not a correctness fix.

**Depends on:**
- Plan 4 C1 ([mesh_config.py](../../src/maxim/peer/mesh_config.py)) — `mesh.yml::nodes` is how operators address doc-drop targets by name
- Plan 4 C2 ([drain_state.py](../../src/maxim/peer/drain_state.py)) — same `~/.maxim/util/` mutable-state layer + `filelock.FileLock` RMW pattern
- Plan 4 C3.3 ([install_core.py](../../src/maxim/peer/install_core.py)) — the admin-endpoint + shared-core + CI-grep allow-list pattern this plan replicates

**Enables (future work that expects this):**
- Multi-agent mesh coordination — the killer use case. Maxim agents on separate nodes today have zero structured channel for "share this context with your sibling agent." Doc-drop is the minimum useful primitive for that.
- Multi-session Claude collaboration — parallel Claude sessions on the operator's leader + peer terminals can exchange context via a standardized inbox/outbox rather than manual clipboard juggling.
- Stage C7 cluster key rotation — a rotation broadcast verb could use doc transport to announce the new key to peers before switching over.
- Stage C4.5 auto-drain announcements — the router could post a doc explaining *why* a node was auto-drained (e.g., "4 consecutive timeouts, backoff triggered at 2026-04-15T12:34:56Z") so operators have an audit trail.
- Mother Maxim precursor — a cross-node memory-fragment exchange is the lightweight version of what Mother Maxim would formalize. Doc transport would be the substrate; Mother Maxim would be a specific consumer.

---

## The problem

The Maxim mesh today has **three** cross-node communication channels and none of them are a general-purpose structured-document exchange:

1. **Inference traffic** through `_MaximPeerBackend` (`/v1/chat/completions`). LLM prompt-in, completion-out. Not a structured-doc channel — the prompt IS the doc, and there's no persistence on the receiving side.
2. **Admin endpoints** like `/v1/admin/install`, `/v1/admin/update`, `/v1/admin/restart`. Operator-explicit one-shot verbs. Persistence is side-effectual (pip install, git pull). Not addressable — there's no way to "drop this JSON to node X and retrieve it later."
3. **Structured logs** via `MAXIM_LOG_FILE` + `maxim peer logs`. Read-only from the operator side; not a channel the nodes themselves use to exchange state.

The gap: **there is no way for an agent on node A to deposit a structured document that an agent on node B can read later.** Today the workaround is:

- Operator manually copies text between terminal sessions (current state for multi-Claude workflows)
- Agents share a git repo and push/pull (heavyweight, requires every agent to be a git client)
- Agents write to a shared filesystem mount (assumes one exists — doesn't on a Cloudflare-tunneled deployment)

None of these scale beyond two nodes or one operator.

---

## Proposed shape

A **standardized small-document KV-drop** addressed by `(namespace, key)` tuples, stored in the C2 mutable-state layer, served by a new admin endpoint family, fronted by a CLI verb.

### Storage layer

- **Root:** `~/.maxim/util/mesh_docs/` (role-scoped would be `~/.maxim/util/mesh_docs.{role}/` if roles need isolation — **open design question, see Q2 below**)
- **Hierarchy:** `mesh_docs/<namespace>/<key>.{json,md}` — namespace is a flat string, key is operator-provided, extension determines content-type
- **Size cap:** 1 MB per doc, 100 MB per namespace (configurable via env var, sensible defaults)
- **Locking:** `filelock.FileLock` per-namespace RMW, same pattern as `drain_state.py` — unifies the mesh state-layer locking approach
- **Atomic writes:** `maxim.utils.atomic_io.atomic_write_text` (or `atomic_write_secret` if the doc contains credentials — **open design question, see Q1 below**)
- **TTL + sweeper:** default 24h TTL; a background sweeper (or doctor-invoked cleanup) removes expired docs. Namespace-wide TTL overrides per-doc TTL.

### HTTP endpoints

All on the leader proxy, bearer auth via cluster key (C7 per-peer identity is future work — documented as a known limitation in v1):

| Verb | Endpoint | Purpose |
|---|---|---|
| `PUT` | `/v1/mesh/docs/<namespace>/<key>` | Store (body: raw JSON or markdown, Content-Type determines extension) |
| `GET` | `/v1/mesh/docs/<namespace>/<key>` | Retrieve single doc |
| `GET` | `/v1/mesh/docs/<namespace>` | List keys in namespace (metadata only — name, size, mtime, TTL expiry) |
| `DELETE` | `/v1/mesh/docs/<namespace>/<key>` | Delete single doc |
| `GET` | `/v1/mesh/docs` | List all namespaces on this node (admin-only) |

Response shapes follow the existing admin-endpoint JSON conventions (`{"status": "ok", ...}` on success, `{"error": "..."}` on failure). Error codes mirror `install_on_target`'s contract (401 auth, 403 disabled, 404 not found, 413 too large, 429 rate limited).

### Shared core (mirroring install_core.py pattern)

New `src/maxim/peer/mesh_doc_core.py`:

- `put_doc(url, key, namespace, doc_key, body, content_type) -> int`
- `get_doc(url, key, namespace, doc_key) -> tuple[int, bytes | None, str | None]` — returns (exit_code, body, content_type)
- `list_docs(url, key, namespace) -> tuple[int, list[DocMetadata]]`
- `delete_doc(url, key, namespace, doc_key) -> int`
- `list_namespaces(url, key) -> tuple[int, list[str]]`

CI grep lock on the `/v1/mesh/docs` endpoint string follows the `/v1/admin/install` precedent — only `mesh_doc_core.py` + its test file + `leader_proxy.py` (server handler) may reference the literal.

### CLI verbs

```bash
# Drop a doc to a named mesh node
maxim peer --node <name> docs put <namespace> <key> < body.md
maxim peer --node <name> docs put <namespace> <key> --file body.md
maxim peer --node <name> docs put <namespace> <key> --json '{"foo": "bar"}'

# Retrieve a doc from a named mesh node
maxim peer --node <name> docs get <namespace> <key>
maxim peer --node <name> docs get <namespace> <key> --output body.md

# List docs in a namespace on a named mesh node
maxim peer --node <name> docs ls <namespace>
maxim peer --node <name> docs ls <namespace> --json

# List all namespaces on a named mesh node
maxim peer --node <name> docs namespaces

# Delete a doc
maxim peer --node <name> docs rm <namespace> <key>
```

The `--node <name>` addressing matches the C1+C2+C3.3 pattern (resolve URL + cluster key from `mesh.yml::nodes` by name). A no-mesh.yml fallback via positional URL (`maxim peer docs put <namespace> <key> --url https://leader.example.com/v1`) is worth supporting for operators who haven't run `init-mesh`.

### Namespace conventions

Proposed (not enforced — convention only unless Q2 picks a different answer):

- `claude-sessions/<session-id>` — multi-Claude coordination inboxes
- `agents/<agent-id>` — Maxim agent-to-agent mailboxes
- `operator-broadcast` — operator messages visible to all agents (read-only for agents)
- `system/<component>` — system-level signals (future C7 key rotation, C4.5 auto-drain reasons, etc.)

### Agent integration (sketch, not part of v1)

The v1 ship is the transport + CLI. Agent integration is a **follow-up** that uses the transport primitive:

- Maxim agents on each node could poll their `agents/<own-id>` namespace every N seconds as part of the idle loop; received docs become SEM input or structured memory fragments
- Claude sessions could use a `/mesh-inbox` slash command (via hook or skill) that `GET`s the namespace and displays new docs; `/mesh-send <node> <body>` `PUT`s to the target's inbox

The transport plan does NOT prescribe how agents consume docs — that's a separate design step. v1 just makes the transport reliable.

---

## Open design questions (must be answered in the pre-design review round)

### Q1 — Secret-bearing docs: `atomic_write_secret` or `atomic_write_text`?

Docs might contain cluster keys (e.g., a C7 key-rotation announcement doc), API tokens, or PII. Options:

- **(a) Always `atomic_write_secret`** — every mesh doc gets 0o600 perms. Simplest rule, paranoid default. Cost: operators who `cat` a non-secret doc from the filesystem get a permission error if they're not the maxim user.
- **(b) Per-namespace policy** — namespaces declare `secret: true` at creation, then all writes in that namespace use `atomic_write_secret`. Requires namespace creation to be a first-class verb with metadata.
- **(c) Per-doc `X-Mesh-Doc-Secret: true` header** — sender decides. Simpler API, but trust boundary is on the sender.

**Recommendation to validate:** (b). Matches the C2 "function name IS the signal" pattern — a namespace marked secret writes through the secret helper at the call site, never via a flag that's easy to forget.

### Q2 — Role-scoping: one `mesh_docs/` dir or per-role `mesh_docs.{role}/`?

C2's drain state is per-role (`drained_nodes.leader.txt` vs `drained_nodes.peer.txt`) because a node's effective role controls its routing semantics. Do mesh docs need the same split?

- **(a) One shared `mesh_docs/` across roles** — simpler, docs survive role flips (leader → peer reconfig), matches the "mesh docs are addressed by node name not role" framing.
- **(b) Per-role `mesh_docs.{role}/`** — matches existing C2 pattern; prevents cross-role bleed during role flips; costs doc loss if roles flip.

**Recommendation to validate:** (a). Docs are addressed by `(namespace, key)`, not by role. A role flip shouldn't delete operator-committed state. The C2 per-role split is specific to drain semantics; generalizing it is wrong.

### Q3 — Delivery semantics: pure pull or optional push?

- **(a) Pure pull** — recipients poll `GET /v1/mesh/docs/<namespace>` to discover new docs. Simpler, matches the existing admin-endpoint shape, no new real-time infrastructure.
- **(b) Long-poll `GET /v1/mesh/docs/<namespace>?wait=30`** — recipient blocks up to N seconds waiting for a new doc. Adds real-time UX without full websocket complexity.
- **(c) Webhook push** — sender's `PUT` triggers a callback URL registered by the recipient. Real-time but introduces bidirectional trust + new auth surface.

**Recommendation to validate:** (a) for v1, (b) as a v2 add. (c) is C7+ territory because per-peer identity is needed before webhooks make sense.

### Q4 — Namespace creation + enforcement

- **(a) Namespaces are implicit** — first `PUT` to `<namespace>/<key>` creates the namespace dir. Free-form, no metadata.
- **(b) Namespaces must be created explicitly** — `maxim peer --node X docs ns create <namespace> [--secret] [--ttl 24h]` first, then `PUT` is allowed. Enables per-namespace policy (Q1b) and forces operator intent.
- **(c) Namespaces are declared in `mesh.yml`** — centralized schema. **Rejected in advance** because this violates the C2 "mesh.yml is declarative, `~/.maxim/util/` is mutable state" invariant — namespaces are runtime state.

**Recommendation to validate:** (b). Explicit namespace creation enables Q1b's secret-namespace policy and gives operators a place to set per-namespace TTL. The friction cost (one extra verb) is small.

### Q5 — Authorization model

v1 ships with the shared cluster key. Anyone with the cluster key can `PUT`/`GET`/`DELETE` any namespace on any node. This is:

- **Fine for the current deployment** (2-node home mesh, one operator)
- **Not fine for any multi-tenant or untrusted-peer scenario**

The v1 plan documents this explicitly as a known limitation. The fix is **C7 per-peer identity** — per-peer keypairs + per-namespace ACLs. This plan is NOT blocking on C7; it ships with the limitation and C7 layers identity over it later.

---

## v1 scope cut (explicit non-goals)

- **No per-peer identity.** v1 uses the cluster key. C7 adds keypairs later.
- **No ACLs on namespaces.** v1 is "cluster-key-authenticated = full access."
- **No webhooks / push.** v1 is pure pull.
- **No broadcast-to-all-nodes primitive.** v1 is addressable by node name only — operators who want broadcast can script it as a loop over `maxim peer list-nodes`.
- **No doc versioning.** v1 is last-write-wins. A `rm` + `put` is how you replace.
- **No binary blob support.** v1 is JSON + markdown only, 1 MB cap. Large binaries are a deferred follow-up — probably streaming-upload territory.
- **No search / query.** v1 list is flat by namespace. Full-text or metadata search is future.
- **No agent consumer integration.** The transport is shipped; how Maxim agents or Claude sessions consume docs is a separate design step.
- **No cross-node replication.** v1 docs live on the node they were `PUT` to. Replication to multiple nodes is a sender-side concern (loop over nodes).

---

## Proposed implementation sequence (when this plan activates)

1. **Pre-design review round** — answer Q1-Q5 above via a 3-lens review of this shell plan. Gate on cross-confirmed findings before any code.
2. **Commit 1:** `mesh_doc_core.py` module with `put_doc` / `get_doc` / `list_docs` / `delete_doc` client functions. Unit tests with mocked `_http.fetch_url`. No server yet — tests exercise the wire shape.
3. **Commit 2:** Server-side endpoint handlers in `leader_proxy.py`. Storage layer at `~/.maxim/util/mesh_docs/`. Size/TTL enforcement. Filelock RMW serialization. Integration tests.
4. **Commit 3:** CLI verbs `maxim peer --node <name> docs put|get|ls|rm` + `maxim peer --node <name> docs namespaces` + `maxim peer --node <name> docs ns create` (if Q4b is chosen). Composition layer tests.
5. **Commit 4:** CI grep allow-list for `/v1/mesh/docs` literal — only `mesh_doc_core.py` + test file + `leader_proxy.py` may reference it.
6. **Commit 5:** Docs — `cli-reference.md` verb rows, `mesh_debug.md` walkthrough, `CLAUDE.md` invariant for the doc-transport single-source-of-truth rule.
7. **3-lens pre-merge review round** — executor + architecture + blast radius lenses. Fold round before PR.
8. **PR + merge.**

Estimated total effort: **~3 sessions** (design + implementation + review fold + docs).

---

## Test plan skeleton (fill in during design round)

- [ ] Wire-level regression tests for every endpoint verb (mirror `test_peer_install.py` shape)
- [ ] Filelock RMW serialization under concurrent writers (mirror `test_drain_state.py::TestThreadingConcurrency`)
- [ ] Size cap enforcement (1 MB doc, 100 MB namespace) — reject with 413 + clear error
- [ ] TTL sweeper deletes expired docs + preserves non-expired
- [ ] Per-namespace secret policy (Q1b) — secret namespaces write via `atomic_write_secret`
- [ ] Namespace creation enforcement (Q4b) — `PUT` to non-existent namespace rejected
- [ ] Unknown node rejection via mesh.yml lookup (mirror C3.3 pattern)
- [ ] Self-drop: dropping a doc to yourself should just work (no self-guard needed — unlike install, self-drop has no "use pip directly" escape hatch)
- [ ] Positional URL fallback when no mesh.yml (mirror C3.3 pattern)
- [ ] CI grep enforcement: new `/v1/mesh/docs` reference outside allow-list fails CI
- [ ] Stacked-failure safety: interruption mid-write should not leave a corrupt doc (atomic_write_text covers this)

---

## Re-check triggers

Revisit this plan when:

- **Multi-agent coordination becomes a real user-visible blocker.** Today it's "Claude sessions on two nodes are annoying to coordinate" — tolerable. When Maxim agents on multiple nodes genuinely need structured exchange (e.g., a Mother Maxim precursor, or auto-drain notification publishing), the plan activates.
- **Any C3.x+ feature needs to send a structured message to peers.** C4.5 auto-drain announcements, C7 cluster-key rotation broadcasts, C6 request-trace aggregation — any of these would benefit from having the doc transport already in place. If two features in a row would want this primitive, ship it first and retrofit both.
- **The "crude" path (operator clipboard juggling) becomes painful.** Currently tolerable at 2 nodes / 2 Claudes. If you add a third node or a third parallel session, revisit.

---

## Related plans

- [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) §4 Stage C9 — this plan's home in the roadmap
- [llm_path_operator_visibility.md](llm_path_operator_visibility.md) — Plan 4 arc, C3.3 patterns this mirrors
- [node_security_simplification.md](node_security_simplification.md) — C7 per-peer identity, the fix for the v1 shared-cluster-key limitation
- [deferred/mother_npc_stimulus_plan.md](deferred/mother_npc_stimulus_plan.md) — Mother Maxim precursor that would consume this transport
