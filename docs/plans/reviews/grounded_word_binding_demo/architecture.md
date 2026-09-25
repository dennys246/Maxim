# Architecture lens — grounded_word_binding_demo.md (v2)

**Verdict: ADOPT WITH CHANGES.** The front-gate is honest: all three additions are real, and most
of the line rides on shipped pieces. But two of the three new mechanisms are specified in a way
that breaks a shipped contract. Front-gate 1 (live ingest) leaves its critical section undefined
against `NAc.load_state`, which takes no lock and replaces state wholesale. Front-gate 3 (server
search that returns signed slices) breaks both the trust model DECISIONS.md adopted 2026-09-19 and
V8's exactly-once idempotence. It can ride on existing infrastructure far more cheaply: pick the
slice locally, from a cached release that has already been verified. Front-gate 2 (the ATL payload)
as written ships raw heard text. Fix those three in the plan and the rest is SHOULD-FIX.

Reviewed against: the plan; CLAUDE.md; ARCHITECTURE.md; DECISIONS.md (2026-09-19 public Oasis);
`maxim_hivemind.md`; `public_oasis.md`; `archive/oasis_ingestion_contract.md`;
`docs/agents/persistence-config.md`, and the source files cited below.

---

## DO-NOT-BUILD

### D1. The live-ingest critical section is not defined, and as written it loses updates (Front-gate 1 / Stage 4)

The plan describes the sequence "dump live state → `ingest_bundle` checks → `substrate_merge` →
load into the live NAc/EC/ATL at a tick boundary". It also has the fetch run off-thread and the
merge "applied at the next boundary". It never says whether the **dump** happens at the same
boundary as the **load**. The code makes that question load-bearing:

- `decisions/nac.py::NAc.load_state` replaces `_links`, `_reward_bias` and the rest **wholesale**.
  Its docstring says it *"does NOT acquire the NAc mutex because callers expect load-time
  quiescence"*. So a dump taken at tick T, then a merge, then a `load_state` at tick T+k silently
  erases every outcome, fear and Welford update learned in between. That is the D43 shape again:
  a merge that reports success and deletes state.
- `memory/atl.py::ATL.load_state` is also a wholesale `clear()`-then-restore.
- `similarity/ec.py::EC.ingest_substrate_nodes` is additive, but it **overwrites** folded node ids
  with the snapshot-time merge result. A text centroid is not frozen
  (`ECConfig.frozen_centroid_modalities` excludes `text`), so it can drift between the snapshot and
  the apply and then get clobbered.
- Cost: `hivemind/merge.py::ec_merge_aligned` is a pure-Python O(N_donor × N_receiver × d) scan.
  It calls `_cosine` through `sum(zip)` and deep-copies every receiver node. With a whole release
  as the donor this cannot fit the loop's ≈1 s tick. Even a small slice against a receiver of about
  1k nodes at 384–768 dimensions needs measuring before anyone assumes it fits.

**Fix (the plan must pick one and state it):** (a) **one critical section.** At the boundary, hold
`nac._lock` (an RLock, so it is re-entrant for the applier), then dump, merge and load with no other
writer running. The merge runs only over a pre-selected **small** donor slice (see D2), and its wall
time is measured at the demo's receiver size. The declared "cautious hold" (open question 1) covers
the pause. Or (b) **delta apply.** Merge off-thread against a snapshot, then apply only the delta
additively at the boundary. The delta carries a version check: if a node or key it touches changed
since the snapshot, re-run. (a) is smaller and honest. (b) needs new additive NAc and ATL apply
methods. Either way, name every thread that writes NAc during a substrate tick and assert there is
no writer off the loop thread, or take the lock.

### D2. Server-side "search → signed situation-scoped slice" breaks the trust model and V8 (Front-gate 3 / Stages 6–7)

Two shipped contracts break:

1. **Trust.** DECISIONS.md 2026-09-19 adopted the public Oasis on one property: *"Consumers verify
   the Queen signature, never the host that served the bytes, so a mirror is untrusted by
   construction."* `hivemind/signing.py::bundle_signing_payload` signs the manifest plus **every
   slice's raw bytes**, so a slice cut on the server cannot carry the contributor's signature. The
   two remaining options in the plan's "Open" list both move trust to the host. A server-signed
   slice makes the host a signer. Per-entry signatures are a new wire format. The first option
   also contradicts `maxim_hivemind.md`'s own principle (*"Oases are full agents, not databases …
   not searching a record store"*).
2. **Idempotence (V8).** `hivemind/ingest.py::ingest_bundle` is exactly-once **per bundle digest**,
   contract §4: *"re-ingestion sums counts and re-walks the mean fold (row J)"*. Slices from the
   server are fresh bundles with fresh digests. Two consults whose answers overlap would fold the
   same foreign entries twice, and V8 cannot see it. Consulting one cached release twice is refused
   outright on the second consult ("already ingested").

**Fix, which is also the front-gate answer (no server mechanism at all):** a consult is a **local
selective ingest from a release that has already been pulled and verified**.

- `hive pull` runs off the loop thread (at session start, or on a timer) and verifies the
  **whole** signed release once, with `require_signed` and `trusted_keys`, the way it does today.
- At consult time, a **pure** `hivemind` function selects the donor sub-state for the current
  situation. It uses the same per-modality matching `ec_merge_aligned` already does
  (`SENSOR_MODALITY_THRESHOLDS`, geometry gate, `strict_geometry`) and keeps only the matched donor
  EC nodes, the NAc keys naming them, and the `NAMES` edges whose endpoints both survive. That
  sub-state is merged through the same `substrate_merge`.
- V8 is amended to **entry-level** idempotence. The journal records `(release digest, donor entry
  id)` pairs as they are admitted, and the selector excludes any pair already admitted. This is a
  dated amendment to the contract, not a bypass.

This removes the plan's whole "Open for the Oasis side" list. There is no slice verifiability
problem, because the whole release is verified. There is no single-bearer-key question, because
`GET /v1/substrate/releases|bundle` are already routed before the bearer check in
`runtime/leader_proxy.py::LeaderProxyHandler.do_GET`. There are no server rate limits and no
server compute DoS from arbitrary 768-d query vectors, and the server learns nothing about the
asker's situation. It is also the more faithful biology, since the plan's own stigmergy
paragraph says the store is *"read **locally**"*. The cost is freshness (a new release reaches the
agent at the next pull) and pulling whole releases. Server search stays deferred, with a named
trigger: release size makes whole-pull impractical.

### D3. "The concepts … at both ends" ships raw heard text (Front-gate 2 / Stage 4)

`memory/atl.py::ATL.activate_substrate_node` creates a `Concept` with `name=text[:80]` and
`definition=text`. For a text node, that text is the heard message verbatim, and game system
messages carry player names (for example "<player> drowned"). So shipping the endpoint *concepts*
ships raw utterances. That is episodic content, and it escapes the load-bearing
"hippocampus-episodes-stay-local" privacy invariant by a side door. `hivemind/identity.py` screens
only NAc event signatures and EC `domain`. Nothing screens ATL labels. The text EC centroid itself
is close to one sentence embedding when a node has a single member, so it can be inverted.

**Fix:** the ATL slice carries **relations only**: `(source_id, target_id, "NAMES", weight,
confidence, count, provenance)`. Endpoint *EC nodes* ride in `ec.json` as they do today. Endpoint
concepts are **re-derived on the receiver**, created on demand by `activate_substrate_node` the
first time the receiver hears the word. No `name`/`definition` field crosses the wire. Text EC
nodes pass through `is_identity_bearing` on whatever label is used for screening at compose, with
the threshold stated, and public_oasis.md Phase 0 item 3 (the privacy read) is re-run for any
release that carries text nodes. Name the inversion residual in the plan.

---

## SHOULD-FIX

### S1. `ingest_bundle` can be reused without a fork, with one small change

`hivemind/ingest.py::ingest_bundle` already takes the **receiver** as dicts and is documented
*"pure with respect to the receiver … nothing is written to disk"*. So dumping live state into it
is sanctioned as it stands. The **donor** is a `bundle_path` (`read_bytes()`, then
`zipfile.ZipFile(path)`). Split it into `_ingest_bundle_bytes(raw, …)` and keep the path function
as a thin wrapper. Keep one pipeline and do not add a "live" twin. Under D2 the donor is a cached
file that has already been verified, so the change is optional for the demo. It becomes required
if a donor is ever held only in memory.

### S2. Layer ownership: pure pieces in `hivemind/`, the live apply in `runtime/`, and the audit should enforce it

ARCHITECTURE.md lists `hivemind/` as "substrate-sharing layer". `utils/audit.py::LAYER_RULES` has
**no `hivemind` row**, so nothing stops `hivemind` from importing `runtime`
(`hivemind/oasis_cli.py` already imports `runtime.leader_proxy` for `serve`, a CLI-level
exception). Recommendation:

- In `hivemind/`, under the mypy gate since the bundle format is a wire boundary: ATL-slice
  validation in the ingest admission pass, the ATL leg of `substrate_merge` (S3), the local slice
  selector (D2), and the entry-level journal.
- In `runtime/`: one thin module (for example `runtime/substrate_consult.py`) that owns the
  off-thread pull, the boundary apply (D1), the snapshot and revert (S7), and the trigger call site.
  `agent_loop.py` gets a single call, not an inlined block. It is already the god-function hotspot.
- Add `"hivemind": {"must_not_import": ["runtime", "agents", "planning", "tools"]}` to
  `LAYER_RULES`, and put the existing `oasis_cli` → `runtime` import in the baseline with a
  rationale, so the audit enforces the split.

### S3. The ATL re-key must live inside `substrate_merge`, not beside it (the D43 seam rule)

`hivemind/merge.py::substrate_merge` exists because D43 left a composition to its callers. An
`atl_merge` that a caller runs after `substrate_merge`, with the returned `id_map`, rebuilds that
exact defect. Add `receiver_atl_relations=` and `donor_atl_relations=` to `substrate_merge`. Re-key
the donor's `NAMES` endpoints through `aligned.id_map`, drop a relation missing an endpoint with a
count (`names_dropped`, the dangling rule), fold duplicate `(src, tgt)` pairs evidence-weighted,
and apply the admission caps (`CAP_FOREIGN_CONFIDENCE`, `MAX_FOREIGN_COUNT` analogues). Add the
counts to `SubstrateMergeResult` (it is frozen, but it is not persisted and not on the wire, so CC3
does not apply).

### S4. The `NAMES` shape: where it lives and two silent-no-op traps

- **Define it once** as a builtin in
  `memory/semantic_types.py::RelationshipRegistry._seed_base_types` (non-symmetric).
  `RelationshipRegistry.from_dict` re-seeds builtins via `cls()`, so every older persisted ATL gains
  `NAMES` on load. It cannot be a runtime-`register()`ed type, because a receiver that never
  proposed it would lack it.
- Bundle code refers to it by the **string** `"NAMES"`. It must not import the memory layer. That
  follows the existing hivemind convention, such as the duplicated `_VALID_VALENCES` and `_cosine`
  (*"keep the hivemind layer free of internal-module imports"*), plus a unit test that pins the
  string equal to the registry builtin. That is not a layering violation.
- **Trap 1:** `memory/semantics.py::Semantics.define` **returns `False` silently** for an
  unregistered type. Every binding write and every ATL apply must check the return value and fail
  loudly. By the three-miss rule this is a candidate to push into the type.
- **Trap 2:** `define` **appends a duplicate edge** on every call (`agents/bus.py::DependencyGraph.add_edge`
  has no dedup), and `DependencyGraph.update_edge` matches the first `(source, target, edge_type)`
  while ignoring `relationship_type` metadata. `NAMES` and `RELATED_TO` both map to
  `EdgeType.ASSOCIATES`, so they collide. "Strengthens a `NAMES` relation" (Stage 2) needs a real
  upsert keyed on `(src, tgt, rel_type)`. Build it in the ATL layer at Stage 2, because the wire
  shape is fixed there, not at Stage 4 as the contract table says.
- No new frozen dataclass is needed, since relations travel as dicts. If one is added, it takes
  CC3 path (a).

### S5. Bump `BUNDLE_SCHEMA_VERSION`, and have ingest hash every declared slice

Today an `atl` entry in `manifest.contents` passes V7 and is then **never read**. The pipeline reads
only `"nac"`/`"ec"`, and the skip adds no note. As a result:

- An **unsigned** ATL bundle into a 1.3.0 receiver drops the bindings silently while admitting their
  text EC nodes.
- A **signed** ATL bundle into a 1.3.0 receiver **fails the signature** with the message
  "tampered or wrong key", because `raw_slices` holds only nac and ec while the signer covered
  every slice.

Fix: have `ingest_bundle` put **every declared slice** into `raw_slices`, the way the standalone
`verify_bundle_signature` already does. Bump `BUNDLE_SCHEMA_VERSION` to 3 and register a v2→v3
migration (trivial, since v2 has no ATL), so an older build refuses loudly at
`_manifest_from_zip` ("schema_version unsupported"). Sequence this against public_oasis.md Phase 0
item 2 (the public format-freeze pass). Either fold the ATL slice shape into the freeze, or record
that the freeze already carries one planned bump. DECISIONS 2026-09-19 notes that after
publication *"format changes now break strangers"*.

### S6. The session-end persist contradicts the contract's write order, and consolidation can erase a pull

- `integration/memory_hub.py::MemoryHub.on_session_end` and the lightweight end path save **NAc
  before EC**, with ATL after. The contract's §3.12 ordering is **EC before NAc**: NAc-first
  "leaves merged biases naming donor clusters the EC file does not yet hold (dangling)". The
  plan's "the session-end save then persists the merged state" inherits the wrong order. The
  amendment must reorder to EC → ATL → NAc, or state that it relies on the D2/D17 load-time
  dangling detection.
- The same session end calls `ATL.consolidate()`, which removes low-retention concepts and their
  edges, and `nac.decay_all(0.95)`. A binding pulled mid-session with no local reinforcement can be
  pruned before it is ever saved. State the retention policy for `NAMES` endpoints (for example,
  a `NAMES` edge counts toward degree, which it already does via `_outgoing`, and whether that is
  enough), and add a guard test that a pulled binding survives one session end.
- The bio-memory brief notes a residual gap: mutations made after the loop's
  `end_bio_session` and before `shutdown()` are not persisted. A pull applied during that window
  is lost. The runtime module must refuse to apply outside an active session.

### S7. Revert snapshot: build on `SessionSnapshot`, and be honest about what revert means

`memory/snapshot.py::SessionSnapshot.restore_into` already does a transactional multi-system apply
with a pre-mutation `dump()` and reverse-order rollback. That is exactly the snapshot and torn-state
rollback the plan wants. **EC is not a snapshot kind** (`SNAPSHOT_KINDS` has atl, hippocampus, nac,
scn, ptb and cross_layer, but no EC), and EC has no in-memory wholesale substrate replace, only
`load(path)` and the additive `ingest_substrate_nodes`. Add EC `dump`/`load_state` for the
substrate block and join it to the protocol. Two more points:

1. Merges are not invertible (counts sum and means fold), so "revert the pull" means **roll back to
   before the pull and lose all learning since**. Say so, or cap the revert window to the consult's
   own hold.
2. The journal must record the revert. Otherwise V8 (or the entry-level ledger from D2) refuses to
   re-admit the reverted entries forever.

### S8. Where the contract §1 amendment goes

`oasis_ingestion_contract.md` is **ARCHIVED** ("not a live plan"). Amending an archived record in
place hides the change. Write a dated **DECISIONS.md** entry, "Mid-session ingest legal only via
the runtime boundary path", and add a pointer line in the archived §1 table's Mid-session row. The
entry keeps file-level `substrate ingest` refused against a live receiver. The new row is legal
only through the runtime path, with D1's critical section, S6's write order and S7's revert. The
threat model's §5 change rule applies if any V-duty semantics change (they do: V8 becomes
entry-level under D2).

### S9. The text merge threshold must equal the text formation threshold

The D43 rule, in `merge.py`'s own comment: *"A sensor modality must align at the threshold its
clusters were FORMED at."* Text clusters form at `ECConfig.pattern_complete_threshold = 0.44`, the
same value the merge falls back to. Calibrating a separate merge threshold while formation stays at
0.44 re-creates the D43 mismatch. Calibrate the **formation** threshold for heard words at Stage 1,
and have the merge read the same constant from one table entry. Use a distinct modality tag for
heard words (for example `heard`) rather than `text`. `text` is also written by the
`MAXIM_SUBSTRATE_PATH` MemoryHub percept-text path, whose clusters mean something different, and a
distinct tag gets its own threshold row and its own frozen-centroid decision. mpnet single-word
cosines sit near 0.44, so `water` and `fire` may collapse together, which is itself a Stage 1
measurement.

### S10. Stage 1's text channel: an abstraction mismatch and a hidden dependency (layering, not invariant, but real)

The text channel has no layering violation: runtime → similarity/memory is allowed. Three things
need stating:

- `runtime/agent_loop.py::_SUBSTRATE_CHANNELS` entries are `ModalityChannel(tag, read_values,
  read_ranges)`, which read **executor sensor state** and encode through `SensorEncoder` at 384-d.
  Heard text is an **event** from the percept stream, encoded by `LinguisticEncoder` at 768-d. It
  is not "one tuple entry". It needs a second channel kind, event-driven with decay, and the
  comment block's "Adding a future modality … is one tuple entry here" does not hold for it.
- "A heard word stays in the situation for a window, then decays" is a decaying activation buffer,
  which is `memory/percept_trace_buffer.py::PerceptTraceBuffer`. So **Stage 1 also depends on R4's
  look-back design**, not only Stage 2. Building a private window here would be the second buffer
  the plan forbids for Stage 2.
- The same comment warns that the summed cluster term in `recommend_action` scales with the number
  of active channels ("a selection-dynamics change; re-check gate calibration"). The arm must
  re-run `scripts/selection_dynamics_rebaseline.py`, and the plan should name it next to the
  Re-run-on triggers. Also, the situation key is `{modality: one cluster_id}`
  (`nac.note_active_clusters`), so two words heard inside one window collide on the single slot.
  Decide between latest-wins and a multi-valued key before the key shape is fixed on the wire (it is
  in the "shared, owned by neither" list). Existing survival cluster ids do **not** change. The
  change is additive: the pain→fear subscriber will also key fear onto the text cluster, which is
  route (a) in Exp A and must be pinned as present.

### S11. Learned trust must not write `hive.json`

`docs/agents/persistence-config.md` exempts `~/.config/maxim/hive.json` from the grep precisely
because it has **no runtime writer**. The Stage 6 "learned trust" follow-up updates trust
mid-session. Route it to the agent's own persisted state (or `~/.maxim/util/`), never
`HiveRegistry`. The operator allowlist (`trusted_sources`) stays the boundary, and learned trust
only discounts **inside** that boundary.

### S12. Stage ordering

- Split Stage 4. **4a:** the ATL payload + file (pre-boot) ingest, with the receiver resolver
  (`cli.py::_resolve_receiver_pair`) extended to include `atl.json` and the write order EC → ATL →
  NAc. **4b:** live ingest (D1, S6–S8). Stage 5's pre-boot arm needs only 4a. The live arm needs 4b.
- The **consult line (4b → 6, Exp C) does not depend on the language line (1 → 3).** Exp C's gate
  can run on today's NAc+EC fear and want bundles (the Exp 61 material), with word bindings as an
  optional passenger. Running it in parallel de-risks the demo's second claim and gives it its own
  may-fail.
- The `NAMES` shape (S4) is decided at Stage 2, where it is first written. Move it out of the
  Stage 4 row of the contract table.

### S13. Scope realism for "the lead-up demo needs Stages 0–7"

The demo scope includes three new experiments (A, B, C), each needing a four-lens design review,
rig time and a prereg. It also includes two engineering subsystems (4a/4b and the consult). It
has an external dependency on R4's `PerceptTraceBuffer` design, which the memory file calls *"the
real gap"*, and a UI surface. That is a multi-release line, not a lead-up demo. Suggested demo cut:
Stages 0–3 + 4a + 5 (pre-boot arm) + Stage 7 as an **explicit, user-initiated** local consult (D2).
The automatic gated trigger and Exp C would follow as the second claim. The claim sentence already
ends at "when and only when the receiver is uncertain", which the cut would not earn, so the
public sentence must drop that clause until Exp C is EARNED.

---

## NIT

- **N1.** The "0.44 generic" phrasing in Stage 4 reads as if 0.44 is arbitrary for text. It is the
  text formation threshold (S9). Reword.
- **N2.** "Trusted signers only" (Stage 7) should name the mechanism: `trusted_keys` +
  `require_signed=True` on the cached-release verify. The harness assertion from Stage 5 (Exp 61
  passed no `--sign`) belongs on the consult path too.
- **N3.** `hivemind/ingest.py::IngestReport` is a mutable `@dataclass`. That is fine because it is
  not persisted, but the new entry-level journal records **are** persisted JSON. They need
  `_format_version` via `with_format_version` and `atomic_write_json`, as `IngestionJournal.save`
  already does.
- **N4.** The mypy gate covers `src/maxim/hivemind/` only. Code placed in `runtime/` escapes it,
  which is one more reason to keep every wire-touching function in `hivemind/` (S2).
- **N5.** `MAX_BUNDLE_ENTRIES = 16` has headroom for an `atl.json` member. Nothing to do.
- **N6.** Stage 7 lets the user bypass the gate. State in Exp C's prereg that Stage 7 consults are
  excluded from the gated arm's counts.

---

## Verified fine

- `ingest_bundle` takes the receiver as dicts and is pure (no disk writes, no mutation), so the
  plan's "same `ingest_bundle` checks" on a live dump is sanctioned by its docstring (S1 covers
  only the donor side).
- `substrate_merge` re-keys the **donor only**. Receiver ids survive, as the plan states.
- `EC.ingest_substrate_nodes` exists as the documented live counterpart that preserves member
  counts, and the contract §1 text already equates a splice with "a live load→ingest→save
  round-trip".
- Text nodes are geometry-stamped at encode (`LinguisticEncoder.geometry_for` →
  `register_substrate_node(..., geometry=)`), and `_migrate_legacy_geometries` stamps legacy text
  nodes exactly. `strict_geometry=True` at ingest therefore works for 768-d text without new code,
  and a donor on the hash fallback will not fold into an mpnet receiver.
- `observed_embedding_dims` is measured per modality from the arrays at compose and re-measured at
  admission (V3), so a `text: [768]` entry needs no manifest change. `_validate_ec_payload` has no
  modality allowlist to extend.
- The `_format_version` + CC3 obligations apply only to the manifest (already stamped) and any new
  persisted journal (N3). No new frozen wire dataclass is needed if relations travel as dicts (S4).
- `ec_merge_aligned`'s `_cosine` returns 0.0 on a dimension mismatch, and alignment is
  per-modality, so the plan's "no cross-modal comparison is needed" is structurally true.
- The binding as an ATL relation joining two ids needs no 384↔768 projection. Correct.
- "Cluster ids ARE ATL concept ids": `SensorEncoder` and `LinguisticEncoder` call
  `ATL.activate_substrate_node(node_id, …)`, which returns the EC node id as the concept id.
- The fear merge is already tighten-only plus foreign-discounted (`FOREIGN_FEAR_DISCOUNT = 0.75`,
  `tighten_negative_biases` inside `substrate_merge`), so "own experience winning" for fear needs
  no new mechanism.
- Public release routes sit before the bearer check. Under D2 the consult needs no authenticated
  route, so public_oasis.md's single-bearer-key gap does not touch this line.
