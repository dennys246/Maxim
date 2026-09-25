# Security / trust lens — grounded_word_binding_demo.md (v2)

**Verdict: ADOPT WITH CHANGES.** Stages 0–3 add no new trust surface beyond S1 below. Stages 4, 6, 7 and
Experiment C do. Four findings are DO-NOT-BUILD as written. **D1 has to be settled before any Stage 4
code**, because it decides what Stage 4 builds. D2 and D3 are prerequisites for the Stage 6 wire format.
D4 is a prerequisite for the Exp C harness.

Reviewed against the plan, `sharing_threat_model.md` (FROZEN, §3–§5), `public_oasis.md`,
`maxim_hivemind.md` §Trust topology, and the code in `src/maxim/hivemind/` (`ingest.py`, `merge.py`,
`signing.py`, `bundle.py`, `hive_cli.py`, `store.py`, `oasis_cli.py`), plus
`runtime/leader_proxy.py`, `decisions/nac.py` and `scripts/minecraft_bridge/index.js`. Line anchors are
by symbol.

---

## DO-NOT-BUILD

### D1 — "Own experience wins" cannot be enforced if a consult is merged into NAc. The consult fires exactly where merge-in gives foreign data full weight.

The trigger fires when the agent is **unfamiliar or uncertain**, which means no fear or want history for
the situation. That is the regime where every existing bound is weakest.

- **Want has no foreign bound.** `merge.py::_merge_mean_clamped` uses a zero-prior rule: a key only one
  side holds keeps that side's value. So a foreign `cluster_reward_bias` lands **at full value** on any
  situation the receiver has no entry for. Unlike fear, there is no want counterpart to
  `FOREIGN_FEAR_DISCOUNT`. `tighten_negative_biases` skips `key not in merged_field` and `rv >= 0.0`
  (quoted in `public_oasis.md` too), so it protects only a want the receiver already held as negative.
  Where the receiver does hold a want, the fold is an unweighted 50/50 mean. One consult halves a +0.9
  learned want, and a −1.0 donor turns it into an aversion. Own evidence mass does not count, so own
  experience does not "win". It gets averaged.
- **Fear is bounded in magnitude but not in behaviour, and nothing can undo it.** Foreign fear is
  multiplied by 0.75. The constant's own comment says it was chosen so that a saturated donor clears the
  0.5 activation floor, so foreign fear is *meant* to act. Once it is merged, the receiver cannot lower
  it through its own experience. `NACConfig` says "counter-conditioning has no producer";
  `record_cluster_fear` only deepens; and fear has no per-tick decay (7-day wall decay only). The agent
  also avoids the situation, so it never collects disconfirming evidence. A poisoned or stale fear on
  `food while hungry`, or on the only exit, is a denial of behaviour lasting about a week, and the
  session-end save makes it persist. Exp C's prediction ("its own experience overrides") therefore
  **cannot come true for fear**. The claim sentence "that gate protects it from a wrong Oasis" is
  structurally false for fear today.
- **Revert and learned trust are blocked by the same thing.** NAc bias and fear dicts carry no
  provenance (threat model V5, and `cluster_reward_source` promotes to `"mixed"`). After a merge, nothing
  can say which entry came from which consult. That rules out a selective revert and rules out the
  plan's learned-trust (bee-rule) follow-up. A pre-merge snapshot reverts **everything** since the
  snapshot, including the agent's own learning. After the session-end save, `prune_nac_cluster_biases`
  cannot reach EC inserts, valences or counts (`public_oasis.md` §missing).

**Fix (choose before Stage 4):** keep consulted material in a **separate, attributed foreign layer**
that is composed at read time instead of merged into the agent's own dicts. The front-gate reason it
needs its own mechanism: the NAc dicts carry no provenance, so merge-in cannot support revert, learned
trust or own-wins. Minimum semantics:

1. Own write paths never touch the layer.
2. Read = own value when own evidence ≥ k, otherwise the discounted foreign value.
3. Foreign **want** is discounted exactly as fear is.
4. Foreign fear is **extinguishable**: own non-aversive visits decay the *foreign* entry. This gives
   "own wins" a producer without inventing counter-conditioning in the own store.
5. Revert = drop the entries tagged with one consult id.
6. The layer persists separately with an expiry. It is not folded into the own session file.

If the owner keeps merge-in, the plan must drop "own experience wins" and the protection clause from the
claim, state that live revert is all-or-nothing and in-session only, and add a foreign-want discount
before Stage 6.

### D2 — Automated consults would send the leader's inference key to every Oasis

`hive_cli.py::_run_pull` does `api_key = args.api_key or read_key()`. The default sends the **local
leader's bearer key**, the one `_check_auth` accepts for `/v1/chat/completions`, to whatever URL is
registered for the Oasis. Pulling by hand from a third-party Oasis already leaks it once. A consult path
that reuses this client leaks it **on every consult** to every Oasis operator, who can then run inference
on the receiver's GPU.

Separately, `public_oasis.md` states that the substrate GET routes are "routed before the proxy's bearer
check … a public read surface by design". The code does not do that: `_handle_substrate_get` calls
`_check_auth()` first. As shipped, every reader therefore needs the one key that also grants inference.
That is exactly the open question the plan lists, and the answer today is "no".

**Fix (prerequisite for Stage 6; recommended for Stage 5):**
- Store a per-Oasis read credential in `hive.json`.
- Never fall back to `read_key()` for a non-loopback Oasis.
- Signed release content needs no auth to read. Make the read tier anonymous and rate-limited, or scope
  it with a read-only token that is distinct from the inference key. This is `public_oasis.md` Phase 2
  condition 5, pulled forward for reads.

### D3 — A slice must stay a Queen-signed object. Server-signed slices break the trust model. `signer_identity` is not covered by the signature.

The signature covers the whole manifest plus the raw slice bytes (`signing.py::bundle_signing_payload`).
Any cut produces new bytes, so the Queen signature is lost. The three options move trust differently:

| Option | Trust moves to | Verdict |
|---|---|---|
| Server-signed slice | An **online** key on the internet-facing rig, which also chooses the content | **Reject.** Compromising the server now means authoring Queen-grade content. That defeats the offline-Queen model and the "channel is untrusted by construction" property `public_oasis.md` relies on. |
| Per-entry signatures | The Queen, per entry | Workable only if each signed entry binds `release_id` + a monotonic release sequence. Otherwise a server can mix entries across releases or serve a superseded (stale) entry. |
| **Situation-scoped release with a signed entry index** | The Queen, once per release | **Recommended.** The Queen signs one manifest that lists each entry's hash and its situation centroids. A slice is a subset of entries, each checked against the signed index. The server can only *omit*. The Queen key stays offline, and the client tracks the newest release sequence it has seen, which blocks rollback. |

The recommended option also enables client-side matching (S5): download the signed index, match
locally, then fetch entries by hash. The server then learns much less, and the rule becomes the
stigmergy read-locally rule the plan's own biology section describes.

**Pre-existing gap the slice design must close:** `_SIGNATURE_FIELDS` excludes `signer_identity` from the
signed payload, and `load_or_create_signer` notes that re-labelling reuses the same key. Anyone can
rewrite `signer_identity` on a signed bundle to any identity that maps to the same public key, and it
still verifies. Bind the identity: include it in the payload, or key `trusted_keys` by public-key
fingerprint. This is what turns D4 from a mistake into an exploit.

### D4 — As written, Exp C's corrupted Oasis would be a genuine Queen release

`substrate export --sign` always signs with **the one persisted key**
(`signing.py::load_or_create_signer`, `~/.config/maxim/hive_signing_key`). `--signer-id` is only a
label, and it sits outside the signed bytes (D3). A corrupted test bundle built on the Queen host is
therefore **cryptographically a real Queen release**. The attack chain:

1. Take the corrupted bundle, from a committed data file or a copied artifact.
2. Rewrite `signer_identity` to the Queen's label. It still verifies.
3. `_build_ingest_argv` trusts `[contributor]` when no allow-list is set, so every default-policy
   receiver admits it.

The registry (`~/.config/maxim/hive.json`) and the `~/.maxim/` sessions are shared machine-wide, so an
Exp C `hive trust … --allow-unsigned` or a registered test Oasis also leaks into real use.

**Fix (prerequisite for the Exp C harness; the harness asserts each item, provenance-assert style):**
- An ephemeral keypair generated in a tmpdir, never `signing_key_path()`.
- A test-only signer and contributor namespace (for example the additive manifest key
  `"purpose": "adversarial-test"` or an id prefix), which `hive pull` and `ingest_bundle` refuse unless a
  harness-only flag is passed.
- `OasisStore` root, `--registry`, journal and receiver session all under the sim's tmpdir sandbox.
- Loopback bind only.
- Corrupted bundles are never committed as `.zip`. Commit the *recipe* (seed + corruption spec) and the
  results, not a signed artifact.
- Never simulate corruption through `--allow-unsigned`. The corrupted arm must pass the same signature
  path as the clean arm, or Exp C measures the signature check instead of the gate.

---

## SHOULD-FIX

**S1 — Any player on the server can trigger the consult.** The bridge forwards `"<username> says:
<message>"` for **every** player (`bot.on("chat")`). With `--system_messages` it also forwards death
messages, which can carry attacker-chosen text through name-tagged mobs and items. Stage 1 puts that text
into the situation key, and Stage 6 reads novelty from it. Another player can therefore:

- make the agent "unfamiliar" at will, by saying new words;
- pick which Oasis entries match, via the text key;
- raise stakes cheaply, by hurting the agent;
- get past the per-situation refractory, by changing the words every time (the situation key changes);
- if the waiting behaviour is a "cautious hold", freeze the agent (denial of service).

Binding has the same exposure: saying "food" while the agent stands in lava binds `food` to that
situation, and Stage 4 then exports it.

Fixes:
- Tag each text percept with its source (system vs chat, plus the speaker).
- Only allowlisted speakers enter the situation key and the binding in experiments. The teacher is an
  allowlisted username.
- Novelty for the trigger comes from world, interoception and audio only.
- Add a **global** consult budget (a token bucket per session) on top of the per-situation refractory.
- Cap the hold, or keep acting while waiting (open question 1, answered from this lens).
- Discard a consult answer whose situation is no longer active when it arrives.

This is the same class as #828: heard input reaching a consequential action.

**S2 — Replay dedup does not survive slicing.** V8 deduplicates on the digest of the *whole bundle*.
Every slice is new bytes, so the same entry is re-admitted on every consult. The 50/50 want fold then
walks toward the donor's value geometrically (threat model row J), and counts add up. Fix: deduplicate
per entry hash across consults, and make applying the same entry twice a no-op.

**S3 — The live merge has a TOCTOU window and an unlocked load.** `NAc.load_state` "does NOT acquire the
NAc mutex because callers expect load-time quiescence", but pain-bus subscribers (`record_cluster_fear`)
write concurrently. The sequence dump → network wait → merge → `load_state` also overwrites any own
learning that happened during the wait. Fix: compute the merge on the loop thread at the boundary, from
state read *at* the boundary, with writers held. D1's overlay avoids this entirely.

**S4 — The ATL payload is a new slice and needs threat-model coverage before Stage 4 ships.**

- Per threat model §2, a new slice needs a `schema_version` bump plus a migration, and changing a §5 duty
  needs a dated amendment. The plan names `_format_version` and CC3 but neither of these.
- New rows the amendment needs:
  - V2 bounds on `NAMES` `weight` and `confidence`. A `confidence` that merges by max is row M: assert
    1.0 once and it is permanent.
  - A V4 identity scrub of text concepts. Heard chat carries usernames ("X says:").
  - A V9 charset check on concept ids, plus V6 caps on relation count.
  - Receiver-stamped provenance (V1) and a foreign discount on `NAMES` weight.
- Never ship raw heard text. Row L covers prompt injection, and #823 is the same untrusted-text class.
  Ship ids and embeddings, or a closed vocabulary.
- **Amplification path:** after re-keying through the aligned-EC id map, a foreign `NAMES` relation can
  point a common word at the **receiver's own deepest-fear cluster**. Hearing the word then fires the
  receiver's full, *undiscounted* own fear, with no foreign fear magnitude needed. Bound it: a foreign
  relation may target only clusters the same slice brought, or it carries the foreign discount into the
  reactivated situation's valence.

**S5 — What the server learns from a consult.** The key is a `{modality: embedding}` map:

- Text embeddings are invertible to a useful degree, so heard chat, including other players' words and
  names, can be recovered.
- Interoception shows distress: low health, drowning, hunger.
- Consult timing shows *when* the agent is in trouble.
- Stage 7 sends the user's typed text.

Preferred: client-side matching over the signed index (D3). The server then sees only which entry
hashes are fetched, or nothing if the whole scoped release is fetched.
Fallback:
- No interoception on the wire, no raw text, no agent, session or contributor id in the query.
- No server-side query logging by default.
- Disclose in the Search UI that the query leaves the machine.

**S6 — Stage 7 input handling.** Typed text is a **query only**. It must never enter the percept stream
or the binding look-back; otherwise a user can bind any word to the agent's current situation. It must
not be parsed for commands (#828), and it gets a length and charset cap. Search also skips the
uncertainty gate, so each search is an always-consult. Show a dry-run preview (the `ingest` report,
before anything is applied), apply only to the D1 layer, and make it one-click revertible.

**S7 — Pin the live path's policy.** The consult path forces `require_signed=True` (it cannot inherit a
per-Oasis `allow_unsigned`) and `inherent_trusted_sources=∅`, so a consult can never install a
decay-exempt safety-floor key. `trusted_sources` comes from the operator allow-list, never from the
`[contributor]` fallback.

**S8 — Server-side limits for search.** Search is compute. Serving blobs is not. `PeerRateLimiter` is
keyed by IP and unlimited by default (`MAXIM_PROXY_RATE_LIMIT_RPM=0`). Search needs a rate limit that is
on by default, a cap on query payload size, and a cap on slice size.

**S9 — Write down the revert semantics.** What the snapshot covers (NAc + EC + ATL + id map). That
reverting after further own learning discards that learning. That revert is impossible after the
session-end save unless D1's layer exists. Revert must mark the journal entry `reverted`.

---

## NIT

- **N1.** `public_oasis.md` §Front-gate says the substrate GETs sit before the bearer check. The code
  authenticates them (`_handle_substrate_get`). Correct the doc in its own PR.
- **N2.** The Stage 5 "assert the signature was verified" should read a structured field. The journal
  entry records it only as a free-text note (`"signature verified (…)"`). Add
  `signature_verified: <signer>` to `journal_entry`.
- **N3.** #824 (DNS rebinding) is open. The consult's outbound fetch goes through `utils/http`. The Oasis
  URL is set by the operator, so the risk is low, but name it.
- **N4.** The claim sentence should carry the fear exception until D1 lands.

---

## Verified fine

- V1–V10 adapter reuse: Stage 4's "the same `ingest_bundle` checks" keeps the front door (a V1
  allow-list refusal), V6 bounded streaming reads, V2 NaN and count caps, V9 charset and reserved
  domains, and V4 receiver-side re-scrub.
- Fear enters foreign-discounted (×0.75), its failure mode is checked against the Wire-4 allowlist, and
  it folds by MIN, so shipping enthusiasm cannot erase a held fear (`tighten_negative_biases` for
  negative wants).
- `OasisStore.publish_release` refuses unsigned bundles. `hive pull` defaults to Queen-only and validates
  release ids before building a path.
- The plan already fetches off the loop thread, excludes any public contribution path (promotion stays
  WRITE-ONLY), keeps web text test-only (Stage 8), runs Exp A raw, and asks for a signature assertion in
  Stage 5 (fixing Exp 61's gap).
- Exp C as a may-fail test of the gate against a corrupted Oasis is the right security experiment, once
  D1 makes a pass possible and D4 isolates it.
