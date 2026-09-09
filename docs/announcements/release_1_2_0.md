# pymaxim 1.2.0 — "Oasis"

**Released 2026-09-09 (UTC — PyPI `upload_time`).** `pip install --upgrade pymaxim`

1.1.4 built the seam where Maxim's substrate meets a world it does not control. 1.2 is the
first release to make a **behavioral claim about that world** — two of them, in fact — and to
ship the plumbing that lets one agent's learning become another's. It is the substrate-sharing
line: an agent can be taught a want, and that want can travel to a genuinely independent agent,
over a real network, and change what the second agent does the first time it meets the
situation. The headline claim earns; the scaling claim earns its per-agent half and honestly
declares its cost.

## The two claims

- **Exp 56 — EARNED (the headline).** Agent A is taught, by a contingent teacher, that one
  specific action pays off at one specific world situation (the Exp 52 operant-credit mechanism,
  moved to the world channel). A's learned substrate is exported as a signed bundle and ingested
  into agent B — *independent by construction*: different `agent_id`, separately built
  `EntorhinalCortex` + `SensorEncoder`, disjoint cluster ids. At B's **first contact** with the
  situation, B chooses A's taught action, and three controls establish that the choice needed the
  taught *representation*, not merely the arrival of a bundle. Four-arm live campaign on a real
  Paper 1.16.5 Minecraft world (not a mock), n = 50 receivers/arm, substrate-primary (no LLM in
  the action path), all four gates PASS. This is the claim the whole "Oasis" idea rests on:
  learning is transferable between minds.
  Write-up: [56_four_arm_sharing.md](https://github.com/dennys246/Maxim/blob/main/docs/experiments/56_four_arm_sharing.md).

- **Exp 57 — PARTIAL (the scaling claim, run as its pre-registered may-fail second claim).**
  Does pooling N independent *partial* learners let each reach criterion in fewer of its own
  trials as N grows? **Yes — MONOTONICITY passes in its robust form** (Jonckheere–Terpstra
  permutation p ≈ 1e-4; survives dropping the fully-censored rungs; graded coverage widening
  0.25 → 0.50 → 0.75 → 0.75). Per-agent trials-to-criterion fall 21 → 21 → 15.5 → 10.5 across
  N = 1/2/4/8: a lone agent never reaches the 3-of-4 criterion in 20 trials, while an agent in a
  crèche of 8 gets there in ~10.5 of its own. **But NOT-JUST-MORE-DATA fails**: the pool spends
  more *total* experience than one agent given all of it (N × τ = 42/62/84 vs 41/46/43). Honest
  reading: collective learning is a **per-participant win, not a total-sample
  free lunch** — the convex-combination merge has a real cost, named as the 1.3 audit target
  (contributors ran sequentially; parallelism is a deployment property this sample-cost gate
  did not measure).
  Shipped as a named PARTIAL rather than an unqualified pass.
  Write-up: [57_dose_response_ladder.md](https://github.com/dennys246/Maxim/blob/main/docs/experiments/57_dose_response_ladder.md).

## The Hivemind — the P2P substrate exchange, end to end

Four slices, each reviewed and merged on its own:

- **Slice A — bundle signing.** `ed25519` release signing + verification for substrate bundles;
  a bundle carries who signed it and refuses to be trusted otherwise.
- **Slice B — the exchange endpoints.** Three authenticated routes on the existing leader-proxy
  server: list Queen-tier releases, download a signed bundle, contribute to the experimental tier.
- **Slice C — the operator CLIs.** `maxim oasis serve/publish/status` and `maxim hive
  add/remove/list/pull/contribute`, over a static registry at `~/.config/maxim/hive.json`
  (name → URL + Queen public keys + subscribed domains). `hive pull` **delegates to the existing
  `substrate ingest` verb** — the verification + V1–V10 gauntlet + journal are reused, not
  reimplemented.
- **Slice D — consumer trust.** Default trust is **Queen-only**: a pull refuses a release not
  signed by a registered Queen key, and refuses the decay-exempt inherent ("safety floor") bias
  class, unless the operator opts in per Oasis with `maxim hive trust`.

**Queen-tier promotion is deliberately not shipped** — its gauntlet battery cannot run yet
(Gauntlet #3 does not exist), so the blocker is recorded in `docs/plans/hivemind_p2p_scope.md`
rather than shipping a gate that cannot gate. `hive contribute` is write-only in 1.2.

## The nulls that bound the claims

Two pre-registered probes resolved as **offline structural nulls** — and they are load-bearing,
because they say precisely what the claims do *not* cover:

- **R1 — CACHE-CONFIRMED.** The shared want is an exact-key cache entry, not a concept: a world
  layout distinct enough to be a genuinely different situation necessarily misses the taught key.
  Cross-layout generalization is architecturally unreachable in the current readout — a 1.3
  design target, not a bug.
- **R2 — PREMISE-NULL.** World-owned drives do not, on their own, move behaviour: the survival
  loop has three specific breaks. Rather than back-fit a survival benchmark onto them, R3/R4
  defer to 1.3, where the loop is *designed* — the three breaks are the build list.

## Gates

Gates 1 (migrate half — `maxim substrate invalidate`), 3 (D8 separated from recall), 7 (typed
bundles have a real composer and a refusing receiver), and 8 (`hivemind/` mypy-clean; the D27/D28
fresh-agent and opt-in-evidence fixes) all closed before the claims ran.

## Scope, stated plainly

Both claims are substrate-primary, in Minecraft, one world layout, teacher-taught wants. Exp 56 is
one campaign at n = 50/arm; Exp 57 is 20 cohorts/rung. Still ahead as their own experiments: the
two-Reachy cross-unit hardware replication, aversion transfer (Exp 55, 1.3), cross-layout
generalization, and self-taught (rather than teacher-taught) world wants. This is a real
behavioral result with real edges — not a general-intelligence claim.

## Upgrading

`pip install --upgrade pymaxim`. Leaders: `maxim peer update && maxim peer restart`. Operators
running the live Minecraft bridge should redeploy it (it now emits `offset_x`/`offset_z`).

Full changelog: [CHANGELOG.md](https://github.com/dennys246/Maxim/blob/main/CHANGELOG.md#120---2026-09-09--oasis).
