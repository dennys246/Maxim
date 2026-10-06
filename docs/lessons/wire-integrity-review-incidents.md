# Wire-integrity evidence: the two incident tables (2026-10-05)

The source data behind [wire-integrity-review.md](wire-integrity-review.md) and [../CODE_REVIEW.md](../CODE_REVIEW.md),
kept so the counts can be checked. Two passes ran independently on 2026-10-05, each by a separate reader:

- **Pass A (history):** issues, PRs and `git log`, searched for de-wiring terms. Each origin was dated with the
  pickaxe (`git log -S/-G`) where possible: **V** = verified (a pickaxe hit, or an explicit statement in the issue, PR
  or commit), **I** = inferred.
- **Pass B (docs):** `docs/bugs/README.md` (D1–D88), the lessons, `docs/wiring/`, `docs/plans/outstanding.md`,
  CHANGELOG. **doc** = stated in a document, **inf** = inferred by the reader.

The passes overlap: 30 + 39 table rows. About a dozen incidents appear in both (D58/d916aea9, #840, #841, #908,
#909/1.2.1, D41, D42, D77/D79, #972, #870, #873, #1085), and several rows group related incidents, so no exact unique
count is claimed. "Review on the breaking change" was looked up only where the breaking commit or PR could be
identified: origins before the review discipline (2026-04-12) read "predates", and post-discipline origins with no
round on record read "none recorded" or "—". **Eight** breaking changes had a recorded two-lens round (#411,
5a1dd499, 8a09ae85, 4619e941, 60702417, #1029, the #985 fixes #973/#983/#986, the 1.2.1 pairing PRs), and every one
missed the de-wiring in its scope (8a09ae85's round covered its results, not its code; #1029 fixed one site of N;
60702417's round is verified but its role as the origin is inferred).
This is a sample of incidents, not a catch rate.

## Pass A: history (30 rows)

| Id | What de-wired | Origin (break) | Fix | Shape | Found by, latency | Review on the breaking change |
|---|---|---|---|---|---|---|
| [#840](https://github.com/dennys246/Maxim/issues/840) | FearCircuitBridge → `NAc.record_event(metadata=)`, TypeError swallowed | 9c890b0a 2026-02-06 **V** | #1080 (Dormant) | signature mismatch + swallow | while fixing #839, ~7.5 mo | predates the discipline |
| [#841](https://github.com/dennys246/Maxim/issues/841) | plan_manager passes a str to `recall_similar` | 1cdcf989 2026-02-16 **V** | #1080 | argument type + swallow | #839 audit, ~7 mo | predates |
| d916aea9 (D58) | `observe_from_action(entity_path=, actual_sensors=)` vs `entity=/actual=`; forward model never trained | c6277e45 2026-04-07 **V** | d916aea9 | kwarg mismatch + swallow; a test-only caller called it correctly | four-lens design dive, ~5 mo | predates |
| [#908](https://github.com/dennys246/Maxim/issues/908) | Cerebellum loaded from a path, saved to `None`; 3 no-op save sites | 4619e941 2026-04-17 **V** | #1106 | load/save path split | engram audit #912, ~5 mo | **two-lens round; its folds added two of the three no-op saves** **V** |
| [#909](https://github.com/dennys246/Maxim/issues/909) | motor engrams / `predict` / programs with no production consumer | c6277e45 **I** | #1106 (Dormant + scan) | consumer never wired | engram audit, ~5 mo | predates |
| [#910](https://github.com/dennys246/Maxim/issues/910) | `[DANGEROUS]` reads `reward_bias`, clamped ≥ 0 | annotators 3c41ec2a 2026-04-24, clamp 734f3ca4 2026-04-12 **V** | #1104 | consumer on the pre-clamp contract | engram audit, ~5 mo | none recorded |
| [#889](https://github.com/dennys246/Maxim/issues/889) | ablation switch gates `distribute_reward`, which had no production caller | replaced 45c1543b 2026-04-24, switch 8a09ae85 2026-05-30 **V** | #891 | producer replaced, guard left on the dead path | R4 look-back #890, ~4 mo | 8a09ae85 claims a two-lens review on results **V** |
| [#1085](https://github.com/dennys246/Maxim/issues/1085) | adding `plan_approval` to the critical contexts made the auto-approve path dead | 1fb09efd / 1c960c80 2026-04-09 **V** | open | constant change starves a branch | found while fixing #1083, ~6 mo | predates |
| [#1083](https://github.com/dennys246/Maxim/issues/1083) | `Proposal` lacks `cluster_id`; the approved path crashes | #411 26d8f901 2026-07-22 **V** | #1087 | producer type lacks fields the consumer reads | mypy widened to the composition layer, 74 d | **two-lens round on #411** **V** |
| [#1042](https://github.com/dennys246/Maxim/issues/1042) (4) | stall suppression asks `tier="large"`; the router registers a cost tier | 5a1dd499 2026-06-04 **V** | #1045 | key/namespace mismatch; tests used `"large"` on both sides **I** | Exp 10 offline replay, ~4 mo | **two-lens round** **V** |
| #1042 (1–3) | narrator prompt names tools it lacks; #1029 fixed 1 path of N | #1029 | #1047 | fix at 1 of N sites | Exp 10 abort, ~1 d | **#1029 exec + arch: "no blockers"** **V** |
| [#1052](https://github.com/dennys246/Maxim/issues/1052) | prompt teaches `ready_to_act:false`; the loop passes no `bio_enrichment_pipeline` | 499151e5 / 0b595938 (April) **V** per #1056 | #1056 | prompt promises a capability the loop lacks | Exp 10 abort (exposed by #1047), ~6 mo | — |
| [#851](https://github.com/dennys246/Maxim/issues/851) | pending entry leaks when pain is dropped; the any-pending guard then disables attribution | broad guard 60702417 2026-04-14 **I** | #1095 | lifecycle owner missing; guard precondition | #847 review, ~5 mo | **two-lens round; both lenses flagged the guard and folded a doc note** **V** |
| [#864](https://github.com/dennys246/Maxim/issues/864) | interactive learning gate fails open on exception | e89a07a7 / 9dba6010 **V** | #867 | guard swallow inverts direction | #863 call-site review | — |
| [#870](https://github.com/dennys246/Maxim/issues/870) | reflex counts `success=False` / `None` as fired | — | #872 | return value ignored | #863 review | — |
| [#871](https://github.com/dennys246/Maxim/issues/871) | YAML declares deltas, the tool sets absolutes, orphan root key | — | #872 | producer/consumer semantic drift | #870 review | — |
| [#873](https://github.com/dennys246/Maxim/issues/873) | `damage_component` falls back to root health and reports success | — | #1096 | fallback reports success | #872 review, ~10 d | — |
| [#1093](https://github.com/dennys246/Maxim/issues/1093) | orchestrator prompt names parts the default body lacks | — | open | prompt/registry drift | #873 executor lens | — |
| [#972](https://github.com/dennys246/Maxim/issues/972) | write-but-don't-read still restores ATL/AG/cross-layer | hub restore not gated **I** | #986 | flag reaches 3 of 6 restores | #939/#950 executor lens | — |
| [#985](https://github.com/dennys246/Maxim/issues/985) | `create_npc_agent` over an existing home saves nothing after run 1 | **caused by the fixes #973/#983/#986** **V** | open | new guard silences an adjacent caller | #972 review | the breaking fixes had two-lens rounds; the next PR's caught it |
| [#982](https://github.com/dennys246/Maxim/issues/982) | `enable_oscillator()` discards the restored SCN oscillator | f22d9236 2026-04-26 **V** | open | restore then overwrite | #971 executor review, ~5 mo | — |
| D42 | `build_bio_stack` builds a pathless `SCN()` | — | ea346e8b era | builder swap drops the path | score card | — |
| D41 / N2 | `shutdown()` closes a session never opened | — | ea346e8b (#572) | lifecycle pair unmatched | blind score card | — |
| D77 / D79 | per-site kwarg threading dropped `embodiment=`, then `cerebellum=` / `entity_map=` | — | 7a8d704c (#638) | builder parameter at 1 of N sites | D77: 1.1.4 PR3 two-lens round (**a review catch**) | — |
| [#822](https://github.com/dennys246/Maxim/issues/822) | internet policy written and persisted, never read by the live tools | — | 82e245ff | writer with no reader | — | — |
| 7eb77e0f | `--clear-memory` unlinks legacy paths after the per-agent migration | migration **I** | 7eb77e0f | stale path after migration | — | — |
| db5a8768 | auto-spawn never stamps the served n_ctx (only hot-swap did) | — | #484 | 1 of 2 paths | Exp 44 blocker | — |
| [#817](https://github.com/dennys246/Maxim/issues/817) | forming-stage transitions have zero callers | bd1723f1 2026-03-09 **V** | #844 | driver never wired | memory-plan audit | predates |
| #845 / #991 / #993 / #861 | readers probe nonexistent attrs via `getattr` default or swallow; `FakeEpisode` has `.content` | f9a25fb7 (#861) **V**, others **I** | #846 / #996 / open / #867 | wrong field + silent default; test fakes masked it | surveys of adjacent fixes | — |
| 1.2.1 pairing | `make_pairing_announcer` shipped "end to end", zero production callers | 0bf407f2 / ba5f3540 **V** | open (1.3.0 notes corrected) | composition unwired | post-release | the PRs had two-lens folds (0e6c8fb1) |

## Pass B: documents (39 rows)

| Id | What de-wired | Contract | Shape | Found by, how long live | Source |
|---|---|---|---|---|---|
| D43 / #590 | `ec_merge` discarded its id_map; consumers still called bare `nac_merge` | return + identity | return discarded; fix had no caller | strict-xfail D44 gate did not flip | doc |
| D58 | Cerebellum kwargs drift, swallowed | signature | kwarg drift + swallow; a test called it right | design dive, ~5 mo | doc |
| #840 | FearCircuitBridge signature, swallowed | signature | the same | while fixing #839 | doc |
| #841 | str passed where a percept expected | signature/type | the same | the same | doc |
| D60 | `getattr(nac, "last_predicted_valence", 0.5)` — attribute never existed | attribute | default hides a missing producer | audit | doc |
| #908 | load/save path split | persisted path | review folds added the no-op saves | engram audit | doc |
| D42 | pathless `SCN()` | builder / path | 1 of N sites drops a collaborator | verifying D41, ~4.5 mo | inf (date) |
| D41 | `shutdown()` on an unopened session; first fix opened it on a hub later swapped out | lifecycle | wired object replaced | score card; round 2 | doc |
| D28, #972 | restore not gated at every site | config gate | gate at some sites only | D17 work; #939 executor lens | doc |
| #939 / #1071 | `create.*` opened empty and saved over a store | persisted path | write without read | owner score card | doc |
| D2, D17 fold | clear table lacked `ec`; NAc restored beside a fresh EC | paired keys | entry missing; pair split | audit | doc |
| MemoryHub | `.connect()` never called at 2 sites; 3 bridges None | builder | post-construction step forgotten | unification audit | doc |
| PainBus | NAc subscriber missing on 3 of 4 CLI paths | bus subscriber | subscriber at N−1 sites | unification audit | doc |
| build_executor | ToolPainBridge forgotten 3 times, 3 more found | builder param | the same | sem_execution_hook | doc |
| reaction bus | `cerebellum_modulator_factory` dropped `reaction_bus=` | factory param | accepted, not threaded | audit | doc |
| D73 | no `permissions=`; `tool_whitelist` written and read by nothing | builder / config | write-only field | sandbox audit | doc |
| D77 → D79 → D86 | the acquisition seam dropped collaborators three times | builder param | the same skeleton | PR3/PR4 executor lens | doc |
| Exp 32 Bug A | `agent_id` default diverged between surfaces | identity | default masks a missing value | experiment | doc |
| D54 / D57 | pain `Reaction` without `agent_id`; distributor returns early on None | payload identity | required field missing | ternary sweep | doc |
| per-agent stash | module globals shared across agents | identity | global instead of keyed map | v1 P4 | doc |
| D45 | bridge reads `observations["position"]`; nothing writes it | dict key | consumer without producer | scoping dive | doc |
| D9 | 5 of 6 event types have no producer; drive emitter swallowed | topic | producer missing | audit | doc |
| D6 | hub stashes a 1-tuple; consumer returns on len < 2 | arity | consumer early return | audit | doc |
| D53, D56(f) | `success=True` while `reached=False`; caller hardcodes success | outcome | polarity collapsed | microduck pass / sweep | doc |
| #870, #873 | unwired reflex counted as acted; damage fallback returns success | outcome | fallback reports success | #847/#872 rounds | doc |
| ternary sweep | `execute_parallel_actions` discarded `side_effects` | payload | dropped by 1 dispatcher of 3 | sweep | doc |
| `nac_merge` | omitted `cluster_reward_source`; `load_state` reset it | return schema | omitted key resets consumer | D43 work | doc |
| #1085 | constant change killed the only `approve` caller | config set | constant starves a branch | while fixing #1083, ~6 mo | doc |
| Exp 60 harness | loop ran at default PLANNING autonomy; no affordance executed | default | default turns it off | experiment | doc |
| #596 | optional `geometry`: omitting it disabled the guard, and a caller did | default param | omission opts out | 1.1.3 round | doc |
| D68 | `geometry` made required; a script broke with no CI | signature | caller outside CI | re-run shakedown | doc |
| D37 | shallow fetch made every diff lint skip with exit 0 | CI env | graceful skip = success | Cluster B executor lens | doc |
| sim-n-ctx, D82, D85, D32 | one setting resolved two ways; roster vs permission; clamps; prompt vs constitution | config / text | duplicated logic drifts | experiments, audits | doc |
| logging lesson | subcommand dispatch skipped `configure_logging` | lifecycle order | early return skips setup | dev accident | doc |
| tool dual schema | JSONSchema values broke the prompt renderer (swallowed) | LLM-facing schema | consumer not migrated | CC9 | doc |
| decay ticks | decay methods never called per tick (pre 2026-04-24) | lifecycle | hook never invoked | — | doc |
| zero-caller compositions | 1.2.1 announcer, gate 7 bundle composer, #909, #1084, D83, `Tool.cancel` | composition | pieces shipped, composition not | audits, owner | doc |
| PR #435, PR #395 | merged without the fold; branch grew after its round | merge | merged ≠ reviewed | broken main | doc |
| mutable globals | extraction re-imported a mutable global by name | module binding | stale binding after migration | — | doc |
