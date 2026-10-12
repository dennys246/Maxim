# B1 design review — environment lens (2026-10-11, on the DRAFT prereg)

Preserved verbatim from the reviewer.

I read the draft prereg, the Exp 60 prereg (apparatus, §Design (iii), Amendments 4–7, runbook), the bridge `index.js`, `minecraft_player.yaml`, `minecraft_harness._loop_kwargs`, `tools/base.py::_validate_input`, `embodiment/backends/minecraft.py`, `setup_world.py` and the run-2 rows in `exp60_trials.jsonl`.

## DO-NOT-BUILD

**E1. The params of a RANDOM or Q action are undefined, and they decide whether exploration can destroy the apparatus.** The draft says "uniform over the selectable actions" but never says what params a drawn action carries.
- **With `params={}`** (what `propose_via_substrate` emits today), the apparatus is safe:
  - `move_to`, `turn`, `mine_block` and `place_block` fail straight away with `Missing required input`. This is `Tool._validate_input`, raised before the bridge.
  - `flee` fails fast underwater (`bot.entity.isInWater`).
  - `attack_nearest` fails because no hostile spawns (`doMobSpawning false`).
  - `eat` throws `Food is full` at food 20.
- **If the hook samples coordinates**, the apparatus can be destroyed:
  - `mine_block` can break the stone shell or the pool floor, and the pool drains or floods the shore. Underwater digging is slow, so it outlives the 15 s Python `action_timeout_s` and is booked `unknown`/neutral while the bridge keeps digging.
  - `move_to` on the shore uses default `Movements` (`canDig=true`), so it can tunnel through the shell.
  - A `goto` survives a Python timeout and a `/tp`.

**Prereg change:** pin "every arm's action carries `params={}`, exactly as FEAR's `recommend_action` output does". Add a unit test that the hook cannot emit params. Add an apparatus re-verify before and after each arm's seed (the water check's pool source cells, lip, shore floor and shell faces, read over RCON), refusing on any mismatch.

**E2. The inherited Exp 60 refusals make RANDOM and Q_EXPLORE unrunnable as written.** Exp 60 refuses a seed on "a positive escape link after the preflight". Any arm that *executes* `escape_water` successfully books NAc positive outcomes through `tool_dispatch.record_outcome`. That happens in RANDOM's pre-probe and in Q_EXPLORE's training. FEAR's post probe alone logged 51. **Prereg change:** re-scope each inherited refusal per arm. Keep the rule's intent (no instrumental escape credit before the probe) only where an arm claims none. For RANDOM and Q_EXPLORE, record the escape links rather than refusing on them.

**E3. The per-seed interleaving cannot be done on the current harness.** `exp60_run.py run` takes one `--arm` per invocation and mints a run_id per invocation. Its seed selection is a prefix (`FROZEN["seeds"][: args.seeds]`), so it cannot run seed 12 alone, and the verdict's `--run-id` logic is per arm. **Prereg change:** either declare a multi-arm driver with one run_id per session and per-seed arm order (a harness requirement for the code review), or drop the interleave and state which drift control replaces it.

## SHOULD-FIX

**S1. H1's expected value rests on the wrong time budget and an unmeasured action set.**
- **The time budget is shorter than the draft assumes.** Escape takes 1.45–1.83 s, so a draw only counts if it starts by about cap − 1.83 ≈ 2.5 s after the teleport, not anywhere in 4.3 s. Run 2's FEAR telemetry shows ticks at irregular spacing (0, 0.52, 1.21, 1.80, 2.52 s), with the first call at about 1 s. That leaves about 4–5 useful draws, not 8.
- **The action set is unmeasured.** The body declares 8 affordances. The ALWAYS_ALLOWED no-op head tools (`move`, `track_target`, …) also sit in the loop's universe under AUTONOMOUS (see the comment in `_loop_kwargs`). The draft's "about 15" is a guess.
- **Failed draws do not cost extra time.** Nearly every non-escape draw fails instantly, so the "less for actions that take time" discount mostly does not apply. The exception is `eat` once food drops below 20, which takes about 1.6 s.

**Prereg change:** add a preflight that enumerates and stamps the exact `available_tools` set RANDOM draws from (after `executor.permits`). Recompute the expectation as 1 − (1 − 1/N)^k with k ≈ 4–5: N = 8 gives about 0.41–0.49, N = 12 gives about 0.29–0.34.

**S2. Food and saturation drift with arm and order.** `escape_water` holds jump, and swimming and jumping add exhaustion. Over about 20 seed-sessions, food and saturation can fall from rest. Both are world-modality sensors, so the cluster ids shift. Below food 20, `eat` starts *succeeding* underwater: it consumes bread (inventory drift, since `keepInventory` holds the 64 bread `prepare` seeded) and blocks about 1.6 s. **Prereg change:** add a per-seed preflight that food = 20 and saturation is at the bridge clamp, restored game-natively with `effect give <bot> minecraft:saturation 1 10` and then verified. Stamp the bread count per seed.

**S3. Q_EXPLORE's rescue must run on the harness clock, not the loop's.** Exp 60 training never executed anything, so the loop never blocked. Under execution, the loop thread sits inside the action (`escape_water` up to its 8 s cap, `eat` about 1.6 s, worst case the 15 s Python timeout). Meanwhile no `evaluate_failures()` tick publishes the "saturating pain" terminal. **Prereg change:** set the Q_EXPLORE episode terminals by bridge truth from the harness thread: `is_in_water` 0, sensed oxygen ≤ the training threshold, or a health drop. Use the geometry probe's cap of onset − 3 s (≈ 13 s), not onset − 1 s, and rescue by RCON before stopping the loop. Send the bridge `stop` at every rescue, because a `goto` persists.

**S4. The specificity read cannot discriminate on this apparatus.** With `params={}`, no action can move the bot *into* water. `flee` goes to the shore anchor and `escape_water` on the shore returns "already at surface". So shore water entry is about 0 for every arm by construction. **Prereg change:** state that up front and report action counts only, or the readout looks like evidence.

**S5. Box contention.** No arm has an LLM consumer (`_loop_kwargs` passes no `llm_worker`), so there is no :8100 contention. The edges in the window (pain at 5.1 s, escape 1.5–1.8 s) are wall-clock physics at the server's tick rate, though. A qwen32b inference from another session on the same unified-memory box could lag the server. **Prereg change:** add a preflight that no inference job is active on the rig (or record its load and the server tick rate), and a contemporaneous apparatus check on the run day. The verdict already binds one apparatus record.

## NIT

- **N1. The run-length estimate is high.** Run 2's timestamps show about 3.3–3.5 min per seed, not 8–9. Four arms × 5 seeds is about 70–90 min with re-verifies.
- **N2. The 1 s shore warm-up acts too.** For RANDOM, actions run during the warm-up; an `eat` in flight at the teleport eats window time. Record any call in flight at the teleport.
- **N3. `eat` equips bread even when it fails.** This changes the held item. It is harmless, but note it.

## Verified

- **Bridge action behaviour**
  - `flee` fast-fails in water.
  - `escape_water` bypasses the pathfinder and caps at 8 s.
  - mineflayer `consume` throws `Food is full` at 20.
  - Missing required params raise in Python before the bridge.
  - The bridge runs actions concurrently, so `stop` works mid-`goto`.
- **World settings:** gamerules `doMobSpawning false`, `keepInventory true`, `doDaylightCycle false`, with bread seeded by `prepare`.
- **No LLM in the loop:** AUTONOMOUS autonomy, and no LLM worker in the harness.
- **Run-2 timings:** escape took about 1.6 s and latency read 3.34 s against a 4.335 s cap. Each seed ran about 200 s.
