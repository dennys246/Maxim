# Exp 56 re-baseline port (Paper 1.16.5 → 1.20.4) — EXECUTOR lens

Reviewed: the uncommitted diff of `scripts/exp56/setup_world.py` + `scripts/exp56/README.md`
(2026-09-18, branch `plan/roadmap-1-4`, on top of `0460a747`). Reference: `scripts/survival_world/setup_world.py`.
No files edited. Everything below was RUN, not read, unless marked "prior knowledge — verify by boot".

## Verdict

**DO-NOT-SHIP as written — one finding (E1).** The port's load-bearing claim ("same superflat
surface y=63, feet y=64 — the frozen neutral; only the server version changes") is predicted FALSE
on 1.20.4 with the layer stack it ships, and neither `setup` nor `verify` would catch it. The fix is
one number plus one `verify` probe, then a boot to confirm. Everything else in the diff runs
correctly in every branch I exercised (exit codes below).

## Findings

### E1 — DO-NOT-SHIP: the 64-block layer stack puts the surface at y=-1 on 1.20.4, not y=63

`scripts/exp56/setup_world.py:79` (generator-settings) keeps
`bedrock(1) + dirt(62) + grass_block(1)` and `:78` keeps the comment
"bedrock + 62 dirt + grass = surface y=63, feet y=64 (the frozen neutral)"; the new docstring at `:19-20`
and `README.md:27-28` assert the same surface.

Prior knowledge (high confidence; the one thing on this page I could not boot to confirm — this host has only
Corretto 8): since Java Edition 1.18 the flat generator stacks its layers from the dimension's
`min_y` (overworld: -64), which is why the DEFAULT superflat surface moved to y=-60 in 1.18+. The vendored
client side agrees (`scripts/minecraft_bridge/node_modules/prismarine-chunk/src/pc/1.18/ChunkColumn.js:21`
`CAVES_UPDATE_MIN_Y`; `mineflayer/lib/plugins/game.js:76-81` reads `min_y` from the dimension codec). On
1.16.5 (min_y 0) this stack gave grass at y=63; on 1.20.4 the same stack gives bedrock at -64, dirt -63..-2,
grass at **y=-1, feet at y=0**.

What that does to the apparatus:
- `prepare` still builds the 5x5 stone pad at y=63 (`:224-226`) and the slot boxes at y=112 (`:186-196`),
  and `settle_until_reflected` RCON-`tp`s the bot to every anchor (`common.py:~707`), so the bot's
  `y_altitude` at rest is still 64 and the bridge's `distance_from_spawn` is horizontal-plane from
  `bot.spawnPoint` (`index.js:93-96`) — the two signature sensors are numerically preserved. But the rest
  anchor becomes a floating 5x5 island 64 blocks above the ground rather than a pad flush with the
  prereg's "superflat surface at y=63", first-join placement lands the bot at y=0 ~94% of the time
  (heightmap spawn within `spawnRadius` 10 off a 5x5 pad), and the re-baseline would be reported as
  "same apparatus" on a world that is not. That is exactly the runbook-only claim E5 is about.
- Worse in 1.20.4 than 1.16: a schema-mismatched `generator-settings` JSON no longer crashes boot —
  `FlatLevelGeneratorSettings.CODEC.parse(...).resultOrPartial(LOGGER::error)` falls back to the DEFAULT
  flat preset (surface y=-60) and the server comes up green. Only a probe of the world can tell.

Exact fix:
1. `:79` — `{"block":"minecraft:dirt","height":62}` → `{"block":"minecraft:dirt","height":126}`
   (1 + 126 + 1 = 128 = 63 - (-64) + 1; bedrock -64, dirt -63..62, grass 63, feet 64). Update the `:78`
   comment and the docstring `:19-20` / `README.md:27-28` to say "126 dirt (1.18+ stacks from y=-64)".
2. Add the surface probe to `cmd_verify` (`:258-266`, beside the rest-pad check) at a point OFF the pad,
   e.g. `execute if block 20 63 20 minecraft:grass_block run seed` and
   `execute unless block 20 64 20 minecraft:air run seed` → failure text
   "surface is not at y=63 — the flat layer stack is wrong for this MC version". This is the regression
   guard the port currently lacks: today `verify` only checks blocks that `prepare` itself placed, so it
   passes on ANY ground plane.
3. Boot once with Java 17 and run `verify` before the re-baseline campaign. If the boot shows grass at
   y=63 with height 62 (i.e. my prior is wrong), keep 62 — the probe in (2) is still the fix, because it
   turns the claim into a measurement either way.

### E2 — SHOULD-FIX: new `SyntaxWarning: invalid escape sequence '\:'` on every import

`scripts/exp56/setup_world.py:76` — the added comment line INSIDE the `SERVER_PROPERTIES` template reads
``(the survival builder's `minecraft\:normal` form`` in a non-raw string. Observed on every run:
`setup_world.py:76: SyntaxWarning: invalid escape sequence '\:'`. `ruff check` passes (W605 is not in
the enabled set), so CI will not catch it, and it becomes a `SyntaxError` in a future CPython.
The line is also emitted verbatim into `server.properties` (`# ... minecraft\:normal ...`; harmless there
as a comment). Fix: write it `minecraft\\:normal` in the template (matching `:74`), or drop the aside.

### E3 — SHOULD-FIX: the printed start command and the README use bare `java`, which on this host is Java 8

`scripts/exp56/setup_world.py:176` prints `cd … && java -Xms1G -Xmx2G -jar paper-1.20.4.jar nogui`;
`README.md:30` the same. On the reviewing host `java -version` → `openjdk version "1.8.0_392"` (the script
itself warned so, then printed the bare-`java` command anyway). The survival builder already documents
this exact trap for this exact operator (`scripts/survival_world/setup_world.py:48-51, 170-173`: "macOS
`java` may resolve to an older default JDK (the pinned exp56 Java 11) even with 17 installed") and pins
per invocation: `"$(/usr/libexec/java_home -v 17)/bin/java" -jar paper-1.20.4.jar nogui`. Mirror that in
both places. (The 1.16.5 server needs the old default to STAY old, so "change the default JDK" is not the fix.)

### E4 — SHOULD-FIX: the Java predicate misses GA (x.0.0) builds, which print no dot

`scripts/exp56/setup_world.py:108` `any(f'"{v}.' in jv for v in ("1","9",…,"16"))`. Measured matrix:

| `java -version` first line | warns? | correct? |
|---|---|---|
| `openjdk version "1.8.0_392"` (this host) | yes | yes |
| `openjdk version "11.0.21" 2023-10-17` | yes | yes |
| `openjdk version "16.0.2" 2021-07-20` | yes | yes |
| `openjdk version "17.0.9"`, `"17"`, `"21"`, `"21.0.1"`, `"22-ea"` | no | yes |
| `openjdk version "11" 2018-09-25` (11 GA) | **no** | **false negative** |
| `openjdk version "16" 2021-03-16` (16 GA) | **no** | **false negative** |
| `openjdk version "9"`, `"15-ea"` | **no** | **false negative** |
| `The operation couldn't be completed. Unable to locate a Java Runtime.` (macOS stub, no JDK) | no | prints `java: The operation…` — misleading (pre-existing) |

No false positives found (`"1.` cannot match `"21.`/`"17.` because the quote is anchored). Fix — parse
instead of substring-match:
```python
m = re.search(r'version "(\d+)(?:\.(\d+))?', jv)
major = int(m.group(1)) if m else None
if major == 1 and m.group(2): major = int(m.group(2))   # "1.8.0_392" → 8
if major is None or major < 17: print("WARNING: … Java 17+ …")
```
which also turns the no-JDK stub into a warning instead of a `java: …` line.

### E5 — SHOULD-FIX: no campaign row records the SERVER version — the re-baseline cannot prove 1.20.4

Grepped `docs/experiments/data/56_four_arm.jsonl`, `56_four_arm_verdict.json`, `56_phase0.json`:
zero occurrences of `1.16.5`. Row keys are `action_map, allow_dirty, approach_latency, arm, bias_decisive,
chose_target, contact_at, first_contact, gated, mock, n_decisions, pair_seed, provenance, slot, target_aff,
ts, working_tree_dirty_src_scripts`; `provenance` (from `scripts/_provenance.py::executed_code_provenance`,
`:343-348`) carries only harness/executed git hashes, the maxim file, pythonpath. `scripts/exp56/common.py`
and `run_campaign.py` have no server/version field; the bridge (`index.js`) never sends `bot.version`.
So the EARNED row's "1.16.5" rests on the runbook, and the re-baseline's "1.20.4" would too — the D5-hash
family of gap R3 just paid for.

Exact seam (one RCON call, the running server answers for itself — stronger than reading a stamp file the
harness cannot locate, since it only knows `--rcon-host/port`):
- `scripts/exp56/run_campaign.py:349-353` (the live branch that builds `world = C.RconControl(...)`):
  `server_version = world.command("version").strip()` (Paper's Bukkit `version` command:
  "This server is running Paper version … (MC: 1.20.4)"); `None` under `--mock`. Refuse (exit 3, the
  Exp 52 shape) if the reply does not contain `MC: 1.20.4` — the frozen apparatus now names its platform.
- `scripts/exp56/run_campaign.py:417-419` (beside `row["provenance"] = provenance`):
  `row["server_version"] = server_version`.
- Same two lines in `scripts/exp56/instrument_check.py` (Phase 0 record) so `56_phase0*.json` carries it.
- `scripts/analyze_exp56.py`: copy the set of `server_version` strings seen into the verdict record and
  refuse a verdict over rows that disagree.
- Optionally `cmd_verify` prints the same `version` reply so the runbook step logs it too.

### E6 — SHOULD-FIX (pre-existing shape, but the port claims to mirror the sibling that fixed it): a truncated jar survives to the next run

`scripts/exp56/setup_world.py:152-171`: `download_to_file(... expected_bytes=...)` "raises HTTPError on
mismatch with the partial file preserved for caller cleanup" (`src/maxim/utils/http.py:1098-1100`); the
`except Exception` branch prints the WARNING but does not unlink, so the NEXT run hits
`if jar.is_file(): print("paper jar already present")` (`:151-152`) on a truncated jar. Only the sha256
branch unlinks. The survival builder handles exactly this (`scripts/survival_world/setup_world.py:152-156`,
`jar.unlink(missing_ok=True)` with the reasoning in its comment). Fix: `jar.unlink(missing_ok=True)` first
thing in the except. (Keeping exit 0 + WARNING here is fine per the brief; survival returns 1 — either is
defensible, but the stale partial jar is not.)

### NITs

- N1 `README.md:24` still says `# Download Paper 1.16.5, write configs` — stale after this diff.
- N2 Stamp shape differs from the sibling: survival writes `{"_format_version": "1.0", "mc_version": …}`
  (`survival_world/setup_world.py:167`); this writes `{"mc_version", "stamped_by"}` with no
  `_format_version` (`setup_world.py:181-183`). CLAUDE.md's persisted-JSON rule; add the key for parity
  (neither uses `atomic_write_json`; both are dev-tool stamps — consistency is the point).
- N3 First boot of a Paperclip 1.20.4 jar downloads the Mojang vanilla jar (~50 MB) at boot time —
  network needed at BOOT, not only at `setup`. Worth one runbook line next to the start command.
- N4 The corrupt-stamp branch (`:120-121`) reports "with no version stamp (the 1.16.5 apparatus predates
  this guard)" for a stamp that exists but is unreadable — the parenthetical is wrong for that case.
  Cosmetic; the refusal itself is right.

## What I verified — (1)–(6) as briefed

**(1) `level-type` / `generator-settings` form.** `.format()` output measured:
`level-type=minecraft\:flat` (the `\\:` in the non-raw template renders as one backslash — same spelling
as `survival_world/setup_world.py:91` `minecraft\\:normal`, confirmed by reading it) and the
generator-settings line round-trips `json.loads` to the intended dict (the `{{ }}` escapes are right).
`java.util.Properties` unescapes `\:` → `:`; Paper writes the value back escaped, so this is the idempotent
form. Prior knowledge (verify by boot): 1.19+ `DedicatedServerProperties` parses `level-type` as a
`ResourceLocation` after a legacy-name map (`default`, `largebiomes`; bare `flat` also resolves to
`minecraft:flat`), so the spelling is accepted; the flat generator's codec still takes `layers[{block,height}]`
+ `biome`, so the JSON is schema-valid — BUT see E1 for what those layers produce, and note the
1.20.4 failure mode for a bad JSON is a logged fallback to the default preset, not the 1.16 boot crash the
comment at `:74-75` describes.

**(2) Java predicate.** Matrix above (E4). Correct on 1.8 / 11.x / 16.x / 17 / 21; false negatives on the
dot-less GA strings `"9"`, `"11"`, `"16"` and `-ea` builds. No false positives.

**(3) Stamp guard wiring.** `level-name=exp56_world` at `:80` matches `world_dir = server_dir / "exp56_world"`
at `:115`. The stamp write is the LAST statement before `return 0` (`:181-184`): after the EULA refusal
(`:135-141`, returns 2 first — measured in F2: stamp stayed `1.16.5`), after the properties write, after the
download block (which cannot return early, so the stamp is also written on the download-WARNING path —
acceptable: it stamps the dir's intent, no world exists yet). `--force-world` is on the `setup` subparser
(`:281-285`), `args.force_world` therefore exists on that namespace; `verify --force-world` →
`unrecognized arguments` exit 2 (measured, H).

**(4) Runs** (`PYTHONPATH=src`, scratch dirs; host Java = Corretto 8):

| case | exit | observed |
|---|---|---|
| A2 fresh dir, `--accept-eula` | 0 | Java-8 WARNING printed; Paper 1.20.4 build 499 (42,781,488 B) downloaded + sha256-verified; `eula.txt`, `server.properties`, `world_version.json` `{"mc_version":"1.20.4","stamped_by":…}` written |
| G2 re-run on that dir after simulated first boot (`exp56_world/` present, 1.20.4 stamp) | 0 | "paper jar already present", proceeds (idempotent) |
| B fresh dir, no `--accept-eula` | 2 | refuses; nothing written |
| C `exp56_world/` + `{"mc_version":"1.16.5"}` stamp | 2 | "holds a world stamped MC 1.16.5 … targets MC 1.20.4"; stamp untouched |
| D `exp56_world/`, no stamp | 2 | "no version stamp (the 1.16.5 apparatus predates this guard)" |
| E `exp56_world/` + `not json` stamp | 2 | same refusal (see N4 wording) |
| F2 1.16.5 stamp + `--force-world`, no `--accept-eula` | 2 | refuses at EULA; stamp still `1.16.5`, no eula/properties written |
| I 1.16.5 stamp + `--force-world --accept-eula` | 0 | proceeds, restamps `1.20.4` |
| H `verify --force-world` | 2 | argparse rejects (flag scoped to `setup`) |

Every run also printed the E2 `SyntaxWarning` and the E3 bare-`java` start hint.
`ruff check` / `ruff format --check` on the file: clean.

**(5) Provenance gap.** Confirmed — E5, with the seam.

**(6) Tests.** `PYTHONPATH=src python -m pytest tests/unit/test_exp56_harness.py -q` → **23 passed in 19.11s**.
(None of them exercise `setup_world.py`; the guard/predicate have no test — the runs above are the only
evidence, which is fine for a dev tool but worth knowing.)

## Not verified on this host (needs Java 17 + a boot)
- The E1 ground-plane prediction (surface y=-1 with height 62) and the E1 fix (y=63 with height 126).
- That Paper 1.20.4 accepts `minecraft\:flat` + this JSON without the default-preset fallback.
- That the Bukkit `version` command answers over RCON with the `(MC: 1.20.4)` suffix (E5's assertion string).
All three are one boot + `verify` away; E1's probe makes the first two self-checking thereafter.
