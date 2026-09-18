# Exp 56 re-baseline port — ARCHITECTURE lens (pre-merge review round)

**Date:** 2026-09-18 · **Scope:** the uncommitted diff of `scripts/exp56/setup_world.py` +
`scripts/exp56/README.md` (branch `exp62/decisions-taken` @ `d7ece4d9`; the `exp62_*` prereg
edit in the same tree is out of scope) · **Purpose under review:** re-baselining the EARNED
2026-09-06 four-arm campaign (Paper 1.16.5, ledger row
`docs/plans/behavioral_graduation_candidates.md:194`) on Paper 1.20.4 per `roadmap_1_3.md:33-36`.

**Verdict: DO-NOT-SHIP as is** — two findings, one in the world geometry the port claims to
preserve and one in the runbook's data path, would each make the re-baseline a *different
apparatus* or *corrupt the EARNED record* while every visible check stays green. Both are
small to fix. The rest is SHOULD-FIX (extract the duplicated builder code now; measured
server-version provenance; the doc/ledger/prereg protocol) and NITs.

The line numbers below are the WORKING-TREE lines of the modified files.

---

## DO-NOT-SHIP

### D1. The 64-layer stack does not put the surface at y=63 on 1.20.4 — the port claims a geometry it does not generate

**Evidence.** `setup_world.py:79` keeps the 1.16.5 layer stack
(`bedrock 1 + dirt 62 + grass_block 1` = 64 layers) and `:78` restates
"= surface y=63, feet y=64 (the frozen neutral)"; the new port paragraph (`:19-20`) asserts
"same superflat surface (y=63, feet y=64 — the frozen neutral)"; `README.md:28-29` repeats it.

**Why it is wrong.** Since Java 1.18 the overworld spans y = −64..319 and the flat generator
lays its layers from the dimension's **minimum** y, not from y=0 (the well-known consequence:
the default "Classic Flat" preset now puts the player at feet y=−60). A 64-layer stack on
1.20.4 therefore occupies y=−64..−1: bedrock at −64, grass at **−1, feet at y=0** — the
bottom edge of the body's declared `y_altitude` range `[0,128]`
(`src/maxim/_data/components/bodies/minecraft_bench.yaml:58-63`), i.e. the *rest-at-extreme*
regime amendment 2 of the prereg named as "the measured-bad regime the body design warns
against" (`exp56_four_arm_sharing_preregistration.md:424-428`).

**Why nothing currently catches it — and why that is the dangerous part.** `prepare`
(`setup_world.py:224-236`) sets the world spawn at the FROZEN rest anchor (0,64,0), force-loads
it and builds a stone pad at y=63. With the wrong layers the bot would stand on a **5×5 pad
floating 64 blocks above a grass plain at y=0**; `y_altitude` would still read 64 at rest, the
slots still sit at y=112, `distance_from_spawn` is unchanged, all roster actions are turns —
so Phase-0 check 1 would very plausibly PASS and the campaign would run to a verdict on an
apparatus that is not the frozen one, with every docstring saying it is. That is the
too-permissive-instrument shape this experiment already paid for once (amendment 2,
`light_level`; write-up `56_four_arm_sharing.md:188`). A passing instrument on the wrong
apparatus is indistinguishable from a passing instrument on the right one.

**Honesty note.** The "layers start at −64" fact is my knowledge of the 1.18 generator, not a
boot of this server — the file's own rule at `:74-75` ("found by booting, not by reading")
applies to me too. The fix below makes the knowledge irrelevant by MEASURING.

**Fix (three parts, all in this diff's files):**

1. `setup_world.py:79` — widen the stack so the surface lands at y=63 regardless of where the
   generator starts (on 1.18+ the world is 384 tall; 128 layers from −64 end at 63):
   ```
   generator-settings={{"layers":[{{"block":"minecraft:bedrock","height":1}},{{"block":"minecraft:dirt","height":126}},{{"block":"minecraft:grass_block","height":1}}],"biome":"minecraft:plains"}}
   ```
   and rewrite the comment at `:78`:
   ```
   # 1.18+ flat worlds generate from the dimension floor (y=-64), so the stack is
   # 1 bedrock + 126 dirt + 1 grass = 128 layers (-64..63): surface y=63, feet y=64
   # (the frozen neutral). The 1.16.5 apparatus used 1+62+1 from y=0 — same surface.
   ```
2. `cmd_verify` (`:245-272`) — add a MEASURED surface check, probed inside the rest anchor's
   force-loaded chunk but OUTSIDE the pad's x±2/z±2 footprint (so `prepare`'s pad cannot mask
   it), refusing with the existing exit-4 convention:
   ```python
   # Surface probe: (8,63,8) is in the rest anchor's force-loaded chunk, outside the pad.
   for (x, y, z), want in (((8, 63, 8), "minecraft:grass_block"), ((8, 64, 8), "minecraft:air")):
       resp = rcon.command(f"execute if block {x} {y} {z} {want} run seed")
       if "seed" not in resp.lower():
           failures.append(
               f"surface probe ({x},{y},{z}) is not {want} — the generated world's surface is "
               f"not at y=63 (1.18+ flat worlds start at y=-64; check generator-settings)"
           )
   ```
   Update the OK line at `:271` to name it: `"verify: OK — surface at y=63 (measured), gamerules set, rest pad + all 4 slot enclosures in place."`
3. `README.md:28-30` — say the surface is *measured*, not assumed:
   `# First boot (generates the superflat world; `verify` MEASURES that the surface is at y=63 — feet y=64, the bench body's neutral). Leave it running:`

**What the re-baseline protocol must record if the measurement disagrees with the model** (it
should not after fix 1, but the rule is the rule): the probe reading (`verify` output), the
bot's `y_altitude` at join (the bridge prints `spawn state: {...}` at
`scripts/minecraft_bridge/index.js:359`), and STOP — the frozen `rest_anchor`/`contingency_slots`
in `common.py::FROZEN` are world coordinates and may not be re-typed; a moved surface is an
apparatus failure to fix in `server.properties`, never a constants edit.

### D2. The shipped runbook writes the re-baseline INTO the EARNED record — and appends

**Evidence.** `README.md:95` still says `--out docs/experiments/data/56_four_arm.jsonl` and
`:98` analyzes that same file; `README.md:63-64` still names `56_phase0.json`, which is
`instrument_check.py:198`'s DEFAULT `--out`. `run_campaign.py:395` opens the output with
`open("a")` — it **appends**. Rows carry no server-version field
(`scripts/_provenance.py:343-348`; the committed first row's keys confirm), so 200 new rows
appended to the 200 EARNED rows would give the analyzer n=100/arm of indistinguishable
1.16.5+1.20.4 mixtures. Separately, `run_campaign.py:415` hard-codes the anti-vacuity kit at
`out_path.parent / "pair0_artifacts"` and `analyze_exp56.py:181` reads the same default; that
kit for the EARNED run is **not committed** (`ls docs/experiments/data/` shows no
`pair0_artifacts`; it is not gitignored either) — it lives only beside the jsonl on the
operator's checkout, and a re-baseline written anywhere under `docs/experiments/data/`
overwrites it.

**Fix (README; no code needed if the subdirectory form is used):** the re-baseline gets its
own evidence directory so the campaign's hard-coded `pair0_artifacts` sibling lands beside the
NEW record, not the old one:
```
python scripts/exp56/instrument_check.py --rcon-password 'CHOOSE_A_PW' \
    --out docs/experiments/data/exp56_rebaseline_1204/56_phase0.json --write-experiment-results
python scripts/exp56/run_campaign.py ... \
    --out docs/experiments/data/exp56_rebaseline_1204/56_four_arm.jsonl ...
python scripts/analyze_exp56.py --in docs/experiments/data/exp56_rebaseline_1204/56_four_arm.jsonl \
    --gate v1 --assert-noop-fails
```
plus one sentence under the campaign block: "**Never point `--out` at `56_four_arm.jsonl` —
the campaign APPENDS and that file is the EARNED 2026-09-06 record.**" And commit the
`pair0_artifacts/` kit with the data PR this time (four small files: `meta.json`, `taught.zip`,
`receiver_pre_nac.json`, `receiver_pre_ec.json`) so ANTI-VACUITY is reproducible from `main`
— the 1.16.5 kit's absence from the repo is a pre-existing gap worth a line in the ledger
annotation (S6), not something this PR can fix retroactively.

---

## SHOULD-FIX

### S1. Port vs fork: this is the second verbatim copy of the server-setup code, and the two copies have ALREADY diverged in both directions — extract NOW

**Evidence of divergence (each copy carries a fix the other lacks):**
- `scripts/exp56/setup_world.py:162-166` verifies the jar's **sha256** against the Fill API
  checksum; `scripts/survival_world/setup_world.py:141-160` only size-checks.
- `scripts/survival_world/setup_world.py:150-160` **unlinks the truncated jar and returns 1**
  on a failed download (its comment: `download_to_file` "writes straight to the final path
  with no .partial staging" — confirmed at `src/maxim/utils/http.py:1092-1099`, mismatch
  raises "with the partial file" left behind). `scripts/exp56/setup_world.py:168-173` prints a
  WARNING and **continues**: `jar.is_file()` at `:150` then treats the truncated jar as present
  on the next run, and the stamp at `:181-183` is written although no usable jar exists (the
  comment at `:116` "written on every successful setup" is false on that path).
- The diff pastes the survival builder's 25-line stale-world guard (`:114-134`) verbatim, minus
  the `_format_version` key the survival stamp writes (`survival:167`).

**Front-gate answer.** "Can it ride on existing infrastructure?" — there is none: no shared
module owns Paper download / EULA / properties / stamp / Java-check; `scripts/_provenance.py`
is the precedent for shared script infrastructure and shows the shape. "Name the reason it
needs its own": CLAUDE.md's band-aid trigger is literally "a fix that would need to be
repeated elsewhere" — the truncated-jar fix and the sha256 check each need repeating in the
other file today, and a THIRD builder (the Exp 62 two-pool world, and any 1.4 world) is on the
roadmap. Extraction is ~60 lines moved, touches no science constant, and both builders become
callers in the same PR.

**Recommended shape** — `scripts/minecraft_bridge/paper_server.py` (beside the bridge the
servers exist for) or `scripts/_paper_server.py` (the `_provenance.py` convention):
```python
def java_major() -> int | None: ...                       # parse `java -version`; None if absent
def refuse_stale_world(server_dir: Path, world_name: str, mc_version: str, *, force: bool) -> str | None:
    """Returns the refusal message (caller prints + exits 2) or None."""
def download_paper(server_dir: Path, mc_version: str) -> Path:
    """Fill v3 lookup, download_to_file, sha256 verify; unlinks the partial file and RAISES on any failure."""
def write_world_stamp(server_dir: Path, mc_version: str, *, stamped_by: str) -> None:
    """atomic_write_json(with_format_version({...})) — see S3."""
```
If the owner defers extraction (file it as owed), then AT MINIMUM in this PR: port the
`jar.unlink(missing_ok=True)` + `return 1` into `:168-173` and move the stamp write above the
download so the "successful setup" comment is true.

### S2. The printed and documented start command uses bare `java` — on this operator's machine that resolves to the pinned Java 11

**Evidence.** `setup_world.py:177` prints `cd {server_dir} && java -Xms1G -Xmx2G -jar ...`;
`README.md:30` and the docstring `:41` say the same. The survival builder documents the trap
explicitly (`survival:48-51`, `:171-173`): "macOS `java` may resolve to an older default JDK
(the pinned exp56 Java 11) even with 17 installed" and pins per invocation. The Java-version
check at `:108-112` is correct as written (it matches `"1.`/`"9.`…`"16.` and not `"17.`), but
it only WARNS — the operator will still copy the printed line.

**Fix.** Use the survival builder's form in all three places:
```
cd ~/exp56_server_1204 && "$(/usr/libexec/java_home -v 17)/bin/java" -Xms1G -Xmx2G -jar paper-1.20.4.jar nogui
```
(`:177` → `f'  cd {server_dir} && "$(/usr/libexec/java_home -v 17)/bin/java" -Xms1G -Xmx2G -jar {jar.name} nogui\n'`).

### S3. The stamp file: no `_format_version`, non-atomic, written on the failure path

`setup_world.py:181-183` writes `{"mc_version", "stamped_by"}` with `Path.write_text`. The
sibling stamp (`survival:167`) carries `"_format_version": "1.0"`; CLAUDE.md's persisted-JSON
invariant asks for it on every persisted JSON and for `atomic_write_json` as the writer. The
two stamps share the `mc_version` key (good — either guard reads either stamp), but a reader
that ever calls `check_format_version` will WARN on the exp56 one. Fix (or fold into S1's
helper):
```python
from maxim.utils.atomic_io import atomic_write_json
from maxim.utils.format_version import with_format_version
atomic_write_json(stamp_path, with_format_version({"mc_version": MC_VERSION, "stamped_by": "scripts/exp56/setup_world.py"}))
```
(`run_campaign.py` already imports `maxim`, so the script family has the dependency; the
lazy `maxim.utils.http` import at `:156` shows the standalone-usable intent — if that matters,
keep the import lazy inside `cmd_setup`.)

### S4. Provenance must carry a MEASURED server version — rows, Phase-0 record, verdict

**Evidence.** `executed_code_provenance` (`scripts/_provenance.py:343-348`) stamps repo hashes
only; nothing in a row, the Phase-0 record or the verdict says which server it ran against
(the 1.16.5 vs 1.20.4 distinction the whole re-baseline is about). `MC_VERSION` in the builder
is a DECLARATION; the row needs the server's own answer. `run_campaign.py:348` already holds
an `RconControl` in live mode, and Paper answers the Bukkit `version` command over RCON
("This server is running Paper version … (MC: 1.20.4)").

**Fix (small, `run_campaign.py` + `instrument_check.py`, not this diff's files — record as the
port's owed follow-up in the same PR series):** in live mode, once at start,
`server_version = world.command("version").strip()` and `row["server_version"] = server_version`
next to `row["provenance"]` (`:419`); same field in the Phase-0 record; the analyzer copies the
distinct set into the verdict and REFUSES (`problems.append`) if a file mixes more than one
value — that is the structural guard against D2's append hazard, cheaper than convention.
Verify the exact `version` response text on the rig before relying on a substring
(`verify-the-instrument`).

### S5. Doc consistency — every stale or unreachable claim in the two files

| where | now | fix |
|---|---|---|
| `README.md:24` | `# Download Paper 1.16.5, write configs.` | `# Download Paper 1.20.4, write configs.` |
| `README.md:28-29` | "generates the superflat world with the surface at y=63" | per D1 fix 3 (measured by `verify`) |
| `README.md:63-64`, `:95`, `:98` | EARNED file names | per D2 (the `exp56_rebaseline_1204/` directory) |
| `README.md:66-69` | "disclose the Phase-0 readings as an amendment entry in the prereg" | for the re-baseline: "as the Re-baseline record entry (not an amendment — see prereg §Re-baseline runs)" (S6) |
| `setup_world.py:16` | cites `roadmap_1_4 T4` | `docs/plans/roadmap_1_4.md` exists only on the unmerged `plan/roadmap-1-4` branch (`git ls-files` at HEAD: absent) and no tracked plan defines a "T4" for Exp 56. Cite what is on `main`: `roadmap_1_3.md` §Phase 0 ("the Exp 56 shared-want fabric IS re-baselined on 1.20.4 … before any claim reuses it") and `survival_world_1_3.md` §"Minecraft version" (the trigger definition). Add the 1.4 cite when that plan merges. |
| `setup_world.py:17` | "ledger row at `9905d4d8`" | `9905d4d8` is the CAMPAIGN commit the ledger row cites, not the row's own commit — say "the campaign at `9905d4d8`". |
| `setup_world.py:74-77` | "1.16+ takes JSON …; 1.19+ spells the level type `minecraft:flat`" | keep, but the JSON on 1.20.4 is untested here — the survival builder only proves `minecraft\:normal`. After D1's measured probe passes on the rig, replace "found by booting" history with the dated 1.20.4 boot. |
| `setup_world.py:21-22` | "The RCON apparatus (`prepare`/`verify`) is version-agnostic." | true for the commands used (`gamerule`, `fill … hollow`, `forceload`, `setworldspawn`, `execute if block … run seed` all exist unchanged in 1.20.4); leave it, but `verify`'s gamerule check reads a substring of the response text, which is the kind of thing a version move changes — first `verify` on the rig is the test. |
| `setup_world.py:130` | "(the 1.16.5 apparatus predates this guard)" | fine; matches the survival wording. |

### S6. Ledger / prereg / write-up consequences — what "Re-baselined <date> on 1.20.4" must rest on

**Is it an amendment?** No. The prereg's amendments (`:398-505`) and Amendment rule
(`:507-511`) govern changes to the DESIGN of the original confirmatory campaign, and its
"ONCE" clause (`:540-544`) governed that campaign; both are discharged and the 2026-09-06
record is untouched. The re-baseline is a **fresh campaign under the frozen protocol on a new
platform**, triggered by the ledger row's own `Re-run on:` list ("Minecraft bridge protocol
change" — `survival_world_1_3.md:306-313` defines a version move as exactly that). The
governing document is the ledger row, so:

1. **Prereg** (`exp56_four_arm_sharing_preregistration.md`) — append a new section AFTER
   "## Amendment rule", not a numbered amendment:
   > **## Re-baseline runs (post-EARNED; not amendments)**
   > A re-baseline is a fresh campaign under this frozen protocol on a new platform, fired by
   > the ledger row's `Re-run on:` triggers. It changes no gate constant, selector knob,
   > schedule constant or slot coordinate, so it is not an amendment and the Amendment rule's
   > structural-invalidity test is not invoked; the "ONCE" clause governed the 2026-09-06
   > confirmatory campaign, whose record is untouched. Each run records: the trigger, the
   > platform delta, the measured apparatus checks, the Phase-0 record, the campaign hash, the
   > data path, and the verdict.
   >
   > **RB-1 — <date>, Paper 1.20.4** (trigger: 1.3 platform move, roadmap_1_3 §Phase 0).
   > Delta: server 1.16.5 → 1.20.4; `level-type` spelling; flat layer stack 1+62+1 → 1+126+1
   > because 1.18+ flat worlds generate from y=−64 (surface HELD at y=63, MEASURED by
   > `setup_world.py verify` at (8,63,8)); Java 17. Phase 0: `data/exp56_rebaseline_1204/56_phase0.json`
   > at `<hash>` (checks 1–5 <readings>). Campaign ONCE at main-reachable `<hash>`, clean tree,
   > `mock: false`, `server_version` "<measured>" on every row →
   > `data/exp56_rebaseline_1204/56_four_arm.jsonl` + `…_verdict.json` (data PR #<n>,
   > merge commit; `pair0_artifacts/` committed). Verdict: <PASS/FAIL per gate>.
2. **Ledger row** (`behavioral_graduation_candidates.md:194`) — a dated annotation in the
   row's established voice, appended after the two DISCHARGED entries:
   > **RE-BASELINED <date> ON PAPER 1.20.4 (trigger "Minecraft bridge protocol change", fired
   > by the 1.3 platform move — the [roadmap_1_3](roadmap_1_3.md) Phase-0 obligation):** same
   > frozen protocol, constants and gates (prereg §Re-baseline runs RB-1); world re-stood by
   > `scripts/exp56/setup_world.py` at 1.20.4 with the surface held at y=63 (measured); Phase 0
   > re-run [56_phase0.json](../experiments/data/exp56_rebaseline_1204/56_phase0.json) all PASS;
   > campaign ONCE at `<hash>` (clean tree, `mock: false`, `server_version` stamped per row),
   > data [56_four_arm.jsonl](…) / [verdict](…) (data PR #<n>, merge commit). **Gates:**
   > TRANSFERRED <x> / ABOVE-FLOOR <x> / WANT-NOT-FILE <x> / BOTH-HALVES <x> / ANTI-VACUITY
   > `kit_pass` <x> (this time the `pair0_artifacts` kit is committed; the 2026-09-06 kit
   > never was — a recorded gap). The 2026-09-06 1.16.5 result stands as the EARNING record;
   > this is the baseline every 1.3 claim that reuses the shared-want fabric cites.
   And state the FAIL branch in the PR body up front, before data exist (literal-vs-structural
   prereg discipline): *if any gate fails, the row does not un-earn — the 1.16.5 result stands —
   but it becomes `Stale` for 1.3 reuse, which by this ledger's own rule blocks 1.3.0, and the
   divergence audit applies before any second attempt.*
3. **Write-up** (`56_four_arm_sharing.md:42-45` "Apparatus (as run)… Paper 1.16.5") — a dated
   "Re-baseline (1.20.4)" subsection pointing at RB-1; the interpretation PR, per the
   structure-or-time rule the README already states (`README.md:105-106`).
4. **Data PR hygiene** (the gated-record contract): prereg §Re-baseline-runs header + the port
   ON `main` before the first data timestamp; campaign from a CLEAN tree at a main-reachable
   hash; `--write-experiment-results`; data PR merge-committed, never squashed.
5. **Branch:** this diff sits on `exp62/decisions-taken` next to an uncommitted `exp62` prereg
   edit. The port must land on its own branch/PR (`exp56/rebaseline-1204`) — a mixed PR splits
   the bisect surface, and the provenance rule needs the harness commit to be reachable on
   `main` independently of the Exp 62 decisions.

---

## NIT

- **N1** `setup_world.py:105` "(coexists with the temurin@11 the 1.16.5 world used)" — the
  operator has `openjdk@11` per the OLD README (`git diff` line 13 removed
  `brew install openjdk@11`); say "the Java 11 the 1.16.5 world used" to avoid naming a cask
  they may not have.
- **N2** `README.md:30` — the trailing comment is 90 columns past the command; move "a FRESH
  dir: the 1.16.5 world is the EARNED apparatus and stays untouched" to its own `#` line
  above.
- **N3** `setup_world.py:284` `--force-world` help — add "(never for the re-baseline: the
  1.16.5 dir is the EARNED apparatus)" so the flag's one legitimate use is not the tempting
  one.
- **N4** `SERVER_PROPERTIES` motd (`:84`) could carry the version ("frozen apparatus, Paper
  1.20.4 re-baseline") — a human joining as spectator (`README.md:72`) then sees which server
  they are on.

---

## CLAUDE.md invariant check (item 5 of the brief)

| invariant | status |
|---|---|
| HTTP via `maxim/utils/http.py` | ✅ `fetch_url` + `download_to_file` (`:156-161`), unchanged |
| No NEW silent swallows | ✅ the added `except (OSError, ValueError)` (`:125-126`) is narrowed and falls through to a printed refusal; the pre-existing `except Exception` at `:168` prints — not new in this diff (its *continue-after-failure* is S1/S2) |
| Env-var rules | ✅ none added |
| `atomic_write_json` + `_format_version` for persisted JSON | ❌ the stamp (`:181-183`) — S3 |
| Removed identifiers / lane tiers | ✅ n/a |
| `scripts/lint_harness_provenance.py` | ✅ clean (run 2026-09-18) — `setup_world.py` spawns no `maxim` |
| `ruff check` / `ruff format --check` on the `.py` | ✅ clean |
| Frozen constants re-typed? | ✅ no — slots/rest still read from `common.py::FROZEN` (`:193`, `:216`) |
| "A fix ships with a caller" | ⚠️ the port's *claim* ("same surface") has no measuring caller until D1's `verify` probe exists |

## Summary for the fold

Two DO-NOT-SHIP: (D1) the 64-layer flat stack lands the surface at y=0 on 1.18+ worlds, not
y=63 — widen to 1+126+1 and make `verify` MEASURE the surface outside the pad footprint, or
the re-baseline runs on a floating pad over the wrong world with every check green; (D2) the
README runbook still points Phase 0 and the campaign at the EARNED files, the campaign
APPENDS, rows carry no server version, and the uncommitted anti-vacuity kit would be
overwritten — give the re-baseline its own `docs/experiments/data/exp56_rebaseline_1204/`
directory and commit the kit. SHOULD-FIX: extract the now-twice-copied, already-diverged
server-setup code into one helper (S1); pin Java 17 in the printed/README start command (S2);
stamp via `atomic_write_json` + `_format_version` (S3); stamp a MEASURED `server_version`
per row / Phase-0 record and refuse mixed files in the analyzer (S4); the doc table (S5); the
protocol: NOT a prereg amendment — a "Re-baseline runs" section + a dated ledger annotation
with the FAIL branch pre-stated, landed from its own branch (S6).
