# Device registry (#1806): round 6 check, adversarial lens (inputs and trust)

Reviewer: adversarial lens, round 6 check, read-only, 2026-11-25, v7 @ 352ea398. Scope: R5-1, R5-2 and R5-3 only (`git diff b0cbd8f7 352ea398`), plus any new fail-open defect those folds introduce. Code read at b7380ee7 (`tools/base.py::git_dir_write_refusal`, `utils/child_env.py`, `hardware/reachy/controller.py`); SDK `reachy_mini.py::_configure_mediamanager` in `.venv`. One offline probe ran under `env -i HOME=$(mktemp -d) python -I` (E1). No network, no robot, no sim.

**Verdict: BUILD. STOP RULE MET.** All three folds are CLOSED. The R5 probe cases (A–F) and nine new ones are all refused by the v7 algorithm as written, and the three negative controls are allowed. The three new findings are fail-closed text corrections in covered classes: a residual that is stated more broadly than it is, a mitigation sentence with no mechanism behind it, and a name rule with no red gate that only it can satisfy.

## Re-check

| Fold | v7 status | Note |
|---|---|---|
| R5-1: SDK-exact `value.lower()` + canonical literal to `ReachyMini(` | **CLOSED** | **SDK normalisation.** The SDK lowercases only: `media_backend.lower() == "no_media"`, then `in ("default","auto")`, then `MediaBackend(lower())`. It does no strip or NFKC.<br><br>**No desync is possible.** v7 classifies on the same `lower()` and passes the matched canonical literal. `" no_media"`, `"ｎｏ_media"` and a Cyrillic-`і` spelling are refused, and `NO_MEDIA`/`WebRTC` map to `no_media`/`webrtc` (E1).<br><br>**Exhaustive scan.** Across all of Unicode, the only non-ASCII code point whose `.lower()` is ASCII is U+212A (Kelvin sign) → `k`. No allowlist word contains `k`.<br><br>**Reconnect.** The SDK reconnect reuses the string that was passed, which is now the canonical one |
| R5-2: env-independent case-folded tail rule + stated residuals | **CLOSED** (text nits N6-1, N6-2, N6-3) | **What the rule does.** Applied to the realpath, the rule refuses the operator's default `~/.maxim/robots.yaml` and `~/.config/maxim/*` even when every root was derived from a re-homed env (E1 K, K2). A custom data home stays open, as stated (K3).<br><br>**Over-deny is fail-closed and narrow.** `ws/robots.yaml`, `~/.config/maximal/x` and `~/.maxim/sim_reports/x.json` are allowed. Only a `.maxim/`-parented robots file, or a path under some `.config/maxim/`, is denied, at any depth |
| R5-3: every-ancestor walk + absent-root anchoring + kept realpath/NFC/casefold compare + red gates | **CLOSED** | **The walk matches #957.** The walk built from §5 as written visits `real` itself and then every ancestor, as #957's loop does (`current = real` first, `base.py::git_dir_write_refusal`). Absent roots are anchored at their deepest existing ancestor plus a tail.<br><br>**R5's fail-open cases.** R5's B, C, D, E and F are refused, as is A. Each is refused by all three tests independently (E1). So are:<br>- `.config` itself absent (anchor = home, tail `.config/maxim`);<br>- case variants of absent and existing paths;<br>- a dangling workspace symlink into the absent config dir;<br>- a workspace symlink to `~/.maxim`.<br><br>**Red gates.** The listed red gates cover R5's four fail-open shapes |

## New findings

| # | Finding | Class | Fail-open / closed | Covered? |
|---|---|---|---|---|
| N6-1 | **The stated residual is broader than the real one.** The name rule matches the realpath tail at any root, so a child under a **different uid** still refuses the operator's **default** `~/.maxim/robots.yaml` and `~/.config/maxim/*` (E1 K, K2). The only residual left is an operator's **non-default** `MAXIM_DATA_HOME`/`XDG_CONFIG_HOME` (K3), in a child that re-points it or runs under another uid. **Fix:** narrow the residual sentence to that case | NF3 (spec accuracy) | fail-closed (over-states) | Yes |
| N6-2 | **A mitigation sentence with no mechanism.** "The parent's roots are passed to children through the #1593 `child_env` seam where the harness uses it." `utils/child_env.py::child_env` strips `PYTHONPATH` site entries and passes every other variable unchanged. It carries no roots, and v7 names no variable, reader or test. A harness that sets `MAXIM_DATA_HOME=<tmp>` overrides the parent's value through `overrides`, which is exactly the residual. **Fix:** delete the sentence, or specify an additive-only inherited roots variable with a reader and a test. A child must only be able to add roots, never replace them. Otherwise the sentence is a fix with no caller | R3-10 / "fix ships with a caller" | fail-closed (no behaviour depends on it; the residual stays as stated) | Yes |
| N6-3 | **The name rule has no isolating red gate, and its wording is loose.** Every listed test (XDG case, `MAXIM_DATA_HOME` child case, NFD case and the four R5-3 gates) is also satisfied by the root union or the walk. So deleting the name rule passes the whole suite, and its deletion probe is vacuous. Separately, "Under any root … wherever they appear" can be misread as "under each denied root", which would be redundant with the union. **Fix:**<br>1. Add one gate that only the name rule satisfies: patch `HOME`, `MAXIM_DATA_HOME` and `pwd` home to a tmp dir, then write `<operator home>/.maxim/robots.yaml` and expect a refusal.<br>2. Reword the rule as "a component-wise suffix match on the realpath, at any depth" | deletion-probe discipline / NF3 | fail-closed today (the rule as specified denies); regression risk only | Yes |

## Evidence

### E1. Probe

Script: `/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/fc699e42-5cb0-4089-8c73-d65cffebfe42/scratchpad/r6/probe_r6.py`, adapted from `probe_r5.py`. It runs three tests, each as v7 specifies:

- **walk:** every existing ancestor including `real`, `samefile` against each denied entry, and each absent entry anchored at its deepest existing ancestor plus a tail; the tail is compared NFC + casefold and component-wise;
- **string:** realpath + NFC + casefold, a prefix compare on component boundaries;
- **name:** case-folded realpath suffix.

The denied set is the config dir plus the three data-home files.

```
H  .config absent, new ~/.config/maxim/config.json      -> refused walk+string+name
H2 ~/.CONFIG/Maxim/api_key                              -> refused walk+string+name
A  new robots.yaml                                      -> refused walk+string+name
B  new util/device_state.json (util exists)             -> refused walk+string+name
C  config dir absent, new config.yaml                   -> refused walk+string+name
D  config dir absent, case variant                      -> refused walk+string+name
I  dangling ws symlink into absent config dir           -> refused walk+string+name
NEG ~/.config/maximal/x                                 -> allowed
NEG ws/robots.yaml                                      -> allowed
NEG ~/.maxim/sim_reports/x.json                         -> allowed
E  new file in existing config subdir                   -> refused walk+string+name
F  EXISTING robots.yaml overwrite                       -> refused walk+string+name
F2 ~/.MAXIM/ROBOTS.YAML (existing)                      -> refused walk+string+name
J  ws symlink -> ~/.maxim, write m/robots.yaml          -> refused walk+string+name
-- roots derived for a re-homed child (denied set = child tmp only)
K  operator ~/.maxim/robots.yaml                        -> refused name
K2 operator ~/.config/maxim/x                           -> refused name
K3 operator CUSTOM data home state/robots.yaml          -> allowed (stated residual)
-- R5-1
non-ASCII chars whose .lower() is ASCII: [U+212A -> 'k']
' no_media' refused · 'NO_MEDIA' -> 'no_media' · 'ｎｏ_media' refused · 'WebRTC' -> 'webrtc'
'no_medіa' (Cyrillic і) refused · 'local' / 'GStreamer' refused · 'Kauto' (Kelvin) refused
```

K and K2 are refused by the name rule alone, which is the basis for N6-1 and N6-3.

### E2. `child_env` carries no roots

`src/maxim/utils/child_env.py::child_env` copies the environment, strips this interpreter's site dirs from `PYTHONPATH`, then applies `overrides`. Nothing in it, or in its callers, propagates protected roots. This is the basis for N6-2.

### Out-of-scope observations

These are not new and were not introduced by the v7 folds; they are noted for the safety lens and for implementation.

- **A pre-existing residual of the same kind #957 carries.** On an **HFS+** volume, where some Unicode-ignorable code points are ignored in name comparison, an absent config dir spelled with a zero-width character slips past the anchor-tail and string compares. APFS, the default home volume since 2017, does not ignore them. Stripping `Default_Ignorable_Code_Point` characters before the compare would close it.
- **§4.2's first allowlist bullet is still scoped to "not `localhost_only`".** The G2 fold makes the onboard proof, not the label, the LOCAL gate. The controller (§4.3) is the chokepoint and refuses LOCAL for a manual tunnel, so this is not fail-open. The load-time scope should still read "does not pass the onboard proof".
- **A non-`str` YAML value** (`media_backend: no` parses as `False`) should be refused at load. That is fail-closed either way, because the SDK's `.lower()` also raises.

## Stop-rule outcome

**STOP RULE MET.** R5-1, R5-2 and R5-3 are CLOSED. All three new findings are fail-closed and in already-covered classes:

- N6-1, a residual stated more broadly than it is (NF3 spec accuracy);
- N6-2, a mitigation sentence with no mechanism (R3-10 / caller rule);
- N6-3, a name rule with no isolating red gate (deletion-probe discipline).

No further adversarial round is owed. N6-1 to N6-3 are one-sentence text folds plus one test row.
