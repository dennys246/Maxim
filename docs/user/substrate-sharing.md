# Substrate Sharing

## Overview

Every Maxim run trains a bio-substrate as a side effect. As an agent acts, the NAc forms causal links from tool outcomes (`tool:X → outcome:Y, confidence Z`) and the EC clusters concepts into centroids. **Substrate sharing** lets you package that learned substrate into a portable bundle, hand it to another Maxim, and merge it into a live system -- so a second instance can start from accumulated experience rather than from zero.

This is the 1.0 foundation for the [Maxim Hivemind](../hivemind.md) -- the peer-to-peer substrate-exchange layer. The 1.0 ship gives you the **bundle format**, the **merge utilities**, and the **`maxim substrate` CLI verbs**. The Oasis software (1.1) and full P2P protocol (1.2) build on these surfaces.

## What's in a bundle

A substrate snapshot bundle is a versioned ZIP archive:

```
maxim-substrate.zip
├── manifest.json   # _format_version, schema_version, contributor_id, domain, signature slots
├── nac.json        # NAc causal links + reward biases + provenance  (1.0)
└── ec.json         # EC concept centroids + cluster metadata        (1.0)
```

That's it. Two payloads, plus a manifest.

**Hippocampus episodes are NEVER bundled.** This is the load-bearing privacy invariant. Episodes are literal experience records -- the full PII surface -- and they are local-only by construction. Only distilled substrate (NAc weights, EC centroids) ever leaves a Maxim. The bundle composer has no code path that reads episodes, so this is enforced structurally, not by policy. ATL, reflexes, and Cerebellum payloads are reserved for 1.1; the 1.0 bundle is NAc + EC only because those are the components with merge math.

## The workflow

```
run a session  ──►  maxim substrate export  ──►  share the .zip
                                                       │
                                                       ▼
live system  ◄──  nac_merge / ec_merge  ◄──  maxim substrate import
```

The CLI handles the round-trip of bytes; the Python merge utilities handle folding the extracted state into a live system. The two halves are deliberately separate -- see [Safety semantics](#safety-semantics) below.

## Exporting a bundle

`maxim substrate export` reads a session's persisted `aut_nac.json` and `aut_ec.json`, applies the identity filter, and writes a ZIP.

```bash
maxim substrate export my-combat-substrate.zip \
  --session 20260408_004219 \
  --contributor-id alice-mac-mini \
  --domain combat
```

| Argument | Required | Description |
|---|---|---|
| `output` (positional) | yes | Path to write the `.zip` bundle to. |
| `--session` | yes | A session ID (resolved under `~/.maxim/sessions/{id}/`) **or** a path to a directory containing `aut_nac.json` / `aut_ec.json`. |
| `--contributor-id` | yes | Opaque ID identifying this Maxim. Must NOT start with `_` (reserved namespace -- see below). |
| `--domain` | no | Substrate-domain tag scoping the bundle (e.g. `combat`, `cooking`). Default: undomained. |
| `--no-identity-filter` | no | Skip the identity-bearing-pattern quarantine. For trusted-internal backups only. |
| `--identity-threshold` | no | Identity heuristic threshold (default `2` -- bundle-stricter than the per-call heuristic default of 1). |

On success it prints a short summary:

```
composed bundle at /path/to/my-combat-substrate.zip
  contributor: alice-mac-mini
  domain:      combat
  slices:      2
  identity_filter: True
```

The `--session` argument accepts either form. A bare ID like `20260408_004219` resolves under `~/.maxim/sessions/`. A path -- absolute or relative -- to any directory holding `aut_nac.json` and/or `aut_ec.json` works for substrate persisted outside `~/.maxim/`.

## Importing a bundle

`maxim substrate import` **extracts** a bundle to a directory. It does NOT merge.

```bash
maxim substrate import my-combat-substrate.zip --output-dir ./imported/
```

| Argument | Required | Description |
|---|---|---|
| `input` (positional) | yes | Path to the `.zip` bundle. |
| `--output-dir` | yes | Directory to extract the bundle into (created if absent). |

Output:

```
extracted bundle to /path/to/imported
  contributor: alice-mac-mini
  domain:      combat
  schema_version: 1
  slices:      ['ec', 'nac']
(use maxim.hivemind.nac_merge / ec_merge to merge into a live system)
```

After extraction you have `manifest.json`, `nac.json`, and `ec.json` (whichever were present) on disk. The next step -- merging into a live system -- is a deliberate, explicit Python call.

## Inspecting a bundle

To read a bundle's manifest without extracting anything:

```bash
maxim substrate inspect my-combat-substrate.zip
```

This prints the manifest as JSON -- contributor, domain, schema version, which slices are present, and the (reserved-null at 1.0) signature fields. Useful for checking a bundle's provenance before you trust its contents.

## Merging into a live system

The CLI extracts; the Python utilities in `maxim.hivemind` merge. Both are pure functions -- they take state dicts and return fresh dicts, never mutating their inputs.

```python
import json
from maxim.hivemind import nac_merge, ec_merge

# Your live system's current state.
local_nac = my_nac.dump()
local_ec = json.loads(open("local_ec.json").read())["substrate_nodes"]

# The extracted bundle's state.
imported_nac = json.loads(open("imported/nac.json").read())
imported_ec = json.loads(open("imported/ec.json").read())["substrate_nodes"]

# Merge — every call requires both contributor sources.
merged_nac = nac_merge(
    local_nac, imported_nac,
    left_source="local", right_source="alice-mac-mini",
)
merged_ec = ec_merge(
    local_ec, imported_ec,
    left_source="local", right_source="alice-mac-mini",
)

# Load the merged state back into the live system.
my_nac.load_state(merged_nac)
```

`nac_merge` consumes `NAc.dump()`-shape dicts and produces a dict that loads cleanly via `NAc.load_state()`. `ec_merge` consumes the `substrate_nodes` slice of `EC.save()`'s payload.

### How merging works

- **Zero prior for unobserved entries.** A contributor that never observed a `(agent, key)` pair contributes no evidence -- its absence neither boosts nor decays the merged value.
- **Causal links aggregate Bayesian-style.** Shared links combine observation counts and confidence; `predicted_value` is an observation-weighted mean. Multi-source confirmation ends up at least as confident as the more-confident single observer.
- **Opposite valences stay separate.** When two contributors disagree about an event's outcome (POSITIVE vs NEGATIVE), each side's link is preserved -- nothing is collapsed. The `contributors` tuple on each link records who voted which way.
- **EC centroids merge by `count`-weighted mean.** A right-side node merges into the best-matching left-side node of the same modality when their cosine similarity meets the threshold (default `0.44`, matching `ECConfig.pattern_complete_threshold`); otherwise it's inserted as a new node.

### Reserved contributor namespace

Contributor IDs in the `_*` namespace are rejected at merge time. `nac_merge`, `ec_merge`, and the bundle composer all route their source/contributor IDs through one validator that rejects non-strings, empty strings, and any ID starting with `_`. This protects the internal sentinels:

- `_consensus` -- marks a link or node aggregated across two or more contributors. When a merged entry has multiple contributors, its `source` becomes this marker and its `contributors` field carries the full order-preserving union.
- `_identity` -- the reserved domain marker for EC nodes the operator wants quarantined from every bundle.

Pick a plain, opaque ID for your Maxim (`alice-mac-mini`, `oasis-abc`, `local`). Never start it with an underscore.

### Frozen-modality handling

`ec_merge` respects EC's frozen-centroid modalities (default: `interoception`). For nodes in a frozen modality the merge accumulates counts and unions contributors but does **not** update the centroid embedding. This preserves the bio-fidelity invariant from the EC centroid-drift fix: a running-mean centroid update across contributors would re-introduce the very drift the frozen-modality contract was built to prevent. Interoceptive embeddings track smooth drive drift, so their prototypes stay frozen.

## The identity filter

Before a bundle ships, the composer quarantines patterns that look identity-bearing -- references to specific named people, places, or unique named roles that may carry PII or impersonation risk. This is a substrate-only heuristic (`maxim.hivemind.identity`), not a full entity-graph walk; it scans label strings for two surface signals:

- **Proper-noun shape** -- a capitalized, multi-character token that isn't a common stop-word.
- **Identity keywords** -- honorifics (`Dr`, `Captain`), ownership markers (`my`, `owner`), and name-introducers (`named`, `called`, `aka`).

When the count of either signal meets the threshold, the label is flagged and dropped. NAc event signatures that trip the filter are removed from the bundle; EC nodes marked with the `_identity` domain are dropped unconditionally.

The bundle composer's default threshold is `2` -- deliberately stricter than the heuristic's own default of `1`. Game and fantasy substrates over-flag at threshold 1: every sentence-cased creature token (`"Dragon roared"`, `"Goblin fled"`) trips the proper-noun signal even though these are generic templates you probably want to share. The conservative tilt means false positives (a generic pattern stays local unnecessarily) are preferred over false negatives (an identity-bearing pattern leaks). Disable the filter with `--no-identity-filter` only for trusted-internal backups.

## Safety semantics

These three properties are load-bearing -- they are why substrate sharing is safe to use today.

1. **`import` extracts only.** The 1.0 `import` verb writes the bundle's files to disk and stops. It never auto-merges into a live system. Merging is a separate, explicit `nac_merge` / `ec_merge` call you make on purpose. This keeps the CLI side-effect-free at the bio-stack layer -- importing a bundle can never silently mutate your agent's learned behavior.

2. **Episodes are never bundled.** Hippocampus episodes have no code path into the bundle composer. The bundle carries distilled substrate only.

3. **ZIP-slip protection.** Every entry in an imported bundle is routed through a path-safety check before any file is written. Absolute paths, `..` traversal, and symlink escape are all rejected. A malicious bundle with one safe slice and one escape slice writes nothing -- the safety pass runs before any disk write. This matters because the 1.2 P2P protocol will exchange bundles between peers, so the threat surface is real even before import is widely used.

The manifest also reserves `signature` and `signature_algorithm` slots. At 1.0 they are always `None` -- the slot exists so 1.1+ verification can land without breaking 1.0 bundles. This build computes no signatures and validates none; if you need signing today, build your own ZIP with a populated signature field and a custom verifier.

## See also

- [Maxim Hivemind + Oasis](../hivemind.md) -- the design rationale and the 1.1/1.2 roadmap this format feeds
- [Memory User Guide](memory-user-guide.md) -- how the substrate is built in the first place
- [Concept Decomposition](concept-decomposition.md) -- finer-grained EC nodes make for cleaner merges
