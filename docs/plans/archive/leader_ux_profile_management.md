# Leader UX — Profile Management for Custom / Larger Models

**Status:** Drafted 2026-05-31. Ships as ONE PR with three folded stages (L1 bundled-profile expansion + L2 user YAML profiles + L3 `maxim model` CLI verbs). Scope decision made by the user in the originating setup conversation: cleaner history > faster unblock. Worktree: `Maxim-wt-leader-ux-profile-management`. Branch: `feat/v1-leader-ux-profile-management`.

**Triggered by:** the first real-user attempt to set up a Mac Mini as a *leader* (not a peer) hit a structural gap. Bundled `_BUILTIN_PROFILES` in [src/maxim/models/language/config.py](../../src/maxim/models/language/config.py) tops out at `qwen2.5-14b-instruct`. Anything larger requires either editing the source dict or running `python -m llama_cpp.server` manually and bypassing Maxim's auto-spawn — the "janky" path the user explicitly objected to. With 48 GB Apple Silicon Macs and multi-GPU consumer rigs becoming common, this is no longer an edge case. **The Mac Mini being interesting at all is also the most likely first-touch with Maxim as a leader** — the rough edge here disproportionately shapes the on-ramp impression.

**Companion docs:** rides into 1.0 alongside the §B5 Hivemind shareability work; both are "leader-as-a-real-thing" surface. Does NOT block Hivemind PR D or the Tier 1 graduation work. Independent track per the [sequenced 1.0 plan](v1_refinement.md).

---

## Front-gate scope pressure (CLAUDE.md Principle 3)

**Question:** does this need to be its own mechanism, or can it ride on existing infrastructure?

**Existing infrastructure surveyed:**

| Candidate | Why insufficient (or sufficient) |
|---|---|
| `_BUILTIN_PROFILES` dict in [models/language/config.py](../../src/maxim/models/language/config.py) | **Rides on it.** L1 adds entries. L2 merges user entries into the same dict at startup. No new dispatch surface — the rest of the runtime sees one profile table. |
| `_GGUF_MODELS` download URL table in [models/download.py](../../src/maxim/models/download.py) | **Rides on it.** L1 adds entries. L2's `download:` block constructs URLs in the same shape. |
| `maxim peer` / `maxim tunnel` / `maxim doctor` subcommand patterns | **Rides on it.** L3's `maxim model` is the fourth subcommand, same dispatch shape as the existing three. |
| Existing `--llm <name>` flag + `MAXIM_LLM_PROFILE` env var | **Rides on it.** L2/L3 don't introduce a new model-selection path; existing flag continues to resolve through the merged profile table. |
| `~/.config/maxim/peer.yml` + `mesh.yml` declarative-config layer | **Rides on it.** `profiles.yml` joins these two as the third declarative config file. Lives at `~/.config/maxim/profiles.yml`, parsed by PyYAML (already a core dep), follows the same "operator edits by hand, runtime reads only" contract per CLAUDE.md's mesh.yml invariant. |
| The hand-rolled `mesh.yml` parser at [peer/mesh_config.py](../../src/maxim/peer/mesh_config.py) | **Cannot ride on it.** CLAUDE.md explicitly forbids extending the mesh parser. Profile loading uses PyYAML directly — supports nested structures the mesh parser deliberately refuses. |

**Verdict:** could-ride-on-existing for L1 (pure additive expansion of an existing data table). New mechanism for L2 + L3, but minimal: one new declarative config file + one new CLI subcommand. The schema is the only frozen-at-1.0 surface.

**Specific reason this earns new mechanism:** there is no existing user-extensibility point for the profile registry. Every other config layer in the project (`peer.yml`, `mesh.yml`, `~/.maxim/util/*`) handles either a single peer/leader pairing or runtime mutable state, not the model catalog. Adding bundled profiles forever (the alternative) scales poorly and pushes every "I want X model" interaction through a PR. The YAML + CLI verb pair is the smallest mechanism that closes the gap for any GGUF model on either backend (llama-cpp or torch).

---

## The three stages, folded into one PR

| Stage | What lands | LOC est. | Frozen at 1.0? |
|---|---|---|---|
| **L1** | 3 new bundled profiles: `qwen2.5-32b-instruct`, `llama-3.1-70b-instruct` (Q4_K_M), `mixtral-8x7b-instruct` | ~60 LOC + tests | No — additive, future minor versions can add more |
| **L2** | `~/.config/maxim/profiles.yml` loader; merges user profiles into `_BUILTIN_PROFILES` at startup with user-wins precedence | ~120 LOC + tests | **Yes — schema** (see below) |
| **L3** | `maxim model {add,remove,list}` CLI verbs; wraps L2's YAML mutation with a CLI surface; chat-format auto-inference | ~200 LOC + tests + docs | Partially — CLI verb names + flag names are stable; auto-inference table is additive |

Total: ~380 LOC + ~150 LOC tests + docs updates in [llm-setup.md](../../user/llm-setup.md), [peer-setup.md](../../user/peer-setup.md), [getting-started.md](../../user/getting-started.md). Roughly 1-2 days plus review.

---

## L1 — Bundled profile expansion

**Goal:** the three most common "I bought a Mac with unified memory for this" or "I have a multi-GPU rig" choices ship in the box.

**Profiles to add** (HF repo verification + arch field values per the existing pattern):

| Profile slug | HF repo | File | n_ctx | Weights (Q4_K_M) | Practical fit |
|---|---|---|---|---|---|
| `qwen2.5-32b-instruct` | `bartowski/Qwen2.5-32B-Instruct-GGUF` | `Qwen2.5-32B-Instruct-Q4_K_M.gguf` | 32768 | ~20 GB | 48 GB Mac comfortable; 24 GB VRAM tight |
| `llama-3.1-70b-instruct` | `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF` | `Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf` | 32768 | ~42 GB | 64 GB Mac comfortable; 48 GB Mac borderline |
| `mixtral-8x7b-instruct` | `bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF` | `Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf` | 32768 | ~26 GB | 48 GB Mac comfortable; multi-GPU friendly |

**Aliases:** `qwen2.5-32b` → `qwen2.5-32b-instruct`; `qwen32b`; `llama-3.1-70b`, `llama3.1-70b`, `llama70b` → `llama-3.1-70b-instruct`; `mixtral`, `mixtral-8x7b` → `mixtral-8x7b-instruct`.

**Arch fields:** verified against the HuggingFace model cards before merge. Required for the VRAM estimation path in [runtime/lane_models.py::estimate_max_ctx](../../src/maxim/runtime/lane_models.py).

**Lives at:** [src/maxim/models/language/config.py](../../src/maxim/models/language/config.py) (new profile entries + aliases) + [src/maxim/models/download.py](../../src/maxim/models/download.py) (URL entries).

**Pass criterion:** `maxim --list-models` shows the three new profiles; `maxim --llm qwen2.5-32b --auto-download` resolves the HF URL and starts a download.

---

## L2 — User profile YAML

**Goal:** any GGUF on HuggingFace (or any local path) becomes a Maxim profile via a hand-edited config file.

### Schema (FROZEN at 1.0)

**File location:** `~/.config/maxim/profiles.yml` (declarative config layer, same dir as `peer.yml` / `mesh.yml`).

```yaml
# ~/.config/maxim/profiles.yml
#
# User-defined Maxim model profiles. Merged into _BUILTIN_PROFILES at startup.
# User profiles WIN on name collision with built-ins (loudly logged at startup).

profiles:
  my-qwen-32b-bigger-quant:
    # REQUIRED
    backend: llama_cpp                  # one of: llama_cpp, pytorch
    prompt_style: chatml                # one of: chatml, mistral_instruct,
                                        # llama3_instruct, llama2_chat, phi3,
                                        # phi, gemma
    # ONE OF download: OR local_path: REQUIRED
    download:
      hf_repo: bartowski/Qwen2.5-32B-Instruct-GGUF
      hf_file: Qwen2.5-32B-Instruct-Q5_K_M.gguf
    # local_path: ~/models/my-custom.gguf

    # OPTIONAL
    n_ctx: 32768                        # default 8192 if omitted
    stop: ["<|im_end|>", "<|endoftext|>"]  # default per prompt_style
    aliases: [qwen-bigger, qwen32-q5]   # default empty
    arch:                               # optional but enables VRAM estimation
      n_layers: 64
      n_kv_heads: 8
      head_dim: 128
      kv_type_bytes: 2
      weights_gb: 23.0
```

**Field semantics:**

| Field | Required? | Default | Purpose |
|---|---|---|---|
| `backend` | Yes | — | Dispatch to llama-cpp-server or PyTorch |
| `prompt_style` | Yes | — | Chat template selection |
| `download` XOR `local_path` | Yes (exactly one) | — | Where the model file comes from |
| `n_ctx` | No | 8192 | Context window |
| `stop` | No | per prompt_style | Stop tokens |
| `aliases` | No | `[]` | CLI shorthand names |
| `arch` | No | `None` (coarse VRAM est.) | Per-layer arch for `estimate_max_ctx` |
| `model` | Auto-derived from profile key | — | Internal slug (NOT user-facing field) |
| `model_base` | Auto-derived from `hf_repo` | — | HF basename (NOT user-facing field) |

**`model:` and `model_base:` are deliberately auto-derived, not user-facing fields,** to keep the schema small. The internal runtime still needs them — the loader fills them in from the profile key + `hf_repo` respectively.

**Schema versioning:** the file does NOT carry `_format_version` at 1.0 (per CC1 the convention applies to files Maxim *writes*; this is user-authored). Future schema changes either add optional fields with defaults (non-breaking) or introduce a top-level `version: 2` field (breaking — requires migration). The 1.0 schema described above is the implicit `version: 1`.

### Merge precedence

| Collision case | Resolution |
|---|---|
| User profile slug = built-in slug | User wins. Startup logs WARNING: `User profile 'qwen2.5-14b-instruct' overrides bundled profile.` |
| User alias = built-in alias | User wins. Same WARNING shape. |
| User alias = different built-in slug | User alias wins (resolves to the user profile). WARNING. |
| Two user profiles claim the same alias | `ConfigurationError` at startup (operator must resolve). |

**Why user-wins:** the user is opting in by writing the file. Falling back to built-in silently would mask the operator's intent (the same silent-no-op pattern the CLAUDE.md "Lessons learned" section flags repeatedly).

### Error paths

| Failure | Behavior |
|---|---|
| `profiles.yml` missing | Loader is a no-op. No warning (the empty case is the common case). |
| `profiles.yml` exists but is empty / null | Loader is a no-op. No warning. |
| YAML syntax error | `ConfigurationError` at startup, with file path + line number from PyYAML. |
| Required field missing | `ConfigurationError` listing the missing field + the profile slug it belongs to. |
| Both `download:` and `local_path:` set | `ConfigurationError` ("specify one, not both"). |
| Neither set | `ConfigurationError`. |
| `local_path:` points at non-existent file | WARNING at load time; `ConfigurationError` only on first inference attempt (so users can pre-stage config before model download). |
| `prompt_style` not in the allowed enum | `ConfigurationError` listing valid values. |
| `backend` not in `{llama_cpp, pytorch}` | `ConfigurationError`. |

### Regression guards

- **Schema-shape test:** `tests/unit/test_user_profiles.py::test_minimum_valid_profile` — loads a profile with only required fields, verifies it merges into the resolved dict with all auto-derived fields populated.
- **Collision-precedence test:** `test_user_profile_overrides_builtin_with_warning` — user profile shadows `qwen2.5-14b-instruct`; assertion that the user version wins AND the warning is emitted.
- **Bad-schema tests:** parametrized cases for each `ConfigurationError` path in the table above.
- **Missing-file test:** `local_path:` to a fake path → load succeeds, first-inference fails cleanly.
- **CI grep:** `grep -nE "yaml.load\b" src/maxim/models/language/profile_loader.py` must return zero matches (PyYAML's `yaml.safe_load` is the only allowed entry point — `yaml.load` is unsafe).

### Lives at

- New module: [src/maxim/models/language/profile_loader.py](../../src/maxim/models/language/profile_loader.py) (loader + merge logic)
- Edit: [src/maxim/models/language/config.py](../../src/maxim/models/language/config.py) — call loader once at module import, after `_BUILTIN_PROFILES` is defined
- Edit: [src/maxim/doctor/checks.py](../../src/maxim/doctor/checks.py) — surface user profiles in tier-detection output with `(user)` suffix
- Edit: `maxim --list-models` output — section header `User Profiles:` after built-in list, with source markers

---

## L3 — `maxim model` CLI verbs

**Goal:** users don't have to write YAML to add a profile.

### CLI surface

```bash
# Add a profile from HF
maxim model add qwen-32b-q5 \
    --hf bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q5_K_M.gguf \
    --chat-format chatml \
    --n-ctx 32768 \
    --alias qwen32q5

# Add a profile from a local file
maxim model add my-local --local ~/models/custom.gguf --chat-format llama3

# Remove
maxim model remove qwen-32b-q5

# List with source markers
maxim model list
# → mirrors `maxim --list-models` output but scoped to profiles only.
```

### Chat-format auto-inference

Substring match on the profile slug OR the HF repo basename (case-insensitive). If `--chat-format` is omitted AND inference succeeds, the inferred style is used + logged. If inference fails, `--chat-format` becomes required and the error lists valid values.

| Substring match | Inferred `prompt_style` |
|---|---|
| `qwen` | `chatml` |
| `llama-3`, `llama3` | `llama3_instruct` |
| `llama-2`, `llama2` | `llama2_chat` |
| `mistral`, `mixtral` | `mistral_instruct` |
| `phi-3`, `phi3` | `phi3` |
| `phi-2`, `phi2` | `phi` |
| `gemma` | `gemma` |
| `smollm` | (no Maxim-bundled chat template — require `--chat-format`) |

### CLI <-> YAML round-trip

`maxim model add` writes `profiles.yml`. `maxim model remove` rewrites it without the removed entry, preserving formatting + comments via `ruamel.yaml`? No — adds a new dep. **Use stdlib + atomic-write semantics:** load with PyYAML, edit the dict, write via `atomic_write_text`. This loses comments on round-trip — an explicit trade-off documented in the CLI's `--help` output ("editing via `maxim model` drops YAML comments; edit the file directly to preserve them").

### Atomic-write contract

`profiles.yml` is NOT a secret — uses `maxim.utils.atomic_io.atomic_write_text`, NOT `atomic_write_secret`. Per the CLAUDE.md mesh.yml invariant, declarative config files use plain atomic writes; only credential-bearing files (api_key, cluster_key) use `atomic_write_secret`.

### Regression guards

- **CLI shape test:** `tests/unit/test_model_cli.py::test_add_writes_yaml_then_list_finds_it` — full add → list → remove cycle.
- **Chat-format inference test:** parametrized cases over the table above + a negative case (`smollm` → error requiring `--chat-format`).
- **`--hf` parsing test:** `repo:file` format roundtrip; missing `:` separator → error; multi-`:` in filename handled correctly.
- **Refuse to overwrite without `--force`:** `add` on an existing profile slug → error + hint, mirroring `maxim peer add-node` behavior.

### Lives at

- New module: [src/maxim/cli/model.py](../../src/maxim/cli/model.py) — subcommand implementation
- Edit: [src/maxim/cli.py](../../src/maxim/cli.py) — register `model` subcommand alongside `peer` / `tunnel` / `doctor`

---

## Edge cases & non-obvious design choices

| Case | Decision | Rationale |
|---|---|---|
| User profile claims `backend: pytorch` but `transformers` not installed | Load succeeds; first inference fails with the existing torch-missing error. | Same shape as bundled pytorch profiles today. |
| User profile's HF download fails (404, auth-walled gated model) | Standard download error path. | No new failure mode introduced. |
| User edits `profiles.yml` while Maxim is running | Picked up on next process startup, NOT mid-process. | Mirrors `peer.yml` / `mesh.yml` semantics. Live reload is out of scope. |
| `arch:` field missing on user profile | VRAM estimation falls back to coarser RAM-only heuristic. WARNING in doctor. | Most users won't supply `arch:` — it's an expert escape hatch. Coarse fallback is fine for "does it fit." |
| Two leaders in a mesh disagree on profile definitions | Each leader uses its own `profiles.yml`. Peers route by `MAXIM_LANE_LARGE_REMOTE_MODEL` string — they don't share profile dicts. | Existing peer routing already works this way for built-ins. |
| `maxim model add` from a peer machine | Writes that peer's `profiles.yml`. Does NOT propagate to the leader. | Profiles are per-machine declarative config, not mesh-wide. If user wants leader profile change, `ssh leader; maxim model add ...` or future Hivemind broadcast. |
| Profile slug contains `/` or `:` (invalid for HF cache pathnames) | `add` rejects with a clear message; YAML loader rejects with same. | Defensive; avoids cache-write failures down the line. |

---

## Doctor integration

- `maxim doctor` Environment section: when user profiles are loaded, log `LLM profiles: N bundled + M user` instead of just `N`.
- Tier-detection check: if a user profile is the active large-tier pick, source marker shows in the message.
- New check: `profiles.yml` syntax check — runs PyYAML parse on the file (if present) and surfaces errors with line numbers.

---

## Two-lens review prompts (run after implementation, before PR)

### Executor lens prompt

> Review the implementation in `src/maxim/models/language/profile_loader.py` + `src/maxim/cli/model.py` + the `_BUILTIN_PROFILES` additions. Focus on: silent-no-op failure modes (does any branch swallow an error?), schema validation completeness (does every required-field error path fire?), atomic-write correctness on `profiles.yml`, CLI argument parsing edge cases (`--hf` with multi-`:` filenames, `--alias` collisions, missing required flags). Verify the chat-format auto-inference table actually matches the existing `_BUILTIN_PROFILES` `prompt_style` values. Flag any new code path that constructs a profile dict without going through the loader's validation.

### Architecture lens prompt

> Review the design against CLAUDE.md invariants. Specifically: (1) does the schema honor the CC3 frozen-dataclass discipline if Maxim ever exposes profiles as a typed dataclass? (2) does `profiles.yml` correctly sit in the `~/.config/maxim/` declarative-config layer per the mesh.yml two-layer-split invariant? (3) does the user-wins collision precedence match the project's existing silent-no-op-avoidance pattern? (4) does the loader avoid mutable-globals-and-module-extraction footguns (CLAUDE.md Lesson L4)? (5) is the auto-derivation of `model:` and `model_base:` from the profile key a leaky abstraction that bites later?

---

## What this does NOT cover (deferred)

- **Live profile reload without restart** — `peer.yml` / `mesh.yml` also require restart. Out of scope.
- **Hivemind / mesh-wide profile broadcast** — profiles stay per-machine. If §B5 grows a profile-sync surface later, that's a future plan.
- **Quantization conversion** — Maxim does NOT bundle a GGUF quantizer. Users bring already-quantized files.
- **GUI for adding profiles** — CLI only.
- **Auto-download from Ollama / LM Studio profile catalogs** — out of scope; users bring HF URLs.
- **Multi-file model support (e.g. split GGUF)** — schema only supports single-file. Defer to a future `download.files: [...]` extension.

---

## Implementation order within the single PR

1. **L1 first** — additive, lowest risk, smallest blast radius if review surfaces issues. Land in commit 1.
2. **L2 schema + loader** — frozen-at-1.0 surface, lands in commit 2. Two-lens architecture review focuses here.
3. **L3 CLI** — wraps L2, lands in commit 3. Two-lens executor review focuses here.
4. **Docs updates** — commit 4. Cross-references between `llm-setup.md`, `peer-setup.md`, `getting-started.md`, and this plan doc.
5. **Doctor integration + final sweep** — commit 5.

Single PR opened against `main` after all five commits + two-lens review folds.