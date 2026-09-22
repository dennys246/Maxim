# Config extensibility — what stays closed, and what opens

**Status:** Drafted 2026-09-22. Not scheduled; Phase 1 is ready to sequence behind the
memory-strength line, Phase 2 wants the `#856` format-version decision first.
**Motivating question:** do the config restrictions we keep adding make Maxim unextendable by
anyone who is not the author?
**Adjacent:** [#856](https://github.com/dennys246/Maxim/issues/856) (config-format downgrade
hazard) · [deferred/bio_system_plugin_plan.md](deferred/bio_system_plugin_plan.md) (the same
question one layer up: bio-systems, not settings).

## Motivation

Every phase of the memory-strength line has added a closed set. Phase 2c-1 added
`memory.strategy`, validated in `__post_init__` against a three-name frozenset, because a typo
(`strenght`) used to score silently as the default — a real defect, caught in review. The
validation is right. But the *shape* of that validation — a hard-coded literal set in core —
means a community strategy cannot be selected even when it is correctly implemented against the
public `MemoryStrategy` ABC.

That pattern will repeat: every future mechanism with a selectable variant gets a frozenset in
core, and each one is another thing a fork is required for.

The concern is not hypothetical strictness. It is that **"unknown to core" and "unknown to
everyone" are currently the same check**, and only the second one is a typo.

## What is actually closed today

Four separate restrictions, in `runtime/config_loader.py` unless noted. They are not equally
justified and should not be relaxed as a group.

| # | Restriction | Site | Verdict |
|---|---|---|---|
| 1 | `resolve_setting` raises on any field path not in `_FIELD_TO_ENV` | `config_loader.py::resolve_setting` | **Blocks extensions.** A third party cannot add a setting at all. |
| 2 | Unknown top-level keys refused unless the file declares a future minor | `config_loader.py::_parse_config_dict` | **Blocks extensions** *and* is the #856 downgrade hazard. |
| 3 | Closed value enums (`memory.strategy`, roles, lane tiers) | `MemoryConfigSection.__post_init__`, `_coerce_role`, … | **Blocks extensions** where a real ABC exists behind the name. |
| 4 | Unknown keys *inside* a known section | `config_loader.py::_parse_typed_section` | **Keep.** This is the typo check; nothing extends a core section. |

Two precedents already point the way. `LaneTierConfig` takes CC3 path (a) — unknown keys land in
an `extra` dict rather than raising (`config_loader.py::_parse_lane_tier`). And robot controllers
are already discoverable by third parties through the `maxim.robots` entry-point group
(`hardware/registry.py::RobotRegistry._discover_entry_point_plugins`). The machinery for both
halves of this plan exists in the codebase; neither needs inventing.

## Scope pressure

*Does this need to be its own mechanism?* **No, and it should not be.** Existing infrastructure
covers it:

- Discovery rides on the **`maxim.robots` entry-point pattern**, copied shape-for-shape into a
  second group. No plugin framework, no `BioSystem` Protocol dependency (that is the deferred
  bio-system plan's problem, and this plan must not grow into it).
- Tolerated config keys ride on the **CC3 path-(a) `extra` dict** already used by lane tiers.

What this plan adds is a naming convention and two small registries, not a platform layer.

## The rule this plan installs

> Strictness exists so a typo cannot silently degrade behaviour. A registry preserves that
> guarantee exactly: a name nobody registered still raises. What it drops is the requirement that
> the registrar be us.

Concretely, validation moves from `name in FROZENSET` to `name in builtins | registered`, and the
error message lists both sets so the failure says whether the plugin is missing or the name is
misspelled.

## Phase 1 — a strategy registry

**Delivers:** a community retention strategy selectable by name from config, with no core edit.

- `memory/strategies.py` gains `register_memory_strategy(name, factory)` and a
  `maxim.memory_strategies` entry-point group discovered once, mirroring the robots registry
  (including its "a plugin that fails to load logs and is skipped" behaviour — but see the
  hard constraint below, which differs).
- `MemoryConfigSection.__post_init__` validates against built-ins ∪ registry.
- `Hippocampus._get_memory_strategy` and `ATL._get_memory_strategy` resolve through the registry
  instead of an `if/elif` chain. Both keep the `TemporalAwareStrategy` wrap when SCN is connected,
  so a plugin strategy inherits temporal awareness for free.
- Built-in names are reserved: registering `access_based` raises rather than shadowing it.

**Hard constraint — a missing plugin is loud.** The robots registry skips a plugin that will not
load, which is right for an optional robot and wrong here: config *names* the strategy, so a
silent skip is a fall-back to the default under a different name — the exact defect 2c-1 fixed in
the selector's `else` branch. An installed-but-unloadable strategy that config names must raise.

**Behaviour tier:** invariant. Default unchanged, byte-identical when nothing is registered.
**Regression guard:** a test that a registered strategy is selectable end-to-end from a written
`config.json`, a test that a named-but-unregistered strategy raises with both sets in the message,
and a test that registering a built-in name raises.

**Sequencing:** after Phase 2c-3 lands `strength`. Doing it before means rewriting the same
selector twice.

## Phase 2 — an extension namespace for settings

**Delivers:** third-party settings that survive a `maxim config set` round trip and do not break
core commands.

- One new top-level key, `extensions`, holding free-form per-namespace objects:
  `{"extensions": {"acme": {"widget_threshold": 0.4}}}`. Core never interprets the contents.
- `resolve_setting("extensions.<ns>.<key>")` resolves through the same four-layer precedence, with
  env `MAXIM_EXT_<NS>_<KEY>`, instead of raising restriction #1.
- Unknown keys *inside* `extensions.<ns>` are tolerated by construction. Unknown keys inside
  `llm`, `memory` and every other core section still raise. Restriction #2 keeps refusing unknown
  *top-level* keys, so `consoel` is still caught.
- The config writer must round-trip the namespace untouched — the failure mode to guard is 2c-1's:
  a value written and then dropped by the next write.

**Open question (decide before building):** typing. The namespace is `dict[str, Any]`, so a
plugin's own values get no coercion — an env var arrives as a string and a JSON value does not.
Either publish that asymmetry plainly, or give registrants a way to declare coercion. The lane
tier `extra` precedent says JSON-serializable values, no schema; simplest is to match it and
document the string-from-env edge.

**Behaviour tier:** invariant.
**Regression guard:** a write→load→write round trip that preserves an unknown namespace verbatim,
plus the existing parse-walk AST test extended to cover the new section.

## Phase 3 — pair with the format-version decision (#856)

Phase 2 adds a top-level key, which is precisely the #856 hazard: an older build reading a newer
file refuses the unknown key and every `maxim` command fails until the file is hand-edited. The
`extensions` key must not ship before that decision, or it ships the bug it is meant to relieve.

Not re-litigated here — #856 owns it. This plan only records the dependency.

## Security notes

Installing a package already executes its code, so an entry point does not create a new trust
boundary — it is discovered from what is installed, not from a path or a config value. That
bounds the exposure, but three constraints still hold:

1. **No filesystem discovery.** No "load the class named in config.json by import path". Config
   names a *registered* name; nothing more.
2. **No shadowing.** A plugin cannot re-register a built-in name, so `access_based` always means
   what the docs say.
3. **Registration is explicit.** In-process registration happens at a call site; there is no
   implicit scan beyond the entry-point group.

The `extensions` namespace carries a fourth: core must never read from it. The moment a core
decision branches on a plugin's value, plugin data is in the trusted path.

## What stays strict — non-negotiable

- Unknown keys inside a known section raise. (Restriction #4, the typo check.)
- A typo in a core field path raises. (Restriction #1 outside the `extensions.*` prefix.)
- A name nobody registered raises, and never falls back to the default.
- Persistence never silently substitutes. A config naming a strategy that is not present is an
  error, not a downgrade to `access_based`.

## Not in scope

- Bio-system plugins ([deferred/bio_system_plugin_plan.md](deferred/bio_system_plugin_plan.md))
  — same question, much larger surface, and it needs the `BioSystem` Protocol first.
- Opening the other closed enums (roles, lane tiers). They name infrastructure, not an ABC a third
  party can implement; open them on demand, with a demonstrated implementer.
- A general plugin lifecycle (ordering, dependencies, health). If Phase 1 needs any of it, that is
  a signal the bio-system plan is the right vehicle and this one is overreaching.
