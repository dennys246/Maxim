- **Importing one Maxim module no longer imports its whole package, and `profiles.yml` is read on first use, not at
  import** (#1315 step 0a, V9; design: `docs/plans/atonia_gate_approach_notes.md` §1A.1).
  - `maxim.runtime`, `maxim.agents`, `maxim.tools` and `maxim.utils` resolve their public names on first access
    (PEP 562). `import maxim.runtime.executor` now loads about 130 modules instead of about 830. `from maxim.runtime
    import X` and every other public name keep working. One break: reaching a submodule as an attribute of a
    package you only imported (`import maxim.runtime` then `maxim.runtime.executor.X`) now raises
    `AttributeError`. Import the submodule itself (`import maxim.runtime.executor`).
  - `~/.config/maxim/profiles.yml` is merged when a profile is first looked up, not when `models/language/config.py`
    is imported. A malformed file still raises `ConfigurationError`, now on every lookup instead of once at import,
    so a caller that swallows the first error cannot silently lose your profiles for the rest of the run. Every run
    still refuses it up front, before choosing a model: every `maxim` run mode stops with `UserProfilesInvalid`,
    naming the file and the fix (exit 2; `maxim roy` records it as a priming abort and exits 1), and the API and
    `create` raise it. Bare `maxim` stops before its menu: run `maxim doctor` directly. The repair commands
    (`maxim doctor`, `config`, `model`, ...) still run, and `maxim doctor` now reports the bad file instead of
    crashing.
  - New profile accessors: `get_profile`, `iter_profiles` and `profile_aliases` in `maxim.models.language.config`,
    and `maxim.models.download.llm_models()`. `maxim.models.LLM_MODELS` stays importable and now includes your
    `profiles.yml` downloads. Every internal reader moved to these. `maxim --list-models` output is unchanged.
  - An executor built without a pain bridge no longer imports `maxim.simulation` on every dispatch.
