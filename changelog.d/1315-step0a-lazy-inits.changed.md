- **Importing one Maxim module no longer imports its whole package, and `profiles.yml` is read on first use, not at
  import** (#1315 step 0a, V9; design: `docs/plans/atonia_gate_approach_notes.md` §1A.1).
  - `maxim.runtime`, `maxim.agents`, `maxim.tools` and `maxim.utils` resolve their public names on first access
    (PEP 562). `import maxim.runtime.executor` now loads about 130 modules instead of about 830. `from maxim.runtime
    import X` and every other public import keep working.
  - `~/.config/maxim/profiles.yml` is merged when a profile is first looked up, not when `models/language/config.py`
    is imported. A malformed file still raises `ConfigurationError`, now on every lookup instead of once at import,
    so a caller that swallows the first error cannot silently lose your profiles for the rest of the run.
  - New profile accessors: `get_profile`, `iter_profiles` and `profile_aliases` in `maxim.models.language.config`,
    and `maxim.models.download.llm_models()`. `maxim.models.LLM_MODELS` stays importable and now includes your
    `profiles.yml` downloads. Every internal reader moved to these. `maxim --list-models` output is unchanged.
  - An executor built without a pain bridge no longer imports `maxim.simulation` on every dispatch.
