- `maxim --foundry` and `--auto-curate` with `--llm` now ask before downloading a missing local model, as the
  agentic path does (#1396). Before, their router reached the backend's lazy load, which never prompts, so at a
  terminal the run fell back to templates without asking. Declined, or not at a terminal with no
  `--auto-download` grant, they say the model is not available and use the template fallback, with no warmup.
