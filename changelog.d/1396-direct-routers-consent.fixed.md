- **A model you named that is not on disk now stops the run up front, and the sim, research, robot and ExecAgent
  routers ask before downloading** (#1396). `--llm X`, `--aut-model X`, research's model, and
  `maxim.run()`, `maxim.create.router()`, `maxim.imagine()`, `maxim.campaign()` and `maxim.research()` (each hands its
  model, `mistral-7b` by default, to the build) used to leave a run going on a router that refused every call when X
  was missing and not downloaded (the dual-LLM AUT quietly measured the shared model instead). Now the run refuses
  with `LocalModelUnavailable` before any router warms up, before the sim's sandbox is set up and before a
  `--aut-model` loads (on a GPU host the shared llama-cpp-server may already have been started); the CLI prints how
  to get the
  model (download it, grant downloads, fix an llm.json `model_path`, or place an unregistered model by hand) and
  exits 2. `LocalModelUnavailable` is a `BackendError` and also a `ConfigurationError`, so an existing
  `except ConfigurationError` around `maxim.run()` or `maxim.create.router()` still catches it.
  A model picked for you (no name given) falls back to a smaller model you have, or runs without the LLM; when
  `MAXIM_LLM_PROFILE` names the declined model, the run goes without the LLM. A router with a `--cloud-fallback`
  (or peer) tail keeps serving from it when its local model is missing, as before. The new router check asks at a
  terminal only on the main thread and never twice for a model declined in the same startup; the existing
  startup check inside the lane build still asks wherever stdin is a terminal. `maxim.run()`, `maxim.create`, the
  console's talk and server mode keep that startup check and its named-model refusal, without the new router
  check. The benchmark's paper writer never asks. The fallback search after a declined model is now handed the model
  the run asked for, not the one just declined. A model alias (`mistral-7b`) is resolved before that check, and
  `lane_decisions.jsonl` now records the `--llm` model as the profile handed to the build for CLI sims.
