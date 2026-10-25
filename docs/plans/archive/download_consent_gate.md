# The model-download consent gate (#1343): approach note

> **ARCHIVED (2026-10-24 plans archive):** #1343 shipped (PR #1400, 2026-10-20). The rule now lives in docs/agents/llm-routing.md §3.

**Status:** approach note for the adversarial design pass (gate rule, CLAUDE.md "Faster without lowering the bar"). Session H, 1.3.3. Owner decision 2026-10-19: H takes the hunks in Session V's `models/download.py` and `models/language/llama_backend.py`; V's item 10 (offline) builds on this one gate.

## The defect

`models/download.py::ensure_available` is the consent gate for a model download. It returns at once if the file is on disk, refuses an unregistered profile, runs the storage preflight, then asks for consent (`auto=` argument, else `MAXIM_AUTO_DOWNLOAD_MODELS`, which `--auto-download` and `llm.auto_download` both set, else a y/N prompt at a tty with a 30 s timeout, else a refusal) and downloads under the advisory lock.

Two internal paths skip it and call `download_llm(profile)` directly:

1. **The CLI agentic path** (`cli.py`, the "model not found" branch after the router is built): a multi-GB GGUF with no consent.
2. **`models/language/llama_backend.py::_LlamaCppBackend._ensure`**, the backend's lazy load. It is reached by EVERY router whose local file is missing (CLI, `maxim.run`, the console, sims), so it is the wider hole. It also takes no lock.

## The fix (folded after the design pass, 2026-10-19)

- **The backend never prompts.** `_LlamaCppBackend._ensure` runs on the `LLMWarmup` thread and inside every `complete()`, while the interactive stdin reader or the agent loop owns stdin; a y/N read there could take the user's reply to the agent ("yes") as consent. So it calls `ensure_available(profile, interactive=False, ...)`, which honours only a standing grant, and remembers a refusal per backend instance, so later calls fail fast instead of asking again under the backend lock.
- **The CLI agentic path asks**, on the main thread, through the new `cli.py::_ensure_agentic_model`. Its router comes from `ExecAgent._ensure_router` (`LLMRouter(load_llm_config())`), which never ran `build_primary_router`'s preflight, so this is a real second path. It is the one prompting site there; the backend's lazy load behind it does not prompt again.
- Both treat `False` as "no local model", as they treated a failed download before.
- **`download_llm` keeps its signature.** It is public API (`maxim.models.download_llm`), and calling it IS the consent on the two explicit-request paths: `api.py::download_model` (the `maxim.download_model` verb) and `python -m maxim.models.download --llm`. A required consent keyword would break pymaxim callers for no gain.
- **The guard is an allow-list, the `config_writer` / `write_mesh_config` pattern:** a test scans `src/maxim/` (AST, every call form: `download_llm(...)`, `module.download_llm(...)`, an alias) and fails on any caller outside `{models/download.py::ensure_available, models/download.py::main, api.py::download_model}`. A new internal auto-download is then a test failure, not a silent bypass. Red on the pre-fix tree (two violations). Widened by the design pass: `download_file` (the same download one layer down) has no caller outside `download.py`, and nothing in `src/` calls the explicit-request paths (`download_model`, `download.main`), which would be an auto-download wearing the user's consent.
- Behavioural red gates: the backend's lazy load with a missing file, a registered profile, no consent and no tty never calls `download_llm`; the CLI branch, extracted into a module-level helper (rule T: `_main_impl` shrinks by the lines moved), the same.

## Inputs the gate consults, and how each could lie

| Input | How it could lie | Answer |
|---|---|---|
| `MAXIM_AUTO_DOWNLOAD_MODELS` | Leaked from a parent process or a test; `--auto-download` sets it process-wide, so sub-sims inherit it | Inheritance is the operator's own grant for the process tree (existing behaviour, unchanged). Tests: the existing conftest scrub, verified to cover it |
| `llm.auto_download` in config.json | Never seen: the gate read only the env var, which config reached only through `build_primary_router` (review round, Executor + Wire) | The gate resolves it itself: env first, then `resolve_setting("llm.auto_download")`; an unreadable config.json grants nothing |
| `sys.stdin.isatty()` | A pty with nobody watching (CI runner with a tty) | The 30 s timeout refuses; unchanged |
| The prompt answer | A line the user typed to the agent, read by a prompt on another thread | Only the main-thread startup sites prompt; the backend passes `interactive=False` |
| `LLM_MODELS` registry | A user profile with no size | The preflight's existing guard (`size_gb` None → 0.0); unchanged |
| The caller | A future internal path calls `download_llm` directly | The allow-list test |
| An aliased import (`from ... import download_llm as dl`) | Escapes a name grep | The AST scan resolves import aliases; `getattr(module, "download_llm")` and `importlib` are a stated blind spot |
| Offline (V's item 10) | — | Out of scope; V adds the offline refusal INSIDE `ensure_available`, so both sites inherit it |

## Not in scope

`ensure_tts_voice` (its own consent rule, #1268 F1); the Whisper/vision/encoder downloads and `transformers_backend.py`'s `from_pretrained` / `snapshot_download` of HF LLM weights (item 10's inventory: this gate covers the GGUF registry, not every LLM download); and a download progress UI. Fail-closed follow-ups from the design pass: the "Another maxim process" message on a same-process lock contention, and the `LLMRouter(...)` builds that reach the backend with no preflight (the CLI's foundry and curate paths, `agents/llm_agent.py`, the benchmark paper writer, the orchestrator / research / robot-runtime fallbacks), which now refuse without a standing grant instead of downloading; whether the terminal ones should ask on their main thread is the follow-up's call. Also: `api.py::download_model` (Session V's file) could call `ensure_available(name, auto=True)` to gain the lock and the storage preflight, shrinking the allow-list to the gate and `main`.
