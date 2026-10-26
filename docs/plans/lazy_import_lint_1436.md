# #1436 approach note: voice input fails loud, and a lint for lazy imports of missing code (Session S, 1.3.4)

Status: ADOPTED 2026-10-25. **Read the two fold sections at the end first:** the adversarial pass and the three-lens round superseded parts of the sections above (the shipped lint is `scripts/lint_dead_imports.py`; a missing maxim module fails at import, not as a typed raise; there is no `listening_off_reason` attribute). Owner decisions, 2026-10-25:
- **Split by cause.** A maxim-internal import failure (a code defect) raises a typed error and stops the session.
  A missing `audio` extra keeps the session running, with ONE ERROR naming `pip install 'pymaxim[audio]'` and a
  recorded `listening: off (reason)` on the runtime.
- **The lint checks modules AND names** for function-level `maxim` imports in `src/` and `scripts/`.
- **Land at zero, no allowlist.**
  - Remove the dead `CommunicationBridge` wiring.
  - Delete the dead `maxim.skills` block in `agentic_runtime.py`, as a shared hunk with Session V.
  - Delete `scripts/rtsp_bridge.py`, and fix docs/user/robot-setup.md.

## The defect, and what the probe found

`db274547` (2026-04-03) deleted `maxim/data/audio/_file_based_transcription.py`, a re-export shim of
`inference/transcribe_audio.py`. Three function-level imports still name it:
- `embodied_runtime/media_loop.py`, at the transcription start inside `live()`, under a broad `except`;
- `embodied_runtime/workers.py`, twice, in the audio writer's `_flush_pending` and in its shutdown flush.

Those two only run once a transcription process exists, so fixing `media_loop.py` alone would move the crash into
the audio-save thread.

A scratch AST probe resolved all 2,035 function-level `maxim` imports in `src/` and `scripts/` against the tree. It
found 8 dead ones: #1436's 3, plus 5 more.
- `agents/maxim_agent.py::wire_communication` imports `maxim.bridges.communication_bridge` (deleted in 3c6ad81d as
  a "dead bridge"). **A live crash:** the robot path calls it with a gateway and a NAc, and the raise kills
  `_start_agentic_runtime` behind one warning.
- `embodied_runtime/agentic_runtime.py` (×3) imports `maxim.skills.*`, deleted in the 1.0 consolidation. A broad
  `except` warns "Failed to register" every run.
- `scripts/rtsp_bridge.py` imports `maxim.tools.rtsp_bridge`, deleted in c5eb49d6, so the script cannot run.

No name-level miss was found (every imported name exists where its module does).

## Fail-loud design (the split)

The transcription start is extracted from `live()` into `MediaLoopMixin._start_transcription_worker(...)`. Rule T
applies, since `live()` is pinned.
- **Code imports at the top of the helper, outside any `except`.** These are `from maxim.inference.transcribe_audio
  import watch_and_transcribe` and `load_whisper_config`. A missing maxim module raises `ImportError` straight
  through `live()`. The lint is what keeps that from ever shipping.
- **The extra:** `optional_dependency_available("faster_whisper")` runs before spawning. If it is missing, the helper
  logs one ERROR from `OptionalDependencyError`'s `fix_hint`, sets `self.listening_off_reason`, and returns no worker.
- **Other start failures** (config, spawn, or the worker exiting immediately): one ERROR, the reason recorded, and
  no worker. These stay as "keep running", because the owner's split names only the two causes above.
- **The record's reader.** No runtime status surface today shows robot listening state, so the ERROR line is the
  user-facing signal. `listening_off_reason` is the record that the device plan's step 5 Listener will own.

  Flagged for the design pass: shipping an attribute with no reader yet.
- **`workers.py`:** both imports point at `maxim.inference.transcribe_audio`.

## The lint (a gate): `scripts/lint_lazy_imports.py`

- **What it checks.**
  - Every `import maxim...` / `from maxim... import ...` (absolute, level 0) inside a function or method body in
    `src/maxim/**.py` and `scripts/**.py`.
  - The module must resolve to `src/maxim/<path>.py` or `<path>/__init__.py`.
  - Each imported name must be a submodule, or a name bound at the target module's top level: def, class,
    assignment, import, or a binding inside a top-level `if`/`try`.
  - A module that defines `__getattr__` accepts any name.
- **Why function-level only.** A top-level import is executed by importing the module, which every test does. A
  function-level one runs only when its branch does, which is how #1436 and the other five hid.
- **Runs in CI's lint job and in prepr** (which reads `jobs.lint`). Fails on any miss and prints `file:line: module
  X does not exist` or `X has no Y`.
- **Tests:** fixtures for a dead module, a dead name, a submodule, a re-export, a `__getattr__` module, a
  `TYPE_CHECKING` binding, and a relative import (ignored: level > 0).

  The guard is proven by deleting the mechanism: re-adding the shim import makes the lint fail.

## Inputs the lint consults, and how each could lie (for the adversarial pass)

1. **The tree on disk.** A stale `.pyc`, or an untracked `.py` that is not committed, would resolve locally but not
   in CI. The lint reads `.py` files only. Should it read only `git ls-files`?
2. **Top-level bindings.** A name bound only inside a function, or through `globals()[...] =` or `exec`, is missed.
   Is a name bound only under `if TYPE_CHECKING:` a false pass (it exists only for type checkers)?
3. **Relative imports** (`from .x import y`) are skipped. Do any function-level relative imports exist?
4. **Dynamic imports.** `importlib.import_module("maxim...")` with a string literal is not covered.
   Should it be?
5. **Imports guarded on purpose.** A `try: from maxim.x import y / except ImportError` that is meant to degrade
   would now fail the lint if `x` were deleted. That is correct: maxim-internal code is never optional.
6. **Re-export chains.** `from maxim.a import b`, where `a` re-exports `b` from `c`, passes if `a` binds `b` at top
   level, even when `c` no longer has `b`. In that case `a`'s own top-level import fails at import time, which tests
   catch.
7. **`scripts/` importing the installed package.** A script run from a checkout imports `src/`, so resolving
   against `src/` is right.
8. **Tests are excluded.** Should `tests/` be in scope?

## Tests for #1436 itself
- **Strict red gate first:** the real `MediaLoopMixin._start_transcription_worker` path with `faster_whisper`
  present (faked) spawns a worker. Today it warns and returns none.
- **Missing extra:** one ERROR naming the fix, `listening_off_reason` set, no raise.
- **`workers.py`:** with a stub transcription process, `_flush_pending` writes a task file instead of raising.
- **`wire_communication`:** with a gateway and a NAc, it wires the gateway and does not raise. The robot-path
  `_start_agentic_runtime` reaches "Communication gateway wired".
- **The skills block is gone:** no "Failed to register" warning for protocols.

## The adversarial design pass (2026-10-25) and its folds

There was no DO-NOT-BUILD finding. The design changes as follows; where it differs from the sections above, this
section wins.

- **Scope is every depth, not function-level only (S2).** The reason given above was false for `scripts/`: no test
  imports a script. It was also incomplete for `src/`, which has 282 conditional top-level imports. The full scope
  catches two more misses:
  - `scripts/behavioral_convergence_exp2.py` has a dead top-level `maxim.energy.reactions` import (the deleted
    `EnergyReactionBridge` family). **Owner: delete the script.**
  - `default_network/behaviors/__init__.py` has a dead `TYPE_CHECKING` import of `maxim.default_network.novelty`.
    The class now lives at `maxim.salience.ThreadSafeNoveltyTracker`.

  The lint resolves relative imports as well, and treats names bound only under `TYPE_CHECKING` as absent at
  runtime. `tests/` stays out of scope: its one hit is an intentional `pytest.raises(ModuleNotFoundError)`.
- **Why this is its own lint and not a mypy-ratchet rule (S1, front-gate):**
  - `scripts/lint_mypy_ratchet.py` only stops a file's count rising. A new `import-not-found` can be offset by a fix
    elsewhere in the same file, and grandfathered errors stay.
  - It does not cover `scripts/`.
  - The AST pass runs in under a second with no allowlist.

  The ratchet does still see `src/` misses, and stays as a second reader.
- **Hoist rather than raise out of `live()` (S3).** `maxim.inference.transcribe_audio` and
  `maxim.models.audio.transcription` import in 0.27 s and pull in no faster_whisper, ctranslate2 or torch (measured).
  The transcription imports in `media_loop.py` and both `create_task_file` sites in `workers.py` move to module top
  level. A missing maxim module then fails when the module is imported, which every test does, and the lint is the
  general guard.
- **A spec-only probe for the extra (S4).** `optional_dependency_available` imports the package, which would load
  faster_whisper, ctranslate2 and CUDA into the parent that hides its GPUs on purpose. A new
  `optional_dependency_installed` on the canonical surface uses `importlib.util.find_spec`; the fix hint comes from
  `EXTRA_FOR_IMPORT`.
- **A late child death is loud (S5).** `watch_and_transcribe` returns with code 0 when the transcriber cannot load,
  typically seconds after spawn, so the 0.1 s `is_alive()` check rarely fires. `workers.py::audio_writer_worker`
  checks `transcribe_process.is_alive()` before queueing a task. If the child is dead, it logs ONE ERROR and stops
  writing transcription tasks, so chunk files no longer pile up with no consumer.
- **Swallows (S6).** The inline nvidia-smi probe becomes the existing `maxim.utils.gpu_compat.detect_blackwell`, so
  no edited `except` handler moves.
- **Listening status (owner, 2026-10-25): the ERROR now, the record later.** The ERROR fires once per `live()` start.
  No `listening_off_reason` attribute until the device plan's step 5 Listener owns listening state, because nothing
  reads it today (N4).
- **N1:** the `wire_communication` crash needs a comms gateway, which `agentic_runtime.py` builds only when
  `MAXIM_COMMS_ENABLED` is set. It is real but opt-in, not on the default robot path. The bridge's NAc wiring goes;
  the unused parameters and the unread `MaximAgent._gateway` are #1466 (`register_promotion_source` has one other
  caller, the statistician, so it stays).
- **N2 (#1467):** deleting the protocol block leaves always-None plumbing in several places:
  - `selfy.py::_protocol_activate`/`_protocol_deactivate` and the `_protocol_registry` initializer;
  - the teardown in `_stop_agentic_runtime`;
  - the `protocol_registry=` arguments in `agent_loop.py`, `loop_setup.py` and `loop_controller.py`;
  - `memory_agent.py`'s `get_active()`, and `concept_context.py::set_skill_registry`.

  These go to a follow-up issue so Session V's shared hunk stays one block.
- **N3:** `docs/plans/deferred/tool_refinement_plan.md` links the deleted `tools/rtsp_bridge.py`; it is fixed with
  the script deletion.
- **N5 (#1468):** a session that stops on an exception still exits 0 (`cli.py`, `log_exception` then
  `break`), which is invisible to supervisors.
- **N6:** the real lint does not inherit the probe's weaknesses. It does not collect names bound inside functions
  nested in top-level `if`/`try`, and it checks a `__getattr__` module's closed literal set rather than exempting it.

## The three-lens round (2026-10-25, tree 409d8f86) and its folds

No BLOCKER beyond an unused import (Executor: ruff F401 in the flipped test). Owner decisions taken from it:
- **The transcriber runs in its own process group** (Executor SF2, Wire S1). In the terminal's group a Ctrl-C stopped
  the child before the parent noticed the stop, and the writer logged "Voice input is OFF" on every ordinary stop.
  It is spawned through `utils.audio.run_in_own_process_group` (a light module, so the child does not import the
  embodied runtime); the parent's teardown stops it. The writer also stays silent once its `stop_event` is set.
- **A per-run chunk directory** (Wire S2): `audio/chunks/<run_id>/`. The shared directory fed an earlier session's
  leftover chunks and tasks to the next transcriber as new speech.
- **Named fence exceptions:** `embodied_runtime/workers.py`, `utils/optional_deps.py`, and the two Exp 2 record notes
  in `docs/experiments/` (Session E's fence).

Folds:
- Executor SF3: a structural test pins `live()` → helper → `audio_writer_worker` and the per-run `chunk_dir` (the
  composition #1436 broke). Executor SF4: the test no longer leaks `MAXIM_TRANSCRIPTION_WORKER_CPU_ONLY`.
- Architecture SF5: a subprocess test pins that importing `media_loop`/`workers` loads no faster_whisper, ctranslate2
  or torch (`inference/transcribe_audio.py` is Session V's file; a top-level faster_whisper there would break it).
- `load_whisper_config` comes from `maxim.inference.transcribe_audio`, not the `models.audio.transcription` re-export
  shim (Wire N1, Executor N6).
- The lint: star re-exports accept any name, `AsyncWith`/`AsyncFor` and loop `else` bodies bind, and a `__getattr__`
  module counts only the strings in its top-level literal tables; fixtures for each. Dynamic `import_module` strings
  stay a recorded blind spot (mechanization backlog M80; none dead today, per two independent scans).
- Docs: `optional_deps.py`'s docstring names the fourth intent and the probe's test moved to
  `tests/unit/test_optional_deps.py`; `embodied_device.md` (R5 fixed; the file-IPC path is the interim path, replaced
  by step 5's Listener before deletion); `ci-testing.md`'s tree-wide lints row; Exp 2's records and the two READMEs
  say what it measured (the Architecture lens re-ran it at fe1124b9: 13/13); `mediaMTX.md` marked stale.
- Follow-up issues filed: #1466 (inbound comms never polled: `Maxim._gateway` is never assigned), #1467 (protocol
  plumbing), #1468 (a session that stops on an exception exits 0).
- The staged M79 "met" note moved to its own branch (`docs/m79-met`), so this PR's diff is #1436's alone.
- Hand-off (CLAUDE.md is the process session's file): extend the "Dead code accumulates silently" lesson with "...and
  an import of a missing module or name fails CI. Regression guard: scripts/lint_dead_imports.py +
  tests/unit/test_lint_dead_imports.py"; add `optional_dependency_installed` to the optional-deps lesson's intents.
