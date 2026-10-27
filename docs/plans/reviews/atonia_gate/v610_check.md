# v6.10 narrow delta check (Executor + Wire)

Scope: `git diff 136b9974 f98acb45 -- tests/`, the folds of `v69_check.md` S1 and S2. Read-only. Both touched gates
XFAIL as expected (`pytest -k "reads_no_process_stop or huge_trained"`: 2 xfailed, each on its "item 1 not built"
AssertionError). `ruff check` and `ruff format --check` are clean on the file.

## Fold S1: `test_closed_local_reads_no_process_stop_signal`

**Does a correct `closed_local` pass it?** Yes. I probed the scan logic on a scratch copy with a module shaped per
v6.10. It imports the shared kwargs builder from `llama_backend`, imports `require_optional_dependency`, and has a
`complete_tokens(..., should_stop)` whose `stopping_criteria` closure calls `should_stop()` and `deadline()`. Its
docstring and a comment both name `is_shutdown_requested` and `current_stop`. The scan finds no hits. Docstrings are
`ast.Constant` and comments are not in the AST, so there are no false positives from prose. `should_stop`,
`stopping_criteria` and the parameter names are not on the list.

**Does a polling `closed_local` fail it?** The probe flagged each of these:
- `from ...cancellation import is_shutdown_requested`;
- `from maxim.models.language import cancellation as c; c.is_shutdown_requested()`;
- `from maxim.utils.cancellation import StopRequest`;
- `anthropic_backend.is_shutdown_requested()`, which reaches the function through a re-export.

The import rule (any module ending in `cancellation`) also covers names the list leaves out: `request_stop`,
`yield_requested` and `StopRequest` (all in `utils/cancellation.py`, per `bounded_stop.md` §5), and `is_cancelled` and
`current_cancel_event` (in that module today). This does not contradict notes §1A.8, which rejects an import-closure
gate. The scan reads only `closed_local`'s own file, so the `runtime/` `StopRequest` imports planned by bounded_stop
cannot make it unflippable.

**Indirection.** At HEAD, `models/language/llama_backend.py` has no `cancellation` import and no shutdown polling.
Neither `_ensure` nor `complete` reads a stop signal. The only code `closed_local` will share with it is the
`Llama(**kwargs)` builder (§1A.3), which runs at `open()` and never during token generation. Generation goes through
`closed_local`'s own `stopping_criteria`, so the scan sees it.

**N1. NIT, fail-open.** The scan is not transitive. A future edit could add an `is_shutdown_requested()` check to the
shared builder in `llama_backend.py` ("for consistency with the other backends"). The observer would then refuse to
`open()` after a process shutdown, and this gate would not see it. The behavioural gate would not see it either,
because it uses the fake. Nothing like that exists today, and such an edit would be a reviewed `src/` change, so this
is a NIT. Cheap hardening: scan both `closed_local.py` and `llama_backend.py` (loop over the two specs).
`llama_backend` passes today, so its half would be a plain assertion inside the xfail test, or a separate non-xfail
test.

**N2. NIT, fail-open (adversarial only).** A string-built lookup evades the scan, for example
`importlib.import_module("maxim.models.language.cancellation")` plus `getattr(m, "is_" + "shutdown_requested")`
(confirmed by probe). Nobody writes that by accident. No action.

**N3. NIT.** If `closed_local` were built as a package, `spec.origin` would be its `__init__.py`, and the scan would
miss its submodules. The notes fix it as `closed_local.py`, so no action is needed. An assert that
`spec.submodule_search_locations is None` would make the assumption explicit.

**N4. NIT (cosmetic).** `import ast` sits after `import hashlib`. Ruff passes because isort ordering is not enforced
here.

## Fold S2: `test_closed_local_clamps_a_huge_trained_context`

The fold is sound.
- The temp file is a real regular file, so an `exists`/`is_file`/regular-file check in `closed_local` or the shared
  builder passes.
- The bytes start with the `GGUF` magic, so a cheap magic-byte check also passes.
- `trained_context` is patched by the name the notes pin (§1A.8 line "`closed_local.trained_context(path)`"), so the
  bytes are never parsed.
- `raised` is recorded and named in the "never constructed" message. A pre-`Llama` refusal is therefore visible
  instead of silent. That addresses v69 S2's concern about the broad `except`.
- The spy records its kwargs before it raises. A builder that swallows the spy's `ScenarioBroken` (as
  `_LlamaCppBackend._ensure`'s `except Exception: warn; return False` does today) still leaves `seen` populated, so the
  n_ctx assert runs.

**N5. NIT, fail-closed (carried from v69 N3).** `llama_backend.Llama` is not patched. Today `llama_backend` resolves
`Llama` at call time (`require_optional_dependency(...).Llama`), so patching `llama_cpp.Llama` reaches it. Only a
future module-top `from llama_cpp import Llama` would escape the spy. The gate would then fail visibly (a real load of
a 9-byte file raises, and `raised` names it), so the failure is fail-closed.

**N6. NIT, fail-closed.** A future pinned-GGUF hash check in `open_closed_local_backend`, run before `Llama`, would
stop the gate from flipping. The notes' entry point takes "the path alone", so this does not apply now. The
`raised!r` message would name it if it ever happened.

## Verdict

No BLOCKER and no SHOULD-FIX. Every finding is a NIT. The fail-open NITs (N1, N2) need a future edit to a file that
has no polling today (N1), or deliberate obfuscation (N2). N1's two-file scan is a cheap optional hardening.

STOP: no further round owed
