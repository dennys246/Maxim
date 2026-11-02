"""The observer's closed llama.cpp backend (#1315 item 1; ``atonia_gate_approach_notes.md`` §4.2, §1A.8, §1A.10).

It is NOT ``LLMRouter``. The router reads the data home (its cost tracker), registers ``atexit`` writes, reads
``MAXIM_*`` knobs and resolves profiles that may download. This module does none of that. Its whole surface:

- :func:`open_closed_local_backend` builds ``Llama`` EAGERLY, from a path alone (no profile, no lazy load, no
  reload, no env read);
- :meth:`ClosedLocalBackend.build_prompt` builds the prompt as token ids, with every untrusted segment defanged;
- :meth:`ClosedLocalBackend.complete_tokens` runs one grammar-constrained completion, bounded per token.

The prompt is token ids, never a string: sanitising strings fails (NFKC turns a fullwidth ``＜｜im＿end｜＞`` into
a real special token, and deleting a token string can recombine one). The trusted template is tokenized with
``special=True``; each untrusted segment with ``special=False``, and then every id it produced is checked by
ATTRIBUTE: ``special=False`` skips only CONTROL and UNKNOWN, so USER_DEFINED tokens still come out of plain text.
An id with the CONTROL, USER_DEFINED or UNKNOWN bit (or the BOS id) is escaped by re-tokenizing the segment one
character at a time, which may change its whitespace; a segment that still yields one is rejected, never dropped
silently (the caller lists it).

The only stop this module reads is the caller's ``should_stop``: never a process-wide stop signal, so an embedding
process's shutdown cannot abandon an observer turn it did not own (§1A.8 "Stopping").

Only ``maxim.advisor`` imports this module (the CI backend-import grep's allow-list).
"""

from __future__ import annotations

import json
import os
import stat
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO

from maxim.models.language.llama_backend import llama_kwargs
from maxim.utils.optional_deps import require_optional_dependency

#: ``n_ctx=None`` resolves to ``min(trained context, DEFAULT_CTX_CAP)``: never the full trained context, whose
#: prefill can take minutes or fail to allocate at ``open()`` (§1A.8 "Limits").
DEFAULT_CTX_CAP = 8192

_GGUF_MAGIC = b"GGUF"
# GGUF scalar value types: (struct format, byte size). 8 (string) and 9 (array) are variable-length.
_GGUF_SCALARS = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}
_GGUF_STRING, _GGUF_ARRAY = 8, 9
# Bounds a hostile header cannot get past: a real model's metadata is a few hundred keys.
_GGUF_MAX_KV = 1 << 16
_GGUF_MAX_STRING = 1 << 24


class ClosedLocalError(ValueError):
    """The model file is not one the closed backend will load: not a regular file, or not a readable GGUF."""


@dataclass(frozen=True)
class PromptIds:
    """A built prompt. ``ids`` is the whole prompt; ``untrusted_spans[i]`` is segment ``i``'s ``(start, end)`` in it,
    or ``None`` when segment ``i`` was rejected (its index is then in ``rejected``). In-process only: never persisted
    or sent, so CC3 does not apply."""

    ids: list[int]
    untrusted_spans: list[tuple[int, int] | None]
    rejected: tuple[int, ...]


@dataclass(frozen=True)
class Completion:
    """One completion. ``deadline_hit`` and ``stopped`` are this module's own flags, set by the per-token criterion:
    llama-cpp reports a criterion stop as ``finish_reason "stop"``, the same as a natural one. In-process only (CC3
    does not apply)."""

    text: str
    deadline_hit: bool
    stopped: bool
    finish_reason: str


def _llama_cpp() -> Any:
    return require_optional_dependency("llama_cpp", extra="llm-llama", feature="the observer's closed backend")


def _require_regular_file(path: str) -> None:
    try:
        st = os.stat(path)
    except OSError as e:
        raise ClosedLocalError(f"the model is not an existing file: {path!r} ({e.strerror})") from e
    if not stat.S_ISREG(st.st_mode):
        raise ClosedLocalError(f"the model is not a regular file: {path!r}")


def _read_exact(fh: BinaryIO, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise ClosedLocalError("the GGUF header ends early")
    return data


def _read_u(fh: BinaryIO, fmt: str, size: int) -> Any:
    return struct.unpack(fmt, _read_exact(fh, size))[0]


def _read_string(fh: BinaryIO) -> bytes:
    n = _read_u(fh, "<Q", 8)
    if n > _GGUF_MAX_STRING:
        raise ClosedLocalError(f"a GGUF string of {n} bytes: not a model header")
    return _read_exact(fh, n)


def _read_value(fh: BinaryIO, vtype: int) -> Any:
    if vtype in _GGUF_SCALARS:
        fmt, size = _GGUF_SCALARS[vtype]
        return _read_u(fh, fmt, size)
    if vtype == _GGUF_STRING:
        return _read_string(fh)
    if vtype == _GGUF_ARRAY:
        item_type = _read_u(fh, "<I", 4)
        count = _read_u(fh, "<Q", 8)
        if item_type in _GGUF_SCALARS:  # skip a fixed-size array without reading it
            fh.seek(count * _GGUF_SCALARS[item_type][1], os.SEEK_CUR)
        elif item_type == _GGUF_STRING:
            for _ in range(count):
                fh.seek(_read_u(fh, "<Q", 8), os.SEEK_CUR)
        else:
            raise ClosedLocalError(f"a GGUF array of type {item_type}")
        return None
    raise ClosedLocalError(f"an unknown GGUF value type {vtype}")


def trained_context(path: str | os.PathLike[str]) -> int:
    """The model's trained context length, read from the GGUF header in pure Python, before anything loads it.

    It looks up ``<general.architecture>.context_length``. A file that is not a GGUF, or a header that runs past its
    bounds, raises :class:`ClosedLocalError`; so does a header with no context length.
    """
    path = os.fspath(path)
    _require_regular_file(path)
    with open(path, "rb") as fh:
        if _read_exact(fh, 4) != _GGUF_MAGIC:
            raise ClosedLocalError(f"not a GGUF file: {path!r}")
        version = _read_u(fh, "<I", 4)
        if version < 2:
            raise ClosedLocalError(f"GGUF version {version} is not supported")
        _read_u(fh, "<Q", 8)  # tensor count
        kv_count = _read_u(fh, "<Q", 8)
        if kv_count > _GGUF_MAX_KV:
            raise ClosedLocalError(f"a GGUF header with {kv_count} keys: not a model header")
        values: dict[bytes, Any] = {}
        for _ in range(kv_count):
            key = _read_string(fh)
            values[key] = _read_value(fh, _read_u(fh, "<I", 4))
    arch = values.get(b"general.architecture")
    length = values.get(arch + b".context_length") if isinstance(arch, bytes) else None
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ClosedLocalError(f"the GGUF header names no context length: {path!r}")
    return int(length)


def _refused_mask(llama_cpp: Any) -> int:
    return (
        int(llama_cpp.LLAMA_TOKEN_ATTR_CONTROL)
        | int(llama_cpp.LLAMA_TOKEN_ATTR_USER_DEFINED)
        | int(llama_cpp.LLAMA_TOKEN_ATTR_UNKNOWN)
    )


def _token_attr(handle: Any, token_id: int) -> int:
    """The one attribute lookup (a bitmask of ``LLAMA_TOKEN_ATTR_*``); tests patch it for the synthetic refuse case."""
    return int(handle._model.token_get_attr(int(token_id)))


class ClosedLocalBackend:
    """A loaded model and the two calls the observer makes on it. Build it with :func:`open_closed_local_backend`."""

    def __init__(self, llama: Any, *, n_ctx: int, refused_mask: int) -> None:
        self._llama = llama
        self.n_ctx = int(n_ctx)
        self._refused_mask = refused_mask
        self._bos = int(llama.token_bos())

    def _refused(self, ids: Sequence[int]) -> bool:
        return any(i == self._bos or _token_attr(self._llama, i) & self._refused_mask for i in ids)

    def _untrusted_ids(self, segment: str) -> list[int] | None:
        """``segment``'s ids, defanged; ``None`` when it cannot be (a character that is itself a refused token)."""
        ids = list(self._llama.tokenize(segment.encode("utf-8"), add_bos=False, special=False))
        if not self._refused(ids):
            return ids
        escaped: list[int] = []
        for ch in segment:  # escape: one character per call, so no refused token can span characters
            char_ids = list(self._llama.tokenize(ch.encode("utf-8"), add_bos=False, special=False))
            if self._refused(char_ids):
                return None
            escaped.extend(char_ids)
        return escaped

    def build_prompt(self, template: str, segments: Sequence[str]) -> PromptIds:
        """The trusted ``template`` (``special=True``, with BOS), then each untrusted segment, defanged."""
        ids = list(self._llama.tokenize(str(template).encode("utf-8"), add_bos=True, special=True))
        spans: list[tuple[int, int] | None] = []
        rejected: list[int] = []
        for index, segment in enumerate(segments):
            seg_ids = self._untrusted_ids(str(segment))
            if seg_ids is None:
                spans.append(None)
                rejected.append(index)
                continue
            spans.append((len(ids), len(ids) + len(seg_ids)))
            ids.extend(seg_ids)
        return PromptIds(ids=ids, untrusted_spans=spans, rejected=tuple(rejected))

    def complete_tokens(
        self,
        prompt_ids: Sequence[int],
        *,
        grammar: dict[str, Any],
        max_tokens: int,
        deadline: float,
        should_stop: Callable[[], bool] | None = None,
    ) -> Completion:
        """One completion over ``prompt_ids``, constrained by the JSON schema ``grammar``.

        ``deadline`` is a ``time.monotonic()`` value. The criterion runs per sampled token: past ``deadline`` it sets
        ``deadline_hit``, and when ``should_stop()`` is true it sets ``stopped``; either ends sampling at that token.
        Prefill cannot be interrupted, so the caller bounds it by capping the prompt's length. Any error from
        llama-cpp propagates.
        """
        llama_cpp = _llama_cpp()
        flags = {"deadline_hit": False, "stopped": False}

        def criterion(_input_ids: Any, _logits: Any) -> bool:
            if should_stop is not None and should_stop():
                flags["stopped"] = True
                return True
            if time.monotonic() >= deadline:
                flags["deadline_hit"] = True
                return True
            return False

        out = self._llama.create_completion(
            prompt=list(prompt_ids),
            max_tokens=int(max_tokens),
            temperature=0.0,
            grammar=llama_cpp.LlamaGrammar.from_json_schema(json.dumps(grammar), verbose=False),
            stopping_criteria=llama_cpp.StoppingCriteriaList([criterion]),
        )
        choice = out["choices"][0]
        return Completion(
            text=str(choice.get("text") or ""),
            deadline_hit=flags["deadline_hit"],
            stopped=flags["stopped"],
            finish_reason=str(choice.get("finish_reason") or ""),
        )


def open_closed_local_backend(
    path: str | os.PathLike[str],
    *,
    n_ctx: int | None,
    n_gpu_layers: int,
    seed: int,
    vocab_only: bool = False,
) -> ClosedLocalBackend:
    """Load ``path`` eagerly and return the backend; the load is the whole construction (there is no reload).

    ``n_ctx=None`` resolves to ``min(trained_context(path), DEFAULT_CTX_CAP)``, read from the header before the load;
    an explicit ``n_ctx`` must be positive. ``vocab_only=True`` loads the tokenizer alone (the defang tests run on
    llama.cpp's vocab-only GGUFs); such a backend can build prompts but not complete.
    """
    path = os.fspath(path)
    _require_regular_file(path)
    if n_ctx is None:
        resolved = min(trained_context(path), DEFAULT_CTX_CAP)
    else:
        resolved = int(n_ctx)
        if resolved <= 0:
            raise ClosedLocalError(f"n_ctx must be positive (llama-cpp reads 0 as unbounded), not {n_ctx!r}")
    llama_cpp = _llama_cpp()
    llama = llama_cpp.Llama(
        **llama_kwargs(path, n_ctx=resolved, n_gpu_layers=n_gpu_layers, seed=seed, vocab_only=vocab_only)
    )
    return ClosedLocalBackend(llama, n_ctx=resolved, refused_mask=_refused_mask(llama_cpp))
