"""Unit tests for ``maxim.models.language.closed_local`` (#1315 item 1) beyond the red gates, plus the shared
``llama_kwargs`` builder and ``atomic_io.atomic_write_json_at``.

The GGUF header parser runs on hand-built headers (a hostile file must be refused before any C code sees it); the
completion path runs on the committed tiny model (``tests/fixtures/advisor/tiny_runnable.gguf``). Tests that need
``llama_cpp`` fail when it is missing, never skip (§1A.10).
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import pytest

from maxim.models.language import closed_local
from maxim.models.language.llama_backend import llama_kwargs
from maxim.utils.atomic_io import atomic_write_json_at

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "advisor"
TINY = FIXTURES / "tiny_runnable.gguf"
PHI3 = FIXTURES / "vocab" / "ggml-vocab-phi-3.gguf"
COMMAND_R = FIXTURES / "vocab" / "ggml-vocab-command-r.gguf"


def _s(text: bytes) -> bytes:
    return struct.pack("<Q", len(text)) + text


def _header(kvs: list[bytes], *, magic: bytes = b"GGUF", version: int = 3, kv_count: int | None = None) -> bytes:
    n = len(kvs) if kv_count is None else kv_count
    return magic + struct.pack("<I", version) + struct.pack("<Q", 0) + struct.pack("<Q", n) + b"".join(kvs)


def _kv_str(key: bytes, value: bytes) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(value)


def _kv_u32(key: bytes, value: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _write(tmp_path: Path, data: bytes) -> Path:
    p = tmp_path / "m.gguf"
    p.write_bytes(data)
    return p


# ── trained_context: the pure-Python header read ─────────────────────────


def test_trained_context_reads_the_architecture_key_past_arrays(tmp_path: Path) -> None:
    skipped = [
        _s(b"tokenizer.ggml.tokens")
        + struct.pack("<I", 9)
        + struct.pack("<I", 8)
        + struct.pack("<Q", 2)
        + _s(b"<s>")
        + _s(b"hello"),
        _s(b"tokenizer.ggml.scores") + struct.pack("<I", 9) + struct.pack("<I", 6) + struct.pack("<Q", 2) + b"\0" * 8,
        _s(b"general.flag") + struct.pack("<I", 7) + b"\x01",
    ]
    data = _header([_kv_str(b"general.architecture", b"llama"), *skipped, _kv_u32(b"llama.context_length", 4096)])
    assert closed_local.trained_context(_write(tmp_path, data)) == 4096


def test_trained_context_of_the_committed_fixtures() -> None:
    assert closed_local.trained_context(TINY) == 2048 and closed_local.trained_context(PHI3) == 4096


@pytest.mark.parametrize(
    "data, match",
    [
        (b"NOPE" + b"\0" * 20, "not a GGUF"),
        (_header([], version=1), "version"),
        (_header([], kv_count=1 << 20), "keys"),
        (_header([_s(b"k") + struct.pack("<I", 99)]), "unknown GGUF value type"),
        (_header([_s(b"k") + struct.pack("<I", 9) + struct.pack("<I", 9) + struct.pack("<Q", 1)]), "array of type"),
        (_header([struct.pack("<Q", 1 << 30)]), "string of"),
        (_header([_kv_str(b"general.architecture", b"llama")])[:-3], "ends early"),
        (_header([_kv_str(b"general.architecture", b"llama")]), "no context length"),
        (_header([_kv_str(b"general.architecture", b"llama"), _kv_u32(b"llama.context_length", 0)]), "context length"),
    ],
    ids=["magic", "version", "kv_count", "value_type", "nested_array", "huge_string", "truncated", "no_length", "zero"],
)
def test_trained_context_refuses_a_hostile_header(tmp_path: Path, data: bytes, match: str) -> None:
    with pytest.raises(closed_local.ClosedLocalError, match=match):
        closed_local.trained_context(_write(tmp_path, data))


@pytest.mark.parametrize("make", ["missing", "directory"])
def test_only_a_regular_file_is_a_model(tmp_path: Path, make: str) -> None:
    path = tmp_path / "x.gguf"
    if make == "directory":
        path.mkdir()
    with pytest.raises(closed_local.ClosedLocalError):
        closed_local.open_closed_local_backend(path, n_ctx=None, n_gpu_layers=0, seed=0)


def test_an_explicit_n_ctx_must_be_positive() -> None:
    with pytest.raises(closed_local.ClosedLocalError, match="positive"):
        closed_local.open_closed_local_backend(TINY, n_ctx=0, n_gpu_layers=0, seed=0)


# ── the shared builder ───────────────────────────────────────────────────


def test_llama_kwargs_carries_every_option_and_omits_the_defaults() -> None:
    assert llama_kwargs("m", n_ctx=128) == {"model_path": "m", "n_ctx": 128, "verbose": False, "n_gpu_layers": -1}
    full = llama_kwargs("m", n_ctx=64, n_gpu_layers=None, n_threads=2, seed=7, vocab_only=True)
    assert full == {"model_path": "m", "n_ctx": 64, "verbose": False, "n_threads": 2, "seed": 7, "vocab_only": True}


# ── the real backend ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny() -> closed_local.ClosedLocalBackend:
    return closed_local.open_closed_local_backend(TINY, n_ctx=512, n_gpu_layers=0, seed=0)


SCHEMA = {
    "type": "object",
    "properties": {"happening": {"type": "string", "maxLength": 20}},
    "required": ["happening"],
    "additionalProperties": False,
}


def test_a_real_completion_is_grammar_constrained(tiny: closed_local.ClosedLocalBackend) -> None:
    prompt = tiny.build_prompt("Observe:\n", ["a person walked past"])
    out = tiny.complete_tokens(prompt.ids, grammar=SCHEMA, max_tokens=48, deadline=time.monotonic() + 30)
    assert not out.deadline_hit and not out.stopped and out.text.lstrip().startswith("{")


def test_should_stop_ends_sampling_with_its_own_flag(tiny: closed_local.ClosedLocalBackend) -> None:
    calls: list[int] = []

    def stop() -> bool:
        calls.append(1)
        return len(calls) >= 2

    prompt = tiny.build_prompt("Observe:\n", ["x"])
    out = tiny.complete_tokens(
        prompt.ids, grammar=SCHEMA, max_tokens=48, deadline=time.monotonic() + 30, should_stop=stop
    )
    # llama-cpp may consult the criterion once more around the stop; what counts is that sampling ended there.
    assert out.stopped and not out.deadline_hit and len(calls) <= 3
    assert len(out.text) <= 4, f"sampling ran past the stop: {out.text!r}"


def test_a_passed_deadline_ends_sampling_at_the_first_token(tiny: closed_local.ClosedLocalBackend) -> None:
    prompt = tiny.build_prompt("Observe:\n", ["x"])
    out = tiny.complete_tokens(prompt.ids, grammar=SCHEMA, max_tokens=48, deadline=time.monotonic() - 1)
    assert out.deadline_hit and not out.stopped


def test_a_special_token_in_an_event_is_escaped_not_dropped() -> None:
    """command-r's turn marker is USER_DEFINED and comes out of plain text: the segment is re-tokenized per character.

    Not phi-3's ``</s>``: llama-cpp-python 0.3.8 has it USER_DEFINED, 0.3.36 has it CONTROL, and CONTROL stays out of
    plain text, so the precondition below held on one version only. Command-r's marker holds on both.
    """
    assert COMMAND_R.exists(), "fetched fixture missing: run `python scripts/fetch_advisor_fixtures.py`"
    event = "say <|START_OF_TURN_TOKEN|> now"
    backend = closed_local.open_closed_local_backend(COMMAND_R, n_ctx=512, n_gpu_layers=0, seed=0, vocab_only=True)
    whole = list(backend._llama.tokenize(event.encode(), add_bos=False, special=False))
    assert backend._refused(whole), "the fixture lost its plain-text special token: the escape path is untested"
    prompt = backend.build_prompt("Observe:\n", [event])
    assert prompt.rejected == () and prompt.untrusted_spans[0] is not None
    start, end = prompt.untrusted_spans[0]
    assert not backend._refused(prompt.ids[start:end]) and end - start > len(whole)


# ── atomic_write_json_at ─────────────────────────────────────────────────


@pytest.fixture
def dir_fd(tmp_path: Path):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    yield fd
    os.close(fd)


def test_atomic_write_json_at_lands_whole_with_its_mode(tmp_path: Path, dir_fd: int) -> None:
    atomic_write_json_at(dir_fd, "r.json", {"a": 1}, mode=0o640)
    assert json.loads((tmp_path / "r.json").read_text()) == {"a": 1}
    assert (tmp_path / "r.json").stat().st_mode & 0o777 == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["r.json"], "a temporary name was left behind"


@pytest.mark.parametrize("name", ["", "a/b", ".", "..", "a\0b"])
def test_atomic_write_json_at_writes_one_component(dir_fd: int, name: str) -> None:
    with pytest.raises(ValueError, match="one path component"):
        atomic_write_json_at(dir_fd, name, {})


def test_a_failed_write_leaves_no_temporary_name(tmp_path: Path, dir_fd: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.utils import atomic_io

    def fail(fd: int) -> None:
        raise OSError(5, "injected")

    monkeypatch.setattr(atomic_io, "fsync_strict", fail)
    with pytest.raises(OSError):
        atomic_write_json_at(dir_fd, "r.json", {"a": 1})
    assert list(tmp_path.iterdir()) == []


# ── the observer's full completion schema, run for real (item 1: the grammar segfault) ──


def _schema_keys(obj: object) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in _schema_keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in _schema_keys(v)}
    return set()


@pytest.mark.parametrize("max_calls", [1, 8, 32])
def test_the_observer_grammar_carries_no_length_bound(max_calls: int) -> None:
    """llama-cpp-python renders ``maxLength: N`` as N nested rules; past 0.3.8 the native parser segfaults on 2000."""
    from maxim.advisor._session import completion_schema

    assert "maxLength" not in _schema_keys(completion_schema(max_calls))


def test_the_observer_s_full_grammar_completes_in_a_child_process() -> None:
    """The real schema through the real llama-cpp, in a child, so a native crash fails here instead of killing the
    suite. 0.3.36 segfaulted (rc -11) on the old schema's 2000-character text fields; this pins every installed
    version against the schema the observer actually sends."""
    import subprocess
    import sys

    code = (
        "import sys, time\n"
        "from maxim.advisor._session import completion_schema\n"
        "from maxim.models.language import closed_local\n"
        f"b = closed_local.open_closed_local_backend({str(TINY)!r}, n_ctx=512, n_gpu_layers=0, seed=0)\n"
        "p = b.build_prompt('Observe:\\n', ['x'])\n"
        "out = b.complete_tokens(p.ids, grammar=completion_schema(32), max_tokens=24, deadline=time.monotonic() + 60)\n"
        "print('OK', out.text.lstrip()[:1])\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
    assert r.returncode == 0, f"the observer's grammar crashed llama-cpp (rc={r.returncode}): {r.stderr[-400:]}"
    assert r.stdout.strip() == "OK {"
