"""``scripts/fetch_advisor_fixtures.py``: every fixture is pinned, a mismatch is refused, and nothing is ever used
unverified (#1315 item 1, owner decision V17). No network: ``urlopen`` is replaced."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]


_PRIVATE_NAME = "_fetch_advisor_fixtures_under_test"  # never the name ``import`` resolves (script_module_guard)


def _script() -> Any:
    if _PRIVATE_NAME in sys.modules:
        return sys.modules[_PRIVATE_NAME]
    spec = importlib.util.spec_from_file_location(_PRIVATE_NAME, REPO / "scripts" / "fetch_advisor_fixtures.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PRIVATE_NAME] = mod  # @dataclass resolves its class's module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _fixture(mod: Any, data: bytes, *, committed: bool = False) -> Any:
    return mod.Fixture(
        "vocab/x.gguf", "https://example.invalid/x.gguf", len(data), hashlib.sha256(data).hexdigest(), committed
    )


def _serve(monkeypatch: pytest.MonkeyPatch, mod: Any, payload: bytes) -> list[str]:
    seen: list[str] = []

    class _Resp(io.BytesIO):
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: Any) -> None:
            self.close()

    def urlopen(url: str, timeout: float) -> _Resp:
        seen.append(url)
        return _Resp(payload)

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    return seen


def test_the_committed_fixtures_match_their_pins() -> None:
    """The two committed files are in the tree and verify; a stale or swapped one fails here, not at a gate."""
    mod = _script()
    for fx in mod.FIXTURES:
        if fx.committed:
            assert mod.verify(fx) is None, fx.path


def test_every_pin_is_a_full_sha256_at_a_fixed_revision() -> None:
    mod = _script()
    assert {f.path for f in mod.FIXTURES} == {
        "tiny_runnable.gguf",
        "vocab/ggml-vocab-phi-3.gguf",
        "vocab/ggml-vocab-command-r.gguf",
        "vocab/ggml-vocab-qwen2.gguf",
        "vocab/ggml-vocab-llama-bpe.gguf",
    }
    for f in mod.FIXTURES:
        assert len(f.sha256) == 64 and int(f.sha256, 16) >= 0, f.path
        assert "/main/" not in f.url and "/master/" not in f.url, f"{f.path} is pinned to a moving branch"


def test_a_fetch_that_matches_lands_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _script()
    data = b"GGUF" + b"\x00" * 100
    fx = _fixture(mod, data)
    seen = _serve(monkeypatch, mod, data)
    mod.fetch(fx, tmp_path)
    assert seen == [fx.url] and mod.verify(fx, tmp_path) is None
    assert not list(tmp_path.rglob("*.part"))


def test_a_fetch_that_does_not_match_is_refused_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _script()
    fx = _fixture(mod, b"GGUF-the-pinned-bytes")
    _serve(monkeypatch, mod, b"GGUF-swapped-bytes!!!")  # same length, different content
    with pytest.raises(ValueError, match="does not match its pin"):
        mod.fetch(fx, tmp_path)
    assert not (tmp_path / fx.path).exists() and not list(tmp_path.rglob("*.part"))


def test_verify_names_what_is_wrong(tmp_path: Path) -> None:
    mod = _script()
    fx = _fixture(mod, b"abcd")
    assert mod.verify(fx, tmp_path) == "missing"
    (tmp_path / "vocab").mkdir()
    (tmp_path / fx.path).write_bytes(b"abc")
    assert mod.verify(fx, tmp_path).startswith("size 3")
    (tmp_path / fx.path).write_bytes(b"abce")
    assert mod.verify(fx, tmp_path).startswith("sha256 ")


def test_check_mode_never_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _script()
    monkeypatch.setattr(mod, "FIXTURE_DIR", tmp_path)
    monkeypatch.setattr(mod, "FIXTURES", (_fixture(mod, b"abcd"),))
    seen = _serve(monkeypatch, mod, b"abcd")
    assert mod.main(["--check"]) == 1 and seen == []
    assert mod.main([]) == 0 and seen == ["https://example.invalid/x.gguf"]
