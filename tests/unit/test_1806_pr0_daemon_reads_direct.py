"""#1806 PR 0: every read of the Reachy daemon is dialled directly at a private address.

The probe (``hardware/reachy/probe.py``) and the controller's fresh readback carry
``address_policy="private", strict=True``. The REST DoA reader is the third daemon read
and had neither: an environment proxy could carry the azimuth the orient loop acts on.
"""

from __future__ import annotations

from typing import Any

from maxim.embodiment.audio_localization import make_reachy_rest_doa_reader


class _Resp:
    def json(self) -> dict[str, Any]:
        return {"angle": 0.5, "speech_detected": True}


def test_the_rest_doa_reader_dials_the_daemon_directly(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def _fake_fetch_url(url: str, **kwargs: Any) -> _Resp:
        seen["url"] = url
        seen.update(kwargs)
        return _Resp()

    monkeypatch.setattr("maxim.utils.http.fetch_url", _fake_fetch_url)
    reader = make_reachy_rest_doa_reader("10.0.0.5")
    assert reader() == (0.5, True)
    assert seen["url"].startswith("http://10.0.0.5:")
    assert seen.get("address_policy") == "private"
    assert seen.get("strict") is True


def test_maxim_diagnostics_docs_read_is_direct(monkeypatch) -> None:
    """#1806 PR 0 (review W-2): maxim-diagnostics' /docs read is a daemon read too."""
    import inspect

    import maxim.utils.reachy_diagnostics as diag

    src = inspect.getsource(diag)
    docs_call = src[src.index('8000/docs"') - 80 : src.index('8000/docs"') + 120]
    assert 'address_policy="private"' in docs_call and "strict=True" in docs_call


def test_edit_file_invalidates_the_result_cache_like_write_file() -> None:
    """#1806 PR 0 (review A13): a successful edit_file drops cached reads of that path."""
    from maxim.runtime.tool_dispatch import _invalidate_cache_after_write

    class _Cache:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def invalidate(self, *, path: str) -> int:
            self.paths.append(path)
            return 1

    for tool in ("write_file", "edit_file"):
        cache = _Cache()
        _invalidate_cache_after_write(cache, {"tool_name": tool, "params": {"path": "a.py"}}, True)
        assert cache.paths == ["a.py"], tool
    cache = _Cache()
    _invalidate_cache_after_write(cache, {"tool_name": "edit_file", "params": {"path": "a.py"}}, False)
    _invalidate_cache_after_write(cache, {"tool_name": "read_file", "params": {"path": "a.py"}}, True)
    assert cache.paths == []
