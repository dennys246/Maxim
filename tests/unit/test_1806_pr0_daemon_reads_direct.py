"""#1806 PR 0: every read of the Reachy daemon is dialled directly at a private address.

The probe (``hardware/reachy/probe.py``) and the controller's fresh readback carry
``address_policy="private", proxy="never"``. The REST DoA reader is the third daemon read
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
    assert seen.get("proxy") == "never"
